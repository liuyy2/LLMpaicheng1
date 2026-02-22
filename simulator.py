"""
仿真器模块 - Rolling Horizon 仿真主循环
"""

from __future__ import annotations

import time
import copy
import json
import csv
import os
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any

from config import Config, DEFAULT_CONFIG
from solver_cpsat import (
    Mission, Operation, Resource, PlanV2_1, OpAssignment,
    SolverResult, SolveStatus,
    solve_v2_1, compute_frozen_ops, SolverConfigV2_1
)
from scenario import Scenario, DisturbanceEvent
from metrics import (
    RollingMetrics, EpisodeMetrics,
    compute_rolling_metrics_ops, compute_episode_metrics_ops,
    compute_plan_drift_ops,
    metrics_to_dict, rolling_metrics_to_dict
)
from features import compute_state_features_ops, build_trcg_summary
from policies.base import BasePolicy, MetaParams

import logging
logger = logging.getLogger(__name__)



def assignment_to_dict(assign: Any) -> Dict[str, Any]:
    if hasattr(assign, 'op_id'):
        return {
            'op_id': assign.op_id,
            'mission_id': assign.mission_id,
            'op_index': assign.op_index,
            'resources': list(assign.resources),
            'start_slot': assign.start_slot,
            'end_slot': assign.end_slot
        }

    return {'id': str(assign)}


@dataclass
class RollingSnapshot:
    t: int
    plan: Optional[Any]
    solve_status: SolveStatus
    solve_time_ms: int
    metrics: RollingMetrics
    meta_params: Optional[MetaParams] = None
    infeasible_reasons: List[str] = field(default_factory=list)
    state_features: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        mp_dict = None
        if self.meta_params:
            mp_dict = {
                'w_delay': self.meta_params.w_delay,
                'w_shift': self.meta_params.w_shift,
                'w_switch': self.meta_params.w_switch,
                'freeze_horizon': self.meta_params.freeze_horizon,
                'use_two_stage': self.meta_params.use_two_stage,
                'epsilon_solver': self.meta_params.epsilon_solver,
                'kappa_win': self.meta_params.kappa_win,
                'kappa_seq': self.meta_params.kappa_seq,
                # TRCG Repair 扩展字段
                'decision_source': self.meta_params.decision_source,
                'fallback_reason': self.meta_params.fallback_reason,
                'attempt_idx': self.meta_params.attempt_idx,
            }
            if self.meta_params.unlock_mission_ids is not None:
                mp_dict['unlock_mission_ids'] = list(self.meta_params.unlock_mission_ids)
            if self.meta_params.root_cause_mission_id:
                mp_dict['root_cause_mission_id'] = self.meta_params.root_cause_mission_id
        return {
            't': self.t,
            'solve_status': self.solve_status.value,
            'solve_time_ms': self.solve_time_ms,
            'plan': self.plan.to_dict() if self.plan else None,
            'metrics': rolling_metrics_to_dict(self.metrics),
            'meta_params': mp_dict,
            'infeasible_reasons': self.infeasible_reasons,
            'state_features': self.state_features
        }


@dataclass
class EpisodeResult:
    seed: int
    policy_name: str
    snapshots: List[RollingSnapshot]
    metrics: EpisodeMetrics
    final_schedule: List[Any]
    completed_tasks: Set[str]
    uncompleted_tasks: Set[str]
    total_runtime_s: float

    def to_dict(self) -> dict:
        return {
            'seed': self.seed,
            'policy_name': self.policy_name,
            'metrics': metrics_to_dict(self.metrics),
            'final_schedule': [assignment_to_dict(a) for a in self.final_schedule],
            'completed_tasks': list(self.completed_tasks),
            'uncompleted_tasks': list(self.uncompleted_tasks),
            'total_runtime_s': round(self.total_runtime_s, 3),
            'num_snapshots': len(self.snapshots)
        }


@dataclass
class SimulationStateOps:
    now: int
    missions: List[Mission]
    resources: List[Resource]
    current_plan: Optional[PlanV2_1]

    started_ops: Set[str] = field(default_factory=set)
    completed_ops: Set[str] = field(default_factory=set)
    applied_events: Set[int] = field(default_factory=set)
    actual_durations: Dict[str, int] = field(default_factory=dict)

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        for m in self.missions:
            if m.mission_id == mission_id:
                return m
        return None

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        for r in self.resources:
            if r.resource_id == resource_id:
                return r
        return None

    def get_operation(self, op_id: str) -> Optional[Operation]:
        for mission in self.missions:
            for op in mission.operations:
                if op.op_id == op_id:
                    return op
        return None

    def is_mission_completed(self, mission: Mission) -> bool:
        launch = mission.get_launch_op()
        return bool(launch and launch.op_id in self.completed_ops)

    def get_schedulable_missions(self, horizon_end: int) -> List[Mission]:
        schedulable = []
        for mission in self.missions:
            if self.is_mission_completed(mission):
                continue
            if mission.release <= horizon_end:
                schedulable.append(mission)
        return schedulable


def _create_initial_state_ops(missions: List[Mission], resources: List[Resource]) -> SimulationStateOps:
    return SimulationStateOps(
        now=0,
        missions=copy.deepcopy(missions),
        resources=copy.deepcopy(resources),
        current_plan=None
    )


def _update_op_status(state: SimulationStateOps, now: int) -> SimulationStateOps:
    if state.current_plan:
        for assign in state.current_plan.op_assignments:
            if assign.op_id in state.completed_ops:
                continue
            if assign.start_slot <= now:
                state.started_ops.add(assign.op_id)
            if assign.end_slot <= now:
                state.completed_ops.add(assign.op_id)
    return state


def _merge_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not windows:
        return []

    sorted_wins = sorted(windows, key=lambda x: x[0])
    merged = [sorted_wins[0]]

    for start, end in sorted_wins[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def _compute_op6_candidate_windows(
    mission_windows: List[Tuple[int, int]],
    range_calendar: Dict[int, List[Tuple[int, int]]],
    op6_duration: int,
    sim_total_slots: int = 960
) -> List[Tuple[int, int]]:
    """
    计算 Op6 的候选窗口：mission_windows 与 range_calendar 的交集
    
    对每个 mission_window，与对应时间范围内的 range_calendar 窗口取交集
    最后过滤掉长度 < op6_duration 的窗口
    
    返回合并后的候选窗口列表
    """
    if not range_calendar:
        # 如果没有 range_calendar，直接返回 mission_windows
        return mission_windows
    
    slots_per_day = 96
    candidate_windows = []
    
    for mw_start, mw_end in mission_windows:
        # 找到 mission_window 覆盖的天
        start_day = mw_start // slots_per_day
        end_day = mw_end // slots_per_day
        
        # 收集这些天的 range_calendar 窗口
        range_windows = []
        for day in range(start_day, end_day + 1):
            if day in range_calendar:
                range_windows.extend(range_calendar[day])
        
        # 计算交集
        for rw_start, rw_end in range_windows:
            inter_start = max(mw_start, rw_start)
            inter_end = min(mw_end, rw_end)
            
            if inter_start < inter_end:
                candidate_windows.append((inter_start, inter_end))
    
    # 过滤掉长度不足的窗口
    candidate_windows = [
        (s, e) for s, e in candidate_windows
        if (e - s) >= op6_duration
    ]
    
    # 合并重叠窗口
    if candidate_windows:
        candidate_windows = _merge_windows(candidate_windows)
    
    return candidate_windows


def _apply_weather_disturbance_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    params = event.params
    delete_ratio = params.get('delete_ratio', 0.3)
    affected_start = params.get('affected_start', event.trigger_time)
    affected_end = params.get('affected_end', event.trigger_time + 12)

    for mission in state.missions:
        launch = mission.get_launch_op()
        if not launch:
            continue
        if launch.op_id in state.completed_ops or launch.op_id in state.started_ops:
            continue

        new_windows = []
        for win_start, win_end in launch.time_windows:
            if win_end < affected_start or win_start > affected_end:
                new_windows.append((win_start, win_end))
            else:
                overlap_start = max(win_start, affected_start)
                overlap_end = min(win_end, affected_end)
                overlap_len = overlap_end - overlap_start + 1
                delete_len = int(overlap_len * delete_ratio)

                if win_start < affected_start:
                    new_windows.append((win_start, affected_start - 1))
                if win_end > affected_end:
                    new_windows.append((affected_end + 1, win_end))

                if delete_len < overlap_len:
                    kept_start = overlap_start + delete_len
                    if kept_start <= overlap_end:
                        new_windows.append((kept_start, overlap_end))

        if new_windows:
            launch.time_windows = _merge_windows(new_windows)
        if not launch.time_windows:
            launch.time_windows = [(affected_end + 1, affected_end + 30)]


def _apply_pad_outage_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    resource_id = event.target_id
    params = event.params
    outage_start = params.get('outage_start', event.trigger_time + 1)
    outage_end = params.get('outage_end', event.trigger_time + 10)

    resource = state.get_resource(resource_id)
    if resource:
        new_interval = (outage_start, outage_end)
        if new_interval not in resource.unavailable:
            resource.unavailable.append(new_interval)
            resource.unavailable = _merge_windows(resource.unavailable)


def _apply_duration_disturbance_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    op_id = event.target_id
    params = event.params
    multiplier = params.get('multiplier', 1.0)

    op = state.get_operation(op_id)
    if op and op_id not in state.started_ops:
        # 跳过 wait op 和 launch op（它们的时长由求解器管理）
        wait_op = None
        launch_op = None
        for mission in state.missions:
            for mop in mission.operations:
                if mop.op_id == op_id:
                    wait_op = mission.get_wait_op()
                    launch_op = mission.get_launch_op()
                    break
            if wait_op or launch_op:
                break
        if (wait_op and op_id == wait_op.op_id) or (launch_op and op_id == launch_op.op_id):
            return
        new_duration = max(1, int(round(op.duration * multiplier)))
        state.actual_durations[op_id] = new_duration
        op.duration = new_duration


def _apply_range_closure_ops(
    state: SimulationStateOps, 
    event: DisturbanceEvent,
    scenario: Any  # Scenario object with range_calendar
) -> None:
    """
    应用 range_closure 扰动到 scenario.range_calendar
    
    实现区间减法：从当天的 range_calendar 窗口中减去 closure 区间
    带可行性护栏：
    1) 不能让当天 range_calendar 变空
    2) 不能让任何未完成任务的 Op6 候选窗口变空
    
    如果违反护栏，则跳过该 closure（记录 skipped）
    """
    params = event.params
    day = params.get('day', 0)
    closure_start = params.get('closure_start', event.trigger_time)
    closure_end = params.get('closure_end', event.trigger_time + 10)
    
    if not hasattr(scenario, 'range_calendar') or day not in scenario.range_calendar:
        return
    
    original_windows = scenario.range_calendar[day]
    
    # 应用区间减法
    new_windows = []
    for win_start, win_end in original_windows:
        if closure_end < win_start or closure_start >= win_end:
            # 无交集
            new_windows.append((win_start, win_end))
        elif closure_start <= win_start and closure_end >= win_end:
            # closure 完全覆盖窗口，删除
            pass
        elif closure_start > win_start and closure_end < win_end:
            # closure 在窗口中间，切成两段，保留较长的一段
            left_len = closure_start - win_start
            right_len = win_end - closure_end
            if left_len >= right_len:
                new_windows.append((win_start, closure_start - 1))
            else:
                new_windows.append((closure_end + 1, win_end))
        elif closure_start <= win_start:
            # closure 覆盖左侧
            new_windows.append((closure_end + 1, win_end))
        else:
            # closure 覆盖右侧
            new_windows.append((win_start, closure_start - 1))
    
    # 护栏检查 A: 当天 range_calendar 不能变空
    if not new_windows:
        # 跳过该 closure
        return
    
    # 护栏检查 B: 对每个未完成任务，检查 Launch 候选窗口是否仍有效
    for mission in state.missions:
        launch = mission.get_launch_op()
        if not launch or launch.op_id in state.completed_ops or launch.op_id in state.started_ops:
            continue
        
        if not launch.time_windows:
            continue
        
        # 计算 candidate_windows = intersect(mission_windows, new_range_calendar)
        # 简化：只检查是否至少有一个 mission_window 与 new_windows 有交集
        has_valid_window = False
        for mw_start, mw_end in launch.time_windows:
            for rw_start, rw_end in new_windows:
                # 检查交集
                inter_start = max(mw_start, rw_start)
                inter_end = min(mw_end, rw_end)
                if inter_start < inter_end and (inter_end - inter_start) >= launch.duration:
                    has_valid_window = True
                    break
            if has_valid_window:
                break
        
        if not has_valid_window:
            # 该任务会失去所有候选窗口，跳过 closure
            return
    
    # 通过所有护栏检查，应用 closure
    scenario.range_calendar[day] = new_windows


def _apply_release_jitter_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    """释放时间扰动：将目标任务的所有工序 release 向后推迟 jitter_slots 个时隙。
    
    只影响尚未开始的任务，避免对已执行的工序产生矛盾。
    推迟后，若当前排程中该任务某工序 start_slot < new_release，
    则 _check_plan_feasibility_ops 会检测到不可行并触发强制重排。
    """
    mission_id = event.target_id
    jitter = event.params.get("jitter_slots", 0)
    if jitter <= 0 or not mission_id:
        return

    mission = state.get_mission(mission_id)
    if not mission:
        return

    # 如果任务中任何工序已开始或已完成，不再施加抖动
    if any(op.op_id in state.started_ops or op.op_id in state.completed_ops
           for op in mission.operations):
        return

    # 推迟 mission release 和所有 operation release
    mission.release += jitter
    for op in mission.operations:
        op.release += jitter


def _apply_disturbance_ops(
    state: SimulationStateOps,
    now: int,
    events: List[DisturbanceEvent],
    scenario: Any,  # Scenario object
    last_now: int = 0
) -> SimulationStateOps:
    for idx, event in enumerate(events):
        if idx in state.applied_events:
            continue
        if event.trigger_time > now:
            continue
        if event.trigger_time <= last_now and last_now > 0:
            continue

        if event.event_type == 'weather':
            _apply_weather_disturbance_ops(state, event)
        elif event.event_type in ('pad_outage', 'closure_change'):
            _apply_pad_outage_ops(state, event)
        elif event.event_type == 'duration':
            _apply_duration_disturbance_ops(state, event)
        elif event.event_type == 'range_closure':
            _apply_range_closure_ops(state, event, scenario)
        elif event.event_type == 'release_jitter':
            _apply_release_jitter_ops(state, event)

        state.applied_events.add(idx)

    return state


def _check_plan_feasibility_ops(
    state: SimulationStateOps,
    now: int
) -> Tuple[bool, List[str]]:
    if state.current_plan is None:
        return True, []

    infeasible = []
    assignments = {a.op_id: a for a in state.current_plan.op_assignments}

    for mission in state.missions:
        for op in mission.operations:
            assign = assignments.get(op.op_id)
            if not assign:
                continue
            if op.op_id in state.completed_ops or op.op_id in state.started_ops:
                continue

            if assign.start_slot < op.release:
                infeasible.append(f'release_{op.op_id}')
                return False, infeasible

            # 检查时长约束：跳过 wait op（弹性时长）
            wait_op = mission.get_wait_op()
            is_wait_op = (wait_op and op.op_id == wait_op.op_id)
            if not is_wait_op and assign.end_slot - assign.start_slot != op.duration:
                infeasible.append(f'duration_{op.op_id}')
                return False, infeasible

            # 检查窗口约束：launch op 必须落在 time_windows 内
            launch_op = mission.get_launch_op()
            is_launch_op = (launch_op and op.op_id == launch_op.op_id)
            if is_launch_op and op.time_windows:
                in_window = any(
                    assign.start_slot >= ws and assign.end_slot <= we
                    for ws, we in op.time_windows
                )
                if not in_window:
                    infeasible.append(f'window_{op.op_id}')
                    return False, infeasible

            for res_id in op.resources:
                resource = state.get_resource(res_id)
                if not resource:
                    continue
                for ua_start, ua_end in resource.unavailable:
                    if not (assign.end_slot <= ua_start or assign.start_slot > ua_end):
                        infeasible.append(f'closure_{res_id}_{op.op_id}')
                        return False, infeasible

    for mission in state.missions:
        for op in mission.operations:
            assign = assignments.get(op.op_id)
            if not assign:
                continue
            if op.op_id in state.completed_ops or op.op_id in state.started_ops:
                continue
            for pred_id in op.precedences:
                pred_assign = assignments.get(pred_id)
                if pred_assign and assign.start_slot < pred_assign.end_slot:
                    infeasible.append(f'precedence_{op.op_id}')
                    return False, infeasible

    return True, []


# ============================================================================
# TRCG Repair 回退链（simulator 侧）
# ============================================================================

def _solve_with_trcg_fallback(
    initial_result: SolverResult,
    meta_params: MetaParams,
    state: "SimulationStateOps",
    now: int,
    config: Config,
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    solver_config_base: SolverConfigV2_1,
    scenario: Any,
    verbose: bool = False,
) -> Tuple[SolverResult, MetaParams, Any]:
    """
    当 TRCGRepairPolicy 提供的 unlock_mission_ids 初次求解失败时，
    调用 solve_with_fallback_chain 执行 3 次降级重试 + 最终全局回退。

    返回 (SolverResult, updated_meta_params, FallbackChainResult)。
    """
    from policies.policy_llm_repair import (
        RepairDecision, solve_with_fallback_chain,
        heuristic_repair_decision, FREEZE_HOURS_TO_SLOTS,
    )

    # 重建 RepairDecision（从 meta_params 反向映射）
    freeze_h_hours = 8  # 默认
    for h, s in FREEZE_HOURS_TO_SLOTS.items():
        if s == (meta_params.freeze_horizon or config.freeze_horizon):
            freeze_h_hours = h
            break

    unlock_ids = list(meta_params.unlock_mission_ids) if meta_params.unlock_mission_ids else []

    decision = RepairDecision(
        root_cause_mission_id=meta_params.root_cause_mission_id or (unlock_ids[0] if unlock_ids else ""),
        unlock_mission_ids=unlock_ids,
        freeze_horizon_hours=freeze_h_hours,
        epsilon_solver=meta_params.epsilon_solver or config.default_epsilon_solver,
        analysis_short="rebuild_from_meta",
        secondary_root_cause_mission_id=meta_params.secondary_root_cause_mission_id,
    )

    # eligible_ids: 活跃且未开始/完成的 mission
    completed_ops = getattr(state, 'completed_ops', set())
    started_ops = getattr(state, 'started_ops', set())

    completed_mission_ids = set()
    started_mission_ids = set()
    for m in missions:
        launch = m.get_launch_op()
        if launch and launch.op_id in completed_ops:
            completed_mission_ids.add(m.mission_id)
            continue
        for op in m.operations:
            if op.op_id in started_ops:
                started_mission_ids.add(m.mission_id)
                break

    eligible_ids = {m.mission_id for m in missions} - started_mission_ids - completed_mission_ids

    # 构建 TRCG dict（用于 expand_unlock）
    try:
        trcg = build_trcg_summary(
            missions=state.missions,
            resources=state.resources,
            plan=state.current_plan,
            now=now,
            config=config,
            started_ops=started_ops,
            completed_ops=completed_ops,
            actual_durations=getattr(state, 'actual_durations', {}),
            frozen_ops=frozen_ops,
        )
        trcg_dict = trcg.to_dict()
    except Exception:
        trcg_dict = {}

    if verbose:
        print(f"   TRCG fallback chain triggered (initial: {initial_result.status.value})")

    chain = solve_with_fallback_chain(
        decision=decision,
        trcg_dict=trcg_dict,
        missions=missions,
        resources=resources,
        horizon=horizon,
        prev_plan=prev_plan,
        frozen_ops=frozen_ops,
        now=now,
        eligible_ids=eligible_ids,
        solver_config_base=solver_config_base,
        compute_frozen_ops_fn=compute_frozen_ops,
        current_plan_for_refreeze=state.current_plan,
        started_ops=started_ops,
        completed_ops=completed_ops,
    )

    # 更新 meta_params 中的回退信息
    if chain.success:
        meta_params.attempt_idx = len(chain.attempts) - 1
        final = chain.attempts[-1] if chain.attempts else None
        if final and final.attempt_name == "final_global_replan":
            meta_params.decision_source = "forced_global"
            meta_params.fallback_reason = "stage2_infeasible_forced_global"
        else:
            meta_params.fallback_reason = f"solver_retry_{chain.final_attempt_name}"
    else:
        meta_params.decision_source = "forced_global"
        meta_params.fallback_reason = "all_fallback_failed"
        meta_params.attempt_idx = len(chain.attempts)

    if verbose:
        print(f"   Fallback chain: success={chain.success}, "
              f"final={chain.final_attempt_name}, calls={chain.total_solver_calls}")

    return chain.solver_result, meta_params, chain


# ============================================================================
# INFEASIBLE 回退机制 —— 确保 solver 失败时新任务仍能进入计划
# ============================================================================

def _fallback_on_infeasible(
    state: SimulationStateOps,
    now: int,
    config: Config,
    missions_to_schedule: List[Mission],
    solver_config: SolverConfigV2_1,
    solver_horizon: int,
    frozen_ops: Dict[str, OpAssignment],
    scenario: Any,
    verbose: bool = False,
) -> SolverResult:
    """
    当 CP-SAT solver 对全部 mission 返回 INFEASIBLE/TIMEOUT 时的回退策略。

    回退链（三级降级）:
      Level 1 — 增量求解：保留旧计划中已排定的 mission 不变，仅对新增 mission 调用 solver
      Level 2 — 宽松求解：关闭 two-stage、去掉 freeze，对全部 mission 重新求解
      Level 3 — Greedy 兜底：使用 EDF Greedy 生成完整计划

    任何一级成功即停止并返回。

    Returns:
        SolverResult（status=OPTIMAL/FEASIBLE 表示成功）
    """
    old_plan = state.current_plan

    # ------------------------------------------------------------------
    # 收集已在旧计划中执行/开始的 mission（这些的 assignments 必须保留）
    # ------------------------------------------------------------------
    planned_mission_ids: Set[str] = set()
    preserved_assignments: List[OpAssignment] = []
    preserved_op_ids: Set[str] = set()

    if old_plan:
        for assign in old_plan.op_assignments:
            planned_mission_ids.add(assign.mission_id)
            # 保留已 completed / started 的 assignments
            if assign.op_id in state.completed_ops or assign.op_id in state.started_ops:
                preserved_assignments.append(assign)
                preserved_op_ids.add(assign.op_id)

    # 分离新增 mission vs 已排定 mission
    new_missions = [m for m in missions_to_schedule
                    if m.mission_id not in planned_mission_ids]
    old_missions = [m for m in missions_to_schedule
                    if m.mission_id in planned_mission_ids]

    if verbose:
        print(f"   [Fallback] total={len(missions_to_schedule)}, "
              f"old_planned={len(old_missions)}, new={len(new_missions)}, "
              f"preserved_ops={len(preserved_assignments)}")

    # ==================================================================
    # Level 1: 增量求解 — 仅对新增 mission 调用 solver
    # ==================================================================
    if new_missions:
        if verbose:
            print(f"   [Fallback L1] Incremental solve for {len(new_missions)} new missions")

        # 为新增 mission 构建独立 solver config（无 prev_plan，无 freeze）
        incr_config = SolverConfigV2_1(
            horizon_slots=solver_config.horizon_slots,
            w_delay=solver_config.w_delay,
            w_shift=0.0,         # 新 mission 无历史，不惩罚 shift
            w_switch=0.0,        # 新 mission 无历史，不惩罚 switch
            time_limit_seconds=solver_config.time_limit_seconds * 0.5,
            num_workers=solver_config.num_workers,
            op5_max_wait_slots=solver_config.op5_max_wait_slots,
            use_two_stage=False,  # 单阶段即可
            epsilon_solver=solver_config.epsilon_solver,
            kappa_win=solver_config.kappa_win,
            kappa_seq=solver_config.kappa_seq,
            stage1_time_ratio=solver_config.stage1_time_ratio,
        )

        # 构建资源占用：已保留 assignments 作为不可用区间叠加到资源上
        augmented_resources = _augment_resources_with_old_plan(
            state.resources, old_plan, preserved_op_ids, state.completed_ops, state.started_ops, now
        )

        incr_result = solve_v2_1(
            missions=new_missions,
            resources=augmented_resources,
            horizon=solver_horizon,
            prev_plan=None,
            frozen_ops={},
            config=incr_config,
            now=now,
        )

        if incr_result.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) and incr_result.plan is not None:
            # 合并：旧计划（全部 assignments 中的活跃部分） + 新增 mission 的 assignments
            incr_plan: PlanV2_1 = incr_result.plan  # type: ignore[assignment]
            merged = _merge_old_and_new_plan(old_plan, incr_plan, state, now)
            if verbose:
                print(f"   [Fallback L1] SUCCESS — merged {len(merged.op_assignments)} ops")
            return SolverResult(
                status=incr_result.status,
                plan=merged,
                objective_value=incr_result.objective_value,
                solve_time_ms=incr_result.solve_time_ms,
                degradation_count=1,
                degradation_actions=["incremental_solve"],
            )
        elif verbose:
            print(f"   [Fallback L1] FAILED — {incr_result.status}")

    # ==================================================================
    # Level 2: 宽松求解 — 关闭 two-stage / freeze，对全部 mission 重新求解
    # ==================================================================
    if verbose:
        print(f"   [Fallback L2] Relaxed solve for all {len(missions_to_schedule)} missions")

    relaxed_config = SolverConfigV2_1(
        horizon_slots=solver_config.horizon_slots,
        w_delay=solver_config.w_delay,
        w_shift=0.0,
        w_switch=0.0,
        time_limit_seconds=solver_config.time_limit_seconds,
        num_workers=solver_config.num_workers,
        op5_max_wait_slots=solver_config.op5_max_wait_slots,
        use_two_stage=False,
        epsilon_solver=0.0,
        kappa_win=solver_config.kappa_win,
        kappa_seq=solver_config.kappa_seq,
        stage1_time_ratio=solver_config.stage1_time_ratio,
    )

    # 只冻结已 started 的 ops（不冻结未来的）
    minimal_frozen = {}
    if old_plan:
        for assign in old_plan.op_assignments:
            if assign.op_id in state.started_ops and assign.op_id not in state.completed_ops:
                minimal_frozen[assign.op_id] = assign

    relaxed_result = solve_v2_1(
        missions=missions_to_schedule,
        resources=state.resources,
        horizon=solver_horizon,
        prev_plan=None,      # 无 prev_plan → 无 shift/switch 约束
        frozen_ops=minimal_frozen,
        config=relaxed_config,
        now=now,
    )

    if relaxed_result.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) and relaxed_result.plan is not None:
        # 将已完成/已开始 ops 的 assignments 注入到新计划中
        relaxed_plan: PlanV2_1 = relaxed_result.plan  # type: ignore[assignment]
        merged = _inject_preserved_assignments(relaxed_plan, preserved_assignments)
        if verbose:
            print(f"   [Fallback L2] SUCCESS — {len(merged.op_assignments)} ops")
        return SolverResult(
            status=relaxed_result.status,
            plan=merged,
            objective_value=relaxed_result.objective_value,
            solve_time_ms=relaxed_result.solve_time_ms,
            degradation_count=2,
            degradation_actions=["incremental_solve_failed", "relaxed_solve"],
        )
    elif verbose:
        print(f"   [Fallback L2] FAILED — {relaxed_result.status}")

    # ==================================================================
    # Level 3: Greedy 兜底 — 使用 EDF Greedy 生成完整计划
    # ==================================================================
    if verbose:
        print(f"   [Fallback L3] Greedy fallback")

    from policies.policy_greedy import GreedyPolicy
    greedy_policy = GreedyPolicy(policy_name="_fallback_greedy")
    _, greedy_plan = greedy_policy.decide(state, now, config)

    if greedy_plan and greedy_plan.op_assignments:
        if verbose:
            print(f"   [Fallback L3] SUCCESS — {len(greedy_plan.op_assignments)} ops")
        return SolverResult(
            status=SolveStatus.FEASIBLE,
            plan=greedy_plan,
            objective_value=0.0,
            solve_time_ms=0,
            degradation_count=3,
            degradation_actions=["incremental_solve_failed", "relaxed_solve_failed", "greedy_fallback"],
        )

    # 所有回退均失败 — 返回 INFEASIBLE（保留旧计划由调用方处理）
    if verbose:
        print(f"   [Fallback] ALL LEVELS FAILED")
    return SolverResult(
        status=SolveStatus.INFEASIBLE,
        plan=None,
        degradation_count=3,
        degradation_actions=["all_fallback_failed"],
    )


def _augment_resources_with_old_plan(
    resources: List[Resource],
    old_plan: Optional[PlanV2_1],
    preserved_op_ids: Set[str],
    completed_ops: Set[str],
    started_ops: Set[str],
    now: int,
) -> List[Resource]:
    """
    将旧计划中仍活跃（未完成且未来要执行）的 assignments 的资源占用
    转换为 Resource.unavailable 区间，避免新的 solver 产生冲突。
    """
    import copy as _copy
    augmented = _copy.deepcopy(resources)

    if not old_plan:
        return augmented

    res_map = {r.resource_id: r for r in augmented}

    for assign in old_plan.op_assignments:
        # 跳过已完成的 ops（资源已释放）
        if assign.op_id in completed_ops:
            continue
        # 只阻断未完成但属于旧计划的 ops
        if assign.end_slot <= now:
            continue
        actual_start = max(now, assign.start_slot)
        for res_id in assign.resources:
            if res_id in res_map:
                res_map[res_id].unavailable.append((actual_start, assign.end_slot))

    return augmented


def _merge_old_and_new_plan(
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    state: SimulationStateOps,
    now: int,
) -> PlanV2_1:
    """
    合并旧计划（活跃部分）和新增 mission 的计划。

    - 旧计划中未完成的 assignments 全部保留
    - 新计划的 assignments 追加
    """
    merged_assignments: List[OpAssignment] = []
    seen_op_ids: Set[str] = set()

    if old_plan:
        for assign in old_plan.op_assignments:
            # 保留所有未来仍需执行的 assignments
            if assign.op_id not in state.completed_ops or assign.end_slot > now:
                merged_assignments.append(assign)
                seen_op_ids.add(assign.op_id)

    for assign in new_plan.op_assignments:
        if assign.op_id not in seen_op_ids:
            merged_assignments.append(assign)
            seen_op_ids.add(assign.op_id)

    return PlanV2_1(timestamp=now, op_assignments=merged_assignments)


def _inject_preserved_assignments(
    plan: PlanV2_1,
    preserved: List[OpAssignment],
) -> PlanV2_1:
    """
    将已 started/completed 的 assignments 注入到新计划中（如果缺失）。
    """
    existing_ids = {a.op_id for a in plan.op_assignments}
    merged = list(plan.op_assignments)
    for assign in preserved:
        if assign.op_id not in existing_ids:
            merged.append(assign)
    return PlanV2_1(timestamp=plan.timestamp, op_assignments=merged)


def simulate_episode(
    policy: BasePolicy,
    scenario: Scenario,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False
) -> EpisodeResult:
    return _simulate_episode_v2_1(policy, scenario, config, verbose)


def _simulate_episode_v2_1(
    policy: BasePolicy,
    scenario: Scenario,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False
) -> EpisodeResult:
    start_time = time.time()

    policy.reset()

    state = _create_initial_state_ops(scenario.missions, scenario.resources)

    snapshots: List[RollingSnapshot] = []
    rolling_metrics_list: List[RollingMetrics] = []
    feature_history = []
    prev_window_slots = None

    executed_assignments: Dict[str, OpAssignment] = {}

    sim_total = config.sim_total_slots
    rolling_interval = config.rolling_interval
    horizon = config.horizon_slots

    now = 0
    last_now = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f" Episode Simulation: seed={scenario.seed}, policy={policy.name}")
        print(f" Total slots: {sim_total}, Rolling interval: {rolling_interval}")
        print(f"{'='*60}")

    while now < sim_total:
        if verbose:
            print(f"\n--- Rolling at t={now} ---")

        state = _apply_disturbance_ops(
            state, now, scenario.disturbance_timeline, scenario, last_now
        )

        state = _update_op_status(state, now)
        state.now = now

        if state.current_plan:
            for assign in state.current_plan.op_assignments:
                if assign.op_id in state.completed_ops and assign.op_id not in executed_assignments:
                    executed_assignments[assign.op_id] = assign

        is_feasible, infeasible_reasons = _check_plan_feasibility_ops(state, now)
        forced_replan = not is_feasible

        if verbose and forced_replan:
            print(f"   Plan infeasible: {infeasible_reasons[:2]}")

        meta_params, direct_plan = policy.decide(state, now, config)

        if meta_params:
            freeze_h = meta_params.freeze_horizon if meta_params.freeze_horizon is not None else config.freeze_horizon
            weights = meta_params.to_weights()
            use_two_stage = config.use_two_stage_solver if meta_params.use_two_stage is None else meta_params.use_two_stage
            epsilon_solver = meta_params.epsilon_solver if meta_params.epsilon_solver is not None else config.default_epsilon_solver
            kappa_win = meta_params.kappa_win if meta_params.kappa_win is not None else config.default_kappa_win
            kappa_seq = meta_params.kappa_seq if meta_params.kappa_seq is not None else config.default_kappa_seq
        else:
            freeze_h = config.freeze_horizon
            weights = (config.default_w_delay, config.default_w_shift, config.default_w_switch)
            use_two_stage = config.use_two_stage_solver
            epsilon_solver = config.default_epsilon_solver
            kappa_win = config.default_kappa_win
            kappa_seq = config.default_kappa_seq

        horizon_end = min(now + horizon, sim_total)
        solver_horizon = sim_total
        missions_to_schedule = state.get_schedulable_missions(horizon_end)

        if state.current_plan:
            for assign in state.current_plan.op_assignments:
                if assign.op_id in state.started_ops and assign.op_id not in state.completed_ops:
                    mission = state.get_mission(assign.mission_id)
                    if mission and mission not in missions_to_schedule:
                        missions_to_schedule.append(mission)

        frozen_ops = compute_frozen_ops(
            state.current_plan,
            now,
            freeze_h,
            state.started_ops,
            state.completed_ops
        )

        if direct_plan and hasattr(direct_plan, 'op_assignments'):
            result = SolverResult(
                status=SolveStatus.OPTIMAL,
                plan=direct_plan,
                objective_value=0.0,
                solve_time_ms=0
            )
            # direct_plan 时也回写日志（无 solver 调用）
            if hasattr(policy, 'update_pending_log_with_solver_result'):
                policy.update_pending_log_with_solver_result(solver_result=result)
        else:
            _trcg_chain_result = None  # FallbackChainResult，用于回写日志
            solver_config = SolverConfigV2_1(
                horizon_slots=solver_horizon,
                w_delay=weights[0],
                w_shift=weights[1],
                w_switch=weights[2],
                time_limit_seconds=config.solver_timeout_s,
                num_workers=config.solver_num_workers,
                op5_max_wait_slots=max(
                    0,
                    int(round(config.op5_max_wait_hours * 60 / config.slot_minutes))
                ),
                use_two_stage=use_two_stage,
                epsilon_solver=epsilon_solver,
                kappa_win=kappa_win,
                kappa_seq=kappa_seq,
                stage1_time_ratio=config.stage1_time_ratio
            )
            
            # V2.5: 在求解前，根据 range_calendar 过滤 Launch 候选窗口
            if hasattr(scenario, 'range_calendar') and scenario.range_calendar:
                for mission in missions_to_schedule:
                    launch = mission.get_launch_op()
                    if launch and launch.time_windows:
                        candidate_windows = _compute_op6_candidate_windows(
                            launch.time_windows,
                            scenario.range_calendar,
                            launch.duration,
                            sim_total
                        )
                        if candidate_windows:
                            launch.time_windows = candidate_windows
                        # 如果候选窗口为空，保留原窗口（避免 infeasible）

            # 提取 TRCG unlock 集合（若策略提供）
            unlock_set = None
            if meta_params and meta_params.unlock_mission_ids is not None:
                unlock_set = set(meta_params.unlock_mission_ids)

            result = solve_v2_1(
                missions=missions_to_schedule,
                resources=state.resources,
                horizon=solver_horizon,
                prev_plan=state.current_plan,
                frozen_ops=frozen_ops,
                config=solver_config,
                unlock_mission_ids=unlock_set,
                now=now,
            )

            # ---- TRCG Anchor Quality Guard ----
            # 当 anchor 求解的 drift 极端异常时才回退，大幅提高阈值以保留锚点效果。
            # 软锚点（weight=500）已经在 solver 内部平衡了 drift vs anchor 约束，
            # 不需要外部 guard 频繁干预。
            _ANCHOR_DRIFT_GUARD_THRESHOLD = 2000.0  # 大幅提高阈值，仅在极端情况触发
            if (unlock_set is not None
                    and result.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
                    and result.anchor_fix_applied_count > 0
                    and state.current_plan is not None):
                _guard_drift_anchor, *_ = compute_plan_drift_ops(
                    state.current_plan, result.plan,
                    state.completed_ops, state.started_ops,
                    set(frozen_ops.keys()),
                    state.missions,
                    kappa_win=kappa_win, kappa_seq=kappa_seq,
                )
                if _guard_drift_anchor > _ANCHOR_DRIFT_GUARD_THRESHOLD:
                    _result_free = solve_v2_1(
                        missions=missions_to_schedule,
                        resources=state.resources,
                        horizon=solver_horizon,
                        prev_plan=state.current_plan,
                        frozen_ops=frozen_ops,
                        config=solver_config,
                        unlock_mission_ids=None,
                        now=now,
                    )
                    if _result_free.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
                        _guard_drift_free, *_ = compute_plan_drift_ops(
                            state.current_plan, _result_free.plan,
                            state.completed_ops, state.started_ops,
                            set(frozen_ops.keys()),
                            state.missions,
                            kappa_win=kappa_win, kappa_seq=kappa_seq,
                        )
                        if _guard_drift_free < _guard_drift_anchor:
                            result = _result_free
                            if meta_params:
                                meta_params = dataclasses.replace(
                                    meta_params,
                                    unlock_mission_ids=None,
                                    decision_source=meta_params.decision_source + "_guard_free",
                                )

            # ---- TRCG Repair 回退链：若 policy 提供了 unlock_mission_ids 且初次求解失败 ----
            # 注意：仅在 TIMEOUT 时触发 TRCG 回退（换配置可能帮助），
            # INFEASIBLE 意味着结构性不可行（frozen_ops 冲突等），
            # TRCG 回退链无法解决 → 直接走通用回退（会移除 frozen_ops）。
            _trcg_chain_result = None  # FallbackChainResult，用于回写日志
            _trcg_should_fallback = (
                meta_params
                and meta_params.unlock_mission_ids is not None
                and result.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
                and result.status != SolveStatus.INFEASIBLE  # 结构性不可行直接走通用回退
            )
            if _trcg_should_fallback:
                result, meta_params, _trcg_chain_result = _solve_with_trcg_fallback(
                    initial_result=result,
                    meta_params=meta_params,
                    state=state,
                    now=now,
                    config=config,
                    missions=missions_to_schedule,
                    resources=state.resources,
                    horizon=solver_horizon,
                    prev_plan=state.current_plan,
                    frozen_ops=frozen_ops,
                    solver_config_base=solver_config,
                    scenario=scenario,
                    verbose=verbose,
                )

            # ---- INFEASIBLE/TIMEOUT 通用回退：增量求解 + 宽松求解 + Greedy 兜底 ----
            if result.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
                if verbose:
                    print(f"   [t={now}] Solver {result.status} → entering fallback chain")
                result = _fallback_on_infeasible(
                    state=state,
                    now=now,
                    config=config,
                    missions_to_schedule=missions_to_schedule,
                    solver_config=solver_config,
                    solver_horizon=solver_horizon,
                    frozen_ops=frozen_ops,
                    scenario=scenario,
                    verbose=verbose,
                )

            # ---- 回写 TRCG repair step log（solver 结果填充） ----
            if hasattr(policy, 'update_pending_log_with_solver_result'):
                policy.update_pending_log_with_solver_result(
                    solver_result=result,
                    chain_result=_trcg_chain_result,
                )

        old_plan = state.current_plan

        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            state.current_plan = result.plan
            if state.current_plan:
                state.current_plan.timestamp = now
            solve_status = result.status
        else:
            if verbose:
                print(f"   Solver returned {result.status}, keeping old plan")
            solve_status = result.status

        rolling_m = compute_rolling_metrics_ops(
            t=now,
            old_plan=old_plan,
            new_plan=state.current_plan,
            completed_ops=state.completed_ops,
            started_ops=state.started_ops,
            frozen_ops=set(frozen_ops.keys()),
            missions=state.missions,
            solve_time_ms=result.solve_time_ms,
            is_feasible=is_feasible,
            forced_replan=forced_replan,
            frozen_count=len(frozen_ops),
            kappa_win=kappa_win,
            kappa_seq=kappa_seq
        )

        rolling_metrics_list.append(rolling_m)

        state_features = None
        try:
            features, prev_window_slots = compute_state_features_ops(
                missions=state.missions,
                resources=state.resources,
                current_plan=state.current_plan,
                now=now,
                config=config,
                completed_ops=state.completed_ops,
                prev_window_slots=prev_window_slots,
                recent_shifts=rolling_m.num_shifts,
                recent_switches=rolling_m.num_switches,
                history=feature_history
            )
            state_features = features.to_dict()
            feature_history.append(features)
        except Exception:
            state_features = None

        snapshot = RollingSnapshot(
            t=now,
            plan=copy.deepcopy(state.current_plan),
            solve_status=solve_status,
            solve_time_ms=result.solve_time_ms,
            metrics=rolling_m,
            meta_params=meta_params,
            infeasible_reasons=infeasible_reasons if forced_replan else [],
            state_features=state_features
        )
        snapshots.append(snapshot)

        if verbose:
            print(f"  Status: {solve_status.value}, Time: {result.solve_time_ms}ms")
            print(f"  Ops scheduled: {rolling_m.num_tasks_scheduled}, Frozen: {rolling_m.num_frozen}")
            print(f"  Drift: {rolling_m.plan_drift:.4f}, Shifts: {rolling_m.num_shifts}, Switches: {rolling_m.num_switches}")

        last_now = now
        now += rolling_interval

        completed_missions = set()
        for m in state.missions:
            _launch = m.get_launch_op()
            if _launch and _launch.op_id in state.completed_ops:
                completed_missions.add(m.mission_id)
        if len(completed_missions) >= len(state.missions):
            if verbose:
                print(f"\n All missions completed at t={now}")
            break

    final_schedule = list(executed_assignments.values())
    if state.current_plan:
        for assign in state.current_plan.op_assignments:
            if assign.op_id not in executed_assignments:
                final_schedule.append(assign)

    completed_missions = set()
    for m in state.missions:
        _launch = m.get_launch_op()
        if _launch and _launch.op_id in state.completed_ops:
            completed_missions.add(m.mission_id)

    episode_metrics = compute_episode_metrics_ops(
        rolling_metrics_list=rolling_metrics_list,
        final_assignments=final_schedule,
        missions=state.missions,
        completed_mission_ids=completed_missions,
        resources=state.resources,
        horizon_slots=sim_total,
        slot_minutes=config.slot_minutes
    )

    total_runtime = time.time() - start_time

    all_mission_ids = {m.mission_id for m in state.missions}
    uncompleted = all_mission_ids - completed_missions

    result = EpisodeResult(
        seed=scenario.seed,
        policy_name=policy.name,
        snapshots=snapshots,
        metrics=episode_metrics,
        final_schedule=final_schedule,
        completed_tasks=completed_missions,
        uncompleted_tasks=uncompleted,
        total_runtime_s=total_runtime
    )

    if verbose:
        print(f"\n{'='*60}")
        print(" Episode Complete")
        print(f"{'='*60}")
        print(f"  Runtime: {total_runtime:.2f}s")
        print(f"  Completed: {episode_metrics.num_completed}/{episode_metrics.num_total}")
        print(f"  On-time rate: {episode_metrics.on_time_rate:.2%}")
        print(f"  Drift/replan: {episode_metrics.drift_per_replan:.6f}  (main)")
        print(f"  Episode drift: {episode_metrics.episode_drift:.4f}  (cumulative)")
        print(f"  Total shifts: {episode_metrics.total_shifts}")
        print(f"  Total switches: {episode_metrics.total_switches}")

    return result


# ============================================================================
# Logs
# ============================================================================

def save_episode_logs(
    result: EpisodeResult,
    output_dir: str,
    scenario: Optional[Scenario] = None
) -> Dict[str, str]:
    """
    保存 Episode 日志到文件
    
    Args:
        result: Episode 结果
        output_dir: 输出目录
        scenario: 场景（可选，用于保存场景信息）
    
    Returns:
        保存的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # 1. 保存 rolling log (JSONL)
    rolling_log_path = os.path.join(output_dir, "rolling_log.jsonl")
    with open(rolling_log_path, 'w', encoding='utf-8') as f:
        for snapshot in result.snapshots:
            f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + '\n')
    saved_files["rolling_log"] = rolling_log_path
    
    # 2. 保存 metrics CSV
    metrics_csv_path = os.path.join(output_dir, "metrics_per_roll.csv")
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "t", "plan_drift", "avg_time_shift_slots", "num_shifts", "num_switches",
            "num_window_switches", "num_sequence_switches",
            "num_tasks_scheduled", "num_frozen", "solve_time_ms",
            "is_feasible", "forced_replan", "num_active_missions"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in result.snapshots:
            writer.writerow(rolling_metrics_to_dict(snapshot.metrics))
    saved_files["metrics_csv"] = metrics_csv_path
    
    # 3. 保存最终排程
    final_schedule_path = os.path.join(output_dir, "final_schedule.json")
    with open(final_schedule_path, 'w', encoding='utf-8') as f:
        json.dump({
            "seed": result.seed,
            "policy": result.policy_name,
            "schema_version": scenario.schema_version if scenario else "v2_1",
            "schedule": [assignment_to_dict(a) for a in result.final_schedule],
            "completed": list(result.completed_tasks),
            "uncompleted": list(result.uncompleted_tasks)
        }, f, indent=2, ensure_ascii=False)
    saved_files["final_schedule"] = final_schedule_path
    
    # 4. 保存 Episode 汇总
    summary_path = os.path.join(output_dir, "episode_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    saved_files["summary"] = summary_path
    
    # 5. 保存场景（如果提供）
    if scenario:
        scenario_path = os.path.join(output_dir, "scenario.json")
        with open(scenario_path, 'w', encoding='utf-8') as f:
            json.dump(scenario.to_dict(), f, indent=2, ensure_ascii=False)
        saved_files["scenario"] = scenario_path
    
    return saved_files


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    from scenario import generate_scenario
    from policies.policy_fixed import FixedWeightPolicy
    
    print("=== Simulator Test ===\n")
    
    # 生成场景
    scenario = generate_scenario(seed=42)
    print(f"Generated scenario: {len(scenario.missions)} missions, "
          f"{len(scenario.disturbance_timeline)} disturbances")
    
    # 创建策略
    policy = FixedWeightPolicy(w_delay=10.0, w_shift=1.0, w_switch=5.0)
    
    # 运行仿真
    result = simulate_episode(policy, scenario, verbose=True)
    
    # 打印结果
    print("\n" + "="*60)
    print(" Final Metrics")
    print("="*60)
    for key, value in metrics_to_dict(result.metrics).items():
        print(f"  {key}: {value}")
