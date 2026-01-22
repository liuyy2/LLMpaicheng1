"""
仿真器模块 - Rolling Horizon 仿真主循环
"""

from __future__ import annotations

import time
import copy
import json
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any

from config import Config, DEFAULT_CONFIG
from solver_cpsat import (
    Task, Pad, Plan, TaskAssignment,
    Mission, Operation, Resource, PlanV2_1, OpAssignment,
    SolverResult, SolveStatus,
    solve, compute_frozen_tasks,
    solve_v2_1, compute_frozen_ops, SolverConfigV2_1
)
from scenario import Scenario, DisturbanceEvent
from disturbance import (
    SimulationState, 
    apply_disturbance, 
    update_task_status,
    get_frozen_assignments,
    check_plan_feasibility,
    create_initial_state
)
from metrics import (
    RollingMetrics, EpisodeMetrics,
    compute_rolling_metrics, compute_episode_metrics,
    compute_rolling_metrics_ops, compute_episode_metrics_ops,
    metrics_to_dict, rolling_metrics_to_dict
)
from features import compute_state_features, compute_state_features_ops
from policies.base import BasePolicy, MetaParams




def assignment_to_dict(assign: Any) -> Dict[str, Any]:
    if hasattr(assign, 'task_id'):
        return {
            'task_id': assign.task_id,
            'pad_id': assign.pad_id,
            'start_slot': assign.start_slot,
            'launch_slot': assign.launch_slot,
            'end_slot': assign.end_slot
        }

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
        return {
            't': self.t,
            'solve_status': self.solve_status.value,
            'solve_time_ms': self.solve_time_ms,
            'plan': self.plan.to_dict() if self.plan else None,
            'metrics': rolling_metrics_to_dict(self.metrics),
            'meta_params': {
                'w_delay': self.meta_params.w_delay,
                'w_shift': self.meta_params.w_shift,
                'w_switch': self.meta_params.w_switch,
                'freeze_horizon': self.meta_params.freeze_horizon
            } if self.meta_params else None,
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
    actual_releases: Dict[str, int] = field(default_factory=dict)

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
        op6 = mission.get_operation(6)
        return bool(op6 and op6.op_id in self.completed_ops)

    def get_schedulable_missions(self, horizon_end: int) -> List[Mission]:
        schedulable = []
        for mission in self.missions:
            if self.is_mission_completed(mission):
                continue
            actual_rel = self.actual_releases.get(mission.mission_id, mission.release)
            if actual_rel <= horizon_end:
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


def _apply_weather_disturbance_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    params = event.params
    delete_ratio = params.get('delete_ratio', 0.3)
    affected_start = params.get('affected_start', event.trigger_time)
    affected_end = params.get('affected_end', event.trigger_time + 12)

    for mission in state.missions:
        op6 = mission.get_operation(6)
        if not op6:
            continue
        if op6.op_id in state.completed_ops or op6.op_id in state.started_ops:
            continue

        new_windows = []
        for win_start, win_end in op6.time_windows:
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
            op6.time_windows = _merge_windows(new_windows)
        if not op6.time_windows:
            op6.time_windows = [(affected_end + 1, affected_end + 30)]


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
        if op.op_index in (5, 6):
            return
        new_duration = max(1, int(round(op.duration * multiplier)))
        state.actual_durations[op_id] = new_duration
        op.duration = new_duration


def _apply_release_disturbance_ops(state: SimulationStateOps, event: DisturbanceEvent) -> None:
    mission_id = event.target_id
    params = event.params
    new_release = params.get('new_release', 0)

    mission = state.get_mission(mission_id)
    if mission:
        mission.release = new_release
        state.actual_releases[mission_id] = new_release
        for op in mission.operations:
            op.release = new_release


def _apply_disturbance_ops(
    state: SimulationStateOps,
    now: int,
    events: List[DisturbanceEvent],
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
        elif event.event_type == 'release':
            _apply_release_disturbance_ops(state, event)

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

            if op.op_index != 5 and assign.end_slot - assign.start_slot != op.duration:
                infeasible.append(f'duration_{op.op_id}')
                return False, infeasible

            if op.op_index == 6 and op.time_windows:
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


def simulate_episode(
    policy: BasePolicy,
    scenario: Scenario,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False
) -> EpisodeResult:
    if getattr(scenario, 'schema_version', 'v1') == 'v2_1':
        return _simulate_episode_v2_1(policy, scenario, config, verbose)
    return _simulate_episode_v1(policy, scenario, config, verbose)


def _simulate_episode_v1(
    policy: BasePolicy,
    scenario: Scenario,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False
) -> EpisodeResult:
    start_time = time.time()

    policy.reset()

    state = create_initial_state(scenario.tasks, scenario.pads)

    snapshots: List[RollingSnapshot] = []
    rolling_metrics_list: List[RollingMetrics] = []
    prev_window_slots = None
    prev_window_slots = None

    executed_assignments: Dict[str, TaskAssignment] = {}

    sim_total = config.sim_total_slots
    rolling_interval = config.rolling_interval
    horizon = config.horizon_slots

    now = 0
    last_now = 0

    if verbose:
        print("\n" + "=" * 60)
        print(f" Episode Simulation: seed={scenario.seed}, policy={policy.name}")
        print(f" Total slots: {sim_total}, Rolling interval: {rolling_interval}")
        print("=" * 60)

    while now < sim_total:
        if verbose:
            print(f"\n--- Rolling at t={now} ---")

        state = apply_disturbance(state, now, scenario.disturbance_timeline, last_now)

        state = update_task_status(state, now)
        state.now = now

        if state.current_plan:
            for assign in state.current_plan.assignments:
                if assign.task_id in state.completed_tasks and assign.task_id not in executed_assignments:
                    executed_assignments[assign.task_id] = assign

        is_feasible, infeasible_reasons = check_plan_feasibility(state, now)
        forced_replan = not is_feasible

        if verbose and forced_replan:
            print(f"  Plan infeasible: {infeasible_reasons[:2]}")

        meta_params, direct_plan = policy.decide(state, now, config)

        if meta_params:
            freeze_h = meta_params.freeze_horizon if meta_params.freeze_horizon is not None else config.freeze_horizon
            weights = meta_params.to_weights()
        else:
            freeze_h = config.freeze_horizon
            weights = (config.default_w_delay, config.default_w_shift, config.default_w_switch)

        horizon_end = now + horizon
        tasks_to_schedule = state.get_schedulable_tasks(horizon_end)

        if state.current_plan:
            for assign in state.current_plan.assignments:
                if assign.task_id in state.started_tasks and assign.task_id not in state.completed_tasks:
                    task = state.get_task(assign.task_id)
                    if task and task not in tasks_to_schedule:
                        tasks_to_schedule.append(task)

        if direct_plan:
            result = SolverResult(
                status=SolveStatus.OPTIMAL,
                plan=direct_plan,
                objective_value=0.0,
                solve_time_ms=0
            )
        else:
            result = solve(
                now=now,
                tasks=tasks_to_schedule,
                pads=state.pads,
                prev_plan=state.current_plan,
                freeze_horizon=freeze_h,
                weights=weights,
                time_limit=config.solver_timeout_s,
                completed_tasks=state.completed_tasks
            )

        old_plan = state.current_plan

        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            state.current_plan = result.plan
            if state.current_plan:
                state.current_plan.timestamp = now
            solve_status = result.status
        else:
            if verbose:
                print(f"  Solver returned {result.status}, keeping old plan")
            solve_status = result.status

        frozen_count = len(get_frozen_assignments(state, now, freeze_h))

        rolling_m = compute_rolling_metrics(
            t=now,
            old_plan=old_plan,
            new_plan=state.current_plan,
            completed_tasks=state.completed_tasks,
            horizon=horizon,
            solve_time_ms=result.solve_time_ms,
            is_feasible=is_feasible,
            forced_replan=forced_replan,
            frozen_count=frozen_count,
            alpha=config.drift_alpha,
            beta=config.drift_beta
        )

        rolling_metrics_list.append(rolling_m)

        state_features = None
        try:
            features, prev_window_slots = compute_state_features(
                tasks=state.tasks,
                pads=state.pads,
                current_plan=state.current_plan,
                now=now,
                config=config,
                completed_tasks=state.completed_tasks,
                prev_window_slots=prev_window_slots,
                recent_shifts=rolling_m.num_shifts,
                recent_switches=rolling_m.num_switches
            )
            state_features = features.to_dict()
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
            print(f"  Tasks scheduled: {rolling_m.num_tasks_scheduled}, Frozen: {rolling_m.num_frozen}")
            print(f"  Drift: {rolling_m.plan_drift:.4f}, Shifts: {rolling_m.num_shifts}, Switches: {rolling_m.num_switches}")
            print(f"  Completed: {len(state.completed_tasks)}/{len(state.tasks)}")

        last_now = now
        now += rolling_interval

        if len(state.completed_tasks) >= len(state.tasks):
            if verbose:
                print(f"\nAll tasks completed at t={now}")
            break

    final_schedule = list(executed_assignments.values())
    if state.current_plan:
        for assign in state.current_plan.assignments:
            if assign.task_id not in executed_assignments:
                final_schedule.append(assign)

    episode_metrics = compute_episode_metrics(
        rolling_metrics_list=rolling_metrics_list,
        final_assignments=final_schedule,
        tasks=scenario.tasks,
        completed_task_ids=state.completed_tasks,
        pad_count=len(scenario.pads),
        horizon_slots=sim_total
    )

    total_runtime = time.time() - start_time

    all_task_ids = {t.task_id for t in scenario.tasks}
    uncompleted = all_task_ids - state.completed_tasks

    result = EpisodeResult(
        seed=scenario.seed,
        policy_name=policy.name,
        snapshots=snapshots,
        metrics=episode_metrics,
        final_schedule=final_schedule,
        completed_tasks=state.completed_tasks,
        uncompleted_tasks=uncompleted,
        total_runtime_s=total_runtime
    )

    if verbose:
        print("\n" + "=" * 60)
        print(" Episode Complete")
        print("=" * 60)
        print(f"  Runtime: {total_runtime:.2f}s")
        print(f"  Completed: {episode_metrics.num_completed}/{episode_metrics.num_total}")
        print(f"  On-time rate: {episode_metrics.on_time_rate:.2%}")
        print(f"  Episode drift: {episode_metrics.episode_drift:.4f}")
        print(f"  Total shifts: {episode_metrics.total_shifts}")
        print(f"  Total switches: {episode_metrics.total_switches}")

    return result


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
            state, now, scenario.disturbance_timeline, last_now
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
        else:
            freeze_h = config.freeze_horizon
            weights = (config.default_w_delay, config.default_w_shift, config.default_w_switch)

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
        else:
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
                )
            )

            result = solve_v2_1(
                missions=missions_to_schedule,
                resources=state.resources,
                horizon=solver_horizon,
                prev_plan=state.current_plan,
                frozen_ops=frozen_ops,
                config=solver_config
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
            horizon=horizon,
            solve_time_ms=result.solve_time_ms,
            is_feasible=is_feasible,
            forced_replan=forced_replan,
            frozen_count=len(frozen_ops),
            alpha=config.drift_alpha,
            beta=config.drift_beta
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
                recent_switches=rolling_m.num_switches
            )
            state_features = features.to_dict()
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

        completed_missions = {
            m.mission_id
            for m in state.missions
            if m.get_operation(6) and m.get_operation(6).op_id in state.completed_ops
        }
        if len(completed_missions) >= len(state.missions):
            if verbose:
                print(f"\n All missions completed at t={now}")
            break

    final_schedule = list(executed_assignments.values())
    if state.current_plan:
        for assign in state.current_plan.op_assignments:
            if assign.op_id not in executed_assignments:
                final_schedule.append(assign)

    completed_missions = {
        m.mission_id
        for m in state.missions
        if m.get_operation(6) and m.get_operation(6).op_id in state.completed_ops
    }

    episode_metrics = compute_episode_metrics_ops(
        rolling_metrics_list=rolling_metrics_list,
        final_assignments=final_schedule,
        missions=state.missions,
        completed_mission_ids=completed_missions,
        resources=state.resources,
        horizon_slots=sim_total
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
        print(f"  Episode drift: {episode_metrics.episode_drift:.4f}")
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
            "t", "plan_drift", "num_shifts", "num_switches",
            "num_tasks_scheduled", "num_frozen", "solve_time_ms",
            "is_feasible", "forced_replan"
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
            "schema_version": scenario.schema_version if scenario else "v1",
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
    print(f"Generated scenario: {len(scenario.tasks)} tasks, "
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
