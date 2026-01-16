"""
仿真器模块 - Rolling Horizon 仿真主循环
"""

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
    SolverResult, SolveStatus,
    solve, compute_frozen_tasks
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
    metrics_to_dict, rolling_metrics_to_dict
)
from policies.base import BasePolicy, MetaParams


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RollingSnapshot:
    """单次 Rolling 的快照"""
    t: int                                        # 时刻
    plan: Optional[Plan]                          # 计划
    solve_status: SolveStatus                     # 求解状态
    solve_time_ms: int                            # 求解时间
    metrics: RollingMetrics                       # 指标
    meta_params: Optional[MetaParams] = None      # 策略元参数
    infeasible_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "solve_status": self.solve_status.value,
            "solve_time_ms": self.solve_time_ms,
            "plan": self.plan.to_dict() if self.plan else None,
            "metrics": rolling_metrics_to_dict(self.metrics),
            "meta_params": {
                "w_delay": self.meta_params.w_delay,
                "w_shift": self.meta_params.w_shift,
                "w_switch": self.meta_params.w_switch,
                "freeze_horizon": self.meta_params.freeze_horizon
            } if self.meta_params else None,
            "infeasible_reasons": self.infeasible_reasons
        }


@dataclass
class EpisodeResult:
    """单次仿真结果"""
    seed: int
    policy_name: str
    snapshots: List[RollingSnapshot]
    metrics: EpisodeMetrics
    final_schedule: List[TaskAssignment]
    completed_tasks: Set[str]
    uncompleted_tasks: Set[str]
    total_runtime_s: float
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "policy_name": self.policy_name,
            "metrics": metrics_to_dict(self.metrics),
            "final_schedule": [
                {
                    "task_id": a.task_id,
                    "pad_id": a.pad_id,
                    "launch_slot": a.launch_slot,
                    "start_slot": a.start_slot
                }
                for a in self.final_schedule
            ],
            "completed_tasks": list(self.completed_tasks),
            "uncompleted_tasks": list(self.uncompleted_tasks),
            "total_runtime_s": round(self.total_runtime_s, 3),
            "num_snapshots": len(self.snapshots)
        }


# ============================================================================
# 仿真主循环
# ============================================================================

def simulate_episode(
    policy: BasePolicy,
    scenario: Scenario,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False
) -> EpisodeResult:
    """
    运行单个 Episode 的完整仿真
    
    Args:
        policy: 策略对象
        scenario: 场景
        config: 配置
        verbose: 是否输出详细信息
    
    Returns:
        EpisodeResult
    """
    start_time = time.time()
    
    # 重置策略
    policy.reset()
    
    # 初始化状态
    state = create_initial_state(scenario.tasks, scenario.pads)
    
    # 记录
    snapshots: List[RollingSnapshot] = []
    rolling_metrics_list: List[RollingMetrics] = []
    
    # 追踪已执行的任务（用于最终排程）
    executed_assignments: Dict[str, TaskAssignment] = {}
    
    # 仿真参数
    sim_total = config.sim_total_slots
    rolling_interval = config.rolling_interval
    horizon = config.horizon_slots
    
    # 时间推进
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
        
        # 1. 应用扰动
        state = apply_disturbance(
            state, now, scenario.disturbance_timeline, last_now
        )
        
        # 2. 更新任务状态
        state = update_task_status(state, now)
        state.now = now
        
        # 记录已完成任务的执行信息
        if state.current_plan:
            for assign in state.current_plan.assignments:
                if assign.task_id in state.completed_tasks and assign.task_id not in executed_assignments:
                    executed_assignments[assign.task_id] = assign
        
        # 3. 检查旧计划可行性
        is_feasible, infeasible_reasons = check_plan_feasibility(state, now)
        forced_replan = not is_feasible
        
        if verbose and forced_replan:
            print(f"  ⚠ Plan infeasible: {infeasible_reasons[:2]}")
        
        # 4. 调用策略获取元参数
        meta_params, direct_plan = policy.decide(state, now, config)
        
        # 5. 确定冻结视野和权重
        if meta_params:
            freeze_h = meta_params.freeze_horizon if meta_params.freeze_horizon is not None else config.freeze_horizon
            weights = meta_params.to_weights()
        else:
            freeze_h = config.freeze_horizon
            weights = (config.default_w_delay, config.default_w_shift, config.default_w_switch)
        
        # 6. 获取需要排程的任务
        horizon_end = now + horizon
        tasks_to_schedule = state.get_schedulable_tasks(horizon_end)
        
        # 添加已开始但未完成的任务（它们需要保持在计划中）
        for assign in (state.current_plan.assignments if state.current_plan else []):
            if assign.task_id in state.started_tasks and assign.task_id not in state.completed_tasks:
                task = state.get_task(assign.task_id)
                if task and task not in tasks_to_schedule:
                    tasks_to_schedule.append(task)
        
        # 7. 求解
        if direct_plan:
            # 策略直接给出计划（如贪心）
            result = SolverResult(
                status=SolveStatus.OPTIMAL,
                plan=direct_plan,
                objective_value=0.0,
                solve_time_ms=0
            )
        else:
            # 使用 CP-SAT 求解
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
        
        # 8. 处理结果
        old_plan = state.current_plan
        
        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            state.current_plan = result.plan
            solve_status = result.status
        else:
            # 不可行或超时，保持旧计划
            if verbose:
                print(f"  ⚠ Solver returned {result.status}, keeping old plan")
            solve_status = result.status
            # 如果完全没有计划，强制重排（简化处理：保持原计划）
        
        # 9. 计算指标
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
        
        # 10. 记录快照
        snapshot = RollingSnapshot(
            t=now,
            plan=copy.deepcopy(state.current_plan),
            solve_status=solve_status,
            solve_time_ms=result.solve_time_ms,
            metrics=rolling_m,
            meta_params=meta_params,
            infeasible_reasons=infeasible_reasons if forced_replan else []
        )
        snapshots.append(snapshot)
        
        if verbose:
            print(f"  Status: {solve_status.value}, Time: {result.solve_time_ms}ms")
            print(f"  Tasks scheduled: {rolling_m.num_tasks_scheduled}, "
                  f"Frozen: {rolling_m.num_frozen}")
            print(f"  Drift: {rolling_m.plan_drift:.4f}, "
                  f"Shifts: {rolling_m.num_shifts}, Switches: {rolling_m.num_switches}")
            print(f"  Completed: {len(state.completed_tasks)}/{len(state.tasks)}")
        
        # 11. 时间推进
        last_now = now
        now += rolling_interval
        
        # 检查是否所有任务完成
        if len(state.completed_tasks) >= len(state.tasks):
            if verbose:
                print(f"\n✓ All tasks completed at t={now}")
            break
    
    # 收集最终结果 - 使用执行记录，而非最终状态的计划
    final_schedule = list(executed_assignments.values())
    
    # 添加当前计划中未完成的任务
    if state.current_plan:
        for assign in state.current_plan.assignments:
            if assign.task_id not in executed_assignments:
                final_schedule.append(assign)
    
    # 计算 Episode 指标
    episode_metrics = compute_episode_metrics(
        rolling_metrics_list=rolling_metrics_list,
        final_assignments=final_schedule,
        tasks=scenario.tasks,
        completed_task_ids=state.completed_tasks
    )
    
    total_runtime = time.time() - start_time
    
    # 未完成任务
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
        print(f"\n{'='*60}")
        print(f" Episode Complete")
        print(f"{'='*60}")
        print(f"  Runtime: {total_runtime:.2f}s")
        print(f"  Completed: {episode_metrics.num_completed}/{episode_metrics.num_total}")
        print(f"  On-time rate: {episode_metrics.on_time_rate:.2%}")
        print(f"  Episode drift: {episode_metrics.episode_drift:.4f}")
        print(f"  Total shifts: {episode_metrics.total_shifts}")
        print(f"  Total switches: {episode_metrics.total_switches}")
    
    return result


# ============================================================================
# 日志保存
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
            "schedule": [
                {
                    "task_id": a.task_id,
                    "pad_id": a.pad_id,
                    "start_slot": a.start_slot,
                    "launch_slot": a.launch_slot
                }
                for a in result.final_schedule
            ],
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
