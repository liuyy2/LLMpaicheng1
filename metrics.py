"""
指标计算模块 - 稳定性与性能指标
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import math

from solver_cpsat import Plan, TaskAssignment, Task


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RollingMetrics:
    """单次 Rolling 的指标"""
    t: int                                    # 时刻
    plan_drift: float                         # PlanDrift_t
    num_shifts: int                           # 时间变化任务数
    num_switches: int                         # Pad 切换任务数
    num_tasks_scheduled: int                  # 排程任务数
    num_frozen: int                           # 冻结任务数
    solve_time_ms: int                        # 求解时间
    is_feasible: bool                         # 是否可行
    forced_replan: bool                       # 是否强制重排


@dataclass
class EpisodeMetrics:
    """Episode 总指标"""
    # 延迟相关
    on_time_rate: float                       # 按期发射率
    total_delay: int                          # 总延迟 (slots)
    avg_delay: float                          # 平均延迟
    max_delay: int                            # 最大延迟
    
    # 稳定性
    episode_drift: float                      # Episode 总 Drift (单标量)
    total_shifts: int                         # 总时间变化次数
    total_switches: int                       # 总 Pad 切换次数
    
    # 求解性能
    total_solve_time_ms: int                  # 总求解时间
    avg_solve_time_ms: float                  # 平均求解时间
    num_replans: int                          # 重排次数
    num_forced_replans: int                   # 强制重排次数
    
    # 其他
    num_completed: int                        # 完成任务数
    num_total: int                            # 总任务数
    completion_rate: float                    # 完成率


# ============================================================================
# 单任务指标
# ============================================================================

def compute_task_time_drift(
    task_id: str,
    old_plan: Optional[Plan],
    new_plan: Plan,
    horizon: int
) -> float:
    """
    计算单任务时间偏移 D_time_k
    
    D_time_k = |start_new - start_old| / horizon
    
    Args:
        task_id: 任务 ID
        old_plan: 上一轮计划
        new_plan: 本轮计划
        horizon: 视野长度（用于归一化）
    
    Returns:
        D_time_k ∈ [0, 1]，若任务不在 old_plan 中返回 0.0
    """
    if old_plan is None:
        return 0.0
    
    old_assign = old_plan.get_assignment(task_id)
    new_assign = new_plan.get_assignment(task_id)
    
    if old_assign is None or new_assign is None:
        return 0.0
    
    diff = abs(new_assign.launch_slot - old_assign.launch_slot)
    return min(1.0, diff / horizon) if horizon > 0 else 0.0


def compute_task_pad_drift(
    task_id: str,
    old_plan: Optional[Plan],
    new_plan: Plan
) -> float:
    """
    计算单任务 Pad 切换 D_pad_k
    
    Returns:
        1.0 若 pad 变化，否则 0.0
        若任务不在 old_plan 中返回 0.0
    """
    if old_plan is None:
        return 0.0
    
    old_assign = old_plan.get_assignment(task_id)
    new_assign = new_plan.get_assignment(task_id)
    
    if old_assign is None or new_assign is None:
        return 0.0
    
    return 1.0 if new_assign.pad_id != old_assign.pad_id else 0.0


def compute_npd(
    task_id: str,
    old_plan: Optional[Plan],
    new_plan: Plan,
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> float:
    """
    计算单任务归一化偏移 NPD_k
    
    NPD_k = α * D_time_k + β * D_pad_k
    
    Args:
        task_id: 任务 ID
        old_plan: 上一轮计划
        new_plan: 本轮计划
        horizon: 视野长度
        alpha: 时间偏移权重（默认 0.7）
        beta: Pad 切换权重（默认 0.3）
    
    Returns:
        NPD_k ∈ [0, 1]
    """
    d_time = compute_task_time_drift(task_id, old_plan, new_plan, horizon)
    d_pad = compute_task_pad_drift(task_id, old_plan, new_plan)
    
    return alpha * d_time + beta * d_pad


# ============================================================================
# Rolling 级指标
# ============================================================================

def compute_plan_drift(
    old_plan: Optional[Plan],
    new_plan: Plan,
    completed_tasks: Set[str],
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Tuple[float, int, int]:
    """
    计算单次 Rolling 的 PlanDrift
    
    PlanDrift_t = (1 / |K_common|) * Σ_{k ∈ K_common} NPD_k
    
    Args:
        old_plan: 上一轮计划
        new_plan: 本轮计划
        completed_tasks: 已完成任务集合（排除）
        horizon: 视野长度
        alpha, beta: 权重
    
    Returns:
        (PlanDrift_t, num_shifts, num_switches)
    """
    if old_plan is None:
        return 0.0, 0, 0
    
    # 找出共同任务（排除已完成）
    old_task_ids = {a.task_id for a in old_plan.assignments}
    new_task_ids = {a.task_id for a in new_plan.assignments}
    
    common_tasks = (old_task_ids & new_task_ids) - completed_tasks
    
    if not common_tasks:
        return 0.0, 0, 0
    
    # 计算 NPD
    total_npd = 0.0
    num_shifts = 0
    num_switches = 0
    
    for task_id in common_tasks:
        npd = compute_npd(task_id, old_plan, new_plan, horizon, alpha, beta)
        total_npd += npd
        
        # 统计 shift 和 switch
        d_time = compute_task_time_drift(task_id, old_plan, new_plan, horizon)
        d_pad = compute_task_pad_drift(task_id, old_plan, new_plan)
        
        if d_time > 0:
            num_shifts += 1
        if d_pad > 0:
            num_switches += 1
    
    plan_drift = total_npd / len(common_tasks)
    
    return plan_drift, num_shifts, num_switches


def compute_rolling_metrics(
    t: int,
    old_plan: Optional[Plan],
    new_plan: Plan,
    completed_tasks: Set[str],
    horizon: int,
    solve_time_ms: int,
    is_feasible: bool,
    forced_replan: bool,
    frozen_count: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> RollingMetrics:
    """
    计算单次 Rolling 的完整指标
    """
    plan_drift, num_shifts, num_switches = compute_plan_drift(
        old_plan, new_plan, completed_tasks, horizon, alpha, beta
    )
    
    return RollingMetrics(
        t=t,
        plan_drift=plan_drift,
        num_shifts=num_shifts,
        num_switches=num_switches,
        num_tasks_scheduled=len(new_plan.assignments) if new_plan else 0,
        num_frozen=frozen_count,
        solve_time_ms=solve_time_ms,
        is_feasible=is_feasible,
        forced_replan=forced_replan
    )


# ============================================================================
# Episode 级指标
# ============================================================================

def compute_episode_drift(
    rolling_metrics_list: List[RollingMetrics]
) -> float:
    """
    计算 Episode 总 Drift（单标量）
    
    EpisodeDrift = (1 / T_rolls) * Σ_t PlanDrift_t
    
    Args:
        rolling_metrics_list: 所有 rolling 的指标列表
    
    Returns:
        EpisodeDrift ∈ [0, 1]
    """
    if not rolling_metrics_list:
        return 0.0
    
    total_drift = sum(m.plan_drift for m in rolling_metrics_list)
    return total_drift / len(rolling_metrics_list)


def compute_delay_metrics(
    final_assignments: List[TaskAssignment],
    tasks: List[Task],
    task_dues: Dict[str, int]
) -> Tuple[float, float, int, int, int]:
    """
    计算延迟相关指标
    
    Args:
        final_assignments: 最终排程
        tasks: 任务列表
        task_dues: 任务截止时间 {task_id: due}
    
    Returns:
        (on_time_rate, avg_delay, max_delay, total_delay, num_on_time)
    """
    if not final_assignments:
        return 1.0, 0.0, 0, 0, 0
    
    total_delay = 0
    max_delay = 0
    num_on_time = 0
    
    for assign in final_assignments:
        due = task_dues.get(assign.task_id, assign.launch_slot)
        delay = max(0, assign.launch_slot - due)
        
        total_delay += delay
        max_delay = max(max_delay, delay)
        
        if delay == 0:
            num_on_time += 1
    
    n = len(final_assignments)
    on_time_rate = num_on_time / n if n > 0 else 1.0
    avg_delay = total_delay / n if n > 0 else 0.0
    
    return on_time_rate, avg_delay, max_delay, total_delay, num_on_time


def compute_episode_metrics(
    rolling_metrics_list: List[RollingMetrics],
    final_assignments: List[TaskAssignment],
    tasks: List[Task],
    completed_task_ids: Set[str]
) -> EpisodeMetrics:
    """
    计算 Episode 的完整指标
    
    Args:
        rolling_metrics_list: 所有 rolling 的指标
        final_assignments: 最终排程
        tasks: 原始任务列表
        completed_task_ids: 已完成任务 ID 集合
    
    Returns:
        EpisodeMetrics
    """
    # 构建 due 字典
    task_dues = {t.task_id: t.due for t in tasks}
    
    # 延迟指标
    on_time_rate, avg_delay, max_delay, total_delay, _ = compute_delay_metrics(
        final_assignments, tasks, task_dues
    )
    
    # 稳定性指标
    episode_drift = compute_episode_drift(rolling_metrics_list)
    total_shifts = sum(m.num_shifts for m in rolling_metrics_list)
    total_switches = sum(m.num_switches for m in rolling_metrics_list)
    
    # 求解性能
    total_solve_time = sum(m.solve_time_ms for m in rolling_metrics_list)
    avg_solve_time = total_solve_time / len(rolling_metrics_list) if rolling_metrics_list else 0.0
    num_replans = len(rolling_metrics_list)
    num_forced = sum(1 for m in rolling_metrics_list if m.forced_replan)
    
    # 完成率
    num_total = len(tasks)
    num_completed = len(completed_task_ids)
    completion_rate = num_completed / num_total if num_total > 0 else 0.0
    
    return EpisodeMetrics(
        on_time_rate=on_time_rate,
        total_delay=total_delay,
        avg_delay=avg_delay,
        max_delay=max_delay,
        episode_drift=episode_drift,
        total_shifts=total_shifts,
        total_switches=total_switches,
        total_solve_time_ms=total_solve_time,
        avg_solve_time_ms=avg_solve_time,
        num_replans=num_replans,
        num_forced_replans=num_forced,
        num_completed=num_completed,
        num_total=num_total,
        completion_rate=completion_rate
    )


# ============================================================================
# 辅助函数
# ============================================================================

def metrics_to_dict(metrics: EpisodeMetrics) -> dict:
    """将 EpisodeMetrics 转换为字典"""
    return {
        "on_time_rate": round(metrics.on_time_rate, 4),
        "total_delay": metrics.total_delay,
        "avg_delay": round(metrics.avg_delay, 2),
        "max_delay": metrics.max_delay,
        "episode_drift": round(metrics.episode_drift, 4),
        "total_shifts": metrics.total_shifts,
        "total_switches": metrics.total_switches,
        "total_solve_time_ms": metrics.total_solve_time_ms,
        "avg_solve_time_ms": round(metrics.avg_solve_time_ms, 2),
        "num_replans": metrics.num_replans,
        "num_forced_replans": metrics.num_forced_replans,
        "num_completed": metrics.num_completed,
        "num_total": metrics.num_total,
        "completion_rate": round(metrics.completion_rate, 4)
    }


def rolling_metrics_to_dict(m: RollingMetrics) -> dict:
    """将 RollingMetrics 转换为字典"""
    return {
        "t": m.t,
        "plan_drift": round(m.plan_drift, 4),
        "num_shifts": m.num_shifts,
        "num_switches": m.num_switches,
        "num_tasks_scheduled": m.num_tasks_scheduled,
        "num_frozen": m.num_frozen,
        "solve_time_ms": m.solve_time_ms,
        "is_feasible": m.is_feasible,
        "forced_replan": m.forced_replan
    }


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    print("=== Metrics Module Test ===\n")
    
    from solver_cpsat import Plan, TaskAssignment
    
    # 模拟两个计划
    old_plan = Plan(timestamp=0, assignments=[
        TaskAssignment("T001", "PAD_A", 20, 14, 20),
        TaskAssignment("T002", "PAD_B", 30, 26, 30),
        TaskAssignment("T003", "PAD_A", 40, 35, 40),
    ])
    
    new_plan = Plan(timestamp=6, assignments=[
        TaskAssignment("T001", "PAD_A", 22, 16, 22),  # shift +2
        TaskAssignment("T002", "PAD_A", 32, 28, 32),  # shift +2, switch
        TaskAssignment("T003", "PAD_A", 40, 35, 40),  # no change
    ])
    
    horizon = 144
    completed = set()
    
    # 计算指标
    drift, shifts, switches = compute_plan_drift(
        old_plan, new_plan, completed, horizon
    )
    
    print(f"Plan Drift: {drift:.4f}")
    print(f"Shifts: {shifts}")
    print(f"Switches: {switches}")
    
    # 单任务测试
    for tid in ["T001", "T002", "T003"]:
        d_time = compute_task_time_drift(tid, old_plan, new_plan, horizon)
        d_pad = compute_task_pad_drift(tid, old_plan, new_plan)
        npd = compute_npd(tid, old_plan, new_plan, horizon)
        print(f"  {tid}: D_time={d_time:.4f}, D_pad={d_pad:.1f}, NPD={npd:.4f}")
