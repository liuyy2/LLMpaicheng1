"""
指标计算模块 - 稳定性与性能指标
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import math

from solver_cpsat import Plan, TaskAssignment, Task, PlanV2_1, OpAssignment, Mission, Resource


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class RollingMetrics:
    """单次 Rolling 的指标"""
    t: int                                    # 时刻
    plan_drift: float                         # PlanDrift_t
    avg_time_shift_slots: float               # 平均时间偏移 (slots)
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
    weighted_tardiness: float              # weighted tardiness
    resource_utilization: float            # resource utilization
    
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

    # ========== 补充指标 ==========
    avg_time_deviation_min: float             # 平均时间偏移(分钟)
    total_resource_switches: int              # 资源切换总数
    makespan_cmax: int                        # 完成时间Cmax (slots)
    feasible_rate: float                      # 可行率
    forced_replan_rate: float                 # 强制重排率
    avg_frozen: float                         # 平均冻结数量
    avg_num_tasks_scheduled: float            # 平均调度任务数
    util_r_pad: float                         # R_pad 利用率


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
) -> Tuple[float, int, int, float]:
    """
    ???? Rolling ? PlanDrift

    PlanDrift_t = (1 / |K_common|) * ?_{k ? K_common} NPD_k

    Args:
        old_plan: ?????
        new_plan: ????
        completed_tasks: ???????????
        horizon: ????
        alpha, beta: ??

    Returns:
        (PlanDrift_t, num_shifts, num_switches, avg_time_shift_slots)
    """
    if old_plan is None:
        return 0.0, 0, 0, 0.0

    # ?????????????
    old_task_ids = {a.task_id for a in old_plan.assignments}
    new_task_ids = {a.task_id for a in new_plan.assignments}

    common_tasks = (old_task_ids & new_task_ids) - completed_tasks

    if not common_tasks:
        return 0.0, 0, 0, 0.0

    # ?? NPD
    total_npd = 0.0
    num_shifts = 0
    num_switches = 0
    total_time_shift = 0.0

    for task_id in common_tasks:
        old_assign = old_plan.get_assignment(task_id)
        new_assign = new_plan.get_assignment(task_id)
        if old_assign is None or new_assign is None:
            continue

        diff = abs(new_assign.launch_slot - old_assign.launch_slot)
        total_time_shift += diff

        d_time = min(1.0, diff / horizon) if horizon > 0 else 0.0
        d_pad = 1.0 if new_assign.pad_id != old_assign.pad_id else 0.0
        npd = alpha * d_time + beta * d_pad
        total_npd += npd

        if diff > 0:
            num_shifts += 1
        if d_pad > 0:
            num_switches += 1

    plan_drift = total_npd / len(common_tasks)
    avg_time_shift_slots = total_time_shift / len(common_tasks)

    return plan_drift, num_shifts, num_switches, avg_time_shift_slots


def compute_op_time_drift(
    op_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    horizon: int
) -> float:
    """Compute op time drift using start time."""
    if old_plan is None:
        return 0.0

    old_assign = old_plan.get_assignment(op_id)
    new_assign = new_plan.get_assignment(op_id)

    if old_assign is None or new_assign is None:
        return 0.0

    diff = abs(new_assign.start_slot - old_assign.start_slot)
    return min(1.0, diff / horizon) if horizon > 0 else 0.0


def compute_op_resource_drift(
    op_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    op6_windows: Dict[str, List[Tuple[int, int]]]
) -> float:
    """Compute op6 window switch drift (0/1) for V2.1."""
    if old_plan is None:
        return 0.0

    old_assign = old_plan.get_assignment(op_id)
    new_assign = new_plan.get_assignment(op_id)

    if old_assign is None or new_assign is None:
        return 0.0

    if old_assign.op_index != 6 or new_assign.op_index != 6:
        return 0.0

    windows = op6_windows.get(op_id, [])
    if not windows:
        return 0.0

    def _window_index(start: int, end: int) -> Optional[int]:
        for idx, (ws, we) in enumerate(windows):
            if start >= ws and end <= we:
                return idx
        return None

    old_idx = _window_index(old_assign.start_slot, old_assign.end_slot)
    new_idx = _window_index(new_assign.start_slot, new_assign.end_slot)
    if old_idx is None or new_idx is None:
        return 0.0

    return 1.0 if old_idx != new_idx else 0.0


def compute_op_npd(
    op_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    op6_windows: Dict[str, List[Tuple[int, int]]],
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> float:
    d_time = compute_op_time_drift(op_id, old_plan, new_plan, horizon)
    d_res = compute_op_resource_drift(op_id, old_plan, new_plan, op6_windows)
    return alpha * d_time + beta * d_res


def compute_plan_drift_ops(
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    completed_ops: Set[str],
    missions: List[Mission],
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> Tuple[float, int, int, float]:
    if old_plan is None:
        return 0.0, 0, 0, 0.0

    old_op_ids = {a.op_id for a in old_plan.op_assignments}
    new_op_ids = {a.op_id for a in new_plan.op_assignments}
    common_ops = (old_op_ids & new_op_ids) - completed_ops

    if not common_ops:
        return 0.0, 0, 0, 0.0

    op6_windows: Dict[str, List[Tuple[int, int]]] = {}
    for mission in missions:
        op6 = mission.get_operation(6)
        if op6:
            op6_windows[op6.op_id] = list(op6.time_windows or [])

    total_npd = 0.0
    num_shifts = 0
    num_switches = 0
    total_time_shift = 0.0

    for op_id in common_ops:
        old_assign = old_plan.get_assignment(op_id)
        new_assign = new_plan.get_assignment(op_id)
        if old_assign is None or new_assign is None:
            continue

        diff = abs(new_assign.start_slot - old_assign.start_slot)
        total_time_shift += diff

        d_time = min(1.0, diff / horizon) if horizon > 0 else 0.0
        d_res = compute_op_resource_drift(op_id, old_plan, new_plan, op6_windows)
        npd = alpha * d_time + beta * d_res
        total_npd += npd

        if diff > 0:
            num_shifts += 1
        if d_res > 0:
            num_switches += 1

    plan_drift = total_npd / len(common_ops)
    avg_time_shift_slots = total_time_shift / len(common_ops)
    return plan_drift, num_shifts, num_switches, avg_time_shift_slots


def compute_rolling_metrics_ops(
    t: int,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    completed_ops: Set[str],
    missions: List[Mission],
    horizon: int,
    solve_time_ms: int,
    is_feasible: bool,
    forced_replan: bool,
    frozen_count: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> RollingMetrics:
    plan_drift, num_shifts, num_switches, avg_time_shift_slots = compute_plan_drift_ops(
        old_plan, new_plan, completed_ops, missions, horizon, alpha, beta
    )

    return RollingMetrics(
        t=t,
        plan_drift=plan_drift,
        avg_time_shift_slots=avg_time_shift_slots,
        num_shifts=num_shifts,
        num_switches=num_switches,
        num_tasks_scheduled=len(new_plan.op_assignments) if new_plan else 0,
        num_frozen=frozen_count,
        solve_time_ms=solve_time_ms,
        is_feasible=is_feasible,
        forced_replan=forced_replan
    )


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
    plan_drift, num_shifts, num_switches, avg_time_shift_slots = compute_plan_drift(
        old_plan, new_plan, completed_tasks, horizon, alpha, beta
    )
    
    return RollingMetrics(
        t=t,
        plan_drift=plan_drift,
        avg_time_shift_slots=avg_time_shift_slots,
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


def compute_delay_metrics_ops(
    final_assignments: List[OpAssignment],
    missions: List[Mission]
) -> Tuple[float, float, int, int, int]:
    """Compute delay metrics using Op6 completion times."""
    if not final_assignments:
        return 1.0, 0.0, 0, 0, 0

    mission_due = {m.mission_id: m.due for m in missions}

    total_delay = 0
    max_delay = 0
    num_on_time = 0
    counted = 0

    for assign in final_assignments:
        if assign.op_index != 6:
            continue
        due = mission_due.get(assign.mission_id, assign.end_slot)
        delay = max(0, assign.end_slot - due)

        total_delay += delay
        max_delay = max(max_delay, delay)
        if delay == 0:
            num_on_time += 1
        counted += 1

    on_time_rate = num_on_time / counted if counted > 0 else 1.0
    avg_delay = total_delay / counted if counted > 0 else 0.0

    return on_time_rate, avg_delay, max_delay, total_delay, num_on_time


def compute_weighted_tardiness_tasks(
    final_assignments: List[TaskAssignment],
    tasks: List[Task]
) -> float:
    priorities = {t.task_id: t.priority for t in tasks}
    dues = {t.task_id: t.due for t in tasks}

    total = 0.0
    for assign in final_assignments:
        due = dues.get(assign.task_id, assign.launch_slot)
        priority = priorities.get(assign.task_id, 1.0)
        delay = max(0, assign.launch_slot - due)
        total += delay * priority

    return total


def compute_weighted_tardiness_ops(
    final_assignments: List[OpAssignment],
    missions: List[Mission]
) -> float:
    priorities = {m.mission_id: m.priority for m in missions}
    dues = {m.mission_id: m.due for m in missions}

    total = 0.0
    for assign in final_assignments:
        if assign.op_index != 6:
            continue
        due = dues.get(assign.mission_id, assign.end_slot)
        priority = priorities.get(assign.mission_id, 1.0)
        delay = max(0, assign.end_slot - due)
        total += delay * priority

    return total


def compute_resource_utilization(
    final_assignments: List[Any],
    total_capacity_slots: int
) -> float:
    if total_capacity_slots <= 0:
        return 0.0

    total_busy = 0
    for assign in final_assignments:
        duration = max(0, assign.end_slot - assign.start_slot)
        if hasattr(assign, 'resources'):
            total_busy += duration * len(assign.resources)
        else:
            total_busy += duration

    return min(1.0, total_busy / total_capacity_slots)


def compute_r_pad_utilization_ops(
    final_assignments: List[OpAssignment],
    resources: List[Resource],
    horizon_slots: int
) -> float:
    pad = next((r for r in resources if r.resource_id == "R_pad"), None)
    if not pad or horizon_slots <= 0:
        return 0.0

    total_capacity = pad.capacity * horizon_slots
    total_busy = 0
    for assign in final_assignments:
        if "R_pad" not in getattr(assign, "resources", []):
            continue
        duration = max(0, assign.end_slot - assign.start_slot)
        total_busy += duration

    return min(1.0, total_busy / total_capacity)


def compute_episode_metrics(
    rolling_metrics_list: List[RollingMetrics],
    final_assignments: List[TaskAssignment],
    tasks: List[Task],
    completed_task_ids: Set[str],
    pad_count: Optional[int] = None,
    horizon_slots: Optional[int] = None,
    slot_minutes: int = 10
) -> EpisodeMetrics:
    """Compute episode metrics for V1 tasks."""
    task_dues = {t.task_id: t.due for t in tasks}

    on_time_rate, avg_delay, max_delay, total_delay, _ = compute_delay_metrics(
        final_assignments, tasks, task_dues
    )

    weighted_tardiness = compute_weighted_tardiness_tasks(final_assignments, tasks)

    episode_drift = compute_episode_drift(rolling_metrics_list)
    total_shifts = sum(m.num_shifts for m in rolling_metrics_list)
    total_switches = sum(m.num_switches for m in rolling_metrics_list)

    total_solve_time = sum(m.solve_time_ms for m in rolling_metrics_list)
    avg_solve_time = total_solve_time / len(rolling_metrics_list) if rolling_metrics_list else 0.0
    num_replans = len(rolling_metrics_list)
    num_forced = sum(1 for m in rolling_metrics_list if m.forced_replan)

    num_total = len(tasks)
    num_completed = len(completed_task_ids)
    completion_rate = num_completed / num_total if num_total > 0 else 0.0

    resource_utilization = 0.0
    if pad_count and horizon_slots:
        resource_utilization = compute_resource_utilization(
            final_assignments, pad_count * horizon_slots
        )

    avg_time_shift_slots = (
        sum(m.avg_time_shift_slots for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_time_deviation_min = avg_time_shift_slots * slot_minutes
    feasible_rate = (
        sum(1.0 for m in rolling_metrics_list if m.is_feasible) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    forced_replan_rate = (
        sum(1.0 for m in rolling_metrics_list if m.forced_replan) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_frozen = (
        sum(m.num_frozen for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_num_tasks_scheduled = (
        sum(m.num_tasks_scheduled for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    makespan_cmax = max((a.end_slot for a in final_assignments), default=0)

    return EpisodeMetrics(
        on_time_rate=on_time_rate,
        total_delay=total_delay,
        avg_delay=avg_delay,
        max_delay=max_delay,
        weighted_tardiness=weighted_tardiness,
        resource_utilization=resource_utilization,
        episode_drift=episode_drift,
        total_shifts=total_shifts,
        total_switches=total_switches,
        total_solve_time_ms=total_solve_time,
        avg_solve_time_ms=avg_solve_time,
        num_replans=num_replans,
        num_forced_replans=num_forced,
        num_completed=num_completed,
        num_total=num_total,
        completion_rate=completion_rate,
        avg_time_deviation_min=avg_time_deviation_min,
        total_resource_switches=total_switches,
        makespan_cmax=makespan_cmax,
        feasible_rate=feasible_rate,
        forced_replan_rate=forced_replan_rate,
        avg_frozen=avg_frozen,
        avg_num_tasks_scheduled=avg_num_tasks_scheduled,
        util_r_pad=resource_utilization
    )


def compute_episode_metrics_ops(
    rolling_metrics_list: List[RollingMetrics],
    final_assignments: List[OpAssignment],
    missions: List[Mission],
    completed_mission_ids: Set[str],
    resources: List[Resource],
    horizon_slots: int,
    slot_minutes: int = 10
) -> EpisodeMetrics:
    """Compute episode metrics for V2.1 missions."""
    on_time_rate, avg_delay, max_delay, total_delay, _ = compute_delay_metrics_ops(
        final_assignments, missions
    )

    weighted_tardiness = compute_weighted_tardiness_ops(final_assignments, missions)

    episode_drift = compute_episode_drift(rolling_metrics_list)
    total_shifts = sum(m.num_shifts for m in rolling_metrics_list)
    total_switches = sum(m.num_switches for m in rolling_metrics_list)

    total_solve_time = sum(m.solve_time_ms for m in rolling_metrics_list)
    avg_solve_time = total_solve_time / len(rolling_metrics_list) if rolling_metrics_list else 0.0
    num_replans = len(rolling_metrics_list)
    num_forced = sum(1 for m in rolling_metrics_list if m.forced_replan)

    num_total = len(missions)
    num_completed = len(completed_mission_ids)
    completion_rate = num_completed / num_total if num_total > 0 else 0.0

    total_capacity = sum(r.capacity for r in resources) * horizon_slots
    resource_utilization = compute_resource_utilization(final_assignments, total_capacity)

    avg_time_shift_slots = (
        sum(m.avg_time_shift_slots for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_time_deviation_min = avg_time_shift_slots * slot_minutes
    feasible_rate = (
        sum(1.0 for m in rolling_metrics_list if m.is_feasible) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    forced_replan_rate = (
        sum(1.0 for m in rolling_metrics_list if m.forced_replan) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_frozen = (
        sum(m.num_frozen for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )
    avg_num_tasks_scheduled = (
        sum(m.num_tasks_scheduled for m in rolling_metrics_list) / len(rolling_metrics_list)
        if rolling_metrics_list else 0.0
    )

    makespan_cmax = max(
        (a.end_slot for a in final_assignments if a.op_index == 6),
        default=0
    )
    util_r_pad = compute_r_pad_utilization_ops(final_assignments, resources, horizon_slots)

    return EpisodeMetrics(
        on_time_rate=on_time_rate,
        total_delay=total_delay,
        avg_delay=avg_delay,
        max_delay=max_delay,
        weighted_tardiness=weighted_tardiness,
        resource_utilization=resource_utilization,
        episode_drift=episode_drift,
        total_shifts=total_shifts,
        total_switches=total_switches,
        total_solve_time_ms=total_solve_time,
        avg_solve_time_ms=avg_solve_time,
        num_replans=num_replans,
        num_forced_replans=num_forced,
        num_completed=num_completed,
        num_total=num_total,
        completion_rate=completion_rate,
        avg_time_deviation_min=avg_time_deviation_min,
        total_resource_switches=total_switches,
        makespan_cmax=makespan_cmax,
        feasible_rate=feasible_rate,
        forced_replan_rate=forced_replan_rate,
        avg_frozen=avg_frozen,
        avg_num_tasks_scheduled=avg_num_tasks_scheduled,
        util_r_pad=util_r_pad
    )


def metrics_to_dict(metrics: EpisodeMetrics) -> dict:
    """将 EpisodeMetrics 转换为字典"""
    return {
        "on_time_rate": round(metrics.on_time_rate, 4),
        "total_delay": metrics.total_delay,
        "avg_delay": round(metrics.avg_delay, 2),
        "max_delay": metrics.max_delay,
        "weighted_tardiness": round(metrics.weighted_tardiness, 2),
        "resource_utilization": round(metrics.resource_utilization, 4),
        "util_r_pad": round(metrics.util_r_pad, 4),
        "episode_drift": round(metrics.episode_drift, 4),
        "total_shifts": metrics.total_shifts,
        "total_switches": metrics.total_switches,
        "avg_time_deviation_min": round(metrics.avg_time_deviation_min, 2),
        "total_resource_switches": metrics.total_resource_switches,
        "makespan_cmax": metrics.makespan_cmax,
        "feasible_rate": round(metrics.feasible_rate, 4),
        "forced_replan_rate": round(metrics.forced_replan_rate, 4),
        "avg_frozen": round(metrics.avg_frozen, 2),
        "avg_num_tasks_scheduled": round(metrics.avg_num_tasks_scheduled, 2),
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
        "avg_time_shift_slots": round(m.avg_time_shift_slots, 2),
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
