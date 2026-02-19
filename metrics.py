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
    """Single rolling metrics."""
    t: int
    plan_drift: float
    avg_time_shift_slots: float
    num_shifts: int
    num_switches: int
    num_window_switches: int
    num_sequence_switches: int
    num_tasks_scheduled: int
    num_frozen: int
    solve_time_ms: int
    is_feasible: bool
    forced_replan: bool
    num_active_missions: int = 0  # 参与 drift 计算的非冻结非完成 mission 数


@dataclass
class EpisodeMetrics:
    """Episode 总指标

    稳定性指标说明：
      episode_drift  = Σ_t PlanDrift_t          （总累积 drift，总量）
      drift_per_replan = episode_drift / num_replans （主稳定性指标，归一化）
    论文主文使用 drift_per_replan 作为可比稳定性指标。
    """
    # 延迟相关
    on_time_rate: float                       # 按期发射率
    total_delay: int                          # 总延迟 (slots)
    avg_delay: float                          # 平均延迟
    max_delay: int                            # 最大延迟
    weighted_tardiness: float              # weighted tardiness
    resource_utilization: float            # resource utilization
    
    # 稳定性
    episode_drift: float                      # Episode 累积 Drift（总量 = Σ PlanDrift_t）
    total_shifts: int                         # 总时间变化次数
    total_switches: int                       # 总 Pad 切换次数

    total_window_switches: int
    total_sequence_switches: int
    
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

    # ========== 归一化 drift（主稳定性指标） ==========
    drift_per_replan: float = 0.0             # episode_drift / num_replans（原主指标）
    drift_per_day: float = 0.0                # episode_drift / num_days（辅助）
    drift_per_active_mission: float = 0.0     # episode_drift / Σ num_active_missions（消除 freeze 偏差）


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


def _get_mission_by_id(missions: List[Mission], mission_id: str) -> Optional[Mission]:
    for mission in missions:
        if mission.mission_id == mission_id:
            return mission
    return None


def compute_launch_drift(
    mission_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    missions: List[Mission]
) -> int:
    if old_plan is None:
        return 0

    mission = _get_mission_by_id(missions, mission_id)
    if not mission:
        return 0

    op6 = mission.get_launch_op()
    if not op6:
        return 0

    old_assign = old_plan.get_assignment(op6.op_id)
    new_assign = new_plan.get_assignment(op6.op_id)
    if old_assign is None or new_assign is None:
        return 0

    return abs(new_assign.start_slot - old_assign.start_slot)


def compute_pad_hold_start_drift(
    mission_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    missions: List[Mission]
) -> int:
    if old_plan is None:
        return 0

    mission = _get_mission_by_id(missions, mission_id)
    if not mission:
        return 0

    op4 = mission.get_pad_hold_op()
    if not op4:
        return 0

    old_assign = old_plan.get_assignment(op4.op_id)
    new_assign = new_plan.get_assignment(op4.op_id)
    if old_assign is None or new_assign is None:
        return 0

    return abs(new_assign.start_slot - old_assign.start_slot)


def compute_window_switch(
    mission_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    missions: List[Mission]
) -> int:
    if old_plan is None:
        return 0

    mission = _get_mission_by_id(missions, mission_id)
    if not mission:
        return 0

    op6 = mission.get_launch_op()
    if not op6 or not op6.time_windows:
        return 0

    old_assign = old_plan.get_assignment(op6.op_id)
    new_assign = new_plan.get_assignment(op6.op_id)
    if old_assign is None or new_assign is None:
        return 0

    def _window_index(start: int, end: int) -> Optional[int]:
        for idx, (ws, we) in enumerate(op6.time_windows):
            if start >= ws and end <= we:
                return idx
        return None

    old_idx = _window_index(old_assign.start_slot, old_assign.end_slot)
    new_idx = _window_index(new_assign.start_slot, new_assign.end_slot)
    if old_idx is None or new_idx is None:
        return 0

    return 1 if old_idx != new_idx else 0


def _pad_sequence(plan: PlanV2_1, missions: List[Mission]) -> List[str]:
    sequence = []
    for mission in missions:
        pad_hold = mission.get_pad_hold_op()
        if not pad_hold:
            continue
        assign = plan.get_assignment(pad_hold.op_id)
        if assign and "R_pad" in assign.resources:
            sequence.append((mission.mission_id, assign.start_slot))
    sequence.sort(key=lambda x: x[1])
    return [m for m, _ in sequence]


def compute_sequence_switch(
    mission_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    missions: List[Mission]
) -> int:
    if old_plan is None:
        return 0

    old_seq = _pad_sequence(old_plan, missions)
    new_seq = _pad_sequence(new_plan, missions)

    if mission_id not in old_seq or mission_id not in new_seq:
        return 0

    old_idx = old_seq.index(mission_id)
    new_idx = new_seq.index(mission_id)

    if old_idx == 0:
        return 0

    old_pred = old_seq[old_idx - 1]
    new_pred = new_seq[new_idx - 1] if new_idx > 0 else None

    if new_pred is None:
        return 0

    return 1 if new_pred != old_pred else 0


def compute_mission_drift_v3(
    mission_id: str,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    missions: List[Mission],
    priority: float = 1.0,
    kappa_win: float = 12.0,
    kappa_seq: float = 6.0
) -> float:
    if old_plan is None:
        return 0.0

    delta_launch = compute_launch_drift(mission_id, old_plan, new_plan, missions)
    delta_pad = compute_pad_hold_start_drift(mission_id, old_plan, new_plan, missions)
    switch_win = compute_window_switch(mission_id, old_plan, new_plan, missions)
    switch_seq = compute_sequence_switch(mission_id, old_plan, new_plan, missions)

    time_drift = 0.7 * delta_launch + 0.3 * delta_pad
    discrete_drift = kappa_win * switch_win + kappa_seq * switch_seq

    return priority * (time_drift + discrete_drift)


def compute_plan_drift_ops(
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    completed_ops: Set[str],
    started_ops: Set[str],
    frozen_ops: Set[str],
    missions: List[Mission],
    kappa_win: float = 12.0,
    kappa_seq: float = 6.0
) -> Tuple[float, int, int, float, int, int, int]:
    if old_plan is None:
        return 0.0, 0, 0, 0.0, 0, 0, 0

    exclude_ops = set(completed_ops) | set(started_ops) | set(frozen_ops)

    active_missions = []
    for mission in missions:
        pad_hold = mission.get_pad_hold_op()
        launch = mission.get_launch_op()
        if not pad_hold or not launch:
            continue
        if pad_hold.op_id in exclude_ops or launch.op_id in exclude_ops:
            continue
        if old_plan.get_assignment(pad_hold.op_id) is None or old_plan.get_assignment(launch.op_id) is None:
            continue
        if new_plan.get_assignment(pad_hold.op_id) is None or new_plan.get_assignment(launch.op_id) is None:
            continue
        active_missions.append(mission)

    if not active_missions:
        return 0.0, 0, 0, 0.0, 0, 0, 0

    total_drift = 0.0
    total_time_drift = 0.0
    num_time_shifts = 0
    num_window_switches = 0
    num_sequence_switches = 0
    num_active_missions = len(active_missions)

    for mission in active_missions:
        mission_drift = compute_mission_drift_v3(
            mission.mission_id,
            old_plan,
            new_plan,
            missions,
            mission.priority,
            kappa_win,
            kappa_seq
        )
        total_drift += mission_drift

        delta_launch = compute_launch_drift(mission.mission_id, old_plan, new_plan, missions)
        delta_pad = compute_pad_hold_start_drift(mission.mission_id, old_plan, new_plan, missions)
        switch_win = compute_window_switch(mission.mission_id, old_plan, new_plan, missions)
        switch_seq = compute_sequence_switch(mission.mission_id, old_plan, new_plan, missions)

        if delta_launch > 0 or delta_pad > 0:
            num_time_shifts += 1
            total_time_drift += 0.7 * delta_launch + 0.3 * delta_pad
        if switch_win > 0:
            num_window_switches += 1
        if switch_seq > 0:
            num_sequence_switches += 1

    avg_time_shift_slots = total_time_drift / len(active_missions)
    total_switches = num_window_switches + num_sequence_switches

    return (
        total_drift,
        num_time_shifts,
        total_switches,
        avg_time_shift_slots,
        num_window_switches,
        num_sequence_switches,
        num_active_missions,
    )



def compute_rolling_metrics_ops(
    t: int,
    old_plan: Optional[PlanV2_1],
    new_plan: PlanV2_1,
    completed_ops: Set[str],
    started_ops: Set[str],
    frozen_ops: Set[str],
    missions: List[Mission],
    solve_time_ms: int,
    is_feasible: bool,
    forced_replan: bool,
    frozen_count: int,
    kappa_win: float = 12.0,
    kappa_seq: float = 6.0
) -> RollingMetrics:
    plan_drift, num_shifts, num_switches, avg_time_shift_slots, num_window_switches, num_sequence_switches, num_active_missions = compute_plan_drift_ops(
        old_plan,
        new_plan,
        completed_ops,
        started_ops,
        frozen_ops,
        missions,
        kappa_win,
        kappa_seq
    )

    return RollingMetrics(
        t=t,
        plan_drift=plan_drift,
        avg_time_shift_slots=avg_time_shift_slots,
        num_shifts=num_shifts,
        num_switches=num_switches,
        num_window_switches=num_window_switches,
        num_sequence_switches=num_sequence_switches,
        num_tasks_scheduled=len(new_plan.op_assignments) if new_plan else 0,
        num_frozen=frozen_count,
        solve_time_ms=solve_time_ms,
        is_feasible=is_feasible,
        forced_replan=forced_replan,
        num_active_missions=num_active_missions,
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
        num_window_switches=0,
        num_sequence_switches=0,
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
    计算 Episode 累积 Drift（总量标量）

    EpisodeDrift = Σ_t PlanDrift_t

    注意：这是总量指标。论文主指标使用 drift_per_replan = EpisodeDrift / num_replans
    进行归一化，确保不同策略之间公平可比。

    Args:
        rolling_metrics_list: 所有 rolling 的指标列表

    Returns:
        EpisodeDrift ≥ 0 （累积总量）
    """
    if not rolling_metrics_list:
        return 0.0

    return sum(m.plan_drift for m in rolling_metrics_list)


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
    missions: List[Mission],
    horizon_slots: int = 960,
) -> Tuple[float, float, int, int, int]:
    """Compute delay metrics using launch operation completion times.
    
    Note: Launch operation is Op6 normally, but Op7 when range_test_asset is enabled.
    We find the highest op_index for each mission to handle both cases.
    
    Uncompleted missions are penalized with delay = horizon_slots - due,
    eliminating survivorship bias.
    """
    if not missions:
        return 1.0, 0.0, 0, 0, 0

    mission_due = {m.mission_id: m.due for m in missions}
    
    # Find the highest op_index for each mission (the launch operation)
    mission_launch_op_index = {}
    for m in missions:
        max_idx = max((op.op_index for op in m.operations), default=6)
        mission_launch_op_index[m.mission_id] = max_idx

    # Collect completed mission delays from final_assignments
    completed_delays: dict = {}
    for assign in final_assignments:
        expected_launch_idx = mission_launch_op_index.get(assign.mission_id, 6)
        if assign.op_index != expected_launch_idx:
            continue
        due = mission_due.get(assign.mission_id, assign.start_slot)
        delay = max(0, assign.start_slot - due)
        completed_delays[assign.mission_id] = delay

    # Iterate ALL missions; penalize uncompleted ones
    total_delay = 0
    max_delay = 0
    num_on_time = 0
    n = len(missions)

    for m in missions:
        if m.mission_id in completed_delays:
            delay = completed_delays[m.mission_id]
        else:
            # Uncompleted mission penalty: horizon - due
            delay = max(0, horizon_slots - m.due)
        total_delay += delay
        max_delay = max(max_delay, delay)
        if delay == 0:
            num_on_time += 1

    on_time_rate = num_on_time / n if n > 0 else 1.0
    avg_delay = total_delay / n if n > 0 else 0.0

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
    missions: List[Mission],
    horizon_slots: int = 960,
) -> float:
    """Compute weighted tardiness.
    
    Uncompleted missions are penalized with tardiness = priority * (horizon - due).
    """
    priorities = {m.mission_id: m.priority for m in missions}
    dues = {m.mission_id: m.due for m in missions}
    launch_op_ids = {}
    for m in missions:
        launch = m.get_launch_op()
        if launch:
            launch_op_ids[launch.op_id] = m.mission_id

    # Collect completed mission tardiness
    completed_tardiness: dict = {}
    for assign in final_assignments:
        if assign.op_id not in launch_op_ids:
            continue
        mid = launch_op_ids[assign.op_id]
        due = dues.get(mid, assign.start_slot)
        priority = priorities.get(mid, 1.0)
        delay = max(0, assign.start_slot - due)
        completed_tardiness[mid] = delay * priority

    # Sum over ALL missions; penalize uncompleted ones
    total = 0.0
    for m in missions:
        if m.mission_id in completed_tardiness:
            total += completed_tardiness[m.mission_id]
        else:
            delay = max(0, horizon_slots - m.due)
            total += delay * priorities.get(m.mission_id, 1.0)

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

    _drift_per_replan = episode_drift / num_replans if num_replans > 0 else 0.0
    _slots_per_day = 96
    _n_days = max(1, (horizon_slots or 960) / _slots_per_day)
    _drift_per_day = episode_drift / _n_days
    _total_active = sum(m.num_active_missions for m in rolling_metrics_list)
    _drift_per_active_mission = episode_drift / _total_active if _total_active > 0 else 0.0

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
        total_window_switches=sum(m.num_window_switches for m in rolling_metrics_list),
        total_sequence_switches=sum(m.num_sequence_switches for m in rolling_metrics_list),
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
        util_r_pad=resource_utilization,
        drift_per_replan=_drift_per_replan,
        drift_per_day=_drift_per_day,
        drift_per_active_mission=_drift_per_active_mission,
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
        final_assignments, missions, horizon_slots=horizon_slots
    )

    weighted_tardiness = compute_weighted_tardiness_ops(final_assignments, missions, horizon_slots=horizon_slots)

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

    # 构建 launch op_ids 集合用于 makespan 计算
    _launch_op_ids = set()
    for m in missions:
        _launch = m.get_launch_op()
        if _launch:
            _launch_op_ids.add(_launch.op_id)

    makespan_cmax = max(
        (a.end_slot for a in final_assignments if a.op_id in _launch_op_ids),
        default=0
    )
    util_r_pad = compute_r_pad_utilization_ops(final_assignments, resources, horizon_slots)

    # 归一化 drift
    drift_per_replan = episode_drift / num_replans if num_replans > 0 else 0.0
    slots_per_day = 96  # 24h * 4 (15min slots)
    num_days = max(1, horizon_slots / slots_per_day)
    drift_per_day = episode_drift / num_days
    total_active_missions = sum(m.num_active_missions for m in rolling_metrics_list)
    drift_per_active_mission = episode_drift / total_active_missions if total_active_missions > 0 else 0.0

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
        total_window_switches=sum(m.num_window_switches for m in rolling_metrics_list),
        total_sequence_switches=sum(m.num_sequence_switches for m in rolling_metrics_list),
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
        util_r_pad=util_r_pad,
        drift_per_replan=drift_per_replan,
        drift_per_day=drift_per_day,
        drift_per_active_mission=drift_per_active_mission,
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
        "total_window_switches": metrics.total_window_switches,
        "total_sequence_switches": metrics.total_sequence_switches,
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
        "completion_rate": round(metrics.completion_rate, 4),
        "drift_per_replan": round(metrics.drift_per_replan, 6),
        "drift_per_day": round(metrics.drift_per_day, 6),
        "drift_per_active_mission": round(metrics.drift_per_active_mission, 6),
    }


def rolling_metrics_to_dict(m: RollingMetrics) -> dict:
    """将 RollingMetrics 转换为字典"""
    return {
        "t": m.t,
        "plan_drift": round(m.plan_drift, 4),
        "avg_time_shift_slots": round(m.avg_time_shift_slots, 2),
        "num_shifts": m.num_shifts,
        "num_switches": m.num_switches,
        "num_window_switches": m.num_window_switches,
        "num_sequence_switches": m.num_sequence_switches,
        "num_tasks_scheduled": m.num_tasks_scheduled,
        "num_frozen": m.num_frozen,
        "solve_time_ms": m.solve_time_ms,
        "is_feasible": m.is_feasible,
        "forced_replan": m.forced_replan,
        "num_active_missions": m.num_active_missions,
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
