"""
特征计算模块 - 为 LLM 策略提供输入特征

提供以下特征：
- window_loss_pct: 未来 H 内 windows 可用 slot 数减少比例
- pad_outage_overlap_hours: 未来 H 内 pad outage 总时长
- delay_increase_minutes: 不重排情况下预计延误增加
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import copy
import statistics

from solver_cpsat import Task, Pad, Plan, TaskAssignment, Mission, Resource, PlanV2_1, OpAssignment
from config import Config, DEFAULT_CONFIG

@dataclass
class StateFeatures:
    """状态特征（作为 LLM/MockLLM 输入）"""
    # 窗口损失
    window_loss_pct: float                    # [0, 1] 窗口减少比例
    
    # Pad 可用性
    pad_outage_overlap_hours: float           # 未来 H 内 outage 总时长（小时）
    
    # 延迟估算
    delay_increase_minutes: float             # 预估延误增加（分钟）
    pad_pressure: float                       # R_pad pressure (demand / capacity)
    slack_min_minutes: float                  # min slack in minutes
    resource_conflict_pressure: float         # R3/R4 conflict pressure
    trend_window_loss: float                  # long-range trend of window loss
    trend_pad_pressure: float                 # long-range trend of pad pressure
    trend_slack_min_minutes: float            # long-range trend of min slack
    trend_delay_increase_minutes: float       # long-range trend of delay increase
    volatility_pad_pressure: float            # volatility of pad pressure
    
    # 任务状态
    num_urgent_tasks: int                     # 紧急任务数（即将到期）
    
    # 稳定性状态
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（便于 JSON 序列化）"""
        return {
            "window_loss_pct": round(self.window_loss_pct, 4),
            "pad_outage_overlap_hours": round(self.pad_outage_overlap_hours, 2),
            "delay_increase_minutes": round(self.delay_increase_minutes, 1),
            "pad_pressure": round(self.pad_pressure, 4),
            "slack_min_minutes": round(self.slack_min_minutes, 1),
            "resource_conflict_pressure": round(self.resource_conflict_pressure, 4),
            "trend_window_loss": round(self.trend_window_loss, 4),
            "trend_pad_pressure": round(self.trend_pad_pressure, 4),
            "trend_slack_min_minutes": round(self.trend_slack_min_minutes, 2),
            "trend_delay_increase_minutes": round(self.trend_delay_increase_minutes, 2),
            "volatility_pad_pressure": round(self.volatility_pad_pressure, 4),
            "num_urgent_tasks": self.num_urgent_tasks,
        }

TREND_WINDOW = 4

def _series_with_current(history, attr, current):
    values = [getattr(h, attr) for h in history] if history else []
    values.append(current)
    return values

def _compute_trend(values, window: int = TREND_WINDOW):
    if len(values) < 2:
        return 0.0
    k = min(window, len(values))
    denom = max(1, k - 1)
    return (values[-1] - values[-k]) / denom

def _compute_volatility(values, window: int = TREND_WINDOW):
    if len(values) < 2:
        return 0.0
    k = min(window, len(values))
    subset = values[-k:]
    if len(subset) < 2:
        return 0.0
    return statistics.pstdev(subset)

def compute_window_loss_pct(
    tasks: List[Task],
    now: int,
    horizon_end: int,
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
    completed_tasks: Optional[Set[str]] = None
) -> Tuple[float, float, Dict[str, Set[int]]]:
    """
    计算窗口损失比例
    
    Args:
        tasks: 当前任务列表
        now: 当前时刻
        horizon_end: 视野结束时刻
        prev_window_slots: 上一次的窗口 slot 集合 {task_id: set of slots}
        completed_tasks: 已完成任务
    
    Returns:
        (window_loss_pct, window_remaining_pct, current_window_slots)
    """
    if completed_tasks is None:
        completed_tasks = set()
    
    # 计算当前每个任务的窗口 slots
    current_window_slots: Dict[str, Set[int]] = {}
    total_current = 0
    
    for task in tasks:
        if task.task_id in completed_tasks:
            continue
        
        slots = set()
        for win_start, win_end in task.windows:
            # 只计算 [now, horizon_end] 范围内的 slots
            effective_start = max(win_start, now)
            effective_end = min(win_end, horizon_end)
            if effective_start <= effective_end:
                for s in range(effective_start, effective_end + 1):
                    slots.add(s)
        
        current_window_slots[task.task_id] = slots
        total_current += len(slots)
    
    # 计算损失
    if prev_window_slots is None:
        # 首次调用，无法计算损失
        return 0.0, 1.0, current_window_slots
    
    total_prev = 0
    total_lost = 0
    
    for task_id, prev_slots in prev_window_slots.items():
        if task_id in completed_tasks:
            continue
        
        total_prev += len(prev_slots)
        
        curr_slots = current_window_slots.get(task_id, set())
        # 损失 = 上一次有但这次没有的 slots
        lost = prev_slots - curr_slots
        total_lost += len(lost)
    
    # 窗口损失比例
    window_loss_pct = total_lost / total_prev if total_prev > 0 else 0.0
    
    # 剩余窗口比例 (相对于总需求的估算)
    # 使用任务数 * 视野长度作为理论最大值的估算
    horizon_len = horizon_end - now
    num_active_tasks = len([t for t in tasks if t.task_id not in completed_tasks])
    theoretical_max = num_active_tasks * horizon_len if num_active_tasks > 0 else 1
    window_remaining_pct = min(1.0, total_current / theoretical_max)
    
    return window_loss_pct, window_remaining_pct, current_window_slots

def compute_window_loss_pct_ops(
    missions: List[Mission],
    now: int,
    horizon_end: int,
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
    completed_missions: Optional[Set[str]] = None
) -> Tuple[float, float, Dict[str, Set[int]]]:
    if completed_missions is None:
        completed_missions = set()

    current_window_slots: Dict[str, Set[int]] = {}
    total_current = 0

    for mission in missions:
        if mission.mission_id in completed_missions:
            continue
        launch = mission.get_launch_op()
        if not launch:
            continue

        slots = set()
        for win_start, win_end in launch.time_windows:
            effective_start = max(win_start, now)
            effective_end = min(win_end, horizon_end)
            if effective_start <= effective_end:
                for s in range(effective_start, effective_end + 1):
                    slots.add(s)

        current_window_slots[mission.mission_id] = slots
        total_current += len(slots)

    if prev_window_slots is None:
        return 0.0, 1.0, current_window_slots

    total_prev = 0
    total_lost = 0

    for mission_id, prev_slots in prev_window_slots.items():
        if mission_id in completed_missions:
            continue

        total_prev += len(prev_slots)

        curr_slots = current_window_slots.get(mission_id, set())
        lost = prev_slots - curr_slots
        total_lost += len(lost)

    window_loss_pct = total_lost / total_prev if total_prev > 0 else 0.0

    horizon_len = horizon_end - now
    num_active = len([m for m in missions if m.mission_id not in completed_missions])
    theoretical_max = num_active * horizon_len if num_active > 0 else 1
    window_remaining_pct = min(1.0, total_current / theoretical_max)

    return window_loss_pct, window_remaining_pct, current_window_slots

def compute_pad_outage_overlap(
    pads: List[Pad],
    now: int,
    horizon_end: int,
    tasks: List[Task],
    current_plan: Optional[Plan],
    slot_minutes: int = 10
) -> Tuple[float, int]:
    """
    计算未来视野内 pad outage 的影响
    
    Args:
        pads: Pad 列表
        now: 当前时刻
        horizon_end: 视野结束时刻
        tasks: 任务列表
        current_plan: 当前计划
        slot_minutes: 每 slot 分钟数
    
    Returns:
        (pad_outage_overlap_hours, affected_task_count)
    """
    # 计算视野内的 outage 总 slot 数
    total_outage_slots = 0
    
    for pad in pads:
        for ua_start, ua_end in pad.unavailable:
            # 计算与 [now, horizon_end] 的交集
            overlap_start = max(ua_start, now)
            overlap_end = min(ua_end, horizon_end)
            if overlap_start <= overlap_end:
                total_outage_slots += (overlap_end - overlap_start + 1)
    
    # 转为小时
    outage_hours = (total_outage_slots * slot_minutes) / 60.0
    
    # 计算受影响的任务数
    affected_tasks = 0
    
    if current_plan:
        for assignment in current_plan.assignments:
            task_pad = next((p for p in pads if p.pad_id == assignment.pad_id), None)
            if task_pad is None:
                continue
            
            # 检查任务是否与 outage 冲突
            for ua_start, ua_end in task_pad.unavailable:
                if not (assignment.launch_slot <= ua_start or assignment.start_slot > ua_end):
                    affected_tasks += 1
                    break
    
    return outage_hours, affected_tasks

def compute_pad_outage_overlap_ops(
    resources: List[Resource],
    now: int,
    horizon_end: int,
    current_plan: Optional[PlanV2_1],
    slot_minutes: int = 10
) -> Tuple[float, int]:
    pad = next((r for r in resources if r.resource_id == "R_pad"), None)
    if not pad:
        return 0.0, 0

    total_outage_slots = _compute_unavailable_slots(pad.unavailable, now, horizon_end)
    outage_hours = (total_outage_slots * slot_minutes) / 60.0

    affected_ops = 0
    if current_plan:
        for assignment in current_plan.op_assignments:
            if "R_pad" not in assignment.resources:
                continue
            for ua_start, ua_end in pad.unavailable:
                if not (assignment.end_slot <= ua_start or assignment.start_slot > ua_end):
                    affected_ops += 1
                    break

    return outage_hours, affected_ops

def _compute_unavailable_slots(intervals, start, end):
    total = 0
    for s, e in intervals:
        if e < start or s > end:
            continue
        total += min(e, end) - max(s, start) + 1
    return total

def compute_pad_pressure_tasks(tasks, pads, now, horizon_end, completed_tasks):
    total_demand = sum(
        t.duration for t in tasks
        if t.task_id not in completed_tasks and t.release <= horizon_end
    )
    total_capacity = max(1, (horizon_end - now + 1) * max(1, len(pads)))
    total_unavail = sum(
        _compute_unavailable_slots(p.unavailable, now, horizon_end) for p in pads
    )
    available = max(1, total_capacity - total_unavail)
    return total_demand / available

def compute_pad_pressure_ops(missions, resources, now, horizon_end, completed_ops):
    pad = next((r for r in resources if r.resource_id == 'R_pad'), None)
    if not pad:
        return 0.0
    total_demand = 0
    for mission in missions:
        for op in mission.operations:
            if op.op_id in completed_ops:
                continue
            if op.release > horizon_end:
                continue
            if 'R_pad' in op.resources:
                total_demand += op.duration
    total_capacity = max(1, (horizon_end - now + 1) * max(1, pad.capacity))
    total_unavail = _compute_unavailable_slots(pad.unavailable, now, horizon_end)
    available = max(1, total_capacity - total_unavail)
    return total_demand / available

def compute_resource_conflict_pressure_ops(missions, resources, now, horizon_end, completed_ops):
    pressures = []
    for res_id in ('R3', 'R4'):
        resource = next((r for r in resources if r.resource_id == res_id), None)
        if not resource:
            continue
        demand = 0
        for mission in missions:
            for op in mission.operations:
                if op.op_id in completed_ops:
                    continue
                if op.release > horizon_end:
                    continue
                if res_id in op.resources:
                    demand += op.duration
        total_capacity = max(1, (horizon_end - now + 1) * max(1, resource.capacity))
        total_unavail = _compute_unavailable_slots(resource.unavailable, now, horizon_end)
        available = max(1, total_capacity - total_unavail)
        pressures.append(demand / available)
    return max(pressures) if pressures else 0.0

def compute_min_slack_minutes_tasks(tasks, now, completed_tasks, slot_minutes):
    slacks = []
    for task in tasks:
        if task.task_id in completed_tasks:
            continue
        slack_slots = task.due - (now + task.duration)
        slacks.append(slack_slots * slot_minutes)
    return min(slacks) if slacks else 0.0

def compute_min_slack_minutes_ops(missions, now, completed_ops, slot_minutes):
    slacks = []
    for mission in missions:
        remaining = sum(
            op.duration for op in mission.operations
            if op.op_id not in completed_ops
        )
        if remaining <= 0:
            continue
        slack_slots = mission.due - (now + remaining)
        slacks.append(slack_slots * slot_minutes)
    return min(slacks) if slacks else 0.0

    pad_resource = next((r for r in resources if r.resource_id == 'R_pad'), None)
    if pad_resource:
        for ua_start, ua_end in pad_resource.unavailable:
            overlap_start = max(ua_start, now)
            overlap_end = min(ua_end, horizon_end)
            if overlap_start <= overlap_end:
                total_outage_slots += (overlap_end - overlap_start + 1)

    outage_hours = (total_outage_slots * slot_minutes) / 60.0

    affected_ops = 0
    if current_plan and pad_resource:
        for assignment in current_plan.op_assignments:
            if 'R_pad' not in assignment.resources:
                continue
            for ua_start, ua_end in pad_resource.unavailable:
                if not (assignment.end_slot <= ua_start or assignment.start_slot > ua_end):
                    affected_ops += 1
                    break

    return outage_hours, affected_ops

def compute_delay_increase(
    tasks: List[Task],
    current_plan: Optional[Plan],
    now: int,
    slot_minutes: int = 10,
    completed_tasks: Optional[Set[str]] = None
) -> Tuple[float, float]:
    """
    估算不重排情况下的延误增加
    
    简单估算：检查当前计划中的任务是否会因窗口/release 变化而延迟
    
    Args:
        tasks: 任务列表
        current_plan: 当前计划
        now: 当前时刻
        slot_minutes: 每 slot 分钟数
        completed_tasks: 已完成任务
    
    Returns:
        (delay_increase_minutes, current_total_delay_minutes)
    """
    if completed_tasks is None:
        completed_tasks = set()
    
    if current_plan is None:
        return 0.0, 0.0
    
    task_map = {t.task_id: t for t in tasks}
    
    current_total_delay = 0
    potential_delay_increase = 0
    
    for assignment in current_plan.assignments:
        if assignment.task_id in completed_tasks:
            continue
        
        task = task_map.get(assignment.task_id)
        if task is None:
            continue
        
        # 当前延迟
        current_delay = max(0, assignment.launch_slot - task.due)
        current_total_delay += current_delay
        
        # 检查当前分配是否仍然可行
        launch = assignment.launch_slot
        
        # 检查是否在有效窗口内
        in_window = False
        for win_start, win_end in task.windows:
            if win_start <= launch <= win_end:
                in_window = True
                break
        
        if not in_window and launch > now:
            # 需要延迟到下一个可用窗口
            earliest_valid = None
            for win_start, win_end in task.windows:
                if win_start > launch:
                    earliest_valid = win_start
                    break
            
            if earliest_valid:
                potential_delay_increase += (earliest_valid - launch)
            else:
                # 没有后续窗口，估算一个大延迟
                potential_delay_increase += 30
    
    # 转为分钟
    delay_increase_minutes = potential_delay_increase * slot_minutes
    current_total_delay_minutes = current_total_delay * slot_minutes
    
    return delay_increase_minutes, current_total_delay_minutes

def compute_delay_increase_ops(
    missions: List[Mission],
    current_plan: Optional[PlanV2_1],
    now: int,
    slot_minutes: int = 10,
    completed_ops: Optional[Set[str]] = None
) -> Tuple[float, float]:
    if completed_ops is None:
        completed_ops = set()

    if current_plan is None:
        return 0.0, 0.0

    mission_map = {m.mission_id: m for m in missions}
    # 构建 launch op_ids 集合
    launch_op_ids = set()
    for m in missions:
        _launch = m.get_launch_op()
        if _launch:
            launch_op_ids.add(_launch.op_id)

    current_total_delay = 0
    potential_delay_increase = 0

    for assignment in current_plan.op_assignments:
        if assignment.op_id not in launch_op_ids:
            continue

        mission = mission_map.get(assignment.mission_id)
        if mission is None:
            continue

        launch = mission.get_launch_op()
        if not launch or launch.op_id in completed_ops:
            continue

        current_delay = max(0, assignment.start_slot - mission.due)
        current_total_delay += current_delay

        in_window = any(
            assignment.start_slot >= ws and assignment.end_slot <= we
            for ws, we in launch.time_windows
        )

        if not in_window and assignment.start_slot > now:
            earliest_valid = None
            for win_start, win_end in launch.time_windows:
                if win_start >= assignment.start_slot and win_start + launch.duration <= win_end:
                    earliest_valid = win_start
                    break

            if earliest_valid is not None:
                potential_delay_increase += (earliest_valid - assignment.start_slot)
            else:
                potential_delay_increase += 30

    delay_increase_minutes = potential_delay_increase * slot_minutes
    current_total_delay_minutes = current_total_delay * slot_minutes

    return delay_increase_minutes, current_total_delay_minutes

def count_urgent_tasks(
    tasks: List[Task],
    now: int,
    urgent_threshold_slots: int = 18,  # 3 小时内到期算紧急
    completed_tasks: Optional[Set[str]] = None
) -> int:
    """
    统计紧急任务数（即将到期）
    
    Args:
        tasks: 任务列表
        now: 当前时刻
        urgent_threshold_slots: 紧急阈值 (slots)
        completed_tasks: 已完成任务
    
    Returns:
        紧急任务数
    """
    if completed_tasks is None:
        completed_tasks = set()
    
    count = 0
    for task in tasks:
        if task.task_id in completed_tasks:
            continue
        
        # 如果 due - now <= threshold，则为紧急
        if task.due - now <= urgent_threshold_slots:
            count += 1
    
    return count

def count_urgent_missions(
    missions: List[Mission],
    now: int,
    urgent_threshold_slots: int = 18,
    completed_missions: Optional[Set[str]] = None
) -> int:
    if completed_missions is None:
        completed_missions = set()

    count = 0
    for mission in missions:
        if mission.mission_id in completed_missions:
            continue
        if mission.due - now <= urgent_threshold_slots:
            count += 1

    return count

def compute_state_features(
    tasks: List[Task],
    pads: List[Pad],
    current_plan: Optional[Plan],
    now: int,
    config: Config,
    completed_tasks: Optional[Set[str]] = None,
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
    recent_shifts: int = 0,
    recent_switches: int = 0,
    history: Optional[List["StateFeatures"]] = None
) -> Tuple[StateFeatures, Dict[str, Set[int]]]:
    """
    计算完整的状态特征
    
    Args:
        tasks: 任务列表
        pads: Pad 列表
        current_plan: 当前计划
        now: 当前时刻
        config: 配置
        completed_tasks: 已完成任务
        prev_window_slots: 上次的窗口 slots
        recent_shifts: 最近的 shift 数
        recent_switches: 最近的 switch 数
    
    Returns:
        (StateFeatures, current_window_slots)
    """
    if completed_tasks is None:
        completed_tasks = set()
    
    horizon_end = now + config.horizon_slots
    
    # 窗口损失
    window_loss_pct, _window_remaining_pct, curr_window_slots = compute_window_loss_pct(
        tasks, now, horizon_end, prev_window_slots, completed_tasks
    )
    
    # Pad outage
    outage_hours, _outage_task_count = compute_pad_outage_overlap(
        pads, now, horizon_end, tasks, current_plan, config.slot_minutes
    )
    
    # 延迟估算
    delay_increase, _current_delay = compute_delay_increase(
        tasks, current_plan, now, config.slot_minutes, completed_tasks
    )
    
    pad_pressure = compute_pad_pressure_tasks(
        tasks, pads, now, horizon_end, completed_tasks
    )

    slack_min = compute_min_slack_minutes_tasks(
        tasks, now, completed_tasks, config.slot_minutes
    )

    resource_conflict_pressure = 0.0
    
    # 紧急任务
    urgent_count = count_urgent_tasks(tasks, now, completed_tasks=completed_tasks)
    
    # 完成率

    window_loss_series = _series_with_current(history, "window_loss_pct", window_loss_pct)
    pad_pressure_series = _series_with_current(history, "pad_pressure", pad_pressure)
    slack_min_series = _series_with_current(history, "slack_min_minutes", slack_min)
    delay_increase_series = _series_with_current(history, "delay_increase_minutes", delay_increase)

    trend_window_loss = _compute_trend(window_loss_series)
    trend_pad_pressure = _compute_trend(pad_pressure_series)
    trend_slack_min = _compute_trend(slack_min_series)
    trend_delay_increase = _compute_trend(delay_increase_series)
    volatility_pad_pressure = _compute_volatility(pad_pressure_series)

    features = StateFeatures(
        window_loss_pct=window_loss_pct,
        pad_outage_overlap_hours=outage_hours,
        delay_increase_minutes=delay_increase,
        pad_pressure=pad_pressure,
        slack_min_minutes=slack_min,
        resource_conflict_pressure=resource_conflict_pressure,
        trend_window_loss=trend_window_loss,
        trend_pad_pressure=trend_pad_pressure,
        trend_slack_min_minutes=trend_slack_min,
        trend_delay_increase_minutes=trend_delay_increase,
        volatility_pad_pressure=volatility_pad_pressure,
        num_urgent_tasks=urgent_count,
    )
    
    return features, curr_window_slots

def compute_state_features_ops(
    missions: List[Mission],
    resources: List[Resource],
    current_plan: Optional[PlanV2_1],
    now: int,
    config: Config,
    completed_ops: Optional[Set[str]] = None,
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
    recent_shifts: int = 0,
    recent_switches: int = 0,
    history: Optional[List["StateFeatures"]] = None
) -> Tuple[StateFeatures, Dict[str, Set[int]]]:
    if completed_ops is None:
        completed_ops = set()

    completed_missions = set()
    for mission in missions:
        launch = mission.get_launch_op()
        if launch and launch.op_id in completed_ops:
            completed_missions.add(mission.mission_id)

    horizon_end = now + config.horizon_slots

    window_loss_pct, _window_remaining_pct, curr_window_slots = compute_window_loss_pct_ops(
        missions, now, horizon_end, prev_window_slots, completed_missions
    )

    outage_hours, _outage_task_count = compute_pad_outage_overlap_ops(
        resources, now, horizon_end, current_plan, config.slot_minutes
    )

    delay_increase, _current_delay = compute_delay_increase_ops(
        missions, current_plan, now, config.slot_minutes, completed_ops
    )

    pad_pressure = compute_pad_pressure_ops(
        missions, resources, now, horizon_end, completed_ops
    )

    slack_min = compute_min_slack_minutes_ops(
        missions, now, completed_ops, config.slot_minutes
    )

    resource_conflict_pressure = compute_resource_conflict_pressure_ops(
        missions, resources, now, horizon_end, completed_ops
    )

    urgent_count = count_urgent_missions(
        missions, now, completed_missions=completed_missions
    )


    window_loss_series = _series_with_current(history, "window_loss_pct", window_loss_pct)
    pad_pressure_series = _series_with_current(history, "pad_pressure", pad_pressure)
    slack_min_series = _series_with_current(history, "slack_min_minutes", slack_min)
    delay_increase_series = _series_with_current(history, "delay_increase_minutes", delay_increase)

    trend_window_loss = _compute_trend(window_loss_series)
    trend_pad_pressure = _compute_trend(pad_pressure_series)
    trend_slack_min = _compute_trend(slack_min_series)
    trend_delay_increase = _compute_trend(delay_increase_series)
    volatility_pad_pressure = _compute_volatility(pad_pressure_series)

    features = StateFeatures(
        window_loss_pct=window_loss_pct,
        pad_outage_overlap_hours=outage_hours,
        delay_increase_minutes=delay_increase,
        pad_pressure=pad_pressure,
        slack_min_minutes=slack_min,
        resource_conflict_pressure=resource_conflict_pressure,
        trend_window_loss=trend_window_loss,
        trend_pad_pressure=trend_pad_pressure,
        trend_slack_min_minutes=trend_slack_min,
        trend_delay_increase_minutes=trend_delay_increase,
        volatility_pad_pressure=volatility_pad_pressure,
        num_urgent_tasks=urgent_count,
    )

    return features, curr_window_slots


# ============================================================================
# TRCG 轻量根因诊断摘要（确定性、无图算法）
# ============================================================================

TRCG_CONFLICT_RESOURCES = ('R_pad', 'R3', 'R_range_test')
_TRCG_RES_KEY_MAP = {
    'R_pad': 'pad_util',
    'R3': 'r3_util',
    'R_range_test': 'range_test_util',
}


@dataclass
class TRCGSummary:
    """
    轻量 TRCG（Temporal-Resource Conflict Graph）根因诊断摘要。

    纯确定性计算、无图算法/GNN、顶级字段 = 8 个。
    用途：构造 LLM prompt → LLM 输出 root_cause + unlock_set。

    每个 rolling step 调用一次 build_trcg_summary() 生成。
    """
    now_slot: int
    horizon_end_slot: int
    bottleneck_pressure: Dict[str, float]           # {pad_util, r3_util, range_test_util}
    top_conflicts: List[Dict[str, Any]]              # 至多 10 条
    conflict_clusters: List[Dict[str, Any]]          # 至多 2 个
    urgent_missions: List[Dict[str, Any]]            # 至多 3 个
    disturbance_summary: Dict[str, Any]
    frozen_summary: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'now_slot': self.now_slot,
            'horizon_end_slot': self.horizon_end_slot,
            'bottleneck_pressure': self.bottleneck_pressure,
            'top_conflicts': self.top_conflicts,
            'conflict_clusters': self.conflict_clusters,
            'urgent_missions': self.urgent_missions,
            'disturbance_summary': self.disturbance_summary,
            'frozen_summary': self.frozen_summary,
        }

    def to_prompt_str(self) -> str:
        """生成用于 LLM prompt 的紧凑 JSON 文本（单行，节省 token）。"""
        import json as _json
        return _json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))


# ----- TRCG 内部辅助函数 -----

def _trcg_bottleneck_pressure(
    missions: List[Mission],
    resources: List[Resource],
    now: int,
    horizon_end: int,
    completed_ops: Set[str],
) -> Dict[str, float]:
    """
    计算 R_pad / R3 / R_range_test 的利用率（demand / effective_capacity）。
    """
    result: Dict[str, float] = {}
    for res_id in TRCG_CONFLICT_RESOURCES:
        resource = next((r for r in resources if r.resource_id == res_id), None)
        if resource is None:
            result[_TRCG_RES_KEY_MAP[res_id]] = 0.0
            continue
        demand = 0
        for mission in missions:
            for op in mission.operations:
                if op.op_id in completed_ops or op.release > horizon_end:
                    continue
                if res_id in op.resources:
                    demand += op.duration
        span = max(1, horizon_end - now)
        capacity = span * max(1, resource.capacity)
        unavail = _compute_unavailable_slots(resource.unavailable, now, horizon_end)
        effective = max(1, capacity - unavail)
        result[_TRCG_RES_KEY_MAP[res_id]] = round(min(1.0, demand / effective), 4)
    return result


def _trcg_project_intervals(
    missions: List[Mission],
    plan: Optional[PlanV2_1],
    started_ops: Set[str],
    completed_ops: Set[str],
    actual_durations: Dict[str, int],
) -> Dict[str, Tuple[int, int]]:
    """
    将 prev_plan 投影为"当前现实下的预计区间"。

    规则（逐 mission、按 op_index 顺序）：
      - 已完成 op：保留原区间（仅用于前驱推算）。
      - 已开始 op：start 不变，end = start + actual_duration；
        若 actual end > planned end → carry_delay 向后传播。
      - 未开始 op：start = planned_start + carry_delay，
        end   = start + planned_duration（从 assign 计算）。

    carry_delay 在 mission 内逐 op 累积，不会跨 mission。
    """
    if plan is None:
        return {}

    assign_map: Dict[str, OpAssignment] = {
        a.op_id: a for a in plan.op_assignments
    }
    projected: Dict[str, Tuple[int, int]] = {}

    for mission in missions:
        ops = sorted(mission.operations, key=lambda o: o.op_index)
        carry = 0  # 本 mission 累积延迟

        for op in ops:
            assign = assign_map.get(op.op_id)
            if assign is None:
                continue

            planned_dur = max(0, assign.end_slot - assign.start_slot)

            if op.op_id in completed_ops:
                projected[op.op_id] = (assign.start_slot, assign.end_slot)
                continue

            if op.op_id in started_ops:
                # 已开始：start 不变，用 actual_duration 修正 end
                s = assign.start_slot
                dur = actual_durations.get(op.op_id, planned_dur)
                e = s + dur
                extra = max(0, e - assign.end_slot)
                carry += extra
            else:
                # 未开始：向后平移 carry
                s = assign.start_slot + carry
                e = s + planned_dur

            projected[op.op_id] = (s, e)

    return projected


def _trcg_detect_conflicts(
    missions: List[Mission],
    projected: Dict[str, Tuple[int, int]],
    completed_ops: Set[str],
    near_gap_slots: int = 8,
    near_gap_weight: float = 0.6,
    max_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    在投影区间上检测资源时间冲突（R_pad / R3 / R_range_test）。

    同一 mission 内的 op 不算冲突；零时长 op 跳过。
    冲突分两类：
    1) overlap: 区间重叠
    2) near_queue: 区间不重叠但间隔 gap <= near_gap_slots（表示紧密排队、低余量）

    severity:
    - overlap: overlap_slots * (priority_a + priority_b)
    - near_queue: (near_gap_slots - gap + 1) * (priority_a + priority_b) * near_gap_weight

    返回 severity 降序排列的前 max_n 条。
    """
    mission_map = {m.mission_id: m for m in missions}

    # 按资源收集 (mission_id, start, end)
    res_entries: Dict[str, List[Tuple[str, int, int]]] = {
        r: [] for r in TRCG_CONFLICT_RESOURCES
    }

    for mission in missions:
        for op in mission.operations:
            if op.op_id in completed_ops or op.op_id not in projected:
                continue
            s, e = projected[op.op_id]
            if e <= s:
                continue
            for rid in op.resources:
                if rid in res_entries:
                    res_entries[rid].append((mission.mission_id, s, e))

    conflict_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for rid, entries in res_entries.items():
        n = len(entries)
        for i in range(n):
            m_a, s_a, e_a = entries[i]
            for j in range(i + 1, n):
                m_b, s_b, e_b = entries[j]
                if m_a == m_b:
                    continue
                key = (min(m_a, m_b), max(m_a, m_b), rid)

                overlap = min(e_a, e_b) - max(s_a, s_b)
                p_a = mission_map[m_a].priority
                p_b = mission_map[m_b].priority

                if overlap > 0:
                    candidate = {
                        'a': m_a,
                        'b': m_b,
                        'resource': rid,
                        'conflict_type': 'overlap',
                        'overlap_slots': overlap,
                        'gap_slots': 0,
                        't_range': [max(s_a, s_b), min(e_a, e_b)],
                        'severity': round(overlap * (p_a + p_b), 2),
                    }
                else:
                    gap = max(s_a, s_b) - min(e_a, e_b)
                    if gap > near_gap_slots:
                        continue
                    tightness = max(1, near_gap_slots - gap + 1)
                    candidate = {
                        'a': m_a,
                        'b': m_b,
                        'resource': rid,
                        'conflict_type': 'near_queue',
                        'overlap_slots': 0,
                        'gap_slots': gap,
                        't_range': [min(e_a, e_b), max(s_a, s_b)],
                        'severity': round(tightness * (p_a + p_b) * near_gap_weight, 2),
                    }

                prev = conflict_by_key.get(key)
                if prev is None or candidate['severity'] > prev['severity']:
                    conflict_by_key[key] = candidate

    conflicts: List[Dict[str, Any]] = list(conflict_by_key.values())
    conflicts.sort(key=lambda c: c['severity'], reverse=True)
    return conflicts[:max_n]


def _trcg_build_clusters(
    conflicts: List[Dict[str, Any]],
    max_clusters: int = 2,
) -> List[Dict[str, Any]]:
    """
    从 top_conflicts 构造至多 *max_clusters* 个冲突簇。

    center = 加权度数（Σ severity）最大的 mission。
    members = center + 其在冲突图中的直接邻居（按加权度数降序）。
    同一 mission 不会出现在两个簇中。
    """
    if not conflicts:
        return []

    degree: Dict[str, float] = {}
    neighbors: Dict[str, Set[str]] = {}
    for c in conflicts:
        for m in (c['a'], c['b']):
            degree[m] = degree.get(m, 0.0) + c['severity']
            neighbors.setdefault(m, set())
        neighbors[c['a']].add(c['b'])
        neighbors[c['b']].add(c['a'])

    clusters: List[Dict[str, Any]] = []
    used: Set[str] = set()

    for center in sorted(degree, key=degree.get, reverse=True):
        if center in used or len(clusters) >= max_clusters:
            break
        members = [center] + sorted(
            (n for n in neighbors.get(center, set()) if n not in used),
            key=lambda m: degree.get(m, 0),
            reverse=True,
        )
        clusters.append({
            'center_mission_id': center,
            'members': members,
            'score': round(degree[center], 2),
        })
        used.update(members)

    return clusters


def _trcg_find_urgent(
    missions: List[Mission],
    plan: Optional[PlanV2_1],
    now: int,
    completed_ops: Set[str],
    max_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    找出最紧迫的 mission（Op6 窗口余裕 / due 接近 / 现有 delay）。

    urgency_score = due_slack + 0.5 * window_slack − 2 * current_delay
    越小越紧迫。返回前 max_n 条。
    """
    assign_map: Dict[str, OpAssignment] = {}
    if plan:
        assign_map = {a.op_id: a for a in plan.op_assignments}

    urgents: List[Dict[str, Any]] = []

    for mission in missions:
        launch = mission.get_launch_op()
        if not launch or launch.op_id in completed_ops:
            continue

        due_slack = mission.due - now

        # Launch 最近可用窗口距 now 的间隔
        win_slack = 9999
        if launch.time_windows:
            for ws, we in sorted(launch.time_windows):
                if we > now:
                    win_slack = max(0, ws - now)
                    break
            else:
                win_slack = 0  # 所有窗口已过

        # 当前计划下的 delay
        current_delay = 0
        assign = assign_map.get(launch.op_id)
        if assign:
            current_delay = max(0, assign.start_slot - mission.due)

        score = due_slack + 0.5 * win_slack - 2.0 * current_delay

        urgents.append({
            'mission_id': mission.mission_id,
            'due_slot': mission.due,
            'due_slack_slots': due_slack,
            'window_slack_slots': win_slack if win_slack < 9999 else -1,
            'current_delay_slots': current_delay,
            'priority': mission.priority,
            'urgency_score': round(score, 1),
        })

    urgents.sort(key=lambda u: u['urgency_score'])
    return urgents[:max_n]


def _trcg_disturbance_summary(
    missions: List[Mission],
    resources: List[Resource],
    now: int,
    horizon_end: int,
    completed_ops: Set[str],
    prev_window_slots: Optional[Dict[str, Set[int]]],
    actual_durations: Dict[str, int],
) -> Dict[str, Any]:
    """
    扰动摘要：range_loss_pct / pad_outage_active / duration_volatility_level。
    """
    # ---- range_loss_pct: Launch 窗口 slot 损失比例 ----
    total_now = 0
    total_prev = 0
    for mission in missions:
        launch = mission.get_launch_op()
        if not launch or launch.op_id in completed_ops:
            continue
        cur_slots: Set[int] = set()
        for ws, we in launch.time_windows:
            for s in range(max(ws, now), min(we, horizon_end) + 1):
                cur_slots.add(s)
        total_now += len(cur_slots)
        if prev_window_slots and mission.mission_id in prev_window_slots:
            total_prev += len(prev_window_slots[mission.mission_id])

    range_loss = max(0.0, 1.0 - total_now / total_prev) if total_prev > 0 else 0.0

    # ---- pad_outage_active ----
    pad_outage_active = False
    pad = next((r for r in resources if r.resource_id == 'R_pad'), None)
    if pad:
        for ua_s, ua_e in pad.unavailable:
            if ua_s <= horizon_end and ua_e >= now:
                pad_outage_active = True
                break

    # ---- duration_volatility_level ----
    n_mod = len(actual_durations)
    if n_mod >= 5:
        vol = 'high'
    elif n_mod >= 2:
        vol = 'medium'
    elif n_mod >= 1:
        vol = 'low'
    else:
        vol = 'none'

    return {
        'range_loss_pct': round(range_loss, 3),
        'pad_outage_active': pad_outage_active,
        'duration_volatility_level': vol,
    }


# ----- TRCG 主入口 -----

def build_trcg_summary(
    missions: List[Mission],
    resources: List[Resource],
    plan: Optional[PlanV2_1],
    now: int,
    config: Config,
    started_ops: Set[str],
    completed_ops: Set[str],
    actual_durations: Dict[str, int],
    frozen_ops: Dict[str, OpAssignment],
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
) -> TRCGSummary:
    """
    构造轻量 TRCG 根因诊断摘要（确定性，每 rolling step 调用一次）。

    Parameters
    ----------
    plan : 上一轮求解器产出的计划（policy.decide 调用时即 state.current_plan）。
    frozen_ops : 当前已计算好的冻结 op 字典（由 compute_frozen_ops 给出）。
    prev_window_slots : 上一步窗口 slot 集合（用于计算 range_loss_pct）。

    Returns
    -------
    TRCGSummary  —— 8 个顶级字段，可直接 .to_dict() 或 .to_prompt_str() 喂给 LLM。
    """
    horizon_end = now + config.horizon_slots

    # 1. 瓶颈压力
    pressure = _trcg_bottleneck_pressure(
        missions, resources, now, horizon_end, completed_ops,
    )

    # 2. 投影 → 冲突检测
    projected = _trcg_project_intervals(
        missions, plan, started_ops, completed_ops, actual_durations,
    )
    conflicts = _trcg_detect_conflicts(missions, projected, completed_ops)

    # 3. 冲突簇
    clusters = _trcg_build_clusters(conflicts)

    # 4. 紧迫 mission
    urgent = _trcg_find_urgent(missions, plan, now, completed_ops)

    # 5. 扰动摘要
    dist_summary = _trcg_disturbance_summary(
        missions, resources, now, horizon_end,
        completed_ops, prev_window_slots, actual_durations,
    )

    # 6. 冻结摘要
    frozen_summary = {
        'num_started_ops': len(started_ops),
        'num_frozen_ops': len(frozen_ops),
        'frozen_horizon_slots': config.freeze_horizon,
    }

    return TRCGSummary(
        now_slot=now,
        horizon_end_slot=horizon_end,
        bottleneck_pressure=pressure,
        top_conflicts=conflicts,
        conflict_clusters=clusters,
        urgent_missions=urgent,
        disturbance_summary=dist_summary,
        frozen_summary=frozen_summary,
    )


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    from scenario import generate_scenario
    
    print("=== Features Module Test ===\n")
    
    scenario = generate_scenario(seed=42)
    features, window_slots = compute_state_features_ops(
        missions=scenario.missions,
        resources=scenario.resources,
        current_plan=None,
        now=0,
        config=DEFAULT_CONFIG,
        completed_ops=set()
    )

    print("Initial features:")
    for key, value in features.to_dict().items():
        print(f"  {key}: {value}")
