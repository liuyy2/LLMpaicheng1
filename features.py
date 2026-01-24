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
        op6 = mission.get_operation(6)
        if not op6:
            continue

        slots = set()
        for win_start, win_end in op6.time_windows:
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

    current_total_delay = 0
    potential_delay_increase = 0

    for assignment in current_plan.op_assignments:
        if assignment.op_index != 6:
            continue

        mission = mission_map.get(assignment.mission_id)
        if mission is None:
            continue

        op6 = mission.get_operation(6)
        if not op6 or op6.op_id in completed_ops:
            continue

        current_delay = max(0, assignment.end_slot - mission.due)
        current_total_delay += current_delay

        in_window = any(
            assignment.start_slot >= ws and assignment.end_slot <= we
            for ws, we in op6.time_windows
        )

        if not in_window and assignment.start_slot > now:
            earliest_valid = None
            for win_start, win_end in op6.time_windows:
                if win_start >= assignment.start_slot and win_start + op6.duration <= win_end:
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
        op6 = mission.get_operation(6)
        if op6 and op6.op_id in completed_ops:
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
# 模块测试
# ============================================================================

if __name__ == "__main__":
    from scenario import generate_scenario
    
    print("=== Features Module Test ===\n")
    
    scenario = generate_scenario(seed=42)

    if getattr(scenario, 'schema_version', 'v1') == 'v2_1':
        features, window_slots = compute_state_features_ops(
            missions=scenario.missions,
            resources=scenario.resources,
            current_plan=None,
            now=0,
            config=DEFAULT_CONFIG
        )
    else:
        features, window_slots = compute_state_features(
            tasks=scenario.tasks,
            pads=scenario.pads,
            current_plan=None,
            now=0,
            config=DEFAULT_CONFIG
        )
    
    print("Initial features:")
    for key, value in features.to_dict().items():
        print(f"  {key}: {value}")