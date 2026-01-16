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

from solver_cpsat import Task, Pad, Plan, TaskAssignment
from config import Config, DEFAULT_CONFIG


@dataclass
class StateFeatures:
    """状态特征（作为 LLM/MockLLM 输入）"""
    # 窗口损失
    window_loss_pct: float                    # [0, 1] 窗口减少比例
    window_remaining_pct: float               # [0, 1] 剩余窗口比例
    
    # Pad 可用性
    pad_outage_overlap_hours: float           # 未来 H 内 outage 总时长（小时）
    pad_outage_task_count: int                # 受 outage 影响的任务数
    
    # 延迟估算
    delay_increase_minutes: float             # 预估延误增加（分钟）
    current_total_delay_minutes: float        # 当前总延迟（分钟）
    
    # 任务状态
    num_tasks_in_horizon: int                 # 视野内任务数
    num_urgent_tasks: int                     # 紧急任务数（即将到期）
    completed_rate: float                     # 已完成比例
    
    # 稳定性状态
    recent_shift_count: int                   # 最近一次重排的时间变化数
    recent_switch_count: int                  # 最近一次重排的 pad 切换数
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（便于 JSON 序列化）"""
        return {
            "window_loss_pct": round(self.window_loss_pct, 4),
            "window_remaining_pct": round(self.window_remaining_pct, 4),
            "pad_outage_overlap_hours": round(self.pad_outage_overlap_hours, 2),
            "pad_outage_task_count": self.pad_outage_task_count,
            "delay_increase_minutes": round(self.delay_increase_minutes, 1),
            "current_total_delay_minutes": round(self.current_total_delay_minutes, 1),
            "num_tasks_in_horizon": self.num_tasks_in_horizon,
            "num_urgent_tasks": self.num_urgent_tasks,
            "completed_rate": round(self.completed_rate, 4),
            "recent_shift_count": self.recent_shift_count,
            "recent_switch_count": self.recent_switch_count
        }


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


def compute_state_features(
    tasks: List[Task],
    pads: List[Pad],
    current_plan: Optional[Plan],
    now: int,
    config: Config,
    completed_tasks: Optional[Set[str]] = None,
    prev_window_slots: Optional[Dict[str, Set[int]]] = None,
    recent_shifts: int = 0,
    recent_switches: int = 0
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
    window_loss_pct, window_remaining_pct, curr_window_slots = compute_window_loss_pct(
        tasks, now, horizon_end, prev_window_slots, completed_tasks
    )
    
    # Pad outage
    outage_hours, outage_task_count = compute_pad_outage_overlap(
        pads, now, horizon_end, tasks, current_plan, config.slot_minutes
    )
    
    # 延迟估算
    delay_increase, current_delay = compute_delay_increase(
        tasks, current_plan, now, config.slot_minutes, completed_tasks
    )
    
    # 视野内任务数
    tasks_in_horizon = [
        t for t in tasks 
        if t.task_id not in completed_tasks and t.release <= horizon_end
    ]
    
    # 紧急任务
    urgent_count = count_urgent_tasks(tasks, now, completed_tasks=completed_tasks)
    
    # 完成率
    total = len(tasks)
    completed_rate = len(completed_tasks) / total if total > 0 else 0.0
    
    features = StateFeatures(
        window_loss_pct=window_loss_pct,
        window_remaining_pct=window_remaining_pct,
        pad_outage_overlap_hours=outage_hours,
        pad_outage_task_count=outage_task_count,
        delay_increase_minutes=delay_increase,
        current_total_delay_minutes=current_delay,
        num_tasks_in_horizon=len(tasks_in_horizon),
        num_urgent_tasks=urgent_count,
        completed_rate=completed_rate,
        recent_shift_count=recent_shifts,
        recent_switch_count=recent_switches
    )
    
    return features, curr_window_slots


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    from scenario import generate_scenario
    
    print("=== Features Module Test ===\n")
    
    scenario = generate_scenario(seed=42)
    
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
