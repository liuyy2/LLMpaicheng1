"""
扰动应用模块 - 在仿真过程中动态修改状态
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import copy

from solver_cpsat import Task, Pad, Plan, TaskAssignment
from scenario import DisturbanceEvent


@dataclass
class SimulationState:
    """仿真状态（可变）"""
    now: int                                      # 当前时刻
    tasks: List[Task]                             # 任务列表（含实时状态）
    pads: List[Pad]                               # Pad 列表（含不可用区间）
    current_plan: Optional[Plan]                  # 当前有效计划
    
    # 任务状态跟踪
    started_tasks: Set[str] = field(default_factory=set)     # 已开始的任务
    completed_tasks: Set[str] = field(default_factory=set)   # 已完成的任务
    
    # 扰动跟踪
    applied_events: Set[int] = field(default_factory=set)    # 已应用的事件索引
    
    # 实际值记录（扰动后）
    actual_durations: Dict[str, int] = field(default_factory=dict)   # task_id -> actual duration
    actual_releases: Dict[str, int] = field(default_factory=dict)    # task_id -> actual release
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None
    
    def get_pad(self, pad_id: str) -> Optional[Pad]:
        for p in self.pads:
            if p.pad_id == pad_id:
                return p
        return None
    
    def get_active_tasks(self) -> List[Task]:
        """获取活跃任务（未完成）"""
        return [t for t in self.tasks if t.task_id not in self.completed_tasks]
    
    def get_schedulable_tasks(self, horizon_end: int) -> List[Task]:
        """获取可排程任务（未完成、未开始、release <= horizon_end）"""
        schedulable = []
        for t in self.tasks:
            if t.task_id in self.completed_tasks:
                continue
            if t.task_id in self.started_tasks:
                continue
            # 使用实际 release
            actual_rel = self.actual_releases.get(t.task_id, t.release)
            if actual_rel <= horizon_end:
                schedulable.append(t)
        return schedulable


def apply_disturbance(
    state: SimulationState,
    now: int,
    events: List[DisturbanceEvent],
    last_now: int = 0
) -> SimulationState:
    """
    应用 (last_now, now] 区间内触发的所有扰动事件
    
    Args:
        state: 当前仿真状态（会被修改）
        now: 当前时刻 slot
        events: 扰动事件列表（已按时间排序）
        last_now: 上一次的时刻
    
    Returns:
        更新后的 state
    """
    for idx, event in enumerate(events):
        # 跳过已应用的事件
        if idx in state.applied_events:
            continue
        
        # 跳过未到触发时刻的事件
        if event.trigger_time > now:
            continue
        
        # 跳过 last_now 之前的事件（仿真开始时 last_now=0 的事件除外）
        if event.trigger_time <= last_now and last_now > 0:
            continue
        
        # 应用事件
        if event.event_type == "weather":
            _apply_weather_disturbance(state, event)
        elif event.event_type == "pad_outage":
            _apply_pad_outage(state, event)
        elif event.event_type == "duration":
            _apply_duration_disturbance(state, event)
        elif event.event_type == "release":
            _apply_release_disturbance(state, event)
        
        state.applied_events.add(idx)
    
    return state


def _apply_weather_disturbance(state: SimulationState, event: DisturbanceEvent) -> None:
    """
    应用天气扰动 - 删除受影响时间段内的 windows slots
    """
    params = event.params
    delete_ratio = params.get("delete_ratio", 0.3)
    affected_start = params.get("affected_start", event.trigger_time)
    affected_end = params.get("affected_end", event.trigger_time + 12)
    
    for task in state.tasks:
        # 跳过已完成或已开始的任务
        if task.task_id in state.completed_tasks or task.task_id in state.started_tasks:
            continue
        
        # 修改 windows
        new_windows = []
        for win_start, win_end in task.windows:
            # 检查是否与受影响区间重叠
            if win_end < affected_start or win_start > affected_end:
                # 不重叠，保留
                new_windows.append((win_start, win_end))
            else:
                # 有重叠，根据比例缩减
                # 策略：按比例删除受影响部分
                overlap_start = max(win_start, affected_start)
                overlap_end = min(win_end, affected_end)
                overlap_len = overlap_end - overlap_start + 1
                delete_len = int(overlap_len * delete_ratio)
                
                # 保留不受影响的部分
                if win_start < affected_start:
                    new_windows.append((win_start, affected_start - 1))
                if win_end > affected_end:
                    new_windows.append((affected_end + 1, win_end))
                
                # 受影响部分按比例保留
                if delete_len < overlap_len:
                    # 保留后半部分
                    kept_start = overlap_start + delete_len
                    if kept_start <= overlap_end:
                        new_windows.append((kept_start, overlap_end))
        
        # 合并相邻窗口
        if new_windows:
            task.windows = _merge_windows(new_windows)
        # 如果窗口全被删除，保留原始（避免不可行）
        if not task.windows:
            task.windows = [(affected_end + 1, affected_end + 30)]


def _merge_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并相邻或重叠的窗口"""
    if not windows:
        return []
    
    sorted_wins = sorted(windows, key=lambda x: x[0])
    merged = [sorted_wins[0]]
    
    for start, end in sorted_wins[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            # 合并
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def _apply_pad_outage(state: SimulationState, event: DisturbanceEvent) -> None:
    """
    应用 Pad 不可用扰动
    """
    pad_id = event.target_id
    params = event.params
    outage_start = params.get("outage_start", event.trigger_time + 1)
    outage_end = params.get("outage_end", event.trigger_time + 10)
    
    pad = state.get_pad(pad_id)
    if pad:
        # 添加不可用区间（避免重复）
        new_interval = (outage_start, outage_end)
        if new_interval not in pad.unavailable:
            pad.unavailable.append(new_interval)
            # 合并重叠区间
            pad.unavailable = _merge_windows(pad.unavailable)


def _apply_duration_disturbance(state: SimulationState, event: DisturbanceEvent) -> None:
    """
    应用 duration 扰动
    """
    task_id = event.target_id
    params = event.params
    multiplier = params.get("multiplier", 1.0)
    
    task = state.get_task(task_id)
    if task and task_id not in state.started_tasks:
        # 计算新 duration
        new_duration = max(1, int(round(task.duration * multiplier)))
        state.actual_durations[task_id] = new_duration
        task.duration = new_duration


def _apply_release_disturbance(state: SimulationState, event: DisturbanceEvent) -> None:
    """
    应用 release 扰动
    """
    task_id = event.target_id
    params = event.params
    new_release = params.get("new_release", 0)
    
    task = state.get_task(task_id)
    if task and task_id not in state.started_tasks:
        state.actual_releases[task_id] = new_release
        task.release = new_release


def update_task_status(
    state: SimulationState,
    now: int
) -> SimulationState:
    """
    更新任务状态：检查哪些任务已开始、已完成
    
    Args:
        state: 当前状态
        now: 当前时刻
    
    Returns:
        更新后的状态
    """
    if state.current_plan is None:
        return state
    
    for assignment in state.current_plan.assignments:
        task_id = assignment.task_id
        
        # 已完成的跳过
        if task_id in state.completed_tasks:
            continue
        
        # 检查是否已开始
        if assignment.start_slot <= now:
            state.started_tasks.add(task_id)
        
        # 检查是否已完成（launch_slot 已过）
        if assignment.launch_slot <= now:
            state.completed_tasks.add(task_id)
            state.started_tasks.add(task_id)  # 确保也在 started 中
    
    return state


def get_frozen_assignments(
    state: SimulationState,
    now: int,
    freeze_horizon: int
) -> Dict[str, TaskAssignment]:
    """
    获取需要冻结的任务分配
    
    规则：
    1. 已开始的任务必须冻结
    2. 在 freeze_horizon 内即将开始的任务也冻结
    
    Args:
        state: 当前状态
        now: 当前时刻
        freeze_horizon: 冻结视野
    
    Returns:
        冻结任务字典 {task_id: TaskAssignment}
    """
    if state.current_plan is None:
        return {}
    
    frozen = {}
    freeze_end = now + freeze_horizon
    
    for assignment in state.current_plan.assignments:
        task_id = assignment.task_id
        
        # 已完成的不需要冻结
        if task_id in state.completed_tasks:
            continue
        
        # 已开始的必须冻结
        if task_id in state.started_tasks:
            frozen[task_id] = assignment
            continue
        
        # 在冻结视野内即将开始的也冻结
        if assignment.start_slot <= freeze_end and assignment.start_slot > now:
            frozen[task_id] = assignment
    
    return frozen


def check_plan_feasibility(
    state: SimulationState,
    now: int
) -> Tuple[bool, List[str]]:
    """
    检查当前计划是否因扰动变得不可行
    
    Returns:
        (is_feasible, reasons)
    """
    if state.current_plan is None:
        return True, []
    
    reasons = []
    
    for assignment in state.current_plan.assignments:
        task_id = assignment.task_id
        
        # 跳过已完成的
        if task_id in state.completed_tasks:
            continue
        
        task = state.get_task(task_id)
        if task is None:
            continue
        
        # 检查 1: launch 是否仍在 windows 内
        launch = assignment.launch_slot
        in_window = False
        for win_start, win_end in task.windows:
            if win_start <= launch <= win_end:
                in_window = True
                break
        
        if not in_window and launch > now:
            reasons.append(f"{task_id}: launch={launch} not in windows {task.windows}")
        
        # 检查 2: pad 是否可用
        pad = state.get_pad(assignment.pad_id)
        if pad:
            for ua_start, ua_end in pad.unavailable:
                # 检查 [start, launch] 是否与不可用区间重叠
                if not (assignment.launch_slot <= ua_start or assignment.start_slot > ua_end):
                    if assignment.start_slot > now:  # 只检查未开始的
                        reasons.append(
                            f"{task_id}: pad {assignment.pad_id} unavailable "
                            f"[{ua_start},{ua_end}] conflicts with [{assignment.start_slot},{assignment.launch_slot}]"
                        )
        
        # 检查 3: release 约束
        actual_release = state.actual_releases.get(task_id, task.release)
        if assignment.start_slot < actual_release and task_id not in state.started_tasks:
            reasons.append(f"{task_id}: start={assignment.start_slot} < release={actual_release}")
    
    return len(reasons) == 0, reasons


def create_initial_state(
    tasks: List[Task],
    pads: List[Pad]
) -> SimulationState:
    """
    创建初始仿真状态
    
    Args:
        tasks: 任务列表（会被深拷贝）
        pads: Pad 列表（会被深拷贝）
    
    Returns:
        初始状态
    """
    return SimulationState(
        now=0,
        tasks=copy.deepcopy(tasks),
        pads=copy.deepcopy(pads),
        current_plan=None,
        started_tasks=set(),
        completed_tasks=set(),
        applied_events=set(),
        actual_durations={},
        actual_releases={}
    )
