"""
贪心策略 - 不调用 CP-SAT，直接生成可行计划

特点：
- 不使用 CP-SAT 优化
- 按 EDF (Earliest Deadline First) 或窗口起点排序
- 依次选择最早可行的 pad + slot
- 计算速度快，但方案质量较差

Baseline B3：用于对比 CP-SAT 优化的价值
"""

from typing import Optional, Tuple, List, Dict, Set, Any
from dataclasses import dataclass

from policies.base import BasePolicy, MetaParams
from disturbance import SimulationState
from config import Config
from solver_cpsat import (
    Task, Pad, Plan, TaskAssignment,
    PlanV2_1, OpAssignment, Mission, Resource, Operation
)


@dataclass
class PadSchedule:
    """单个 Pad 的占用时间线"""
    pad_id: str
    occupied_intervals: List[Tuple[int, int]]  # [(start, end), ...] 已占用区间
    
    def is_available(self, start: int, end: int, unavailable: List[Tuple[int, int]]) -> bool:
        """检查 [start, end] 是否可用"""
        # 检查与已占用区间冲突
        for occ_start, occ_end in self.occupied_intervals:
            if not (end <= occ_start or start >= occ_end):
                return False
        
        # 检查与不可用区间冲突
        for ua_start, ua_end in unavailable:
            if not (end <= ua_start or start > ua_end):
                return False
        
        return True
    
    def add_occupation(self, start: int, end: int):
        """添加占用"""
        self.occupied_intervals.append((start, end))
        self.occupied_intervals.sort()


def find_earliest_slot_in_windows(
    task: Task,
    pad_schedule: PadSchedule,
    pad_unavailable: List[Tuple[int, int]],
    now: int
) -> Optional[int]:
    """
    在任务窗口内找到最早可用的发射时刻
    
    Args:
        task: 任务
        pad_schedule: Pad 占用情况
        pad_unavailable: Pad 不可用区间
        now: 当前时刻
    
    Returns:
        最早可用的 launch_slot，或 None
    """
    # 从 release + duration 开始搜索
    min_start = max(now, task.release)
    min_launch = min_start + task.duration
    
    for win_start, win_end in sorted(task.windows):
        # 跳过已过去的窗口
        if win_end < min_launch:
            continue
        
        # 在此窗口内搜索
        search_start = max(win_start, min_launch)
        search_end = win_end
        
        for launch in range(search_start, search_end + 1):
            start = launch - task.duration
            if start < min_start:
                continue
            
            # 检查是否可用
            if pad_schedule.is_available(start, launch, pad_unavailable):
                return launch
    
    return None


class GreedyPolicy(BasePolicy):
    """
    贪心策略 (Baseline B3)
    
    算法：
    1. 按 due 时间（EDF）或窗口起点对任务排序
    2. 对每个任务，尝试每个 pad
    3. 在每个 pad 上找最早可行的 launch slot
    4. 选择最优的 (pad, slot) 组合
    5. 直接生成 Plan，不调用 CP-SAT
    
    特点：
    - O(n * p * w) 复杂度，速度快
    - 不考虑全局最优，可能产生较差方案
    - 无稳定性控制（每次完全重排）
    """
    
    def __init__(
        self,
        sort_by: str = "due",  # "due" or "window_start"
        prefer_pad_switch: bool = False,  # 是否倾向于保持原 pad
        policy_name: str = "greedy"
    ):
        """
        Args:
            sort_by: 排序方式
                - "due": EDF，按 due 时间升序
                - "window_start": 按第一个窗口起点升序
            prefer_pad_switch: 是否在同等条件下倾向于保持原 pad
            policy_name: 策略名称
        """
        self._sort_by = sort_by
        self._prefer_pad_switch = prefer_pad_switch
        self._policy_name = policy_name
        
        # 统计
        self._call_count = 0
        self._total_tasks_scheduled = 0
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[Optional[MetaParams], Optional[Plan]]:
        """
        直接生成贪心计划
        
        Returns:
            (None, Plan) - 第一项为 None 表示不使用 CP-SAT
        """
        self._call_count += 1
        if hasattr(state, 'missions') and hasattr(state, 'resources'):
            plan = self._greedy_plan_ops(state, now, config)
            return None, plan

        
        # 获取需要排程的任务
        horizon_end = now + config.horizon_slots
        tasks_to_schedule = self._get_tasks_to_schedule(state, now, horizon_end)
        
        if not tasks_to_schedule:
            return None, Plan(timestamp=now, assignments=[])
        
        # 初始化每个 pad 的占用情况
        pad_schedules: Dict[str, PadSchedule] = {}
        for pad in state.pads:
            pad_schedules[pad.pad_id] = PadSchedule(pad_id=pad.pad_id, occupied_intervals=[])
        
        # 添加已开始任务的占用
        if state.current_plan:
            for assign in state.current_plan.assignments:
                if assign.task_id in state.started_tasks and assign.task_id not in state.completed_tasks:
                    pad_schedules[assign.pad_id].add_occupation(assign.start_slot, assign.launch_slot)
        
        # 排序任务
        sorted_tasks = self._sort_tasks(tasks_to_schedule)
        
        # 贪心分配
        assignments: List[TaskAssignment] = []
        
        for task in sorted_tasks:
            best_assignment = self._find_best_assignment(
                task, pad_schedules, state.pads, now, state.current_plan
            )
            
            if best_assignment:
                assignments.append(best_assignment)
                pad_schedules[best_assignment.pad_id].add_occupation(
                    best_assignment.start_slot, best_assignment.launch_slot
                )
                self._total_tasks_scheduled += 1
        
        # 添加已开始但未完成的任务（保持不变）
        if state.current_plan:
            for assign in state.current_plan.assignments:
                if assign.task_id in state.started_tasks and assign.task_id not in state.completed_tasks:
                    # 检查是否已在 assignments 中
                    if not any(a.task_id == assign.task_id for a in assignments):
                        assignments.append(assign)
        
        plan = Plan(timestamp=now, assignments=assignments)
        return None, plan

    def _greedy_plan_ops(
        self,
        state: Any,
        now: int,
        config: Config
    ) -> PlanV2_1:
        """Greedy plan for V2.1 ops without CP-SAT."""
        horizon_end = min(now + config.horizon_slots, config.sim_total_slots)
        missions = state.get_schedulable_missions(horizon_end)

        if not missions:
            return PlanV2_1(timestamp=now, op_assignments=[])

        resource_schedules = self._init_resource_schedules(state)
        assignments: List[OpAssignment] = []
        fixed_end: Dict[str, int] = {}

        if state.current_plan:
            for assign in state.current_plan.op_assignments:
                if assign.op_id in state.completed_ops or assign.op_id in state.started_ops:
                    assignments.append(assign)
                    fixed_end[assign.op_id] = assign.end_slot
                    self._reserve_interval(resource_schedules, assign.resources, assign.start_slot, assign.end_slot)

        missions_sorted = sorted(
            missions, key=lambda m: (m.due, m.release, m.mission_id)
        )

        for mission in missions_sorted:
            ops_by_idx = {op.op_index: op for op in mission.operations}
            mission_release = state.actual_releases.get(mission.mission_id, mission.release)
            prev_end = max(now, mission_release)

            for op_index in sorted(ops_by_idx.keys()):
                op = ops_by_idx[op_index]
                if op.op_id in fixed_end:
                    prev_end = max(prev_end, fixed_end[op.op_id])
                    continue

                op_release = max(op.release, mission_release)
                earliest = max(now, op_release, prev_end)

                if op_index == 4 and 5 in ops_by_idx and 6 in ops_by_idx:
                    op5 = ops_by_idx[5]
                    op6 = ops_by_idx[6]
                    if op5.op_id not in fixed_end and op6.op_id not in fixed_end:
                        start4 = self._find_block_start(
                            resource_schedules,
                            earliest,
                            op,
                            op5,
                            op6,
                            config.sim_total_slots
                        )
                        if start4 is not None:
                            end4 = start4 + op.duration
                            start5 = end4
                            end5 = start5 + op5.duration
                            start6 = end5
                            end6 = start6 + op6.duration
                            assignments.append(OpAssignment(
                                op_id=op.op_id,
                                mission_id=op.mission_id,
                                op_index=op.op_index,
                                resources=op.resources,
                                start_slot=start4,
                                end_slot=end4
                            ))
                            assignments.append(OpAssignment(
                                op_id=op5.op_id,
                                mission_id=op5.mission_id,
                                op_index=op5.op_index,
                                resources=op5.resources,
                                start_slot=start5,
                                end_slot=end5
                            ))
                            assignments.append(OpAssignment(
                                op_id=op6.op_id,
                                mission_id=op6.mission_id,
                                op_index=op6.op_index,
                                resources=op6.resources,
                                start_slot=start6,
                                end_slot=end6
                            ))
                            fixed_end[op.op_id] = end4
                            fixed_end[op5.op_id] = end5
                            fixed_end[op6.op_id] = end6
                            self._reserve_interval(resource_schedules, op.resources, start4, end4)
                            self._reserve_interval(resource_schedules, op5.resources, start5, end5)
                            self._reserve_interval(resource_schedules, op6.resources, start6, end6)
                            prev_end = end6
                            continue

                if op_index == 6 and op.time_windows:
                    start = self._find_earliest_start_in_windows(
                        resource_schedules,
                        op.resources,
                        earliest,
                        op.duration,
                        op.time_windows,
                        config.sim_total_slots
                    )
                else:
                    start = self._find_earliest_start(
                        resource_schedules,
                        op.resources,
                        earliest,
                        op.duration,
                        config.sim_total_slots
                    )

                if start is None:
                    continue

                end = start + op.duration
                assignments.append(OpAssignment(
                    op_id=op.op_id,
                    mission_id=op.mission_id,
                    op_index=op.op_index,
                    resources=op.resources,
                    start_slot=start,
                    end_slot=end
                ))
                fixed_end[op.op_id] = end
                self._reserve_interval(resource_schedules, op.resources, start, end)
                prev_end = end

        return PlanV2_1(timestamp=now, op_assignments=assignments)

    def _init_resource_schedules(self, state: Any) -> Dict[str, List[Tuple[int, int]]]:
        schedules: Dict[str, List[Tuple[int, int]]] = {}
        for res in state.resources:
            intervals = []
            for start, end in res.unavailable:
                intervals.append((start, end + 1))
            schedules[res.resource_id] = intervals
        return schedules

    def _is_available(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        resources: List[str],
        start: int,
        end: int
    ) -> bool:
        for res_id in resources:
            for occ_start, occ_end in schedules.get(res_id, []):
                if not (end <= occ_start or start >= occ_end):
                    return False
        return True

    def _reserve_interval(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        resources: List[str],
        start: int,
        end: int
    ) -> None:
        for res_id in resources:
            schedules.setdefault(res_id, []).append((start, end))

    def _find_earliest_start(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        resources: List[str],
        earliest: int,
        duration: int,
        max_time: int
    ) -> Optional[int]:
        t = earliest
        while t + duration <= max_time:
            if self._is_available(schedules, resources, t, t + duration):
                return t
            t += 1
        return None

    def _find_earliest_start_in_windows(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        resources: List[str],
        earliest: int,
        duration: int,
        windows: List[Tuple[int, int]],
        max_time: int
    ) -> Optional[int]:
        for ws, we in sorted(windows):
            t = max(earliest, ws)
            while t + duration <= we and t + duration <= max_time:
                if self._is_available(schedules, resources, t, t + duration):
                    return t
                t += 1
        return None

    def _block_available(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        op4: Operation,
        op5: Operation,
        op6: Operation,
        start4: int
    ) -> bool:
        end4 = start4 + op4.duration
        start5 = end4
        end5 = start5 + op5.duration
        start6 = end5
        end6 = start6 + op6.duration

        if not self._is_available(schedules, op4.resources, start4, end4):
            return False
        if not self._is_available(schedules, op5.resources, start5, end5):
            return False
        if not self._is_available(schedules, op6.resources, start6, end6):
            return False
        return True

    def _find_block_start(
        self,
        schedules: Dict[str, List[Tuple[int, int]]],
        earliest: int,
        op4: Operation,
        op5: Operation,
        op6: Operation,
        max_time: int
    ) -> Optional[int]:
        if not op6.time_windows:
            return None
        total_pre = op4.duration + op5.duration
        for ws, we in sorted(op6.time_windows):
            t6 = max(ws, earliest + total_pre)
            while t6 + op6.duration <= we and t6 + op6.duration <= max_time:
                start4 = t6 - total_pre
                if self._block_available(schedules, op4, op5, op6, start4):
                    return start4
                t6 += 1
        return None
    
    def _get_tasks_to_schedule(
        self,
        state: SimulationState,
        now: int,
        horizon_end: int
    ) -> List[Task]:
        """获取需要排程的任务"""
        tasks = []
        for task in state.tasks:
            # 跳过已完成
            if task.task_id in state.completed_tasks:
                continue
            # 跳过已开始
            if task.task_id in state.started_tasks:
                continue
            # 检查 release
            actual_release = state.actual_releases.get(task.task_id, task.release)
            if actual_release <= horizon_end:
                tasks.append(task)
        return tasks
    
    def _sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """按策略排序任务"""
        if self._sort_by == "due":
            # EDF: Earliest Deadline First
            return sorted(tasks, key=lambda t: (t.due, t.priority * -1))
        elif self._sort_by == "window_start":
            # 按第一个窗口起点
            return sorted(tasks, key=lambda t: (
                min(w[0] for w in t.windows) if t.windows else float('inf'),
                t.due
            ))
        else:
            return tasks
    
    def _find_best_assignment(
        self,
        task: Task,
        pad_schedules: Dict[str, PadSchedule],
        pads: List[Pad],
        now: int,
        prev_plan: Optional[Plan]
    ) -> Optional[TaskAssignment]:
        """为任务找最佳分配"""
        best: Optional[TaskAssignment] = None
        best_launch: int = float('inf')
        
        # 获取之前的分配（用于判断是否保持原 pad）
        prev_pad_id = None
        if prev_plan and self._prefer_pad_switch:
            prev_assign = prev_plan.get_assignment(task.task_id)
            if prev_assign:
                prev_pad_id = prev_assign.pad_id
        
        for pad in pads:
            pad_schedule = pad_schedules[pad.pad_id]
            
            launch = find_earliest_slot_in_windows(
                task, pad_schedule, pad.unavailable, now
            )
            
            if launch is not None:
                # 判断是否更优
                is_better = False
                
                if best is None:
                    is_better = True
                elif launch < best_launch:
                    is_better = True
                elif launch == best_launch and self._prefer_pad_switch:
                    # 同样早，优先选择原 pad
                    if pad.pad_id == prev_pad_id:
                        is_better = True
                
                if is_better:
                    best_launch = launch
                    best = TaskAssignment(
                        task_id=task.task_id,
                        pad_id=pad.pad_id,
                        launch_slot=launch,
                        start_slot=launch - task.duration,
                        end_slot=launch
                    )
        
        return best
    
    def reset(self) -> None:
        """重置策略状态"""
        self._call_count = 0
        self._total_tasks_scheduled = 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "policy_name": self._policy_name,
            "sort_by": self._sort_by,
            "call_count": self._call_count,
            "total_tasks_scheduled": self._total_tasks_scheduled
        }
    
    def __repr__(self) -> str:
        return f"GreedyPolicy(name={self.name}, sort_by={self._sort_by})"


class EDFGreedyPolicy(GreedyPolicy):
    """EDF 贪心策略的快捷创建"""
    
    def __init__(self, policy_name: str = "greedy_edf"):
        super().__init__(sort_by="due", prefer_pad_switch=True, policy_name=policy_name)


class WindowGreedyPolicy(GreedyPolicy):
    """按窗口起点贪心的快捷创建"""
    
    def __init__(self, policy_name: str = "greedy_window"):
        super().__init__(sort_by="window_start", prefer_pad_switch=False, policy_name=policy_name)
