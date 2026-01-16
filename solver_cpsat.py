"""
CP-SAT 求解器模块 - 火箭发射排程
OR-Tools CP-SAT 实现，支持 pad 分配、窗口约束、冻结、稳定性惩罚
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import time

from ortools.sat.python import cp_model


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Task:
    """任务定义"""
    task_id: str                              # 唯一标识
    release: int                              # 最早可开始 slot
    duration: int                             # 占用 slot 数
    windows: List[Tuple[int, int]]            # 允许发射窗口 [(start, end), ...]，闭区间
    due: int                                  # 软截止 slot
    priority: float = 1.0                     # 优先级权重 [0.1, 1.0]
    preferred_pad: Optional[str] = None       # 偏好 pad


@dataclass
class Pad:
    """发射台定义"""
    pad_id: str                               # 唯一标识
    unavailable: List[Tuple[int, int]] = field(default_factory=list)
                                              # 不可用区间 [(start, end), ...]


@dataclass
class TaskAssignment:
    """单任务分配结果"""
    task_id: str
    pad_id: str
    launch_slot: int                          # 发射时刻
    start_slot: int                           # 开始占用 pad 时刻 = launch - duration
    end_slot: int                             # 结束时刻 = launch_slot


@dataclass
class Plan:
    """排程计划"""
    timestamp: int                            # 生成时刻 (now)
    assignments: List[TaskAssignment] = field(default_factory=list)
    
    def get_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """按 task_id 查找分配"""
        for a in self.assignments:
            if a.task_id == task_id:
                return a
        return None
    
    def to_dict(self) -> dict:
        """序列化"""
        return {
            "timestamp": self.timestamp,
            "assignments": [
                {
                    "task_id": a.task_id,
                    "pad_id": a.pad_id,
                    "launch_slot": a.launch_slot,
                    "start_slot": a.start_slot,
                    "end_slot": a.end_slot
                }
                for a in self.assignments
            ]
        }


class SolveStatus(Enum):
    """求解状态"""
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class SolverResult:
    """求解结果"""
    status: SolveStatus
    plan: Optional[Plan]
    objective_value: Optional[float] = None
    solve_time_ms: int = 0
    num_variables: int = 0
    num_constraints: int = 0
    degradation_count: int = 0                # 降级次数
    degradation_actions: List[str] = field(default_factory=list)


@dataclass
class SolverConfig:
    """求解器配置"""
    # 时间参数
    slot_minutes: int = 10
    horizon_slots: int = 144                  # 24h
    sim_total_slots: int = 432                # 72h
    
    # 冻结参数
    freeze_horizon: int = 12                  # 2h = 12 slots
    
    # 权重参数
    w_delay: float = 10.0                     # delay 权重
    w_shift: float = 1.0                      # shift 权重
    w_switch: float = 5.0                     # pad switch 权重
    
    # 求解器参数
    time_limit_seconds: float = 30.0
    num_workers: int = 4
    
    # 惩罚上界
    max_delay: int = 200                      # 最大延迟 slots
    max_shift: int = 100                      # 最大偏移 slots


# ============================================================================
# 辅助函数
# ============================================================================

def expand_windows_to_slots(
    windows: List[Tuple[int, int]],
    min_slot: int,
    max_slot: int
) -> List[int]:
    """
    将窗口列表展开为离散 slot 列表
    
    Args:
        windows: 窗口列表 [(start, end), ...]，闭区间
        min_slot: 最小有效 slot
        max_slot: 最大有效 slot
    
    Returns:
        允许的 slot 列表
    """
    allowed = set()
    for start, end in windows:
        for s in range(max(start, min_slot), min(end, max_slot) + 1):
            allowed.add(s)
    return sorted(allowed)


def get_pad_available_intervals(
    pad: Pad,
    min_slot: int,
    max_slot: int
) -> List[Tuple[int, int]]:
    """
    获取 pad 可用区间列表
    
    Args:
        pad: Pad 对象
        min_slot: 最小 slot
        max_slot: 最大 slot
    
    Returns:
        可用区间列表 [(start, end), ...]
    """
    if not pad.unavailable:
        return [(min_slot, max_slot)]
    
    # 按开始时间排序不可用区间
    unavail_sorted = sorted(pad.unavailable, key=lambda x: x[0])
    
    available = []
    current = min_slot
    
    for ua_start, ua_end in unavail_sorted:
        if ua_start > current:
            available.append((current, ua_start - 1))
        current = max(current, ua_end + 1)
    
    if current <= max_slot:
        available.append((current, max_slot))
    
    return available


# ============================================================================
# 核心求解函数
# ============================================================================

def solve_cpsat(
    now: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan],
    frozen_tasks: Optional[Dict[str, TaskAssignment]],
    config: SolverConfig
) -> SolverResult:
    """
    构建并求解 CP-SAT 模型
    
    Args:
        now: 当前时刻 slot
        tasks: 需要排程的任务列表
        pads: 可用 pad 列表
        prev_plan: 上一轮计划（用于计算 shift/switch）
        frozen_tasks: 冻结的任务分配 {task_id: TaskAssignment}
        config: 求解器配置
    
    Returns:
        SolverResult: 包含状态、计划、求解时间等
    """
    if frozen_tasks is None:
        frozen_tasks = {}
    
    start_time = time.time()
    
    # 第一次尝试完整求解
    result = _solve_with_config(now, tasks, pads, prev_plan, frozen_tasks, config)
    
    # 如果不可行，尝试降级
    if result.status == SolveStatus.INFEASIBLE:
        result = _solve_with_degradation(now, tasks, pads, prev_plan, frozen_tasks, config)
    
    result.solve_time_ms = int((time.time() - start_time) * 1000)
    return result


def _solve_with_config(
    now: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan],
    frozen_tasks: Dict[str, TaskAssignment],
    config: SolverConfig
) -> SolverResult:
    """内部求解函数（单次尝试）"""
    
    if not tasks:
        return SolverResult(
            status=SolveStatus.OPTIMAL,
            plan=Plan(timestamp=now, assignments=[]),
            objective_value=0.0
        )
    
    model = cp_model.CpModel()
    
    # 计算时间边界
    horizon_end = now + config.horizon_slots
    max_slot = config.sim_total_slots
    
    pad_ids = [p.pad_id for p in pads]
    pad_index = {p.pad_id: i for i, p in enumerate(pads)}
    num_pads = len(pads)
    
    # ========== 决策变量 ==========
    
    # launch[k]: 任务 k 的发射时刻
    launch_vars: Dict[str, cp_model.IntVar] = {}
    
    # pad_assign[k, p]: 任务 k 是否分配到 pad p
    pad_assign_vars: Dict[Tuple[str, int], cp_model.IntVar] = {}
    
    # interval[k, p]: 可选区间变量，用于 NoOverlap
    interval_vars: Dict[Tuple[str, int], cp_model.IntervalVar] = {}
    
    # delay[k]: 任务 k 的延迟（软约束）
    delay_vars: Dict[str, cp_model.IntVar] = {}
    
    # shift_abs[k]: |launch - launch_prev|
    shift_abs_vars: Dict[str, cp_model.IntVar] = {}
    
    # switched[k]: pad 是否切换
    switched_vars: Dict[str, cp_model.IntVar] = {}
    
    for task in tasks:
        tid = task.task_id
        
        # 计算 launch 的有效范围
        # start = launch - duration，start >= release，所以 launch >= release + duration
        # 同时 launch 必须在 windows 内
        min_launch = max(now, task.release + task.duration)
        max_launch = max_slot
        
        # 展开 windows 为离散 slot
        allowed_launch_slots = expand_windows_to_slots(
            task.windows, min_launch, max_launch
        )
        
        if not allowed_launch_slots:
            # 没有有效的发射窗口
            return SolverResult(status=SolveStatus.INFEASIBLE, plan=None)
        
        # 创建 launch 变量
        launch_vars[tid] = model.NewIntVar(
            min(allowed_launch_slots), 
            max(allowed_launch_slots), 
            f"launch_{tid}"
        )
        
        # 窗口约束：launch 必须在允许的 slot 中
        model.AddAllowedAssignments(
            [launch_vars[tid]], 
            [[s] for s in allowed_launch_slots]
        )
        
        # Pad 分配变量
        for p_idx in range(num_pads):
            pad_assign_vars[(tid, p_idx)] = model.NewBoolVar(f"pad_{tid}_{p_idx}")
        
        # 每个任务恰好分配一个 pad
        model.AddExactlyOne([pad_assign_vars[(tid, p_idx)] for p_idx in range(num_pads)])
        
        # 创建可选区间变量（用于 NoOverlap）
        for p_idx, pad in enumerate(pads):
            # start = launch - duration
            start_var = model.NewIntVar(0, max_slot, f"start_{tid}_{p_idx}")
            model.Add(start_var == launch_vars[tid] - task.duration)
            
            # 区间: [start, start + duration)
            interval_vars[(tid, p_idx)] = model.NewOptionalIntervalVar(
                start_var,
                task.duration,
                launch_vars[tid],  # end = launch
                pad_assign_vars[(tid, p_idx)],
                f"interval_{tid}_{p_idx}"
            )
        
        # Delay 变量
        delay_vars[tid] = model.NewIntVar(0, config.max_delay, f"delay_{tid}")
        # delay = max(0, launch - due)
        model.AddMaxEquality(delay_vars[tid], [0, launch_vars[tid] - task.due])
        
        # ========== 稳定性变量 ==========
        
        prev_assignment = prev_plan.get_assignment(tid) if prev_plan else None
        
        if prev_assignment:
            prev_launch = prev_assignment.launch_slot
            prev_pad_idx = pad_index.get(prev_assignment.pad_id, 0)
            
            # shift_abs = |launch - prev_launch|
            shift_abs_vars[tid] = model.NewIntVar(0, config.max_shift, f"shift_{tid}")
            diff_var = model.NewIntVar(-config.max_shift, config.max_shift, f"diff_{tid}")
            model.Add(diff_var == launch_vars[tid] - prev_launch)
            model.AddAbsEquality(shift_abs_vars[tid], diff_var)
            
            # switched = 1 if pad changed
            switched_vars[tid] = model.NewBoolVar(f"switched_{tid}")
            # switched = 1 - pad_assign[tid, prev_pad_idx]
            model.Add(switched_vars[tid] == 1).OnlyEnforceIf(
                pad_assign_vars[(tid, prev_pad_idx)].Not()
            )
            model.Add(switched_vars[tid] == 0).OnlyEnforceIf(
                pad_assign_vars[(tid, prev_pad_idx)]
            )
        else:
            # 新任务，无稳定性惩罚
            shift_abs_vars[tid] = model.NewConstant(0)
            switched_vars[tid] = model.NewConstant(0)
    
    # ========== Hard Freeze 约束 ==========
    
    for tid, frozen_assign in frozen_tasks.items():
        if tid not in launch_vars:
            continue
        
        # 强制 launch = frozen_launch
        model.Add(launch_vars[tid] == frozen_assign.launch_slot)
        
        # 强制 pad = frozen_pad
        frozen_pad_idx = pad_index.get(frozen_assign.pad_id)
        if frozen_pad_idx is not None:
            model.Add(pad_assign_vars[(tid, frozen_pad_idx)] == 1)
    
    # ========== Pad 容量约束 (NoOverlap) ==========
    
    for p_idx, pad in enumerate(pads):
        # 收集该 pad 上的所有可选区间
        pad_intervals = [
            interval_vars[(task.task_id, p_idx)] 
            for task in tasks 
            if (task.task_id, p_idx) in interval_vars
        ]
        
        if pad_intervals:
            model.AddNoOverlap(pad_intervals)
        
        # Pad 不可用约束 - 使用更简洁的建模
        # 为每个不可用区间创建一个"占位"固定区间，加入 NoOverlap
        for ua_idx, (ua_start, ua_end) in enumerate(pad.unavailable):
            ua_duration = ua_end - ua_start + 1
            # 创建一个固定的不可用区间
            # Note: OR-Tools 9.10+ 使用 NewFixedSizeIntervalVar (无 "d")
            unavail_interval = model.NewFixedSizeIntervalVar(
                ua_start, ua_duration, f"unavail_{p_idx}_{ua_idx}"
            )
            pad_intervals.append(unavail_interval)
        
        # 重新添加 NoOverlap（包含不可用区间）
        if pad_intervals:
            model.AddNoOverlap(pad_intervals)
    
    # ========== 目标函数 ==========
    
    objective_terms = []
    
    # Delay 惩罚
    for task in tasks:
        tid = task.task_id
        # priority * delay
        objective_terms.append(
            int(config.w_delay * task.priority * 100) * delay_vars[tid]
        )
    
    # Shift 惩罚
    for tid in shift_abs_vars:
        objective_terms.append(int(config.w_shift * 100) * shift_abs_vars[tid])
    
    # Pad switch 惩罚
    for tid in switched_vars:
        objective_terms.append(int(config.w_switch * 100) * switched_vars[tid])
    
    model.Minimize(sum(objective_terms))
    
    # ========== 求解 ==========
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers
    
    status = solver.Solve(model)
    
    # ========== 解析结果 ==========
    
    if status == cp_model.OPTIMAL:
        solve_status = SolveStatus.OPTIMAL
    elif status == cp_model.FEASIBLE:
        solve_status = SolveStatus.FEASIBLE
    elif status == cp_model.INFEASIBLE:
        return SolverResult(
            status=SolveStatus.INFEASIBLE,
            plan=None,
            num_variables=model.Proto().variables.__len__(),
            num_constraints=model.Proto().constraints.__len__()
        )
    else:
        return SolverResult(
            status=SolveStatus.TIMEOUT,
            plan=None,
            num_variables=model.Proto().variables.__len__(),
            num_constraints=model.Proto().constraints.__len__()
        )
    
    # 构建计划
    assignments = []
    for task in tasks:
        tid = task.task_id
        launch_slot = solver.Value(launch_vars[tid])
        
        # 找到分配的 pad
        assigned_pad_idx = None
        for p_idx in range(num_pads):
            if solver.Value(pad_assign_vars[(tid, p_idx)]) == 1:
                assigned_pad_idx = p_idx
                break
        
        if assigned_pad_idx is None:
            continue
        
        assignments.append(TaskAssignment(
            task_id=tid,
            pad_id=pad_ids[assigned_pad_idx],
            launch_slot=launch_slot,
            start_slot=launch_slot - task.duration,
            end_slot=launch_slot
        ))
    
    plan = Plan(timestamp=now, assignments=assignments)
    
    return SolverResult(
        status=solve_status,
        plan=plan,
        objective_value=solver.ObjectiveValue() / 100.0,  # 还原缩放
        num_variables=model.Proto().variables.__len__(),
        num_constraints=model.Proto().constraints.__len__()
    )


def _solve_with_degradation(
    now: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan],
    frozen_tasks: Dict[str, TaskAssignment],
    config: SolverConfig
) -> SolverResult:
    """
    降级求解策略
    当完整模型不可行时，逐步放宽约束
    
    降级顺序:
    1. 缩短 freeze horizon (减半)
    2. 完全取消 freeze
    3. 放宽 delay 上界
    4. 最终: 逐个移除优先级最低的任务
    """
    
    degradation_count = 0
    degradation_actions = []
    
    # 策略 1: 缩短 freeze horizon
    if frozen_tasks:
        degradation_count += 1
        degradation_actions.append("reduce_freeze_horizon_50%")
        
        # 只保留一半的冻结任务（按距离 now 最近的）
        sorted_frozen = sorted(
            frozen_tasks.items(),
            key=lambda x: x[1].launch_slot
        )
        reduced_frozen = dict(sorted_frozen[:len(sorted_frozen) // 2])
        
        result = _solve_with_config(now, tasks, pads, prev_plan, reduced_frozen, config)
        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            result.degradation_count = degradation_count
            result.degradation_actions = degradation_actions
            return result
    
    # 策略 2: 完全取消 freeze
    if frozen_tasks:
        degradation_count += 1
        degradation_actions.append("remove_all_freeze")
        
        result = _solve_with_config(now, tasks, pads, prev_plan, {}, config)
        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            result.degradation_count = degradation_count
            result.degradation_actions = degradation_actions
            return result
    
    # 策略 3: 放宽 delay 上界
    degradation_count += 1
    degradation_actions.append("increase_max_delay")
    
    relaxed_config = SolverConfig(
        slot_minutes=config.slot_minutes,
        horizon_slots=config.horizon_slots,
        sim_total_slots=config.sim_total_slots,
        freeze_horizon=config.freeze_horizon,
        w_delay=config.w_delay,
        w_shift=config.w_shift,
        w_switch=config.w_switch,
        time_limit_seconds=config.time_limit_seconds,
        num_workers=config.num_workers,
        max_delay=config.max_delay * 2,  # 翻倍
        max_shift=config.max_shift * 2
    )
    
    result = _solve_with_config(now, tasks, pads, prev_plan, {}, relaxed_config)
    if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
        result.degradation_count = degradation_count
        result.degradation_actions = degradation_actions
        return result
    
    # 策略 4: 移除优先级最低的任务
    remaining_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    while len(remaining_tasks) > 1:
        degradation_count += 1
        removed_task = remaining_tasks.pop()
        degradation_actions.append(f"remove_task_{removed_task.task_id}")
        
        result = _solve_with_config(now, remaining_tasks, pads, prev_plan, {}, relaxed_config)
        if result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
            result.degradation_count = degradation_count
            result.degradation_actions = degradation_actions
            return result
    
    # 全部策略失败
    return SolverResult(
        status=SolveStatus.INFEASIBLE,
        plan=None,
        degradation_count=degradation_count,
        degradation_actions=degradation_actions
    )


# ============================================================================
# 便捷接口
# ============================================================================

def compute_frozen_tasks(
    current_plan: Optional[Plan],
    now: int,
    freeze_horizon: int,
    completed_tasks: Optional[Set[str]] = None
) -> Dict[str, TaskAssignment]:
    """
    计算需要冻结的任务
    
    Args:
        current_plan: 当前计划
        now: 当前时刻
        freeze_horizon: 冻结视野 (slots)
        completed_tasks: 已完成的任务集合
    
    Returns:
        冻结任务字典 {task_id: TaskAssignment}
    """
    if current_plan is None:
        return {}
    
    if completed_tasks is None:
        completed_tasks = set()
    
    frozen = {}
    freeze_end = now + freeze_horizon
    
    for assignment in current_plan.assignments:
        tid = assignment.task_id
        
        # 跳过已完成任务
        if tid in completed_tasks:
            continue
        
        # 如果任务的 launch 在冻结视野内，且尚未开始（start > now）
        # 则冻结
        if assignment.start_slot > now and assignment.launch_slot <= freeze_end:
            frozen[tid] = assignment
    
    return frozen


def solve(
    now: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan] = None,
    freeze_horizon: int = 12,
    weights: Tuple[float, float, float] = (10.0, 1.0, 5.0),
    time_limit: float = 30.0,
    completed_tasks: Optional[Set[str]] = None
) -> SolverResult:
    """
    简化接口：一站式求解
    
    Args:
        now: 当前时刻 slot
        tasks: 任务列表
        pads: pad 列表
        prev_plan: 上一轮计划
        freeze_horizon: 冻结视野 slots
        weights: (w_delay, w_shift, w_switch)
        time_limit: 求解时限 (秒)
        completed_tasks: 已完成任务集合
    
    Returns:
        SolverResult
    
    Example:
        >>> result = solve(
        ...     now=0,
        ...     tasks=[task1, task2],
        ...     pads=[pad_a, pad_b],
        ...     prev_plan=None,
        ...     weights=(10.0, 1.0, 5.0)
        ... )
        >>> print(result.status, result.plan)
    """
    config = SolverConfig(
        freeze_horizon=freeze_horizon,
        w_delay=weights[0],
        w_shift=weights[1],
        w_switch=weights[2],
        time_limit_seconds=time_limit
    )
    
    # 计算冻结任务
    frozen_tasks = compute_frozen_tasks(prev_plan, now, freeze_horizon, completed_tasks)
    
    return solve_cpsat(now, tasks, pads, prev_plan, frozen_tasks, config)


# ============================================================================
# 模块测试入口
# ============================================================================

if __name__ == "__main__":
    # 简单自测
    print("=== solver_cpsat.py 模块自测 ===")
    
    # 创建测试数据
    tasks = [
        Task("T1", release=0, duration=3, windows=[(5, 20)], due=10, priority=1.0),
        Task("T2", release=2, duration=4, windows=[(8, 25)], due=15, priority=0.8),
    ]
    pads = [Pad("PAD_A"), Pad("PAD_B")]
    
    result = solve(now=0, tasks=tasks, pads=pads)
    
    print(f"Status: {result.status}")
    print(f"Objective: {result.objective_value}")
    print(f"Solve time: {result.solve_time_ms} ms")
    
    if result.plan:
        for a in result.plan.assignments:
            print(f"  {a.task_id}: pad={a.pad_id}, launch={a.launch_slot}, "
                  f"start={a.start_slot}")
