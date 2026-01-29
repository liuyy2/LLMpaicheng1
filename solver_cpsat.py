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
class Operation:
    """Operation definition (V2.1)"""
    op_id: str
    mission_id: str
    op_index: int
    duration: int
    resources: List[str]
    precedences: List[str]
    time_windows: List[Tuple[int, int]] = field(default_factory=list)
    release: int = 0


@dataclass
class Mission:
    """Mission definition (V2.1)"""
    mission_id: str
    operations: List[Operation]
    release: int
    due: int
    priority: float = 1.0

    def get_operation(self, op_index: int) -> Optional[Operation]:
        for op in self.operations:
            if op.op_index == op_index:
                return op
        return None


@dataclass
class Resource:
    """Resource definition (V2.1)"""
    resource_id: str
    capacity: int = 1
    unavailable: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class OpAssignment:
    """Operation assignment (V2.1)"""
    op_id: str
    mission_id: str
    op_index: int
    resources: List[str]
    start_slot: int
    end_slot: int


@dataclass
class PlanV2_1:
    """Schedule plan (V2.1)"""
    timestamp: int
    schema_version: str = "v2_1"
    op_assignments: List[OpAssignment] = field(default_factory=list)

    def get_assignment(self, op_id: str) -> Optional[OpAssignment]:
        for a in self.op_assignments:
            if a.op_id == op_id:
                return a
        return None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "op_assignments": [
                {
                    "op_id": a.op_id,
                    "mission_id": a.mission_id,
                    "op_index": a.op_index,
                    "resources": a.resources,
                    "start_slot": a.start_slot,
                    "end_slot": a.end_slot
                }
                for a in self.op_assignments
            ]
        }

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
            # Note: OR-Tools 9.9 使用 NewFixedSizedIntervalVar (有 "d")
            #       OR-Tools 9.10+ 使用 NewFixedSizeIntervalVar (无 "d")
            try:
                # 尝试 9.9 版本的 API（有 "d"）
                unavail_interval = model.NewFixedSizedIntervalVar(
                    ua_start, ua_duration, f"unavail_{p_idx}_{ua_idx}"
                )
            except AttributeError:
                # 回退到 9.10+ 版本的 API（无 "d"）
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

# ============================================================================
# V2.1 Solver
# ============================================================================


def compute_frozen_ops(
    current_plan: Optional[PlanV2_1],
    now: int,
    freeze_horizon: int,
    started_ops: Optional[Set[str]] = None,
    completed_ops: Optional[Set[str]] = None
) -> Dict[str, OpAssignment]:
    if current_plan is None:
        return {}

    if started_ops is None:
        started_ops = set()
    if completed_ops is None:
        completed_ops = set()

    frozen = {}
    freeze_end = now + freeze_horizon

    for assignment in current_plan.op_assignments:
        op_id = assignment.op_id

        if op_id in completed_ops:
            continue

        if op_id in started_ops:
            frozen[op_id] = assignment
            continue

        if assignment.start_slot > now and assignment.start_slot <= freeze_end:
            frozen[op_id] = assignment

    return frozen


@dataclass
class SolverConfigV2_1:
    """V2.1 solver config"""
    horizon_slots: int = 336
    w_delay: float = 10.0
    w_shift: float = 1.0
    w_switch: float = 5.0
    time_limit_seconds: float = 60.0
    num_workers: int = 4
    op5_max_wait_slots: int = 144
    use_two_stage: bool = True
    epsilon_solver: float = 0.05
    kappa_win: float = 12.0
    kappa_seq: float = 6.0
    stage1_time_ratio: float = 0.4


def solve_v2_1(
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1] = None,
    frozen_ops: Optional[Dict[str, OpAssignment]] = None,
    config: Optional[SolverConfigV2_1] = None
) -> SolverResult:
    if config is None:
        config = SolverConfigV2_1(horizon_slots=horizon)
    if frozen_ops is None:
        frozen_ops = {}

    start_time = time.time()
    if config.use_two_stage:
        result = _solve_v2_1_two_stage(
            missions, resources, horizon, prev_plan, frozen_ops, config
        )
    else:
        result = _solve_v2_1_with_config(missions, resources, horizon, prev_plan, frozen_ops, config)
    result.solve_time_ms = int((time.time() - start_time) * 1000)
    return result


def _solve_v2_1_two_stage(
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    config: SolverConfigV2_1
) -> SolverResult:
    if not missions:
        return SolverResult(
            status=SolveStatus.OPTIMAL,
            plan=PlanV2_1(timestamp=0, op_assignments=[]),
            objective_value=0.0
        )

    stage1_time = max(0.1, config.time_limit_seconds * config.stage1_time_ratio)
    stage1_config = SolverConfigV2_1(
        horizon_slots=config.horizon_slots,
        w_delay=config.w_delay,
        w_shift=0.0,
        w_switch=0.0,
        time_limit_seconds=stage1_time,
        num_workers=config.num_workers,
        op5_max_wait_slots=config.op5_max_wait_slots,
        use_two_stage=False,
        epsilon_solver=config.epsilon_solver,
        kappa_win=config.kappa_win,
        kappa_seq=config.kappa_seq,
        stage1_time_ratio=config.stage1_time_ratio
    )

    result_stage1 = _solve_v2_1_with_config(
        missions, resources, horizon, prev_plan, frozen_ops, stage1_config
    )
    if result_stage1.status not in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
        return result_stage1

    total_delay_stage1 = 0.0
    if result_stage1.plan:
        for mission in missions:
            op6 = mission.get_operation(6)
            if not op6:
                continue
            assign = result_stage1.plan.get_assignment(op6.op_id)
            if not assign:
                continue
            delay = max(0, assign.start_slot - mission.due)
            total_delay_stage1 += mission.priority * delay

    delay_bound = total_delay_stage1 * (1 + config.epsilon_solver)
    stage2_time = max(0.1, config.time_limit_seconds * (1 - config.stage1_time_ratio))
    stage2_config = SolverConfigV2_1(
        horizon_slots=config.horizon_slots,
        w_delay=config.w_delay,
        w_shift=config.w_shift,
        w_switch=config.w_switch,
        time_limit_seconds=stage2_time,
        num_workers=config.num_workers,
        op5_max_wait_slots=config.op5_max_wait_slots,
        use_two_stage=False,
        epsilon_solver=config.epsilon_solver,
        kappa_win=config.kappa_win,
        kappa_seq=config.kappa_seq,
        stage1_time_ratio=config.stage1_time_ratio
    )

    result_stage2 = _solve_v2_1_stage2_with_delay_bound(
        missions, resources, horizon, prev_plan, frozen_ops, stage2_config, delay_bound
    )
    if result_stage2.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE]:
        return result_stage2

    return result_stage1


def _solve_v2_1_stage2_with_delay_bound(
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    config: SolverConfigV2_1,
    delay_bound: float
) -> SolverResult:
    if not missions:
        return SolverResult(
            status=SolveStatus.OPTIMAL,
            plan=PlanV2_1(timestamp=0, op_assignments=[]),
            objective_value=0.0
        )

    model = cp_model.CpModel()

    all_ops = []
    for mission in missions:
        all_ops.extend(mission.operations)

    start_vars: Dict[str, cp_model.IntVar] = {}
    end_vars: Dict[str, cp_model.IntVar] = {}
    interval_vars: Dict[str, cp_model.IntervalVar] = {}

    op5_max_wait_slots = max(0, config.op5_max_wait_slots)
    for op in all_ops:
        start_vars[op.op_id] = model.NewIntVar(
            op.release, horizon, f"start_{op.op_id}"
        )
        if op.op_index == 5:
            min_duration = max(0, op.duration)
            max_duration = min_duration + op5_max_wait_slots
            duration_var = model.NewIntVar(
                min_duration, max_duration, f"dur_{op.op_id}"
            )
            end_vars[op.op_id] = model.NewIntVar(
                op.release + min_duration, horizon + max_duration, f"end_{op.op_id}"
            )
            model.Add(end_vars[op.op_id] == start_vars[op.op_id] + duration_var)
            interval_vars[op.op_id] = model.NewIntervalVar(
                start_vars[op.op_id],
                duration_var,
                end_vars[op.op_id],
                f"interval_{op.op_id}"
            )
        else:
            end_vars[op.op_id] = model.NewIntVar(
                op.release + op.duration, horizon + op.duration, f"end_{op.op_id}"
            )
            model.Add(end_vars[op.op_id] == start_vars[op.op_id] + op.duration)
            interval_vars[op.op_id] = model.NewIntervalVar(
                start_vars[op.op_id],
                op.duration,
                end_vars[op.op_id],
                f"interval_{op.op_id}"
            )

    for op in all_ops:
        for pred_id in op.precedences:
            if pred_id in end_vars:
                model.Add(start_vars[op.op_id] >= end_vars[pred_id])

    for mission in missions:
        op4 = mission.get_operation(4)
        op5 = mission.get_operation(5)
        op6 = mission.get_operation(6)
        if not op4 or not op5 or not op6:
            continue
        if op4.op_id in end_vars and op5.op_id in start_vars:
            model.Add(start_vars[op5.op_id] == end_vars[op4.op_id])
        if op5.op_id in end_vars and op6.op_id in start_vars:
            model.Add(start_vars[op6.op_id] == end_vars[op5.op_id])

    resource_intervals: Dict[str, List[cp_model.IntervalVar]] = {
        r.resource_id: [] for r in resources
    }

    for op in all_ops:
        for res_id in op.resources:
            if res_id in resource_intervals:
                resource_intervals[res_id].append(interval_vars[op.op_id])

    for resource in resources:
        for closure_idx, (cs, ce) in enumerate(resource.unavailable):
            duration = ce - cs + 1
            try:
                blocker = model.NewFixedSizedIntervalVar(
                    cs, duration, f"closure_{resource.resource_id}_{closure_idx}"
                )
            except AttributeError:
                blocker = model.NewFixedSizeIntervalVar(
                    cs, duration, f"closure_{resource.resource_id}_{closure_idx}"
                )
            resource_intervals[resource.resource_id].append(blocker)

    for res_id, intervals in resource_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    window_choice_vars: Dict[str, List[cp_model.BoolVar]] = {}
    prev_window_index: Dict[str, Optional[int]] = {}
    for mission in missions:
        op6 = mission.get_operation(6)
        if not op6 or not op6.time_windows:
            continue
        window_choice = []
        for win_idx, (ws, we) in enumerate(op6.time_windows):
            in_window = model.NewBoolVar(f"op6_{op6.op_id}_window_{win_idx}")
            window_choice.append(in_window)
            model.Add(start_vars[op6.op_id] >= ws).OnlyEnforceIf(in_window)
            model.Add(end_vars[op6.op_id] <= we).OnlyEnforceIf(in_window)
        model.AddExactlyOne(window_choice)
        window_choice_vars[op6.op_id] = window_choice
        prev_idx = None
        if prev_plan:
            prev_assign = prev_plan.get_assignment(op6.op_id)
            if prev_assign:
                for idx, (ws, we) in enumerate(op6.time_windows):
                    if prev_assign.start_slot >= ws and prev_assign.end_slot <= we:
                        prev_idx = idx
                        break
        prev_window_index[op6.op_id] = prev_idx

    for op_id, frozen in frozen_ops.items():
        if op_id in start_vars:
            model.Add(start_vars[op_id] == frozen.start_slot)
            model.Add(end_vars[op_id] == frozen.end_slot)

    delay_vars = {}
    for mission in missions:
        op6 = mission.get_operation(6)
        if not op6 or op6.op_id not in start_vars:
            continue
        delay_var = model.NewIntVar(0, horizon, f"delay_{mission.mission_id}")
        model.AddMaxEquality(delay_var, [0, start_vars[op6.op_id] - mission.due])
        delay_vars[mission.mission_id] = delay_var

    total_delay_terms = []
    for mission in missions:
        if mission.mission_id in delay_vars:
            weight = int(round(mission.priority * 100))
            total_delay_terms.append(weight * delay_vars[mission.mission_id])

    if total_delay_terms:
        model.Add(sum(total_delay_terms) <= int(round(delay_bound * 100)))

    objective_terms = []
    max_shift = max(0, horizon)

    shift_abs_vars: Dict[str, cp_model.IntVar] = {}
    window_switch_vars: Dict[str, cp_model.IntVar] = {}
    seq_switch_vars: Dict[str, cp_model.IntVar] = {}

    if prev_plan:
        for mission in missions:
            op6 = mission.get_operation(6)
            op4 = mission.get_operation(4)
            if op6 and op6.op_id in start_vars:
                prev_assign = prev_plan.get_assignment(op6.op_id)
                if prev_assign:
                    shift_abs = model.NewIntVar(0, max_shift, f"shift_{op6.op_id}")
                    diff_var = model.NewIntVar(-max_shift, max_shift, f"diff_{op6.op_id}")
                    model.Add(diff_var == start_vars[op6.op_id] - prev_assign.start_slot)
                    model.AddAbsEquality(shift_abs, diff_var)
                    shift_abs_vars[op6.op_id] = shift_abs
            if op4 and op4.op_id in start_vars:
                prev_assign = prev_plan.get_assignment(op4.op_id)
                if prev_assign:
                    shift_abs = model.NewIntVar(0, max_shift, f"shift_{op4.op_id}")
                    diff_var = model.NewIntVar(-max_shift, max_shift, f"diff_{op4.op_id}")
                    model.Add(diff_var == start_vars[op4.op_id] - prev_assign.start_slot)
                    model.AddAbsEquality(shift_abs, diff_var)
                    shift_abs_vars[op4.op_id] = shift_abs

    for op_id, choice in window_choice_vars.items():
        prev_idx = prev_window_index.get(op_id)
        if prev_idx is not None and 0 <= prev_idx < len(choice):
            switched = model.NewBoolVar(f"window_switch_{op_id}")
            model.Add(switched == 0).OnlyEnforceIf(choice[prev_idx])
            model.Add(switched == 1).OnlyEnforceIf(choice[prev_idx].Not())
            window_switch_vars[op_id] = switched
        else:
            window_switch_vars[op_id] = model.NewConstant(0)

    prev_pred: Dict[str, str] = {}
    if prev_plan:
        prev_order = []
        for mission in missions:
            op4 = mission.get_operation(4)
            if not op4:
                continue
            prev_assign = prev_plan.get_assignment(op4.op_id)
            if prev_assign and "R_pad" in prev_assign.resources:
                prev_order.append((mission.mission_id, prev_assign.start_slot))
        prev_order.sort(key=lambda x: x[1])
        for idx in range(1, len(prev_order)):
            prev_pred[prev_order[idx][0]] = prev_order[idx - 1][0]

    start_op4: Dict[str, cp_model.IntVar] = {}
    for mission in missions:
        op4 = mission.get_operation(4)
        if op4 and op4.op_id in start_vars:
            start_op4[mission.mission_id] = start_vars[op4.op_id]

    for mission in missions:
        pred_id = prev_pred.get(mission.mission_id)
        if not pred_id:
            continue
        if pred_id not in start_op4 or mission.mission_id not in start_op4:
            continue

        pred_kept = model.NewBoolVar(f"pred_kept_{mission.mission_id}")
        seq_switch = model.NewBoolVar(f"seq_switch_{mission.mission_id}")
        model.Add(pred_kept + seq_switch == 1)

        start_p = start_op4[pred_id]
        start_m = start_op4[mission.mission_id]

        pred_before = model.NewBoolVar(f"pred_before_{mission.mission_id}")
        model.Add(start_p <= start_m - 1).OnlyEnforceIf(pred_before)
        model.Add(start_p >= start_m).OnlyEnforceIf(pred_before.Not())
        model.Add(pred_before == 1).OnlyEnforceIf(pred_kept)

        for other_id, start_q in start_op4.items():
            if other_id in (mission.mission_id, pred_id):
                continue
            q_before_p = model.NewBoolVar(f"q_before_{other_id}_p_{mission.mission_id}")
            model.Add(start_q <= start_p).OnlyEnforceIf(q_before_p)
            model.Add(start_q >= start_p + 1).OnlyEnforceIf(q_before_p.Not())

            q_after_m = model.NewBoolVar(f"q_after_{other_id}_m_{mission.mission_id}")
            model.Add(start_q >= start_m).OnlyEnforceIf(q_after_m)
            model.Add(start_q <= start_m - 1).OnlyEnforceIf(q_after_m.Not())

            model.AddBoolOr([q_before_p, q_after_m]).OnlyEnforceIf(pred_kept)

        seq_switch_vars[mission.mission_id] = seq_switch

    scale = 100
    for mission in missions:
        op6 = mission.get_operation(6)
        op4 = mission.get_operation(4)
        if op6 and op6.op_id in shift_abs_vars:
            weight = int(round(mission.priority * 0.7 * scale))
            objective_terms.append(weight * shift_abs_vars[op6.op_id])
        if op4 and op4.op_id in shift_abs_vars:
            weight = int(round(mission.priority * 0.3 * scale))
            objective_terms.append(weight * shift_abs_vars[op4.op_id])

        if op6 and op6.op_id in window_switch_vars:
            weight = int(round(mission.priority * config.kappa_win * scale))
            objective_terms.append(weight * window_switch_vars[op6.op_id])

        if mission.mission_id in seq_switch_vars:
            weight = int(round(mission.priority * config.kappa_seq * scale))
            objective_terms.append(weight * seq_switch_vars[mission.mission_id])

    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers

    status = solver.Solve(model)

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

    op_assignments = []
    for op in all_ops:
        if op.op_id in start_vars:
            start = solver.Value(start_vars[op.op_id])
            end = solver.Value(end_vars[op.op_id])
            op_assignments.append(OpAssignment(
                op_id=op.op_id,
                mission_id=op.mission_id,
                op_index=op.op_index,
                resources=op.resources,
                start_slot=start,
                end_slot=end
            ))

    plan = PlanV2_1(timestamp=0, op_assignments=op_assignments)

    return SolverResult(
        status=solve_status,
        plan=plan,
        objective_value=solver.ObjectiveValue() / 100.0 if objective_terms else 0.0,
        num_variables=model.Proto().variables.__len__(),
        num_constraints=model.Proto().constraints.__len__()
    )


def _solve_v2_1_with_config(
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    config: SolverConfigV2_1
) -> SolverResult:
    if not missions:
        return SolverResult(
            status=SolveStatus.OPTIMAL,
            plan=PlanV2_1(timestamp=0, op_assignments=[]),
            objective_value=0.0
        )

    model = cp_model.CpModel()

    all_ops = []
    for mission in missions:
        all_ops.extend(mission.operations)

    start_vars: Dict[str, cp_model.IntVar] = {}
    end_vars: Dict[str, cp_model.IntVar] = {}
    interval_vars: Dict[str, cp_model.IntervalVar] = {}

    op5_max_wait_slots = max(0, config.op5_max_wait_slots)
    for op in all_ops:
        start_vars[op.op_id] = model.NewIntVar(
            op.release, horizon, f"start_{op.op_id}"
        )
        if op.op_index == 5:
            min_duration = max(0, op.duration)
            max_duration = min_duration + op5_max_wait_slots
            duration_var = model.NewIntVar(
                min_duration, max_duration, f"dur_{op.op_id}"
            )
            end_vars[op.op_id] = model.NewIntVar(
                op.release + min_duration, horizon + max_duration, f"end_{op.op_id}"
            )
            model.Add(end_vars[op.op_id] == start_vars[op.op_id] + duration_var)
            interval_vars[op.op_id] = model.NewIntervalVar(
                start_vars[op.op_id],
                duration_var,
                end_vars[op.op_id],
                f"interval_{op.op_id}"
            )
        else:
            end_vars[op.op_id] = model.NewIntVar(
                op.release + op.duration, horizon + op.duration, f"end_{op.op_id}"
            )
            model.Add(end_vars[op.op_id] == start_vars[op.op_id] + op.duration)
            interval_vars[op.op_id] = model.NewIntervalVar(
                start_vars[op.op_id],
                op.duration,
                end_vars[op.op_id],
                f"interval_{op.op_id}"
            )

    for op in all_ops:
        for pred_id in op.precedences:
            if pred_id in end_vars:
                model.Add(start_vars[op.op_id] >= end_vars[pred_id])

    # Pad holding: Op4 -> Op5 -> Op6 must be contiguous
    for mission in missions:
        op4 = mission.get_operation(4)
        op5 = mission.get_operation(5)
        op6 = mission.get_operation(6)
        if not op4 or not op5 or not op6:
            continue
        if op4.op_id in end_vars and op5.op_id in start_vars:
            model.Add(start_vars[op5.op_id] == end_vars[op4.op_id])
        if op5.op_id in end_vars and op6.op_id in start_vars:
            model.Add(start_vars[op6.op_id] == end_vars[op5.op_id])
    resource_intervals: Dict[str, List[cp_model.IntervalVar]] = {
        r.resource_id: [] for r in resources
    }

    for op in all_ops:
        for res_id in op.resources:
            if res_id in resource_intervals:
                resource_intervals[res_id].append(interval_vars[op.op_id])

    for resource in resources:
        for closure_idx, (cs, ce) in enumerate(resource.unavailable):
            duration = ce - cs + 1
            try:
                blocker = model.NewFixedSizedIntervalVar(
                    cs, duration, f"closure_{resource.resource_id}_{closure_idx}"
                )
            except AttributeError:
                blocker = model.NewFixedSizeIntervalVar(
                    cs, duration, f"closure_{resource.resource_id}_{closure_idx}"
                )
            resource_intervals[resource.resource_id].append(blocker)

    for res_id, intervals in resource_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    window_choice_vars: Dict[str, List[cp_model.BoolVar]] = {}
    prev_window_index: Dict[str, Optional[int]] = {}
    for mission in missions:
        op6 = mission.get_operation(6)
        if not op6 or not op6.time_windows:
            continue
        window_choice = []
        for win_idx, (ws, we) in enumerate(op6.time_windows):
            in_window = model.NewBoolVar(f"op6_{op6.op_id}_window_{win_idx}")
            window_choice.append(in_window)
            model.Add(start_vars[op6.op_id] >= ws).OnlyEnforceIf(in_window)
            model.Add(end_vars[op6.op_id] <= we).OnlyEnforceIf(in_window)
        model.AddExactlyOne(window_choice)
        window_choice_vars[op6.op_id] = window_choice
        prev_idx = None
        if prev_plan:
            prev_assign = prev_plan.get_assignment(op6.op_id)
            if prev_assign:
                for idx, (ws, we) in enumerate(op6.time_windows):
                    if prev_assign.start_slot >= ws and prev_assign.end_slot <= we:
                        prev_idx = idx
                        break
        prev_window_index[op6.op_id] = prev_idx

    for op_id, frozen in frozen_ops.items():
        if op_id in start_vars:
            model.Add(start_vars[op_id] == frozen.start_slot)
            model.Add(end_vars[op_id] == frozen.end_slot)

    objective_terms = []
    max_shift = max(0, horizon)
    shift_abs_vars: Dict[str, cp_model.IntVar] = {}
    switch_vars: Dict[str, cp_model.IntVar] = {}

    for mission in missions:
        op6 = mission.get_operation(6)
        if not op6 or op6.op_id not in end_vars:
            continue
        delay_var = model.NewIntVar(0, horizon, f"delay_{mission.mission_id}")
        model.AddMaxEquality(delay_var, [0, start_vars[op6.op_id] - mission.due])
        weight = int(config.w_delay * mission.priority * 100)
        objective_terms.append(weight * delay_var)

    if prev_plan:
        for op in all_ops:
            prev_assign = prev_plan.get_assignment(op.op_id)
            if prev_assign and op.op_id in start_vars:
                shift_abs = model.NewIntVar(0, max_shift, f"shift_{op.op_id}")
                diff_var = model.NewIntVar(-max_shift, max_shift, f"diff_{op.op_id}")
                model.Add(diff_var == start_vars[op.op_id] - prev_assign.start_slot)
                model.AddAbsEquality(shift_abs, diff_var)
                shift_abs_vars[op.op_id] = shift_abs
            else:
                shift_abs_vars[op.op_id] = model.NewConstant(0)
    else:
        for op in all_ops:
            shift_abs_vars[op.op_id] = model.NewConstant(0)

    for op_id, choice in window_choice_vars.items():
        prev_idx = prev_window_index.get(op_id)
        if prev_idx is not None and 0 <= prev_idx < len(choice):
            switched = model.NewBoolVar(f"window_switch_{op_id}")
            model.Add(switched == 0).OnlyEnforceIf(choice[prev_idx])
            model.Add(switched == 1).OnlyEnforceIf(choice[prev_idx].Not())
            switch_vars[op_id] = switched
        else:
            switch_vars[op_id] = model.NewConstant(0)

    if config.w_shift != 0:
        for op_id, var in shift_abs_vars.items():
            objective_terms.append(int(config.w_shift * 100) * var)
    if config.w_switch != 0:
        for op_id, var in switch_vars.items():
            objective_terms.append(int(config.w_switch * 100) * var)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.time_limit_seconds
    solver.parameters.num_search_workers = config.num_workers

    status = solver.Solve(model)

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

    op_assignments = []
    for op in all_ops:
        if op.op_id in start_vars:
            start = solver.Value(start_vars[op.op_id])
            end = solver.Value(end_vars[op.op_id])
            op_assignments.append(OpAssignment(
                op_id=op.op_id,
                mission_id=op.mission_id,
                op_index=op.op_index,
                resources=op.resources,
                start_slot=start,
                end_slot=end
            ))

    plan = PlanV2_1(timestamp=0, op_assignments=op_assignments)

    return SolverResult(
        status=solve_status,
        plan=plan,
        objective_value=solver.ObjectiveValue() / 100.0 if objective_terms else 0.0,
        num_variables=model.Proto().variables.__len__(),
        num_constraints=model.Proto().constraints.__len__()
    )


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
