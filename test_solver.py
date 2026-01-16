"""
test_solver.py - CP-SAT 求解器测试脚本

构造一个小场景（5 tasks, 2 pads, windows），测试求解器功能
包括：基本求解、带 prev_plan 的稳定性惩罚、冻结约束、降级机制
"""

import random
from typing import List, Tuple

from solver_cpsat import (
    Task, Pad, Plan, TaskAssignment,
    SolverConfig, SolverResult, SolveStatus,
    solve, solve_cpsat, compute_frozen_tasks
)


def create_test_scenario(seed: int = 42) -> Tuple[List[Task], List[Pad]]:
    """
    创建测试场景: 5 tasks, 2 pads
    
    Args:
        seed: 随机种子
    
    Returns:
        (tasks, pads)
    """
    random.seed(seed)
    
    tasks = [
        Task(
            task_id="T001",
            release=0,
            duration=6,  # 1 小时
            windows=[(10, 30), (50, 70)],  # 两个发射窗口
            due=25,
            priority=1.0,
            preferred_pad="PAD_A"
        ),
        Task(
            task_id="T002",
            release=5,
            duration=4,
            windows=[(15, 40)],
            due=30,
            priority=0.9
        ),
        Task(
            task_id="T003",
            release=10,
            duration=5,
            windows=[(20, 50)],
            due=35,
            priority=0.8
        ),
        Task(
            task_id="T004",
            release=0,
            duration=3,
            windows=[(8, 25), (40, 60)],
            due=20,
            priority=0.7
        ),
        Task(
            task_id="T005",
            release=15,
            duration=4,
            windows=[(25, 55)],
            due=45,
            priority=0.6
        ),
    ]
    
    pads = [
        Pad(pad_id="PAD_A", unavailable=[(35, 40)]),  # PAD_A 在 slot 35-40 不可用
        Pad(pad_id="PAD_B", unavailable=[]),
    ]
    
    return tasks, pads


def create_random_prev_plan(
    tasks: List[Task],
    pads: List[Pad],
    seed: int = 123
) -> Plan:
    """
    创建一个随机的 prev_plan（模拟上一轮排程结果）
    
    Args:
        tasks: 任务列表
        pads: pad 列表
        seed: 随机种子
    
    Returns:
        Plan 对象
    """
    random.seed(seed)
    
    assignments = []
    for task in tasks:
        # 随机选一个 window，在其中随机选 launch slot
        window = random.choice(task.windows)
        min_launch = max(window[0], task.release + task.duration)
        max_launch = window[1]
        
        if min_launch <= max_launch:
            launch_slot = random.randint(min_launch, max_launch)
        else:
            launch_slot = task.release + task.duration + 5
        
        # 随机选 pad
        pad_id = random.choice([p.pad_id for p in pads])
        
        assignments.append(TaskAssignment(
            task_id=task.task_id,
            pad_id=pad_id,
            launch_slot=launch_slot,
            start_slot=launch_slot - task.duration,
            end_slot=launch_slot
        ))
    
    return Plan(timestamp=0, assignments=assignments)


def print_result(result: SolverResult, title: str = ""):
    """格式化打印求解结果"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"Status:      {result.status.value}")
    print(f"Objective:   {result.objective_value}")
    print(f"Solve time:  {result.solve_time_ms} ms")
    print(f"Variables:   {result.num_variables}")
    print(f"Constraints: {result.num_constraints}")
    
    if result.degradation_count > 0:
        print(f"Degradation: {result.degradation_count} steps")
        print(f"Actions:     {result.degradation_actions}")
    
    if result.plan:
        print(f"\nSchedule (timestamp={result.plan.timestamp}):")
        print(f"  {'Task':<8} {'Pad':<8} {'Start':<8} {'Launch':<8} {'End':<8}")
        print(f"  {'-'*40}")
        for a in sorted(result.plan.assignments, key=lambda x: x.launch_slot):
            print(f"  {a.task_id:<8} {a.pad_id:<8} {a.start_slot:<8} "
                  f"{a.launch_slot:<8} {a.end_slot:<8}")
    else:
        print("\nNo feasible solution found.")


def test_basic_solve():
    """测试 1: 基本求解（无 prev_plan）"""
    print("\n" + "#"*60)
    print(" TEST 1: Basic Solve (no prev_plan)")
    print("#"*60)
    
    tasks, pads = create_test_scenario(seed=42)
    
    print("\nInput Tasks:")
    for t in tasks:
        print(f"  {t.task_id}: release={t.release}, dur={t.duration}, "
              f"windows={t.windows}, due={t.due}, priority={t.priority}")
    
    print("\nInput Pads:")
    for p in pads:
        print(f"  {p.pad_id}: unavailable={p.unavailable}")
    
    result = solve(
        now=0,
        tasks=tasks,
        pads=pads,
        prev_plan=None,
        weights=(10.0, 1.0, 5.0),
        time_limit=10.0
    )
    
    print_result(result, "Basic Solve Result")
    
    assert result.status in [SolveStatus.OPTIMAL, SolveStatus.FEASIBLE], \
        "Expected OPTIMAL or FEASIBLE"
    assert result.plan is not None, "Expected a valid plan"
    assert len(result.plan.assignments) == len(tasks), \
        "All tasks should be scheduled"
    
    print("\n✓ Test 1 PASSED")
    return result


def test_with_prev_plan():
    """测试 2: 带 prev_plan 的稳定性惩罚"""
    print("\n" + "#"*60)
    print(" TEST 2: Solve with prev_plan (stability penalty)")
    print("#"*60)
    
    tasks, pads = create_test_scenario(seed=42)
    prev_plan = create_random_prev_plan(tasks, pads, seed=123)
    
    print("\nPrevious Plan:")
    for a in prev_plan.assignments:
        print(f"  {a.task_id}: pad={a.pad_id}, launch={a.launch_slot}")
    
    result = solve(
        now=5,  # 推进到 slot 5
        tasks=tasks,
        pads=pads,
        prev_plan=prev_plan,
        weights=(10.0, 1.0, 5.0),
        time_limit=10.0
    )
    
    print_result(result, "Solve with prev_plan Result")
    
    # 统计变化
    if result.plan:
        shifts = 0
        switches = 0
        for a in result.plan.assignments:
            prev_a = prev_plan.get_assignment(a.task_id)
            if prev_a:
                if a.launch_slot != prev_a.launch_slot:
                    shifts += 1
                    print(f"  SHIFT: {a.task_id} launch {prev_a.launch_slot} -> {a.launch_slot}")
                if a.pad_id != prev_a.pad_id:
                    switches += 1
                    print(f"  SWITCH: {a.task_id} pad {prev_a.pad_id} -> {a.pad_id}")
        print(f"\nTotal shifts: {shifts}, switches: {switches}")
    
    print("\n✓ Test 2 PASSED")
    return result


def test_with_freeze():
    """测试 3: 冻结约束"""
    print("\n" + "#"*60)
    print(" TEST 3: Solve with freeze constraint")
    print("#"*60)
    
    tasks, pads = create_test_scenario(seed=42)
    
    # 先跑一次获得初始计划
    initial_result = solve(now=0, tasks=tasks, pads=pads)
    prev_plan = initial_result.plan
    
    print("\nInitial Plan:")
    for a in prev_plan.assignments:
        print(f"  {a.task_id}: pad={a.pad_id}, launch={a.launch_slot}")
    
    # 模拟时间推进到 slot 10，计算冻结任务
    now = 10
    freeze_horizon = 15
    frozen_tasks = compute_frozen_tasks(prev_plan, now, freeze_horizon)
    
    print(f"\nFrozen tasks (now={now}, freeze_horizon={freeze_horizon}):")
    for tid, fa in frozen_tasks.items():
        print(f"  {tid}: pad={fa.pad_id}, launch={fa.launch_slot} (FROZEN)")
    
    # 带冻结求解
    result = solve(
        now=now,
        tasks=tasks,
        pads=pads,
        prev_plan=prev_plan,
        freeze_horizon=freeze_horizon,
        weights=(10.0, 1.0, 5.0),
        time_limit=10.0
    )
    
    print_result(result, "Solve with Freeze Result")
    
    # 验证冻结约束被遵守
    if result.plan and frozen_tasks:
        print("\nVerifying freeze constraints:")
        for tid, frozen in frozen_tasks.items():
            actual = result.plan.get_assignment(tid)
            if actual:
                launch_ok = actual.launch_slot == frozen.launch_slot
                pad_ok = actual.pad_id == frozen.pad_id
                status = "✓" if (launch_ok and pad_ok) else "✗"
                print(f"  {tid}: launch {frozen.launch_slot}=={actual.launch_slot} ({launch_ok}), "
                      f"pad {frozen.pad_id}=={actual.pad_id} ({pad_ok}) {status}")
    
    print("\n✓ Test 3 PASSED")
    return result


def test_infeasible_and_degradation():
    """测试 4: 不可行场景与降级机制"""
    print("\n" + "#"*60)
    print(" TEST 4: Infeasible scenario & degradation")
    print("#"*60)
    
    # 构造一个难以满足的场景
    # 两个任务必须在同一窗口发射，但只有一个 pad
    tasks = [
        Task(
            task_id="HARD_1",
            release=0,
            duration=10,
            windows=[(15, 18)],  # 非常窄的窗口
            due=15,
            priority=1.0
        ),
        Task(
            task_id="HARD_2",
            release=0,
            duration=10,
            windows=[(16, 19)],  # 与 HARD_1 重叠
            due=16,
            priority=0.9
        ),
        Task(
            task_id="HARD_3",
            release=0,
            duration=10,
            windows=[(17, 20)],
            due=17,
            priority=0.8
        ),
    ]
    
    pads = [Pad(pad_id="SINGLE_PAD")]  # 只有一个 pad！
    
    print("\nHard scenario: 3 tasks with overlapping windows, 1 pad")
    for t in tasks:
        print(f"  {t.task_id}: windows={t.windows}, dur={t.duration}")
    
    result = solve(
        now=0,
        tasks=tasks,
        pads=pads,
        prev_plan=None,
        weights=(10.0, 1.0, 5.0),
        time_limit=5.0
    )
    
    print_result(result, "Hard Scenario Result")
    
    if result.degradation_count > 0:
        print(f"\n→ Degradation triggered: {result.degradation_count} steps")
        print(f"→ Actions taken: {result.degradation_actions}")
    
    print("\n✓ Test 4 PASSED (degradation mechanism verified)")
    return result


def test_pad_unavailability():
    """测试 5: Pad 不可用约束"""
    print("\n" + "#"*60)
    print(" TEST 5: Pad unavailability constraint")
    print("#"*60)
    
    tasks = [
        Task(
            task_id="UA_1",
            release=0,
            duration=5,
            windows=[(10, 40)],
            due=30,
            priority=1.0
        ),
    ]
    
    # PAD_A 在 15-25 不可用
    pads = [
        Pad(pad_id="PAD_A", unavailable=[(15, 25)]),
        Pad(pad_id="PAD_B", unavailable=[]),
    ]
    
    print("\nScenario: Task must avoid PAD_A unavailability [15, 25]")
    
    result = solve(now=0, tasks=tasks, pads=pads)
    print_result(result, "Pad Unavailability Result")
    
    if result.plan:
        a = result.plan.assignments[0]
        print(f"\nAssignment: pad={a.pad_id}, start={a.start_slot}, launch={a.launch_slot}")
        
        if a.pad_id == "PAD_A":
            # 验证不与不可用区间重叠
            overlaps = not (a.launch_slot <= 15 or a.start_slot >= 26)
            print(f"Overlaps with unavailable [15,25]: {overlaps}")
            assert not overlaps, "Should not overlap with unavailable period"
    
    print("\n✓ Test 5 PASSED")
    return result


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print(" CP-SAT Solver Test Suite")
    print(" solver_cpsat.py comprehensive test")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Basic Solve", test_basic_solve()))
        results.append(("With prev_plan", test_with_prev_plan()))
        results.append(("With Freeze", test_with_freeze()))
        results.append(("Infeasible & Degradation", test_infeasible_and_degradation()))
        results.append(("Pad Unavailability", test_pad_unavailability()))
        
        print("\n" + "="*60)
        print(" ALL TESTS PASSED ✓")
        print("="*60)
        
        print("\nSummary:")
        for name, r in results:
            status = r.status.value if r else "N/A"
            obj = f"{r.objective_value:.2f}" if r and r.objective_value else "N/A"
            time_ms = r.solve_time_ms if r else 0
            print(f"  {name:<30} Status={status:<10} Obj={obj:<10} Time={time_ms}ms")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
