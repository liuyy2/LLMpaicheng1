"""
Range Calendar + Range Closure 功能测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from scenario import generate_scenario
from simulator import _compute_op6_candidate_windows, _apply_range_closure_ops, SimulationStateOps


def test_range_calendar_generation():
    """测试 Range Calendar 生成"""
    print("=== Test 1: Range Calendar Generation ===")
    
    config = Config(
        enable_range_calendar=True,
        sim_total_slots=960,  # 10 days
        num_missions=10
    )
    
    scenario = generate_scenario(seed=42, config=config)
    
    # 验证 range_calendar 存在
    assert hasattr(scenario, 'range_calendar'), "Scenario should have range_calendar"
    assert scenario.range_calendar, "range_calendar should not be empty"
    
    # 验证每天至少有一个窗口
    slots_per_day = 96
    num_days = (config.sim_total_slots + slots_per_day - 1) // slots_per_day
    
    for day in range(num_days):
        assert day in scenario.range_calendar, f"Day {day} should be in range_calendar"
        windows = scenario.range_calendar[day]
        assert len(windows) > 0, f"Day {day} should have at least one window"
        
        # 验证窗口格式
        for win_start, win_end in windows:
            assert win_start < win_end, f"Window [{win_start}, {win_end}) should be valid"
            assert win_end - win_start >= 4, f"Window should be at least 4 slots long"
    
    print(f"✓ Range calendar generated for {num_days} days")
    print(f"  Example day 0 windows: {scenario.range_calendar[0]}")
    print()


def test_op6_candidate_windows():
    """测试 Op6 候选窗口计算（mission_windows ∩ range_calendar）"""
    print("=== Test 2: Op6 Candidate Windows Computation ===")
    
    # 构造测试数据
    mission_windows = [(100, 150), (200, 250)]
    range_calendar = {
        1: [(96, 120), (150, 180)],  # day 1
        2: [(192, 220), (240, 260)]  # day 2
    }
    op6_duration = 8
    
    candidate_windows = _compute_op6_candidate_windows(
        mission_windows, range_calendar, op6_duration
    )
    
    print(f"  Mission windows: {mission_windows}")
    print(f"  Range calendar: {range_calendar}")
    print(f"  Candidate windows: {candidate_windows}")
    
    # 验证候选窗口不为空
    assert len(candidate_windows) > 0, "Should have at least one candidate window"
    
    # 验证所有候选窗口长度 >= op6_duration
    for win_start, win_end in candidate_windows:
        assert win_end - win_start >= op6_duration, \
            f"Candidate window [{win_start}, {win_end}) should be >= {op6_duration}"
    
    print("✓ Candidate windows computed correctly")
    print()


def test_range_closure_feasibility_guard():
    """测试 Range closure 可行性护栏"""
    print("=== Test 3: Range Closure Feasibility Guard ===")
    
    config = Config(
        enable_range_calendar=True,
        weather_mode="range_closure",
        sim_total_slots=960,
        num_missions=5
    )
    
    scenario = generate_scenario(seed=123, config=config)
    
    # 创建模拟状态
    state = SimulationStateOps(
        now=0,
        missions=scenario.missions,
        resources=scenario.resources,
        current_plan=None
    )
    
    # 测试 closure 事件（应用前记录）
    from scenario import DisturbanceEvent
    
    day = 0
    original_windows = scenario.range_calendar[day].copy()
    
    # 创建一个会覆盖所有窗口的 closure（应被护栏阻止）
    closure_event = DisturbanceEvent(
        event_type="range_closure",
        trigger_time=0,
        target_id=None,
        params={
            "day": day,
            "closure_start": 0,
            "closure_end": 96  # 覆盖整天
        }
    )
    
    # 应用 closure
    _apply_range_closure_ops(state, closure_event, scenario)
    
    # 验证：窗口不应该被清空（护栏保护）
    assert len(scenario.range_calendar[day]) > 0, \
        "Range calendar should not be empty (guard should prevent)"
    
    print(f"  Day {day} original windows: {original_windows}")
    print(f"  After full-day closure: {scenario.range_calendar[day]}")
    print("✓ Feasibility guard prevented window clearing")
    print()


def test_op3b_generation():
    """测试 Op3b 工序生成"""
    print("=== Test 4: Op3b Operation Generation ===")
    
    config = Config(
        enable_range_test_asset=True,
        op3b_duration_slots=2,
        num_missions_range=(5, 5)
    )
    
    scenario = generate_scenario(seed=456, config=config)
    
    # 验证 R_range_test 资源存在
    range_test_resource = None
    for res in scenario.resources:
        if res.resource_id == config.range_test_resource_id:
            range_test_resource = res
            break
    
    assert range_test_resource is not None, "R_range_test resource should exist"
    print(f"✓ Resource {config.range_test_resource_id} created with capacity={range_test_resource.capacity}")
    
    # 验证每个任务都有 Op3b
    for mission in scenario.missions:
        op3b = None
        for op in mission.operations:
            if op.op_id.endswith("Op3b"):
                op3b = op
                break
        
        assert op3b is not None, f"Mission {mission.mission_id} should have Op3b"
        assert op3b.op_index == 4, "Op3b should have index 4"
        assert op3b.duration == config.op3b_duration_slots, \
            f"Op3b duration should be {config.op3b_duration_slots}"
        assert "R3" in op3b.resources, "Op3b should use R3"
        assert config.range_test_resource_id in op3b.resources, \
            f"Op3b should use {config.range_test_resource_id}"
        
        # 验证前序关系：Op3 -> Op3b -> Op4
        assert f"{mission.mission_id}_Op3" in op3b.precedences, \
            "Op3b should have Op3 as precedence"
        
        # 查找 Op4（op_id 为 "M???_Op4"）
        op4 = None
        for op in mission.operations:
            if op.op_id == f"{mission.mission_id}_Op4":
                op4 = op
                break
        
        if op4:
            assert f"{mission.mission_id}_Op3b" in op4.precedences, \
                "Op4 should have Op3b as precedence"
    
    print(f"✓ All {len(scenario.missions)} missions have Op3b with correct setup")
    print()


def test_range_closure_events():
    """测试 range_closure 事件生成"""
    print("=== Test 6: Range Closure Events Generation ===")
    
    config = Config(
        weather_mode="range_closure",
        enable_range_calendar=True,
        num_missions_range=(10, 15)
    )
    
    scenario = generate_scenario(seed=999, config=config)
    
    # 检查是否有 range_closure 事件
    closure_events = [e for e in scenario.disturbance_timeline 
                      if e.event_type == "range_closure"]
    
    print(f"  Generated {len(closure_events)} range_closure events")
    
    if closure_events:
        # 验证事件格式
        for event in closure_events[:3]:  # 显示前 3 个
            assert 'day' in event.params, "Event should have 'day' parameter"
            assert 'closure_start' in event.params, "Event should have 'closure_start'"
            assert 'closure_end' in event.params, "Event should have 'closure_end'"
            print(f"    Event: day={event.params['day']}, "
                  f"closure=[{event.params['closure_start']}, {event.params['closure_end']})")
    
    print("✓ Range closure events generated correctly")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Range Calendar + Range Closure Feature Tests")
    print("=" * 70)
    print()
    
    try:
        test_range_calendar_generation()
        test_op6_candidate_windows()
        test_range_closure_feasibility_guard()
        test_op3b_generation()
        test_range_closure_events()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
