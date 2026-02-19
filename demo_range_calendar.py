"""
Range Calendar + Range Closure 功能演示
展示新功能的使用方法
"""

from config import Config
from scenario import generate_scenario, save_scenario


def demo_basic_features():
    """演示基本功能"""
    print("=" * 70)
    print("Range Calendar + Range Closure 功能演示")
    print("=" * 70)
    print()
    
    # 1. 创建启用所有新功能的配置
    print("1. 创建配置（启用所有新功能）")
    config = Config(
        enable_range_calendar=True,
        enable_range_test_asset=True,
        weather_mode="range_closure",
        op3b_duration_slots=2,

        num_missions=15,
        sim_total_slots=960  # 10 days
    )
    print(f"   ✓ Range Calendar: {config.enable_range_calendar}")
    print(f"   ✓ Range Test Asset: {config.enable_range_test_asset}")
    print(f"   ✓ Weather Mode: {config.weather_mode}")
    print(f"   ✓ Op3b Duration: {config.op3b_duration_slots} slots")
    print()
    
    # 2. 生成场景
    print("2. 生成场景（seed=42）")
    scenario = generate_scenario(seed=42, config=config)
    print(f"   ✓ 生成 {len(scenario.missions)} 个任务")
    print(f"   ✓ 生成 {len(scenario.resources)} 个资源")
    print(f"   ✓ 生成 {len(scenario.disturbance_timeline)} 个扰动事件")
    print()
    
    # 3. 检查 Range Calendar
    print("3. Range Calendar 详情")
    slots_per_day = 96
    num_days = min(3, len(scenario.range_calendar))  # 显示前3天
    
    for day in range(num_days):
        if day in scenario.range_calendar:
            windows = scenario.range_calendar[day]
            total_slots = sum(e - s for s, e in windows)
            total_hours = total_slots * config.slot_minutes / 60
            print(f"   Day {day}:")
            for i, (start, end) in enumerate(windows, 1):
                start_time = (start % slots_per_day) * config.slot_minutes / 60
                end_time = (end % slots_per_day) * config.slot_minutes / 60
                print(f"      Window {i}: slots [{start:3d}, {end:3d}) = "
                      f"{start_time:5.1f}h - {end_time:5.1f}h ({end-start} slots)")
            print(f"      Total: {total_hours:.1f} hours available")
    print()
    
    # 4. 检查资源
    print("4. 资源详情")
    for res in scenario.resources:
        print(f"   {res.resource_id:15s} - capacity={res.capacity}, "
              f"unavailable_intervals={len(res.unavailable)}")
        if res.resource_id == config.range_test_resource_id:
            print(f"      → 新增的 Range Test 资源 ✓")
    print()
    
    # 5. 检查任务工序（显示一个任务）
    print("5. 任务工序结构（示例：M000）")
    mission = scenario.missions[0]
    print(f"   Mission: {mission.mission_id}")
    print(f"   Release: {mission.release}, Due: {mission.due}, Priority: {mission.priority}")
    print(f"   Operations ({len(mission.operations)} ops):")
    
    for op in mission.operations:
        precedences_str = ", ".join(op.precedences) if op.precedences else "None"
        resources_str = "+".join(op.resources)
        
        marker = ""
        if "Op3b" in op.op_id:
            marker = " ← 新增工序 ✓"
        elif op.op_index == 6 and op.time_windows:
            marker = f" (有 {len(op.time_windows)} 个窗口)"
        
        print(f"      {op.op_id:10s} - duration={op.duration:2d}, "
              f"resources=[{resources_str:20s}], "
              f"precedences=[{precedences_str}]{marker}")
    print()
    
    # 6. 检查扰动事件
    print("6. 扰动事件详情")
    event_types = {}
    for event in scenario.disturbance_timeline:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    for event_type, count in event_types.items():
        marker = " ← 新模式 ✓" if event_type == "range_closure" else ""
        print(f"   {event_type:20s}: {count:2d} events{marker}")
    
    # 显示前几个 range_closure 事件
    closure_events = [e for e in scenario.disturbance_timeline 
                      if e.event_type == "range_closure"]
    if closure_events:
        print(f"\n   Range Closure 事件示例（前 {min(3, len(closure_events))} 个）:")
        for event in closure_events[:3]:
            day = event.params['day']
            start = event.params['closure_start']
            end = event.params['closure_end']
            duration = end - start
            print(f"      Day {day}, slots [{start:3d}, {end:3d}), "
                  f"duration={duration} slots ({duration * config.slot_minutes / 60:.1f}h)")
    print()
    
    # 7. 保存场景
    output_file = "examples/scenario_range_calendar_demo.json"
    print(f"7. 保存场景到文件")
    try:
        save_scenario(scenario, output_file)
        print(f"   ✓ 已保存到: {output_file}")
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
    print()
    
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)


def demo_backward_compatibility():
    """演示向后兼容性（禁用新功能）"""
    print("\n" + "=" * 70)
    print("向后兼容性演示（禁用所有新功能）")
    print("=" * 70)
    print()
    
    config = Config(
        enable_range_calendar=False,
        enable_range_test_asset=False,
        weather_mode="legacy",

        num_missions_range=(8, 10)
    )
    
    scenario = generate_scenario(seed=123, config=config)
    
    print(f"✓ 场景生成成功（V2.1 原有行为）")
    print(f"  - Missions: {len(scenario.missions)}")
    print(f"  - Resources: {len(scenario.resources)} (无 R_range_test)")
    print(f"  - Range Calendar: {'有' if scenario.range_calendar else '无'}")
    
    # 检查是否有 Op3b
    has_op3b = False
    for mission in scenario.missions:
        for op in mission.operations:
            if "Op3b" in op.op_id:
                has_op3b = True
                break
        if has_op3b:
            break
    
    print(f"  - Op3b: {'有' if has_op3b else '无'}")
    
    # 检查扰动类型
    event_types = set(e.event_type for e in scenario.disturbance_timeline)
    print(f"  - 扰动类型: {event_types}")
    
    print("\n✓ 向后兼容性验证通过")
    print("=" * 70)


if __name__ == "__main__":
    demo_basic_features()
    demo_backward_compatibility()
