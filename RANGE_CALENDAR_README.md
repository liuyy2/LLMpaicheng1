# Range Calendar + Range Closure 功能说明

## 概述

本次更新为排程系统添加了 **Range 日历** 和 **Range closure 扰动** 功能，同时引入了新的 **Op3b 联测工序** 和 **R_range_test 资源**。

## 核心功能

### 1. Range Availability Calendar（Range 日历）

#### 设计目标
为 Range 设施提供全局共享的可用时间窗口，模拟真实场景中 Range 的有限开放时间。

#### 实现细节
- **数据结构**: `range_calendar: Dict[day_index, List[Tuple[start_slot, end_slot]]]`
- **窗口配置**（默认参数）:
  - 每天 3 个可用窗口：
    - W1 = [day+12, day+28)  （12 slots = 3h）
    - W2 = [day+40, day+56)  （16 slots = 4h）
    - W3 = [day+68, day+84)  （16 slots = 4h）
  - 每天总可用时间：12 小时
  - 每窗口长度：16 slots（4小时）

#### 可行性硬校验
为防止 infeasible 实例，实现了以下校验机制：

1. **窗口长度校验**：
   ```python
   min_required_length = Op6_duration + 4 slots
   ```
   - 如果窗口长度 < `min_required_length`，自动扩大窗口
   - 如果无法扩大（越界），则将该天设为全天可用（兜底策略）

2. **交集逻辑**：
   ```
   candidate_windows_m = intersect(mission_windows_m, range_calendar)
   ```
   - Op6 的实际候选窗口 = 任务自带窗口 ∩ Range 日历窗口
   - 过滤掉长度 < Op6_duration 的窗口

### 2. Weather → Range Closure 扰动

#### 改动说明
- **旧模式**：Weather 扰动导致资源 unavailable（通用 downtime）
- **新模式**：Weather 扰动改为 **Range closure**，直接作用于 Range 日历窗口

#### 触发机制
- **概率配置**：
  - Light: 5% (p_weather_light)
  - Medium: 7% (p_weather_medium)
  - Heavy: 10% (p_weather_heavy)
- **Closure 生成**：
  ```python
  closure_len = UniformInt[6, 18] slots
  closure_start = UniformInt[day_start, day_end - closure_len]
  closure = [closure_start, closure_start + closure_len)
  ```

#### 应用方式（区间减法）
对当天 `range_calendar` 的每个窗口执行区间减法：

1. **无交集**：窗口不变
2. **完全覆盖**：窗口删除（cancel）
3. **部分覆盖**：窗口缩短（shrink）
4. **中间切分**：保留较长的一段（避免碎片化）

示例：
```python
原窗口: [(12, 28), (40, 56), (68, 84)]
Closure: [20, 50)
结果:   [(12, 19), (50, 56), (68, 84)]
```

#### 可行性护栏（必须满足）

**护栏 A**：当天 `range_calendar` 不能变空
- 如果 closure 后当天无窗口 → 重采样 closure（最多 10 次）
- 仍失败 → 跳过该 closure（记录 skipped）

**护栏 B**：不能让任何任务的 Op6 候选窗口变空
- 对每个未完成任务，检查：
  ```python
  W_final_m = intersect(mission_windows_m, range_calendar_after_closure)
  ```
- 如果任何任务 `W_final_m` 为空 → 重采样/跳过

### 3. Range Test 资源与 Op3b 工序

#### 新增资源
- **资源 ID**: `R_range_test`
- **Capacity**: 1（默认）
- **用途**: 用于 Range 联测工序

#### 新增工序 Op3b
- **位置**: Op3 → **Op3b** → Op4
- **持续时间**: 默认 2 slots（0.5h，可配置）
- **资源需求**: `R3` + `R_range_test`
- **前序关系**:
  ```
  Op1(R1) → Op2(R2) → Op3(R3) → Op3b(R3+R_range_test) 
    → Op4(R_pad+R4) → Op5(R_pad) → Op6(R_pad+R3)
  ```

#### CP-SAT 模型更新
- Op3b 自动通过 `precedences` 字段处理前序约束
- 资源冲突检测：`R_range_test` 加入 NoOverlap 约束
- Op6 窗口约束使用交集后的 `candidate_windows`

### 4. 删除/禁用 Release 扰动

- **配置开关**: `enable_release_disturbance = False`（默认关闭）
- **原因**: 简化扰动复杂度，优先保证可行性
- **保留**：代码逻辑仍存在，可通过配置重新启用

## 配置参数

### config.py 新增配置

```python
# Range Calendar
enable_range_calendar: bool = True
range_calendar_windows_per_day: int = 3
range_calendar_window_length: int = 16  # slots
range_calendar_window_starts: List[int] = [12, 40, 68]

# Range Test Asset
enable_range_test_asset: bool = True
range_test_resource_id: str = "R_range_test"
range_test_capacity: int = 1
op3b_duration_slots: int = 2

# Weather Mode
weather_mode: str = "range_closure"  # "legacy" | "range_closure"
range_closure_duration_range: Tuple[int, int] = (6, 18)
max_resample_attempts_for_closure: int = 10

# Probabilities
p_weather_light: float = 0.05
p_weather_medium: float = 0.07
p_weather_heavy: float = 0.10

# Release Disturbance
enable_release_disturbance: bool = False  # 默认禁用
```

## 可复现性

- **随机种子**: 所有随机生成（窗口、扰动、closure）均基于固定 seed
- **确定性**: 同 seed 保证完全相同的场景和扰动序列

## 验收标准

### 场景生成
✓ 每天 `range_calendar` 至少有一个窗口  
✓ 每个未完成任务至少有一个 `candidate_windows`（否则 closure 被跳过）

### 求解器
✓ Light/Medium/Heavy 各抽样 30 个 seed（10-15 missions，5天仿真）  
✓ Infeasible 比例 < 3%  
✓ 运行时间不显著增加（相比未加 Op3b 版本）

### 事件记录
✓ `analyze` 输出能区分：`pad_outage`、`range_closure`、`duration`

## 使用示例

### 基础场景生成
```python
from config import Config
from scenario import generate_scenario

config = Config(
    enable_range_calendar=True,
    enable_range_test_asset=True,
    weather_mode="range_closure",
    num_missions_range=(10, 15),
    sim_total_slots=480  # 5 days
)

scenario = generate_scenario(seed=42, config=config)

# 查看 Range 日历
print(scenario.range_calendar)
# {0: [(12, 28), (40, 56), (68, 84)], 1: [...], ...}

# 查看 R_range_test 资源
range_test_res = scenario.get_resource("R_range_test")
print(f"Capacity: {range_test_res.capacity}")
```

### 运行仿真
```python
from simulator import run_simulation
from policies.baseline import FixedParamPolicy

policy = FixedParamPolicy()
result = run_simulation(scenario, policy, config)

print(f"Completed missions: {len(result.completed_tasks)}")
print(f"Final makespan: {result.metrics.makespan}")
```

## 测试

运行测试套件：
```bash
python test_range_calendar.py
```

测试覆盖：
1. Range Calendar 生成
2. Op6 候选窗口交集计算
3. Range closure 可行性护栏
4. Op3b 工序生成
5. Release 扰动禁用验证
6. Range closure 事件生成

## 注意事项

1. **Op3b 索引**：内部使用 `op_index=4` 表示 Op3b，Op4-Op6 分别为 5、6、7
2. **窗口过滤**：在每次求解前动态过滤 Op6 窗口，不修改原始 `mission.operations`
3. **护栏日志**：被跳过的 closure 事件会记录在 `applied_events` 中（但不应用）
4. **向后兼容**：可通过配置关闭新功能，回退到 V2.1 行为：
   ```python
   enable_range_calendar=False,
   enable_range_test_asset=False,
   weather_mode="legacy"
   ```

## 性能优化建议

- **大规模实例**（>30 missions）：考虑增加 `op3b_duration_slots`（减少资源冲突）
- **高扰动频率**：适当降低 `p_weather_*` 概率
- **窗口碎片化**：调整 `range_calendar_window_starts` 使窗口更分散

## 常见问题

**Q: 为什么某些 closure 事件没有生效？**  
A: 可能触发了可行性护栏，被自动跳过以避免 infeasible。检查日志中的 `skipped` 标记。

**Q: 如何增加 Range 可用时间？**  
A: 增加 `range_calendar_window_length` 或添加更多 `range_calendar_window_starts`。

**Q: Op3b 资源冲突导致求解慢？**  
A: 增加 `range_test_capacity` 或减少 Op3b 持续时间 `op3b_duration_slots`。

**Q: 想恢复旧的 weather 扰动行为？**  
A: 设置 `weather_mode="legacy"`。

---

**版本**: V2.5  
**最后更新**: 2026-02-04  
**作者**: AI Assistant
