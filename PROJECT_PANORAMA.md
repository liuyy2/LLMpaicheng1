# 火箭发射排程系统 - 项目全景文档

> **文档目的**：为 AI 助手和新读者提供完整的项目理解框架，重点突出场景设计、约束建模、扰动机制和评价体系。

**最后更新**：2026年1月27日  
**核心技术栈**：Python 3.12 + OR-Tools CP-SAT + OpenAI兼容LLM API  
**研究方向**：LLM元参数调控 + 约束规划求解器 + Rolling Horizon动态重调度

---

## 📚 快速导航

1. [项目概览](#1-项目概览) - 研究目标与创新点
2. [场景设计详解](#2-场景设计详解) ⭐ **重点**
3. [扰动机制](#3-扰动机制) - 动态不确定性建模
4. [求解器模型](#4-求解器模型) - CP-SAT约束与目标
5. [仿真框架](#5-仿真框架) - Rolling Horizon闭环
6. [策略体系](#6-策略体系) - LLM vs Baseline
7. [评价指标](#7-评价指标) - 性能与稳定性
8. [代码结构](#8-代码结构) - 模块组织
9. [实验指南](#9-实验指南) - 如何运行

---

## 1. 项目概览

## 1. 场景抽象

### 1.1 任务 (Task) 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `task_id` | str | 唯一标识，如 `"T001"` |
| `release` | int | 最早可开始 slot（0-based），任务就绪时刻 |
| `duration` | int | 占用 pad 的 slot 数量（含集成/转运/上塔/加注/倒计时的总和） |
| `windows` | List[Tuple[int,int]] | 允许发射的时间窗口列表，每个 `(start_slot, end_slot)` 为闭区间 |
| `due` | int | 软截止 slot，超过产生 delay penalty |
| `priority` | float | 优先级权重，范围 `[0.1, 1.0]`，用于加权 delay cost |
| `preferred_pad` | Optional[str] | 偏好 pad（可选），切换产生 switch cost |

### 1.2 发射台 (Pad) 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `pad_id` | str | 唯一标识，如 `"PAD_A"` |
| `capacity` | int | 同时可服务任务数（本模型恒为 1） |
| `unavailable` | List[Tuple[int,int]] | 不可用区间列表（pad outage） |

### 1.3 可选扩展

| 字段 | 说明 |
|------|------|
| `crew_required` | 任务所需 crew 类型（本版本不启用） |
| `separation` | 任务间最小间隔 slot（本版本不启用） |

### 1.4 时间离散化参数（可配置）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `SLOT_MINUTES` | 15 | 每个 slot 的分钟数（当前 V2.5） |
| `ROLLING_INTERVAL` | 12 | 每隔多少 slot 触发一次 rolling（当前 V2.5：3h） |
| `HORIZON_SLOTS` | 96 | 单次求解视野 H（当前 V2.5：24h=96 slots） |
| `SIM_TOTAL_SLOTS` | 480 | 仿真总长 T（当前 V2.5：5 天=480 slots） |

---





## 3. Rolling 闭环逻辑

### 3.1 时间轴

```
t=0      t=6     t=12    t=18    ...    t=432
 |--------|--------|--------|----------|
 rolling  rolling  rolling           END
```

### 3.2 每次 Tick 执行流程

```
1. 推进时间: now += ROLLING_INTERVAL
2. 应用扰动: apply_disturbance(state, now)
   - 检查 [now-ROLLING_INTERVAL, now) 内是否有天气/outage 事件
   - 更新 task.windows, pad.unavailable
   - 对刚完成任务标记 completed
3. 检查冻结: 
   - frozen_tasks = {task | task.start <= now + FREEZE_HORIZON}
   - frozen 任务的 (pad, start) 不可更改
4. 调用策略: policy.decide(state, now) -> meta_params (仅 LLM 策略)
5. 调用求解器: solve_cpsat(now, HORIZON, tasks, pads, prev_plan, frozen, weights)
6. 检查状态:
   - OPTIMAL/FEASIBLE: 更新 current_plan
   - INFEASIBLE: 执行强制重排（放宽约束/延后任务）
7. 记录日志: 保存 plan snapshot, metrics, solve_time
```

### 3.3 冻结逻辑 (Hard Freeze)

```
FREEZE_HORIZON: int  # 默认 12 slots = 2h

规则:
- 若 task.start_slot <= now + FREEZE_HORIZON:
    - start_slot 固定为当前值（不可变）
    - assigned_pad 固定为当前值（不可变）
- CP-SAT 中对应变量设为常量约束
```

### 3.4 强制重排 (Forced Reschedule)

当 CP-SAT 返回 INFEASIBLE 时：

```
1. 尝试放宽: 取消 soft due 约束，仅保留 hard 约束
2. 若仍不可行: 
   - 按 priority 降序逐个延后任务 release += HORIZON
   - 重新求解
3. 记录: infeasible_count++, forced_delay_tasks
```

---



## 4. CP-SAT 模型

### 4.1 决策变量

```
对每个任务 k ∈ Tasks:
  start[k]: IntVar, domain = [release[k], SIM_TOTAL_SLOTS - duration[k]]
  pad[k,p]: BoolVar, 对每个 pad p，表示是否分配到该 pad
  
  # 线性化辅助变量
  delay[k]: IntVar, domain = [0, MAX_DELAY]
  shifted[k]: BoolVar, 表示 start 是否相对上一轮变化
  switched[k]: BoolVar, 表示 pad 是否相对上一轮变化
```

### 4.2 约束

```
# C1: 每个任务恰好分配一个 pad
∀k: Σ_p pad[k,p] = 1

# C2: 窗口约束（必须在某个 window 内发射）
∀k: start[k] + duration[k] - 1 ∈ ⋃ windows[k]
    (用 AddAllowedAssignments 或分段约束实现)

# C3: Pad 容量（同一 pad 上任务不重叠）
∀p, ∀(k1,k2) where k1<k2:
    pad[k1,p] ∧ pad[k2,p] → NoOverlap(start[k1], dur[k1], start[k2], dur[k2])
    (用 AddNoOverlap 配合 IntervalVar)

# C4: Pad 不可用区间
∀k, ∀p, ∀unavail ∈ pad[p].unavailable:
    pad[k,p] → start[k] + duration[k] <= unavail.start 
             ∨ start[k] >= unavail.end

# C5: 冻结约束（hard constraint）
∀k ∈ frozen_tasks:
    start[k] = frozen_start[k]
    pad[k, frozen_pad[k]] = 1

# C6: Delay 定义
∀k: delay[k] >= start[k] + duration[k] - due[k]
    delay[k] >= 0
```

### 4.3 目标函数

```
Minimize:
    w_delay * Σ_k (priority[k] * delay[k])
  + w_shift * Σ_k shifted[k]
  + w_switch * Σ_k switched[k]

其中:
  - w_delay, w_shift, w_switch 为权重（LLM 可调）
  - 默认: w_delay=10, w_shift=1, w_switch=5
```

### 4.4 线性化技巧

**shifted[k] 的线性化:**

```
prev_start[k] 为上一轮计划的 start（常量）
delta_start[k]: IntVar  # 可正可负

delta_start[k] = start[k] - prev_start[k]

# |delta_start| > 0 → shifted = 1
# 使用 indicator 约束:
model.Add(delta_start[k] >= 1).OnlyEnforceIf(shifted[k])
model.Add(delta_start[k] <= -1).OnlyEnforceIf(shifted[k])
model.Add(delta_start[k] == 0).OnlyEnforceIf(shifted[k].Not())

# 更简洁: AddAbsEquality + 阈值
abs_delta[k] = |delta_start[k]|
shifted[k] = (abs_delta[k] >= 1)
```

**switched[k] 的线性化:**

```
prev_pad[k] 为上一轮分配的 pad index（常量）

# 若 pad[k, prev_pad[k]] = 0 则 switched[k] = 1
model.Add(switched[k] == 1 - pad[k, prev_pad[k]])
```

---



## 5. 稳定性指标 (Plan Stability Metrics)

### 5.1 单任务时间偏移

```
D_time_k = |start_k^{new} - start_k^{old}| / HORIZON_SLOTS

归一化: 除以视野长度，使其 ∈ [0, 1]
若任务为新增（上轮不存在）: D_time_k = 0
```

### 5.2 单任务 Pad 切换

```
D_pad_k = 1  if pad_k^{new} ≠ pad_k^{old}
        = 0  otherwise

若任务为新增: D_pad_k = 0
```

### 5.3 单任务归一化偏移

```
NPD_k = α * D_time_k + β * D_pad_k

默认 α=0.7, β=0.3
```

### 5.4 单次 Rolling 的 Plan Drift

```
PlanDrift_t = (1 / |K_common|) * Σ_{k ∈ K_common} NPD_k

K_common = 上轮与本轮都存在的任务集合（排除已完成和新增）
若 |K_common| = 0: PlanDrift_t = 0
```

### 5.5 Episode 总 Drift（单标量）

```
EpisodeDrift = (1 / T_rolls) * Σ_t PlanDrift_t

T_rolls = 仿真过程中 rolling 的总次数
```

---


---

# V2.1 / V2.5 场景补充（当前主线）

相对 V1 的关键差异：
- 场景实体为 Mission -> 6 个 Operation（Op1..Op6），带串行先后约束。
- 资源为 R_pad, R1, R2, R3, R4，capacity=1。
- 只有 Op6 有时间窗（多窗口），其他操作由先后约束 + 资源约束决定。
- 扰动类型略有差异（V2.1 用 closure_change 替代 pad_outage）。

操作-资源绑定（`generate_missions_v2_1`）：
- Op1 -> R1
- Op2 -> R2
- Op3 -> R3
- Op4 -> R_pad + R4
- Op5 -> R_pad
- Op6 -> R_pad + R3

Op6 窗口生成（`_generate_op6_windows`）：
- 窗口数量：`config.op6_windows_range`（默认 2..5）
- 窗口长度：`op6 duration + extra`，extra 来自 `config.op6_window_length_range`（默认 2..6）
- 窗口分布：基于 release 与累计时长确定的 earliest_start ~ latest_end 之间均匀分段生成

V2.1 扰动（`generate_disturbance_timeline_v2_1`）：
- weather：删除 Op6 时间窗片段
- closure_change：更新 R_pad 不可用区间
- duration：扰动 Op6 duration
- release：扰动 mission release

当前 drift 定义（`metrics.py`）：
- drift = 0.7 * 时间偏移归一化 + 0.3 * Op6 window switch

场景补强入口（优先改动点）：
- `scenario.py`: generate_missions_v2_1, _generate_op6_windows, _generate_closure_intervals
- `config.py`: slot_minutes/rolling_interval/horizon/sim_total + 扰动参数 + mission 范围
- `disturbance.py`: _apply_weather_disturbance/_apply_duration_disturbance/_apply_release_disturbance

建议阅读顺序：
1) scenario.py
2) disturbance.py
3) simulator.py
4) solver_cpsat.py
5) metrics.py
6) run_experiments.py

---

## 10. 场景补强与实验行动清单（交给 AI / 开发者）

下面给出一个可交付给 AI 的清单，便于后续自动化改进场景与基线选择：

- **优先级 P0（立刻执行）**
  - 增强 Op6 多窗口生成逻辑，允许异步多模态分布（短窗口/长窗口混合）。参见 [scenario.py](scenario.py)
  - 将 `Config` 中的 `slot_minutes`、`rolling_interval`、`horizon_slots` 作为命令行可配置项，并记录在 scenario.snapshot（见 [config.py](config.py)）。
  - 在 `disturbance.py` 中增加天气按空间（pad 级）差异化影响，支持部分窗口只受影响的实现（见 [disturbance.py](disturbance.py)）。

- **优先级 P1（短期改进）**
  - 为 V2.1 场景引入 mission-level failure 模式（随机取消/延后整次 mission），并统计对 downstream Ops 的影响。修改点：`scenario.py` 和 `disturbance.py`。
  - 在 `scenario.generate_missions_v2_1` 中加入任务簇（clustered launches）选项，增加高并发压力场景。
  - 添加基于历史窗口损失的自适应扰动（使天气概率随时间或事件累积变化）。

- **优先级 P2（中期 / 研究级）**
  - 引入 crew / resource skill constraints（`Resource` 扩展），并在 CP-SAT 中建模资源切换惩罚（见 [solver_cpsat.py](solver_cpsat.py)）。
  - 支持不同优先级策略集成：在线学习的 LLM 元参数更新（策略中添加训练 / 回传接口）和贝叶斯优化调参。

## 11. 推荐 Baseline（用于比较）

- **Greedy Earliest Due First（现有）**：快速启发式，作为最低门槛基线（文件：[policies/policy_greedy.py](policies/policy_greedy.py)）。
- **Fixed-weights CP-SAT（现有）**：不使用 LLM 元参数的 CP-SAT 调用，固定 w_delay/w_shift/w_switch。代码在 [policies/policy_fixed.py](policies/policy_fixed.py)。
- **No-Freeze CP-SAT**：禁用冻结逻辑以观察不稳定性与可行率差异（[policies/policy_nofreeze.py](policies/policy_nofreeze.py)）。
- **LLM 元参数策略（研究目标）**：使用 `policy_llm_meta.py`，由 LLM 输出 `MetaParams` 控制权重与 freeze_horizon。

比较建议：对每个 baseline 在同一 `seed` 下运行 N=30 次 episode，统计 EpisodeMetrics 中的 `episode_drift`, `on_time_rate`, `total_delay`, `feasible_rate`, `forced_replan_rate`。

## 12. 实验清单与验收准则

- 实验目标：验证新的场景设定是否能放大策略间差异并揭示稳定性-性能权衡。
- 基础实验：
  1. 固定 seeds（至少 30），分别运行：Greedy / Fixed-CP-SAT / NoFreeze / LLM-meta。
  2. 记录 `EpisodeMetrics`、`RollingSnapshot` 时间序列、求解时间分布。
  3. 可视化：PlanDrift_t 随时间变化、累积延迟分布、强制重排时间点统计。
- 验收准则（示例）：
  - 若 LLM-meta 在多数 seeds 下显著降低 PlanDrift（p<0.05）且 on_time_rate 不劣于 Fixed-CP-SAT，则视为成功。
  - 若 ForcedReplanRate 显著上升 >10%，需检查 LLM 输出是否过度保守/激进。

## 13. 代码地图（快速定位）

- 场景与生成: [scenario.py](scenario.py)
- 扰动应用: [disturbance.py](disturbance.py)
- 仿真主循环: [simulator.py](simulator.py)
- 求解器模型: [solver_cpsat.py](solver_cpsat.py)
- 策略集合: [policies](policies)
- 特征工程: [features.py](features.py)
- LLM 封装: [llm_client.py](llm_client.py)
- 指标与可视化数据: [metrics.py](metrics.py)
- 实验运行脚本: [run_experiments.py](run_experiments.py)

