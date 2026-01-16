# 火箭发射排程仿真系统 (Launch Scheduling Simulation)

> **LLM 元参数调控 + CP-SAT 求解 + Rolling Horizon 仿真**

---

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
| `SLOT_MINUTES` | 10 | 每个 slot 的分钟数 |
| `ROLLING_INTERVAL` | 6 | 每隔多少 slot 触发一次 rolling (60min) |
| `HORIZON_SLOTS` | 144 | 单次求解视野 H = 24h = 144 slots |
| `SIM_TOTAL_SLOTS` | 432 | 仿真总长 T = 72h = 432 slots |

---

## 2. 扰动模型 (Disturbance Model)

所有扰动在仿真开始前由 `seed` 预生成完整时间线，保证可复现。

### 2.1 天气扰动 → 窗口删除

```
触发: 每个 slot 独立以 p_weather 概率触发天气事件
影响: 删除未来 [t, t + weather_duration] 内的所有 windows
参数:
  - p_weather: 每 slot 触发概率，默认 0.02
  - weather_duration: 影响持续 slot 数，默认 U(6, 18) 即 1~3h
  - affected_ratio: 被删除 window 比例，默认 1.0（全删）
```

### 2.2 Pad Outage → 区间不可用

```
触发: 每个 slot 独立以 p_pad_outage 概率触发
影响: 该 pad 在 [t, t + outage_duration] 不可用
参数:
  - p_pad_outage: 每 slot 触发概率，默认 0.01
  - outage_duration: 不可用持续 slot 数，默认 U(3, 12) 即 30min~2h
```

### 2.3 Duration 噪声 → 乘性扰动

```
触发: 任务实际执行时
影响: actual_duration = ceil(nominal_duration * (1 + ε))
参数:
  - ε ~ N(0, σ_dur²)，默认 σ_dur = 0.1
  - 截断: ε ∈ [-0.3, +0.5]（不能缩太多、可以延长）
```

### 2.4 Release 噪声 → 加性扰动

```
触发: 仿真开始时对每个任务采样
影响: actual_release = nominal_release + δ
参数:
  - δ ~ N(0, σ_rel²)，默认 σ_rel = 2 slots
  - 截断: δ ∈ [-3, +6] slots
  - 下限: actual_release >= 0
```

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

## 6. 实验设计

### 6.1 Baseline / Ablation 定义

| ID | 名称 | 描述 |
|----|------|------|
| B1 | `fixed` | 固定权重 w_delay=10, w_shift=1, w_switch=5，无 LLM |
| B2 | `nofreeze` | 同 B1，但 FREEZE_HORIZON=0（无冻结） |
| B3 | `greedy` | 每次 rolling 贪心分配：按 due 排序，依次塞最早可用 slot |
| B4 | `tuned` | 在训练集上网格搜索最优固定权重，测试集评估 |
| L1 | `llm_meta` | LLM 每次 rolling 输出元参数 JSON，CP-SAT 求解 |

### 6.2 训练集 / 测试集划分

```
Seeds 划分:
  - 训练集: seeds 0-49 (50 episodes)
  - 测试集: seeds 50-99 (50 episodes)

配对原则:
  - 所有算法在同一 seed 下使用完全相同的扰动序列
  - 扰动由 seed 确定性生成，与算法无关
```

### 6.3 调参范围（训练集）

```
网格搜索:
  - w_delay ∈ [1, 5, 10, 20, 50]
  - w_shift ∈ [0.1, 0.5, 1, 2, 5]
  - w_switch ∈ [1, 2, 5, 10, 20]
  - FREEZE_HORIZON ∈ [0, 6, 12, 18, 24]

评估指标: avg_delay + λ * episode_drift, λ=5
```

---

## 7. 输出文件

### 7.1 每 Episode 日志

```
logs/
  episode_{seed}/
    rolling_log.jsonl      # 每次 rolling 的快照
    plan_snapshots.jsonl   # 每次计划的完整内容
    metrics_per_roll.csv   # 每次 rolling 的指标
    final_schedule.json    # 最终执行的完整排程
```

**rolling_log.jsonl 单行格式:**
```json
{
  "t": 12,
  "solve_status": "OPTIMAL",
  "solve_time_ms": 152,
  "num_tasks_in_horizon": 8,
  "num_frozen": 2,
  "plan_drift": 0.034,
  "infeasible": false
}
```

### 7.2 汇总文件

```
results/
  summary.csv              # 所有 episode 的汇总指标
  comparison.csv           # 各算法对比表
  
summary.csv 字段:
  seed, policy, avg_delay, max_delay, episode_drift, 
  total_switches, total_shifts, infeasible_count,
  avg_solve_time_ms, total_runtime_s
```

### 7.3 图表

```
figures/
  delay_drift_scatter.png     # X=avg_delay, Y=episode_drift, 按 policy 着色
  drift_distribution.png      # 各 policy 的 drift 箱线图
  solve_time_trend.png        # 求解时间随 t 的变化
  switch_histogram.png        # pad 切换次数分布
```

---

## 8. 运行命令

```bash
# 安装依赖
pip install ortools numpy pandas matplotlib

# 生成场景（可选，会自动生成）
python scenario.py --seeds 0-99 --output data/scenarios/

# 运行单个 episode
python simulator.py --policy fixed --seed 42 --output logs/

# 运行完整实验 (120 episodes: 60 train + 60 test)
python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/

# 快速测试模式 (18 episodes: 9 train + 9 test)
python run_experiments.py --quick --output results_quick/

# 自定义参数
python run_experiments.py --train-seeds 30 --test-seeds 30 \
                          --solver-timeout 15.0 \
                          --lambda 5.0 \
                          --output results/

# 分析结果并生成图表
python analyze.py --input results/ --output figures/

# 显示图表（不仅保存）
python analyze.py --input results/ --output figures/ --show
```

---

## 9. 实验运行详细说明

### 9.1 批量实验流程 (run_experiments.py)

```
┌─────────────────────────────────────────────────────────────────┐
│                     实验流程概览                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. 生成种子分配                                                  │
│     - 训练集: seeds 0 ~ (N_train-1)                             │
│     - 测试集: seeds N_train ~ (N_train + N_test - 1)            │
│     - 每个扰动级别各占 1/3                                        │
├─────────────────────────────────────────────────────────────────┤
│  2. 网格搜索调参 (仅训练集)                                       │
│     - freeze_horizon_hours: [0, 2, 6, 12]                       │
│     - w_shift: [0, 0.2, 1, 2]                                   │
│     - w_switch: [0, 60, 180, 600]                               │
│     - 综合目标: avg_delay + 5.0 × episode_drift                 │
│     → 选出最优参数 → fixed_tuned 策略                            │
├─────────────────────────────────────────────────────────────────┤
│  3. 测试集配对评估                                                │
│     - 策略列表: fixed_tuned, fixed_default, nofreeze, greedy    │
│     - 同一 seed 运行所有策略，保证配对公平性                       │
├─────────────────────────────────────────────────────────────────┤
│  4. 输出文件                                                     │
│     - results_per_episode.csv: 每 episode 详细指标               │
│     - summary.csv: 按策略/扰动级别汇总统计                        │
│     - tuning_results.csv: 网格搜索所有组合结果                    │
│     - best_params.json: 最优参数                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 扰动强度定义

| 级别 | p_weather | p_pad_outage | σ_duration | σ_release |
|------|-----------|--------------|------------|-----------|
| light | 0.01 | 0.005 | 0.05 | 1.0 slots |
| medium | 0.02 | 0.01 | 0.1 | 2.0 slots |
| heavy | 0.04 | 0.02 | 0.15 | 3.0 slots |

### 9.3 分析脚本 (analyze.py)

生成图表：
- `delay_vs_drift_scatter.png`: Delay × PlanDrift 散点图
- `replans_boxplot.png`: 重排次数箱线图
- `switches_boxplot.png`: 切换次数箱线图  
- `policy_comparison_bars.png`: 各策略综合对比条形图
- `metric_by_disturbance_avg_delay.png`: 按扰动级别对比 Delay
- `metric_by_disturbance_episode_drift.png`: 按扰动级别对比 Drift
- `solve_time_comparison.png`: 求解耗时对比
- `tuning_heatmap.png`: 调参热力图 (如有调参结果)

统计输出：
- `summary_with_tests.csv`: 含配对 t 检验 p-value 的汇总表

---

## 10. 预期输出文件列表

```
LLMpaicheng1/
├── README.md                 # 项目说明文档
├── INTERFACES.md             # 接口规范文档
├── config.py                 # 全局配置常量
├── scenario.py               # 场景生成
├── disturbance.py            # 扰动模型
├── solver_cpsat.py           # CP-SAT 求解器
├── simulator.py              # Rolling 仿真主循环
├── metrics.py                # 指标计算
├── features.py               # 特征提取器（LLM 输入）
├── policies/
│   ├── __init__.py           # 策略工厂
│   ├── base.py               # 策略基类
│   ├── fixed.py              # 固定权重策略
│   ├── nofreeze.py           # 无冻结策略
│   ├── greedy.py             # 贪心策略
│   └── mockllm.py            # Mock LLM 策略
├── run_experiments.py        # 批量实验 + 调参脚本
├── analyze.py                # 结果分析与绘图
├── run_one_episode.py        # 单 episode 运行
├── run_compare_policies_once.py  # 单次策略对比
├── test_solver.py            # 求解器单元测试
├── logs/                     # Episode 日志
│   └── episode_{policy}_{seed}/
│       ├── plan_t{tick}.json
│       └── metrics.json
├── results/                  # 实验结果
│   ├── results_per_episode.csv
│   ├── summary.csv
│   ├── tuning_results.csv
│   └── best_params.json
└── figures/                  # 图表
    ├── delay_vs_drift_scatter.png
    ├── replans_boxplot.png
    ├── switches_boxplot.png
    ├── policy_comparison_bars.png
    ├── metric_by_disturbance_avg_delay.png
    ├── metric_by_disturbance_episode_drift.png
    ├── solve_time_comparison.png
    └── tuning_heatmap.png
```
