# 火箭发射排程仿真系统 (Launch Scheduling Simulation)

> **LLM 元参数调控 + CP-SAT 求解 + Rolling Horizon 仿真**

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
  - w_delay ∈ [5, 10, 20, 50]
  - w_shift ∈ [0, 0.2, 1, 2]
  - w_switch ∈ [0, 60, 180, 600]
  - FREEZE_HORIZON ∈ [0, 2, 6, 12] (小时)

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

# 【新增】分阶段执行 - 仅训练（调参）
python run_experiments.py --mode train-only --train-seeds 60 --output results/

# 【新增】分阶段执行 - 仅测试（评估）
python run_experiments.py --mode test-only --test-seeds 60 --output results/

# 【新增】测试阶段指定参数文件
python run_experiments.py --mode test-only --test-seeds 60 --load-params results/best_params.json --output results/

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

### 9.1 运行模式说明

#### **模式 1: 完整流程 (--mode full, 默认)**

```bash
python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/
```

一次性完成：
1. 训练集调参（60 episodes × 256 组合 = 15,360 次仿真）
2. 保存最优参数
3. 测试集评估（60 episodes × 5 策略 = 300 次仿真）
4. 训练集完整评估（60 episodes × 5 策略 = 300 次仿真）
5. 生成所有结果文件

**优点**: 一次性完成，无需手动操作  
**缺点**: 耗时较长（~15,960 次仿真）

---

#### **模式 2: 分阶段执行（推荐用于大规模实验）**

**阶段 1 - 训练调参 (--mode train-only)**

```bash
python run_experiments.py --mode train-only --train-seeds 60 --output results/
```

执行内容：
- ✅ 网格搜索 256 组参数
- ✅ 保存 `best_params.json`
- ✅ 保存 `tuning_results.csv`
- ✅ 生成训练集的 `results_per_episode.csv` 和 `summary.csv`

输出文件：
```
results/
├── best_params.json          # 最优参数 ⭐
├── tuning_results.csv        # 所有 256 组合的结果
├── results_per_episode.csv   # 训练集每个 episode 详细指标
└── summary.csv               # 训练集汇总统计
```

**阶段 2 - 测试评估 (--mode test-only)**

```bash
python run_experiments.py --mode test-only --test-seeds 60 --output results/
```

执行内容：
- ✅ 自动加载 `results/best_params.json`
- ✅ 在测试集上评估 5 个策略
- ✅ 合并训练集和测试集数据
- ✅ 更新 `results_per_episode.csv` 和 `summary.csv`

**优点**: 
- 可以在不同时间执行（例如训练在白天，测试在夜间）
- 可以修改测试集大小而无需重新调参
- 可以使用不同的硬件资源

---

#### **模式 3: 指定参数文件测试**

如果你想使用自定义参数或其他训练结果：

```bash
python run_experiments.py --mode test-only --test-seeds 60 \
                          --load-params /path/to/custom_params.json \
                          --output results_custom/
```

---

### 9.2 批量实验流程 (完整流程)

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
│     - w_delay: [5, 10, 20, 50]                                  │
│     - w_shift: [0, 0.2, 1, 2]                                   │
│     - w_switch: [0, 60, 180, 600]                               │
│     - 综合目标: avg_delay + 5.0 × episode_drift                 │
│     → 选出最优参数 → fixed_tuned 策略                            │
├─────────────────────────────────────────────────────────────────┤
│  3. 测试集配对评估                                                │
│     - 策略列表: fixed_tuned, fixed_default, nofreeze, greedy,   │
│                mockllm                                           │
│     - 同一 seed 运行所有策略，保证配对公平性                       │
├─────────────────────────────────────────────────────────────────┤
│  4. 输出文件                                                     │
│     - results_per_episode.csv: 每 episode 详细指标               │
│     - summary.csv: 按策略/扰动级别汇总统计                        │
│     - tuning_results.csv: 网格搜索所有组合结果                    │
│     - best_params.json: 最优参数                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 分阶段执行示例

**场景：大规模实验，需要分开执行**

```bash
# 第一步：训练阶段（耗时较长，可在服务器上运行）
python run_experiments.py --mode train-only \
                          --train-seeds 60 \
                          --workers 8 \
                          --output results/

# 输出: results/best_params.json (例如：w_delay=20, w_shift=1.0, w_switch=180, freeze_horizon=12)

# 第二步：测试阶段（可在本地或不同时间运行）
python run_experiments.py --mode test-only \
                          --test-seeds 60 \
                          --workers 8 \
                          --output results/

# 第三步：生成图表分析
python analyze.py --input results/ --output figures/
```

**场景：使用不同的测试集大小**

```bash
# 训练一次
python run_experiments.py --mode train-only --train-seeds 60 --output results/

# 小规模测试
python run_experiments.py --mode test-only --test-seeds 30 --output results_test30/

# 大规模测试
python run_experiments.py --mode test-only --test-seeds 120 --output results_test120/
```

**场景：使用自定义参数跳过调参**

```bash
# 手动创建 custom_params.json
echo '{"w_delay": 15.0, "w_shift": 0.5, "w_switch": 100, "freeze_horizon": 18}' > custom_params.json

# 直接在测试集上评估
python run_experiments.py --mode test-only \
                          --test-seeds 60 \
                          --load-params custom_params.json \
                          --output results_custom/
```

---

### 9.4 扰动强度定义

| 级别 | p_weather | p_pad_outage | σ_duration | σ_release |
|------|-----------|--------------|------------|-----------|
| light | 0.01 | 0.005 | 0.05 | 1.0 slots |
| medium | 0.02 | 0.01 | 0.1 | 2.0 slots |
| heavy | 0.04 | 0.02 | 0.15 | 3.0 slots |

### 9.5 分析脚本 (analyze.py)

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
├── simulator.py            # 扰动模型
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
