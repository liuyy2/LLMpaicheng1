# Mission-6Ops + 单Pad + Op6 windows + rolling+freeze

本文档是“可执行 plan”的落地说明：需要改动的内容、验证方法、以及判断是否有效的标准。
不保留旧版本（以本文为准）。

## 1. Drift 定义（执行版）

**锚点（只算两个承诺点）**
- PadHoldStart = start(Op4)
- LaunchTime = start(Op6)
- PadHoldEnd = end(Op6)（可选，仅用于可视化/统计，不强制进 drift）

**连续型漂移（单位：slot，15min）**
- Launch drift: |s^t_{m,6} - s^{t-1}_{m,6}|
- Pad drift: |s^t_{m,4} - s^{t-1}_{m,4}|
- Drift_time(m) = 0.7 * Launch drift + 0.3 * Pad drift

**离散型漂移（两类 switch）**
- Window switch: 1[win^t_m != win^{t-1}_m]
- Sequence switch（单 pad 排队关系变化）:
  - 以 PadHoldStart 排序，定义 pred(m) = pad 上直接前驱
  - Switch_seq(m) = 1[pred^t(m) != pred^{t-1}(m)]

**“可避免”处理（只惩罚策略造成的变化）**
- 如果上一轮窗口在当前扰动下不可行，则 Window switch = 0
- 仅统计：未开始且不在 freeze 内的任务/工序
- 仅统计：上一轮与当前计划都存在该任务（避免不存在时误罚）

**任务 drift（带优先级）**
Drift_m = p_m * (Drift_time(m) + kappa_win * Switch_win(m) + kappa_seq * Switch_seq(m))

**建议常数**
- kappa_win = 12 （约 3 小时）
- kappa_seq = 6  （约 1.5 小时）

## 2. 目标函数与求解（Lexicographic / epsilon-constraint）

**Stage 1（交付优先）**
- 目标：min sum_m p_m * Delay_m
- Delay_m = max(0, start(Op6) - due_m)
- 得到 D_opt

**Stage 2（稳定性优先，但不显著增加延迟）**
- 约束：total_delay <= (1 + epsilon_solver) * D_opt
- 目标：min sum_m p_m * [Drift_time + kappa_win*Switch_win + kappa_seq*Switch_seq]

**外层调参（两层 epsilon）**
- 搜索空间：freeze_horizon_hours × epsilon_solver_values（4 × 4 = 16）
- 选择准则：avg_delay <= baseline_delay * (1 + epsilon_tune) 时，取 avg_drift 最小

## 3. 落地改动清单（不保留旧版本）

1) config.py
- 增加默认项：use_two_stage_solver, epsilon_solver, kappa_win, kappa_seq, stage1_time_ratio

2) policies/base.py + policies/policy_fixed.py
- MetaParams 增加：use_two_stage / epsilon_solver / kappa_win / kappa_seq
- 固定策略透传这些参数

3) solver_cpsat.py
- SolverConfigV2_1 增加：use_two_stage / epsilon_solver / kappa_win / kappa_seq / stage1_time_ratio
- solve_v2_1：在 use_two_stage=True 时走两阶段求解
- Stage1：仅优化 delay（基于 Op6 start）
- Stage2：加入 total_delay 上界，优化 drift
- Sequence switch：基于 Op4 start 的“直接前驱”变化

4) metrics.py
- drift_v3：采用 launch/pad 两锚点 + 窗口/序列 switch
- 处理不可避免切换（窗口不可行 -> 不计）
- 只统计未开始/未冻结/未完成的任务
- RollingMetrics / EpisodeMetrics 新增 switch 统计

5) features.py
- delay 与 due 统一用 Op6 start

6) simulator.py
- 透传 solver 参数与 kappa
- 记录 rolling_metrics 的 switch 字段

7) run_experiments.py
- 调参维度从权重 w_* 转为 epsilon_solver
- 输出记录包含 window/sequence switch
- 选择规则：delay 约束内最小 drift

## 4. 验证步骤（建议顺序）

**基础导入/字段检查**
- 运行：`python test_run_experiments.py`
- 期望：EpisodeMetricsRecord 字段包含新 drift/switch 指标

**快速冒烟实验**
- 运行：`python run_experiments.py --quick --output results_quick --workers 1`
- 期望：生成 best_params.json、episode_results.csv、rolling_metrics.csv

**核心一致性检查**
- Delay 定义：核对 delay 是否基于 Op6 start（发射时刻）
- Drift 组成：确认 avg_time_shift_slots、num_window_switches、num_sequence_switches 出现在结果中
- Two-stage 约束：随机抽一条 episode，计算 total_delay，检查 <= (1+epsilon_solver)*D_opt

## 5. 判断“有效”的标准（量化）

- 在 avg_delay <= baseline_delay * (1 + epsilon_tune) 条件下，avg_drift 更低
- Window switch 与 Sequence switch 的数量合理减少
- Drift 不被 Op1-Op3 的内部变动放大（指标更稳定、解释更清晰）
- 在相同扰动水平下，计划稳定性更高且完成率不下降

---

如需按此文档逐步落地，请按第 3 节顺序执行并在第 4 节验证。
