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

V2.1 / V2.5 场景补充（当前主线）

场景要点：
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
- `simulator.py`: _apply_*_disturbance_ops handlers

建议阅读顺序：
1) scenario.py
2) simulator.py
3) solver_cpsat.py
4) metrics.py
5) run_experiments.py

---

## 10. 场景补强与实验行动清单（交给 AI / 开发者）

下面给出一个可交付给 AI 的清单，便于后续自动化改进场景与基线选择：

- **优先级 P0（立刻执行）**
  - 增强 Op6 多窗口生成逻辑，允许异步多模态分布（短窗口/长窗口混合）。参见 [scenario.py](scenario.py)
  - 将 `Config` 中的 `slot_minutes`、`rolling_interval`、`horizon_slots` 作为命令行可配置项，并记录在 scenario.snapshot（见 [config.py](config.py)）。
  - 在 `simulator.py` 中增加天气按空间（pad 级）差异化影响，支持部分窗口只受影响的实现（见 [simulator.py](simulator.py)）。

- **优先级 P1（短期改进）**
  - 为 V2.1 场景引入 mission-level failure 模式（随机取消/延后整次 mission），并统计对 downstream Ops 的影响。修改点：`scenario.py` 和 `simulator.py`。
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
- 扰动应用: [simulator.py](simulator.py)
- 仿真主循环: [simulator.py](simulator.py)
- 求解器模型: [solver_cpsat.py](solver_cpsat.py)
- 策略集合: [policies](policies)
- 特征工程: [features.py](features.py)
- LLM 封装: [llm_client.py](llm_client.py)
- 指标与可视化数据: [metrics.py](metrics.py)
- 实验运行脚本: [run_experiments.py](run_experiments.py)

