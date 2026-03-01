# 论文研究概要

---

## 摘要

针对发射场多任务共享稀缺资源场景下高频扰动（天气窗口收缩、设备故障、工序延误）带来的动态调度重排需求，提出一种基于大语言模型（LLM）根因诊断与局部修复的火箭发射动态排程方法。首先，构建带靶场日历窗口约束与多类扰动的滚动重排仿真环境，形成以准时率、加权延迟、计划漂移（Drift）为核心的评价指标体系；然后，引入时序资源冲突图对扰动后的计划进行根因定位，提取冲突簇、瓶颈压力、紧急度等结构化特征，并以此为输入驱动 LLM 进行零样本推理，输出需要"解锁"的最小任务集；最后，基于 CP-SAT 求解器的 Anchor Fix-and-Optimize机制对解锁任务实施局部精确修复，辅以多级回退链保证方法在任意扰动下的可行性，并通过跨多难度等级、多随机种子的大规模对比实验表明所提方法在准时性与计划稳定性上的综合优越性。

---

## 一、研究问题

**研究问题**：如何在充满高频扰动（天气窗口收缩、设备故障、工序延误）的真实发射场环境下，对多任务共享稀缺资源的火箭发射调度计划实施高质量的在线动态重排？

**内涵**：问题本质是带发射窗口约束与资源竞争的动态资源约束项目调度（Dynamic RCPSP），核心矛盾在于"准时性（最小化延迟）"与"计划稳定性（最小化重排漂移 Drift）"的两目标权衡，同时对决策实时性有严格要求。

---

## 二、研究意义

**工程意义**：发射场资源高度稀缺，一次调度决策失误可导致轨道窗口丢失或多任务链式延误，自动化在线重排可大幅降低人工介入成本并提升发射成功率。

**学术意义**：首次将 LLM 作为"零样本元决策器"引入工业级动态调度场景，并提出以 TRCG 驱动的根因诊断-局部修复范式，为 LLM 辅助运筹优化提供了方法论参考。

---

## 三、国内外研究现状

**传统方法**：国内外在 RCPSP/动态调度领域已有大量 CP/MIP 精确求解与启发式（遗传算法、禁忌搜索、LNS 大邻域搜索）研究及 Rolling Horizon 框架，但普遍依赖手工调参且对多类扰动的泛化能力弱。

**LLM 与调度交叉**：近年涌现出将 LLM 用于组合优化提示（FunSearch、Hyper-Heuristic Prompting）的探索，但落地于复杂多约束动态调度并与精确求解器深度融合、具备根因诊断能力的工作尚属空白。

---

## 四、研究思路

以 Rolling Horizon 仿真框架为基座，设计 TRCG-LLM 局部修复架构：扰动发生后，TRCG 冲突图提取冲突簇与瓶颈特征，LLM 零样本推理最小解锁任务集，CP-SAT 的 Anchor Fix-and-Optimize 对解锁任务实施局部精确修复，辅以确定性启发式回退与四级降级链保证可行性；对照基线（FixedWeight、Greedy、GARepairPolicy）形成完整消融体系。

---

## 五、验证方法

在含 Range Calendar 与 Range Closure 扰动的高保真仿真器上，跨三种难度（Light/Medium/Heavy）、多随机种子批量运行（10 天周期），以加权延迟、计划漂移（`drift_per_replan`）、Pad 切换次数、求解时间为核心指标进行定量对比，辅以 Episode Case Study 双泳道甘特图做定性可视化分析。

---

## 六、问题建模

### 6.1 调度场景定义

- **任务（Mission）**：每个任务包含 7 个串行工序 Op1→Op2→Op3→Op3b→Op4→Op5→Op6，最后一道工序 Op6（发射）须落入离散发射窗口内。
- **资源**：发射台 $R_\text{pad}$（容量 1，关键瓶颈）、测试设备 $R_1, R_2, R_3, R_4$、Range 联测设备 $R_\text{range}$（容量 1）。
- **Range Calendar**：全局共享的发射场开放窗口，每天 3 段，Op6 的候选时间为任务轨道窗口与当日 Range Calendar 的交集。
- **Op5→Op6 最大间隔**：不超过 24 小时（燃料稳定性约束）。

### 6.2 两阶段词典序优化模型

**Stage 1**（准时性优先）：

$$\min \sum_{m} \text{priority}_m \cdot \max\!\left(0,\ s_{m,\text{Op6}} - d_m\right)$$

**Stage 2**（稳定性优先，$\varepsilon$-约束）：

$$\text{s.t.} \quad \sum_m \text{priority}_m \cdot \text{delay}_m \leq (1+\varepsilon) \cdot D^*$$

$$\min \sum_m \text{priority}_m \cdot \text{Drift}_m$$

**Drift 定义**（V3）：

$$\text{Drift}_m = 0.7\,\bigl|s_{\text{new},m}^{\text{Op6}} - s_{\text{old},m}^{\text{Op6}}\bigr| + 0.3\,\bigl|s_{\text{new},m}^{\text{Op4}} - s_{\text{old},m}^{\text{Op4}}\bigr| + \kappa_\text{win} \cdot \mathbf{1}[\text{窗口切换}] + \kappa_\text{seq} \cdot \mathbf{1}[\text{Pad 排序变化}]$$

其中 $\kappa_\text{win}=12$，$\kappa_\text{seq}=6$（等效 slot 惩罚）；仅对未开始、未冻结、未完成任务计算"可避免漂移"。

### 6.3 冻结机制（Freeze Horizon）

$$\text{frozen\_ops} = \left\{\text{op} \mid \text{op 已开始} \;\lor\; s_\text{op} \leq t_\text{now} + H_\text{freeze}\right\}$$

冻结工序的时间与资源分配在本次重排中保持不变，$H_\text{freeze} \in \{0,4,8,16,24\}$ 小时。

---

## 七、方法设计

### 7.1 整体架构

```
扰动事件
    │
    ▼
TRCG 诊断（features.py）
    │  冲突图 + 聚类 + 紧急度 + 瓶颈压力
    ▼
LLM 零样本推理（policy_llm_repair.py）
    │  输入：TRCGSummary（结构化文本）
    │  输出：RepairDecision（unlock 集合 + 修复参数）
    ▼
四级校验（validate_repair_decision）
    │  失败 → 确定性启发式回退（heuristic_repair_decision）
    ▼
Anchor Fix-and-Optimize（solver_cpsat.py）
    │  非 unlock 任务的 Op4/Op6 锚定 → Pseudo-LNS
    │  失败 → 四级降级链（扩大 unlock → 降低 freeze → 放宽 ε → 全局重排）
    ▼
新计划
```

### 7.2 TRCG 根因诊断

**时序资源冲突图（TRCG）**：节点为活跃任务的关键工序投影区间，边表示在共享资源（$R_\text{pad}$、$R_3$、$R_\text{range}$）上的时间重叠冲突，边权为重叠时长。

**输出 TRCGSummary（8 字段）**：

| 字段 | 含义 |
|------|------|
| `bottleneck_pressure` | $R_\text{pad}/R_3/R_\text{range}$ 利用率 |
| `top_conflicts` | 冲突边列表（最多 20 条） |
| `conflict_clusters` | 加权度数聚类（中心 = max degree mission） |
| `urgent_missions` | 紧急度评分：$\text{urgency} = \text{due\_slack} + 0.5\,\text{window\_slack} - 2\,\text{delay}$ |
| `disturbance_summary` | range\_loss\_pct / pad\_outage\_active / duration\_volatility |
| `frozen_summary` | num\_frozen\_ops / freeze\_horizon |

### 7.3 LLM 决策结构

```json
{
  "root_cause_mission_id": "M003",
  "unlock_mission_ids": ["M003", "M005", "M007"],
  "freeze_horizon_hours": 8,
  "epsilon_solver": 0.05,
  "analysis_short": "M003 冲突簇中心，窗口剩余 1 个，紧急度最高",
  "secondary_root_cause_mission_id": "M005"
}
```

- **unlock 集规模**：1–5 个任务（相比全局 20 任务，仅解锁 5–25% 变量）
- **四级校验**：JSON 解析 → 必需字段 → 枚举范围 → 业务规则（root∈unlock, unlock⊆active）
- **确定性回退**：加权度数最大任务为 root，K=3（普通）或 K=5（高压/紧急）为 unlock 集

### 7.4 Anchor Fix-and-Optimize（伪 LNS）

对非解锁任务的 Op4（上塔台面）和 Op6（发射）施加锚点约束：

$$s_{\text{op}} = s_{\text{op}}^{\text{prev}}$$

锚点前执行四级可行性检查，扰动导致不可行的锚点自动跳过（记录 `anchor_fix_skipped`）。  
相较全局重排，搜索空间由 $O(n^2)$ 降至 $O(k^2)$，$k \approx 3$，求解速度提升 30–50%。

### 7.5 四级降级回退链

| 级别 | 操作 |
|------|------|
| attempt 1 | unlock 集扩充 +2 个冲突邻居 |
| attempt 2 | freeze_horizon 8→4→0 小时 |
| attempt 3 | $\varepsilon$ 放宽 0.0→0.02→0.05→0.10 |
| final | 无锚点全局重排（保证可行） |

---

## 八、对比基线

| 策略 | 类别 | 特征 |
|------|------|------|
| FixedWeightPolicy | 规则基线 | 固定 $H_\text{freeze}=8$ h、$\varepsilon=0.05$，每次全局重排 |
| GreedyPolicy（EDF/Window） | 启发式基线 | 不调用 CP-SAT，速度最快，质量最低 |
| RealLLMPolicy | LLM 元参数基线 | LLM 推理 $(H_\text{freeze}, \varepsilon)$，全局重排，无局部修复 |
| GARepairPolicy | Matheuristic 基线 | 遗传算法搜索最优 unlock 集，CP-SAT 局部修复，**无 LLM** |
| **TRCGRepairPolicy（本方法）** | **LLM 局部修复** | TRCG 根因诊断 + LLM 零样本 + Anchor LNS + 回退链 |

---

## 九、评价指标体系

### 9.1 核心指标（8 个，支撑主论文 4 图 1 表）

#### 准时性
| 指标 | 定义 |
|------|------|
| `avg_delay` | 平均延迟（slots） |
| `weighted_tardiness` | $\sum_m \text{priority}_m \cdot \max(0, s_m^{\text{Op6}} - d_m)$ |

#### 稳定性
| 指标 | 定义 |
|------|------|
| `episode_drift` | 全 Episode 累计 Drift |
| `drift_per_replan` | $\text{episode\_drift} / \text{num\_replans}$（归一化） |
| `total_switches` | 总 Pad 切换次数 |

#### 求解性能
| 指标 | 定义 |
|------|------|
| `avg_solve_time_ms` | 平均求解时间（毫秒） |
| `num_replans` | 重排总次数 |

#### 完成度
| 指标 | 定义 |
|------|------|
| `completion_rate` | 完成任务比例（须≈100%方可公平比较其他指标） |

### 9.2 次要指标（补充分析）

| 指标 | 用途 |
|------|------|
| `on_time_rate` | 按期发射率，与 avg_delay 互补 |
| `feasible_rate` | 鲁棒性证据 |
| `avg_frozen` | 证明冻结机制有效性 |
| `fallback_rate` | $\text{num\_forced\_global} / \text{num\_replans}$，回退链激活频率 |
| `unlock_size_avg` | 平均解锁集大小，衡量修复局部性 |
| `llm_time_total_ms` | LLM 推理总耗时（成本分析） |

### 9.3 删除的冗余指标

`total_delay`（可推导）、`total_solve_time_ms`（可推导）、`forced_replan_rate`（可推导）、`total_resource_switches`（与 total_switches 重复）、`num_completed/num_total`（作为注释非指标）。

---

## 十、实验设计

### 10.1 仿真环境

- **时间单位**：1 slot = 30 分钟；实验周期 10 天（480 slots）
- **任务规模**：20 个任务/Episode，每任务 7 道工序
- **Range Calendar**：每天 3 段固定窗口（W1∈[12,28)、W2∈[40,56)、W3∈[68,84) slots）
- **扰动强度分组**：

| 扰动类型 | Light | Medium | Heavy |
|---------|-------|--------|-------|
| Range Closure 概率 | 5% | 7% | 10% |
| Pad 故障概率 | 2% | 3% | 5% |
| 工序延迟标准差 | 12% | 20% | 30% |

### 10.2 LLM 配置

- **模型**：Qwen3-32B（阿里云 DashScope API）
- **推理参数**：temperature=0（确定性输出）
- **缓存机制**：相同 prompt hash 命中缓存，避免重复调用
- **超时保护 + 自动重试**：llm_client.py 实现，最多 3 次重试

### 10.3 论文图表规划

| 图表 | 类型 | 核心指标 | 目的 |
|------|------|---------|------|
| Figure 1 | 概念示意图 | — | Rolling Horizon + TRCG 流程说明 |
| Figure 2 | 双泳道甘特图 + 折线图 | `plan_drift`, `num_switches` | 单 Episode Case Study（定性机制证据） |
| Figure 3 | ECDF / Box Plot | `avg_delay`, `episode_drift` | 多策略多 seed 总体效果分布对比 |
| Figure 4 | Pareto 散点图 | `weighted_tardiness` × `drift_per_replan` | 准时性–稳定性权衡前沿 |
| Figure 5 | 分组柱状图 | `avg_delay`, `episode_drift` × 扰动强度 | 扰动分层鲁棒性分析 |
| Table 1 | 实验配置表 | `avg_solve_time_ms`, `feasible_rate` | 基线公平性验证 |
| Table 2 | 结果汇总表 | 8 个核心指标 mean±std | 主要实验结论 |

### 10.4 消融研究规划

| 消融项 | 对比维度 |
|--------|---------|
| 有/无 TRCG 诊断 | 根因定位的必要性 |
| 有/无 Anchor LNS | 局部搜索的加速效果 |
| LLM 决策 vs 启发式回退 | LLM 推理的增益 |
| unlock_set 大小 K=1/3/5 | 解锁集规模的影响 |

---

## 十一、关键术语速查

| 术语 | 含义 |
|------|------|
| RCPSP | 资源约束项目调度问题（Resource-Constrained Project Scheduling Problem） |
| Rolling Horizon | 滚动时域重排：每隔固定时间步重新求解未来窗口 |
| TRCG | 时序资源冲突图（Temporal Resource Conflict Graph），识别扰动根因 |
| Anchor Fix-and-Optimize | 锚定非扰动任务、仅对解锁任务重排的局部搜索（伪 LNS） |
| Drift | 重排导致计划变动量的加权归一化度量，衡量计划稳定性 |
| CP-SAT | Google OR-Tools 约束规划求解器，两阶段词典序优化 |
| GARepairPolicy | 用遗传算法搜索最优解锁集的非 LLM Matheuristic 基线 |
| $\varepsilon$-constraint | 第二阶段允许延迟相对最优值放宽的比例上界 |
| Freeze Horizon | 冻结视野：当前时刻起 $H_\text{freeze}$ 小时内的工序锁定不移动 |
| Op6 | 发射工序（关键路径末端，须落入 Range Calendar ∩ 轨道窗口交集）|
