# 火箭发射调度优化系统：基于LLM的自适应动态排程研究项目（V2.5）

## 项目概览 (Project Overview)

### 研究背景与动机

本项目研究**火箭发射调度问题**（Launch Scheduling Problem），这是一个复杂的**动态资源约束型调度问题**（Resource-Constrained Project Scheduling Problem, RCPSP）。在真实运行环境中，发射任务面临多重挑战：

1. **严格的时间窗口约束**：每次发射仅在特定时间窗口有效（如轨道窗口、气象窗口）
2. **共享资源竞争**：多个任务竞争有限的发射台、测试设备、序列等关键资源
3. **高频扰动**：天气突变、设备故障、任务延误等不确定性事件频繁发生
4. **多目标权衡**：需要在**准时交付**（minimizing delay）和**计划稳定性**（minimizing schedule disruption）之间寻求平衡

### 核心研究问题

**如何利用大语言模型（LLM）的推理能力，实现动态调度策略的自适应参数调整与根因诊断修复？**

- **第一代方法（RealLLMPolicy）**：使用固定权重优化求解器（CP-SAT），利用LLM根据状态特征在线推理最优元参数（freeze_horizon、epsilon_solver等）
- **第二代方法（TRCGRepairPolicy，V2.5+）**：引入**时序资源冲突图（TRCG）**根因诊断，LLM推理需要"解锁"的冲突任务，结合Anchor Fix-and-Optimize（伪LNS）实现局部修复 + 多级回退链（8次尝试）确保鲁棒性
- **对照方法（GARepairPolicy，V2.5+）**：用遗传算法搜索最优unlock子集 + CP-SAT局部修复，作为局部修复的非LLM Matheuristic Baseline
- **对照方法（ALNSRepairPolicy，V2.5+）**：用自适应大邻域搜索（ALNS）替代遗传算法，更轻量的非LLM Matheuristic Baseline，与GARepairPolicy共同构成对照组
- **创新点**：
  1. 首次将LLM作为"元策略"应用于工业级调度问题（零样本决策）
  2. 首创TRCG因果分析框架，将"全局重排"升级为"根因驱动的局部修复"
  3. Anchor Fix-and-Optimize显著降低求解空间（20任务→3解锁 = 17%变量）

### V2.5 核心特性速览

#### 🎯 研究方法演进
| 维度 | V2.1 (RealLLMPolicy) | V2.5 (TRCGRepairPolicy) | V2.5 (GARepairPolicy) | V2.5 (ALNSRepairPolicy) |
|------|----------------------|-------------------------|-----------------------|------------------------|
| **LLM角色** | 元参数调整器 | 根因诊断 + 局部修复决策器 | **无LLM**（对照组） | **无LLM**（对照组） |
| **输入** | 12维状态特征 | TRCG诊断摘要（冲突图+聚类） | TRCG候选池 | TRCG候选池 |
| **输出** | (freeze, epsilon) | (unlock_ids, root_cause) | GA搜索最优unlock子集 | ALNS搜索最优unlock子集 |
| **求解范式** | 全局重排（所有任务） | 局部修复（3-5个解锁任务） | 局部修复（K=5个解锁任务） | 局部修复（K=4个解锁任务） |
| **计算复杂度** | $O(n^2)$ (20任务) | $O(k^2)$ (3任务) | $O(\text{pop} \times k^2)$ GA搜索 | $O(\text{iter} \times k^2)$ ALNS搜索 |
| **鲁棒性** | 单次求解（成功/失败） | 多级回退链（保证可行） | 三级回退链（保证可行） | 三级回退链（保证可行） |

#### 🌐 Range Calendar系统（工业真实性增强）
- **全局共享窗口**：模拟Range设施的有限开放时间（每天3段，共12小时）
- **Range Closure扰动**：天气导致窗口动态收缩（取代旧的通用资源downtime）
- **可行性护栏**：双重校验确保任何时刻都有可行解（护栏A/B）
- **Op3b联测工序**：新增Range测试资产（R_range_test），增加资源竞争复杂度

#### 📊 指标体系升级（论文就绪）
| 新增指标 | 公式 | 论文价值 |
|---------|------|---------|
| `drift_per_replan` | $\frac{\text{episode_drift}}{\text{num_replans}}$ | 归一化比较不同重排频率策略 |
| `drift_per_day` | $\frac{\text{episode_drift}}{\text{sim_days}}$ | 适配多天实验横向对比 |
| `unlock_size_avg` | $\bar{|\text{unlock_ids}|}$ | 衡量修复局部性 |
| `fallback_rate` | $\frac{\text{num_forced_global}}{\text{num_replans}}$ | 回退链鲁棒性指标 |

#### 🧪 实验框架完善
- **run_batch_10day.py**：长周期测试（10天×3难度×3 baseline×N个seeds），输出结构化 CSV
- **run_compare_policies_once.py**：单次多策略对比（Fixed/NoFreeze/MockLLM/LLM），快速冒烟验证
- **merge_results.py**：合并 LLM 与 baseline 的 `results_per_episode.csv`（自动对齐表头差异列）
- **analyze_llm_deviation_bins.py**：分析 LLM 偏离基准参数时各状态特征区间与 episode 指标差异，生成分桶汇总 CSV
- **Episode Case Study**：双泳道Gantt图可视化（Baseline vs Ours）
- **8份文档**：从功能说明到测试指南，覆盖完整开发周期

#### 🔬 Phase 4 实验运行与代码迭代（2026-02-14 至今）
- **Qwen3-32B 实际LLM实验**：`results_V2.5/{BL, LLM}` 目录含多轮种子匹配实验（共 40+ 批次）
- **RepairStepLog 3-way可观测性**：`llm_http_ok` / `llm_parse_ok` / `llm_decision_ok`（移除旧 `llm_call_ok` 单布尔字段）
- **_auto_correct_llm_output**：自动纠正LLM选出的非活跃/已完成 mission_id，从TRCG候选池补齐
- **_trcg_find_urgent回归修复**：移除错误的 started_ops 过滤，保留已启动但即将到期的任务
- **unlock_mission_ids 激活**：确保 Anchor Fix-and-Optimize 实际传递给求解器生效
- **difficulty 档位配置**：新增 `MISSIONS_BY_DIFFICULTY`（light=15/medium=20/heavy=25）、`DIFFICULTY_DISTURBANCE`、`SLACK_MULTIPLIER_BY_DIFFICULTY` 三套映射
- **NoFreezePolicy / MinimalFreezePolicy**：新增无冻结基线策略，用于对比冻结机制效果

---

### 项目架构（V2.5）

```
┌─────────────────────────────────────────────────────────────────┐
│                      实验框架 (Experiment Framework)              │
│  run_experiments.py: 批量实验 + 策略调参 + 统计分析               │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼──────────┐   ┌────────▼──────────┐
│  场景生成          │   │  仿真器 (V2.5)     │
│  (scenario.py)    │──▶│  (simulator.py)   │
│                   │   │                   │
│ - 任务序列生成     │   │ - Rolling Horizon │
│ - 扰动事件生成     │   │ - Range Calendar  │
│ - 资源约束定义     │   │ - TRCG回退链      │
└───────────────────┘   └────────┬──────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
        ┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
        │  策略引擎      │ │  求解器     │ │  指标系统   │
        │  (policies/)  │ │ (solver_)   │ │ (metrics.py)│
        │               │ │ cpsat.py)   │ │            │
        │ - Fixed       │ │            │ │ - Delay    │
        │ - NoFreeze    │ │ CP-SAT     │ │ - Drift    │
        │ - Greedy      │ │ 两阶段求解  │ │ - Switch   │
        │ - RealLLM     │ │ +Anchor LNS │ │ - Features │
        │ - TRCGRepair  │ │            │ │            │
        │ - GARepair    │ │            │ │            │
        │ - ALNSRepair  │ │            │ │            │
        └───────┬───────┘ └────┬───────┘ └────────────┘
                │              │
        ┌───────▼────────┐ ┌───▼──────────┐
        │  TRCG诊断       │ │ LLM 客户端   │
        │  (features.py) │ │(llm_client.py)│
        │                │ │              │
        │ - 根因分析      │ │ - OpenAI API │
        │ - 冲突聚类      │ │ - 缓存机制   │
        │ - 紧急度评分    │ │ - 重试逻辑   │
        └────────────────┘ └──────────────┘
```

---

## 1. 核心模块详解

### 1.1 场景生成器 (scenario.py)

#### 数据模型（V2.5 Schema：Range Calendar + Range Closure）

**Mission（任务）**：
- `mission_id`: 任务唯一标识（如 "M001"）
- `release`: 任务释放时间（最早开始 slot）
- `due`: 软截止时间（发射 deadline）
- `priority`: 优先级权重（0.1-1.0）
- `operations`: 包含 7 个工序（Op1-Op6 + Op3b）

**Operation（工序）**：
- 每个任务由 **7 个串行工序** 组成：
    1. **Op1**: 任务准备（资源 R1）
    2. **Op2**: 总装集成（资源 R2）
    3. **Op3**: 系统检测（资源 R3）
    4. **Op3b**: **联测工序**（资源 R3 + R_range_test，duration=2 slots）
    5. **Op4**: 上塔台面准备（资源 R_pad + R4，**重要锚点**）
    6. **Op5**: 台面占用（资源 R_pad，duration=0，用于约束Op5→Op6最大间隔）
    7. **Op6**: **加注/最后检查/倒计时/窗口执行**（资源 R_pad + R3，**关键锚点**）
- 关键特性：
  - **Op6 有时间窗口**：每个任务2-5个发射窗口（轨道窗口）
  - **Op5→Op6 最大间隔**：24小时（燃料稳定性要求）
  - **前序约束**：Op_i 必须在 Op_{i-1} 完成后开始

**Resource（资源）**：
- `R_pad`: **发射台资源**（容量1，关键瓶颈）
- `R1, R2, R3, R4`: 测试设备
- `R_range_test`: **Range 联测设备**（容量1，Op3b 使用）
- `unavailable`: 资源不可用时间段（维护窗口）

**Range Calendar（日历窗口）**：
- `range_calendar: Dict[day, List[Tuple[start, end]]]`
- 默认每天 3 段固定窗口：W1=[12,28), W2=[40,56), W3=[68,84)
- 硬校验：窗口长度 ≥ (Op6_duration + 4)，不足时扩展或兜底全天

#### 扰动生成

**三种扰动强度**（用于实验分组，对应 `config.py` 的 `DIFFICULTY_DISTURBANCE`）：

| 扰动类型 | Light | Medium | Heavy |
|---------|-------|--------|-------|
| 天气中断概率（`p_weather`） | 4% | 6% | 8% |
| Pad故障概率（`p_pad_outage`） | 1% | 1.5% | 2% |
| 工序延迟标准差（`sigma_duration`） | 12% | 18% | 26% |
| 释放时间扰动（`release_jitter_slots`） | 1 slot | 2 slots | 2 slots |

**扰动事件类型**：
1. **weather**: 天气中断（6-18 slots）
2. **range_closure**: **Range Closure（窗口收缩）**，对 range_calendar 进行区间减法
3. **pad_outage**: Pad故障（3-12 slots）
4. **duration**: 工序实际耗时偏差（±σ）
5. **release**: 任务释放时间延迟（默认禁用）

**Range Closure 可行性护栏**：
- 护栏 A：不允许当天 range_calendar 变为空
- 护栏 B：不允许任何任务的 Op6 候选窗口被清空
- 触发护栏时跳过该次 closure 事件（保持可行性）

---

### 1.2 求解器 (solver_cpsat.py)

#### 核心算法：两阶段 Lexicographic 优化

**问题建模**：
- **决策变量**：
  - `start[op]`: 工序开始时间
  - `end[op]`: 工序结束时间
  - `interval[op]`: 工序执行区间（用于资源冲突检测）

- **硬约束**：
  1. 前序约束：`start[Op_i] ≥ end[Op_{i-1}]`
    2. 时间窗口：`start[Op6] ∈ windows`（求解前会与 `range_calendar` 取交集得到候选窗口）
  3. 资源容量：`Σ interval[op] ≤ capacity[r]`（NoOverlap）
  4. Op5→Op6 最大间隔：`start[Op6] - end[Op5] ≤ 24h`
  5. 冻结约束：已开始或冻结区内的工序不可移动

**Stage 1 - 准时性优先**：
```
minimize: Σ priority[m] × max(0, start[Op6_m] - due[m])
```
获得最优延迟值 `D_opt`

**Stage 2 - 稳定性优先**（加入 ε-constraint）：
```
subject to: Σ priority[m] × delay[m] ≤ (1 + ε_solver) × D_opt
minimize: Σ priority[m] × Drift[m]
```

**Drift 定义（V3版本）**：
```python
Drift[m] = (
    0.7 × |start_new[Op6] - start_old[Op6]|  # Launch时间偏移
  + 0.3 × |start_new[Op4] - start_old[Op4]|  # Pad占用时间偏移
  + κ_win × I[window切换]                    # 窗口切换惩罚（κ_win=12）
  + κ_seq × I[Pad排队顺序变化]               # 序列切换惩罚（κ_seq=6）
)
```

**可避免性过滤**（Avoidable Switch）：
- 仅统计"策略造成的变化"，不惩罚"扰动不可避免的变化"
- 若旧窗口在当前扰动下不可行 → Window switch = 0
- 仅统计未开始 & 未冻结 & 未完成的任务

#### 冻结机制 (Freeze Horizon)

**动机**：避免短期计划频繁变化，提升操作稳定性

**实现**：
```python
frozen_ops = {op | op已开始 OR start[op] ≤ now + freeze_horizon}
```
冻结的工序保持时间 & 资源分配不变

**调参维度**：`freeze_horizon ∈ {0, 4, 8, 16, 24} hours`

---

### 1.3 策略系统 (policies/)

#### 策略接口 (base.py)

```python
class BasePolicy(ABC):
    @abstractmethod
    def decide(state, now, config) -> (MetaParams, Plan):
        """
        返回：
        - MetaParams: 元参数（传给CP-SAT求解器）
        - Plan: 直接计划（贪心策略使用）
        """

@dataclass
class MetaParams:
    w_delay: float            # 延迟权重（Stage 1弃用，保留兼容性）
    w_shift: float            # 偏移权重（Stage 2弃用）
    w_switch: float           # 切换权重（Stage 2弃用）
    freeze_horizon: int       # 冻结视野（hours → slots）
    use_two_stage: bool       # 是否启用两阶段
    epsilon_solver: float     # Stage 2 延迟容差
    kappa_win: float          # 窗口切换等效 slot 数
    kappa_seq: float          # 序列切换等效 slot 数
    
    # ========== TRCG Repair 扩展字段（V2.5+）==========
    unlock_mission_ids: Optional[Tuple[str, ...]] = None   # 解锁集（传给 solver）
    root_cause_mission_id: Optional[str] = None            # 根因 mission
    secondary_root_cause_mission_id: Optional[str] = None  # 次根因
    decision_source: str = "default"                       # llm|heuristic_fallback|forced_global
    fallback_reason: Optional[str] = None                  # 回退原因
    attempt_idx: int = 0                                   # 回退链尝试序号
```

#### 策略实现

**1. FixedWeightPolicy（固定参数策略，Baseline B1）**
```python
# 使用预设的固定参数
params = MetaParams(
    freeze_horizon=12,      # 固定12 slots冻结（2小时）  
    epsilon_solver=0.05,    # 固定5%延迟容差
    use_two_stage=True,
    kappa_win=12.0,
    kappa_seq=6.0
)
```

**2. NoFreezePolicy / MinimalFreezePolicy（无冻结策略，Baseline B2）**
```python
params = MetaParams(
    freeze_horizon=0,       # 无冻结（或 freeze_horizon=4 for Minimal）
    epsilon_solver=0.05,
    use_two_stage=True,
)
```
- 用于对照冻结机制的效果，体现无冻结时的高 drift / 低 delay 特性

**3. GreedyPolicy（启发式策略）**
- **EDFGreedy**: Earliest Due First（最早截止优先）
- **WindowGreedy**: 优先分配窗口最少的任务
- **特点**：不使用CP-SAT，直接构造可行解（速度快，质量低）

**4. RealLLMPolicy（LLM元策略，第一代方法）**

**工作流程**：
```
1. 提取状态特征 → features.py
   ├─ window_loss_pct: 窗口损失比例
   ├─ pad_pressure: Pad资源压力
   ├─ delay_increase_minutes: 预估延误增量
   ├─ trend_window_loss: 窗口损失趋势
   └─ num_urgent_tasks: 紧急任务数

2. 构造 Prompt → policy_llm_meta.py
   ├─ 系统提示：角色定义（调度专家）
   ├─ 上下文：当前状态特征
   ├─ 任务：推理最优参数组合
   └─ 输出格式：JSON schema（freeze_horizon, epsilon_solver）

3. LLM 推理 → llm_client.py
   ├─ 调用 Qwen3-32B API
   ├─ 磁盘缓存（SHA256 key）
   ├─ 指数退避重试（5次）
   └─ JSON 三层抽取（code fence → thinking → raw）

4. 参数验证 & 返回
   └─ 返回 MetaParams 给求解器
```

**Prompt 模板**（简化版）：
```
You are an expert scheduler for rocket launch operations.

Current State:
- Window Loss: 25.3% (increasing trend +0.5%/step)
- Pad Pressure: 0.85 (high utilization)
- Delay Increase: 45 minutes (if no replan)
- Urgent Tasks: 3

Task:
Decide the optimal scheduling parameters:
- freeze_horizon: [0, 4, 8, 16, 24] hours
- epsilon_solver: [0.0, 0.02, 0.05, 0.10]

Reasoning:
1. High window loss → prefer smaller freeze (more responsive)
2. High pad pressure → need tighter delay constraint (smaller epsilon)
3. Urgent tasks → balance between stability and timeliness

Output JSON:
{
  "freeze_horizon_hours": 4,
  "epsilon_solver": 0.02,
  "reasoning": "..."
}
```

**4. TRCGRepairPolicy（TRCG修复策略，第二代方法，V2.5+）**

**核心改进**：从"元参数调整"升级为"根因诊断 + 局部修复 + 回退链"。

**工作流程**：
```
1. TRCG根因诊断 → features.build_trcg_summary()
   ├─ 构建时序资源冲突图（Temporal Resource Conflict Graph）
   ├─ 瓶颈压力分析（Pad/窗口/序列资源）
   ├─ 冲突检测与聚类（同一Pad/窗口的冲突组）
   ├─ 紧急度评分（剩余窗口时长×任务优先级）
   └─ 输出TRCGSummary：根因任务、次根因、拥堵分析

2. LLM修复决策 → llm_client.call_llm_for_repair()
   ├─ 输入：TRCGSummary + 约束条件
   ├─ 任务：推理需要"解锁"哪些任务（打破锚定）
   ├─ 输出：RepairDecision（unlock_mission_ids）
   └─ 回退：LLM失败 → 启发式决策（解锁根因+次根因）

3. Anchor Fix-and-Optimize → solver_cpsat.py
   ├─ 固定非解锁任务的Op4/Op6到prev_plan（伪LNS）
   ├─ 仅重排解锁任务的发射时刻
   ├─ Stage1: min Σdelay   Stage2: min Σdrift (s.t. Stage1最优值±ε)
   └─ 显著降低求解空间（20任务→3解锁 = 17%变量）

4. 多级回退链 → policy_llm_repair.solve_with_fallback_chain()
   Level 0 (initial): 初始解锁集（LLM/启发式）
   ├─ 失败 ↓
   Level 1a (expand_unlock): 扩大解锁集（+2任务，max 8）
   ├─ 失败 ↓
   Level 1b (expand_unlock_wide): 再次扩大（+4任务，max 12）
   ├─ 失败 ↓
   Level 2a (reduce_freeze): 减小冻结视野（一档降级）
   ├─ 失败 ↓
   Level 2b (reduce_freeze_deep): 再次减小冻结视野
   ├─ 失败 ↓
   Level 3a (relax_epsilon): 放松延迟容差（epsilon阶梯提升）
   ├─ 失败 ↓
   Level 3b (wide_unlock_relaxed): 大范围解锁（~45%任务）+ 放松参数
   ├─ 失败 ↓
   Level 3c (partial_global_anchor): 部分全局解锁（~70%任务）
   ├─ 失败 ↓
   Final (global_replan): 强制全局重排（全量解锁，保证可行）
```

**Prompt示例**（TRCG修复场景）：
```
You are an expert repair agent for rocket launch scheduling.

Current Conflict (TRCG Diagnosis):
- Root Cause Mission: M007 (Op4=slot 120, due=125, urgent=HIGH)
- Secondary Root Cause: M012 (Op4=slot 118, same Pad_A)
- Bottleneck: Pad_A pressure=1.2 (oversubscribed)
- Conflict Cluster: {M007, M012, M018} all need Pad_A in [118-125]

Previous Plan (Anchored):
- M007: Op4=120 → violates due date
- M012: Op4=118 → blocks M007
- M018: Op4=122 → chains with M007

Task:
Decide which missions to "unlock" (allow re-optimization):
- Unlocking = allow Op4/Op6 to move freely (breaking anchor)
- Goal: resolve conflict while minimizing plan change

Output JSON:
{
  "unlock_mission_ids": ["M007", "M012"],
  "reasoning": "M007 must move earlier to meet due date. M012 blocks it on Pad_A, so unlock both. M018 can stay anchored."
}
```

**关键设计**：
- **局部性**：只重排3-5个冲突任务，其余锚定 → 计划稳定性高
- **因果推理**：TRCG暴露"谁阻塞谁"，LLM推理"谁需要让路"
- **鲁棒性**：多级回退链（8次尝试 + 最终全局重排）确保最终总有可行解

**5. GARepairPolicy（GA修复策略，Matheuristic Baseline，V2.5+）**

**核心思想**：使用遗传算法（Genetic Algorithm）搜索最优解锁任务子集，结合CP-SAT Anchor Fix-and-Optimize实现局部修复。**定位为非LLM的局部修复baseline**，用于对比TRCGRepairPolicy（LLM驱动）的性能。

**工作流程**：
```
1. TRCG根因诊断 → features.build_trcg_summary()
   ├─ 复用TRCG诊断框架（与TRCGRepairPolicy共享）
   ├─ 生成候选解锁池（高紧急度任务）
   └─ 准备遗传算法搜索空间

2. 遗传算法搜索 → _ga_search_unlock_set_v2()
   ├─ 初始化种群：随机生成N个解锁子集（每个大小K=5）
   ├─ 适应度评估：并行调用CP-SAT求解并计算Fitness
   │   Fitness = -avg_delay (Stage1) 或 -episode_drift (Stage2)
   ├─ 选择：Roulette Wheel Selection（轮盘赌）
   ├─ 交叉：单点交叉（保持子集大小K）
   ├─ 变异：随机替换1-2个任务（概率mutation_rate=0.2）
   └─ 早停：连续patience代无改进则终止

3. Anchor Fix-and-Optimize → solver_cpsat.py
   ├─ 与TRCGRepairPolicy相同机制
   ├─ 固定非解锁任务的Op4/Op6
   ├─ 仅重排GA选出的K个任务
   └─ Stage1: min Σdelay   Stage2: min Σdrift

4. 回退机制 → 三级回退链（简化版）
   Level 0: GA搜索的最优解锁集
   ├─ 失败 ↓
   Level 1: 启发式解锁集（根因+次根因）
   ├─ 失败 ↓
   Level 2: 全局重排
```

**关键参数（V2加速版）**：
```python
# 基础GA参数
pop_size = 12              # 种群大小
generations = 5            # 最大进化代数
K = 5                      # 解锁子集大小
mutation_rate = 0.2        # 变异概率
candidate_pool_size = 15   # 候选池大小（从TRCG提取）

# V2加速特性
n_jobs = 8                 # 并行worker数量（适应度评估）
eval_budget = 48           # 硬约束：最大评估次数
early_stop_patience = 2    # 早停：连续N代无改进
eval_timeout_s = 0.5       # 评估阶段单次CP-SAT超时
final_timeout_s = 2.0      # 最终求解超时（默认=config.solver_timeout_s）
enable_cache = True        # 适应度缓存（避免重复评估）
```

**算法伪代码**：
```python
def ga_search_unlock_set(candidate_pool, prev_plan, state, K=5):
    # 初始化种群
    population = [random_sample(candidate_pool, K) for _ in range(pop_size)]
    
    best_fitness = -∞
    patience_counter = 0
    eval_count = 0
    
    for gen in range(generations):
        # 并行评估适应度（硬预算约束）
        fitness_scores = parallel_map(
            lambda unlock_set: evaluate_with_cpsat(unlock_set, prev_plan, state),
            population,
            n_jobs=n_jobs
        )
        eval_count += len(population)
        
        # 更新最优解
        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_unlock_set = population[argmax(fitness_scores)]
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= early_stop_patience or eval_count >= eval_budget:
            break
        
        # 选择、交叉、变异
        parents = roulette_wheel_selection(population, fitness_scores)
        offspring = single_point_crossover(parents)
        offspring = mutate(offspring, mutation_rate, candidate_pool)
        population = offspring
    
    return best_unlock_set
```

**V2加速优化**：
| 优化项 | V1（基础版） | V2（加速版） | 提升 |
|--------|--------------|--------------|------|
| 并行评估 | ❌串行（1 job） | ✅并行（8 jobs） | **8×加速** |
| 进化预算 | ❌无限（最多pop×gen=80） | ✅硬约束（12次最大） | **-85%评估** |
| 早停机制 | ❌固定gen=5 | ✅patience=2（动态） | **-40%平均代数** |
| 适应度缓存 | ❌重复求解 | ✅SHA256缓存 | **-20%重复计算** |
| CP-SAT超时 | ❌固定20s | ✅两段式（0.5s/2.0s） | **-75%评估耗时** |

**关键设计**：
- **确定性baseline**：相比TRCGRepairPolicy（依赖LLM推理），GA策略完全确定性，便于对比实验
- **搜索效率**：通过并行评估+早停+预算约束，将GA搜索时间控制在可接受范围（通常<10s）
- **局部性**：与TRCGRepairPolicy相同，仅重排K=5个任务，保持计划稳定性
- **鲁棒性**：回退链确保总有可行解（最差情况=全局重排）

**实验价值**：
- **对照组**：验证LLM推理 vs 随机搜索在局部修复场景的效果差异
- **性能基准**：GA作为成熟的Matheuristic方法，提供公平的性能对比标准
- **消融研究**：可通过调整GA参数（pop_size、generations）分析搜索预算与修复质量的权衡

**6. NoFreezePolicy / MinimalFreezePolicy（无冻结/极短冻结基线，V2.5+）**

**核心思想**：完全去除冻结机制（freeze_horizon=0），作为对照基线用于量化冻结机制的稳定性收益。

```python
class NoFreezePolicy(BasePolicy):
    """
    行为：freeze_horizon=0，每次 rolling 可完全重新安排所有任务
    用途：与 FixedWeightPolicy 对比，评估冻结机制的价值
    """
    def decide(state, now, config):
        return MetaParams(
            freeze_horizon=0,          # 无冻结
            epsilon_solver=0.05,
            use_two_stage=True,
            kappa_win=12.0,
            kappa_seq=6.0
        )
```

`MinimalFreezePolicy` 与之类似，但设置 `freeze_horizon=4`（1小时），体现最小保护区间。

**实验价值**：
- 对照无冻结时的高 drift / 低 delay 特性，证明冻结机制的稳定性价值
- 作为 Baseline B2，与 FixedWeightPolicy（B1）形成完整的冻结粒度对比谱

**7. MockLLMPolicy（模拟LLM策略，用于调试）**
- 使用硬编码规则模拟 LLM 决策逻辑（if-else）
- 用于快速验证框架正确性，无需真实 API Key

**8. ALNSRepairPolicy（ALNS修复策略，轻量Matheuristic Baseline，V2.5+）**

**核心思想**：复用与 GARepairPolicy 相同的 TRCG 候选池、CP-SAT 局部修复和回退链，仅将"搜索算法"替换为**自适应大邻域搜索（ALNS）**小循环。相比 GA，ALNS 每轮迭代更轻量（无种群），便于在较少 CP-SAT 调用预算内快速探索 unlock 子集。

**工作流程**：
```
1. TRCG根因诊断 → features.build_trcg_summary()（复用）
   └─ 生成候选解锁池（高紧急度任务，默认大小15）

2. ALNS搜索 → _alns_search_unlock_set()
   ├─ 初始解：随机抽取K=4个候选任务
   ├─ 每轮：随机扰动（替换1-2个任务）+ CP-SAT评估
   ├─ 接受准则：更优解必接受 + 以概率10%接受劣解（模拟退火式）
   └─ 最大迭代次数10次（eval_timeout_s=0.5s，final_timeout_s可配置）

3. Anchor Fix-and-Optimize → solver_cpsat.py（复用）
   ├─ 与GARepairPolicy相同机制
   └─ 固定非解锁任务的Op4/Op6，仅重排ALNS选出的K个任务

4. 回退机制 → 三级回退链（简化版，与GARepairPolicy相同）
```

**关键参数**：
```python
K = 4                        # 解锁子集大小（略小于GA的5）
candidate_pool_size = 15     # TRCG候选池大小
max_iterations = 10          # ALNS最大迭代次数
accept_worse_prob = 0.10     # 接受劣解概率（模拟退火）
eval_timeout_s = 0.5         # 评估阶段单次CP-SAT超时
drift_lambda = 5.0           # 联合目标中drift的权重系数
```

**与GARepairPolicy对比**：
- **优势**：无种群初始化开销，每轮仅1次 CP-SAT 调用，总预算通常更少
- **劣势**：无并行评估，搜索多样性略低于遗传算法
- **定位**：更轻量的 Matheuristic Baseline，与 GARepairPolicy 共同构成非 LLM 对照组

---

### 1.4 仿真器 (simulator.py)

#### Rolling Horizon 框架

**参数设置**：
- `slot_minutes = 15`: 时间离散化粒度（15分钟/slot）
- `rolling_interval = 12`: 重规划间隔（12 slots = 3小时）
- `horizon_slots = 96`: 规划视野（96 slots = 24小时）
- `sim_total_slots = 960`: 仿真总时长（960 slots = 10天，V2.5标准实验时长）

**Op6 候选窗口过滤**：
- 每次求解前动态计算：`candidate_windows = mission_windows ∩ range_calendar`
- 过滤掉长度 < `op6_duration` 的窗口
- 若过滤导致某任务窗口为空，会触发 Range Closure 护栏避免不可行

**仿真循环**：
```python
for t in range(0, sim_total_slots, rolling_interval):
    # 1. 应用扰动事件
    apply_disturbances(state, t, t + rolling_interval)
    
    # 2. 更新已完成工序
    update_started_and_completed(state, t)
    
    # 3. 策略决策
    meta_params, direct_plan = policy.decide(state, t, config)
    
    # 4. 求解器生成新计划（如果策略返回 meta_params）
    if meta_params:
        plan = solve_v2_1(
            state.missions, 
            state.resources,
            now=t,
            horizon_end=t + horizon_slots,
            prev_plan=state.current_plan,
            freeze_horizon=meta_params.freeze_horizon,
            epsilon_solver=meta_params.epsilon_solver,
            # ...
        )
    
    # 5. 计算指标
    metrics = compute_rolling_metrics(state, plan, prev_plan)
    
    # 6. 更新状态
    state.current_plan = plan
    
    # 7. 执行计划（时间前进到 t + rolling_interval）
```

#### 状态管理

**SimulationStateOps**：
```python
@dataclass
class SimulationStateOps:
    now: int                      # 当前时刻
    missions: List[Mission]       # 任务列表（动态更新）
    resources: List[Resource]     # 资源列表（动态更新）
    current_plan: Plan            # 当前执行中的计划
    
    started_ops: Set[str]         # 已开始工序集合
    completed_ops: Set[str]       # 已完成工序集合
    applied_events: Set[int]      # 已应用扰动事件索引
    actual_durations: Dict        # 实际耗时（扰动后）
    actual_releases: Dict         # 实际释放时间（扰动后）
```

---

### 1.5 指标系统 (metrics.py)

#### Rolling Metrics（单步指标）

**Delay 指标**：
- 基于 **Op6 start**（发射时刻）计算：
  ```python
  delay[m] = max(0, start[Op6] - due[m])
  ```

**Drift 指标（V3 定义，V2.5扩展）**：
```python
# 1. 时间偏移（两锚点加权）
time_shift = 0.7 * |start_new[Op6] - start_old[Op6]|
           + 0.3 * |start_new[Op4] - start_old[Op4]|

# 2. 窗口切换（可避免性过滤）
window_switch = 1 if (window_new != window_old AND 旧窗口仍可行) else 0

# 3. 序列切换（Pad排队顺序）
pred_old = Pad上Op4直接前驱任务（按Op4 start排序）
pred_new = 当前Pad上Op4直接前驱任务
sequence_switch = 1 if pred_new != pred_old else 0

# 4. 加权 Drift
drift[m] = priority[m] * (
    time_shift 
    + κ_win * window_switch 
    + κ_seq * sequence_switch
)
```

**其他指标**：
- `num_frozen`: 冻结工序数量
- `solve_time_ms`: 求解耗时（毫秒）
- `is_feasible`: 是否可行

#### Episode Metrics（全局指标，V2.5扩展）

**性能指标（Timeliness）**：
- `avg_delay`: 平均延迟（slots）
- `on_time_rate`: 按期交付率（delay=0的任务占比）
- `weighted_tardiness`: 加权延误（考虑优先级）
- `max_delay`: 最大延迟（鲁棒性分析用）

**稳定性指标（Stability，V2.5新增归一化指标）**：
- `episode_drift`: 全局 Drift（所有步骤累加）
- **`drift_per_replan`**: 平均每次重排的drift（**V2.5核心指标**）
  - 计算：`episode_drift / num_replans`
  - 意义：归一化比较不同重排频率策略的稳定性
- **`drift_per_day`**: 平均每天的drift（**V2.5扩展指标**）
  - 计算：`episode_drift / (sim_total_slots / 96)`
  - 意义：适配多天实验的横向对比
- `total_shifts`: 总时间变化次数
- `total_switches`: 总 Pad 切换次数
- `total_window_switches`: 时间窗切换次数
- `total_sequence_switches`: Pad 序列切换次数

**效率指标**：
- `avg_solve_time_ms`: 平均求解时间
- `total_solve_time_ms`: 总求解时间
- `num_replans`: 重排次数
- `num_forced_replans`: 强制重排次数（不可行触发）
- `feasible_rate`: 可行率（成功求解占比）
- `resource_utilization`: 总资源利用率
- `util_r_pad`: Pad资源利用率（关键瓶颈）

**完成度指标**：
- `completion_rate`: 完成率（必须=100%才能比较其他指标）
- `makespan_cmax`: 完成时间Cmax（如涉及makespan优化）

---

### 1.6 特征提取 (features.py)

#### 核心特征

**1. window_loss_pct（窗口损失比例）**
```python
# 定义：未来 H 内可用窗口 slot 减少比例
prev_slots = 上一次统计的所有可用窗口slot集合
curr_slots = 本次统计的所有可用窗口slot集合
loss_slots = prev_slots - curr_slots
window_loss_pct = len(loss_slots) / len(prev_slots)
```

**2. pad_pressure（Pad资源压力）**
```python
# 定义：Pad需求 / Pad容量
demand = Σ [duration[Op4] + duration[Op6]] for schedulable missions
capacity = len(pads) * horizon_slots
pad_pressure = min(1.0, demand / capacity)
```

**3. delay_increase_minutes（延误增量预估）**
```python
# 不重排情况下，因窗口失效导致的延误增加
for mission in urgent_missions:
    old_launch = prev_plan[mission].start[Op6]
    if old_launch 被本次扰动破坏:
        next_available = 下一个可行窗口
        delay_increase += (next_available - old_launch)
```

**4. 趋势特征（Trend Features）**
```python
# 使用滑动窗口（4步）计算变化趋势
trend_window_loss = (current - past_4_steps) / 4
trend_pad_pressure = (current - past_4_steps) / 4
# 用于判断态势恶化 vs 改善
```

**5. 波动性特征（Volatility）**
```python
# 标准差衡量状态波动程度
volatility_pad_pressure = std_dev(past_4_steps)
# 高波动 → 需要更保守策略
```

---

### 1.7 LLM 客户端 (llm_client.py)

#### 功能特性

**1. OpenAI 兼容 API**
```python
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)
response = client.chat.completions.create(
    model="qwen3-32b",
    messages=[...],
    temperature=0.0,
    max_tokens=256
)
```

**2. 磁盘缓存（并发安全）**
```python
# 缓存键生成
cache_key = sha256(
    model + json.dumps(messages, sort_keys=True)
).hexdigest()

# 原子写入（避免竞态条件）
temp_file = f"{cache_dir}/{cache_key}.tmp.{random_suffix}"
json.dump(result, temp_file)
os.replace(temp_file, f"{cache_dir}/{cache_key}.json")
```

**3. 指数退避重试**
```python
for attempt in range(max_retries):
    try:
        return api_call()
    except RateLimitError:
        delay = min(
            retry_base_delay * (2 ** attempt) * (1 + random() * jitter),
            retry_max_delay
        )
        time.sleep(delay)
```

**4. JSON 三层抽取**
```python
# 层级 1: 尝试提取 ```json code fence
if '```json' in response:
    return extract_code_fence(response)

# 层级 2: 尝试提取 "thinking" 外的 JSON
if '"thinking"' in response:
    return extract_without_thinking(response)

# 层级 3: 直接解析原始文本
return json.loads(response)
```

**5. Schema 校验**
```python
def validate_schema(data: dict, schema: dict) -> bool:
    for key, expected_type in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False
    return True
```

---

## 2. 实验框架 (run_experiments.py)

### 2.1 实验流程

```
┌─────────────────────────────────────────────────────────┐
│ 阶段1: 数据集生成                                        │
│ - Train Set: 60 scenarios (20 light + 20 medium + 20 heavy) │
│ - Test Set:  60 scenarios (20 light + 20 medium + 20 heavy) │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ 阶段2: Baseline 调参（仅在 Train Set）                   │
│ 网格搜索：                                               │
│ - freeze_horizon: [0, 4, 8, 16, 24] hours (5种)        │
│ - epsilon_solver: [0.0, 0.02, 0.05, 0.10] (4种)        │
│ 共 5×4 = 20 组合                                         │
│                                                          │
│ 选择准则：ε-constraint                                   │
│ 1. 筛选满足 avg_delay ≤ baseline * (1 + ε) 的参数组     │
│ 2. 在满足条件的组合中，选择 episode_drift 最小的        │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ 阶段3: 策略对比（在 Test Set）                           │
│ 对比策略：                                               │
│ - FixedWeightPolicy (固定参数，Baseline B1)              │
│ - NoFreezePolicy (无冻结对照，Baseline B2)               │
│ - GreedyPolicy (EDFGreedy / WindowGreedy)               │
│ - RealLLMPolicy (zero-shot，第一代LLM方法)              │
│ - GARepairPolicy (Matheuristic Baseline，无LLM)          │
│ - TRCGRepairPolicy (第二代LLM方法，本方法)               │
│                                                          │
│ 配对比较：每个 seed 在相同扰动下运行所有策略              │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│ 阶段4: 结果分析                                          │
│ 输出文件：                                               │
│ - best_params.json: 最优参数                             │
│ - tuning_results.csv: 调参详细结果                       │
│ - episode_results.csv: 每个episode的汇总指标             │
│ - rolling_metrics.csv: 每步的detailed metrics            │
│ - llm_logs/*.jsonl: LLM调用日志                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 调参策略

**ε-constraint 方法**：
```python
# 定义基准：固定参数策略的性能
baseline_delay = avg_delay(FixedWeightPolicy(freeze=0, epsilon=0))

# 筛选可接受的参数组合
acceptable_configs = [
    config for config in all_configs
    if avg_delay(config) <= baseline_delay * (1 + epsilon_tune)
]

# 选择最优参数（稳定性最优）
best_config = min(acceptable_configs, key=lambda c: episode_drift(c))
```

**关键参数**：
- `epsilon_tune = 0.10`: 延迟容差（相对baseline最多增加10%）
- `tuning_lambda = 5.0`: 综合目标权重（legacy，已弃用）

### 2.3 并行化

**多线程执行**：
```python
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(simulate_episode, scenario, policy, config): (seed, policy)
        for seed in train_seeds
        for policy in candidate_policies
    }
    
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
```

**注意事项**：
- LLM 策略**强制单线程**（避免 API 速率限制）
- Baseline 策略可并行（纯计算，无外部调用）

---

## 3. 关键技术细节

### 3.1 两阶段求解的数学模型

**Stage 1（Lexicographic 第一优先级）**：

决策变量：
- $s_{m,i}$: 任务 $m$ 的工序 $i$ 的开始时间
- $e_{m,i}$: 任务 $m$ 的工序 $i$ 的结束时间
- $w_{m,6}$: 任务 $m$ 选择的发射窗口索引

目标函数：
$$
\text{minimize} \quad \sum_{m \in M} p_m \cdot \max(0, s_{m,6} - d_m)
$$

约束条件：
1. 前序约束：$s_{m,i} \geq e_{m,i-1}, \forall m, i \geq 2$
2. 工序时长：$e_{m,i} = s_{m,i} + \text{dur}_{m,i}$
3. 窗口约束：$s_{m,6} \in \text{windows}_{m,w_{m,6}}$
4. 资源容量：$\text{NoOverlap}(\{\text{interval}_{m,i} : \text{res}_{m,i} = r\})$
5. Op5→Op6 间隔：$s_{m,6} - e_{m,5} \leq 96$ (24h)
6. 冻结约束：$s_{m,i} = \bar{s}_{m,i}, \forall (m,i) \in F$

**Stage 2（加入稳定性）**：

获得 Stage 1 最优值 $D^* = \sum_m p_m \cdot \text{delay}_m$

新增约束：
$$
\sum_{m \in M} p_m \cdot \max(0, s_{m,6} - d_m) \leq (1 + \varepsilon) \cdot D^*
$$

新目标函数：
$$
\text{minimize} \quad \sum_{m \in M} p_m \cdot \text{Drift}_m
$$

其中：
$$
\begin{align}
\text{Drift}_m = &\ 0.7 \cdot |s^t_{m,6} - s^{t-1}_{m,6}| \\
                 &+ 0.3 \cdot |s^t_{m,4} - s^{t-1}_{m,4}| \\
                 &+ \kappa_{\text{win}} \cdot \mathbb{1}[w^t_m \neq w^{t-1}_m \land \text{旧窗口可行}] \\
                 &+ \kappa_{\text{seq}} \cdot \mathbb{1}[\text{pred}^t(m) \neq \text{pred}^{t-1}(m)]
\end{align}
$$

序列切换定义：
$$
\text{pred}^t(m) = \arg\max_{m' : s^t_{m',4} < s^t_{m,4} \land \text{same pad}} s^t_{m',4}
$$

### 3.2 可避免性判断算法

```python
def is_window_switch_avoidable(
    mission: Mission,
    old_plan: Plan,
    new_plan: Plan,
    state: SimulationState
) -> bool:
    """判断窗口切换是否可避免"""
    
    # 1. 获取旧窗口
    old_window_idx = old_plan.get_window(mission.mission_id)
    old_window = mission.operations[5].time_windows[old_window_idx]
    
    # 2. 检查旧窗口在当前扰动下是否仍可行
    # （考虑资源可用性、前序依赖、时间约束）
    if not is_window_still_feasible(old_window, state):
        return False  # 不可避免的切换（扰动导致）
    
    # 3. 检查任务是否已开始或冻结
    if mission.mission_id in state.started_ops:
        return False  # 已执行的不计入
    
    op4_start = new_plan.get_assignment(f"{mission.mission_id}_Op4").start_slot
    if op4_start <= state.now + freeze_horizon:
        return False  # 冻结区内的不计入
    
    # 4. 新旧窗口不同 → 可避免的切换
    new_window_idx = new_plan.get_window(mission.mission_id)
    return new_window_idx != old_window_idx
```

### 3.3 LLM Prompt Engineering

**System Prompt（角色定义）**：
```
You are an AI expert in dynamic scheduling for rocket launch operations.
Your task is to analyze the current system state and recommend optimal 
scheduling parameters (freeze horizon and epsilon solver) that balance 
timeliness and stability.

Key Principles:
1. High urgency → Prefer smaller freeze (more responsive)
2. High resource pressure → Prefer smaller epsilon (tighter delay control)
3. Stable trend → Can use larger freeze (reduce replanning)
4. High volatility → Prefer smaller freeze (stay adaptive)
```

**User Prompt（状态上下文）**：
```
Current State (T={now}):
----------------------------------------
URGENCY INDICATORS:
- Window Loss: {window_loss_pct:.1%} (trend: {trend_window_loss:+.2%}/step)
- Urgent Tasks (due within 12h): {num_urgent_tasks}
- Projected Delay Increase: {delay_increase_minutes:.0f} minutes

RESOURCE PRESSURE:
- Pad Pressure: {pad_pressure:.2f} (demand/capacity ratio)
- Resource Conflict Level: {resource_conflict_pressure:.2f}
- Min Slack: {slack_min_minutes:.0f} minutes

SYSTEM STABILITY:
- Pad Pressure Volatility: {volatility_pad_pressure:.3f}
- Trend (Window Loss): {trend_window_loss:+.2%}
- Trend (Pad Pressure): {trend_pad_pressure:+.3f}

DECISION CONTEXT:
- Last 4 steps average metrics: (from history)
- Current replan triggered by: [automatic interval / forced by disturbance]

TASK:
Recommend optimal parameters:
1. freeze_horizon_hours: Choose from [0, 4, 8, 16, 24]
2. epsilon_solver: Choose from [0.0, 0.02, 0.05, 0.10]

OUTPUT FORMAT (JSON):
{
  "freeze_horizon_hours": <int>,
  "epsilon_solver": <float>,
  "reasoning": "<brief explanation of your choice>"
}
```

**Few-Shot Examples（可选）**：
```
Example 1 (High Urgency):
Input: {window_loss: 35%, urgent_tasks: 5, trend: +1.2%}
Output: {freeze: 0, epsilon: 0.02, reasoning: "High urgency requires immediate response"}

Example 2 (Stable State):
Input: {window_loss: 5%, urgent_tasks: 1, trend: -0.1%}
Output: {freeze: 16, epsilon: 0.10, reasoning: "Stable state allows longer freeze for continuity"}
```

---

### 3.4 V2.5 Phase 4 代码迭代改进

基于实际LLM实验运行（Qwen3-32B），以下代码改进在2026-02-14后完成：

#### RepairStepLog 可观测性升级

**问题**：原`llm_call_ok`（单布尔）无法区分HTTP失败 vs JSON解析失败 vs 业务逻辑校验失败。

**改进**：3-way LLM可观测性字段：
```python
@dataclass
class RepairStepLog:
    llm_http_ok: bool = False        # HTTP请求是否成功（网络/API层）
    llm_parse_ok: bool = False       # JSON解析是否成功
    llm_decision_ok: bool = False    # 业务规则校验是否通过
    llm_error: dict = field(default_factory=dict)  # 结构化错误信息
    # llm_call_ok: bool  # 已移除 ← 原单字段
```
日志分析中可用以下方式定位失败类型：
```
decision_source=heuristic_fallback + llm_http_ok=True + llm_parse_ok=True + llm_decision_ok=False
→ LLM HTTP调通但输出不合法（如解锁了already-started的mission）
```

#### LLM输出自动纠正（_auto_correct_llm_output）

**问题**：LLM偶尔选出已完成或已启动的 mission_id（不在活跃集合中）。

**解决方案**：
```python
def _auto_correct_llm_output(decision, active_mission_ids, started_ops):
    """将LLM输出中不合法的mission_id替换为TRCG诊断推荐的合法候选"""
    valid_ids = set(active_mission_ids) - get_started_missions(started_ops)
    corrected = [m for m in decision.unlock_mission_ids if m in valid_ids]
    if len(corrected) < len(decision.unlock_mission_ids):
        # 补充启发式候选填满unlock集
        corrected += pick_from_trcg_urgent(valid_ids - set(corrected))
    return corrected
```

#### _trcg_find_urgent 回归修复

**问题**：误加 `started_ops` 过滤导致高紧急度但已开始准备的任务被排除在urgent列表外，造成unlock集为空。

**修复**：`_trcg_find_urgent()` 不过滤 started missions，保留所有"即将到期"的任务（包括已部分启动的）。

#### unlock_mission_ids 激活修复

**问题**：`MetaParams.unlock_mission_ids` 默认为 `None`，导致 solver 按全局重排处理（Anchor Fix-and-Optimize 未生效）。

**修复**：`TRCGRepairPolicy.decide()` 始终返回非 `None` 的 `unlock_mission_ids`，确保 Anchor 约束实际传递给求解器。

#### Difficulty 档位体系完善

**新增三套难度映射**（`config.py`）：

```python
MISSIONS_BY_DIFFICULTY = {"light": 15, "medium": 20, "heavy": 25}

DIFFICULTY_DISTURBANCE = {
    "light":  {"p_weather": 0.04, "sigma_duration": 0.12, ...},
    "medium": {"p_weather": 0.06, "sigma_duration": 0.18, ...},
    "heavy":  {"p_weather": 0.08, "sigma_duration": 0.26, ...},
}

SLACK_MULTIPLIER_BY_DIFFICULTY = {"light": 1.5, "medium": 1.2, "heavy": 1.0}
```

`make_config_for_difficulty(difficulty, num_missions_override=None, scenario_profile="default", **kwargs)` 工厂函数统一创建符合难度档位的 Config 实例，`run_batch_10day.py` 直接调用。

**场景配置文件（SCENARIO_PROFILES）**：`config.py` 还定义了 `SCENARIO_PROFILES` 字典，支持通过 `scenario_profile` 参数切换场景配置模板：
- `"default"`：标准配置（默认）
- `"local_repair"`：为评测局部修复策略特化的配置，采用 Wave 式释放模式（任务集中在几个波次到来，加剧 Pad 冲突），更大的 Op6 窗口（3–5个），更短的 Range Closure 持续时间，以及难度相关的 slack_multiplier 覆盖值——这些设计使 TRCG 根因更加显著，适合放大 TRCGRepairPolicy / GARepairPolicy / ALNSRepairPolicy 之间的性能差异。

---

## 4. 研究假设与验证方法

### 4.1 核心假设

**H1（主假设）**：LLM 策略在动态调度中能够实现与调优后的固定策略相当或更优的性能

**H2**：LLM 策略在应对高强度扰动时表现出更强的鲁棒性（相对性能下降更小）

**H3**：LLM 策略能够通过上下文学习（in-context learning）实现 zero-shot 决策，无需历史数据训练

### 4.2 评估指标

**主指标（Primary Metrics）**：
1. **Avg Delay**（平均延迟）：$\frac{1}{|M|}\sum_{m \in M} \max(0, \text{actual\_launch}_m - \text{due}_m)$
2. **Episode Drift**（全局稳定性）：$\sum_{t, m} p_m \cdot \text{Drift}^t_m$

**次级指标（Secondary Metrics）**：
3. **On-Time Rate**：$\frac{|\{m : \text{delay}_m = 0\}|}{|M|}$
4. **Window Switch Rate**：$\frac{\text{total\_window\_switches}}{|M| \times \text{num\_replans}}$
5. **Feasible Rate**：$\frac{\text{num\_feasible\_replans}}{\text{total\_replans}}$
6. **Avg Solve Time**：求解器平均耗时（评估计算效率）

**相对性能（Relative Performance）**：
$$
\text{Rel}_{\text{metric}} = \frac{\text{Metric}_{\text{LLM}} - \text{Metric}_{\text{Baseline}}}{\text{Metric}_{\text{Baseline}}} \times 100\%
$$

### 4.3 统计检验

**配对 t 检验**（Paired t-test）：
```python
from scipy.stats import ttest_rel

# 在相同 seeds 上配对比较
delays_baseline = [result['avg_delay'] for result in baseline_results]
delays_llm = [result['avg_delay'] for result in llm_results]

t_stat, p_value = ttest_rel(delays_baseline, delays_llm)
```

**显著性水平**：$\alpha = 0.05$

**效应量**（Effect Size）：
$$
d = \frac{\bar{x}_{\text{LLM}} - \bar{x}_{\text{Baseline}}}{s_{\text{pooled}}}
$$

---

## 5. 完整约束体系（Constraint Taxonomy）

本节系统汇总项目在 CP-SAT 模型（`solver_cpsat.py`）及仿真层（`simulator.py`）中涉及的**所有约束**，按类别分层展示实际代码实现与数学形式。

---

### 5.1 硬约束（Hard Constraints）

硬约束在 CP-SAT 模型中以 `model.Add(...)` 或内置传播形式施加，**不可违反**，违反则为 INFEASIBLE。

#### C1 释放时间约束（Release Time Constraint）

**适用范围**：全部工序（Op1–Op6 + Op3b）

```python
# solver 变量定义时直接编码下界
start_vars[op.op_id] = model.NewIntVar(op.release, horizon, f"start_{op.op_id}")
```

$$
s_{m,i} \geq r_{m,i}, \quad \forall m \in M,\ i \in \{1,\ldots,7\}
$$

其中 $r_{m,i}$ 为工序 $(m,i)$ 的最早可开始时间，继承自任务释放时间与前序工序累计完工时间的最大值。

---

#### C2 工序时长约束（Duration Constraint）

**适用范围**：全部工序（除 Op5 外均为固定时长）

```python
end_vars[op.op_id] = model.NewIntVar(...)
model.Add(end_vars[op.op_id] == start_vars[op.op_id] + op.duration)
```

$$
e_{m,i} = s_{m,i} + \text{dur}_{m,i}, \quad \forall m,i \text{ (非 Op5)}
$$

---

#### C3 前序约束（Precedence Constraint）

**适用范围**：工序链 Op1→Op2→Op3→Op3b→Op4→Op5→Op6（通过 `Operation.precedences` 字段载入）

```python
for op in all_ops:
    for pred_id in op.precedences:
        if pred_id in end_vars:
            model.Add(start_vars[op.op_id] >= end_vars[pred_id])
```

$$
s_{m,i} \geq e_{m,i-1}, \quad \forall m \in M,\ i \geq 2
$$

> **注意**：Op3b 以 `precedences` 字段自动嵌入链，无需手工区分索引。前序链顺序为：Op1→Op2→Op3→Op3b→Op4→Op5→Op6。

---

#### C4 Pad 块三元组连续性约束（Contiguous Pad Block Constraint）⚠️ **新**

**适用范围**：每任务的 (Op4, Op5, Op6) 三元组

这是一项**文档中此前未明确说明的隐式关键约束**。Op4（pad_hold）、Op5（wait）、Op6（launch）三个工序在 Pad 资源上必须**紧接连续执行，无任何间隙**：

```python
# Pad block contiguity: pad_hold -> wait -> launch must be contiguous
for mission in missions:
    pad_hold, wait, launch = mission.get_pad_block()
    model.Add(start_vars[wait.op_id] == end_vars[pad_hold.op_id])   # Op5紧跟Op4
    model.Add(start_vars[launch.op_id] == end_vars[wait.op_id])      # Op6紧跟Op5
```

$$
s_{m,5} = e_{m,4}, \quad s_{m,6} = e_{m,5}, \quad \forall m \in M
$$

**工程含义**：一旦火箭上塔（Op4 开始），Pad 就被连续占用直至发射（Op6 结束），中间不允许释放再占用。这是"一旦上塔即锁定"的现实运营规则。

---

#### C5 Op5 可变时长约束（Variable-Duration Wait Constraint）⚠️ **新**

**适用范围**：Op5（台面占用 / 等待工序）

Op5 是全系统**唯一具有可变工序时长的工序**。其时长是决策变量，范围由最小标称时长与最大等待窗口共同约束：

```python
wait_op_ids = {mission.get_wait_op().op_id for mission in missions}
if op.op_id in wait_op_ids:
    min_duration = max(0, op.duration)
    max_duration = min_duration + op5_max_wait_slots        # 默认 op5_max_wait_slots=96(24h)
    duration_var = model.NewIntVar(min_duration, max_duration, f"dur_{op.op_id}")
    model.Add(end_vars[op.op_id] == start_vars[op.op_id] + duration_var)
```

$$
\text{dur}_{m,5}^{\min} \leq \text{dur}_{m,5} \leq \text{dur}_{m,5}^{\min} + H_{\text{wait}}, \quad H_{\text{wait}} = 96\ \text{slots}\ (24\text{h})
$$

**关系澄清**：文档此前描述的"Op5→Op6 最大间隔 24h"等价于 $\text{dur}_{m,5} \leq \text{dur}_{m,5}^{\min} + 96$，而非 $s_{m,6} - e_{m,5} \leq 96$（因为 C4 保证了 $s_{m,6} = e_{m,5}$，两者等价）。**可变时长**是 Op5 "弹性等待" 的建模机制，既保证连续性又允许选择不同发射窗口。

---

#### C6 Op6 发射窗口约束（Launch Window Constraint）

**适用范围**：Op6（发射工序）

每个任务必须**恰好选择一个**发射窗口，Op6 的时间范围必须完整落在所选窗口内：

```python
window_choice = []
for win_idx, (ws, we) in enumerate(launch.time_windows):
    in_window = model.NewBoolVar(...)
    window_choice.append(in_window)
    model.Add(start_vars[launch.op_id] >= ws).OnlyEnforceIf(in_window)
    model.Add(end_vars[launch.op_id] <= we).OnlyEnforceIf(in_window)
model.AddExactlyOne(window_choice)   # ← 恰好一个窗口
```

$$
\exists!\ k \in W_m:\ s_{m,6} \geq \text{ws}_{m,k}\ \land\ e_{m,6} \leq \text{we}_{m,k}
$$

**预处理**：传入 solver 之前，`time_windows` 已经过 Range Calendar 交集过滤（见 C10）。

---

#### C7 资源容量约束（Resource Capacity / No-Overlap Constraint）

**适用范围**：全部资源 R_pad、R1、R2、R3、R4、R_range_test（每种容量均为 1）

```python
for res_id, intervals in resource_intervals.items():
    if intervals:
        model.AddNoOverlap(intervals)   # 容量=1 → NoOverlap
```

$$
\forall r \in \mathcal{R},\ \text{interval}_{m,i} \cap \text{interval}_{m',i'} = \emptyset,\ \text{若}\ r \in \text{res}_{m,i} \cap \text{res}_{m',i'}
$$

**Op3b 多资源同步约束**：Op3b 同时占用 R3 和 R_range_test，其区间变量被分别注册到两个资源的 NoOverlap 集合，从而自动施加双重无冲突约束：

```python
# Op3b.resources = ["R3", "R_range_test"]
for res_id in op.resources:  # 遍历 ["R3", "R_range_test"]
    resource_intervals[res_id].append(interval_vars[op.op_id])
```

---

#### C8 资源不可用区间约束（Resource Unavailability / Closure Blocker）

**适用范围**：所有资源中标记为 `unavailable` 的区间（Pad Outage、Range Closure 注入的封闭窗口）

通过在 NoOverlap 中插入定长固定区间（"Blocker"）实现：

```python
for resource in resources:
    for closure_idx, (cs, ce) in enumerate(resource.unavailable):
        duration = ce - cs + 1
        blocker = model.NewFixedSizedIntervalVar(cs, duration, f"closure_...")
        resource_intervals[resource.resource_id].append(blocker)
```

$$
\text{NoOverlap}\big(\{\text{op\_intervals}\} \cup \{\text{unavailability\_blockers}\}\big)
$$

**Range Closure 注入路径**：
1. 仿真器检测到 `range_closure` 扰动事件
2. 对 `range_calendar[day]` 执行区间减法（线段差运算）
3. 护栏 A/B 确保不清空当天窗口或任意任务的 Op6 候选窗口
4. 缩短后的 Range 开放区间"外侧"部分转化为 `resource.unavailable` 注入 C8

---

#### C9 冻结约束（Freeze Horizon Constraint）

**适用范围**：已开始工序（`started_ops`）及冻结视野内的工序

```python
for op_id, frozen in frozen_ops.items():
    if op_id in start_vars:
        model.Add(start_vars[op_id] == frozen.start_slot)
        model.Add(end_vars[op_id] == frozen.end_slot)
```

$$
s_{m,i} = \bar{s}_{m,i},\ e_{m,i} = \bar{e}_{m,i}, \quad \forall (m,i) \in F
$$

其中冻结集合 $F$ 的确定规则（`compute_frozen_ops`）：

$$
F = \{(m,i) \mid \text{op}_{m,i} \in \text{started\_ops}\} \cup \{(m,i) \mid s_{m,i} > t_{\text{now}} \land s_{m,i} \leq t_{\text{now}} + H_{\text{freeze}}\}
$$

已启动工序**无条件冻结**（不受 $H_{\text{freeze}}$ 限制）。

---

#### C10 Anchor Fix-and-Optimize 约束（V2.5 新增）⚠️ **新**

**适用范围**：TRCGRepairPolicy / GARepairPolicy 局部修复模式中，**非解锁任务** 的 Op4（pad_hold）和 Op6（launch）

这是 V2.5 核心创新之一，将大邻域搜索（LNS）中的 "fix" 操作以 CP-SAT 硬约束实现：

```python
# Stage 2 中对非 unlock missions 施加硬位置锁定
for op_id, anchor_start in anchor_fixes.items():
    if op_id in start_vars and op_id not in frozen_ops:
        model.Add(start_vars[op_id] == anchor_start)   # 硬位置锚点
```

锚点候选集在施加前通过 `_check_anchor_feasibility()` 进行 **5 步可行性预检**：

| 步骤 | 检验内容 | 跳过原因字段 |
|------|---------|------------|
| 1 | prev_plan 中存在 Op4/Op6 分配 | `skip_missing_prev_assignment` |
| 2 | Op4/Op6 的 prev 位置在当前 horizon 内 | `skip_out_of_horizon` |
| 3 | Op6 prev 位置仍落在 time_windows 内 | `skip_window_mismatch` |
| 4 | Op4/Op6 prev 位置不与资源不可用区间重叠 | `skip_resource_unavailable_pad/launch` |
| 5 | Op4→Op5→Op6 隐含等待时长满足 Op5 可变范围 | `skip_wait_violation` |

通过预检的锚点进一步做**无冲突筛选**（与已冻结区间或其他锚点不重叠）后才最终施加：

$$
s_{m,4} = \bar{s}_{m,4}^{\text{prev}},\ s_{m,6} = \bar{s}_{m,6}^{\text{prev}}, \quad \forall m \notin \mathcal{U}\ \land\ \text{预检通过}
$$

其中 $\mathcal{U}$ 为 LLM/GA 推断出的解锁集（unlock_mission_ids）。

**Stage 1 不施加锚点**（关键设计决策）：Stage 1 纯延迟最优化不受锚点约束，确保 `delay_bound` 是真实最优值，不因锚点偏紧导致 Stage 2 过度受限。

---

### 5.2 软约束与目标函数（Soft Constraints / Objectives）

#### O1 Stage 1 目标：最小化加权延误

$$
\min \sum_{m \in M} p_m \cdot \max\!\left(0,\ s_{m,6} - d_m\right)
$$

| 参数 | 说明 | 默认值 |
|------|------|--------|
| $p_m$ | 任务优先级权重 | 0.1–1.0 |
| $d_m$ | 软截止时间（due slot） | 场景生成 |

#### O2 Stage 2 延迟上界约束（ε-Constraint，兼硬约束）

Stage 2 引入 ε-constraint 将 Stage 1 最优延误 $D^*$ 作为上界，放宽比例为 $\varepsilon$：

```python
delay_bound = total_delay_stage1 * (1 + config.epsilon_solver)
model.Add(sum(total_delay_terms) <= int(round(delay_bound * 100)))
```

$$
\sum_{m \in M} p_m \cdot \max\!\left(0,\ s_{m,6} - d_m\right) \leq (1 + \varepsilon) \cdot D^*
$$

调参范围：$\varepsilon \in \{0.0,\ 0.02,\ 0.05,\ 0.10\}$

#### O3 Stage 2 目标：最小化加权 Drift

$$
\min \sum_{m \in M} p_m \cdot \text{Drift}_m
$$

**Drift 四项分解**（V3 定义，代码实现于 `_solve_v2_1_stage2_with_delay_bound`）：

| 项目 | CP-SAT 建模 | 权重 | 语义 |
|------|------------|------|------|
| Op6 时间偏移 | `AddAbsEquality(shift_abs, diff_var)` | $0.7 \times p_m$ | 发射时刻漂移 |
| Op4 时间偏移 | `AddAbsEquality(shift_abs, diff_var)` | $0.3 \times p_m$ | Pad 占用时刻漂移 |
| 窗口切换 | `BoolVar window_switch` = $1 - \text{choice}[\text{prev\_idx}]$ | $\kappa_{\text{win}} \times p_m$ | 发射窗口变化 |
| 序列切换 | `BoolVar seq_switch` 通过前驱保持约束编码 | $\kappa_{\text{seq}} \times p_m$ | Pad 排队顺序倒换 |

$$
\text{Drift}_m = 0.7 \cdot |s^t_{m,6} - s^{t-1}_{m,6}|
             + 0.3 \cdot |s^t_{m,4} - s^{t-1}_{m,4}|
             + \kappa_{\text{win}} \cdot \mathbb{1}\!\left[w^t_m \neq w^{t-1}_m \land \text{旧窗口可行}\right]
             + \kappa_{\text{seq}} \cdot \mathbb{1}\!\left[\text{pred}^t(m) \neq \text{pred}^{t-1}(m)\right]
$$

其中参数默认值：$\kappa_{\text{win}} = 12$（slots/switch），$\kappa_{\text{seq}} = 6$（slots/switch）。

**Warm Start 优化**：Stage 2 对 `start_vars` / `end_vars` / `window_choice_vars` 添加来自 `prev_plan` 的 `AddHint()`，帮助求解器在时限内更快收敛到高质量解。

---

### 5.3 预处理约束（Pre-processing Constraints，仿真层）

预处理约束在 **仿真器层（simulator.py）** 而非 CP-SAT 模型内执行，但对求解器所接收的有效输入空间构成硬性限制。

#### P1 Op6 候选窗口 Range Calendar 交集过滤

每次重排前动态计算 Op6 的有效窗口集合：

```python
candidate_windows[m] = mission_windows[m] ∩ range_calendar[current_state]
candidate_windows[m] = [w for w in candidate_windows[m] if w.length >= op6_duration]
```

$$
W_m^{\text{candidate}} = \left\{w \cap w_r \mid w \in W_m,\ w_r \in \mathcal{RC}_t,\ |w \cap w_r| \geq \text{dur}_{m,6}\right\}
$$

若过滤后 $W_m^{\text{candidate}} = \emptyset$，求解器直接报 INFEASIBLE。

#### P2 Range Closure 可行性护栏（Feasibility Guard）

扰动生成时限制 range_closure 事件的幅度，防止产生不可行场景：

- **护栏 A**：单日 `range_calendar[day]` 不允许被清空（至少保留 1 个窗口段）
- **护栏 B**：任意任务 $m$ 的 Op6 候选窗口不允许被完全清空
- 触发护栏时跳过本次 closure 事件，维持上一状态

---

### 5.4 约束关系一览

```
                    ┌────────────────────────────────────────────┐
                    │             仿真层预处理（P1, P2）           │
                    │  range_calendar ∩ mission_windows → W_eff  │
                    └──────────────────┬─────────────────────────┘
                                       │ 传入有效候选窗口
                    ┌──────────────────▼─────────────────────────┐
                    │              CP-SAT 模型                     │
                    │                                              │
                    │  硬约束：                                    │
                    │   C1 释放时间   start[op] ≥ release[op]    │
                    │   C2 工序时长   end = start + dur           │
                    │   C3 前序       s_i ≥ e_{i-1}              │
                    │   C4 Pad连续性  s[Op5]=e[Op4], s[Op6]=e[Op5]│
                    │   C5 Op5可变长  dur[Op5]∈[min,min+max_wait] │
                    │   C6 窗口选择   ExactlyOne(window_choice)  │
                    │   C7 资源NoOverlap  (含Op3b多资源)          │
                    │   C8 资源不可用 Blocker闭区间               │
                    │   C9 冻结       started OR in freeze_horizon│
                    │   C10 锚点LNS   非unlock task硬位置固定      │
                    │                                              │
                    │  Stage 1 目标：min Σ p_m·delay_m           │
                    │  Stage 2 ε约束：Σdelay ≤ (1+ε)·D*          │
                    │  Stage 2 目标：min Σ p_m·Drift_m            │
                    └──────────────────────────────────────────────┘
```

---

### 5.5 约束参数速查表

| 约束 | 参数名 | 默认值 | 可调范围 | 说明 |
|------|--------|--------|---------|------|
| C5 Op5最大等待 | `op5_max_wait_slots` | 96 slots (24h) | — | 对应 `config.op5_max_wait_hours=24` |
| C9 冻结视野 | `freeze_horizon` | 12 slots (2h) | {0,4,8,16,24}h | 实验可调参数 |
| O2 延迟容差 | `epsilon_solver` | 0.05 | {0.0,0.02,0.05,0.10} | 实验可调参数 |
| O3 窗口切换惩罚 | `kappa_win` | 12.0 slots | — | Drift 公式权重 |
| O3 序列切换惩罚 | `kappa_seq` | 6.0 slots | — | Drift 公式权重 |
| O3 Op6偏移权重 | `drift_alpha` | 0.7 | — | Drift = α·Op6shift + β·Op4shift |
| O3 Op4偏移权重 | `drift_beta` | 0.3 | — | 同上 |
| C10 Anchor锁定 | `unlock_mission_ids` | None→全局重排 | LLM/GA/heuristic | V2.5 TRCGRepair/GARepair |
| — | `solver_timeout_s` | 30.0 s | — | CP-SAT总时限 |
| — | `stage1_time_ratio` | 0.4 | — | Stage1占40%时限 |
| — | `num_workers` | 4 | — | CP-SAT并行搜索线程数 |

---

## 6. 实验工具与脚本说明

### 6.1 批量实验脚本

#### run_batch_10day.py（主实验脚本）
```bash
python run_batch_10day.py [--seeds 0 1 2] [--policies static,fixed_tuned,full_unlock,ga_repair,alns_repair] [--output results/batch_10day]
```
- 运行 `light/medium/heavy × seeds × policies` 完整矩阵
- 内置 5 个策略：`static`（不重排）、`fixed_tuned`（固定参数+二阶段）、`full_unlock`（全解锁）、`ga_repair`（GA局部修复）、`alns_repair`（ALNS局部修复）
- 输出 CSV 含 20 项指标：`on_time_rate`, `avg_delay`, `episode_drift`, `drift_per_replan`, `drift_per_day`, `drift_per_active_mission` 等
- 每个 episode 同步写入 JSON 日志到 `output_dir/difficulty_policy_seed/`

#### run_compare_policies_once.py（快速对比脚本）
```bash
python run_compare_policies_once.py [--seed 42] [--difficulty medium]
```
- 单次运行 Fixed / NoFreeze / MockLLM / LLMInterface 四种策略的对比
- 输出控制台表格 + 可选 JSON 日志
- 适合冒烟测试和快速机制验证

#### run_one_episode.py（单 Episode 调试脚本）
- 单次仿真 + 详细日志输出，便于 debug 单个 case

#### run_experiments.py（参数调优脚本）
- 在 Train Set（60 scenarios）上做网格搜索调参
- 5×4 = 20 组 `(freeze_horizon, epsilon_solver)` 组合
- 结果写入 `best_params.json`, `tuning_results.csv`, `episode_results.csv`

### 6.2 结果分析工具

#### merge_results.py（合并 CSV）
```bash
python merge_results.py --llm results_V2.5/LLM/3.05_2 --baseline results_V2.5/BL/305_2 --output merged/
```
- 合并 LLM 与 baseline 的 `results_per_episode.csv`
- 自动对齐表头差异列（LLM-only 列或 baseline-only 列用空字符串填充）
- 支持任意两个结果目录合并

#### analyze_llm_deviation_bins.py（LLM偏离分析）
```bash
python analyze_llm_deviation_bins.py --llm_dir results_V2.5/LLM --results_csv merged/results_per_episode.csv
```
- 分析哪些状态特征区间更容易触发 LLM 偏离提示词基准参数
- 计算偏离发生时的 episode 指标差异 (`delta_avg_delay`, `delta_episode_drift`)
- 输出分桶汇总 CSV，支持论文 Figure 分析

#### analyze.py（通用分析脚本）
- 读取 `rolling_metrics.csv` / `episode_results.csv`
- 计算统计摘要、生成对比表

### 6.3 结果目录结构

```
results_V2.5/
├── BL/          # Baseline 实验结果（固定参数策略，多批次，40+轮次）
│   ├── 304_1/   # 格式：月日_批次
│   ├── 305_2/
│   └── ...
├── LLM/         # LLM 实验结果（TRCGRepairPolicy，多批次 Qwen3-32B）
│   ├── 3.04_1/  # 格式：月.日_批次
│   ├── 3.05_2/
│   └── ...
├── baseline/    # 早期 baseline 存档
└── env/         # 环境配置存档
```

每个实验目录内包含：
- `results_per_episode.csv`：每个 episode 的汇总指标（20列）
- `rolling_metrics.csv`：逐 slot 的详细指标
- `episode_*/`：单 episode 的 JSON 日志（扰动序列、计划快照、指标）

---

## 7. 论文研究框架摘要

### 7.1 研究问题与意义

**研究问题**：如何在充满高频扰动（天气窗口收缩、设备故障、工序延误）的真实发射场环境下，对多任务共享稀缺资源的火箭发射调度计划实施高质量的在线动态重排？

**工程意义**：发射场资源高度稀缺，一次调度决策失误可导致轨道窗口丢失或多任务链式延误，自动化在线重排可大幅降低人工介入成本并提升发射成功率。

**学术意义**：首次将 LLM 作为"零样本元决策器"引入工业级动态调度场景，并提出以 TRCG 驱动的根因诊断-局部修复范式，为 LLM 辅助运筹优化提供了方法论参考。

### 7.2 论文摘要

针对发射场多任务共享稀缺资源场景下高频扰动（天气窗口收缩、设备故障、工序延误）带来的动态调度重排需求，提出一种基于大语言模型（LLM）根因诊断与局部修复的火箭发射动态排程方法。首先，构建带靶场日历窗口约束与多类扰动的滚动重排仿真环境，形成以准时率、加权延迟、计划漂移（Drift）为核心的评价指标体系；然后，引入时序资源冲突图对扰动后的计划进行根因定位，提取冲突簇、瓶颈压力、紧急度等结构化特征，并以此为输入驱动 LLM 进行零样本推理，输出需要"解锁"的最小任务集；最后，基于 CP-SAT 求解器的 Anchor Fix-and-Optimize 机制对解锁任务实施局部精确修复，辅以多级回退链保证方法在任意扰动下的可行性，并通过跨多难度等级、多随机种子的大规模对比实验表明所提方法在准时性与计划稳定性上的综合优越性。

### 7.3 对比基线体系

| 策略 | 类别 | 特征 |
|------|------|------|
| FixedWeightPolicy | 规则基线 B1 | 固定 $H_\text{freeze}=12$ slots、$\varepsilon=0.05$，每次全局重排 |
| NoFreezePolicy | 规则基线 B2 | $H_\text{freeze}=0$，无冻结保护，每次全局重排 |
| GreedyPolicy（EDF/Window） | 启发式基线 | 不调用 CP-SAT，速度最快，质量最低 |
| RealLLMPolicy | LLM 元参数基线 | LLM 推理 $(H_\text{freeze}, \varepsilon)$，全局重排，无局部修复 |
| GARepairPolicy | Matheuristic 基线 | 遗传算法搜索最优 unlock 集，CP-SAT 局部修复，**无 LLM** |
| ALNSRepairPolicy | Matheuristic 基线 | ALNS 搜索最优 unlock 集，CP-SAT 局部修复，**无 LLM**（更轻量） |
| **TRCGRepairPolicy（本方法）** | **LLM 局部修复** | TRCG 根因诊断 + LLM 零样本 + Anchor LNS + 多级回退链 |

### 7.4 核心指标体系（论文就绪）

#### 准时性指标（Timeliness）

| 指标 | 定义 | 论文用途 |
|------|------|---------|
| `avg_delay` | 平均延迟（slots） | Figure 3, Table 2 核心 |
| `weighted_tardiness` | $\sum_m p_m \cdot \max(0, s_{m,6} - d_m)$ | Figure 4 Pareto 轴 |
| `on_time_rate` | delay=0 任务比例 | Figure 3 补充 |
| `completion_rate` | 完成率（须≈100%方可比较） | 可行性验证 |

#### 稳定性指标（Stability）

| 指标 | 定义 | 论文用途 |
|------|------|---------|
| `episode_drift` | 全 Episode 累计 Drift | Figure 2, 3, 4 核心 |
| `drift_per_replan` | $\text{episode\_drift} / \text{num\_replans}$（归一化） | **核心**：跨策略公平比较 |
| `drift_per_day` | $\text{episode\_drift} / \text{sim\_days}$ | 多天实验横向对比 |
| `total_switches` | 总 Pad 切换次数 | Figure 2 机制分析 |

#### 求解性能指标

| 指标 | 定义 | 论文用途 |
|------|------|---------|
| `avg_solve_time_ms` | 平均求解时间（ms） | Table 1 效率对比 |
| `num_replans` | 重排总次数 | Table 1 |
| `feasible_rate` | 成功求解比例 | 鲁棒性证据 |

#### LLM/修复专属指标（TRCGRepairPolicy）

| 指标 | 定义 | 论文用途 |
|------|------|---------|
| `unlock_size_avg` | 平均解锁集大小 | 衡量修复局部性 |
| `fallback_rate` | $\text{num\_forced\_global} / \text{num\_replans}$ | 回退链激活频率 |
| `llm_time_total_ms` | LLM 推理总耗时 | 成本分析 |

### 7.5 论文图表规划

| 图表 | 类型 | 核心指标 | 目的 |
|------|------|---------|------|
| Figure 1 | 概念示意图 | — | Rolling Horizon + TRCG 流程说明 |
| Figure 2 | 双泳道甘特图 + 折线图 | `plan_drift`, `num_switches` | 单 Episode Case Study（定性机制证据） |
| Figure 3 | ECDF / Box Plot | `avg_delay`, `episode_drift` | 多策略多 seed 总体效果分布对比 |
| Figure 4 | Pareto 散点图 | `weighted_tardiness` × `drift_per_replan` | 准时性–稳定性权衡前沿 |
| Figure 5 | 分组柱状图 | `avg_delay`, `episode_drift` × 扰动强度 | 扰动分层鲁棒性分析 |
| Table 1 | 实验配置表 | `avg_solve_time_ms`, `feasible_rate` | 基线公平性验证 |
| Table 2 | 结果汇总表 | 8 个核心指标 mean±std | 主要实验结论 |

### 7.6 消融研究规划

| 消融项 | 对比维度 |
|--------|---------|
| 有/无 TRCG 诊断 | 根因定位的必要性 |
| 有/无 Anchor LNS | 局部搜索的加速效果 |
| LLM 决策 vs 启发式回退 | LLM 推理的增益 |
| unlock_set 大小 K=1/3/5 | 解锁集规模的影响 |
| TRCGRepair vs GARepair | LLM 推理 vs 随机搜索 |

---

## 8. 关键术语速查

| 术语 | 含义 |
|------|------|
| RCPSP | 资源约束项目调度问题（Resource-Constrained Project Scheduling Problem） |
| Rolling Horizon | 滚动时域重排：每隔固定时间步重新求解未来窗口（3小时间隔，24小时视野） |
| TRCG | 时序资源冲突图（Temporal Resource Conflict Graph），识别扰动根因 |
| Anchor Fix-and-Optimize | 锚定非扰动任务、仅对解锁任务重排的局部搜索（伪 LNS） |
| Drift | 重排导致计划变动量的加权归一化度量，衡量计划稳定性 |
| CP-SAT | Google OR-Tools 约束规划求解器，两阶段词典序优化 |
| GARepairPolicy | 用遗传算法搜索最优解锁集的非 LLM Matheuristic 基线 |
| ALNSRepairPolicy | 用自适应大邻域搜索（ALNS）搜索最优解锁集的非 LLM Matheuristic 基线（更轻量） |
| $\varepsilon$-constraint | 第二阶段允许延迟相对最优值放宽的比例上界 |
| Freeze Horizon | 冻结视野：当前时刻起 $H_\text{freeze}$ 小时内的工序锁定不移动 |
| Op6 | 发射工序（关键路径末端，须落入 Range Calendar ∩ 轨道窗口交集） |
| Range Calendar | 全局共享的发射场开放时间段（每天 3 段，W1/W2/W3） |
| Range Closure | 天气导致的 Range 窗口动态收缩扰动，对 range_calendar 执行区间减法 |
| Op3b | 联测工序（R3 + R_range_test），增加资源竞争复杂度 |
| Pad Block | Op4（上塔）→Op5（等待）→Op6（发射）三元组，Pad 资源连续占用 |
| unlock_mission_ids | LLM/GA/ALNS/启发式决定的解锁任务集，仅此集合任务可被重排 |
| decision_source | 决策来源标记：`llm` / `heuristic_fallback` / `forced_global` |
| fallback_chain | 多级降级策略（最多8次尝试）：扩大 unlock → 减小 freeze → 放宽 ε → 强制全局重排 |

