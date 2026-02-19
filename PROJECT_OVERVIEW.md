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
- **第二代方法（TRCGRepairPolicy，V2.5+）**：引入**时序资源冲突图（TRCG）**根因诊断，LLM推理需要"解锁"的冲突任务，结合Anchor Fix-and-Optimize（伪LNS）实现局部修复 + 四级回退链确保鲁棒性
- **对照方法（GARepairPolicy，V2.5+）**：用遗传算法搜索最优unlock子集 + CP-SAT局部修复，作为局部修复的非LLM Matheuristic Baseline
- **创新点**：
  1. 首次将LLM作为"元策略"应用于工业级调度问题（零样本决策）
  2. 首创TRCG因果分析框架，将"全局重排"升级为"根因驱动的局部修复"
  3. Anchor Fix-and-Optimize显著降低求解空间（20任务→3解锁 = 17%变量）

### V2.5 核心特性速览

#### 🎯 研究方法演进
| 维度 | V2.1 (RealLLMPolicy) | V2.5 (TRCGRepairPolicy) | V2.5 (GARepairPolicy) |
|------|----------------------|-------------------------|-----------------------|
| **LLM角色** | 元参数调整器 | 根因诊断 + 局部修复决策器 | **无LLM**（对照组） |
| **输入** | 12维状态特征 | TRCG诊断摘要（冲突图+聚类） | TRCG候选池 |
| **输出** | (freeze, epsilon) | (unlock_ids, root_cause) | GA搜索最优unlock子集 |
| **求解范式** | 全局重排（所有任务） | 局部修复（3-5个解锁任务） | 局部修复（K=5个解锁任务） |
| **计算复杂度** | $O(n^2)$ (20任务) | $O(k^2)$ (3任务) | $O(\text{pop} \times k^2)$ GA搜索 |
| **鲁棒性** | 单次求解（成功/失败） | 四级回退链（保证可行） | 三级回退链（保证可行） |

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
- **run_batch_10day.py**：长周期测试（10天×3难度×3baseline×N个seeds）
- **Episode Case Study**：双泳道Gantt图可视化（Baseline vs Ours）
- **7份新增文档**：从功能说明到测试指南，覆盖完整开发周期

#### 🔬 Phase 4 实验运行与代码迭代（2026-02-14至今）
- **Qwen3-32B实际LLM实验**：results_V2.5/{BL, LLM}目录含多轮种子匹配实验
- **RepairStepLog 3-way可观测性**：`llm_http_ok`/`llm_parse_ok`/`llm_decision_ok`（移除旧`llm_call_ok`）
- **_auto_correct_llm_output**：自动纠正LLM选出的非活跃/已完成 mission_id
- **_trcg_find_urgent回归修复**：移除错误的 started_ops 过滤
- **unlock_mission_ids激活**：确保 Anchor Fix-and-Optimize 实际生效

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
        │ - Greedy      │ │ CP-SAT     │ │ - Drift    │
        │ - RealLLM     │ │ 两阶段求解  │ │ - Switch   │
        │ - TRCGRepair  │ │ +Anchor LNS │ │ - Features │
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

**三种扰动强度**（用于实验分组）：

| 扰动类型 | Light | Medium | Heavy |
|---------|-------|--------|-------|
| 天气中断概率 | 5% | 7% | 10% |
| Pad故障概率 | 2% | 3% | 5% |
| 工序延迟标准差 | 12% | 20% | 30% |
| 释放时间扰动 | 2 slots | 3 slots | 4 slots |

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

**1. FixedWeightPolicy（固定参数策略，Baseline）**
```python
# 使用预设的固定参数
params = MetaParams(
    freeze_horizon=8,       # 固定8小时冻结
    epsilon_solver=0.05,    # 固定5%延迟容差
    use_two_stage=True,
    kappa_win=12.0,
    kappa_seq=6.0
)
```

**2. GreedyPolicy（启发式策略）**
- **EDFGreedy**: Earliest Due First（最早截止优先）
- **WindowGreedy**: 优先分配窗口最少的任务
- **特点**：不使用CP-SAT，直接构造可行解（速度快，质量低）

**3. RealLLMPolicy（LLM元策略，第一代方法）**

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

4. 三级回退链 → policy_llm_repair.solve_with_fallback_chain()
   Level 0: 初始解锁集（LLM/启发式）
   ├─ 失败 ↓
   Level 1: 扩大解锁集（+瓶颈关联任务）
   ├─ 失败 ↓
   Level 2: 减小冻结视野（freeze_horizon//2）
   ├─ 失败 ↓
   Level 3: 放松延迟容差（epsilon_solver×2）
   ├─ 失败 ↓
   Level 4: 强制全局重排（unlock all missions）
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
- **鲁棒性**：四级回退确保最终总有可行解（最差情况=全局重排）

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
pop_size = 16              # 种群大小
generations = 5            # 最大进化代数
K = 5                      # 解锁子集大小
mutation_rate = 0.2        # 变异概率
candidate_pool_size = 15   # 候选池大小（从TRCG提取）

# V2加速特性
n_jobs = 8                 # 并行worker数量（适应度评估）
eval_budget = 12           # 硬约束：最大评估次数
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

**6. MockLLMPolicy（模拟LLM策略，用于调试）**
- 使用硬编码规则模拟 LLM 决策逻辑（if-else）
- 用于快速验证框架正确性

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
│ - FixedWeightPolicy (最优参数)                           │
│ - GreedyPolicy (EDFGreedy / WindowGreedy)               │
│ - RealLLMPolicy (zero-shot)                             │
│ - MockLLMPolicy (规则模拟)                               │
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

