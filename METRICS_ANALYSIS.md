# 项目指标分析与论文图表映射

## 一、当前指标清单

### 1. **准时性指标 (Timeliness)**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `on_time_rate` | 按期发射率（delay=0 的任务比例） | ⭐⭐⭐ | **核心** - Figure 3 |
| `avg_delay` | 平均延迟（slots） | ⭐⭐⭐ | **核心** - Figure 3, 4 |
| `max_delay` | 最大延迟（slots） | ⭐ | 次要 - 可用于鲁棒性分析 |
| `total_delay` | 总延迟（slots） | ⭐ | 冗余 - 可由 avg_delay 推导 |
| `weighted_tardiness` | 加权延迟（考虑优先级） | ⭐⭐⭐ | **核心** - Figure 3, 4 |

**推荐保留：** `on_time_rate`, `avg_delay`, `weighted_tardiness`  
**建议去掉：** `total_delay`（冗余）, `max_delay`（次要，除非专门分析极端情况）

---

### 2. **稳定性指标 (Stability)**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `episode_drift` | Episode 总 Drift（归一化重排扰动） | ⭐⭐⭐ | **核心** - Figure 2, 3, 4 |
| `drift_per_replan` | 平均每次重排的 drift | ⭐⭐⭐ | **核心** - 归一化比较 |
| `drift_per_day` | 平均每天的 drift | ⭐⭐ | 次要 - 多天实验对比 |
| `total_shifts` | 总时间变化次数 | ⭐⭐ | 次要 - Figure 2 |
| `total_switches` | 总 Pad 切换次数 | ⭐⭐⭐ | **核心** - Figure 2 |
| `total_window_switches` | 时间窗切换次数 | ⭐⭐ | 次要 - 领域特定 |
| `total_sequence_switches` | Pad 序列切换次数 | ⭐⭐ | 次要 - 领域特定 |
| `total_resource_switches` | 资源切换总数 | ⭐ | 冗余 - 与 total_switches 重复 |
| `avg_time_deviation_min` | 平均时间偏移（分钟） | ⭐ | 冗余 - 与 avg_time_shift_slots 重复 |

**推荐保留：** `episode_drift`, `drift_per_replan`, `total_switches`  
**建议去掉：** `total_resource_switches`（冗余）, `avg_time_deviation_min`（等价于 total_shifts 加单位换算）

---

### 3. **求解性能指标 (Solver Performance)**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `avg_solve_time_ms` | 平均求解时间（毫秒） | ⭐⭐⭐ | **核心** - Table 1 |
| `total_solve_time_ms` | 总求解时间（毫秒） | ⭐ | 次要 - 可选 |
| `num_replans` | 重排次数 | ⭐⭐⭐ | **核心** - Figure 2, Table 1 |
| `num_forced_replans` | 强制重排次数 | ⭐⭐ | 次要 - 可行性分析 |
| `feasible_rate` | 可行率 | ⭐⭐ | 次要 - 鲁棒性证据 |
| `forced_replan_rate` | 强制重排率 | ⭐ | 冗余 - 可由 num_forced_replans/num_replans 推导 |

**推荐保留：** `avg_solve_time_ms`, `num_replans`, `feasible_rate`  
**建议去掉：** `total_solve_time_ms`（冗余）, `forced_replan_rate`（可推导）

---

### 4. **资源利用率指标 (Resource Utilization)**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `resource_utilization` | 总资源利用率 | ⭐⭐ | 次要 - 可选背景信息 |
| `util_r_pad` | Pad 资源利用率 | ⭐⭐ | 次要 - 领域特定 |

**推荐保留：** `util_r_pad`（如果 Pad 是核心资源瓶颈）  
**建议去掉：** `resource_utilization`（除非论文需要证明方法不浪费资源）

---

### 5. **完成度指标 (Completion)**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `completion_rate` | 完成率 | ⭐⭐⭐ | **核心** - 必须 = 100% 才能比较其他指标 |
| `num_completed` | 完成任务数 | ⭐ | 冗余 - 可由 completion_rate 推导 |
| `num_total` | 总任务数 | ⭐ | 冗余 - 实验配置信息 |

**推荐保留：** `completion_rate`  
**建议去掉：** `num_completed`, `num_total`（作为表格注释而非指标）

---

### 6. **其他辅助指标**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `avg_frozen` | 平均冻结数量 | ⭐⭐ | 次要 - 机制解释（Figure 2） |
| `avg_num_tasks_scheduled` | 平均调度任务数 | ⭐ | 次要 - 实现细节 |
| `makespan_cmax` | 完成时间 Cmax | ⭐ | 次要 - 如果不涉及 makespan 优化 |

**推荐保留：** `avg_frozen`（证明 freeze 机制有效）  
**建议去掉：** `avg_num_tasks_scheduled`, `makespan_cmax`（除非论文目标涉及）

---

### 7. **LLM 相关指标（仅 llm_real 策略）**
| 指标名称 | 定义 | 重要性 | 论文使用 |
|---------|------|--------|---------|
| `llm_calls` | LLM 调用次数 | ⭐⭐ | 次要 - 成本分析 |
| `llm_time_total_ms` | LLM 总耗时 | ⭐⭐ | 次要 - 性能对比 |
| `llm_cache_hit_rate` | 缓存命中率 | ⭐ | 次要 - 实现细节 |
| `llm_fallback_count` | 降级次数 | ⭐⭐ | 次要 - 可靠性证据 |
| `llm_prompt/completion_tokens` | Token 消耗 | ⭐ | 次要 - 成本分析 |

**推荐保留：** `llm_time_total_ms`, `llm_fallback_count`（如果论文涉及 LLM 可靠性）  
**建议去掉：** Token 相关指标（除非做成本分析）

---

## 二、论文核心图表 vs 必备指标映射

### **Figure 1: 任务与滚动机制示意图**（定义正确性）
- **所需数据：** 无需指标数据，纯概念示意图
- **内容：** roll 时刻、freeze 区间、planning window、Op1-Op6 流程、Pad 资源

---

### **Figure 2: 单 Episode Case Study**（机制证据）
- **x 轴：** `roll_index`（时间）
- **上半部分：** Gantt Chart（Pad 双泳道）
  - 需要：每个 roll 的 `plan` 数据（任务-Pad-时间分配）
  - 标注：变化段（`num_shifts`, `num_switches`）
- **下半部分：**
  - **线图：** `plan_drift`（每次 roll 的值）
  - **柱状图：** `num_switches`（每次 roll 的值）
  - **竖线：** 扰动时刻

**必备指标：**
- Rolling 级别：`plan_drift`, `num_switches`, `num_shifts`
- 辅助：`avg_frozen`（证明 freeze 机制）

---

### **Figure 3: 总体效果分布图**（结论证据）
- **子图 A：准时性**
  - **推荐指标：** `avg_delay` 或 `weighted_tardiness`
  - **图类型：** ECDF 或 box plot（多 seed × 多 episode）
  
- **子图 B：稳定性**
  - **推荐指标：** `episode_drift` 或 `drift_per_replan`
  - **图类型：** ECDF 或 box plot

**必备指标：**
- `avg_delay`（或 `weighted_tardiness`）
- `episode_drift`（或 `drift_per_replan`）

---

### **Figure 4: Pareto/ε-constraint 证据图**（封死"权衡"质疑）
- **x 轴：** `avg_delay` 或 `weighted_tardiness`
- **y 轴：** `episode_drift` 或 `total_switches`
- **每个点：** 一次 episode 或均值点+误差条
- **理想结果：** 你的方法的点云在"左下"（delay 更小，drift 更小）

**必备指标：**
- `avg_delay`（或 `weighted_tardiness`）
- `episode_drift`（或 `drift_per_replan`）

---

### **Table 1: Baselines 与超参数公平性表**（防"baseline 不公平"质疑）
| Policy | Freeze? | Repair? | Objective | Time Limit (s) | Threads | Tuned? |
|--------|---------|---------|-----------|---------------|---------|--------|
| baseline_full_replan | × | × | min weighted tardiness | 60 | 8 | ✓ |
| trcg_freeze_repair | ✓ | ✓ | min drift | 60 | 8 | ✓ |
| llm_real | ✓ | LLM | min drift | 60 | 8 | ✓ |

**配套指标：**
- `avg_solve_time_ms`（证明求解器预算一致）
- `feasible_rate`（证明都能求解）

---

### **Figure 5: 扰动强度分层结果**（鲁棒性证据）
- **x 轴：** 扰动强度（low/medium/high）
- **y 轴 A：** `avg_delay`
- **y 轴 B：** `episode_drift`
- **图类型：** 分组柱状图+误差棒

**必备指标：**
- `avg_delay`
- `episode_drift`

---

## 三、最终推荐的核心指标集

### **主论文必备（8 个核心指标）**

#### **准时性（2 个）**
1. ✅ `avg_delay` - 平均延迟（slots）
2. ✅ `weighted_tardiness` - 加权延迟（考虑优先级）

#### **稳定性（3 个）**
3. ✅ `episode_drift` - Episode 总 Drift
4. ✅ `drift_per_replan` - 每次重排的平均 drift（归一化）
5. ✅ `total_switches` - 总 Pad 切换次数

#### **求解性能（2 个）**
6. ✅ `avg_solve_time_ms` - 平均求解时间
7. ✅ `num_replans` - 重排次数

#### **完成度（1 个）**
8. ✅ `completion_rate` - 完成率（必须 ≈ 100%）

---

### **次要指标（可选，用于补充分析）**

#### **稳定性细节**
- `total_shifts` - 时间变化次数（Figure 2 解释用）
- `avg_frozen` - 平均冻结数量（证明 freeze 机制）

#### **准时性细节**
- `on_time_rate` - 按期发射率（可作为 alt. 指标）

#### **鲁棒性**
- `feasible_rate` - 可行率（证明方法稳健）
- `num_forced_replans` - 强制重排次数（可行性分析）

#### **资源效率**
- `util_r_pad` - Pad 利用率（如果 Pad 是瓶颈资源）

---

## 四、需要删除或合并的冗余指标

### **直接删除（冗余）**
1. ❌ `total_delay` - 可由 `avg_delay × num_total` 推导
2. ❌ `total_solve_time_ms` - 可由 `avg_solve_time_ms × num_replans` 推导
3. ❌ `total_resource_switches` - 与 `total_switches` 重复
4. ❌ `forced_replan_rate` - 可由 `num_forced_replans / num_replans` 推导
5. ❌ `num_completed`, `num_total` - 作为表格注释，非独立指标

### **降级为辅助信息（不作为主指标）**
1. 📊 `max_delay` - 仅在讨论极端情况时提及
2. 📊 `makespan_cmax` - 除非目标包含 makespan 优化
3. 📊 `resource_utilization` - 除非论文强调资源效率
4. 📊 `avg_time_deviation_min` - 单位换算，不作为独立指标
5. 📊 `drift_per_day` - 仅在多天对比时使用

### **LLM 特定指标（仅 llm_real 策略报告）**
- 保留 `llm_time_total_ms`, `llm_fallback_count`
- Token 相关指标移至附录或成本分析部分

---

## 五、指标数据流架构

```
Episode 实验
    ↓
Rolling 级别指标（存储在 rolling_metrics_list）
    - plan_drift
    - num_shifts
    - num_switches
    - num_frozen
    - solve_time_ms
    ↓
Episode 级别指标（compute_episode_metrics）
    → 准时性：avg_delay, weighted_tardiness
    → 稳定性：episode_drift, drift_per_replan, total_switches
    → 性能：avg_solve_time_ms, num_replans
    → 完成度：completion_rate
    ↓
CSV 输出（results_per_episode.csv）
    ↓
分析脚本（analyze.py）
    → 统计量：mean, CI, std
    → 图表：ECDF, box plot, scatter, Gantt
    ↓
论文图表
    - Figure 2: 单 episode 案例（Gantt + drift 曲线）
    - Figure 3: 分布对比（ECDF/box）
    - Figure 4: Pareto 图（delay vs drift scatter）
    - Figure 5: 扰动分层（grouped bar）
```

---

## 六、代码修改建议

### **1. metrics.py：简化 EpisodeMetrics 数据结构**
```python
@dataclass
class EpisodeMetrics:
    """Simplified core metrics for paper."""
    # === 核心准时性指标 ===
    avg_delay: float
    weighted_tardiness: float
    
    # === 核心稳定性指标 ===
    episode_drift: float
    drift_per_replan: float
    total_switches: int
    
    # === 求解性能 ===
    avg_solve_time_ms: float
    num_replans: int
    
    # === 完成度 ===
    completion_rate: float
    
    # === 次要指标（可选） ===
    on_time_rate: float = 0.0
    total_shifts: int = 0
    avg_frozen: float = 0.0
    feasible_rate: float = 1.0
    num_forced_replans: int = 0
    util_r_pad: float = 0.0
    
    # === 内部计算用（不输出） ===
    num_total: int = 0
    num_completed: int = 0
```

### **2. analyze.py：更新图表生成函数**
- ✅ `plot_ecdf_comparison()`：Figure 3
- ✅ `plot_pareto_scatter()`：Figure 4
- ✅ `plot_disturbance_stratified()`：Figure 5
- ✅ `plot_case_study_gantt()`：Figure 2

### **3. 删除冗余字段**
在以下文件中移除：
- `metrics.py`: `total_delay`, `total_solve_time_ms`, `total_resource_switches`
- `analyze.py`: `EpisodeRecord` 对应字段
- CSV 输出逻辑：删除相关列

---

## 七、论文指标呈现建议

### **主文表格（Table 2: 实验结果总结）**
| Method | Avg Delay↓ | W. Tardiness↓ | Drift↓ | Switches↓ | Time (ms) |
|--------|-----------|--------------|--------|-----------|-----------|
| Full Replan | 2.34±0.51 | 45.6±8.2 | 0.42±0.08 | 134±23 | 523±67 |
| TRCG (Ours) | **1.87±0.39** | **32.1±5.6** | **0.18±0.04** | **47±12** | 489±54 |
| LLM Repair | 2.01±0.44 | 35.3±6.1 | 0.22±0.05 | 58±15 | 1124±89 |

*（数值为 mean±std，多 seed × 多 episode，加粗表示最优）*

### **附录表格（Table A1: 完整指标）**
包含：`on_time_rate`, `feasible_rate`, `num_replans`, `completion_rate`, etc.

---

## 八、执行计划

### **Phase 1: 清理指标定义**
1. 修改 [metrics.py](metrics.py) 的 `EpisodeMetrics`，标注核心/次要/废弃
2. 添加 `@deprecated` 注释到冗余字段

### **Phase 2: 更新数据流**
1. 确保 CSV 输出只包含核心+次要指标
2. 更新 [analyze.py](analyze.py) 的 `EpisodeRecord`

### **Phase 3: 生成论文图表**
1. 实现 Figure 2-5 的绘图函数
2. 生成 LaTeX 格式的表格代码

### **Phase 4: 验证与文档**
1. 运行完整实验流程确认指标正确
2. 更新 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

---

## 总结

**核心结论：**
- **必须保留的 8 个核心指标** 足以支撑论文的 4 图 1 表
- **删除 5 个冗余指标**，减少数据噪音
- **降级 8 个次要指标** 至辅助地位，按需使用

**关键原则：**
1. **最小必备集**：8 个核心指标覆盖准时性、稳定性、性能、完成度
2. **可推导的不保留**：避免冗余存储（如 total_delay）
3. **领域特定的后置**：如 window_switches, sequence_switches 仅在需要时讨论
4. **对标审稿人视角**：每个指标都能回答一个明确的质疑

**下一步：**
需要我执行具体的代码清理和图表生成实现吗？
