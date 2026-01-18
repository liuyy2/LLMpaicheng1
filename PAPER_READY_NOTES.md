# Paper-Ready Notes: LLM-Guided Rocket Launch Scheduling

本文档提供将实验结果写入学术论文时的关键注意事项、报告规范和局限性说明。

---

## 1. 时间指标分离报告

### 1.1 为什么要分离 LLM Time 与 Solver Time

在报告实验结果时，必须将 **LLM 推理时间** 与 **求解器计算时间** 分开报告，原因如下：

1. **可比性**：Baseline 策略（fixed, greedy, nofreeze）不使用 LLM，直接比较 wall time 不公平
2. **瓶颈分析**：帮助读者理解性能瓶颈在 LLM 还是求解器
3. **可复现性**：LLM 时间受网络延迟影响，solver time 更稳定可复现
4. **成本估算**：LLM 调用有 token 成本，需单独计量

### 1.2 报告格式建议

**表格格式**（推荐）：

| Policy | Avg Delay ↓ | Plan Drift ↓ | Solver Time (ms) | LLM Time (ms) | Wall Time (ms) |
|--------|-------------|--------------|------------------|---------------|----------------|
| Fixed (Tuned) | 0.12 ± 0.05 | 0.005 ± 0.002 | 625 ± 42 | - | 625 ± 42 |
| Greedy | 0.08 ± 0.03 | 0.012 ± 0.003 | 0 | - | 15 ± 3 |
| NoFreeze | 0.15 ± 0.06 | 0.008 ± 0.002 | 618 ± 38 | - | 618 ± 38 |
| **LLM-Guided** | **0.06 ± 0.02** | **0.004 ± 0.001** | 612 ± 35 | 3200 ± 580 | 3812 ± 590 |

**文字描述**：

> The LLM-guided policy achieves the lowest average delay (0.06 slots) with minimal plan drift (0.004).
> While the total wall-clock time (3.8s) is higher than baselines due to LLM inference latency (3.2s),
> the solver computation time (0.6s) remains comparable to other CP-SAT-based methods.

### 1.3 可用指标字段

从 `results_per_episode.csv` 提取：

| 指标 | CSV 字段 | 说明 |
|------|----------|------|
| Solver Time | `solver_time_total_ms` | CP-SAT 求解器总耗时 |
| LLM Time | `llm_time_total_ms` | LLM API 调用总耗时（含网络） |
| LLM Latency | `llm_latency_total_ms` | LLM 推理延迟（不含预处理） |
| Wall Time | `wall_time_total_ms` | Episode 总耗时 |
| Token Usage | `llm_total_tokens` | 总 token 消耗 |

---

## 2. Fallback 策略报告

### 2.1 什么是 Fallback

当 LLM 返回无效输出（非 JSON、格式错误、数值越界）时，系统自动回退到默认参数：

```python
DEFAULT_PARAMS = {
    "freeze_horizon_slots": 3,
    "w_delay": 10.0,
    "w_shift": 1.0,
    "w_switch": 5.0
}
```

### 2.2 报告 Fallback 率

**必须报告的指标**：

| 指标 | 计算方式 | 期望值 |
|------|----------|--------|
| Fallback Rate | `llm_fallback_count / llm_calls` | < 5% |
| Total Fallbacks | `sum(llm_fallback_count)` | 越低越好 |

**论文报告示例**：

> The LLM-guided policy achieved a fallback rate of 2.3% (47 fallbacks out of 2,040 LLM calls),
> indicating that the structured prompt design effectively elicits valid JSON responses from the model.

### 2.3 Fallback 对结果的影响

在论文中应说明：

1. **透明性**：Fallback 使用的是经过调参的默认值，而非随机值
2. **保守性**：Fallback 参数倾向于保守策略，优先保证约束满足
3. **上界估计**：高 Fallback 率时，LLM 策略退化为 Fixed 策略

---

## 3. 缓存与可复现性

### 3.1 缓存机制说明

在 Method 章节应说明缓存设计：

> To ensure reproducibility of LLM-based experiments, we implement a deterministic caching mechanism.
> Each LLM query is cached using a hash of the input prompt, model name, and temperature setting.
> Subsequent runs with identical inputs retrieve cached responses, eliminating non-determinism
> from network latency and potential model version updates.

### 3.2 缓存使用声明

**实验设置章节**：

> All reported results use cached LLM responses to ensure reproducibility.
> The cache was populated during the initial experiment run on [DATE] using [MODEL_VERSION].
> Cache files are available in the supplementary materials for verification.

### 3.3 如何共享缓存

发布论文时，应在补充材料中包含：

```
supplementary/
├── llm_cache/           # 完整缓存目录
│   ├── ab/ab1234...json
│   └── ...
├── cache_manifest.json  # 缓存文件清单
└── README.md            # 使用说明
```

**cache_manifest.json 示例**：

```json
{
  "model": "qwen3-32b",
  "temperature": 0.0,
  "cache_date": "2026-01-15",
  "total_entries": 2040,
  "total_tokens": 512000
}
```

---

## 4. 公平性声明

### 4.1 Baseline 调参说明

**训练/测试集分离**：

> Hyperparameters for baseline policies (Fixed, NoFreeze) were tuned on a training set
> of 60 episodes (20 seeds × 3 disturbance levels) using grid search to minimize
> the combined objective: `delay + 5 × drift`. The test set consists of 30 distinct
> episodes with non-overlapping seeds.

**调参搜索空间**：

| 参数 | 搜索范围 | 最优值 |
|------|----------|--------|
| freeze_horizon_slots | [1, 2, 3, 4, 5] | 3 |
| w_delay | [1, 5, 10, 20] | 10 |
| w_shift | [0.5, 1, 2, 5] | 1 |
| w_switch | [1, 2, 5, 10] | 5 |

### 4.2 LLM 策略的公平性

**顺序执行防限流**：

> To ensure fair comparison and avoid API rate limiting effects, LLM-guided experiments
> were executed sequentially (workers=1) while baseline policies were run in parallel (workers=4).
> This design choice prioritizes result reliability over execution speed.

**无调参声明**：

> The LLM-guided policy uses zero-shot prompting without task-specific fine-tuning or
> prompt optimization. The prompt template remained fixed throughout all experiments,
> ensuring that reported performance reflects the model's out-of-box capabilities.

### 4.3 随机种子固定

> All experiments use fixed random seeds (1-30 for test set) to ensure deterministic
> disturbance generation. Combined with LLM response caching, this guarantees that
> reported results are fully reproducible.

---

## 5. 局限性声明

### 5.1 LLM 非确定性

**问题**：即使设置 `temperature=0`，LLM 输出仍可能存在微小变化。

**论文表述**：

> **Limitation 1: LLM Non-determinism.** While we set temperature=0 to minimize randomness,
> LLM outputs may exhibit slight variations across different API calls due to:
> (1) floating-point precision in softmax computation,
> (2) potential model updates by the provider, and
> (3) hardware-level non-determinism in GPU inference.
> Our caching mechanism mitigates this issue for reported results, but users reproducing
> experiments without the provided cache may observe minor variations (typically <1% in metrics).

### 5.2 网络延迟变化

**问题**：LLM 调用时间受网络条件影响。

**论文表述**：

> **Limitation 2: Network Latency Variability.** The reported LLM inference times include
> network round-trip latency, which varies based on geographic location, time of day,
> and API server load. Users in different regions may experience 50-200% variation in
> LLM time metrics. We recommend focusing on solver time for algorithmic comparisons
> and treating LLM time as an implementation-dependent overhead.

### 5.3 供应商/模型变化

**问题**：API 模型可能被更新或废弃。

**论文表述**：

> **Limitation 3: Model Version Dependency.** Our experiments use [MODEL_NAME] via
> [PROVIDER] API as of [DATE]. Model capabilities and behaviors may change with
> provider updates. We document the exact model version and provide cached responses
> in supplementary materials. Future work should consider self-hosted or versioned
> models for long-term reproducibility.

### 5.4 成本约束

**问题**：大规模实验的 API 成本限制。

**论文表述**：

> **Limitation 4: API Cost Constraints.** Each episode requires approximately 40-50 LLM
> calls, consuming ~1,200 tokens per episode. For our test set of 30 episodes, the total
> cost is approximately $X.XX using [MODEL] pricing. This limits the feasibility of
> extensive hyperparameter search or large-scale ablation studies for LLM-based policies.

---

## 6. 推荐论文结构

### 6.1 实验设置章节

```markdown
## Experimental Setup

### Environment
- Hardware: [CPU/GPU specs]
- Software: Python 3.10, OR-Tools 9.x, OpenAI SDK 1.x

### LLM Configuration
- Model: Qwen3-32B via DashScope API
- Temperature: 0.0
- Max tokens: 256
- Caching: Enabled (SHA-256 hash of prompt)

### Baselines
- Fixed: Tuned on training set (freeze=3, w_delay=10, w_shift=1, w_switch=5)
- Greedy: Earliest-deadline-first heuristic
- NoFreeze: CP-SAT without freeze horizon

### Evaluation Metrics
- Average Delay (slots): Mean task completion delay
- Plan Drift: Normalized schedule change magnitude
- Combined Score: delay + 5 × drift
```

### 6.2 结果表格模板

```markdown
| Policy | Delay ↓ | Drift ↓ | Combined ↓ | Solver (ms) | LLM (ms) | Fallback % |
|--------|---------|---------|------------|-------------|----------|------------|
| Fixed  | 0.12±0.05 | 0.005±0.002 | 0.15±0.06 | 625±42 | - | - |
| Greedy | 0.08±0.03 | 0.012±0.003 | 0.14±0.04 | 0 | - | - |
| NoFreeze | 0.15±0.06 | 0.008±0.002 | 0.19±0.07 | 618±38 | - | - |
| **LLM** | **0.06±0.02** | **0.004±0.001** | **0.08±0.03** | 612±35 | 3200±580 | 2.3% |
```

### 6.3 统计检验报告

> Paired t-tests comparing LLM-guided against the best baseline (Fixed-Tuned) show
> statistically significant improvements in both delay (t=3.42, p<0.01) and drift
> (t=2.89, p<0.05) on the test set (n=30 episodes).

---

## 7. 补充材料清单

建议在论文补充材料中提供：

| 文件 | 内容 |
|------|------|
| `code.zip` | 完整代码库 |
| `llm_cache.zip` | LLM 响应缓存 |
| `results_per_episode.csv` | 原始实验数据 |
| `summary.csv` | 汇总统计 |
| `figures/` | 高分辨率图表 |
| `prompts/` | 完整 prompt 模板 |
| `README.md` | 复现指南 |

---

## 8. 常见审稿问题预案

### Q1: LLM 调用成本是否实用？

> 每 episode 约 40 次调用、1200 tokens，按 Qwen3-32B 定价约 ¥0.02/episode。
> 对于关键任务调度（如航天发射），此成本相对于潜在收益可忽略不计。

### Q2: 如何保证 LLM 输出的安全性？

> 所有 LLM 建议的参数经过边界检查（freeze_horizon ∈ [1,10], w_* ∈ [0.1, 100]）。
> 超出范围的值触发 fallback 机制，使用保守的默认参数。

### Q3: 为什么不微调 LLM？

> 本研究聚焦于 zero-shot 能力评估。微调需要大量领域数据且可能过拟合。
> 我们的结果表明，通用 LLM 通过合理的 prompt 设计即可提供有效的调度建议。

### Q4: 如何处理 LLM 的幻觉问题？

> (1) 结构化输出（JSON）约束响应格式
> (2) 数值边界检查防止极端值
> (3) Fallback 机制保底
> (4) 最终决策由 CP-SAT 求解器执行，LLM 仅提供参数建议

---

## 9. 引用格式

如果使用本代码库，请引用：

```bibtex
@misc{llm_rocket_scheduling_2026,
  title={LLM-Guided Adaptive Scheduling for Rocket Launch Operations},
  author={[Author Names]},
  year={2026},
  howpublished={\url{https://github.com/xxx/LLMpaicheng1}},
  note={Code and cached LLM responses available}
}
```

---

*文档版本: 1.0 | 更新日期: 2026-01-17*
