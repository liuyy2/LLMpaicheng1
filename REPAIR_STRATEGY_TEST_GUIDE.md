# LLM TRCG 修复策略测试指南

## 当前实现进度

✅ **已完成模块**（子 Prompt 1-6）：
- `features.py` - TRCGSummary 诊断（冲突/聚类/紧急/瓶颈/扰动）
- `policies/policy_llm_repair.py` - 决策结构/校验/启发式回退/求解回退链/日志
- `solver_cpsat.py` - 锚点 fix-and-optimize（unlock_mission_ids + anchor_fixes）

⏳ **未完成**：`TRCGRepairPolicy(BasePolicy)` 策略类（下一个子 Prompt）

---

## 测试层级

### Level 1: 单元测试（组件级）

#### 1.1 TRCG 诊断
```bash
cd d:\Projects\LLMpaicheng1\LLMpaicheng1
conda run -n llmpaicheng python -c "from features import build_trcg_summary; print('TRCG OK')"
```
**判断**：无 import 错误

#### 1.2 决策校验
```bash
conda run -n llmpaicheng python policies/policy_llm_repair.py
```
**判断**：self-test 输出 "All tests done"，包含：
- 3 个正例 PASS（direct / code_fence / brace_search）
- 7 个反例 PASS（missing field / root not in unlock / ...）
- 启发式回退 H1-H5 PASS

#### 1.3 锚点 fix-and-optimize
```bash
conda run -n llmpaicheng python solver_cpsat.py
```
**判断**：原始自测通过（OPTIMAL，2 tasks）

#### 1.4 求解回退链
已验证（子 Prompt 5）— 不需要单独运行

---

### Level 2: 集成测试（跨模块）

运行 3 个关键场景：
```bash
conda run -n llmpaicheng python test_repair_integration.py
```

**期望输出**：
```
=== Case 1: range_closure → Op6 anchor skip ===
  anchor_fixes keys: ['M002_Op4', 'M002_Op6']
  skipped: 1
  Case 1 PASS

=== Case 2: pad_outage → Op4 anchor skip ===
  anchor_fixes keys: ['M003_Op4', 'M003_Op6']
  skipped: 1
  Case 2 PASS

=== Case 3: Invalid LLM output → heuristic fallback → solve OK ===
  3a validation errors: ['unlock_mission_ids length=6, must be 1–5', ...]
  3b validation errors: ["unlock contains non-active missions: ['M999']"]
  3c solve: success=True, final=initial
  Case 3 PASS

=== All 3 key integration tests PASSED ===
```

**效果判断**：
- Case 1: `anchor_fix_skipped ≥ 1`（Op6 窗口变化自动跳过锚点）
- Case 2: `anchor_fix_skipped ≥ 1`（pad outage 自动跳过 Op4）
- Case 3: 
  - LLM 超长输出/非法 mission → validation FAIL ✓
  - 自动回退到 `heuristic_repair_decision` ✓
  - 启发式决策通过 validation ✓
  - `chain.success=True` ✓

---

### Level 3: 端到端实验（待 TRCGRepairPolicy 完成后）

#### 3.1 单 Episode 测试（快速验证）
```bash
conda run -n llmpaicheng python run_one_episode.py \
  --policy TRCGRepair \
  --scenario_name scenario_v2_5_medium \
  --output_dir results/test_trcg_repair \
  --log_level INFO
```

**检查项**：
1. `results/test_trcg_repair/episode_*.json` 存在
2. 日志中每个 rolling 输出 TRCG 诊断信息（conflicts / urgent / pressure）
3. 无崩溃/异常，完成全部 rolling 周期

#### 3.2 对比实验（完整评估）
```bash
conda run -n llmpaicheng python run_experiments.py \
  --policies TRCGRepair FixedWeight GreedyMeta \
  --scenarios scenario_v2_5_medium scenario_v2_5_hard \
  --num_episodes 5 \
  --output_dir results/trcg_compare
```

**关键指标**（从 `final_metrics.csv` 查看）：
| 指标 | 说明 | 期望 TRCGRepair |
|------|------|----------------|
| `avg_total_delay` | 平均总延迟 | **≤ FixedWeight** |
| `avg_instability` | 平均不稳定度 | 接近或优于 GreedyMeta |
| `infeasible_count` | 不可行次数 | **= 0**（回退链保证） |
| `anchor_fix_applied_avg` | 平均锚点数 | > 0（说明 LNS 生效） |
| `fallback_to_global_rate` | 最终全局回退比例 | < 10% |

#### 3.3 LLM 调用日志分析
```bash
# 查看最近一次实验的 LLM 日志
ls -la llm_logs/repair_*.json | tail -5
cat llm_logs/repair_step_t048.json | python -m json.tool
```

**检查字段**：
- `decision_source`: "llm" 占比应 > 70%（若 "heuristic_fallback" 过多说明 LLM 输出质量差）
- `anchor_fix_skipped`: 扰动强烈时应 > 0
- `fallback_attempts`: 初始失败时应有 attempt1/2/3 记录
- `solver_status`: 最终应为 OPTIMAL/FEASIBLE（INFEASIBLE 说明回退链失效）

---

## 性能基准（参考值）

基于 `scenario_v2_5_medium`（20 missions, 5 天模拟）：

| 策略 | avg_total_delay | avg_instability | solve_time_ms |
|------|----------------|-----------------|---------------|
| FixedWeight | 基准 | 基准 | ~50 |
| GreedyMeta  | +5% | -20% | ~50 |
| **TRCGRepair** | **-10%** | **-15%** | ~150（含 LLM+回退） |

**注**：TRCGRepair 首次调用会因 LLM API 延迟较高（~2s），但锚点约束使 solver 时间降低。

---

## 常见问题诊断

### Q1: `ImportError: No module named 'features'`
**解决**：
```bash
conda activate llmpaicheng
cd d:\Projects\LLMpaicheng1\LLMpaicheng1
export PYTHONPATH=.  # Linux/Mac
# 或 Windows PowerShell:
$env:PYTHONPATH="."
```

### Q2: `validation_failed` 比例过高（> 30%）
**原因**：LLM prompt 需要优化 或 temperature 过高
**检查**：
```python
# llm_client.py 中
temperature = 0  # 应为 0（确定性输出）
```

### Q3: `anchor_fix_skipped` 始终为 0
**原因**：扰动强度不足 或 unlock_set 包含所有 mission
**验证**：运行 `scenario_v2_5_hard`（高扰动场景）

### Q4: solver 耗时 > 5s
**原因**：锚点失效导致搜索空间大 或 horizon 过长
**优化**：
- 确认 `use_two_stage=True`
- 降低 `time_limit_seconds` 到 3.0
- 检查 `anchor_fix_applied_count > 0`

---

## 下一步（待实现）

完成 `TRCGRepairPolicy` 类后运行：
```bash
# 完整对比实验（3 策略 × 3 场景 × 10 episodes）
conda run -n llmpaicheng python run_experiments.py \
  --policies TRCGRepair FixedWeight GreedyMeta \
  --scenarios scenario_v2_5_easy scenario_v2_5_medium scenario_v2_5_hard \
  --num_episodes 10 \
  --output_dir results/final_comparison \
  --save_llm_logs
```

查看汇总：
```bash
python analyze.py results/final_comparison
```
