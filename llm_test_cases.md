# TRCG Repair LLM 策略 — 模型测试用例集

## 背景说明

本文档包含 `trcg_repair_llm` 策略在不同调度场景下发送给 LLM 的**真实 prompt**。
你可以将下面每个测试用例的 **System Prompt** 和 **User Prompt** 分别复制到不同模型的网页端，
观察各模型的输出质量、格式合规性和推理准确性，以选择最适合调用的模型 API。

### 任务概述
- **领域**: 火箭发射排程调度（Rolling Horizon 仿真）
- **LLM 的角色**: 根因诊断 + 修复锚点选择
- **输入**: TRCG（Temporal-Resource Conflict Graph）诊断摘要
- **输出**: 严格 JSON（4 个字段），指定根因任务和需要解锁重排的任务集

### 期望的输出格式
```json
{
  "root_cause_mission_id": "M00X",
  "unlock_mission_ids": ["M00X", "M00Y"],
  "secondary_root_cause_mission_id": "M00Y" 或 null,
  "analysis_short": "≤120字符的根因分析简述"
}
```

### 评估维度
1. **格式合规**: 是否输出纯 JSON，无多余文字/markdown
2. **字段完整**: 4 个 required 字段是否齐全
3. **业务逻辑**: root_cause 是否在 unlock 内；unlock 是否为 active 子集
4. **推理质量**: 根因识别是否准确；unlock 集合大小是否合理
5. **响应速度**: 延迟是否可接受（生产中每步 rolling 都要调一次）

---

## 测试用例 1: 真实仿真-早期无冲突（pad/R3满载）

| 属性 | 值 |
|------|-----|
| 场景种子 | 42 |
| 难度 | medium |
| 当前时刻 | slot 48 (第 12.0 小时) |
| 活跃任务数 | 9 |
| 活跃任务 | M000, M001, M002, M004, M006, M011, M012, M018, M019 |
| Pad利用率 | 100.00% |
| R3利用率 | 100.00% |
| 冲突数量 | 0 |
| 冲突簇数 | 0 |
| 紧迫任务 | M004, M002, M001 |
| 窗口损失率 | 0.0% |
| Pad故障 | 否 |

### System Prompt

```
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic summary, identify the root cause of scheduling conflicts and decide which missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–8 mission IDs. MUST include root_cause_mission_id. Only include missions that are currently ACTIVE (not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (soft):
- If conflicts are few / mild → small unlock set (1–2). Stability is king.
- If a bottleneck resource (pad/R3) is saturated → unlock the cluster around it (3–5 missions sharing that resource).
- If pad_outage is active → prefer not unlocking missions already frozen on pad.
- If range_loss_pct > 0.3 → unlock affected missions whose windows are shrinking.
- If no conflicts at all → unlock only 1 urgent mission for minor adjustment.
- When uncertain, be conservative: unlock only the conflict cluster center + 1 neighbor.
```

### User Prompt

```
TRCG diagnostic summary (JSON):
{"now_slot":48,"horizon_end_slot":144,"bottleneck_pressure":{"pad_util":1.0,"r3_util":1.0,"range_test_util":0.1875},"top_conflicts":[],"conflict_clusters":[],"urgent_missions":[{"mission_id":"M004","due_slot":168,"due_slack_slots":120,"window_slack_slots":99,"current_delay_slots":0,"priority":0.45,"urgency_score":169.5},{"mission_id":"M002","due_slot":188,"due_slack_slots":140,"window_slack_slots":116,"current_delay_slots":0,"priority":0.37,"urgency_score":198.0},{"mission_id":"M001","due_slot":209,"due_slack_slots":161,"window_slack_slots":138,"current_delay_slots":0,"priority":0.71,"urgency_score":230.0}],"disturbance_summary":{"range_loss_pct":0.0,"pad_outage_active":false,"duration_volatility_level":"low"},"frozen_summary":{"num_started_ops":0,"num_frozen_ops":0,"frozen_horizon_slots":12}}

Active (schedulable, not started/completed) missions: [M000, M001, M002, M004, M006, M011, M012, M018, M019]

unlock_mission_ids must be a SUBSET of the active missions above, size 1–8, and must include root_cause_mission_id.

Output the JSON now (4 keys only: root_cause_mission_id, unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):
```

### 参考答案（启发式基线的决策，仅供对比）

```json
{
  "root_cause_mission_id": "M004",
  "unlock_mission_ids": [
    "M004"
  ],
  "secondary_root_cause_mission_id": "M002",
  "analysis_short": "heuristic: root=M004 conflicts=0 pad_util=1.00"
}
```

---

## 测试用例 2: 真实仿真-中期高资源压力+已过期任务

| 属性 | 值 |
|------|-----|
| 场景种子 | 42 |
| 难度 | heavy |
| 当前时刻 | slot 192 (第 48.0 小时) |
| 活跃任务数 | 16 |
| 活跃任务 | M000, M001, M002, M004, M006, M007, M011, M012, M013, M015, M016, M018, M019, M022, M023, M024 |
| Pad利用率 | 100.00% |
| R3利用率 | 100.00% |
| 冲突数量 | 0 |
| 冲突簇数 | 0 |
| 紧迫任务 | M001, M013, M002 |
| 窗口损失率 | 0.0% |
| Pad故障 | 否 |

### System Prompt

```
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic summary, identify the root cause of scheduling conflicts and decide which missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–8 mission IDs. MUST include root_cause_mission_id. Only include missions that are currently ACTIVE (not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (soft):
- If conflicts are few / mild → small unlock set (1–2). Stability is king.
- If a bottleneck resource (pad/R3) is saturated → unlock the cluster around it (3–5 missions sharing that resource).
- If pad_outage is active → prefer not unlocking missions already frozen on pad.
- If range_loss_pct > 0.3 → unlock affected missions whose windows are shrinking.
- If no conflicts at all → unlock only 1 urgent mission for minor adjustment.
- When uncertain, be conservative: unlock only the conflict cluster center + 1 neighbor.
```

### User Prompt

```
TRCG diagnostic summary (JSON):
{"now_slot":192,"horizon_end_slot":288,"bottleneck_pressure":{"pad_util":1.0,"r3_util":1.0,"range_test_util":0.3333},"top_conflicts":[],"conflict_clusters":[],"urgent_missions":[{"mission_id":"M001","due_slot":207,"due_slack_slots":15,"window_slack_slots":0,"current_delay_slots":0,"priority":0.71,"urgency_score":15.0},{"mission_id":"M013","due_slot":257,"due_slack_slots":65,"window_slack_slots":35,"current_delay_slots":0,"priority":0.98,"urgency_score":82.5},{"mission_id":"M002","due_slot":186,"due_slack_slots":-6,"window_slack_slots":229,"current_delay_slots":0,"priority":0.37,"urgency_score":108.5}],"disturbance_summary":{"range_loss_pct":0.0,"pad_outage_active":false,"duration_volatility_level":"medium"},"frozen_summary":{"num_started_ops":0,"num_frozen_ops":0,"frozen_horizon_slots":12}}

Active (schedulable, not started/completed) missions: [M000, M001, M002, M004, M006, M007, M011, M012, M013, M015, M016, M018, M019, M022, M023, M024]

unlock_mission_ids must be a SUBSET of the active missions above, size 1–8, and must include root_cause_mission_id.

Output the JSON now (4 keys only: root_cause_mission_id, unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):
```

### 参考答案（启发式基线的决策，仅供对比）

```json
{
  "root_cause_mission_id": "M001",
  "unlock_mission_ids": [
    "M001"
  ],
  "secondary_root_cause_mission_id": "M013",
  "analysis_short": "heuristic: root=M001 conflicts=0 pad_util=1.00"
}
```

---

## 测试用例 3: Pad资源冲突-3任务竞争（合成）

| 属性 | 值 |
|------|-----|
| 场景种子 | N/A (合成数据) |
| 难度 | heavy |
| 当前时刻 | slot 120 (第 30.0 小时) |
| 活跃任务数 | 8 |
| 活跃任务 | M001, M003, M005, M007, M008, M010, M012, M015 |
| Pad利用率 | 95.00% |
| R3利用率 | 72.00% |
| 冲突数量 | 4 |
| 冲突簇数 | 1 |
| 紧迫任务 | M003, M005, M001 |
| 窗口损失率 | 5.0% |
| Pad故障 | 否 |

### System Prompt

```
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic summary, identify the root cause of scheduling conflicts and decide which missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–8 mission IDs. MUST include root_cause_mission_id. Only include missions that are currently ACTIVE (not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (soft):
- If conflicts are few / mild → small unlock set (1–2). Stability is king.
- If a bottleneck resource (pad/R3) is saturated → unlock the cluster around it (3–5 missions sharing that resource).
- If pad_outage is active → prefer not unlocking missions already frozen on pad.
- If range_loss_pct > 0.3 → unlock affected missions whose windows are shrinking.
- If no conflicts at all → unlock only 1 urgent mission for minor adjustment.
- When uncertain, be conservative: unlock only the conflict cluster center + 1 neighbor.
```

### User Prompt

```
TRCG diagnostic summary (JSON):
{"now_slot":120,"horizon_end_slot":216,"bottleneck_pressure":{"pad_util":0.95,"r3_util":0.72,"range_test_util":0.18},"top_conflicts":[{"a":"M003","b":"M005","resource":"R_pad","overlap_slots":14,"t_range":[152,166],"severity":23.8},{"a":"M003","b":"M007","resource":"R_pad","overlap_slots":8,"t_range":[158,166],"severity":14.4},{"a":"M005","b":"M007","resource":"R_pad","overlap_slots":6,"t_range":[160,166],"severity":9.6},{"a":"M001","b":"M010","resource":"R3","overlap_slots":4,"t_range":[140,144],"severity":5.2}],"conflict_clusters":[{"center_mission_id":"M003","members":["M003","M005","M007"],"score":38.2}],"urgent_missions":[{"mission_id":"M003","due_slot":200,"due_slack_slots":80,"window_slack_slots":32,"current_delay_slots":12,"priority":0.85,"urgency_score":72.0},{"mission_id":"M005","due_slot":230,"due_slack_slots":110,"window_slack_slots":40,"current_delay_slots":8,"priority":0.62,"urgency_score":114.0},{"mission_id":"M001","due_slot":180,"due_slack_slots":60,"window_slack_slots":22,"current_delay_slots":0,"priority":0.71,"urgency_score":71.0}],"disturbance_summary":{"range_loss_pct":0.05,"pad_outage_active":false,"duration_volatility_level":"low"},"frozen_summary":{"num_started_ops":4,"num_frozen_ops":6,"frozen_horizon_slots":12}}

Active (schedulable, not started/completed) missions: [M001, M003, M005, M007, M008, M010, M012, M015]

unlock_mission_ids must be a SUBSET of the active missions above, size 1–8, and must include root_cause_mission_id.

Output the JSON now (4 keys only: root_cause_mission_id, unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):
```

### 参考答案（启发式基线的决策，仅供对比）

```json
{
  "root_cause_mission_id": "M003",
  "unlock_mission_ids": [
    "M003",
    "M005",
    "M007"
  ],
  "secondary_root_cause_mission_id": "M005",
  "analysis_short": "heuristic: root=M003 conflicts=4 pad_util=0.95"
}
```

---

## 测试用例 4: Pad故障+R3冲突+窗口缩减（合成）

| 属性 | 值 |
|------|-----|
| 场景种子 | N/A (合成数据) |
| 难度 | heavy |
| 当前时刻 | slot 240 (第 60.0 小时) |
| 活跃任务数 | 10 |
| 活跃任务 | M002, M004, M006, M008, M009, M011, M014, M016, M018, M020 |
| Pad利用率 | 100.00% |
| R3利用率 | 88.00% |
| 冲突数量 | 6 |
| 冲突簇数 | 2 |
| 紧迫任务 | M004, M011, M006 |
| 窗口损失率 | 35.0% |
| Pad故障 | 是 |

### System Prompt

```
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic summary, identify the root cause of scheduling conflicts and decide which missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–8 mission IDs. MUST include root_cause_mission_id. Only include missions that are currently ACTIVE (not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (soft):
- If conflicts are few / mild → small unlock set (1–2). Stability is king.
- If a bottleneck resource (pad/R3) is saturated → unlock the cluster around it (3–5 missions sharing that resource).
- If pad_outage is active → prefer not unlocking missions already frozen on pad.
- If range_loss_pct > 0.3 → unlock affected missions whose windows are shrinking.
- If no conflicts at all → unlock only 1 urgent mission for minor adjustment.
- When uncertain, be conservative: unlock only the conflict cluster center + 1 neighbor.
```

### User Prompt

```
TRCG diagnostic summary (JSON):
{"now_slot":240,"horizon_end_slot":336,"bottleneck_pressure":{"pad_util":1.0,"r3_util":0.88,"range_test_util":0.35},"top_conflicts":[{"a":"M004","b":"M006","resource":"R_pad","overlap_slots":18,"t_range":[260,278],"severity":32.4},{"a":"M004","b":"M009","resource":"R_pad","overlap_slots":10,"t_range":[268,278],"severity":17.0},{"a":"M006","b":"M009","resource":"R_pad","overlap_slots":7,"t_range":[271,278],"severity":11.9},{"a":"M011","b":"M016","resource":"R3","overlap_slots":12,"t_range":[280,292],"severity":22.8},{"a":"M008","b":"M011","resource":"R3","overlap_slots":5,"t_range":[285,290],"severity":8.5},{"a":"M002","b":"M014","resource":"R_range_test","overlap_slots":3,"t_range":[300,303],"severity":4.2}],"conflict_clusters":[{"center_mission_id":"M004","members":["M004","M006","M009"],"score":49.4},{"center_mission_id":"M011","members":["M011","M016","M008"],"score":31.3}],"urgent_missions":[{"mission_id":"M004","due_slot":280,"due_slack_slots":40,"window_slack_slots":18,"current_delay_slots":20,"priority":0.92,"urgency_score":9.0},{"mission_id":"M011","due_slot":310,"due_slack_slots":70,"window_slack_slots":30,"current_delay_slots":15,"priority":0.78,"urgency_score":55.0},{"mission_id":"M006","due_slot":325,"due_slack_slots":85,"window_slack_slots":40,"current_delay_slots":10,"priority":0.65,"urgency_score":85.0}],"disturbance_summary":{"range_loss_pct":0.35,"pad_outage_active":true,"duration_volatility_level":"high"},"frozen_summary":{"num_started_ops":8,"num_frozen_ops":12,"frozen_horizon_slots":12}}

Active (schedulable, not started/completed) missions: [M002, M004, M006, M008, M009, M011, M014, M016, M018, M020]

unlock_mission_ids must be a SUBSET of the active missions above, size 1–8, and must include root_cause_mission_id.

Output the JSON now (4 keys only: root_cause_mission_id, unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):
```

### 参考答案（启发式基线的决策，仅供对比）

```json
{
  "root_cause_mission_id": "M004",
  "unlock_mission_ids": [
    "M004",
    "M006",
    "M009"
  ],
  "secondary_root_cause_mission_id": "M006",
  "analysis_short": "heuristic: root=M004 conflicts=6 pad_util=1.00"
}
```

---

## 测试用例 5: 大规模冲突簇-5任务连锁冲突（合成）

| 属性 | 值 |
|------|-----|
| 场景种子 | N/A (合成数据) |
| 难度 | heavy |
| 当前时刻 | slot 360 (第 90.0 小时) |
| 活跃任务数 | 12 |
| 活跃任务 | M001, M002, M005, M006, M008, M010, M013, M015, M017, M019, M021, M023 |
| Pad利用率 | 100.00% |
| R3利用率 | 95.00% |
| 冲突数量 | 8 |
| 冲突簇数 | 2 |
| 紧迫任务 | M005, M008, M010 |
| 窗口损失率 | 12.0% |
| Pad故障 | 否 |

### System Prompt

```
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic summary, identify the root cause of scheduling conflicts and decide which missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–8 mission IDs. MUST include root_cause_mission_id. Only include missions that are currently ACTIVE (not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (soft):
- If conflicts are few / mild → small unlock set (1–2). Stability is king.
- If a bottleneck resource (pad/R3) is saturated → unlock the cluster around it (3–5 missions sharing that resource).
- If pad_outage is active → prefer not unlocking missions already frozen on pad.
- If range_loss_pct > 0.3 → unlock affected missions whose windows are shrinking.
- If no conflicts at all → unlock only 1 urgent mission for minor adjustment.
- When uncertain, be conservative: unlock only the conflict cluster center + 1 neighbor.
```

### User Prompt

```
TRCG diagnostic summary (JSON):
{"now_slot":360,"horizon_end_slot":456,"bottleneck_pressure":{"pad_util":1.0,"r3_util":0.95,"range_test_util":0.42},"top_conflicts":[{"a":"M005","b":"M008","resource":"R_pad","overlap_slots":22,"t_range":[380,402],"severity":39.6},{"a":"M005","b":"M010","resource":"R_pad","overlap_slots":16,"t_range":[386,402],"severity":28.8},{"a":"M008","b":"M010","resource":"R_pad","overlap_slots":11,"t_range":[391,402],"severity":19.8},{"a":"M005","b":"M013","resource":"R3","overlap_slots":9,"t_range":[375,384],"severity":16.2},{"a":"M008","b":"M013","resource":"R3","overlap_slots":7,"t_range":[378,385],"severity":12.6},{"a":"M010","b":"M015","resource":"R_pad","overlap_slots":8,"t_range":[395,403],"severity":12.0},{"a":"M002","b":"M019","resource":"R3","overlap_slots":5,"t_range":[410,415],"severity":7.5},{"a":"M001","b":"M006","resource":"R_range_test","overlap_slots":4,"t_range":[420,424],"severity":5.2}],"conflict_clusters":[{"center_mission_id":"M005","members":["M005","M008","M010","M013","M015"],"score":84.6},{"center_mission_id":"M002","members":["M002","M019"],"score":7.5}],"urgent_missions":[{"mission_id":"M005","due_slot":410,"due_slack_slots":50,"window_slack_slots":14,"current_delay_slots":25,"priority":0.95,"urgency_score":7.0},{"mission_id":"M008","due_slot":430,"due_slack_slots":70,"window_slack_slots":20,"current_delay_slots":18,"priority":0.88,"urgency_score":44.0},{"mission_id":"M010","due_slot":450,"due_slack_slots":90,"window_slack_slots":28,"current_delay_slots":10,"priority":0.72,"urgency_score":84.0}],"disturbance_summary":{"range_loss_pct":0.12,"pad_outage_active":false,"duration_volatility_level":"medium"},"frozen_summary":{"num_started_ops":12,"num_frozen_ops":16,"frozen_horizon_slots":12}}

Active (schedulable, not started/completed) missions: [M001, M002, M005, M006, M008, M010, M013, M015, M017, M019, M021, M023]

unlock_mission_ids must be a SUBSET of the active missions above, size 1–8, and must include root_cause_mission_id.

Output the JSON now (4 keys only: root_cause_mission_id, unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):
```

### 参考答案（启发式基线的决策，仅供对比）

```json
{
  "root_cause_mission_id": "M005",
  "unlock_mission_ids": [
    "M005",
    "M008",
    "M010",
    "M013"
  ],
  "secondary_root_cause_mission_id": "M008",
  "analysis_short": "heuristic: root=M005 conflicts=8 pad_util=1.00"
}
```

---

## 附录：模型评测打分表

| 评估项 | 权重 | 模型A | 模型B | 模型C | 模型D |
|--------|------|-------|-------|-------|-------|
| 纯JSON输出（无多余文字） | 25% | | | | |
| 4字段完整且类型正确 | 20% | | | | |
| root_cause∈unlock∈active | 20% | | | | |
| unlock集合大小合理（1-8） | 10% | | | | |
| 根因识别准确性 | 15% | | | | |
| 响应延迟（秒） | 10% | | | | |
| **总分** | **100%** | | | | |

### 打分标准
- **纯JSON**: 输出仅包含JSON对象得满分；带markdown/解释扣50%；完全不合格0分
- **字段完整**: 4个required字段齐全且类型正确得满分；缺字段每个扣25%
- **业务规则**: root∈unlock∈active全部满足得满分；每违反一条扣33%
- **集合大小**: 冲突多时3-5个合理；冲突少时1-2个合理；过大过小酌情扣分
- **根因准确**: 与冲突簇中心一致得满分；选了次优但合理的扣20%；完全不相关0分
- **响应延迟**: <5秒满分；5-15秒扣20%；15-30秒扣50%；>30秒扣80%
