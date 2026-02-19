"""
LLM TRCG 修复策略 — Prompt 模板、输出 Schema、校验器、启发式回退、求解回退链、日志

本模块定义：
1. RepairDecision  dataclass   — LLM 输出的 6 字段决策
2. REPAIR_DECISION_SCHEMA      — 严格 JSON schema（枚举 + 长度限制）
3. REPAIR_SYSTEM_PROMPT        — system prompt（角色 + 硬约束）
4. build_repair_user_prompt()  — user prompt 模板（TRCGSummary 注入）
5. validate_repair_decision()  — 多级校验（schema → 存在性 → 业务规则）
6. heuristic_repair_decision() — 确定性启发式回退（LLM 不可用时）
7. solve_with_fallback_chain() — 3 次重试 + 最终全局回退求解链路
8. RepairStepLog / build_repair_step_log() — 每个 rolling 步骤的结构化日志

后续子 Prompt 将在此文件中追加 TRCGRepairPolicy(BasePolicy) 实际策略类。
"""

import json
import re
import logging
import time as _time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Set

# 复用已有 JSON 三层抽取
try:
    from llm_client import extract_json_from_text
    _HAS_LLM_EXTRACT = True
except ImportError:
    _HAS_LLM_EXTRACT = False


# ============================================================================
# RepairDecision 数据结构
# ============================================================================

# freeze_horizon_hours → slots 转换表 (slot=15min)
FREEZE_HOURS_TO_SLOTS: Dict[int, int] = {
    0: 0,
    4: 16,
    8: 32,
    16: 64,
    24: 96,
}

ALLOWED_FREEZE_HOURS: List[int] = [0, 4, 8, 16, 24]
ALLOWED_EPSILON: List[float] = [0.0, 0.02, 0.05, 0.10]


@dataclass
class RepairDecision:
    """
    TRCG 修复决策（LLM 输出 4 字段）。

    LLM / 启发式决策字段：
    - root_cause_mission_id: 根因 mission（1 个）
    - unlock_mission_ids:    解锁集（1~8，必须包含 root）
    - analysis_short:        根因简述（≤120 字符，仅日志用）
    - secondary_root_cause_mission_id: 次根因（回退扩大 unlock 用，可为 None）

    freeze_horizon_hours / epsilon_solver 不由 LLM 输出，也不用规则引擎。
    固定为 grid search 最优值（与 fixed_tuned 一致），从 config 中读取。
    保留这两个字段仅供日志记录和回退链降级使用。
    """
    root_cause_mission_id: str
    unlock_mission_ids: List[str]
    analysis_short: str = ""
    secondary_root_cause_mission_id: Optional[str] = None
    # 以下不由 LLM 输出，保留供回退链使用
    freeze_horizon_hours: int = 24
    epsilon_solver: float = 0.10

    # ---------- 派生属性 ----------

    @property
    def freeze_horizon_slots(self) -> int:
        return FREEZE_HOURS_TO_SLOTS.get(self.freeze_horizon_hours, 0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# JSON Schema（供文档 & 运行时校验双重用途）
# ============================================================================

REPAIR_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "root_cause_mission_id",
        "unlock_mission_ids",
        "secondary_root_cause_mission_id",
        "analysis_short",
    ],
    "properties": {
        "root_cause_mission_id": {
            "type": "string",
            "pattern": r"^M\d{3}$",
            "description": "根因 mission（1 个）",
        },
        "unlock_mission_ids": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^M\d{3}$"},
            "minItems": 1,
            "maxItems": 8,
            "description": "解锁集（1~5，必须包含 root_cause；保持最小化以减少 drift）",
        },
        "secondary_root_cause_mission_id": {
            "type": ["string", "null"],
            "description": "次要根因（可为 null）",
        },
        "analysis_short": {
            "type": "string",
            "maxLength": 120,
            "description": "根因简述（≤120 字符）",
        },
    },
}


# ============================================================================
# System Prompt
# ============================================================================

REPAIR_SYSTEM_PROMPT = """\
You are an expert rocket-launch scheduling advisor.

Your ONLY job: given a Temporal-Resource Conflict Graph (TRCG) diagnostic \
summary, identify the root cause of scheduling conflicts and decide the \
MINIMAL set of missions to unlock for local repair (anchor fix-and-optimize).

You do NOT choose solver parameters (freeze, epsilon). Those are set by \
a deterministic rule engine. You ONLY choose the repair anchor set.

HARD RULES — violating ANY rule makes your output INVALID:
1. Output ONLY the JSON object. No explanation, no markdown, no code fence.
2. The JSON must contain EXACTLY these 4 keys:
   root_cause_mission_id, unlock_mission_ids,
   secondary_root_cause_mission_id, analysis_short.
3. root_cause_mission_id: exactly ONE mission ID (format "M###") that is \
the primary source of the scheduling conflict.
4. unlock_mission_ids: array of 1–5 mission IDs. MUST include \
root_cause_mission_id. Only include missions that are currently ACTIVE \
(not started, not completed).
5. secondary_root_cause_mission_id: ONE mission ID or null. Should be the \
next most impactful mission in the conflict cluster.
6. analysis_short: ≤120 characters, concise root-cause statement for logging.

STRATEGY GUIDANCE (primary goal: MINIMIZE schedule drift / maximize stability):
- SCHEDULE STABILITY is the primary objective. Keep unlock sets as SMALL as \
possible. Missions NOT in unlock_mission_ids will be anchored to their \
current planned positions, producing ZERO drift for those missions.
- Identify the EXACT root-cause mission(s) creating the conflict. Unlock \
ONLY the minimal cluster needed to resolve it (usually 1–3 missions).
- If one mission caused the conflict: unlock only that mission (1 in set).
- If two missions conflict directly: unlock both (2 in set).
- If a bottleneck resource (pad/R3) forces a cluster: unlock the MINIMUM \
subset needed (2–3 missions, NOT the entire cluster).
- If pad_outage is active → unlock ONLY the 1–2 missions forced to reschedule.
- If range_loss_pct > 0.3 → unlock the 1–2 most affected missions only.
- If no conflicts → unlock exactly 1 urgent mission for minor adjustment.
- ALWAYS prefer fewer unlocked missions. Every additional unlocked mission \
increases schedule instability. Be precise and conservative.\
"""


# ============================================================================
# User Prompt 模板
# ============================================================================

def build_repair_user_prompt(
    trcg_dict: Dict[str, Any],
    active_mission_ids: List[str],
) -> str:
    """
    构建 user prompt，将 TRCGSummary + 候选 mission 列表注入模板。

    Parameters
    ----------
    trcg_dict : TRCGSummary.to_dict() 的输出。
    active_mission_ids : 当前可调度且未开始/未完成的 mission ID 列表。
    """
    trcg_json = json.dumps(trcg_dict, ensure_ascii=False, separators=(',', ':'))
    ids_str = ', '.join(active_mission_ids)

    return (
        f"TRCG diagnostic summary (JSON):\n{trcg_json}\n\n"
        f"Active (schedulable, not started/completed) missions: [{ids_str}]\n\n"
        "unlock_mission_ids must be a SUBSET of the active missions above, "
        "size 1–8, and must include root_cause_mission_id.\n\n"
        "Output the JSON now (4 keys only: root_cause_mission_id, "
        "unlock_mission_ids, secondary_root_cause_mission_id, analysis_short):"
    )


# ============================================================================
# 三层 JSON 抽取（内部 fallback，与 llm_client 解耦）
# ============================================================================

def _fallback_extract_json(text: str) -> Tuple[Optional[str], str]:
    """llm_client 不可用时的本地三层 JSON 抽取。"""
    if not text:
        return None, "failed"
    text = text.strip()

    # 1. direct
    if text.startswith('{') and text.endswith('}'):
        try:
            json.loads(text)
            return text, "direct"
        except json.JSONDecodeError:
            pass

    # 2. code fence
    for pat in (r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```'):
        m = re.search(pat, text)
        if m:
            c = m.group(1).strip()
            if c.startswith('{'):
                try:
                    json.loads(c)
                    return c, "code_fence"
                except json.JSONDecodeError:
                    pass

    # 3. brace search
    start = text.find('{')
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"' and not esc:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate, "brace_search"
                    except json.JSONDecodeError:
                        pass
                    break
    return None, "failed"


def _extract_json(text: str) -> Tuple[Optional[str], str]:
    """统一抽取入口：优先用 llm_client，否则本地 fallback。"""
    if _HAS_LLM_EXTRACT:
        return extract_json_from_text(text)
    return _fallback_extract_json(text)


# ============================================================================
# 校验器
# ============================================================================

@dataclass
class RepairValidationResult:
    """校验结果（与 policy_llm_meta.ValidationResult 对称但独立）。"""
    is_valid: bool
    decision: Optional[RepairDecision] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_method: str = "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'decision': self.decision.to_dict() if self.decision else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'extraction_method': self.extraction_method,
        }


def validate_repair_decision(
    raw_text: str,
    active_mission_ids: Set[str],
    started_mission_ids: Set[str],
    completed_mission_ids: Set[str],
) -> RepairValidationResult:
    """
    对 LLM 原始输出做多级校验，返回 RepairValidationResult。

    校验层级（按顺序，任一层失败 → is_valid=False）：
      L1. JSON 抽取（三层：direct / code_fence / brace_search）
      L2. 必需字段存在性（4 个：root_cause, unlock, secondary, analysis）
      L3. 类型校验（string / array / string|null / string）
      L4. 业务规则：
          - root_cause_mission_id ∈ active_mission_ids
          - root_cause_mission_id ∈ unlock_mission_ids
          - unlock_mission_ids ⊆ active_mission_ids
          - len(unlock_mission_ids) ∈ [1, 8]
          - 不能包含已开始 / 已完成 mission
          - secondary_root_cause（若非 null）∈ active_mission_ids
          - analysis_short ≤ 120 字符

    注：freeze_horizon_hours 和 epsilon_solver 不由 LLM 输出，
    固定为 grid search 最优值（与 fixed_tuned 一致）。
    若 LLM 输出包含这两个字段会被静默忽略（向后兼容）。
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ---- L1: JSON 抽取 ----
    json_str, extraction_method = _extract_json(raw_text)
    if json_str is None:
        return RepairValidationResult(
            is_valid=False,
            errors=[f"JSON extraction failed from: {raw_text[:120]}..."],
            extraction_method=extraction_method,
        )

    try:
        data: Dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return RepairValidationResult(
            is_valid=False,
            errors=[f"JSON parse error: {exc}"],
            extraction_method=extraction_method,
        )

    if not isinstance(data, dict):
        return RepairValidationResult(
            is_valid=False,
            errors=["JSON root must be an object"],
            extraction_method=extraction_method,
        )

    # ---- L2: 必需字段 ----
    required = REPAIR_DECISION_SCHEMA["required"]
    for key in required:
        if key not in data:
            errors.append(f"missing required field: {key}")
    if errors:
        return RepairValidationResult(
            is_valid=False, errors=errors, extraction_method=extraction_method,
        )

    # ---- L3: 类型校验 ----
    # 注：freeze_horizon_hours / epsilon_solver 不再由 LLM 输出，若存在则忽略
    if "freeze_horizon_hours" in data:
        warnings.append("freeze_horizon_hours present in LLM output — ignored (rule-based)")
    if "epsilon_solver" in data:
        warnings.append("epsilon_solver present in LLM output — ignored (rule-based)")

    # root_cause_mission_id
    root = data["root_cause_mission_id"]
    if not isinstance(root, str):
        errors.append(f"root_cause_mission_id must be str, got {type(root).__name__}")

    # unlock_mission_ids
    unlock = data["unlock_mission_ids"]
    if not isinstance(unlock, list):
        errors.append(f"unlock_mission_ids must be array, got {type(unlock).__name__}")
    elif not all(isinstance(x, str) for x in unlock):
        errors.append("unlock_mission_ids items must all be str")

    # secondary_root_cause_mission_id (str | null)
    secondary = data["secondary_root_cause_mission_id"]
    if secondary is not None and not isinstance(secondary, str):
        errors.append(f"secondary_root_cause_mission_id must be str or null")

    # analysis_short
    analysis = data["analysis_short"]
    if not isinstance(analysis, str):
        errors.append(f"analysis_short must be str, got {type(analysis).__name__}")
    elif len(analysis) > 120:
        warnings.append(f"analysis_short truncated from {len(analysis)} to 120 chars")
        analysis = analysis[:120]
        data["analysis_short"] = analysis

    if errors:
        return RepairValidationResult(
            is_valid=False, errors=errors, warnings=warnings,
            extraction_method=extraction_method,
        )

    # ---- L4: 业务规则 ----

    root = str(data["root_cause_mission_id"])
    unlock = [str(x) for x in data["unlock_mission_ids"]]
    secondary = data["secondary_root_cause_mission_id"]
    if secondary is not None:
        secondary = str(secondary)

    # 4a. unlock 大小
    if len(unlock) < 1 or len(unlock) > 8:
        errors.append(f"unlock_mission_ids length={len(unlock)}, must be 1–8")

    # 4b. root ∈ unlock
    if root not in unlock:
        errors.append(f"root_cause {root} not in unlock_mission_ids")

    # 4c. root ∈ active
    if root not in active_mission_ids:
        errors.append(f"root_cause {root} not in active missions")

    # 4d. unlock ⊆ active
    invalid_unlock = [m for m in unlock if m not in active_mission_ids]
    if invalid_unlock:
        errors.append(f"unlock contains non-active missions: {invalid_unlock}")

    # 4e. unlock 不得包含已开始/已完成的 mission
    started_in_unlock = [m for m in unlock if m in started_mission_ids]
    if started_in_unlock:
        errors.append(f"unlock contains started missions: {started_in_unlock}")
    completed_in_unlock = [m for m in unlock if m in completed_mission_ids]
    if completed_in_unlock:
        errors.append(f"unlock contains completed missions: {completed_in_unlock}")

    # 4f. secondary（若非 null）∈ active & 不在 started/completed
    if secondary is not None:
        if secondary not in active_mission_ids:
            warnings.append(f"secondary_root {secondary} not active, will be ignored")
            secondary = None
            data["secondary_root_cause_mission_id"] = None
        elif secondary in started_mission_ids or secondary in completed_mission_ids:
            warnings.append(f"secondary_root {secondary} already started/completed, ignored")
            secondary = None
            data["secondary_root_cause_mission_id"] = None

    # 4g. 去重 unlock
    deduped = list(dict.fromkeys(unlock))
    if len(deduped) != len(unlock):
        warnings.append("unlock_mission_ids had duplicates, deduplicated")
        unlock = deduped
        data["unlock_mission_ids"] = unlock

    if errors:
        return RepairValidationResult(
            is_valid=False, errors=errors, warnings=warnings,
            extraction_method=extraction_method,
        )

    # ---- 构造 RepairDecision ----
    # freeze/epsilon 使用默认值，与 fixed_tuned 基线一致，由 config 控制
    decision = RepairDecision(
        root_cause_mission_id=root,
        unlock_mission_ids=unlock,
        analysis_short=str(data["analysis_short"]),
        secondary_root_cause_mission_id=secondary,
    )

    return RepairValidationResult(
        is_valid=True,
        decision=decision,
        warnings=warnings,
        extraction_method=extraction_method,
    )


# ============================================================================
# 方案一回退：确定性启发式 root cause + unlock set
# ============================================================================

_HEURISTIC_DEFAULT_K = 3   # 默认解锁 3 个任务（糟定性优先）
_HEURISTIC_HEAVY_K = 6    # 重压力场景下最大解锁 6 个
_PAD_PRESSURE_HEAVY_THRESH = 0.80  # bottleneck 压力阈值
_URGENT_HEAVY_THRESH = 3   # urgent 数量阈值


def heuristic_repair_decision(
    trcg_dict: Dict[str, Any],
    active_mission_ids: Set[str],
    started_mission_ids: Set[str],
    completed_mission_ids: Set[str],
    fallback_reason: str = "heuristic",
) -> RepairDecision:
    """
    确定性启发式生成 RepairDecision（方案一回退）。

    当 LLM 不可用 / 输出非法 / 超时 / 解码失败时调用。
    保证：输出永远合法（满足 validate_repair_decision 的全部 L1–L4 规则）。

    算法
    ----
    1. **root 选择**
       a) 若 top_conflicts 非空：对所有冲突中出现的 mission 按加权度数
          (Σ severity) 降序排列；tie-break 按 mission_id 字典序升序。
          取度数最大且仍 active 的 mission 为 root。
       b) 否则：取 urgent_missions 中 urgency_score 最小且 active 的。
       c) 否则：取 active_mission_ids 字典序最小的。

    2. **unlock_set 构造**
       - K = 3（默认）；若 pad_pressure > 0.80 或 urgent 数 ≥ 3 → K = 7。
       - unlock = {root}。
       - 从 top_conflicts 中找 root 的邻居，按 severity 降序逐个加入。
       - 若 unlock 仍只有 root 且 conflict_clusters 非空，从 cluster 补充。

    3. **secondary_root 选择**
       unlock 中除 root 外度数最大的 mission。

    注：freeze_horizon_hours 和 epsilon_solver 不在此设置，
    固定为 grid search 最优值（与 fixed_tuned 一致），从 config 读取。

    Parameters
    ----------
    trcg_dict : TRCGSummary.to_dict() 的输出。
    active_mission_ids : 当前可调度且未开始/未完成的 mission ID 集合。
    started_mission_ids : 已开始的 mission ID 集合。
    completed_mission_ids : 已完成的 mission ID 集合。
    fallback_reason : 写入 analysis_short 的前缀。

    Returns
    -------
    RepairDecision — 保证合法（freeze/epsilon 为默认值，后续覆盖）。
    """
    conflicts: List[Dict[str, Any]] = trcg_dict.get('top_conflicts', [])
    clusters: List[Dict[str, Any]] = trcg_dict.get('conflict_clusters', [])
    urgents: List[Dict[str, Any]] = trcg_dict.get('urgent_missions', [])
    pressure = trcg_dict.get('bottleneck_pressure', {})

    pad_pressure = pressure.get('pad_util', 0.0)
    num_urgent = len(urgents)

    # 可选的 mission 集合（active 且非 started/completed）
    eligible = sorted(
        active_mission_ids - started_mission_ids - completed_mission_ids
    )
    if not eligible:
        # 极端兜底：如果连 eligible 都为空，用 active 的第一个
        eligible = sorted(active_mission_ids) if active_mission_ids else ["M000"]

    # ------------------------------------------------------------------
    # 1. root 选择
    # ------------------------------------------------------------------
    degree: Dict[str, float] = {}      # mission → Σ severity
    neighbors: Dict[str, List[Tuple[float, str]]] = {}  # mission → [(severity, neighbor)]

    for c in conflicts:
        a, b = c.get('a', ''), c.get('b', '')
        sev = float(c.get('severity', 0))
        for m in (a, b):
            degree[m] = degree.get(m, 0.0) + sev
        neighbors.setdefault(a, []).append((sev, b))
        neighbors.setdefault(b, []).append((sev, a))

    root: Optional[str] = None

    if degree:
        # 按 (-degree, mission_id) 排序 → 确定性 tie-break
        candidates = sorted(degree.keys(), key=lambda m: (-degree[m], m))
        for m in candidates:
            if m in eligible:
                root = m
                break

    if root is None and urgents:
        # 按 urgency_score 升序（越小越紧迫），tie-break mission_id 升序
        for u in sorted(urgents, key=lambda u: (u.get('urgency_score', 9999),
                                                 u.get('mission_id', ''))):
            mid = u.get('mission_id', '')
            if mid in eligible:
                root = mid
                break

    if root is None:
        root = eligible[0]

    # ------------------------------------------------------------------
    # 2. unlock_set
    # ------------------------------------------------------------------
    is_heavy = (pad_pressure > _PAD_PRESSURE_HEAVY_THRESH
                or num_urgent >= _URGENT_HEAVY_THRESH)
    K = _HEURISTIC_HEAVY_K if is_heavy else _HEURISTIC_DEFAULT_K

    unlock_set: List[str] = [root]
    seen: Set[str] = {root}

    # 从 root 的冲突邻居按 severity 降序、tie-break mission_id 升序取
    root_neighbors = neighbors.get(root, [])
    # 排序：(-severity, mission_id)
    root_neighbors_sorted = sorted(root_neighbors, key=lambda t: (-t[0], t[1]))

    for _sev, nbr in root_neighbors_sorted:
        if len(unlock_set) >= K:
            break
        if nbr in seen or nbr not in eligible:
            continue
        unlock_set.append(nbr)
        seen.add(nbr)

    # 若 unlock 仍只有 root，尝试从 clusters 补充
    if len(unlock_set) == 1 and clusters:
        for cl in clusters:
            for member in cl.get('members', []):
                if len(unlock_set) >= K:
                    break
                if member in seen or member not in eligible:
                    continue
                unlock_set.append(member)
                seen.add(member)
            if len(unlock_set) >= K:
                break

    # ------------------------------------------------------------------
    # 3. secondary_root
    # ------------------------------------------------------------------
    secondary: Optional[str] = None
    if len(unlock_set) > 1:
        # unlock 中除 root 外度数最大的，tie-break mission_id 升序
        others = [m for m in unlock_set if m != root]
        others.sort(key=lambda m: (-degree.get(m, 0.0), m))
        secondary = others[0]

    # ------------------------------------------------------------------
    # 4. analysis_short
    # ------------------------------------------------------------------
    n_conflicts = len(conflicts)
    analysis = (
        f"[{fallback_reason}] root={root} K={len(unlock_set)} "
        f"conflicts={n_conflicts} pad={pad_pressure:.2f} urgent={num_urgent}"
    )
    if len(analysis) > 120:
        analysis = analysis[:120]

    # freeze/epsilon 使用默认值，与 fixed_tuned 基线一致
    return RepairDecision(
        root_cause_mission_id=root,
        unlock_mission_ids=unlock_set,
        analysis_short=analysis,
        secondary_root_cause_mission_id=secondary,
    )


# ============================================================================
# 求解回退链路：最多 3 次重试 + 最终全局回退
# ============================================================================

logger = logging.getLogger("repair_fallback")

# freeze 降级序列：当前值 → 下一个更低值
_FREEZE_STEP_DOWN: Dict[int, int] = {24: 16, 16: 8, 8: 4, 4: 0, 0: 0}

# epsilon 升级序列：当前值 → 下一个更宽松值
_EPSILON_STEP_UP: Dict[float, float] = {0.0: 0.02, 0.02: 0.05, 0.05: 0.10, 0.10: 0.10}


@dataclass
class FallbackAttempt:
    """单次重试的日志记录。"""
    attempt_name: str           # "initial" / "attempt1_expand_unlock" / ...
    unlock_ids: List[str]
    freeze_hours: int
    epsilon: float
    use_anchor: bool
    solver_status: str          # "OPTIMAL" / "FEASIBLE" / "INFEASIBLE" / ...
    solve_time_ms: int = 0
    anchor_applied: int = 0
    anchor_skipped: int = 0
    anchor_applied_missions: List[str] = field(default_factory=list)
    anchor_applied_vars: Dict[str, int] = field(default_factory=dict)
    anchor_skip_reason: str = ""


@dataclass
class FallbackChainResult:
    """
    回退链路的完整输出。

    Attributes
    ----------
    solver_result : SolverResult — 最终采用的求解结果
    success : bool — 是否成功求解（OPTIMAL/FEASIBLE）
    attempts : List[FallbackAttempt] — 所有尝试记录（含 initial）
    final_attempt_name : str — 最终采用的尝试名称
    total_solver_calls : int — solver 调用总次数
    """
    solver_result: Any                  # SolverResult
    success: bool
    attempts: List[FallbackAttempt] = field(default_factory=list)
    final_attempt_name: str = "initial"
    total_solver_calls: int = 0


def _expand_unlock_from_conflicts(
    current_unlock: List[str],
    trcg_dict: Dict[str, Any],
    eligible_ids: Set[str],
    expand_count: int = 2,
) -> List[str]:
    """
    从 TRCG conflict edges 中按 severity 次高顺序扩大 unlock_set。

    策略：
    1. 收集所有冲突边中涉及的 mission（但不在当前 unlock 中）
    2. 按该 mission 的 Σ severity 降序排列
    3. 取前 expand_count 个加入
    4. 不超过 5 个总上限
    """
    conflicts: List[Dict[str, Any]] = trcg_dict.get('top_conflicts', [])
    current_set = set(current_unlock)

    # 计算每个 mission 的 severity 总和
    degree: Dict[str, float] = {}
    for c in conflicts:
        for m in (c.get('a', ''), c.get('b', '')):
            if m and m not in current_set and m in eligible_ids:
                degree[m] = degree.get(m, 0.0) + float(c.get('severity', 0))

    # 按 (-degree, mission_id) 排序
    candidates = sorted(degree.keys(), key=lambda m: (-degree[m], m))

    expanded = list(current_unlock)
    for m in candidates:
        if len(expanded) >= 8:   # unlock_mission_ids 上限
            break
        if m not in current_set:
            expanded.append(m)
            current_set.add(m)
            expand_count -= 1
            if expand_count <= 0:
                break

    return expanded


def solve_with_fallback_chain(
    decision: RepairDecision,
    trcg_dict: Dict[str, Any],
    missions: List,               # List[Mission]
    resources: List,              # List[Resource]
    horizon: int,
    prev_plan: Any,               # Optional[PlanV2_1]
    frozen_ops: Dict,             # Dict[str, OpAssignment]
    now: int,
    eligible_ids: Set[str],
    solver_config_base: Any,      # SolverConfigV2_1 — 基础配置模板
    compute_frozen_ops_fn=None,   # 可选：重新计算 frozen_ops 的函数
    current_plan_for_refreeze: Any = None,  # 降 freeze 时用的 current_plan
    started_ops: Optional[Set[str]] = None,
    completed_ops: Optional[Set[str]] = None,
) -> FallbackChainResult:
    """
    带回退链路的求解。最多 3+1 次 solver 调用（initial + 3 attempts），
    若全部失败则最终回退到全局重排。

    流程
    ----
    0. initial：按 decision 的参数求解（带锚点）
    1. attempt1_expand_unlock：扩大 unlock_set +2
    2. attempt2_reduce_freeze：降低 freeze_horizon 一档
    3. attempt3_relax_epsilon：放宽 epsilon 一档
    4. final_global_replan：保留原始 freeze_horizon, eps=0.10, 全 unlock, 无锚点
       （最坏情况=fixed_tuned，不会产生比固定基线更差的灾难性 drift 事件）

    Parameters
    ----------
    decision : RepairDecision
    trcg_dict : TRCGSummary.to_dict()
    missions, resources, horizon, prev_plan, frozen_ops, now : solver 参数
    eligible_ids : 可解锁的 mission 集合
    solver_config_base : SolverConfigV2_1 基础模板
    compute_frozen_ops_fn : Optional callable(current_plan, now, freeze_slots,
                            started_ops, completed_ops) → Dict[str, OpAssignment]
    current_plan_for_refreeze : 重新计算 frozen_ops 时使用的 plan
    started_ops, completed_ops : 已开始/完成 ops 集合

    Returns
    -------
    FallbackChainResult
    """
    # 延迟导入 solver（避免循环依赖）
    from solver_cpsat import (
        solve_v2_1, SolverConfigV2_1, SolveStatus,
        compute_frozen_ops as _default_compute_frozen_ops,
    )
    if compute_frozen_ops_fn is None:
        compute_frozen_ops_fn = _default_compute_frozen_ops
    if started_ops is None:
        started_ops = set()
    if completed_ops is None:
        completed_ops = set()

    attempts: List[FallbackAttempt] = []
    total_calls = 0

    def _is_success(status) -> bool:
        return status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)

    def _make_config(eps: float) -> SolverConfigV2_1:
        """基于 solver_config_base 派生新 config，仅修改 epsilon。"""
        return SolverConfigV2_1(
            horizon_slots=solver_config_base.horizon_slots,
            w_delay=solver_config_base.w_delay,
            w_shift=solver_config_base.w_shift,
            w_switch=solver_config_base.w_switch,
            time_limit_seconds=solver_config_base.time_limit_seconds,
            num_workers=solver_config_base.num_workers,
            op5_max_wait_slots=solver_config_base.op5_max_wait_slots,
            use_two_stage=solver_config_base.use_two_stage,
            epsilon_solver=eps,
            kappa_win=solver_config_base.kappa_win,
            kappa_seq=solver_config_base.kappa_seq,
            stage1_time_ratio=solver_config_base.stage1_time_ratio,
        )

    def _solve_once(
        attempt_name: str,
        unlock_ids: List[str],
        freeze_hours: int,
        epsilon: float,
        use_anchor: bool,
        frozen_ops_override: Optional[Dict] = None,
    ) -> Tuple[Any, FallbackAttempt]:
        """执行一次求解并记录日志。"""
        nonlocal total_calls
        total_calls += 1

        cfg = _make_config(epsilon)
        fops = frozen_ops_override if frozen_ops_override is not None else frozen_ops

        unlock_set = set(unlock_ids) if use_anchor else None

        result = solve_v2_1(
            missions=missions,
            resources=resources,
            horizon=horizon,
            prev_plan=prev_plan,
            frozen_ops=fops,
            config=cfg,
            unlock_mission_ids=unlock_set,
            now=now,
        )

        attempt = FallbackAttempt(
            attempt_name=attempt_name,
            unlock_ids=list(unlock_ids),
            freeze_hours=freeze_hours,
            epsilon=epsilon,
            use_anchor=use_anchor,
            solver_status=result.status.value,
            solve_time_ms=result.solve_time_ms,
            anchor_applied=result.anchor_fix_applied_count,
            anchor_skipped=result.anchor_fix_skipped_count,
            anchor_applied_missions=getattr(result, 'anchor_fix_applied_missions', []),
            anchor_applied_vars=getattr(result, 'anchor_fix_applied_vars', {}),
            anchor_skip_reason=getattr(result, 'anchor_fix_skip_reason', ''),
        )

        logger.info(
            "fallback_chain %s: status=%s time=%dms anchors=%d/%d "
            "unlock=%s freeze_h=%d eps=%.3f",
            attempt_name, result.status.value, result.solve_time_ms,
            result.anchor_fix_applied_count, result.anchor_fix_skipped_count,
            unlock_ids, freeze_hours, epsilon,
        )

        return result, attempt

    # 当前工作参数（逐步变异）
    cur_unlock = list(decision.unlock_mission_ids)
    cur_freeze_h = decision.freeze_horizon_hours
    cur_epsilon = decision.epsilon_solver

    # ================================================================
    # Step 0: initial 求解
    # ================================================================
    result, attempt = _solve_once(
        "initial", cur_unlock, cur_freeze_h, cur_epsilon, use_anchor=True
    )
    attempts.append(attempt)

    if _is_success(result.status):
        return FallbackChainResult(
            solver_result=result, success=True, attempts=attempts,
            final_attempt_name="initial", total_solver_calls=total_calls,
        )

    # ================================================================
    # Step 1: attempt1 — 扩大 unlock_set (+2)
    # ================================================================
    cur_unlock = _expand_unlock_from_conflicts(
        cur_unlock, trcg_dict, eligible_ids, expand_count=2
    )

    result, attempt = _solve_once(
        "attempt1_expand_unlock", cur_unlock, cur_freeze_h, cur_epsilon,
        use_anchor=True,
    )
    attempts.append(attempt)

    if _is_success(result.status):
        return FallbackChainResult(
            solver_result=result, success=True, attempts=attempts,
            final_attempt_name="attempt1_expand_unlock",
            total_solver_calls=total_calls,
        )

    # ================================================================
    # Step 2: attempt2 — 降低 freeze_horizon 一档
    # ================================================================
    new_freeze_h = _FREEZE_STEP_DOWN.get(cur_freeze_h, 0)
    if new_freeze_h == cur_freeze_h:
        new_freeze_h = 0  # 已经是 0 就保持 0

    # 重新计算 frozen_ops
    new_freeze_slots = FREEZE_HOURS_TO_SLOTS.get(new_freeze_h, 0)
    recomputed_frozen = frozen_ops
    if current_plan_for_refreeze is not None and new_freeze_slots != FREEZE_HOURS_TO_SLOTS.get(cur_freeze_h, 0):
        recomputed_frozen = compute_frozen_ops_fn(
            current_plan_for_refreeze, now, new_freeze_slots,
            started_ops, completed_ops,
        )

    cur_freeze_h = new_freeze_h

    result, attempt = _solve_once(
        "attempt2_reduce_freeze", cur_unlock, cur_freeze_h, cur_epsilon,
        use_anchor=True, frozen_ops_override=recomputed_frozen,
    )
    attempts.append(attempt)

    if _is_success(result.status):
        return FallbackChainResult(
            solver_result=result, success=True, attempts=attempts,
            final_attempt_name="attempt2_reduce_freeze",
            total_solver_calls=total_calls,
        )

    # ================================================================
    # Step 3: attempt3 — 放宽 epsilon 一档
    # ================================================================
    cur_epsilon = _EPSILON_STEP_UP.get(cur_epsilon, 0.10)

    result, attempt = _solve_once(
        "attempt3_relax_epsilon", cur_unlock, cur_freeze_h, cur_epsilon,
        use_anchor=True, frozen_ops_override=recomputed_frozen,
    )
    attempts.append(attempt)

    if _is_success(result.status):
        return FallbackChainResult(
            solver_result=result, success=True, attempts=attempts,
            final_attempt_name="attempt3_relax_epsilon",
            total_solver_calls=total_calls,
        )

    # ================================================================
    # Step 4: 最终回退 — 保留 freeze + 使用全 unlock（仍有软锚点惩罚作用）
    # ================================================================
    logger.warning(
        "fallback_chain: all 3 attempts failed, falling back to graceful global replan"
    )

    # 使用原始 freeze_horizon（不是 0！），只全部 unlock
    # 即使全 unlock，solver Stage 2 的 drift 目标函数仍然惩罚移动，
    # 比完全没有 anchor penalty 的 None 更好。
    original_freeze_slots = FREEZE_HOURS_TO_SLOTS.get(decision.freeze_horizon_hours, 0)
    global_frozen = compute_frozen_ops_fn(
        current_plan_for_refreeze, now, original_freeze_slots,
        started_ops, completed_ops,
    ) if current_plan_for_refreeze is not None else {}

    result, attempt = _solve_once(
        "final_global_replan",
        sorted(eligible_ids),  # 全 unlock（所有 mission 可移动）
        decision.freeze_horizon_hours,  # 保留原始 freeze 小时数（非 0）
        0.15,   # epsilon 放宽但仍有约束
        use_anchor=True,   # 仍使用锚点（全 unlock = 所有 mission 跳过锚点检查，但保留框架）
        frozen_ops_override=global_frozen,
    )
    attempts.append(attempt)

    return FallbackChainResult(
        solver_result=result,
        success=_is_success(result.status),
        attempts=attempts,
        final_attempt_name="final_global_replan",
        total_solver_calls=total_calls,
    )


# ============================================================================
# Rolling 步骤结构化日志
# ============================================================================

@dataclass
class RepairStepLog:
    """
    每次 rolling 步骤的完整诊断日志。

    由 TRCGRepairPolicy.decide() 在一次 rolling 调用结束时构建。
    可序列化为 JSON 写入 llm_logs/ 或嵌入 rolling_metrics。

    字段分组
    --------
    [时间]  now_slot, wall_clock_ms
    [TRCG]  trcg_pressure, trcg_top_conflicts, trcg_urgent_ids
    [LLM]   llm_raw_output (截断), llm_http_ok, llm_parse_ok, llm_decision_ok, llm_error
    [决策]  decision_json, decision_source
    [回退]  fallback_reason, fallback_attempts, final_attempt_name
    [求解]  solver_status, solver_time_ms, anchor_fix_applied, anchor_fix_skipped,
            anchor_fix_applied_missions, anchor_fix_applied_vars, anchor_fix_skip_reason
    """
    # ---- 时间 ----
    now_slot: int = 0
    wall_clock_ms: int = 0

    # ---- TRCG 摘要（精简版，避免日志膨胀） ----
    trcg_pressure: Dict[str, float] = field(default_factory=dict)
    trcg_top_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    trcg_urgent_ids: List[str] = field(default_factory=list)

    # ---- LLM 原始输出 ----
    llm_raw_output: str = ""            # 截断至 500 字符
    llm_http_ok: bool = False           # HTTP 请求是否成功（无网络/API 错误）
    llm_parse_ok: bool = False          # schema 校验是否通过
    llm_decision_ok: bool = False       # 最终 decision 是否来自 LLM
    llm_error: Dict[str, Any] = field(default_factory=dict)  # 结构化错误信息

    # ---- 最终使用的决策 ----
    decision_json: Dict[str, Any] = field(default_factory=dict)
    decision_source: str = "unknown"
    #   "llm"                — LLM 输出通过校验，直接使用
    #   "heuristic_fallback" — LLM 失败后启发式生成
    #   "forced_global"      — 求解回退链路全部失败，最终全局重排

    # ---- 回退信息 ----
    fallback_reason: str = ""           # 为何回退（validation_failed / llm_timeout / ...）
    fallback_attempts: List[Dict[str, Any]] = field(default_factory=list)
    final_attempt_name: str = ""        # 回退链最终采用的 attempt
    total_solver_calls: int = 0

    # ---- 求解结果 ----
    solver_status: str = ""             # OPTIMAL / FEASIBLE / INFEASIBLE / ...
    solver_time_ms: int = 0
    anchor_fix_applied: int = 0
    anchor_fix_skipped: int = 0
    anchor_fix_applied_missions: List[str] = field(default_factory=list)
    anchor_fix_applied_vars: Dict[str, int] = field(default_factory=dict)
    anchor_fix_skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


_LOG_RAW_OUTPUT_MAX_LEN = 500


def build_repair_step_log(
    now_slot: int,
    trcg_dict: Optional[Dict[str, Any]] = None,
    llm_raw_output: Optional[str] = None,
    llm_http_ok: bool = False,
    llm_parse_ok: bool = False,
    llm_decision_ok: bool = False,
    llm_error: Optional[Dict[str, Any]] = None,
    decision: Optional[RepairDecision] = None,
    decision_source: str = "unknown",
    fallback_reason: str = "",
    chain_result: Optional[FallbackChainResult] = None,
    wall_clock_ms: int = 0,
) -> RepairStepLog:
    """
    构建一次 rolling 步骤的结构化日志。

    Parameters
    ----------
    now_slot : 当前时刻
    trcg_dict : TRCGSummary.to_dict() 输出
    llm_raw_output : LLM 原始文本（会截断至 500 字符）
    llm_http_ok : HTTP 请求是否成功
    llm_parse_ok : schema 校验是否通过
    llm_decision_ok : 最终 decision 是否来自 LLM
    llm_error : 结构化错误信息 dict
    decision : 最终使用的 RepairDecision
    decision_source : "llm" / "heuristic_fallback" / "forced_global"
    fallback_reason : 回退原因
    chain_result : FallbackChainResult
    wall_clock_ms : 整个 decide() 的 wall-clock 毫秒

    Returns
    -------
    RepairStepLog
    """
    log = RepairStepLog(now_slot=now_slot, wall_clock_ms=wall_clock_ms)

    # ---- TRCG 精简 ----
    if trcg_dict:
        log.trcg_pressure = trcg_dict.get('bottleneck_pressure', {})
        # top_conflicts 保留前 5 条
        raw_conflicts = trcg_dict.get('top_conflicts', [])
        log.trcg_top_conflicts = raw_conflicts[:5]
        # urgent 只存 mission_id 列表
        log.trcg_urgent_ids = [
            u.get('mission_id', '') for u in trcg_dict.get('urgent_missions', [])
        ]

    # ---- LLM ----
    log.llm_http_ok = llm_http_ok
    log.llm_parse_ok = llm_parse_ok
    log.llm_decision_ok = llm_decision_ok
    log.llm_error = llm_error or {}
    if llm_raw_output:
        log.llm_raw_output = llm_raw_output[:_LOG_RAW_OUTPUT_MAX_LEN]
    else:
        log.llm_raw_output = ""

    # ---- 决策 ----
    log.decision_source = decision_source
    if decision:
        log.decision_json = decision.to_dict()

    # ---- 回退 ----
    log.fallback_reason = fallback_reason
    if chain_result:
        log.fallback_attempts = [asdict(a) for a in chain_result.attempts]
        log.final_attempt_name = chain_result.final_attempt_name
        log.total_solver_calls = chain_result.total_solver_calls
        # 求解结果
        if chain_result.solver_result:
            log.solver_status = chain_result.solver_result.status.value
            log.solver_time_ms = chain_result.solver_result.solve_time_ms
            log.anchor_fix_applied = chain_result.solver_result.anchor_fix_applied_count
            log.anchor_fix_skipped = chain_result.solver_result.anchor_fix_skipped_count
            log.anchor_fix_applied_missions = getattr(
                chain_result.solver_result, 'anchor_fix_applied_missions', []
            )
            log.anchor_fix_applied_vars = getattr(
                chain_result.solver_result, 'anchor_fix_applied_vars', {}
            )
            log.anchor_fix_skip_reason = getattr(
                chain_result.solver_result, 'anchor_fix_skip_reason', ''
            )

    return log


# ============================================================================
# 模块自测
# ============================================================================

if __name__ == "__main__":
    print("=== policy_llm_repair.py self-test ===\n")

    active = {"M001", "M002", "M003", "M007", "M010"}
    started = {"M004", "M005"}
    completed = {"M006", "M008"}

    # ---- 正例 ----
    good_cases = [
        # 1. 纯净 JSON
        '{"freeze_horizon_hours":8,"epsilon_solver":0.05,'
        '"root_cause_mission_id":"M001","unlock_mission_ids":["M001","M002"],'
        '"secondary_root_cause_mission_id":"M003",'
        '"analysis_short":"R_pad contention between M001/M002 near slot 60"}',

        # 2. 带 code fence
        '```json\n{"freeze_horizon_hours":0,"epsilon_solver":0.10,'
        '"root_cause_mission_id":"M007","unlock_mission_ids":["M007"],'
        '"secondary_root_cause_mission_id":null,'
        '"analysis_short":"Pad outage forces M007 reschedule"}\n```',

        # 3. 前后有杂文
        'Thinking...\n{"freeze_horizon_hours":24,"epsilon_solver":0.0,'
        '"root_cause_mission_id":"M010","unlock_mission_ids":["M010","M003"],'
        '"secondary_root_cause_mission_id":"M001",'
        '"analysis_short":"Low conflict; freeze most"}\nDone.',
    ]

    for idx, text in enumerate(good_cases, 1):
        r = validate_repair_decision(text, active, started, completed)
        tag = "PASS" if r.is_valid else "FAIL"
        print(f"  Good#{idx} [{tag}] method={r.extraction_method} "
              f"warnings={r.warnings}")
        if r.decision:
            print(f"         root={r.decision.root_cause_mission_id} "
                  f"unlock={r.decision.unlock_mission_ids} "
                  f"freeze_slots={r.decision.freeze_horizon_slots}")

    # ---- 反例 ----
    bad_cases = [
        ("missing field",
         '{"freeze_horizon_hours":8,"epsilon_solver":0.05}'),
        ("root not in unlock",
         '{"freeze_horizon_hours":8,"epsilon_solver":0.05,'
         '"root_cause_mission_id":"M001","unlock_mission_ids":["M002"],'
         '"secondary_root_cause_mission_id":null,"analysis_short":"x"}'),
        ("unlock contains started mission",
         '{"freeze_horizon_hours":8,"epsilon_solver":0.05,'
         '"root_cause_mission_id":"M001","unlock_mission_ids":["M001","M004"],'
         '"secondary_root_cause_mission_id":null,"analysis_short":"x"}'),
        ("root not active",
         '{"freeze_horizon_hours":8,"epsilon_solver":0.05,'
         '"root_cause_mission_id":"M006","unlock_mission_ids":["M006"],'
         '"secondary_root_cause_mission_id":null,"analysis_short":"x"}'),
        ("garbage",
         'I think we should delay M001 by 3 slots.'),
    ]

    # 向后兼容 case（LLM 输出含 freeze/epsilon 但仍合法）
    compat_cases = [
        ("freeze ignored (backward compat)",
         '{"freeze_horizon_hours":5,"epsilon_solver":0.05,'
         '"root_cause_mission_id":"M001","unlock_mission_ids":["M001"],'
         '"secondary_root_cause_mission_id":null,"analysis_short":"x"}'),
        ("unlock dedup ok",
         '{"freeze_horizon_hours":8,"epsilon_solver":0.05,'
         '"root_cause_mission_id":"M001",'
         '"unlock_mission_ids":["M001","M002","M003","M007","M010","M001"],'
         '"secondary_root_cause_mission_id":null,"analysis_short":"x"}'),
    ]

    print()
    for label, text in bad_cases:
        r = validate_repair_decision(text, active, started, completed)
        tag = "PASS" if not r.is_valid else "FAIL"
        print(f"  Bad[{label}] [{tag}] errors={r.errors[:2]}")

    for label, text in compat_cases:
        r = validate_repair_decision(text, active, started, completed)
        tag = "PASS" if r.is_valid else "FAIL"
        print(f"  Compat[{label}] [{tag}] warnings={r.warnings[:2]}")

    # ---- prompt 构造 ----
    print("\n--- Prompt sample ---")
    from features import TRCGSummary
    dummy_trcg = TRCGSummary(
        now_slot=48, horizon_end_slot=144,
        bottleneck_pressure={'pad_util': 0.85, 'r3_util': 0.6, 'range_test_util': 0.2},
        top_conflicts=[{'a': 'M001', 'b': 'M002', 'resource': 'R_pad',
                        'overlap_slots': 4, 't_range': [58, 62], 'severity': 6.0}],
        conflict_clusters=[{'center_mission_id': 'M001', 'members': ['M001', 'M002'],
                            'score': 6.0}],
        urgent_missions=[{'mission_id': 'M001', 'due_slot': 120, 'due_slack_slots': 72,
                          'window_slack_slots': 10, 'current_delay_slots': 0,
                          'priority': 1.0, 'urgency_score': 77.0}],
        disturbance_summary={'range_loss_pct': 0.0, 'pad_outage_active': True,
                             'duration_volatility_level': 'medium'},
        frozen_summary={'num_started_ops': 12, 'num_frozen_ops': 5,
                        'frozen_horizon_slots': 12},
    )
    prompt = build_repair_user_prompt(dummy_trcg.to_dict(), sorted(active))
    print(f"  system prompt length: {len(REPAIR_SYSTEM_PROMPT)} chars")
    print(f"  user prompt length:   {len(prompt)} chars")
    print(f"  user prompt preview:\n{prompt[:300]}...")

    # ---- heuristic_repair_decision 测试 ----
    print("\n--- Heuristic Fallback Tests ---")

    # Case H1: 有冲突 → root = degree 最大
    trcg_h1 = dummy_trcg.to_dict()
    h1 = heuristic_repair_decision(trcg_h1, active, started, completed)
    r_h1 = validate_repair_decision(
        json.dumps(asdict(h1)), active, started, completed
    )
    print(f"  H1 root={h1.root_cause_mission_id} unlock={h1.unlock_mission_ids} "
          f"freeze_h={h1.freeze_horizon_hours} eps={h1.epsilon_solver} "
          f"valid={r_h1.is_valid}")
    assert r_h1.is_valid, f"H1 validation failed: {r_h1.errors}"
    assert h1.root_cause_mission_id == "M001", "H1: expected root=M001"

    # Case H2: 无冲突 → 走 urgent 分支
    trcg_h2 = dict(trcg_h1)
    trcg_h2['top_conflicts'] = []
    trcg_h2['conflict_clusters'] = []
    h2 = heuristic_repair_decision(trcg_h2, active, started, completed)
    r_h2 = validate_repair_decision(
        json.dumps(asdict(h2)), active, started, completed
    )
    print(f"  H2 root={h2.root_cause_mission_id} unlock={h2.unlock_mission_ids} "
          f"freeze_h={h2.freeze_horizon_hours} eps={h2.epsilon_solver} "
          f"valid={r_h2.is_valid}")
    assert r_h2.is_valid, f"H2 validation failed: {r_h2.errors}"
    assert h2.root_cause_mission_id == "M001", "H2: expected root=M001 (urgent)"

    # Case H3: 无冲突且无 urgent → 字典序最小
    trcg_h3 = dict(trcg_h2)
    trcg_h3['urgent_missions'] = []
    trcg_h3['bottleneck_pressure'] = {'pad_util': 0.3, 'r3_util': 0.2,
                                       'range_test_util': 0.1}
    h3 = heuristic_repair_decision(trcg_h3, active, started, completed)
    r_h3 = validate_repair_decision(
        json.dumps(asdict(h3)), active, started, completed
    )
    print(f"  H3 root={h3.root_cause_mission_id} unlock={h3.unlock_mission_ids} "
          f"freeze_h={h3.freeze_horizon_hours} eps={h3.epsilon_solver} "
          f"valid={r_h3.is_valid}")
    assert r_h3.is_valid, f"H3 validation failed: {r_h3.errors}"
    assert h3.root_cause_mission_id == "M001", "H3: expected root=M001 (lexicographic)"
    assert h3.freeze_horizon_hours == 24, "H3: expected freeze=24h (fixed default)"

    # Case H4: 重度压力 → K=5, freeze=0, eps=0.02
    trcg_h4 = dict(trcg_h1)
    trcg_h4['bottleneck_pressure'] = {'pad_util': 0.90, 'r3_util': 0.7,
                                       'range_test_util': 0.5}
    trcg_h4['top_conflicts'] = [
        {'a': 'M001', 'b': 'M002', 'resource': 'R_pad', 'overlap_slots': 4,
         't_range': [58, 62], 'severity': 6.0},
        {'a': 'M001', 'b': 'M003', 'resource': 'R3',   'overlap_slots': 2,
         't_range': [50, 55], 'severity': 3.0},
        {'a': 'M002', 'b': 'M007', 'resource': 'R_pad', 'overlap_slots': 3,
         't_range': [60, 63], 'severity': 4.5},
    ]
    trcg_h4['urgent_missions'] = [
        {'mission_id': f'M00{i}', 'due_slot': 120, 'due_slack_slots': 30,
         'window_slack_slots': 5, 'current_delay_slots': 2, 'priority': 1.0,
         'urgency_score': 40.0}
        for i in range(1, 5)
    ]
    h4 = heuristic_repair_decision(trcg_h4, active, started, completed)
    r_h4 = validate_repair_decision(
        json.dumps(asdict(h4)), active, started, completed
    )
    print(f"  H4 root={h4.root_cause_mission_id} unlock={h4.unlock_mission_ids} "
          f"freeze_h={h4.freeze_horizon_hours} eps={h4.epsilon_solver} "
          f"sec={h4.secondary_root_cause_mission_id} valid={r_h4.is_valid}")
    assert r_h4.is_valid, f"H4 validation failed: {r_h4.errors}"
    # M001: 6+3=9, M002: 6+4.5=10.5 → M002 is root
    assert h4.root_cause_mission_id == "M002", f"H4: expected root=M002 (degree=10.5)"
    assert h4.freeze_horizon_hours == 24, "H4: freeze=24h (fixed default)"
    assert h4.epsilon_solver == 0.10, "H4: eps=0.10 (fixed default)"
    assert len(h4.unlock_mission_ids) >= 3, "H4: expected K>=3"

    # Case H5: 确定性 — 同一输入多次调用结果一致
    h5a = heuristic_repair_decision(trcg_h4, active, started, completed)
    h5b = heuristic_repair_decision(trcg_h4, active, started, completed)
    assert asdict(h5a) == asdict(h5b), "H5: determinism check FAILED"
    print("  H5 determinism check PASS")

    print("\nAll tests done.")
