"""
TRCGRepairPolicy — 基于 TRCG 根因诊断 + LLM 决策 + CP-SAT 锚点修复策略

将以下组件串联为完整策略类：
- features.build_trcg_summary()        → TRCG 诊断
- policy_llm_repair: prompt / schema   → LLM 调用 & JSON 校验
- policy_llm_repair: heuristic_repair_decision → 启发式回退
- policy_llm_repair: RepairStepLog     → 结构化日志

decide() 返回 (MetaParams, None)，由 simulator 调用 solve_v2_1。
"""

import json
import os
import logging
import time as _time
from typing import Optional, Tuple, Dict, Any, Set, List

from policies.base import BasePolicy, MetaParams
from config import Config

from policies.policy_llm_repair import (
    RepairDecision,
    REPAIR_SYSTEM_PROMPT,
    build_repair_user_prompt,
    validate_repair_decision,
    heuristic_repair_decision,
    RepairStepLog,
    build_repair_step_log,
    FallbackChainResult,
)
from features import build_trcg_summary, TRCGSummary
from solver_cpsat import compute_frozen_ops

# LLM Client（可选：未安装 openai 时仍可运行，走纯启发式）
try:
    from llm_client import LLMClient, LLMConfig, LLMCallResult
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False

logger = logging.getLogger("TRCGRepairPolicy")


class TRCGRepairPolicy(BasePolicy):
    """
    TRCG 根因修复策略 —— LLM 诊断 + 启发式回退 + MetaParams 驱动 CP-SAT。

    生命周期
    --------
    1. 构造时传入 LLMClient（可选）。若无 client → 全部走启发式。
    2. decide() 每个 rolling step 调用一次：
       a) build_trcg_summary → 诊断
       b) LLM 调用 → validate → 成功用 LLM 决策
       c) validate 失败 → heuristic_repair_decision
       d) 输出 MetaParams（含 unlock_mission_ids 等扩展字段）
    3. simulator 拿到 MetaParams 后调用 solve_with_fallback（回退链在 simulator 侧）。

    日志
    ----
    每次 decide() 构建 RepairStepLog，写入 llm_logs/ 目录。
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_config: Optional[Any] = None,
        policy_name: str = "trcg_repair",
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        episode_id: str = "",
        # 默认权重（与 run_experiments 中 baseline 对齐：纯延迟最小化）
        w_delay: float = 1.0,
        w_shift: float = 0.0,
        w_switch: float = 0.0,
    ):
        self._policy_name = policy_name
        self._log_dir = log_dir
        self._enable_logging = enable_logging
        self._episode_id = episode_id

        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch

        # LLM Client
        self._llm_client = llm_client
        self._llm_config = llm_config
        if self._llm_client is None and self._llm_config is not None and HAS_LLM_CLIENT:
            self._llm_client = LLMClient(self._llm_config)

        # 状态（per-episode）
        self._prev_window_slots: Optional[Dict[str, Set[int]]] = None
        self._call_count = 0
        self._llm_ok_count = 0
        self._heuristic_count = 0
        self._skip_count = 0          # 方案 A: 无冲突跳过 LLM 次數
        self._stable_skip_count = 0   # 状态稳定跳过次数
        self._step_logs: List[RepairStepLog] = []
        # 延迟日志：decide() 構建 pending log，solver 完成后由 simulator 回写
        self._pending_step_log: Optional[RepairStepLog] = None
        self._pending_now: Optional[int] = None
        # 状态指纹：用于检测 TRCG 状态是否稳定（稳定时复用上次决策，跳过 LLM）
        self._last_state_fingerprint: Optional[str] = None
        self._last_llm_decision: Optional[RepairDecision] = None
        self._last_decision_source: Optional[str] = None

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._policy_name

    def reset(self) -> None:
        self._prev_window_slots = None
        self._call_count = 0
        self._llm_ok_count = 0
        self._heuristic_count = 0
        self._skip_count = 0
        self._stable_skip_count = 0
        self._step_logs = []
        self._pending_step_log = None
        self._pending_now = None
        self._last_state_fingerprint = None
        self._last_llm_decision = None
        self._last_decision_source = None

    def set_episode_id(self, episode_id: str) -> None:
        self._episode_id = episode_id

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    def decide(
        self,
        state: Any,
        now: int,
        config: Config,
    ) -> Tuple[MetaParams, None]:
        """
        策略决策入口。

        Returns
        -------
        (MetaParams, None) — MetaParams 携带 unlock_mission_ids 等 TRCG 修复字段；
        direct_plan=None（交由 simulator 侧 solver 求解）。
        """
        t0 = _time.time()
        self._call_count += 1

        # ============================================================
        # Step 1: 计算 frozen_ops（供 TRCG 诊断使用）
        # ============================================================
        freeze_slots_for_trcg = config.freeze_horizon
        frozen_ops = compute_frozen_ops(
            state.current_plan, now, freeze_slots_for_trcg,
            getattr(state, 'started_ops', set()),
            getattr(state, 'completed_ops', set()),
        )

        # ============================================================
        # Step 2: 构造 TRCGSummary
        # ============================================================
        trcg = build_trcg_summary(
            missions=state.missions,
            resources=state.resources,
            plan=state.current_plan,
            now=now,
            config=config,
            started_ops=getattr(state, 'started_ops', set()),
            completed_ops=getattr(state, 'completed_ops', set()),
            actual_durations=getattr(state, 'actual_durations', {}),
            frozen_ops=frozen_ops,
            prev_window_slots=self._prev_window_slots,
        )
        trcg_dict = trcg.to_dict()

        # ============================================================
        # Step 2.5: 条件跳过 — 无冲突 + 低压力 → 跳过 LLM（方案 A）
        #   当 TRCG 诊断显示无资源冲突且系统无紧迫压力时，LLM 调用
        #   不会产生有意义的 unlock 决策，跳过后返回全解锁 MetaParams
        #   （等价于 fixed_tuned 基线行为），不影响有冲突时的 LLM 质量。
        # ============================================================
        _should_skip = self._check_skip_conditions(trcg_dict)

        # 更新 prev_window_slots：记录本次 Launch 窗口 slots，供下次计算 range_loss_pct
        horizon_end = now + config.horizon_slots
        _completed_ops = getattr(state, 'completed_ops', set())
        current_window_slots: Dict[str, Set[int]] = {}
        for m in state.missions:
            launch = m.get_launch_op()
            if not launch or launch.op_id in _completed_ops:
                continue
            slots: Set[int] = set()
            for ws, we in launch.time_windows:
                for s in range(max(ws, now), min(we, horizon_end) + 1):
                    slots.add(s)
            current_window_slots[m.mission_id] = slots
        self._prev_window_slots = current_window_slots

        # ============================================================
        # Step 3: 确定 active / started / completed mission 集合
        # ============================================================
        started_ops: Set[str] = getattr(state, 'started_ops', set())
        completed_ops: Set[str] = getattr(state, 'completed_ops', set())

        # mission 粒度：一个 mission "started" 若其 任一 op 已 started
        started_mission_ids: Set[str] = set()
        completed_mission_ids: Set[str] = set()
        all_mission_ids: Set[str] = set()
        horizon_end = now + config.horizon_slots

        for m in state.missions:
            all_mission_ids.add(m.mission_id)
            # completed: Launch op 完成
            launch = m.get_launch_op()
            if launch and launch.op_id in completed_ops:
                completed_mission_ids.add(m.mission_id)
                continue
            # started: 任何 op 已 started 且未全部完成
            for op in m.operations:
                if op.op_id in started_ops:
                    started_mission_ids.add(m.mission_id)
                    break

        # active = schedulable 且 not started 且 not completed
        schedulable_ids = {
            m.mission_id for m in state.get_schedulable_missions(horizon_end)
        }
        active_mission_ids = schedulable_ids - started_mission_ids - completed_mission_ids

        # 如果 active 为空（所有 mission 都在进行或完成），回退到 schedulable
        if not active_mission_ids:
            active_mission_ids = schedulable_ids - completed_mission_ids

        # ============================================================
        # Step 4: LLM 调用 → 校验 → 启发式回退
        #   若 Step 2.5 判定 skip，直接构造全解锁 MetaParams 返回。
        # ============================================================
        if _should_skip:
            return self._build_skip_return(now, t0, trcg_dict, config)

        # ============================================================
        # Step 4.1: 状态指纹跳过 — 若 TRCG 核心状态未变，复用上次 LLM 决策
        #   fingerprint = (sorted_urgent_ids, sorted_conflict_pairs, sorted_active_ids)
        #   连续相似状态下避免重复调用 LLM（每次 ~40s），显著加速。
        # ============================================================
        _state_fp = self._compute_state_fingerprint(trcg_dict, active_mission_ids)
        if (self._last_state_fingerprint is not None
                and _state_fp == self._last_state_fingerprint
                and self._last_llm_decision is not None):
            return self._build_stable_skip_return(
                now, t0, trcg_dict, config,
                self._last_llm_decision, self._last_decision_source or "llm",
                active_mission_ids, started_mission_ids, completed_mission_ids,
            )

        llm_raw_output: Optional[str] = None
        llm_http_ok = False
        llm_parse_ok = False
        llm_decision_ok = False
        llm_error: Dict[str, Any] = {}
        decision: Optional[RepairDecision] = None
        decision_source = "heuristic_fallback"
        fallback_reason: Optional[str] = None

        if self._llm_client is not None and active_mission_ids:
            # 4a. 构建 prompt
            user_prompt = build_repair_user_prompt(
                trcg_dict, sorted(active_mission_ids)
            )

            # 4b. 调用 LLM
            try:
                llm_result = self._llm_client.call(
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                )
                llm_http_ok = llm_result.success
                llm_raw_output = llm_result.raw_text or ""
                if not llm_result.success:
                    llm_error = {
                        "error_type": llm_result.error_type or "unknown",
                        "http_status": getattr(llm_result, 'http_status_code', None),
                        "is_timeout": "timeout" in (llm_result.error_type or "").lower(),
                        "message": llm_result.error_message or "",
                        "response_snippet": (llm_raw_output or "")[:200],
                        "retries": llm_result.retries,
                    }
            except Exception as exc:
                llm_http_ok = False
                llm_error = {
                    "error_type": type(exc).__name__,
                    "http_status": None,
                    "is_timeout": "timeout" in type(exc).__name__.lower(),
                    "message": str(exc),
                    "response_snippet": "",
                    "retries": 0,
                }
                llm_raw_output = ""

            # 4c. 自动修正 + 校验
            #   LLM 可能选中 started mission（它在 TRCG urgent/conflict 中可见），
            #   直接校验会 fail → fallback。这里先尝试自动修正：
            #   - 从 unlock 中移除 started/completed mission
            #   - 如果 root_cause 是 started，替换为 unlock 中最高优先级 active mission
            #   修正后再校验，大幅降低不必要的 fallback。
            if llm_http_ok and llm_raw_output:
                corrected_output = self._auto_correct_llm_output(
                    llm_raw_output,
                    active_mission_ids,
                    started_mission_ids,
                    completed_mission_ids,
                    trcg_dict,
                )
                vr = validate_repair_decision(
                    corrected_output,
                    active_mission_ids,
                    started_mission_ids,
                    completed_mission_ids,
                )
                if vr.is_valid and vr.decision is not None:
                    llm_parse_ok = True
                    llm_decision_ok = True
                    decision = vr.decision
                    decision_source = "llm"
                    fallback_reason = None
                    self._llm_ok_count += 1
                else:
                    llm_parse_ok = False
                    llm_error = {
                        "error_type": "validation_failed",
                        "http_status": None,
                        "is_timeout": False,
                        "message": "; ".join(vr.errors[:3]),
                        "response_snippet": (llm_raw_output or "")[:200],
                        "validation_errors": vr.errors[:5],
                    }
                    fallback_reason = f"validation_failed: {'; '.join(vr.errors[:3])}"
            else:
                fallback_reason = llm_error.get("message", "") or "llm_call_failed"

        elif not active_mission_ids:
            fallback_reason = "no_active_missions"
        else:
            fallback_reason = "no_llm_client"

        # 4d. 启发式回退
        if decision is None and active_mission_ids:
            decision = heuristic_repair_decision(
                trcg_dict,
                active_mission_ids,
                started_mission_ids,
                completed_mission_ids,
                fallback_reason=fallback_reason or "heuristic",
            )
            decision_source = "heuristic_fallback"
            self._heuristic_count += 1

        # 4e. 极端兜底（所有 mission 都 started/completed）
        if decision is None:
            decision = RepairDecision(
                root_cause_mission_id="",
                unlock_mission_ids=[],

                analysis_short="no_active_missions_all_started_or_completed",
                secondary_root_cause_mission_id=None,
            )
            decision_source = "heuristic_fallback"
            fallback_reason = "no_active_missions"

        # 4f. 更新状态指纹 —— 记住本次 LLM 决策，供下一步稳定跳过
        self._last_state_fingerprint = _state_fp
        self._last_llm_decision = decision
        self._last_decision_source = decision_source

        # ============================================================
        # Step 5: freeze / epsilon 固定为 grid search 最优值
        #   核心优化：
        #   1. freeze_horizon 使用 RepairDecision 的值（24h=96slots），与 fixed_tuned best_params 一致。
        #      之前用 config.freeze_horizon=12 导致 drift 楼比 FT 多 22%——
        #      因为 freeze 决定了多少 mission 被屏蔽在 drift 计算之外。
        #   2. epsilon_solver 使用 RepairDecision 的值（0.10），
        #      给 Stage 2 更多空间全局最小化 drift_v3。
        #   回退链（Level 1–4）会在 solver INFEASIBLE 时自动降级。
        # ============================================================
        # freeze = RepairDecision 的 24h = 96 slots（匹配 fixed_tuned best_params）
        # freeze = RepairDecision 的 24h = 96 slots（匹配 fixed_tuned best_params）
        # TRCG selective freezing 已在 simulator 中实现，此处使用标准 freeze_horizon
        freeze_h_slots = max(
            int(decision.freeze_horizon_hours * 60 / config.slot_minutes),
            config.freeze_horizon,
        )
        # ---- 自适应 epsilon ----
        # TRCG 的核心优势：根据场景冲突严重程度调节 epsilon。
        # 低冲突 → 更高 epsilon → Stage 2 有更宽松的延迟预算 → 更低 drift。
        # 高冲突 → 保守 epsilon → 优先保证延迟不过大。
        # fixed_tuned 使用固定 epsilon=0.10，无法根据场景调节。
        _conflicts = trcg_dict.get('top_conflicts', [])
        _urgents = trcg_dict.get('urgent_missions', [])
        _pressure = trcg_dict.get('bottleneck_pressure', {})
        _max_pressure = max(_pressure.values(), default=0.0) if _pressure else 0.0

        base_epsilon = max(0.10, decision.epsilon_solver, config.default_epsilon_solver)
        if len(_conflicts) == 0 and len(_urgents) <= 1 and _max_pressure < 0.6:
            adaptive_epsilon = min(0.12, base_epsilon + 0.02)
        elif len(_conflicts) <= 2 and _max_pressure < 0.8:
            adaptive_epsilon = min(0.11, base_epsilon + 0.01)
        else:
            adaptive_epsilon = base_epsilon

        # Keep delay budget close to fixed_tuned (0.10) to avoid delay inflation.
        epsilon_solver = min(0.12, adaptive_epsilon)

        # 策略确定 epsilon: 使用自适应 epsilon 而非固定值
        # TRCG selective freezing 已在 simulator 层面保证低 drift，
        # 此处使用标准 epsilon。

        # ============================================================
        # Step 6: 构造 MetaParams
        #   核心改动：启用 unlock_mission_ids（锚点约束）
        #   —— 将 LLM 选出的 unlock 集合传给 solver，对非 unlock 的 mission
        #   施加锚点约束（固定为 prev_plan 的位置），使其不产生 drift。
        #   只有 unlock 集中的 mission 才参与重排。
        #
        #   这是 trcg_repair_llm 区别于 fixed_tuned 的核心机制：
        #   fixed_tuned 每次全量解锁（所有 mission 都可能移动 → 高 drift），
        #   而 LLM 精准定位需要移动的 mission → 其余锚定 → 低 drift。
        #
        #   当锚点导致求解不可行时，simulator 侧的 _solve_with_trcg_fallback
        #   会自动触发回退链（扩大 unlock → 降 freeze → 全局重排），
        #   最坏情况退化为 fixed_tuned（不会更差）。
        # ============================================================

        # 构建 unlock 集合：LLM/heuristic 决策的 unlock + started/completed 过滤
        _raw_unlock = decision.unlock_mission_ids or []
        _valid_unlock = [
            mid for mid in _raw_unlock
            if mid in active_mission_ids
        ]

        # ---- 扩展 unlock 集合 ----
        # LLM 通常只返回 1 个 unlock mission，导致 solver 搜索空间过小。
        # 从 TRCG 冲突邻居和 urgent missions 中补充，使 unlock 集合至少包含 3 个 mission，
        # 给 solver 更多优化空间。cap 到 active 的 50% 以保留 anchor 引导效果。
        # 降低 unlock 上限：更少 mission 被解锁 → 更多被锚定 → 更低 drift
        _MIN_UNLOCK = 3
        _MAX_UNLOCK_RATIO = 0.25  # 最多解锁 25% active mission
        _max_unlock = max(_MIN_UNLOCK, int(len(active_mission_ids) * _MAX_UNLOCK_RATIO))
        _unlock_set = set(_valid_unlock)
        _eligible = active_mission_ids - started_mission_ids - completed_mission_ids
        _target_unlock = min(len(_eligible), _MIN_UNLOCK)

        # 即使 LLM 只给了 1 个 root，也要补齐局部邻域，避免单点解锁退化为“尾部追加”。
        if len(_unlock_set) < _target_unlock:
            conflicts = trcg_dict.get('top_conflicts', [])
            for c in sorted(conflicts, key=lambda x: -float(x.get('severity', 0))):
                for m in (c.get('a', ''), c.get('b', '')):
                    if m in _eligible and m not in _unlock_set and len(_unlock_set) < _max_unlock:
                        _unlock_set.add(m)
                if len(_unlock_set) >= _target_unlock:
                    break

        if len(_unlock_set) < _target_unlock:
            urgents = trcg_dict.get('urgent_missions', [])
            for u in sorted(urgents, key=lambda x: x.get('urgency_score', 9999)):
                mid = u.get('mission_id', '')
                if mid in _eligible and mid not in _unlock_set and len(_unlock_set) < _max_unlock:
                    _unlock_set.add(mid)
                if len(_unlock_set) >= _target_unlock:
                    break

        _valid_unlock = list(_unlock_set)

        # 如果过滤后 unlock 为空，使用空元组（= 全部锚定）而非 None（= 全解锁）
        # 这是关键改动：即使 LLM 没有选出需要解锁的 mission，
        # 也不应退化为 fixed_tuned 的全解锁模式，而是锚定所有 mission。
        use_unlock = tuple(_valid_unlock)

        meta = MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=freeze_h_slots,
            use_two_stage=True,
            epsilon_solver=epsilon_solver,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
            unlock_mission_ids=use_unlock,
            root_cause_mission_id=decision.root_cause_mission_id or None,
            secondary_root_cause_mission_id=decision.secondary_root_cause_mission_id,
            decision_source=decision_source,
            fallback_reason=fallback_reason,
            attempt_idx=0,
        )

        # ============================================================
        # Step 7: 构建日志（延迟写入，等 solver 完成后回写）
        # ============================================================
        wall_ms = int((_time.time() - t0) * 1000)
        step_log = build_repair_step_log(
            now_slot=now,
            trcg_dict=trcg_dict,
            llm_raw_output=llm_raw_output,
            llm_http_ok=llm_http_ok,
            llm_parse_ok=llm_parse_ok,
            llm_decision_ok=llm_decision_ok,
            llm_error=llm_error,
            decision=decision,
            decision_source=decision_source,
            fallback_reason=fallback_reason or "",
            wall_clock_ms=wall_ms,
        )
        self._step_logs.append(step_log)
        # 暂存 pending，等 simulator 调 update_pending_log_with_solver_result() 后落盘
        self._pending_step_log = step_log
        self._pending_now = now

        logger.info(
            "t=%d source=%s root=%s unlock=%s freeze_h=%d eps=%.2f wall=%dms",
            now, decision_source,
            decision.root_cause_mission_id,
            decision.unlock_mission_ids,
            decision.freeze_horizon_hours,
            decision.epsilon_solver,
            wall_ms,
        )

        return meta, None

    # ------------------------------------------------------------------
    # LLM 输出自动修正 —— 将 started/completed mission 替换为 active
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_correct_llm_output(
        raw_text: str,
        active_mission_ids: Set[str],
        started_mission_ids: Set[str],
        completed_mission_ids: Set[str],
        trcg_dict: Dict[str, Any],
    ) -> str:
        """
        对 LLM 原始输出做自动修正，避免因选中 started mission 导致校验失败。

        修正逻辑：
        1. 解析 JSON
        2. 从 unlock_mission_ids 中移除 started/completed mission
        3. 如果 root_cause_mission_id 是 started/completed：
           a) 用 unlock 中剩余的第一个 active mission 替代
           b) 如果 unlock 为空，从冲突中选最高严重度的 active mission
           c) 如果仍然为空，选 active 中字典序最小的
        4. 同理修正 secondary_root_cause_mission_id
        5. 重新序列化为 JSON 字符串

        如果解析失败或无法修正，返回原始文本（交给 validate 处理错误）。
        """
        import json as _json

        # 尝试解析 JSON
        try:
            from policies.policy_llm_repair import _extract_json
            json_str, _method = _extract_json(raw_text)
            if json_str is None:
                return raw_text
            data = _json.loads(json_str)
            if not isinstance(data, dict):
                return raw_text
        except Exception:
            return raw_text

        invalid_ids = started_mission_ids | completed_mission_ids
        modified = False

        # 修正 unlock_mission_ids
        unlock = data.get('unlock_mission_ids')
        if isinstance(unlock, list):
            clean_unlock = [m for m in unlock if str(m) not in invalid_ids]
            if len(clean_unlock) < len(unlock):
                modified = True
                data['unlock_mission_ids'] = clean_unlock
            unlock = clean_unlock
        else:
            unlock = []

        # 当 unlock 被清空时，主动从 TRCG 冲突中补充 active mission
        # 避免 unlock=[] 导致后续退化为全解锁（=fixed_tuned 无区别）
        if not unlock:
            _eligible = active_mission_ids - invalid_ids
            conflicts = trcg_dict.get('top_conflicts', [])
            _degree: Dict[str, float] = {}
            for c in conflicts:
                for m in (c.get('a', ''), c.get('b', '')):
                    if m in _eligible:
                        _degree[m] = _degree.get(m, 0.0) + float(c.get('severity', 0))
            if _degree:
                # 从冲突中选度数最高的 active mission 作为替代 unlock
                _top = sorted(_degree, key=lambda x: (-_degree[x], x))
                refill = _top[:3]    # 最多补充 3 个
                data['unlock_mission_ids'] = refill
                unlock = refill
                modified = True
            else:
                # 从 urgent missions 补充
                urgents = trcg_dict.get('urgent_missions', [])
                refill = []
                for u in sorted(urgents, key=lambda x: x.get('urgency_score', 9999)):
                    mid = u.get('mission_id', '')
                    if mid in _eligible and mid not in refill:
                        refill.append(mid)
                    if len(refill) >= 3:
                        break
                if refill:
                    data['unlock_mission_ids'] = refill
                    unlock = refill
                    modified = True

        # 修正 root_cause_mission_id
        root = data.get('root_cause_mission_id', '')
        if isinstance(root, str) and root in invalid_ids:
            modified = True
            new_root = None
            # 优先从剩余 unlock 中选
            if unlock:
                new_root = unlock[0]
            else:
                # 从冲突中选最高严重度的 active mission
                conflicts = trcg_dict.get('top_conflicts', [])
                degree: Dict[str, float] = {}
                for c in conflicts:
                    for m in (c.get('a', ''), c.get('b', '')):
                        if m in active_mission_ids:
                            degree[m] = degree.get(m, 0.0) + float(c.get('severity', 0))
                if degree:
                    new_root = max(degree, key=degree.get)
                else:
                    # 最后手段：取 active 中字典序最小的
                    new_root = min(active_mission_ids) if active_mission_ids else root

            if new_root:
                data['root_cause_mission_id'] = new_root
                # 确保 root 在 unlock 中
                if new_root not in data.get('unlock_mission_ids', []):
                    data['unlock_mission_ids'] = [new_root] + data.get('unlock_mission_ids', [])

        # 修正 secondary_root_cause_mission_id
        secondary = data.get('secondary_root_cause_mission_id')
        if isinstance(secondary, str) and secondary in invalid_ids:
            data['secondary_root_cause_mission_id'] = None
            modified = True

        if modified:
            return _json.dumps(data, ensure_ascii=False)
        return raw_text

    # ------------------------------------------------------------------
    # 方案 A: 条件跳过 LLM —— 无冲突 + 低压力时全锚定
    # ------------------------------------------------------------------

    # 跳过阈值常量（放宽以提高跳过率，实际场景 urgent 通常≥3，压力≥0.8）
    _SKIP_PRESSURE_THRESH = 0.85  # bottleneck 压力低于此值视为"宽松"
    _SKIP_MAX_URGENT = 2          # urgent mission ≤ 此值时可跳过

    def _check_skip_conditions(self, trcg_dict: Dict[str, Any]) -> bool:
        """
        判断当前 TRCG 状态是否满足跳过 LLM 的条件。

        跳过条件（全部满足时跳过）：
        1. top_conflicts 为空（无资源冲突）
        2. urgent_missions 数量 ≤ _SKIP_MAX_URGENT
        3. bottleneck_pressure 全部 < _SKIP_PRESSURE_THRESH

        Returns True 表示可以跳过 LLM 调用。
        """
        # 条件 1: 无冲突
        conflicts = trcg_dict.get('top_conflicts', [])
        if conflicts:
            return False

        # 条件 2: 紧急 mission 少
        urgents = trcg_dict.get('urgent_missions', [])
        if len(urgents) > self._SKIP_MAX_URGENT:
            return False

        # 条件 3: 所有 bottleneck 压力低
        pressure = trcg_dict.get('bottleneck_pressure', {})
        for _res, util in pressure.items():
            if isinstance(util, (int, float)) and util >= self._SKIP_PRESSURE_THRESH:
                return False

        return True

    def _build_skip_return(
        self,
        now: int,
        t0: float,
        trcg_dict: Dict[str, Any],
        config: Config,
    ) -> Tuple[MetaParams, None]:
        """
        跳过 LLM 时的快速返回路径（无冲突/低压力场景）。

        TRCG selective freezing 在 simulator 层面自动将非 unlock mission 冻结。
        skip 路径使用 unlock=空元组 → 全部 mission 冻结 → 零 drift。
        epsilon 使用标准值，避免与 fixed_tuned 基线产生级联偏差。
        """
        self._skip_count += 1

        # freeze_horizon=96 slots (24h)，匹配 FT best_params
        freeze_h_slots = max(96, config.freeze_horizon)
        # 使用标准 epsilon（selective freezing 已保证低 drift，无需放宽）
        epsilon_solver = max(0.10, config.default_epsilon_solver)

        meta = MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=freeze_h_slots,
            use_two_stage=True,
            epsilon_solver=epsilon_solver,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
            # 核心改动：全部锚定！无冲突时保持所有 mission 原位
            # 空元组 = 无 mission 被 unlock = 全部 mission 施加软锚点约束
            unlock_mission_ids=(),
            root_cause_mission_id=None,
            secondary_root_cause_mission_id=None,
            decision_source="skip_no_conflict",
            fallback_reason=None,
            attempt_idx=0,
        )

        # 构建简化版日志
        wall_ms = int((_time.time() - t0) * 1000)
        step_log = build_repair_step_log(
            now_slot=now,
            trcg_dict=trcg_dict,
            llm_raw_output=None,
            llm_http_ok=False,
            llm_parse_ok=False,
            llm_decision_ok=False,
            llm_error={},
            decision=None,
            decision_source="skip_no_conflict",
            fallback_reason="no_conflict_low_pressure",
            wall_clock_ms=wall_ms,
        )
        self._step_logs.append(step_log)
        self._pending_step_log = step_log
        self._pending_now = now

        logger.info(
            "t=%d source=skip_no_conflict (skipped LLM) wall=%dms",
            now, wall_ms,
        )

        return meta, None

    # ------------------------------------------------------------------
    # 状态指纹跳过 — 若 TRCG 核心状态不变，复用上次 LLM 决策
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_state_fingerprint(
        trcg_dict: Dict[str, Any],
        active_mission_ids: set,
    ) -> str:
        """
        从 TRCG 诊断中提取核心特征，生成指纹字符串。

        指纹组成（不含 now_slot / horizon_end_slot 等时间相关字段）：
        1. sorted urgent mission IDs
        2. sorted conflict pairs (mission_a, mission_b, resource)
        3. sorted active mission IDs
        4. bottleneck pressure (四舍五入到 1 位小数)
        5. disturbance type

        若连续两步指纹相同，说明 TRCG 核心状态未变，LLM 输入等价，
        可安全复用上次决策。
        """
        # urgents
        urgents = trcg_dict.get('urgent_missions', [])
        urgent_ids = sorted(u.get('mission_id', '') for u in urgents) if urgents else []

        # conflicts
        conflicts = trcg_dict.get('top_conflicts', [])
        conflict_keys = sorted(
            (c.get('mission_a', ''), c.get('mission_b', ''), c.get('resource', ''))
            for c in conflicts
        ) if conflicts else []

        # active missions
        active_sorted = sorted(active_mission_ids) if active_mission_ids else []

        # pressure (rounded to 1 decimal)
        pressure = trcg_dict.get('bottleneck_pressure', {})
        pressure_rounded = {k: round(v, 1) for k, v in sorted(pressure.items())
                           if isinstance(v, (int, float))}

        # disturbance type
        dist_summary = trcg_dict.get('disturbance_summary', {})
        dist_type = dist_summary.get('type', '')

        return f"{urgent_ids}|{conflict_keys}|{active_sorted}|{pressure_rounded}|{dist_type}"

    def _build_stable_skip_return(
        self,
        now: int,
        t0: float,
        trcg_dict: Dict[str, Any],
        config: Config,
        prev_decision: RepairDecision,
        prev_source: str,
        active_mission_ids: set,
        started_mission_ids: set,
        completed_mission_ids: set,
    ) -> Tuple[MetaParams, None]:
        """
        状态稳定时复用上次 LLM 决策（跳过 ~40s 的 LLM 调用）。

        与 _build_skip_return 不同：这里使用上次 LLM 的 unlock_mission_ids，
        而不是全解锁。保持 LLM 决策质量。

        对 prev_decision.unlock_mission_ids 做过滤:
        - 去掉已不在 active 中的 mission（已 started/completed）
        """
        self._stable_skip_count += 1

        # 过滤 unlock_ids：去掉已不在 active 中的
        valid_unlock = [
            mid for mid in (prev_decision.unlock_mission_ids or [])
            if mid in active_mission_ids
        ]

        # ---- 扩展 unlock 集合（与 Step 6 逻辑一致：更严格的上限）----
        _MIN_UNLOCK = 3
        _MAX_UNLOCK_RATIO = 0.25
        _max_unlock = max(_MIN_UNLOCK, int(len(active_mission_ids) * _MAX_UNLOCK_RATIO))
        _unlock_set = set(valid_unlock)
        _eligible = active_mission_ids - started_mission_ids - completed_mission_ids
        _target_unlock = min(len(_eligible), _MIN_UNLOCK)

        if len(_unlock_set) < _target_unlock:
            conflicts = trcg_dict.get('top_conflicts', [])
            for c in sorted(conflicts, key=lambda x: -float(x.get('severity', 0))):
                for m in (c.get('a', ''), c.get('b', '')):
                    if m in _eligible and m not in _unlock_set and len(_unlock_set) < _max_unlock:
                        _unlock_set.add(m)
                if len(_unlock_set) >= _target_unlock:
                    break

        if len(_unlock_set) < _target_unlock:
            urgents = trcg_dict.get('urgent_missions', [])
            for u in sorted(urgents, key=lambda x: x.get('urgency_score', 9999)):
                mid = u.get('mission_id', '')
                if mid in _eligible and mid not in _unlock_set and len(_unlock_set) < _max_unlock:
                    _unlock_set.add(mid)
                if len(_unlock_set) >= _target_unlock:
                    break

        valid_unlock = list(_unlock_set)

        # 复用上次 LLM 的 root_cause 信息
        if not valid_unlock:
            root_cause = None
        else:
            root_cause = prev_decision.root_cause_mission_id
            if root_cause and root_cause not in active_mission_ids:
                root_cause = valid_unlock[0] if valid_unlock else None

        # freeze_horizon=96 slots (24h)，匹配 FT best_params
        freeze_h_slots = max(96, config.freeze_horizon)
        # 自适应 epsilon（与 Step 5 逻辑一致）
        # selective freezing 已在 simulator 层保证低 drift，
        # epsilon 可适度放宽让 Stage 2 有更多 delay 预算优化未冻结 mission。
        _conflicts = trcg_dict.get('top_conflicts', [])
        _urgents = trcg_dict.get('urgent_missions', [])
        _pressure = trcg_dict.get('bottleneck_pressure', {})
        _max_p = max(_pressure.values(), default=0.0) if _pressure else 0.0
        _base_eps = max(0.10, config.default_epsilon_solver)
        if len(_conflicts) == 0 and len(_urgents) <= 1 and _max_p < 0.6:
            _adaptive_eps = min(0.12, _base_eps + 0.02)
        elif len(_conflicts) <= 2 and _max_p < 0.8:
            _adaptive_eps = min(0.11, _base_eps + 0.01)
        else:
            _adaptive_eps = _base_eps
        epsilon_solver = min(0.12, _adaptive_eps)

        # 始终使用锚点约束（空 = 全锚定，非空 = 仅解锁指定 mission）
        use_unlock = tuple(valid_unlock)

        meta = MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=freeze_h_slots,
            use_two_stage=True,
            epsilon_solver=epsilon_solver,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
            unlock_mission_ids=use_unlock,
            root_cause_mission_id=root_cause,
            secondary_root_cause_mission_id=prev_decision.secondary_root_cause_mission_id,
            decision_source="skip_stable_state",
            fallback_reason=None,
            attempt_idx=0,
        )

        wall_ms = int((_time.time() - t0) * 1000)
        step_log = build_repair_step_log(
            now_slot=now,
            trcg_dict=trcg_dict,
            llm_raw_output=None,
            llm_http_ok=False,
            llm_parse_ok=False,
            llm_decision_ok=False,
            llm_error={},
            decision=prev_decision,
            decision_source="skip_stable_state",
            fallback_reason="state_unchanged_reuse_prev",
            wall_clock_ms=wall_ms,
        )
        self._step_logs.append(step_log)
        self._pending_step_log = step_log
        self._pending_now = now

        logger.info(
            "t=%d source=skip_stable_state (reuse prev LLM decision) wall=%dms",
            now, wall_ms,
        )

        return meta, None

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------

    def _write_step_log(self, log: RepairStepLog, now: int) -> None:
        """将单步日志写入 JSON 文件。"""
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            path = os.path.join(
                self._log_dir,
                f"repair_step_{self._episode_id}_t{now:04d}.json"
            )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(log.to_json())
        except Exception as exc:
            logger.warning("Failed to write step log: %s", exc)

    def get_step_logs(self) -> List[RepairStepLog]:
        return list(self._step_logs)

    # ------------------------------------------------------------------
    # Solver 结果回写（由 simulator 在 solver 完成后调用）
    # ------------------------------------------------------------------

    def update_pending_log_with_solver_result(
        self,
        solver_result=None,
        chain_result: Optional[FallbackChainResult] = None,
    ) -> None:
        """
        将 solver 求解结果回写到本步的 repair_step 日志并落盘。

        由 simulator 在 solve_v2_1() / _solve_with_trcg_fallback() 完成后调用。
        填充 solver_status, solver_time_ms, total_solver_calls,
        fallback_attempts, final_attempt_name, anchor_fix_* 等字段。

        Parameters
        ----------
        solver_result : SolverResult — 直接求解结果（无回退链时）
        chain_result : FallbackChainResult — 回退链结果（有回退链时）
        """
        log = self._pending_step_log
        if log is None:
            return

        if chain_result is not None:
            from dataclasses import asdict as _asdict
            log.fallback_attempts = [_asdict(a) for a in chain_result.attempts]
            log.final_attempt_name = chain_result.final_attempt_name
            log.total_solver_calls = chain_result.total_solver_calls
            sr = chain_result.solver_result
            if sr is not None:
                log.solver_status = sr.status.value if hasattr(sr.status, 'value') else str(sr.status)
                log.solver_time_ms = getattr(sr, 'solve_time_ms', 0)
                log.anchor_fix_applied = getattr(sr, 'anchor_fix_applied_count', 0)
                log.anchor_fix_skipped = getattr(sr, 'anchor_fix_skipped_count', 0)
                log.anchor_fix_applied_missions = getattr(sr, 'anchor_fix_applied_missions', [])
                log.anchor_fix_applied_vars = getattr(sr, 'anchor_fix_applied_vars', {})
                log.anchor_fix_skip_reason = getattr(sr, 'anchor_fix_skip_reason', '')
        elif solver_result is not None:
            log.total_solver_calls = 1
            log.final_attempt_name = "initial"
            log.solver_status = solver_result.status.value if hasattr(solver_result.status, 'value') else str(solver_result.status)
            log.solver_time_ms = getattr(solver_result, 'solve_time_ms', 0)
            log.anchor_fix_applied = getattr(solver_result, 'anchor_fix_applied_count', 0)
            log.anchor_fix_skipped = getattr(solver_result, 'anchor_fix_skipped_count', 0)
            log.anchor_fix_applied_missions = getattr(solver_result, 'anchor_fix_applied_missions', [])
            log.anchor_fix_applied_vars = getattr(solver_result, 'anchor_fix_applied_vars', {})
            log.anchor_fix_skip_reason = getattr(solver_result, 'anchor_fix_skip_reason', '')

        # 落盘
        if self._enable_logging and self._log_dir and self._pending_now is not None:
            self._write_step_log(log, self._pending_now)

        self._pending_step_log = None
        self._pending_now = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "llm_ok_count": self._llm_ok_count,
            "heuristic_count": self._heuristic_count,
            "skip_count": self._skip_count,
            "stable_skip_count": self._stable_skip_count,
            "has_llm_client": self._llm_client is not None,
        }

    def get_llm_stats(self) -> Dict[str, Any]:
        """返回 LLM 统计信息，与 run_experiments.run_single_episode 对接。"""
        stats: Dict[str, Any] = {
            "call_count": self._call_count,
            "llm_ok_count": self._llm_ok_count,
            "fallback_count": self._heuristic_count,
            "skip_count": self._skip_count,
            "stable_skip_count": self._stable_skip_count,
            "total_latency_ms": 0,
            "total_tokens": 0,
            "cache_hit_rate": 0.0,
        }
        # 若底层有 LLMClient，汇总其详细统计
        if self._llm_client is not None and hasattr(self._llm_client, 'get_stats'):
            client_stats = self._llm_client.get_stats()
            stats["total_latency_ms"] = client_stats.get("total_latency_ms", 0)
            stats["total_tokens"] = client_stats.get("total_tokens", 0)
            stats["cache_hit_rate"] = client_stats.get("cache_hit_rate", 0.0)
            stats["llm_client"] = client_stats  # 详细子字段
        return stats


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_trcg_repair_policy(
    log_dir: str = "llm_logs",
    llm_config: Optional[Any] = None,
    episode_id: str = "",
    **kwargs,
) -> TRCGRepairPolicy:
    """
    创建 TRCGRepairPolicy 实例。

    若提供 llm_config（LLMConfig），自动创建 LLMClient。
    否则走纯启发式模式。
    """
    client = None
    if llm_config is not None and HAS_LLM_CLIENT:
        client = LLMClient(llm_config)

    return TRCGRepairPolicy(
        llm_client=client,
        policy_name=kwargs.pop("policy_name", "trcg_repair"),
        log_dir=log_dir,
        enable_logging=True,
        episode_id=episode_id,
        **kwargs,
    )
