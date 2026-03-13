"""
ALNSRepairPolicy - lightweight ALNS unlock-set search baseline.

This policy reuses the same TRCG candidate pool, CP-SAT local repair, and
fallback chain as GARepairPolicy. The only change is how unlock_mission_ids are
chosen: a small ALNS loop searches over fixed-size unlock sets.
"""

import json
import logging
import os
import random
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from config import Config
from features import build_trcg_summary
from metrics import compute_delay_metrics_ops, compute_plan_drift_ops, compute_weighted_tardiness_ops
from policies.base import BasePolicy, MetaParams
from policies.policy_ga_repair import GA_BIG_M, _build_candidate_pool
from policies.policy_llm_repair import FREEZE_HOURS_TO_SLOTS, heuristic_repair_decision
from solver_cpsat import (
    Mission,
    OpAssignment,
    PlanV2_1,
    Resource,
    SolveStatus,
    SolverConfigV2_1,
    compute_frozen_ops,
    solve_v2_1,
)

logger = logging.getLogger("ALNSRepairPolicy")

ALNS_DEFAULT_K: int = 4
ALNS_DEFAULT_CANDIDATE_POOL_SIZE: int = 15
ALNS_DEFAULT_MAX_ITERATIONS: int = 10
ALNS_DEFAULT_ACCEPT_WORSE_PROB: float = 0.10
ALNS_DEFAULT_DRIFT_LAMBDA: float = 5.0
ALNS_DEFAULT_EVAL_TIMEOUT_S: float = 0.5
ALNS_DEFAULT_FINAL_TIMEOUT_S: Optional[float] = None
ALNS_DEFAULT_EVAL_CP_WORKERS: int = 1
ALNS_DEFAULT_FINAL_CP_WORKERS: Optional[int] = None


@dataclass
class ALNSEvalResult:
    unlock_set: Tuple[str, ...]
    score: float
    solve_time_ms: int
    feasible: bool
    avg_delay: float = 0.0
    weighted_tardiness: float = 0.0
    drift: float = 0.0


@dataclass
class ALNSStats:
    iterations_run: int = 0
    total_evaluations: int = 0
    cache_hits: int = 0
    accepted_worse: int = 0
    feasible_count: int = 0
    infeasible_count: int = 0
    init_source: str = ""
    init_score: float = GA_BIG_M
    best_score: float = GA_BIG_M
    best_unlock_set: Optional[Tuple[str, ...]] = None
    unlock_size: int = 0
    candidate_pool_size: int = 0
    wall_time_ms: int = 0
    final_recompute_ms: int = 0
    final_recompute_score: float = GA_BIG_M
    destroy_usage: Dict[str, int] = field(default_factory=dict)
    destroy_improve: Dict[str, int] = field(default_factory=dict)
    repair_usage: Dict[str, int] = field(default_factory=dict)
    repair_improve: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations_run": self.iterations_run,
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "accepted_worse": self.accepted_worse,
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "init_source": self.init_source,
            "init_score": self.init_score if self.init_score < GA_BIG_M else "BIG_M",
            "best_score": self.best_score if self.best_score < GA_BIG_M else "BIG_M",
            "best_unlock_set": list(self.best_unlock_set) if self.best_unlock_set else [],
            "unlock_size": self.unlock_size,
            "candidate_pool_size": self.candidate_pool_size,
            "wall_time_ms": self.wall_time_ms,
            "final_recompute_ms": self.final_recompute_ms,
            "final_recompute_score": (
                self.final_recompute_score
                if self.final_recompute_score < GA_BIG_M
                else "BIG_M"
            ),
            "destroy_usage": dict(self.destroy_usage),
            "destroy_improve": dict(self.destroy_improve),
            "repair_usage": dict(self.repair_usage),
            "repair_improve": dict(self.repair_improve),
        }


def _build_priority_scores(
    trcg_dict: Dict[str, Any],
    candidate_pool: List[str],
    heuristic_unlock: List[str],
    root_cause_id: Optional[str],
    secondary_root_id: Optional[str],
) -> Dict[str, float]:
    scores = {mid: 0.0 for mid in candidate_pool}

    for idx, mid in enumerate(candidate_pool):
        scores[mid] += max(0.0, float(len(candidate_pool) - idx)) * 0.05

    for idx, item in enumerate(trcg_dict.get("urgent_missions", []) or []):
        mid = item.get("mission_id", "")
        if mid in scores:
            urgency = float(item.get("urgency_score", 0.0) or item.get("score", 0.0) or 0.0)
            scores[mid] += 10.0 + urgency + max(0.0, 5.0 - idx)

    for item in trcg_dict.get("top_conflicts", []) or []:
        severity = float(item.get("severity", 0.0) or 0.0)
        for mid in (item.get("a", ""), item.get("b", "")):
            if mid in scores:
                scores[mid] += max(0.0, severity)

    if root_cause_id in scores:
        scores[root_cause_id] += 20.0
    if secondary_root_id in scores:
        scores[secondary_root_id] += 8.0

    for idx, mid in enumerate(heuristic_unlock):
        if mid in scores:
            scores[mid] += max(0.0, 6.0 - idx)

    return scores


def _normalize_unlock_set(
    unlock_ids: List[str],
    candidate_pool: List[str],
    target_k: int,
    priority_scores: Dict[str, float],
    rng: random.Random,
    fill_mode: str = "urgent",
) -> Tuple[str, ...]:
    seen: Set[str] = set()
    cleaned: List[str] = []
    pool_set = set(candidate_pool)

    for mid in unlock_ids:
        if mid in pool_set and mid not in seen:
            cleaned.append(mid)
            seen.add(mid)

    if len(cleaned) > target_k:
        cleaned = sorted(
            cleaned,
            key=lambda mid: (-priority_scores.get(mid, 0.0), mid),
        )[:target_k]

    remaining = [mid for mid in candidate_pool if mid not in seen]
    if len(cleaned) < target_k and remaining:
        if fill_mode == "random":
            rng.shuffle(remaining)
        else:
            remaining = sorted(
                remaining,
                key=lambda mid: (-priority_scores.get(mid, 0.0), mid),
            )
        cleaned.extend(remaining[: max(0, target_k - len(cleaned))])

    if not cleaned and candidate_pool:
        cleaned.append(candidate_pool[0])

    return tuple(sorted(cleaned))


def _evaluate_unlock_set(
    unlock_set_frozen: FrozenSet[str],
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    solver_config_dict: Dict[str, Any],
    now: int,
    completed_ops: Set[str],
    started_ops: Set[str],
    drift_lambda: float,
    kappa_win: float,
    kappa_seq: float,
) -> ALNSEvalResult:
    unlock_set = set(unlock_set_frozen)
    try:
        solver_config = SolverConfigV2_1(**solver_config_dict)
        result = solve_v2_1(
            missions=missions,
            resources=resources,
            horizon=horizon,
            prev_plan=prev_plan,
            frozen_ops=frozen_ops,
            config=solver_config,
            unlock_mission_ids=unlock_set,
            now=now,
        )
    except Exception as exc:
        logger.warning("ALNS eval exception for %s: %s", sorted(unlock_set_frozen), exc)
        return ALNSEvalResult(
            unlock_set=tuple(sorted(unlock_set_frozen)),
            score=GA_BIG_M,
            solve_time_ms=0,
            feasible=False,
        )

    if result.status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) or result.plan is None:
        return ALNSEvalResult(
            unlock_set=tuple(sorted(unlock_set_frozen)),
            score=GA_BIG_M,
            solve_time_ms=result.solve_time_ms,
            feasible=False,
        )

    _, avg_delay, _, _, _ = compute_delay_metrics_ops(
        result.plan.op_assignments,
        missions,
        horizon_slots=horizon,
    )
    weighted_tardiness = compute_weighted_tardiness_ops(
        result.plan.op_assignments,
        missions,
        horizon_slots=horizon,
    )
    drift, *_ = compute_plan_drift_ops(
        prev_plan,
        result.plan,
        completed_ops,
        started_ops,
        set(frozen_ops.keys()),
        missions,
        kappa_win=kappa_win,
        kappa_seq=kappa_seq,
    )
    score = weighted_tardiness + drift_lambda * drift
    return ALNSEvalResult(
        unlock_set=tuple(sorted(unlock_set_frozen)),
        score=score,
        solve_time_ms=result.solve_time_ms,
        feasible=True,
        avg_delay=avg_delay,
        weighted_tardiness=weighted_tardiness,
        drift=drift,
    )


def run_alns_search(
    candidate_pool: List[str],
    heuristic_seed: List[str],
    priority_scores: Dict[str, float],
    K: int,
    max_iterations: int,
    accept_worse_prob: float,
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    solver_config_dict: Dict[str, Any],
    now: int,
    completed_ops: Set[str],
    started_ops: Set[str],
    drift_lambda: float,
    kappa_win: float,
    kappa_seq: float,
    alns_seed: Optional[int] = None,
) -> ALNSStats:
    rng = random.Random(alns_seed)
    t0 = _time.time()
    stats = ALNSStats(
        unlock_size=min(K, len(candidate_pool)),
        candidate_pool_size=len(candidate_pool),
    )
    if not candidate_pool:
        stats.wall_time_ms = int((_time.time() - t0) * 1000)
        return stats

    target_k = min(K, len(candidate_pool))
    eval_cache: Dict[FrozenSet[str], ALNSEvalResult] = {}

    def evaluate(unlock_ids: Tuple[str, ...]) -> ALNSEvalResult:
        key = frozenset(unlock_ids)
        cached = eval_cache.get(key)
        if cached is not None:
            stats.cache_hits += 1
            return cached

        result = _evaluate_unlock_set(
            unlock_set_frozen=key,
            missions=missions,
            resources=resources,
            horizon=horizon,
            prev_plan=prev_plan,
            frozen_ops=frozen_ops,
            solver_config_dict=solver_config_dict,
            now=now,
            completed_ops=completed_ops,
            started_ops=started_ops,
            drift_lambda=drift_lambda,
            kappa_win=kappa_win,
            kappa_seq=kappa_seq,
        )
        eval_cache[key] = result
        stats.total_evaluations += 1
        if result.feasible:
            stats.feasible_count += 1
        else:
            stats.infeasible_count += 1
        return result

    heuristic_tuple = _normalize_unlock_set(
        unlock_ids=heuristic_seed,
        candidate_pool=candidate_pool,
        target_k=target_k,
        priority_scores=priority_scores,
        rng=rng,
        fill_mode="urgent",
    )
    random_tuple = _normalize_unlock_set(
        unlock_ids=rng.sample(candidate_pool, target_k),
        candidate_pool=candidate_pool,
        target_k=target_k,
        priority_scores=priority_scores,
        rng=rng,
        fill_mode="random",
    )

    heuristic_eval = evaluate(heuristic_tuple)
    random_eval = evaluate(random_tuple)
    if heuristic_eval.score <= random_eval.score:
        current = heuristic_eval
        stats.init_source = "heuristic_seed"
    else:
        current = random_eval
        stats.init_source = "random_seed"

    best = current
    stats.init_score = current.score
    stats.best_score = best.score
    stats.best_unlock_set = best.unlock_set

    destroy_ops = ("random_remove", "low_priority_remove")
    repair_ops = ("random_add", "urgent_add")
    stats.destroy_usage = {name: 0 for name in destroy_ops}
    stats.destroy_improve = {name: 0 for name in destroy_ops}
    stats.repair_usage = {name: 0 for name in repair_ops}
    stats.repair_improve = {name: 0 for name in repair_ops}

    for iteration in range(max_iterations):
        stats.iterations_run = iteration + 1
        destroy_name = rng.choice(destroy_ops)
        repair_name = rng.choice(repair_ops)
        stats.destroy_usage[destroy_name] += 1
        stats.repair_usage[repair_name] += 1

        current_list = list(current.unlock_set)
        if destroy_name == "random_remove":
            remove_count = min(len(current_list) - 1, rng.randint(1, 2)) if len(current_list) > 1 else 0
            if remove_count > 0:
                keep = current_list[:]
                rng.shuffle(keep)
                partial = keep[remove_count:]
            else:
                partial = current_list[:]
        else:
            remove_count = min(len(current_list) - 1, rng.randint(1, 2)) if len(current_list) > 1 else 0
            ranked = sorted(current_list, key=lambda mid: (priority_scores.get(mid, 0.0), mid))
            partial = ranked[remove_count:] if remove_count > 0 else ranked

        if repair_name == "random_add":
            candidate = _normalize_unlock_set(
                unlock_ids=partial,
                candidate_pool=candidate_pool,
                target_k=target_k,
                priority_scores=priority_scores,
                rng=rng,
                fill_mode="random",
            )
        else:
            working = list(dict.fromkeys(partial))
            remaining = [mid for mid in candidate_pool if mid not in working]
            while len(working) < target_k and remaining:
                ranked_remaining = sorted(
                    remaining,
                    key=lambda mid: (-priority_scores.get(mid, 0.0), mid),
                )
                shortlist = ranked_remaining[: min(3, len(ranked_remaining))]
                best_add_mid: Optional[str] = None
                best_add_eval: Optional[ALNSEvalResult] = None
                for add_mid in shortlist:
                    trial = _normalize_unlock_set(
                        unlock_ids=working + [add_mid],
                        candidate_pool=candidate_pool,
                        target_k=target_k,
                        priority_scores=priority_scores,
                        rng=rng,
                        fill_mode="urgent",
                    )
                    trial_eval = evaluate(trial)
                    if best_add_eval is None or (
                        trial_eval.score,
                        -priority_scores.get(add_mid, 0.0),
                        add_mid,
                    ) < (
                        best_add_eval.score,
                        -priority_scores.get(best_add_mid or "", 0.0),
                        best_add_mid or "",
                    ):
                        best_add_mid = add_mid
                        best_add_eval = trial_eval

                if best_add_mid is None:
                    break
                working.append(best_add_mid)
                remaining.remove(best_add_mid)

            candidate = _normalize_unlock_set(
                unlock_ids=working,
                candidate_pool=candidate_pool,
                target_k=target_k,
                priority_scores=priority_scores,
                rng=rng,
                fill_mode="urgent",
            )

        candidate_eval = evaluate(candidate)

        if candidate_eval.score + 1e-9 < best.score:
            best = candidate_eval
            stats.best_score = best.score
            stats.best_unlock_set = best.unlock_set
            stats.destroy_improve[destroy_name] += 1
            stats.repair_improve[repair_name] += 1

        accept = False
        if candidate_eval.score + 1e-9 < current.score:
            accept = True
        elif rng.random() < accept_worse_prob:
            accept = True
            stats.accepted_worse += 1

        if accept:
            current = candidate_eval

    stats.wall_time_ms = int((_time.time() - t0) * 1000)
    return stats


class ALNSRepairPolicy(BasePolicy):
    def __init__(
        self,
        policy_name: str = "alns_repair",
        K: int = ALNS_DEFAULT_K,
        candidate_pool_size: int = ALNS_DEFAULT_CANDIDATE_POOL_SIZE,
        max_iterations: int = ALNS_DEFAULT_MAX_ITERATIONS,
        accept_worse_prob: float = ALNS_DEFAULT_ACCEPT_WORSE_PROB,
        drift_lambda: float = ALNS_DEFAULT_DRIFT_LAMBDA,
        eval_timeout_s: float = ALNS_DEFAULT_EVAL_TIMEOUT_S,
        final_timeout_s: Optional[float] = ALNS_DEFAULT_FINAL_TIMEOUT_S,
        eval_cp_workers: int = ALNS_DEFAULT_EVAL_CP_WORKERS,
        final_cp_workers: Optional[int] = ALNS_DEFAULT_FINAL_CP_WORKERS,
        w_delay: float = 10.0,
        w_shift: float = 1.0,
        w_switch: float = 5.0,
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        episode_id: str = "",
    ):
        self._policy_name = policy_name
        self._K = K
        self._candidate_pool_size = candidate_pool_size
        self._max_iterations = max_iterations
        self._accept_worse_prob = accept_worse_prob
        self._drift_lambda = drift_lambda
        self._eval_timeout_s = eval_timeout_s
        self._final_timeout_s = final_timeout_s
        self._eval_cp_workers = eval_cp_workers
        self._final_cp_workers = final_cp_workers
        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch
        self._log_dir = log_dir
        self._enable_logging = enable_logging
        self._episode_id = episode_id

        self._prev_window_slots: Optional[Dict[str, Set[int]]] = None
        self._call_count = 0
        self._alns_stats_history: List[ALNSStats] = []

    @property
    def name(self) -> str:
        return self._policy_name

    def reset(self) -> None:
        self._prev_window_slots = None
        self._call_count = 0
        self._alns_stats_history = []

    def set_episode_id(self, episode_id: str) -> None:
        self._episode_id = episode_id

    def decide(
        self,
        state: Any,
        now: int,
        config: Config,
    ) -> Tuple[MetaParams, None]:
        t0 = _time.time()
        self._call_count += 1

        freeze_slots = config.freeze_horizon
        frozen_ops = compute_frozen_ops(
            state.current_plan,
            now,
            freeze_slots,
            getattr(state, "started_ops", set()),
            getattr(state, "completed_ops", set()),
        )

        trcg = build_trcg_summary(
            missions=state.missions,
            resources=state.resources,
            plan=state.current_plan,
            now=now,
            config=config,
            started_ops=getattr(state, "started_ops", set()),
            completed_ops=getattr(state, "completed_ops", set()),
            actual_durations=getattr(state, "actual_durations", {}),
            frozen_ops=frozen_ops,
            prev_window_slots=self._prev_window_slots,
        )
        trcg_dict = trcg.to_dict()

        started_ops: Set[str] = getattr(state, "started_ops", set())
        completed_ops: Set[str] = getattr(state, "completed_ops", set())
        started_mission_ids: Set[str] = set()
        completed_mission_ids: Set[str] = set()
        horizon_end = now + config.horizon_slots

        for mission in state.missions:
            launch = mission.get_launch_op()
            if launch and launch.op_id in completed_ops:
                completed_mission_ids.add(mission.mission_id)
                continue
            for op in mission.operations:
                if op.op_id in started_ops:
                    started_mission_ids.add(mission.mission_id)
                    break

        schedulable_ids = {
            mission.mission_id for mission in state.get_schedulable_missions(horizon_end)
        }
        active_mission_ids = schedulable_ids - started_mission_ids - completed_mission_ids
        if not active_mission_ids:
            active_mission_ids = schedulable_ids - completed_mission_ids

        candidate_pool = _build_candidate_pool(
            trcg_dict=trcg_dict,
            active_mission_ids=active_mission_ids,
            started_mission_ids=started_mission_ids,
            completed_mission_ids=completed_mission_ids,
            max_pool_size=self._candidate_pool_size,
        )

        heuristic_decision = None
        heuristic_seed: List[str] = []
        fallback_reason: Optional[str] = None
        root_cause_id: Optional[str] = None
        secondary_root_id: Optional[str] = None

        if active_mission_ids:
            heuristic_decision = heuristic_repair_decision(
                trcg_dict=trcg_dict,
                active_mission_ids=active_mission_ids,
                started_mission_ids=started_mission_ids,
                completed_mission_ids=completed_mission_ids,
                fallback_reason="alns_seed",
            )
            heuristic_seed = list(heuristic_decision.unlock_mission_ids[: self._K])
            root_cause_id = heuristic_decision.root_cause_mission_id
            secondary_root_id = heuristic_decision.secondary_root_cause_mission_id

        decision_source = "alns"
        alns_stats = ALNSStats()
        unlock_ids: Optional[List[str]] = None

        if candidate_pool and active_mission_ids:
            priority_scores = _build_priority_scores(
                trcg_dict=trcg_dict,
                candidate_pool=candidate_pool,
                heuristic_unlock=heuristic_seed,
                root_cause_id=root_cause_id,
                secondary_root_id=secondary_root_id,
            )
            solver_horizon = config.sim_total_slots
            solver_config_dict = {
                "horizon_slots": solver_horizon,
                "w_delay": self._w_delay,
                "w_shift": self._w_shift,
                "w_switch": self._w_switch,
                "time_limit_seconds": self._eval_timeout_s,
                "num_workers": self._eval_cp_workers,
                "op5_max_wait_slots": max(
                    0,
                    int(round(config.op5_max_wait_hours * 60 / config.slot_minutes)),
                ),
                "use_two_stage": config.use_two_stage_solver,
                "epsilon_solver": config.default_epsilon_solver,
                "kappa_win": config.default_kappa_win,
                "kappa_seq": config.default_kappa_seq,
                "stage1_time_ratio": config.stage1_time_ratio,
            }

            missions_to_schedule = state.get_schedulable_missions(horizon_end)
            if state.current_plan:
                for assign in state.current_plan.op_assignments:
                    if assign.op_id in started_ops and assign.op_id not in completed_ops:
                        mission = state.get_mission(assign.mission_id)
                        if mission and mission not in missions_to_schedule:
                            missions_to_schedule.append(mission)

            alns_stats = run_alns_search(
                candidate_pool=candidate_pool,
                heuristic_seed=heuristic_seed,
                priority_scores=priority_scores,
                K=self._K,
                max_iterations=self._max_iterations,
                accept_worse_prob=self._accept_worse_prob,
                missions=missions_to_schedule,
                resources=state.resources,
                horizon=solver_horizon,
                prev_plan=state.current_plan,
                frozen_ops=frozen_ops,
                solver_config_dict=solver_config_dict,
                now=now,
                completed_ops=completed_ops,
                started_ops=started_ops,
                drift_lambda=self._drift_lambda,
                kappa_win=config.default_kappa_win,
                kappa_seq=config.default_kappa_seq,
                alns_seed=self._call_count,
            )

            if alns_stats.best_score < GA_BIG_M and alns_stats.best_unlock_set:
                final_config_dict = dict(solver_config_dict)
                final_config_dict["time_limit_seconds"] = self._final_timeout_s or config.solver_timeout_s
                final_config_dict["num_workers"] = self._final_cp_workers or config.solver_num_workers

                t_recompute = _time.time()
                final_eval = _evaluate_unlock_set(
                    unlock_set_frozen=frozenset(alns_stats.best_unlock_set),
                    missions=missions_to_schedule,
                    resources=state.resources,
                    horizon=solver_horizon,
                    prev_plan=state.current_plan,
                    frozen_ops=frozen_ops,
                    solver_config_dict=final_config_dict,
                    now=now,
                    completed_ops=completed_ops,
                    started_ops=started_ops,
                    drift_lambda=self._drift_lambda,
                    kappa_win=config.default_kappa_win,
                    kappa_seq=config.default_kappa_seq,
                )
                alns_stats.final_recompute_score = final_eval.score
                alns_stats.final_recompute_ms = int((_time.time() - t_recompute) * 1000)
                if final_eval.score >= GA_BIG_M:
                    alns_stats.best_score = GA_BIG_M
                    fallback_reason = "alns_final_recompute_infeasible"
            else:
                fallback_reason = "alns_no_feasible_solution"
        else:
            fallback_reason = "no_active_missions_or_empty_pool"

        if alns_stats.best_score < GA_BIG_M and alns_stats.best_unlock_set:
            unlock_ids = list(alns_stats.best_unlock_set)
            if not root_cause_id:
                root_cause_id = unlock_ids[0] if unlock_ids else None
            if not secondary_root_id and len(unlock_ids) > 1:
                secondary_root_id = unlock_ids[1]
        else:
            decision_source = "heuristic_fallback"
            if heuristic_decision is not None:
                unlock_ids = list(heuristic_decision.unlock_mission_ids)
                root_cause_id = heuristic_decision.root_cause_mission_id
                secondary_root_id = heuristic_decision.secondary_root_cause_mission_id
            else:
                unlock_ids = []
                root_cause_id = ""
                fallback_reason = fallback_reason or "no_active_missions"

        pressure = trcg_dict.get("bottleneck_pressure", {})
        pad_pressure = pressure.get("pad_util", 0.0)
        num_urgent = len(trcg_dict.get("urgent_missions", []))
        if pad_pressure > 0.80 or num_urgent >= 3:
            freeze_h_hours = 0
            epsilon = 0.02
        elif pad_pressure > 0.50 or num_urgent >= 2:
            freeze_h_hours = 4
            epsilon = 0.05
        else:
            freeze_h_hours = 8
            epsilon = 0.05

        meta = MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=FREEZE_HOURS_TO_SLOTS.get(freeze_h_hours, config.freeze_horizon),
            use_two_stage=True,
            epsilon_solver=epsilon,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
            unlock_mission_ids=tuple(unlock_ids) if unlock_ids else None,
            root_cause_mission_id=root_cause_id,
            secondary_root_cause_mission_id=secondary_root_id,
            decision_source=decision_source,
            fallback_reason=fallback_reason,
            attempt_idx=0,
        )

        wall_ms = int((_time.time() - t0) * 1000)
        alns_stats.wall_time_ms = wall_ms
        self._alns_stats_history.append(alns_stats)

        if self._enable_logging and self._log_dir:
            self._write_alns_log(now, alns_stats, meta, wall_ms)

        logger.info(
            "t=%d src=%s score=%.2f unlock=%s iters=%d evals=%d cache=%d wall=%dms",
            now,
            decision_source,
            alns_stats.best_score if alns_stats.best_score < GA_BIG_M else float("inf"),
            unlock_ids,
            alns_stats.iterations_run,
            alns_stats.total_evaluations,
            alns_stats.cache_hits,
            wall_ms,
        )
        return meta, None

    def _write_alns_log(self, now: int, alns_stats: ALNSStats, meta: MetaParams, wall_ms: int) -> None:
        if not self._log_dir:
            return
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            payload = {
                "policy_name": self._policy_name,
                "episode_id": self._episode_id,
                "call_count": self._call_count,
                "now_slot": now,
                "unlock_size": len(meta.unlock_mission_ids) if meta.unlock_mission_ids else 0,
                "decision_source": meta.decision_source,
                "fallback_reason": meta.fallback_reason,
                "unlock_mission_ids": list(meta.unlock_mission_ids) if meta.unlock_mission_ids else [],
                "alns_iterations": alns_stats.iterations_run,
                "alns_init_score": alns_stats.init_score if alns_stats.init_score < GA_BIG_M else None,
                "alns_best_score": alns_stats.best_score if alns_stats.best_score < GA_BIG_M else None,
                "alns_stats": alns_stats.to_dict(),
                "wall_time_ms": wall_ms,
            }
            path = os.path.join(
                self._log_dir,
                f"alns_step_{self._episode_id}_t{now:04d}.json",
            )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to write ALNS log: %s", exc)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "alns_stats_history": [item.to_dict() for item in self._alns_stats_history],
            "total_evaluations": sum(item.total_evaluations for item in self._alns_stats_history),
            "total_cache_hits": sum(item.cache_hits for item in self._alns_stats_history),
        }

    def get_alns_stats_history(self) -> List[ALNSStats]:
        return list(self._alns_stats_history)


def create_alns_repair_policy(
    log_dir: str = "llm_logs",
    episode_id: str = "",
    **kwargs,
) -> ALNSRepairPolicy:
    return ALNSRepairPolicy(
        policy_name=kwargs.pop("policy_name", "alns_repair"),
        log_dir=log_dir,
        enable_logging=True,
        episode_id=episode_id,
        **kwargs,
    )
