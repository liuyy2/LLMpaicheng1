#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate bucketed tables that answer:
1) Which state feature ranges tend to trigger deviations from the prompt baseline params
2) When deviations happen in those ranges, how episode-level metrics differ vs a baseline policy

Inputs:
- LLM decision logs: */episode_<seed>_llm_real/llm_decisions.jsonl
- Episode results CSV: results_per_episode.csv (contains llm_real + baseline rows)

Output:
- A single CSV table with one row per (feature, bin)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PROMPT_BASELINE_PARAMS = {
    "w_delay": 50.0,
    "w_shift": 0.0,
    "w_switch": 180.0,
    "freeze_horizon": 0,
}


@dataclass(frozen=True)
class EpisodeDelta:
    seed: int
    dataset: str
    disturbance_level: str
    baseline_match: str  # "paired" | "disturbance_mean" | "overall_mean" | "missing"
    delta_avg_delay: float
    delta_episode_drift: float
    delta_on_time_rate: float
    delta_combined: float


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _params_equal(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-9) -> bool:
    return (
        abs(_safe_float(a.get("w_delay")) - _safe_float(b.get("w_delay"))) <= tol
        and abs(_safe_float(a.get("w_shift")) - _safe_float(b.get("w_shift"))) <= tol
        and abs(_safe_float(a.get("w_switch")) - _safe_float(b.get("w_switch"))) <= tol
        and _safe_int(a.get("freeze_horizon")) == _safe_int(b.get("freeze_horizon"))
    )


def _bin_numeric(value: float, edges: Sequence[float]) -> str:
    """
    Bin numeric value using half-open intervals: [e0,e1),...,[e_{n-2},e_{n-1}), last is [e_{n-1}, +inf)
    """
    if not edges:
        return "all"
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if value >= lo and value < hi:
            return f"[{lo},{hi})"
    return f"[{edges[-1]},+inf)"


def _bin_discrete(value: int, mapping: Sequence[Tuple[Callable[[int], bool], str]]) -> str:
    for pred, label in mapping:
        if pred(value):
            return label
    return "other"


def _read_episode_results(
    results_csv: Path,
    dataset: str,
    llm_policy: str,
    baseline_policy: str,
    tuning_lambda: float,
) -> Dict[Tuple[int, str], EpisodeDelta]:
    """
    Returns dict keyed by (seed, disturbance_level) for llm_policy episodes.
    Deltas are computed as (llm_policy - baseline_policy) for avg_delay, episode_drift, on_time_rate, combined.
    """
    llm_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    baseline_rows: Dict[Tuple[int, str], Dict[str, Any]] = {}
    baseline_by_dist: Dict[str, List[Dict[str, Any]]] = {}
    baseline_all: List[Dict[str, Any]] = []

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != dataset:
                continue
            seed = _safe_int(row.get("seed"))
            dist = str(row.get("disturbance_level") or "")
            policy = str(row.get("policy_name") or "")
            key = (seed, dist)
            if policy == llm_policy:
                llm_rows[key] = row
            if policy == baseline_policy:
                baseline_rows[key] = row
                baseline_all.append(row)
                baseline_by_dist.setdefault(dist, []).append(row)

    def mean_row(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {
                "avg_delay": math.nan,
                "episode_drift": math.nan,
                "on_time_rate": math.nan,
            }
        return {
            "avg_delay": sum(_safe_float(r.get("avg_delay")) for r in rows) / len(rows),
            "episode_drift": sum(_safe_float(r.get("episode_drift")) for r in rows) / len(rows),
            "on_time_rate": sum(_safe_float(r.get("on_time_rate")) for r in rows) / len(rows),
        }

    baseline_means_by_dist = {d: mean_row(rows) for d, rows in baseline_by_dist.items()}
    baseline_mean_all = mean_row(baseline_all)

    deltas: Dict[Tuple[int, str], EpisodeDelta] = {}
    for key, llm in llm_rows.items():
        seed, dist = key
        baseline_match = "missing"
        base_avg_delay = math.nan
        base_drift = math.nan
        base_on_time = math.nan

        if key in baseline_rows:
            baseline_match = "paired"
            b = baseline_rows[key]
            base_avg_delay = _safe_float(b.get("avg_delay"), math.nan)
            base_drift = _safe_float(b.get("episode_drift"), math.nan)
            base_on_time = _safe_float(b.get("on_time_rate"), math.nan)
        elif dist in baseline_means_by_dist:
            baseline_match = "disturbance_mean"
            b = baseline_means_by_dist[dist]
            base_avg_delay = b["avg_delay"]
            base_drift = b["episode_drift"]
            base_on_time = b["on_time_rate"]
        elif baseline_all:
            baseline_match = "overall_mean"
            base_avg_delay = baseline_mean_all["avg_delay"]
            base_drift = baseline_mean_all["episode_drift"]
            base_on_time = baseline_mean_all["on_time_rate"]

        llm_avg_delay = _safe_float(llm.get("avg_delay"), math.nan)
        llm_drift = _safe_float(llm.get("episode_drift"), math.nan)
        llm_on_time = _safe_float(llm.get("on_time_rate"), math.nan)

        llm_combined = llm_avg_delay + tuning_lambda * llm_drift
        base_combined = base_avg_delay + tuning_lambda * base_drift

        deltas[key] = EpisodeDelta(
            seed=seed,
            dataset=dataset,
            disturbance_level=dist,
            baseline_match=baseline_match,
            delta_avg_delay=llm_avg_delay - base_avg_delay,
            delta_episode_drift=llm_drift - base_drift,
            delta_on_time_rate=llm_on_time - base_on_time,
            delta_combined=llm_combined - base_combined,
        )

    return deltas


def _iter_llm_decisions(log_root: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    Yields (seed, decision_obj) for each line in llm_decisions.jsonl.
    """
    # Accept roots that either directly contain episode_* dirs, or contain logs/episode_* dirs
    candidates: List[Path] = []
    if (log_root / "logs").exists():
        candidates.append(log_root / "logs")
    candidates.append(log_root)

    episode_re = re.compile(r"episode_(\d+)_llm_real$")
    for base in candidates:
        if not base.exists():
            continue
        for episode_dir in base.glob("episode_*_llm_real"):
            m = episode_re.match(episode_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            fp = episode_dir / "llm_decisions.jsonl"
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield seed, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _default_feature_specs() -> List[Tuple[str, Callable[[Any], str]]]:
    """
    Return (feature_name, binner(feature_value)->bin_label).
    """
    return [
        ("window_loss_pct", lambda v: _bin_numeric(_safe_float(v), [0.0, 0.05, 0.2, 0.5, 1.0])),
        ("window_remaining_pct", lambda v: _bin_numeric(_safe_float(v), [0.0, 0.2, 0.5, 0.8, 1.0])),
        ("pad_outage_overlap_hours", lambda v: _bin_numeric(_safe_float(v), [0.0, 0.1, 0.5, 1.0, 2.0])),
        (
            "pad_outage_task_count",
            lambda v: _bin_discrete(
                _safe_int(v),
                [
                    (lambda x: x == 0, "0"),
                    (lambda x: x == 1, "1"),
                    (lambda x: x == 2, "2"),
                    (lambda x: x >= 3, ">=3"),
                ],
            ),
        ),
        (
            "delay_increase_minutes",
            lambda v: _bin_discrete(
                int(round(_safe_float(v))),
                [
                    (lambda x: x == 0, "0"),
                    (lambda x: 1 <= x <= 10, "1-10"),
                    (lambda x: 11 <= x <= 30, "11-30"),
                    (lambda x: 31 <= x <= 60, "31-60"),
                    (lambda x: x > 60, ">60"),
                ],
            ),
        ),
        (
            "num_urgent_tasks",
            lambda v: _bin_discrete(
                _safe_int(v),
                [
                    (lambda x: x == 0, "0"),
                    (lambda x: 1 <= x <= 2, "1-2"),
                    (lambda x: 3 <= x <= 5, "3-5"),
                    (lambda x: x >= 6, ">=6"),
                ],
            ),
        ),
        (
            "recent_shift_count",
            lambda v: _bin_discrete(
                _safe_int(v),
                [
                    (lambda x: x == 0, "0"),
                    (lambda x: 1 <= x <= 2, "1-2"),
                    (lambda x: 3 <= x <= 5, "3-5"),
                    (lambda x: x >= 6, ">=6"),
                ],
            ),
        ),
        (
            "recent_switch_count",
            lambda v: _bin_discrete(
                _safe_int(v),
                [
                    (lambda x: x == 0, "0"),
                    (lambda x: 1 <= x <= 2, "1-2"),
                    (lambda x: x >= 3, ">=3"),
                ],
            ),
        ),
    ]


@dataclass
class BinAgg:
    feature: str
    bin_label: str
    decisions_total: int = 0
    decisions_deviation: int = 0
    episodes_in_bin: Set[Tuple[int, str]] = None  # (seed, disturbance_level)
    episodes_with_dev_in_bin: Set[Tuple[int, str]] = None
    dev_param_sum: Dict[str, float] = None
    dev_param_count: int = 0

    def __post_init__(self) -> None:
        if self.episodes_in_bin is None:
            self.episodes_in_bin = set()
        if self.episodes_with_dev_in_bin is None:
            self.episodes_with_dev_in_bin = set()
        if self.dev_param_sum is None:
            self.dev_param_sum = {"w_delay": 0.0, "w_shift": 0.0, "w_switch": 0.0, "freeze_horizon": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=str,
        default="results_llm_chat_30_env1/logs",
        help="Directory containing episode_*_llm_real/llm_decisions.jsonl (or a parent that contains logs/).",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="results_30_env1/results_per_episode.csv",
        help="CSV containing per-episode metrics for llm_real and baseline.",
    )
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--llm-policy", type=str, default="llm_real")
    parser.add_argument("--baseline-policy", type=str, default="fixed_tuned")
    parser.add_argument("--lambda", dest="tuning_lambda", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=str,
        default="results_30_env1/figures/llm_deviation_bucket_table.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    results_csv = Path(args.results_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episode_deltas = _read_episode_results(
        results_csv=results_csv,
        dataset=args.dataset,
        llm_policy=args.llm_policy,
        baseline_policy=args.baseline_policy,
        tuning_lambda=float(args.tuning_lambda),
    )

    feature_specs = _default_feature_specs()
    aggs: Dict[Tuple[str, str], BinAgg] = {}
    total_decisions_all = 0
    total_deviation_decisions_all = 0
    episodes_seen: Set[Tuple[int, str]] = set()
    episodes_with_any_dev: Set[Tuple[int, str]] = set()

    # Build a helper for mapping seed -> disturbance_level using episode_deltas keys
    dist_by_seed: Dict[int, str] = {}
    for (seed, dist), _ in episode_deltas.items():
        dist_by_seed[seed] = dist

    for seed, obj in _iter_llm_decisions(log_root):
        dist = dist_by_seed.get(seed, "")
        episode_key = (seed, dist)
        episodes_seen.add(episode_key)

        features = obj.get("features") or {}
        final_params = obj.get("final_params") or {}
        is_deviation = not _params_equal(final_params, PROMPT_BASELINE_PARAMS)
        total_decisions_all += 1
        if is_deviation:
            total_deviation_decisions_all += 1
            episodes_with_any_dev.add(episode_key)

        for feature_name, binner in feature_specs:
            bin_label = binner(features.get(feature_name))
            key = (feature_name, bin_label)
            agg = aggs.get(key)
            if agg is None:
                agg = BinAgg(feature=feature_name, bin_label=bin_label)
                aggs[key] = agg

            agg.decisions_total += 1
            agg.episodes_in_bin.add(episode_key)
            if is_deviation:
                agg.decisions_deviation += 1
                agg.episodes_with_dev_in_bin.add(episode_key)
                agg.dev_param_sum["w_delay"] += _safe_float(final_params.get("w_delay"))
                agg.dev_param_sum["w_shift"] += _safe_float(final_params.get("w_shift"))
                agg.dev_param_sum["w_switch"] += _safe_float(final_params.get("w_switch"))
                agg.dev_param_sum["freeze_horizon"] += float(_safe_int(final_params.get("freeze_horizon")))
                agg.dev_param_count += 1

    # Prepare output rows
    def mean_delta(keys: Iterable[Tuple[int, str]]) -> Dict[str, float]:
        vals = [episode_deltas[k] for k in keys if k in episode_deltas]
        if not vals:
            return {
                "n_episodes": 0,
                "mean_delta_avg_delay": math.nan,
                "mean_delta_episode_drift": math.nan,
                "mean_delta_on_time_rate": math.nan,
                "mean_delta_combined": math.nan,
            }
        n = len(vals)
        return {
            "n_episodes": n,
            "mean_delta_avg_delay": sum(v.delta_avg_delay for v in vals) / n,
            "mean_delta_episode_drift": sum(v.delta_episode_drift for v in vals) / n,
            "mean_delta_on_time_rate": sum(v.delta_on_time_rate for v in vals) / n,
            "mean_delta_combined": sum(v.delta_combined for v in vals) / n,
        }

    global_mean = mean_delta(episode_deltas.keys())
    overall_dev_rate_decisions = (
        total_deviation_decisions_all / total_decisions_all if total_decisions_all else 0.0
    )
    episodes_seen_with_metrics = {k for k in episodes_seen if k in episode_deltas}
    overall_dev_rate_episodes = (
        len({k for k in episodes_with_any_dev if k in episode_deltas}) / len(episodes_seen_with_metrics)
        if episodes_seen_with_metrics
        else 0.0
    )

    def diff(a: float, b: float) -> float:
        if a is None or b is None:
            return math.nan
        if isinstance(a, float) and math.isnan(a):
            return math.nan
        if isinstance(b, float) and math.isnan(b):
            return math.nan
        return a - b

    rows_out: List[Dict[str, Any]] = []
    for (_, _), agg in sorted(aggs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        episodes_in_bin = {k for k in agg.episodes_in_bin if k in episode_deltas}
        episodes_with_dev = {k for k in agg.episodes_with_dev_in_bin if k in episode_deltas}
        episodes_without_dev = episodes_in_bin - episodes_with_dev

        m_with = mean_delta(episodes_with_dev)
        m_without = mean_delta(episodes_without_dev)

        dev_rate_decisions = agg.decisions_deviation / agg.decisions_total if agg.decisions_total else 0.0
        dev_rate_episodes = (
            len(episodes_with_dev) / len(episodes_in_bin) if episodes_in_bin else 0.0
        )

        if agg.dev_param_count:
            mean_w_delay = agg.dev_param_sum["w_delay"] / agg.dev_param_count
            mean_w_shift = agg.dev_param_sum["w_shift"] / agg.dev_param_count
            mean_w_switch = agg.dev_param_sum["w_switch"] / agg.dev_param_count
            mean_freeze = agg.dev_param_sum["freeze_horizon"] / agg.dev_param_count
        else:
            mean_w_delay = math.nan
            mean_w_shift = math.nan
            mean_w_switch = math.nan
            mean_freeze = math.nan

        row = {
            "feature": agg.feature,
            "bin": agg.bin_label,
            "overall_deviation_rate_decisions": overall_dev_rate_decisions,
            "overall_deviation_rate_episodes": overall_dev_rate_episodes,
            "decisions_total": agg.decisions_total,
            "decisions_deviation": agg.decisions_deviation,
            "deviation_rate_decisions": dev_rate_decisions,
            "deviation_rate_lift_decisions": (
                (dev_rate_decisions / overall_dev_rate_decisions) if overall_dev_rate_decisions > 0 else math.nan
            ),
            "episodes_in_bin": len(episodes_in_bin),
            "episodes_with_deviation_in_bin": len(episodes_with_dev),
            "deviation_rate_episodes": dev_rate_episodes,
            "deviation_rate_lift_episodes": (
                (dev_rate_episodes / overall_dev_rate_episodes) if overall_dev_rate_episodes > 0 else math.nan
            ),
            "mean_dev_params_w_delay": mean_w_delay,
            "mean_dev_params_w_shift": mean_w_shift,
            "mean_dev_params_w_switch": mean_w_switch,
            "mean_dev_params_freeze_horizon": mean_freeze,
            "global_mean_delta_avg_delay": global_mean["mean_delta_avg_delay"],
            "global_mean_delta_episode_drift": global_mean["mean_delta_episode_drift"],
            "global_mean_delta_on_time_rate": global_mean["mean_delta_on_time_rate"],
            "global_mean_delta_combined": global_mean["mean_delta_combined"],
            "with_dev_n_episodes": m_with["n_episodes"],
            "with_dev_mean_delta_avg_delay": m_with["mean_delta_avg_delay"],
            "with_dev_mean_delta_episode_drift": m_with["mean_delta_episode_drift"],
            "with_dev_mean_delta_on_time_rate": m_with["mean_delta_on_time_rate"],
            "with_dev_mean_delta_combined": m_with["mean_delta_combined"],
            "with_dev_minus_global_delta_avg_delay": diff(
                m_with["mean_delta_avg_delay"], global_mean["mean_delta_avg_delay"]
            ),
            "with_dev_minus_global_delta_episode_drift": diff(
                m_with["mean_delta_episode_drift"], global_mean["mean_delta_episode_drift"]
            ),
            "with_dev_minus_global_delta_on_time_rate": diff(
                m_with["mean_delta_on_time_rate"], global_mean["mean_delta_on_time_rate"]
            ),
            "with_dev_minus_global_delta_combined": diff(
                m_with["mean_delta_combined"], global_mean["mean_delta_combined"]
            ),
            "without_dev_n_episodes": m_without["n_episodes"],
            "without_dev_mean_delta_avg_delay": m_without["mean_delta_avg_delay"],
            "without_dev_mean_delta_episode_drift": m_without["mean_delta_episode_drift"],
            "without_dev_mean_delta_on_time_rate": m_without["mean_delta_on_time_rate"],
            "without_dev_mean_delta_combined": m_without["mean_delta_combined"],
            "without_dev_minus_global_delta_avg_delay": diff(
                m_without["mean_delta_avg_delay"], global_mean["mean_delta_avg_delay"]
            ),
            "without_dev_minus_global_delta_episode_drift": diff(
                m_without["mean_delta_episode_drift"], global_mean["mean_delta_episode_drift"]
            ),
            "without_dev_minus_global_delta_on_time_rate": diff(
                m_without["mean_delta_on_time_rate"], global_mean["mean_delta_on_time_rate"]
            ),
            "without_dev_minus_global_delta_combined": diff(
                m_without["mean_delta_combined"], global_mean["mean_delta_combined"]
            ),
        }

        # Contribution proxy: (with_dev - without_dev) within the same bin's episode set
        # (diff is already defined above)
        row["contrib_proxy_delta_avg_delay"] = diff(
            row["with_dev_mean_delta_avg_delay"], row["without_dev_mean_delta_avg_delay"]
        )
        row["contrib_proxy_delta_episode_drift"] = diff(
            row["with_dev_mean_delta_episode_drift"], row["without_dev_mean_delta_episode_drift"]
        )
        row["contrib_proxy_delta_on_time_rate"] = diff(
            row["with_dev_mean_delta_on_time_rate"], row["without_dev_mean_delta_on_time_rate"]
        )
        row["contrib_proxy_delta_combined"] = diff(
            row["with_dev_mean_delta_combined"], row["without_dev_mean_delta_combined"]
        )

        rows_out.append(row)

    # Write CSV
    fieldnames = list(rows_out[0].keys()) if rows_out else [
        "feature",
        "bin",
        "decisions_total",
        "decisions_deviation",
        "deviation_rate_decisions",
        "episodes_in_bin",
        "episodes_with_deviation_in_bin",
        "deviation_rate_episodes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote: {out_path} ({len(rows_out)} rows)")
    print(f"Prompt baseline params: {PROMPT_BASELINE_PARAMS}")
    print(f"Episode deltas computed for: {len(episode_deltas)} episodes (dataset={args.dataset})")
    print(f"Overall deviation rate (decisions): {overall_dev_rate_decisions:.4f}")
    print(f"Overall deviation rate (episodes): {overall_dev_rate_episodes:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
