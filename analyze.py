#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析与绘图脚本 - 读取实验结果 CSV，计算统计，生成图表

功能：
1. 读取 results_per_episode.csv（含 LLM 相关字段）
2. 输出 summary.csv 包含：
   - mean_delay, CI_delay, mean_drift, CI_drift
   - mean_replans, mean_switch
   - mean_solver_time_ms, mean_wall_time_ms
   - fallback_rate, cache_hit_rate
   - mean_llm_time_ms, mean_llm_total_tokens
3. 生成图表：
   - Delay vs PlanDrift scatter
   - 重排次数/切换次数分布
   - LLM time vs Solver time 对比
4. 对 llm_real 策略输出 reliability_report.md

用法：
    python analyze.py --input results/ --output figures/
    python analyze.py --input results/ --output figures/ --show
"""

import argparse
import csv
import json
import os
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EpisodeRecord:
    """单 episode 记录（支持 LLM 字段）"""
    seed: int
    disturbance_level: str
    policy_name: str
    dataset: str
    
    # 核心指标
    completed: int
    total: int
    on_time_rate: float
    avg_delay: float
    max_delay: int
    weighted_tardiness: float
    resource_utilization: float
    episode_drift: float
    
    # 稳定性
    total_shifts: int
    total_switches: int
    num_replans: int
    num_forced_replans: int
    
    # 性能
    avg_solve_time_ms: float
    total_runtime_s: float

    # 补充指标
    avg_time_deviation_min: float = 0.0
    total_resource_switches: int = 0
    makespan_cmax: int = 0
    feasible_rate: float = 0.0
    forced_replan_rate: float = 0.0
    avg_frozen: float = 0.0
    avg_num_tasks_scheduled: float = 0.0
    util_r_pad: float = 0.0
    
    # LLM 相关字段（可选，旧版 CSV 可能不包含）
    llm_calls: int = 0
    llm_time_total_ms: int = 0
    llm_latency_total_ms: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cache_hit_rate: float = 0.0
    llm_fallback_count: int = 0
    solver_time_total_ms: int = 0
    wall_time_total_ms: int = 0


# ============================================================================
# 数据加载
# ============================================================================

def load_episode_results(filepath: str) -> List[EpisodeRecord]:
    """加载 episode 结果 CSV（兼容新旧格式）"""
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(EpisodeRecord(
                seed=int(row['seed']),
                disturbance_level=row['disturbance_level'],
                policy_name=row['policy_name'],
                dataset=row['dataset'],
                completed=int(row['completed']),
                total=int(row['total']),
                on_time_rate=float(row['on_time_rate']),
                avg_delay=float(row['avg_delay']),
                max_delay=int(row['max_delay']),
                weighted_tardiness=float(row.get('weighted_tardiness', 0.0)),
                resource_utilization=float(row.get('resource_utilization', 0.0)),
                episode_drift=float(row['episode_drift']),
                total_shifts=int(row['total_shifts']),
                total_switches=int(row['total_switches']),
                num_replans=int(row['num_replans']),
                num_forced_replans=int(row['num_forced_replans']),
                avg_solve_time_ms=float(row['avg_solve_time_ms']),
                total_runtime_s=float(row['total_runtime_s']),
                avg_time_deviation_min=float(row.get('avg_time_deviation_min', 0.0)),
                total_resource_switches=int(float(row.get('total_resource_switches', 0))),
                makespan_cmax=int(float(row.get('makespan_cmax', 0))),
                feasible_rate=float(row.get('feasible_rate', 0.0)),
                forced_replan_rate=float(row.get('forced_replan_rate', 0.0)),
                avg_frozen=float(row.get('avg_frozen', 0.0)),
                avg_num_tasks_scheduled=float(row.get('avg_num_tasks_scheduled', 0.0)),
                util_r_pad=float(row.get('util_r_pad', 0.0)),
                # LLM 字段（兼容旧版 CSV）
                llm_calls=int(row.get('llm_calls', 0)),
                llm_time_total_ms=int(row.get('llm_time_total_ms', 0)),
                llm_latency_total_ms=int(row.get('llm_latency_total_ms', 0)),
                llm_prompt_tokens=int(row.get('llm_prompt_tokens', 0)),
                llm_completion_tokens=int(row.get('llm_completion_tokens', 0)),
                llm_total_tokens=int(row.get('llm_total_tokens', 0)),
                llm_cache_hit_rate=float(row.get('llm_cache_hit_rate', 0.0)),
                llm_fallback_count=int(row.get('llm_fallback_count', 0)),
                solver_time_total_ms=int(row.get('solver_time_total_ms', 0)),
                wall_time_total_ms=int(row.get('wall_time_total_ms', 0))
            ))
    
    return records


def load_tuning_results(filepath: str) -> List[Dict[str, Any]]:
    """?????? CSV?????freeze ? epsilon_solver?"""
    results = []
    
    if not os.path.exists(filepath):
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "freeze_horizon_slots": int(row.get('freeze_horizon_slots', 0)),
                "epsilon_solver": float(row.get('epsilon_solver', 0.0)),
                "avg_delay": float(row.get('avg_delay', 0.0)),
                "avg_drift": float(row.get('avg_drift', 0.0)),
                "avg_time_deviation_min": float(row.get('avg_time_deviation_min', 0.0)),
                "avg_forced_replan_rate": float(row.get('avg_forced_replan_rate', 0.0)),
                "avg_feasible_rate": float(row.get('avg_feasible_rate', 0.0)),
                "delay_ratio": float(row.get('delay_ratio', 0.0)),
                "drift_ratio": float(row.get('drift_ratio', 0.0)),
                "time_dev_ratio": float(row.get('time_dev_ratio', 0.0)),
                "forced_replan_ratio": float(row.get('forced_replan_ratio', 0.0)),
                "delay_ok": row.get('delay_ok', 'False') == 'True',
                "feasible_ok": row.get('feasible_ok', 'False') == 'True',
                "combined_score": float(row.get('combined_score', 0.0)),
                "epsilon_metric": row.get('epsilon_metric', ''),
                "epsilon_threshold": float(row.get('epsilon_threshold', 0.0)),
                "epsilon_value": float(row.get('epsilon_value', 0.0)),
                "epsilon_ok": row.get('epsilon_ok', 'False') == 'True',
                "epsilon_violation": float(row.get('epsilon_violation', 0.0)),
                "num_episodes": int(row.get('num_episodes', 0)),
                "avg_solve_time_ms": float(row.get('avg_solve_time_ms', 0.0))
            })
    
    return results



# ============================================================================
# 统计计算
# ============================================================================

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def ci95(values: List[float]) -> float:
    """95% 置信区间半宽"""
    if len(values) < 2:
        return 0.0
    return 1.96 * std(values) / math.sqrt(len(values))


def compute_epsilon_threshold(
    records: List["EpisodeRecord"],
    dataset: str,
    epsilon_metric: str,
    epsilon_relative: str,
    epsilon_value: float,
    baseline_policy: str = "fixed_default"
) -> Optional[float]:
    if epsilon_metric != "avg_delay":
        raise ValueError(f"Unsupported epsilon_metric: {epsilon_metric}")
    baseline_delays = [
        r.avg_delay
        for r in records
        if r.dataset == dataset and r.policy_name == baseline_policy
    ]
    if not baseline_delays:
        return None
    baseline_delay = mean(baseline_delays)
    if epsilon_relative == "baseline":
        return baseline_delay * (1 + epsilon_value)
    if epsilon_relative == "absolute":
        return epsilon_value
    raise ValueError(f"Unsupported epsilon_relative: {epsilon_relative}")


def percentile(values: List[float], p: float) -> float:
    """计算百分位数"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def paired_t_test(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """
    配对 t 检验
    
    Returns:
        (t_statistic, p_value)
    """
    if len(values1) != len(values2) or len(values1) < 2:
        return 0.0, 1.0
    
    n = len(values1)
    diffs = [v1 - v2 for v1, v2 in zip(values1, values2)]
    
    d_bar = mean(diffs)
    s_d = std(diffs)
    
    if s_d == 0:
        return 0.0, 1.0
    
    t_stat = d_bar / (s_d / math.sqrt(n))
    
    # 简化 p 值计算（使用正态近似）
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    
    return t_stat, p_value


def compute_summary_stats(
    records: List[EpisodeRecord],
    tuning_lambda: float = 5.0
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    计算汇总统计（含 LLM 指标）
    
    Returns:
        {(dataset, policy_name): {metric: value, ...}}
    """
    # 按 (dataset, policy_name) 分组
    groups: Dict[Tuple[str, str], List[EpisodeRecord]] = defaultdict(list)
    for rec in records:
        key = (rec.dataset, rec.policy_name)
        groups[key].append(rec)
    
    # 基线策略优先级，用于归一化综合成本
    baseline_priority = ["fixed_default", "fixed_tuned"]
    baseline_means: Dict[str, Tuple[float, float]] = {}
    datasets = sorted({k[0] for k in groups.keys()})
    for dataset in datasets:
        baseline_recs = None
        for name in baseline_priority:
            baseline_recs = groups.get((dataset, name))
            if baseline_recs:
                break
        if not baseline_recs and groups:
            # 回退到该数据集的任一策略
            for (ds, _), recs in groups.items():
                if ds == dataset:
                    baseline_recs = recs
                    break
        if baseline_recs:
            base_delay = mean([r.avg_delay for r in baseline_recs])
            base_drift = mean([r.episode_drift for r in baseline_recs])
            baseline_means[dataset] = (base_delay, base_drift)

    stats = {}
    for key, recs in groups.items():
        dataset, policy_name = key
        
        # 核心指标
        delays = [r.avg_delay for r in recs]
        drifts = [r.episode_drift for r in recs]
        on_times = [r.on_time_rate for r in recs]
        weighted_tardinesses = [r.weighted_tardiness for r in recs]
        resource_utils = [r.resource_utilization for r in recs]
        shifts = [r.total_shifts for r in recs]
        switches = [r.total_switches for r in recs]
        replans = [r.num_replans for r in recs]
        time_devs = [r.avg_time_deviation_min for r in recs]
        resource_switches = [r.total_resource_switches for r in recs]
        makespans = [r.makespan_cmax for r in recs]
        feasible_rates = [r.feasible_rate for r in recs]
        forced_replan_rates = [r.forced_replan_rate for r in recs]
        avg_frozens = [r.avg_frozen for r in recs]
        avg_tasks_scheduled = [r.avg_num_tasks_scheduled for r in recs]
        util_r_pads = [r.util_r_pad for r in recs]
        
        # 时间指标
        solve_times = [r.avg_solve_time_ms for r in recs]
        solver_total_times = [r.solver_time_total_ms for r in recs]
        wall_times = [r.wall_time_total_ms for r in recs]
        
        # LLM 指标
        llm_calls = [r.llm_calls for r in recs]
        llm_times = [r.llm_time_total_ms for r in recs]
        llm_tokens = [r.llm_total_tokens for r in recs]
        llm_fallbacks = [r.llm_fallback_count for r in recs]
        llm_cache_hits = [r.llm_cache_hit_rate for r in recs]
        
        # 计算 fallback_rate = sum(fallback_count) / sum(calls)
        total_calls = sum(llm_calls)
        total_fallbacks = sum(llm_fallbacks)
        fallback_rate = total_fallbacks / total_calls if total_calls > 0 else 0.0
        
        # 平均 cache_hit_rate（加权平均）
        weighted_cache_hit = sum(
            r.llm_cache_hit_rate * r.llm_calls for r in recs
        )
        avg_cache_hit_rate = weighted_cache_hit / total_calls if total_calls > 0 else 0.0
        
        # 归一化综合成本：相对基线的加权比例
        base_delay, base_drift = baseline_means.get(dataset, (0.0, 0.0))
        eps = 1e-9
        base_delay = base_delay if base_delay > eps else eps
        base_drift = base_drift if base_drift > eps else eps
        w1, w2 = 0.5, 0.5
        combined = [
            w1 * (d / base_delay) + w2 * (dr / base_drift)
            for d, dr in zip(delays, drifts)
        ]
        
        stats[key] = {
            "dataset": dataset,
            "policy_name": policy_name,
            "n": len(recs),
            
            # 核心指标
            "mean_delay": mean(delays),
            "CI_delay": ci95(delays),
            "mean_drift": mean(drifts),
            "CI_drift": ci95(drifts),
            "combined_mean": mean(combined),
            "combined_ci95": ci95(combined),
            "on_time_mean": mean(on_times),
            "mean_weighted_tardiness": mean(weighted_tardinesses),
            "CI_weighted_tardiness": ci95(weighted_tardinesses),
            "mean_resource_utilization": mean(resource_utils),
            "CI_resource_utilization": ci95(resource_utils),
            
            # 稳定性
            "mean_replans": mean(replans),
            "mean_switch": mean(switches),
            "mean_shifts": mean(shifts),
            "mean_avg_time_deviation_min": mean(time_devs),
            "CI_avg_time_deviation_min": ci95(time_devs),
            "mean_total_resource_switches": mean(resource_switches),
            "CI_total_resource_switches": ci95(resource_switches),
            "mean_makespan_cmax": mean(makespans),
            "CI_makespan_cmax": ci95(makespans),
            "mean_feasible_rate": mean(feasible_rates),
            "mean_forced_replan_rate": mean(forced_replan_rates),
            "mean_avg_frozen": mean(avg_frozens),
            "mean_avg_tasks_scheduled": mean(avg_tasks_scheduled),
            "mean_util_r_pad": mean(util_r_pads),
            "CI_util_r_pad": ci95(util_r_pads),
            
            # 时间
            "mean_solver_time_ms": mean(solve_times),
            "mean_solver_total_ms": mean(solver_total_times),
            "mean_wall_time_ms": mean(wall_times),
            
            # LLM 指标
            "mean_llm_calls": mean(llm_calls),
            "mean_llm_time_ms": mean(llm_times),
            "mean_llm_total_tokens": mean(llm_tokens),
            "fallback_rate": fallback_rate,
            "cache_hit_rate": avg_cache_hit_rate,
            "total_llm_calls": total_calls,
            "total_fallbacks": total_fallbacks
        }
    
    return stats


# ============================================================================
# Baseline-only analysis helpers (no LLM required)
# ============================================================================

def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def _parse_episode_dir_name(name: str) -> Optional[Tuple[int, str]]:
    if not name.startswith("episode_"):
        return None
    rest = name[len("episode_"):]
    parts = rest.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        seed = int(parts[0])
    except ValueError:
        return None
    policy = parts[1]
    return seed, policy


def build_record_index(
    records: List[EpisodeRecord],
    dataset: Optional[str] = None
) -> Dict[Tuple[int, str], EpisodeRecord]:
    index = {}
    for rec in records:
        if dataset and rec.dataset != dataset:
            continue
        index[(rec.seed, rec.policy_name)] = rec
    return index


def load_metrics_per_roll(metrics_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "t": int(float(row.get("t", 0))),
                "plan_drift": float(row.get("plan_drift", 0.0)),
                "num_shifts": int(float(row.get("num_shifts", 0))),
                "num_switches": int(float(row.get("num_switches", 0))),
                "solve_time_ms": float(row.get("solve_time_ms", 0.0)),
                "is_feasible": _to_bool(row.get("is_feasible", True)),
                "forced_replan": _to_bool(row.get("forced_replan", False))
            })
    return rows


def collect_roll_metrics(
    logs_dir: str,
    record_index: Dict[Tuple[int, str], EpisodeRecord]
) -> List[Dict[str, Any]]:
    roll_rows: List[Dict[str, Any]] = []
    if not os.path.isdir(logs_dir):
        return roll_rows

    for root, _, files in os.walk(logs_dir):
        if "metrics_per_roll.csv" not in files:
            continue
        episode_dir = os.path.basename(root)
        parsed = _parse_episode_dir_name(episode_dir)
        if not parsed:
            continue
        seed, policy = parsed
        rec = record_index.get((seed, policy))
        if not rec:
            continue

        metrics_path = os.path.join(root, "metrics_per_roll.csv")
        for row in load_metrics_per_roll(metrics_path):
            roll_rows.append({
                "seed": seed,
                "policy_name": policy,
                "disturbance_level": rec.disturbance_level,
                "t": row["t"],
                "plan_drift": row["plan_drift"],
                "num_shifts": row["num_shifts"],
                "num_switches": row["num_switches"],
                "solve_time_ms": row["solve_time_ms"],
                "is_feasible": row["is_feasible"],
                "forced_replan": row["forced_replan"]
            })

    return roll_rows


def compute_roll_stats(
    roll_rows: List[Dict[str, Any]],
    policies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in roll_rows:
        if policies and row["policy_name"] not in policies:
            continue
        key = (row["disturbance_level"], row["policy_name"])
        groups[key].append(row)

    rows = []
    for (level, policy), items in sorted(groups.items()):
        feasible_rate = mean([1.0 if r["is_feasible"] else 0.0 for r in items])
        forced_rate = mean([1.0 if r["forced_replan"] else 0.0 for r in items])
        solve_times = [r["solve_time_ms"] for r in items]
        rows.append({
            "disturbance_level": level,
            "policy_name": policy,
            "n_rolls": len(items),
            "feasible_rate": round(feasible_rate, 4),
            "forced_replan_rate": round(forced_rate, 4),
            "solve_time_mean_ms": round(mean(solve_times), 2),
            "solve_time_p95_ms": round(percentile(solve_times, 95), 2)
        })
    return rows


def compute_main_table_by_disturbance(
    records: List[EpisodeRecord],
    dataset: str,
    delay_budget: Optional[float],
    policies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[EpisodeRecord]] = defaultdict(list)
    for rec in records:
        if rec.dataset != dataset:
            continue
        if policies and rec.policy_name not in policies:
            continue
        key = (rec.disturbance_level, rec.policy_name)
        groups[key].append(rec)

    rows = []
    for (level, policy), recs in sorted(groups.items()):
        avg_delay_vals = [r.avg_delay for r in recs]
        max_delay_vals = [r.max_delay for r in recs]
        drift_vals = [r.episode_drift for r in recs]
        shifts_vals = [r.total_shifts for r in recs]
        switches_vals = [r.total_switches for r in recs]
        feasible_vals = [r.feasible_rate for r in recs]
        solve_vals = [r.avg_solve_time_ms for r in recs]
        mean_delay = mean(avg_delay_vals)

        rows.append({
            "disturbance_level": level,
            "policy_name": policy,
            "n": len(recs),
            "avg_delay_mean": round(mean_delay, 3),
            "avg_delay_std": round(std(avg_delay_vals), 3),
            "max_delay_mean": round(mean(max_delay_vals), 2),
            "max_delay_std": round(std(max_delay_vals), 2),
            "drift_mean": round(mean(drift_vals), 4),
            "drift_std": round(std(drift_vals), 4),
            "total_shifts_mean": round(mean(shifts_vals), 2),
            "total_shifts_std": round(std(shifts_vals), 2),
            "total_switches_mean": round(mean(switches_vals), 2),
            "total_switches_std": round(std(switches_vals), 2),
            "feasible_rate_mean": round(mean(feasible_vals), 4),
            "feasible_rate_std": round(std(feasible_vals), 4),
            "avg_solve_time_ms_mean": round(mean(solve_vals), 2),
            "avg_solve_time_ms_std": round(std(solve_vals), 2),
            "delay_budget": round(delay_budget, 3) if delay_budget is not None else "",
            "budget_ok": (
                "True" if delay_budget is not None and mean_delay <= delay_budget else "False"
            )
        })
    return rows


def save_main_table_by_disturbance(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        print("Skip main table: no rows")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved main table: {output_path}")


def compute_delay_budget(
    records: List[EpisodeRecord],
    dataset: str,
    baseline_policy: str,
    budget_source: str,
    budget_multiplier: float,
    budget_value: Optional[float],
    best_params_path: Optional[str]
) -> Optional[float]:
    if budget_source == "manual":
        return budget_value
    if budget_source == "best_params" and best_params_path and os.path.exists(best_params_path):
        try:
            with open(best_params_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "epsilon_threshold" in data:
                return float(data["epsilon_threshold"])
            base = float(data.get("baseline_avg_delay", 0.0))
            eps = float(data.get("epsilon_value", 0.0))
            if base > 0:
                return base * (1 + eps)
        except Exception:
            return None
    if budget_source == "baseline":
        baseline_delays = [
            r.avg_delay for r in records
            if r.dataset == dataset and r.policy_name == baseline_policy
        ]
        if not baseline_delays:
            return None
        return mean(baseline_delays) * budget_multiplier
    return None


def save_budgeted_best_drift(
    tuning_results: List[Dict[str, Any]],
    records: List[EpisodeRecord],
    dataset: str,
    delay_budget: Optional[float],
    baseline_policy: str,
    output_path: str
) -> None:
    if delay_budget is None:
        print("Skip budgeted best drift: delay budget not available")
        return
    baseline_delays = [
        r.avg_delay for r in records
        if r.dataset == dataset and r.policy_name == baseline_policy
    ]
    baseline_drifts = [
        r.episode_drift for r in records
        if r.dataset == dataset and r.policy_name == baseline_policy
    ]
    base_delay = mean(baseline_delays) if baseline_delays else 0.0
    base_drift = mean(baseline_drifts) if baseline_drifts else 0.0

    eligible = [r for r in tuning_results if r.get("avg_delay", 0.0) <= delay_budget]
    best = None
    if eligible:
        best = min(eligible, key=lambda r: r.get("avg_drift", float("inf")))

    row = {
        "delay_budget": round(delay_budget, 3),
        "baseline_policy": baseline_policy,
        "baseline_avg_delay": round(base_delay, 3),
        "baseline_avg_drift": round(base_drift, 5),
        "eligible_count": len(eligible)
    }
    if best:
        row.update({
            "best_avg_delay": round(best.get("avg_delay", 0.0), 3),
            "best_avg_drift": round(best.get("avg_drift", 0.0), 5),
            "best_freeze_horizon": best.get("freeze_horizon_slots", ""),
            "best_epsilon_solver": best.get("epsilon_solver", "")
        })
    else:
        row.update({
            "best_avg_delay": "",
            "best_avg_drift": "",
            "best_freeze_horizon": "",
            "best_epsilon_solver": ""
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved budgeted best drift: {output_path}")


def save_roll_feasibility_stats(
    roll_rows: List[Dict[str, Any]],
    output_path: str,
    policies: Optional[List[str]] = None
) -> None:
    stats = compute_roll_stats(roll_rows, policies=policies)
    if not stats:
        print("Skip roll feasibility stats: no roll data")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(stats[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    print(f"Saved roll feasibility stats: {output_path}")


def _load_schedule_items(schedule_path: str, max_items: int = 120) -> List[Tuple[str, int, int]]:
    with open(schedule_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    schedule = data.get("schedule") or data.get("final_schedule") or []
    items = []
    for entry in schedule:
        start = entry.get("start_slot")
        end = entry.get("end_slot")
        label = entry.get("op_id") or entry.get("task_id") or entry.get("id")
        if start is None or end is None or label is None:
            continue
        items.append((label, int(start), int(end)))
    items.sort(key=lambda x: x[1])
    return items[:max_items]


def plot_episode_compare(
    logs_dir: str,
    seed: int,
    policy_a: str,
    policy_b: str,
    output_path: str,
    max_items: int = 120
) -> None:
    if not os.path.isdir(logs_dir):
        print("Skip episode compare: logs dir not found")
        return

    target_a = f"episode_{seed}_{policy_a}"
    target_b = f"episode_{seed}_{policy_b}"
    dir_a = None
    dir_b = None
    for root, dirs, _ in os.walk(logs_dir):
        if target_a in dirs:
            dir_a = os.path.join(root, target_a)
        if target_b in dirs:
            dir_b = os.path.join(root, target_b)

    if not dir_a or not dir_b:
        print("Skip episode compare: episode dirs not found")
        return

    schedule_path = os.path.join(dir_a, "final_schedule.json")
    metrics_a = os.path.join(dir_a, "metrics_per_roll.csv")
    metrics_b = os.path.join(dir_b, "metrics_per_roll.csv")
    if not os.path.exists(metrics_a) or not os.path.exists(metrics_b):
        print("Skip episode compare: missing metrics_per_roll.csv")
        return

    rows_a = load_metrics_per_roll(metrics_a)
    rows_b = load_metrics_per_roll(metrics_b)
    if not rows_a or not rows_b:
        print("Skip episode compare: empty roll metrics")
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    if os.path.exists(schedule_path):
        items = _load_schedule_items(schedule_path, max_items=max_items)
        labels = [i[0] for i in items]
        starts = [i[1] for i in items]
        durations = [max(0, i[2] - i[1]) for i in items]
        y_positions = range(len(items))
        ax_top.barh(y_positions, durations, left=starts, color="#3498db", alpha=0.8)
        ax_top.set_yticks(list(y_positions))
        ax_top.set_yticklabels(labels, fontsize=7)
        ax_top.invert_yaxis()
        ax_top.set_xlabel("Slot")
        ax_top.set_title(f"Gantt (seed={seed}, policy={policy_a})")
        ax_top.grid(True, axis="x", alpha=0.3)
    else:
        ax_top.axis("off")
        ax_top.set_title("Gantt not available")

    t_a = [r["t"] for r in rows_a]
    t_b = [r["t"] for r in rows_b]
    drift_a = [r["plan_drift"] for r in rows_a]
    drift_b = [r["plan_drift"] for r in rows_b]
    shifts_a = [r["num_shifts"] for r in rows_a]
    shifts_b = [r["num_shifts"] for r in rows_b]

    ax_bottom.plot(t_a, drift_a, color="#2ecc71", label=f"{policy_a} drift")
    ax_bottom.plot(t_b, drift_b, color="#e74c3c", label=f"{policy_b} drift")
    ax_bottom.set_xlabel("Rolling step")
    ax_bottom.set_ylabel("Plan drift")
    ax_bottom.grid(True, alpha=0.3)

    ax2 = ax_bottom.twinx()
    ax2.plot(t_a, shifts_a, color="#2ecc71", linestyle="--", label=f"{policy_a} changes")
    ax2.plot(t_b, shifts_b, color="#e74c3c", linestyle="--", label=f"{policy_b} changes")
    ax2.set_ylabel("#changes (num_shifts)")

    lines, labels = ax_bottom.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_bottom.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved episode compare: {output_path}")


# ============================================================================
# 绘图样式配置
# ============================================================================

POLICY_STYLES = {
    "fixed_tuned": {"color": "#2ecc71", "marker": "o", "label": "Fixed-Balanced"},
    "fixed_default": {"color": "#3498db", "marker": "s", "label": "Delay-Only"},
    "nofreeze": {"color": "#e74c3c", "marker": "^", "label": "No-Freeze"},
    "mockllm": {"color": "#f39c12", "marker": "v", "label": "MockLLM"},
    "llm_real": {"color": "#1abc9c", "marker": "P", "label": "LLM (Real)"},
    "fixed": {"color": "#3498db", "marker": "s", "label": "Fixed"}
}


def get_policy_style(policy_name: str) -> dict:
    """获取策略样式"""
    return POLICY_STYLES.get(policy_name, {
        "color": "#7f8c8d", "marker": "x", "label": policy_name
    })


# ============================================================================
# 图表 1: Delay vs PlanDrift Scatter
# ============================================================================

def plot_delay_vs_drift_scatter(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test",
    title: str = "Delay vs Plan Drift",
    epsilon_threshold: Optional[float] = None
):
    """
    绘制 Delay vs PlanDrift 散点图
    
    每个策略不同颜色和标记
    """
    # 按策略分组
    policy_data: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        policy_data[rec.policy_name][0].append(rec.avg_delay)
        policy_data[rec.policy_name][1].append(rec.episode_drift)
    
    if epsilon_threshold is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        # 绘制每个策略
        for policy_name, (delays, drifts) in policy_data.items():
            style = get_policy_style(policy_name)
            ax.scatter(
                delays, drifts,
                c=style["color"],
                marker=style["marker"],
                label=style["label"],
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )
        ax.set_xlabel("Average Delay (slots)", fontsize=12)
        ax.set_ylabel("Episode Plan Drift", fontsize=12)
        ax.set_title(f"{title} ({dataset} set)", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        # 添加参考线
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        panels = [("All", False), ("Feasible Only", True)]
        for ax, (panel_title, feasible_only) in zip(axes, panels):
            for policy_name, (delays, drifts) in policy_data.items():
                style = get_policy_style(policy_name)
                pts = [
                    (d, dr)
                    for d, dr in zip(delays, drifts)
                    if (d <= epsilon_threshold or not feasible_only)
                ]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(
                    xs, ys,
                    c=style["color"],
                    marker=style["marker"],
                    label=style["label"],
                    alpha=0.7,
                    s=50,
                    edgecolors='white',
                    linewidths=0.5
                )
            ax.set_xlabel("Average Delay (slots)", fontsize=11)
            ax.set_title(f"{title} ({dataset} set) - {panel_title}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(
                x=epsilon_threshold,
                color='black',
                linestyle='--',
                alpha=0.7,
                label='epsilon threshold' if not feasible_only else None
            )
            ax.axvspan(0, epsilon_threshold, color='green', alpha=0.04)
        axes[0].set_ylabel("Episode Plan Drift", fontsize=11)
        axes[0].legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"保存图表: {output_path}")


# ============================================================================
# 图表 2: 重排/切换次数分布（箱线图）
# ============================================================================

def plot_replans_switches_boxplot(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """绘制重排次数和切换次数分布图（双子图箱线图）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按策略分组
    replan_data: Dict[str, List[int]] = defaultdict(list)
    switch_data: Dict[str, List[int]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        replan_data[rec.policy_name].append(rec.num_replans)
        switch_data[rec.policy_name].append(rec.total_switches)
    
    # 过滤空数据的策略
    policy_names = sorted([p for p in replan_data.keys() if len(replan_data[p]) > 0])
    
    if len(policy_names) == 0:
        print(f"警告: {dataset} 数据集没有找到任何策略数据，跳过绘制 {output_path}")
        plt.close()
        return
    
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    # 子图 1: 重排次数
    ax1 = axes[0]
    data1 = [replan_data[p] for p in policy_names]
    bp1 = ax1.boxplot(data1, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel("Number of Replans", fontsize=12)
    ax1.set_title(f"Replan Count Distribution ({dataset} set)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # 子图 2: 切换次数
    ax2 = axes[1]
    data2 = [switch_data[p] for p in policy_names]
    bp2 = ax2.boxplot(data2, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel("Number of Pad Switches", fontsize=12)
    ax2.set_title(f"Pad Switch Count Distribution ({dataset} set)", fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


# ============================================================================
# 图表 3: LLM Time vs Solver Time
# ============================================================================

def plot_llm_vs_solver_time(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """
    绘制 LLM 时间 vs Solver 时间对比图
    
    使用分组条形图展示各策略的时间分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按策略分组
    llm_times: Dict[str, List[int]] = defaultdict(list)
    solver_times: Dict[str, List[int]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        llm_times[rec.policy_name].append(rec.llm_time_total_ms)
        solver_times[rec.policy_name].append(rec.solver_time_total_ms)
    
    policy_names = sorted(llm_times.keys())
    
    if not policy_names:
        print(f"警告: 没有数据，跳过绘制 {output_path}")
        plt.close()
        return
    
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    # 子图 1: 箱线图对比
    ax1 = axes[0]
    
    # 准备数据 - LLM 时间和 Solver 时间并排
    positions = []
    box_data = []
    box_colors = []
    
    for i, policy in enumerate(policy_names):
        pos_llm = i * 2.5
        pos_solver = i * 2.5 + 1
        positions.extend([pos_llm, pos_solver])
        box_data.append(llm_times[policy])
        box_data.append(solver_times[policy])
        box_colors.extend([colors[i], colors[i]])
    
    bp = ax1.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True)
    
    for j, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7 if j % 2 == 0 else 0.4)  # LLM 深色，Solver 浅色
    
    # X 轴标签
    tick_positions = [i * 2.5 + 0.5 for i in range(len(policy_names))]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title(f"LLM Time (dark) vs Solver Time (light) ({dataset} set)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 子图 2: 散点图 LLM vs Solver
    ax2 = axes[1]
    
    for policy in policy_names:
        style = get_policy_style(policy)
        llm_vals = llm_times[policy]
        solver_vals = solver_times[policy]
        
        ax2.scatter(
            solver_vals, llm_vals,
            c=style["color"],
            marker=style["marker"],
            label=style["label"],
            alpha=0.6,
            s=40
        )
    
    # 添加对角线
    max_val = max(
        max(max(llm_times[p]) for p in policy_names if llm_times[p]) if any(llm_times[p] for p in policy_names) else 0,
        max(max(solver_times[p]) for p in policy_names if solver_times[p]) if any(solver_times[p] for p in policy_names) else 0
    )
    if max_val > 0:
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')
    
    ax2.set_xlabel("Solver Time Total (ms)", fontsize=12)
    ax2.set_ylabel("LLM Time Total (ms)", fontsize=12)
    ax2.set_title(f"LLM vs Solver Time Scatter ({dataset} set)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


# ============================================================================
# 额外图表
# ============================================================================

def plot_policy_comparison_bars(
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str,
    dataset: str = "test",
    metric: str = "combined_mean",
    ylabel: str = "Combined Score (delay + 5*drift)"
):
    """绘制策略对比条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    filtered = {k: v for k, v in stats.items() if k[0] == dataset}
    
    if not filtered:
        print(f"No data for dataset={dataset}")
        plt.close()
        return
    
    policy_names = sorted([k[1] for k in filtered.keys()])
    values = [filtered[(dataset, p)].get(metric, 0) for p in policy_names]
    
    # CI 如果有
    # CI mapping for known metrics
    metric_ci_map = {
        "mean_delay": "CI_delay",
        "mean_drift": "CI_drift",
        "combined_mean": "combined_ci95",
        "mean_weighted_tardiness": "CI_weighted_tardiness",
        "mean_resource_utilization": "CI_resource_utilization",
        "mean_avg_time_deviation_min": "CI_avg_time_deviation_min",
        "mean_total_resource_switches": "CI_total_resource_switches",
        "mean_makespan_cmax": "CI_makespan_cmax",
        "mean_util_r_pad": "CI_util_r_pad"
    }
    cis = []
    for p in policy_names:
        s = filtered[(dataset, p)]
        ci_key = metric_ci_map.get(metric)
        ci_val = s.get(ci_key, 0) if ci_key else 0
        cis.append(ci_val)
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    x = range(len(policy_names))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    if any(c > 0 for c in cis):
        ax.errorbar(x, values, yerr=cis, fmt='none', color='black', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Policy Comparison ({dataset} set)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def plot_metric_by_disturbance(
    records: List[EpisodeRecord],
    output_path: str,
    metric: str = "avg_delay",
    ylabel: str = "Average Delay",
    dataset: str = "test"
):
    """按扰动级别绘制指标对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        key = (rec.policy_name, rec.disturbance_level)
        groups[key].append(getattr(rec, metric))
    
    levels = ["light", "medium", "heavy"]
    policies = sorted(set(k[0] for k in groups.keys()))
    
    x = range(len(levels))
    width = 0.12
    
    for i, policy in enumerate(policies):
        style = get_policy_style(policy)
        values = [mean(groups.get((policy, level), [0])) for level in levels]
        offset = (i - len(policies)/2 + 0.5) * width
        
        ax.bar([xi + offset for xi in x], values,
               width=width, label=style["label"],
               color=style["color"], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Light", "Medium", "Heavy"])
    ax.set_xlabel("Disturbance Level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} by Disturbance Level ({dataset} set)", fontsize=14)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def _find_rolling_logs(input_dir):
    log_paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name == 'rolling_log.jsonl':
                log_paths.append(os.path.join(root, name))
    return log_paths


def _bucket_value(value, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return f"[{bins[i]}, {bins[i+1]})"
    return f"[{bins[-2]}, {bins[-1]}+)"


def build_feature_bucket_table(input_dir):
    feature_bins = {
        'window_loss_pct': [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, float('inf')],
        'pad_pressure': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float('inf')],
        'slack_min_minutes': [float('-inf'), 0.0, 60.0, 120.0, 240.0, float('inf')],
        'resource_conflict_pressure': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float('inf')],
        'delay_increase_minutes': [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, float('inf')]
    }

    bucket_stats = defaultdict(lambda: {
        'count': 0,
        'plan_drift': 0.0,
        'num_shifts': 0,
        'num_switches': 0,
        'infeasible': 0
    })

    for log_path in _find_rolling_logs(input_dir):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                features = entry.get('state_features')
                metrics = entry.get('metrics', {})
                if not isinstance(features, dict):
                    continue

                plan_drift = float(metrics.get('plan_drift', 0.0))
                num_shifts = int(metrics.get('num_shifts', 0))
                num_switches = int(metrics.get('num_switches', 0))
                is_feasible = metrics.get('is_feasible', True)
                infeasible = 0 if is_feasible else 1

                for feature_name, bins in feature_bins.items():
                    if feature_name not in features:
                        continue
                    value = float(features.get(feature_name, 0.0))
                    bucket = _bucket_value(value, bins)
                    key = (feature_name, bucket)
                    stat = bucket_stats[key]
                    stat['count'] += 1
                    stat['plan_drift'] += plan_drift
                    stat['num_shifts'] += num_shifts
                    stat['num_switches'] += num_switches
                    stat['infeasible'] += infeasible

    rows = []
    for (feature, bucket), stat in sorted(bucket_stats.items()):
        count = stat['count']
        if count == 0:
            continue
        rows.append({
            'feature': feature,
            'bucket': bucket,
            'count': count,
            'avg_plan_drift': round(stat['plan_drift'] / count, 6),
            'avg_num_shifts': round(stat['num_shifts'] / count, 2),
            'avg_num_switches': round(stat['num_switches'] / count, 2),
            'infeasible_rate': round(stat['infeasible'] / count, 4)
        })

    return rows


def save_feature_bucket_table(input_dir, output_path):
    rows = build_feature_bucket_table(input_dir)
    if not rows:
        print('No rolling logs with state_features found for bucket table')
        return

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fieldnames = [
        'feature', 'bucket', 'count',
        'avg_plan_drift', 'avg_num_shifts', 'avg_num_switches', 'infeasible_rate'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved bucket table: {output_path}")


def _find_final_schedule(input_dir):
    for root, _, files in os.walk(input_dir):
        if 'final_schedule.json' in files:
            return os.path.join(root, 'final_schedule.json')
    return None


def plot_gantt_from_schedule(input_dir, output_path, max_items=120):
    schedule_path = _find_final_schedule(input_dir)
    if not schedule_path:
        print('No final_schedule.json found for Gantt plot')
        return

    with open(schedule_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    schedule = data.get('schedule', [])
    if not schedule:
        print('No schedule entries found for Gantt plot')
        return

    items = []
    for entry in schedule:
        start = entry.get('start_slot')
        end = entry.get('end_slot')
        label = entry.get('op_id') or entry.get('task_id') or entry.get('id')
        if start is None or end is None or label is None:
            continue
        items.append((label, start, end))

    items.sort(key=lambda x: x[1])
    items = items[:max_items]

    labels = [i[0] for i in items]
    starts = [i[1] for i in items]
    durations = [max(0, i[2] - i[1]) for i in items]

    fig, ax = plt.subplots(figsize=(12, max(4, len(items) * 0.2)))
    y_positions = range(len(items))
    ax.barh(y_positions, durations, left=starts, color='#3498db', alpha=0.8)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Slot')
    ax.set_title('Gantt Chart (sample schedule)')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Gantt chart: {output_path}")

def plot_tuning_heatmap(
    tuning_results: List[Dict[str, Any]],
    output_path: str
):
    """??????????freeze ? epsilon_solver?"""
    if not tuning_results:
        print("No tuning results to plot")
        return
    if any("epsilon_solver" not in r or "freeze_horizon_slots" not in r for r in tuning_results):
        print("Skip tuning heatmap: missing epsilon_solver/freeze_horizon_slots")
        return
    
    freeze_values = sorted(set(r["freeze_horizon_slots"] for r in tuning_results))
    eps_values = sorted(set(r["epsilon_solver"] for r in tuning_results))
    
    matrix = {}
    for r in tuning_results:
        key = (r["freeze_horizon_slots"], r["epsilon_solver"])
        if key not in matrix:
            matrix[key] = []
        matrix[key].append(r["combined_score"])
    
    data = []
    for freeze in freeze_values:
        row = []
        for eps in eps_values:
            key = (freeze, eps)
            vals = matrix.get(key, [0])
            row.append(mean(vals))
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(eps_values)))
    ax.set_xticklabels([f"{v}" for v in eps_values])
    ax.set_yticks(range(len(freeze_values)))
    ax.set_yticklabels([f"{v}" for v in freeze_values])
    
    ax.set_xlabel("epsilon_solver", fontsize=12)
    ax.set_ylabel("freeze_horizon (slots)", fontsize=12)
    ax.set_title("Tuning: Combined Score (lower is better)", fontsize=14)
    
    plt.colorbar(im, ax=ax, label="Combined Score")
    
    for i in range(len(freeze_values)):
        for j in range(len(eps_values)):
            ax.text(j, i, f'{data[i][j]:.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if data[i][j] > mean([mean(row) for row in data]) else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"????: {output_path}")

def save_summary_csv(
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str
):
    """
    保存汇总 CSV（包含所有要求的字段）
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fieldnames = [
        "dataset", "policy_name", "n",
        "mean_delay", "CI_delay",
        "mean_drift", "CI_drift",
        "combined_mean", "combined_ci95",
        "on_time_mean",
        "mean_weighted_tardiness", "CI_weighted_tardiness",
        "mean_resource_utilization", "CI_resource_utilization",
        "mean_replans", "mean_switch", "mean_shifts",
        "mean_avg_time_deviation_min", "CI_avg_time_deviation_min",
        "mean_total_resource_switches", "CI_total_resource_switches",
        "mean_makespan_cmax", "CI_makespan_cmax",
        "mean_feasible_rate", "mean_forced_replan_rate",
        "mean_avg_frozen", "mean_avg_tasks_scheduled",
        "mean_util_r_pad", "CI_util_r_pad",
        "mean_solver_time_ms", "mean_wall_time_ms",
        "fallback_rate", "cache_hit_rate",
        "mean_llm_time_ms", "mean_llm_total_tokens",
        "total_llm_calls", "total_fallbacks"
    ]
    
    rows = []
    for key in sorted(stats.keys()):
        s = stats[key]
        row = {
            "dataset": s["dataset"],
            "policy_name": s["policy_name"],
            "n": s["n"],
            "mean_delay": round(s["mean_delay"], 3),
            "CI_delay": round(s["CI_delay"], 3),
            "mean_drift": round(s["mean_drift"], 5),
            "CI_drift": round(s["CI_drift"], 5),
            "combined_mean": round(s["combined_mean"], 3),
            "combined_ci95": round(s["combined_ci95"], 3),
            "on_time_mean": round(s["on_time_mean"], 4),
            "mean_replans": round(s["mean_replans"], 2),
            "mean_switch": round(s["mean_switch"], 2),
            "mean_shifts": round(s["mean_shifts"], 2),
            "mean_avg_time_deviation_min": round(s["mean_avg_time_deviation_min"], 2),
            "CI_avg_time_deviation_min": round(s["CI_avg_time_deviation_min"], 2),
            "mean_total_resource_switches": round(s["mean_total_resource_switches"], 2),
            "CI_total_resource_switches": round(s["CI_total_resource_switches"], 2),
            "mean_makespan_cmax": round(s["mean_makespan_cmax"], 2),
            "CI_makespan_cmax": round(s["CI_makespan_cmax"], 2),
            "mean_feasible_rate": round(s["mean_feasible_rate"], 4),
            "mean_forced_replan_rate": round(s["mean_forced_replan_rate"], 4),
            "mean_avg_frozen": round(s["mean_avg_frozen"], 2),
            "mean_avg_tasks_scheduled": round(s["mean_avg_tasks_scheduled"], 2),
            "mean_util_r_pad": round(s["mean_util_r_pad"], 4),
            "CI_util_r_pad": round(s["CI_util_r_pad"], 4),
            "mean_solver_time_ms": round(s["mean_solver_time_ms"], 1),
            "mean_wall_time_ms": round(s["mean_wall_time_ms"], 1),
            "fallback_rate": round(s["fallback_rate"], 4),
            "cache_hit_rate": round(s["cache_hit_rate"], 4),
            "mean_llm_time_ms": round(s["mean_llm_time_ms"], 1),
            "mean_llm_total_tokens": round(s["mean_llm_total_tokens"], 0),
            "total_llm_calls": s["total_llm_calls"],
            "total_fallbacks": s["total_fallbacks"]
        }
        rows.append(row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"保存汇总 CSV: {output_path}")


def save_enhanced_summary_with_tests(
    records: List[EpisodeRecord],
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str,
    tuning_lambda: float = 5.0
):
    """保存增强的汇总 CSV（含配对检验）"""
    
    test_records = [r for r in records if r.dataset == "test"]
    
    by_seed: Dict[int, Dict[str, EpisodeRecord]] = defaultdict(dict)
    for rec in test_records:
        by_seed[rec.seed][rec.policy_name] = rec
    
    baseline_priority = ["fixed_default", "fixed_tuned"]
    baseline = "fixed_tuned"
    baseline_recs = None
    for name in baseline_priority:
        baseline_recs = [r for r in test_records if r.policy_name == name]
        if baseline_recs:
            baseline = name
            break
    if not baseline_recs:
        baseline_recs = []

    base_delay = mean([r.avg_delay for r in baseline_recs]) if baseline_recs else 0.0
    base_drift = mean([r.episode_drift for r in baseline_recs]) if baseline_recs else 0.0
    eps = 1e-9
    base_delay = base_delay if base_delay > eps else eps
    base_drift = base_drift if base_drift > eps else eps
    w1, w2 = 0.5, 0.5
    comparisons = {}
    
    for policy in set(r.policy_name for r in test_records):
        if policy == baseline:
            continue
        
        baseline_vals = []
        other_vals = []
        
        for seed, policies in by_seed.items():
            if baseline in policies and policy in policies:
                b = policies[baseline]
                o = policies[policy]
                b_combined = w1 * (b.avg_delay / base_delay) + w2 * (b.episode_drift / base_drift)
                o_combined = w1 * (o.avg_delay / base_delay) + w2 * (o.episode_drift / base_drift)
                baseline_vals.append(b_combined)
                other_vals.append(o_combined)
        
        if baseline_vals:
            t_stat, p_val = paired_t_test(baseline_vals, other_vals)
            comparisons[policy] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "n_pairs": len(baseline_vals)
            }
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    fieldnames = [
        "dataset", "policy_name", "n",
        "mean_delay", "CI_delay",
        "mean_drift", "CI_drift",
        "combined_mean", "combined_ci95",
        "on_time_mean",
        "mean_weighted_tardiness", "CI_weighted_tardiness",
        "mean_resource_utilization", "CI_resource_utilization",
        "mean_replans", "mean_switch",
        "mean_avg_time_deviation_min", "CI_avg_time_deviation_min",
        "mean_total_resource_switches", "CI_total_resource_switches",
        "mean_makespan_cmax", "CI_makespan_cmax",
        "mean_feasible_rate", "mean_forced_replan_rate",
        "mean_avg_frozen", "mean_avg_tasks_scheduled",
        "mean_util_r_pad", "CI_util_r_pad",
        "mean_solver_time_ms", "mean_wall_time_ms",
        "fallback_rate", "cache_hit_rate",
        "mean_llm_time_ms", "mean_llm_total_tokens",
        "vs_tuned_t_stat", "vs_tuned_p_value"
    ]
    
    rows = []
    for key in sorted(stats.keys()):
        s = stats[key]
        row = {
            "dataset": s["dataset"],
            "policy_name": s["policy_name"],
            "n": s["n"],
            "mean_delay": round(s["mean_delay"], 3),
            "CI_delay": round(s["CI_delay"], 3),
            "mean_drift": round(s["mean_drift"], 5),
            "CI_drift": round(s["CI_drift"], 5),
            "combined_mean": round(s["combined_mean"], 3),
            "combined_ci95": round(s["combined_ci95"], 3),
            "on_time_mean": round(s["on_time_mean"], 4),
            "mean_replans": round(s["mean_replans"], 2),
            "mean_switch": round(s["mean_switch"], 2),
            "mean_avg_time_deviation_min": round(s["mean_avg_time_deviation_min"], 2),
            "CI_avg_time_deviation_min": round(s["CI_avg_time_deviation_min"], 2),
            "mean_total_resource_switches": round(s["mean_total_resource_switches"], 2),
            "CI_total_resource_switches": round(s["CI_total_resource_switches"], 2),
            "mean_makespan_cmax": round(s["mean_makespan_cmax"], 2),
            "CI_makespan_cmax": round(s["CI_makespan_cmax"], 2),
            "mean_feasible_rate": round(s["mean_feasible_rate"], 4),
            "mean_forced_replan_rate": round(s["mean_forced_replan_rate"], 4),
            "mean_avg_frozen": round(s["mean_avg_frozen"], 2),
            "mean_avg_tasks_scheduled": round(s["mean_avg_tasks_scheduled"], 2),
            "mean_util_r_pad": round(s["mean_util_r_pad"], 4),
            "CI_util_r_pad": round(s["CI_util_r_pad"], 4),
            "mean_solver_time_ms": round(s["mean_solver_time_ms"], 1),
            "mean_wall_time_ms": round(s["mean_wall_time_ms"], 1),
            "fallback_rate": round(s["fallback_rate"], 4),
            "cache_hit_rate": round(s["cache_hit_rate"], 4),
            "mean_llm_time_ms": round(s["mean_llm_time_ms"], 1),
            "mean_llm_total_tokens": round(s["mean_llm_total_tokens"], 0),
            "vs_tuned_t_stat": "",
            "vs_tuned_p_value": ""
        }
        
        if s["dataset"] == "test" and s["policy_name"] in comparisons:
            comp = comparisons[s["policy_name"]]
            row["vs_tuned_t_stat"] = round(comp["t_statistic"], 3)
            row["vs_tuned_p_value"] = round(comp["p_value"], 4)
        
        rows.append(row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"保存增强汇总: {output_path}")


# ============================================================================
# LLM Reliability Report
# ============================================================================

def generate_llm_reliability_report(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """
    为 llm_real 策略生成可靠性报告（Markdown 格式）
    """
    # 筛选 llm_real 数据
    llm_records = [r for r in records if r.policy_name == "llm_real" and r.dataset == dataset]
    
    if not llm_records:
        print(f"没有 llm_real 策略的数据，跳过 reliability report")
        return
    
    # 统计计算
    total_calls = sum(r.llm_calls for r in llm_records)
    total_fallbacks = sum(r.llm_fallback_count for r in llm_records)
    total_tokens = sum(r.llm_total_tokens for r in llm_records)
    total_prompt_tokens = sum(r.llm_prompt_tokens for r in llm_records)
    total_completion_tokens = sum(r.llm_completion_tokens for r in llm_records)
    
    # 计算率
    fallback_rate = total_fallbacks / total_calls if total_calls > 0 else 0.0
    
    # cache_hit_rate 加权平均
    weighted_cache = sum(r.llm_cache_hit_rate * r.llm_calls for r in llm_records)
    cache_hit_rate = weighted_cache / total_calls if total_calls > 0 else 0.0
    
    # Token 分布
    tokens_per_call = [r.llm_total_tokens / r.llm_calls if r.llm_calls > 0 else 0 for r in llm_records]
    tokens_per_episode = [r.llm_total_tokens for r in llm_records]
    
    # LLM 时间分布
    llm_times = [r.llm_time_total_ms for r in llm_records]
    latencies = [r.llm_latency_total_ms for r in llm_records]
    
    # 生成 Markdown
    report = f"""# LLM Real Strategy Reliability Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Dataset:** {dataset}  
**Episodes:** {len(llm_records)}

---

## 1. Summary Statistics

| Metric | Value |
|--------|-------|
| Total LLM Calls | {total_calls:,} |
| Total Fallbacks | {total_fallbacks:,} |
| **Fallback Rate** | **{fallback_rate:.2%}** |
| **Cache Hit Rate** | **{cache_hit_rate:.2%}** |

---

## 2. Token Usage

| Metric | Value |
|--------|-------|
| Total Tokens | {total_tokens:,} |
| Prompt Tokens | {total_prompt_tokens:,} |
| Completion Tokens | {total_completion_tokens:,} |
| Avg Tokens/Episode | {mean(tokens_per_episode):.1f} |
| Avg Tokens/Call | {mean(tokens_per_call):.1f} |

### Token Distribution (per episode)

| Percentile | Tokens |
|------------|--------|
| Min | {min(tokens_per_episode) if tokens_per_episode else 0} |
| 25% | {percentile(tokens_per_episode, 25):.0f} |
| 50% (Median) | {percentile(tokens_per_episode, 50):.0f} |
| 75% | {percentile(tokens_per_episode, 75):.0f} |
| Max | {max(tokens_per_episode) if tokens_per_episode else 0} |

---

## 3. Latency Analysis

| Metric | Value |
|--------|-------|
| Mean LLM Time (ms) | {mean(llm_times):.1f} |
| Std LLM Time (ms) | {std(llm_times):.1f} |
| Mean Latency (ms) | {mean(latencies):.1f} |
| Min Latency (ms) | {min(latencies) if latencies else 0} |
| Max Latency (ms) | {max(latencies) if latencies else 0} |

### Latency Distribution (per episode)

| Percentile | Latency (ms) |
|------------|--------------|
| 25% | {percentile(latencies, 25):.0f} |
| 50% | {percentile(latencies, 50):.0f} |
| 75% | {percentile(latencies, 75):.0f} |
| 95% | {percentile(latencies, 95):.0f} |
| 99% | {percentile(latencies, 99):.0f} |

---

## 4. Performance Metrics

| Metric | Mean | Std | CI (95%) |
|--------|------|-----|----------|
| Avg Delay | {mean([r.avg_delay for r in llm_records]):.2f} | {std([r.avg_delay for r in llm_records]):.2f} | ±{ci95([r.avg_delay for r in llm_records]):.2f} |
| Episode Drift | {mean([r.episode_drift for r in llm_records]):.4f} | {std([r.episode_drift for r in llm_records]):.4f} | ±{ci95([r.episode_drift for r in llm_records]):.4f} |
| On-time Rate | {mean([r.on_time_rate for r in llm_records]):.2%} | - | - |
| Replans | {mean([r.num_replans for r in llm_records]):.1f} | {std([r.num_replans for r in llm_records]):.1f} | - |

---

## 5. Reliability Assessment

"""
    
    # 可靠性评估
    if fallback_rate < 0.05:
        reliability_grade = "A (Excellent)"
        reliability_note = "Fallback rate is very low, indicating stable LLM performance."
    elif fallback_rate < 0.10:
        reliability_grade = "B (Good)"
        reliability_note = "Fallback rate is acceptable for production use."
    elif fallback_rate < 0.20:
        reliability_grade = "C (Fair)"
        reliability_note = "Consider investigating fallback causes."
    else:
        reliability_grade = "D (Poor)"
        reliability_note = "High fallback rate indicates LLM reliability issues."
    
    report += f"""### Overall Grade: **{reliability_grade}**

{reliability_note}

### Recommendations

"""
    
    if fallback_rate > 0.05:
        report += "- Review LLM output parsing logic for edge cases\n"
    if cache_hit_rate < 0.3:
        report += "- Consider warming up cache with common scenarios\n"
    if mean(latencies) > 5000:
        report += "- High latency detected, consider timeout optimization\n"
    if total_tokens / len(llm_records) > 500:
        report += "- High token usage, consider prompt optimization\n"
    
    report += "\n---\n\n*Report generated by analyze.py*\n"
    
    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"保存 LLM 可靠性报告: {output_path}")


# ============================================================================
# 控制台输出
# ============================================================================

def print_summary_table(stats: Dict[Tuple[str, str], Dict[str, Any]], dataset: str = "test"):
    """打印汇总表格"""
    print(f"\n{'='*100}")
    print(f" Summary Statistics ({dataset} set)")
    print(f"{'='*100}")
    
    filtered = {k: v for k, v in stats.items() if k[0] == dataset}
    
    if not filtered:
        print("No data available")
        return
    
    # 简化表头
    headers = ["Policy", "N", "Delay", "Drift", "Replans", "Switch", "Fallback%", "Cache%"]
    widths = [15, 5, 15, 15, 8, 8, 10, 10]
    
    header_line = "│"
    for h, w in zip(headers, widths):
        header_line += f" {h:^{w}} │"
    
    sep = "├" + "┼".join(["─" * (w + 2) for w in widths]) + "┤"
    top = "┌" + "┬".join(["─" * (w + 2) for w in widths]) + "┐"
    bottom = "└" + "┴".join(["─" * (w + 2) for w in widths]) + "┘"
    
    print(top)
    print(header_line)
    print(sep)
    
    for key in sorted(filtered.keys(), key=lambda x: x[1]):
        s = filtered[key]
        policy_label = get_policy_style(s["policy_name"])["label"]
        
        row = [
            policy_label[:15],
            str(s["n"]),
            f"{s['mean_delay']:.2f}±{s['CI_delay']:.2f}",
            f"{s['mean_drift']:.4f}±{s['CI_drift']:.4f}",
            f"{s['mean_replans']:.1f}",
            f"{s['mean_switch']:.1f}",
            f"{s['fallback_rate']:.1%}",
            f"{s['cache_hit_rate']:.1%}"
        ]
        
        row_line = "│"
        for cell, w in zip(row, widths):
            row_line += f" {cell:^{w}} │"
        print(row_line)
    
    print(bottom)


# ============================================================================
# 主流程
# ============================================================================

def run_analysis(
    input_dir: str,
    output_dir: str,
    show_plots: bool = False,
    tuning_lambda: float = 5.0,
    epsilon_metric: str = "avg_delay",
    epsilon_relative: str = "baseline",
    epsilon_value: float = 0.10,
    include_llm: bool = False,
    policies: Optional[List[str]] = None,
    logs_dir: Optional[str] = None,
    budget_source: str = "baseline",
    budget_multiplier: float = 1.05,
    budget_value: Optional[float] = None,
    best_params_path: Optional[str] = None,
    episode_seed: Optional[int] = None,
    episode_policy_a: str = "fixed_tuned",
    episode_policy_b: str = "nofreeze"
):
    """
    运行完整分析流程
    """
    print("\n" + "="*70)
    print(" 实验结果分析 (含 LLM 指标)")
    print("="*70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    episode_path = os.path.join(input_dir, "results_per_episode.csv")
    tuning_path = os.path.join(input_dir, "tuning_results.csv")
    
    if not os.path.exists(episode_path):
        print(f"Error: {episode_path} not found")
        return
    
    records = load_episode_results(episode_path)
    tuning_results = load_tuning_results(tuning_path)
    
    print(f"加载 {len(records)} episode 记录")
    print(f"加载 {len(tuning_results)} 调参结果")
    
    # 检测可用的数据集和策略
    available_datasets = set(rec.dataset for rec in records)
    available_policies = set(rec.policy_name for rec in records)
    print(f"可用数据集: {sorted(available_datasets)}")
    print(f"可用策略: {sorted(available_policies)}")
    
    primary_dataset = "test" if "test" in available_datasets else "train"
    print(f"主数据集: {primary_dataset}")

    if policies is None:
        policies = ["fixed_default", "fixed_tuned", "nofreeze"]
    policies = [p for p in policies if p in available_policies]
    if not policies:
        policies = sorted(available_policies)

    logs_dir = logs_dir or os.path.join(input_dir, "logs")
    baseline_policy = "fixed_default"

    delay_budget = compute_delay_budget(
        records=records,
        dataset=primary_dataset,
        baseline_policy=baseline_policy,
        budget_source=budget_source,
        budget_multiplier=budget_multiplier,
        budget_value=budget_value,
        best_params_path=best_params_path
    )

    epsilon_threshold = delay_budget if delay_budget is not None else compute_epsilon_threshold(
        records,
        primary_dataset,
        epsilon_metric,
        epsilon_relative,
        epsilon_value
    )
    if epsilon_threshold is None:
        print("Warning: delay budget not available, skip constraint line")
    else:
        print(f"Delay budget: {epsilon_threshold:.3f} (source={budget_source})")
    
    # 2. 计算统计
    stats = compute_summary_stats(records, tuning_lambda)
    
    # 3. 打印汇总
    for ds in sorted(available_datasets):
        print_summary_table(stats, ds)
    
    # 4. 保存 summary.csv
    summary_path = os.path.join(output_dir, "summary.csv")
    save_summary_csv(stats, summary_path)
    
    # 5. 保存增强汇总（含 t 检验）
    enhanced_path = os.path.join(output_dir, "summary_with_tests.csv")
    save_enhanced_summary_with_tests(records, stats, enhanced_path, tuning_lambda)

    # Baseline-only evidence package (no LLM required)
    main_table_path = os.path.join(output_dir, "main_table_by_disturbance.csv")
    main_rows = compute_main_table_by_disturbance(
        records=records,
        dataset=primary_dataset,
        delay_budget=delay_budget,
        policies=policies
    )
    save_main_table_by_disturbance(main_rows, main_table_path)

    budgeted_path = os.path.join(output_dir, "budgeted_best_drift.csv")
    save_budgeted_best_drift(
        tuning_results=tuning_results,
        records=records,
        dataset=primary_dataset,
        delay_budget=delay_budget,
        baseline_policy=baseline_policy,
        output_path=budgeted_path
    )

    record_index = build_record_index(records, dataset=primary_dataset)
    roll_rows = collect_roll_metrics(logs_dir, record_index)
    roll_stats_path = os.path.join(output_dir, "roll_feasibility_stats.csv")
    save_roll_feasibility_stats(roll_rows, roll_stats_path, policies=policies)

    if episode_seed is not None:
        episode_compare_path = os.path.join(output_dir, "episode_compare.png")
        plot_episode_compare(
            logs_dir=logs_dir,
            seed=episode_seed,
            policy_a=episode_policy_a,
            policy_b=episode_policy_b,
            output_path=episode_compare_path
        )
    
    # 6. 生成图表
    print("\n生成图表...")
    
    # 图表 1: Delay vs Drift 散点图
    plot_delay_vs_drift_scatter(
        records,
        os.path.join(output_dir, "delay_vs_drift_scatter.png"),
        dataset=primary_dataset,
        epsilon_threshold=epsilon_threshold
    )
    
    # 图表 2: 重排/切换次数分布
    plot_replans_switches_boxplot(
        records,
        os.path.join(output_dir, "replans_switches_boxplot.png"),
        dataset=primary_dataset
    )
    
    # 图表 3: LLM Time vs Solver Time
    if include_llm:
        plot_llm_vs_solver_time(
            records,
            os.path.join(output_dir, "llm_vs_solver_time.png"),
            dataset=primary_dataset
        )
    
    # 额外图表
    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_combined.png"),
        dataset=primary_dataset,
        metric="combined_mean",
        ylabel="Normalized Cost (baseline=1.0)"
    )
    
    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_delay.png"),
        dataset=primary_dataset,
        metric="mean_delay",
        ylabel="Average Delay (slots)"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_weighted_tardiness.png"),
        dataset=primary_dataset,
        metric="mean_weighted_tardiness",
        ylabel="Weighted Tardiness"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_resource_utilization.png"),
        dataset=primary_dataset,
        metric="mean_resource_utilization",
        ylabel="Resource Utilization"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_util_r_pad.png"),
        dataset=primary_dataset,
        metric="mean_util_r_pad",
        ylabel="Utilization (R_pad)"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_makespan_cmax.png"),
        dataset=primary_dataset,
        metric="mean_makespan_cmax",
        ylabel="Makespan Cmax (slots)"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_avg_time_deviation_min.png"),
        dataset=primary_dataset,
        metric="mean_avg_time_deviation_min",
        ylabel="Avg Time Deviation (min)"
    )

    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_total_resource_switches.png"),
        dataset=primary_dataset,
        metric="mean_total_resource_switches",
        ylabel="Total Resource Switches"
    )
    
    plot_metric_by_disturbance(
        records,
        os.path.join(output_dir, "delay_by_disturbance.png"),
        metric="avg_delay",
        ylabel="Average Delay",
        dataset=primary_dataset
    )
    
    plot_metric_by_disturbance(
        records,
        os.path.join(output_dir, "drift_by_disturbance.png"),
        metric="episode_drift",
        ylabel="Episode Drift",
        dataset=primary_dataset
    )
    
    # ??????? tuning_results ????
    can_plot_tuning = bool(tuning_results) and all(
        "epsilon_solver" in r and "freeze_horizon_slots" in r for r in tuning_results
    )
    if can_plot_tuning:
        plot_tuning_heatmap(
            tuning_results,
            os.path.join(output_dir, "tuning_heatmap.png")
        )
    else:
        print("Skip tuning heatmap: tuning_results empty or missing fields")

    # Feature buckets and Gantt chart (if logs exist)
    save_feature_bucket_table(input_dir, os.path.join(output_dir, "feature_bucket_table.csv"))
    plot_gantt_from_schedule(input_dir, os.path.join(output_dir, "gantt_chart.png"))
    
    # 7. LLM Reliability Report（如果有 llm_real 数据）
    if include_llm and "llm_real" in available_policies:
        report_path = os.path.join(output_dir, "reliability_report.md")
        generate_llm_reliability_report(records, report_path, dataset=primary_dataset)
    
    # 完成
    print(f"\n{'='*70}")
    print(" 分析完成！")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {summary_path}")
    print(f"  - {enhanced_path}")
    print(f"  - {output_dir}/main_table_by_disturbance.csv")
    print(f"  - {output_dir}/budgeted_best_drift.csv")
    print(f"  - {output_dir}/roll_feasibility_stats.csv")
    if episode_seed is not None:
        print(f"  - {output_dir}/episode_compare.png")
    print(f"  - {output_dir}/delay_vs_drift_scatter.png")
    print(f"  - {output_dir}/replans_switches_boxplot.png")
    if include_llm:
        print(f"  - {output_dir}/llm_vs_solver_time.png")
    print(f"  - {output_dir}/policy_comparison_*.png")
    print(f"  - {output_dir}/policy_comparison_util_r_pad.png")
    print(f"  - {output_dir}/policy_comparison_makespan_cmax.png")
    print(f"  - {output_dir}/policy_comparison_avg_time_deviation_min.png")
    print(f"  - {output_dir}/policy_comparison_total_resource_switches.png")
    print(f"  - {output_dir}/delay_by_disturbance.png")
    print(f"  - {output_dir}/drift_by_disturbance.png")
    if can_plot_tuning:
        print(f"  - {output_dir}/tuning_heatmap.png")
    if include_llm and "llm_real" in available_policies:
        print(f"  - {output_dir}/reliability_report.md")
    
    if show_plots:
        plt.show()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验结果分析与绘图（支持 LLM 指标）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本分析
  python analyze.py --input results/ --output figures/

  # 显示图表
  python analyze.py --input results/ --output figures/ --show

  # 自定义 lambda
  python analyze.py --input results/ --output figures/ --lambda 10.0
"""
    )
    parser.add_argument(
        "--input", "-i", type=str, default="results",
        help="输入目录（包含 results_per_episode.csv）(default: results/)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="figures",
        help="输出目录（保存图表和报告）(default: figures/)"
    )
    parser.add_argument(
        "--lambda", dest="tuning_lambda", type=float, default=5.0,
        help="综合目标中 drift 的权重 (default: 5.0)"
    )
    parser.add_argument(
        "--epsilon-metric", type=str, default="avg_delay",
        choices=["avg_delay"],
        help="epsilon constraint metric (default: avg_delay)"
    )
    parser.add_argument(
        "--epsilon-relative", type=str, default="baseline",
        choices=["baseline", "absolute"],
        help="epsilon threshold type: baseline or absolute (default: baseline)"
    )
    parser.add_argument(
        "--epsilon-value", type=float, default=0.10,
        help="baseline: relative tolerance (e.g. 0.10); absolute: threshold value"
    )
    parser.add_argument(
        "--include-llm", action="store_true",
        help="include LLM-related outputs (default: off)"
    )
    parser.add_argument(
        "--policies", type=str, default="fixed_default,fixed_tuned,nofreeze",
        help="comma-separated policy filter for baseline tables"
    )
    parser.add_argument(
        "--logs-dir", type=str, default=None,
        help="logs directory containing metrics_per_roll.csv (default: input/logs)"
    )
    parser.add_argument(
        "--budget-source", type=str, default="baseline",
        choices=["baseline", "best_params", "manual"],
        help="delay budget source (default: baseline)"
    )
    parser.add_argument(
        "--budget-multiplier", type=float, default=1.05,
        help="budget multiplier on baseline avg_delay (default: 1.05)"
    )
    parser.add_argument(
        "--budget-value", type=float, default=None,
        help="manual budget value (used when budget-source=manual)"
    )
    parser.add_argument(
        "--best-params", type=str, default=None,
        help="best_params.json path (used when budget-source=best_params)"
    )
    parser.add_argument(
        "--episode-seed", type=int, default=None,
        help="seed for episode compare plot (optional)"
    )
    parser.add_argument(
        "--episode-policy-a", type=str, default="fixed_tuned",
        help="policy A for episode compare plot"
    )
    parser.add_argument(
        "--episode-policy-b", type=str, default="nofreeze",
        help="policy B for episode compare plot"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="显示图表窗口"
    )
    
    args = parser.parse_args()

    policies = [p.strip() for p in (args.policies or "").split(",") if p.strip()]
    
    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        show_plots=args.show,
        tuning_lambda=args.tuning_lambda,
        epsilon_metric=args.epsilon_metric,
        epsilon_relative=args.epsilon_relative,
        epsilon_value=args.epsilon_value,
        include_llm=args.include_llm,
        policies=policies,
        logs_dir=args.logs_dir,
        budget_source=args.budget_source,
        budget_multiplier=args.budget_multiplier,
        budget_value=args.budget_value,
        best_params_path=args.best_params,
        episode_seed=args.episode_seed,
        episode_policy_a=args.episode_policy_a,
        episode_policy_b=args.episode_policy_b
    )


if __name__ == "__main__":
    main()
