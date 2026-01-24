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
    """加载调参结果 CSV"""
    results = []
    
    if not os.path.exists(filepath):
        return results
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "freeze_horizon_slots": int(row['freeze_horizon_slots']),
                "w_delay": float(row['w_delay']),
                "w_shift": float(row['w_shift']),
                "w_switch": float(row['w_switch']),
                "avg_delay": float(row['avg_delay']),
                "avg_drift": float(row['avg_drift']),
                "combined_score": float(row['combined_score'])
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
    baseline_priority = ["fixed_default", "fixed_tuned", "fixed"]
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
# 绘图样式配置
# ============================================================================

POLICY_STYLES = {
    "fixed_tuned": {"color": "#2ecc71", "marker": "o", "label": "Fixed (Tuned)"},
    "fixed_default": {"color": "#3498db", "marker": "s", "label": "Fixed (Default)"},
    "nofreeze": {"color": "#e74c3c", "marker": "^", "label": "NoFreeze"},
    "greedy": {"color": "#9b59b6", "marker": "D", "label": "Priority Rule (EDD)"},
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
    title: str = "Delay vs Plan Drift"
):
    """
    绘制 Delay vs PlanDrift 散点图
    
    每个策略不同颜色和标记
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 按策略分组
    policy_data: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        policy_data[rec.policy_name][0].append(rec.avg_delay)
        policy_data[rec.policy_name][1].append(rec.episode_drift)
    
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
    """绘制调参结果热力图"""
    if not tuning_results:
        print("No tuning results to plot")
        return
    
    freeze_values = sorted(set(r["freeze_horizon_slots"] for r in tuning_results))
    wshift_values = sorted(set(r["w_shift"] for r in tuning_results))
    
    matrix = {}
    for r in tuning_results:
        key = (r["freeze_horizon_slots"], r["w_shift"])
        if key not in matrix:
            matrix[key] = []
        matrix[key].append(r["combined_score"])
    
    data = []
    for freeze in freeze_values:
        row = []
        for wshift in wshift_values:
            key = (freeze, wshift)
            vals = matrix.get(key, [0])
            row.append(mean(vals))
        data.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(len(wshift_values)))
    ax.set_xticklabels([f"{v}" for v in wshift_values])
    ax.set_yticks(range(len(freeze_values)))
    ax.set_yticklabels([f"{v}" for v in freeze_values])
    
    ax.set_xlabel("w_shift", fontsize=12)
    ax.set_ylabel("freeze_horizon (slots)", fontsize=12)
    ax.set_title("Tuning: Combined Score (lower is better)", fontsize=14)
    
    plt.colorbar(im, ax=ax, label="Combined Score")
    
    for i in range(len(freeze_values)):
        for j in range(len(wshift_values)):
            ax.text(j, i, f'{data[i][j]:.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if data[i][j] > mean([mean(row) for row in data]) else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


# ============================================================================
# Summary CSV 输出
# ============================================================================

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
    
    baseline_priority = ["fixed_default", "fixed_tuned", "fixed"]
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
    tuning_lambda: float = 5.0
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
    
    # 6. 生成图表
    print("\n生成图表...")
    
    # 图表 1: Delay vs Drift 散点图
    plot_delay_vs_drift_scatter(
        records,
        os.path.join(output_dir, "delay_vs_drift_scatter.png"),
        dataset=primary_dataset
    )
    
    # 图表 2: 重排/切换次数分布
    plot_replans_switches_boxplot(
        records,
        os.path.join(output_dir, "replans_switches_boxplot.png"),
        dataset=primary_dataset
    )
    
    # 图表 3: LLM Time vs Solver Time
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
    
    # 调参热力图
    if tuning_results:
        plot_tuning_heatmap(
            tuning_results,
            os.path.join(output_dir, "tuning_heatmap.png")
        )

    # Feature buckets and Gantt chart (if logs exist)
    save_feature_bucket_table(input_dir, os.path.join(output_dir, "feature_bucket_table.csv"))
    plot_gantt_from_schedule(input_dir, os.path.join(output_dir, "gantt_chart.png"))
    
    # 7. LLM Reliability Report（如果有 llm_real 数据）
    if "llm_real" in available_policies:
        report_path = os.path.join(output_dir, "reliability_report.md")
        generate_llm_reliability_report(records, report_path, dataset=primary_dataset)
    
    # 完成
    print(f"\n{'='*70}")
    print(" 分析完成！")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {summary_path}")
    print(f"  - {enhanced_path}")
    print(f"  - {output_dir}/delay_vs_drift_scatter.png")
    print(f"  - {output_dir}/replans_switches_boxplot.png")
    print(f"  - {output_dir}/llm_vs_solver_time.png")
    print(f"  - {output_dir}/policy_comparison_*.png")
    print(f"  - {output_dir}/policy_comparison_util_r_pad.png")
    print(f"  - {output_dir}/policy_comparison_makespan_cmax.png")
    print(f"  - {output_dir}/policy_comparison_avg_time_deviation_min.png")
    print(f"  - {output_dir}/policy_comparison_total_resource_switches.png")
    print(f"  - {output_dir}/delay_by_disturbance.png")
    print(f"  - {output_dir}/drift_by_disturbance.png")
    if tuning_results:
        print(f"  - {output_dir}/tuning_heatmap.png")
    if "llm_real" in available_policies:
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
        "--show", action="store_true",
        help="显示图表窗口"
    )
    
    args = parser.parse_args()
    
    run_analysis(
        input_dir=args.input,
        output_dir=args.output,
        show_plots=args.show,
        tuning_lambda=args.tuning_lambda
    )


if __name__ == "__main__":
    main()
