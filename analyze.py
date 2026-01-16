#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析与绘图脚本 - 读取实验结果 CSV，计算统计，生成图表

功能：
1. 读取 results_per_episode.csv
2. 计算汇总统计（均值、95% CI、配对 t 检验）
3. 生成图表：
   - Delay vs PlanDrift scatter
   - 重排次数分布
   - 切换次数分布
   - 各策略对比条形图

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
from dataclasses import dataclass
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据加载
# ============================================================================

@dataclass
class EpisodeRecord:
    """单 episode 记录"""
    seed: int
    disturbance_level: str
    policy_name: str
    dataset: str
    completed: int
    total: int
    on_time_rate: float
    avg_delay: float
    max_delay: int
    episode_drift: float
    total_shifts: int
    total_switches: int
    num_replans: int
    num_forced_replans: int
    avg_solve_time_ms: float
    total_runtime_s: float


def load_episode_results(filepath: str) -> List[EpisodeRecord]:
    """加载 episode 结果 CSV"""
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
                episode_drift=float(row['episode_drift']),
                total_shifts=int(row['total_shifts']),
                total_switches=int(row['total_switches']),
                num_replans=int(row['num_replans']),
                num_forced_replans=int(row['num_forced_replans']),
                avg_solve_time_ms=float(row['avg_solve_time_ms']),
                total_runtime_s=float(row['total_runtime_s'])
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
    # 对于 n > 30，t 分布近似正态
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    
    return t_stat, p_value


def compute_summary_stats(
    records: List[EpisodeRecord],
    tuning_lambda: float = 5.0
) -> Dict[str, Dict[str, Any]]:
    """
    计算汇总统计
    
    Returns:
        {(dataset, policy_name): {metric: value, ...}}
    """
    # 按 (dataset, policy_name) 分组
    groups: Dict[Tuple[str, str], List[EpisodeRecord]] = defaultdict(list)
    for rec in records:
        key = (rec.dataset, rec.policy_name)
        groups[key].append(rec)
    
    stats = {}
    for key, recs in groups.items():
        dataset, policy_name = key
        
        delays = [r.avg_delay for r in recs]
        drifts = [r.episode_drift for r in recs]
        on_times = [r.on_time_rate for r in recs]
        shifts = [r.total_shifts for r in recs]
        switches = [r.total_switches for r in recs]
        replans = [r.num_replans for r in recs]
        solve_times = [r.avg_solve_time_ms for r in recs]
        
        combined = [d + tuning_lambda * dr for d, dr in zip(delays, drifts)]
        
        stats[key] = {
            "dataset": dataset,
            "policy_name": policy_name,
            "n": len(recs),
            "delay_mean": mean(delays),
            "delay_std": std(delays),
            "delay_ci95": ci95(delays),
            "drift_mean": mean(drifts),
            "drift_std": std(drifts),
            "drift_ci95": ci95(drifts),
            "combined_mean": mean(combined),
            "combined_ci95": ci95(combined),
            "on_time_mean": mean(on_times),
            "shifts_mean": mean(shifts),
            "switches_mean": mean(switches),
            "replans_mean": mean(replans),
            "solve_time_mean": mean(solve_times)
        }
    
    return stats


# ============================================================================
# 绘图函数
# ============================================================================

# 策略颜色和标记配置
POLICY_STYLES = {
    "fixed_tuned": {"color": "#2ecc71", "marker": "o", "label": "Fixed (Tuned)"},
    "fixed_default": {"color": "#3498db", "marker": "s", "label": "Fixed (Default)"},
    "nofreeze": {"color": "#e74c3c", "marker": "^", "label": "NoFreeze"},
    "greedy": {"color": "#9b59b6", "marker": "D", "label": "Greedy"},
    "mockllm": {"color": "#f39c12", "marker": "v", "label": "MockLLM"},
    "fixed": {"color": "#3498db", "marker": "s", "label": "Fixed"}
}


def get_policy_style(policy_name: str) -> dict:
    """获取策略样式"""
    return POLICY_STYLES.get(policy_name, {
        "color": "#7f8c8d", "marker": "x", "label": policy_name
    })


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
            s=50,
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


def plot_replans_distribution(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """绘制重排次数分布图（箱线图）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按策略分组
    policy_data: Dict[str, List[int]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        policy_data[rec.policy_name].append(rec.num_replans)
    
    # 排序策略
    policy_names = sorted(policy_data.keys())
    data = [policy_data[p] for p in policy_names]
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    # 绘制箱线图
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Number of Replans", fontsize=12)
    ax.set_title(f"Replan Count Distribution ({dataset} set)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def plot_switches_distribution(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """绘制 Pad 切换次数分布图（箱线图）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按策略分组
    policy_data: Dict[str, List[int]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        policy_data[rec.policy_name].append(rec.total_switches)
    
    policy_names = sorted(policy_data.keys())
    data = [policy_data[p] for p in policy_names]
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Number of Pad Switches", fontsize=12)
    ax.set_title(f"Pad Switch Count Distribution ({dataset} set)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def plot_policy_comparison_bars(
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str,
    dataset: str = "test",
    metric: str = "combined_mean",
    ylabel: str = "Combined Score (delay + 5*drift)"
):
    """绘制策略对比条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 筛选指定数据集
    filtered = {k: v for k, v in stats.items() if k[0] == dataset}
    
    if not filtered:
        print(f"No data for dataset={dataset}")
        return
    
    # 排序
    policy_names = sorted([k[1] for k in filtered.keys()])
    values = [filtered[(dataset, p)][metric] for p in policy_names]
    
    # CI 如果有
    ci_key = metric.replace("_mean", "_ci95")
    cis = []
    for p in policy_names:
        s = filtered[(dataset, p)]
        cis.append(s.get(ci_key, 0))
    
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    x = range(len(policy_names))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # 添加误差线
    if any(c > 0 for c in cis):
        ax.errorbar(x, values, yerr=cis, fmt='none', color='black', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Policy Comparison ({dataset} set)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
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
    
    # 按 (policy, level) 分组
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        key = (rec.policy_name, rec.disturbance_level)
        groups[key].append(getattr(rec, metric))
    
    levels = ["light", "medium", "heavy"]
    policies = sorted(set(k[0] for k in groups.keys()))
    
    x = range(len(levels))
    width = 0.15
    
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
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def plot_tuning_heatmap(
    tuning_results: List[Dict[str, Any]],
    output_path: str
):
    """绘制调参结果热力图"""
    if not tuning_results:
        print("No tuning results to plot")
        return
    
    # 提取唯一值
    freeze_values = sorted(set(r["freeze_horizon_slots"] for r in tuning_results))
    wshift_values = sorted(set(r["w_shift"] for r in tuning_results))
    
    # 构建矩阵（取 w_switch 的平均）
    matrix = {}
    for r in tuning_results:
        key = (r["freeze_horizon_slots"], r["w_shift"])
        if key not in matrix:
            matrix[key] = []
        matrix[key].append(r["combined_score"])
    
    # 转为 2D 数组
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
    
    # 添加数值
    for i in range(len(freeze_values)):
        for j in range(len(wshift_values)):
            ax.text(j, i, f'{data[i][j]:.1f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if data[i][j] > mean([mean(row) for row in data]) else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


def plot_solve_time_comparison(
    records: List[EpisodeRecord],
    output_path: str,
    dataset: str = "test"
):
    """绘制求解时间对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policy_data: Dict[str, List[float]] = defaultdict(list)
    
    for rec in records:
        if rec.dataset != dataset:
            continue
        policy_data[rec.policy_name].append(rec.avg_solve_time_ms)
    
    policy_names = sorted(policy_data.keys())
    means = [mean(policy_data[p]) for p in policy_names]
    stds = [std(policy_data[p]) for p in policy_names]
    colors = [get_policy_style(p)["color"] for p in policy_names]
    labels = [get_policy_style(p)["label"] for p in policy_names]
    
    x = range(len(policy_names))
    ax.bar(x, means, color=colors, alpha=0.8, yerr=stds, capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Average Solve Time (ms)", fontsize=12)
    ax.set_title(f"Solver Performance ({dataset} set)", fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"保存图表: {output_path}")


# ============================================================================
# 汇总输出
# ============================================================================

def print_summary_table(stats: Dict[Tuple[str, str], Dict[str, Any]], dataset: str = "test"):
    """打印汇总表格"""
    print(f"\n{'='*80}")
    print(f" Summary Statistics ({dataset} set)")
    print(f"{'='*80}")
    
    filtered = {k: v for k, v in stats.items() if k[0] == dataset}
    
    if not filtered:
        print("No data available")
        return
    
    # 表头
    headers = ["Policy", "N", "Delay", "Drift", "Combined", "OnTime%", "Shifts", "Switches"]
    widths = [15, 5, 15, 15, 15, 10, 8, 10]
    
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
            f"{s['delay_mean']:.2f}±{s['delay_ci95']:.2f}",
            f"{s['drift_mean']:.4f}±{s['drift_ci95']:.4f}",
            f"{s['combined_mean']:.2f}±{s['combined_ci95']:.2f}",
            f"{s['on_time_mean']:.1%}",
            f"{s['shifts_mean']:.1f}",
            f"{s['switches_mean']:.1f}"
        ]
        
        row_line = "│"
        for cell, w in zip(row, widths):
            row_line += f" {cell:^{w}} │"
        print(row_line)
    
    print(bottom)


def save_enhanced_summary(
    records: List[EpisodeRecord],
    stats: Dict[Tuple[str, str], Dict[str, Any]],
    output_path: str,
    tuning_lambda: float = 5.0
):
    """保存增强的汇总 CSV（含配对检验）"""
    
    # 配对 t 检验：fixed_tuned vs 其他策略
    test_records = [r for r in records if r.dataset == "test"]
    
    # 按 seed 匹配
    by_seed: Dict[int, Dict[str, EpisodeRecord]] = defaultdict(dict)
    for rec in test_records:
        by_seed[rec.seed][rec.policy_name] = rec
    
    # 计算配对检验
    baseline = "fixed_tuned"
    comparisons = {}
    
    for policy in set(r.policy_name for r in test_records):
        if policy == baseline:
            continue
        
        baseline_vals = []
        other_vals = []
        
        for seed, policies in by_seed.items():
            if baseline in policies and policy in policies:
                b_combined = policies[baseline].avg_delay + tuning_lambda * policies[baseline].episode_drift
                o_combined = policies[policy].avg_delay + tuning_lambda * policies[policy].episode_drift
                baseline_vals.append(b_combined)
                other_vals.append(o_combined)
        
        if baseline_vals:
            t_stat, p_val = paired_t_test(baseline_vals, other_vals)
            comparisons[policy] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "n_pairs": len(baseline_vals),
                "baseline_mean": mean(baseline_vals),
                "other_mean": mean(other_vals)
            }
    
    # 输出文件
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    rows = []
    for key, s in sorted(stats.items()):
        row = {
            "dataset": s["dataset"],
            "policy_name": s["policy_name"],
            "n": s["n"],
            "delay_mean": round(s["delay_mean"], 3),
            "delay_ci95": round(s["delay_ci95"], 3),
            "drift_mean": round(s["drift_mean"], 5),
            "drift_ci95": round(s["drift_ci95"], 5),
            "combined_mean": round(s["combined_mean"], 3),
            "combined_ci95": round(s["combined_ci95"], 3),
            "on_time_mean": round(s["on_time_mean"], 4),
            "shifts_mean": round(s["shifts_mean"], 2),
            "switches_mean": round(s["switches_mean"], 2),
            "solve_time_mean": round(s["solve_time_mean"], 1)
        }
        
        # 添加配对检验结果
        if s["dataset"] == "test" and s["policy_name"] in comparisons:
            comp = comparisons[s["policy_name"]]
            row["vs_tuned_t_stat"] = round(comp["t_statistic"], 3)
            row["vs_tuned_p_value"] = round(comp["p_value"], 4)
        else:
            row["vs_tuned_t_stat"] = ""
            row["vs_tuned_p_value"] = ""
        
        rows.append(row)
    
    fieldnames = list(rows[0].keys()) if rows else []
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"保存增强汇总: {output_path}")


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
    print(" 实验结果分析")
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
    
    # 2. 计算统计
    stats = compute_summary_stats(records, tuning_lambda)
    
    # 3. 打印汇总
    print_summary_table(stats, "test")
    print_summary_table(stats, "train")
    
    # 4. 保存增强汇总
    enhanced_summary_path = os.path.join(output_dir, "summary_with_tests.csv")
    save_enhanced_summary(records, stats, enhanced_summary_path, tuning_lambda)
    
    # 5. 生成图表
    print("\n生成图表...")
    
    # Delay vs Drift 散点图
    plot_delay_vs_drift_scatter(
        records,
        os.path.join(output_dir, "delay_vs_drift_scatter.png"),
        dataset="test"
    )
    
    # 重排次数分布
    plot_replans_distribution(
        records,
        os.path.join(output_dir, "replans_distribution.png"),
        dataset="test"
    )
    
    # 切换次数分布
    plot_switches_distribution(
        records,
        os.path.join(output_dir, "switches_distribution.png"),
        dataset="test"
    )
    
    # 策略对比条形图
    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_combined.png"),
        dataset="test",
        metric="combined_mean",
        ylabel="Combined Score (delay + 5*drift)"
    )
    
    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_delay.png"),
        dataset="test",
        metric="delay_mean",
        ylabel="Average Delay (slots)"
    )
    
    plot_policy_comparison_bars(
        stats,
        os.path.join(output_dir, "policy_comparison_drift.png"),
        dataset="test",
        metric="drift_mean",
        ylabel="Episode Plan Drift"
    )
    
    # 按扰动级别的对比
    plot_metric_by_disturbance(
        records,
        os.path.join(output_dir, "delay_by_disturbance.png"),
        metric="avg_delay",
        ylabel="Average Delay",
        dataset="test"
    )
    
    plot_metric_by_disturbance(
        records,
        os.path.join(output_dir, "drift_by_disturbance.png"),
        metric="episode_drift",
        ylabel="Episode Drift",
        dataset="test"
    )
    
    # 求解时间对比
    plot_solve_time_comparison(
        records,
        os.path.join(output_dir, "solve_time_comparison.png"),
        dataset="test"
    )
    
    # 调参热力图
    if tuning_results:
        plot_tuning_heatmap(
            tuning_results,
            os.path.join(output_dir, "tuning_heatmap.png")
        )
    
    print(f"\n{'='*70}")
    print(" 分析完成！")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {enhanced_summary_path}")
    print(f"  - {output_dir}/delay_vs_drift_scatter.png")
    print(f"  - {output_dir}/replans_distribution.png")
    print(f"  - {output_dir}/switches_distribution.png")
    print(f"  - {output_dir}/policy_comparison_*.png")
    print(f"  - {output_dir}/delay_by_disturbance.png")
    print(f"  - {output_dir}/drift_by_disturbance.png")
    print(f"  - {output_dir}/solve_time_comparison.png")
    if tuning_results:
        print(f"  - {output_dir}/tuning_heatmap.png")
    
    if show_plots:
        plt.show()


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验结果分析与绘图"
    )
    parser.add_argument(
        "--input", "-i", type=str, default="results",
        help="输入目录（包含 results_per_episode.csv）"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="figures",
        help="输出目录（保存图表）"
    )
    parser.add_argument(
        "--lambda", dest="tuning_lambda", type=float, default=5.0,
        help="综合目标中 drift 的权重"
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
