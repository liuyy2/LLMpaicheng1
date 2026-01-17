#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验脚本 - 训练集调参 + 测试集评估

功能：
1. 生成指定数量的 episodes（轻/中/重扰动各占 1/3）
2. baseline 调参：在训练集上网格搜索最优参数
3. 测试集评估：配对比较所有策略
4. 输出 CSV 结果文件

用法：
    python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/
    python run_experiments.py --quick  # 快速测试模式 (较少 seeds)
"""

import argparse
import csv
import json
import os
import time
import itertools
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

from config import Config, DEFAULT_CONFIG
from scenario import generate_scenario, Scenario
from simulator import simulate_episode, EpisodeResult
from policies import (
    FixedWeightPolicy, NoFreezePolicy, GreedyPolicy, MockLLMPolicy,
    create_policy
)


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集
    num_train_seeds: int = 60
    num_test_seeds: int = 60
    
    # 扰动强度分布（轻/中/重各 1/3）
    disturbance_levels: List[str] = field(default_factory=lambda: ["light", "medium", "heavy"])
    
    # 调参网格
    freeze_horizon_hours: List[float] = field(default_factory=lambda: [0, 2, 6, 12])
    w_delay_values: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 50.0])
    w_shift_values: List[float] = field(default_factory=lambda: [0.0, 0.2, 1.0, 2.0])
    w_switch_values: List[float] = field(default_factory=lambda: [0, 60, 180, 600])
    trigger_window_loss_pct: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.4])
    
    # 综合目标权重
    tuning_lambda: float = 5.0  # delay + lambda * drift
    
    # 求解器
    solver_timeout_s: float = 10.0  # 每次求解限时
    
    # 并行处理
    max_workers: int = 1  # 并行进程数（1为串行，设置>1启用并行）
    
    # 输出目录
    output_dir: str = "results"


# 扰动强度对应的 Config 参数
DISTURBANCE_CONFIGS = {
    "light": {
        "p_weather": 0.01,
        "p_pad_outage": 0.005,
        "sigma_duration": 0.05,
        "sigma_release": 1.0
    },
    "medium": {
        "p_weather": 0.02,
        "p_pad_outage": 0.01,
        "sigma_duration": 0.1,
        "sigma_release": 2.0
    },
    "heavy": {
        "p_weather": 0.04,
        "p_pad_outage": 0.02,
        "sigma_duration": 0.15,
        "sigma_release": 3.0
    }
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EpisodeMetricsRecord:
    """单 episode 指标记录"""
    seed: int
    disturbance_level: str
    policy_name: str
    dataset: str  # "train" or "test"
    
    # 核心指标
    completed: int
    total: int
    on_time_rate: float
    avg_delay: float
    max_delay: int
    episode_drift: float
    
    # 稳定性
    total_shifts: int
    total_switches: int
    num_replans: int
    num_forced_replans: int
    
    # 性能
    avg_solve_time_ms: float
    total_runtime_s: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TuningResult:
    """调参结果记录"""
    freeze_horizon_slots: int
    w_delay: float
    w_shift: float
    w_switch: float
    
    # 训练集指标
    avg_delay: float
    avg_drift: float
    combined_score: float  # delay + lambda * drift
    
    # 统计
    num_episodes: int
    avg_solve_time_ms: float


# ============================================================================
# 场景生成
# ============================================================================

def create_config_for_disturbance(level: str, solver_timeout: float = 10.0) -> Config:
    """根据扰动强度创建配置"""
    params = DISTURBANCE_CONFIGS.get(level, DISTURBANCE_CONFIGS["medium"])
    
    return Config(
        p_weather=params["p_weather"],
        p_pad_outage=params["p_pad_outage"],
        sigma_duration=params["sigma_duration"],
        sigma_release=params["sigma_release"],
        solver_timeout_s=solver_timeout
    )


def generate_seed_assignments(
    num_train: int,
    num_test: int,
    levels: List[str] = ["light", "medium", "heavy"]
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    生成种子分配，确保轻/中/重扰动各占 1/3
    
    Returns:
        (train_assignments, test_assignments)
        每个元素为 (seed, disturbance_level)
    """
    num_levels = len(levels)
    
    # 训练集
    train_per_level = num_train // num_levels
    train_assignments = []
    for i, level in enumerate(levels):
        for j in range(train_per_level):
            seed = i * train_per_level + j
            train_assignments.append((seed, level))
    # 处理余数
    for j in range(num_train % num_levels):
        seed = num_levels * train_per_level + j
        train_assignments.append((seed, levels[j % num_levels]))
    
    # 测试集
    test_per_level = num_test // num_levels
    test_assignments = []
    base_seed = num_train  # 测试集 seed 从训练集之后开始
    for i, level in enumerate(levels):
        for j in range(test_per_level):
            seed = base_seed + i * test_per_level + j
            test_assignments.append((seed, level))
    # 处理余数
    for j in range(num_test % num_levels):
        seed = base_seed + num_levels * test_per_level + j
        test_assignments.append((seed, levels[j % num_levels]))
    
    return train_assignments, test_assignments


# ============================================================================
# 单次运行
# ============================================================================

def run_single_episode(
    seed: int,
    disturbance_level: str,
    policy_name: str,
    policy_params: Dict[str, Any],
    dataset: str,
    solver_timeout: float = 10.0
) -> EpisodeMetricsRecord:
    """
    运行单个 episode
    
    Args:
        seed: 随机种子
        disturbance_level: 扰动强度
        policy_name: 策略名称
        policy_params: 策略参数
        dataset: 数据集标识 ("train" / "test")
        solver_timeout: 求解超时
    
    Returns:
        EpisodeMetricsRecord
    """
    # 创建配置
    config = create_config_for_disturbance(disturbance_level, solver_timeout)
    
    # 生成场景
    scenario = generate_scenario(seed=seed, config=config)
    
    # 创建策略
    # fixed, fixed_tuned, fixed_default 都使用 FixedWeightPolicy
    if policy_name in ("fixed", "fixed_tuned", "fixed_default"):
        policy = FixedWeightPolicy(
            w_delay=policy_params.get("w_delay", 10.0),
            w_shift=policy_params.get("w_shift", 1.0),
            w_switch=policy_params.get("w_switch", 5.0),
            freeze_horizon=policy_params.get("freeze_horizon", 12),
            policy_name=policy_name
        )
    elif policy_name == "nofreeze":
        policy = NoFreezePolicy()
    elif policy_name == "greedy":
        policy = GreedyPolicy(sort_by="due", policy_name="greedy")
    elif policy_name == "mockllm":
        policy = MockLLMPolicy(policy_name="mockllm", enable_logging=False)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    # 运行仿真
    result = simulate_episode(policy, scenario, config, verbose=False)
    
    # 提取指标
    m = result.metrics
    return EpisodeMetricsRecord(
        seed=seed,
        disturbance_level=disturbance_level,
        policy_name=policy_name,
        dataset=dataset,
        completed=m.num_completed,
        total=m.num_total,
        on_time_rate=m.on_time_rate,
        avg_delay=m.avg_delay,
        max_delay=m.max_delay,
        episode_drift=m.episode_drift,
        total_shifts=m.total_shifts,
        total_switches=m.total_switches,
        num_replans=m.num_replans,
        num_forced_replans=m.num_forced_replans,
        avg_solve_time_ms=m.avg_solve_time_ms,
        total_runtime_s=result.total_runtime_s
    )


# ============================================================================
# 并行执行辅助函数
# ============================================================================

def run_single_episode_wrapper(args):
    """
    包装函数用于并行执行（ProcessPoolExecutor需要可pickle的参数）
    """
    return run_single_episode(*args)


# ============================================================================
# 调参
# ============================================================================

def grid_search_tuning(
    train_assignments: List[Tuple[int, str]],
    exp_config: ExperimentConfig,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[TuningResult]]:
    """
    在训练集上网格搜索最优参数（使用并行处理）
    
    参数空间：
    - freeze_horizon_hours
    - w_shift
    - w_switch
    - trigger_window_loss_pct (暂不使用，保留接口)
    
    Returns:
        (best_params, all_tuning_results)
    """
    if verbose:
        print("\n" + "="*70)
        print(" 开始网格搜索调参（并行处理）")
        print("="*70)
    
    # 构建参数网格
    # freeze_horizon: hours -> slots (1h = 6 slots)
    freeze_slots = [int(h * 6) for h in exp_config.freeze_horizon_hours]
    
    param_grid = list(itertools.product(
        freeze_slots,
        exp_config.w_delay_values,
        exp_config.w_shift_values,
        exp_config.w_switch_values
    ))
    
    if verbose:
        print(f"参数组合数: {len(param_grid)}")
        print(f"训练集 episodes: {len(train_assignments)}")
        print(f"并行进程数: {exp_config.max_workers}")
        print(f"总任务数: {len(param_grid) * len(train_assignments)}")
    
    tuning_results: List[TuningResult] = []
    best_score = float('inf')
    best_params = None
    
    total_combos = len(param_grid)
    
    for combo_idx, (freeze_h, w_delay, w_shift, w_switch) in enumerate(param_grid):
        if verbose:
            print(f"\n[{combo_idx+1}/{total_combos}] freeze={freeze_h}, "
                  f"w_delay={w_delay}, w_shift={w_shift}, w_switch={w_switch}")
        
        policy_params = {
            "w_delay": w_delay,
            "w_shift": w_shift,
            "w_switch": w_switch,
            "freeze_horizon": freeze_h
        }
        
        # 准备并行任务参数
        tasks = [
            (seed, level, "fixed", policy_params, "train", exp_config.solver_timeout_s)
            for seed, level in train_assignments
        ]
        
        # 并行执行所有训练 episodes
        delays = []
        drifts = []
        solve_times = []
        
        with ProcessPoolExecutor(max_workers=exp_config.max_workers) as executor:
            futures = {executor.submit(run_single_episode_wrapper, task): task for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                    delays.append(record.avg_delay)
                    drifts.append(record.episode_drift)
                    solve_times.append(record.avg_solve_time_ms)
                except Exception as e:
                    seed = task[0]
                    if verbose:
                        print(f"  Warning: seed {seed} failed: {e}")
                    delays.append(100.0)  # 惩罚值
                    drifts.append(1.0)
                    solve_times.append(0.0)
                
                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"  进度: {completed}/{len(tasks)}")
        
        # 计算平均值
        avg_delay = sum(delays) / len(delays) if delays else float('inf')
        avg_drift = sum(drifts) / len(drifts) if drifts else float('inf')
        avg_solve = sum(solve_times) / len(solve_times) if solve_times else 0.0
        
        # 综合得分
        combined = avg_delay + exp_config.tuning_lambda * avg_drift
        
        result = TuningResult(
            freeze_horizon_slots=freeze_h,
            w_delay=w_delay,
            w_shift=w_shift,
            w_switch=w_switch,
            avg_delay=avg_delay,
            avg_drift=avg_drift,
            combined_score=combined,
            num_episodes=len(train_assignments),
            avg_solve_time_ms=avg_solve
        )
        tuning_results.append(result)
        
        if verbose:
            print(f"  avg_delay={avg_delay:.2f}, avg_drift={avg_drift:.4f}, "
                  f"combined={combined:.2f}")
        
        # 更新最优
        if combined < best_score:
            best_score = combined
            best_params = {
                "w_delay": w_delay,
                "w_shift": w_shift,
                "w_switch": w_switch,
                "freeze_horizon": freeze_h
            }
    
    if verbose:
        print(f"\n最优参数: {best_params}")
        print(f"最优得分: {best_score:.2f}")
    
    return best_params, tuning_results


# ============================================================================
# 批量评估
# ============================================================================

def run_evaluation(
    assignments: List[Tuple[int, str]],
    policies: Dict[str, Dict[str, Any]],
    dataset: str,
    solver_timeout: float = 10.0,
    max_workers: int = 8,
    verbose: bool = True
) -> List[EpisodeMetricsRecord]:
    """
    在指定数据集上评估多个策略（使用并行处理）
    
    Args:
        assignments: [(seed, level), ...]
        policies: {policy_name: policy_params}
        dataset: "train" / "test"
        solver_timeout: 求解超时
        max_workers: 并行进程数
        verbose: 是否打印进度
    
    Returns:
        所有 episode 的指标记录
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f" 评估 {dataset} 集: {len(assignments)} episodes × {len(policies)} policies")
        print(f" 并行进程数: {max_workers}")
        print(f"{'='*70}")
    
    # 准备所有任务
    tasks = []
    for seed, level in assignments:
        for policy_name, policy_params in policies.items():
            tasks.append((
                seed, level, policy_name, policy_params, dataset, solver_timeout
            ))
    
    total = len(tasks)
    all_records: List[EpisodeMetricsRecord] = []
    
    # 并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_episode_wrapper, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            seed, level, policy_name = task[0], task[1], task[2]
            
            try:
                record = future.result()
                all_records.append(record)
            except Exception as e:
                print(f"  Error: seed={seed}, policy={policy_name}: {e}")
                # 记录失败
                all_records.append(EpisodeMetricsRecord(
                    seed=seed,
                    disturbance_level=level,
                    policy_name=policy_name,
                    dataset=task[4],
                    completed=0, total=0,
                    on_time_rate=0.0, avg_delay=float('inf'), max_delay=0,
                    episode_drift=0.0, total_shifts=0, total_switches=0,
                    num_replans=0, num_forced_replans=0,
                    avg_solve_time_ms=0.0, total_runtime_s=0.0
                ))
            
            completed += 1
            if verbose and completed % 50 == 0:
                print(f"  进度: {completed}/{total}")
    
    if verbose:
        print(f"  完成: {total}/{total}")
    
    return all_records


# ============================================================================
# 保存结果
# ============================================================================

def save_episode_results(records: List[EpisodeMetricsRecord], filepath: str):
    """保存每 episode 结果到 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    fieldnames = [
        "seed", "disturbance_level", "policy_name", "dataset",
        "completed", "total", "on_time_rate", "avg_delay", "max_delay",
        "episode_drift", "total_shifts", "total_switches",
        "num_replans", "num_forced_replans", "avg_solve_time_ms", "total_runtime_s"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())
    
    print(f"保存 episode 结果: {filepath}")


def save_tuning_results(results: List[TuningResult], filepath: str):
    """保存调参结果到 CSV"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    fieldnames = [
        "freeze_horizon_slots", "w_delay", "w_shift", "w_switch",
        "avg_delay", "avg_drift", "combined_score",
        "num_episodes", "avg_solve_time_ms"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    
    print(f"保存调参结果: {filepath}")


def save_summary(
    records: List[EpisodeMetricsRecord],
    filepath: str,
    tuning_lambda: float = 5.0
):
    """计算并保存汇总统计"""
    import math
    
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    # 按 (dataset, policy_name) 分组
    groups: Dict[Tuple[str, str], List[EpisodeMetricsRecord]] = {}
    for record in records:
        key = (record.dataset, record.policy_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    # 计算统计
    def mean(values):
        return sum(values) / len(values) if values else 0.0
    
    def std(values):
        if len(values) < 2:
            return 0.0
        m = mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    
    def ci95(values):
        """95% 置信区间半宽"""
        if len(values) < 2:
            return 0.0
        return 1.96 * std(values) / math.sqrt(len(values))
    
    rows = []
    for (dataset, policy_name), recs in sorted(groups.items()):
        delays = [r.avg_delay for r in recs]
        drifts = [r.episode_drift for r in recs]
        on_times = [r.on_time_rate for r in recs]
        shifts = [r.total_shifts for r in recs]
        switches = [r.total_switches for r in recs]
        solve_times = [r.avg_solve_time_ms for r in recs]
        
        combined = [d + tuning_lambda * dr for d, dr in zip(delays, drifts)]
        
        rows.append({
            "dataset": dataset,
            "policy_name": policy_name,
            "num_episodes": len(recs),
            "avg_delay_mean": round(mean(delays), 3),
            "avg_delay_ci95": round(ci95(delays), 3),
            "episode_drift_mean": round(mean(drifts), 5),
            "episode_drift_ci95": round(ci95(drifts), 5),
            "combined_score_mean": round(mean(combined), 3),
            "combined_score_ci95": round(ci95(combined), 3),
            "on_time_rate_mean": round(mean(on_times), 4),
            "total_shifts_mean": round(mean(shifts), 2),
            "total_switches_mean": round(mean(switches), 2),
            "avg_solve_time_ms_mean": round(mean(solve_times), 1)
        })
    
    fieldnames = list(rows[0].keys()) if rows else []
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    print(f"保存汇总统计: {filepath}")


def save_best_params(params: Dict[str, Any], filepath: str):
    """保存最优参数"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"保存最优参数: {filepath}")


# ============================================================================
# 主流程
# ============================================================================

def run_full_experiment(exp_config: ExperimentConfig, verbose: bool = True):
    """
    运行完整实验流程
    
    流程：
    1. 生成种子分配
    2. 在训练集上调参
    3. 在测试集上评估所有策略
    4. 保存结果
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" 火箭发射排程仿真 - 完整实验")
    print("="*70)
    print(f"训练集: {exp_config.num_train_seeds} episodes")
    print(f"测试集: {exp_config.num_test_seeds} episodes")
    print(f"扰动级别: {exp_config.disturbance_levels}")
    print(f"输出目录: {exp_config.output_dir}")
    
    # 1. 生成种子分配
    train_assignments, test_assignments = generate_seed_assignments(
        exp_config.num_train_seeds,
        exp_config.num_test_seeds,
        exp_config.disturbance_levels
    )
    
    print(f"\n训练集种子: {[a[0] for a in train_assignments[:5]]}... (共 {len(train_assignments)})")
    print(f"测试集种子: {[a[0] for a in test_assignments[:5]]}... (共 {len(test_assignments)})")
    
    # 2. 网格搜索调参（只用固定策略）
    best_params, tuning_results = grid_search_tuning(
        train_assignments, exp_config, verbose=verbose
    )
    
    # 保存调参结果
    tuning_path = os.path.join(exp_config.output_dir, "tuning_results.csv")
    save_tuning_results(tuning_results, tuning_path)
    
    best_params_path = os.path.join(exp_config.output_dir, "best_params.json")
    save_best_params(best_params, best_params_path)
    
    # 3. 定义所有策略
    policies = {
        "fixed_tuned": best_params,  # 调优后的固定策略
        "fixed_default": {
            "w_delay": 10.0, "w_shift": 1.0, "w_switch": 5.0, "freeze_horizon": 12
        },
        "nofreeze": {},
        "greedy": {},
        "mockllm": {}
    }
    
    # 4. 在测试集上评估
    test_records = run_evaluation(
        test_assignments, policies, "test",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose
    )
    
    # 5. 在训练集上也用最终策略跑一遍（用于完整记录）
    train_records = run_evaluation(
        train_assignments, policies, "train",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose
    )
    
    # 6. 合并并保存
    all_records = train_records + test_records
    
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    save_episode_results(all_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(all_records, summary_path, exp_config.tuning_lambda)
    
    # 完成
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" 实验完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {episode_path}")
    print(f"  - {summary_path}")
    print(f"  - {tuning_path}")
    print(f"  - {best_params_path}")
    
    return all_records


def run_train_only(exp_config: ExperimentConfig, verbose: bool = True):
    """
    仅运行训练阶段：调参 + 保存最优参数
    
    输出文件：
    - tuning_results.csv
    - best_params.json
    - results_per_episode.csv (仅训练集)
    - summary.csv (仅训练集)
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" 火箭发射排程仿真 - 训练阶段（调参）")
    print("="*70)
    print(f"训练集: {exp_config.num_train_seeds} episodes")
    print(f"输出目录: {exp_config.output_dir}")
    
    # 生成训练集种子
    train_assignments, _ = generate_seed_assignments(
        exp_config.num_train_seeds,
        0,  # 测试集为0
        exp_config.disturbance_levels
    )
    
    print(f"\n训练集种子: {[a[0] for a in train_assignments[:5]]}... (共 {len(train_assignments)})")
    
    # 网格搜索调参
    best_params, tuning_results = grid_search_tuning(
        train_assignments, exp_config, verbose=verbose
    )
    
    # 保存调参结果
    os.makedirs(exp_config.output_dir, exist_ok=True)
    
    tuning_path = os.path.join(exp_config.output_dir, "tuning_results.csv")
    save_tuning_results(tuning_results, tuning_path)
    
    best_params_path = os.path.join(exp_config.output_dir, "best_params.json")
    save_best_params(best_params, best_params_path)
    
    # 定义所有策略（使用最优参数）
    policies = {
        "fixed_tuned": best_params,
        "fixed_default": {
            "w_delay": 10.0, "w_shift": 1.0, "w_switch": 5.0, "freeze_horizon": 12
        },
        "nofreeze": {},
        "greedy": {},
        "mockllm": {}
    }
    
    # 在训练集上评估所有策略（完整记录）
    train_records = run_evaluation(
        train_assignments, policies, "train",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose
    )
    
    # 保存结果
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    save_episode_results(train_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(train_records, summary_path, exp_config.tuning_lambda)
    
    # 完成
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" 训练阶段完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {best_params_path}")
    print(f"  - {tuning_path}")
    print(f"  - {episode_path}")
    print(f"  - {summary_path}")
    print(f"\n提示: 运行测试阶段请使用:")
    print(f"  python run_experiments.py --mode test-only --test-seeds {exp_config.num_test_seeds} --output {exp_config.output_dir}")


def run_test_only(exp_config: ExperimentConfig, params_path: str, verbose: bool = True):
    """
    仅运行测试阶段：加载最优参数 + 测试集评估
    
    输入文件：
    - best_params.json (或通过 --load-params 指定)
    
    输出文件：
    - results_per_episode.csv (追加测试集数据)
    - summary.csv (包含训练集+测试集)
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" 火箭发射排程仿真 - 测试阶段（评估）")
    print("="*70)
    print(f"测试集: {exp_config.num_test_seeds} episodes")
    print(f"参数文件: {params_path}")
    print(f"输出目录: {exp_config.output_dir}")
    
    # 加载最优参数
    if not os.path.exists(params_path):
        print(f"错误: 参数文件不存在: {params_path}")
        return
    
    with open(params_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    
    print(f"\n加载的最优参数: {best_params}")
    
    # 生成测试集种子（从训练集之后开始）
    _, test_assignments = generate_seed_assignments(
        exp_config.num_train_seeds,  # 需要知道训练集大小以确定测试集起始种子
        exp_config.num_test_seeds,
        exp_config.disturbance_levels
    )
    
    print(f"测试集种子: {[a[0] for a in test_assignments[:5]]}... (共 {len(test_assignments)})")
    
    # 定义所有策略
    policies = {
        "fixed_tuned": best_params,
        "fixed_default": {
            "w_delay": 10.0, "w_shift": 1.0, "w_switch": 5.0, "freeze_horizon": 12
        },
        "nofreeze": {},
        "greedy": {},
        "mockllm": {}
    }
    
    # 在测试集上评估
    test_records = run_evaluation(
        test_assignments, policies, "test",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose
    )
    
    # 读取已有的训练集结果（如果存在）
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    all_records = test_records
    
    if os.path.exists(episode_path):
        print(f"\n检测到已有结果文件，合并训练集和测试集数据...")
        existing_records = []
        with open(episode_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_records.append(EpisodeMetricsRecord(
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
        
        # 合并：保留训练集数据，添加测试集数据
        train_records = [r for r in existing_records if r.dataset == "train"]
        all_records = train_records + test_records
        print(f"  训练集记录: {len(train_records)}")
        print(f"  测试集记录: {len(test_records)}")
    
    # 保存合并后的结果
    save_episode_results(all_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(all_records, summary_path, exp_config.tuning_lambda)
    
    # 完成
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f" 测试阶段完成！总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")
    print(f"输出文件:")
    print(f"  - {episode_path}")
    print(f"  - {summary_path}")
    print(f"\n提示: 生成图表请使用:")
    print(f"  python analyze.py --input {exp_config.output_dir} --output figures/")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="火箭发射排程仿真 - 批量实验脚本"
    )
    parser.add_argument(
        "--train-seeds", type=int, default=60,
        help="训练集 episode 数量 (default: 60)"
    )
    parser.add_argument(
        "--test-seeds", type=int, default=60,
        help="测试集 episode 数量 (default: 60)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results",
        help="输出目录 (default: results/)"
    )
    parser.add_argument(
        "--solver-timeout", type=float, default=10.0,
        help="每次 CP-SAT 求解超时秒数 (default: 10.0)"
    )
    parser.add_argument(
        "--lambda", dest="tuning_lambda", type=float, default=5.0,
        help="综合目标中 drift 的权重 (default: 5.0)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式 (train=9, test=9, 减少调参组合)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印详细信息"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1,
        help="并行进程数，1为串行 (default: 1)"
    )
    parser.add_argument(
        "--mode", type=str, default="full", 
        choices=["full", "train-only", "test-only"],
        help="运行模式: full=完整流程, train-only=仅训练调参, test-only=仅测试评估 (需提供 best_params.json)"
    )
    parser.add_argument(
        "--load-params", type=str, default=None,
        help="加载已有的最优参数文件路径 (用于 test-only 模式)"
    )
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        args.train_seeds = 9
        args.test_seeds = 9
        print("快速测试模式: train=9, test=9")
    
    # 创建配置
    exp_config = ExperimentConfig(
        num_train_seeds=args.train_seeds,
        num_test_seeds=args.test_seeds,
        output_dir=args.output,
        solver_timeout_s=args.solver_timeout,
        tuning_lambda=args.tuning_lambda,
        max_workers=args.workers
    )
    
    # 快速模式下减少调参组合
    if args.quick:
        exp_config.freeze_horizon_hours = [0, 6]  # 2 instead of 4
        exp_config.w_delay_values = [5.0, 20.0]   # 2 instead of 4
        exp_config.w_shift_values = [0.0, 1.0]    # 2 instead of 4
        exp_config.w_switch_values = [0, 180]     # 2 instead of 4
        print("  调参网格缩减为 2×2×2×2=16 组合")
    
    # 根据模式运行
    if args.mode == "full":
        run_full_experiment(exp_config, verbose=args.verbose or args.quick)
    
    elif args.mode == "train-only":
        run_train_only(exp_config, verbose=args.verbose or args.quick)
    
    elif args.mode == "test-only":
        if not args.load_params:
            # 尝试从输出目录加载
            default_params_path = os.path.join(args.output, "best_params.json")
            if os.path.exists(default_params_path):
                args.load_params = default_params_path
                print(f"自动加载参数文件: {args.load_params}")
            else:
                print("错误: test-only 模式需要提供 --load-params 或在输出目录中存在 best_params.json")
                return
        
        run_test_only(exp_config, args.load_params, verbose=args.verbose or args.quick)


if __name__ == "__main__":
    main()
