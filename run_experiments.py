#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量实验脚本 - 训练集调参 + 测试集评估 + LLM 策略支持

功能：
1. 生成指定数量的 episodes（轻/中/重扰动各占 1/3）
2. baseline 调参：在训练集上网格搜索最优参数
3. 测试集评估：配对比较所有策略
4. 支持 Real LLM 策略（Qwen3-32B）
5. 输出 CSV 结果文件和 JSONL 日志

用法：
    # Baseline 策略（可并行）
    python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/ --workers 8
    
    # LLM 策略（强制单线程）
    python run_experiments.py --mode test-only --policy llm_real --test-seeds 60 \\
        --llm-base-url https://api-inference.modelscope.cn/v1 \\
        --llm-model Qwen/Qwen3-32B --llm-key-env DASHSCOPE_API_KEY \\
        --output results_llm/
"""

import argparse
import csv
import json
import os
import time
import itertools
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from config import Config, DEFAULT_CONFIG
from scenario import generate_scenario, Scenario
from simulator import simulate_episode, EpisodeResult, save_episode_logs
from policies import (
    FixedWeightPolicy, NoFreezePolicy, GreedyPolicy, MockLLMPolicy,
    create_policy, MetaParams
)

# 可选导入 RealLLMPolicy（需要 llm_client）
try:
    from policies import RealLLMPolicy, create_real_llm_policy
    from llm_client import LLMConfig
    HAS_REAL_LLM = True
except ImportError:
    HAS_REAL_LLM = False
    RealLLMPolicy = None
    create_real_llm_policy = None
    LLMConfig = None


# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    
    # 调参网格（扩展上界，保持4×4×4×4=256组合）
    freeze_horizon_hours: List[float] = field(default_factory=lambda: [0, 4, 8, 16])  # 增加上界到16小时
    w_delay_values: List[float] = field(default_factory=lambda: [10.0, 15.0, 20.0, 30.0])  # 增加上界到30
    w_shift_values: List[float] = field(default_factory=lambda: [0.0, 1.0, 3.0, 8.0])  # 增加上界到2.0
    w_switch_values: List[float] = field(default_factory=lambda: [60, 120, 180, 240])  # 增加上界到100
    trigger_window_loss_pct: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.4])
    
    # 综合目标权重（软约束）
    tuning_lambda: float = 5.0  # legacy: delay + lambda * drift
    tuning_delay_tolerance: float = 0.10  # 放宽到10%容忍度
    tuning_weight_delay: float = 0.50  # 准时性（最重要）
    tuning_weight_drift: float = 0.20  # 稳定性
    tuning_weight_time_dev: float = 0.10  # 时间偏差
    tuning_weight_forced_replan: float = 0.10  # 强制重排
    tuning_weight_feasible: float = 0.10  # 可行性惩罚
    
    # 求解器
    solver_timeout_s: float = 20.0  # 每次求解限时
    
    # 并行处理
    max_workers: int = 1  # 并行进程数（1为串行，设置>1启用并行）

    # ========== ε-constraint（稳定性约束） ==========
    epsilon_metric: str = "avg_delay"  # 目前仅支持 avg_delay
    epsilon_relative: str = "baseline"  # "baseline" | "absolute"
    epsilon_value: float = 0.10  # baseline: 相对阈值(10%); absolute: 绝对阈值
    
    # 输出目录
    output_dir: str = "results"
    
    # ========== LLM 配置 ==========
    llm_base_url: str = "https://api-inference.modelscope.cn/v1"
    llm_model: str = "Qwen/Qwen3-32B"
    llm_key_env: str = "DASHSCOPE_API_KEY"
    llm_cache_dir: Optional[str] = None  # None = output_dir/llm_cache
    llm_log_path: Optional[str] = None   # None = output_dir/llm_logs
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 5
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0


BASELINE_FIXED_DEFAULT = {
    "w_delay": 10.0,
    "w_shift": 1.0,
    "w_switch": 5.0,
    "freeze_horizon": 12
}

# 扰动强度对应的 Config 参数
DISTURBANCE_CONFIGS = {
    "light": {
        "p_weather": 0.05,
        "p_pad_outage": 0.02,
        "sigma_duration": 0.12,
        "sigma_release": 2.0
    },
    "medium": {
        "p_weather": 0.07,
        "p_pad_outage": 0.03,
        "sigma_duration": 0.2,
        "sigma_release": 3.0
    },
    "heavy": {
        "p_weather": 0.10,
        "p_pad_outage": 0.05,
        "sigma_duration": 0.3,
        "sigma_release": 4.0
    }
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class EpisodeMetricsRecord:
    """单 episode 指标记录（扩展版，包含 LLM 相关字段）"""
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
    weighted_tardiness: float
    resource_utilization: float
    episode_drift: float
    
    # 稳定性
    total_shifts: int
    total_switches: int
    num_replans: int
    num_forced_replans: int
    
    # 性能（时间单位统一为 ms/s）
    avg_solve_time_ms: float
    total_runtime_s: float

    # ========== 补充指标 ==========
    avg_time_deviation_min: float
    total_resource_switches: int
    makespan_cmax: int
    feasible_rate: float
    forced_replan_rate: float
    avg_frozen: float
    avg_num_tasks_scheduled: float
    util_r_pad: float
    
    # ========== LLM 相关指标（新增） ==========
    llm_calls: int = 0                    # LLM 调用次数
    llm_time_total_ms: int = 0            # LLM 总耗时（含重试）
    llm_latency_total_ms: int = 0         # LLM 网络延迟总计
    llm_prompt_tokens: int = 0            # Prompt token 数
    llm_completion_tokens: int = 0        # Completion token 数
    llm_total_tokens: int = 0             # 总 token 数
    llm_cache_hit_rate: float = 0.0       # 缓存命中率
    llm_fallback_count: int = 0           # Fallback 次数
    solver_time_total_ms: int = 0         # 求解器总耗时
    wall_time_total_ms: int = 0           # 实际墙上时间总计
    
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
    avg_time_deviation_min: float
    avg_forced_replan_rate: float
    avg_feasible_rate: float
    delay_ratio: float
    drift_ratio: float
    time_dev_ratio: float
    forced_replan_ratio: float
    delay_ok: bool
    feasible_ok: bool
    combined_score: float  # normalized weighted score

    # ε-constraint 结果
    epsilon_metric: str
    epsilon_threshold: float
    epsilon_value: float
    epsilon_ok: bool
    epsilon_violation: float
    
    # 统计
    num_episodes: int
    avg_solve_time_ms: float


@dataclass
class RollingLogEntry:
    """单次 rolling 的详细日志"""
    episode_id: str
    t_now: int
    trigger_reason: str            # "scheduled" | "forced_replan" | "initial"
    freeze_horizon: int
    weights: Dict[str, float]      # {"w_delay": ..., "w_shift": ..., "w_switch": ...}
    plan_diff: Dict[str, Any]      # {"num_shifts": ..., "num_switches": ..., "drift": ...}
    drift_components: Dict[str, float]  # {"time_drift": ..., "pad_drift": ...}
    solve_status: str
    solve_time_ms: int
    
    # LLM 相关（仅 llm_real 策略有值）
    llm_cache_hit: Optional[bool] = None
    llm_latency_ms: Optional[int] = None
    llm_tokens: Optional[int] = None
    llm_fallback: Optional[bool] = None
    llm_raw_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 场景生成
# ============================================================================

def create_config_for_disturbance(level: str, solver_timeout: float = 20.0) -> Config:
    """根据扰动强度创建配置"""
    params = DISTURBANCE_CONFIGS.get(level, DISTURBANCE_CONFIGS["medium"])
    
    return Config(
        schema_version=DEFAULT_CONFIG.schema_version,
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
    solver_timeout: float = 20.0,
    exp_config: Optional[ExperimentConfig] = None,
    output_dir: Optional[str] = None
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
        exp_config: 实验配置（用于 LLM 参数）
        output_dir: 日志输出目录
    
    Returns:
        EpisodeMetricsRecord
    """
    wall_start_time = time.time()
    
    # 创建配置
    config = create_config_for_disturbance(disturbance_level, solver_timeout)
    
    # 生成场景
    scenario = generate_scenario(seed=seed, config=config)
    
    # Episode ID
    episode_id = f"episode_{seed}_{policy_name}"
    
    # 创建策略
    policy = None
    llm_stats = None  # 用于记录 LLM 统计
    
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
        log_dir = None
        if output_dir:
            log_dir = os.path.join(output_dir, "logs", episode_id)
            os.makedirs(log_dir, exist_ok=True)
        policy = MockLLMPolicy(
            policy_name="mockllm",
            enable_logging=True,
            log_dir=log_dir,
            episode_id=episode_id
        )
    elif policy_name == "llm_real":
        if not HAS_REAL_LLM:
            raise ImportError(
                "llm_real 策略需要 llm_client 模块，请确保 llm_client.py 存在且已安装 openai 库"
            )
        
        # 准备日志和缓存目录
        log_dir = None
        cache_dir = None
        
        if output_dir:
            log_dir = os.path.join(output_dir, "logs", episode_id)
            os.makedirs(log_dir, exist_ok=True)
        
        if exp_config:
            cache_dir = exp_config.llm_cache_dir
            if cache_dir is None:
                cache_dir = os.path.join(output_dir or ".", "llm_cache")
        
        # 创建 LLM 配置
        llm_config = LLMConfig(
            api_key_env=exp_config.llm_key_env if exp_config else "DASHSCOPE_API_KEY",
            base_url=exp_config.llm_base_url if exp_config else "https://api-inference.modelscope.cn/v1",
            model=exp_config.llm_model if exp_config else "Qwen/Qwen3-32B",
            temperature=exp_config.llm_temperature if exp_config else 0.0,
            top_p=exp_config.llm_top_p if exp_config else 1.0,
            timeout_s=exp_config.llm_timeout_s if exp_config else 30.0,
            max_retries=exp_config.llm_max_retries if exp_config else 5,
            cache_dir=cache_dir,
            log_file=os.path.join(log_dir, "llm_raw_calls.jsonl") if log_dir else None,
            enable_thinking=False
        )
        
        fallback_params = None
        if output_dir:
            best_params_path = os.path.join(output_dir, "best_params.json")
            if os.path.exists(best_params_path):
                try:
                    with open(best_params_path, "r", encoding="utf-8") as f:
                        best = json.load(f)
                    fallback_params = MetaParams(
                        w_delay=best.get("w_delay", 10.0),
                        w_shift=best.get("w_shift", 1.0),
                        w_switch=best.get("w_switch", 5.0),
                        freeze_horizon=best.get("freeze_horizon", 12)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load best_params.json for fallback: {e}")

        policy = RealLLMPolicy(
            llm_config=llm_config,
            policy_name="llm_real",
            log_dir=log_dir,
            enable_logging=True,
            episode_id=episode_id,
            fallback_params=fallback_params
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    # 运行仿真
    result = simulate_episode(policy, scenario, config, verbose=False)

    # save rolling log/final schedule for batch analysis
    if output_dir:
        log_dir = os.path.join(output_dir, "logs", episode_id)
        save_episode_logs(result, log_dir, scenario)

    
    # 收集 LLM 统计（如果是 LLM 策略）
    llm_calls = 0
    llm_time_total_ms = 0
    llm_latency_total_ms = 0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_total_tokens = 0
    llm_cache_hit_rate = 0.0
    llm_fallback_count = 0
    
    if policy_name in ("mockllm", "llm_real"):
        if hasattr(policy, 'get_llm_stats'):
            llm_stats = policy.get_llm_stats()
            llm_calls = llm_stats.get("call_count", 0)
            llm_time_total_ms = llm_stats.get("total_latency_ms", 0)
            llm_latency_total_ms = llm_stats.get("total_latency_ms", 0)
            llm_total_tokens = llm_stats.get("total_tokens", 0)
            llm_cache_hit_rate = llm_stats.get("cache_hit_rate", 0.0)
            llm_fallback_count = llm_stats.get("fallback_count", 0)
            
            # 对于 LLM client，尝试获取更详细的 token 统计
            if "llm_client" in llm_stats:
                client_stats = llm_stats["llm_client"]
                llm_prompt_tokens = client_stats.get("total_prompt_tokens", 0)
                llm_completion_tokens = client_stats.get("total_completion_tokens", 0)
    
    # 计算求解器总耗时
    solver_time_total_ms = sum(
        snap.solve_time_ms for snap in result.snapshots
    )
    
    # 墙上时间
    wall_time_total_ms = int((time.time() - wall_start_time) * 1000)
    
    # 写入 episode 日志（JSONL 格式）
    if output_dir and policy_name in ("mockllm", "llm_real"):
        log_dir = os.path.join(output_dir, "logs", episode_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # 写入 episode_log.jsonl
        episode_log_path = os.path.join(log_dir, "episode_log.jsonl")
        _write_episode_rolling_log(
            result, scenario, policy, episode_id, episode_log_path
        )
    
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
        weighted_tardiness=m.weighted_tardiness,
        resource_utilization=m.resource_utilization,
        episode_drift=m.episode_drift,
        total_shifts=m.total_shifts,
        total_switches=m.total_switches,
        num_replans=m.num_replans,
        num_forced_replans=m.num_forced_replans,
        avg_solve_time_ms=m.avg_solve_time_ms,
        total_runtime_s=result.total_runtime_s,
        avg_time_deviation_min=m.avg_time_deviation_min,
        total_resource_switches=m.total_resource_switches,
        makespan_cmax=m.makespan_cmax,
        feasible_rate=m.feasible_rate,
        forced_replan_rate=m.forced_replan_rate,
        avg_frozen=m.avg_frozen,
        avg_num_tasks_scheduled=m.avg_num_tasks_scheduled,
        util_r_pad=m.util_r_pad,
        # LLM ??
        llm_calls=llm_calls,
        llm_time_total_ms=llm_time_total_ms,
        llm_latency_total_ms=llm_latency_total_ms,
        llm_prompt_tokens=llm_prompt_tokens,
        llm_completion_tokens=llm_completion_tokens,
        llm_total_tokens=llm_total_tokens,
        llm_cache_hit_rate=llm_cache_hit_rate,
        llm_fallback_count=llm_fallback_count,
        solver_time_total_ms=solver_time_total_ms,
        wall_time_total_ms=wall_time_total_ms
    )



def _write_episode_rolling_log(
    result: EpisodeResult,
    scenario: Scenario,
    policy: Any,
    episode_id: str,
    filepath: str
) -> None:
    """
    写入每次 rolling 的详细日志
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, snapshot in enumerate(result.snapshots):
            # 确定触发原因
            trigger_reason = "scheduled"
            if i == 0:
                trigger_reason = "initial"
            elif snapshot.infeasible_reasons:
                trigger_reason = "forced_replan"
            
            # 获取权重
            weights = {
                "w_delay": 10.0,
                "w_shift": 1.0,
                "w_switch": 5.0
            }
            freeze_horizon = 12
            
            if snapshot.meta_params:
                weights = {
                    "w_delay": snapshot.meta_params.w_delay,
                    "w_shift": snapshot.meta_params.w_shift,
                    "w_switch": snapshot.meta_params.w_switch
                }
                freeze_horizon = snapshot.meta_params.freeze_horizon
            
            # Plan diff
            plan_diff = {
                "num_shifts": snapshot.metrics.num_shifts,
                "num_switches": snapshot.metrics.num_switches,
                "drift": snapshot.metrics.plan_drift
            }
            
            # Drift components
            drift_components = {
                "time_drift": snapshot.metrics.time_drift if hasattr(snapshot.metrics, 'time_drift') else 0.0,
                "pad_drift": snapshot.metrics.pad_drift if hasattr(snapshot.metrics, 'pad_drift') else 0.0
            }
            
            # 构建日志条目
            entry = RollingLogEntry(
                episode_id=episode_id,
                t_now=snapshot.t,
                trigger_reason=trigger_reason,
                freeze_horizon=freeze_horizon,
                weights=weights,
                plan_diff=plan_diff,
                drift_components=drift_components,
                solve_status=snapshot.solve_status.value,
                solve_time_ms=snapshot.solve_time_ms
            )
            
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + '\n')


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
    Grid search on train set with epsilon-constraint and min drift.
    """
    if verbose:
        print("\n" + "="*70)
        print(" Grid search tuning (train set)")
        print("="*70)

    slot_minutes = DEFAULT_CONFIG.slot_minutes
    freeze_slots = [
        int(round(h * 60 / slot_minutes))
        for h in exp_config.freeze_horizon_hours
    ]

    param_grid = list(itertools.product(
        freeze_slots,
        exp_config.w_delay_values,
        exp_config.w_shift_values,
        exp_config.w_switch_values
    ))

    def _evaluate_policy(policy_name: str, policy_params: Dict[str, Any]) -> Dict[str, float]:
        tasks = [
            (seed, level, policy_name, policy_params, "train", exp_config.solver_timeout_s, None, None)
            for seed, level in train_assignments
        ]
        delays = []
        drifts = []
        time_devs = []
        forced_replans = []
        feasible_rates = []
        solve_times = []

        if exp_config.max_workers <= 1:
            completed = 0
            for task in tasks:
                try:
                    record = run_single_episode_wrapper(task)
                    delays.append(record.avg_delay)
                    drifts.append(record.episode_drift)
                    time_devs.append(record.avg_time_deviation_min)
                    forced_replans.append(record.forced_replan_rate)
                    feasible_rates.append(record.feasible_rate)
                    solve_times.append(record.avg_solve_time_ms)
                except Exception as e:
                    seed = task[0]
                    if verbose:
                        print(f"  Warning: seed {seed} failed: {e}")
                    delays.append(100.0)
                    drifts.append(1.0)
                    time_devs.append(999.0)
                    forced_replans.append(1.0)
                    feasible_rates.append(0.0)
                    solve_times.append(0.0)

                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"  progress: {completed}/{len(tasks)}")
        else:
            with ThreadPoolExecutor(max_workers=exp_config.max_workers) as executor:
                futures = {executor.submit(run_single_episode_wrapper, task): task for task in tasks}

                completed = 0
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                        delays.append(record.avg_delay)
                        drifts.append(record.episode_drift)
                        time_devs.append(record.avg_time_deviation_min)
                        forced_replans.append(record.forced_replan_rate)
                        feasible_rates.append(record.feasible_rate)
                        solve_times.append(record.avg_solve_time_ms)
                    except Exception as e:
                        seed = task[0]
                        if verbose:
                            print(f"  Warning: seed {seed} failed: {e}")
                        delays.append(100.0)
                        drifts.append(1.0)
                        time_devs.append(999.0)
                        forced_replans.append(1.0)
                        feasible_rates.append(0.0)
                        solve_times.append(0.0)

                    completed += 1
                    if verbose and completed % 10 == 0:
                        print(f"  progress: {completed}/{len(tasks)}")

        def _mean(values, default=0.0):
            return sum(values) / len(values) if values else default

        return {
            "avg_delay": _mean(delays, float('inf')),
            "avg_drift": _mean(drifts, float('inf')),
            "avg_time_deviation_min": _mean(time_devs, float('inf')),
            "avg_forced_replan_rate": _mean(forced_replans, 1.0),
            "avg_feasible_rate": _mean(feasible_rates, 0.0),
            "avg_solve_time_ms": _mean(solve_times, 0.0)
        }

    if verbose:
        print(f"Param combos: {len(param_grid)}")
        print(f"Train episodes: {len(train_assignments)}")
        print(f"Max workers: {exp_config.max_workers}")
        print(f"Total tasks: {len(param_grid) * len(train_assignments)}")

    baseline_fixed = _evaluate_policy("fixed", BASELINE_FIXED_DEFAULT)

    eps = 1e-9
    base_delay = max(eps, baseline_fixed["avg_delay"])
    base_drift = max(eps, baseline_fixed["avg_drift"])
    base_time_dev = max(eps, baseline_fixed["avg_time_deviation_min"])
    base_forced = max(eps, baseline_fixed["avg_forced_replan_rate"])
    base_feasible = baseline_fixed["avg_feasible_rate"]

    if exp_config.epsilon_metric != "avg_delay":
        raise ValueError(f"Unsupported epsilon_metric: {exp_config.epsilon_metric}")

    if exp_config.epsilon_relative == "baseline":
        epsilon_threshold = base_delay * (1 + exp_config.epsilon_value)
    elif exp_config.epsilon_relative == "absolute":
        epsilon_threshold = exp_config.epsilon_value
    else:
        raise ValueError(f"Unsupported epsilon_relative: {exp_config.epsilon_relative}")

    w_delay = exp_config.tuning_weight_delay
    w_drift = exp_config.tuning_weight_drift
    w_time = exp_config.tuning_weight_time_dev
    w_forced = exp_config.tuning_weight_forced_replan
    w_sum = w_delay + w_drift + w_time + w_forced
    if w_sum > 0:
        w_delay, w_drift, w_time, w_forced = [w / w_sum for w in (w_delay, w_drift, w_time, w_forced)]

    if verbose:
        print("\nBaseline (fixed_default) train stats:")
        print(f"  delay={baseline_fixed['avg_delay']:.3f}, drift={baseline_fixed['avg_drift']:.5f}, time_dev={baseline_fixed['avg_time_deviation_min']:.3f}, forced={baseline_fixed['avg_forced_replan_rate']:.3f}, feasible={baseline_fixed['avg_feasible_rate']:.3f}")
        print("Baseline thresholds:")
        print(f"  delay<= {base_delay * (1 + exp_config.tuning_delay_tolerance):.3f} (tol={exp_config.tuning_delay_tolerance*100:.1f}%)")
        print(f"  feasible>= {base_feasible:.3f}")
        print("Epsilon-constraint:")
        print(f"  metric={exp_config.epsilon_metric}, threshold={epsilon_threshold:.3f} ({exp_config.epsilon_relative}, value={exp_config.epsilon_value})")

    tuning_results: List[TuningResult] = []
    best_params = None
    best_drift = float('inf')
    best_violation = float('inf')
    best_delay = float('inf')
    feasible_found = False

    total_combos = len(param_grid)

    for combo_idx, (freeze_h, w_d, w_s, w_sw) in enumerate(param_grid):
        if verbose:
            print(f"\n[{combo_idx+1}/{total_combos}] freeze={freeze_h}, w_delay={w_d}, w_shift={w_s}, w_switch={w_sw}")

        policy_params = {
            "w_delay": w_d,
            "w_shift": w_s,
            "w_switch": w_sw,
            "freeze_horizon": freeze_h
        }

        stats = _evaluate_policy("fixed", policy_params)

        # 归一化指标（相对baseline）
        delay_ratio = stats["avg_delay"] / base_delay
        drift_ratio = stats["avg_drift"] / base_drift
        time_dev_ratio = stats["avg_time_deviation_min"] / base_time_dev
        forced_ratio = stats["avg_forced_replan_rate"] / base_forced

        # 可行性惩罚（如果低于baseline）
        feasible_penalty = max(0, (base_feasible - stats["avg_feasible_rate"]) / max(base_feasible, 1e-9))

        # 软约束：全部用加权求和（移除硬约束）
        combined = (
            w_delay * delay_ratio +
            w_drift * drift_ratio +
            w_time * time_dev_ratio +
            w_forced * forced_ratio +
            exp_config.tuning_weight_feasible * feasible_penalty  # 新增可行性惩罚
        )
        
        # 记录是否满足基本要求（用于分析，不影响选择）
        delay_ok = stats["avg_delay"] <= base_delay * (1 + exp_config.tuning_delay_tolerance)
        feasible_ok = stats["avg_feasible_rate"] >= base_feasible - 1e-9

        epsilon_value = stats["avg_delay"]
        epsilon_ok = epsilon_value <= epsilon_threshold
        epsilon_violation = max(0.0, epsilon_value - epsilon_threshold)

        result = TuningResult(
            freeze_horizon_slots=freeze_h,
            w_delay=w_d,
            w_shift=w_s,
            w_switch=w_sw,
            avg_delay=stats["avg_delay"],
            avg_drift=stats["avg_drift"],
            avg_time_deviation_min=stats["avg_time_deviation_min"],
            avg_forced_replan_rate=stats["avg_forced_replan_rate"],
            avg_feasible_rate=stats["avg_feasible_rate"],
            delay_ratio=delay_ratio,
            drift_ratio=drift_ratio,
            time_dev_ratio=time_dev_ratio,
            forced_replan_ratio=forced_ratio,
            delay_ok=delay_ok,
            feasible_ok=feasible_ok,
            combined_score=combined,
            epsilon_metric=exp_config.epsilon_metric,
            epsilon_threshold=epsilon_threshold,
            epsilon_value=epsilon_value,
            epsilon_ok=epsilon_ok,
            epsilon_violation=epsilon_violation,
            num_episodes=len(train_assignments),
            avg_solve_time_ms=stats["avg_solve_time_ms"]
        )
        tuning_results.append(result)

        if verbose:
            print(
                f"  delay={stats['avg_delay']:.3f}, drift={stats['avg_drift']:.5f}, "
                f"time_dev={stats['avg_time_deviation_min']:.3f}, forced={stats['avg_forced_replan_rate']:.3f}, "
                f"feasible={stats['avg_feasible_rate']:.3f}, combined={combined:.4f}, eps_ok={epsilon_ok}"
            )

        if epsilon_ok:
            feasible_found = True
            if (stats["avg_drift"] < best_drift) or (
                stats["avg_drift"] == best_drift and stats["avg_delay"] < best_delay
            ):
                best_drift = stats["avg_drift"]
                best_delay = stats["avg_delay"]
                best_params = {
                    "w_delay": w_d,
                    "w_shift": w_s,
                    "w_switch": w_sw,
                    "freeze_horizon": freeze_h,
                    "epsilon_metric": exp_config.epsilon_metric,
                    "epsilon_relative": exp_config.epsilon_relative,
                    "epsilon_value": exp_config.epsilon_value,
                    "epsilon_threshold": epsilon_threshold,
                    "baseline_avg_delay": base_delay
                }
        elif not feasible_found:
            if (epsilon_violation < best_violation) or (
                epsilon_violation == best_violation and stats["avg_drift"] < best_drift
            ):
                best_violation = epsilon_violation
                best_drift = stats["avg_drift"]
                best_delay = stats["avg_delay"]
                best_params = {
                    "w_delay": w_d,
                    "w_shift": w_s,
                    "w_switch": w_sw,
                    "freeze_horizon": freeze_h,
                    "epsilon_metric": exp_config.epsilon_metric,
                    "epsilon_relative": exp_config.epsilon_relative,
                    "epsilon_value": exp_config.epsilon_value,
                    "epsilon_threshold": epsilon_threshold,
                    "baseline_avg_delay": base_delay
                }

    if verbose:
        print(f"\nBest params: {best_params}")
        if feasible_found:
            print(f"Best drift (feasible): {best_drift:.5f}")
        else:
            print(f"WARNING: no feasible params, best violation={best_violation:.3f}, drift={best_drift:.5f}")
        
        # 边界检查警告
        slot_minutes = DEFAULT_CONFIG.slot_minutes
        freeze_slots = [int(round(h * 60 / slot_minutes)) for h in exp_config.freeze_horizon_hours]
        at_boundary = []
        if best_params.get("freeze_horizon") == max(freeze_slots):
            at_boundary.append("freeze_horizon (max)")
        if best_params.get("w_delay") == max(exp_config.w_delay_values):
            at_boundary.append("w_delay (max)")
        if best_params.get("w_shift") == max(exp_config.w_shift_values):
            at_boundary.append("w_shift (max)")
        if best_params.get("w_switch") == max(exp_config.w_switch_values):
            at_boundary.append("w_switch (max)")
        
        if at_boundary:
            print(f"\n⚠️  WARNING: 最优参数触及网格边界: {', '.join(at_boundary)}")
            print(f"   建议: 扩大搜索空间或检查参数设置")

    return best_params, tuning_results


def run_evaluation(
    assignments: List[Tuple[int, str]],
    policies: Dict[str, Dict[str, Any]],
    dataset: str,
    solver_timeout: float = 20.0,
    max_workers: int = 8,
    verbose: bool = True,
    exp_config: Optional[ExperimentConfig] = None,
    output_dir: Optional[str] = None
) -> List[EpisodeMetricsRecord]:
    """
    在指定数据集上评估多个策略（使用并行处理）
    
    注意：llm_real 策略会自动降级为单线程执行
    
    Args:
        assignments: [(seed, level), ...]
        policies: {policy_name: policy_params}
        dataset: "train" / "test"
        solver_timeout: 求解超时
        max_workers: 并行进程数
        verbose: 是否打印进度
        exp_config: 实验配置（用于 LLM 参数）
        output_dir: 输出目录
    
    Returns:
        所有 episode 的指标记录
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f" 评估 {dataset} 集: {len(assignments)} episodes × {len(policies)} policies")
        print(f" 并行进程数: {max_workers}")
        print(f"{'='*70}")
    
    all_records: List[EpisodeMetricsRecord] = []
    
    # 分离 LLM 策略和非 LLM 策略
    llm_policies = {}
    non_llm_policies = {}
    
    for policy_name, policy_params in policies.items():
        if policy_name == "llm_real":
            llm_policies[policy_name] = policy_params
        else:
            non_llm_policies[policy_name] = policy_params
    
    # 1. 并行执行非 LLM 策略
    if non_llm_policies:
        if verbose:
            print(f"\n[1/2] 评估非 LLM 策略: {list(non_llm_policies.keys())}")
        
        tasks = []
        for seed, level in assignments:
            for policy_name, policy_params in non_llm_policies.items():
                tasks.append((
                    seed, level, policy_name, policy_params, dataset, 
                    solver_timeout, exp_config, output_dir
                ))
        
        total = len(tasks)
        
        if max_workers <= 1:
            completed = 0
            for task in tasks:
                seed, level, policy_name = task[0], task[1], task[2]
                try:
                    record = run_single_episode_wrapper(task)
                    all_records.append(record)
                except Exception as e:
                    logger.error(f"Error: seed={seed}, policy={policy_name}: {e}")
                    all_records.append(_create_failed_record(
                        seed, level, policy_name, dataset
                    ))

                completed += 1
                if verbose and completed % 50 == 0:
                    print(f"  进度: {completed}/{total}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_single_episode_wrapper, task): task for task in tasks}

                completed = 0
                for future in as_completed(futures):
                    task = futures[future]
                    seed, level, policy_name = task[0], task[1], task[2]

                    try:
                        record = future.result()
                        all_records.append(record)
                    except Exception as e:
                        logger.error(f"Error: seed={seed}, policy={policy_name}: {e}")
        # ???????
                        all_records.append(_create_failed_record(
                            seed, level, policy_name, dataset
                        ))

                    completed += 1
                    if verbose and completed % 50 == 0:
                        print(f"  进度: {completed}/{total}")
        if verbose:
            print(f"  完成: {total}/{total}")
    
    # 2. 串行执行 LLM 策略
    if llm_policies:
        if verbose:
            print(f"\n[2/2] 评估 LLM 策略（强制串行）: {list(llm_policies.keys())}")
            print(f"  原因: LLM API 调用需要顺序执行以保证缓存一致性和避免速率限制")
        
        for policy_name, policy_params in llm_policies.items():
            completed = 0
            total = len(assignments)
            
            for seed, level in assignments:
                try:
                    record = run_single_episode(
                        seed=seed,
                        disturbance_level=level,
                        policy_name=policy_name,
                        policy_params=policy_params,
                        dataset=dataset,
                        solver_timeout=solver_timeout,
                        exp_config=exp_config,
                        output_dir=output_dir
                    )
                    all_records.append(record)
                except Exception as e:
                    logger.error(f"LLM Error: seed={seed}, policy={policy_name}: {e}")
                    # LLM 策略失败不中断实验，记录失败
                    all_records.append(_create_failed_record(
                        seed, level, policy_name, dataset
                    ))
                
                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"  进度: {completed}/{total}")
            
            if verbose:
                print(f"  完成: {total}/{total}")
    
    return all_records


def _create_failed_record(
    seed: int,
    level: str,
    policy_name: str,
    dataset: str
) -> EpisodeMetricsRecord:
    """创建失败记录"""
    return EpisodeMetricsRecord(
        seed=seed,
        disturbance_level=level,
        policy_name=policy_name,
        dataset=dataset,
        completed=0, total=0,
        on_time_rate=0.0, avg_delay=float('inf'), max_delay=0,
        weighted_tardiness=0.0, resource_utilization=0.0,
        episode_drift=0.0, total_shifts=0, total_switches=0,
        num_replans=0, num_forced_replans=0,
        avg_solve_time_ms=0.0, total_runtime_s=0.0,
        avg_time_deviation_min=0.0,
        total_resource_switches=0,
        makespan_cmax=0,
        feasible_rate=0.0,
        forced_replan_rate=0.0,
        avg_frozen=0.0,
        avg_num_tasks_scheduled=0.0,
        util_r_pad=0.0,
        llm_calls=0, llm_time_total_ms=0, llm_latency_total_ms=0,
        llm_prompt_tokens=0, llm_completion_tokens=0, llm_total_tokens=0,
        llm_cache_hit_rate=0.0, llm_fallback_count=0,
        solver_time_total_ms=0, wall_time_total_ms=0
    )


# ============================================================================
# 保存结果
# ============================================================================

def save_episode_results(records: List[EpisodeMetricsRecord], filepath: str):
    """保存每 episode 结果到 CSV（包含 LLM 相关字段）"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    fieldnames = [
        "seed", "disturbance_level", "policy_name", "dataset",
        "completed", "total", "on_time_rate", "avg_delay", "max_delay", "weighted_tardiness", "resource_utilization",
        "util_r_pad",
        "episode_drift", "total_shifts", "total_switches",
        "avg_time_deviation_min", "total_resource_switches", "makespan_cmax",
        "feasible_rate", "forced_replan_rate", "avg_frozen", "avg_num_tasks_scheduled",
        "num_replans", "num_forced_replans", "avg_solve_time_ms", "total_runtime_s",
        # LLM ????
        "llm_calls", "llm_time_total_ms", "llm_latency_total_ms",
        "llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens",
        "llm_cache_hit_rate", "llm_fallback_count",
        "solver_time_total_ms", "wall_time_total_ms"

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
        "avg_delay", "avg_drift", "avg_time_deviation_min",
        "avg_forced_replan_rate", "avg_feasible_rate",
        "delay_ratio", "drift_ratio", "time_dev_ratio", "forced_replan_ratio",
        "delay_ok", "feasible_ok", "combined_score",
        "epsilon_metric", "epsilon_threshold", "epsilon_value",
        "epsilon_ok", "epsilon_violation",
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
    tuning_lambda: float = 5.0,
    epsilon_metric: str = "avg_delay",
    epsilon_relative: str = "baseline",
    epsilon_value: float = 0.10
):
    """Compute and save summary statistics."""
    import math

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # group by (dataset, policy_name)
    groups: Dict[Tuple[str, str], List[EpisodeMetricsRecord]] = {}
    for record in records:
        key = (record.dataset, record.policy_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def std(values):
        if len(values) < 2:
            return 0.0
        m = mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    def ci95(values):
        if len(values) < 2:
            return 0.0
        return 1.96 * std(values) / math.sqrt(len(values))

    rows = []
    for (dataset, policy_name), recs in sorted(groups.items()):
        baseline_recs = groups.get((dataset, "fixed_default"), [])
        baseline_delay = mean([r.avg_delay for r in baseline_recs]) if baseline_recs else 0.0
        if epsilon_metric != "avg_delay":
            raise ValueError(f"Unsupported epsilon_metric: {epsilon_metric}")
        if epsilon_relative == "baseline":
            epsilon_threshold = baseline_delay * (1 + epsilon_value)
        elif epsilon_relative == "absolute":
            epsilon_threshold = epsilon_value
        else:
            raise ValueError(f"Unsupported epsilon_relative: {epsilon_relative}")

        delays = [r.avg_delay for r in recs]
        drifts = [r.episode_drift for r in recs]
        on_times = [r.on_time_rate for r in recs]
        shifts = [r.total_shifts for r in recs]
        switches = [r.total_switches for r in recs]
        solve_times = [r.avg_solve_time_ms for r in recs]
        time_devs = [r.avg_time_deviation_min for r in recs]
        resource_switches = [r.total_resource_switches for r in recs]
        makespans = [r.makespan_cmax for r in recs]
        feasible_rates = [r.feasible_rate for r in recs]
        forced_replan_rates = [r.forced_replan_rate for r in recs]
        avg_frozens = [r.avg_frozen for r in recs]
        avg_tasks_scheduled = [r.avg_num_tasks_scheduled for r in recs]
        util_r_pads = [r.util_r_pad for r in recs]

        combined = [
            d + tuning_lambda * dr
            for d, dr in zip(delays, drifts)
        ]

        llm_calls = [r.llm_calls for r in recs]
        llm_tokens = [r.llm_total_tokens for r in recs]
        llm_fallbacks = [r.llm_fallback_count for r in recs]
        llm_cache_hits = [r.llm_cache_hit_rate for r in recs]

        row = {
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
            "avg_time_deviation_min_mean": round(mean(time_devs), 2),
            "avg_time_deviation_min_ci95": round(ci95(time_devs), 2),
            "total_resource_switches_mean": round(mean(resource_switches), 2),
            "total_resource_switches_ci95": round(ci95(resource_switches), 2),
            "makespan_cmax_mean": round(mean(makespans), 2),
            "makespan_cmax_ci95": round(ci95(makespans), 2),
            "feasible_rate_mean": round(mean(feasible_rates), 4),
            "forced_replan_rate_mean": round(mean(forced_replan_rates), 4),
            "avg_frozen_mean": round(mean(avg_frozens), 2),
            "avg_num_tasks_scheduled_mean": round(mean(avg_tasks_scheduled), 2),
            "util_r_pad_mean": round(mean(util_r_pads), 4),
            "util_r_pad_ci95": round(ci95(util_r_pads), 4),
            "avg_solve_time_ms_mean": round(mean(solve_times), 1),
            "llm_calls_mean": round(mean(llm_calls), 1),
            "llm_tokens_mean": round(mean(llm_tokens), 0),
            "llm_fallback_mean": round(mean(llm_fallbacks), 2),
            "llm_cache_hit_rate_mean": round(mean(llm_cache_hits), 4),
            "epsilon_metric": epsilon_metric,
            "epsilon_threshold": round(epsilon_threshold, 3),
            "epsilon_ok_rate": round(
                mean([1.0 if d <= epsilon_threshold else 0.0 for d in delays]), 4
            ),
            "epsilon_violation_mean": round(
                mean([max(0.0, d - epsilon_threshold) for d in delays]), 3
            )
        }
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved summary: {filepath}")

def save_best_params(params: Dict[str, Any], filepath: str):
    """保存最优参数"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"保存最优参数: {filepath}")


def save_seed_manifest(
    train_assignments: List[Tuple[int, str]],
    test_assignments: List[Tuple[int, str]],
    filepath: str
):
    """保存训练/测试种子分配清单，便于跨实验对齐"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def summarize(assignments: List[Tuple[int, str]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for _, level in assignments:
            counts[level] = counts.get(level, 0) + 1
        return {
            "total": len(assignments),
            "by_level": counts
        }

    data = {
        "train": {
            "summary": summarize(train_assignments),
            "assignments": [{"seed": s, "disturbance_level": l} for s, l in train_assignments]
        },
        "test": {
            "summary": summarize(test_assignments),
            "assignments": [{"seed": s, "disturbance_level": l} for s, l in test_assignments]
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"保存种子清单: {filepath}")


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
    seed_manifest_path = os.path.join(exp_config.output_dir, "seed_manifest.json")
    save_seed_manifest(train_assignments, test_assignments, seed_manifest_path)
    
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
        "fixed_default": BASELINE_FIXED_DEFAULT,
        "nofreeze": {},
        "greedy": {},
    }
    
    # 4. 在测试集上评估
    test_records = run_evaluation(
        test_assignments, policies, "test",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose,
        exp_config=exp_config,
        output_dir=exp_config.output_dir
    )
    
    # 5. 在训练集上也用最终策略跑一遍（用于完整记录）
    train_records = run_evaluation(
        train_assignments, policies, "train",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose,
        exp_config=exp_config,
        output_dir=exp_config.output_dir
    )
    
    # 6. 合并并保存
    all_records = train_records + test_records
    
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    save_episode_results(all_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(
        all_records,
        summary_path,
        exp_config.tuning_lambda,
        exp_config.epsilon_metric,
        exp_config.epsilon_relative,
        exp_config.epsilon_value
    )
    
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
        "fixed_default": BASELINE_FIXED_DEFAULT,
        "nofreeze": {},
        "greedy": {},
    }
    
    # 在训练集上评估所有策略（完整记录）
    train_records = run_evaluation(
        train_assignments, policies, "train",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose,
        exp_config=exp_config,
        output_dir=exp_config.output_dir
    )
    
    # 保存结果
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    save_episode_results(train_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(
        train_records,
        summary_path,
        exp_config.tuning_lambda,
        exp_config.epsilon_metric,
        exp_config.epsilon_relative,
        exp_config.epsilon_value
    )
    
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


def run_test_only(
    exp_config: ExperimentConfig,
    params_path: str,
    selected_policies: List[str],
    verbose: bool = True
):
    """
    仅运行测试阶段：加载最优参数 + 测试集评估
    
    Args:
        exp_config: 实验配置
        params_path: 最优参数文件路径
        selected_policies: 要评估的策略列表
        verbose: 是否打印详细信息
    
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
    print(f"选定策略: {selected_policies}")
    print("Note: for fair comparison, keep the same test seed count/assignment as fixed_tuned, or include fixed_tuned in the same run.")
    print(f"输出目录: {exp_config.output_dir}")
    
    # 加载最优参数
    if not os.path.exists(params_path):
        print(f"错误: 参数文件不存在: {params_path}")
        return
    
    with open(params_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    
    print(f"\n加载的最优参数: {best_params}")
    
    # 生成测试集种子（从训练集之后开始）
    train_assignments, test_assignments = generate_seed_assignments(
        exp_config.num_train_seeds,  # 需要知道训练集大小以确定测试集起始种子
        exp_config.num_test_seeds,
        exp_config.disturbance_levels
    )
    
    print(f"测试集种子: {[a[0] for a in test_assignments[:5]]}... (共 {len(test_assignments)})")
    
    seed_manifest_path = os.path.join(exp_config.output_dir, "seed_manifest.json")
    save_seed_manifest(train_assignments, test_assignments, seed_manifest_path)


    all_policies = {
        "fixed_tuned": best_params,
        "fixed_default": BASELINE_FIXED_DEFAULT,
        "nofreeze": {},
        "greedy": {},
        "mockllm": {},
        "llm_real": {}  # LLM params from exp_config
    }
    
    # 过滤选定的策略
    policies = {
        name: params
        for name, params in all_policies.items()
        if name in selected_policies
    }
    
    # 检查 llm_real 是否可用
    if "llm_real" in policies and not HAS_REAL_LLM:
        print("警告: llm_real 策略不可用（缺少 llm_client 模块），已跳过")
        del policies["llm_real"]
    
    # 检查 llm_real 的并行设置
    if "llm_real" in policies and exp_config.max_workers > 1:
        logger.warning(
            f"检测到 llm_real 策略，强制将 workers 从 {exp_config.max_workers} 降为 1。"
            f"原因: LLM API 调用需要顺序执行以保证缓存一致性和避免速率限制。"
        )
        exp_config.max_workers = 1
    
    print(f"实际评估策略: {list(policies.keys())}")
    
    # 在测试集上评估
    test_records = run_evaluation(
        test_assignments, policies, "test",
        solver_timeout=exp_config.solver_timeout_s,
        max_workers=exp_config.max_workers,
        verbose=verbose,
        exp_config=exp_config,
        output_dir=exp_config.output_dir
    )
    
    # 读取已有的训练集结果（如果存在）
    episode_path = os.path.join(exp_config.output_dir, "results_per_episode.csv")
    all_records = test_records
    
    if os.path.exists(episode_path):
        print(f"\n检测到已有结果文件，合并训练集和测试集数据...")
        existing_records = _load_existing_records(episode_path)
        
        # 合并：保留训练集数据，添加测试集数据
        train_records = [r for r in existing_records if r.dataset == "train"]
        all_records = train_records + test_records
        print(f"  训练集记录: {len(train_records)}")
        print(f"  测试集记录: {len(test_records)}")
    
    # 保存合并后的结果
    save_episode_results(all_records, episode_path)
    
    summary_path = os.path.join(exp_config.output_dir, "summary.csv")
    save_summary(
        all_records,
        summary_path,
        exp_config.tuning_lambda,
        exp_config.epsilon_metric,
        exp_config.epsilon_relative,
        exp_config.epsilon_value
    )
    
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


def _load_existing_records(filepath: str) -> List[EpisodeMetricsRecord]:
    """加载已有的 episode 记录"""
    records = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理可能缺失的 LLM 字段
            record = EpisodeMetricsRecord(
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
                util_r_pad=float(row.get('util_r_pad', 0.0)),
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
                # LLM ???????????????
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
            )
            records.append(record)

    
    return records


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="火箭发射排程仿真 - 批量实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 跑 baseline 策略（可并行）
  python run_experiments.py --train-seeds 60 --test-seeds 60 --output results/ --workers 8

  # 仅跑 llm_real 策略（强制单线程）
  python run_experiments.py --mode test-only --policy llm_real --test-seeds 60 \\
      --llm-base-url https://api-inference.modelscope.cn/v1 \\
      --llm-model Qwen/Qwen3-32B --llm-key-env DASHSCOPE_API_KEY \\
      --output results_llm/

  # 跑所有策略包括 llm_real
  python run_experiments.py --mode test-only --policy fixed_tuned,nofreeze,llm_real \\
      --test-seeds 30 --output results/
"""
    )
    
    # 基本参数
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
        "--solver-timeout", type=float, default=20.0,
        help="每次 CP-SAT 求解超时秒数 (default: 10.0)"
    )
    parser.add_argument(
        "--lambda", dest="tuning_lambda", type=float, default=5.0,
        help="drift weight in combined score (default: 5.0)"
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
        help="运行模式: full=完整流程, train-only=仅训练调参, test-only=仅测试评估"
    )
    parser.add_argument(
        "--load-params", type=str, default=None,
        help="加载已有的最优参数文件路径 (用于 test-only 模式)"
    )
    
    # 策略选择
    parser.add_argument(
        "--policy", type=str, default=None,
        help="要评估的策略，逗号分隔 (default: 全部)。"
             "可选: fixed_tuned, fixed_default, nofreeze, greedy, mockllm, llm_real"
    )
    
    # ========== LLM 参数 ==========
    parser.add_argument(
        "--llm-base-url", type=str, default="https://api-inference.modelscope.cn/v1",
        help="LLM API 基础 URL (default: ModelScope)"
    )
    parser.add_argument(
        "--llm-model", type=str, default="Qwen/Qwen3-32B",
        help="LLM 模型名称 (default: Qwen/Qwen3-32B)"
    )
    parser.add_argument(
        "--llm-key-env", type=str, default="DASHSCOPE_API_KEY",
        help="API Key 环境变量名 (default: DASHSCOPE_API_KEY)"
    )
    parser.add_argument(
        "--llm-cache-dir", type=str, default=None,
        help="LLM 缓存目录 (default: output_dir/llm_cache)"
    )
    parser.add_argument(
        "--llm-log-path", type=str, default=None,
        help="LLM 日志路径 (default: output_dir/llm_logs)"
    )
    parser.add_argument(
        "--llm-timeout-s", type=float, default=30.0,
        help="LLM 调用超时秒数 (default: 30.0)"
    )
    parser.add_argument(
        "--llm-max-retries", type=int, default=5,
        help="LLM 最大重试次数 (default: 5)"
    )
    parser.add_argument(
        "--llm-temperature", type=float, default=0.0,
        help="LLM temperature 参数 (default: 0.0)"
    )
    parser.add_argument(
        "--llm-top-p", type=float, default=1.0,
        help="LLM top_p 参数 (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # 快速模式
    if args.quick:
        args.train_seeds = 9
        args.test_seeds = 9
        print("快速测试模式: train=9, test=9")
    
    # 解析策略列表
    all_baseline_policies = ["fixed_tuned", "fixed_default", "nofreeze", "greedy"]
    if args.policy:
        selected_policies = [p.strip() for p in args.policy.split(",")]
    else:
        selected_policies = all_baseline_policies  # 默认不包含 llm_real
    
    # 创建配置
    exp_config = ExperimentConfig(
        num_train_seeds=args.train_seeds,
        num_test_seeds=args.test_seeds,
        output_dir=args.output,
        solver_timeout_s=args.solver_timeout,
        tuning_lambda=args.tuning_lambda,
        max_workers=args.workers,
        epsilon_metric=args.epsilon_metric,
        epsilon_relative=args.epsilon_relative,
        epsilon_value=args.epsilon_value,
        # LLM 参数
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_key_env=args.llm_key_env,
        llm_cache_dir=args.llm_cache_dir,
        llm_log_path=args.llm_log_path,
        llm_timeout_s=args.llm_timeout_s,
        llm_max_retries=args.llm_max_retries,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p
    )
    
    # 检查 llm_real 策略的并行设置
    if "llm_real" in selected_policies and args.workers > 1:
        logger.warning(
            f"检测到 llm_real 策略，强制将 workers 从 {args.workers} 降为 1。"
            f"原因: LLM API 调用需要顺序执行以保证缓存一致性和避免速率限制。"
        )
        exp_config.max_workers = 1
    
    # 快速模式下减少调参组合
    if args.quick:
        exp_config.freeze_horizon_hours = [2, 4, 6]
        exp_config.w_delay_values = [10.0, 12.0]
        exp_config.w_shift_values = [0.6, 1.0]
        exp_config.w_switch_values = [10, 20]
        print("  调参网格缩减为 3×2×2×2=24 组合")
    
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
        
        run_test_only(
            exp_config,
            args.load_params,
            selected_policies,
            verbose=args.verbose or args.quick
        )


if __name__ == "__main__":
    main()
