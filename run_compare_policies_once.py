#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单次策略对比脚本 - 同一个 scenario seed 运行所有策略，打印指标对比

用法:
    python run_compare_policies_once.py --seed 42
    python run_compare_policies_once.py --seed 42 --verbose
    python run_compare_policies_once.py --seed 42 --output results/compare_42.json
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass

from config import Config, DEFAULT_CONFIG
from scenario import generate_scenario, Scenario
from simulator import simulate_episode, EpisodeResult, save_episode_logs
from policies import (
    FixedWeightPolicy,
    NoFreezePolicy,
    GreedyPolicy,
    MockLLMPolicy,
    create_policy
)


@dataclass
class PolicyComparisonResult:
    """单策略对比结果"""
    policy_name: str
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "completed": self.completed,
            "total": self.total,
            "on_time_rate": round(self.on_time_rate, 4),
            "avg_delay": round(self.avg_delay, 2),
            "max_delay": self.max_delay,
            "episode_drift": round(self.episode_drift, 4),
            "total_shifts": self.total_shifts,
            "total_switches": self.total_switches,
            "num_replans": self.num_replans,
            "num_forced_replans": self.num_forced_replans,
            "avg_solve_time_ms": round(self.avg_solve_time_ms, 1),
            "total_runtime_s": round(self.total_runtime_s, 2)
        }


def run_policy_comparison(
    seed: int,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = False,
    output_dir: str = None
) -> List[PolicyComparisonResult]:
    """
    运行策略对比
    
    Args:
        seed: 场景随机种子
        config: 配置
        verbose: 是否打印详细信息
        output_dir: 日志输出目录（可选）
    
    Returns:
        各策略的对比结果列表
    """
    # 1. 生成场景
    print(f"\n{'='*70}")
    print(f" 策略对比实验 - Seed: {seed}")
    print(f"{'='*70}")
    
    scenario = generate_scenario(seed=seed, config=config)
    print(f"\n场景生成完成:")
    print(f"  - 任务数: {len(scenario.tasks)}")
    print(f"  - Pad 数: {len(scenario.pads)}")
    print(f"  - 扰动事件数: {len(scenario.disturbance_timeline)}")
    
    # 2. 定义策略
    policies = [
        FixedWeightPolicy(
            w_delay=10.0, w_shift=1.0, w_switch=5.0,
            freeze_horizon=12, policy_name="fixed"
        ),
        NoFreezePolicy(
            w_delay=10.0, w_shift=0.2, w_switch=1.0,
            freeze_horizon=0, policy_name="nofreeze"
        ),
        GreedyPolicy(
            sort_by="due", prefer_pad_switch=True,
            policy_name="greedy"
        ),
        MockLLMPolicy(
            policy_name="mockllm",
            enable_logging=True
        )
    ]
    
    # 3. 运行各策略
    results: List[PolicyComparisonResult] = []
    
    for policy in policies:
        print(f"\n{'─'*50}")
        print(f" 运行策略: {policy.name}")
        print(f"{'─'*50}")
        
        start_time = time.time()
        
        try:
            episode_result = simulate_episode(
                policy=policy,
                scenario=scenario,
                config=config,
                verbose=verbose
            )
            
            # 提取指标
            m = episode_result.metrics
            comparison_result = PolicyComparisonResult(
                policy_name=policy.name,
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
                total_runtime_s=episode_result.total_runtime_s
            )
            results.append(comparison_result)
            
            # 打印简要结果
            print(f"  ✓ 完成: {m.num_completed}/{m.num_total}")
            print(f"  ✓ 准时率: {m.on_time_rate:.2%}")
            print(f"  ✓ 平均延迟: {m.avg_delay:.2f} slots")
            print(f"  ✓ Episode Drift: {m.episode_drift:.4f}")
            print(f"  ✓ 运行时间: {episode_result.total_runtime_s:.2f}s")
            
            # 保存日志
            if output_dir:
                policy_log_dir = os.path.join(output_dir, f"episode_{seed}_{policy.name}")
                save_episode_logs(episode_result, policy_log_dir, scenario)
                
                # 如果是 MockLLM，保存决策日志
                if hasattr(policy, 'save_logs'):
                    llm_log_path = os.path.join(policy_log_dir, "llm_decisions.jsonl")
                    policy.save_logs(llm_log_path)
                    
        except Exception as e:
            print(f"  ✗ 策略执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建空结果
            results.append(PolicyComparisonResult(
                policy_name=policy.name,
                completed=0, total=len(scenario.tasks),
                on_time_rate=0.0, avg_delay=float('inf'),
                max_delay=0, episode_drift=0.0,
                total_shifts=0, total_switches=0,
                num_replans=0, num_forced_replans=0,
                avg_solve_time_ms=0.0,
                total_runtime_s=time.time() - start_time
            ))
    
    return results


def print_comparison_table(results: List[PolicyComparisonResult]):
    """打印对比表格"""
    print(f"\n{'='*70}")
    print(f" 策略对比结果汇总")
    print(f"{'='*70}")
    
    # 表头
    headers = [
        "策略", "完成", "准时率", "平均延迟", "Drift", 
        "Shifts", "Switches", "重排次数", "运行时间"
    ]
    
    col_widths = [12, 8, 10, 10, 10, 8, 10, 10, 10]
    
    # 打印表头
    header_line = "│"
    for i, h in enumerate(headers):
        header_line += f" {h:^{col_widths[i]}} │"
    
    separator = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
    top_line = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
    bottom_line = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
    
    print(top_line)
    print(header_line)
    print(separator)
    
    # 打印数据
    for r in results:
        row = [
            r.policy_name[:12],
            f"{r.completed}/{r.total}",
            f"{r.on_time_rate:.1%}",
            f"{r.avg_delay:.1f}",
            f"{r.episode_drift:.4f}",
            str(r.total_shifts),
            str(r.total_switches),
            str(r.num_replans),
            f"{r.total_runtime_s:.2f}s"
        ]
        
        row_line = "│"
        for i, cell in enumerate(row):
            row_line += f" {cell:^{col_widths[i]}} │"
        print(row_line)
    
    print(bottom_line)
    
    # 找出最优指标
    print("\n最优策略（按指标）:")
    
    # 最高准时率
    best_ontime = max(results, key=lambda x: x.on_time_rate)
    print(f"  - 准时率最高: {best_ontime.policy_name} ({best_ontime.on_time_rate:.2%})")
    
    # 最低延迟
    best_delay = min(results, key=lambda x: x.avg_delay)
    print(f"  - 平均延迟最低: {best_delay.policy_name} ({best_delay.avg_delay:.2f})")
    
    # 最稳定（最低 drift）
    best_drift = min(results, key=lambda x: x.episode_drift)
    print(f"  - 最稳定 (Drift): {best_drift.policy_name} ({best_drift.episode_drift:.4f})")
    
    # 最快
    best_speed = min(results, key=lambda x: x.total_runtime_s)
    print(f"  - 最快: {best_speed.policy_name} ({best_speed.total_runtime_s:.2f}s)")


def save_comparison_results(
    results: List[PolicyComparisonResult],
    filepath: str,
    seed: int
):
    """保存对比结果到文件"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    output = {
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n对比结果已保存至: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="单次策略对比 - 同一场景运行多个策略并比较结果"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="场景随机种子 (default: 42)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印详细仿真信息"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="结果输出目录 (default: None)"
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="对比结果 JSON 文件路径"
    )
    
    args = parser.parse_args()
    
    # 运行对比
    output_dir = args.output or f"logs/compare_{args.seed}"
    
    results = run_policy_comparison(
        seed=args.seed,
        verbose=args.verbose,
        output_dir=output_dir
    )
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存结果
    if args.save_json:
        save_comparison_results(results, args.save_json, args.seed)
    else:
        # 默认保存位置
        default_json = os.path.join(output_dir, "comparison.json")
        save_comparison_results(results, default_json, args.seed)


if __name__ == "__main__":
    main()
