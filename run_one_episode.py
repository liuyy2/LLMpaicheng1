"""
run_one_episode.py - 运行单个 Episode 的演示脚本

使用 FixedWeightPolicy 运行一次完整仿真，输出关键指标和日志文件。

Usage:
    python run_one_episode.py [--seed SEED] [--verbose] [--output DIR]
"""

import argparse
import os
import sys
import json
from datetime import datetime

from config import Config, DEFAULT_CONFIG
from scenario import generate_scenario, save_scenario
from simulator import simulate_episode, save_episode_logs
from policies.policy_fixed import FixedWeightPolicy
from metrics import metrics_to_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single episode simulation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for scenario generation (default: 42)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed simulation progress"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="logs",
        help="Output directory for logs (default: logs)"
    )
    parser.add_argument(
        "--w-delay", type=float, default=10.0,
        help="Delay weight (default: 10.0)"
    )
    parser.add_argument(
        "--w-shift", type=float, default=1.0,
        help="Shift weight (default: 1.0)"
    )
    parser.add_argument(
        "--w-switch", type=float, default=5.0,
        help="Switch weight (default: 5.0)"
    )
    return parser.parse_args()


def print_header():
    print("=" * 70)
    print("  Launch Scheduling Simulation - Single Episode")
    print("=" * 70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_scenario_info(scenario):
    print("-" * 50)
    print(" Scenario Information")
    print("-" * 50)
    print(f"  Seed:           {scenario.seed}")
    print(f"  Missions:       {len(scenario.missions)}")
    print(f"  Resources:      {len(scenario.resources)}")
    print(f"  Disturbances:   {len(scenario.disturbance_timeline)}")

    event_counts = {}
    for e in scenario.disturbance_timeline:
        event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
    print(f"  Event breakdown: {event_counts}")
    print()


def print_policy_info(policy, config):
    print("-" * 50)
    print(" Policy & Configuration")
    print("-" * 50)
    print(f"  Policy:         {policy.name}")
    print(f"  Weights:        delay={policy._w_delay}, shift={policy._w_shift}, switch={policy._w_switch}")
    print(f"  Freeze horizon: {config.freeze_horizon} slots ({config.freeze_horizon * config.slot_minutes} min)")
    print(f"  Rolling interval: {config.rolling_interval} slots ({config.rolling_interval * config.slot_minutes} min)")
    print(f"  Horizon:        {config.horizon_slots} slots ({config.horizon_slots * config.slot_minutes // 60} h)")
    print(f"  Total sim:      {config.sim_total_slots} slots ({config.sim_total_slots * config.slot_minutes // 60} h)")
    print()


def print_results(result):
    print("-" * 50)
    print(" Episode Results")
    print("-" * 50)
    
    m = result.metrics
    
    print(f"   Performance Metrics:")
    print(f"     On-time rate:     {m.on_time_rate:.2%}")
    print(f"     Total delay:      {m.total_delay} slots")
    print(f"     Avg delay:        {m.avg_delay:.2f} slots")
    print(f"     Max delay:        {m.max_delay} slots")
    print()
    
    print(f"   Stability Metrics:")
    print(f"     Episode drift:    {m.episode_drift:.4f}")
    print(f"     Total shifts:     {m.total_shifts}")
    print(f"     Total switches:   {m.total_switches}")
    print()
    
    print(f"   Solver Performance:")
    print(f"     Replans:          {m.num_replans}")
    print(f"     Forced replans:   {m.num_forced_replans}")
    print(f"     Total solve time: {m.total_solve_time_ms} ms")
    print(f"     Avg solve time:   {m.avg_solve_time_ms:.2f} ms")
    print()
    
    print(f"   Completion:")
    print(f"     Completed:        {m.num_completed}/{m.num_total} ({m.completion_rate:.2%})")
    print(f"     Runtime:          {result.total_runtime_s:.2f} s")
    print()


def print_schedule(result, max_show=10):
    print("-" * 50)
    print(" Final Schedule")
    print("-" * 50)

    if not result.final_schedule:
        print("  No schedule available")
        return

    if hasattr(result.final_schedule[0], 'op_id'):
        sorted_schedule = sorted(result.final_schedule, key=lambda x: x.start_slot)
        print(f"  {'Op':<12} {'Mission':<10} {'Start':<10} {'End':<10}")
        print(f"  {'-'*48}")
        for i, a in enumerate(sorted_schedule):
            if i >= max_show:
                print(f"  ... and {len(sorted_schedule) - max_show} more ops")
                break
            print(f"  {a.op_id:<12} {a.mission_id:<10} {a.start_slot:<10} {a.end_slot:<10}")
    else:
        sorted_schedule = sorted(result.final_schedule, key=lambda x: x.launch_slot)
        print(f"  {'Task':<10} {'Pad':<10} {'Start':<10} {'Launch':<10}")
        print(f"  {'-'*40}")
        for i, a in enumerate(sorted_schedule):
            if i >= max_show:
                print(f"  ... and {len(sorted_schedule) - max_show} more ops")
                break
            print(f"  {a.task_id:<10} {a.pad_id:<10} {a.start_slot:<10} {a.launch_slot:<10}")

    print()


def print_saved_files(saved_files, output_dir):
    print("-" * 50)
    print(" Saved Files")
    print("-" * 50)
    print(f"  Output directory: {os.path.abspath(output_dir)}")
    for name, path in saved_files.items():
        print(f"    - {name}: {os.path.basename(path)}")
    print()


def main():
    args = parse_args()
    
    print_header()
    
    # 1. 生成场景
    print(" Generating scenario...")
    config = DEFAULT_CONFIG
    scenario = generate_scenario(seed=args.seed, config=config)
    print_scenario_info(scenario)
    
    # 2. 创建策略
    policy = FixedWeightPolicy(
        w_delay=args.w_delay,
        w_shift=args.w_shift,
        w_switch=args.w_switch,
        policy_name="fixed"
    )
    print_policy_info(policy, config)
    
    # 3. 运行仿真
    print(" Running simulation...")
    if args.verbose:
        print()
    
    result = simulate_episode(
        policy=policy,
        scenario=scenario,
        config=config,
        verbose=args.verbose
    )
    
    if not args.verbose:
        print("   Simulation complete")
    print()
    
    # 4. 打印结果
    print_results(result)
    print_schedule(result)
    
    # 5. 保存日志
    output_dir = os.path.join(args.output, f"episode_{args.seed}_{policy.name}")
    print(f" Saving logs to {output_dir}...")
    
    saved_files = save_episode_logs(result, output_dir, scenario)
    print_saved_files(saved_files, output_dir)
    
    # 6. 打印汇总 JSON
    print("-" * 50)
    print(" Summary JSON")
    print("-" * 50)
    summary = {
        "seed": result.seed,
        "policy": result.policy_name,
        "on_time_rate": round(result.metrics.on_time_rate, 4),
        "episode_drift": round(result.metrics.episode_drift, 4),
        "total_delay": result.metrics.total_delay,
        "total_shifts": result.metrics.total_shifts,
        "total_switches": result.metrics.total_switches,
        "completion_rate": round(result.metrics.completion_rate, 4),
        "runtime_s": round(result.total_runtime_s, 3)
    }
    print(json.dumps(summary, indent=2))
    print()
    
    print("=" * 70)
    print("  Episode completed successfully!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
