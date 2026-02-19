#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路A 10天 batch 测试脚本
运行 light/medium/heavy × seeds 0..2，输出汇总 CSV

Usage:
    python run_batch_10day.py [--seeds 0 1 2] [--output results/batch_10day]
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Dict, Any

from config import Config, make_config_for_difficulty, MISSIONS_BY_DIFFICULTY
from scenario import generate_scenario
from simulator import simulate_episode, save_episode_logs
from policies.policy_fixed import FixedWeightPolicy
from metrics import metrics_to_dict


DIFFICULTIES = ["light", "medium", "heavy"]

# Baseline 策略定义
BASELINES = {
    "static": {
        "desc": "Static (一次排定，不重排)",
        "freeze_horizon": 9999,  # 全冻结 = 不重排
        "use_two_stage": False,
        "epsilon_solver": None,
    },
    "full_unlock": {
        "desc": "Full-Unlock (全解锁重排)",
        "freeze_horizon": 0,
        "use_two_stage": True,
        "epsilon_solver": 0.05,
    },
    "fixed_tuned": {
        "desc": "Fixed-Tuned (固定权重+二阶段)",
        "freeze_horizon": 12,
        "use_two_stage": True,
        "epsilon_solver": 0.05,
    },
}


def run_one(seed: int, difficulty: str, policy_name: str, policy_params: dict,
            output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """运行单个 episode 并返回结果字典"""
    config = make_config_for_difficulty(difficulty)
    scenario = generate_scenario(seed=seed, config=config)

    policy = FixedWeightPolicy(
        w_delay=10.0,
        w_shift=1.0 if policy_name != "static" else 0.0,
        w_switch=5.0 if policy_name != "static" else 0.0,
        freeze_horizon=policy_params.get("freeze_horizon", 12),
        policy_name=policy_name,
        use_two_stage=policy_params.get("use_two_stage", True),
        epsilon_solver=policy_params.get("epsilon_solver"),
    )

    result = simulate_episode(
        policy=policy,
        scenario=scenario,
        config=config,
        verbose=verbose,
    )

    # 保存日志
    ep_dir = os.path.join(output_dir, f"{difficulty}_{policy_name}_seed{seed}")
    save_episode_logs(result, ep_dir, scenario)

    m = result.metrics
    return {
        "seed": seed,
        "difficulty": difficulty,
        "policy": policy_name,
        "num_missions": len(scenario.missions),
        "sim_total_slots": config.sim_total_slots,
        "on_time_rate": round(m.on_time_rate, 4),
        "avg_delay": round(m.avg_delay, 2),
        "max_delay": m.max_delay,
        "episode_drift": round(m.episode_drift, 4),
        "drift_per_replan": round(m.drift_per_replan, 6),
        "drift_per_day": round(m.drift_per_day, 6),
        "drift_per_active_mission": round(m.drift_per_active_mission, 6),
        "total_shifts": m.total_shifts,
        "total_switches": m.total_switches,
        "num_replans": m.num_replans,
        "num_forced_replans": m.num_forced_replans,
        "completion_rate": round(m.completion_rate, 4),
        "feasible_rate": round(m.feasible_rate, 4),
        "avg_solve_time_ms": round(m.avg_solve_time_ms, 2),
        "total_runtime_s": round(result.total_runtime_s, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="路A 10天 batch 实验")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=str, default="results/batch_10day")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    total = len(DIFFICULTIES) * len(BASELINES) * len(args.seeds)
    idx = 0

    for difficulty in DIFFICULTIES:
        for policy_name, params in BASELINES.items():
            for seed in args.seeds:
                idx += 1
                print(f"\n[{idx}/{total}] difficulty={difficulty}, policy={policy_name}, seed={seed}")
                t0 = time.time()
                record = run_one(seed, difficulty, policy_name, params, args.output, args.verbose)
                elapsed = time.time() - t0
                print(f"  -> on_time={record['on_time_rate']}, drift={record['episode_drift']}, "
                      f"drift/replan={record['drift_per_replan']}, "
                      f"drift/day={record['drift_per_day']}, "
                      f"completed={record['completion_rate']}, time={elapsed:.1f}s")
                all_records.append(record)

    # 写 CSV
    csv_path = os.path.join(args.output, "batch_summary.csv")
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n✓ 汇总 CSV 已写入: {csv_path}")

    # 写 JSON
    json_path = os.path.join(args.output, "batch_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"✓ 汇总 JSON 已写入: {json_path}")

    # 按 difficulty 汇总均值/方差
    print("\n" + "=" * 80)
    print("  Aggregated Results (mean ± std)")
    print("=" * 80)
    import statistics
    for difficulty in DIFFICULTIES:
        for policy_name in BASELINES:
            subset = [r for r in all_records
                      if r["difficulty"] == difficulty and r["policy"] == policy_name]
            if not subset:
                continue
            n = len(subset)
            for metric in ["on_time_rate", "episode_drift", "drift_per_replan", "drift_per_day",
                           "completion_rate", "avg_solve_time_ms", "total_runtime_s"]:
                vals = [r[metric] for r in subset]
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if n > 1 else 0.0
                if metric == "on_time_rate":
                    print(f"  {difficulty}/{policy_name}: ", end="")
                print(f"{metric}={mean:.4f}±{std:.4f}  ", end="")
            print()

    print(f"\n✓ 全部 {total} 个 episode 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
