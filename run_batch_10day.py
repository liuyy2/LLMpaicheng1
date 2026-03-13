#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

from config import make_config_for_difficulty
from run_experiments import run_single_episode


DIFFICULTIES = ["light", "medium", "heavy"]

BASELINES: Dict[str, Dict[str, Any]] = {
    "static": {
        "freeze_horizon": 9999,
        "use_two_stage": False,
        "epsilon_solver": None,
    },
    "fixed_tuned": {
        "freeze_horizon": 12,
        "use_two_stage": True,
        "epsilon_solver": 0.05,
    },
    "full_unlock": {
        "freeze_horizon": 0,
        "use_two_stage": True,
        "epsilon_solver": 0.05,
    },
    "ga_repair": {},
    "alns_repair": {},
}


def run_one(
    seed: int,
    difficulty: str,
    policy_name: str,
    policy_params: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    record = run_single_episode(
        seed=seed,
        disturbance_level=difficulty,
        policy_name="fixed_default" if policy_name == "static" else policy_name,
        policy_params=policy_params,
        dataset="batch",
        output_dir=output_dir,
    )
    config = make_config_for_difficulty(difficulty)
    return {
        "seed": seed,
        "difficulty": difficulty,
        "policy": policy_name,
        "num_missions": config.num_missions,
        "sim_total_slots": config.sim_total_slots,
        "on_time_rate": round(record.on_time_rate, 4),
        "avg_delay": round(record.avg_delay, 2),
        "weighted_tardiness": round(record.weighted_tardiness, 2),
        "episode_drift": round(record.episode_drift, 4),
        "drift_per_replan": round(record.drift_per_replan, 6),
        "completion_rate": round(record.completed / record.total if record.total else 0.0, 4),
        "feasible_rate": round(record.feasible_rate, 4),
        "avg_solve_time_ms": round(record.avg_solve_time_ms, 2),
        "total_runtime_s": round(record.total_runtime_s, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch policy runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--policies", type=str, default="static,fixed_tuned,full_unlock,ga_repair,alns_repair")
    parser.add_argument("--output", type=str, default="results/batch_10day")
    args = parser.parse_args()

    selected_policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = [name for name in selected_policies if name not in BASELINES]
    if unknown:
        raise ValueError(f"Unknown policies: {unknown}")

    os.makedirs(args.output, exist_ok=True)
    all_records: List[Dict[str, Any]] = []
    total = len(DIFFICULTIES) * len(selected_policies) * len(args.seeds)
    idx = 0

    for difficulty in DIFFICULTIES:
        for policy_name in selected_policies:
            params = BASELINES[policy_name]
            for seed in args.seeds:
                idx += 1
                print(f"\n[{idx}/{total}] difficulty={difficulty}, policy={policy_name}, seed={seed}")
                t0 = time.time()
                record = run_one(seed, difficulty, policy_name, params, args.output)
                elapsed = time.time() - t0
                print(
                    f"  -> on_time={record['on_time_rate']:.4f}, avg_delay={record['avg_delay']:.2f}, "
                    f"drift={record['episode_drift']:.4f}, solve_ms={record['avg_solve_time_ms']:.1f}, "
                    f"time={elapsed:.1f}s"
                )
                all_records.append(record)

    csv_path = os.path.join(args.output, "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    json_path = os.path.join(args.output, "batch_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Aggregated Results")
    print("=" * 80)
    for difficulty in DIFFICULTIES:
        for policy_name in selected_policies:
            subset = [row for row in all_records if row["difficulty"] == difficulty and row["policy"] == policy_name]
            if not subset:
                continue
            print(
                f"{difficulty}/{policy_name}: "
                f"avg_delay={statistics.mean(row['avg_delay'] for row in subset):.2f}, "
                f"weighted_tardiness={statistics.mean(row['weighted_tardiness'] for row in subset):.2f}, "
                f"drift={statistics.mean(row['episode_drift'] for row in subset):.4f}, "
                f"solve_ms={statistics.mean(row['avg_solve_time_ms'] for row in subset):.1f}"
            )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
