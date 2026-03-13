#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from config import Config, DEFAULT_CONFIG, SCENARIO_PROFILES, make_config_for_difficulty
from scenario import generate_scenario
from simulator import save_episode_logs, simulate_episode
from policies import create_policy


@dataclass
class PolicyComparisonResult:
    policy_name: str
    completed: int
    total: int
    on_time_rate: float
    avg_delay: float
    weighted_tardiness: float
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
            "weighted_tardiness": round(self.weighted_tardiness, 2),
            "max_delay": self.max_delay,
            "episode_drift": round(self.episode_drift, 4),
            "total_shifts": self.total_shifts,
            "total_switches": self.total_switches,
            "num_replans": self.num_replans,
            "num_forced_replans": self.num_forced_replans,
            "avg_solve_time_ms": round(self.avg_solve_time_ms, 1),
            "total_runtime_s": round(self.total_runtime_s, 2),
        }


def build_policy(name: str, output_dir: str, episode_id: str):
    kwargs: Dict[str, Any] = {}
    if name in ("trcg_repair", "ga_repair", "alns_repair", "mockllm"):
        kwargs["log_dir"] = os.path.join(output_dir, "logs", episode_id)
        kwargs["episode_id"] = episode_id
        kwargs["enable_logging"] = True
        os.makedirs(kwargs["log_dir"], exist_ok=True)
    return create_policy(name, **kwargs)


def run_policy_comparison(
    seed: int,
    config: Config = DEFAULT_CONFIG,
    policy_names: List[str] = None,
    verbose: bool = False,
    output_dir: str = None,
) -> List[PolicyComparisonResult]:
    scenario = generate_scenario(seed=seed, config=config)
    print("")
    print("Scenario generated:")
    print(f"  Missions: {len(scenario.missions)}")
    print(f"  Resources: {len(scenario.resources)}")
    print(f"  Disturbance events: {len(scenario.disturbance_timeline)}")

    if not policy_names:
        policy_names = ["full_unlock", "ga_repair", "alns_repair"]

    results: List[PolicyComparisonResult] = []
    for policy_name in policy_names:
        episode_id = f"compare_seed{seed}_{policy_name}"
        policy = build_policy(policy_name, output_dir or "logs", episode_id)
        print(f"\n{'=' * 50}")
        print(f"Running policy: {policy.name}")
        print(f"{'=' * 50}")
        start_time = time.time()
        try:
            episode_result = simulate_episode(
                policy=policy,
                scenario=scenario,
                config=config,
                verbose=verbose,
            )
            if output_dir:
                policy_dir = os.path.join(output_dir, episode_id)
                save_episode_logs(episode_result, policy_dir, scenario)

            m = episode_result.metrics
            item = PolicyComparisonResult(
                policy_name=policy.name,
                completed=m.num_completed,
                total=m.num_total,
                on_time_rate=m.on_time_rate,
                avg_delay=m.avg_delay,
                weighted_tardiness=m.weighted_tardiness,
                max_delay=m.max_delay,
                episode_drift=m.episode_drift,
                total_shifts=m.total_shifts,
                total_switches=m.total_switches,
                num_replans=m.num_replans,
                num_forced_replans=m.num_forced_replans,
                avg_solve_time_ms=m.avg_solve_time_ms,
                total_runtime_s=episode_result.total_runtime_s,
            )
            results.append(item)
            print(
                f"  completed={m.num_completed}/{m.num_total} "
                f"on_time={m.on_time_rate:.2%} avg_delay={m.avg_delay:.2f} "
                f"drift={m.episode_drift:.4f} runtime={episode_result.total_runtime_s:.2f}s"
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            results.append(
                PolicyComparisonResult(
                    policy_name=policy.name,
                    completed=0,
                    total=len(scenario.missions),
                    on_time_rate=0.0,
                    avg_delay=float("inf"),
                    weighted_tardiness=float("inf"),
                    max_delay=0,
                    episode_drift=0.0,
                    total_shifts=0,
                    total_switches=0,
                    num_replans=0,
                    num_forced_replans=0,
                    avg_solve_time_ms=0.0,
                    total_runtime_s=time.time() - start_time,
                )
            )
    return results


def print_comparison_table(results: List[PolicyComparisonResult]) -> None:
    print("\n" + "=" * 88)
    print("Policy Comparison")
    print("=" * 88)
    header = (
        f"{'policy':<14} {'done':<8} {'on_time':<10} {'avg_delay':<10} "
        f"{'wtard':<10} {'drift':<10} {'solve_ms':<10} {'runtime_s':<10}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.policy_name:<14} {f'{item.completed}/{item.total}':<8} "
            f"{item.on_time_rate:<10.2%} {item.avg_delay:<10.2f} "
            f"{item.weighted_tardiness:<10.2f} {item.episode_drift:<10.4f} "
            f"{item.avg_solve_time_ms:<10.1f} {item.total_runtime_s:<10.2f}"
        )


def save_comparison_results(results: List[PolicyComparisonResult], filepath: str, seed: int) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "seed": seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [item.to_dict() for item in results],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved comparison JSON: {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scenario with multiple policies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str, default="logs/compare_once")
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--difficulty", type=str, default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--num-missions", type=int, default=None)
    parser.add_argument("--scenario-profile", type=str, default="default", choices=sorted(SCENARIO_PROFILES.keys()))
    parser.add_argument("--policies", type=str, default="full_unlock,ga_repair,alns_repair")
    args = parser.parse_args()

    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    config = make_config_for_difficulty(
        difficulty=args.difficulty,
        num_missions_override=args.num_missions,
        scenario_profile=args.scenario_profile,
    )
    results = run_policy_comparison(
        seed=args.seed,
        config=config,
        policy_names=policy_names,
        verbose=args.verbose,
        output_dir=args.output,
    )
    print_comparison_table(results)
    save_path = args.save_json or os.path.join(args.output, "comparison.json")
    save_comparison_results(results, save_path, args.seed)


if __name__ == "__main__":
    main()
