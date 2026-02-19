#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resume batch runner: skip already-completed episodes, run rest, write summary.
"""
import sys, os, json, csv, time, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode, save_episode_logs
from policies.policy_fixed import FixedWeightPolicy

BASELINES = {
    "static":      {"freeze_horizon": 9999, "use_two_stage": False, "epsilon_solver": None},
    "full_unlock": {"freeze_horizon": 0,    "use_two_stage": True,  "epsilon_solver": 0.05},
    "fixed_tuned": {"freeze_horizon": 12,   "use_two_stage": True,  "epsilon_solver": 0.05},
}
DIFFICULTIES = ["light", "medium", "heavy"]
SEEDS = [0, 1, 2]
OUTPUT_DIR = "results/batch_10day"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records = []
    total = len(DIFFICULTIES) * len(BASELINES) * len(SEEDS)
    idx = 0

    for diff in DIFFICULTIES:
        for pname, params in BASELINES.items():
            for seed in SEEDS:
                idx += 1
                ep_dir = os.path.join(OUTPUT_DIR, f"{diff}_{pname}_seed{seed}")
                summary_path = os.path.join(ep_dir, "episode_summary.json")

                # Skip if already done
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path) as f:
                            d = json.load(f)
                        if "metrics" in d:
                            m = d["metrics"]
                            all_records.append({
                                "seed": seed, "difficulty": diff, "policy": pname,
                                "num_missions": m.get("num_total", d.get("num_missions", 0)),
                                "sim_total_slots": 960,
                                "on_time_rate": m["on_time_rate"],
                                "avg_delay": m["avg_delay"],
                                "max_delay": m["max_delay"],
                                "episode_drift": m["episode_drift"],
                                "drift_per_replan": m["drift_per_replan"],
                                "drift_per_day": m["drift_per_day"],
                                "total_shifts": m["total_shifts"],
                                "total_switches": m["total_switches"],
                                "num_replans": m["num_replans"],
                                "num_forced_replans": m["num_forced_replans"],
                                "completion_rate": m["completion_rate"],
                                "feasible_rate": m["feasible_rate"],
                                "avg_solve_time_ms": m["avg_solve_time_ms"],
                                "total_runtime_s": d.get("total_runtime_s", 0),
                            })
                            print(f"[{idx}/{total}] SKIP {diff}/{pname}/seed{seed} (cached)")
                            continue
                    except Exception:
                        pass

                print(f"[{idx}/{total}] RUN {diff}/{pname}/seed{seed} ...", end=" ", flush=True)
                t0 = time.time()
                config = make_config_for_difficulty(diff)
                scenario = generate_scenario(seed=seed, config=config)
                policy = FixedWeightPolicy(
                    w_delay=10.0,
                    w_shift=1.0 if pname != "static" else 0.0,
                    w_switch=5.0 if pname != "static" else 0.0,
                    freeze_horizon=params["freeze_horizon"],
                    policy_name=pname,
                    use_two_stage=params["use_two_stage"],
                    epsilon_solver=params["epsilon_solver"],
                )
                result = simulate_episode(policy=policy, scenario=scenario, config=config, verbose=False)
                save_episode_logs(result, ep_dir, scenario)
                r = result.metrics
                rec = {
                    "seed": seed, "difficulty": diff, "policy": pname,
                    "num_missions": len(scenario.missions),
                    "sim_total_slots": config.sim_total_slots,
                    "on_time_rate": round(r.on_time_rate, 4),
                    "avg_delay": round(r.avg_delay, 2),
                    "max_delay": r.max_delay,
                    "episode_drift": round(r.episode_drift, 4),
                    "drift_per_replan": round(r.drift_per_replan, 6),
                    "drift_per_day": round(r.drift_per_day, 6),
                    "total_shifts": r.total_shifts,
                    "total_switches": r.total_switches,
                    "num_replans": r.num_replans,
                    "num_forced_replans": r.num_forced_replans,
                    "completion_rate": round(r.completion_rate, 4),
                    "feasible_rate": round(r.feasible_rate, 4),
                    "avg_solve_time_ms": round(r.avg_solve_time_ms, 2),
                    "total_runtime_s": round(result.total_runtime_s, 3),
                }
                elapsed = time.time() - t0
                otr = rec["on_time_rate"]
                dr = rec["episode_drift"]
                print(f"done {elapsed:.1f}s, on_time={otr}, drift={dr}")
                all_records.append(rec)

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "batch_summary.csv")
    fieldnames = list(all_records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_records)
    print(f"\nCSV: {csv_path}")

    # Write JSON
    json_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")

    # Aggregated
    print()
    print("=" * 80)
    print("  Aggregated Results (mean +/- std)")
    print("=" * 80)
    for diff in DIFFICULTIES:
        for pname in BASELINES:
            subset = [r for r in all_records if r["difficulty"] == diff and r["policy"] == pname]
            if not subset:
                continue
            n = len(subset)

            def mean_of(k):
                return statistics.mean([r[k] for r in subset])

            def std_of(k):
                return statistics.stdev([r[k] for r in subset]) if n > 1 else 0.0

            print(
                f"  {diff:8s}/{pname:12s}: "
                f"on_time={mean_of('on_time_rate'):.3f}+/-{std_of('on_time_rate'):.3f}  "
                f"drift={mean_of('episode_drift'):.4f}+/-{std_of('episode_drift'):.4f}  "
                f"drift/replan={mean_of('drift_per_replan'):.6f}  "
                f"drift/day={mean_of('drift_per_day'):.6f}  "
                f"complete={mean_of('completion_rate'):.3f}  "
                f"time={mean_of('total_runtime_s'):.1f}s"
            )
    print()
    print(f"Total: {len(all_records)} episodes completed")


if __name__ == "__main__":
    main()
