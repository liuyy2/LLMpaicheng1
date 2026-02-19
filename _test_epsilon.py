#!/usr/bin/env python3
"""Test: adaptive epsilon ONLY (no anchors) to isolate epsilon's contribution."""
import sys
import time
import statistics

from config import Config, make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies import FixedWeightPolicy

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700]
DIFFICULTY = "medium"


def run_one(seed, policy, config):
    scenario = generate_scenario(seed=seed, config=config)
    result = simulate_episode(
        scenario=scenario,
        policy=policy,
        config=config,
        verbose=False,
    )
    dpr = result.metrics.drift_per_replan
    return dpr


def main():
    config = make_config_for_difficulty(DIFFICULTY)

    ft010_dprs = []
    ft025_dprs = []

    for seed in SEEDS:
        # FT with epsilon=0.10 (standard)
        ft010 = FixedWeightPolicy(
            w_delay=1.0, w_shift=0.0, w_switch=0.0,
            freeze_horizon=96,
            use_two_stage=True,
            policy_name="ft_eps010",
            epsilon_solver=0.10,
        )
        dpr010 = run_one(seed, ft010, config)

        # FT with epsilon=0.25 (higher, same as TRCG skip path)
        ft025 = FixedWeightPolicy(
            w_delay=1.0, w_shift=0.0, w_switch=0.0,
            freeze_horizon=96,
            use_two_stage=True,
            policy_name="ft_eps025",
            epsilon_solver=0.25,
        )
        dpr025 = run_one(seed, ft025, config)

        ratio = dpr025 / dpr010 if dpr010 > 0 else float('inf')
        sys.stdout.write(f"seed={seed:4d}  eps010={dpr010:8.2f}  eps025={dpr025:8.2f}  ratio={ratio:.3f}  {'WIN' if ratio < 1 else 'LOSE'}\n")
        sys.stdout.flush()
        ft010_dprs.append(dpr010)
        ft025_dprs.append(dpr025)

    avg010 = statistics.mean(ft010_dprs)
    avg025 = statistics.mean(ft025_dprs)
    ratio_avg = avg025 / avg010 if avg010 > 0 else 0
    wins = sum(1 for a, b in zip(ft025_dprs, ft010_dprs) if a < b)
    sys.stdout.write(f"\n--- Epsilon Comparison ({DIFFICULTY}) ---\n")
    sys.stdout.write(f"FT(eps=0.10) avg dpr = {avg010:.2f}\n")
    sys.stdout.write(f"FT(eps=0.25) avg dpr = {avg025:.2f}\n")
    sys.stdout.write(f"Ratio = {ratio_avg:.3f}\n")
    sys.stdout.write(f"Win rate = {wins}/{len(SEEDS)}\n")
    sys.stdout.write(f"Improvement = {100*(1-ratio_avg):.1f}%\n")
    sys.stdout.flush()


if __name__ == "__main__":
    t0 = time.time()
    main()
    sys.stdout.write(f"\nTotal time: {time.time()-t0:.1f}s\n")
    sys.stdout.flush()
