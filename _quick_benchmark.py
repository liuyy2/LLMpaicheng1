#!/usr/bin/env python3
"""Quick local benchmark: trcg_repair (heuristic) vs fixed_tuned
Tests soft anchor + dual-solve changes.
"""
import sys
import time
import statistics

from config import Config, make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies import FixedWeightPolicy
from policies.policy_llm_trcg_repair import TRCGRepairPolicy

SEEDS = [42, 100, 200, 300, 400, 500, 600, 700]
DIFFICULTY = "medium"


def run_one(seed, policy, config, difficulty):
    scenario = generate_scenario(seed=seed, config=config)
    result = simulate_episode(
        scenario=scenario,
        policy=policy,
        config=config,
        verbose=False,
    )
    dpr = result.metrics.drift_per_replan
    return dpr, result.metrics.episode_drift, result.metrics.num_replans


def main():
    config = make_config_for_difficulty(DIFFICULTY)
    # V2.5 best_params
    best_params = {
        'freeze_horizon': 96,
        'w_delay': 1.0,
        'w_shift': 0.0,
        'w_switch': 0.0,
        'use_two_stage': True,
    }

    ft_dprs = []
    trcg_dprs = []

    for seed in SEEDS:
        # fixed_tuned
        ft_policy = FixedWeightPolicy(
            w_delay=best_params['w_delay'],
            w_shift=best_params['w_shift'],
            w_switch=best_params['w_switch'],
            freeze_horizon=best_params['freeze_horizon'],
            use_two_stage=best_params['use_two_stage'],
            policy_name="fixed_tuned",
        )
        ft_dpr, ft_drift, ft_replans = run_one(seed, ft_policy, config, DIFFICULTY)

        # trcg_repair (heuristic, uses same w_delay/w_shift/w_switch)
        trcg_policy = TRCGRepairPolicy(
            policy_name="trcg_repair",
            log_dir=None,
            enable_logging=False,
            episode_id=f"bench_s{seed}",
            llm_client=None,  # heuristic mode
        )
        trcg_dpr, trcg_drift, trcg_replans = run_one(seed, trcg_policy, config, DIFFICULTY)

        ratio = trcg_dpr / ft_dpr if ft_dpr > 0 else float('inf')
        sys.stdout.write(f"seed={seed:4d}  FT_dpr={ft_dpr:8.2f}  TRCG_dpr={trcg_dpr:8.2f}  ratio={ratio:.3f}  {'WIN' if ratio < 1 else 'LOSE'}\n")
        sys.stdout.flush()
        ft_dprs.append(ft_dpr)
        trcg_dprs.append(trcg_dpr)

    ft_avg = statistics.mean(ft_dprs)
    trcg_avg = statistics.mean(trcg_dprs)
    ratio_avg = trcg_avg / ft_avg if ft_avg > 0 else 0
    wins = sum(1 for t, f in zip(trcg_dprs, ft_dprs) if t < f)
    sys.stdout.write(f"\n--- Summary ({DIFFICULTY}) ---\n")
    sys.stdout.write(f"FT  avg dpr = {ft_avg:.2f}\n")
    sys.stdout.write(f"TRCG avg dpr = {trcg_avg:.2f}\n")
    sys.stdout.write(f"Ratio (TRCG/FT) = {ratio_avg:.3f}\n")
    sys.stdout.write(f"Win rate = {wins}/{len(SEEDS)} ({100*wins/len(SEEDS):.0f}%)\n")
    sys.stdout.write(f"Improvement = {100*(1-ratio_avg):.1f}%\n")
    sys.stdout.flush()


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    sys.stdout.write(f"\nTotal time: {elapsed:.1f}s\n")
    sys.stdout.flush()
