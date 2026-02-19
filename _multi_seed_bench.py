"""Multi-seed benchmark: TRCG (anchor) vs Fixed (full unlock)."""
import sys, time, logging
logging.disable(logging.CRITICAL)

from scenario import generate_scenario
from config import DEFAULT_CONFIG
from simulator import simulate_episode
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
from policies.policy_fixed import FixedWeightPolicy

seeds = [42, 55, 100, 101, 102, 103, 110, 123]
results = []

for seed in seeds:
    # TRCG (anchors enabled)
    s1 = generate_scenario(seed=seed)
    p1 = TRCGRepairPolicy(policy_name=f't_{seed}')
    r1 = simulate_episode(p1, s1, DEFAULT_CONFIG, verbose=False)
    
    # Fixed (no anchors)
    s2 = generate_scenario(seed=seed)
    p2 = FixedWeightPolicy(w_delay=10.0, w_shift=1.0, w_switch=5.0)
    r2 = simulate_episode(p2, s2, DEFAULT_CONFIG, verbose=False)
    
    steps1 = len(r1.snapshots) or 1
    steps2 = len(r2.snapshots) or 1
    dpr1 = r1.metrics.episode_drift / steps1
    dpr2 = r2.metrics.episode_drift / steps2
    
    results.append({
        'seed': seed,
        'trcg_dpr': dpr1,
        'fixed_dpr': dpr2,
        'trcg_drift': r1.metrics.episode_drift,
        'fixed_drift': r2.metrics.episode_drift,
        'steps': steps1,
    })
    print(f"seed={seed}: TRCG_dpr={dpr1:.2f} Fixed_dpr={dpr2:.2f} ratio={dpr1/dpr2:.3f}")

# Summary
avg_trcg = sum(r['trcg_dpr'] for r in results) / len(results)
avg_fixed = sum(r['fixed_dpr'] for r in results) / len(results)
print(f"\nAverage: TRCG={avg_trcg:.2f} Fixed={avg_fixed:.2f}")
print(f"Average drift reduction: {(1 - avg_trcg/avg_fixed)*100:.1f}%")
print(f"TRCG wins: {sum(1 for r in results if r['trcg_dpr'] < r['fixed_dpr'])}/{len(results)}")
