"""
Multi-seed benchmark: TRCG selective freezing vs fixed_tuned.
Tests across multiple seeds to validate robustness.
"""
import sys
sys.path.insert(0, '.')

from config import make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies.policy_fixed import FixedWeightPolicy
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
from metrics import metrics_to_dict

test_cases = [
    (80, "medium"),
]

results = []

for seed, level in test_cases:
    config = make_config_for_difficulty(level)
    config.solver_timeout_s = 15.0
    
    print(f"\n{'='*60}")
    print(f"Seed={seed}, Level={level}")
    print(f"{'='*60}")
    
    # Fixed tuned baseline
    scenario_ft = generate_scenario(seed, config)
    ft_policy = FixedWeightPolicy(
        w_delay=10.0, w_shift=1.0, w_switch=5.0,
        freeze_horizon=96, use_two_stage=True, epsilon_solver=0.10,
    )
    ft_result = simulate_episode(ft_policy, scenario_ft, config, verbose=False)
    ft_m = metrics_to_dict(ft_result.metrics)
    
    # TRCG selective freezing
    scenario_trcg = generate_scenario(seed, config)
    trcg_policy = TRCGRepairPolicy(
        llm_client=None, policy_name="trcg_repair_llm",
        log_dir="test_output_temp/logs", enable_logging=False,
        episode_id=f"{seed}_{level}",
        w_delay=10.0, w_shift=1.0, w_switch=5.0,
    )
    trcg_result = simulate_episode(trcg_policy, scenario_trcg, config, verbose=False)
    trcg_m = metrics_to_dict(trcg_result.metrics)
    
    ft_dpr = ft_m['drift_per_replan']
    trcg_dpr = trcg_m['drift_per_replan']
    improvement = (ft_dpr - trcg_dpr) / ft_dpr * 100 if ft_dpr > 0 else 0
    delay_chg = trcg_m['avg_delay'] - ft_m['avg_delay']
    
    print(f"  fixed_tuned: dpr={ft_dpr:.2f} delay={ft_m['avg_delay']:.2f} win={ft_m['total_window_switches']} seq={ft_m['total_sequence_switches']}")
    print(f"  trcg_repair: dpr={trcg_dpr:.2f} delay={trcg_m['avg_delay']:.2f} win={trcg_m['total_window_switches']} seq={trcg_m['total_sequence_switches']}")
    print(f"  Drift reduction: {improvement:+.1f}%, Delay change: {delay_chg:+.2f}")
    
    results.append({
        'seed': seed, 'level': level,
        'ft_dpr': ft_dpr, 'trcg_dpr': trcg_dpr,
        'improvement': improvement,
        'ft_delay': ft_m['avg_delay'], 'trcg_delay': trcg_m['avg_delay'],
        'delay_chg': delay_chg,
    })

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
avg_improvement = sum(r['improvement'] for r in results) / len(results)
avg_delay_chg = sum(r['delay_chg'] for r in results) / len(results)
print(f"Average drift reduction: {avg_improvement:+.1f}%")
print(f"Average delay change: {avg_delay_chg:+.2f} slots")
for r in results:
    print(f"  seed={r['seed']:>3} {r['level']:>6}: drift {r['improvement']:+.1f}%, delay {r['delay_chg']:+.2f}")
print("\nDone!")
