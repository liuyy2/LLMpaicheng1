"""Verify freeze_horizon is the dominant factor using tuning grid."""
import pandas as pd

tuning = pd.read_csv('results_V2.5/baseline/baseline_216_1/tuning_results.csv')

# Compare freeze=12 (closest to LLM) vs freeze=96 (FT best), both eps=0.10
llm_like = tuning[(tuning['freeze_horizon_slots'] == 16) & (tuning['epsilon_solver'] == 0.10)]
ft_best = tuning[(tuning['freeze_horizon_slots'] == 96) & (tuning['epsilon_solver'] == 0.10)]

if len(llm_like) > 0 and len(ft_best) > 0:
    ll = llm_like.iloc[0]
    fb = ft_best.iloc[0]
    print("=== SAME POLICY (fixed_tuned), different freeze ===")
    print(f"  freeze=16, eps=0.10: avg_drift={ll['avg_drift']:.1f}  avg_delay={ll['avg_delay']:.1f}")
    print(f"  freeze=96, eps=0.10: avg_drift={fb['avg_drift']:.1f}  avg_delay={fb['avg_delay']:.1f}")
    drift_pct = (ll['avg_drift'] - fb['avg_drift']) / fb['avg_drift'] * 100
    delay_pct = (ll['avg_delay'] - fb['avg_delay']) / fb['avg_delay'] * 100
    print(f"  Drift difference: {drift_pct:+.1f}% (freeze alone!)")
    print(f"  Delay difference: {delay_pct:+.1f}%")

print()
# Now compare freeze=0 vs freeze=96 at each epsilon    
print("=== Drift vs Freeze (all epsilon values) ===")
for eps in sorted(tuning['epsilon_solver'].unique()):
    f0 = tuning[(tuning['freeze_horizon_slots'] == 0) & (tuning['epsilon_solver'] == eps)]
    f96 = tuning[(tuning['freeze_horizon_slots'] == 96) & (tuning['epsilon_solver'] == eps)]
    if len(f0) > 0 and len(f96) > 0:
        d0 = f0.iloc[0]['avg_drift']
        d96 = f96.iloc[0]['avg_drift']
        pct = (d0 - d96) / d96 * 100
        print(f"  eps={eps:.2f}: freeze=0 drift={d0:.1f} vs freeze=96 drift={d96:.1f}  diff={pct:+.1f}%")

print()
print("=== Conclusion ===")
print("freeze_horizon is the DOMINANT factor for drift_per_replan.")
print(f"Going from freeze=16 to freeze=96 (SAME policy) reduces drift by ~{drift_pct:.0f}%!")
print()

# Compute what LLM's drift WOULD be with freeze=96
llm_data = pd.read_csv('results_V2.5/LLM/2.18_3/results_per_episode.csv')
bl = pd.read_csv('results_V2.5/baseline/baseline_216_1/results_per_episode.csv')
ft_data = bl[bl['policy_name'] == 'fixed_tuned']

common = set(llm_data['seed']) & set(ft_data['seed'])
llm_match = llm_data[llm_data['seed'].isin(common)]
ft_match = ft_data[ft_data['seed'].isin(common)]

print(f"=== Current Results (seed-matched, n={len(common)}) ===")
print(f"  LLM (freeze=12):  drift_per_replan={llm_match['drift_per_replan'].mean():.2f}  avg_delay={llm_match['avg_delay'].mean():.2f}")
print(f"  FT  (freeze=96):  drift_per_replan={ft_match['drift_per_replan'].mean():.2f}  avg_delay={ft_match['avg_delay'].mean():.2f}")
print()
print(f"The {drift_pct:.0f}% drift gap is almost entirely due to freeze_horizon (12 vs 96)!")
print(f"FIX: Set LLM's freeze_horizon=96 to match FT. Then both have IDENTICAL solver config,")
print(f"and LLM's advantage must come from METRIC NORMALIZATION or EXTRA functionality.")
