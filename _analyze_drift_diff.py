"""Analyze drift differences between LLM 2.18_3 and fixed_tuned baseline."""
import pandas as pd
import numpy as np

llm = pd.read_csv('results_V2.5/LLM/2.18_3/results_per_episode.csv')
bl = pd.read_csv('results_V2.5/baseline/baseline_216_1/results_per_episode.csv')
ft = bl[bl['policy_name'] == 'fixed_tuned']

# Seed-matched comparison
common_seeds = set(llm['seed'].unique()) & set(ft['seed'].unique())
print(f"Common seeds: {len(common_seeds)}")
print(f"LLM seeds: {len(llm['seed'].unique())}, FT seeds: {len(ft['seed'].unique())}")

llm_m = llm[llm['seed'].isin(common_seeds)].set_index('seed')
ft_m = ft[ft['seed'].isin(common_seeds)].set_index('seed')

# Decompose drift difference
for col in ['episode_drift', 'drift_per_replan', 'avg_delay', 'total_sequence_switches',
            'total_window_switches', 'total_shifts', 'avg_time_deviation_min', 'num_replans']:
    diff = llm_m[col] - ft_m[col]
    lm = llm_m[col].mean()
    fm = ft_m[col].mean()
    pct = (lm - fm) / abs(fm) * 100 if fm != 0 else 0
    print(f"  {col:35s}  LLM={lm:10.2f} FT={fm:10.2f} diff={pct:+.1f}%  (seed_diff_mean={diff.mean():+.2f} std={diff.std():.2f})")

print()
print("=== Drift Decomposition (estimated) ===")
seq_sw_llm = llm_m['total_sequence_switches'].mean()
seq_sw_ft = ft_m['total_sequence_switches'].mean()
win_sw_llm = llm_m['total_window_switches'].mean()
win_sw_ft = ft_m['total_window_switches'].mean()

kappa_seq = 6.0
kappa_win = 12.0
switch_drift_diff = (seq_sw_llm - seq_sw_ft) * kappa_seq + (win_sw_llm - win_sw_ft) * kappa_win
total_drift_diff = llm_m['episode_drift'].mean() - ft_m['episode_drift'].mean()

print(f"  Extra drift from seq_switches:  ({seq_sw_llm:.1f} - {seq_sw_ft:.1f}) x 6.0 = {(seq_sw_llm-seq_sw_ft)*6:.1f}")
print(f"  Extra drift from win_switches:  ({win_sw_llm:.1f} - {win_sw_ft:.1f}) x 12.0 = {(win_sw_llm-win_sw_ft)*12:.1f}")
print(f"  Total switch-related drift diff: {switch_drift_diff:.1f}")
print(f"  Total drift difference:          {total_drift_diff:.1f}")
print(f"  Remaining (from time shifts):    {total_drift_diff - switch_drift_diff:.1f}")
print(f"  Switch % of total diff:          {switch_drift_diff/total_drift_diff*100:.1f}%")

# Per-seed: worst seeds
diff_drift = (llm_m['drift_per_replan'] - ft_m['drift_per_replan']).sort_values(ascending=False)
print()
print("=== Top 10 worst seeds (LLM drift - FT drift) ===")
for s in diff_drift.head(10).index:
    lvl = llm_m.loc[s, 'disturbance_level']
    ld = llm_m.loc[s, 'drift_per_replan']
    fd = ft_m.loc[s, 'drift_per_replan']
    lsw = llm_m.loc[s, 'total_sequence_switches']
    fsw = ft_m.loc[s, 'total_sequence_switches']
    lsh = llm_m.loc[s, 'total_shifts']
    fsh = ft_m.loc[s, 'total_shifts']
    ld_delay = llm_m.loc[s, 'avg_delay']
    fd_delay = ft_m.loc[s, 'avg_delay']
    print(f"  seed={s:3d} level={lvl:6s} "
          f"drift LLM={ld:6.1f} FT={fd:6.1f} diff={ld-fd:+6.1f}  "
          f"seq_sw LLM={lsw:3.0f}/FT={fsw:3.0f}  "
          f"shifts LLM={lsh:3.0f}/FT={fsh:3.0f}  "
          f"delay LLM={ld_delay:.1f}/FT={fd_delay:.1f}")

# Check: how many seeds LLM wins vs loses
llm_wins = (diff_drift < 0).sum()
ft_wins = (diff_drift > 0).sum()
ties = (diff_drift == 0).sum()
print(f"\n=== Win/Lose (seed-matched) ===")
print(f"  LLM wins (lower drift): {llm_wins}")
print(f"  FT wins (lower drift):  {ft_wins}")
print(f"  Ties:                    {ties}")

# Check avg_delay for seeds where LLM wins on drift
llm_wins_seeds = diff_drift[diff_drift < 0].index
ft_wins_seeds = diff_drift[diff_drift > 0].index
if len(llm_wins_seeds) > 0:
    print(f"\n  Where LLM wins on drift: avg_delay LLM={llm_m.loc[llm_wins_seeds, 'avg_delay'].mean():.1f}, FT={ft_m.loc[llm_wins_seeds, 'avg_delay'].mean():.1f}")
if len(ft_wins_seeds) > 0:
    print(f"  Where FT wins on drift:  avg_delay LLM={llm_m.loc[ft_wins_seeds, 'avg_delay'].mean():.1f}, FT={ft_m.loc[ft_wins_seeds, 'avg_delay'].mean():.1f}")
