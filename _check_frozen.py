"""Check frozen ops and FT tuning results."""
import pandas as pd

llm = pd.read_csv('results_V2.5/LLM/2.18_3/results_per_episode.csv')
bl = pd.read_csv('results_V2.5/baseline/baseline_216_1/results_per_episode.csv')
ft = bl[bl['policy_name'] == 'fixed_tuned']
common_seeds = set(llm['seed'].unique()) & set(ft['seed'].unique())
llm_m = llm[llm['seed'].isin(common_seeds)].set_index('seed')
ft_m = ft[ft['seed'].isin(common_seeds)].set_index('seed')

# Check frozen and forced_replan
print("=== Frozen and Forced Replan ===")
for col in ['avg_frozen', 'avg_num_tasks_scheduled', 'forced_replan_rate', 'num_forced_replans', 'num_replans']:
    lm = llm_m[col].mean()
    fm = ft_m[col].mean()
    print(f"  {col:35s}  LLM={lm:10.2f}  FT={fm:10.2f}")

# Check FT's tuning_results
print()
tuning = pd.read_csv('results_V2.5/baseline/baseline_216_1/tuning_results.csv')
print("=== Tuning Results Columns ===")
print(tuning.columns.tolist()[:20])
print()

ftc = [c for c in tuning.columns if 'freeze' in c.lower() or 'epsilon' in c.lower() or 'drift' in c.lower() or 'delay' in c.lower()]
print(f"Relevant columns: {ftc}")
print()

if 'freeze_horizon' in tuning.columns:
    print("=== FT tuning: freeze x epsilon x drift x delay ===")
    for _, row in tuning.iterrows():
        fh = row.get('freeze_horizon', '?')
        es = row.get('epsilon_solver', '?')
        dpr = row.get('drift_per_replan', '?')
        ad = row.get('avg_delay', '?')
        print(f"  freeze={fh:>5}  eps={es:>6}  drift_per_replan={dpr:>8}  avg_delay={ad:>8}")
else:
    print("No freeze_horizon column in tuning_results. Showing head:")
    print(tuning.head(10).to_string())
