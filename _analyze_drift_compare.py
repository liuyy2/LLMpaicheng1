import pandas as pd

df = pd.read_csv('results_V2.5/BL/218_7/results_per_episode.csv')

print('=== Drift per Replan by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    dpr = sub['drift_per_replan'].dropna()
    print(f'  {pol}: mean={dpr.mean():.2f}, median={dpr.median():.2f}, count={len(sub)}')

print('\n=== Avg Delay by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["avg_delay"].mean():.2f}')

print('\n=== Episode Drift by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["episode_drift"].mean():.2f}')

print('\n=== avg_frozen by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["avg_frozen"].mean():.2f}')

print('\n=== avg_time_deviation_min by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["avg_time_deviation_min"].mean():.2f}')

print('\n=== total_shifts by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["total_shifts"].mean():.2f}')

print('\n=== total_switches by Policy ===')
for pol in df['policy_name'].unique():
    sub = df[df['policy_name']==pol]
    print(f'  {pol}: mean={sub["total_switches"].mean():.2f}')

print('\n=== By disturbance_level ===')
for lvl in ['light', 'medium', 'heavy']:
    print(f'\n--- {lvl} ---')
    sub = df[df['disturbance_level']==lvl]
    for pol in sub['policy_name'].unique():
        s = sub[sub['policy_name']==pol]
        dpr = s['drift_per_replan'].dropna()
        print(f'  {pol}: drift_per_replan={dpr.mean():.2f}, avg_delay={s["avg_delay"].mean():.2f}, episode_drift={s["episode_drift"].mean():.2f}, avg_frozen={s["avg_frozen"].mean():.2f}, shifts={s["total_shifts"].mean():.1f}, switches={s["total_switches"].mean():.1f}')

# Also test-only comparison
print('\n=== TEST ONLY ===')
test = df[df['dataset']=='test']
for pol in test['policy_name'].unique():
    sub = test[test['policy_name']==pol]
    dpr = sub['drift_per_replan'].dropna()
    print(f'  {pol}: drift_per_replan={dpr.mean():.2f}, avg_delay={sub["avg_delay"].mean():.2f}, episode_drift={sub["episode_drift"].mean():.2f}')
