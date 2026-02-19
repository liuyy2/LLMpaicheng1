"""分析 trcg_repair 异常 episode"""
import csv

with open('results_V2.5/baseline/baseline_213/results_per_episode.csv', 'r') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

# 对比异常 seed 在所有策略中的表现
for seed in ['71', '66', '77']:
    rows = [r for r in all_rows if r['seed'] == seed]
    print(f'=== seed={seed} (disturbance={rows[0]["disturbance_level"]}) 所有策略对比 ===')
    for r in sorted(rows, key=lambda x: x['policy_name']):
        print(f'  {r["policy_name"]:15s} delay={float(r["avg_delay"]):7.3f} '
              f'otr={float(r["on_time_rate"]):.3f} '
              f'drift={float(r["drift_per_replan"]):.4f} '
              f'forced={r["num_forced_replans"]} '
              f'replans={r["num_replans"]} '
              f'makespan={r["makespan_cmax"]} '
              f'feasible={float(r["feasible_rate"]):.3f}')
    print()

# trcg_repair vs ga_repair 全局对比
print('=== trcg_repair vs ga_repair 全局对比 ===')
for dist in ['light', 'medium', 'heavy']:
    trcg = [r for r in all_rows if r['policy_name'] == 'trcg_repair' and r['disturbance_level'] == dist]
    ga = [r for r in all_rows if r['policy_name'] == 'ga_repair' and r['disturbance_level'] == dist]
    
    trcg_delay = sum(float(r['avg_delay']) for r in trcg) / len(trcg)
    ga_delay = sum(float(r['avg_delay']) for r in ga) / len(ga)
    trcg_otr = sum(float(r['on_time_rate']) for r in trcg) / len(trcg)
    ga_otr = sum(float(r['on_time_rate']) for r in ga) / len(ga)
    trcg_drift = sum(float(r['drift_per_replan']) for r in trcg) / len(trcg)
    ga_drift = sum(float(r['drift_per_replan']) for r in ga) / len(ga)
    trcg_forced = sum(int(r['num_forced_replans']) for r in trcg)
    ga_forced = sum(int(r['num_forced_replans']) for r in ga)
    
    print(f'{dist:8s} trcg: delay={trcg_delay:.3f} otr={trcg_otr:.3f} drift={trcg_drift:.4f} total_forced={trcg_forced}')
    print(f'         ga:   delay={ga_delay:.3f} otr={ga_otr:.3f} drift={ga_drift:.4f} total_forced={ga_forced}')
    print()

# 检查 trcg_repair 是否有 feasible_rate < 1 的 episode
print('=== trcg_repair feasible_rate < 1 的 episodes ===')
infeasible = [r for r in all_rows if r['policy_name'] == 'trcg_repair' and float(r['feasible_rate']) < 1.0]
if infeasible:
    for r in infeasible:
        print(f'  seed={r["seed"]} dist={r["disturbance_level"]} feasible={float(r["feasible_rate"]):.3f} delay={float(r["avg_delay"]):.3f}')
else:
    print('  无')

# 检查 trcg_repair 和 ga_repair 逐 episode 差异
print()
print('=== trcg_repair vs ga_repair 逐 episode 对比 (delay差值最大的 top10) ===')
diffs = []
for r in all_rows:
    if r['policy_name'] != 'trcg_repair':
        continue
    seed = r['seed']
    dist = r['disturbance_level']
    ga_row = [x for x in all_rows if x['policy_name'] == 'ga_repair' and x['seed'] == seed and x['disturbance_level'] == dist]
    if ga_row:
        g = ga_row[0]
        diff = float(r['avg_delay']) - float(g['avg_delay'])
        diffs.append((seed, dist, float(r['avg_delay']), float(g['avg_delay']), diff,
                       float(r['drift_per_replan']), float(g['drift_per_replan']),
                       int(r['num_forced_replans']), int(g['num_forced_replans'])))

diffs.sort(key=lambda x: abs(x[4]), reverse=True)
for s, d, td, gd, diff, tdr, gdr, tf, gf in diffs[:15]:
    marker = '<<< TRCG worse' if diff > 0.5 else ('>>> GA worse' if diff < -0.5 else '')
    print(f'  seed={s:3s} {d:7s} trcg_delay={td:6.3f} ga_delay={gd:6.3f} diff={diff:+7.3f} '
          f'trcg_drift={tdr:.4f} ga_drift={gdr:.4f} '
          f'trcg_forced={tf} ga_forced={gf} {marker}')
