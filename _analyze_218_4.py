"""Analyze 218_4 results: seed-matched LLM vs FT comparison."""
import csv, statistics

rows = list(csv.DictReader(open(r'results_V2.5\BL\218_4\results_per_episode.csv')))

llm = {int(r['seed']): r for r in rows if r['policy_name'] == 'trcg_repair_llm'}
ft = {int(r['seed']): r for r in rows if r['policy_name'] == 'fixed_tuned'}
common = sorted(set(llm) & set(ft))
print(f'Seed-matched episodes: {len(common)}')

llm_wins = 0
ft_wins = 0
diffs = []
levels = {}
worst_seeds = []  # LLM loses most

for s in common:
    ld = float(llm[s]['drift_per_replan'])
    fd = float(ft[s]['drift_per_replan'])
    diff = ld - fd
    diffs.append(diff)
    if ld < fd:
        llm_wins += 1
    else:
        ft_wins += 1
    worst_seeds.append((diff, s, llm[s]['disturbance_level'], ld, fd))
    
    lev = llm[s]['disturbance_level']
    if lev not in levels:
        levels[lev] = {'llm_d': [], 'ft_d': [], 'llm_w': 0, 'ft_w': 0,
                       'llm_shifts': [], 'ft_shifts': [], 'llm_sw': [], 'ft_sw': []}
    levels[lev]['llm_d'].append(ld)
    levels[lev]['ft_d'].append(fd)
    levels[lev]['llm_shifts'].append(int(llm[s]['total_shifts']))
    levels[lev]['ft_shifts'].append(int(ft[s]['total_shifts']))
    levels[lev]['llm_sw'].append(int(llm[s]['total_switches']))
    levels[lev]['ft_sw'].append(int(ft[s]['total_switches']))
    if ld < fd:
        levels[lev]['llm_w'] += 1
    else:
        levels[lev]['ft_w'] += 1

print(f'LLM wins: {llm_wins}/{len(common)} ({llm_wins/len(common)*100:.0f}%)')
print(f'FT wins:  {ft_wins}/{len(common)} ({ft_wins/len(common)*100:.0f}%)')
print(f'Avg diff (LLM-FT): {statistics.mean(diffs):.4f}')

print('\n=== Per-level ===')
for lev in ['light', 'medium', 'heavy']:
    if lev not in levels:
        continue
    l = levels[lev]
    n = len(l['llm_d'])
    lm = statistics.mean(l['llm_d'])
    fm = statistics.mean(l['ft_d'])
    print(f'{lev:7s} (n={n:2d}): LLM={lm:.3f}, FT={fm:.3f}, diff={lm-fm:+.3f}, LLM_win={l["llm_w"]}/{n}')
    print(f'          shifts: LLM={statistics.mean(l["llm_shifts"]):.1f}, FT={statistics.mean(l["ft_shifts"]):.1f}')
    print(f'          sw:     LLM={statistics.mean(l["llm_sw"]):.1f}, FT={statistics.mean(l["ft_sw"]):.1f}')

# Top 10 seeds where LLM loses most
worst_seeds.sort(reverse=True)
print('\n=== Top 10 seeds where LLM loses most ===')
for diff, s, lev, ld, fd in worst_seeds[:10]:
    llm_shifts = int(llm[s]['total_shifts'])
    ft_shifts = int(ft[s]['total_shifts'])
    llm_sw = int(llm[s]['total_switches'])
    ft_sw = int(ft[s]['total_switches'])
    print(f'  seed={s:3d} [{lev:6s}] LLM={ld:.2f} FT={fd:.2f} diff={diff:+.2f} | shifts: {llm_shifts}/{ft_shifts}, sw: {llm_sw}/{ft_sw}')

# Top 10 seeds where LLM wins most
worst_seeds.sort()
print('\n=== Top 10 seeds where LLM wins most ===')
for diff, s, lev, ld, fd in worst_seeds[:10]:
    print(f'  seed={s:3d} [{lev:6s}] LLM={ld:.2f} FT={fd:.2f} diff={diff:+.2f}')

# Drift decomposition: switches vs time shifts
print('\n=== Drift decomposition (avg) ===')
for p_name, p_data in [('LLM', llm), ('FT', ft)]:
    sds = [p_data[s] for s in common]
    avg_ws = statistics.mean([int(r['total_window_switches']) for r in sds])
    avg_ss = statistics.mean([int(r['total_sequence_switches']) for r in sds])
    avg_shifts = statistics.mean([int(r['total_shifts']) for r in sds])
    avg_td = statistics.mean([float(r['avg_time_deviation_min']) for r in sds])
    print(f'{p_name}: shifts={avg_shifts:.1f}, win_sw={avg_ws:.1f}, seq_sw={avg_ss:.1f}, time_dev={avg_td:.1f}min')
