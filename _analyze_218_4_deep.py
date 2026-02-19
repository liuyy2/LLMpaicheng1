"""Deep analysis: what decisions cause LLM drift to exceed FT on losing seeds."""
import json, os, glob, csv, statistics

# Load rolling metrics per step for seed 116
llm_dir = r'results_V2.5\LLM\2.18_4\logs\episode_116_trcg_repair_llm'
ft_dir = r'results_V2.5\baseline\baseline_216_1\logs\episode_116_fixed_tuned'

def load_rolling(d):
    path = os.path.join(d, 'metrics_per_roll.csv')
    if not os.path.exists(path):
        path2 = os.path.join(d, 'rolling_log.jsonl')
        if os.path.exists(path2):
            entries = []
            for line in open(path2):
                e = json.loads(line)
                m = e.get('metrics', e)
                entries.append(m)
            return entries
        return None
    return list(csv.DictReader(open(path)))

def load_repair_steps(d):
    steps = {}
    for fp in glob.glob(os.path.join(d, 'repair_step_*.json')):
        with open(fp) as f:
            s = json.load(f)
        steps[s['now_slot']] = s
    return steps

# ---- Analyze all losing seeds ----
rows = list(csv.DictReader(open(r'results_V2.5\BL\218_4\results_per_episode.csv')))
llm_map = {int(r['seed']): r for r in rows if r['policy_name'] == 'trcg_repair_llm'}
ft_map = {int(r['seed']): r for r in rows if r['policy_name'] == 'fixed_tuned'}

# Find seeds where LLM's drift > FT's drift by > 1.0
losing_seeds = []
for s in sorted(set(llm_map) & set(ft_map)):
    ld = float(llm_map[s]['drift_per_replan'])
    fd = float(ft_map[s]['drift_per_replan'])
    if ld > fd + 0.5:
        losing_seeds.append((ld - fd, s))
losing_seeds.sort(reverse=True)

print(f"Seeds where LLM drift > FT by >0.5: {len(losing_seeds)}")
print()

# Analyze what differs on those seeds
for diff, seed in losing_seeds[:5]:
    print(f"=== seed={seed} diff=+{diff:.2f} ===")
    llm_log_dir = f'results_V2.5/LLM/2.18_4/logs/episode_{seed}_trcg_repair_llm'
    
    if not os.path.exists(llm_log_dir):
        print("  (no logs)")
        continue
    
    steps = load_repair_steps(llm_log_dir)
    
    # Count decision sources
    sources = {}
    unlock_sizes = []
    for t, s in sorted(steps.items()):
        src = s.get('decision_source', '?')
        sources[src] = sources.get(src, 0) + 1
        unlock = s.get('decision_json', {}).get('unlock_mission_ids', [])
        if unlock:
            unlock_sizes.append(len(unlock))
        
    print(f"  Decision sources: {sources}")
    if unlock_sizes:
        print(f"  Unlock sizes: avg={statistics.mean(unlock_sizes):.1f}, max={max(unlock_sizes)}")
    
    # Load rolling metrics
    rolling = load_rolling(llm_log_dir)
    if rolling:
        drifts = []
        for r in rolling:
            d = float(r.get('plan_drift', 0))
            if d > 0:
                drifts.append(d)
        if drifts:
            print(f"  Non-zero drift steps: {len(drifts)}/{len(rolling)}")
            print(f"  Max drift step: {max(drifts):.2f}")
            top5 = sorted(drifts, reverse=True)[:5]
            print(f"  Top-5 drift steps: {[round(d,2) for d in top5]}")
    
    # Find forced replans
    forced = [s for t, s in sorted(steps.items()) 
              if s.get('solver_status') not in ('OPTIMAL', 'FEASIBLE', None)]
    if forced:
        print(f"  Non-optimal solves: {len(forced)}")
    
    # Check if LLM produced different plan from what FT would
    # Look at fallback count
    fallbacks = sum(1 for s in steps.values() if s.get('fallback_reason'))
    if fallbacks:
        print(f"  Fallback count: {fallbacks}")
    print()

# Overall: what fraction of LLM steps are skipped?
print("=== Decision source distribution (all 60 episodes) ===")
all_sources = {}
all_unlock_sizes = []
all_forced_replans = 0
total_steps = 0

for seed in sorted(llm_map.keys()):
    llm_log_dir = f'results_V2.5/LLM/2.18_4/logs/episode_{seed}_trcg_repair_llm'
    if not os.path.exists(llm_log_dir):
        continue
    steps = load_repair_steps(llm_log_dir)
    total_steps += len(steps)
    for t, s in steps.items():
        src = s.get('decision_source', '?')
        all_sources[src] = all_sources.get(src, 0) + 1
        unlock = s.get('decision_json', {}).get('unlock_mission_ids', [])
        if unlock:
            all_unlock_sizes.append(len(unlock))

for src, cnt in sorted(all_sources.items(), key=lambda x: -x[1]):
    print(f"  {src:25s}: {cnt:5d} ({cnt/total_steps*100:.1f}%)")
print(f"  Total: {total_steps}")
if all_unlock_sizes:
    print(f"\nUnlock sizes: avg={statistics.mean(all_unlock_sizes):.2f}, "
          f"median={statistics.median(all_unlock_sizes):.0f}, "
          f"max={max(all_unlock_sizes)}, "
          f"dist: 1={all_unlock_sizes.count(1)}, 2={sum(1 for x in all_unlock_sizes if x==2)}, "
          f"3+={sum(1 for x in all_unlock_sizes if x>=3)}")
