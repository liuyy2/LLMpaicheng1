"""Analyze heuristic_fallback: what triggers it and what it does vs LLM."""
import json, os, glob, csv, statistics
from collections import Counter

# Analyze all fallback steps across all episodes
all_fallback_reasons = Counter()
all_fallback_attempts = Counter()
fallback_has_conflict = 0
fallback_no_conflict = 0
llm_has_conflict = 0
llm_no_conflict = 0

losing_fb_drifts = []  # drift values at fallback steps on losing seeds
winning_fb_drifts = []
total_fb_steps = 0
total_llm_steps = 0

rows = list(csv.DictReader(open(r'results_V2.5\BL\218_4\results_per_episode.csv')))
llm_map = {int(r['seed']): r for r in rows if r['policy_name'] == 'trcg_repair_llm'}
ft_map = {int(r['seed']): r for r in rows if r['policy_name'] == 'fixed_tuned'}

for seed in sorted(llm_map.keys()):
    llm_log_dir = f'results_V2.5/LLM/2.18_4/logs/episode_{seed}_trcg_repair_llm'
    if not os.path.exists(llm_log_dir):
        continue
    
    # Load rolling metrics
    roll_path = os.path.join(llm_log_dir, 'metrics_per_roll.csv')
    roll_data = {}
    if os.path.exists(roll_path):
        for r in csv.DictReader(open(roll_path)):
            roll_data[int(r['t'])] = float(r['plan_drift'])
    
    is_losing = float(llm_map[seed]['drift_per_replan']) > float(ft_map[seed]['drift_per_replan'])
    
    for fp in glob.glob(os.path.join(llm_log_dir, 'repair_step_*.json')):
        with open(fp) as f:
            step = json.load(f)
        
        src = step.get('decision_source', '')
        t = step['now_slot']
        conflicts = step.get('trcg_top_conflicts', [])
        fb_reason = step.get('fallback_reason', '')
        fb_attempts = step.get('fallback_attempts', [])
        
        drift_at_step = roll_data.get(t, 0.0)
        
        if src == 'heuristic_fallback':
            total_fb_steps += 1
            if fb_reason:
                all_fallback_reasons[fb_reason] += 1
            for att in fb_attempts:
                all_fallback_attempts[att.get('name', '?')] += 1
            if conflicts:
                fallback_has_conflict += 1
            else:
                fallback_no_conflict += 1
            if is_losing:
                losing_fb_drifts.append(drift_at_step)
            else:
                winning_fb_drifts.append(drift_at_step)
        elif src == 'llm':
            total_llm_steps += 1
            if conflicts:
                llm_has_conflict += 1
            else:
                llm_no_conflict += 1

print("=== Fallback Analysis ===")
print(f"Total fallback steps: {total_fb_steps}")
print(f"Total LLM steps: {total_llm_steps}")
print()
print("Fallback reasons:")
for reason, cnt in all_fallback_reasons.most_common(20):
    print(f"  {reason}: {cnt}")
print()
print("Fallback attempt names:")
for name, cnt in all_fallback_attempts.most_common(20):
    print(f"  {name}: {cnt}")
print()
print(f"Fallback with conflicts: {fallback_has_conflict}, without: {fallback_no_conflict}")
print(f"LLM with conflicts: {llm_has_conflict}, without: {llm_no_conflict}")
print()
if losing_fb_drifts:
    print(f"Fallback drift (losing seeds): avg={statistics.mean(losing_fb_drifts):.2f}, n={len(losing_fb_drifts)}")
if winning_fb_drifts:
    print(f"Fallback drift (winning seeds): avg={statistics.mean(winning_fb_drifts):.2f}, n={len(winning_fb_drifts)}")

# ---- Analyze skip_stable_state and skip_no_conflict ----
print("\n=== Skip Analysis ===")
skip_stable_drifts = []
skip_noconf_drifts = []

for seed in sorted(llm_map.keys()):
    llm_log_dir = f'results_V2.5/LLM/2.18_4/logs/episode_{seed}_trcg_repair_llm'
    if not os.path.exists(llm_log_dir):
        continue
    
    roll_path = os.path.join(llm_log_dir, 'metrics_per_roll.csv')
    roll_data = {}
    if os.path.exists(roll_path):
        for r in csv.DictReader(open(roll_path)):
            roll_data[int(r['t'])] = float(r['plan_drift'])
    
    for fp in glob.glob(os.path.join(llm_log_dir, 'repair_step_*.json')):
        with open(fp) as f:
            step = json.load(f)
        src = step.get('decision_source', '')
        t = step['now_slot']
        d = roll_data.get(t, 0.0)
        if src == 'skip_stable_state':
            skip_stable_drifts.append(d)
        elif src == 'skip_no_conflict':
            skip_noconf_drifts.append(d)

if skip_stable_drifts:
    nz = [d for d in skip_stable_drifts if d > 0]
    print(f"skip_stable_state: n={len(skip_stable_drifts)}, nonzero_drift={len(nz)}, avg_drift={statistics.mean(skip_stable_drifts):.2f}")
    if nz:
        print(f"  nonzero avg={statistics.mean(nz):.2f}, max={max(nz):.2f}")
if skip_noconf_drifts:
    nz = [d for d in skip_noconf_drifts if d > 0]
    print(f"skip_no_conflict: n={len(skip_noconf_drifts)}, nonzero_drift={len(nz)}, avg_drift={statistics.mean(skip_noconf_drifts):.2f}")
    if nz:
        print(f"  nonzero avg={statistics.mean(nz):.2f}, max={max(nz):.2f}")
