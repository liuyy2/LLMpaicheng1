"""Compare 218_5 (with started-filter fix) vs 218_4 (before fix) results."""
import csv
import statistics

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

# Load 218_5 results
llm_5 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_5\results_per_episode.csv")
bl_5 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\BL\218_5\results_per_episode.csv")

# Load 218_4 results for comparison
llm_4 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_4\results_per_episode.csv")
bl_4 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\BL\218_4\results_per_episode.csv")

def get_policy_data(rows, policy):
    return [r for r in rows if r['policy_name'] == policy]

def avg(vals):
    return sum(vals) / len(vals) if vals else 0

# ===== 218_5 overall comparison =====
print("=" * 70)
print("218_5 OVERALL COMPARISON")
print("=" * 70)

# Separate LLM and FT from 218_5
llm_rows_5 = llm_5  # all trcg_repair_llm
ft_rows_5 = get_policy_data(bl_5, 'fixed_tuned')

# Separate LLM and FT from BL (contains both policies)
if not ft_rows_5:
    # BL might contain both policies mixed
    ft_rows_5 = [r for r in bl_5 if r['policy_name'] == 'fixed_tuned']
    llm_from_bl_5 = [r for r in bl_5 if r['policy_name'] == 'trcg_repair_llm']
    if llm_from_bl_5:
        llm_rows_5 = llm_from_bl_5

print(f"LLM episodes: {len(llm_rows_5)}, FT episodes: {len(ft_rows_5)}")

# Get unique policies in BL
policies_bl = set(r['policy_name'] for r in bl_5)
print(f"Policies in BL/218_5: {policies_bl}")
policies_llm = set(r['policy_name'] for r in llm_5)
print(f"Policies in LLM/218_5: {policies_llm}")

# Compute averages
for label, rows in [("trcg_repair_llm (218_5)", llm_rows_5), ("fixed_tuned (218_5)", ft_rows_5)]:
    if not rows:
        print(f"\n{label}: NO DATA")
        continue
    drifts = [float(r['drift_per_replan']) for r in rows if r.get('drift_per_replan')]
    delays = [float(r['avg_delay']) for r in rows if r.get('avg_delay')]
    shifts = [float(r['total_shifts']) for r in rows if r.get('total_shifts')]
    switches = [float(r['total_switches']) for r in rows if r.get('total_switches')]
    fallbacks = [float(r.get('llm_fallback_count', 0)) for r in rows]
    replans = [float(r['num_replans']) for r in rows if r.get('num_replans')]
    
    print(f"\n{label} (n={len(rows)}):")
    print(f"  drift_per_replan: mean={avg(drifts):.2f}, median={statistics.median(drifts):.2f}, std={statistics.stdev(drifts):.2f}")
    print(f"  avg_delay:        mean={avg(delays):.2f}")
    print(f"  total_shifts:     mean={avg(shifts):.2f}")
    print(f"  total_switches:   mean={avg(switches):.2f}")
    print(f"  llm_fallback:     mean={avg(fallbacks):.2f}, total={sum(fallbacks):.0f}")
    print(f"  num_replans:      mean={avg(replans):.2f}")

# ===== 218_4 for comparison =====
print("\n" + "=" * 70)
print("218_4 COMPARISON (BEFORE FIX)")
print("=" * 70)

llm_rows_4 = llm_4
ft_rows_4 = get_policy_data(bl_4, 'fixed_tuned')
if not ft_rows_4:
    ft_rows_4 = [r for r in bl_4 if r['policy_name'] == 'fixed_tuned']
    llm_from_bl_4 = [r for r in bl_4 if r['policy_name'] == 'trcg_repair_llm']
    if llm_from_bl_4:
        llm_rows_4 = llm_from_bl_4

for label, rows in [("trcg_repair_llm (218_4)", llm_rows_4), ("fixed_tuned (218_4)", ft_rows_4)]:
    if not rows:
        print(f"\n{label}: NO DATA")
        continue
    drifts = [float(r['drift_per_replan']) for r in rows if r.get('drift_per_replan')]
    delays = [float(r['avg_delay']) for r in rows if r.get('avg_delay')]
    fallbacks = [float(r.get('llm_fallback_count', 0)) for r in rows]
    
    print(f"\n{label} (n={len(rows)}):")
    print(f"  drift_per_replan: mean={avg(drifts):.2f}, median={statistics.median(drifts):.2f}")
    print(f"  avg_delay:        mean={avg(delays):.2f}")
    print(f"  llm_fallback:     mean={avg(fallbacks):.2f}, total={sum(fallbacks):.0f}")

# ===== Seed-matched comparison 218_5 =====
print("\n" + "=" * 70)
print("218_5 SEED-MATCHED COMPARISON (LLM vs FT)")
print("=" * 70)

# Build seed-level dicts
def build_seed_dict(rows, key='drift_per_replan'):
    d = {}
    for r in rows:
        seed = r['seed']
        level = r['disturbance_level']
        d[(seed, level)] = float(r[key]) if r.get(key) else None
    return d

llm_drift_5 = build_seed_dict(llm_rows_5)
ft_drift_5 = build_seed_dict(ft_rows_5)

common_5 = set(llm_drift_5.keys()) & set(ft_drift_5.keys())
print(f"Common seeds: {len(common_5)}")

if common_5:
    diffs = []
    wins = losses = ties = 0
    for key in sorted(common_5):
        ld = llm_drift_5[key]
        fd = ft_drift_5[key]
        if ld is not None and fd is not None:
            diff = ld - fd
            diffs.append(diff)
            if diff < -0.5: wins += 1
            elif diff > 0.5: losses += 1
            else: ties += 1
    
    print(f"LLM wins: {wins}, FT wins: {losses}, ties: {ties}")
    print(f"Avg diff (LLM - FT): {avg(diffs):.2f}")
    print(f"Median diff: {statistics.median(diffs):.2f}")
    
    # Per level
    for level in ['light', 'medium', 'heavy']:
        level_diffs = []
        for key in common_5:
            if key[1] == level:
                ld = llm_drift_5[key]
                fd = ft_drift_5[key]
                if ld is not None and fd is not None:
                    level_diffs.append(ld - fd)
        if level_diffs:
            print(f"  {level}: avg_diff={avg(level_diffs):+.2f}, n={len(level_diffs)}")

# ===== Compare 218_4 vs 218_5 LLM drift directly =====
print("\n" + "=" * 70)
print("LLM DRIFT: 218_4 vs 218_5 (same seeds)")
print("=" * 70)

llm_drift_4 = {}
for r in llm_rows_4:
    seed = r['seed']
    level = r['disturbance_level']
    llm_drift_4[(seed, level)] = float(r['drift_per_replan']) if r.get('drift_per_replan') else None

common_45 = set(llm_drift_4.keys()) & set(llm_drift_5.keys())
print(f"Common seeds between 218_4 and 218_5 LLM: {len(common_45)}")

if common_45:
    diffs_45 = []
    better = worse = same = 0
    for key in sorted(common_45):
        d4 = llm_drift_4[key]
        d5 = llm_drift_5[key]
        if d4 is not None and d5 is not None:
            diff = d5 - d4  # positive = 218_5 is worse
            diffs_45.append(diff)
            if diff < -0.5: better += 1
            elif diff > 0.5: worse += 1
            else: same += 1
    
    print(f"218_5 better: {better}, 218_5 worse: {worse}, same: {same}")
    print(f"Avg change (218_5 - 218_4): {avg(diffs_45):+.2f}")
    
    # Worst regressions and best improvements
    keyed = [(key, d5 - d4) for key, d4, d5 in 
             ((k, llm_drift_4[k], llm_drift_5[k]) for k in common_45 
              if llm_drift_4[k] is not None and llm_drift_5[k] is not None)]
    keyed.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 regressions (218_5 worse):")
    for (seed, level), diff in keyed[:5]:
        print(f"  seed={seed} level={level}: {llm_drift_4[(seed,level)]:.2f} -> {llm_drift_5[(seed,level)]:.2f} (change={diff:+.2f})")
    
    print("\nTop 5 improvements (218_5 better):")
    for (seed, level), diff in keyed[-5:]:
        print(f"  seed={seed} level={level}: {llm_drift_4[(seed,level)]:.2f} -> {llm_drift_5[(seed,level)]:.2f} (change={diff:+.2f})")

# ===== Fallback analysis =====
print("\n" + "=" * 70)
print("FALLBACK COMPARISON: 218_4 vs 218_5")
print("=" * 70)

fb_4 = sum(float(r.get('llm_fallback_count', 0)) for r in llm_rows_4)
fb_5 = sum(float(r.get('llm_fallback_count', 0)) for r in llm_rows_5)
print(f"218_4 total fallbacks: {fb_4:.0f}")
print(f"218_5 total fallbacks: {fb_5:.0f}")
print(f"Change: {fb_5 - fb_4:+.0f}")

# ===== Decision source from logs =====
print("\n" + "=" * 70)
print("DECISION SOURCE ANALYSIS (218_5 logs)")
print("=" * 70)

import os, json
log_dir = r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_5\logs"
if os.path.exists(log_dir):
    sources = {}
    total_steps = 0
    for fn in os.listdir(log_dir)[:10]:  # sample first 10
        if fn.endswith('.jsonl'):
            with open(os.path.join(log_dir, fn), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        src = entry.get('decision_source', 'unknown')
                        sources[src] = sources.get(src, 0) + 1
                        total_steps += 1
                    except: pass
    
    print(f"From first 10 log files ({total_steps} steps):")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({100*cnt/total_steps:.1f}%)")
else:
    print("No log directory found")
