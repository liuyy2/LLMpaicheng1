"""Deep analysis: understand WHY filtering started missions hurt LLM quality.
Also analyze decision flow to find the real improvement opportunity."""
import csv, json, os, statistics

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

# ===== 1. Compare skip rates between 218_4 and 218_5 =====
print("=" * 70)
print("HYPOTHESIS 1: Did skip_no_conflict / skip_stable_state increase?")
print("=" * 70)

# Check logs for 218_5
log_dir_5 = r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_5\logs"
log_dir_BL5 = r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\BL\218_5"

# Check what's in the LLM log directory
if os.path.exists(log_dir_5):
    files = os.listdir(log_dir_5)
    print(f"Log files in 218_5/logs: {len(files)} files")
    print(f"Sample: {files[:3]}")
else:
    print("No 218_5 log dir")

# Also check BL logs
bl_log_dir = os.path.join(log_dir_BL5, 'logs')
if os.path.exists(bl_log_dir):
    print(f"BL 218_5 has logs dir")
else:
    print("No BL/218_5/logs dir")

# Check for log files in the episode CSVs themselves
llm_5 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_5\results_per_episode.csv")
llm_4 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\LLM\2.18_4\results_per_episode.csv")

# ===== 2. Compare LLM call counts =====
print("\n" + "=" * 70)
print("HYPOTHESIS 2: LLM call counts changed?")
print("=" * 70)

calls_4 = [float(r.get('llm_calls', 0)) for r in llm_4]
calls_5 = [float(r.get('llm_calls', 0)) for r in llm_5]
fb_4 = [float(r.get('llm_fallback_count', 0)) for r in llm_4]
fb_5 = [float(r.get('llm_fallback_count', 0)) for r in llm_5]
replans_4 = [float(r.get('num_replans', 0)) for r in llm_4]
replans_5 = [float(r.get('num_replans', 0)) for r in llm_5]

print(f"218_4: avg llm_calls={sum(calls_4)/len(calls_4):.1f}, avg fallbacks={sum(fb_4)/len(fb_4):.1f}")
print(f"218_5: avg llm_calls={sum(calls_5)/len(calls_5):.1f}, avg fallbacks={sum(fb_5)/len(fb_5):.1f}")
print(f"218_4: avg replans={sum(replans_4)/len(replans_4):.1f}")
print(f"218_5: avg replans={sum(replans_5)/len(replans_5):.1f}")
print(f"218_4: effective LLM decisions = calls - fallbacks = {sum(calls_4)/len(calls_4) - sum(fb_4)/len(fb_4):.1f}")
print(f"218_5: effective LLM decisions = calls - fallbacks = {sum(calls_5)/len(calls_5) - sum(fb_5)/len(fb_5):.1f}")

# ===== 3. Per-seed regression analysis =====
print("\n" + "=" * 70)
print("REGRESSION ROOTS: what changed in top5 regressing seeds?")
print("=" * 70)

regressed_seeds = [('112','heavy'), ('119','heavy'), ('109','heavy'), ('102','heavy'), ('114','heavy')]

for seed, level in regressed_seeds:
    r4 = next((r for r in llm_4 if r['seed']==seed and r['disturbance_level']==level), None)
    r5 = next((r for r in llm_5 if r['seed']==seed and r['disturbance_level']==level), None)
    if r4 and r5:
        print(f"\n  seed={seed} ({level}):")
        for key in ['drift_per_replan', 'total_shifts', 'total_switches', 'total_window_switches', 
                     'total_sequence_switches', 'llm_calls', 'llm_fallback_count', 'num_replans',
                     'avg_delay', 'feasible_rate', 'num_forced_replans']:
            v4 = r4.get(key, 'N/A')
            v5 = r5.get(key, 'N/A')
            try:
                diff = float(v5) - float(v4)
                print(f"    {key:30s}: {v4:>10s} -> {v5:>10s}  ({diff:+.2f})")
            except:
                print(f"    {key:30s}: {v4:>10s} -> {v5:>10s}")

# ===== 4. Component decomposition of drift change =====
print("\n" + "=" * 70)
print("DRIFT COMPONENT DECOMPOSITION (avg across all 60)")
print("=" * 70)

def avg(vals):
    return sum(vals) / len(vals) if vals else 0

for label, rows in [("218_4", llm_4), ("218_5", llm_5)]:
    shifts = [float(r['total_shifts']) for r in rows]
    w_sw = [float(r.get('total_window_switches', 0)) for r in rows]
    s_sw = [float(r.get('total_sequence_switches', 0)) for r in rows]
    t_sw = [float(r.get('total_switches', 0)) for r in rows]
    dpr = [float(r['drift_per_replan']) for r in rows]
    ep_drift = [float(r['episode_drift']) for r in rows]
    
    print(f"\n{label}:")
    print(f"  drift_per_replan: {avg(dpr):.2f}")
    print(f"  episode_drift:    {avg(ep_drift):.2f}")
    print(f"  total_shifts:     {avg(shifts):.2f}")
    print(f"  total_switches:   {avg(t_sw):.2f}")
    print(f"  window_switches:  {avg(w_sw):.2f} (weight=12)")
    print(f"  sequence_switches:{avg(s_sw):.2f} (weight=6)")

# ===== 5. FT comparison to understand the baseline =====
print("\n" + "=" * 70)
print("FT BASELINE: are FT results identical in 218_4 and 218_5?")
print("=" * 70)

bl_4 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\BL\218_4\results_per_episode.csv")
bl_5 = load_csv(r"d:\Projects\LLMpaicheng1\LLMpaicheng1\results_V2.5\BL\218_5\results_per_episode.csv")

ft_4 = [r for r in bl_4 if r['policy_name'] == 'fixed_tuned']
ft_5 = [r for r in bl_5 if r['policy_name'] == 'fixed_tuned']

ft4_drifts = [float(r['drift_per_replan']) for r in ft_4]
ft5_drifts = [float(r['drift_per_replan']) for r in ft_5]
print(f"FT 218_4: n={len(ft_4)}, mean_drift={avg(ft4_drifts):.2f}")
print(f"FT 218_5: n={len(ft_5)}, mean_drift={avg(ft5_drifts):.2f}")
print(f"FT results identical? {abs(avg(ft4_drifts) - avg(ft5_drifts)) < 0.01}")
