import json, os, glob

# Check LLM log files from episodes
base_log = 'results_V2.5/LLM/2.18_7/logs'
ep_dirs = sorted(glob.glob(os.path.join(base_log, 'episode_*')))

decision_sources = {}
total = 0
unlock_sizes = []
anchor_applied_list = []
anchor_skipped_list = []
fallback_count = 0
anchor_skip_reasons = {}

for ep_dir in ep_dirs:
    files = sorted(glob.glob(os.path.join(ep_dir, '*.json')))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        total += 1
        src = d.get('decision_source', 'unknown')
        decision_sources[src] = decision_sources.get(src, 0) + 1
        
        uids = d.get('unlock_mission_ids', [])
        if uids:
            unlock_sizes.append(len(uids))
        
        ap = d.get('anchor_fix_applied', 0) or 0
        sk = d.get('anchor_fix_skipped', 0) or 0
        anchor_applied_list.append(ap)
        anchor_skipped_list.append(sk)
        
        reason = d.get('anchor_fix_skip_reason', '')
        if reason:
            anchor_skip_reasons[reason] = anchor_skip_reasons.get(reason, 0) + 1
        
        fa = d.get('fallback_attempts', [])
        if fa:
            fallback_count += 1

print(f"Total logs across {len(ep_dirs)} episodes: {total}")
print(f"Decision sources: {decision_sources}")
if unlock_sizes:
    print(f"Unlock sizes: min={min(unlock_sizes)} max={max(unlock_sizes)} avg={sum(unlock_sizes)/len(unlock_sizes):.1f} count={len(unlock_sizes)}")
if anchor_applied_list:
    nonzero = [x for x in anchor_applied_list if x > 0]
    print(f"Anchor applied: nonzero_count={len(nonzero)}/{total} avg_when_applied={sum(nonzero)/len(nonzero) if nonzero else 0:.1f}")
if anchor_skipped_list:
    nonzero = [x for x in anchor_skipped_list if x > 0]
    print(f"Anchor skipped: nonzero_count={len(nonzero)}/{total} avg_when_skipped={sum(nonzero)/len(nonzero) if nonzero else 0:.1f}")
print(f"Anchor skip reasons: {anchor_skip_reasons}")
print(f"Fallback chains triggered: {fallback_count}/{total}")

# Show a few specific logs  
sample_ep = ep_dirs[0] if ep_dirs else None
if sample_ep:
    files = sorted(glob.glob(os.path.join(sample_ep, '*.json')))
    for f in files[:3]:
        print(f"\n=== {os.path.basename(f)} (from {os.path.basename(sample_ep)}) ===")
        with open(f) as fh:
            d = json.load(fh)
        print(f"  decision_source: {d.get('decision_source')}")
        print(f"  unlock: {d.get('unlock_mission_ids')}")
        print(f"  root_cause: {d.get('root_cause_mission_id')}")
        print(f"  anchor_applied: {d.get('anchor_fix_applied')}")
        print(f"  anchor_skipped: {d.get('anchor_fix_skipped')}")
        print(f"  anchor_skip_reason: {d.get('anchor_fix_skip_reason')}")
        print(f"  solver_status: {d.get('solver_status')}")
        fa = d.get('fallback_attempts', [])
        if fa:
            print(f"  fallback_attempts: {len(fa)}")
            for a in fa:
                print(f"    {a.get('attempt_name')}: status={a.get('solver_status')}, anchor={a.get('anchor_applied')}/{a.get('anchor_skipped')}, skip_reason={a.get('anchor_skip_reason')}")
