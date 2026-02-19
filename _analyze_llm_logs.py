import json, os, glob

# Check LLM log files from episode
log_dir = 'results_V2.5/LLM/2.18_7/logs'
if os.path.exists(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, '*.json')))[:10]
    decision_sources = {}
    skip_count = 0
    total = 0
    unlock_sizes = []
    anchor_applied = []
    anchor_skipped = []
    
    all_files = sorted(glob.glob(os.path.join(log_dir, '*.json')))
    for f in all_files:
        with open(f) as fh:
            d = json.load(fh)
        total += 1
        src = d.get('decision_source', 'unknown')
        decision_sources[src] = decision_sources.get(src, 0) + 1
        
        uids = d.get('unlock_mission_ids', [])
        if uids:
            unlock_sizes.append(len(uids))
        
        ap = d.get('anchor_fix_applied', 0)
        sk = d.get('anchor_fix_skipped', 0)
        if ap or sk:
            anchor_applied.append(ap)
            anchor_skipped.append(sk)
    
    print(f"Total logs: {total}")
    print(f"Decision sources: {decision_sources}")
    print(f"Unlock sizes: min={min(unlock_sizes) if unlock_sizes else 0} max={max(unlock_sizes) if unlock_sizes else 0} avg={sum(unlock_sizes)/len(unlock_sizes) if unlock_sizes else 0:.1f}")
    print(f"Anchor applied: count={len(anchor_applied)} avg={sum(anchor_applied)/len(anchor_applied) if anchor_applied else 0:.1f}")
    print(f"Anchor skipped: count={len(anchor_skipped)} avg={sum(anchor_skipped)/len(anchor_skipped) if anchor_skipped else 0:.1f}")
    
    # Show first few logs detail
    for f in files[:3]:
        print(f"\n=== {os.path.basename(f)} ===")
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
else:
    for d in os.listdir('results_V2.5/LLM/2.18_7'):
        print(d)
