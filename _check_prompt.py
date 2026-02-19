"""Check: why does LLM pick started missions? Look at prompt content."""
import json, os, glob

# Check a failing step from seed=116 — find one with heuristic_fallback
llm_dir = r'results_V2.5/LLM/2.18_4/logs/episode_116_trcg_repair_llm'

for fp in sorted(glob.glob(os.path.join(llm_dir, 'repair_step_*.json'))):
    with open(fp) as f:
        step = json.load(f)
    if step.get('decision_source') == 'heuristic_fallback':
        t = step['now_slot']
        fb = step.get('fallback_reason', '')
        raw = step.get('llm_raw_output', '')
        urgents = step.get('trcg_urgent_ids', [])
        conflicts = step.get('trcg_top_conflicts', [])
        decision_json = step.get('decision_json', {})
        
        print(f"=== t={t}, fallback_reason: {fb[:200]} ===")
        print(f"  trcg_urgent_ids: {urgents}")
        print(f"  trcg_top_conflicts: {conflicts}")
        print(f"  LLM raw output: {raw[:300]}")
        print(f"  LLM decision: {decision_json}")
        print()
        
        # The key question: what was in the active_mission_ids sent to LLM?
        # LLM picked a mission that's in started_mission_ids
        # Let's check the failed mission
        errors = step.get('llm_error', {})
        print(f"  errors: {errors}")
        print()
        break

# Now look at the llm_raw_calls.jsonl to see prompt content for that step
print("=== LLM Prompt Content (first failing call) ===")
raw_calls = os.path.join(llm_dir, 'llm_raw_calls.jsonl')
with open(raw_calls) as f:
    for i, line in enumerate(f):
        call = json.loads(line)
        raw = call.get('raw_text', '')
        if 'started' in str(call.get('error_message', '')):
            print(f"Call {i}: tokens={call.get('tokens_total')}, raw={raw[:500]}")
            break
    else:
        # Just show first few calls
        f.seek(0)
        for i, line in enumerate(f):
            if i >= 3:
                break
            call = json.loads(line)
            print(f"Call {i}: tokens_prompt={call.get('tokens_prompt')}, raw_text={call.get('raw_text', '')[:300]}")
