"""Check 2.18_4 cache coverage and verify the fix."""
import os, glob, traceback

# 1. Cache entries
cache_dir = r'results_V2.5\LLM\2.18_4\llm_cache'
files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
print(f'Cache entries in 2.18_4: {len(files)}')

# 2. LLM raw call log analysis
logs = glob.glob(r'results_V2.5\LLM\2.18_4\logs\**\llm_raw_calls.jsonl', recursive=True)
total_calls = 0
cache_hits = 0
for p in logs:
    for line in open(p, encoding='utf-8'):
        if not line.strip():
            continue
        total_calls += 1
        if '"cache_hit": true' in line:
            cache_hits += 1
print(f'LLM log entries: {total_calls}  cache_hits={cache_hits}')
print(f'Episodes with logs: {len(logs)}')

# 3. Verify simulation works now
print('\n--- Verify fix ---')
from simulator import simulate_episode
from scenario import generate_scenario
from config import Config
from policies.policy_llm_trcg_repair import TRCGRepairPolicy

config = Config()
scenario = generate_scenario(seed=60, config=config)
policy = TRCGRepairPolicy(policy_name='trcg_repair', log_dir=None, enable_logging=False, episode_id='test')
result = simulate_episode(policy, scenario, config, verbose=False)
m = result.metrics
print(f'num_completed={m.num_completed}, num_total={m.num_total}')
print(f'episode_drift={m.episode_drift:.2f}, drift_per_active_mission={m.drift_per_active_mission:.4f}')
print('Fix verified OK' if m.num_total > 0 else 'STILL BROKEN')
