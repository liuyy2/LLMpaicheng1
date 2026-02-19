"""Quick analysis: estimate runtime overhead of fallback chains."""
from scenario import generate_scenario
from config import DEFAULT_CONFIG
from simulator import simulate_episode
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
import time

# Run with seed that generates disturbance
scenario = generate_scenario(seed=42)
config = DEFAULT_CONFIG

policy = TRCGRepairPolicy(policy_name="timing_test")

t0 = time.time()
result = simulate_episode(policy, scenario, config, verbose=False)
elapsed = time.time() - t0

# Analyze snapshots
total_steps = len(result.snapshots)
fallback_count = 0
forced_global_count = 0
anchor_applied_steps = 0

for snap in result.snapshots:
    mp = snap.meta_params
    if mp:
        src = getattr(mp, 'decision_source', '')
        if 'forced_global' in str(src):
            forced_global_count += 1
        if getattr(mp, 'unlock_mission_ids', None) is not None:
            anchor_applied_steps += 1

stats = policy.get_stats()

print(f"Total steps: {total_steps}")
print(f"Runtime: {elapsed:.2f}s ({elapsed/total_steps:.3f}s per step)")
print(f"Steps with anchors: {anchor_applied_steps}")
print(f"Forced global replans: {forced_global_count}")
print(f"Policy stats: {stats}")
print(f"Drift: {result.metrics.episode_drift:.2f}")
print(f"Completed: {result.metrics.num_completed}/{result.metrics.num_total}")

# Compare with fixed policy
from policies.base import FixedWeightPolicy
scenario2 = generate_scenario(seed=42)
policy_fixed = FixedWeightPolicy(w_delay=10.0, w_shift=1.0, w_switch=5.0)

t1 = time.time()
result_fixed = simulate_episode(policy_fixed, scenario2, config, verbose=False)
elapsed_fixed = time.time() - t1

print(f"\n--- Comparison ---")
print(f"TRCG drift: {result.metrics.episode_drift:.2f}, runtime: {elapsed:.2f}s")
print(f"Fixed drift: {result_fixed.metrics.episode_drift:.2f}, runtime: {elapsed_fixed:.2f}s")
print(f"Drift reduction: {(1 - result.metrics.episode_drift / result_fixed.metrics.episode_drift) * 100:.1f}%")
print(f"Runtime overhead: {(elapsed / elapsed_fixed - 1) * 100:.1f}%")
