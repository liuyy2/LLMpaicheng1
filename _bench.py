import sys, time, logging
logging.disable(logging.CRITICAL)

from scenario import generate_scenario
from config import DEFAULT_CONFIG
from simulator import simulate_episode
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
from policies.base import MetaParams
from policies.policy_fixed import FixedWeightPolicy

# TRCG w/ anchors
scenario1 = generate_scenario(seed=123)
policy1 = TRCGRepairPolicy(policy_name='t1')
t0 = time.time()
r1 = simulate_episode(policy1, scenario1, DEFAULT_CONFIG, verbose=False)
t1_elapsed = time.time() - t0

# Fixed (baseline)
scenario2 = generate_scenario(seed=123)
policy2 = FixedWeightPolicy(w_delay=10.0, w_shift=1.0, w_switch=5.0)
t0 = time.time()
r2 = simulate_episode(policy2, scenario2, DEFAULT_CONFIG, verbose=False)
t2_elapsed = time.time() - t0

sys.stderr = open('NUL', 'w')
print(f"TRCG: drift={r1.metrics.episode_drift:.2f}, time={t1_elapsed:.2f}s, steps={len(r1.snapshots)}")
print(f"Fixed: drift={r2.metrics.episode_drift:.2f}, time={t2_elapsed:.2f}s, steps={len(r2.snapshots)}")
print(f"Drift reduction: {(1-r1.metrics.episode_drift/r2.metrics.episode_drift)*100:.1f}%")
print(f"Runtime overhead: {(t1_elapsed/t2_elapsed-1)*100:.1f}%")
