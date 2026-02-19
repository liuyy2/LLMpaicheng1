"""
Quick single-episode test of the anchor optimization changes.
Run trcg_repair_llm on one light + one medium + one heavy seed.
Compare drift metrics vs fixed_tuned.
"""
import sys, copy
sys.path.insert(0, '.')

from config import make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies.policy_fixed import FixedWeightPolicy
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
from metrics import metrics_to_dict

# Use LLM client
try:
    from llm_client import LLMClient, LLMConfig
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Test seeds (from the BL dataset)
test_cases = [
    (66, "light"),
]

for seed, level in test_cases:
    print(f"\n{'='*60}")
    print(f"Seed={seed}, Level={level}")
    print(f"{'='*60}")
    
    config = make_config_for_difficulty(level)
    config.solver_timeout_s = 15.0  # Faster iteration
    
    # Create SEPARATE scenarios to avoid in-place mutation during simulation
    scenario_ft = generate_scenario(seed, config)
    scenario_trcg = generate_scenario(seed, config)  # Fresh copy
    
    # Run fixed_tuned
    ft_policy = FixedWeightPolicy(
        w_delay=10.0, w_shift=1.0, w_switch=5.0,
        freeze_horizon=96,
        use_two_stage=True,
        epsilon_solver=0.10,
    )
    ft_result = simulate_episode(ft_policy, scenario_ft, config, verbose=False)
    ft_metrics = metrics_to_dict(ft_result.metrics)
    
    print(f"  fixed_tuned:")
    print(f"    drift_per_replan = {ft_metrics.get('drift_per_replan', 0):.2f}")
    print(f"    episode_drift    = {ft_metrics.get('episode_drift', 0):.2f}")
    print(f"    avg_delay        = {ft_metrics.get('avg_delay', 0):.2f}")
    print(f"    win_switches     = {ft_metrics.get('total_window_switches', 0)}")
    print(f"    seq_switches     = {ft_metrics.get('total_sequence_switches', 0)}")
    print(f"    total_shifts     = {ft_metrics.get('total_shifts', 0)}")
    
    # Run trcg_repair (heuristic mode — same anchor/epsilon changes apply)
    # Use SAME weights as fixed_tuned to isolate anchor effect
    trcg_policy = TRCGRepairPolicy(
        llm_client=None,  # Heuristic only
        policy_name="trcg_repair_llm",
        log_dir="test_output_temp/logs",
        enable_logging=False,
        episode_id=f"{seed}_{level}",
        w_delay=10.0, w_shift=1.0, w_switch=5.0,  # SAME as fixed_tuned
    )
    trcg_result = simulate_episode(trcg_policy, scenario_trcg, config, verbose=False)
    trcg_metrics = metrics_to_dict(trcg_result.metrics)
    
    print(f"  trcg_repair_llm:")
    print(f"    drift_per_replan = {trcg_metrics.get('drift_per_replan', 0):.2f}")
    print(f"    episode_drift    = {trcg_metrics.get('episode_drift', 0):.2f}")
    print(f"    avg_delay        = {trcg_metrics.get('avg_delay', 0):.2f}")
    print(f"    win_switches     = {trcg_metrics.get('total_window_switches', 0)}")
    print(f"    seq_switches     = {trcg_metrics.get('total_sequence_switches', 0)}")
    print(f"    total_shifts     = {trcg_metrics.get('total_shifts', 0)}")
    
    ft_dpr = ft_metrics.get('drift_per_replan', 0)
    trcg_dpr = trcg_metrics.get('drift_per_replan', 0)
    if ft_dpr > 0:
        improvement = (ft_dpr - trcg_dpr) / ft_dpr * 100
        print(f"  Improvement: {improvement:.1f}% drift reduction")
    
    delay_change = trcg_metrics.get('avg_delay', 0) - ft_metrics.get('avg_delay', 0)
    print(f"  Delay change: {delay_change:+.2f}")

print("\nDone!")
