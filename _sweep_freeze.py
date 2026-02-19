"""
Sweep freeze_horizon values to find optimal drift vs delay trade-off.
Tests trcg_repair_llm with different freeze values against fixed_tuned baseline.
"""
import sys, importlib
sys.path.insert(0, '.')

from config import make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies.policy_fixed import FixedWeightPolicy
from policies.policy_llm_trcg_repair import TRCGRepairPolicy
from metrics import metrics_to_dict
import policies.policy_llm_trcg_repair as trcg_module

SEED = 80
LEVEL = "medium"
FREEZE_VALUES = [96, 108, 120, 132, 144, 160]

config = make_config_for_difficulty(LEVEL)
config.solver_timeout_s = 15.0

# Run fixed_tuned baseline ONCE
print(f"Running fixed_tuned baseline (seed={SEED}, level={LEVEL})...")
scenario_ft = generate_scenario(SEED, config)
ft_policy = FixedWeightPolicy(
    w_delay=10.0, w_shift=1.0, w_switch=5.0,
    freeze_horizon=96,
    use_two_stage=True,
    epsilon_solver=0.10,
)
ft_result = simulate_episode(ft_policy, scenario_ft, config, verbose=False)
ft_metrics = metrics_to_dict(ft_result.metrics)
ft_dpr = ft_metrics['drift_per_replan']
ft_delay = ft_metrics['avg_delay']
print(f"  fixed_tuned: drift_per_replan={ft_dpr:.2f}, avg_delay={ft_delay:.2f}")
print(f"    win={ft_metrics['total_window_switches']} seq={ft_metrics['total_sequence_switches']} shifts={ft_metrics['total_shifts']}")
print()

# Sweep freeze values
print(f"{'freeze':>8} | {'dpr':>8} | {'delay':>8} | {'drift_chg':>10} | {'delay_chg':>10} | {'win':>4} | {'seq':>4} | {'shifts':>6}")
print("-" * 80)

for fv in FREEZE_VALUES:
    # Monkey-patch the freeze minimum
    # We need to override the _TRCG_FREEZE_MIN inside the policy's decide() 
    # This is a hack - we read the source and know the variable name
    # Better: pass via constructor or config
    
    scenario_trcg = generate_scenario(SEED, config)
    trcg_policy = TRCGRepairPolicy(
        llm_client=None,
        policy_name="trcg_repair_llm",
        log_dir="test_output_temp/logs",
        enable_logging=False,
        episode_id=f"{SEED}_{LEVEL}_f{fv}",
        w_delay=10.0, w_shift=1.0, w_switch=5.0,
    )
    
    # Monkey-patch: override the freeze value in the decide method
    original_decide = trcg_policy.decide
    def make_patched_decide(freeze_val, orig):
        def patched_decide(state, now, cfg):
            meta, plan = orig(state, now, cfg)
            if meta:
                # Override freeze_horizon in MetaParams
                import dataclasses
                meta = dataclasses.replace(meta, freeze_horizon=freeze_val)
            return meta, plan
        return patched_decide
    
    trcg_policy.decide = make_patched_decide(fv, original_decide)
    
    trcg_result = simulate_episode(trcg_policy, scenario_trcg, config, verbose=False)
    trcg_metrics = metrics_to_dict(trcg_result.metrics)
    trcg_dpr = trcg_metrics['drift_per_replan']
    trcg_delay = trcg_metrics['avg_delay']
    
    drift_chg = (trcg_dpr - ft_dpr) / ft_dpr * 100 if ft_dpr > 0 else 0
    delay_chg = trcg_delay - ft_delay
    
    print(f"{fv:>8} | {trcg_dpr:>8.2f} | {trcg_delay:>8.2f} | {drift_chg:>+9.1f}% | {delay_chg:>+9.2f} | {trcg_metrics['total_window_switches']:>4} | {trcg_metrics['total_sequence_switches']:>4} | {trcg_metrics['total_shifts']:>6}")

print("\nDone!")
