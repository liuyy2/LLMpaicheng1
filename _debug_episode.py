"""Run one episode with the LLM policy path (using cache) and capture any exception."""
import traceback, sys, os

# Simulate what run_experiments.py does for trcg_repair_llm
sys.path.insert(0, '.')

from run_experiments import run_single_episode, ExperimentConfig

# Build minimal exp_config matching the 2.18_4 command
exp_config = ExperimentConfig(
    num_test_seeds=1,
    output_dir='test_output_temp/debug_episode',
    llm_base_url='https://api.deepseek.com',  
    llm_model='deepseek-chat',
    llm_key_env='DASHSCOPE_API_KEY',
    llm_cache_dir=r'results_V2.5\LLM\2.18_4\llm_cache',
)

seed, level, policy_name, policy_params, dataset = 60, 'light', 'trcg_repair_llm', {}, 'test'
solver_timeout = 30

print('=== Running run_single_episode directly (no try/except) ===')
try:
    record = run_single_episode(
        seed, level, policy_name, policy_params, dataset,
        solver_timeout, exp_config, 'test_output_temp/debug_episode'
    )
    print(f'\nSUCCESS: completed={record.completed}, total={record.total}')
    print(f'  episode_drift={record.episode_drift:.2f}')
    print(f'  drift_per_active_mission={record.drift_per_active_mission:.4f}')
except Exception:
    print('\n!!! EXCEPTION:')
    traceback.print_exc()
