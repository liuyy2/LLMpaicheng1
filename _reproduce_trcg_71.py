"""复现 seed=71 light 下 trcg_repair 异常 delay=13"""
import csv
from config import Config, make_config_for_difficulty, MISSIONS_BY_DIFFICULTY
from scenario import generate_scenario
from simulator import simulate_episode
from policies import create_policy
from metrics import metrics_to_dict

seed = 71
difficulty = "light"
cfg = make_config_for_difficulty(difficulty)
cfg.sim_num_missions = MISSIONS_BY_DIFFICULTY[difficulty]

print(f"=== 场景 seed={seed}, difficulty={difficulty} ===")
scenario = generate_scenario(seed, cfg)

# 检查场景中有哪些 mission 的 Op 结构
for m in scenario.missions[:3]:
    print(f"\nMission {m.mission_id}:")
    for op in m.operations:
        tw_str = f" windows={op.time_windows}" if op.time_windows else ""
        print(f"  Op{op.op_index}: dur={op.duration}, release={op.release}, "
              f"resources={op.resources}{tw_str}")

# 运行各策略对比
for policy_name in ["fixed_default", "trcg_repair", "ga_repair"]:
    kwargs = {"log_dir": "test_output_temp"} if policy_name not in ("fixed_default", "fixed_tuned") else {}
    policy = create_policy(policy_name, **kwargs)
    result = simulate_episode(policy, scenario, cfg, verbose=False)
    m = metrics_to_dict(result.metrics)
    
    # 统计 op 分配
    op_counts = {}
    for a in result.final_schedule:
        op_counts[a.op_index] = op_counts.get(a.op_index, 0) + 1
    
    print(f"\n--- {policy_name} ---")
    print(f"  delay={m['avg_delay']:.3f} otr={m['on_time_rate']:.3f} "
          f"drift={m['episode_drift']:.2f} replans={m['num_replans']} "
          f"forced={m['num_forced_replans']}")
    print(f"  Op assignments: {dict(sorted(op_counts.items()))}")
    
    # 检查每个 mission 的 completion
    completed = result.completed_tasks
    uncompleted = result.uncompleted_tasks
    print(f"  completed={len(completed)} uncompleted={len(uncompleted)}")
    if uncompleted:
        print(f"  uncompleted_missions: {sorted(uncompleted)}")
    
    # 检查 delay 分布
    if hasattr(result.metrics, 'per_mission_delays') and result.metrics.per_mission_delays:
        delays = result.metrics.per_mission_delays
        nonzero = {k: v for k, v in delays.items() if v > 0}
        if nonzero:
            print(f"  per_mission_delays (nonzero): {nonzero}")

# 也运行verbose模式的 trcg_repair 看决策过程
print("\n\n=== trcg_repair VERBOSE 运行 ===")
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
policy = create_policy("trcg_repair", log_dir="test_output_temp")
result = simulate_episode(policy, scenario, cfg, verbose=True)
