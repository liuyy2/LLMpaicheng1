"""快速测试 greedy 策略是否能正确处理 V2.5 结构"""
import sys
from config import Config, make_config_for_difficulty, MISSIONS_BY_DIFFICULTY
from scenario import generate_scenario
from simulator import simulate_episode
from policies.policy_greedy import GreedyPolicy
from metrics import metrics_to_dict

def test_greedy():
    # 生成轻度扰动场景
    seed = 100
    difficulty = "light"
    
    cfg = make_config_for_difficulty(difficulty)
    cfg.sim_num_missions = MISSIONS_BY_DIFFICULTY[difficulty]
    
    print(f"生成场景: seed={seed}, difficulty={difficulty}, missions={cfg.sim_num_missions}")
    scenario = generate_scenario(seed, cfg)
    
    # 运行 greedy 策略
    policy = GreedyPolicy()
    print(f"\n运行 Greedy 策略...")
    result = simulate_episode(policy, scenario, cfg, verbose=False)
    
    # 输出关键指标
    metrics_dict = metrics_to_dict(result.metrics)
    print(f"\n=== 仿真结果 ===")
    print(f"完成任务数: {len(result.completed_tasks)} / {cfg.sim_num_missions}")
    print(f"平均延迟: {metrics_dict['avg_delay']:.2f} slots")
    print(f"总耗时: {metrics_dict['makespan_cmax']} slots")
    print(f"重规划次数: {metrics_dict['num_replans']}")
    
    # 统计 Op 分配情况
    op_counts = {}
    for assignment in result.final_schedule:
        idx = assignment.op_index
        op_counts[idx] = op_counts.get(idx, 0) + 1
    
    print(f"\n=== Operation 分配统计 ===")
    for idx in sorted(op_counts.keys()):
        print(f"  Op{idx}: {op_counts[idx]} 个分配")
    
    # 验证是否成功调度 Op4-7
    has_op4_7 = any(idx >= 4 for idx in op_counts)
    if has_op4_7:
        print(f"\n✅ 测试通过: Greedy 策略成功调度 Op4-7 (V2.5 支持)")
        return 0
    else:
        print(f"\n❌ 测试失败: Greedy 策略未调度任何 Op4-7")
        return 1

if __name__ == "__main__":
    sys.exit(test_greedy())
