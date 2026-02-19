"""
快速测试 INFEASIBLE fallback 机制。
使用 heavy (25 missions) + seed=100（已知该组合导致 CP-SAT INFEASIBLE）。
"""
import time
from config import make_config_for_difficulty
from scenario import generate_scenario
from simulator import simulate_episode
from policies.policy_fixed import FixedWeightPolicy
from metrics import metrics_to_dict

def main():
    seed = 100
    difficulty = "heavy"
    config = make_config_for_difficulty(difficulty=difficulty)
    scenario = generate_scenario(seed=seed, config=config)

    # 使用 fixed_default baseline (与之前实验相同)
    policy = FixedWeightPolicy(
        w_delay=1.0,
        w_shift=0.0,
        w_switch=0.0,
        freeze_horizon=0,
        use_two_stage=False,
        policy_name="fixed_default"
    )

    print(f"Running episode: seed={seed}, difficulty={difficulty}, "
          f"missions={len(scenario.missions)}, policy={policy.name}")
    print("=" * 60)

    t0 = time.time()
    result = simulate_episode(
        scenario=scenario,
        policy=policy,
        config=config,
        verbose=True,
    )
    elapsed = time.time() - t0

    m = result.metrics
    print("\n" + "=" * 60)
    print(f"RESULT: seed={seed}, difficulty={difficulty}")
    print(f"  completed: {m.num_completed}/{m.num_total} ({m.completion_rate:.2%})")
    print(f"  avg_delay: {m.avg_delay:.2f}")
    print(f"  total_delay: {m.total_delay}")
    print(f"  drift_per_replan: {m.drift_per_replan:.6f}")
    print(f"  on_time_rate: {m.on_time_rate:.2%}")
    print(f"  num_replans: {m.num_replans}")
    print(f"  runtime: {elapsed:.1f}s")

    # 检查 rolling snapshots 中的 solve status 报告
    from solver_cpsat import SolveStatus
    status_counts = {}
    for snap in result.snapshots:
        s = snap.solve_status.name if hasattr(snap.solve_status, 'name') else str(snap.solve_status)
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"\n  Solve status distribution: {status_counts}")

    # 若之前是 completed=16/25, 现在应该接近 25/25
    if m.num_completed >= 20:
        print(f"\n  >>> SUCCESS: fallback mechanism works! ({m.num_completed}/{m.num_total} completed) <<<")
    else:
        print(f"\n  >>> WARNING: only {m.num_completed}/{m.num_total} completed, fallback may not be working <<<")


if __name__ == "__main__":
    main()
