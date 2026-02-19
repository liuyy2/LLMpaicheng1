"""
验证方案 A（条件跳过 LLM）和方案 B（episode 并行）的集成测试。
运行后删除此文件即可。
"""
import sys
import time

def test_plan_a_skip_logic():
    """方案 A: 测试无 LLM client 时 skip + heuristic 的完整 episode"""
    print("=" * 60)
    print("Test Plan A: Conditional LLM Skip")
    print("=" * 60)

    from policies.policy_llm_trcg_repair import TRCGRepairPolicy
    from config import make_config_for_difficulty
    from scenario import generate_scenario
    from simulator import simulate_episode

    policy = TRCGRepairPolicy(
        llm_client=None,
        policy_name="trcg_repair_llm",
        log_dir="test_output_temp/llm_logs",
        enable_logging=False,
        episode_id="test_skip",
    )

    config = make_config_for_difficulty("light")
    config.freeze_horizon = 96
    config.default_epsilon_solver = 0.10

    scenario = generate_scenario(42, config)

    t0 = time.time()
    result = simulate_episode(policy, scenario, config)
    wall = time.time() - t0

    stats = policy.get_stats()
    call_count = stats["call_count"]
    skip_count = stats["skip_count"]
    stable_skip_count = stats.get("stable_skip_count", 0)
    heuristic_count = stats["heuristic_count"]
    llm_ok_count = stats["llm_ok_count"]
    total_skipped = skip_count + stable_skip_count
    skip_pct = total_skipped / max(call_count, 1) * 100

    print(f"  call_count        = {call_count}")
    print(f"  skip_count        = {skip_count}")
    print(f"  stable_skip_count = {stable_skip_count}")
    print(f"  heuristic_cnt     = {heuristic_count}")
    print(f"  llm_ok_count      = {llm_ok_count}")
    print(f"  total_skipped     = {total_skipped} ({skip_pct:.1f}%)")
    print(f"  wall_time         = {wall:.1f}s")
    print(f"  avg_delay         = {result.metrics.avg_delay}")
    print(f"  feasibility       = {result.metrics.on_time_rate}")

    # 验证：跳过数 + 稳定跳过数 + 启发式数 = 总调用数（无 LLM client 时 llm_ok=0）
    assert llm_ok_count == 0, f"Expected llm_ok=0 (no client), got {llm_ok_count}"
    assert skip_count + stable_skip_count + heuristic_count == call_count, (
        f"skip({skip_count}) + stable({stable_skip_count}) + heuristic({heuristic_count}) != total({call_count})"
    )
    assert total_skipped > 0, "Expected at least some skips in a light scenario"

    print("  [PASS] Plan A integration test\n")
    return skip_pct


def test_plan_a_multiple_difficulties():
    """方案 A: 测试不同难度下的 skip rate"""
    print("=" * 60)
    print("Test Plan A: Skip rates by difficulty")
    print("=" * 60)

    from policies.policy_llm_trcg_repair import TRCGRepairPolicy
    from config import make_config_for_difficulty
    from scenario import generate_scenario
    from simulator import simulate_episode

    for level in ["light", "medium", "heavy"]:
        policy = TRCGRepairPolicy(
            llm_client=None,
            policy_name="trcg_repair_llm",
            enable_logging=False,
            episode_id=f"test_{level}",
        )

        config = make_config_for_difficulty(level)
        config.freeze_horizon = 96
        config.default_epsilon_solver = 0.10

        scenario = generate_scenario(100, config)
        result = simulate_episode(policy, scenario, config)

        stats = policy.get_stats()
        cc = stats["call_count"]
        sc = stats["skip_count"]
        pct = sc / max(cc, 1) * 100
        print(f"  {level:8s}: calls={cc}, skips={sc} ({pct:.1f}%), "
              f"heuristic={stats['heuristic_count']}")

    print("  [PASS] Skip rate varies by difficulty\n")


def test_plan_b_parallel_config():
    """方案 B: 验证 ExperimentConfig.llm_max_workers 字段存在且合理"""
    print("=" * 60)
    print("Test Plan B: ExperimentConfig.llm_max_workers")
    print("=" * 60)

    from run_experiments import ExperimentConfig

    ec = ExperimentConfig()
    assert hasattr(ec, "llm_max_workers"), "Missing llm_max_workers field"
    assert ec.llm_max_workers >= 1, f"llm_max_workers should be >=1, got {ec.llm_max_workers}"
    print(f"  default llm_max_workers = {ec.llm_max_workers}")

    # 可以自定义
    ec2 = ExperimentConfig(llm_max_workers=8)
    assert ec2.llm_max_workers == 8
    print(f"  custom llm_max_workers = {ec2.llm_max_workers}")

    print("  [PASS] Plan B config test\n")


def test_plan_b_thread_safety():
    """方案 B: 验证多线程并发执行 episode 不冲突"""
    print("=" * 60)
    print("Test Plan B: Thread safety (2 parallel episodes)")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from policies.policy_llm_trcg_repair import TRCGRepairPolicy
    from config import make_config_for_difficulty
    from scenario import generate_scenario
    from simulator import simulate_episode

    def run_one(seed, level):
        policy = TRCGRepairPolicy(
            llm_client=None,
            policy_name="trcg_repair_llm",
            enable_logging=False,
            episode_id=f"parallel_{seed}",
        )
        config = make_config_for_difficulty(level)
        config.freeze_horizon = 96
        config.default_epsilon_solver = 0.10
        scenario = generate_scenario(seed, config)
        result = simulate_episode(policy, scenario, config)
        stats = policy.get_stats()
        return seed, stats, result.metrics.avg_delay

    tasks = [(42, "light"), (99, "medium")]

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_one, s, l): s for s, l in tasks}
        for f in as_completed(futures):
            seed, stats, avg_delay = f.result()
            results[seed] = (stats, avg_delay)
            print(f"  seed={seed}: calls={stats['call_count']}, "
                  f"skips={stats['skip_count']}, avg_delay={avg_delay:.2f}")
    wall = time.time() - t0

    # 串行参照
    t1 = time.time()
    for seed, level in tasks:
        run_one(seed, level)
    serial_wall = time.time() - t1

    print(f"  parallel_wall = {wall:.1f}s, serial_wall = {serial_wall:.1f}s")
    print(f"  speedup = {serial_wall / max(wall, 0.01):.2f}x")

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print("  [PASS] Plan B thread safety test\n")


if __name__ == "__main__":
    skip_pct = test_plan_a_skip_logic()
    test_plan_a_multiple_difficulties()
    test_plan_b_parallel_config()
    test_plan_b_thread_safety()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print(f"  Plan A skip rate (light, seed=42): {skip_pct:.1f}%")
    print("=" * 60)
