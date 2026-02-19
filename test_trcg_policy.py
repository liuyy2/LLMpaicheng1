#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRCGRepairPolicy 端到端集成测试

测试内容：
1. 基础导入 & MetaParams 向后兼容
2. TRCGRepairPolicy.decide() 返回正确 MetaParams
3. simulator 完整 episode 运行（纯启发式模式，无 LLM）
4. 与 FixedWeightPolicy 对比（验证不崩）
5. 回退链 simulator 侧触发验证
"""
import sys
import os
import json
import time

sys.path.insert(0, '.')

from config import Config, DEFAULT_CONFIG
from scenario import generate_scenario
from simulator import simulate_episode, save_episode_logs
from policies.base import MetaParams
from policies import (
    FixedWeightPolicy, TRCGRepairPolicy,
    create_policy, AVAILABLE_POLICIES,
)


def test_1_metaparams_backward_compat():
    """MetaParams 新字段不影响旧策略。"""
    print("=== Test 1: MetaParams backward compatibility ===")
    # 旧风格构造
    mp_old = MetaParams(w_delay=10.0, w_shift=1.0, w_switch=5.0)
    assert mp_old.unlock_mission_ids is None
    assert mp_old.decision_source == "default"
    assert mp_old.attempt_idx == 0
    assert mp_old.to_weights() == (10.0, 1.0, 5.0)

    # 新风格构造
    mp_new = MetaParams(
        w_delay=10.0, w_shift=1.0, w_switch=5.0,
        unlock_mission_ids=("M001", "M002"),
        root_cause_mission_id="M001",
        decision_source="llm",
    )
    assert mp_new.unlock_mission_ids == ("M001", "M002")
    assert mp_new.root_cause_mission_id == "M001"
    assert mp_new.decision_source == "llm"
    print("  PASS")


def test_2_decide_returns_correct_meta():
    """TRCGRepairPolicy.decide() 输出合法 MetaParams。"""
    print("\n=== Test 2: decide() returns correct MetaParams ===")
    scenario = generate_scenario(seed=42)
    policy = TRCGRepairPolicy(policy_name="trcg_test")
    config = DEFAULT_CONFIG

    # 模拟 state
    from simulator import _create_initial_state_ops
    state = _create_initial_state_ops(scenario.missions, scenario.resources)
    state.now = 0

    meta, direct_plan = policy.decide(state, 0, config)

    assert direct_plan is None, "direct_plan must be None"
    assert meta is not None, "meta must not be None"
    assert isinstance(meta, MetaParams)
    assert meta.use_two_stage == True
    assert meta.decision_source in ("llm", "heuristic_fallback"), \
        f"unexpected source: {meta.decision_source}"
    # 无 LLM client 时应走启发式
    assert meta.decision_source == "heuristic_fallback"

    # unlock_mission_ids 应为 tuple
    if meta.unlock_mission_ids is not None:
        assert isinstance(meta.unlock_mission_ids, tuple)
        assert len(meta.unlock_mission_ids) >= 1
        assert len(meta.unlock_mission_ids) <= 5

    print(f"  source={meta.decision_source}")
    print(f"  unlock={meta.unlock_mission_ids}")
    print(f"  root={meta.root_cause_mission_id}")
    print(f"  freeze={meta.freeze_horizon} eps={meta.epsilon_solver}")
    print(f"  fallback_reason={meta.fallback_reason}")

    stats = policy.get_stats()
    assert stats['heuristic_count'] == 1
    assert stats['llm_ok_count'] == 0
    print("  PASS")


def test_3_full_episode_heuristic():
    """完整 episode 仿真（纯启发式模式）。"""
    print("\n=== Test 3: Full episode simulation (heuristic mode) ===")
    scenario = generate_scenario(seed=42)
    policy = TRCGRepairPolicy(
        policy_name="trcg_test_ep",
        log_dir="llm_logs/test_trcg",
        enable_logging=True,
        episode_id="test42",
    )
    config = DEFAULT_CONFIG

    t0 = time.time()
    result = simulate_episode(policy, scenario, config, verbose=False)
    elapsed = time.time() - t0

    print(f"  Runtime: {elapsed:.2f}s")
    print(f"  Completed: {result.metrics.num_completed}/{result.metrics.num_total}")
    print(f"  On-time: {result.metrics.on_time_rate:.2%}")
    print(f"  Drift: {result.metrics.episode_drift:.4f}")
    print(f"  Snapshots: {len(result.snapshots)}")

    # 验证不崩
    assert result.metrics.num_completed > 0, "Should complete at least some missions"
    assert len(result.snapshots) > 0, "Should have rolling snapshots"

    # 验证 snapshot 中 meta_params 包含 TRCG 字段
    for snap in result.snapshots[:3]:
        d = snap.to_dict()
        if d['meta_params']:
            assert 'decision_source' in d['meta_params']
            assert 'attempt_idx' in d['meta_params']
            break

    # 验证 policy 统计
    stats = policy.get_stats()
    print(f"  Policy stats: {stats}")
    assert stats['call_count'] > 0
    # 无 LLM → 走启发式 + 状态稳定跳过 + Plan A 跳过
    # heuristic + stable_skip + skip 应覆盖几乎所有步
    effective_decisions = (
        stats['heuristic_count']
        + stats.get('stable_skip_count', 0)
        + stats.get('skip_count', 0)
    )
    assert effective_decisions >= stats['call_count'] - 1

    # 验证日志
    logs = policy.get_step_logs()
    assert len(logs) == stats['call_count']
    sample_log = logs[0]
    # 首步可能是 heuristic_fallback 或 skip_no_conflict（取决于 TRCG 压力）
    assert sample_log.decision_source in (
        "heuristic_fallback", "skip_no_conflict", "skip_stable_state"
    )
    assert sample_log.now_slot == 0
    print(f"  Step logs: {len(logs)} entries")
    print("  PASS")


def test_4_compare_with_fixed():
    """与 FixedWeightPolicy 对比运行（验证数据通路完整性）。"""
    print("\n=== Test 4: Compare TRCGRepair vs Fixed ===")
    scenario = generate_scenario(seed=123)
    config = DEFAULT_CONFIG

    # Fixed baseline
    policy_fixed = FixedWeightPolicy(
        w_delay=10.0, w_shift=1.0, w_switch=5.0
    )
    result_fixed = simulate_episode(policy_fixed, scenario, config, verbose=False)

    # TRCG Repair
    # 重新生成场景（深拷贝 scenario 在 simulate_episode 中已 deepcopy missions）
    scenario2 = generate_scenario(seed=123)
    policy_trcg = TRCGRepairPolicy(policy_name="trcg_compare")
    result_trcg = simulate_episode(policy_trcg, scenario2, config, verbose=False)

    print(f"  Fixed:  completed={result_fixed.metrics.num_completed}/{result_fixed.metrics.num_total} "
          f"drift={result_fixed.metrics.episode_drift:.4f}")
    print(f"  TRCG:   completed={result_trcg.metrics.num_completed}/{result_trcg.metrics.num_total} "
          f"drift={result_trcg.metrics.episode_drift:.4f}")

    # 两者都应完成任务（不崩）
    assert result_fixed.metrics.num_completed > 0
    assert result_trcg.metrics.num_completed > 0
    print("  PASS (both completed without crash)")


def test_5_create_policy_registry():
    """验证 create_policy 能正确创建 TRCGRepairPolicy。"""
    print("\n=== Test 5: create_policy registry ===")
    assert "trcg_repair" in AVAILABLE_POLICIES

    p = create_policy("trcg_repair")
    assert isinstance(p, TRCGRepairPolicy)
    assert p.name == "trcg_repair"
    print("  PASS")


def test_6_snapshot_serialization():
    """验证 RollingSnapshot 序列化包含 TRCG 字段。"""
    print("\n=== Test 6: Snapshot serialization with TRCG fields ===")
    scenario = generate_scenario(seed=55)
    policy = TRCGRepairPolicy(policy_name="trcg_serial")
    config = DEFAULT_CONFIG

    result = simulate_episode(policy, scenario, config, verbose=False)

    # 序列化第一个 snapshot
    snap_dict = result.snapshots[0].to_dict()
    mp = snap_dict.get('meta_params')
    assert mp is not None, "meta_params should not be None"

    # 新字段存在
    assert 'decision_source' in mp
    assert 'fallback_reason' in mp
    assert 'attempt_idx' in mp
    assert mp['decision_source'] in ("heuristic_fallback", "llm", "forced_global", "default")

    # 完整 JSON 序列化不报错
    json_str = json.dumps(snap_dict, ensure_ascii=False, default=str)
    assert len(json_str) > 0

    print(f"  Snapshot[0] meta: source={mp['decision_source']}, "
          f"attempt={mp['attempt_idx']}")
    if 'unlock_mission_ids' in mp:
        print(f"  unlock={mp['unlock_mission_ids']}")
    print("  PASS")


if __name__ == "__main__":
    test_1_metaparams_backward_compat()
    test_2_decide_returns_correct_meta()
    test_3_full_episode_heuristic()
    test_4_compare_with_fixed()
    test_5_create_policy_registry()
    test_6_snapshot_serialization()
    print("\n" + "="*60)
    print(" All 6 tests PASSED ")
    print("="*60)
