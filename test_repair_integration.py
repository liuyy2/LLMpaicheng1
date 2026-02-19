"""
3 个关键集成测试（子 Prompt 6 交付物）

Case 1: range_closure 清空旧窗口 → Op6 anchor 必须自动跳过
Case 2: pad_outage 覆盖旧 Op4 区间 → Op4 anchor 跳过
Case 3: LLM 输出 unlock_set 超长/包含不存在任务 → fallback 启发式且能跑通
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(name)s %(levelname)s %(message)s")

from solver_cpsat import (
    Mission, Operation, Resource, PlanV2_1, OpAssignment,
    SolverConfigV2_1, SolveStatus, solve_v2_1,
    _check_anchor_feasibility, compute_frozen_ops,
)
from policies.policy_llm_repair import (
    RepairDecision, RepairStepLog, FallbackChainResult,
    validate_repair_decision, heuristic_repair_decision,
    solve_with_fallback_chain, build_repair_step_log,
)


# ======================= helpers =======================

def make_mission(mid, release=0, due=200, priority=1.0,
                 op4_dur=4, op6_dur=2, op6_windows=None):
    ops = []
    preds = []
    for idx in range(1, 7):
        if idx == 1:   dur, res = 2, ["R1"]
        elif idx == 2: dur, res = 2, ["R2"]
        elif idx == 3: dur, res = 3, ["R3"]
        elif idx == 4: dur, res = op4_dur, ["R_pad", "R4"]
        elif idx == 5: dur, res = 0, ["R_pad"]
        elif idx == 6: dur, res = op6_dur, ["R_pad", "R3"]
        else: continue
        op_preds = [f"{mid}_Op{idx-1}"] if idx > 1 else []
        op = Operation(
            op_id=f"{mid}_Op{idx}", mission_id=mid, op_index=idx,
            duration=dur, resources=res, precedences=op_preds,
            time_windows=op6_windows if idx == 6 else [], release=release,
        )
        ops.append(op)
    return Mission(mission_id=mid, operations=ops, release=release,
                   due=due, priority=priority)


def make_resources(pad_unavail=None):
    return [
        Resource("R1", capacity=10), Resource("R2", capacity=10),
        Resource("R3", capacity=2), Resource("R4", capacity=10),
        Resource("R_pad", capacity=1, unavailable=pad_unavail or []),
        Resource("R_range_test", capacity=1),
    ]


# baseline solve to get prev_plan
m1 = make_mission("M001", op6_windows=[(30, 100)])
m2 = make_mission("M002", op6_windows=[(50, 120)])
m3 = make_mission("M003", op6_windows=[(70, 150)])
missions_base = [m1, m2, m3]
resources_clean = make_resources()
cfg = SolverConfigV2_1(horizon_slots=200, time_limit_seconds=10,
                        use_two_stage=False, epsilon_solver=0.05)

r_base = solve_v2_1(missions_base, resources_clean, horizon=200, config=cfg)
assert r_base.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
prev_plan = r_base.plan
print(f"Baseline plan: {[(a.op_id, a.start_slot, a.end_slot) for a in prev_plan.op_assignments if 'Op4' in a.op_id or 'Op6' in a.op_id]}")


# ===========================================================================
# Case 1: range_closure 清空旧 Op6 窗口 → anchor 必须自动跳过
# ===========================================================================
print("\n=== Case 1: range_closure → Op6 anchor skip ===")

prev_op6_m1 = prev_plan.get_assignment("M001_Op6")
print(f"  prev M001 Op6: start={prev_op6_m1.start_slot}, end={prev_op6_m1.end_slot}")

# 模拟 range_closure：新窗口完全不覆盖旧 Op6 位置
# 把 M001 的 Op6 window 移到远处
old_start = prev_op6_m1.start_slot
m1_shifted = make_mission("M001", op6_windows=[(old_start + 30, old_start + 60)])
m2_keep = make_mission("M002", op6_windows=[(50, 120)])
m3_keep = make_mission("M003", op6_windows=[(70, 150)])
missions_c1 = [m1_shifted, m2_keep, m3_keep]

# anchor check: unlock only M003 → M001 & M002 should be anchored
# but M001's Op6 window moved → anchor infeasible → skip
anchor_fixes, skipped, anchored_ids = _check_anchor_feasibility(
    missions_c1, resources_clean, prev_plan,
    unlock_mission_ids={"M003"},
    frozen_ops={}, now=0, horizon=200, op5_max_wait_slots=144,
)

assert "M001_Op6" not in anchor_fixes, \
    f"Case1 FAIL: M001_Op6 should be skipped (window moved), but in anchor_fixes"
assert skipped >= 1, f"Case1 FAIL: expected >=1 skipped, got {skipped}"
print(f"  anchor_fixes keys: {list(anchor_fixes.keys())}")
print(f"  skipped: {skipped}")

# 现在通过 solve_with_fallback_chain 端到端验证
decision_c1 = RepairDecision(
    root_cause_mission_id="M003",
    unlock_mission_ids=["M003"],
    freeze_horizon_hours=8,
    epsilon_solver=0.05,
    analysis_short="range_closure test",
)
chain_c1 = solve_with_fallback_chain(
    decision=decision_c1,
    trcg_dict={'top_conflicts': [], 'conflict_clusters': [],
               'urgent_missions': [], 'bottleneck_pressure': {'pad_util': 0.5},
               'disturbance_summary': {}},
    missions=missions_c1, resources=resources_clean, horizon=200,
    prev_plan=prev_plan, frozen_ops={}, now=0,
    eligible_ids={"M001", "M002", "M003"},
    solver_config_base=cfg,
    current_plan_for_refreeze=prev_plan,
)

# 构建 step log
log_c1 = build_repair_step_log(
    now_slot=0, trcg_dict={'top_conflicts': [], 'urgent_missions': [],
                            'bottleneck_pressure': {'pad_util': 0.5}},
    decision=decision_c1, decision_source="llm",
    chain_result=chain_c1,
)
assert log_c1.anchor_fix_skipped >= 1, \
    f"Case1 log: expected anchor_fix_skipped >=1, got {log_c1.anchor_fix_skipped}"
print(f"  log.anchor_fix_skipped={log_c1.anchor_fix_skipped}, "
      f"solver_status={log_c1.solver_status}")
print("  Case 1 PASS")


# ===========================================================================
# Case 2: pad_outage 覆盖旧 Op4 区间 → Op4 anchor 跳过
# ===========================================================================
print("\n=== Case 2: pad_outage → Op4 anchor skip ===")

prev_op4_m2 = prev_plan.get_assignment("M002_Op4")
print(f"  prev M002 Op4: start={prev_op4_m2.start_slot}, end={prev_op4_m2.end_slot}")

# 放一个 pad outage 恰好覆盖 M002 Op4 的区间
outage_start = prev_op4_m2.start_slot
outage_end = prev_op4_m2.end_slot
resources_outage = make_resources(pad_unavail=[(outage_start, outage_end)])

# anchor check: unlock M001 → M002 & M003 need anchor
# M002's Op4 overlaps outage → skip
anchor_fixes_c2, skipped_c2, anchored_ids_c2 = _check_anchor_feasibility(
    missions_base, resources_outage, prev_plan,
    unlock_mission_ids={"M001"},
    frozen_ops={}, now=0, horizon=200, op5_max_wait_slots=144,
)

assert "M002_Op4" not in anchor_fixes_c2, \
    f"Case2 FAIL: M002_Op4 should be skipped (pad outage)"
assert skipped_c2 >= 1
print(f"  anchor_fixes keys: {list(anchor_fixes_c2.keys())}")
print(f"  skipped: {skipped_c2}")

# 端到端 fallback chain
decision_c2 = RepairDecision(
    root_cause_mission_id="M001",
    unlock_mission_ids=["M001"],
    freeze_horizon_hours=4,
    epsilon_solver=0.02,
    analysis_short="pad_outage test",
)
chain_c2 = solve_with_fallback_chain(
    decision=decision_c2,
    trcg_dict={'top_conflicts': [{'a': 'M001', 'b': 'M002', 'resource': 'R_pad',
                                   'severity': 5.0}],
               'conflict_clusters': [], 'urgent_missions': [],
               'bottleneck_pressure': {'pad_util': 0.7},
               'disturbance_summary': {}},
    missions=missions_base, resources=resources_outage, horizon=200,
    prev_plan=prev_plan, frozen_ops={}, now=0,
    eligible_ids={"M001", "M002", "M003"},
    solver_config_base=cfg,
    current_plan_for_refreeze=prev_plan,
)

log_c2 = build_repair_step_log(
    now_slot=0, decision=decision_c2, decision_source="llm",
    chain_result=chain_c2,
)
print(f"  chain success={chain_c2.success}, final={chain_c2.final_attempt_name}, "
      f"calls={chain_c2.total_solver_calls}")
print(f"  log.anchor_fix_skipped={log_c2.anchor_fix_skipped}")
print("  Case 2 PASS")


# ===========================================================================
# Case 3: LLM 输出 unlock_set 超长 / 包含不存在任务 → fallback 启发式能跑通
# ===========================================================================
print("\n=== Case 3: Invalid LLM output → heuristic fallback → solve OK ===")

active = {"M001", "M002", "M003"}
started = set()
completed = set()

# 3a: unlock_set 超长（6 个，上限 5）
bad_llm_output_a = json.dumps({
    "root_cause_mission_id": "M001",
    "unlock_mission_ids": ["M001", "M002", "M003", "M004", "M005", "M006"],
    "freeze_horizon_hours": 8,
    "epsilon_solver": 0.05,
    "analysis_short": "too many unlocks",
    "secondary_root_cause_mission_id": None,
})
vr_a = validate_repair_decision(bad_llm_output_a, active, started, completed)
assert not vr_a.is_valid, "Case3a: expected validation FAIL for len>5"
print(f"  3a validation errors: {vr_a.errors[:2]}")

# 3b: unlock_set 包含不存在的 mission
bad_llm_output_b = json.dumps({
    "root_cause_mission_id": "M001",
    "unlock_mission_ids": ["M001", "M999"],
    "freeze_horizon_hours": 8,
    "epsilon_solver": 0.05,
    "analysis_short": "ghost mission",
    "secondary_root_cause_mission_id": None,
})
vr_b = validate_repair_decision(bad_llm_output_b, active, started, completed)
assert not vr_b.is_valid, "Case3b: expected validation FAIL for M999"
print(f"  3b validation errors: {vr_b.errors[:2]}")

# 3c: 验证回退路径：validate fails → heuristic_repair_decision → solve OK
trcg_c3 = {
    'top_conflicts': [
        {'a': 'M001', 'b': 'M002', 'resource': 'R_pad',
         'overlap_slots': 3, 'severity': 5.0},
    ],
    'conflict_clusters': [
        {'center_mission_id': 'M001', 'members': ['M001', 'M002'], 'score': 5.0},
    ],
    'urgent_missions': [
        {'mission_id': 'M001', 'due_slot': 120, 'urgency_score': 50.0},
    ],
    'bottleneck_pressure': {'pad_util': 0.6, 'r3_util': 0.3},
    'disturbance_summary': {'pad_outage_active': False},
}

# Simulate: LLM failed → fallback to heuristic
heuristic_dec = heuristic_repair_decision(
    trcg_c3, active, started, completed, fallback_reason="llm_validation_failed"
)
# Validate the heuristic output
vr_heuristic = validate_repair_decision(
    json.dumps(heuristic_dec.to_dict()), active, started, completed
)
assert vr_heuristic.is_valid, f"Case3c: heuristic output invalid: {vr_heuristic.errors}"
print(f"  3c heuristic decision: root={heuristic_dec.root_cause_mission_id}, "
      f"unlock={heuristic_dec.unlock_mission_ids}")

# Now solve with this heuristic decision
chain_c3 = solve_with_fallback_chain(
    decision=heuristic_dec,
    trcg_dict=trcg_c3,
    missions=missions_base, resources=resources_clean, horizon=200,
    prev_plan=prev_plan, frozen_ops={}, now=0,
    eligible_ids=active,
    solver_config_base=cfg,
    current_plan_for_refreeze=prev_plan,
)
assert chain_c3.success, f"Case3c: expected solve success, got {chain_c3.final_attempt_name}"
print(f"  3c solve: success={chain_c3.success}, final={chain_c3.final_attempt_name}")

# Build full step log for the LLM-failed→heuristic scenario
log_c3 = build_repair_step_log(
    now_slot=48,
    trcg_dict=trcg_c3,
    llm_raw_output=bad_llm_output_b,
    llm_http_ok=True,           # HTTP request succeeded
    llm_parse_ok=False,         # schema validation failed
    llm_decision_ok=False,      # decision did not come from LLM
    llm_error={"error_type": "validation_failed", "message": "unlock contains M999"},
    decision=heuristic_dec,
    decision_source="heuristic_fallback",
    fallback_reason="llm_validation_failed",
    chain_result=chain_c3,
    wall_clock_ms=150,
)

# Verify log fields
assert log_c3.decision_source == "heuristic_fallback"
assert log_c3.fallback_reason == "llm_validation_failed"
assert log_c3.now_slot == 48
assert log_c3.llm_http_ok is True
assert log_c3.llm_parse_ok is False
assert log_c3.llm_decision_ok is False
assert log_c3.llm_error["error_type"] == "validation_failed"
assert len(log_c3.llm_raw_output) > 0
assert log_c3.solver_status in ("OPTIMAL", "FEASIBLE")
assert log_c3.trcg_pressure.get('pad_util') == 0.6
assert "M001" in log_c3.trcg_urgent_ids

# Verify serialization roundtrip
log_dict = log_c3.to_dict()
log_json = log_c3.to_json()
assert isinstance(log_dict, dict)
assert isinstance(json.loads(log_json), dict)
print(f"  3c log JSON preview (first 200 chars): {log_json[:200]}...")
print("  Case 3 PASS")


print("\n=== All 3 key integration tests PASSED ===")
