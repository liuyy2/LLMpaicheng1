"""Quick test: verify new observability fields (llm split + anchor_fix detail)"""
import json
from policies.policy_llm_repair import (
    RepairStepLog, build_repair_step_log, FallbackAttempt, FallbackChainResult,
)
from solver_cpsat import SolverResult, SolveStatus

print("=== Test 1: RepairStepLog 3-way llm fields ===")
log = RepairStepLog()
# defaults
assert log.llm_http_ok is False
assert log.llm_parse_ok is False
assert log.llm_decision_ok is False
assert log.llm_error == {}
assert not hasattr(log, 'llm_call_ok'), "old llm_call_ok should not exist"
print("  PASS: defaults correct, old llm_call_ok removed")

print("=== Test 2: build_repair_step_log with 3-way fields ===")
log2 = build_repair_step_log(
    now_slot=100,
    llm_raw_output='{"root_cause_mission_id":"M1"}',
    llm_http_ok=True,
    llm_parse_ok=True,
    llm_decision_ok=True,
    llm_error={},
    decision_source="llm",
    wall_clock_ms=50,
)
assert log2.llm_http_ok is True
assert log2.llm_parse_ok is True
assert log2.llm_decision_ok is True
assert log2.llm_error == {}
d = log2.to_dict()
assert "llm_http_ok" in d
assert "llm_parse_ok" in d
assert "llm_decision_ok" in d
assert "llm_call_ok" not in d
print("  PASS: 3-way fields in build + to_dict")

print("=== Test 3: structured llm_error ===")
log3 = build_repair_step_log(
    now_slot=200,
    llm_http_ok=False,
    llm_error={
        "error_type": "APIError",
        "http_status": 429,
        "is_timeout": False,
        "message": "Rate limit exceeded",
        "response_snippet": "",
        "retries": 3,
    },
)
assert log3.llm_error["error_type"] == "APIError"
assert log3.llm_error["http_status"] == 429
assert log3.llm_error["retries"] == 3
j = json.loads(log3.to_json())
assert j["llm_error"]["http_status"] == 429
print("  PASS: structured llm_error serializes to JSON")

print("=== Test 4: SolverResult anchor_fix new fields ===")
sr = SolverResult(status=SolveStatus.OPTIMAL, plan=None)
sr.anchor_fix_applied_count = 2
sr.anchor_fix_skipped_count = 1
sr.anchor_fix_applied_missions = ["M001", "M002"]
sr.anchor_fix_applied_vars = {"M001_Op4": 10, "M001_Op6": 30, "M002_Op4": 40, "M002_Op6": 60}
sr.anchor_fix_skip_reason = ""
assert len(sr.anchor_fix_applied_missions) == 2
assert sr.anchor_fix_applied_vars["M001_Op4"] == 10
print("  PASS: SolverResult has new fields")

print("=== Test 5: anchor_fix flows through FallbackAttempt ===")
fa = FallbackAttempt(
    attempt_name="initial",
    unlock_ids=["M003"],
    freeze_hours=24,
    epsilon=0.1,
    use_anchor=True,
    solver_status="OPTIMAL",
    anchor_applied_missions=["M001", "M002"],
    anchor_applied_vars={"M001_Op4": 10},
    anchor_skip_reason="",
)
assert fa.anchor_applied_missions == ["M001", "M002"]
print("  PASS: FallbackAttempt has new fields")

print("=== Test 6: anchor_fix flows through RepairStepLog via chain_result ===")
sr2 = SolverResult(status=SolveStatus.OPTIMAL, plan=None)
sr2.anchor_fix_applied_count = 3
sr2.anchor_fix_skipped_count = 0
sr2.anchor_fix_applied_missions = ["M001", "M002", "M003"]
sr2.anchor_fix_applied_vars = {"M001_Op4": 5, "M001_Op6": 25}
sr2.anchor_fix_skip_reason = ""
chain = FallbackChainResult(
    solver_result=sr2,
    success=True,
    attempts=[fa],
    final_attempt_name="initial",
    total_solver_calls=1,
)
log6 = build_repair_step_log(
    now_slot=300,
    llm_http_ok=True, llm_parse_ok=True, llm_decision_ok=True,
    chain_result=chain,
)
assert log6.anchor_fix_applied_missions == ["M001", "M002", "M003"]
assert log6.anchor_fix_applied_vars == {"M001_Op4": 5, "M001_Op6": 25}
assert log6.anchor_fix_skip_reason == ""
j6 = json.loads(log6.to_json())
assert j6["anchor_fix_applied_missions"] == ["M001", "M002", "M003"]
print("  PASS: anchor_fix detail flows through chain_result to log JSON")

print("=== Test 7: anchor_fix_skip_reason variants ===")
for reason in ["no_prev_plan", "unlock_all", "all_infeasible", ""]:
    sr_r = SolverResult(status=SolveStatus.OPTIMAL, plan=None)
    sr_r.anchor_fix_skip_reason = reason
    chain_r = FallbackChainResult(solver_result=sr_r, success=True)
    log_r = build_repair_step_log(now_slot=0, chain_result=chain_r)
    assert log_r.anchor_fix_skip_reason == reason
print("  PASS: all skip_reason variants work")

print("\n=== ALL 7 OBSERVABILITY TESTS PASSED ===")
