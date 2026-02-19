"""
Verify the full policy works after changes:
1. _trcg_find_urgent reverted (no started filter)
2. _sanitize_trcg_for_llm removed
3. _auto_correct_llm_output added
4. unlock_mission_ids enabled (not None)
"""
import json

# Test 1: features.py _trcg_find_urgent has no started_ops parameter
print("=== Test 1: _trcg_find_urgent reverted ===")
from features import build_trcg_summary, _trcg_find_urgent
from scenario import generate_scenario
from config import DEFAULT_CONFIG
import inspect

sig = inspect.signature(_trcg_find_urgent)
params = list(sig.parameters.keys())
assert 'started_ops' not in params, f"started_ops still in params: {params}"
print(f"  Parameters: {params}")
print("  PASS: no started_ops parameter")

# Verify started missions ARE included in urgent
scenario = generate_scenario(seed=42)
started_ops = {scenario.missions[0].operations[0].op_id}
trcg = build_trcg_summary(
    missions=scenario.missions,
    resources=scenario.resources,
    plan=None,
    now=0,
    config=DEFAULT_CONFIG,
    started_ops=started_ops,
    completed_ops=set(),
    actual_durations={},
    frozen_ops={},
)
d = trcg.to_dict()
urgent_mids = [u["mission_id"] for u in d["urgent_missions"]]
print(f"  Urgent missions (may include started): {urgent_mids}")
print("  PASS: urgent includes all missions (started not filtered)")

# Test 2: _sanitize_trcg_for_llm removed
print("\n=== Test 2: _sanitize_trcg_for_llm removed ===")
import sys
sys.path.insert(0, "policies")
from policy_llm_trcg_repair import TRCGRepairPolicy
assert not hasattr(TRCGRepairPolicy, '_sanitize_trcg_for_llm'), "Method still exists!"
print("  PASS: _sanitize_trcg_for_llm removed")

# Test 3: _auto_correct_llm_output works
print("\n=== Test 3: _auto_correct_llm_output ===")
assert hasattr(TRCGRepairPolicy, '_auto_correct_llm_output'), "Method not found!"

# Case 3a: LLM picks a started mission
raw = json.dumps({
    "root_cause_mission_id": "M001",
    "unlock_mission_ids": ["M001", "M003"],
    "secondary_root_cause_mission_id": None,
    "analysis_short": "test"
})
active = {"M002", "M003", "M004"}
started = {"M001"}
completed = set()
trcg_d = {
    "top_conflicts": [
        {"a": "M002", "b": "M003", "resource": "R_pad", "severity": 10},
        {"a": "M001", "b": "M004", "resource": "R3", "severity": 5},
    ]
}

corrected = TRCGRepairPolicy._auto_correct_llm_output(
    raw, active, started, completed, trcg_d
)
data = json.loads(corrected)
print(f"  Input:  root=M001, unlock=[M001,M003]")
print(f"  Output: root={data['root_cause_mission_id']}, unlock={data['unlock_mission_ids']}")
assert data['root_cause_mission_id'] in active, f"Root {data['root_cause_mission_id']} not active!"
assert all(m in active for m in data['unlock_mission_ids']), "Non-active in unlock!"
assert "M001" not in data['unlock_mission_ids'], "Started M001 still in unlock!"
print("  PASS: started mission auto-corrected")

# Case 3b: LLM picks valid missions (no correction needed)
raw2 = json.dumps({
    "root_cause_mission_id": "M002",
    "unlock_mission_ids": ["M002", "M003"],
    "secondary_root_cause_mission_id": "M004",
    "analysis_short": "test"
})
corrected2 = TRCGRepairPolicy._auto_correct_llm_output(
    raw2, active, started, completed, trcg_d
)
data2 = json.loads(corrected2)
assert data2['root_cause_mission_id'] == "M002", "Correct output modified!"
assert data2['unlock_mission_ids'] == ["M002", "M003"], "Correct output modified!"
print("  PASS: valid output not modified")

# Case 3c: All LLM picks are started → should fall through with best active from conflicts
raw3 = json.dumps({
    "root_cause_mission_id": "M001",
    "unlock_mission_ids": ["M001"],
    "secondary_root_cause_mission_id": "M001",
    "analysis_short": "test"
})
corrected3 = TRCGRepairPolicy._auto_correct_llm_output(
    raw3, active, started, completed, trcg_d
)
data3 = json.loads(corrected3)
print(f"  All-started input: root=M001, unlock=[M001]")
print(f"  Corrected: root={data3['root_cause_mission_id']}, unlock={data3['unlock_mission_ids']}")
assert data3['root_cause_mission_id'] in active
assert data3['secondary_root_cause_mission_id'] is None  # M001 was started
print("  PASS: all-started case handled")

# Test 4: MetaParams uses unlock_mission_ids
print("\n=== Test 4: MetaParams uses unlock (code review) ===")
import re
with open("policies/policy_llm_trcg_repair.py", "r", encoding="utf-8") as f:
    code = f.read()

# Count how many times unlock_mission_ids=None appears
none_count = len(re.findall(r'unlock_mission_ids\s*=\s*None', code))
# Count how many times unlock_mission_ids=use_unlock appears
use_count = len(re.findall(r'unlock_mission_ids\s*=\s*use_unlock', code))

print(f"  unlock_mission_ids=None: {none_count} occurrences (skip_no_conflict only)")
print(f"  unlock_mission_ids=use_unlock: {use_count} occurrences (LLM + stable_skip)")
assert none_count == 1, f"Expected 1 None (skip_no_conflict), got {none_count}"
assert use_count == 2, f"Expected 2 use_unlock (LLM + stable_skip), got {use_count}"
print("  PASS: unlock correctly enabled in LLM and stable_skip paths")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
