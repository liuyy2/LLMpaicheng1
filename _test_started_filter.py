"""Quick test: verify _trcg_find_urgent filters started missions."""
from features import build_trcg_summary, _trcg_find_urgent
from scenario import generate_scenario
from config import DEFAULT_CONFIG

scenario = generate_scenario(seed=42)
# Simulate some started ops
started_ops = {scenario.missions[0].operations[0].op_id}
completed_ops = set()

trcg = build_trcg_summary(
    missions=scenario.missions,
    resources=scenario.resources,
    plan=None,
    now=0,
    config=DEFAULT_CONFIG,
    started_ops=started_ops,
    completed_ops=completed_ops,
    actual_durations={},
    frozen_ops={},
)
d = trcg.to_dict()
started_mid = scenario.missions[0].mission_id
urgent_mids = [u["mission_id"] for u in d["urgent_missions"]]
print(f"Started mission: {started_mid}")
print(f"Urgent missions: {urgent_mids}")
assert started_mid not in urgent_mids, "FAIL: started mission still in urgent!"
print("PASS: _trcg_find_urgent correctly filters started missions")

# Test without started_ops — should include mission 0
trcg2 = build_trcg_summary(
    missions=scenario.missions,
    resources=scenario.resources,
    plan=None,
    now=0,
    config=DEFAULT_CONFIG,
    started_ops=set(),
    completed_ops=set(),
    actual_durations={},
    frozen_ops={},
)
d2 = trcg2.to_dict()
urgent_mids2 = [u["mission_id"] for u in d2["urgent_missions"]]
print(f"Without filter, urgent: {urgent_mids2}")

# Test _sanitize_trcg_for_llm
import sys
sys.path.insert(0, "policies")
from policy_llm_trcg_repair import TRCGRepairPolicy

# Fake trcg_dict with started missions in conflicts/clusters/urgent
fake_dict = {
    "urgent_missions": [
        {"mission_id": "M001"}, {"mission_id": "M002"}, {"mission_id": "M003"}
    ],
    "top_conflicts": [
        {"a": "M001", "b": "M002", "resource": "R_pad", "severity": 10},
        {"a": "M003", "b": "M004", "resource": "R3", "severity": 5},
    ],
    "conflict_clusters": [
        {"center_mission_id": "M001", "members": ["M001", "M002"], "score": 10},
        {"center_mission_id": "M003", "members": ["M003", "M004"], "score": 5},
    ],
}
started = {"M001"}
sanitized = TRCGRepairPolicy._sanitize_trcg_for_llm(fake_dict, started)

assert "M001" not in [u["mission_id"] for u in sanitized["urgent_missions"]]
assert all(c["a"] != "M001" and c["b"] != "M001" for c in sanitized["top_conflicts"])
assert all(cl["center_mission_id"] != "M001" for cl in sanitized["conflict_clusters"])
print("PASS: _sanitize_trcg_for_llm correctly removes started missions")
print(f"  Urgent: {[u['mission_id'] for u in sanitized['urgent_missions']]}")
print(f"  Conflicts: {sanitized['top_conflicts']}")
print(f"  Clusters: {sanitized['conflict_clusters']}")
print("\nAll tests passed!")
