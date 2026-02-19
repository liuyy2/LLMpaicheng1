"""验证 V2.5 solver 建模问题"""
from config import make_config_for_difficulty, MISSIONS_BY_DIFFICULTY
from scenario import generate_scenario

seed = 71
cfg = make_config_for_difficulty("light")
cfg.sim_num_missions = MISSIONS_BY_DIFFICULTY["light"]
scenario = generate_scenario(seed, cfg)

# 检查 Op 结构
m0 = scenario.missions[0]
print("=== Mission 0 的 Op 结构 ===")
for op in m0.operations:
    tw_str = f" time_windows={op.time_windows}" if op.time_windows else " (no windows)"
    prec_str = f" prec={op.precedences}" if op.precedences else ""
    print(f"  op_index={op.op_index} op_id={op.op_id} dur={op.duration} res={op.resources}{tw_str}{prec_str}")

print()
print("=== get_operation 行为测试 ===")
for idx in range(1, 8):
    op = m0.get_operation(idx)
    if op:
        print(f"  get_operation({idx}) => op_id={op.op_id} op_index={op.op_index} dur={op.duration} windows={bool(op.time_windows)}")
    else:
        print(f"  get_operation({idx}) => None")

# 验证 solver 建模：检查 op_index==5 弹性工期被加在哪
print()
print("=== Solver 建模验证 ===")
from solver_cpsat import SolverConfigV2_1

# 检查 op_index==5 弹性工期
for m in scenario.missions[:3]:
    op5_by_idx = m.get_operation(5)
    op6_by_idx = m.get_operation(6)
    op7_by_idx = m.get_operation(7)
    
    print(f"\n{m.mission_id}:")
    if op5_by_idx:
        print(f"  op_index=5: {op5_by_idx.op_id} dur={op5_by_idx.duration} res={op5_by_idx.resources}")
        print(f"    -> Solver gives elastic duration to THIS op (WRONG: this is pad hold, not wait)")
    if op6_by_idx:
        print(f"  op_index=6: {op6_by_idx.op_id} dur={op6_by_idx.duration} windows={bool(op6_by_idx.time_windows)}")
        print(f"    -> Solver gets 'op6' = THIS (wait/dummy), windows={op6_by_idx.time_windows}")
        print(f"    -> Solver checks op6.time_windows: {'SKIPPED (no windows)' if not op6_by_idx.time_windows else 'APPLIED'}")
    if op7_by_idx:
        print(f"  op_index=7: {op7_by_idx.op_id} dur={op7_by_idx.duration} windows={op7_by_idx.time_windows}")
        print(f"    -> Solver NEVER checks this op's time_windows (not referenced as 'op6')")

# 验证 contiguity: solver 强制 op4->op5->op6 连续
print()
print("=== Contiguity Constraint 验证 ===")
for m in scenario.missions[:2]:
    op4g = m.get_operation(4)  # V2.5: range test
    op5g = m.get_operation(5)  # V2.5: pad hold
    op6g = m.get_operation(6)  # V2.5: wait dummy
    op7g = m.get_operation(7)  # V2.5: launch
    print(f"\n{m.mission_id}:")
    print(f"  Solver creates: start[{op5g.op_id}(pad)] == end[{op4g.op_id}(range_test)]  <- WRONG")
    print(f"  Solver creates: start[{op6g.op_id}(wait)] == end[{op5g.op_id}(pad)]  <- CORRECT (coincidence)")
    print(f"  MISSING: start[{op7g.op_id}(launch)] == end[{op6g.op_id}(wait)]")
    
    # 检查是否有 precedence 链接 op6->op7
    if op7g.precedences:
        print(f"  BUT: op7 has precedences={op7g.precedences}")
        has_op6_prec = any(p == op6g.op_id for p in op7g.precedences)
        print(f"  Has precedence from wait(op6)? {has_op6_prec}")
        if has_op6_prec:
            print(f"  -> start[op7] >= end[op6] enforced by 通用 precedence (但不是严格等式)")
