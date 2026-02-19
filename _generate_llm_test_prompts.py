#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_generate_llm_test_prompts.py

生成 trcg_repair_llm 策略在不同场景下 LLM 面临的真实决策情境。
输出完整的 System Prompt + User Prompt，可直接复制到各模型网页端进行测试。

Usage:
    python _generate_llm_test_prompts.py
    python _generate_llm_test_prompts.py --output llm_test_cases.md
"""

import json
import copy
import argparse
from typing import List, Dict, Any, Set, Optional

from config import Config, DEFAULT_CONFIG, make_config_for_difficulty
from scenario import generate_scenario
from simulator import (
    SimulationStateOps, _create_initial_state_ops,
    _apply_disturbance_ops,
)
from solver_cpsat import (
    solve_v2_1, compute_frozen_ops, SolverConfigV2_1, SolveStatus
)
from features import build_trcg_summary
from policies.policy_llm_repair import (
    REPAIR_SYSTEM_PROMPT,
    build_repair_user_prompt,
)


def simulate_to_step(scenario, config, target_now: int):
    """
    快速模拟到 target_now 时刻，返回 (state, plan)。
    简化版：只做求解+扰动，不执行 op 开始/完成逻辑（足够生成 TRCG）。
    """
    state = _create_initial_state_ops(scenario.missions, scenario.resources)
    state.now = 0

    plan = None
    last_now = 0
    for t in range(0, target_now + 1, config.rolling_interval):
        state.now = t
        horizon_end = t + config.horizon_slots

        # 应用扰动（使用 simulator 内置函数）
        state = _apply_disturbance_ops(state, t, scenario.disturbance_timeline, scenario, last_now)
        last_now = t

        # 求解
        schedulable = state.get_schedulable_missions(horizon_end)
        if not schedulable:
            continue

        frozen_ops = compute_frozen_ops(
            plan, t, config.freeze_horizon,
            state.started_ops, state.completed_ops,
        )

        solver_cfg = SolverConfigV2_1(
            w_delay=config.default_w_delay,
            w_shift=config.default_w_shift,
            w_switch=config.default_w_switch,
            use_two_stage=True,
            epsilon_solver=config.default_epsilon_solver,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
        )

        result = solve_v2_1(
            missions=schedulable,
            resources=state.resources,
            horizon=horizon_end,
            config=solver_cfg,
            prev_plan=plan,
            frozen_ops=frozen_ops,
            now=t,
        )

        if result.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
            plan = result.plan
            state.current_plan = plan

    return state, plan


def build_trcg_for_state(state, config, prev_window_slots=None):
    """从 state 构造 TRCGSummary dict"""
    frozen_ops = compute_frozen_ops(
        state.current_plan, state.now, config.freeze_horizon,
        state.started_ops, state.completed_ops,
    )
    trcg = build_trcg_summary(
        missions=state.missions,
        resources=state.resources,
        plan=state.current_plan,
        now=state.now,
        config=config,
        started_ops=state.started_ops,
        completed_ops=state.completed_ops,
        actual_durations=state.actual_durations,
        frozen_ops=frozen_ops,
        prev_window_slots=prev_window_slots,
    )
    return trcg.to_dict()


def get_active_mission_ids(state, config) -> List[str]:
    """获取当前活跃的 mission IDs"""
    horizon_end = state.now + config.horizon_slots
    started_mission_ids = set()
    completed_mission_ids = set()

    for m in state.missions:
        launch = m.get_launch_op()
        if launch and launch.op_id in state.completed_ops:
            completed_mission_ids.add(m.mission_id)
            continue
        for op in m.operations:
            if op.op_id in state.started_ops:
                started_mission_ids.add(m.mission_id)
                break

    schedulable_ids = {
        m.mission_id for m in state.get_schedulable_missions(horizon_end)
    }
    active = schedulable_ids - started_mission_ids - completed_mission_ids
    if not active:
        active = schedulable_ids - completed_mission_ids
    return sorted(active)


def generate_test_case(seed, difficulty, step_slot, case_id):
    """生成一个完整的测试用例"""
    config = make_config_for_difficulty(difficulty)
    scenario = generate_scenario(seed=seed, config=config)

    state, plan = simulate_to_step(scenario, config, step_slot)
    trcg_dict = build_trcg_for_state(state, config)
    active_ids = get_active_mission_ids(state, config)

    if not active_ids:
        return None

    user_prompt = build_repair_user_prompt(trcg_dict, active_ids)

    return {
        "case_id": case_id,
        "seed": seed,
        "difficulty": difficulty,
        "now_slot": step_slot,
        "now_hours": step_slot * 15 / 60,
        "num_active_missions": len(active_ids),
        "active_mission_ids": active_ids,
        "trcg_dict": trcg_dict,
        "system_prompt": REPAIR_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }


# ============================================================
# 预定义测试场景
# ============================================================
TEST_SCENARIOS = [
    # Case 1: 早期阶段，中等难度，资源压力初现
    {"seed": 42, "difficulty": "medium", "step_slot": 48,
     "label": "早期阶段-资源压力初现"},

    # Case 2: 中期，高负载，资源冲突激烈
    {"seed": 42, "difficulty": "heavy", "step_slot": 192,
     "label": "中期-高负载资源冲突"},

    # Case 3: 中期，轻负载，几乎无冲突
    {"seed": 100, "difficulty": "light", "step_slot": 96,
     "label": "轻负载-低冲突"},

    # Case 4: 扰动密集期，pad outage + 窗口缩减
    {"seed": 55, "difficulty": "heavy", "step_slot": 288,
     "label": "扰动密集-pad故障+窗口缩减"},

    # Case 5: 后期阶段，多任务完成，少量紧迫任务
    {"seed": 77, "difficulty": "medium", "step_slot": 600,
     "label": "后期-少量紧迫任务"},
]


# ============================================================
# 合成的高冲突测试场景（手工构造，覆盖 LLM 推理的关键分支）
# ============================================================

def build_synthetic_cases() -> List[Dict]:
    """
    构造若干包含冲突、冲突簇、pad outage、窗口损失等典型情况的合成 TRCG 数据。
    这些是真实仿真中会出现的结构，只是手工设置了数值以覆盖 LLM 的不同推理路径。
    """
    cases = []

    # ------- 合成 Case A: Pad 资源严重冲突，3 任务争夺同一 Pad -------
    cases.append({
        "case_id": "synthetic_A",
        "label": "Pad资源冲突-3任务竞争（合成）",
        "seed": "N/A (合成数据)",
        "difficulty": "heavy",
        "now_slot": 120,
        "now_hours": 30.0,
        "num_active_missions": 8,
        "active_mission_ids": ["M001", "M003", "M005", "M007", "M008", "M010", "M012", "M015"],
        "trcg_dict": {
            "now_slot": 120,
            "horizon_end_slot": 216,
            "bottleneck_pressure": {"pad_util": 0.95, "r3_util": 0.72, "range_test_util": 0.18},
            "top_conflicts": [
                {"a": "M003", "b": "M005", "resource": "R_pad", "overlap_slots": 14, "t_range": [152, 166], "severity": 23.8},
                {"a": "M003", "b": "M007", "resource": "R_pad", "overlap_slots": 8, "t_range": [158, 166], "severity": 14.4},
                {"a": "M005", "b": "M007", "resource": "R_pad", "overlap_slots": 6, "t_range": [160, 166], "severity": 9.6},
                {"a": "M001", "b": "M010", "resource": "R3", "overlap_slots": 4, "t_range": [140, 144], "severity": 5.2},
            ],
            "conflict_clusters": [
                {"center_mission_id": "M003", "members": ["M003", "M005", "M007"], "score": 38.2},
            ],
            "urgent_missions": [
                {"mission_id": "M003", "due_slot": 200, "due_slack_slots": 80, "window_slack_slots": 32, "current_delay_slots": 12, "priority": 0.85, "urgency_score": 72.0},
                {"mission_id": "M005", "due_slot": 230, "due_slack_slots": 110, "window_slack_slots": 40, "current_delay_slots": 8, "priority": 0.62, "urgency_score": 114.0},
                {"mission_id": "M001", "due_slot": 180, "due_slack_slots": 60, "window_slack_slots": 22, "current_delay_slots": 0, "priority": 0.71, "urgency_score": 71.0},
            ],
            "disturbance_summary": {"range_loss_pct": 0.05, "pad_outage_active": False, "duration_volatility_level": "low"},
            "frozen_summary": {"num_started_ops": 4, "num_frozen_ops": 6, "frozen_horizon_slots": 12},
        },
        "system_prompt": REPAIR_SYSTEM_PROMPT,
        "user_prompt": None,  # 后面生成
    })

    # ------- 合成 Case B: Pad outage 叠加 R3 冲突 + 高窗口损失 -------
    cases.append({
        "case_id": "synthetic_B",
        "label": "Pad故障+R3冲突+窗口缩减（合成）",
        "seed": "N/A (合成数据)",
        "difficulty": "heavy",
        "now_slot": 240,
        "now_hours": 60.0,
        "num_active_missions": 10,
        "active_mission_ids": ["M002", "M004", "M006", "M008", "M009", "M011", "M014", "M016", "M018", "M020"],
        "trcg_dict": {
            "now_slot": 240,
            "horizon_end_slot": 336,
            "bottleneck_pressure": {"pad_util": 1.0, "r3_util": 0.88, "range_test_util": 0.35},
            "top_conflicts": [
                {"a": "M004", "b": "M006", "resource": "R_pad", "overlap_slots": 18, "t_range": [260, 278], "severity": 32.4},
                {"a": "M004", "b": "M009", "resource": "R_pad", "overlap_slots": 10, "t_range": [268, 278], "severity": 17.0},
                {"a": "M006", "b": "M009", "resource": "R_pad", "overlap_slots": 7, "t_range": [271, 278], "severity": 11.9},
                {"a": "M011", "b": "M016", "resource": "R3", "overlap_slots": 12, "t_range": [280, 292], "severity": 22.8},
                {"a": "M008", "b": "M011", "resource": "R3", "overlap_slots": 5, "t_range": [285, 290], "severity": 8.5},
                {"a": "M002", "b": "M014", "resource": "R_range_test", "overlap_slots": 3, "t_range": [300, 303], "severity": 4.2},
            ],
            "conflict_clusters": [
                {"center_mission_id": "M004", "members": ["M004", "M006", "M009"], "score": 49.4},
                {"center_mission_id": "M011", "members": ["M011", "M016", "M008"], "score": 31.3},
            ],
            "urgent_missions": [
                {"mission_id": "M004", "due_slot": 280, "due_slack_slots": 40, "window_slack_slots": 18, "current_delay_slots": 20, "priority": 0.92, "urgency_score": 9.0},
                {"mission_id": "M011", "due_slot": 310, "due_slack_slots": 70, "window_slack_slots": 30, "current_delay_slots": 15, "priority": 0.78, "urgency_score": 55.0},
                {"mission_id": "M006", "due_slot": 325, "due_slack_slots": 85, "window_slack_slots": 40, "current_delay_slots": 10, "priority": 0.65, "urgency_score": 85.0},
            ],
            "disturbance_summary": {"range_loss_pct": 0.35, "pad_outage_active": True, "duration_volatility_level": "high"},
            "frozen_summary": {"num_started_ops": 8, "num_frozen_ops": 12, "frozen_horizon_slots": 12},
        },
        "system_prompt": REPAIR_SYSTEM_PROMPT,
        "user_prompt": None,
    })

    # ------- 合成 Case C: 大规模冲突簇，需要较大 unlock 集 -------
    cases.append({
        "case_id": "synthetic_C",
        "label": "大规模冲突簇-5任务连锁冲突（合成）",
        "seed": "N/A (合成数据)",
        "difficulty": "heavy",
        "now_slot": 360,
        "now_hours": 90.0,
        "num_active_missions": 12,
        "active_mission_ids": ["M001", "M002", "M005", "M006", "M008", "M010", "M013", "M015", "M017", "M019", "M021", "M023"],
        "trcg_dict": {
            "now_slot": 360,
            "horizon_end_slot": 456,
            "bottleneck_pressure": {"pad_util": 1.0, "r3_util": 0.95, "range_test_util": 0.42},
            "top_conflicts": [
                {"a": "M005", "b": "M008", "resource": "R_pad", "overlap_slots": 22, "t_range": [380, 402], "severity": 39.6},
                {"a": "M005", "b": "M010", "resource": "R_pad", "overlap_slots": 16, "t_range": [386, 402], "severity": 28.8},
                {"a": "M008", "b": "M010", "resource": "R_pad", "overlap_slots": 11, "t_range": [391, 402], "severity": 19.8},
                {"a": "M005", "b": "M013", "resource": "R3", "overlap_slots": 9, "t_range": [375, 384], "severity": 16.2},
                {"a": "M008", "b": "M013", "resource": "R3", "overlap_slots": 7, "t_range": [378, 385], "severity": 12.6},
                {"a": "M010", "b": "M015", "resource": "R_pad", "overlap_slots": 8, "t_range": [395, 403], "severity": 12.0},
                {"a": "M002", "b": "M019", "resource": "R3", "overlap_slots": 5, "t_range": [410, 415], "severity": 7.5},
                {"a": "M001", "b": "M006", "resource": "R_range_test", "overlap_slots": 4, "t_range": [420, 424], "severity": 5.2},
            ],
            "conflict_clusters": [
                {"center_mission_id": "M005", "members": ["M005", "M008", "M010", "M013", "M015"], "score": 84.6},
                {"center_mission_id": "M002", "members": ["M002", "M019"], "score": 7.5},
            ],
            "urgent_missions": [
                {"mission_id": "M005", "due_slot": 410, "due_slack_slots": 50, "window_slack_slots": 14, "current_delay_slots": 25, "priority": 0.95, "urgency_score": 7.0},
                {"mission_id": "M008", "due_slot": 430, "due_slack_slots": 70, "window_slack_slots": 20, "current_delay_slots": 18, "priority": 0.88, "urgency_score": 44.0},
                {"mission_id": "M010", "due_slot": 450, "due_slack_slots": 90, "window_slack_slots": 28, "current_delay_slots": 10, "priority": 0.72, "urgency_score": 84.0},
            ],
            "disturbance_summary": {"range_loss_pct": 0.12, "pad_outage_active": False, "duration_volatility_level": "medium"},
            "frozen_summary": {"num_started_ops": 12, "num_frozen_ops": 16, "frozen_horizon_slots": 12},
        },
        "system_prompt": REPAIR_SYSTEM_PROMPT,
        "user_prompt": None,
    })

    # 为合成用例生成 user_prompt
    for case in cases:
        case["user_prompt"] = build_repair_user_prompt(
            case["trcg_dict"], case["active_mission_ids"]
        )

    return cases


def format_markdown_output(cases: List[Dict]) -> str:
    """格式化为 Markdown 文档"""
    lines = []
    lines.append("# TRCG Repair LLM 策略 — 模型测试用例集\n")
    lines.append("## 背景说明\n")
    lines.append("""\
本文档包含 `trcg_repair_llm` 策略在不同调度场景下发送给 LLM 的**真实 prompt**。
你可以将下面每个测试用例的 **System Prompt** 和 **User Prompt** 分别复制到不同模型的网页端，
观察各模型的输出质量、格式合规性和推理准确性，以选择最适合调用的模型 API。

### 任务概述
- **领域**: 火箭发射排程调度（Rolling Horizon 仿真）
- **LLM 的角色**: 根因诊断 + 修复锚点选择
- **输入**: TRCG（Temporal-Resource Conflict Graph）诊断摘要
- **输出**: 严格 JSON（4 个字段），指定根因任务和需要解锁重排的任务集

### 期望的输出格式
```json
{
  "root_cause_mission_id": "M00X",
  "unlock_mission_ids": ["M00X", "M00Y"],
  "secondary_root_cause_mission_id": "M00Y" 或 null,
  "analysis_short": "≤120字符的根因分析简述"
}
```

### 评估维度
1. **格式合规**: 是否输出纯 JSON，无多余文字/markdown
2. **字段完整**: 4 个 required 字段是否齐全
3. **业务逻辑**: root_cause 是否在 unlock 内；unlock 是否为 active 子集
4. **推理质量**: 根因识别是否准确；unlock 集合大小是否合理
5. **响应速度**: 延迟是否可接受（生产中每步 rolling 都要调一次）

---
""")

    for i, case in enumerate(cases, 1):
        if case is None:
            continue
        lines.append(f"## 测试用例 {i}: {case.get('label', case['case_id'])}\n")
        lines.append(f"| 属性 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 场景种子 | {case['seed']} |")
        lines.append(f"| 难度 | {case['difficulty']} |")
        lines.append(f"| 当前时刻 | slot {case['now_slot']} (第 {case['now_hours']:.1f} 小时) |")
        lines.append(f"| 活跃任务数 | {case['num_active_missions']} |")
        lines.append(f"| 活跃任务 | {', '.join(case['active_mission_ids'])} |")

        # TRCG 关键指标
        trcg = case['trcg_dict']
        pressure = trcg.get('bottleneck_pressure', {})
        conflicts = trcg.get('top_conflicts', [])
        clusters = trcg.get('conflict_clusters', [])
        urgents = trcg.get('urgent_missions', [])
        dist = trcg.get('disturbance_summary', {})

        lines.append(f"| Pad利用率 | {pressure.get('pad_util', 0):.2%} |")
        lines.append(f"| R3利用率 | {pressure.get('r3_util', 0):.2%} |")
        lines.append(f"| 冲突数量 | {len(conflicts)} |")
        lines.append(f"| 冲突簇数 | {len(clusters)} |")
        lines.append(f"| 紧迫任务 | {', '.join(u['mission_id'] for u in urgents)} |")
        lines.append(f"| 窗口损失率 | {dist.get('range_loss_pct', 0):.1%} |")
        lines.append(f"| Pad故障 | {'是' if dist.get('pad_outage_active') else '否'} |")
        lines.append("")

        # System Prompt
        lines.append("### System Prompt\n")
        lines.append("```")
        lines.append(case['system_prompt'])
        lines.append("```\n")

        # User Prompt
        lines.append("### User Prompt\n")
        lines.append("```")
        lines.append(case['user_prompt'])
        lines.append("```\n")

        # 期望输出参考
        if clusters:
            center = clusters[0].get('center_mission_id', '')
            members = clusters[0].get('members', [])
            ref_unlock = members[:min(4, len(members))]
            secondary = members[1] if len(members) > 1 else None
        elif urgents:
            center = urgents[0]['mission_id']
            ref_unlock = [center]
            secondary = urgents[1]['mission_id'] if len(urgents) > 1 else None
        else:
            center = case['active_mission_ids'][0] if case['active_mission_ids'] else "M000"
            ref_unlock = [center]
            secondary = None

        lines.append("### 参考答案（启发式基线的决策，仅供对比）\n")
        lines.append("```json")
        ref_answer = {
            "root_cause_mission_id": center,
            "unlock_mission_ids": ref_unlock,
            "secondary_root_cause_mission_id": secondary,
            "analysis_short": f"heuristic: root={center} conflicts={len(conflicts)} pad_util={pressure.get('pad_util', 0):.2f}"
        }
        lines.append(json.dumps(ref_answer, indent=2, ensure_ascii=False))
        lines.append("```\n")

        lines.append("---\n")

    # 附录：评测打分表
    lines.append("## 附录：模型评测打分表\n")
    lines.append("| 评估项 | 权重 | 模型A | 模型B | 模型C | 模型D |")
    lines.append("|--------|------|-------|-------|-------|-------|")
    lines.append("| 纯JSON输出（无多余文字） | 25% | | | | |")
    lines.append("| 4字段完整且类型正确 | 20% | | | | |")
    lines.append("| root_cause∈unlock∈active | 20% | | | | |")
    lines.append("| unlock集合大小合理（1-8） | 10% | | | | |")
    lines.append("| 根因识别准确性 | 15% | | | | |")
    lines.append("| 响应延迟（秒） | 10% | | | | |")
    lines.append("| **总分** | **100%** | | | | |\n")

    lines.append("""\
### 打分标准
- **纯JSON**: 输出仅包含JSON对象得满分；带markdown/解释扣50%；完全不合格0分
- **字段完整**: 4个required字段齐全且类型正确得满分；缺字段每个扣25%
- **业务规则**: root∈unlock∈active全部满足得满分；每违反一条扣33%
- **集合大小**: 冲突多时3-5个合理；冲突少时1-2个合理；过大过小酌情扣分
- **根因准确**: 与冲突簇中心一致得满分；选了次优但合理的扣20%；完全不相关0分
- **响应延迟**: <5秒满分；5-15秒扣20%；15-30秒扣50%；>30秒扣80%
""")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成 LLM 测试用例")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径（默认: 打印到控制台）")
    args = parser.parse_args()

    print("正在生成测试用例...")

    # 1. 从真实仿真生成的用例（0 冲突，但有不同的紧迫度和压力信息）
    sim_scenarios = [
        {"seed": 42, "difficulty": "medium", "step_slot": 48,
         "label": "真实仿真-早期无冲突（pad/R3满载）"},
        {"seed": 42, "difficulty": "heavy", "step_slot": 192,
         "label": "真实仿真-中期高资源压力+已过期任务"},
    ]

    cases = []
    for i, spec in enumerate(sim_scenarios, 1):
        print(f"  [仿真 {i}/{len(sim_scenarios)}] seed={spec['seed']}, "
              f"difficulty={spec['difficulty']}, t={spec['step_slot']}")
        case = generate_test_case(
            seed=spec['seed'],
            difficulty=spec['difficulty'],
            step_slot=spec['step_slot'],
            case_id=f"sim_{i}",
        )
        if case:
            case['label'] = spec['label']
        cases.append(case)

    # 2. 合成的高冲突用例
    print("  [合成用例] 生成 3 个带冲突的合成场景...")
    synthetic = build_synthetic_cases()
    cases.extend(synthetic)

    valid_cases = [c for c in cases if c is not None]
    print(f"\n成功生成 {len(valid_cases)} 个测试用例 "
          f"({len(sim_scenarios)} 仿真 + {len(synthetic)} 合成)")

    md = format_markdown_output(valid_cases)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"已保存到: {args.output}")
    else:
        print("\n" + "=" * 80)
        print(md)


if __name__ == "__main__":
    main()
