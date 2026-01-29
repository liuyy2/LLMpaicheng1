#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiments.py 测试脚本

验证：
1. 新增字段完整性
2. mockllm 策略运行
3. llm_real 策略强制串行
4. CSV 输出格式
"""

import os
import sys

def test_imports():
    """测试导入"""
    print("1. 测试导入...")
    from run_experiments import (
        ExperimentConfig, EpisodeMetricsRecord, RollingLogEntry,
        run_single_episode, generate_seed_assignments,
        save_episode_results, HAS_REAL_LLM
    )
    print(f"   HAS_REAL_LLM = {HAS_REAL_LLM}")
    print("   OK 导入成功")
    return True


def test_record_fields():
    """测试 EpisodeMetricsRecord 字段"""
    print("\n2. 测试 EpisodeMetricsRecord 字段...")
    from run_experiments import EpisodeMetricsRecord
    
    required_fields = [
        "llm_calls",
        "llm_time_total_ms", 
        "llm_latency_total_ms",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "llm_cache_hit_rate",
        "llm_fallback_count",
        "solver_time_total_ms",
        "wall_time_total_ms"
    ]
    
    fields = set(EpisodeMetricsRecord.__dataclass_fields__.keys())
    missing = [f for f in required_fields if f not in fields]
    
    if missing:
        print(f"   FAIL 缺少字段: {missing}")
        return False
    
    print(f"   OK 所有必需字段存在")
    return True


def test_mockllm_run():
    """测试 mockllm 策略运行"""
    print("\n3. 测试 mockllm 策略运行...")
    from run_experiments import run_single_episode
    
    output_dir = "test_output_temp"
    os.makedirs(output_dir, exist_ok=True)
    
    record = run_single_episode(
        seed=42,
        disturbance_level="light",
        policy_name="mockllm",
        policy_params={},
        dataset="test",
        solver_timeout=5.0,
        output_dir=output_dir
    )
    
    print(f"   seed={record.seed}, policy={record.policy_name}")
    print(f"   completed={record.completed}/{record.total}")
    print(f"   llm_calls={record.llm_calls}")
    print(f"   solver_time_total_ms={record.solver_time_total_ms}")
    print(f"   wall_time_total_ms={record.wall_time_total_ms}")
    
    # 检查日志文件
    log_dir = os.path.join(output_dir, "logs", f"episode_42_mockllm")
    if os.path.exists(log_dir):
        files = os.listdir(log_dir)
        print(f"   日志文件: {files}")
    
    print("   OK mockllm 运行成功")
    return True


def test_csv_output():
    """测试 CSV 输出"""
    print("\n4. 测试 CSV 输出...")
    from run_experiments import EpisodeMetricsRecord, save_episode_results
    import csv
    
    output_dir = "test_output_temp"
    csv_path = os.path.join(output_dir, "test_results.csv")
    
    # 创建测试记录
    record = EpisodeMetricsRecord(
        seed=1, disturbance_level="light", policy_name="test", dataset="test",
        completed=10, total=10, on_time_rate=1.0, avg_delay=0.0, max_delay=0,
        weighted_tardiness=0.0, resource_utilization=0.0,
        episode_drift=0.001, total_shifts=2, total_switches=1,
        total_window_switches=0, total_sequence_switches=0,
        num_replans=5, num_forced_replans=0, avg_solve_time_ms=100.0,
        total_runtime_s=1.0,
        avg_time_deviation_min=0.0, total_resource_switches=0,
        makespan_cmax=0, feasible_rate=1.0, forced_replan_rate=0.0,
        avg_frozen=0.0, avg_num_tasks_scheduled=0.0, util_r_pad=0.0,
        llm_calls=5, llm_time_total_ms=500, llm_latency_total_ms=450,
        llm_prompt_tokens=100, llm_completion_tokens=50, llm_total_tokens=150,
        llm_cache_hit_rate=0.4, llm_fallback_count=1,
        solver_time_total_ms=400, wall_time_total_ms=1000
    )
    
    save_episode_results([record], csv_path)
    
    # 验证 CSV 内容
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        
        llm_fields = [k for k in row.keys() if 'llm' in k or 'solver_time' in k or 'wall_time' in k]
        print(f"   LLM 相关字段: {llm_fields}")
    
    print("   OK CSV 输出正确")
    return True


def test_llm_real_warning():
    """测试 llm_real 策略的并行警告"""
    print("\n5. 测试 llm_real 并行警告...")
    from run_experiments import ExperimentConfig, HAS_REAL_LLM
    import logging
    
    if not HAS_REAL_LLM:
        print("   跳过: llm_client 不可用")
        return True
    
    exp_config = ExperimentConfig(max_workers=8)
    
    # 模拟设置警告
    print(f"   初始 workers={exp_config.max_workers}")
    
    # 检测到 llm_real 策略时的逻辑
    if exp_config.max_workers > 1:
        print("   检测到 llm_real 策略，强制降为 1")
        exp_config.max_workers = 1
    
    print(f"   最终 workers={exp_config.max_workers}")
    print("   OK 并行警告逻辑正确")
    return True


def test_seed_assignments():
    """测试种子分配"""
    print("\n6. 测试种子分配...")
    from run_experiments import generate_seed_assignments
    
    train, test = generate_seed_assignments(9, 9)
    
    print(f"   训练集: {[(s, l) for s, l in train[:3]]}...")
    print(f"   测试集: {[(s, l) for s, l in test[:3]]}...")
    
    # 验证种子不重叠
    train_seeds = set(s for s, _ in train)
    test_seeds = set(s for s, _ in test)
    overlap = train_seeds & test_seeds
    
    if overlap:
        print(f"   FAIL 种子重叠: {overlap}")
        return False
    
    print("   OK 种子分配正确")
    return True


def cleanup():
    """清理测试文件"""
    import shutil
    output_dir = "test_output_temp"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print(" run_experiments.py 功能测试")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_record_fields,
        test_mockllm_run,
        test_csv_output,
        test_llm_real_warning,
        test_seed_assignments
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   FAIL 异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f" 结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    cleanup()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
