#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Policy 演示脚本

功能：
1. 构造假的状态特征
2. 调用 MockLLMPolicy / RealLLMPolicy 获取元参数 JSON
3. 打印结果并写日志

用法：
    # 测试 MockLLMPolicy（无需 API Key）
    python demo_llm_policy.py --mock

    # 测试 RealLLMPolicy（需要 API Key）
    set DASHSCOPE_API_KEY=your_key
    python demo_llm_policy.py --real
    
    # 测试两者
    python demo_llm_policy.py --both
"""

import os
import sys
import json
import argparse
import shutil
from typing import Dict, Any

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import StateFeatures
from policies.policy_llm_meta import (
    MockLLMPolicy,
    RealLLMPolicy,
    validate_meta_params_json,
    build_user_prompt,
    SYSTEM_PROMPT,
    LLMDecisionLog
)


def create_fake_features(scenario: str = "normal") -> StateFeatures:
    """
    创建假的状态特征
    
    Args:
        scenario: 场景类型
            - "normal": 正常场景
            - "high_loss": 高窗口损失
            - "many_urgent": 多紧急任务
            - "pad_outage": Pad 故障
    """
    base = {
        "window_loss_pct": 0.1,
        "pad_outage_overlap_hours": 0.0,
        "delay_increase_minutes": 10,
        "pad_pressure": 0.8,
        "slack_min_minutes": 120,
        "resource_conflict_pressure": 0.2,
        "trend_window_loss": 0.0,
        "trend_pad_pressure": 0.0,
        "trend_slack_min_minutes": 0.0,
        "trend_delay_increase_minutes": 0.0,
        "volatility_pad_pressure": 0.05,
        "num_urgent_tasks": 1,
    }
    
    if scenario == "high_loss":
        base.update({
            "window_loss_pct": 0.5,
            "delay_increase_minutes": 60,
            "trend_window_loss": 0.08
        })
    elif scenario == "many_urgent":
        base.update({
            "num_urgent_tasks": 6,
            "delay_increase_minutes": 90,
            "trend_slack_min_minutes": -15
        })
    elif scenario == "pad_outage":
        base.update({
            "pad_outage_overlap_hours": 3.0,
            "volatility_pad_pressure": 0.2,
            "trend_pad_pressure": 0.1
        })
    
    return StateFeatures(**base)


def print_features(features: StateFeatures) -> None:
    """打印特征"""
    print("\n  特征:")
    for key, value in features.to_dict().items():
        print(f"    {key}: {value}")


def print_result(result: Dict[str, Any], log: LLMDecisionLog) -> None:
    """打印决策结果"""
    print("\n  决策结果:")
    print(f"    cache_hit: {log.llm_cache_hit}")
    print(f"    latency_ms: {log.llm_latency_ms}")
    print(f"    tokens: {log.usage_tokens}")
    print(f"    parsed_ok: {log.parsed_ok}")
    print(f"    fallback_used: {log.fallback_used}")
    if log.fallback_reason:
        print(f"    fallback_reason: {log.fallback_reason}")
    print(f"    extraction_method: {log.extraction_method}")
    print(f"\n  最终参数: {json.dumps(log.final_params, indent=4)}")
    
    if log.raw_output:
        print(f"\n  原始输出:\n    {log.raw_output[:200]}...")


def test_mock_policy(log_dir: str) -> bool:
    """测试 MockLLMPolicy"""
    print("\n" + "="*60)
    print(" 测试 MockLLMPolicy")
    print("="*60)
    
    # 清理旧日志
    mock_log_dir = os.path.join(log_dir, "mock")
    if os.path.exists(mock_log_dir):
        shutil.rmtree(mock_log_dir)
    os.makedirs(mock_log_dir, exist_ok=True)
    
    policy = MockLLMPolicy(
        policy_name="mockllm_demo",
        log_dir=mock_log_dir,
        enable_logging=True,
        episode_id="demo_001"
    )
    
    scenarios = ["normal", "high_loss", "many_urgent", "pad_outage"]
    
    for scenario in scenarios:
        print(f"\n--- 场景: {scenario} ---")
        
        features = create_fake_features(scenario)
        print_features(features)
        
        # 生成 JSON
        raw_json = policy._generate_mock_json(features)
        print(f"\n  生成的 JSON:\n{raw_json}")
        
        # 校验
        validation = validate_meta_params_json(raw_json)
        print(f"\n  校验: valid={validation.is_valid}, method={validation.extraction_method}")
        if validation.warnings:
            print(f"  warnings: {validation.warnings}")
    
    # 保存日志
    log_file = os.path.join(mock_log_dir, "decisions.jsonl")
    policy.save_logs(log_file)
    
    print(f"\n  统计: {policy.get_stats()}")
    print(f"  日志保存到: {log_file}")
    
    return True


def test_real_policy(log_dir: str) -> bool:
    """测试 RealLLMPolicy"""
    print("\n" + "="*60)
    print(" 测试 RealLLMPolicy")
    print("="*60)
    
    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("  ⚠ 未设置 DASHSCOPE_API_KEY 环境变量")
        print("  设置方法:")
        print("    Windows: set DASHSCOPE_API_KEY=your_key")
        print("    Linux:   export DASHSCOPE_API_KEY=your_key")
        return False
    
    print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # 清理旧日志
    real_log_dir = os.path.join(log_dir, "real")
    if os.path.exists(real_log_dir):
        shutil.rmtree(real_log_dir)
    os.makedirs(real_log_dir, exist_ok=True)
    
    try:
        from llm_client import LLMConfig
        
        llm_config = LLMConfig(
            api_key=api_key,
            base_url="https://api-inference.modelscope.cn/v1",
            model="Qwen/Qwen3-32B",
            temperature=0.0,
            max_tokens=256,
            timeout_s=30.0,
            cache_dir=os.path.join(real_log_dir, "llm_cache"),
            log_file=os.path.join(real_log_dir, "llm_raw_calls.jsonl"),
            enable_thinking=False
        )
        
        policy = RealLLMPolicy(
            llm_config=llm_config,
            policy_name="qwen3_demo",
            log_dir=real_log_dir,
            enable_logging=True,
            episode_id="demo_002"
        )
        
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ✗ 策略创建失败: {e}")
        return False
    
    # 测试场景
    scenarios = ["normal", "high_loss"]
    
    for scenario in scenarios:
        print(f"\n--- 场景: {scenario} ---")
        
        features = create_fake_features(scenario)
        print_features(features)
        
        # 构建 prompt 展示
        user_prompt = build_user_prompt(features)
        print(f"\n  Prompt 长度: {len(user_prompt)} 字符")
        print(f"  System Prompt 长度: {len(SYSTEM_PROMPT)} 字符")
        
        # 模拟调用（我们需要创建一个假的 SimulationState）
        # 这里直接测试 LLM 客户端
        print("\n  调用 LLM...")
        
        try:
            client = policy._ensure_client()
            result = client.call(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=SYSTEM_PROMPT
            )
            
            print(f"    success: {result.success}")
            print(f"    cache_hit: {result.cache_hit}")
            print(f"    latency_ms: {result.latency_ms}")
            print(f"    tokens_total: {result.tokens_total}")
            
            if result.success:
                print(f"\n  原始响应:\n    {result.raw_text[:300]}..." if len(result.raw_text or "") > 300 else f"\n  原始响应:\n    {result.raw_text}")
                
                # 校验
                validation = validate_meta_params_json(result.raw_text or "")
                print(f"\n  校验: valid={validation.is_valid}, method={validation.extraction_method}")
                if validation.params:
                    print(f"  解析参数: {validation.params}")
                if validation.warnings:
                    print(f"  warnings: {validation.warnings}")
                if validation.errors:
                    print(f"  errors: {validation.errors}")
            else:
                print(f"    error: {result.error_type}: {result.error_message}")
                
        except Exception as e:
            print(f"    ✗ 调用异常: {e}")
            import traceback
            traceback.print_exc()
    
    # 统计
    print(f"\n  策略统计: {policy.get_stats()}")
    
    # 测试缓存命中
    print("\n--- 测试缓存命中 ---")
    features = create_fake_features("normal")
    user_prompt = build_user_prompt(features)
    
    result2 = client.call(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=SYSTEM_PROMPT
    )
    
    print(f"  cache_hit: {result2.cache_hit}")
    print(f"  latency_ms: {result2.latency_ms}")
    
    if result2.cache_hit:
        print("  ✓ 缓存命中验证通过")
    else:
        print("  ✗ 缓存未命中（可能是首次调用）")
    
    # 检查日志文件
    log_file = os.path.join(real_log_dir, "llm_raw_calls.jsonl")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"\n  日志文件: {log_file}")
        print(f"  日志行数: {len(lines)}")
    
    # 检查缓存
    cache_dir = os.path.join(real_log_dir, "llm_cache")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
        print(f"  缓存文件数: {len(cache_files)}")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM Policy 演示脚本")
    parser.add_argument("--mock", action="store_true", help="测试 MockLLMPolicy")
    parser.add_argument("--real", action="store_true", help="测试 RealLLMPolicy")
    parser.add_argument("--both", action="store_true", help="测试两者")
    parser.add_argument("--log-dir", type=str, default="./llm_logs/demo", help="日志目录")
    
    args = parser.parse_args()
    
    # 默认测试 mock
    if not args.mock and not args.real and not args.both:
        args.mock = True
    
    print("="*60)
    print(" LLM Policy 演示脚本")
    print("="*60)
    print(f" 工作目录: {os.getcwd()}")
    print(f" 日志目录: {args.log_dir}")
    
    results = {}
    
    if args.mock or args.both:
        results["mock"] = test_mock_policy(args.log_dir)
    
    if args.real or args.both:
        results["real"] = test_real_policy(args.log_dir)
    
    # 总结
    print("\n" + "="*60)
    print(" 测试总结")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/SKIP"
        print(f"  {status}: {name}")
    
    print(f"\n  日志目录: {args.log_dir}")
    print("  可查看以下文件:")
    print("    - mock/decisions.jsonl (MockLLM 决策日志)")
    print("    - real/llm_raw_calls.jsonl (真实 LLM 调用日志)")
    print("    - real/llm_cache/*.json (LLM 缓存)")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
