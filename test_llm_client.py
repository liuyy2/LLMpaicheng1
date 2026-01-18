#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Client 自测脚本

功能：
1. 从环境变量读取 API Key
2. 测试 LLM 调用
3. 验证缓存命中
4. 输出日志到 ./llm_logs/test.jsonl

用法：
    # 设置环境变量
    set DASHSCOPE_API_KEY=your_api_key   # Windows
    export DASHSCOPE_API_KEY=your_api_key  # Linux/Mac
    
    # 运行测试
    python test_llm_client.py
"""

import os
import sys
import json
import shutil

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import (
    LLMClient, LLMConfig, LLMCache,
    extract_json_from_text, create_llm_client
)


def test_json_extraction():
    """测试 JSON 三层抽取"""
    print("\n" + "="*60)
    print(" 测试 1: JSON 三层抽取")
    print("="*60)
    
    test_cases = [
        # (输入, 期望方法, 期望能解析)
        ('{"w_delay": 10, "w_shift": 1}', "direct", True),
        ('```json\n{"w_delay": 15}\n```', "code_fence", True),
        ('```\n{"w_delay": 20}\n```', "code_fence", True),
        ('我来分析一下...\n\n{"w_delay": 25, "w_shift": 2}\n\n以上是建议', "brace_search", True),
        ('{"nested": {"a": 1, "b": 2}}', "direct", True),
        ('thinking...\n\n```json\n{"result": 42}\n```\n\ndone', "code_fence", True),
        ('no valid json here', "failed", False),
        ('just some {incomplete', "failed", False),
    ]
    
    passed = 0
    for text, expected_method, should_parse in test_cases:
        result, method = extract_json_from_text(text)
        
        method_ok = (method == expected_method)
        parse_ok = (result is not None) == should_parse
        
        if method_ok and parse_ok:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
        
        # 截断显示
        display = text[:40].replace('\n', '\\n')
        if len(text) > 40:
            display += "..."
        
        print(f"  {status}: method={method:12} expected={expected_method:12} | {display}")
    
    print(f"\n  结果: {passed}/{len(test_cases)} 通过")
    return passed == len(test_cases)


def test_cache_key():
    """测试缓存 key 计算"""
    print("\n" + "="*60)
    print(" 测试 2: 缓存 Key 计算")
    print("="*60)
    
    # 相同输入应该生成相同 key
    key1 = LLMCache.compute_cache_key(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "测试"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )
    
    key2 = LLMCache.compute_cache_key(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "测试"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )
    
    # 不同输入应该生成不同 key
    key3 = LLMCache.compute_cache_key(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "测试2"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )
    
    key4 = LLMCache.compute_cache_key(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "测试"}],
        temperature=0.5,  # 不同温度
        top_p=1.0,
        max_tokens=256
    )
    
    tests = [
        ("相同输入生成相同 key", key1 == key2),
        ("不同内容生成不同 key", key1 != key3),
        ("不同温度生成不同 key", key1 != key4),
        ("Key 长度为 64 (SHA256)", len(key1) == 64),
    ]
    
    passed = 0
    for desc, result in tests:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {desc}")
        if result:
            passed += 1
    
    print(f"\n  Key 示例: {key1[:32]}...")
    print(f"  结果: {passed}/{len(tests)} 通过")
    return passed == len(tests)


def test_cache_operations():
    """测试缓存读写"""
    print("\n" + "="*60)
    print(" 测试 3: 缓存读写操作")
    print("="*60)
    
    # 使用临时目录
    test_cache_dir = "./llm_logs/.test_cache"
    
    try:
        # 清理旧缓存
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)
        
        cache = LLMCache(test_cache_dir)
        
        # 测试写入
        test_key = "test_key_12345"
        test_data = {"raw_text": "test response", "tokens_total": 100}
        
        write_ok = cache.set(test_key, test_data)
        print(f"  写入缓存: {'✓' if write_ok else '✗'}")
        
        # 测试读取
        read_data = cache.get(test_key)
        read_ok = read_data is not None and read_data.get("raw_text") == "test response"
        print(f"  读取缓存: {'✓' if read_ok else '✗'}")
        
        # 测试不存在的 key
        missing = cache.get("nonexistent_key")
        missing_ok = missing is None
        print(f"  缺失返回 None: {'✓' if missing_ok else '✗'}")
        
        # 测试统计
        stats = cache.stats()
        stats_ok = stats["num_entries"] == 1
        print(f"  统计正确: {'✓' if stats_ok else '✗'} (entries={stats['num_entries']})")
        
        # 测试清空
        cleared = cache.clear()
        clear_ok = cleared == 1 and cache.get(test_key) is None
        print(f"  清空缓存: {'✓' if clear_ok else '✗'} (cleared={cleared})")
        
        passed = sum([write_ok, read_ok, missing_ok, stats_ok, clear_ok])
        print(f"\n  结果: {passed}/5 通过")
        
        return passed == 5
        
    finally:
        # 清理
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)


def test_llm_api():
    """测试真实 LLM API 调用"""
    print("\n" + "="*60)
    print(" 测试 4: 真实 LLM API 调用")
    print("="*60)
    
    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("  ⚠ 未设置 DASHSCOPE_API_KEY 环境变量，跳过 API 测试")
        print("  设置方法:")
        print("    Windows: set DASHSCOPE_API_KEY=your_key")
        print("    Linux:   export DASHSCOPE_API_KEY=your_key")
        return None  # 跳过但不算失败
    
    print(f"  API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # 配置
    cache_dir = "./llm_logs/test_cache"
    log_file = "./llm_logs/test.jsonl"
    
    # 清理旧日志和缓存
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
    # 创建客户端
    config = LLMConfig(
        api_key=api_key,
        base_url="https://api-inference.modelscope.cn/v1",
        model="Qwen/Qwen3-32B",
        temperature=0.0,
        max_tokens=64,
        timeout_s=30.0,
        max_retries=3,
        cache_dir=cache_dir,
        log_file=log_file,
        enable_thinking=False
    )
    
    print(f"\n  配置:")
    print(f"    Model: {config.model}")
    print(f"    Base URL: {config.base_url}")
    print(f"    Cache Dir: {cache_dir}")
    print(f"    Log File: {log_file}")
    
    try:
        client = LLMClient(config)
    except Exception as e:
        print(f"\n  ✗ 客户端创建失败: {e}")
        return False
    
    # 测试消息（极短）
    test_messages = [
        {"role": "user", "content": "请用一个词回答：1+1=?"}
    ]
    
    # ========== 第一次调用 ==========
    print("\n  --- 第一次调用 (应该调用 API) ---")
    
    try:
        result1 = client.call(messages=test_messages)
        
        print(f"    success: {result1.success}")
        print(f"    cache_hit: {result1.cache_hit}")
        print(f"    latency_ms: {result1.latency_ms}")
        print(f"    tokens_prompt: {result1.tokens_prompt}")
        print(f"    tokens_completion: {result1.tokens_completion}")
        print(f"    tokens_total: {result1.tokens_total}")
        print(f"    finish_reason: {result1.finish_reason}")
        print(f"    raw_text: {result1.raw_text[:100] if result1.raw_text else 'None'}...")
        
        if not result1.success:
            print(f"    error: {result1.error_type}: {result1.error_message}")
            return False
        
        call1_ok = result1.success and not result1.cache_hit
        print(f"    验证: {'✓' if call1_ok else '✗'} (success=True, cache_hit=False)")
        
    except Exception as e:
        print(f"    ✗ 调用异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 第二次调用（相同内容，应命中缓存）==========
    print("\n  --- 第二次调用 (应该命中缓存) ---")
    
    try:
        result2 = client.call(messages=test_messages)
        
        print(f"    success: {result2.success}")
        print(f"    cache_hit: {result2.cache_hit}")
        print(f"    latency_ms: {result2.latency_ms}")
        print(f"    raw_text: {result2.raw_text[:100] if result2.raw_text else 'None'}...")
        
        call2_ok = result2.success and result2.cache_hit
        print(f"    验证: {'✓' if call2_ok else '✗'} (success=True, cache_hit=True)")
        
        # 验证缓存内容一致
        content_ok = result1.raw_text == result2.raw_text
        print(f"    内容一致: {'✓' if content_ok else '✗'}")
        
    except Exception as e:
        print(f"    ✗ 调用异常: {e}")
        return False
    
    # ========== 第三次调用（不同内容，应调用 API）==========
    print("\n  --- 第三次调用 (不同内容，应调用 API) ---")
    
    try:
        result3 = client.call(
            messages=[{"role": "user", "content": "请用一个词回答：2+2=?"}]
        )
        
        print(f"    success: {result3.success}")
        print(f"    cache_hit: {result3.cache_hit}")
        print(f"    latency_ms: {result3.latency_ms}")
        
        call3_ok = result3.success and not result3.cache_hit
        print(f"    验证: {'✓' if call3_ok else '✗'} (success=True, cache_hit=False)")
        
    except Exception as e:
        print(f"    ✗ 调用异常: {e}")
        return False
    
    # ========== 统计 ==========
    print("\n  --- 客户端统计 ---")
    stats = client.get_stats()
    print(f"    total_calls: {stats['total_calls']}")
    print(f"    cache_hits: {stats['cache_hits']}")
    print(f"    api_calls: {stats['api_calls']}")
    print(f"    cache_hit_rate: {stats['cache_hit_rate']:.2%}")
    print(f"    total_tokens: {stats['total_tokens']}")
    
    stats_ok = (
        stats['total_calls'] == 3 and
        stats['cache_hits'] == 1 and
        stats['api_calls'] == 2
    )
    print(f"    验证: {'✓' if stats_ok else '✗'} (calls=3, hits=1, api=2)")
    
    # ========== 日志验证 ==========
    print("\n  --- 日志验证 ---")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"    日志文件存在: ✓")
        print(f"    日志行数: {len(lines)}")
        
        log_ok = len(lines) == 3
        print(f"    验证: {'✓' if log_ok else '✗'} (应该有 3 行)")
        
        # 显示日志内容
        print("\n    日志内容预览:")
        for i, line in enumerate(lines):
            log_entry = json.loads(line)
            print(f"      [{i+1}] cache_hit={log_entry['cache_hit']}, "
                  f"latency={log_entry['latency_ms']}ms, "
                  f"tokens={log_entry['tokens_total']}")
    else:
        print(f"    日志文件不存在: ✗")
        log_ok = False
    
    # ========== 总结 ==========
    all_passed = call1_ok and call2_ok and content_ok and call3_ok and stats_ok and log_ok
    print(f"\n  结果: {'全部通过 ✓' if all_passed else '存在失败 ✗'}")
    
    return all_passed


def main():
    """主测试函数"""
    print("="*60)
    print(" LLM Client 自测脚本")
    print("="*60)
    print(f" 工作目录: {os.getcwd()}")
    print(f" Python: {sys.version.split()[0]}")
    
    results = {}
    
    # 测试 1: JSON 抽取
    results["json_extraction"] = test_json_extraction()
    
    # 测试 2: 缓存 key
    results["cache_key"] = test_cache_key()
    
    # 测试 3: 缓存操作
    results["cache_operations"] = test_cache_operations()
    
    # 测试 4: API 调用
    api_result = test_llm_api()
    if api_result is not None:
        results["api_call"] = api_result
    
    # 总结
    print("\n" + "="*60)
    print(" 测试总结")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n  总计: {total_passed}/{total_tests} 通过")
    
    if total_passed == total_tests:
        print("\n  🎉 所有测试通过!")
        return 0
    else:
        print("\n  ⚠ 存在测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
