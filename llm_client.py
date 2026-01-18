#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Client - 封装 OpenAI 兼容 API 调用

功能：
1. 支持 ModelScope Qwen3-32B（OpenAI 兼容接口）
2. 磁盘缓存（SHA256 key，并发安全）
3. 指数退避重试（429/5xx/超时）
4. 完整调用日志（JSONL 格式）
5. JSON 三层抽取与 Schema 校验

使用方式：
    from llm_client import LLMClient, LLMConfig
    
    config = LLMConfig(api_key="your_key")
    client = LLMClient(config)
    result = client.call(messages=[{"role": "user", "content": "Hello"}])
"""

import os
import json
import time
import hashlib
import random
import tempfile
import shutil
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    # Dummy exceptions for type hints when openai not installed
    class APIError(Exception): pass
    class APITimeoutError(Exception): pass
    class RateLimitError(Exception): pass
    class APIConnectionError(Exception): pass


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class LLMConfig:
    """LLM 客户端配置"""
    
    # API 配置
    api_key: str = ""                          # API Key（可从环境变量读取）
    api_key_env: str = "DASHSCOPE_API_KEY"     # 环境变量名
    base_url: str = "https://api-inference.modelscope.cn/v1"
    model: str = "Qwen/Qwen3-32B"
    
    # 生成参数
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    
    # 超时与重试
    timeout_s: float = 30.0
    max_retries: int = 5
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 60.0
    retry_jitter: float = 0.5                  # jitter 比例 (0-1)
    
    # 缓存
    cache_dir: Optional[str] = None            # None = 不缓存
    cache_enabled: bool = True
    
    # 日志
    log_file: Optional[str] = None             # JSONL 日志文件路径
    
    # Qwen3 特殊参数
    enable_thinking: bool = False              # Qwen3 非流式必须关闭
    
    def get_api_key(self) -> str:
        """获取 API Key（优先使用直接配置，否则从环境变量读取）"""
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "")
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（隐藏 api_key）"""
        d = asdict(self)
        d["api_key"] = "***" if self.api_key else "(from env)"
        return d


# ============================================================================
# 调用结果数据类
# ============================================================================

@dataclass
class LLMCallResult:
    """单次 LLM 调用结果"""
    
    # 基本信息
    success: bool
    model: str
    finish_reason: Optional[str] = None
    
    # 响应内容
    raw_text: Optional[str] = None             # 原始响应文本
    
    # 性能指标
    latency_ms: int = 0                        # 调用延迟（毫秒）
    cache_hit: bool = False                    # 是否命中缓存
    retries: int = 0                           # 重试次数
    
    # Token 使用
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    
    # 错误信息
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # 元数据
    timestamp: str = ""
    cache_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return asdict(self)
    
    def to_log_dict(self) -> Dict[str, Any]:
        """转为日志字典（精简版）"""
        return {
            "timestamp": self.timestamp,
            "success": self.success,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "raw_text": self.raw_text,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "retries": self.retries,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "tokens_total": self.tokens_total,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "cache_key": self.cache_key[:16] if self.cache_key else None
        }


# ============================================================================
# 缓存管理
# ============================================================================

class LLMCache:
    """LLM 缓存管理器（磁盘级，并发安全）"""
    
    def __init__(self, cache_dir: str):
        """
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    @staticmethod
    def compute_cache_key(
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int
    ) -> str:
        """
        计算缓存 key
        
        key = SHA256(规范化 JSON(model + messages + params))
        """
        # 构建规范化对象
        cache_obj = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        # 规范化 JSON（排序 key，无空格）
        canonical = json.dumps(cache_obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        
        # SHA256
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存
        
        Returns:
            缓存数据 or None
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # 缓存损坏，删除
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None
    
    def set(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        写入缓存（并发安全：临时文件 + rename）
        
        Args:
            cache_key: 缓存 key
            data: 缓存数据
        
        Returns:
            是否成功
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # 写入临时文件
            fd, tmp_path = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix=f".tmp_{cache_key[:8]}_",
                suffix=".json"
            )
            
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 原子 rename（Windows 上需要先删除目标）
                if os.name == 'nt' and os.path.exists(cache_path):
                    os.remove(cache_path)
                shutil.move(tmp_path, cache_path)
                return True
                
            except Exception:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
                
        except Exception as e:
            print(f"[LLMCache] 写入缓存失败: {e}")
            return False
    
    def clear(self) -> int:
        """
        清空缓存
        
        Returns:
            删除的文件数
        """
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json') and not filename.startswith('.'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
                except OSError:
                    pass
        return count
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        files = [f for f in os.listdir(self.cache_dir) 
                 if f.endswith('.json') and not f.startswith('.')]
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f)) 
            for f in files
        )
        return {
            "cache_dir": self.cache_dir,
            "num_entries": len(files),
            "total_size_kb": round(total_size / 1024, 2)
        }


# ============================================================================
# JSON 抽取工具
# ============================================================================

def extract_json_from_text(text: str) -> Tuple[Optional[str], str]:
    """
    三层 JSON 抽取
    
    层级：
    1. 直接尝试 json.loads（如果整个文本就是 JSON）
    2. 去除 ```json ... ``` 或 ``` ... ``` code fence
    3. 正则搜索第一个 {...} 块（处理嵌套）
    
    Args:
        text: 原始文本
    
    Returns:
        (extracted_json_str, extraction_method)
        extraction_method: "direct" | "code_fence" | "brace_search" | "failed"
    """
    if not text:
        return None, "failed"
    
    text = text.strip()
    
    # 层级 1: 直接尝试（整个文本就是 JSON）
    if text.startswith('{') and text.endswith('}'):
        try:
            json.loads(text)
            return text, "direct"
        except json.JSONDecodeError:
            pass
    
    # 层级 2: Code fence 抽取
    # 匹配 ```json ... ``` 或 ``` ... ```
    code_fence_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```'
    ]
    
    for pattern in code_fence_patterns:
        match = re.search(pattern, text)
        if match:
            content = match.group(1).strip()
            if content.startswith('{') and content.endswith('}'):
                try:
                    json.loads(content)
                    return content, "code_fence"
                except json.JSONDecodeError:
                    pass
    
    # 层级 3: Brace 搜索（处理嵌套）
    def find_json_object(s: str) -> Optional[str]:
        """找到第一个完整的 JSON 对象"""
        start = s.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape = False
        
        for i, c in enumerate(s[start:], start):
            if escape:
                escape = False
                continue
            
            if c == '\\' and in_string:
                escape = True
                continue
            
            if c == '"' and not escape:
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        
        return None
    
    json_str = find_json_object(text)
    if json_str:
        try:
            json.loads(json_str)
            return json_str, "brace_search"
        except json.JSONDecodeError:
            pass
    
    return None, "failed"


# ============================================================================
# LLM Client 主类
# ============================================================================

class LLMClient:
    """
    LLM 调用客户端
    
    特性：
    - OpenAI 兼容接口
    - 磁盘缓存（并发安全）
    - 指数退避重试
    - 完整日志
    """
    
    # 可重试的异常类型
    RETRYABLE_EXCEPTIONS = (
        APITimeoutError,
        RateLimitError,
        APIConnectionError,
    )
    
    # 可重试的 HTTP 状态码
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    
    def __init__(self, config: LLMConfig):
        """
        Args:
            config: LLM 配置
        """
        if not HAS_OPENAI:
            raise ImportError(
                "需要安装 openai 库: pip install openai>=1.0.0"
            )
        
        self.config = config
        
        # 初始化 OpenAI 客户端
        api_key = config.get_api_key()
        if not api_key:
            raise ValueError(
                f"未提供 API Key，请设置环境变量 {config.api_key_env} 或直接传入 api_key"
            )
        
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_s
        )
        
        # 初始化缓存
        self._cache: Optional[LLMCache] = None
        if config.cache_enabled and config.cache_dir:
            self._cache = LLMCache(config.cache_dir)
        
        # 统计
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "retries": 0,
            "failures": 0,
            "total_tokens": 0,
            "total_latency_ms": 0
        }
    
    def call(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: Optional[bool] = None
    ) -> LLMCallResult:
        """
        调用 LLM
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            system_prompt: 系统提示（可选，会插入到 messages 最前面）
            temperature: 温度（覆盖配置）
            top_p: top_p（覆盖配置）
            max_tokens: 最大 token（覆盖配置）
            use_cache: 是否使用缓存（覆盖配置）
        
        Returns:
            LLMCallResult
        """
        self._stats["total_calls"] += 1
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # 参数处理
        temp = temperature if temperature is not None else self.config.temperature
        tp = top_p if top_p is not None else self.config.top_p
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        should_cache = use_cache if use_cache is not None else self.config.cache_enabled
        
        # 构建完整 messages
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # 计算缓存 key
        cache_key = LLMCache.compute_cache_key(
            model=self.config.model,
            messages=full_messages,
            temperature=temp,
            top_p=tp,
            max_tokens=max_tok
        )
        
        # 检查缓存
        if should_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                latency_ms = int((time.time() - start_time) * 1000)
                
                result = LLMCallResult(
                    success=True,
                    model=self.config.model,
                    finish_reason=cached.get("finish_reason"),
                    raw_text=cached.get("raw_text"),
                    latency_ms=latency_ms,
                    cache_hit=True,
                    retries=0,
                    tokens_prompt=cached.get("tokens_prompt", 0),
                    tokens_completion=cached.get("tokens_completion", 0),
                    tokens_total=cached.get("tokens_total", 0),
                    timestamp=timestamp,
                    cache_key=cache_key
                )
                
                # 写日志
                self._append_log(result)
                return result
        
        # API 调用（带重试）
        self._stats["api_calls"] += 1
        result = self._call_with_retry(
            messages=full_messages,
            temperature=temp,
            top_p=tp,
            max_tokens=max_tok,
            cache_key=cache_key,
            timestamp=timestamp
        )
        
        # 写入缓存
        if result.success and should_cache and self._cache:
            cache_data = {
                "raw_text": result.raw_text,
                "finish_reason": result.finish_reason,
                "tokens_prompt": result.tokens_prompt,
                "tokens_completion": result.tokens_completion,
                "tokens_total": result.tokens_total,
                "model": result.model,
                "cached_at": timestamp
            }
            self._cache.set(cache_key, cache_data)
        
        # 更新统计
        self._stats["total_tokens"] += result.tokens_total
        self._stats["total_latency_ms"] += result.latency_ms
        if not result.success:
            self._stats["failures"] += 1
        
        # 写日志
        self._append_log(result)
        
        return result
    
    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        cache_key: str,
        timestamp: str
    ) -> LLMCallResult:
        """
        带重试的 API 调用
        
        指数退避策略：delay = base * (2 ^ retry) + random_jitter
        """
        last_error: Optional[Exception] = None
        retries = 0
        
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                retries = attempt
                self._stats["retries"] += 1
                
                # 计算退避延迟
                delay = min(
                    self.config.retry_base_delay_s * (2 ** (attempt - 1)),
                    self.config.retry_max_delay_s
                )
                # 添加 jitter
                jitter = delay * self.config.retry_jitter * random.random()
                actual_delay = delay + jitter
                
                print(f"[LLMClient] 重试 {attempt}/{self.config.max_retries}，等待 {actual_delay:.2f}s...")
                time.sleep(actual_delay)
            
            start_time = time.time()
            
            try:
                # 构建请求参数
                request_params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                
                # Qwen3 特殊参数
                if "qwen" in self.config.model.lower():
                    request_params["extra_body"] = {
                        "enable_thinking": self.config.enable_thinking
                    }
                
                # 调用 API
                response = self._client.chat.completions.create(**request_params)
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # 解析响应
                choice = response.choices[0] if response.choices else None
                raw_text = choice.message.content if choice else ""
                finish_reason = choice.finish_reason if choice else None
                
                # Token 使用
                usage = response.usage
                tokens_prompt = usage.prompt_tokens if usage else 0
                tokens_completion = usage.completion_tokens if usage else 0
                tokens_total = usage.total_tokens if usage else 0
                
                return LLMCallResult(
                    success=True,
                    model=self.config.model,
                    finish_reason=finish_reason,
                    raw_text=raw_text,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    retries=retries,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    tokens_total=tokens_total,
                    timestamp=timestamp,
                    cache_key=cache_key
                )
                
            except self.RETRYABLE_EXCEPTIONS as e:
                last_error = e
                print(f"[LLMClient] 可重试错误: {type(e).__name__}: {e}")
                continue
                
            except APIError as e:
                # 检查是否是可重试的状态码
                status_code = getattr(e, 'status_code', None)
                if status_code in self.RETRYABLE_STATUS_CODES:
                    last_error = e
                    print(f"[LLMClient] HTTP {status_code} 错误，重试...")
                    continue
                
                # 不可重试的 API 错误
                latency_ms = int((time.time() - start_time) * 1000)
                return LLMCallResult(
                    success=False,
                    model=self.config.model,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    retries=retries,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=timestamp,
                    cache_key=cache_key
                )
                
            except Exception as e:
                # 未知错误，不重试
                latency_ms = int((time.time() - start_time) * 1000)
                return LLMCallResult(
                    success=False,
                    model=self.config.model,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    retries=retries,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    timestamp=timestamp,
                    cache_key=cache_key
                )
        
        # 所有重试都失败
        latency_ms = int((time.time() - start_time) * 1000)
        return LLMCallResult(
            success=False,
            model=self.config.model,
            latency_ms=latency_ms,
            cache_hit=False,
            retries=retries,
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            error_message=str(last_error) if last_error else "Max retries exceeded",
            timestamp=timestamp,
            cache_key=cache_key
        )
    
    def _append_log(self, result: LLMCallResult) -> None:
        """追加日志到 JSONL 文件"""
        if not self.config.log_file:
            return
        
        try:
            # 确保目录存在
            log_dir = os.path.dirname(self.config.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # 追加写入（并发安全：使用 append 模式）
            with open(self.config.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result.to_log_dict(), ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"[LLMClient] 写入日志失败: {e}")
    
    def append_log_to_file(self, result: LLMCallResult, filepath: str) -> bool:
        """
        追加日志到指定文件
        
        Args:
            result: 调用结果
            filepath: 日志文件路径
        
        Returns:
            是否成功
        """
        try:
            log_dir = os.path.dirname(filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result.to_log_dict(), ensure_ascii=False) + '\n')
            return True
            
        except Exception as e:
            print(f"[LLMClient] 写入日志失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        # 计算派生指标
        if stats["total_calls"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_calls"]
            stats["failure_rate"] = stats["failures"] / stats["total_calls"]
        else:
            stats["cache_hit_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        if stats["api_calls"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["api_calls"]
            stats["avg_retries"] = stats["retries"] / stats["api_calls"]
        else:
            stats["avg_latency_ms"] = 0.0
            stats["avg_retries"] = 0.0
        
        # 添加缓存统计
        if self._cache:
            stats["cache"] = self._cache.stats()
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "retries": 0,
            "failures": 0,
            "total_tokens": 0,
            "total_latency_ms": 0
        }
    
    def clear_cache(self) -> int:
        """
        清空缓存
        
        Returns:
            删除的缓存条目数
        """
        if self._cache:
            return self._cache.clear()
        return 0


# ============================================================================
# 便捷函数
# ============================================================================

def create_llm_client(
    api_key: Optional[str] = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = "https://api-inference.modelscope.cn/v1",
    model: str = "Qwen/Qwen3-32B",
    cache_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    便捷函数：创建 LLM 客户端
    
    Args:
        api_key: API Key（可选，默认从环境变量读取）
        api_key_env: 环境变量名
        base_url: API 端点
        model: 模型名
        cache_dir: 缓存目录
        log_file: 日志文件
        **kwargs: 其他 LLMConfig 参数
    
    Returns:
        LLMClient
    """
    config = LLMConfig(
        api_key=api_key or "",
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        cache_dir=cache_dir,
        log_file=log_file,
        **kwargs
    )
    return LLMClient(config)


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    print("=== LLM Client Module Test ===\n")
    
    # 测试 JSON 抽取
    print("1. JSON 抽取测试")
    
    test_cases = [
        ('{"a": 1}', "direct"),
        ('```json\n{"a": 2}\n```', "code_fence"),
        ('思考中...\n{"a": 3}\n好的', "brace_search"),
        ('```\n{"a": 4}\n```', "code_fence"),
        ('no json here', "failed"),
    ]
    
    for text, expected_method in test_cases:
        result, method = extract_json_from_text(text)
        status = "✓" if method == expected_method else "✗"
        print(f"  {status} {expected_method:12} -> {method:12}: {text[:30]}...")
    
    # 测试缓存 key
    print("\n2. 缓存 key 测试")
    key1 = LLMCache.compute_cache_key(
        "model-a", [{"role": "user", "content": "hello"}], 0.0, 1.0, 256
    )
    key2 = LLMCache.compute_cache_key(
        "model-a", [{"role": "user", "content": "hello"}], 0.0, 1.0, 256
    )
    key3 = LLMCache.compute_cache_key(
        "model-a", [{"role": "user", "content": "hello!"}], 0.0, 1.0, 256
    )
    
    print(f"  相同输入 key 相同: {key1 == key2}")
    print(f"  不同输入 key 不同: {key1 != key3}")
    print(f"  Key 示例: {key1[:16]}...")
    
    print("\n✓ 模块测试完成")
