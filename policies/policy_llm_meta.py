"""
LLM 元参数策略 - 使用 LLM（或模拟 LLM）输出元参数 JSON

特点：
- LLM 只输出元参数 JSON（触发阈值/冻结时长/惩罚权重）
- CP-SAT 仍然是唯一的排程优化器
- 提供 JSON schema 校验（三层抽取）
- 校验失败时 fallback 到固定策略
- 支持真实 LLM 接口（Qwen3-32B via ModelScope）

策略类型：
- MockLLMPolicy: 根据特征确定性输出 JSON（温度=0 等价，用于离线复现）
- RealLLMPolicy: 真实 LLM 调用接口（调用 Qwen3-32B）

严禁：LLM 直接输出任务级排程
"""

import json
import os
from typing import Optional, Tuple, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# LLM Client
try:
    from llm_client import LLMClient, LLMConfig, LLMCallResult, extract_json_from_text
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False

from policies.base import BasePolicy, MetaParams
from disturbance import SimulationState
from config import Config
from solver_cpsat import Plan
from features import compute_state_features, StateFeatures


# ============================================================================
# JSON Schema 定义
# ============================================================================

META_PARAMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["w_delay", "w_shift", "w_switch"],
    "properties": {
        "w_delay": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 100.0,
            "default": 10.0,
            "description": "延迟惩罚权重"
        },
        "w_shift": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 50.0,
            "default": 1.0,
            "description": "时间偏移惩罚权重"
        },
        "w_switch": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 50.0,
            "default": 5.0,
            "description": "Pad 切换惩罚权重"
        },
        "freeze_horizon": {
            "type": "integer",
            "minimum": 0,
            "maximum": 72,
            "default": 12,
            "description": "冻结视野（slots）"
        }
    }
}

# 默认值
DEFAULT_META_PARAMS: Dict[str, Any] = {
    "w_delay": 10.0,
    "w_shift": 1.0,
    "w_switch": 5.0,
    "freeze_horizon": 12
}


# ============================================================================
# JSON 校验与解析
# ============================================================================

@dataclass
class ValidationResult:
    """校验结果"""
    is_valid: bool
    params: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extraction_method: str = ""  # "direct" | "code_fence" | "brace_search" | "failed"


def _simple_json_extract(text: str) -> Tuple[Optional[str], str]:
    """
    简单 JSON 抽取（当 llm_client 不可用时的回退）
    
    三层抽取：
    1. 直接尝试（整个文本就是 JSON）
    2. Code fence 抽取
    3. Brace 搜索
    """
    import re
    
    if not text:
        return None, "failed"
    
    text = text.strip()
    
    # 层级 1: 直接尝试
    if text.startswith('{') and text.endswith('}'):
        try:
            json.loads(text)
            return text, "direct"
        except json.JSONDecodeError:
            pass
    
    # 层级 2: Code fence
    patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            content = match.group(1).strip()
            if content.startswith('{'):
                try:
                    json.loads(content)
                    return content, "code_fence"
                except json.JSONDecodeError:
                    pass
    
    # 层级 3: Brace search（处理嵌套）
    start = text.find('{')
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, c in enumerate(text[start:], start):
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
                    try:
                        json.loads(text[start:i+1])
                        return text[start:i+1], "brace_search"
                    except json.JSONDecodeError:
                        pass
                    break
    
    return None, "failed"


def validate_meta_params_json(
    raw_text: str,
    schema: Dict[str, Any] = META_PARAMS_SCHEMA
) -> ValidationResult:
    """
    校验元参数 JSON（三层抽取 + Schema 校验）
    
    校验流程：
    1. 三层 JSON 抽取（direct / code fence / brace search）
    2. 必需字段存在性检查
    3. 字段类型检查
    4. 值范围检查（超出范围则截断并警告）
    5. 缺失字段补默认值
    
    Args:
        raw_text: 原始文本（可能包含 code fence 或其他内容）
        schema: JSON Schema
    
    Returns:
        ValidationResult
    """
    errors: List[str] = []
    warnings: List[str] = []
    extraction_method = "failed"
    
    # 1. 三层 JSON 抽取
    if HAS_LLM_CLIENT:
        json_str, extraction_method = extract_json_from_text(raw_text)
    else:
        json_str, extraction_method = _simple_json_extract(raw_text)
    
    if json_str is None:
        return ValidationResult(
            is_valid=False,
            errors=[f"无法从文本中抽取 JSON: {raw_text[:100]}..."],
            extraction_method=extraction_method
        )
    
    # 2. 解析 JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"JSON 解析失败: {e}"],
            extraction_method=extraction_method
        )
    
    if not isinstance(data, dict):
        return ValidationResult(
            is_valid=False,
            errors=["JSON 根元素必须是对象"],
            extraction_method=extraction_method
        )
    
    # 3. 检查必需字段
    required = schema.get("required", [])
    for field_name in required:
        if field_name not in data:
            errors.append(f"缺少必需字段: {field_name}")
    
    # 4. 校验每个字段（类型 + 范围）
    props = schema.get("properties", {})
    validated_params: Dict[str, Any] = {}
    
    for field_name, field_schema in props.items():
        if field_name in data:
            value = data[field_name]
            field_type = field_schema.get("type")
            
            # 类型检查与转换
            if field_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"{field_name} 必须是数值类型，got {type(value).__name__}")
                    continue
                value = float(value)
            elif field_type == "integer":
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                    warnings.append(f"{field_name} 已从 float 转换为 integer")
                elif not isinstance(value, int):
                    errors.append(f"{field_name} 必须是整数类型，got {type(value).__name__}")
                    continue
            
            # 范围检查（超出则截断）
            min_val = field_schema.get("minimum")
            max_val = field_schema.get("maximum")
            
            if min_val is not None and value < min_val:
                warnings.append(f"{field_name}={value} < min={min_val}，已截断")
                value = min_val
            
            if max_val is not None and value > max_val:
                warnings.append(f"{field_name}={value} > max={max_val}，已截断")
                value = max_val
            
            validated_params[field_name] = value
        else:
            # 使用默认值
            default = field_schema.get("default")
            if default is not None:
                validated_params[field_name] = default
                warnings.append(f"{field_name} 使用默认值 {default}")
    
    # 如果有严重错误（必需字段缺失或类型错误），返回失败
    if errors:
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            extraction_method=extraction_method
        )
    
    return ValidationResult(
        is_valid=True,
        params=validated_params,
        warnings=warnings,
        extraction_method=extraction_method
    )


def json_to_meta_params(validated_params: Dict[str, Any]) -> MetaParams:
    """将校验后的参数转为 MetaParams"""
    return MetaParams(
        w_delay=validated_params.get("w_delay", DEFAULT_META_PARAMS["w_delay"]),
        w_shift=validated_params.get("w_shift", DEFAULT_META_PARAMS["w_shift"]),
        w_switch=validated_params.get("w_switch", DEFAULT_META_PARAMS["w_switch"]),
        freeze_horizon=validated_params.get("freeze_horizon", DEFAULT_META_PARAMS["freeze_horizon"])
    )


# ============================================================================
# LLM 策略日志
# ============================================================================

@dataclass
class LLMDecisionLog:
    """单次 LLM 决策日志"""
    timestamp: str
    episode_id: str
    t_now: int
    features: Dict[str, Any]
    
    # LLM 调用信息
    llm_cache_hit: bool
    llm_latency_ms: int
    usage_tokens: int
    
    # 解析信息
    raw_output: str
    extraction_method: str
    parsed_json: Optional[Dict[str, Any]]
    parsed_ok: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    
    # Fallback 信息
    fallback_used: bool
    fallback_reason: Optional[str]
    
    # 最终参数
    final_params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMPolicyLogger:
    """LLM 策略日志记录器"""
    
    def __init__(self, log_dir: Optional[str] = None, episode_id: str = ""):
        self.log_dir = log_dir
        self.episode_id = episode_id
        self.logs: List[LLMDecisionLog] = []
    
    def set_episode_id(self, episode_id: str) -> None:
        """设置 episode ID"""
        self.episode_id = episode_id
    
    def log_decision(
        self,
        t_now: int,
        features: StateFeatures,
        raw_output: str,
        extraction_method: str,
        validation_result: ValidationResult,
        llm_cache_hit: bool,
        llm_latency_ms: int,
        usage_tokens: int,
        fallback_used: bool,
        fallback_reason: Optional[str],
        final_params: MetaParams
    ) -> LLMDecisionLog:
        """记录一次决策"""
        log = LLMDecisionLog(
            timestamp=datetime.now().isoformat(),
            episode_id=self.episode_id,
            t_now=t_now,
            features=features.to_dict(),
            llm_cache_hit=llm_cache_hit,
            llm_latency_ms=llm_latency_ms,
            usage_tokens=usage_tokens,
            raw_output=raw_output,
            extraction_method=extraction_method,
            parsed_json=validation_result.params,
            parsed_ok=validation_result.is_valid,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            final_params={
                "w_delay": final_params.w_delay,
                "w_shift": final_params.w_shift,
                "w_switch": final_params.w_switch,
                "freeze_horizon": final_params.freeze_horizon
            }
        )
        self.logs.append(log)
        return log
    
    def save_logs(self, filepath: str) -> None:
        """保存日志到 JSONL 文件"""
        if not self.logs:
            return
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for log in self.logs:
                f.write(json.dumps(log.to_dict(), ensure_ascii=False) + '\n')
    
    def append_log(self, filepath: str, log: LLMDecisionLog) -> None:
        """追加单条日志到文件"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + '\n')
    
    def clear(self) -> None:
        """清空日志"""
        self.logs = []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        if not self.logs:
            return {"total": 0}
        
        cache_hits = sum(1 for log in self.logs if log.llm_cache_hit)
        fallbacks = sum(1 for log in self.logs if log.fallback_used)
        parse_fails = sum(1 for log in self.logs if not log.parsed_ok)
        total_tokens = sum(log.usage_tokens for log in self.logs)
        total_latency = sum(log.llm_latency_ms for log in self.logs)
        
        return {
            "total": len(self.logs),
            "cache_hits": cache_hits,
            "fallbacks": fallbacks,
            "parse_failures": parse_fails,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "cache_hit_rate": cache_hits / len(self.logs) if self.logs else 0,
            "fallback_rate": fallbacks / len(self.logs) if self.logs else 0
        }


# ============================================================================
# Prompt 构建
# ============================================================================

SYSTEM_PROMPT = """你是一个火箭发射排程优化的“元参数调参器”。你的目标是：在不显著增加延迟的前提下，尽量降低计划漂移（drift）。

重要规则：
1. 你只能输出元参数 JSON，严禁输出具体的任务排程
2. 参数将传递给 CP-SAT 优化器进行实际排程
3. 只输出 JSON，不要解释，不要代码块

默认基线（无强信号时直接返回）：
{"w_delay": 10.0, "w_shift": 1.0, "w_switch": 5.0, "freeze_horizon": 12}

建议输出范围（优先在此范围内微调）：
- w_delay: 5.0 - 30.0
- w_shift: 0.5 - 5.0
- w_switch: 2.0 - 15.0
- freeze_horizon: 6 - 24

输出格式（严格 JSON）：
{"w_delay": <number>, "w_shift": <number>, "w_switch": <number>, "freeze_horizon": <int>}"""


def build_user_prompt(features: StateFeatures) -> str:
    """构建用户 Prompt"""
    feature_text = json.dumps(features.to_dict(), indent=2, ensure_ascii=False)
    
    prompt = f"""当前状态特征：
```json
{feature_text}
```

特征说明：
- window_loss_pct: 窗口可用性损失比例 (0-1)
- window_remaining_pct: 剩余窗口比例 (0-1)
- pad_outage_overlap_hours: 未来视野内 Pad 不可用时长（小时）
- pad_outage_task_count: 受 outage 影响的任务数
- delay_increase_minutes: 预估延误增加（分钟）
- current_total_delay_minutes: 当前累计延误（分钟）
- num_tasks_in_horizon: 视野内任务数
- num_urgent_tasks: 紧急任务数
- completed_rate: 已完成任务比例
- recent_shift_count: 最近一次重排的时间变化数
- recent_switch_count: 最近一次重排的 pad 切换数

决策原则（优先稳定性，少量微调）：
- 默认返回基线：w_delay=10, w_shift=1, w_switch=5, freeze_horizon=12
- window_loss_pct 高 或 window_remaining_pct 低 → 稳定优先：w_shift↑, w_switch↑, freeze_horizon↑
- pad_outage_overlap_hours 高 且 pad_outage_task_count>0 → 允许切换：w_switch↓（但不低于 2）
- num_urgent_tasks 高 或 delay_increase_minutes 高 → 提高时效：w_delay↑
- recent_shift_count / recent_switch_count 高 → 稳定优先：w_shift↑, w_switch↑, freeze_horizon↑
- completed_rate 高 → 更保守：w_shift↑, w_switch↑

输出仅 JSON（不要代码块/解释）："""
    
    return prompt


# ============================================================================
# Mock LLM 策略 (确定性，用于离线复现)
# ============================================================================

class MockLLMPolicy(BasePolicy):
    """
    模拟 LLM 策略 - 根据特征确定性输出 JSON
    
    行为：
    - 接收状态特征作为输入
    - 根据规则确定性输出元参数 JSON（等价于 temperature=0）
    - 经过 JSON schema 校验
    - 校验失败时 fallback 到固定参数
    - 记录每次决策的输入特征和输出 JSON
    
    用途：
    - 离线复现实验
    - Baseline 对比
    """
    
    def __init__(
        self,
        policy_name: str = "mockllm",
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        episode_id: str = ""
    ):
        """
        Args:
            policy_name: 策略名称
            log_dir: 日志目录
            enable_logging: 是否记录日志
            episode_id: Episode 标识
        """
        self._policy_name = policy_name
        self._enable_logging = enable_logging
        self._log_dir = log_dir
        self._logger = LLMPolicyLogger(log_dir, episode_id) if enable_logging else None
        
        # 状态追踪
        self._prev_window_slots: Optional[Dict[str, Set[int]]] = None
        self._recent_shifts = 0
        self._recent_switches = 0
        
        # Fallback 参数
        self._fallback_params = MetaParams(
            w_delay=10.0,
            w_shift=1.0,
            w_switch=5.0,
            freeze_horizon=12
        )
        
        # 统计
        self._call_count = 0
        self._fallback_count = 0
        self._invalid_json_count = 0
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def set_episode_id(self, episode_id: str) -> None:
        """设置 episode ID"""
        if self._logger:
            self._logger.set_episode_id(episode_id)
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """根据特征决策元参数"""
        self._call_count += 1
        
        # 1. 计算特征
        features, curr_window_slots = compute_state_features(
            tasks=state.tasks,
            pads=state.pads,
            current_plan=state.current_plan,
            now=now,
            config=config,
            completed_tasks=state.completed_tasks,
            prev_window_slots=self._prev_window_slots,
            recent_shifts=self._recent_shifts,
            recent_switches=self._recent_switches
        )
        
        # 更新状态
        self._prev_window_slots = curr_window_slots
        
        # 2. 生成 JSON（模拟 LLM 输出）
        raw_json = self._generate_mock_json(features)
        
        # 3. 校验 JSON
        validation = validate_meta_params_json(raw_json)
        
        # 4. 确定最终参数
        fallback_used = False
        fallback_reason: Optional[str] = None
        
        if validation.is_valid:
            final_params = json_to_meta_params(validation.params)
        else:
            final_params = self._fallback_params
            fallback_used = True
            fallback_reason = "; ".join(validation.errors)
            self._fallback_count += 1
            self._invalid_json_count += 1
        
        # 5. 记录日志
        if self._logger:
            log = self._logger.log_decision(
                t_now=now,
                features=features,
                raw_output=raw_json,
                extraction_method=validation.extraction_method,
                validation_result=validation,
                llm_cache_hit=True,  # Mock 始终等价于缓存命中
                llm_latency_ms=0,
                usage_tokens=0,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                final_params=final_params
            )
            
            # 实时追加日志
            if self._log_dir:
                log_file = os.path.join(self._log_dir, "llm_decisions.jsonl")
                self._logger.append_log(log_file, log)
        
        return final_params, None
    
    def _generate_mock_json(self, features: StateFeatures) -> str:
        """
        根据特征生成元参数 JSON（确定性规则）
        """
        # 基础参数
        w_delay = 10.0
        w_shift = 1.0
        w_switch = 5.0
        freeze_horizon = 12
        
        # 规则 1: 高窗口损失 -> 增大冻结视野
        if features.window_loss_pct > 0.3:
            freeze_horizon = min(24, freeze_horizon + 6)
            w_shift = min(5.0, w_shift * 2)
        
        # 规则 2: Pad outage 影响任务 -> 降低 switch 惩罚
        if features.pad_outage_task_count > 0:
            w_switch = max(1.0, w_switch - 2.0)
        
        # 规则 3: 高 outage 时长 -> 缩短冻结视野
        if features.pad_outage_overlap_hours > 1.0:
            freeze_horizon = max(6, freeze_horizon - 3)
        
        # 规则 4: 紧急任务多 -> 增大 delay 惩罚
        if features.num_urgent_tasks > 3:
            w_delay = min(30.0, w_delay + features.num_urgent_tasks * 2)
        
        # 规则 5: 延迟增加大 -> 放宽稳定性约束
        if features.delay_increase_minutes > 60:
            w_shift = max(0.1, w_shift - 0.3)
            w_switch = max(1.0, w_switch - 1.0)
        
        # 规则 6: 完成率高 -> 可以更激进
        if features.completed_rate > 0.7:
            freeze_horizon = max(6, freeze_horizon - 3)
        
        # 规则 7: 最近变动大 -> 增强稳定性
        if features.recent_shift_count > 3 or features.recent_switch_count > 2:
            w_shift = min(5.0, w_shift + 1.0)
            w_switch = min(10.0, w_switch + 2.0)
            freeze_horizon = min(24, freeze_horizon + 3)
        
        # 生成 JSON
        params = {
            "w_delay": round(w_delay, 2),
            "w_shift": round(w_shift, 2),
            "w_switch": round(w_switch, 2),
            "freeze_horizon": freeze_horizon
        }
        
        return json.dumps(params, indent=2)
    
    def reset(self) -> None:
        """重置策略状态"""
        self._prev_window_slots = None
        self._recent_shifts = 0
        self._recent_switches = 0
        self._call_count = 0
        self._fallback_count = 0
        self._invalid_json_count = 0
        if self._logger:
            self._logger.clear()
    
    def save_logs(self, filepath: str) -> None:
        """保存日志"""
        if self._logger:
            self._logger.save_logs(filepath)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "fallback_count": self._fallback_count,
            "invalid_json_count": self._invalid_json_count,
            "fallback_rate": self._fallback_count / self._call_count if self._call_count > 0 else 0.0,
            "cache_hit_count": self._call_count,  # Mock 全部算缓存命中
            "total_tokens": 0,
            "total_latency_ms": 0
        }
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """获取 LLM 调用统计（与 RealLLMPolicy 接口一致）"""
        return self.get_stats()
    
    def __repr__(self) -> str:
        return f"MockLLMPolicy(name={self.name})"


# ============================================================================
# 真实 LLM 策略（调用 Qwen3-32B）
# ============================================================================

class RealLLMPolicy(BasePolicy):
    """
    真实 LLM 策略 - 调用 Qwen3-32B API
    
    特点：
    - 在"重排检查点"调用 LLM
    - 输入为特征摘要（数值特征）
    - 输出为元参数 JSON
    - 严禁：输出任务级排程
    - 失败时 fallback 到固定参数
    - 完整日志记录
    """
    
    def __init__(
        self,
        llm_config: Optional["LLMConfig"] = None,
        policy_name: str = "qwen3",
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        episode_id: str = "",
        fallback_params: Optional[MetaParams] = None
    ):
        """
        Args:
            llm_config: LLM 客户端配置
            policy_name: 策略名称
            log_dir: 日志目录
            enable_logging: 是否记录日志
            episode_id: Episode 标识
            fallback_params: 失败时的回退参数
        """
        if not HAS_LLM_CLIENT:
            raise ImportError(
                "RealLLMPolicy 需要 llm_client 模块，请确保 llm_client.py 存在"
            )
        
        self._policy_name = policy_name
        self._enable_logging = enable_logging
        self._log_dir = log_dir
        self._logger = LLMPolicyLogger(log_dir, episode_id) if enable_logging else None
        
        # LLM 客户端配置
        if llm_config is None:
            llm_config = LLMConfig(
                cache_dir=os.path.join(log_dir, "llm_cache") if log_dir else None,
                log_file=os.path.join(log_dir, "llm_raw_calls.jsonl") if log_dir else None
            )
        
        self._llm_config = llm_config
        self._llm_client: Optional[LLMClient] = None  # 延迟初始化
        
        # 状态追踪
        self._prev_window_slots: Optional[Dict[str, Set[int]]] = None
        self._recent_shifts = 0
        self._recent_switches = 0
        
        # Fallback 参数
        self._fallback_params = fallback_params or MetaParams(
            w_delay=10.0,
            w_shift=1.0,
            w_switch=5.0,
            freeze_horizon=12
        )
        
        # 统计
        self._call_count = 0
        self._fallback_count = 0
        self._invalid_json_count = 0
        self._cache_hit_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
    
    def _ensure_client(self) -> "LLMClient":
        """确保 LLM 客户端已初始化"""
        if self._llm_client is None:
            self._llm_client = LLMClient(self._llm_config)
        return self._llm_client
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def set_episode_id(self, episode_id: str) -> None:
        """设置 episode ID"""
        if self._logger:
            self._logger.set_episode_id(episode_id)
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """
        在重排检查点调用 LLM 决策
        
        流程：
        1. 计算状态特征
        2. 构建 Prompt
        3. 调用 LLM（带缓存/重试）
        4. 解析 + Schema 校验
        5. 失败则 Fallback
        6. 记录日志
        """
        self._call_count += 1
        
        # 1. 计算特征
        features, curr_window_slots = compute_state_features(
            tasks=state.tasks,
            pads=state.pads,
            current_plan=state.current_plan,
            now=now,
            config=config,
            completed_tasks=state.completed_tasks,
            prev_window_slots=self._prev_window_slots,
            recent_shifts=self._recent_shifts,
            recent_switches=self._recent_switches
        )
        
        # 更新状态
        self._prev_window_slots = curr_window_slots
        
        # 2. 调用 LLM
        client = self._ensure_client()
        user_prompt = build_user_prompt(features)
        
        llm_result = client.call(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=SYSTEM_PROMPT
        )
        
        # 更新统计
        self._total_latency_ms += llm_result.latency_ms
        self._total_tokens += llm_result.tokens_total
        if llm_result.cache_hit:
            self._cache_hit_count += 1
        
        # 3. 处理 LLM 结果
        fallback_used = False
        fallback_reason: Optional[str] = None
        
        if not llm_result.success:
            # LLM 调用失败
            fallback_used = True
            fallback_reason = f"LLM调用失败: {llm_result.error_type}: {llm_result.error_message}"
            self._fallback_count += 1
            
            validation = ValidationResult(
                is_valid=False,
                errors=[fallback_reason],
                extraction_method="failed"
            )
            final_params = self._fallback_params
        else:
            # 4. 校验 JSON
            raw_output = llm_result.raw_text or ""
            validation = validate_meta_params_json(raw_output)
            
            if validation.is_valid:
                final_params = json_to_meta_params(validation.params)
            else:
                # 校验失败，使用 fallback
                fallback_used = True
                fallback_reason = "; ".join(validation.errors)
                self._fallback_count += 1
                self._invalid_json_count += 1
                final_params = self._fallback_params
        
        # 5. 记录日志
        if self._logger:
            log = self._logger.log_decision(
                t_now=now,
                features=features,
                raw_output=llm_result.raw_text or "",
                extraction_method=validation.extraction_method,
                validation_result=validation,
                llm_cache_hit=llm_result.cache_hit,
                llm_latency_ms=llm_result.latency_ms,
                usage_tokens=llm_result.tokens_total,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                final_params=final_params
            )
            
            # 实时追加日志
            if self._log_dir:
                log_file = os.path.join(self._log_dir, "llm_decisions.jsonl")
                self._logger.append_log(log_file, log)
        
        return final_params, None
    
    def reset(self) -> None:
        """重置策略状态"""
        self._prev_window_slots = None
        self._recent_shifts = 0
        self._recent_switches = 0
        self._call_count = 0
        self._fallback_count = 0
        self._invalid_json_count = 0
        self._cache_hit_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        if self._logger:
            self._logger.clear()
        if self._llm_client:
            self._llm_client.reset_stats()
    
    def save_logs(self, filepath: str) -> None:
        """保存日志"""
        if self._logger:
            self._logger.save_logs(filepath)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "fallback_count": self._fallback_count,
            "invalid_json_count": self._invalid_json_count,
            "cache_hit_count": self._cache_hit_count,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "fallback_rate": self._fallback_count / self._call_count if self._call_count > 0 else 0.0,
            "cache_hit_rate": self._cache_hit_count / self._call_count if self._call_count > 0 else 0.0,
            "avg_latency_ms": self._total_latency_ms / self._call_count if self._call_count > 0 else 0.0
        }
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """获取 LLM 调用统计"""
        stats = self.get_stats()
        
        # 合并 LLM 客户端统计
        if self._llm_client:
            client_stats = self._llm_client.get_stats()
            stats["llm_client"] = client_stats
        
        return stats
    
    def __repr__(self) -> str:
        return f"RealLLMPolicy(name={self.name}, model={self._llm_config.model})"


# 别名（兼容旧代码）
LLMInterfacePolicy = RealLLMPolicy


# ============================================================================
# 便捷创建函数
# ============================================================================

def create_mock_llm_policy(
    log_dir: Optional[str] = None,
    enable_logging: bool = True,
    episode_id: str = ""
) -> MockLLMPolicy:
    """创建 Mock LLM 策略"""
    return MockLLMPolicy(
        policy_name="mockllm",
        log_dir=log_dir,
        enable_logging=enable_logging,
        episode_id=episode_id
    )


def create_real_llm_policy(
    api_key: Optional[str] = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = "https://api-inference.modelscope.cn/v1",
    model: str = "Qwen/Qwen3-32B",
    log_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    policy_name: str = "qwen3",
    episode_id: str = ""
) -> RealLLMPolicy:
    """
    便捷函数：创建真实 LLM 策略
    
    Args:
        api_key: API Key（可选，默认从环境变量读取）
        api_key_env: 环境变量名
        base_url: API 端点
        model: 模型名
        log_dir: 日志目录
        cache_dir: 缓存目录（默认为 log_dir/llm_cache）
        policy_name: 策略名称
        episode_id: Episode 标识
    
    Returns:
        RealLLMPolicy
    """
    if not HAS_LLM_CLIENT:
        raise ImportError("需要 llm_client 模块")
    
    if cache_dir is None and log_dir:
        cache_dir = os.path.join(log_dir, "llm_cache")
    
    llm_config = LLMConfig(
        api_key=api_key or "",
        api_key_env=api_key_env,
        base_url=base_url,
        model=model,
        cache_dir=cache_dir,
        log_file=os.path.join(log_dir, "llm_raw_calls.jsonl") if log_dir else None,
        temperature=0.0,
        max_tokens=256,
        enable_thinking=False
    )
    
    return RealLLMPolicy(
        llm_config=llm_config,
        policy_name=policy_name,
        log_dir=log_dir,
        enable_logging=True,
        episode_id=episode_id
    )


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    print("=== LLM Meta Policy Test ===\n")
    
    # 测试 JSON 校验
    print("1. JSON 校验测试")
    
    test_cases = [
        ('{"w_delay": 15.0, "w_shift": 2.0, "w_switch": 8.0, "freeze_horizon": 18}', True),
        ('```json\n{"w_delay": 20.0, "w_shift": 1.0, "w_switch": 5.0}\n```', True),
        ('thinking...\n{"w_delay": 10, "w_shift": 1, "w_switch": 5}\n\ndone', True),
        ('{"w_delay": 200, "w_shift": -5, "w_switch": 10, "freeze_horizon": 100}', True),  # 会截断
        ('{"w_delay": 10}', False),  # 缺少必需字段
        ('{invalid json}', False),
    ]
    
    for text, should_pass in test_cases:
        result = validate_meta_params_json(text)
        status = "✓" if result.is_valid == should_pass else "✗"
        print(f"  {status} valid={result.is_valid}, method={result.extraction_method}: {text[:50]}...")
        if result.warnings:
            print(f"      warnings: {result.warnings[:2]}")
        if result.errors:
            print(f"      errors: {result.errors[:2]}")
    
    print("\n2. MockLLMPolicy 测试")
    
    # 创建模拟特征
    features = StateFeatures(
        window_loss_pct=0.4,
        window_remaining_pct=0.6,
        pad_outage_overlap_hours=2.0,
        pad_outage_task_count=3,
        delay_increase_minutes=30,
        current_total_delay_minutes=0,
        num_tasks_in_horizon=10,
        num_urgent_tasks=5,
        completed_rate=0.2,
        recent_shift_count=2,
        recent_switch_count=1
    )
    
    policy = MockLLMPolicy(enable_logging=False)
    raw_json = policy._generate_mock_json(features)
    print(f"  生成的 JSON:\n{raw_json}")
    
    # 校验生成的 JSON
    validation = validate_meta_params_json(raw_json)
    print(f"  校验结果: valid={validation.is_valid}, params={validation.params}")
    
    print("\n✓ 测试完成")
