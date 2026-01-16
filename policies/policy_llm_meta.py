"""
LLM 元参数策略 - 使用 LLM（或模拟 LLM）输出元参数 JSON

特点：
- LLM 只输出元参数 JSON（触发阈值/冻结时长/惩罚权重）
- CP-SAT 仍然是唯一的排程优化器
- 提供 JSON schema 校验
- 校验失败时 fallback 到固定策略
- 预留真实 LLM 接口（默认不启用）

策略类型：
- MockLLMPolicy: 根据特征确定性输出 JSON（温度=0 等价）
- LLMInterfacePolicy: 真实 LLM 调用接口（预留，不启用）
"""

import json
import os
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from policies.base import BasePolicy, MetaParams
from disturbance import SimulationState
from config import Config
from solver_cpsat import Plan
from features import compute_state_features, StateFeatures


# ============================================================================
# JSON Schema 定义
# ============================================================================

META_PARAMS_SCHEMA = {
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
DEFAULT_META_PARAMS = {
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


def validate_meta_params_json(
    json_str: str,
    schema: dict = META_PARAMS_SCHEMA
) -> ValidationResult:
    """
    校验元参数 JSON
    
    校验内容：
    1. JSON 语法正确性
    2. 必需字段存在
    3. 字段类型正确
    4. 值在有效范围内
    5. 缺失字段补默认值
    
    Args:
        json_str: JSON 字符串
        schema: JSON Schema
    
    Returns:
        ValidationResult
    """
    errors = []
    warnings = []
    
    # 1. 解析 JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"JSON 解析失败: {e}"]
        )
    
    if not isinstance(data, dict):
        return ValidationResult(
            is_valid=False,
            errors=["JSON 根元素必须是对象"]
        )
    
    # 2. 检查必需字段
    required = schema.get("required", [])
    for field_name in required:
        if field_name not in data:
            errors.append(f"缺少必需字段: {field_name}")
    
    # 3. 校验每个字段
    props = schema.get("properties", {})
    validated_params = {}
    
    for field_name, field_schema in props.items():
        if field_name in data:
            value = data[field_name]
            field_type = field_schema.get("type")
            
            # 类型检查
            if field_type == "number":
                if not isinstance(value, (int, float)):
                    errors.append(f"{field_name} 必须是数值类型")
                    continue
                value = float(value)
            elif field_type == "integer":
                if not isinstance(value, int):
                    if isinstance(value, float) and value.is_integer():
                        value = int(value)
                        warnings.append(f"{field_name} 已转换为整数")
                    else:
                        errors.append(f"{field_name} 必须是整数类型")
                        continue
            
            # 范围检查
            min_val = field_schema.get("minimum")
            max_val = field_schema.get("maximum")
            
            if min_val is not None and value < min_val:
                warnings.append(f"{field_name}={value} 小于最小值 {min_val}，已截断")
                value = min_val
            
            if max_val is not None and value > max_val:
                warnings.append(f"{field_name}={value} 大于最大值 {max_val}，已截断")
                value = max_val
            
            validated_params[field_name] = value
        else:
            # 使用默认值
            default = field_schema.get("default")
            if default is not None:
                validated_params[field_name] = default
                warnings.append(f"{field_name} 使用默认值 {default}")
    
    # 如果有严重错误（必需字段缺失），返回失败
    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    return ValidationResult(
        is_valid=True,
        params=validated_params,
        warnings=warnings
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
    now: int
    features: Dict[str, Any]
    raw_output: str
    validated_params: Optional[Dict[str, Any]]
    validation_errors: List[str]
    validation_warnings: List[str]
    used_fallback: bool
    final_params: Dict[str, Any]


class LLMPolicyLogger:
    """LLM 策略日志记录器"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.logs: List[LLMDecisionLog] = []
    
    def log_decision(
        self,
        now: int,
        features: StateFeatures,
        raw_output: str,
        validation_result: ValidationResult,
        used_fallback: bool,
        final_params: MetaParams
    ):
        """记录一次决策"""
        log = LLMDecisionLog(
            timestamp=datetime.now().isoformat(),
            now=now,
            features=features.to_dict(),
            raw_output=raw_output,
            validated_params=validation_result.params,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            used_fallback=used_fallback,
            final_params={
                "w_delay": final_params.w_delay,
                "w_shift": final_params.w_shift,
                "w_switch": final_params.w_switch,
                "freeze_horizon": final_params.freeze_horizon
            }
        )
        self.logs.append(log)
    
    def save_logs(self, filepath: str):
        """保存日志到文件"""
        if not self.logs:
            return
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for log in self.logs:
                f.write(json.dumps(asdict(log), ensure_ascii=False) + '\n')
    
    def clear(self):
        """清空日志"""
        self.logs = []


# ============================================================================
# Mock LLM 策略 (确定性)
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
    
    规则示例：
    - 高窗口损失 (window_loss_pct > 0.3): 增大 freeze_horizon
    - 高 pad outage: 降低 w_switch 以允许切换
    - 紧急任务多: 增大 w_delay
    """
    
    def __init__(
        self,
        policy_name: str = "mockllm",
        log_dir: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Args:
            policy_name: 策略名称
            log_dir: 日志目录
            enable_logging: 是否记录日志
        """
        self._policy_name = policy_name
        self._enable_logging = enable_logging
        self._logger = LLMPolicyLogger(log_dir) if enable_logging else None
        
        # 状态追踪
        self._prev_window_slots = None
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
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """
        根据特征决策元参数
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
        
        # 2. 生成 JSON（模拟 LLM 输出）
        raw_json = self._generate_mock_json(features)
        
        # 3. 校验 JSON
        validation = validate_meta_params_json(raw_json)
        
        # 4. 确定最终参数
        if validation.is_valid:
            final_params = json_to_meta_params(validation.params)
            used_fallback = False
        else:
            final_params = self._fallback_params
            used_fallback = True
            self._fallback_count += 1
        
        # 5. 记录日志
        if self._logger:
            self._logger.log_decision(
                now=now,
                features=features,
                raw_output=raw_json,
                validation_result=validation,
                used_fallback=used_fallback,
                final_params=final_params
            )
        
        return final_params, None
    
    def _generate_mock_json(self, features: StateFeatures) -> str:
        """
        根据特征生成元参数 JSON（确定性规则）
        
        这是一个简单的规则引擎，模拟 LLM 的行为
        """
        # 基础参数
        w_delay = 10.0
        w_shift = 1.0
        w_switch = 5.0
        freeze_horizon = 12
        
        # 规则 1: 高窗口损失 -> 增大冻结视野（保护已有计划）
        if features.window_loss_pct > 0.3:
            freeze_horizon = min(24, freeze_horizon + 6)
            w_shift = min(5.0, w_shift * 2)  # 增大 shift 惩罚
        
        # 规则 2: Pad outage 影响任务 -> 降低 switch 惩罚
        if features.pad_outage_task_count > 0:
            w_switch = max(1.0, w_switch - 2.0)
        
        # 规则 3: 高 outage 时长 -> 缩短冻结视野（允许重排）
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
        if self._logger:
            self._logger.clear()
    
    def save_logs(self, filepath: str):
        """保存日志"""
        if self._logger:
            self._logger.save_logs(filepath)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / self._call_count if self._call_count > 0 else 0.0
        }
    
    def __repr__(self) -> str:
        return f"MockLLMPolicy(name={self.name})"


# ============================================================================
# 真实 LLM 接口（预留，默认不启用）
# ============================================================================

class LLMInterfacePolicy(BasePolicy):
    """
    真实 LLM 接口策略（预留）
    
    ⚠️ 警告：默认不启用，启用会破坏可复现性
    
    接口设计：
    - 调用外部 LLM API
    - 传入状态特征作为 prompt
    - 解析 LLM 返回的 JSON
    - 提供 fallback 机制
    
    使用前提：
    - 必须设置 enable=True
    - 必须提供 API 配置
    - 结果不可复现（非确定性）
    """
    
    def __init__(
        self,
        api_endpoint: str = "",
        api_key: str = "",
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        enable: bool = False,  # 默认不启用
        policy_name: str = "llm_interface",
        log_dir: Optional[str] = None
    ):
        """
        Args:
            api_endpoint: LLM API 端点
            api_key: API 密钥
            model_name: 模型名称
            temperature: 温度参数
            enable: 是否启用（默认 False）
            policy_name: 策略名称
            log_dir: 日志目录
        """
        self._api_endpoint = api_endpoint
        self._api_key = api_key
        self._model_name = model_name
        self._temperature = temperature
        self._enable = enable
        self._policy_name = policy_name
        self._logger = LLMPolicyLogger(log_dir)
        
        # 状态
        self._prev_window_slots = None
        self._call_count = 0
        self._fallback_count = 0
        
        # Fallback
        self._fallback_params = MetaParams(
            w_delay=10.0,
            w_shift=1.0,
            w_switch=5.0,
            freeze_horizon=12
        )
        
        if not enable:
            print(f"⚠️ {self.__class__.__name__} 未启用，将使用 fallback 参数")
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """决策"""
        self._call_count += 1
        
        # 如果未启用，直接返回 fallback
        if not self._enable:
            self._fallback_count += 1
            return self._fallback_params, None
        
        # 计算特征
        features, curr_window_slots = compute_state_features(
            tasks=state.tasks,
            pads=state.pads,
            current_plan=state.current_plan,
            now=now,
            config=config,
            completed_tasks=state.completed_tasks,
            prev_window_slots=self._prev_window_slots
        )
        self._prev_window_slots = curr_window_slots
        
        # 调用 LLM（预留）
        try:
            raw_json = self._call_llm(features)
            validation = validate_meta_params_json(raw_json)
            
            if validation.is_valid:
                final_params = json_to_meta_params(validation.params)
                used_fallback = False
            else:
                final_params = self._fallback_params
                used_fallback = True
                self._fallback_count += 1
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            raw_json = f"ERROR: {e}"
            validation = ValidationResult(is_valid=False, errors=[str(e)])
            final_params = self._fallback_params
            used_fallback = True
            self._fallback_count += 1
        
        # 记录日志
        self._logger.log_decision(
            now=now,
            features=features,
            raw_output=raw_json,
            validation_result=validation,
            used_fallback=used_fallback,
            final_params=final_params
        )
        
        return final_params, None
    
    def _call_llm(self, features: StateFeatures) -> str:
        """
        调用 LLM API（预留实现）
        
        实际使用时需要：
        1. 构建 prompt
        2. 调用 API
        3. 解析响应
        """
        # 预留实现 - 实际调用时需要实现
        raise NotImplementedError(
            "LLM API 调用未实现。请：\n"
            "1. 实现 _call_llm 方法\n"
            "2. 或使用 MockLLMPolicy 进行确定性测试"
        )
    
    def _build_prompt(self, features: StateFeatures) -> str:
        """构建 prompt（预留）"""
        feature_text = json.dumps(features.to_dict(), indent=2, ensure_ascii=False)
        
        prompt = f"""你是一个火箭发射排程优化助手。根据当前状态特征，输出元参数 JSON。

## 当前状态特征
```json
{feature_text}
```

## 输出要求
请输出一个 JSON 对象，包含以下字段：
- w_delay: 延迟惩罚权重 (0.1-100)
- w_shift: 时间偏移惩罚权重 (0-50)
- w_switch: Pad 切换惩罚权重 (0-50)
- freeze_horizon: 冻结视野 slots (0-72)

## 决策原则
- 高窗口损失时增大冻结视野
- Pad outage 多时降低切换惩罚
- 紧急任务多时增大延迟惩罚
- 保持计划稳定性

请只输出 JSON，不要其他内容。
"""
        return prompt
    
    def reset(self) -> None:
        """重置"""
        self._prev_window_slots = None
        self._call_count = 0
        self._fallback_count = 0
        self._logger.clear()
    
    def save_logs(self, filepath: str):
        """保存日志"""
        self._logger.save_logs(filepath)
    
    def get_stats(self) -> dict:
        return {
            "policy_name": self._policy_name,
            "enabled": self._enable,
            "call_count": self._call_count,
            "fallback_count": self._fallback_count
        }
    
    def __repr__(self) -> str:
        return f"LLMInterfacePolicy(name={self.name}, enabled={self._enable})"


# ============================================================================
# 便捷创建函数
# ============================================================================

def create_mock_llm_policy(
    log_dir: Optional[str] = None,
    enable_logging: bool = True
) -> MockLLMPolicy:
    """创建 Mock LLM 策略"""
    return MockLLMPolicy(
        policy_name="mockllm",
        log_dir=log_dir,
        enable_logging=enable_logging
    )


# ============================================================================
# 模块测试
# ============================================================================

if __name__ == "__main__":
    print("=== LLM Meta Policy Test ===\n")
    
    # 测试 JSON 校验
    print("1. JSON 校验测试")
    
    # 有效 JSON
    valid_json = '{"w_delay": 15.0, "w_shift": 2.0, "w_switch": 8.0, "freeze_horizon": 18}'
    result = validate_meta_params_json(valid_json)
    print(f"  有效 JSON: valid={result.is_valid}, params={result.params}")
    
    # 缺少字段
    partial_json = '{"w_delay": 20.0}'
    result = validate_meta_params_json(partial_json)
    print(f"  缺少字段: valid={result.is_valid}, errors={result.errors[:1]}")
    
    # 无效 JSON
    invalid_json = '{w_delay: 10}'
    result = validate_meta_params_json(invalid_json)
    print(f"  无效 JSON: valid={result.is_valid}, errors={result.errors[:1]}")
    
    # 范围越界
    overflow_json = '{"w_delay": 200, "w_shift": -5, "w_switch": 10, "freeze_horizon": 100}'
    result = validate_meta_params_json(overflow_json)
    print(f"  范围越界: valid={result.is_valid}, warnings={result.warnings[:2]}")
    
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
    
    print("\n✓ 测试完成")
