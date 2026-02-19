"""
策略模块 - 提供各种排程策略

策略类型：
- FixedWeightPolicy (B1): 固定权重 + 固定冻结
- NoFreezePolicy (B2): 无冻结 + 低稳定惩罚
- MockLLMPolicy (L1): 模拟 LLM，根据特征确定性输出元参数
- TRCGRepairPolicy: TRCG 根因修复 + LLM 决策 + 锚点 fix-and-optimize

所有策略都实现 BasePolicy 接口，可被 simulator.py 调用
"""

from policies.base import BasePolicy, MetaParams
from policies.policy_fixed import FixedWeightPolicy
from policies.policy_nofreeze import NoFreezePolicy, MinimalFreezePolicy
from policies.policy_greedy import GreedyPolicy, EDFGreedyPolicy
from policies.policy_llm_meta import (
    MockLLMPolicy,
    LLMInterfacePolicy,
    RealLLMPolicy,
    META_PARAMS_SCHEMA,
    validate_meta_params_json,
    json_to_meta_params,
    create_mock_llm_policy,
    create_real_llm_policy,
    ValidationResult,
    LLMDecisionLog,
    LLMPolicyLogger,
    build_user_prompt,
    SYSTEM_PROMPT
)
from policies.policy_llm_trcg_repair import (
    TRCGRepairPolicy,
    create_trcg_repair_policy,
)
from policies.policy_ga_repair import (
    GARepairPolicy,
    create_ga_repair_policy,
)


__all__ = [
    # 基类
    "BasePolicy",
    "MetaParams",
    
    # 固定权重策略
    "FixedWeightPolicy",
    
    # 无冻结策略
    "NoFreezePolicy",
    "MinimalFreezePolicy",
    
    # 贪心策略
    "GreedyPolicy",
    "EDFGreedyPolicy",
    
    # LLM 策略
    "MockLLMPolicy",
    "LLMInterfacePolicy",
    "RealLLMPolicy",
    "META_PARAMS_SCHEMA",
    "validate_meta_params_json",
    "json_to_meta_params",
    "create_mock_llm_policy",
    "create_real_llm_policy",
    "ValidationResult",
    "LLMDecisionLog",
    "LLMPolicyLogger",
    "build_user_prompt",
    "SYSTEM_PROMPT",
    
    # TRCG 修复策略
    "TRCGRepairPolicy",
    "create_trcg_repair_policy",
    
    # GA 修复策略
    "GARepairPolicy",
    "create_ga_repair_policy",
]


# 便捷创建函数
def create_policy(name: str, **kwargs) -> BasePolicy:
    """
    根据名称创建策略
    
    Args:
        name: 策略名称
            - "fixed": FixedWeightPolicy (默认参数)
            - "fixed_default": FixedWeightPolicy (默认参数)
            - "fixed_tuned": FixedWeightPolicy (可传入调优参数)
            - "nofreeze": NoFreezePolicy
            - "mockllm": MockLLMPolicy
            - "trcg_repair": TRCGRepairPolicy (纯启发式模式)
            - "trcg_repair_llm": TRCGRepairPolicy (真实 LLM 模式, 需提供 llm_config)
        **kwargs: 传递给策略构造函数的参数
    
    Returns:
        策略对象
    
    Example:
        >>> policy = create_policy("fixed", w_delay=15.0)
        >>> policy = create_policy("trcg_repair", log_dir="llm_logs")
    """
    name_lower = name.lower()
    
    if name_lower in ("fixed", "fixed_default"):
        return FixedWeightPolicy(policy_name=name_lower, **kwargs)
    elif name_lower == "fixed_tuned":
        return FixedWeightPolicy(policy_name="fixed_tuned", **kwargs)
    elif name_lower == "nofreeze":
        return NoFreezePolicy(policy_name="nofreeze", **kwargs)
    elif name_lower in ("greedy", "greedy_edf"):
        return GreedyPolicy(policy_name=name_lower, **kwargs)
    elif name_lower == "mockllm":
        return MockLLMPolicy(policy_name="mockllm", **kwargs)
    elif name_lower in ("trcg_repair", "trcgrepair"):
        return TRCGRepairPolicy(policy_name="trcg_repair", **kwargs)
    elif name_lower in ("trcg_repair_llm", "trcgrepair_llm"):
        return create_trcg_repair_policy(**kwargs)
    elif name_lower in ("ga_repair", "garepair", "ga-repair"):
        return GARepairPolicy(policy_name="ga_repair", **kwargs)
    else:
        raise ValueError(f"Unknown policy: {name}")


# 获取所有可用策略名称
AVAILABLE_POLICIES = [
    "fixed", "fixed_default", "fixed_tuned", "nofreeze", "greedy", "greedy_edf", "mockllm",
    "trcg_repair", "trcg_repair_llm", "ga_repair",
]
