"""
固定权重策略 - 使用固定的元参数

特点：
- 固定触发阈值（每个 rolling interval 都重新求解）
- 固定 freeze_horizon
- 固定权重 (w_delay, w_shift, w_switch)
"""

from typing import Optional, Tuple, Any

from policies.base import BasePolicy, MetaParams
from config import Config


class FixedWeightPolicy(BasePolicy):
    """
    固定权重策略 (Baseline B1)
    
    行为：
    - 每次 rolling 都调用 CP-SAT 求解
    - 使用固定的权重配置
    - 使用固定的冻结视野
    - 无条件触发（不检查特征）
    
    默认配置：
    - w_delay=10, w_shift=1, w_switch=5
    - freeze_horizon 使用配置默认值 (12 slots = 2h)
    """
    
    def __init__(
        self,
        w_delay: float = 10.0,
        w_shift: float = 1.0,
        w_switch: float = 5.0,
        freeze_horizon: Optional[int] = None,
        policy_name: str = "fixed"
    ):
        """
        Args:
            w_delay: delay 权重，惩罚超期任务
            w_shift: shift 权重，惩罚时间变化
            w_switch: switch 权重，惩罚 pad 切换
            freeze_horizon: 冻结视野（None 则使用配置默认值）
            policy_name: 策略名称（用于日志和结果标识）
        """
        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch
        self._freeze_horizon = freeze_horizon
        self._policy_name = policy_name
        
        # 统计信息（可选）
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    @property
    def w_delay(self) -> float:
        return self._w_delay
    
    @property
    def w_shift(self) -> float:
        return self._w_shift
    
    @property
    def w_switch(self) -> float:
        return self._w_switch
    
    @property
    def freeze_horizon(self) -> Optional[int]:
        return self._freeze_horizon
    
    def decide(
        self,
        state: Any,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """
        返回固定的元参数，由 CP-SAT 进行优化
        
        Returns:
            (MetaParams, None) - 第二项为 None 表示不直接生成计划
        """
        self._call_count += 1
        
        # 确定实际使用的 freeze_horizon
        effective_freeze = (
            self._freeze_horizon 
            if self._freeze_horizon is not None 
            else config.freeze_horizon
        )
        
        return MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=effective_freeze
        ), None
    
    def reset(self) -> None:
        """重置策略状态"""
        self._call_count = 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "w_delay": self._w_delay,
            "w_shift": self._w_shift,
            "w_switch": self._w_switch,
            "freeze_horizon": self._freeze_horizon
        }
    
    def __repr__(self) -> str:
        return (f"FixedWeightPolicy(name={self.name}, "
                f"w_delay={self._w_delay}, w_shift={self._w_shift}, "
                f"w_switch={self._w_switch}, freeze={self._freeze_horizon})")
