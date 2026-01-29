"""
无冻结策略 - freeze_horizon = 0 或极短

特点：
- freeze_horizon = 0（完全无冻结）或极短
- 稳定惩罚权重较低（更追求减少延误）
- 每个 rolling 完全重新优化，不保护已有分配

Baseline B2：用于对比冻结机制的效果
"""

from typing import Optional, Tuple, Any

from policies.base import BasePolicy, MetaParams
from config import Config


class NoFreezePolicy(BasePolicy):
    """
    无冻结策略 (Baseline B2)
    
    行为：
    - freeze_horizon = 0，不冻结任何任务
    - 稳定惩罚权重较低（shift/switch 惩罚减少）
    - 每次 rolling 都可以完全重新安排所有任务
    
    用途：
    - 与 fixed 策略对比，评估冻结机制的价值
    - 展示无冻结导致的计划频繁变动
    """
    
    def __init__(
        self,
        w_delay: float = 1.0,
        w_shift: float = 0.0,      # 较低的 shift 惩罚
        w_switch: float = 0.0,     # 较低的 switch 惩罚
        freeze_horizon: int = 0,   # 无冻结
        policy_name: str = "nofreeze",
        use_two_stage: Optional[bool] = True,
        epsilon_solver: Optional[float] = None,
        kappa_win: Optional[float] = None,
        kappa_seq: Optional[float] = None
    ):
        """
        Args:
            w_delay: delay 权重（通常保持较高以追求准时）
            w_shift: shift 权重（设置较低，允许频繁变更）
            w_switch: switch 权重（设置较低，允许换 pad）
            freeze_horizon: 冻结视野，默认 0（无冻结）
            policy_name: 策略名称
        """
        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch
        self._freeze_horizon = freeze_horizon
        self._policy_name = policy_name
        self._use_two_stage = use_two_stage
        self._epsilon_solver = epsilon_solver
        self._kappa_win = kappa_win
        self._kappa_seq = kappa_seq
        
        # 统计
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
    def freeze_horizon(self) -> int:
        return self._freeze_horizon
    
    def decide(
        self,
        state: Any,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """
        返回低稳定惩罚、无冻结的元参数
        """
        self._call_count += 1
        
        return MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=self._freeze_horizon,
            use_two_stage=self._use_two_stage,
            epsilon_solver=self._epsilon_solver,
            kappa_win=self._kappa_win,
            kappa_seq=self._kappa_seq
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
        return (f"NoFreezePolicy(name={self.name}, "
                f"w_delay={self._w_delay}, w_shift={self._w_shift}, "
                f"w_switch={self._w_switch}, freeze={self._freeze_horizon})")


class MinimalFreezePolicy(BasePolicy):
    """
    极短冻结策略 - freeze_horizon 极短（1-2 slots）
    
    比完全无冻结略保守，但仍允许大量重排
    """
    
    def __init__(
        self,
        w_delay: float = 1.0,
        w_shift: float = 0.5,
        w_switch: float = 2.0,
        freeze_horizon: int = 2,   # 极短冻结（20分钟）
        policy_name: str = "minimal_freeze"
    ):
        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch
        self._freeze_horizon = freeze_horizon
        self._policy_name = policy_name
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return self._policy_name
    
    def decide(
        self,
        state: Any,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        self._call_count += 1
        
        return MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=self._freeze_horizon,
            use_two_stage=self._use_two_stage,
            epsilon_solver=self._epsilon_solver,
            kappa_win=self._kappa_win,
            kappa_seq=self._kappa_seq
        ), None
    
    def reset(self) -> None:
        self._call_count = 0
    
    def __repr__(self) -> str:
        return f"MinimalFreezePolicy(freeze={self._freeze_horizon})"
