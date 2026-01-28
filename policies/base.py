"""
策略基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Any

# 避免循环导入，使用字符串类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Config
    from solver_cpsat import PlanV2_1


@dataclass
class MetaParams:
    """策略输出的元参数"""
    w_delay: float                            # delay 权重
    w_shift: float                            # shift 权重
    w_switch: float                           # switch 权重
    freeze_horizon: Optional[int] = None      # 可选覆盖冻结视野
    
    def to_weights(self) -> Tuple[float, float, float]:
        """返回权重元组"""
        return (self.w_delay, self.w_shift, self.w_switch)


class BasePolicy(ABC):
    """策略抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    @abstractmethod
    def decide(
        self,
        state: Any,
        now: int,
        config: "Config"
    ) -> Tuple[Optional[MetaParams], Optional["PlanV2_1"]]:
        """
        策略决策
        
        Args:
            state: 当前仿真状态
            now: 当前时刻
            config: 配置
        
        Returns:
            (meta_params, direct_plan)
            - 对于 CP-SAT 策略: 返回 (MetaParams, None)
            - 对于贪心策略: 返回 (None, Plan)
        """
        pass
    
    def reset(self) -> None:
        """重置策略状态（新 episode 开始时调用）"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
