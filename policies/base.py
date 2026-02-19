"""
策略基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    use_two_stage: Optional[bool] = None      # 是否启用二阶段
    epsilon_solver: Optional[float] = None    # Stage 2延迟容差
    kappa_win: Optional[float] = None         # 窗口切换等效slot
    kappa_seq: Optional[float] = None         # 序列切换等效slot

    # ========== TRCG Repair 扩展字段（默认值，向后兼容） ==========
    unlock_mission_ids: Optional[Tuple[str, ...]] = None   # 解锁集（传给 solver）
    root_cause_mission_id: Optional[str] = None            # 根因 mission
    secondary_root_cause_mission_id: Optional[str] = None  # 次根因 mission
    decision_source: str = "default"                       # llm|heuristic_fallback|forced_global|default
    fallback_reason: Optional[str] = None                  # 回退原因
    attempt_idx: int = 0                                   # 回退链当前尝试序号

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
