"""
配置模块 - 全局常量与默认配置
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    """全局配置"""
    schema_version: str = "v2_1"             # "v1" | "v2_1"
    # 时间离散化
    slot_minutes: int = 15                   # 15分钟/格 (比10分钟更整洁，比30分钟更精确)
    rolling_interval: int = 12                # 12 * 15m = 180m = 3小时 (符合"半个班次"的决策频率)
    horizon_slots: int = 96                  # 96 * 15m = 1440m = 24小时 (经典的 Rolling Horizon 长度)
    sim_total_slots: int = 480                # 480 * 15m = 120小时 = 5天
    
    # 冻结
    freeze_horizon: int = 12                  # 2h = 12 slots
    
    # 默认权重
    default_w_delay: float = 10.0
    default_w_shift: float = 1.0
    default_w_switch: float = 5.0
    
    # 扰动参数
    p_weather: float = 0.02
    weather_duration_range: Tuple[int, int] = (6, 18)
    p_pad_outage: float = 0.01
    outage_duration_range: Tuple[int, int] = (3, 12)
    sigma_duration: float = 0.1
    sigma_release: float = 2.0
    
    # 求解器
    solver_timeout_s: float = 30.0
    solver_num_workers: int = 4
    
    # 指标权重
    drift_alpha: float = 0.7                  # 时间偏移权重
    drift_beta: float = 0.3                   # pad 切换权重
    
    # 场景生成
    num_tasks_range: Tuple[int, int] = (20, 30)
    num_pads: int = 2
    windows_per_task_range: Tuple[int, int] = (1, 2)

    # ========== V2.1 ???? ==========
    resource_ids: List[str] = field(default_factory=lambda: [
        "R_pad", "R1", "R2", "R3", "R4"
    ])
    num_missions_range: Tuple[int, int] = (15, 20)
    ops_per_mission: int = 6
    op_duration_range: Tuple[int, int] = (4, 12)
    op6_windows_range: Tuple[int, int] = (2, 5)
    op6_window_length_range: Tuple[int, int] = (2, 6)
    closure_count_range: Tuple[int, int] = (0, 1)
    closure_duration_hours_range: Tuple[int, int] = (4, 12)
    op5_max_wait_hours: int = 24


# 默认配置实例
DEFAULT_CONFIG = Config()
