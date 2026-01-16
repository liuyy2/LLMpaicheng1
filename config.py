"""
配置模块 - 全局常量与默认配置
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """全局配置"""
    # 时间离散化
    slot_minutes: int = 10
    rolling_interval: int = 6                 # slots (60min)
    horizon_slots: int = 144                  # 24h
    sim_total_slots: int = 432                # 72h
    
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
    num_tasks_range: Tuple[int, int] = (10, 20)
    num_pads: int = 3
    windows_per_task_range: Tuple[int, int] = (1, 3)


# 默认配置实例
DEFAULT_CONFIG = Config()
