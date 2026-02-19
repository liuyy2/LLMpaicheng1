"""
配置模块 - 全局常量与默认配置
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict


@dataclass
class Config:
    """全局配置"""
    schema_version: str = "v2_1"             # V2.5 (v2_1) only
    # 时间离散化
    slot_minutes: int = 15                   # 15分钟/格 (比10分钟更整洁，比30分钟更精确)
    rolling_interval: int = 12                # 12 * 15m = 180m = 3小时 (符合"半个班次"的决策频率)
    horizon_slots: int = 96                  # 96 * 15m = 1440m = 24小时 (经典的 Rolling Horizon 长度)
    sim_total_slots: int = 960                # 960 * 15m = 240小时 = 10天
    
    # 冻结
    freeze_horizon: int = 12                  # 2h = 12 slots
    
    # 默认权重
    default_w_delay: float = 10.0
    default_w_shift: float = 1.0
    default_w_switch: float = 5.0

    # ========== V3 二阶段参数 ==========
    use_two_stage_solver: bool = True
    default_epsilon_solver: float = 0.05
    default_kappa_win: float = 12.0
    default_kappa_seq: float = 6.0
    stage1_time_ratio: float = 0.4
    
    # 扰动参数
    p_weather: float = 0.02
    weather_duration_range: Tuple[int, int] = (6, 18)
    p_pad_outage: float = 0.01
    outage_duration_range: Tuple[int, int] = (3, 12)
    sigma_duration: float = 0.1
    
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

    # ========== V2.1 基础配置 ==========
    resource_ids: List[str] = field(default_factory=lambda: [
        "R_pad", "R1", "R2", "R3", "R4"
    ])
    num_missions: Optional[int] = None          # 若设置则固定任务数（优先级最高）
    num_missions_range: Tuple[int, int] = (15, 20)  # 已废弃：仅当 num_missions=None 且无 difficulty 时回退
    ops_per_mission: int = 7  # 增加为 7 (含 Op3b)
    op_duration_range: Tuple[int, int] = (4, 12)
    op6_windows_range: Tuple[int, int] = (2, 5)
    op6_window_length_range: Tuple[int, int] = (2, 6)
    closure_count_range: Tuple[int, int] = (0, 1)
    closure_duration_hours_range: Tuple[int, int] = (4, 12)
    op5_max_wait_hours: int = 24
    
    # ========== V2.5 Range Calendar + Range Test ==========
    enable_range_calendar: bool = True
    enable_range_test_asset: bool = True
    weather_mode: str = "range_closure"  # "legacy" | "range_closure"
    
    # Range Calendar: 每天 3 个窗口，每窗口 16 slots (4h)
    range_calendar_windows_per_day: int = 3
    range_calendar_window_length: int = 16  # slots (= 4h)
    range_calendar_window_starts: List[int] = field(default_factory=lambda: [12, 40, 68])  # 相对每天起始
    
    # Range Test 资源
    range_test_resource_id: str = "R_range_test"
    range_test_capacity: int = 1
    
    # Op3b 参数
    op3b_duration_slots: int = 2  # 默认 0.5h (2*15min)
    
    # Range closure 扰动
    max_resample_attempts_for_closure: int = 10
    range_closure_duration_range: Tuple[int, int] = (6, 18)  # slots
    
    # 扰动概率（weather 触发 range_closure）
    p_weather_light: float = 0.05
    p_weather_medium: float = 0.07
    p_weather_heavy: float = 0.10
    
    # 释放时间扰动幅度 (slots)，0 = 不启用
    release_jitter_slots: int = 0
    



# ========== Difficulty Preset 映射 ==========
# 按扰动档位固定任务数量（路A 10天配置）
MISSIONS_BY_DIFFICULTY: Dict[str, int] = {
    "light": 15,
    "medium": 20,
    "heavy": 25,
}

DIFFICULTY_DISTURBANCE: Dict[str, Dict[str, float]] = {
    "light": {
        "p_weather": 0.05,
        "p_pad_outage": 0.02,
        "sigma_duration": 0.12,
        "release_jitter_slots": 2,
    },
    "medium": {
        "p_weather": 0.07,
        "p_pad_outage": 0.03,
        "sigma_duration": 0.2,
        "release_jitter_slots": 3,
    },
    "heavy": {
        "p_weather": 0.10,
        "p_pad_outage": 0.05,
        "sigma_duration": 0.3,
        "release_jitter_slots": 5,
    },
}

# Slack 倍率（按档位控制 due 的宽松度）
SLACK_MULTIPLIER_BY_DIFFICULTY: Dict[str, float] = {
    "light": 1.5,
    "medium": 1.2,
    "heavy": 1.0,
}


def make_config_for_difficulty(difficulty: str, num_missions_override: Optional[int] = None, **kwargs) -> "Config":
    """
    根据 difficulty 档位创建 Config 实例。
    - difficulty: 'light' | 'medium' | 'heavy'
    - num_missions_override: 若非 None 则覆盖档位默认值
    - kwargs 中的同名参数会覆盖档位默认值
    """
    if difficulty not in MISSIONS_BY_DIFFICULTY:
        raise ValueError(f"Unknown difficulty: {difficulty!r}. Must be one of {list(MISSIONS_BY_DIFFICULTY.keys())}")
    dist = DIFFICULTY_DISTURBANCE[difficulty]
    n_missions = num_missions_override if num_missions_override is not None else MISSIONS_BY_DIFFICULTY[difficulty]

    # 构建基础参数，kwargs 优先覆盖
    base = dict(
        num_missions=n_missions,
        p_weather=dist["p_weather"],
        p_weather_light=dist["p_weather"],
        p_weather_medium=dist["p_weather"],
        p_weather_heavy=dist["p_weather"],
        p_pad_outage=dist["p_pad_outage"],
        sigma_duration=dist["sigma_duration"],
        release_jitter_slots=int(dist.get("release_jitter_slots", 0)),
    )
    base.update(kwargs)
    return Config(**base)


# 默认配置实例
DEFAULT_CONFIG = Config()
