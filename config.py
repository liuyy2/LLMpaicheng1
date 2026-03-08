"""
Configuration helpers for scenario generation and experiment presets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Config:
    """Global simulation configuration."""

    schema_version: str = "v2_1"

    # Time discretization
    slot_minutes: int = 15
    rolling_interval: int = 12
    horizon_slots: int = 96
    sim_total_slots: int = 960

    # Freeze horizon
    freeze_horizon: int = 12

    # Default weights
    default_w_delay: float = 10.0
    default_w_shift: float = 1.0
    default_w_switch: float = 5.0

    # Two-stage solver
    use_two_stage_solver: bool = True
    default_epsilon_solver: float = 0.05
    default_kappa_win: float = 12.0
    default_kappa_seq: float = 6.0
    stage1_time_ratio: float = 0.4

    # Disturbances
    p_weather: float = 0.02
    weather_duration_range: Tuple[int, int] = (6, 18)
    p_pad_outage: float = 0.01
    outage_duration_range: Tuple[int, int] = (3, 12)
    sigma_duration: float = 0.1
    p_duration_disturbance: float = 0.05
    p_release_jitter: float = 0.15

    # Solver
    solver_timeout_s: float = 30.0
    solver_num_workers: int = 4

    # Metrics weights
    drift_alpha: float = 0.7
    drift_beta: float = 0.3

    # Legacy scenario fields
    num_tasks_range: Tuple[int, int] = (20, 30)
    num_pads: int = 2
    windows_per_task_range: Tuple[int, int] = (1, 2)

    # Scenario profile controls
    scenario_profile: str = "default"
    release_pattern: str = "uniform"      # "uniform" | "wave"
    release_upper_ratio: float = 0.65
    release_wave_count: int = 0
    release_wave_spread_slots: int = 18
    slack_multiplier_override: Optional[float] = None

    # Mission / operation generation
    resource_ids: List[str] = field(default_factory=lambda: [
        "R_pad", "R1", "R2", "R3", "R4"
    ])
    num_missions: Optional[int] = None
    num_missions_range: Tuple[int, int] = (15, 20)
    ops_per_mission: int = 7
    op_duration_range: Tuple[int, int] = (4, 12)
    op6_windows_range: Tuple[int, int] = (2, 5)
    op6_window_length_range: Tuple[int, int] = (2, 6)
    closure_count_range: Tuple[int, int] = (0, 1)
    closure_duration_hours_range: Tuple[int, int] = (4, 12)
    op5_max_wait_hours: int = 24

    # Range calendar + range test
    enable_range_calendar: bool = True
    enable_range_test_asset: bool = True
    weather_mode: str = "range_closure"
    range_calendar_windows_per_day: int = 3
    range_calendar_window_length: int = 16
    range_calendar_window_starts: List[int] = field(default_factory=lambda: [12, 40, 68])
    range_test_resource_id: str = "R_range_test"
    range_test_capacity: int = 1
    op3b_duration_slots: int = 2
    max_resample_attempts_for_closure: int = 10
    range_closure_duration_range: Tuple[int, int] = (3, 10)

    # Per-difficulty disturbance probabilities
    p_weather_light: float = 0.05
    p_weather_medium: float = 0.07
    p_weather_heavy: float = 0.10
    release_jitter_slots: int = 0


MISSIONS_BY_DIFFICULTY: Dict[str, int] = {
    "light": 15,
    "medium": 20,
    "heavy": 25,
}

DIFFICULTY_DISTURBANCE: Dict[str, Dict[str, float]] = {
    "light": {
        "p_weather": 0.04,
        "p_pad_outage": 0.01,
        "sigma_duration": 0.12,
        "release_jitter_slots": 1,
        "p_duration_disturbance": 0.06,
        "p_release_jitter": 0.10,
    },
    "medium": {
        "p_weather": 0.06,
        "p_pad_outage": 0.015,
        "sigma_duration": 0.18,
        "release_jitter_slots": 2,
        "p_duration_disturbance": 0.08,
        "p_release_jitter": 0.12,
    },
    "heavy": {
        "p_weather": 0.08,
        "p_pad_outage": 0.02,
        "sigma_duration": 0.26,
        "release_jitter_slots": 2,
        "p_duration_disturbance": 0.10,
        "p_release_jitter": 0.14,
    },
}

SLACK_MULTIPLIER_BY_DIFFICULTY: Dict[str, float] = {
    "light": 1.5,
    "medium": 1.2,
    "heavy": 1.0,
}

SCENARIO_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {},
    "local_repair": {
        "release_pattern": "wave",
        "release_upper_ratio": 0.78,
        "release_wave_count": {
            "light": 3,
            "medium": 4,
            "heavy": 5,
        },
        "release_wave_spread_slots": {
            "light": 16,
            "medium": 18,
            "heavy": 20,
        },
        "op6_windows_range": (3, 5),
        "op6_window_length_range": (6, 12),
        "range_calendar_window_length": 20,
        "range_calendar_window_starts": [8, 36, 64],
        "closure_duration_hours_range": (2, 6),
        "range_closure_duration_range": (2, 6),
        "slack_multiplier_override": {
            "light": 1.7,
            "medium": 1.45,
            "heavy": 1.25,
        },
    },
}


def _resolve_profile_value(value: Any, difficulty: str) -> Any:
    if isinstance(value, dict):
        return value.get(difficulty)
    return value


def _get_profile_overrides(profile: str, difficulty: str) -> Dict[str, Any]:
    if profile not in SCENARIO_PROFILES:
        raise ValueError(
            f"Unknown scenario_profile: {profile!r}. "
            f"Must be one of {list(SCENARIO_PROFILES.keys())}"
        )
    raw = SCENARIO_PROFILES[profile]
    resolved: Dict[str, Any] = {}
    for key, value in raw.items():
        resolved_value = _resolve_profile_value(value, difficulty)
        if resolved_value is not None:
            resolved[key] = resolved_value
    return resolved


def make_config_for_difficulty(
    difficulty: str,
    num_missions_override: Optional[int] = None,
    scenario_profile: str = "default",
    **kwargs,
) -> Config:
    """Create a Config using difficulty preset plus optional profile overlay."""
    if difficulty not in MISSIONS_BY_DIFFICULTY:
        raise ValueError(
            f"Unknown difficulty: {difficulty!r}. Must be one of {list(MISSIONS_BY_DIFFICULTY.keys())}"
        )

    dist = DIFFICULTY_DISTURBANCE[difficulty]
    n_missions = num_missions_override if num_missions_override is not None else MISSIONS_BY_DIFFICULTY[difficulty]

    base = dict(
        scenario_profile=scenario_profile,
        num_missions=n_missions,
        p_weather=dist["p_weather"],
        p_weather_light=dist["p_weather"],
        p_weather_medium=dist["p_weather"],
        p_weather_heavy=dist["p_weather"],
        p_pad_outage=dist["p_pad_outage"],
        sigma_duration=dist["sigma_duration"],
        release_jitter_slots=int(dist.get("release_jitter_slots", 0)),
        p_duration_disturbance=float(dist.get("p_duration_disturbance", 0.05)),
        p_release_jitter=float(dist.get("p_release_jitter", 0.15)),
    )
    base.update(_get_profile_overrides(scenario_profile, difficulty))
    base.update(kwargs)
    return Config(**base)


DEFAULT_CONFIG = Config()
