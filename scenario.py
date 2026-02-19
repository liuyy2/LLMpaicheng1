"""
场景生成模块 - 生成任务序列（V2.1 mission/operation）与扰动时间线
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from config import Config, DEFAULT_CONFIG, SLACK_MULTIPLIER_BY_DIFFICULTY
from solver_cpsat import Mission, Operation, Resource


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DisturbanceEvent:
    """扰动事件"""
    event_type: str                  # "weather" | "pad_outage" | "closure_change" | "duration" | "range_closure" | "release_jitter"
    trigger_time: int                # 触发 slot
    target_id: Optional[str]         # task_id 或 pad_id
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "trigger_time": self.trigger_time,
            "target_id": self.target_id,
            "params": self.params
        }


@dataclass
class Scenario:
    """Complete scenario (V2.1)"""
    seed: int
    schema_version: str = "v2_1"

    # V2.1 fields
    missions: List[Mission] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)

    disturbance_timeline: List[DisturbanceEvent] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # V2.5 Range Calendar: dict[day_index] -> list of (start_slot, end_slot)
    range_calendar: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    def get_mission(self, mission_id: str) -> Optional[Mission]:
        for m in self.missions:
            if m.mission_id == mission_id:
                return m
        return None

    def get_resource(self, resource_id: str) -> Optional[Resource]:
        for r in self.resources:
            if r.resource_id == resource_id:
                return r
        return None

    def to_dict(self) -> dict:
        base = {
            "seed": self.seed,
            "schema_version": self.schema_version,
            "disturbance_timeline": [e.to_dict() for e in self.disturbance_timeline],
            "config_snapshot": self.config_snapshot,
            "range_calendar": {
                str(day): [(s, e) for s, e in windows]
                for day, windows in self.range_calendar.items()
            }
        }
        base["missions"] = [
            {
                "mission_id": m.mission_id,
                "release": m.release,
                "due": m.due,
                "priority": m.priority,
                "operations": [
                    {
                        "op_id": op.op_id,
                        "mission_id": op.mission_id,
                        "op_index": op.op_index,
                        "duration": op.duration,
                        "resources": op.resources,
                        "precedences": op.precedences,
                        "time_windows": op.time_windows,
                        "release": op.release
                    }
                    for op in m.operations
                ]
            }
            for m in self.missions
        ]
        base["resources"] = [
            {
                "resource_id": r.resource_id,
                "capacity": r.capacity,
                "unavailable": r.unavailable
            }
            for r in self.resources
        ]

        return base

# ============================================================================
# 场景生成
# ============================================================================

def _compute_due_with_slack(
    release: int,
    cumulative_duration: int,
    sim_total: int,
    slack_multiplier: float,
    rng: random.Random,
    first_window_end: int | None = None,
) -> int:
    """根据链长 + slack 计算 due，保证 due < sim_total。

    如果提供了 first_window_end（发射窗口第一个窗口的结束 slot），
    due 不会早于 first_window_end + base_slack，以避免结构性不可行。
    """
    base_slack = max(4, int(cumulative_duration * 0.3 * slack_multiplier))
    jitter = rng.randint(0, max(1, base_slack // 2))
    due_base = release + cumulative_duration + base_slack + jitter
    # 如果有时间窗约束，due 不能早于第一个窗口结束 + slack
    if first_window_end is not None:
        due_base = max(due_base, first_window_end + base_slack)
    return min(due_base, sim_total - 1)


def generate_missions_v2_1(
    seed: int,
    num_missions: int,
    config: Config,
    slack_multiplier: float = 1.2,
) -> List[Mission]:
    rng = random.Random(seed)
    missions = []

    sim_total = config.sim_total_slots
    # release 分布在 [0, 0.65 * sim_total] 内（前 65% 区间，避免前期拥堵/后期空转）
    release_upper = int(sim_total * 0.65)

    for i in range(num_missions):
        mission_id = f"M{i:03d}"

        release = rng.randint(0, release_upper)
        operations = []
        cumulative_duration = 0

        # 生成 Op1, Op2, Op3
        for op_idx in [1, 2, 3]:
            op_id = f"{mission_id}_Op{op_idx}"
            duration = rng.randint(*config.op_duration_range)
            cumulative_duration += duration
            
            if op_idx == 1:
                resources = ["R1"]
                precedences = []
            elif op_idx == 2:
                resources = ["R2"]
                precedences = [f"{mission_id}_Op1"]
            elif op_idx == 3:
                resources = ["R3"]
                precedences = [f"{mission_id}_Op2"]
            
            operations.append(Operation(
                op_id=op_id,
                mission_id=mission_id,
                op_index=op_idx,
                duration=duration,
                resources=resources,
                precedences=precedences,
                time_windows=[],
                release=release
            ))
        
        # 生成 Op3b（新增，使用特殊标识）
        if config.enable_range_test_asset:
            op_id = f"{mission_id}_Op3b"
            duration = config.op3b_duration_slots
            cumulative_duration += duration
            resources = ["R3", config.range_test_resource_id]
            precedences = [f"{mission_id}_Op3"]
            
            # Op3b 使用 op_index = 3.5 的概念，但实际用整数，我们用负数或特殊值
            # 为了兼容性，使用 op_index=4 但在 solver 中特殊处理
            operations.append(Operation(
                op_id=op_id,
                mission_id=mission_id,
                op_index=4,  # Op3b 的特殊索引
                duration=duration,
                resources=resources,
                precedences=precedences,
                time_windows=[],
                release=release
            ))
        
        # 生成 Op4, Op5, Op6（保持原有索引 4,5,6 或调整为 5,6,7）
        # 为了向后兼容，我们将它们保持为原索引
        # 但在查询时需要注意 Op3b 的存在
        for op_idx in [4, 5, 6]:
            # 调整实际索引：如果启用了 Op3b，则 Op4-Op6 变成索引 5-7
            if config.enable_range_test_asset:
                actual_idx = op_idx + 1  # 4->5, 5->6, 6->7
            else:
                actual_idx = op_idx
            
            op_id = f"{mission_id}_Op{op_idx}"
            
            if op_idx == 4:
                resources = ["R_pad", "R4"]
                duration = rng.randint(*config.op_duration_range)
                # 前序：如果有 Op3b 则依赖 Op3b，否则依赖 Op3
                if config.enable_range_test_asset:
                    precedences = [f"{mission_id}_Op3b"]
                else:
                    precedences = [f"{mission_id}_Op3"]
            elif op_idx == 5:
                resources = ["R_pad"]
                duration = 0
                precedences = [f"{mission_id}_Op4"]
            elif op_idx == 6:
                resources = ["R_pad", "R3"]
                duration = rng.randint(*config.op_duration_range)
                precedences = [f"{mission_id}_Op5"]
            
            cumulative_duration += duration
            
            time_windows = []
            if op_idx == 6:  # Op6
                time_windows = _generate_op6_windows(
                    rng, release, cumulative_duration, duration, sim_total, config
                )
            
            operations.append(Operation(
                op_id=op_id,
                mission_id=mission_id,
                op_index=actual_idx,
                duration=duration,
                resources=resources,
                precedences=precedences,
                time_windows=time_windows,
                release=release
            ))

        op6 = operations[-1]  # 最后一个是发射操作（Op6 或 Op7）
        # 获取发射窗口的第一个窗口结束时间，用于合理化 due
        first_window_end = None
        if op6.time_windows:
            first_window_end = op6.time_windows[0][1]
        # 基于链长 + slack + 时间窗约束 计算 due
        due = _compute_due_with_slack(
            release, cumulative_duration, sim_total, slack_multiplier, rng,
            first_window_end=first_window_end
        )

        priority = round(rng.uniform(0.3, 1.0), 2)

        missions.append(Mission(
            mission_id=mission_id,
            operations=operations,
            release=release,
            due=due,
            priority=priority
        ))

    return missions


def _generate_op6_windows(
    rng: random.Random,
    release: int,
    cumulative_duration: int,
    op6_duration: int,
    sim_total: int,
    config: Config
) -> List[Tuple[int, int]]:
    num_windows = rng.randint(*config.op6_windows_range)
    min_extra = max(0, config.op6_window_length_range[0])
    max_extra = max(min_extra, config.op6_window_length_range[1])

    earliest_start = release + cumulative_duration - op6_duration
    latest_end = min(sim_total - 1, earliest_start + int(sim_total * 0.8))

    available_span = latest_end - earliest_start
    if available_span < num_windows * (op6_duration + min_extra):
        latest_end = min(
            sim_total - 1,
            earliest_start + num_windows * (op6_duration + max_extra)
        )

    windows = []
    segment_size = (latest_end - earliest_start) // max(1, num_windows)

    for w in range(num_windows):
        segment_center = earliest_start + segment_size * w + segment_size // 2
        window_length = op6_duration + rng.randint(min_extra, max_extra)

        window_start = max(earliest_start, segment_center - window_length // 2)
        window_end = min(latest_end, window_start + window_length)

        if window_start < window_end and window_end - window_start >= op6_duration:
            windows.append((window_start, window_end))

    if not windows:
        window_start = earliest_start + 5
        window_end = min(latest_end, window_start + op6_duration + max(2, min_extra))
        windows = [(window_start, window_end)]

    return windows


def _generate_closure_intervals(
    rng: random.Random,
    sim_total: int,
    slot_minutes: int,
    count_range: Tuple[int, int],
    duration_hours_range: Tuple[int, int]
) -> List[Tuple[int, int]]:
    count = rng.randint(count_range[0], count_range[1])
    if count <= 0 or sim_total <= 0:
        return []

    min_len = max(1, int(duration_hours_range[0] * 60 / slot_minutes))
    max_len = max(min_len, int(duration_hours_range[1] * 60 / slot_minutes))

    intervals: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = count * 10 + 10

    while len(intervals) < count and attempts < max_attempts:
        attempts += 1
        length = rng.randint(min_len, max_len)
        if sim_total <= length:
            break
        start = rng.randint(0, sim_total - length)
        end = start + length - 1

        overlap = False
        for s, e in intervals:
            if not (end < s or start > e):
                overlap = True
                break
        if overlap:
            continue

        intervals.append((start, end))

    intervals.sort(key=lambda x: x[0])
    return intervals


def _generate_range_calendar(
    config: Config
) -> Dict[int, List[Tuple[int, int]]]:
    """
    生成 Range 日历：每天 3 个固定窗口
    W1=[day+12, day+28), W2=[day+40, day+56), W3=[day+68, day+84)
    每段长度 16 slots (=4h)，每天总可用 12h
    
    返回: dict[day_index] -> list of (start_slot, end_slot)
    """
    range_calendar: Dict[int, List[Tuple[int, int]]] = {}
    
    sim_total = config.sim_total_slots
    slots_per_day = 96  # 24h * 4 slots/h
    num_days = (sim_total + slots_per_day - 1) // slots_per_day
    
    window_length = config.range_calendar_window_length
    window_starts = config.range_calendar_window_starts
    
    # 计算 Op6 的最小窗口要求（假设 Op6 duration 最大为 op_duration_range 的上界）
    min_op6_duration = config.op_duration_range[0]
    min_required_length = min_op6_duration + 4  # Op6_duration + 4 slots buffer
    
    for day in range(num_days):
        day_start = day * slots_per_day
        day_end = min(sim_total, (day + 1) * slots_per_day)
        
        windows = []
        for offset in window_starts:
            win_start = day_start + offset
            win_end = min(day_end, win_start + window_length)
            
            # 硬校验：窗口长度必须 >= min_required_length
            actual_length = win_end - win_start
            if actual_length < min_required_length:
                # 尝试扩大窗口
                needed = min_required_length - actual_length
                # 先尝试向后扩展
                if win_end + needed <= day_end:
                    win_end = win_end + needed
                # 否则向前扩展
                elif win_start - needed >= day_start:
                    win_start = win_start - needed
                else:
                    # 无法扩展，设为全天可用（兜底）
                    windows = [(day_start, day_end)]
                    break
            
            if win_start < day_end:
                windows.append((win_start, min(win_end, day_end)))
        
        # 如果没有因为兜底而设为全天，则保留正常窗口
        if windows:
            range_calendar[day] = windows
    
    return range_calendar


def generate_resources_v2_1(
    seed: int,
    config: Config
) -> List[Resource]:
    resources = []
    rng = random.Random(seed + 2000)
    
    # 基础资源
    for resource_id in config.resource_ids:
        unavailable = []
        if resource_id == "R_pad":
            unavailable = _generate_closure_intervals(
                rng,
                config.sim_total_slots,
                config.slot_minutes,
                config.closure_count_range,
                config.closure_duration_hours_range
            )
        resources.append(Resource(
            resource_id=resource_id,
            capacity=1,
            unavailable=unavailable
        ))
    
    # 添加 R_range_test 资源
    if config.enable_range_test_asset:
        resources.append(Resource(
            resource_id=config.range_test_resource_id,
            capacity=config.range_test_capacity,
            unavailable=[]
        ))
    
    return resources


def generate_disturbance_timeline_v2_1(
    seed: int,
    missions: List[Mission],
    resources: List[Resource],
    config: Config
) -> List[DisturbanceEvent]:
    """
    Generate a lightweight disturbance timeline for V2.1.
    V2.5: 支持 weather_mode = "range_closure"
    """
    rng = random.Random(seed + 4000)
    events: List[DisturbanceEvent] = []

    sim_total = config.sim_total_slots
    slot_minutes = config.slot_minutes

    outage_min = max(1, int(round(config.outage_duration_range[0] * 60 / slot_minutes)))
    outage_max = max(outage_min, int(round(config.outage_duration_range[1] * 60 / slot_minutes)))

    pad_ids = [r.resource_id for r in resources if r.resource_id.startswith("R_pad")]
    step = max(1, config.rolling_interval)

    # Weather 扰动 -> Range closure (V2.5)
    if config.weather_mode == "range_closure":
        closure_min = config.range_closure_duration_range[0]
        closure_max = config.range_closure_duration_range[1]
        
        # 根据时间段决定概率（简化：按天采样）
        slots_per_day = 96
        num_days = (sim_total + slots_per_day - 1) // slots_per_day
        
        for day in range(num_days):
            day_start = day * slots_per_day
            day_end = min(sim_total, (day + 1) * slots_per_day)
            
            # 简化：每天按一定概率触发一次 range_closure
            # 使用 p_weather_light 作为基础概率
            if rng.random() < config.p_weather_light:
                closure_len = rng.randint(closure_min, closure_max)
                # closure_start 在当天范围内
                if day_end - day_start > closure_len:
                    closure_start = rng.randint(day_start, day_end - closure_len)
                else:
                    closure_start = day_start
                closure_end = min(day_end, closure_start + closure_len)
                
                events.append(DisturbanceEvent(
                    event_type="range_closure",
                    trigger_time=closure_start,
                    target_id=None,
                    params={
                        "day": day,
                        "closure_start": closure_start,
                        "closure_end": closure_end
                    }
                ))
    else:
        # Legacy weather mode (保留原有逻辑)
        weather_min = max(1, int(round(config.weather_duration_range[0] * 60 / slot_minutes)))
        weather_max = max(weather_min, int(round(config.weather_duration_range[1] * 60 / slot_minutes)))
        
        for t in range(0, sim_total, step):
            if rng.random() < config.p_weather:
                dur = rng.randint(weather_min, weather_max)
                end = min(sim_total - 1, t + dur)
                events.append(DisturbanceEvent(
                    event_type="weather",
                    trigger_time=t,
                    target_id=None,
                    params={
                        "delete_ratio": rng.uniform(0.2, 0.6),
                        "affected_start": t,
                        "affected_end": end
                    }
                ))

    # Pad outage 扰动（保留）
    for t in range(0, sim_total, step):
        if pad_ids and rng.random() < config.p_pad_outage:
            dur = rng.randint(outage_min, outage_max)
            end = min(sim_total - 1, t + dur)
            events.append(DisturbanceEvent(
                event_type="pad_outage",
                trigger_time=t,
                target_id=rng.choice(pad_ids),
                params={
                    "outage_start": min(sim_total - 1, t + 1),
                    "outage_end": end
                }
            ))

    # Duration 扰动（只对 Op1-Op3）
    for mission in missions:
        if rng.random() < 0.05:
            candidates = [op for op in mission.operations if op.op_index in (1, 2, 3)]
            if candidates:
                op = rng.choice(candidates)
                multiplier = max(0.5, min(1.5, rng.gauss(1.0, config.sigma_duration)))
                events.append(DisturbanceEvent(
                    event_type="duration",
                    trigger_time=min(sim_total - 1, mission.release),
                    target_id=op.op_id,
                    params={"multiplier": multiplier}
                ))

    # Release jitter 扰动（释放时间向后推迟）
    jitter_max = getattr(config, 'release_jitter_slots', 0)
    if jitter_max > 0:
        for mission in missions:
            # 约 15% 的任务受影响，概率适中不会严重影响完成率
            if rng.random() < 0.15:
                jitter = rng.randint(1, jitter_max)
                events.append(DisturbanceEvent(
                    event_type="release_jitter",
                    trigger_time=mission.release,
                    target_id=mission.mission_id,
                    params={"jitter_slots": jitter}
                ))

    events.sort(key=lambda e: e.trigger_time)
    return events


def generate_scenario(
    seed: int,
    config: Config = DEFAULT_CONFIG
) -> Scenario:
    """
    Generate complete scenario (V2.1).
    """
    if config.schema_version != "v2_1":
        raise ValueError(f"Unsupported schema_version: {config.schema_version}")
    return _generate_scenario_v2_1(seed, config)


def _generate_scenario_v2_1(
    seed: int,
    config: Config
) -> Scenario:
    rng = random.Random(seed)

    # 优先级：config.num_missions > num_missions_range 随机采样
    if config.num_missions is not None:
        num_missions = config.num_missions
    else:
        num_missions = rng.randint(
            config.num_missions_range[0],
            config.num_missions_range[1]
        )

    # 根据 difficulty 获取 slack_multiplier（默认 1.2）
    _slack = 1.2
    try:
        from config import SLACK_MULTIPLIER_BY_DIFFICULTY as _slk
        for _d, _s in _slk.items():
            from config import MISSIONS_BY_DIFFICULTY as _mbd
            if _mbd.get(_d) == num_missions:
                _slack = _s
                break
    except Exception:
        pass

    missions = generate_missions_v2_1(seed, num_missions, config, slack_multiplier=_slack)
    resources = generate_resources_v2_1(seed, config)
    disturbances = generate_disturbance_timeline_v2_1(seed, missions, resources, config)
    
    # V2.5: 生成 Range Calendar
    range_calendar = {}
    if config.enable_range_calendar:
        range_calendar = _generate_range_calendar(config)

    config_snapshot = {
        "schema_version": "v2_1",
        "slot_minutes": config.slot_minutes,
        "rolling_interval": config.rolling_interval,
        "horizon_slots": config.horizon_slots,
        "sim_total_slots": config.sim_total_slots,
        "num_missions": num_missions,
        "ops_per_mission": config.ops_per_mission,
        "num_resources": len(resources),
        "enable_range_calendar": config.enable_range_calendar,
        "enable_range_test_asset": config.enable_range_test_asset,
        "weather_mode": config.weather_mode
    }

    return Scenario(
        seed=seed,
        schema_version="v2_1",
        missions=missions,
        resources=resources,
        disturbance_timeline=disturbances,
        config_snapshot=config_snapshot,
        range_calendar=range_calendar
    )


# ============================================================================
# 序列化
# ============================================================================

import json

def save_scenario(scenario: Scenario, filepath: str) -> None:
    """保存场景到 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(scenario.to_dict(), f, indent=2, ensure_ascii=False)


def load_scenario(filepath: str) -> Scenario:
    """Load scenario from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    schema_version = data.get('schema_version', 'v2_1')
    if schema_version != 'v2_1':
        raise ValueError(f"Unsupported schema_version: {schema_version}")
    return _load_scenario_v2_1(data)


def _load_scenario_v2_1(data: Dict[str, Any]) -> Scenario:
    missions = []
    for m in data.get("missions", []):
        ops = [
            Operation(
                op_id=op["op_id"],
                mission_id=op["mission_id"],
                op_index=op["op_index"],
                duration=op["duration"],
                resources=list(op.get("resources", [])),
                precedences=list(op.get("precedences", [])),
                time_windows=[tuple(w) for w in op.get("time_windows", [])],
                release=op.get("release", 0)
            )
            for op in m.get("operations", [])
        ]
        missions.append(Mission(
            mission_id=m["mission_id"],
            release=m.get("release", 0),
            due=m.get("due", 0),
            priority=m.get("priority", 1.0),
            operations=ops
        ))

    resources = [
        Resource(
            resource_id=r["resource_id"],
            capacity=r.get("capacity", 1),
            unavailable=[tuple(u) for u in r.get("unavailable", [])]
        )
        for r in data.get("resources", [])
    ]

    disturbances = [
        DisturbanceEvent(
            event_type=e["event_type"],
            trigger_time=e["trigger_time"],
            target_id=e["target_id"],
            params=e["params"]
        )
        for e in data.get("disturbance_timeline", [])
    ]
    
    # V2.5: 加载 range_calendar
    range_calendar_data = data.get("range_calendar", {})
    range_calendar = {}
    for day_str, windows in range_calendar_data.items():
        range_calendar[int(day_str)] = [tuple(w) for w in windows]

    return Scenario(
        seed=data["seed"],
        schema_version='v2_1',
        missions=missions,
        resources=resources,
        disturbance_timeline=disturbances,
        config_snapshot=data.get("config_snapshot", {}),
        range_calendar=range_calendar
    )


# ============================================================================
# ??????
# ============================================================================

if __name__ == "__main__":
    print("=== Scenario Generation Test ===\n")

    scenario = generate_scenario(seed=42)

    print(f"Seed: {scenario.seed}")
    print(f"Schema: {scenario.schema_version}")

    print(f"Missions: {len(scenario.missions)}")
    print(f"Resources: {len(scenario.resources)}")
    print(f"Disturbance events: {len(scenario.disturbance_timeline)}")

    print("\nMissions:")
    for m in scenario.missions[:5]:
        print(f"  {m.mission_id}: release={m.release}, due={m.due}, priority={m.priority}")
    if len(scenario.missions) > 5:
        print(f"  ... and {len(scenario.missions) - 5} more")

    print("\nResources:")
    for r in scenario.resources:
        print(f"  {r.resource_id} cap={r.capacity}")

    print("\nDisturbance events (first 10):")
    for e in scenario.disturbance_timeline[:10]:
        print(f"  t={e.trigger_time}: {e.event_type} -> {e.target_id}, {e.params}")

    event_counts = {}
    for e in scenario.disturbance_timeline:
        event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
    print(f"\nEvent counts: {event_counts}")
