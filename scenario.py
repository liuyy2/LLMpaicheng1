"""
场景生成模块 - 生成任务序列（V2.1 mission/operation）与扰动时间线
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from config import Config, DEFAULT_CONFIG
from solver_cpsat import Mission, Operation, Resource


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DisturbanceEvent:
    """扰动事件"""
    event_type: str                  # "weather" | "pad_outage" | "closure_change" | "duration" | "release"
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
            "config_snapshot": self.config_snapshot
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

def generate_missions_v2_1(
    seed: int,
    num_missions: int,
    config: Config
) -> List[Mission]:
    rng = random.Random(seed)
    missions = []

    sim_total = config.sim_total_slots

    for i in range(num_missions):
        mission_id = f"M{i:03d}"

        release = rng.randint(0, int(sim_total * 0.5))
        operations = []
        cumulative_duration = 0

        for op_idx in range(1, 7):
            op_id = f"{mission_id}_Op{op_idx}"
            if op_idx == 5:
                duration = 0
            else:
                duration = rng.randint(*config.op_duration_range)
            cumulative_duration += duration

            if op_idx == 1:
                resources = ["R1"]
            elif op_idx == 2:
                resources = ["R2"]
            elif op_idx == 3:
                resources = ["R3"]
            elif op_idx == 4:
                resources = ["R_pad", "R4"]
            elif op_idx == 5:
                resources = ["R_pad"]
            elif op_idx == 6:
                resources = ["R_pad", "R3"]

            precedences = [f"{mission_id}_Op{op_idx-1}"] if op_idx > 1 else []

            time_windows = []
            if op_idx == 6:
                time_windows = _generate_op6_windows(
                    rng, release, cumulative_duration, duration, sim_total, config
                )

            operations.append(Operation(
                op_id=op_id,
                mission_id=mission_id,
                op_index=op_idx,
                duration=duration,
                resources=resources,
                precedences=precedences,
                time_windows=time_windows,
                release=release
            ))

        op6_windows = operations[5].time_windows
        if op6_windows:
            due = op6_windows[0][0] + rng.randint(0, 6)
        else:
            due = release + cumulative_duration + rng.randint(10, 30)

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


def generate_resources_v2_1(
    seed: int,
    config: Config
) -> List[Resource]:
    resources = []
    rng = random.Random(seed + 2000)
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
    return resources


def generate_disturbance_timeline_v2_1(
    seed: int,
    missions: List[Mission],
    resources: List[Resource],
    config: Config
) -> List[DisturbanceEvent]:
    """
    Generate a lightweight disturbance timeline for V2.1.
    Keep it simple and deterministic per seed.
    """
    rng = random.Random(seed + 4000)
    events: List[DisturbanceEvent] = []

    sim_total = config.sim_total_slots
    slot_minutes = config.slot_minutes

    weather_min = max(1, int(round(config.weather_duration_range[0] * 60 / slot_minutes)))
    weather_max = max(weather_min, int(round(config.weather_duration_range[1] * 60 / slot_minutes)))
    outage_min = max(1, int(round(config.outage_duration_range[0] * 60 / slot_minutes)))
    outage_max = max(outage_min, int(round(config.outage_duration_range[1] * 60 / slot_minutes)))

    pad_ids = [r.resource_id for r in resources if r.resource_id.startswith("R_pad")]
    step = max(1, config.rolling_interval)

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

    # Light release/duration perturbations (optional)
    for mission in missions:
        if rng.random() < 0.05:
            shift = int(round(rng.gauss(0.0, config.sigma_release)))
            if shift != 0:
                new_release = max(0, mission.release + shift)
                events.append(DisturbanceEvent(
                    event_type="release",
                    trigger_time=min(sim_total - 1, new_release),
                    target_id=mission.mission_id,
                    params={"new_release": new_release}
                ))

        if rng.random() < 0.05:
            candidates = [op for op in mission.operations if op.op_index not in (5, 6)]
            if candidates:
                op = rng.choice(candidates)
                multiplier = max(0.5, min(1.5, rng.gauss(1.0, config.sigma_duration)))
                events.append(DisturbanceEvent(
                    event_type="duration",
                    trigger_time=min(sim_total - 1, mission.release),
                    target_id=op.op_id,
                    params={"multiplier": multiplier}
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

    num_missions = rng.randint(
        config.num_missions_range[0],
        config.num_missions_range[1]
    )

    missions = generate_missions_v2_1(seed, num_missions, config)
    resources = generate_resources_v2_1(seed, config)
    disturbances = generate_disturbance_timeline_v2_1(seed, missions, resources, config)

    config_snapshot = {
        "schema_version": "v2_1",
        "slot_minutes": config.slot_minutes,
        "rolling_interval": config.rolling_interval,
        "horizon_slots": config.horizon_slots,
        "sim_total_slots": config.sim_total_slots,
        "num_missions": num_missions,
        "ops_per_mission": config.ops_per_mission,
        "num_resources": len(resources)
    }

    return Scenario(
        seed=seed,
        schema_version="v2_1",
        missions=missions,
        resources=resources,
        disturbance_timeline=disturbances,
        config_snapshot=config_snapshot
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

    return Scenario(
        seed=data["seed"],
        schema_version='v2_1',
        missions=missions,
        resources=resources,
        disturbance_timeline=disturbances,
        config_snapshot=data.get("config_snapshot", {})
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
