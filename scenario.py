"""
场景生成模块 - 生成任务、Pad 和扰动时间线
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from config import Config, DEFAULT_CONFIG
from solver_cpsat import Task, Pad


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class DisturbanceEvent:
    """扰动事件"""
    event_type: str                  # "weather" | "pad_outage" | "duration" | "release"
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
    """完整场景"""
    seed: int
    tasks: List[Task]
    pads: List[Pad]
    disturbance_timeline: List[DisturbanceEvent]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None
    
    def get_pad(self, pad_id: str) -> Optional[Pad]:
        for p in self.pads:
            if p.pad_id == pad_id:
                return p
        return None
    
    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "release": t.release,
                    "duration": t.duration,
                    "windows": t.windows,
                    "due": t.due,
                    "priority": t.priority,
                    "preferred_pad": t.preferred_pad
                }
                for t in self.tasks
            ],
            "pads": [
                {"pad_id": p.pad_id, "unavailable": p.unavailable}
                for p in self.pads
            ],
            "disturbance_timeline": [e.to_dict() for e in self.disturbance_timeline],
            "config_snapshot": self.config_snapshot
        }


# ============================================================================
# 场景生成
# ============================================================================

def generate_tasks(
    seed: int,
    num_tasks: int,
    config: Config
) -> List[Task]:
    """
    生成任务列表
    
    Args:
        seed: 随机种子
        num_tasks: 任务数量
        config: 配置
    
    Returns:
        任务列表
    """
    rng = random.Random(seed)
    tasks = []
    
    sim_total = config.sim_total_slots
    horizon = config.horizon_slots
    
    for i in range(num_tasks):
        task_id = f"T{i:03d}"
        
        # release: 在仿真前 2/3 时间内随机分布
        release = rng.randint(0, int(sim_total * 0.6))
        
        # duration: 3-12 slots (30min - 2h)
        duration = rng.randint(3, 12)
        
        # windows: 1-3 个窗口段
        num_windows = rng.randint(
            config.windows_per_task_range[0],
            config.windows_per_task_range[1]
        )
        
        windows = []
        # 窗口开始点：在 release + duration 之后
        window_start_base = release + duration + rng.randint(5, 20)
        
        for w in range(num_windows):
            # 每个窗口 6-24 slots (1-4h)
            window_duration = rng.randint(6, 24)
            window_start = window_start_base + w * rng.randint(30, 60)
            window_end = min(window_start + window_duration, sim_total - 1)
            
            if window_start < sim_total and window_start < window_end:
                windows.append((window_start, window_end))
        
        # 确保至少有一个有效窗口
        if not windows:
            window_start = release + duration + 10
            window_end = min(window_start + 30, sim_total - 1)
            windows = [(window_start, window_end)]
        
        # due: 第一个窗口结束后一段时间
        first_window_end = windows[0][1]
        due = first_window_end + rng.randint(0, 20)
        
        # priority: 0.3 - 1.0
        priority = round(rng.uniform(0.3, 1.0), 2)
        
        tasks.append(Task(
            task_id=task_id,
            release=release,
            duration=duration,
            windows=windows,
            due=due,
            priority=priority,
            preferred_pad=None
        ))
    
    return tasks


def generate_pads(
    num_pads: int
) -> List[Pad]:
    """
    生成 Pad 列表
    
    Args:
        num_pads: Pad 数量
    
    Returns:
        Pad 列表（初始无不可用区间）
    """
    pads = []
    for i in range(num_pads):
        pad_id = f"PAD_{chr(65 + i)}"  # PAD_A, PAD_B, PAD_C, ...
        pads.append(Pad(pad_id=pad_id, unavailable=[]))
    return pads


def generate_disturbance_timeline(
    seed: int,
    tasks: List[Task],
    pads: List[Pad],
    config: Config
) -> List[DisturbanceEvent]:
    """
    预生成整个仿真周期的扰动时间线
    
    Args:
        seed: 随机种子
        tasks: 任务列表
        pads: Pad 列表
        config: 配置
    
    Returns:
        按 trigger_time 排序的扰动事件列表
    """
    rng = random.Random(seed + 1000)  # 不同于任务生成的种子
    events = []
    
    sim_total = config.sim_total_slots
    
    # 1. 天气扰动事件
    for t in range(sim_total):
        if rng.random() < config.p_weather:
            # 天气强度：轻(0.2)/中(0.4)/重(0.6) 删除比例
            severity = rng.choice(["light", "medium", "heavy"])
            delete_ratio = {"light": 0.2, "medium": 0.4, "heavy": 0.6}[severity]
            
            # 影响持续时间
            duration = rng.randint(
                config.weather_duration_range[0],
                config.weather_duration_range[1]
            )
            
            events.append(DisturbanceEvent(
                event_type="weather",
                trigger_time=t,
                target_id=None,
                params={
                    "severity": severity,
                    "delete_ratio": delete_ratio,
                    "duration": duration,
                    "affected_start": t,
                    "affected_end": min(t + duration, sim_total - 1)
                }
            ))
    
    # 2. Pad outage 事件
    for t in range(sim_total):
        if rng.random() < config.p_pad_outage:
            # 随机选择一个 pad
            pad_id = rng.choice([p.pad_id for p in pads])
            
            # 不可用持续时间
            duration = rng.randint(
                config.outage_duration_range[0],
                config.outage_duration_range[1]
            )
            
            events.append(DisturbanceEvent(
                event_type="pad_outage",
                trigger_time=t,
                target_id=pad_id,
                params={
                    "outage_start": t + 1,  # 下一个 slot 开始
                    "outage_end": min(t + 1 + duration, sim_total - 1)
                }
            ))
    
    # 3. Duration 扰动（在任务实际开始前应用）
    for task in tasks:
        # 50% 概率产生 duration 扰动
        if rng.random() < 0.5:
            # 触发时刻：release 之后，预计开始之前
            trigger = task.release + rng.randint(1, 10)
            
            # 乘性噪声
            sigma = config.sigma_duration
            epsilon = rng.gauss(0, sigma)
            epsilon = max(-0.3, min(0.5, epsilon))  # 截断
            
            events.append(DisturbanceEvent(
                event_type="duration",
                trigger_time=trigger,
                target_id=task.task_id,
                params={
                    "multiplier": 1.0 + epsilon,
                    "epsilon": epsilon
                }
            ))
    
    # 4. Release 扰动
    for task in tasks:
        # 40% 概率产生 release 扰动
        if rng.random() < 0.4:
            # 触发时刻：仿真开始时
            trigger = 0
            
            # 加性噪声
            sigma = config.sigma_release
            delta = int(round(rng.gauss(0, sigma)))
            delta = max(-3, min(6, delta))  # 截断
            
            events.append(DisturbanceEvent(
                event_type="release",
                trigger_time=trigger,
                target_id=task.task_id,
                params={
                    "delta": delta,
                    "new_release": max(0, task.release + delta)
                }
            ))
    
    # 按触发时间排序
    events.sort(key=lambda e: e.trigger_time)
    
    return events


def generate_scenario(
    seed: int,
    config: Config = DEFAULT_CONFIG
) -> Scenario:
    """
    生成完整场景
    
    Args:
        seed: 随机种子，确保可复现
        config: 配置对象
    
    Returns:
        Scenario: 包含 tasks, pads, disturbance_timeline
    """
    rng = random.Random(seed)
    
    # 确定任务数量
    num_tasks = rng.randint(
        config.num_tasks_range[0],
        config.num_tasks_range[1]
    )
    
    # 生成组件
    tasks = generate_tasks(seed, num_tasks, config)
    pads = generate_pads(config.num_pads)
    disturbances = generate_disturbance_timeline(seed, tasks, pads, config)
    
    # 保存配置快照
    config_snapshot = {
        "slot_minutes": config.slot_minutes,
        "rolling_interval": config.rolling_interval,
        "horizon_slots": config.horizon_slots,
        "sim_total_slots": config.sim_total_slots,
        "num_tasks": num_tasks,
        "num_pads": config.num_pads
    }
    
    return Scenario(
        seed=seed,
        tasks=tasks,
        pads=pads,
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
    """从 JSON 文件加载场景"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = [
        Task(
            task_id=t["task_id"],
            release=t["release"],
            duration=t["duration"],
            windows=[tuple(w) for w in t["windows"]],
            due=t["due"],
            priority=t["priority"],
            preferred_pad=t.get("preferred_pad")
        )
        for t in data["tasks"]
    ]
    
    pads = [
        Pad(
            pad_id=p["pad_id"],
            unavailable=[tuple(u) for u in p["unavailable"]]
        )
        for p in data["pads"]
    ]
    
    disturbances = [
        DisturbanceEvent(
            event_type=e["event_type"],
            trigger_time=e["trigger_time"],
            target_id=e["target_id"],
            params=e["params"]
        )
        for e in data["disturbance_timeline"]
    ]
    
    return Scenario(
        seed=data["seed"],
        tasks=tasks,
        pads=pads,
        disturbance_timeline=disturbances,
        config_snapshot=data.get("config_snapshot", {})
    )


# ============================================================================
# 测试入口
# ============================================================================

if __name__ == "__main__":
    print("=== Scenario Generation Test ===\n")
    
    scenario = generate_scenario(seed=42)
    
    print(f"Seed: {scenario.seed}")
    print(f"Tasks: {len(scenario.tasks)}")
    print(f"Pads: {len(scenario.pads)}")
    print(f"Disturbance events: {len(scenario.disturbance_timeline)}")
    
    print("\nTasks:")
    for t in scenario.tasks[:5]:
        print(f"  {t.task_id}: release={t.release}, dur={t.duration}, "
              f"windows={t.windows}, due={t.due}, priority={t.priority}")
    if len(scenario.tasks) > 5:
        print(f"  ... and {len(scenario.tasks) - 5} more")
    
    print("\nPads:")
    for p in scenario.pads:
        print(f"  {p.pad_id}")
    
    print("\nDisturbance events (first 10):")
    for e in scenario.disturbance_timeline[:10]:
        print(f"  t={e.trigger_time}: {e.event_type} -> {e.target_id}, {e.params}")
    
    # 统计扰动类型
    event_counts = {}
    for e in scenario.disturbance_timeline:
        event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
    print(f"\nEvent counts: {event_counts}")
