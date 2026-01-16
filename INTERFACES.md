# Interface Specification (接口契约)

> **Version 1.0 | Milestone 1**

---

## 1. 模块职责划分

| 文件 | 职责 | 依赖 |
|------|------|------|
| `config.py` | 全局常量与默认配置 | 无 |
| `scenario.py` | 场景生成（任务、pad、扰动时间线） | config |
| `disturbance.py` | 扰动应用逻辑 | config |
| `solver_cpsat.py` | CP-SAT 模型构建与求解 | ortools, config |
| `simulator.py` | Rolling 仿真主循环 | scenario, disturbance, solver, metrics |
| `metrics.py` | 稳定性与性能指标计算 | 无 |
| `policies/base.py` | 策略抽象基类 | 无 |
| `policies/fixed.py` | 固定权重策略 | base |
| `policies/greedy.py` | 贪心策略（不用 CP-SAT） | base |
| `policies/llm_meta.py` | LLM 元参数策略 | base, json schema |
| `run_experiments.py` | 批量实验与调参 | simulator, policies |
| `analyze.py` | 结果汇总与绘图 | pandas, matplotlib |

---

## 2. 核心数据结构

### 2.1 Task（任务）

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Task:
    task_id: str                              # 唯一标识 "T001"
    release: int                              # 最早可开始 slot
    duration: int                             # 占用 slot 数
    windows: List[Tuple[int, int]]            # 允许发射窗口 [(start, end), ...]
    due: int                                  # 软截止 slot
    priority: float                           # 优先级 [0.1, 1.0]
    preferred_pad: Optional[str] = None       # 偏好 pad（可选）
    
    # 运行时状态（仿真过程中更新）
    actual_release: Optional[int] = None      # 扰动后实际 release
    actual_duration: Optional[int] = None     # 扰动后实际 duration
    completed: bool = False                   # 是否已完成
    completion_time: Optional[int] = None     # 实际完成 slot
```

### 2.2 Pad（发射台）

```python
@dataclass
class Pad:
    pad_id: str                               # 唯一标识 "PAD_A"
    capacity: int = 1                         # 同时服务数（恒为1）
    unavailable: List[Tuple[int, int]] = field(default_factory=list)
                                              # 不可用区间 [(start, end), ...]
```

### 2.3 Plan（排程计划）

```python
@dataclass
class TaskAssignment:
    task_id: str
    pad_id: str
    start_slot: int
    end_slot: int                             # = start_slot + duration - 1

@dataclass
class Plan:
    timestamp: int                            # 生成时刻 (now)
    assignments: List[TaskAssignment]
    
    def get_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """按 task_id 查找分配"""
        ...
    
    def to_dict(self) -> dict:
        """序列化为 JSON-compatible dict"""
        ...
```

### 2.4 Scenario（场景）

```python
@dataclass
class DisturbanceEvent:
    event_type: str                           # "weather" | "pad_outage" | "duration" | "release"
    trigger_time: int                         # 触发 slot
    target_id: Optional[str]                  # 影响的 task_id 或 pad_id
    params: dict                              # 事件参数

@dataclass
class Scenario:
    seed: int
    tasks: List[Task]
    pads: List[Pad]
    disturbance_timeline: List[DisturbanceEvent]  # 按时间排序
    config: dict                              # 生成时的配置快照
    
    def get_task(self, task_id: str) -> Optional[Task]:
        ...
    
    def get_pad(self, pad_id: str) -> Optional[Pad]:
        ...
```

### 2.5 SolverResult（求解结果）

```python
from enum import Enum

class SolveStatus(Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"

@dataclass
class SolverResult:
    status: SolveStatus
    plan: Optional[Plan]                      # 仅 OPTIMAL/FEASIBLE 时有效
    objective_value: Optional[float]
    solve_time_ms: int
    num_variables: int
    num_constraints: int
    gap: Optional[float]                      # 最优性 gap（若有）
```

### 2.6 EpisodeResult（单次仿真结果）

```python
@dataclass
class RollingSnapshot:
    t: int                                    # 当前时刻
    plan: Plan
    solve_status: SolveStatus
    solve_time_ms: int
    plan_drift: float                         # 本次 rolling 的 drift
    num_frozen: int
    num_tasks_in_horizon: int
    infeasible_forced: bool                   # 是否触发强制重排
    meta_params: Optional[dict]               # LLM 策略时的元参数

@dataclass
class EpisodeResult:
    seed: int
    policy_name: str
    snapshots: List[RollingSnapshot]
    
    # 汇总指标
    avg_delay: float
    max_delay: int
    episode_drift: float                      # 单标量
    total_shifts: int
    total_switches: int
    infeasible_count: int
    avg_solve_time_ms: float
    total_runtime_s: float
    
    # 最终排程
    final_schedule: List[TaskAssignment]
    completed_tasks: List[str]
    uncompleted_tasks: List[str]
```

### 2.7 MetaParams（LLM 元参数）

```python
@dataclass
class MetaParams:
    """LLM 输出的元参数，需 schema 校验"""
    w_delay: float                            # delay 权重 [1, 100]
    w_shift: float                            # shift 权重 [0.1, 10]
    w_switch: float                           # switch 权重 [0.1, 20]
    freeze_horizon: Optional[int] = None      # 可选覆盖冻结视野 [0, 36]
    
    # 扩展（未来）
    urgency_threshold: Optional[float] = None # 紧急任务触发阈值
    replan_trigger: Optional[str] = None      # 重排触发条件

# JSON Schema for validation
META_PARAMS_SCHEMA = {
    "type": "object",
    "required": ["w_delay", "w_shift", "w_switch"],
    "properties": {
        "w_delay": {"type": "number", "minimum": 1, "maximum": 100},
        "w_shift": {"type": "number", "minimum": 0.1, "maximum": 10},
        "w_switch": {"type": "number", "minimum": 0.1, "maximum": 20},
        "freeze_horizon": {"type": "integer", "minimum": 0, "maximum": 36}
    },
    "additionalProperties": False
}
```

### 2.8 Config（配置）

```python
@dataclass
class Config:
    # 时间离散化
    slot_minutes: int = 10
    rolling_interval: int = 6                 # slots (60min)
    horizon_slots: int = 144                  # 24h
    sim_total_slots: int = 432                # 72h
    
    # 冻结
    freeze_horizon: int = 12                  # 2h
    
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
```

---

## 3. 函数签名

### 3.1 scenario.py

```python
def generate_scenario(
    seed: int,
    config: Config
) -> Scenario:
    """
    生成完整场景，包括任务、pad 和扰动时间线。
    
    Args:
        seed: 随机种子，确保可复现
        config: 配置对象
    
    Returns:
        Scenario: 包含 tasks, pads, disturbance_timeline
    
    Raises:
        ValueError: 配置参数非法
    """
    ...

def save_scenario(
    scenario: Scenario,
    filepath: str
) -> None:
    """序列化场景到 JSON 文件"""
    ...

def load_scenario(
    filepath: str
) -> Scenario:
    """从 JSON 文件加载场景"""
    ...
```

### 3.2 disturbance.py

```python
def apply_disturbance(
    state: "SimulationState",
    now: int,
    events: List[DisturbanceEvent]
) -> "SimulationState":
    """
    应用 [last_tick, now) 区间内的所有扰动事件。
    
    Args:
        state: 当前仿真状态（会被修改）
        now: 当前时刻 slot
        events: 扰动事件列表（已按时间排序）
    
    Returns:
        更新后的 state
    
    Side Effects:
        - 修改 task.windows（天气扰动）
        - 修改 pad.unavailable（pad outage）
        - 修改 task.actual_duration（duration 噪声在执行时应用）
    """
    ...

def generate_disturbance_timeline(
    seed: int,
    tasks: List[Task],
    pads: List[Pad],
    config: Config
) -> List[DisturbanceEvent]:
    """
    预生成整个仿真周期的扰动时间线。
    
    Args:
        seed: 随机种子
        tasks: 任务列表
        pads: pad 列表
        config: 配置
    
    Returns:
        按 trigger_time 排序的扰动事件列表
    """
    ...
```

### 3.3 solver_cpsat.py

```python
def solve_cpsat(
    now: int,
    horizon: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan],
    frozen_tasks: Dict[str, TaskAssignment],  # task_id -> frozen assignment
    weights: Tuple[float, float, float],      # (w_delay, w_shift, w_switch)
    config: Config
) -> SolverResult:
    """
    构建并求解 CP-SAT 模型。
    
    Args:
        now: 当前时刻
        horizon: 求解视野（slot 数）
        tasks: 需要排程的任务（未完成 + release <= now+horizon）
        pads: 可用 pad 列表
        prev_plan: 上一轮计划（用于计算 shift/switch）
        frozen_tasks: 冻结的任务分配（不可更改）
        weights: 目标函数权重 (w_delay, w_shift, w_switch)
        config: 配置
    
    Returns:
        SolverResult: 包含状态、计划、求解时间等
    """
    ...

def build_model(
    now: int,
    horizon: int,
    tasks: List[Task],
    pads: List[Pad],
    prev_plan: Optional[Plan],
    frozen_tasks: Dict[str, TaskAssignment],
    weights: Tuple[float, float, float],
    config: Config
) -> Tuple["cp_model.CpModel", dict]:
    """
    构建 CP-SAT 模型（不求解）。
    
    Returns:
        (model, variables_dict): 模型对象和变量字典
    """
    ...
```

### 3.4 simulator.py

```python
@dataclass
class SimulationState:
    """仿真状态（可变）"""
    now: int                                  # 当前时刻
    tasks: List[Task]                         # 任务列表（含状态）
    pads: List[Pad]                           # pad 列表（含 unavailable）
    current_plan: Optional[Plan]              # 当前有效计划
    events_applied: Set[int]                  # 已应用的事件索引

def simulate_episode(
    policy: "BasePolicy",
    scenario: Scenario,
    config: Config
) -> EpisodeResult:
    """
    运行单个 episode 的完整仿真。
    
    Args:
        policy: 策略对象
        scenario: 场景（任务、pad、扰动）
        config: 配置
    
    Returns:
        EpisodeResult: 完整仿真结果
    """
    ...

def run_single_rolling(
    state: SimulationState,
    policy: "BasePolicy",
    scenario: Scenario,
    config: Config
) -> Tuple[SimulationState, RollingSnapshot]:
    """
    执行单次 rolling。
    
    Returns:
        (updated_state, snapshot)
    """
    ...

def get_frozen_tasks(
    current_plan: Optional[Plan],
    now: int,
    freeze_horizon: int
) -> Dict[str, TaskAssignment]:
    """
    获取需要冻结的任务分配。
    
    Returns:
        Dict[task_id, TaskAssignment] 冻结任务
    """
    ...

def handle_infeasible(
    state: SimulationState,
    config: Config
) -> Tuple[SimulationState, Plan]:
    """
    处理不可行情况，返回强制重排结果。
    """
    ...
```

### 3.5 metrics.py

```python
def compute_task_time_drift(
    task_id: str,
    old_plan: Plan,
    new_plan: Plan,
    horizon: int
) -> float:
    """
    计算单任务时间偏移 D_time_k。
    
    Returns:
        D_time_k = |start_new - start_old| / horizon
        若任务不在 old_plan 中，返回 0.0
    """
    ...

def compute_task_pad_drift(
    task_id: str,
    old_plan: Plan,
    new_plan: Plan
) -> float:
    """
    计算单任务 pad 切换 D_pad_k。
    
    Returns:
        1.0 若 pad 变化，否则 0.0
        若任务不在 old_plan 中，返回 0.0
    """
    ...

def compute_npd(
    task_id: str,
    old_plan: Plan,
    new_plan: Plan,
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> float:
    """
    计算单任务归一化偏移 NPD_k。
    
    Returns:
        NPD_k = alpha * D_time_k + beta * D_pad_k
    """
    ...

def compute_plan_drift(
    old_plan: Plan,
    new_plan: Plan,
    completed_tasks: Set[str],
    horizon: int,
    alpha: float = 0.7,
    beta: float = 0.3
) -> float:
    """
    计算单次 rolling 的 PlanDrift。
    
    Args:
        old_plan: 上一轮计划
        new_plan: 本轮计划
        completed_tasks: 已完成任务集合（排除）
        horizon: 视野长度
        alpha, beta: 权重
    
    Returns:
        PlanDrift_t = mean(NPD_k) for k in common_tasks
    """
    ...

def compute_episode_drift(
    snapshots: List[RollingSnapshot]
) -> float:
    """
    计算 episode 总 drift（单标量）。
    
    Returns:
        EpisodeDrift = mean(plan_drift_t) for all t
    """
    ...

def compute_delay_metrics(
    final_schedule: List[TaskAssignment],
    tasks: List[Task]
) -> Tuple[float, int]:
    """
    计算延迟指标。
    
    Returns:
        (avg_delay, max_delay)
    """
    ...

def compute_all_metrics(
    result: EpisodeResult
) -> dict:
    """
    计算所有指标的汇总字典。
    
    Returns:
        {
            "avg_delay": float,
            "max_delay": int,
            "episode_drift": float,
            "total_shifts": int,
            "total_switches": int,
            ...
        }
    """
    ...
```

### 3.6 policies/base.py

```python
from abc import ABC, abstractmethod

class BasePolicy(ABC):
    """策略抽象基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        ...
    
    @abstractmethod
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[Optional[MetaParams], Optional[Plan]]:
        """
        策略决策。
        
        Returns:
            (meta_params, direct_plan)
            - 对于 CP-SAT 策略: 返回 (MetaParams, None)
            - 对于贪心策略: 返回 (None, Plan)
        """
        ...
    
    def reset(self) -> None:
        """重置策略状态（新 episode 开始时调用）"""
        pass
```

### 3.7 policies/fixed.py

```python
class FixedWeightPolicy(BasePolicy):
    """固定权重策略"""
    
    def __init__(
        self,
        w_delay: float = 10.0,
        w_shift: float = 1.0,
        w_switch: float = 5.0,
        freeze_horizon: Optional[int] = None
    ):
        ...
    
    @property
    def name(self) -> str:
        return "fixed"
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """始终返回固定的元参数"""
        ...
```

### 3.8 policies/greedy.py

```python
class GreedyPolicy(BasePolicy):
    """贪心策略（不使用 CP-SAT）"""
    
    @property
    def name(self) -> str:
        return "greedy"
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[None, Plan]:
        """
        贪心分配：
        1. 按 due 升序排序任务
        2. 依次为每个任务分配最早可用 slot
        """
        ...
```

### 3.9 policies/llm_meta.py

```python
class LLMMetaPolicy(BasePolicy):
    """LLM 元参数策略"""
    
    def __init__(
        self,
        llm_client: "LLMClient",  # 抽象 LLM 接口
        fallback_params: MetaParams,
        max_retries: int = 2
    ):
        ...
    
    @property
    def name(self) -> str:
        return "llm_meta"
    
    def decide(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> Tuple[MetaParams, None]:
        """
        调用 LLM 获取元参数：
        1. 构建 prompt（当前状态摘要）
        2. 调用 LLM
        3. 解析 JSON 并 schema 校验
        4. 校验失败则 fallback
        """
        ...
    
    def _build_prompt(
        self,
        state: SimulationState,
        now: int,
        config: Config
    ) -> str:
        """构建 LLM prompt"""
        ...
    
    def _parse_and_validate(
        self,
        llm_response: str
    ) -> Optional[MetaParams]:
        """解析并校验 LLM 输出"""
        ...
```

### 3.10 run_experiments.py

```python
def run_single_experiment(
    policy: BasePolicy,
    seed: int,
    config: Config,
    output_dir: str
) -> EpisodeResult:
    """运行单个实验"""
    ...

def run_batch_experiments(
    policies: List[BasePolicy],
    seeds: List[int],
    config: Config,
    output_dir: str,
    parallel: bool = False
) -> List[EpisodeResult]:
    """批量运行实验"""
    ...

def grid_search_tuning(
    train_seeds: List[int],
    param_grid: dict,
    config: Config,
    output_dir: str
) -> Tuple[dict, pd.DataFrame]:
    """
    网格搜索调参。
    
    Returns:
        (best_params, results_df)
    """
    ...

def save_results(
    results: List[EpisodeResult],
    output_dir: str
) -> None:
    """保存结果到 CSV"""
    ...
```

### 3.11 analyze.py

```python
def load_results(
    results_dir: str
) -> pd.DataFrame:
    """加载结果 CSV"""
    ...

def compute_comparison_table(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    生成对比表。
    
    Returns:
        DataFrame with columns: policy, avg_delay_mean, avg_delay_std,
        episode_drift_mean, episode_drift_std, ...
    """
    ...

def plot_delay_drift_scatter(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """绘制 Delay-Drift 散点图"""
    ...

def plot_drift_boxplot(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """绘制 Drift 箱线图"""
    ...

def plot_solve_time_trend(
    results: List[EpisodeResult],
    output_path: str
) -> None:
    """绘制求解时间趋势图"""
    ...

def generate_report(
    results_dir: str,
    output_dir: str
) -> None:
    """生成完整分析报告"""
    ...
```

---

## 4. 文件路径约定

```
项目根目录: LLMpaicheng1/

配置文件:
  config.py                    # 可 import

数据目录:
  data/scenarios/              # 预生成场景
  data/scenarios/scenario_{seed}.json

日志目录:
  logs/episode_{seed}_{policy}/
  logs/episode_{seed}_{policy}/rolling_log.jsonl
  logs/episode_{seed}_{policy}/plan_snapshots.jsonl
  logs/episode_{seed}_{policy}/metrics_per_roll.csv
  logs/episode_{seed}_{policy}/final_schedule.json

结果目录:
  results/
  results/summary.csv
  results/comparison.csv
  results/tuning_results.csv

图表目录:
  figures/
  figures/delay_drift_scatter.png
  figures/drift_distribution.png
  figures/solve_time_trend.png
  figures/switch_histogram.png
```

---

## 5. 错误处理约定

### 5.1 LLM Schema 校验失败

```python
# 失败时 fallback 到默认参数
try:
    params = parse_and_validate(llm_response)
except (json.JSONDecodeError, ValidationError) as e:
    logger.warning(f"LLM output invalid: {e}, using fallback")
    params = fallback_params
```

### 5.2 CP-SAT 超时/不可行

```python
if result.status == SolveStatus.TIMEOUT:
    # 使用上一轮计划 + 警告
    logger.warning("Solver timeout, keeping previous plan")
    plan = prev_plan

elif result.status == SolveStatus.INFEASIBLE:
    # 触发强制重排
    logger.warning("Infeasible, triggering forced reschedule")
    state, plan = handle_infeasible(state, config)
```

### 5.3 日志记录

```python
# 所有模块使用统一 logger
import logging
logger = logging.getLogger("launch_scheduling")
logger.setLevel(logging.INFO)
```

---

## 6. 版本与扩展点

### 6.1 当前版本范围

- ✅ 单 pad 容量 (capacity=1)
- ✅ 无 crew 约束
- ✅ 无任务间 separation 约束
- ✅ 固定 slot 时长

### 6.2 预留扩展接口

```python
# Task 扩展字段（当前不使用）
class Task:
    crew_required: Optional[List[str]] = None
    separation_after: Optional[int] = None
    
# Config 扩展参数（当前使用默认）
class Config:
    enable_crew_constraint: bool = False
    enable_separation_constraint: bool = False
```

---

## 7. 依赖版本

```
Python >= 3.9
ortools >= 9.6
numpy >= 1.21
pandas >= 1.3
matplotlib >= 3.5
jsonschema >= 4.0  # for LLM output validation
```
