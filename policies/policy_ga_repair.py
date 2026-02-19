"""
GARepairPolicy — 基于 GA 搜索子集 + CP-SAT 局部修复的 Matheuristic 调度策略

核心架构
--------
- GA **不负责** 排程生成。GA 只在离散空间内搜索最优的 unlock_mission_ids 子集。
- 排程质量与可行性完全由现有的 solver_cpsat.solve_v2_1 负责（Anchor Fix-and-Optimize）。
- GA 每评估一个个体，调用一次 Anchor Fix-and-Optimize 接口。
- 严格遵守解锁集合大小上限 K。

定位
----
作为 TRCGRepairPolicy（LLM 驱动）和 RealLLMPolicy（元参数驱动）的核心对比 Baseline。
"""

import logging
import os
import random
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from policies.base import BasePolicy, MetaParams
from config import Config

from features import build_trcg_summary, TRCGSummary
from solver_cpsat import (
    Mission,
    Resource,
    PlanV2_1,
    SolverConfigV2_1,
    SolverResult,
    SolveStatus,
    compute_frozen_ops,
    solve_v2_1,
    OpAssignment,
)

from policies.policy_llm_repair import (
    heuristic_repair_decision,
    FREEZE_HOURS_TO_SLOTS,
)

logger = logging.getLogger("GARepairPolicy")

# ============================================================================
# GA 默认超参（V2 加速版）
# ============================================================================
GA_DEFAULT_POP_SIZE: int = 12           # B: 小种群 → 同预算可跑更多代
GA_DEFAULT_GENERATIONS: int = 5         # B: 从 10 降至 5
GA_DEFAULT_K: int = 5
GA_DEFAULT_MUTATION_RATE: float = 0.2
GA_DEFAULT_CANDIDATE_POOL_SIZE: int = 15
GA_DEFAULT_TOURNAMENT_SIZE: int = 3
GA_DEFAULT_ELITISM_COUNT: int = 2
GA_DEFAULT_PARALLEL_WORKERS: int = 4    # 保留兼容
GA_BIG_M: float = 1e9

# V2 加速默认参数
GA_DEFAULT_N_JOBS: int = 8             # A: 并行评估 job 数
GA_DEFAULT_EVAL_BUDGET: int = 48       # B: 每 replan点最大评估次数(≥3代进化)
GA_DEFAULT_EARLY_STOP_PATIENCE: int = 2  # C: 早停 patience
GA_DEFAULT_EVAL_TIMEOUT_S: float = 0.5   # E: 评估阶段 solver timeout
GA_DEFAULT_FINAL_TIMEOUT_S: float = 2.0  # E: 最终复算 solver timeout
GA_DEFAULT_EVAL_CP_WORKERS: int = 1      # A: 评估阶段 CP-SAT workers
GA_DEFAULT_FINAL_CP_WORKERS: int = 8     # A: 复算阶段 CP-SAT workers
GA_DEFAULT_ENABLE_CACHE: bool = True     # D: 启用缓存


# ============================================================================
# GA 统计信息（V2 扩展版）
# ============================================================================
@dataclass
class GAStats:
    """GA 搜索统计信息（含 V2 加速诊断字段）"""
    generations_run: int = 0
    total_evaluations: int = 0
    cache_hits: int = 0
    best_fitness: float = GA_BIG_M
    best_unlock_set: Optional[Tuple[str, ...]] = None
    pop_size: int = 0
    candidate_pool_size: int = 0
    wall_time_ms: int = 0
    feasible_count: int = 0
    infeasible_count: int = 0

    # V2 扩展字段
    ga_wall_ms: int = 0                    # GA 搜索阶段总耗时
    final_recompute_ms: int = 0            # 最终复算耗时
    eval_solve_times_ms: List[int] = field(default_factory=list)  # 每次 eval 的 solve time
    eval_budget: int = 0                   # 评估预算
    n_jobs: int = 1                        # 并行 job 数
    cp_sat_workers_eval: int = 1           # eval 阶段 CP-SAT workers
    cp_sat_workers_final: int = 8          # final 阶段 CP-SAT workers
    eval_timeout_s: float = 0.5            # eval 阶段 timeout
    final_timeout_s: float = 2.0           # final 阶段 timeout
    stop_reason: str = "budget_exhausted"  # patience|best_reached|budget_exhausted
    cache_hit_rate: float = 0.0            # 缓存命中率
    final_recompute_fitness: float = GA_BIG_M  # 复算后的 fitness

    def to_dict(self) -> Dict[str, Any]:
        eval_times = self.eval_solve_times_ms
        eval_avg = sum(eval_times) / len(eval_times) if eval_times else 0
        eval_p95 = sorted(eval_times)[int(len(eval_times) * 0.95)] if eval_times else 0

        total_attempts = self.total_evaluations + self.cache_hits
        hit_rate = self.cache_hits / total_attempts if total_attempts > 0 else 0.0

        return {
            "generations_run": self.generations_run,
            "total_evaluations": self.total_evaluations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(hit_rate, 3),
            "best_fitness": self.best_fitness if self.best_fitness < GA_BIG_M else "BIG_M",
            "best_unlock_set": list(self.best_unlock_set) if self.best_unlock_set else [],
            "pop_size": self.pop_size,
            "candidate_pool_size": self.candidate_pool_size,
            "wall_time_ms": self.wall_time_ms,
            "ga_wall_ms": self.ga_wall_ms,
            "final_recompute_ms": self.final_recompute_ms,
            "final_recompute_fitness": (
                self.final_recompute_fitness
                if self.final_recompute_fitness < GA_BIG_M
                else "BIG_M"
            ),
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "eval_solve_time_ms_avg": round(eval_avg, 1),
            "eval_solve_time_ms_p95": eval_p95,
            "eval_budget": self.eval_budget,
            "n_jobs": self.n_jobs,
            "cp_sat_workers_eval": self.cp_sat_workers_eval,
            "cp_sat_workers_final": self.cp_sat_workers_final,
            "eval_timeout_s": self.eval_timeout_s,
            "final_timeout_s": self.final_timeout_s,
            "stop_reason": self.stop_reason,
        }


# ============================================================================
# Fitness 评估函数
# ============================================================================

def _evaluate_individual_internal(
    unlock_set_frozen: FrozenSet[str],
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    solver_config_dict: dict,
    now: int,
) -> Tuple[FrozenSet[str], float, int]:
    """
    评估单个个体的 fitness。返回 (unlock_set, fitness, solve_time_ms)。

    Fitness = objective_value（Minimization）。
    若求解不可行/超时 → BIG_M。
    """
    unlock_set = set(unlock_set_frozen)
    solve_time_ms = 0

    try:
        solver_config = SolverConfigV2_1(**solver_config_dict)
        result = solve_v2_1(
            missions=missions,
            resources=resources,
            horizon=horizon,
            prev_plan=prev_plan,
            frozen_ops=frozen_ops,
            config=solver_config,
            unlock_mission_ids=unlock_set,
            now=now,
        )
        solve_time_ms = result.solve_time_ms

        if result.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
            fitness = result.objective_value if result.objective_value is not None else GA_BIG_M
        else:
            fitness = GA_BIG_M
    except Exception as exc:
        logger.warning("Fitness eval exception for %s: %s", unlock_set_frozen, exc)
        fitness = GA_BIG_M

    return unlock_set_frozen, fitness, solve_time_ms


# multiprocessing 兼容的顶层包装器
def _evaluate_individual(args: Tuple) -> Tuple[FrozenSet[str], float, int]:
    """multiprocessing.Pool 兼容的包装器。"""
    return _evaluate_individual_internal(*args)


# ============================================================================
# 候选池构造
# ============================================================================

def _build_candidate_pool(
    trcg_dict: Dict[str, Any],
    active_mission_ids: Set[str],
    started_mission_ids: Set[str],
    completed_mission_ids: Set[str],
    max_pool_size: int = GA_DEFAULT_CANDIDATE_POOL_SIZE,
) -> List[str]:
    """
    基于 TRCG 诊断信息构建候选任务池。

    候选来源（按优先级）：
    1. Root/Secondary：从 heuristic_repair_decision 的输出
    2. 冲突簇 Top-N：conflict_clusters 中的成员
    3. 瓶颈资源相关任务：top_conflicts 中涉及的 mission
    4. urgent_missions 中的 mission

    Returns
    -------
    List[str] — 候选 mission_id 列表（去重，大小 ≤ max_pool_size）
    """
    eligible = active_mission_ids - started_mission_ids - completed_mission_ids
    if not eligible:
        eligible = active_mission_ids - completed_mission_ids

    pool_set: Set[str] = set()

    # 1. 从冲突中提取高度数 mission
    conflicts = trcg_dict.get("top_conflicts", [])
    degree: Dict[str, float] = {}
    for c in conflicts:
        a, b = c.get("a", ""), c.get("b", "")
        sev = float(c.get("severity", 0))
        for m in (a, b):
            if m in eligible:
                degree[m] = degree.get(m, 0.0) + sev

    # 按度数降序排列
    sorted_by_degree = sorted(degree.keys(), key=lambda m: (-degree[m], m))
    pool_set.update(sorted_by_degree)

    # 2. 从冲突簇中提取
    clusters = trcg_dict.get("conflict_clusters", [])
    for cluster in clusters:
        members = cluster.get("members", [])
        for m in members:
            if m in eligible:
                pool_set.add(m)

    # 3. urgent missions
    urgents = trcg_dict.get("urgent_missions", [])
    for u in urgents:
        mid = u.get("mission_id", "")
        if mid in eligible:
            pool_set.add(mid)

    # 4. 若池太小，从 eligible 中补充
    if len(pool_set) < max_pool_size:
        remaining = sorted(eligible - pool_set)
        pool_set.update(remaining[: max_pool_size - len(pool_set)])

    # 截断并排序
    pool_list = sorted(pool_set)[:max_pool_size]
    return pool_list


# ============================================================================
# GA 遗传算子
# ============================================================================

def _init_population(
    candidate_pool: List[str],
    K: int,
    pop_size: int,
    elite_seeds: Optional[List[List[str]]] = None,
    rng: Optional[random.Random] = None,
) -> List[FrozenSet[str]]:
    """
    初始化种群。

    - 多个精英种子（若提供）依次注入。
    - 其余个体从候选池中随机采样 K 个任务。
    """
    if rng is None:
        rng = random.Random()

    population: List[FrozenSet[str]] = []
    seen: Set[FrozenSet[str]] = set()

    # 注入精英种子（支持多个）
    if elite_seeds:
        for seed in elite_seeds:
            if len(population) >= pop_size:
                break
            seed_set = frozenset(seed[:K])
            if seed_set and seed_set not in seen:
                population.append(seed_set)
                seen.add(seed_set)

    # 生成剩余个体
    actual_k = min(K, len(candidate_pool))
    max_attempts = pop_size * 10  # 避免无限循环
    attempts = 0
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        individual = frozenset(rng.sample(candidate_pool, actual_k))
        if individual not in seen:
            population.append(individual)
            seen.add(individual)

    # 如果仍然不够（候选池太小），允许重复
    while len(population) < pop_size:
        individual = frozenset(rng.sample(candidate_pool, actual_k))
        population.append(individual)

    return population


def _crossover(
    parent1: FrozenSet[str],
    parent2: FrozenSet[str],
    K: int,
    rng: random.Random,
) -> FrozenSet[str]:
    """
    交叉算子：取两个父代的并集，从中随机采样 K 个任务。
    """
    union = list(parent1 | parent2)
    actual_k = min(K, len(union))
    return frozenset(rng.sample(union, actual_k))


def _mutate(
    individual: FrozenSet[str],
    candidate_pool: List[str],
    K: int,
    mutation_rate: float,
    rng: random.Random,
) -> FrozenSet[str]:
    """
    变异算子：随机 替换/新增/移除 候选池中的任务，
    始终保持子集大小 ≤ K。
    """
    if rng.random() > mutation_rate:
        return individual

    ind_list = list(individual)
    pool_remaining = [m for m in candidate_pool if m not in individual]

    # 选择变异类型
    op = rng.choice(["replace", "add", "remove"])

    if op == "replace" and ind_list and pool_remaining:
        # 替换：移除一个，加入一个新的
        remove_idx = rng.randint(0, len(ind_list) - 1)
        ind_list.pop(remove_idx)
        ind_list.append(rng.choice(pool_remaining))
    elif op == "add" and pool_remaining and len(ind_list) < K:
        # 新增一个
        ind_list.append(rng.choice(pool_remaining))
    elif op == "remove" and len(ind_list) > 1:
        # 移除一个（保证至少 1 个）
        remove_idx = rng.randint(0, len(ind_list) - 1)
        ind_list.pop(remove_idx)

    # 确保大小 ≤ K
    if len(ind_list) > K:
        ind_list = rng.sample(ind_list, K)

    return frozenset(ind_list)


def _tournament_select(
    population: List[FrozenSet[str]],
    fitness_map: Dict[FrozenSet[str], float],
    tournament_size: int,
    rng: random.Random,
) -> FrozenSet[str]:
    """
    锦标赛选择：从种群中随机选取 tournament_size 个个体，返回 fitness 最小者。
    """
    actual_size = min(tournament_size, len(population))
    contenders = rng.sample(population, actual_size)
    return min(contenders, key=lambda ind: fitness_map.get(ind, GA_BIG_M))


# ============================================================================
# GA 主循环（V2 加速版：并行 + 早停 + 缓存 + 预算 + 两段式 timeout）
# ============================================================================

def run_ga_search(
    candidate_pool: List[str],
    elite_seeds: Optional[List[List[str]]],
    K: int,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    # Solver 参数
    missions: List[Mission],
    resources: List[Resource],
    horizon: int,
    prev_plan: Optional[PlanV2_1],
    frozen_ops: Dict[str, OpAssignment],
    solver_config_dict: dict,
    now: int,
    # V2 加速参数
    n_jobs: int = GA_DEFAULT_N_JOBS,
    eval_budget: int = GA_DEFAULT_EVAL_BUDGET,
    early_stop_patience: int = GA_DEFAULT_EARLY_STOP_PATIENCE,
    eval_timeout_s: float = GA_DEFAULT_EVAL_TIMEOUT_S,
    eval_cp_workers: int = GA_DEFAULT_EVAL_CP_WORKERS,
    enable_cache: bool = GA_DEFAULT_ENABLE_CACHE,
    # 兼容旧接口
    parallel_eval: bool = True,
    num_workers: int = GA_DEFAULT_PARALLEL_WORKERS,
    # 随机种子
    ga_seed: Optional[int] = None,
) -> GAStats:
    """
    执行 GA 搜索（V2 加速版）。

    加速特性：
    - A. 并行评估（ThreadPoolExecutor, eval 阶段 CP-SAT workers=1）
    - B. 评估预算硬约束（eval_budget 超过即截断）
    - C. 早停（patience 代无改善 / best=0 立即停止）
    - D. Fitness 缓存（frozenset 去重）
    - E. 两段式 timeout（eval 阶段短 timeout）

    注：最终复算在此函数外部由 GARepairPolicy.decide() 执行。
    """
    rng = random.Random(ga_seed)
    t0 = _time.time()

    # 合并兼容：旧 parallel_eval/num_workers → 新 n_jobs
    effective_n_jobs = n_jobs if n_jobs > 1 else (num_workers if parallel_eval and num_workers > 1 else 1)

    stats = GAStats(
        pop_size=pop_size,
        candidate_pool_size=len(candidate_pool),
        eval_budget=eval_budget,
        n_jobs=effective_n_jobs,
        cp_sat_workers_eval=eval_cp_workers,
        eval_timeout_s=eval_timeout_s,
    )

    if not candidate_pool:
        logger.warning("GA: empty candidate pool, returning BIG_M")
        stats.stop_reason = "empty_pool"
        stats.wall_time_ms = int((_time.time() - t0) * 1000)
        stats.ga_wall_ms = stats.wall_time_ms
        return stats

    # ── 构建 eval 阶段 solver config（短 timeout + 单线程 CP-SAT）──
    eval_solver_config = dict(solver_config_dict)
    eval_solver_config["time_limit_seconds"] = eval_timeout_s
    eval_solver_config["num_workers"] = eval_cp_workers

    # ── 初始化种群 ──
    population = _init_population(candidate_pool, K, pop_size, elite_seeds, rng)

    # ── Fitness 缓存 ──
    fitness_cache: Dict[FrozenSet[str], float] = {}
    budget_remaining = eval_budget

    def _batch_evaluate(individuals: List[FrozenSet[str]]) -> Dict[FrozenSet[str], float]:
        """批量评估个体，带缓存、预算截断、并行支持。"""
        nonlocal budget_remaining

        to_eval: List[FrozenSet[str]] = []
        results: Dict[FrozenSet[str], float] = {}

        for ind in individuals:
            if enable_cache and ind in fitness_cache:
                results[ind] = fitness_cache[ind]
                stats.cache_hits += 1
            else:
                if budget_remaining > 0:
                    to_eval.append(ind)
                else:
                    # 预算耗尽，未缓存的个体赋 BIG_M（不消耗预算）
                    results[ind] = GA_BIG_M

        if not to_eval:
            return results

        # 截断到剩余预算
        to_eval = to_eval[:budget_remaining]

        def _run_one(ind: FrozenSet[str]) -> Tuple[FrozenSet[str], float, int]:
            return _evaluate_individual_internal(
                ind, missions, resources, horizon,
                prev_plan, frozen_ops, eval_solver_config, now,
            )

        raw_results: List[Tuple[FrozenSet[str], float, int]] = []

        if effective_n_jobs > 1 and len(to_eval) > 1:
            # 使用 ThreadPoolExecutor（CP-SAT 在 C++ 层释放 GIL）
            n = min(effective_n_jobs, len(to_eval))
            try:
                with ThreadPoolExecutor(max_workers=n) as executor:
                    futures = {executor.submit(_run_one, ind): ind for ind in to_eval}
                    for future in as_completed(futures):
                        try:
                            raw_results.append(future.result())
                        except Exception as exc:
                            ind_key = futures[future]
                            logger.warning("Parallel eval error for %s: %s", ind_key, exc)
                            raw_results.append((ind_key, GA_BIG_M, 0))
            except Exception as exc:
                logger.warning("ThreadPool failed (%s), falling back to sequential", exc)
                raw_results = [_run_one(ind) for ind in to_eval]
        else:
            raw_results = [_run_one(ind) for ind in to_eval]

        for ind_key, fit_val, solve_ms in raw_results:
            fitness_cache[ind_key] = fit_val
            results[ind_key] = fit_val
            stats.total_evaluations += 1
            stats.eval_solve_times_ms.append(solve_ms)
            budget_remaining -= 1
            if fit_val < GA_BIG_M:
                stats.feasible_count += 1
            else:
                stats.infeasible_count += 1

        return results

    # ── GA 主循环 ──
    tournament_size = GA_DEFAULT_TOURNAMENT_SIZE
    elitism_count = min(GA_DEFAULT_ELITISM_COUNT, pop_size)
    no_improve_count = 0
    prev_best = GA_BIG_M

    for gen in range(generations):
        stats.generations_run = gen + 1

        # 评估当前种群
        fitness_map = _batch_evaluate(population)

        # 排序（Minimization）
        sorted_pop = sorted(population, key=lambda ind: fitness_map.get(ind, GA_BIG_M))

        # 更新全局最优
        best_ind = sorted_pop[0]
        best_fit = fitness_map.get(best_ind, GA_BIG_M)
        if best_fit < stats.best_fitness:
            stats.best_fitness = best_fit
            stats.best_unlock_set = tuple(sorted(best_ind))

        logger.debug(
            "GA gen=%d best_fit=%.2f pop_unique=%d evals=%d budget_left=%d",
            gen, best_fit, len(set(population)), stats.total_evaluations, budget_remaining,
        )

        # ── C. 早停检查 ──
        # C1: best_fit = 0.0（理论最优）→ 立即停止
        if best_fit <= 1e-9:
            stats.stop_reason = "best_reached"
            break

        # C2: patience 代无改善
        if best_fit < prev_best - 1e-9:
            no_improve_count = 0
            prev_best = best_fit
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            stats.stop_reason = "patience"
            break

        # C3: 预算耗尽
        if budget_remaining <= 0:
            stats.stop_reason = "budget_exhausted"
            break

        # ── 产生下一代 ──
        next_gen: List[FrozenSet[str]] = []

        # 精英保留
        for i in range(min(elitism_count, len(sorted_pop))):
            next_gen.append(sorted_pop[i])

        # 交叉 + 变异
        while len(next_gen) < pop_size:
            p1 = _tournament_select(sorted_pop, fitness_map, tournament_size, rng)
            p2 = _tournament_select(sorted_pop, fitness_map, tournament_size, rng)
            child = _crossover(p1, p2, K, rng)
            child = _mutate(child, candidate_pool, K, mutation_rate, rng)
            next_gen.append(child)

        population = next_gen[:pop_size]
    else:
        # 循环正常结束（没有 break），最终评估
        fitness_map = _batch_evaluate(population)
        sorted_pop = sorted(population, key=lambda ind: fitness_map.get(ind, GA_BIG_M))
        best_ind = sorted_pop[0]
        best_fit = fitness_map.get(best_ind, GA_BIG_M)
        if best_fit < stats.best_fitness:
            stats.best_fitness = best_fit
            stats.best_unlock_set = tuple(sorted(best_ind))
        stats.stop_reason = "budget_exhausted"

    stats.ga_wall_ms = int((_time.time() - t0) * 1000)
    stats.wall_time_ms = stats.ga_wall_ms
    return stats


# ============================================================================
# GARepairPolicy 策略类（V2 加速版）
# ============================================================================

class GARepairPolicy(BasePolicy):
    """
    GA 搜索子集 + CP-SAT 局部修复的 Matheuristic 策略（V2 加速版）。

    加速特性：
    - A. 并行评估（n_jobs, eval 阶段 CP-SAT workers=1）
    - B. 进化预算硬约束（eval_budget, pop_size=16, gens=5）
    - C. 早停（patience / best_reached / budget_exhausted）
    - D. Fitness 缓存（per-replan frozenset 去重）
    - E. 两段式 solver timeout（eval 阶段短 timeout + 最终复算正常预算）
    - F. 细粒度计时与诊断日志
    """

    def __init__(
        self,
        policy_name: str = "ga_repair",
        # GA 超参
        pop_size: int = GA_DEFAULT_POP_SIZE,
        generations: int = GA_DEFAULT_GENERATIONS,
        K: int = GA_DEFAULT_K,
        mutation_rate: float = GA_DEFAULT_MUTATION_RATE,
        candidate_pool_size: int = GA_DEFAULT_CANDIDATE_POOL_SIZE,
        # V2 加速参数
        n_jobs: int = GA_DEFAULT_N_JOBS,
        eval_budget: int = GA_DEFAULT_EVAL_BUDGET,
        early_stop_patience: int = GA_DEFAULT_EARLY_STOP_PATIENCE,
        eval_timeout_s: float = GA_DEFAULT_EVAL_TIMEOUT_S,
        final_timeout_s: Optional[float] = None,  # None = 沿用 solver_timeout_s
        eval_cp_workers: int = GA_DEFAULT_EVAL_CP_WORKERS,
        final_cp_workers: Optional[int] = None,   # None = 沿用 config.solver_num_workers
        enable_cache: bool = GA_DEFAULT_ENABLE_CACHE,
        # 兼容旧接口
        parallel_eval: bool = True,
        num_workers: int = GA_DEFAULT_PARALLEL_WORKERS,
        # Solver 默认权重
        w_delay: float = 10.0,
        w_shift: float = 1.0,
        w_switch: float = 5.0,
        # 日志
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        episode_id: str = "",
    ):
        self._policy_name = policy_name
        self._pop_size = pop_size
        self._generations = generations
        self._K = K
        self._mutation_rate = mutation_rate
        self._candidate_pool_size = candidate_pool_size

        # V2 加速参数
        self._n_jobs = n_jobs
        self._eval_budget = eval_budget
        self._early_stop_patience = early_stop_patience
        self._eval_timeout_s = eval_timeout_s
        self._final_timeout_s = final_timeout_s
        self._eval_cp_workers = eval_cp_workers
        self._final_cp_workers = final_cp_workers
        self._enable_cache = enable_cache

        # 兼容旧接口
        self._parallel_eval = parallel_eval
        self._num_workers = num_workers

        self._w_delay = w_delay
        self._w_shift = w_shift
        self._w_switch = w_switch

        self._log_dir = log_dir
        self._enable_logging = enable_logging
        self._episode_id = episode_id

        # 状态（per-episode）
        self._prev_window_slots: Optional[Dict[str, Set[int]]] = None
        self._call_count = 0
        self._ga_stats_history: List[GAStats] = []

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._policy_name

    def reset(self) -> None:
        self._prev_window_slots = None
        self._call_count = 0
        self._ga_stats_history = []

    def set_episode_id(self, episode_id: str) -> None:
        self._episode_id = episode_id

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    def decide(
        self,
        state: Any,
        now: int,
        config: Config,
    ) -> Tuple[MetaParams, None]:
        """
        策略决策入口。

        Returns
        -------
        (MetaParams, None) — MetaParams 携带 unlock_mission_ids, decision_source="ga" 等。
        """
        t0 = _time.time()
        self._call_count += 1

        # ============================================================
        # Step 1: 计算 frozen_ops
        # ============================================================
        freeze_slots = config.freeze_horizon
        frozen_ops = compute_frozen_ops(
            state.current_plan, now, freeze_slots,
            getattr(state, "started_ops", set()),
            getattr(state, "completed_ops", set()),
        )

        # ============================================================
        # Step 2: 构造 TRCGSummary
        # ============================================================
        trcg = build_trcg_summary(
            missions=state.missions,
            resources=state.resources,
            plan=state.current_plan,
            now=now,
            config=config,
            started_ops=getattr(state, "started_ops", set()),
            completed_ops=getattr(state, "completed_ops", set()),
            actual_durations=getattr(state, "actual_durations", {}),
            frozen_ops=frozen_ops,
            prev_window_slots=self._prev_window_slots,
        )
        trcg_dict = trcg.to_dict()

        # ============================================================
        # Step 3: 确定 active / started / completed mission 集合
        # ============================================================
        started_ops: Set[str] = getattr(state, "started_ops", set())
        completed_ops: Set[str] = getattr(state, "completed_ops", set())

        started_mission_ids: Set[str] = set()
        completed_mission_ids: Set[str] = set()
        all_mission_ids: Set[str] = set()
        horizon_end = now + config.horizon_slots

        for m in state.missions:
            all_mission_ids.add(m.mission_id)
            launch = m.get_launch_op()
            if launch and launch.op_id in completed_ops:
                completed_mission_ids.add(m.mission_id)
                continue
            for op in m.operations:
                if op.op_id in started_ops:
                    started_mission_ids.add(m.mission_id)
                    break

        schedulable_ids = {
            m.mission_id for m in state.get_schedulable_missions(horizon_end)
        }
        active_mission_ids = schedulable_ids - started_mission_ids - completed_mission_ids
        if not active_mission_ids:
            active_mission_ids = schedulable_ids - completed_mission_ids

        # ============================================================
        # Step 4: 构建候选池 & 精英种子
        # ============================================================
        candidate_pool = _build_candidate_pool(
            trcg_dict=trcg_dict,
            active_mission_ids=active_mission_ids,
            started_mission_ids=started_mission_ids,
            completed_mission_ids=completed_mission_ids,
            max_pool_size=self._candidate_pool_size,
        )

        # 精英种子：由 TRCG-Heuristic 生成 + root/secondary 相关集合
        elite_seeds: List[List[str]] = []
        heuristic_decision = None
        fallback_reason: Optional[str] = None

        if active_mission_ids:
            heuristic_decision = heuristic_repair_decision(
                trcg_dict=trcg_dict,
                active_mission_ids=active_mission_ids,
                started_mission_ids=started_mission_ids,
                completed_mission_ids=completed_mission_ids,
                fallback_reason="ga_seed",
            )
            # 种子 1: 启发式推荐的 unlock_set
            elite_seeds.append(heuristic_decision.unlock_mission_ids[:self._K])

            # 种子 2: root_cause 单独作为解锁集
            if heuristic_decision.root_cause_mission_id:
                root_seed = [heuristic_decision.root_cause_mission_id]
                if heuristic_decision.secondary_root_cause_mission_id:
                    root_seed.append(heuristic_decision.secondary_root_cause_mission_id)
                if root_seed != elite_seeds[0]:
                    elite_seeds.append(root_seed)

        # ============================================================
        # Step 5: 运行 GA 搜索
        # ============================================================
        decision_source = "ga"
        ga_stats = GAStats()

        if candidate_pool and active_mission_ids:
            # 构建 solver config dict（序列化友好）
            solver_horizon = config.sim_total_slots
            solver_config_dict = {
                "horizon_slots": solver_horizon,
                "w_delay": self._w_delay,
                "w_shift": self._w_shift,
                "w_switch": self._w_switch,
                "time_limit_seconds": config.solver_timeout_s,
                "num_workers": config.solver_num_workers,
                "op5_max_wait_slots": max(
                    0,
                    int(round(config.op5_max_wait_hours * 60 / config.slot_minutes)),
                ),
                "use_two_stage": config.use_two_stage_solver,
                "epsilon_solver": config.default_epsilon_solver,
                "kappa_win": config.default_kappa_win,
                "kappa_seq": config.default_kappa_seq,
                "stage1_time_ratio": config.stage1_time_ratio,
            }

            # 获取当前可调度 missions & resources
            missions_to_schedule = state.get_schedulable_missions(horizon_end)
            # 包含已 started 但未 completed 的 mission
            if state.current_plan:
                for assign in state.current_plan.op_assignments:
                    if assign.op_id in started_ops and assign.op_id not in completed_ops:
                        mission = state.get_mission(assign.mission_id)
                        if mission and mission not in missions_to_schedule:
                            missions_to_schedule.append(mission)

            ga_stats = run_ga_search(
                candidate_pool=candidate_pool,
                elite_seeds=elite_seeds,
                K=self._K,
                pop_size=self._pop_size,
                generations=self._generations,
                mutation_rate=self._mutation_rate,
                missions=missions_to_schedule,
                resources=state.resources,
                horizon=solver_horizon,
                prev_plan=state.current_plan,
                frozen_ops=frozen_ops,
                solver_config_dict=solver_config_dict,
                now=now,
                # V2 加速参数
                n_jobs=self._n_jobs,
                eval_budget=self._eval_budget,
                early_stop_patience=self._early_stop_patience,
                eval_timeout_s=self._eval_timeout_s,
                eval_cp_workers=self._eval_cp_workers,
                enable_cache=self._enable_cache,
                # 兼容旧接口
                parallel_eval=self._parallel_eval,
                num_workers=self._num_workers,
                ga_seed=self._call_count,  # 每步不同随机种子
            )

            # ============================================================
            # Step 5b: 最终复算（E. 两段式 timeout）
            # ============================================================
            # 对 GA 选出的 best unlock_set 用正常预算做一次最终求解，
            # 保证计划质量与直接用 simulator 解出的一致。
            if ga_stats.best_fitness < GA_BIG_M and ga_stats.best_unlock_set:
                final_timeout = self._final_timeout_s or config.solver_timeout_s
                final_workers = self._final_cp_workers or config.solver_num_workers

                final_config_dict = dict(solver_config_dict)
                final_config_dict["time_limit_seconds"] = final_timeout
                final_config_dict["num_workers"] = final_workers

                ga_stats.cp_sat_workers_final = final_workers
                ga_stats.final_timeout_s = final_timeout

                t_recomp = _time.time()
                try:
                    _, recomp_fitness, recomp_ms = _evaluate_individual_internal(
                        frozenset(ga_stats.best_unlock_set),
                        missions_to_schedule,
                        state.resources,
                        solver_horizon,
                        state.current_plan,
                        frozen_ops,
                        final_config_dict,
                        now,
                    )
                    ga_stats.final_recompute_fitness = recomp_fitness
                    ga_stats.final_recompute_ms = int((_time.time() - t_recomp) * 1000)

                    # 如果复算也是 BIG_M（正常预算下也不可行），标记回退
                    if recomp_fitness >= GA_BIG_M:
                        logger.warning(
                            "GA best unlock_set infeasible on final recompute, "
                            "falling back to heuristic"
                        )
                        ga_stats.best_fitness = GA_BIG_M
                except Exception as exc:
                    logger.warning("Final recompute failed: %s", exc)
                    ga_stats.final_recompute_ms = int((_time.time() - t_recomp) * 1000)
        else:
            fallback_reason = "no_active_missions_or_empty_pool"

        # ============================================================
        # Step 6: 提取结果 / 回退
        # ============================================================
        unlock_ids: Optional[List[str]] = None
        root_cause_id: Optional[str] = None
        secondary_root_id: Optional[str] = None

        if ga_stats.best_fitness < GA_BIG_M and ga_stats.best_unlock_set:
            # GA 找到可行解
            unlock_ids = list(ga_stats.best_unlock_set)
            decision_source = "ga"

            # root_cause: 取 GA 最优集合中度数最高的 mission（基于 TRCG 冲突）
            conflicts = trcg_dict.get("top_conflicts", [])
            degree: Dict[str, float] = {}
            for c in conflicts:
                a, b = c.get("a", ""), c.get("b", "")
                sev = float(c.get("severity", 0))
                for mid in (a, b):
                    if mid in unlock_ids:
                        degree[mid] = degree.get(mid, 0.0) + sev

            if degree:
                root_cause_id = max(degree, key=lambda m: (degree[m], m))
            else:
                root_cause_id = unlock_ids[0]

            # secondary_root: unlock 中除 root 外度数最高者
            remaining_degree = {m: d for m, d in degree.items() if m != root_cause_id}
            if remaining_degree:
                secondary_root_id = max(remaining_degree, key=lambda m: (remaining_degree[m], m))
            elif len(unlock_ids) > 1:
                secondary_root_id = [m for m in unlock_ids if m != root_cause_id][0]

        else:
            # GA 未找到可行解 → 回退到 TRCG-Heuristic
            decision_source = "heuristic_fallback"
            fallback_reason = fallback_reason or "ga_no_feasible_solution"
            if heuristic_decision is not None:
                unlock_ids = heuristic_decision.unlock_mission_ids
                root_cause_id = heuristic_decision.root_cause_mission_id
                secondary_root_id = heuristic_decision.secondary_root_cause_mission_id
            else:
                # 极端兜底
                unlock_ids = []
                root_cause_id = ""
                fallback_reason = "no_active_missions"

        # ============================================================
        # Step 7: 构造 MetaParams
        # ============================================================
        # freeze / epsilon 由规则确定（与 heuristic 一致）
        pressure = trcg_dict.get("bottleneck_pressure", {})
        pad_pressure = pressure.get("pad_util", 0.0)
        num_urgent = len(trcg_dict.get("urgent_missions", []))

        if pad_pressure > 0.80 or num_urgent >= 3:
            freeze_h_hours = 0
            epsilon = 0.02
        elif pad_pressure > 0.50 or num_urgent >= 2:
            freeze_h_hours = 4
            epsilon = 0.05
        else:
            freeze_h_hours = 8
            epsilon = 0.05

        freeze_h_slots = FREEZE_HOURS_TO_SLOTS.get(freeze_h_hours, config.freeze_horizon)

        meta = MetaParams(
            w_delay=self._w_delay,
            w_shift=self._w_shift,
            w_switch=self._w_switch,
            freeze_horizon=freeze_h_slots,
            use_two_stage=True,
            epsilon_solver=epsilon,
            kappa_win=config.default_kappa_win,
            kappa_seq=config.default_kappa_seq,
            # TRCG Repair 扩展字段
            unlock_mission_ids=tuple(unlock_ids) if unlock_ids else None,
            root_cause_mission_id=root_cause_id,
            secondary_root_cause_mission_id=secondary_root_id,
            decision_source=decision_source,
            fallback_reason=fallback_reason,
            attempt_idx=0,
        )

        # ============================================================
        # Step 8: 日志 & 统计
        # ============================================================
        wall_ms = int((_time.time() - t0) * 1000)
        ga_stats.wall_time_ms = wall_ms

        self._ga_stats_history.append(ga_stats)

        if self._enable_logging and self._log_dir:
            self._write_ga_log(now, ga_stats, meta, wall_ms)

        logger.info(
            "t=%d src=%s fit=%.2f unlock=%s gens=%d evals=%d cache=%d stop=%s wall=%dms",
            now,
            decision_source,
            ga_stats.best_fitness if ga_stats.best_fitness < GA_BIG_M else float("inf"),
            unlock_ids,
            ga_stats.generations_run,
            ga_stats.total_evaluations,
            ga_stats.cache_hits,
            ga_stats.stop_reason,
            wall_ms,
        )

        return meta, None

    # ------------------------------------------------------------------
    # 日志辅助
    # ------------------------------------------------------------------

    def _write_ga_log(self, now: int, ga_stats: GAStats, meta: MetaParams, wall_ms: int) -> None:
        """写 GA 搜索日志到 JSON 文件。"""
        import json
        if not self._log_dir:
            return
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            log_data = {
                "now_slot": now,
                "episode_id": self._episode_id,
                "call_count": self._call_count,
                "decision_source": meta.decision_source,
                "unlock_mission_ids": list(meta.unlock_mission_ids) if meta.unlock_mission_ids else [],
                "root_cause": meta.root_cause_mission_id,
                "ga_stats": ga_stats.to_dict(),
                "wall_time_ms": wall_ms,
            }
            path = os.path.join(
                self._log_dir,
                f"ga_step_{self._episode_id}_t{now:04d}.json",
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to write GA log: %s", exc)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "policy_name": self._policy_name,
            "call_count": self._call_count,
            "ga_stats_history": [s.to_dict() for s in self._ga_stats_history],
            "total_evaluations": sum(s.total_evaluations for s in self._ga_stats_history),
            "total_cache_hits": sum(s.cache_hits for s in self._ga_stats_history),
        }

    def get_ga_stats_history(self) -> List[GAStats]:
        return list(self._ga_stats_history)


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_ga_repair_policy(
    log_dir: str = "llm_logs",
    episode_id: str = "",
    **kwargs,
) -> GARepairPolicy:
    """创建 GARepairPolicy 实例。"""
    return GARepairPolicy(
        policy_name=kwargs.pop("policy_name", "ga_repair"),
        log_dir=log_dir,
        enable_logging=True,
        episode_id=episode_id,
        **kwargs,
    )


# ============================================================================
# 自测入口
# ============================================================================

if __name__ == "__main__":
    """
    构造一个包含 8 个任务的小场景，验证 GA V2 加速版能否在有限代内通过 Solver
    产出合法的 unlock_set 和非 NaN 的 drift 值。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from scenario import generate_scenario
    from simulator import simulate_episode
    from config import Config, make_config_for_difficulty

    print("=" * 60)
    print(" GARepairPolicy V2 Self-Test (加速版)")
    print("=" * 60)

    # 创建小场景: 8 个任务, light 扰动
    cfg = make_config_for_difficulty("light")
    cfg.num_missions = 8
    cfg.sim_total_slots = 192        # 2 天 (缩短仿真)
    cfg.solver_timeout_s = 10.0

    scenario = generate_scenario(seed=42, config=cfg)

    print(f"\nScenario: {len(scenario.missions)} missions, "
          f"{len(scenario.resources)} resources, "
          f"sim_total={cfg.sim_total_slots} slots")

    # ---- V2 加速版 ----
    policy_v2 = GARepairPolicy(
        policy_name="ga_repair_v2_test",
        pop_size=12,
        generations=5,
        K=3,
        mutation_rate=0.3,
        candidate_pool_size=8,
        n_jobs=4,
        eval_budget=48,
        early_stop_patience=2,
        eval_timeout_s=0.5,
        final_timeout_s=2.0,
        eval_cp_workers=1,
        final_cp_workers=4,
        enable_cache=True,
        log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_output_temp"),
        enable_logging=True,
        episode_id="ga_selftest_v2",
    )

    print(f"\nPolicy: {policy_v2.name}")
    print(f"  pop_size={policy_v2._pop_size}, gens={policy_v2._generations}, "
          f"K={policy_v2._K}, n_jobs={policy_v2._n_jobs}, "
          f"eval_budget={policy_v2._eval_budget}, eval_timeout={policy_v2._eval_timeout_s}s")

    # 运行仿真
    print("\nRunning simulation...")
    t_v2_start = _time.time()
    result_v2 = simulate_episode(policy_v2, scenario, cfg, verbose=False)
    t_v2_ms = int((_time.time() - t_v2_start) * 1000)

    m = result_v2.metrics
    print(f"\n{'='*40} Results {'='*40}")
    print(f"  total_delay       = {m.total_delay}")
    print(f"  episode_drift     = {m.episode_drift:.4f}")
    print(f"  drift_per_replan  = {m.drift_per_replan:.4f}")
    print(f"  total_switches    = {m.total_switches}")
    print(f"  feasible_rate     = {m.feasible_rate:.4f}")
    print(f"  num_replans       = {m.num_replans}")
    print(f"  wall_time_ms      = {t_v2_ms}")

    # 验证 drift 非 NaN
    import math
    assert not math.isnan(m.episode_drift), "episode_drift is NaN!"
    assert not math.isnan(m.drift_per_replan), "drift_per_replan is NaN!"

    # 打印 GA 统计
    stats = policy_v2.get_stats()
    print(f"\nGA Stats:")
    print(f"  call_count        = {stats['call_count']}")
    print(f"  total_evaluations = {stats['total_evaluations']}")
    print(f"  total_cache_hits  = {stats['total_cache_hits']}")

    for i, gs in enumerate(policy_v2.get_ga_stats_history()):
        d = gs.to_dict()
        print(f"\n  Step {i+1}: gens={d['generations_run']} evals={d['total_evaluations']} "
              f"cache={d['cache_hits']} hit_rate={d['cache_hit_rate']:.2f} "
              f"stop={d['stop_reason']} "
              f"best={d['best_fitness']} "
              f"ga_wall={d['ga_wall_ms']}ms final_recomp={d['final_recompute_ms']}ms "
              f"eval_avg={d['eval_solve_time_ms_avg']}ms eval_p95={d['eval_solve_time_ms_p95']}ms")

    print(f"\n{'='*60}")
    print(" Self-Test PASSED!")
    print(f"{'='*60}")
