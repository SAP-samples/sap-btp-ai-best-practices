"""Build the disclosed five-point portfolio frontier for demo scenarios.

The portfolio master does not return a global candidate-library frontier.  It
chooses one of the unique block partitions supplied by the five core scenario
results, subject to a visible :math:`J_{CH}` epsilon budget.  Example::

    points = build_portfolio_frontier(core_results, config)
    assert len(points) == 5
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from production_wheel.candidates import CandidateMember, CandidatePool
from production_wheel.candidates.models import BlockKey, stable_hash
from production_wheel.optimization import (
    BlockSessionMetrics,
    BlockSolverSession,
    BlockSolveEvidence,
    ObjectiveLevel,
    SelectedCandidate,
    SolveResult,
    solve_candidate_pools,
)
from production_wheel.schemas import CoverageMode, RunConfig
from production_wheel.rule_models import SelectionBound

PORTFOLIO_FRONTIER_LABEL = "frontier_over_disclosed_core_block_partitions"
GREENFIELD_FRONTIER_LABEL = "greenfield_frontier_over_block_nondominated_partitions"
_TOLERANCE = 1e-7
_PROOF_GAP_TOLERANCE = 1e-9
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class PortfolioOption:
    """One unique complete partition for one plant/SEFI block."""

    block_key: BlockKey
    partition_hash: str
    selected: tuple[SelectedCandidate, ...]
    maximum_coverage_days: float
    j_ch: float
    source_scenarios: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One epsilon-constrained portfolio result with honest solver evidence."""

    point_index: int
    epsilon_j_ch: float | None
    status: str
    result_class: str
    selected: tuple[SelectedCandidate, ...] = ()
    maximum_coverage_days: float | None = None
    j_ch: float | None = None
    portfolio_hash: str | None = None
    source_scenarios: tuple[str, ...] = ()
    termination_condition: str = "not_solved"
    relative_gap: float | None = None
    solver_method: str = "pyomo_appsi_highs_portfolio_master"
    frontier_label: str = PORTFOLIO_FRONTIER_LABEL
    error_type: str | None = None
    error_message: str | None = None
    demand_weighted_mean_coverage_days: float | None = None
    proof_scope: str = "disclosed-scenario-library"
    block_option_count: int = 0
    solve_result: SolveResult | None = None
    coverage_anchor_match: bool | None = None
    operations_anchor_match: bool | None = None


@dataclass(frozen=True, slots=True)
class GreenfieldBlockOption:
    """One nondominated exact-cover partition generated for a single block."""

    block_key: BlockKey
    partition_hash: str
    selected: tuple[SelectedCandidate, ...]
    coverage_numerator: float
    demand_litres: float
    j_ch: float
    source: str
    proof_scope: str
    status: str
    termination_condition: str
    relative_gap: float | None


@dataclass(frozen=True, slots=True)
class GreenfieldBlockSolveAudit:
    """One requested anchor, epsilon solve, or unsolved interval stop."""

    block_key: BlockKey
    request_kind: str
    requested_epsilon_j_ch: float | None
    actual_j_ch: float | None
    coverage_numerator: float | None
    partition_hash: str | None
    status: str
    termination_condition: str
    relative_gap: float | None
    refinement_reason: str
    warm_start_kind: str
    solver_call_count: int
    solver_optimize_seconds: float
    heuristic_kind: str | None = None
    heuristic_iterations: int | None = None
    heuristic_moves: int | None = None
    heuristic_construction_seconds: float | None = None
    heuristic_coverage_numerator: float | None = None
    heuristic_j_ch: float | None = None
    heuristic_valid: bool | None = None
    solver_improved_heuristic: bool | None = None


@dataclass(frozen=True, slots=True)
class GreenfieldBlockSessionAudit:
    """Final persistent-model instrumentation for one completed block."""

    block_key: BlockKey
    candidate_count: int
    metrics: BlockSessionMetrics


@dataclass(frozen=True, slots=True)
class GreenfieldBlockFailure:
    """Explicit worker failure retained beside successful block evidence."""

    block_key: BlockKey
    candidate_count: int
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class GreenfieldBlockOptionsResult:
    """Nondominated block options plus complete adaptive-solve evidence."""

    options_by_block: dict[BlockKey, tuple[GreenfieldBlockOption, ...]]
    solve_audits: tuple[GreenfieldBlockSolveAudit, ...]
    session_audits: tuple[GreenfieldBlockSessionAudit, ...]
    failures: tuple[GreenfieldBlockFailure, ...] = ()
    execution_mode: str = "sequential"
    worker_count: int = 1


@dataclass(frozen=True, slots=True)
class GreenfieldGlobalSolveAudit:
    """One low-epsilon-biased compact-master request and its returned portfolio."""

    request_index: int
    normalized_position: float
    schedule_exponent: float
    requested_epsilon_j_ch: float
    actual_j_ch: float
    demand_weighted_mean_coverage_days: float
    portfolio_hash: str
    status: str
    termination_condition: str
    primary_relative_gap: float | None
    final_relative_gap: float | None
    proof_scope: str
    retained_nondominated: bool = False


@dataclass(frozen=True, slots=True)
class GreenfieldFrontierResult:
    """Greenfield anchors, block options, and nondominated portfolio points."""

    config: RunConfig
    pools: tuple[CandidatePool, ...]
    members: tuple[CandidateMember, ...]
    block_options: tuple[GreenfieldBlockOption, ...]
    points: tuple[ParetoPoint, ...]
    coverage_anchor_hash: str
    operations_anchor_hash: str
    elapsed_seconds: float
    per_block_stage_seconds: float | None = None
    per_block_total_seconds: float | None = None
    block_solve_audits: tuple[GreenfieldBlockSolveAudit, ...] = ()
    block_session_audits: tuple[GreenfieldBlockSessionAudit, ...] = ()
    block_failures: tuple[GreenfieldBlockFailure, ...] = ()
    block_execution_mode: str = "sequential"
    block_worker_count: int = 1
    large_block_candidate_threshold: int = 250_000
    global_epsilon_exponent: float = 2.0
    global_solve_audits: tuple[GreenfieldGlobalSolveAudit, ...] = ()

    @property
    def baseline_available(self) -> bool:
        """Return whether every modeled FINI carries a historical group label."""

        return bool(self.members) and all(member.baseline_group for member in self.members)


@dataclass(frozen=True, slots=True)
class _MasterResult:
    """Internal portfolio-master incumbent and proof metadata."""

    status: str
    termination: str
    optimal: bool
    selected_options: tuple[PortfolioOption, ...]
    maximum_coverage_days: float | None
    j_ch: float | None
    relative_gap: float | None


def collect_portfolio_options(
    core_results: Mapping[str, SolveResult],
) -> dict[BlockKey, tuple[PortfolioOption, ...]]:
    """Deduplicate block partitions supplied by successful core scenarios.

    Args:
        core_results: Scenario identifier to successful or failed solve result.

    Returns:
        Stable block mapping whose options record every contributing scenario.
    """

    collected: dict[BlockKey, dict[str, tuple[list[SelectedCandidate], set[str]]]] = {}
    for scenario_id, result in core_results.items():
        if not result.has_incumbent:
            continue
        by_block: dict[BlockKey, list[SelectedCandidate]] = {}
        for selected in result.selected:
            by_block.setdefault(selected.candidate.block_key, []).append(selected)
        for block, values in by_block.items():
            ordered = sorted(values, key=lambda item: item.candidate.candidate_hash)
            identity = [item.candidate.candidate_hash for item in ordered]
            partition_hash = stable_hash({"block": block, "candidates": identity})
            existing = collected.setdefault(block, {}).get(partition_hash)
            if existing is None:
                collected[block][partition_hash] = (ordered, {scenario_id})
            else:
                existing[1].add(scenario_id)

    result: dict[BlockKey, tuple[PortfolioOption, ...]] = {}
    for block in sorted(collected):
        options = []
        for partition_hash, (selected, sources) in sorted(collected[block].items()):
            options.append(
                PortfolioOption(
                    block_key=block,
                    partition_hash=partition_hash,
                    selected=tuple(selected),
                    maximum_coverage_days=max(
                        item.coefficients.coverage_days for item in selected
                    ),
                    j_ch=sum(item.coefficients.j_ch for item in selected),
                    source_scenarios=tuple(sorted(sources)),
                )
            )
        result[block] = tuple(options)
    return result


def _relative_gap(incumbent: float | None, bound: float | None) -> float | None:
    """Calculate a non-negative minimization gap when proof values are finite."""

    if incumbent is None or bound is None or not math.isfinite(incumbent + bound):
        return None
    return abs(incumbent - bound) / max(abs(incumbent), 1e-10)


def _solve_master(
    options_by_block: Mapping[BlockKey, Sequence[PortfolioOption]],
    config: RunConfig,
    objective: str,
    epsilon_j_ch: float | None = None,
) -> _MasterResult:
    """Solve one compact partition-choice master and apply a deterministic tie tier."""

    options = tuple(
        option
        for block in sorted(options_by_block)
        for option in options_by_block[block]
    )
    if not options_by_block or any(not values for values in options_by_block.values()):
        return _MasterResult("infeasible", "missing_block_options", False, (), None, None, None)
    model = pyo.ConcreteModel(name="disclosed_partition_portfolio")
    model.O = pyo.RangeSet(0, len(options) - 1)
    model.x = pyo.Var(model.O, domain=pyo.Binary)
    model.cover = pyo.ConstraintList()
    for block in sorted(options_by_block):
        indexes = [index for index, option in enumerate(options) if option.block_key == block]
        model.cover.add(pyo.quicksum(model.x[index] for index in indexes) == 1)
    model.maximum_coverage = pyo.Var(domain=pyo.NonNegativeReals)
    for index, option in enumerate(options):
        model.cover.add(
            model.maximum_coverage >= option.maximum_coverage_days * model.x[index]
        )
    j_ch = pyo.quicksum(options[index].j_ch * model.x[index] for index in model.O)
    if epsilon_j_ch is not None:
        model.cover.add(j_ch <= epsilon_j_ch + _TOLERANCE * max(1.0, epsilon_j_ch))
    primary = j_ch if objective == "j_ch" else model.maximum_coverage
    model.objective = pyo.Objective(expr=primary, sense=pyo.minimize)
    solver = Highs()
    solver.config.load_solution = False
    solver.config.time_limit = config.solver_limits.time_limit_seconds
    solver.config.mip_gap = config.solver_limits.mip_gap
    solver.config.stream_solver = False
    solver.highs_options = {
        "random_seed": config.solver_limits.random_seed,
        "threads": config.solver_limits.threads,
        "parallel": "off" if config.solver_limits.threads == 1 else "on",
    }

    solved = solver.solve(model)
    incumbent = solved.best_feasible_objective
    if incumbent is None:
        return _MasterResult(
            "no_incumbent", solved.termination_condition.name, False, (), None, None, None
        )
    primals = solved.solution_loader.get_primals(
        vars_to_load=[model.x[index] for index in model.O]
    )
    optimal = solved.termination_condition is TerminationCondition.optimal
    # Coverage points use J_CH as an explicit secondary objective. This removes
    # dominated choices without inventing a blended score.
    if objective == "coverage" and optimal:
        tolerance = _TOLERANCE * max(1.0, float(incumbent))
        model.cover.add(model.maximum_coverage <= float(incumbent) + tolerance)
        model.objective.set_value(j_ch)
        solved = solver.solve(model)
        if solved.best_feasible_objective is not None:
            primals = solved.solution_loader.get_primals(
                vars_to_load=[model.x[index] for index in model.O]
            )
            optimal = solved.termination_condition is TerminationCondition.optimal
    selected_options = tuple(
        options[index] for index in model.O if float(primals[model.x[index]]) > 0.5
    )
    maximum = max(option.maximum_coverage_days for option in selected_options)
    total_j_ch = sum(option.j_ch for option in selected_options)
    return _MasterResult(
        "optimal" if optimal else "feasible_limit",
        solved.termination_condition.name,
        optimal,
        selected_options,
        maximum,
        total_j_ch,
        _relative_gap(solved.best_feasible_objective, solved.best_objective_bound),
    )


def _point(index: int, epsilon: float, solved: _MasterResult) -> ParetoPoint:
    """Convert a master result into the public point contract."""

    selected = tuple(item for option in solved.selected_options for item in option.selected)
    identity = [option.partition_hash for option in solved.selected_options]
    return ParetoPoint(
        point_index=index,
        epsilon_j_ch=epsilon,
        status=solved.status,
        result_class=(
            "portfolio-optimal"
            if solved.optimal
            else "portfolio-bounded"
            if solved.relative_gap is not None and solved.relative_gap <= 0.01
            else "portfolio-exploratory"
            if solved.relative_gap is not None and solved.relative_gap <= 0.05
            else "portfolio-heuristic"
        ),
        selected=selected,
        maximum_coverage_days=solved.maximum_coverage_days,
        j_ch=solved.j_ch,
        portfolio_hash=stable_hash(identity) if identity else None,
        source_scenarios=tuple(
            sorted({source for option in solved.selected_options for source in option.source_scenarios})
        ),
        termination_condition=solved.termination,
        relative_gap=0.0 if solved.optimal else solved.relative_gap,
    )


def build_portfolio_frontier(
    core_results: Mapping[str, SolveResult],
    config: RunConfig,
    point_count: int = 5,
) -> tuple[ParetoPoint, ...]:
    """Build visible max-coverage/J_CH trade-offs over disclosed partitions.

    Failed core scenarios are ignored when other successful scenarios disclose
    a partition for every block. If a frontier cannot be built, one failed
    placeholder is returned for each requested point so suite cardinality and
    partial-run evidence remain intact.
    """

    if point_count < 1:
        raise ValueError("point_count must be positive")
    try:
        options = collect_portfolio_options(core_results)
        operational = _solve_master(options, config, "j_ch")
        coverage = _solve_master(options, config, "coverage")
        if not operational.selected_options or not coverage.selected_options:
            raise RuntimeError("core outcomes do not disclose a complete feasible portfolio")
        low = float(operational.j_ch)  # type: ignore[arg-type]
        high = max(low, float(coverage.j_ch))  # type: ignore[arg-type]
        budgets = (
            [low]
            if point_count == 1
            else [low + (high - low) * index / (point_count - 1) for index in range(point_count)]
        )
        return tuple(
            _point(index + 1, budget, _solve_master(options, config, "coverage", budget))
            for index, budget in enumerate(budgets)
        )
    except Exception as exc:  # Suite orchestration must preserve partial evidence.
        return tuple(
            ParetoPoint(
                point_index=index,
                epsilon_j_ch=None,
                status="failed",
                result_class="portfolio-unavailable",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            for index in range(1, point_count + 1)
        )


def _selected_axes(
    selected: Iterable[SelectedCandidate],
) -> tuple[float, float, float]:
    """Return coverage numerator, represented demand, and total ``J_CH``."""

    values = tuple(selected)
    return (
        sum(
            item.coefficients.coverage_days
            * item.coefficients.group_demand_litres
            for item in values
        ),
        sum(item.coefficients.group_demand_litres for item in values),
        sum(item.coefficients.j_ch for item in values),
    )


def _greenfield_proof_scope(result: SolveResult, pool: CandidatePool) -> str:
    """Classify one block partition as exact, restricted, or runtime-limited."""

    if result.status != "optimal":
        return "runtime-limited"
    return "exact" if pool.completeness == "complete" else "restricted-library"


def _block_option(
    pool: CandidatePool, result: SolveResult, source: str
) -> GreenfieldBlockOption:
    """Convert one block incumbent into a stable option for coordination."""

    if not result.has_incumbent:
        raise RuntimeError(f"{source} produced no incumbent for block {pool.block_key}")
    selected = tuple(
        sorted(result.selected, key=lambda item: item.candidate.candidate_hash)
    )
    numerator, demand, j_ch = _selected_axes(selected)
    identity = [item.candidate.candidate_hash for item in selected]
    return GreenfieldBlockOption(
        block_key=pool.block_key,
        partition_hash=stable_hash({"block": pool.block_key, "candidates": identity}),
        selected=selected,
        coverage_numerator=numerator,
        demand_litres=demand,
        j_ch=j_ch,
        source=source,
        proof_scope=_greenfield_proof_scope(result, pool),
        status=result.status,
        termination_condition=result.termination_condition,
        relative_gap=result.relative_gap,
    )


def _nondominated_block_options(
    options: Iterable[GreenfieldBlockOption],
) -> tuple[GreenfieldBlockOption, ...]:
    """Deduplicate partitions and remove coverage/``J_CH`` dominated options."""

    unique: dict[str, GreenfieldBlockOption] = {}
    for option in options:
        unique.setdefault(option.partition_hash, option)
    ordered = sorted(
        unique.values(),
        key=lambda item: (item.j_ch, item.coverage_numerator, item.partition_hash),
    )
    retained: list[GreenfieldBlockOption] = []
    seen_axes: set[tuple[float, float]] = set()
    for option in ordered:
        axes = (option.j_ch, option.coverage_numerator)
        if axes in seen_axes:
            continue
        seen_axes.add(axes)
        if any(
            other.j_ch <= option.j_ch + _TOLERANCE
            and other.coverage_numerator <= option.coverage_numerator + _TOLERANCE
            and (
                other.j_ch < option.j_ch - _TOLERANCE
                or other.coverage_numerator
                < option.coverage_numerator - _TOLERANCE
            )
            for other in ordered
        ):
            continue
        retained.append(option)
    return tuple(retained)


def _greenfield_stage_time_limits(
    pools: Sequence[CandidatePool], total_seconds: float
) -> dict[BlockKey, float]:
    """Allocate a scenario-wide stage budget with a one-second solve floor.

    Tiny exact blocks otherwise receive sub-millisecond proportional limits and
    appear runtime-limited despite solving immediately once HiGHS initializes.
    The floor is reduced when the requested total is below the block count, so
    the returned allocations always sum to the visible scenario-wide budget.
    """

    if total_seconds <= 0:
        raise ValueError("greenfield stage time budget must be positive")
    if not pools:
        return {}
    floor = min(1.0, total_seconds / len(pools))
    remaining = max(total_seconds - floor * len(pools), 0.0)
    weights = {pool.block_key: max(len(pool.candidates), 1) for pool in pools}
    total_weight = sum(weights.values())
    return {
        block_key: floor + remaining * weight / total_weight
        for block_key, weight in weights.items()
    }


def _block_solve_audit(
    pool: CandidatePool,
    request_kind: str,
    epsilon: float | None,
    result: SolveResult,
    option: GreenfieldBlockOption | None,
    reason: str,
    before: BlockSessionMetrics,
    after: BlockSessionMetrics,
) -> GreenfieldBlockSolveAudit:
    """Convert one persistent solve and its point-local metric delta to audit."""

    heuristic = result.heuristic_evidence
    return GreenfieldBlockSolveAudit(
        block_key=pool.block_key,
        request_kind=request_kind,
        requested_epsilon_j_ch=epsilon,
        actual_j_ch=option.j_ch if option else None,
        coverage_numerator=option.coverage_numerator if option else None,
        partition_hash=option.partition_hash if option else None,
        status=result.status,
        termination_condition=result.termination_condition,
        relative_gap=result.relative_gap,
        refinement_reason=reason,
        warm_start_kind=result.warm_start_kind,
        solver_call_count=after.solver_call_count - before.solver_call_count,
        solver_optimize_seconds=(
            after.solver_optimize_seconds - before.solver_optimize_seconds
        ),
        heuristic_kind=heuristic.kind if heuristic else None,
        heuristic_iterations=heuristic.iterations if heuristic else None,
        heuristic_moves=heuristic.moves if heuristic else None,
        heuristic_construction_seconds=(
            heuristic.construction_seconds if heuristic else None
        ),
        heuristic_coverage_numerator=(
            heuristic.coverage_numerator if heuristic else None
        ),
        heuristic_j_ch=heuristic.j_ch if heuristic else None,
        heuristic_valid=(
            heuristic.exact_cover_valid and heuristic.within_epsilon
            if heuristic
            else None
        ),
        solver_improved_heuristic=(
            heuristic.solver_improved if heuristic else None
        ),
    )


def _interval_stop_audit(
    block_key: BlockKey,
    left: GreenfieldBlockOption,
    right: GreenfieldBlockOption,
    reason: str,
) -> GreenfieldBlockSolveAudit:
    """Record an adaptive interval that stopped without requesting HiGHS."""

    return GreenfieldBlockSolveAudit(
        block_key=block_key,
        request_kind="interval_stop",
        requested_epsilon_j_ch=(left.j_ch + right.j_ch) / 2,
        actual_j_ch=None,
        coverage_numerator=None,
        partition_hash=None,
        status="not_solved",
        termination_condition="not_solved",
        relative_gap=None,
        refinement_reason=reason,
        warm_start_kind="none",
        solver_call_count=0,
        solver_optimize_seconds=0.0,
    )


def _build_one_greenfield_block_options(
    pool: CandidatePool,
    block_members: Sequence[CandidateMember],
    config: RunConfig,
    option_count: int,
    stage_seconds: float,
) -> tuple[
    tuple[GreenfieldBlockOption, ...],
    tuple[GreenfieldBlockSolveAudit, ...],
    GreenfieldBlockSessionAudit,
]:
    """Build one adaptive block frontier on one persistent solver session."""

    local_config = config.model_copy(
        update={
            "coverage_mode": CoverageMode.PARETO,
            "solver_limits": config.solver_limits.model_copy(
                update={"time_limit_seconds": stage_seconds}
            ),
        }
    )
    session = BlockSolverSession(pool, block_members, local_config)
    audits: list[GreenfieldBlockSolveAudit] = []

    def request(
        mode: CoverageMode,
        kind: str,
        epsilon: float | None = None,
        warm_start: Iterable[str] = (),
    ) -> GreenfieldBlockOption | None:
        """Run and audit one anchor or adaptive epsilon request."""

        before = session.metrics
        solved = session.solve(
            mode,
            j_ch_epsilon=epsilon,
            warm_start_candidate_hashes=warm_start,
        )
        after = session.metrics
        option = _block_option(pool, solved, kind) if solved.has_incumbent else None
        audits.append(
            _block_solve_audit(
                pool,
                kind,
                epsilon,
                solved,
                option,
                "anchor" if epsilon is None else "pending_refinement",
                before,
                after,
            )
        )
        return option

    operations = request(CoverageMode.GREENFIELD_OPERATIONS, "operations_anchor")
    coverage = request(CoverageMode.GREENFIELD_COVERAGE, "coverage_anchor")
    if operations is None or coverage is None:
        raise RuntimeError(f"greenfield anchor produced no incumbent for {pool.block_key}")
    options = [operations, coverage]
    same_anchor = (
        operations.partition_hash == coverage.partition_hash
        or (
            math.isclose(operations.j_ch, coverage.j_ch, abs_tol=_TOLERANCE)
            and math.isclose(
                operations.coverage_numerator,
                coverage.coverage_numerator,
                abs_tol=_TOLERANCE,
            )
        )
    )
    intervals: list[tuple[GreenfieldBlockOption, GreenfieldBlockOption]] = []
    if same_anchor:
        audits[-1] = replace(
            audits[-1], refinement_reason="anchors_duplicate_stop"
        )
    else:
        intervals.append((operations, coverage))
    solve_requests = 2
    epsilon_index = 0
    while intervals and solve_requests < option_count:
        intervals.sort(
            key=lambda item: (
                -(item[1].j_ch - item[0].j_ch),
                item[0].j_ch,
                item[0].partition_hash,
                item[1].partition_hash,
            )
        )
        left, right = intervals.pop(0)
        if right.j_ch - left.j_ch <= _TOLERANCE:
            audits.append(
                _interval_stop_audit(
                    pool.block_key, left, right, "j_ch_interval_below_tolerance"
                )
            )
            continue
        if left.coverage_numerator - right.coverage_numerator <= _TOLERANCE:
            audits.append(
                _interval_stop_audit(
                    pool.block_key, left, right, "coverage_improvement_below_tolerance"
                )
            )
            continue
        epsilon_index += 1
        epsilon = (left.j_ch + right.j_ch) / 2
        warm_start = tuple(
            item.candidate.candidate_hash for item in left.selected
        )
        option = request(
            CoverageMode.PARETO,
            f"epsilon_adaptive_{epsilon_index:02d}",
            epsilon,
            warm_start,
        )
        solve_requests += 1
        if option is None:
            audits[-1] = replace(
                audits[-1], refinement_reason="no_incumbent_stop"
            )
            continue
        if option.partition_hash in {
            left.partition_hash,
            right.partition_hash,
        }:
            audits[-1] = replace(
                audits[-1], refinement_reason="duplicate_endpoint_stop"
            )
            continue
        local_frontier = _nondominated_block_options((left, option, right))
        if option.partition_hash not in {
            item.partition_hash for item in local_frontier
        }:
            audits[-1] = replace(
                audits[-1], refinement_reason="dominated_partition_stop"
            )
            continue
        if not (
            left.j_ch + _TOLERANCE
            < option.j_ch
            < right.j_ch - _TOLERANCE
        ):
            audits[-1] = replace(
                audits[-1], refinement_reason="endpoint_axis_stop"
            )
            options.append(option)
            continue
        audits[-1] = replace(
            audits[-1], refinement_reason="new_nondominated_partition_refine"
        )
        options.append(option)
        intervals.extend(((left, option), (option, right)))
    for left, right in intervals:
        audits.append(
            _interval_stop_audit(
                pool.block_key, left, right, "solve_budget_reached"
            )
        )
    return (
        _nondominated_block_options(options),
        tuple(audits),
        GreenfieldBlockSessionAudit(
            block_key=pool.block_key,
            candidate_count=len(pool.candidates),
            metrics=session.metrics,
        ),
    )


def build_greenfield_block_options(
    pools: Sequence[CandidatePool],
    members: Iterable[CandidateMember],
    config: RunConfig,
    option_count: int = 5,
    per_block_stage_seconds: float | None = None,
    per_block_total_seconds: float | None = None,
    worker_count: int = 2,
    large_block_candidate_threshold: int = 250_000,
    progress: ProgressCallback | None = None,
) -> GreenfieldBlockOptionsResult:
    """Generate adaptive nondominated partitions with one model per block.

    option_count is the maximum number of anchor-plus-epsilon solve requests
    per block. Unproductive intervals stop early and remain visible in the
    returned audit evidence.
    """

    if option_count < 2:
        raise ValueError("greenfield block option count must be at least two")
    if not config.is_greenfield:
        raise ValueError("greenfield block options require a greenfield mode")
    if per_block_stage_seconds is not None and per_block_stage_seconds <= 0:
        raise ValueError("per-block stage time must be positive")
    if per_block_total_seconds is not None and per_block_total_seconds <= 0:
        raise ValueError("per-block total time must be positive")
    if per_block_stage_seconds is not None and per_block_total_seconds is not None:
        raise ValueError("per-block stage and total time limits are mutually exclusive")
    if worker_count < 1:
        raise ValueError("block worker count must be positive")
    if large_block_candidate_threshold < 1:
        raise ValueError("large block candidate threshold must be positive")
    population = tuple(members)
    ordered_pools = tuple(sorted(pools, key=lambda item: item.block_key))
    explicit_stage_seconds = (
        per_block_stage_seconds
        if per_block_stage_seconds is not None
        else per_block_total_seconds / (2 * option_count)
        if per_block_total_seconds is not None
        else None
    )
    time_limits = (
        {pool.block_key: explicit_stage_seconds for pool in ordered_pools}
        if explicit_stage_seconds is not None
        else _greenfield_stage_time_limits(
            ordered_pools, config.solver_limits.time_limit_seconds
        )
    )
    from production_wheel.parallel_blocks import solve_block_frontiers_parallel

    execution = solve_block_frontiers_parallel(
        ordered_pools,
        population,
        config,
        option_count,
        time_limits,
        worker_count,
        large_block_candidate_threshold,
        progress,
    )
    options_by_block: dict[BlockKey, tuple[GreenfieldBlockOption, ...]] = {}
    solve_audits: list[GreenfieldBlockSolveAudit] = []
    session_audits: list[GreenfieldBlockSessionAudit] = []
    for block_key, options, audits, session_audit in execution.completed:
        options_by_block[block_key] = options
        solve_audits.extend(audits)
        session_audits.append(session_audit)
    return GreenfieldBlockOptionsResult(
        options_by_block=options_by_block,
        solve_audits=tuple(solve_audits),
        session_audits=tuple(session_audits),
        failures=execution.failures,
        execution_mode=execution.execution_mode,
        worker_count=execution.worker_count,
    )


@dataclass(frozen=True, slots=True)
class _GreenfieldMasterResult:
    """Internal partition-master incumbent and proof evidence."""

    selected_options: tuple[GreenfieldBlockOption, ...]
    termination: str
    exact: bool
    coverage_numerator: float | None
    j_ch: float | None
    primary_bound: float | None
    primary_gap: float | None
    final_bound: float | None
    final_gap: float | None


def _solve_greenfield_master(
    options_by_block: Mapping[BlockKey, Sequence[GreenfieldBlockOption]],
    config: RunConfig,
    epsilon_j_ch: float,
) -> _GreenfieldMasterResult:
    """Minimize coverage over block partitions under one visible ``J_CH`` cap."""

    options = tuple(
        option
        for block in sorted(options_by_block)
        for option in options_by_block[block]
    )
    if not options_by_block or any(not values for values in options_by_block.values()):
        return _GreenfieldMasterResult((), "missing_block_options", False, None, None, None, None, None, None)
    model = pyo.ConcreteModel(name="greenfield_partition_portfolio")
    model.O = pyo.RangeSet(0, len(options) - 1)
    model.x = pyo.Var(model.O, domain=pyo.Binary)
    model.constraints = pyo.ConstraintList()
    for block in sorted(options_by_block):
        indexes = [
            index for index, option in enumerate(options) if option.block_key == block
        ]
        model.constraints.add(pyo.quicksum(model.x[index] for index in indexes) == 1)
        model.x[indexes[0]].value = 1.0
        for index in indexes[1:]:
            model.x[index].value = 0.0
    coverage = pyo.quicksum(
        options[index].coverage_numerator * model.x[index] for index in model.O
    )
    j_ch = pyo.quicksum(options[index].j_ch * model.x[index] for index in model.O)
    model.constraints.add(
        j_ch <= epsilon_j_ch + _TOLERANCE * max(1.0, abs(epsilon_j_ch))
    )
    model.objective = pyo.Objective(expr=coverage, sense=pyo.minimize)
    solver = Highs()
    solver.config.load_solution = False
    solver.config.warmstart = True
    solver.config.time_limit = config.solver_limits.time_limit_seconds
    solver.config.mip_gap = config.solver_limits.mip_gap
    solver.config.stream_solver = False
    solver.highs_options = {
        "random_seed": config.solver_limits.random_seed,
        "threads": config.solver_limits.threads,
        "parallel": "off" if config.solver_limits.threads == 1 else "on",
    }
    primary = solver.solve(model)
    if primary.best_feasible_objective is None:
        return _GreenfieldMasterResult(
            (), primary.termination_condition.name, False, None, None, None, None, None, None
        )
    primary_incumbent = float(primary.best_feasible_objective)
    primary_bound = (
        float(primary.best_objective_bound)
        if primary.best_objective_bound is not None
        else None
    )
    primary_gap = _relative_gap(primary_incumbent, primary_bound)
    primary_exact = (
        primary.termination_condition is TerminationCondition.optimal
        and primary_gap is not None
        and primary_gap <= _PROOF_GAP_TOLERANCE
    )
    primals = primary.solution_loader.get_primals(
        vars_to_load=[model.x[index] for index in model.O]
    )
    final = primary
    final_gap = primary_gap
    if primary_exact:
        model.constraints.add(
            coverage
            <= primary_incumbent
            + _TOLERANCE * max(1.0, abs(primary_incumbent))
        )
        model.objective.set_value(j_ch)
        secondary = solver.solve(model)
        if secondary.best_feasible_objective is not None:
            final = secondary
            final_gap = _relative_gap(
                float(secondary.best_feasible_objective),
                (
                    float(secondary.best_objective_bound)
                    if secondary.best_objective_bound is not None
                    else None
                ),
            )
            primals = secondary.solution_loader.get_primals(
                vars_to_load=[model.x[index] for index in model.O]
            )
    selected = tuple(
        options[index]
        for index in model.O
        if float(primals[model.x[index]]) > 0.5
    )
    coverage_value = sum(option.coverage_numerator for option in selected)
    j_ch_value = sum(option.j_ch for option in selected)
    final_exact = (
        final.termination_condition is TerminationCondition.optimal
        and final_gap is not None
        and final_gap <= _PROOF_GAP_TOLERANCE
    )
    final_bound = (
        float(final.best_objective_bound)
        if final.best_objective_bound is not None
        else None
    )
    return _GreenfieldMasterResult(
        selected,
        final.termination_condition.name if final_exact else (
            "gap_accepted"
            if final.termination_condition is TerminationCondition.optimal
            else final.termination_condition.name
        ),
        primary_exact and final_exact,
        coverage_value,
        j_ch_value,
        primary_bound,
        primary_gap,
        final_bound,
        final_gap,
    )


def _portfolio_hash(options: Iterable[GreenfieldBlockOption]) -> str:
    """Return a stable portfolio digest from ordered block partition hashes."""

    return stable_hash(
        {
            "partitions": [
                option.partition_hash
                for option in sorted(options, key=lambda item: item.block_key)
            ]
        }
    )


def _frontier_proof_scope(
    master: _GreenfieldMasterResult,
) -> str:
    """Return the strongest proof label supported by master and block options."""

    scopes = {option.proof_scope for option in master.selected_options}
    if not master.exact or "runtime-limited" in scopes:
        return "runtime-limited"
    if "restricted-library" in scopes:
        return "restricted-library"
    return "exact"


def _greenfield_point(
    index: int,
    epsilon: float,
    master: _GreenfieldMasterResult,
    pools: Sequence[CandidatePool],
) -> ParetoPoint:
    """Convert a greenfield master incumbent into a validation-ready point."""

    options = master.selected_options
    selected = tuple(item for option in options for item in option.selected)
    total_demand = sum(option.demand_litres for option in options)
    mean = (
        float(master.coverage_numerator) / total_demand
        if master.coverage_numerator is not None and total_demand > 0
        else None
    )
    proof_scope = _frontier_proof_scope(master)
    completeness = (
        "complete"
        if pools and all(pool.completeness == "complete" for pool in pools)
        else "restricted"
    )
    primary_bound = (
        master.primary_bound / total_demand
        if master.primary_bound is not None and total_demand > 0
        else None
    )
    levels = (
        ObjectiveLevel(
            "demand_weighted_mean_coverage_days",
            float(mean),
            master.termination,
            primary_bound,
            master.primary_gap,
        ),
        ObjectiveLevel(
            "j_ch",
            float(master.j_ch),
            master.termination,
            master.final_bound,
            master.final_gap,
        ),
    )
    pool_by_block = {pool.block_key: pool for pool in pools}
    block_evidence = tuple(
        BlockSolveEvidence(
            block_key=option.block_key,
            status=option.status,
            termination_condition=option.termination_condition,
            result_class=f"greenfield-{option.proof_scope}",
            objective_levels=(),
            primary_objective=option.coverage_numerator / option.demand_litres,
            primary_best_bound=None,
            primary_relative_gap=option.relative_gap,
            relative_gap=option.relative_gap,
            candidate_count=len(pool_by_block[option.block_key].candidates),
        )
        for option in sorted(options, key=lambda item: item.block_key)
    )
    status = "optimal" if proof_scope != "runtime-limited" else "feasible_limit"
    result_class = f"greenfield-{proof_scope}"
    solve_result = SolveResult(
        status=status,
        termination_condition=master.termination,
        has_incumbent=bool(selected),
        result_class=result_class,
        selected=selected,
        objective_levels=levels,
        primary_objective=mean,
        primary_best_bound=primary_bound,
        primary_relative_gap=master.primary_gap,
        incumbent_objective=master.j_ch,
        best_bound=master.final_bound,
        relative_gap=0.0 if proof_scope != "runtime-limited" else master.final_gap,
        candidate_count=sum(len(pool.candidates) for pool in pools),
        pool_completeness=completeness,
        solver_method="pyomo_appsi_highs_greenfield_partition_master",
        block_evidence=block_evidence,
        warm_start_kind="block_partition_options",
        warm_start_group_count=len(selected),
        epsilon_j_ch=epsilon,
    )
    return ParetoPoint(
        point_index=index,
        epsilon_j_ch=epsilon,
        status=status,
        result_class=result_class,
        selected=selected,
        maximum_coverage_days=max(
            item.coefficients.coverage_days for item in selected
        ),
        j_ch=master.j_ch,
        portfolio_hash=_portfolio_hash(options),
        source_scenarios=tuple(sorted({option.source for option in options})),
        termination_condition=master.termination,
        relative_gap=0.0 if proof_scope != "runtime-limited" else master.final_gap,
        solver_method="pyomo_appsi_highs_greenfield_partition_master",
        frontier_label=GREENFIELD_FRONTIER_LABEL,
        demand_weighted_mean_coverage_days=mean,
        proof_scope=proof_scope,
        block_option_count=len(options),
        solve_result=solve_result,
    )


def _nondominated_points(points: Iterable[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Keep one representative per KPI tradeoff, then remove dominated points."""

    unique: dict[str, ParetoPoint] = {}
    for point in points:
        if point.portfolio_hash:
            unique.setdefault(point.portfolio_hash, point)
    # Different line assignments can have identical KPI axes. Prefer the
    # strongest proof for that tradeoff instead of presenting duplicate choices.
    values = []
    for point in sorted(unique.values(), key=lambda p: (
        {"exact": 0, "restricted-library": 1, "runtime-limited": 2}.get(p.proof_scope, 3),
        p.status != "optimal", p.portfolio_hash,
    )):
        if not any(abs(point.j_ch - other.j_ch) <= _TOLERANCE and
                   abs(point.demand_weighted_mean_coverage_days - other.demand_weighted_mean_coverage_days) <= _TOLERANCE
                   for other in values):
            values.append(point)
    retained = [
        point
        for point in values
        if not any(
            other.j_ch is not None
            and point.j_ch is not None
            and other.demand_weighted_mean_coverage_days is not None
            and point.demand_weighted_mean_coverage_days is not None
            and other.j_ch <= point.j_ch + _TOLERANCE
            and other.demand_weighted_mean_coverage_days
            <= point.demand_weighted_mean_coverage_days + _TOLERANCE
            and (
                other.j_ch < point.j_ch - _TOLERANCE
                or other.demand_weighted_mean_coverage_days
                < point.demand_weighted_mean_coverage_days - _TOLERANCE
            )
            for other in values
        )
    ]
    return tuple(
        replace(point, point_index=index)
        for index, point in enumerate(
            sorted(retained, key=lambda item: (float(item.j_ch), float(item.demand_weighted_mean_coverage_days))),
            start=1,
        )
    )


def _low_epsilon_budgets(
    low: float,
    high: float,
    point_count: int,
    exponent: float,
) -> tuple[float, ...]:
    """Return deterministic anchor-inclusive budgets denser near low ``J_CH``.

    Args:
        low: Pure operations-anchor ``J_CH``.
        high: Pure coverage-anchor ``J_CH``.
        point_count: Number of requested compact-master solves including anchors.
        exponent: Power applied to normalized positions; values above one make
            low-epsilon steps smaller than high-epsilon steps.

    Returns:
        Strictly ordered visible epsilon budgets, or repeated anchors when both
        objective extremes have the same ``J_CH``.
    """

    if point_count < 2:
        raise ValueError("global frontier requires at least two points")
    if not math.isfinite(exponent) or exponent < 1.0:
        raise ValueError("global epsilon exponent must be finite and at least one")
    width = max(high - low, 0.0)
    return tuple(
        low + width * (index / (point_count - 1)) ** exponent
        for index in range(point_count)
    )


def build_greenfield_frontier(
    pools: Sequence[CandidatePool],
    members: Iterable[CandidateMember],
    config: RunConfig,
    point_count: int = 17,
    block_option_count: int = 5,
    per_block_stage_seconds: float | None = None,
    per_block_total_seconds: float | None = None,
    block_worker_count: int = 2,
    large_block_candidate_threshold: int = 250_000,
    global_epsilon_exponent: float = 2.0,
    progress: ProgressCallback | None = None,
) -> GreenfieldFrontierResult:
    """Build the scalable demand-weighted coverage/``J_CH`` frontier.

    The function first generates nondominated exact-cover partitions inside
    each block, then chooses one partition per block for every global epsilon.
    Historical groups are not read by feasibility, option generation, or the
    portfolio master.
    """

    if point_count < 2:
        raise ValueError("greenfield frontier requires at least two points")
    if not math.isfinite(global_epsilon_exponent) or global_epsilon_exponent < 1:
        raise ValueError("global epsilon exponent must be finite and at least one")
    started = time.monotonic()
    population = tuple(members)
    singleton_keys = {(*c.block_key, c.member_ids[0]) for p in pools for c in p.candidates if len(c.member_ids) == 1}
    # Group predicates already filter candidate pools and do not couple blocks.
    # Preserve the scalable block search and its coverage-improving starts.
    if any(isinstance(rule, SelectionBound) for rule in config.business_rules) or config.preferred_line or any((*m.block_key, m.fini_id) not in singleton_keys for m in population):
        from production_wheel.joint_frontier import build_joint_frontier
        return build_joint_frontier(pools, population, config, point_count=point_count,
            total_seconds=per_block_total_seconds, stage_seconds=per_block_stage_seconds,
            exponent=global_epsilon_exponent, progress=progress)
    block_result = build_greenfield_block_options(
        pools,
        population,
        config,
        option_count=block_option_count,
        per_block_stage_seconds=per_block_stage_seconds,
        per_block_total_seconds=per_block_total_seconds,
        worker_count=block_worker_count,
        large_block_candidate_threshold=large_block_candidate_threshold,
        progress=progress,
    )
    options_by_block = block_result.options_by_block
    flat_options = tuple(
        option
        for block in sorted(options_by_block)
        for option in options_by_block[block]
    )
    if block_result.failures:
        return GreenfieldFrontierResult(
            config=config,
            pools=tuple(pools),
            members=population,
            block_options=flat_options,
            points=(),
            coverage_anchor_hash="",
            operations_anchor_hash="",
            elapsed_seconds=time.monotonic() - started,
            per_block_stage_seconds=per_block_stage_seconds,
            per_block_total_seconds=per_block_total_seconds,
            block_solve_audits=block_result.solve_audits,
            block_session_audits=block_result.session_audits,
            block_failures=block_result.failures,
            block_execution_mode=block_result.execution_mode,
            block_worker_count=block_result.worker_count,
            large_block_candidate_threshold=large_block_candidate_threshold,
            global_epsilon_exponent=global_epsilon_exponent,
        )
    coverage_options = tuple(
        min(
            options_by_block[block],
            key=lambda item: (
                item.coverage_numerator,
                item.j_ch,
                item.partition_hash,
            ),
        )
        for block in sorted(options_by_block)
    )
    operations_options = tuple(
        min(
            options_by_block[block],
            key=lambda item: (
                item.j_ch,
                item.coverage_numerator,
                item.partition_hash,
            ),
        )
        for block in sorted(options_by_block)
    )
    coverage_anchor_hash = _portfolio_hash(coverage_options)
    operations_anchor_hash = _portfolio_hash(operations_options)
    low = sum(option.j_ch for option in operations_options)
    high = max(low, sum(option.j_ch for option in coverage_options))
    budgets = _low_epsilon_budgets(
        low,
        high,
        point_count,
        global_epsilon_exponent,
    )
    raw_points = []
    global_audits: list[GreenfieldGlobalSolveAudit] = []
    coverage_anchor_axes = (
        sum(option.coverage_numerator for option in coverage_options),
        sum(option.j_ch for option in coverage_options),
    )
    operations_anchor_axes = (
        sum(option.coverage_numerator for option in operations_options),
        sum(option.j_ch for option in operations_options),
    )
    for index, budget in enumerate(budgets, start=1):
        master = _solve_greenfield_master(options_by_block, config, budget)
        if not master.selected_options:
            raise RuntimeError(f"portfolio master produced no incumbent at J_CH {budget}")
        point = _greenfield_point(index, budget, master, pools)
        axes = (float(master.coverage_numerator), float(master.j_ch))
        raw_points.append(
            replace(
                point,
                coverage_anchor_match=all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)
                    for left, right in zip(axes, coverage_anchor_axes, strict=True)
                ),
                operations_anchor_match=all(
                    math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)
                    for left, right in zip(axes, operations_anchor_axes, strict=True)
                ),
            )
        )
        global_audits.append(
            GreenfieldGlobalSolveAudit(
                request_index=index,
                normalized_position=(index - 1) / (len(budgets) - 1),
                schedule_exponent=global_epsilon_exponent,
                requested_epsilon_j_ch=budget,
                actual_j_ch=float(master.j_ch),
                demand_weighted_mean_coverage_days=float(
                    point.demand_weighted_mean_coverage_days
                ),
                portfolio_hash=str(point.portfolio_hash),
                status=point.status,
                termination_condition=point.termination_condition,
                primary_relative_gap=master.primary_gap,
                final_relative_gap=master.final_gap,
                proof_scope=point.proof_scope,
            )
        )
        if progress:
            progress("pareto", index, len(budgets))
    points = _nondominated_points(raw_points)
    retained_hashes = {
        point.portfolio_hash for point in points if point.portfolio_hash
    }
    marked_hashes: set[str] = set()
    marked_audits = []
    for audit in global_audits:
        retained = (
            audit.portfolio_hash in retained_hashes
            and audit.portfolio_hash not in marked_hashes
        )
        if retained:
            marked_hashes.add(audit.portfolio_hash)
        marked_audits.append(
            replace(audit, retained_nondominated=retained)
        )
    return GreenfieldFrontierResult(
        config=config,
        pools=tuple(pools),
        members=population,
        block_options=flat_options,
        points=points,
        coverage_anchor_hash=coverage_anchor_hash,
        operations_anchor_hash=operations_anchor_hash,
        elapsed_seconds=time.monotonic() - started,
        per_block_stage_seconds=per_block_stage_seconds,
        per_block_total_seconds=per_block_total_seconds,
        block_solve_audits=block_result.solve_audits,
        block_session_audits=block_result.session_audits,
        block_failures=block_result.failures,
        block_execution_mode=block_result.execution_mode,
        block_worker_count=block_result.worker_count,
        large_block_candidate_threshold=large_block_candidate_threshold,
        global_epsilon_exponent=global_epsilon_exponent,
        global_solve_audits=tuple(marked_audits),
    )
