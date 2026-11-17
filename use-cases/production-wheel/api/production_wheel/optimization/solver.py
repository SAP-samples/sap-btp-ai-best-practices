"""Solve deterministic candidate libraries with Pyomo APPSI and HiGHS.

The model is a binary set partitioning master: every modeled FINI is covered
by exactly one selected candidate. Examples::

    from production_wheel.optimization import solve_candidate_pools
    result = solve_candidate_pools(pools, members, config)
    assert result.has_incumbent

Candidate generation owns structural feasibility. This module defensively
rechecks the active cap and HARD matrix mode, precomputes every KPI coefficient,
then applies the configured business objectives as exact lexicographic solves.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Iterable, Sequence

import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from production_wheel.candidates import Candidate, CandidateMember, CandidatePool
from production_wheel.metrics import (
    changeover_contribution,
    group_coverage,
    group_frequency,
    pallet_allocations,
    proportional_allocations,
)
from production_wheel.matrix import normalize_volume
from production_wheel.schemas import CoverageMode, MatrixMode, RunConfig
from production_wheel.business_rules import linear_rows

_LOCK_TOLERANCE = 1e-7
_PROOF_GAP_TOLERANCE = 1e-9
_DINKELBACH_TOLERANCE = 1e-8
_DINKELBACH_LIMIT = 100
_BASELINE_TOLERANCE = 1e-9
_BASELINE_CONSTRAINED_MODES = {
    CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
    CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
}


@dataclass(frozen=True, slots=True)
class BaselineConstraintLimits:
    """Frozen same-settings limits compiled into a constrained candidate master."""

    target_violation_count: int
    target_worst_excess_days: float
    target_total_excess_days: float
    demand_weighted_mean_coverage_days: float
    p90_coverage_days: float
    group_count: int
    singleton_group_count: int
    j_ch_limit: float
    matrix_exception_group_count: int
    matrix_exception_pair_count: int
    matrix_exception_distinct_volume_pair_count: int

    def as_dict(self) -> dict[str, object]:
        """Return stable field names and values for audit artifacts."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BaselineConstraintContext:
    """Full limits and frozen contributions outside one locally solved block."""

    limits: BaselineConstraintLimits
    outside_target_violation_count: int
    outside_target_total_excess_days: float
    outside_weighted_coverage_numerator: float
    total_demand_litres: float
    outside_p90_exceedance_count: int
    outside_group_count: int
    outside_singleton_group_count: int
    outside_j_ch: float
    outside_matrix_exception_group_count: int
    outside_matrix_exception_pair_count: int
    outside_distinct_volume_pairs: frozenset[tuple[Decimal, Decimal]]


@dataclass(frozen=True, slots=True)
class CandidateCoefficients:
    """Linear KPI coefficients for one candidate under one run configuration.

    Args:
        coverage_days: Selected coverage interpretation in days.
        group_demand_litres: Canonical demand represented by the group.
        j_ch: Count-based recurring changeover contribution.
        target_violation: One when coverage is outside the target band.
        target_excess_days: Distance to the nearest target-band boundary.
        matrix_exceptions: Number of incompatible FINI pairs in the candidate.
        relaxed_group: One when the candidate exceeds the historical cap seven.
        size_excess: Number of members above the historical cap seven.
        stable_rank: Canonical deterministic rank used only after business tiers.
    """

    coverage_days: float
    group_demand_litres: float
    j_ch: float
    target_violation: int
    target_excess_days: float
    matrix_exceptions: int
    relaxed_group: int
    size_excess: int
    stable_rank: int


@dataclass(frozen=True, slots=True)
class HeuristicEvidence:
    """Audit evidence for one deterministic baseline-free incumbent.

    Args:
        kind: Stable heuristic identifier.
        iterations: Candidate moves inspected by the bounded construction.
        moves: Feasible improving moves applied.
        construction_seconds: Wall time spent constructing and validating the start.
        coverage_numerator: Demand-weighted coverage numerator of the start.
        demand_weighted_mean_coverage_days: Coverage numerator divided by block demand.
        j_ch: Manufacturing-proxy value of the start.
        exact_cover_valid: Whether the selected indexes cover every FINI exactly once.
        within_epsilon: Whether the start respects the active ``J_CH`` epsilon.
        solver_improved: Whether HiGHS improved the active primary objective.
    """

    kind: str
    iterations: int
    moves: int
    construction_seconds: float
    coverage_numerator: float
    demand_weighted_mean_coverage_days: float
    j_ch: float
    exact_cover_valid: bool
    within_epsilon: bool
    solver_improved: bool | None = None


@dataclass(frozen=True, slots=True)
class BlockSessionMetrics:
    """Cumulative structural and solver timings for one persistent block model."""

    candidate_order_seconds: float
    coefficient_preparation_seconds: float
    model_construction_seconds: float
    solver_transfer_seconds: float
    solver_update_seconds: float
    solver_optimize_seconds: float
    model_build_count: int
    solver_call_count: int
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class ObjectiveLevel:
    """One solved lexicographic level and its solver evidence."""

    name: str
    value: float
    termination_condition: str = "optimal"
    best_bound: float | None = None
    relative_gap: float | None = None
    wallclock_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    """A selected candidate paired with independently precomputed coefficients."""

    candidate: Candidate
    coefficients: CandidateCoefficients


@dataclass(frozen=True, slots=True)
class BlockSolveEvidence:
    """Persistable objective-tier evidence for one decomposed block solve."""

    block_key: tuple[str, str]
    status: str
    termination_condition: str
    result_class: str
    objective_levels: tuple[ObjectiveLevel, ...]
    primary_objective: float | None
    primary_best_bound: float | None
    primary_relative_gap: float | None
    relative_gap: float | None
    candidate_count: int


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Solver outcome, proof metadata, and selected exact-cover groups.

    ``result_class`` is ``exact`` only for an optimal complete library and
    ``pool-optimal`` for an optimal result containing any restricted block.
    Non-optimal incumbents are classified from their within-pool MIP gap.
    """

    status: str
    termination_condition: str
    has_incumbent: bool
    result_class: str
    selected: tuple[SelectedCandidate, ...]
    objective_levels: tuple[ObjectiveLevel, ...]
    primary_objective: float | None
    primary_best_bound: float | None
    primary_relative_gap: float | None
    # These three fields describe the final solved lexicographic tier.
    incumbent_objective: float | None
    best_bound: float | None
    relative_gap: float | None
    candidate_count: int
    pool_completeness: str
    solver_method: str = "pyomo_appsi_highs"
    block_evidence: tuple[BlockSolveEvidence, ...] = ()
    baseline_constraint_limits: BaselineConstraintLimits | None = None
    baseline_constraint_scope: str = "none"
    warm_start_kind: str = "none"
    warm_start_group_count: int = 0
    epsilon_j_ch: float | None = None
    heuristic_evidence: HeuristicEvidence | None = None
    session_metrics: BlockSessionMetrics | None = None


@dataclass(slots=True)
class _SolveSnapshot:
    """Internal result of one lexicographic HiGHS call."""

    termination: TerminationCondition
    incumbent: float | None
    bound: float | None
    gap: float | None
    primals: dict[int, float]
    wallclock_seconds: float | None

    @property
    def feasible(self) -> bool:
        """Return whether HiGHS supplied a loadable incumbent."""

        return self.incumbent is not None

    @property
    def optimal(self) -> bool:
        """Return whether HiGHS closed the current objective level."""

        return (
            self.termination is TerminationCondition.optimal
            and self.gap is not None
            and self.gap <= _PROOF_GAP_TOLERANCE
        )

    @property
    def tier_termination(self) -> str:
        """Return an honest tier status when HiGHS accepts a configured MIP gap."""

        if self.termination is TerminationCondition.optimal and not self.optimal:
            return "gap_accepted"
        return self.termination.name


def _member_key(member: CandidateMember) -> tuple[str, str, str]:
    """Return the globally unambiguous plant-SEFI-FINI key."""

    return member.plant, member.sefi, member.fini_id


def _candidate_member_keys(candidate: Candidate) -> tuple[tuple[str, str, str], ...]:
    """Return exact-cover keys represented by one candidate."""

    plant, sefi = candidate.block_key
    return tuple((plant, sefi, fini_id) for fini_id in candidate.member_ids)


def _eligible_candidates(
    pools: Sequence[CandidatePool], config: RunConfig
) -> tuple[Candidate, ...]:
    """Return sorted candidates respecting active solver-side safeguards."""

    candidates = {
        candidate.candidate_hash: candidate
        for pool in pools
        for candidate in pool.candidates
        if len(candidate.member_ids) <= config.cap_for_block(*candidate.block_key)
        and not (
            config.matrix_mode is MatrixMode.HARD
            and candidate.matrix_exception_pairs
        )
    }
    return tuple(
        sorted(
            candidates.values(),
            key=lambda item: (
                item.block_key,
                item.member_ids,
                item.pv_id,
                item.nominal_lot_litres,
                item.candidate_hash,
            ),
        )
    )


def precompute_coefficients(
    candidates: Sequence[Candidate],
    members: Iterable[CandidateMember],
    config: RunConfig,
) -> tuple[CandidateCoefficients, ...]:
    """Precompute coverage, target-band, changeover, and tie-break coefficients.

    Args:
        candidates: Candidate configurations in canonical solver order.
        members: Canonical member primitives for every candidate FINI.
        config: Coverage basis, pallet formula, target band, and calendar settings.

    Returns:
        Coefficients aligned one-to-one with ``candidates``.

    Raises:
        ValueError: A candidate references an unknown FINI or inconsistent demand.
    """

    by_key = {_member_key(member): member for member in members}
    if len(by_key) == 0:
        raise ValueError("members cannot be empty")
    result: list[CandidateCoefficients] = []
    for rank, candidate in enumerate(candidates, start=1):
        selected_members: list[CandidateMember] = []
        for key in _candidate_member_keys(candidate):
            if key not in by_key:
                raise ValueError(f"candidate references unknown FINI: {key}")
            selected_members.append(by_key[key])
        demands = {
            member.fini_id: Decimal(str(member.demand_litres))
            for member in selected_members
        }
        nominal = proportional_allocations(demands, candidate.effective_batch_litres)
        adjusted = pallet_allocations(
            nominal,
            {
                member.fini_id: Decimal(str(member.pallet_litres))
                for member in selected_members
            },
            config.pallet_formula,
        )
        coverage = float(
            group_coverage(
                config.coverage_basis,
                demands,
                candidate.effective_batch_litres,
                adjusted,
                config.demand_days,
            )
        )
        group_demand = sum(float(value) for value in demands.values())
        if not math.isclose(
            group_demand,
            candidate.group_demand_litres,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"candidate demand does not match member primitives: {candidate.candidate_hash}"
            )
        frequency = group_frequency(
            group_demand, candidate.effective_batch_litres, config.productive_weeks
        )
        excess = max(
            config.target_band.lower_days - coverage,
            coverage - config.target_band.upper_days,
            0.0,
        )
        result.append(
            CandidateCoefficients(
                coverage_days=coverage,
                group_demand_litres=group_demand,
                j_ch=float(changeover_contribution(frequency, len(candidate.member_ids))),
                target_violation=int(excess > 0),
                target_excess_days=excess,
                matrix_exceptions=len(candidate.matrix_exception_pairs),
                relaxed_group=int(len(candidate.member_ids) > 7),
                size_excess=max(len(candidate.member_ids) - 7, 0),
                stable_rank=rank,
            )
        )
    return tuple(result)


def _relative_gap(incumbent: float | None, bound: float | None) -> float | None:
    """Return a non-negative relative minimization gap when both values exist."""

    if incumbent is None or bound is None or not all(
        math.isfinite(value) for value in (incumbent, bound)
    ):
        return None
    return abs(incumbent - bound) / max(abs(incumbent), 1e-10)


def _configure_solver(config: RunConfig) -> Highs:
    """Create a deterministic APPSI HiGHS solver with solution auto-load disabled."""

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
        "presolve_reduction_limit": config.solver_limits.presolve_reduction_limit,
    }
    return solver


def _baseline_warm_start_indexes(
    candidates: Sequence[Candidate], members: Sequence[CandidateMember]
) -> frozenset[int]:
    """Return a complete baseline candidate cover when it exists in the pool."""

    baseline_groups: dict[tuple[str, str, str], list[CandidateMember]] = {}
    for member in members:
        if not member.baseline_group:
            return frozenset()
        baseline_groups.setdefault(
            (member.plant, member.sefi, member.baseline_group), []
        ).append(member)
    candidate_index = {
        (candidate.block_key, candidate.member_ids, candidate.pv_id): index
        for index, candidate in enumerate(candidates)
    }
    selected: set[int] = set()
    for (plant, sefi, _), group_members in baseline_groups.items():
        aliases = {member.fixed_pv for member in group_members}
        if len(aliases) != 1 or None in aliases or "" in aliases:
            return frozenset()
        key = (
            (plant, sefi),
            tuple(sorted(member.fini_id for member in group_members)),
            str(next(iter(aliases))),
        )
        if key not in candidate_index:
            return frozenset()
        selected.add(candidate_index[key])
    covered = {
        key
        for index in selected
        for key in _candidate_member_keys(candidates[index])
    }
    expected = {_member_key(member) for member in members}
    return frozenset(selected) if covered == expected else frozenset()


def _singleton_warm_start_indexes(
    candidates: Sequence[Candidate], members: Sequence[CandidateMember]
) -> frozenset[int]:
    """Return the first deterministic singleton exact cover when available."""

    first_by_member: dict[tuple[str, str, str], int] = {}
    for index, candidate in enumerate(candidates):
        if len(candidate.member_ids) != 1:
            continue
        key = (*candidate.block_key, candidate.member_ids[0])
        first_by_member.setdefault(key, index)
    expected = {_member_key(member) for member in members}
    if set(first_by_member) != expected:
        return frozenset()
    return frozenset(first_by_member[key] for key in sorted(expected))


def _normalized_exception_volume_pairs(
    candidate: Candidate,
) -> frozenset[tuple[Decimal, Decimal]]:
    """Return stable unordered package-volume pairs represented by one candidate."""

    return frozenset(
        tuple(sorted((normalize_volume(volume_a), normalize_volume(volume_b))))
        for _, _, volume_a, volume_b in candidate.matrix_exception_pairs
    )


def _baseline_constraint_limits(
    candidates: Sequence[Candidate],
    coefficients: Sequence[CandidateCoefficients],
    baseline_indexes: frozenset[int],
    config: RunConfig,
) -> BaselineConstraintLimits:
    """Derive all eleven hard limits from the frozen same-settings partition."""

    baseline_coefficients = [coefficients[index] for index in sorted(baseline_indexes)]
    baseline_candidates = [candidates[index] for index in sorted(baseline_indexes)]
    coverages = sorted(item.coverage_days for item in baseline_coefficients)
    group_count = len(baseline_candidates)
    p90_index = max(math.ceil(0.9 * group_count) - 1, 0)
    total_demand = sum(item.group_demand_litres for item in baseline_coefficients)
    weighted = sum(
        item.coverage_days * item.group_demand_litres
        for item in baseline_coefficients
    ) / total_demand
    distinct_pairs = {
        pair
        for candidate in baseline_candidates
        for pair in _normalized_exception_volume_pairs(candidate)
    }
    baseline_j_ch = sum(item.j_ch for item in baseline_coefficients)
    return BaselineConstraintLimits(
        target_violation_count=sum(item.target_violation for item in baseline_coefficients),
        target_worst_excess_days=max(
            (item.target_excess_days for item in baseline_coefficients), default=0.0
        ),
        target_total_excess_days=sum(
            item.target_excess_days for item in baseline_coefficients
        ),
        demand_weighted_mean_coverage_days=weighted,
        p90_coverage_days=coverages[p90_index],
        group_count=group_count,
        singleton_group_count=sum(
            len(candidate.member_ids) == 1 for candidate in baseline_candidates
        ),
        j_ch_limit=config.baseline_guardrails.j_ch_limit(baseline_j_ch),
        matrix_exception_group_count=sum(
            bool(candidate.matrix_exception_pairs) for candidate in baseline_candidates
        ),
        matrix_exception_pair_count=sum(
            len(candidate.matrix_exception_pairs) for candidate in baseline_candidates
        ),
        matrix_exception_distinct_volume_pair_count=len(distinct_pairs),
    )


def _solve_once(model: pyo.ConcreteModel, solver: Highs) -> _SolveSnapshot:
    """Solve one objective level and read primals only after an incumbent check."""

    results = solver.solve(model)
    incumbent = results.best_feasible_objective
    bound = results.best_objective_bound
    finite_incumbent = (
        float(incumbent)
        if incumbent is not None and math.isfinite(float(incumbent))
        else None
    )
    finite_bound = (
        float(bound)
        if bound is not None and math.isfinite(float(bound))
        else None
    )
    primals: dict[int, float] = {}
    if finite_incumbent is not None:
        results.solution_loader.load_vars()
        primals = {index: float(pyo.value(model.x[index])) for index in model.C}
    return _SolveSnapshot(
        termination=results.termination_condition,
        incumbent=finite_incumbent,
        bound=finite_bound,
        gap=_relative_gap(finite_incumbent, finite_bound),
        primals=primals,
        wallclock_seconds=(
            None
            if results.wallclock_time is None
            else float(results.wallclock_time)
        ),
    )


def _linear(coefficients: Sequence[float | int], model: pyo.ConcreteModel) -> object:
    """Return a linear candidate-selection expression."""

    return pyo.quicksum(coefficients[index] * model.x[index] for index in model.C)


def _maximum(
    coefficients: Sequence[float | int], model: pyo.ConcreteModel
) -> pyo.Var:
    """Create an epigraph variable equal to the maximum selected coefficient."""

    maximum = model.auxiliary_variables.add()
    started = [
        float(coefficients[index])
        for index in model.C
        if model.x[index].value is not None and model.x[index].value > 0.5
    ]
    if started:
        maximum.value = max(started)
    for index in model.C:
        model.auxiliary_constraints.add(maximum >= coefficients[index] * model.x[index])
    return maximum


def _lock(model: pyo.ConcreteModel, expression: object, optimum: float) -> None:
    """Lock a proven lexicographic optimum with a scale-aware numeric tolerance."""

    tolerance = _LOCK_TOLERANCE * max(1.0, abs(optimum))
    model.lexicographic_locks.add(expression <= optimum + tolerance)


def _result_class(
    completeness: str,
    optimal: bool,
    gap: float | None,
) -> str:
    """Classify exact, pool-optimal, bounded, exploratory, or heuristic results."""

    if optimal:
        return "exact" if completeness == "complete" else "pool-optimal"
    if gap is None or gap > 0.05:
        return "heuristic"
    if gap <= 0.01:
        return "bounded-near-optimal"
    return "exploratory"


def _infeasible_result(
    *,
    completeness: str,
    candidate_count: int,
    status: str,
    termination: str,
    baseline_constraint_limits: BaselineConstraintLimits | None = None,
    baseline_constraint_scope: str = "none",
    warm_start_kind: str = "none",
    warm_start_group_count: int = 0,
    epsilon_j_ch: float | None = None,
) -> SolveResult:
    """Build a no-incumbent result with its feasibility and warm-start evidence."""

    if status in {"uncovered_precheck", "infeasible"}:
        result_class = (
            "modeled_infeasible" if completeness == "complete" else "pool_infeasible"
        )
    elif status == "no_incumbent":
        result_class = "no-incumbent-within-limit"
    elif status == "baseline_infeasible":
        result_class = "baseline-infeasible-under-active-rules"
    else:
        result_class = "solver-error"
    return SolveResult(
        status=status,
        termination_condition=termination,
        has_incumbent=False,
        result_class=result_class,
        selected=(),
        objective_levels=(),
        primary_objective=None,
        primary_best_bound=None,
        primary_relative_gap=None,
        incumbent_objective=None,
        best_bound=None,
        relative_gap=None,
        candidate_count=candidate_count,
        pool_completeness=completeness,
        baseline_constraint_limits=baseline_constraint_limits,
        baseline_constraint_scope=baseline_constraint_scope,
        warm_start_kind=warm_start_kind,
        warm_start_group_count=warm_start_group_count,
        epsilon_j_ch=epsilon_j_ch,
    )


def _business_tiers(
    config: RunConfig,
    coefficients: Sequence[CandidateCoefficients],
    model: pyo.ConcreteModel,
) -> list[tuple[str, object, float]]:
    """Return named business objectives and their report-value scale factors."""

    coverages = [item.coverage_days for item in coefficients]
    demands = [item.group_demand_litres for item in coefficients]
    demand_numerator = _linear(
        [coverage * demand for coverage, demand in zip(coverages, demands, strict=True)],
        model,
    )
    tiers: list[tuple[str, object, float]] = []
    if (
        config.coverage_mode not in _BASELINE_CONSTRAINED_MODES
        and config.matrix_mode is MatrixMode.DIAGNOSTIC
        and any(item.matrix_exceptions for item in coefficients)
    ):
        tiers.append(
            (
                "matrix_exception_pairs",
                _linear([item.matrix_exceptions for item in coefficients], model),
                1.0,
            )
        )
    mode = config.coverage_mode
    if mode is CoverageMode.MAX:
        tiers.append(("maximum_coverage_days", _maximum(coverages, model), 1.0))
    elif mode is CoverageMode.DEMAND_WEIGHTED_MEAN:
        tiers.append(("demand_weighted_coverage_numerator", demand_numerator, 1.0))
    elif mode in (CoverageMode.TARGET_BAND, CoverageMode.OPERATIONS_FIRST):
        violations = [item.target_violation for item in coefficients]
        excesses = [item.target_excess_days for item in coefficients]
        tiers.extend(
            [
                ("target_violation_count", _linear(violations, model), 1.0),
                ("target_worst_excess_days", _maximum(excesses, model), 1.0),
                ("target_total_excess_days", _linear(excesses, model), 1.0),
            ]
        )
        if mode is CoverageMode.OPERATIONS_FIRST:
            tiers.append(
                ("j_ch", _linear([item.j_ch for item in coefficients], model), 1.0)
            )
        tiers.append(("demand_weighted_coverage_numerator", demand_numerator, 1.0))
    elif mode is CoverageMode.BASELINE_CONSTRAINED_COVERAGE:
        tiers.append(("demand_weighted_coverage_numerator", demand_numerator, 1.0))
    elif mode is CoverageMode.BASELINE_CONSTRAINED_OPERATIONS:
        tiers.extend(
            [
                ("j_ch", _linear([item.j_ch for item in coefficients], model), 1.0),
                ("demand_weighted_coverage_numerator", demand_numerator, 1.0),
            ]
        )
    elif mode in {CoverageMode.GREENFIELD_COVERAGE, CoverageMode.PARETO}:
        tiers.extend(
            [
                ("demand_weighted_coverage_numerator", demand_numerator, 1.0),
                ("j_ch", _linear([item.j_ch for item in coefficients], model), 1.0),
            ]
        )
    elif mode is CoverageMode.GREENFIELD_OPERATIONS:
        tiers.extend(
            [
                ("j_ch", _linear([item.j_ch for item in coefficients], model), 1.0),
                ("demand_weighted_coverage_numerator", demand_numerator, 1.0),
            ]
        )
    return tiers




def solve_candidate_pools(
    pools: Sequence[CandidatePool],
    members: Iterable[CandidateMember],
    config: RunConfig,
    *,
    baseline_constraint_context: BaselineConstraintContext | None = None,
    j_ch_epsilon: float | None = None,
    preferred_line_minimum: float | None = None,
) -> SolveResult:
    """Solve one global exact-cover master over one or more block libraries.

    The global master is intentionally compact and guarantees that portfolio
    maximums and the GROUP_MEAN fractional objective are coordinated across
    blocks. Candidate pools remain independently generated and auditable.

    Args:
        pools: Complete or restricted candidate libraries by plant-SEFI block.
        members: Modeled FINIs that must each be covered exactly once.
        config: Validated objective, KPI, matrix, and solver settings.
        baseline_constraint_context: Optional full-portfolio residual budgets
            when this master represents one block in a neighbourhood trial.
        j_ch_epsilon: Visible upper bound used only by ``PARETO`` coverage solves.

    Returns:
        Selected groups plus solver incumbent, bound, gap, and result class.
    """

    if config.coverage_mode is CoverageMode.PARETO:
        if j_ch_epsilon is None or not math.isfinite(j_ch_epsilon) or j_ch_epsilon < 0:
            raise ValueError("PARETO mode requires a finite non-negative J_CH epsilon")
    elif j_ch_epsilon is not None:
        raise ValueError("J_CH epsilon is valid only in PARETO mode")

    expected_fingerprint = config.structural_ruleset_fingerprint()
    stale = [
        pool.block_key
        for pool in pools
        if pool.config_fingerprint != expected_fingerprint
    ]
    if stale:
        raise ValueError(
            "candidate pools were generated under different structural settings: "
            f"{stale}"
        )
    ordered_members = tuple(sorted(members, key=_member_key))
    member_keys = tuple(_member_key(member) for member in ordered_members)
    if len(set(member_keys)) != len(member_keys):
        raise ValueError("modeled FINI keys must be unique")
    completeness = (
        "complete" if all(pool.completeness == "complete" for pool in pools) else "restricted"
    )
    candidates = _eligible_candidates(pools, config)
    coefficients = precompute_coefficients(candidates, ordered_members, config) if candidates else ()
    constrained = config.coverage_mode in _BASELINE_CONSTRAINED_MODES
    baseline_start = (
        _baseline_warm_start_indexes(candidates, ordered_members)
        if constrained
        else frozenset()
    )
    if baseline_constraint_context is not None and not constrained:
        raise ValueError(
            "baseline constraint context requires a constrained coverage mode"
        )
    constraint_scope = (
        "one_block_neighbourhood"
        if baseline_constraint_context is not None
        else "global_master"
    )
    if constrained and not baseline_start:
        return _infeasible_result(
            completeness=completeness,
            candidate_count=len(candidates),
            status="baseline_infeasible",
            termination="not_solved",
            baseline_constraint_scope=constraint_scope,
            epsilon_j_ch=j_ch_epsilon,
        )
    baseline_limits = (
        baseline_constraint_context.limits
        if baseline_constraint_context is not None
        else _baseline_constraint_limits(candidates, coefficients, baseline_start, config)
        if constrained
        else None
    )
    incidence_lists: dict[tuple[str, str, str], list[int]] = {
        key: [] for key in member_keys
    }
    for index, candidate in enumerate(candidates):
        for key in _candidate_member_keys(candidate):
            incidence_lists[key].append(index)
    incidence = {key: tuple(indexes) for key, indexes in incidence_lists.items()}
    if not member_keys or any(not indexes for indexes in incidence.values()):
        return _infeasible_result(
            completeness=completeness,
            candidate_count=len(candidates),
            status="uncovered_precheck",
            termination="not_solved",
            epsilon_j_ch=j_ch_epsilon,
        )

    model = pyo.ConcreteModel(name="candidate_set_partitioning")
    model.C = pyo.RangeSet(0, len(candidates) - 1)
    model.x = pyo.Var(model.C, domain=pyo.Binary)
    singleton_start = (
        frozenset()
        if baseline_start
        else _singleton_warm_start_indexes(candidates, ordered_members)
    )
    warm_start = baseline_start or singleton_start
    warm_start_kind = (
        "baseline" if baseline_start else "singleton" if singleton_start else "none"
    )
    if warm_start:
        for index in model.C:
            model.x[index].value = 1.0 if index in warm_start else 0.0
    model.cover = pyo.ConstraintList()
    for key in member_keys:
        model.cover.add(pyo.quicksum(model.x[index] for index in incidence[key]) == 1)
    # Every row sees the same joint exact-cover model, including plant totals
    # that couple otherwise independent SEFI blocks.
    model.business_rules = pyo.ConstraintList()
    for row in linear_rows(candidates, ordered_members, config):
        values = [row["coefficients"].get(c.candidate_hash, 0.0) for c in candidates]
        expression = _linear(values, model)
        for bound, lower in ((row["lower"], True), (row["upper"], False)):
            if bound is None:
                continue
            if not any(values):
                valid = 0 >= bound if lower else 0 <= bound
                model.business_rules.add(pyo.Constraint.Feasible if valid else pyo.Constraint.Infeasible)
            else:
                model.business_rules.add(expression >= bound if lower else expression <= bound)
    preferred_values = [c.group_demand_litres if c.selected_line == config.preferred_line else 0.0 for c in candidates]
    if preferred_line_minimum is not None:
        model.business_rules.add(_linear(preferred_values, model) >= preferred_line_minimum - 1e-7)
    model.auxiliary_variables = pyo.VarList(domain=pyo.NonNegativeReals)
    model.auxiliary_constraints = pyo.ConstraintList()
    model.lexicographic_locks = pyo.ConstraintList()
    model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    if j_ch_epsilon is not None:
        tolerance = _LOCK_TOLERANCE * max(1.0, abs(j_ch_epsilon))
        model.pareto_epsilon = pyo.Constraint(
            expr=_linear([item.j_ch for item in coefficients], model)
            <= j_ch_epsilon + tolerance
        )
    if baseline_limits is not None:
        outside = baseline_constraint_context
        model.baseline_guardrails = pyo.ConstraintList()
        model.baseline_guardrails.add(
            _linear([item.target_violation for item in coefficients], model)
            <= baseline_limits.target_violation_count
            - (outside.outside_target_violation_count if outside else 0)
        )
        for index, item in enumerate(coefficients):
            if (
                item.target_excess_days
                > baseline_limits.target_worst_excess_days + _BASELINE_TOLERANCE
            ):
                model.x[index].fix(0.0)
        model.baseline_guardrails.add(
            _linear([item.target_excess_days for item in coefficients], model)
            <= baseline_limits.target_total_excess_days + _BASELINE_TOLERANCE
            - (outside.outside_target_total_excess_days if outside else 0.0)
        )
        total_modeled_demand = (
            outside.total_demand_litres
            if outside
            else sum(member.demand_litres for member in ordered_members)
        )
        model.baseline_guardrails.add(
            _linear(
                [
                    item.coverage_days * item.group_demand_litres
                    for item in coefficients
                ],
                model,
            )
            <= (
                baseline_limits.demand_weighted_mean_coverage_days
                + _BASELINE_TOLERANCE
            )
            * total_modeled_demand
            - (outside.outside_weighted_coverage_numerator if outside else 0.0)
        )
        group_count_expression = _linear([1 for _ in coefficients], model)
        p90_exceedances = _linear(
            [
                int(
                    item.coverage_days
                    > baseline_limits.p90_coverage_days + _BASELINE_TOLERANCE
                )
                for item in coefficients
            ],
            model,
        )
        outside_p90 = outside.outside_p90_exceedance_count if outside else 0
        outside_groups = outside.outside_group_count if outside else 0
        model.baseline_guardrails.add(
            10 * (outside_p90 + p90_exceedances)
            <= outside_groups + group_count_expression
        )
        model.baseline_guardrails.add(
            group_count_expression
            <= baseline_limits.group_count - outside_groups
        )
        model.baseline_guardrails.add(
            _linear(
                [int(len(candidate.member_ids) == 1) for candidate in candidates], model
            )
            <= baseline_limits.singleton_group_count
            - (outside.outside_singleton_group_count if outside else 0)
        )
        model.baseline_guardrails.add(
            _linear([item.j_ch for item in coefficients], model)
            <= baseline_limits.j_ch_limit + _BASELINE_TOLERANCE
            - (outside.outside_j_ch if outside else 0.0)
        )
        model.baseline_guardrails.add(
            _linear(
                [int(bool(candidate.matrix_exception_pairs)) for candidate in candidates],
                model,
            )
            <= baseline_limits.matrix_exception_group_count
            - (outside.outside_matrix_exception_group_count if outside else 0)
        )
        model.baseline_guardrails.add(
            _linear(
                [len(candidate.matrix_exception_pairs) for candidate in candidates], model
            )
            <= baseline_limits.matrix_exception_pair_count
            - (outside.outside_matrix_exception_pair_count if outside else 0)
        )
        outside_pairs = outside.outside_distinct_volume_pairs if outside else frozenset()
        pair_incidence: dict[tuple[Decimal, Decimal], list[int]] = {}
        for index, candidate in enumerate(candidates):
            for pair in _normalized_exception_volume_pairs(candidate):
                if pair not in outside_pairs:
                    pair_incidence.setdefault(pair, []).append(index)
        pair_keys = tuple(sorted(pair_incidence))
        model.exception_pair_used = pyo.Var(range(len(pair_keys)), domain=pyo.Binary)
        for pair_index, pair in enumerate(pair_keys):
            indexes = pair_incidence[pair]
            occurrences = pyo.quicksum(model.x[index] for index in indexes)
            big_m = min(len(indexes), baseline_limits.group_count)
            model.baseline_guardrails.add(
                occurrences <= big_m * model.exception_pair_used[pair_index]
            )
            if warm_start:
                model.exception_pair_used[pair_index].value = float(
                    any(index in warm_start for index in indexes)
                )
        if pair_keys:
            model.baseline_guardrails.add(
                pyo.quicksum(
                    model.exception_pair_used[index]
                    for index in range(len(pair_keys))
                )
                <= baseline_limits.matrix_exception_distinct_volume_pair_count
                - len(outside_pairs)
            )
    solver = _configure_solver(config)
    levels: list[ObjectiveLevel] = []
    last: _SolveSnapshot | None = None
    matrix_tier_not_required = False

    def solve_and_lock(name: str, expression: object, display_divisor: float = 1.0) -> bool:
        """Solve, record, and lock one level; stop cleanly on no proof/incumbent."""

        nonlocal last
        model.objective.set_value(expression)
        last = _solve_once(model, solver)
        if not last.feasible:
            return False
        optimum = float(last.incumbent)
        levels.append(
            ObjectiveLevel(
                name,
                optimum / display_divisor,
                last.tier_termination,
                (
                    None
                    if last.bound is None
                    else last.bound / display_divisor
                ),
                last.gap,
                last.wallclock_seconds,
            )
        )
        if not last.optimal:
            return False
        _lock(model, expression, optimum)
        return True

    if config.preferred_line:
        solve_and_lock("preferred_line_demand_litres", _linear([-v for v in preferred_values], model), -1.0)
    if (last is None or last.optimal) and (
        config.coverage_mode not in _BASELINE_CONSTRAINED_MODES
        and config.matrix_mode is MatrixMode.DIAGNOSTIC
    ):
        matrix_values = [item.matrix_exceptions for item in coefficients]
        if any(matrix_values):
            if not solve_and_lock(
                "matrix_exception_pairs",
                _linear(matrix_values, model),
            ):
                pass
        else:
            matrix_tier_not_required = True

    stopped = last is not None and not last.optimal
    if not stopped and config.coverage_mode is CoverageMode.GROUP_MEAN:
        numerator = _linear([item.coverage_days for item in coefficients], model)
        denominator = _linear([1 for _ in coefficients], model)
        ratio = 0.0
        for _ in range(_DINKELBACH_LIMIT):
            model.objective.set_value(numerator - ratio * denominator)
            last = _solve_once(model, solver)
            if not last.feasible or not last.optimal:
                stopped = True
                break
            selected_indexes = [index for index, value in last.primals.items() if value > 0.5]
            group_count = len(selected_indexes)
            current_numerator = sum(coefficients[index].coverage_days for index in selected_indexes)
            next_ratio = current_numerator / group_count
            residual = current_numerator - ratio * group_count
            ratio = next_ratio
            if abs(residual) <= _DINKELBACH_TOLERANCE:
                levels.append(
                    ObjectiveLevel(
                        "group_mean_coverage_days",
                        ratio,
                        last.tier_termination,
                        last.bound,
                        last.gap,
                        last.wallclock_seconds,
                    )
                )
                _lock(model, numerator - ratio * denominator, 0.0)
                break
        else:
            raise RuntimeError("GROUP_MEAN Dinkelbach iteration did not converge")
    elif not stopped:
        total_modeled_demand = sum(member.demand_litres for member in ordered_members)
        for name, expression, divisor in _business_tiers(config, coefficients, model):
            if name == "matrix_exception_pairs" and config.matrix_mode is MatrixMode.DIAGNOSTIC:
                continue
            if name == "demand_weighted_coverage_numerator":
                divisor = total_modeled_demand
                name = "demand_weighted_mean_coverage_days"
            if not solve_and_lock(name, expression, divisor):
                stopped = True
                break

    if not stopped:
        post_business = (
            ("relaxed_group_count", [item.relaxed_group for item in coefficients]),
            ("group_size_total_excess", [item.size_excess for item in coefficients]),
        )
        for name, values in post_business:
            if not any(values):
                levels.append(ObjectiveLevel(name, 0.0, "not_required"))
                continue
            if not solve_and_lock(name, _linear(values, model)):
                stopped = True
                break

    if matrix_tier_not_required:
        levels.append(ObjectiveLevel("matrix_exception_pairs", 0.0, "not_required"))

    if last is None or not last.feasible:
        termination = "unknown" if last is None else last.termination.name
        return _infeasible_result(
            completeness=completeness,
            candidate_count=len(candidates),
            status=(
                "infeasible"
                if last is not None
                and last.termination
                in {TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded}
                else "no_incumbent"
            ),
            termination=termination,
            baseline_constraint_limits=baseline_limits,
            baseline_constraint_scope=(constraint_scope if constrained else "none"),
            warm_start_kind=warm_start_kind,
            warm_start_group_count=len(warm_start),
            epsilon_j_ch=j_ch_epsilon,
        )

    selected_indexes = tuple(
        index for index, value in sorted(last.primals.items()) if value > 0.5
    )
    selected = tuple(
        SelectedCandidate(candidates[index], coefficients[index])
        for index in selected_indexes
    )
    # A stable-rank MILP tie-break is disproportionately expensive on the real
    # snapshot. Deterministic candidate ordering and HiGHS options already make
    # repeated runs reproducible, so record the selected rank without another
    # branch-and-bound pass.
    levels.append(
        ObjectiveLevel(
            "stable_candidate_rank_diagnostic",
            float(sum(coefficients[index].stable_rank for index in selected_indexes)),
            "deterministic_selection",
        )
    )
    optimal = last.optimal and not stopped
    primary = levels[0] if levels else None
    return SolveResult(
        status="optimal" if optimal else "feasible_limit",
        termination_condition=last.tier_termination,
        has_incumbent=True,
        result_class=_result_class(
            completeness,
            optimal,
            None
            if config.coverage_mode is CoverageMode.GROUP_MEAN and not optimal
            else last.gap,
        ),
        selected=selected,
        objective_levels=tuple(levels),
        primary_objective=primary.value if primary else None,
        primary_best_bound=primary.best_bound if primary else None,
        primary_relative_gap=primary.relative_gap if primary else None,
        incumbent_objective=last.incumbent,
        best_bound=last.bound,
        relative_gap=0.0 if optimal else last.gap,
        candidate_count=len(candidates),
        pool_completeness=completeness,
        baseline_constraint_limits=baseline_limits,
        baseline_constraint_scope=(constraint_scope if constrained else "none"),
        warm_start_kind=warm_start_kind,
        warm_start_group_count=len(warm_start),
        epsilon_j_ch=j_ch_epsilon,
    )


def solve_block_pool(
    pool: CandidatePool,
    members: Iterable[CandidateMember],
    config: RunConfig,
) -> SolveResult:
    """Solve one block through the same global-coordinator implementation."""

    block_members = tuple(member for member in members if member.block_key == pool.block_key)
    return solve_candidate_pools((pool,), block_members, config)


def _combined_tier_termination(
    name: str, block_results: Sequence[SolveResult]
) -> str:
    """Summarize whether one lexicographic tier was reached across all blocks."""

    observed = [
        next((level for level in result.objective_levels if level.name == name), None)
        for result in block_results
    ]
    reached = [level for level in observed if level is not None]
    if not reached:
        return "not_reached"
    if len(reached) != len(block_results):
        return "partially_reached"
    terminations = {level.termination_condition for level in reached}
    if terminations == {"not_required"}:
        return "not_required"
    proved = {"optimal", "not_required", "deterministic_selection"}
    if terminations <= proved:
        return "decomposed_optimal"
    return "decomposed_feasible_limit"


def _combined_levels(
    selected: Sequence[SelectedCandidate],
    config: RunConfig,
    block_results: Sequence[SolveResult],
) -> tuple[ObjectiveLevel, ...]:
    """Recompute KPI values while retaining block-level tier reachability."""

    coefficients = [item.coefficients for item in selected]
    coverage = [item.coverage_days for item in coefficients]
    demand = [item.group_demand_litres for item in coefficients]
    total_demand = sum(demand)
    weighted = (
        sum(value * weight for value, weight in zip(coverage, demand, strict=True))
        / total_demand
    )
    levels: list[ObjectiveLevel] = []
    matrix_tier_required = any(
        level.name == "matrix_exception_pairs"
        and level.termination_condition != "not_required"
        for result in block_results
        for level in result.objective_levels
    )

    def add(name: str, value: float) -> None:
        """Append a portfolio KPI with honest decomposed reach/proof status."""

        levels.append(
            ObjectiveLevel(
                name,
                value,
                _combined_tier_termination(name, block_results),
            )
        )

    if config.matrix_mode is MatrixMode.DIAGNOSTIC and matrix_tier_required:
        add(
            "matrix_exception_pairs",
            float(sum(item.matrix_exceptions for item in coefficients)),
        )
    mode = config.coverage_mode
    if mode is CoverageMode.MAX:
        add("maximum_coverage_days", max(coverage))
    elif mode is CoverageMode.DEMAND_WEIGHTED_MEAN:
        add("demand_weighted_mean_coverage_days", weighted)
    elif mode is CoverageMode.GROUP_MEAN:
        add("group_mean_coverage_days", sum(coverage) / len(coverage))
    elif mode in (CoverageMode.TARGET_BAND, CoverageMode.OPERATIONS_FIRST):
        add(
            "target_violation_count",
            float(sum(item.target_violation for item in coefficients)),
        )
        add(
            "target_worst_excess_days",
            max(item.target_excess_days for item in coefficients),
        )
        add(
            "target_total_excess_days",
            sum(item.target_excess_days for item in coefficients),
        )
        if mode is CoverageMode.OPERATIONS_FIRST:
            add("j_ch", sum(item.j_ch for item in coefficients))
        add("demand_weighted_mean_coverage_days", weighted)
    elif mode in {CoverageMode.GREENFIELD_COVERAGE, CoverageMode.PARETO}:
        add("demand_weighted_mean_coverage_days", weighted)
        add("j_ch", sum(item.j_ch for item in coefficients))
    elif mode is CoverageMode.GREENFIELD_OPERATIONS:
        add("j_ch", sum(item.j_ch for item in coefficients))
        add("demand_weighted_mean_coverage_days", weighted)
    add(
        "relaxed_group_count",
        float(sum(item.relaxed_group for item in coefficients)),
    )
    add(
        "group_size_total_excess",
        float(sum(item.size_excess for item in coefficients)),
    )
    add(
        "stable_candidate_rank_diagnostic",
        float(sum(item.stable_rank for item in coefficients)),
    )
    if config.matrix_mode is MatrixMode.DIAGNOSTIC and not matrix_tier_required:
        add("matrix_exception_pairs", 0.0)
    return tuple(levels)


def _decomposed_stage_time_limits(
    pools: Sequence[CandidatePool], total_seconds: float
) -> dict[tuple[str, str], float]:
    """Allocate one scenario-wide stage budget across decomposed blocks.

    Args:
        pools: Candidate libraries to solve independently.
        total_seconds: Positive budget for one lexicographic stage across all
            blocks.

    Returns:
        Deterministic positive block budgets proportional to candidate count.
        Empty libraries retain unit weight so their precheck receives a valid
        solver-limit value without inflating the total budget.
    """

    if total_seconds <= 0:
        raise ValueError("decomposed stage time budget must be positive")
    if not pools:
        return {}
    weights = {pool.block_key: max(len(pool.candidates), 1) for pool in pools}
    total_weight = sum(weights.values())
    return {
        block_key: total_seconds * weight / total_weight
        for block_key, weight in weights.items()
    }


def _block_solve_evidence(
    pools: Sequence[CandidatePool], results: Sequence[SolveResult]
) -> tuple[BlockSolveEvidence, ...]:
    """Convert completed decomposed results into persistable block evidence."""

    return tuple(
        BlockSolveEvidence(
            block_key=pool.block_key,
            status=result.status,
            termination_condition=result.termination_condition,
            result_class=result.result_class,
            objective_levels=result.objective_levels,
            primary_objective=result.primary_objective,
            primary_best_bound=result.primary_best_bound,
            primary_relative_gap=result.primary_relative_gap,
            relative_gap=result.relative_gap,
            candidate_count=result.candidate_count,
        )
        for pool, result in zip(pools, results, strict=True)
    )


def _combined_primary_gap(results: Sequence[SolveResult]) -> float | None:
    """Return a conservative aggregate gap only when every block has a bound."""

    gaps = [result.primary_relative_gap for result in results]
    if not gaps or any(gap is None for gap in gaps):
        return None
    return max(float(gap) for gap in gaps)


def solve_decomposed_pools(
    pools: Sequence[CandidatePool],
    members: Iterable[CandidateMember],
    config: RunConfig,
) -> SolveResult:
    """Solve blocks independently for scalable prototype scenarios.

    The exact global master remains the semantic oracle. The decomposed path is
    used for the full demonstrator when the objective is block-separable. MAX
    and target worst-excess are conservatively minimized inside every block;
    this preserves their global primary optimum while possibly over-optimizing
    non-binding blocks during later tie levels. GROUP_MEAN remains global because
    its endogenous denominator couples blocks. Baseline-constrained modes use a
    one-block neighbourhood over multiple pools and retain the global master as
    the one-pool semantic oracle. The configured per-stage time limit is a
    scenario-wide budget: blocks receive deterministic shares proportional to
    their candidate-library size, rather than each receiving the full limit.
    """

    if config.business_rules or config.preferred_line:
        return solve_candidate_pools(pools, members, config)
    if config.coverage_mode in _BASELINE_CONSTRAINED_MODES and len(pools) > 1:
        from .baseline_neighborhood import solve_baseline_neighborhood

        return solve_baseline_neighborhood(pools, members, config)
    if config.coverage_mode in {
        CoverageMode.GROUP_MEAN,
        CoverageMode.PARETO,
        *_BASELINE_CONSTRAINED_MODES,
    }:
        return solve_candidate_pools(pools, members, config)
    ordered_members = tuple(members)
    results: list[SolveResult] = []
    ordered_pools = tuple(sorted(pools, key=lambda item: item.block_key))
    block_time_limits = _decomposed_stage_time_limits(
        ordered_pools, config.solver_limits.time_limit_seconds
    )
    for pool in ordered_pools:
        block_config = config.model_copy(
            update={
                "solver_limits": config.solver_limits.model_copy(
                    update={
                        "time_limit_seconds": block_time_limits[pool.block_key]
                    }
                )
            }
        )
        result = solve_block_pool(pool, ordered_members, block_config)
        results.append(result)
        if not result.has_incumbent:
            warm_start_kinds = {item.warm_start_kind for item in results}
            return SolveResult(
                status=result.status,
                termination_condition=result.termination_condition,
                has_incumbent=False,
                result_class=result.result_class,
                selected=(),
                objective_levels=(),
                primary_objective=None,
                primary_best_bound=None,
                primary_relative_gap=None,
                incumbent_objective=None,
                best_bound=None,
                relative_gap=result.relative_gap,
                candidate_count=sum(len(item.candidates) for item in pools),
                pool_completeness=(
                    "complete"
                    if all(item.completeness == "complete" for item in pools)
                    else "restricted"
                ),
                solver_method="pyomo_appsi_highs_block_decomposed",
                block_evidence=_block_solve_evidence(
                    ordered_pools[: len(results)], results
                ),
                warm_start_kind=(
                    next(iter(warm_start_kinds))
                    if len(warm_start_kinds) == 1
                    else "mixed"
                ),
                warm_start_group_count=sum(
                    item.warm_start_group_count for item in results
                ),
            )
    selected = tuple(
        item
        for result in results
        for item in result.selected
    )
    all_optimal = all(result.status == "optimal" for result in results)
    completeness = (
        "complete" if all(pool.completeness == "complete" for pool in pools) else "restricted"
    )
    gaps = [result.relative_gap for result in results if result.relative_gap is not None]
    combined_gap = max(gaps, default=None)
    if all_optimal and config.coverage_mode is CoverageMode.DEMAND_WEIGHTED_MEAN:
        result_class = _result_class(completeness, True, 0.0)
    elif all_optimal:
        result_class = (
            "decomposed-primary-exact"
            if completeness == "complete"
            else "decomposed-pool-primary-optimal"
        )
    else:
        result_class = _result_class(completeness, False, combined_gap)
    levels = _combined_levels(selected, config, results)
    primary = levels[0] if levels else None
    combined_primary_gap = _combined_primary_gap(results)
    reached_business_levels = [
        level
        for level in levels
        if level.name
        not in {"stable_candidate_rank_diagnostic", "matrix_exception_pairs"}
        and level.termination_condition not in {"not_reached", "not_required"}
    ]
    warm_start_kinds = {result.warm_start_kind for result in results}
    return SolveResult(
        status="optimal" if all_optimal else "feasible_limit",
        termination_condition=(
            "decomposed_optimal" if all_optimal else "decomposed_feasible_limit"
        ),
        has_incumbent=True,
        result_class=result_class,
        selected=selected,
        objective_levels=levels,
        primary_objective=primary.value if primary else None,
        primary_best_bound=primary.value if primary and all_optimal else None,
        primary_relative_gap=(
            0.0 if primary and all_optimal else combined_primary_gap
        ),
        incumbent_objective=(
            reached_business_levels[-1].value if reached_business_levels else None
        ),
        best_bound=None,
        relative_gap=0.0 if all_optimal else combined_gap,
        candidate_count=sum(result.candidate_count for result in results),
        pool_completeness=completeness,
        solver_method="pyomo_appsi_highs_block_decomposed",
        block_evidence=_block_solve_evidence(ordered_pools, results),
        warm_start_kind=(
            next(iter(warm_start_kinds))
            if len(warm_start_kinds) == 1
            else "mixed"
        ),
        warm_start_group_count=sum(
            result.warm_start_group_count for result in results
        ),
    )
