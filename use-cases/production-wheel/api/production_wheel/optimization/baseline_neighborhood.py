"""Evaluate scalable one-block substitutions around the frozen baseline."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from production_wheel.candidates import CandidateMember, CandidatePool
from production_wheel.schemas import CoverageMode, RunConfig

from .solver import (
    BaselineConstraintContext,
    BaselineConstraintLimits,
    ObjectiveLevel,
    SelectedCandidate,
    SolveResult,
    _BASELINE_CONSTRAINED_MODES,
    _BASELINE_TOLERANCE,
    _baseline_constraint_limits,
    _baseline_warm_start_indexes,
    _block_solve_evidence,
    _decomposed_stage_time_limits,
    _eligible_candidates,
    _infeasible_result,
    _member_key,
    _normalized_exception_volume_pairs,
    precompute_coefficients,
    solve_candidate_pools,
)


def _weighted_coverage(selected: Sequence[SelectedCandidate]) -> float:
    """Return demand-weighted mean coverage for a complete selected portfolio."""

    demand = sum(item.coefficients.group_demand_litres for item in selected)
    return sum(
        item.coefficients.coverage_days * item.coefficients.group_demand_litres
        for item in selected
    ) / demand


def _passes_baseline_limits(
    selected: Sequence[SelectedCandidate], limits: BaselineConstraintLimits
) -> bool:
    """Independently recompute and check all eleven portfolio guardrails."""

    coefficients = [item.coefficients for item in selected]
    candidates = [item.candidate for item in selected]
    coverages = sorted(item.coverage_days for item in coefficients)
    if not coverages:
        return False
    p90 = coverages[max(math.ceil(0.9 * len(coverages)) - 1, 0)]
    distinct_pairs = {
        pair
        for candidate in candidates
        for pair in _normalized_exception_volume_pairs(candidate)
    }

    def no_more(value: float, limit: float) -> bool:
        """Compare one floating KPI with the solver's baseline tolerance."""

        return value <= limit + _BASELINE_TOLERANCE

    return all(
        (
            sum(item.target_violation for item in coefficients)
            <= limits.target_violation_count,
            no_more(
                max(item.target_excess_days for item in coefficients),
                limits.target_worst_excess_days,
            ),
            no_more(
                sum(item.target_excess_days for item in coefficients),
                limits.target_total_excess_days,
            ),
            no_more(
                _weighted_coverage(selected),
                limits.demand_weighted_mean_coverage_days,
            ),
            no_more(p90, limits.p90_coverage_days),
            len(candidates) <= limits.group_count,
            sum(len(candidate.member_ids) == 1 for candidate in candidates)
            <= limits.singleton_group_count,
            no_more(sum(item.j_ch for item in coefficients), limits.j_ch_limit),
            sum(bool(candidate.matrix_exception_pairs) for candidate in candidates)
            <= limits.matrix_exception_group_count,
            sum(len(candidate.matrix_exception_pairs) for candidate in candidates)
            <= limits.matrix_exception_pair_count,
            len(distinct_pairs) <= limits.matrix_exception_distinct_volume_pair_count,
        )
    )


def _objective_key(
    selected: Sequence[SelectedCandidate], mode: CoverageMode
) -> tuple[float, ...]:
    """Return the governed lexicographic key for one feasible portfolio."""

    weighted = _weighted_coverage(selected)
    if mode is CoverageMode.BASELINE_CONSTRAINED_OPERATIONS:
        return sum(item.coefficients.j_ch for item in selected), weighted
    return (weighted,)


def _portfolio_levels(
    selected: Sequence[SelectedCandidate], mode: CoverageMode
) -> tuple[ObjectiveLevel, ...]:
    """Report portfolio objectives without claiming a global optimization proof."""

    key = _objective_key(selected, mode)
    names = (
        ("j_ch", "demand_weighted_mean_coverage_days")
        if mode is CoverageMode.BASELINE_CONSTRAINED_OPERATIONS
        else ("demand_weighted_mean_coverage_days",)
    )
    return tuple(
        ObjectiveLevel(name, value, "baseline_neighbourhood_witness")
        for name, value in zip(names, key, strict=True)
    )


def _constraint_context(
    baseline: Sequence[SelectedCandidate],
    held_out_block: tuple[str, str],
    limits: BaselineConstraintLimits,
) -> BaselineConstraintContext:
    """Return full-limit residual context for one locally replaced block."""

    outside = [
        item for item in baseline if item.candidate.block_key != held_out_block
    ]
    coefficients = [item.coefficients for item in outside]
    candidates = [item.candidate for item in outside]
    return BaselineConstraintContext(
        limits=limits,
        outside_target_violation_count=sum(
            item.target_violation for item in coefficients
        ),
        outside_target_total_excess_days=sum(
            item.target_excess_days for item in coefficients
        ),
        outside_weighted_coverage_numerator=sum(
            item.coverage_days * item.group_demand_litres for item in coefficients
        ),
        total_demand_litres=sum(
            item.coefficients.group_demand_litres for item in baseline
        ),
        outside_p90_exceedance_count=sum(
            item.coverage_days > limits.p90_coverage_days + _BASELINE_TOLERANCE
            for item in coefficients
        ),
        outside_group_count=len(candidates),
        outside_singleton_group_count=sum(
            len(candidate.member_ids) == 1 for candidate in candidates
        ),
        outside_j_ch=sum(item.j_ch for item in coefficients),
        outside_matrix_exception_group_count=sum(
            bool(candidate.matrix_exception_pairs) for candidate in candidates
        ),
        outside_matrix_exception_pair_count=sum(
            len(candidate.matrix_exception_pairs) for candidate in candidates
        ),
        outside_distinct_volume_pairs=frozenset(
            pair
            for candidate in candidates
            for pair in _normalized_exception_volume_pairs(candidate)
        ),
    )


def solve_baseline_neighborhood(
    pools: Sequence[CandidatePool],
    members: Iterable[CandidateMember],
    config: RunConfig,
) -> SolveResult:
    """Choose the best valid one-block substitution around the full baseline.

    Each block is solved with the existing constrained master and a deterministic
    proportional share of the scenario-wide stage budget. Its result replaces
    only that block in the frozen partition; all eleven portfolio limits are
    then recomputed before the trial can compete with the baseline fallback.
    """

    if config.coverage_mode not in _BASELINE_CONSTRAINED_MODES:
        raise ValueError("baseline neighborhood requires a constrained coverage mode")
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

    ordered_pools = tuple(sorted(pools, key=lambda item: item.block_key))
    ordered_members = tuple(sorted(members, key=_member_key))
    candidates = _eligible_candidates(ordered_pools, config)
    coefficients = precompute_coefficients(candidates, ordered_members, config)
    baseline_indexes = _baseline_warm_start_indexes(candidates, ordered_members)
    completeness = (
        "complete"
        if all(pool.completeness == "complete" for pool in ordered_pools)
        else "restricted"
    )
    if not baseline_indexes:
        return _infeasible_result(
            completeness=completeness,
            candidate_count=len(candidates),
            status="baseline_infeasible",
            termination="not_solved",
            baseline_constraint_scope="one_block_neighbourhood",
        )

    baseline = tuple(
        SelectedCandidate(candidates[index], coefficients[index])
        for index in sorted(baseline_indexes)
    )
    limits = _baseline_constraint_limits(
        candidates, coefficients, baseline_indexes, config
    )
    by_block = {
        pool.block_key: tuple(
            item for item in baseline if item.candidate.block_key == pool.block_key
        )
        for pool in ordered_pools
    }
    allocations = _decomposed_stage_time_limits(
        ordered_pools, config.solver_limits.time_limit_seconds
    )
    block_results: list[SolveResult] = []
    trials: list[tuple[tuple[str, str] | None, tuple[SelectedCandidate, ...]]] = [
        (None, baseline)
    ]
    for pool in ordered_pools:
        block_config = config.model_copy(
            update={
                "solver_limits": config.solver_limits.model_copy(
                    update={
                        "time_limit_seconds": allocations[pool.block_key]
                    }
                )
            }
        )
        block_members = tuple(
            member for member in ordered_members if member.block_key == pool.block_key
        )
        result = solve_candidate_pools(
            (pool,),
            block_members,
            block_config,
            baseline_constraint_context=_constraint_context(
                baseline, pool.block_key, limits
            ),
        )
        block_results.append(result)
        if not result.has_incumbent:
            continue
        trial = tuple(
            item
            for other_pool in ordered_pools
            for item in (
                result.selected
                if other_pool.block_key == pool.block_key
                else by_block[other_pool.block_key]
            )
        )
        if _passes_baseline_limits(trial, limits):
            trials.append((pool.block_key, trial))

    _, (changed_block, selected) = min(
        enumerate(trials),
        key=lambda item: (_objective_key(item[1][1], config.coverage_mode), item[0]),
    )
    levels = _portfolio_levels(selected, config.coverage_mode)
    search_complete = len(block_results) == len(ordered_pools) and all(
        result.status == "optimal" for result in block_results
    )
    if changed_block is not None:
        status = "feasible_witness"
        termination = "baseline_neighbourhood_witness"
        result_class = "baseline-neighbourhood-witness"
    elif search_complete:
        status = "baseline_nondominated_in_evaluated_neighbourhood"
        termination = "baseline_neighbourhood_proved_no_improvement"
        result_class = "baseline-neighbourhood-nondominated"
    else:
        status = "baseline_no_better_witness_found"
        termination = "baseline_neighbourhood_search_incomplete"
        result_class = "baseline-neighbourhood-inconclusive"
    return SolveResult(
        status=status,
        termination_condition=termination,
        has_incumbent=True,
        result_class=result_class,
        selected=selected,
        objective_levels=levels,
        primary_objective=levels[0].value,
        primary_best_bound=None,
        primary_relative_gap=None,
        incumbent_objective=levels[-1].value,
        best_bound=None,
        relative_gap=None,
        candidate_count=len(candidates),
        pool_completeness=completeness,
        solver_method="pyomo_appsi_highs_baseline_neighbourhood",
        block_evidence=_block_solve_evidence(ordered_pools, block_results),
        baseline_constraint_limits=limits,
        baseline_constraint_scope="one_block_neighbourhood",
        warm_start_kind="baseline",
        warm_start_group_count=len(baseline),
    )
