"""Focused tests for demo scenario orchestration and portfolio trade-offs."""

from __future__ import annotations

import csv
import multiprocessing
from dataclasses import replace
from pathlib import Path

import production_wheel.scenarios as scenario_module
import pytest
from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.pareto import (
    _greenfield_proof_scope,
    _greenfield_stage_time_limits,
    _low_epsilon_budgets,
    build_greenfield_block_options,
    build_greenfield_frontier,
)
from production_wheel.schemas import (
    BaselineAcceptanceStatus,
    BaselineGuardrailPolicy,
    CoverageBasis,
    CoverageMode,
    MatrixMode,
    PalletFormula,
    PoolLimits,
    PVMode,
    RunConfig,
    VersionIdentifiers,
)
from production_wheel.scenarios import (
    CORE_SCENARIO_IDS,
    CanonicalInputs,
    assess_baseline_guardrails,
    demo_configurations,
    load_canonical_inputs,
    named_configurations,
    run_single_scenario,
    run_scenario_suite,
)


def _member(fini_id: str, baseline: str) -> CandidateMember:
    """Create one small canonical modeled member for suite tests."""

    return CandidateMember(
        fini_id=fini_id,
        plant="P1",
        sefi="S1",
        eligible_lines=frozenset({"L1"}),
        demand_litres=1_000.0,
        pallet_litres=25.0,
        package_volume=0.25,
        fixed_pv="PV1",
        baseline_group=baseline,
        pck_code="PCK1",
    )


def _inputs(tmp_path: Path) -> CanonicalInputs:
    """Create an in-memory two-FINI canonical suite input."""

    return CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(
            {"material": "F1", "model_status": "modeled"},
            {"material": "F2", "model_status": "modeled"},
            {"material": "X1", "model_status": "excluded"},
        ),
        members=(_member("F1", "B1"), _member("F2", "B2")),
        production_versions=(ProductionVersion("P1", "S1", "PV1", 100.0),),
    )


def test_demo_contract_has_19_unique_hashed_configurations() -> None:
    """The suite deduplicates default base/minimum target against the core run."""

    configs = demo_configurations()
    assert len(configs) == len({config.configuration_id() for config in configs}) == 19
    assert tuple(config.scenario_id for config in configs[:5]) == CORE_SCENARIO_IDS
    assert sum(config.coverage_mode is CoverageMode.TARGET_BAND for config in configs) == 13
    assert {
        (config.coverage_basis, config.pallet_formula)
        for config in configs
            if config.coverage_mode is CoverageMode.TARGET_BAND
            and config.group_size.effective_cap == 7
            and config.matrix_mode is MatrixMode.DIAGNOSTIC
            and config.pv_mode is PVMode.FIXED
    } >= {
        (basis, formula) for basis in CoverageBasis for formula in PalletFormula
    }
    synthetic = next(
        config
        for config in configs
        if config.scenario_id == "sensitivity_synthetic_hard_matrix"
    )
    assert synthetic.matrix_mode is MatrixMode.HARD
    assert synthetic.versions.matrix_version == "SYNTHETIC_VOLUME_MATRIX_V1"
    assert all(
        config.scenario_id != "target_matrix_diagnostic" for config in configs
    )
    customer = {
        config.scenario_id: config
        for config in configs
        if config.scenario_id.startswith("customer_families_")
    }
    assert set(customer) == {
        "customer_families_hard_target",
        "customer_families_flexible_target",
        "customer_families_hard_operations",
        "customer_families_flexible_operations",
    }
    assert {
        config.versions.matrix_version for config in customer.values()
    } == {"CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"}
    assert (
        customer["customer_families_hard_target"].structural_ruleset_fingerprint()
        == customer[
            "customer_families_hard_operations"
        ].structural_ruleset_fingerprint()
    )
    assert (
        customer[
            "customer_families_flexible_target"
        ].structural_ruleset_fingerprint()
        == customer[
            "customer_families_flexible_operations"
        ].structural_ruleset_fingerprint()
    )


def test_named_registry_adds_greenfield_and_comparison_runs_without_demo_change() -> None:
    """CLI-only greenfield anchors and baseline comparisons stay outside the demo."""

    named = named_configurations()

    assert len(demo_configurations()) == 19
    assert len(named) == 23
    assert named[
        "customer_families_greenfield_coverage"
    ].coverage_mode is CoverageMode.GREENFIELD_COVERAGE
    assert named[
        "customer_families_greenfield_operations"
    ].coverage_mode is CoverageMode.GREENFIELD_OPERATIONS
    assert named[
        "customer_families_baseline_constrained_coverage"
    ].coverage_mode is CoverageMode.BASELINE_CONSTRAINED_COVERAGE
    assert named[
        "customer_families_baseline_constrained_operations"
    ].coverage_mode is CoverageMode.BASELINE_CONSTRAINED_OPERATIONS
    assert all(
        named[scenario_id].matrix_mode is MatrixMode.FLEXIBLE
        for scenario_id in (
            "customer_families_greenfield_coverage",
            "customer_families_greenfield_operations",
            "customer_families_baseline_constrained_coverage",
            "customer_families_baseline_constrained_operations",
        )
    )


def test_greenfield_frontier_is_nondominated_and_matches_both_anchors(
    tmp_path: Path,
) -> None:
    """Block options and the portfolio master preserve both pure endpoints."""

    members = tuple(replace(item, baseline_group=None) for item in _inputs(tmp_path).members)
    config = RunConfig(
        scenario_id="greenfield-test",
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )

    frontier = build_greenfield_frontier(
        pools,
        members,
        config,
        point_count=5,
        block_option_count=5,
    )

    assert any(point.operations_anchor_match for point in frontier.points)
    assert any(point.coverage_anchor_match for point in frontier.points)
    assert all(point.proof_scope == "exact" for point in frontier.points)
    assert all(
        not (
            left.j_ch <= right.j_ch
            and left.demand_weighted_mean_coverage_days
            <= right.demand_weighted_mean_coverage_days
            and (
                left.j_ch < right.j_ch
                or left.demand_weighted_mean_coverage_days
                < right.demand_weighted_mean_coverage_days
            )
        )
        for left in frontier.points
        for right in frontier.points
        if left is not right
    )


def test_global_epsilon_schedule_is_denser_at_low_values() -> None:
    """A quadratic schedule preserves anchors and increases step size."""

    budgets = _low_epsilon_budgets(0.0, 100.0, 5, 2.0)
    uniform = _low_epsilon_budgets(0.0, 100.0, 5, 1.0)

    assert budgets == pytest.approx((0.0, 6.25, 25.0, 56.25, 100.0))
    assert uniform == pytest.approx((0.0, 25.0, 50.0, 75.0, 100.0))
    steps = [right - left for left, right in zip(budgets, budgets[1:])]
    assert steps == sorted(steps)


def test_global_solve_audit_keeps_every_requested_budget(tmp_path: Path) -> None:
    """Duplicate portfolios remain auditable while reported points deduplicate."""

    members = tuple(
        replace(item, baseline_group=None) for item in _inputs(tmp_path).members
    )
    config = RunConfig(
        scenario_id="global-epsilon-test",
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )

    frontier = build_greenfield_frontier(
        pools,
        members,
        config,
        point_count=9,
        block_option_count=5,
        global_epsilon_exponent=2.0,
        block_worker_count=1,
    )

    assert len(frontier.global_solve_audits) == 9
    budgets = [
        audit.requested_epsilon_j_ch for audit in frontier.global_solve_audits
    ]
    steps = [right - left for left, right in zip(budgets, budgets[1:])]
    assert budgets == sorted(budgets)
    assert steps[0] < steps[-1]
    assert sum(
        audit.retained_nondominated for audit in frontier.global_solve_audits
    ) == len(frontier.points)
    assert any(point.operations_anchor_match for point in frontier.points)
    assert any(point.coverage_anchor_match for point in frontier.points)


def test_adaptive_block_frontier_stops_on_duplicate_anchors() -> None:
    """A degenerate frontier builds once and skips three fixed-grid requests."""

    members = (_member("F1", "B1"),)
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )

    result = build_greenfield_block_options(
        pools,
        members,
        config,
        option_count=5,
    )

    solved = [audit for audit in result.solve_audits if audit.status != "not_solved"]
    assert len(solved) == 2
    assert {audit.request_kind for audit in solved} == {
        "operations_anchor",
        "coverage_anchor",
    }
    assert solved[-1].refinement_reason == "anchors_duplicate_stop"
    assert len(result.options_by_block[("P1", "S1")]) == 1
    assert result.session_audits[0].metrics.model_build_count == 1
    assert result.session_audits[0].metrics.solver_call_count == 4


def test_adaptive_block_frontier_refines_and_audits_every_request() -> None:
    """New partitions split intervals while duplicate endpoints stop them."""

    members = tuple(
        replace(_member(f"F{index}", f"B{index}"), demand_litres=100.0 * index)
        for index in range(1, 4)
    )
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )

    result = build_greenfield_block_options(
        pools,
        tuple(reversed(members)),
        config,
        option_count=5,
    )
    options = result.options_by_block[("P1", "S1")]
    solved = [audit for audit in result.solve_audits if audit.status != "not_solved"]

    assert len(solved) == 5
    assert solved[0].request_kind == "operations_anchor"
    assert solved[1].request_kind == "coverage_anchor"
    assert all(
        audit.requested_epsilon_j_ch is not None
        and audit.actual_j_ch is not None
        and audit.partition_hash
        and audit.heuristic_valid
        for audit in solved[2:]
    )
    assert any(
        audit.refinement_reason == "new_nondominated_partition_refine"
        for audit in solved
    )
    assert any(audit.warm_start_kind == "previous_incumbent" for audit in solved)
    assert [option.j_ch for option in options] == sorted(
        option.j_ch for option in options
    )
    assert len({option.partition_hash for option in options}) == len(options)


def test_adaptive_block_frontier_uses_fewer_requests_than_fixed_grid() -> None:
    """A duplicate midpoint ends a two-FINI frontier after three requests."""

    members = (_member("F1", "B1"), _member("F2", "B2"))
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )

    result = build_greenfield_block_options(
        pools,
        members,
        config,
        option_count=5,
    )
    solved = [audit for audit in result.solve_audits if audit.status != "not_solved"]

    assert len(solved) == 3 < 5
    assert solved[-1].refinement_reason == "duplicate_endpoint_stop"


def test_greenfield_frontier_is_identical_with_null_or_populated_baseline() -> None:
    """Historical labels change no pool, heuristic, option, or portfolio evidence."""

    labeled = tuple(_member(f"F{index}", f"OLD-{index}") for index in range(1, 4))
    blank = tuple(replace(member, baseline_group=None) for member in labeled)
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.FLEXIBLE,
        versions=VersionIdentifiers(
            matrix_version="CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
        ),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 100.0),)
    labeled_pools = generate_candidate_pools(labeled, versions, config)
    blank_pools = generate_candidate_pools(blank, versions, config)

    labeled_frontier = build_greenfield_frontier(
        labeled_pools, labeled, config, point_count=5, block_option_count=5
    )
    blank_frontier = build_greenfield_frontier(
        blank_pools, tuple(reversed(blank)), config, point_count=5, block_option_count=5
    )

    assert [pool.pool_hash for pool in labeled_pools] == [
        pool.pool_hash for pool in blank_pools
    ]
    assert [option.partition_hash for option in labeled_frontier.block_options] == [
        option.partition_hash for option in blank_frontier.block_options
    ]
    assert [point.portfolio_hash for point in labeled_frontier.points] == [
        point.portfolio_hash for point in blank_frontier.points
    ]
    assert [
        (
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.demand_weighted_mean_coverage_days,
            audit.portfolio_hash,
            audit.proof_scope,
            audit.retained_nondominated,
        )
        for audit in labeled_frontier.global_solve_audits
    ] == [
        (
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.demand_weighted_mean_coverage_days,
            audit.portfolio_hash,
            audit.proof_scope,
            audit.retained_nondominated,
        )
        for audit in blank_frontier.global_solve_audits
    ]
    assert [
        (point.demand_weighted_mean_coverage_days, point.j_ch, point.proof_scope)
        for point in labeled_frontier.points
    ] == [
        (point.demand_weighted_mean_coverage_days, point.j_ch, point.proof_scope)
        for point in blank_frontier.points
    ]
    assert [
        (
            audit.request_kind,
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.partition_hash,
            audit.refinement_reason,
            audit.heuristic_kind,
            audit.heuristic_iterations,
            audit.heuristic_moves,
            audit.heuristic_coverage_numerator,
            audit.heuristic_j_ch,
        )
        for audit in labeled_frontier.block_solve_audits
    ] == [
        (
            audit.request_kind,
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.partition_hash,
            audit.refinement_reason,
            audit.heuristic_kind,
            audit.heuristic_iterations,
            audit.heuristic_moves,
            audit.heuristic_coverage_numerator,
            audit.heuristic_j_ch,
        )
        for audit in blank_frontier.block_solve_audits
    ]


def test_sequential_and_parallel_greenfield_frontiers_are_identical() -> None:
    """Process completion order cannot change pools, options, points, or proofs."""

    members = (
        _member("A1", "A1"),
        _member("A2", "A2"),
        replace(_member("B1", "B1"), plant="P2"),
        replace(_member("B2", "B2"), plant="P2"),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    versions = (
        ProductionVersion("P1", "S1", "PV1", 100.0),
        ProductionVersion("P2", "S1", "PV1", 100.0),
    )
    pools = generate_candidate_pools(members, versions, config)

    sequential = build_greenfield_frontier(
        pools,
        members,
        config,
        block_worker_count=1,
    )
    parallel = build_greenfield_frontier(
        tuple(reversed(pools)),
        tuple(reversed(members)),
        config,
        block_worker_count=2,
    )

    expected_options = [
        (
            option.block_key,
            option.partition_hash,
            option.coverage_numerator,
            option.j_ch,
            option.proof_scope,
        )
        for option in sequential.block_options
    ]
    actual_options = [
        (
            option.block_key,
            option.partition_hash,
            option.coverage_numerator,
            option.j_ch,
            option.proof_scope,
        )
        for option in parallel.block_options
    ]
    assert actual_options == expected_options
    assert [
        (point.portfolio_hash, point.demand_weighted_mean_coverage_days, point.j_ch, point.proof_scope)
        for point in parallel.points
    ] == [
        (point.portfolio_hash, point.demand_weighted_mean_coverage_days, point.j_ch, point.proof_scope)
        for point in sequential.points
    ]
    assert [
        (
            audit.block_key,
            audit.request_kind,
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.partition_hash,
            audit.refinement_reason,
        )
        for audit in parallel.block_solve_audits
    ] == [
        (
            audit.block_key,
            audit.request_kind,
            audit.requested_epsilon_j_ch,
            audit.actual_j_ch,
            audit.partition_hash,
            audit.refinement_reason,
        )
        for audit in sequential.block_solve_audits
    ]
    if "fork" in multiprocessing.get_all_start_methods():
        assert parallel.block_execution_mode == "process_fork_shared_candidate_pools"
        assert parallel.block_worker_count == 2


def test_parallel_block_failure_preserves_successful_block_evidence() -> None:
    """One stale worker remains explicit while its successful sibling is retained."""

    members = (
        _member("A1", "A1"),
        _member("A2", "A2"),
        replace(_member("B1", "B1"), plant="P2"),
        replace(_member("B2", "B2"), plant="P2"),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    versions = (
        ProductionVersion("P1", "S1", "PV1", 100.0),
        ProductionVersion("P2", "S1", "PV1", 100.0),
    )
    valid_pools = generate_candidate_pools(members, versions, config)
    pools = (
        valid_pools[0],
        replace(valid_pools[1], config_fingerprint="stale-fixture"),
    )

    frontier = build_greenfield_frontier(
        pools,
        members,
        config,
        block_worker_count=2,
    )

    assert frontier.points == ()
    assert {option.block_key for option in frontier.block_options} == {("P1", "S1")}
    assert len(frontier.block_failures) == 1
    assert frontier.block_failures[0].block_key == ("P2", "S1")
    assert frontier.block_failures[0].error_type == "ValueError"


def test_greenfield_time_allocation_keeps_tiny_blocks_solvable() -> None:
    """The visible stage total includes a one-second floor per normal run block."""

    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    members = (_member("F1", "B1"), _member("F2", "B2"))
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )
    second = replace(pools[0], block_key=("P2", "S1"), candidates=pools[0].candidates * 10)

    limits = _greenfield_stage_time_limits((pools[0], second), 60.0)

    assert sum(limits.values()) == pytest.approx(60.0)
    assert min(limits.values()) >= 1.0


def test_greenfield_proof_labels_distinguish_restricted_and_runtime() -> None:
    """A closed restricted pool is not exact, and an open tier is runtime-limited."""

    members = (_member("F1", "B1"), _member("F2", "B2"))
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
        pool_limits=PoolLimits(
            exact_member_subsets=1,
            exact_pv_configurations=1,
            restricted_pv_configurations=10,
        ),
    )
    pools = generate_candidate_pools(
        members,
        (ProductionVersion("P1", "S1", "PV1", 100.0),),
        config,
    )
    frontier = build_greenfield_frontier(pools, members, config)

    assert pools[0].completeness == "restricted"
    assert all(point.proof_scope == "restricted-library" for point in frontier.points)
    solved = frontier.points[0].solve_result
    assert solved is not None
    assert _greenfield_proof_scope(solved, pools[0]) == "restricted-library"
    assert (
        _greenfield_proof_scope(replace(solved, status="feasible_limit"), pools[0])
        == "runtime-limited"
    )


def test_default_config_is_empirical_base_group_diagnostic() -> None:
    """Primary settings use the baseline-derived matrix as diagnostic evidence."""

    config = RunConfig()
    assert config.coverage_basis is CoverageBasis.BASE_GROUP
    assert config.pallet_formula is PalletFormula.MINIMUM_ONLY
    assert config.matrix_mode is MatrixMode.DIAGNOSTIC
    assert config.versions.matrix_version == "BASELINE_EMPIRICAL_MATRIX_V1"
    assert config.versions.ruleset_version == "PROTOTYPE_RULESET_V5"
    assert config.versions.schema_version == "PROTOTYPE_SCHEMA_V9"


def test_baseline_guardrails_classify_tradeoffs_and_synthetic_conflicts(
    tmp_path: Path,
) -> None:
    """Acceptance distinguishes KPI tradeoffs from an infeasible frozen baseline."""

    low_runners = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=(
            replace(_member("F1", "B1"), demand_litres=100.0),
            replace(_member("F2", "B2"), demand_litres=100.0),
        ),
        production_versions=(ProductionVersion("P1", "S1", "PV1", 1_000.0),),
    )
    pareto = run_single_scenario(low_runners, RunConfig())
    assert pareto.acceptance is not None
    assert pareto.acceptance.status is BaselineAcceptanceStatus.PARETO_REVIEW_REQUIRED
    assert pareto.acceptance.failed_guardrails == ("j_ch",)

    cross_volume = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=(
            _member("F1", "B1"),
            replace(_member("F2", "B1"), package_volume=5.0),
        ),
        production_versions=(ProductionVersion("P1", "S1", "PV1", 100.0),),
    )
    hard = run_single_scenario(
        cross_volume,
        RunConfig(
            scenario_id="sensitivity_synthetic_hard_matrix",
            matrix_mode=MatrixMode.HARD,
            versions=VersionIdentifiers(matrix_version="SYNTHETIC_VOLUME_MATRIX_V1"),
        ),
    )
    assert hard.acceptance is not None
    assert (
        hard.acceptance.status
        is BaselineAcceptanceStatus.BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES
    )
    assert hard.solve_result is not None
    without_incumbent = assess_baseline_guardrails(
        cross_volume,
        hard.pools,
        hard.config or RunConfig(),
        replace(hard.solve_result, has_incumbent=False),
    )
    assert (
        without_incumbent.status
        is BaselineAcceptanceStatus.BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES
    )


def test_acceptance_checks_all_three_matrix_exception_guardrails(
    tmp_path: Path,
) -> None:
    """Independent acceptance rejects newly introduced exception exposure."""

    inputs = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=(
            replace(_member("F1", "B1"), package_volume=0.25),
            replace(_member("F2", "B2"), package_volume=5.0),
        ),
        production_versions=(ProductionVersion("P1", "S1", "PV1", 100.0),),
    )
    outcome = run_single_scenario(
        inputs,
        RunConfig(
            matrix_mode=MatrixMode.FLEXIBLE,
            versions=VersionIdentifiers(
                matrix_version="CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
            ),
        ),
    )

    assert outcome.acceptance is not None
    assert {
        "matrix_exception_group_count",
        "matrix_exception_pair_count",
        "matrix_exception_distinct_volume_pair_count",
    } <= set(outcome.acceptance.failed_guardrails)
    fields = outcome.acceptance.as_dict()
    assert fields["baseline_guardrail_matrix_exception_pair_count"] == 0
    assert fields["proposal_guardrail_matrix_exception_pair_count"] == 1


def test_absolute_j_ch_tolerance_can_accept_an_otherwise_equal_profile(
    tmp_path: Path,
) -> None:
    """An explicit absolute J_CH allowance is applied to the derived baseline."""

    outcome = run_single_scenario(
        _inputs(tmp_path),
        RunConfig(
            baseline_guardrails=BaselineGuardrailPolicy(
                j_ch_absolute_tolerance=1.0
            )
        ),
    )
    assert outcome.acceptance is not None
    assert (
        outcome.acceptance.status
        is BaselineAcceptanceStatus.ACCEPTED_BASELINE_GUARDRAILS
    )


def test_fragmentation_and_majority_coverage_require_pareto_review(
    tmp_path: Path,
) -> None:
    """A lower J_CH cannot hide worse portfolio coverage or singleton growth."""

    inputs = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=(_member("F1", "B1"), _member("F2", "B1")),
        production_versions=(ProductionVersion("P1", "S1", "PV1", 100.0),),
    )

    outcome = run_single_scenario(
        inputs, RunConfig(coverage_mode=CoverageMode.OPERATIONS_FIRST)
    )

    assert outcome.acceptance is not None
    assert (
        outcome.acceptance.status
        is BaselineAcceptanceStatus.PARETO_REVIEW_REQUIRED
    )
    assert {
        "demand_weighted_mean_coverage_days",
        "p90_coverage_days",
        "group_count",
        "singleton_group_count",
    } <= set(outcome.acceptance.failed_guardrails)


def test_acceptance_rejects_solver_coefficient_tampering(tmp_path: Path) -> None:
    """Baseline governance consumes independently recomputed group evidence."""

    inputs = _inputs(tmp_path)
    outcome = run_single_scenario(inputs, RunConfig())
    assert outcome.solve_result is not None
    selected = outcome.solve_result.selected[0]
    corrupt = replace(
        outcome.solve_result,
        selected=(
            replace(
                selected,
                coefficients=replace(
                    selected.coefficients,
                    coverage_days=selected.coefficients.coverage_days + 1,
                ),
            ),
            *outcome.solve_result.selected[1:],
        ),
    )

    assessment = assess_baseline_guardrails(
        inputs, outcome.pools, outcome.config or RunConfig(), corrupt
    )

    assert assessment.status is BaselineAcceptanceStatus.NOT_ACCEPTABLE
    assert assessment.failed_guardrails == (
        "proposal_independent_validation_failed",
    )


def test_optimized_pv_assessment_keeps_the_planning_fixed_pv_baseline(
    tmp_path: Path,
) -> None:
    """PV sensitivity never overwrites the frozen baseline's planning PV."""

    members = (_member("F1", "B1"), _member("F2", "B1"))
    config = RunConfig(pv_mode=PVMode.OPTIMIZED)
    inputs = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=members,
        production_versions=(
            ProductionVersion("P1", "S1", "PV1", 100.0),
            ProductionVersion("P1", "S1", "PV2", 1_000.0),
        ),
    )
    pools = generate_candidate_pools(
        inputs.members, inputs.production_versions, config
    )
    baseline = scenario_module._baseline_result(inputs, pools, config)
    assert {item.candidate.pv_id for item in baseline.selected} == {"PV1"}

    unavailable = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(),
        members=members,
        production_versions=(ProductionVersion("P1", "S1", "PV2", 1_000.0),),
    )
    outcome = run_single_scenario(unavailable, config)
    assert outcome.acceptance is not None
    assert (
        outcome.acceptance.status
        is BaselineAcceptanceStatus.BASELINE_INFEASIBLE_UNDER_ACTIVE_RULES
    )


def test_loader_preserves_all_fini_rows_and_types_only_modeled(tmp_path: Path) -> None:
    """CSV ingestion keeps excluded rows visible without sending them to optimization."""

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    fini_fields = [
        "plant",
        "material",
        "sefi",
        "eligible_lines",
        "forecast_litres_12m",
        "pallet_litres_resolved",
        "package_volume",
        "fixed_pv",
        "baseline_group_key",
        "pck_code",
        "model_status",
        "optimized_pv_model_status",
    ]
    fini_rows = [
        {
            "plant": "P1",
            "material": "F1",
            "sefi": "S1",
            "eligible_lines": "1|2",
            "forecast_litres_12m": "1000",
            "pallet_litres_resolved": "50",
            "package_volume": "0.25",
            "fixed_pv": "PV1",
            "baseline_group_key": "B1",
            "pck_code": "PCK1",
            "model_status": "modeled",
            "optimized_pv_model_status": "modeled",
        },
        {
            "plant": "P1",
            "material": "X1",
            "sefi": "S1",
            "eligible_lines": "1",
            "forecast_litres_12m": "20",
            "pallet_litres_resolved": "50",
            "package_volume": "0.25",
            "fixed_pv": "",
            "baseline_group_key": "",
            "pck_code": "PCK1",
            "model_status": "excluded",
            "optimized_pv_model_status": "modeled",
        },
    ]
    with (extracted / "fini_master.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fini_fields)
        writer.writeheader()
        writer.writerows(fini_rows)
    with (extracted / "production_versions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["plant", "sefi", "production_version", "lot_size_litres"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "plant": "P1",
                "sefi": "S1",
                "production_version": "PV1",
                "lot_size_litres": "100",
            }
        )

    loaded = load_canonical_inputs(tmp_path)
    assert tuple(row["material"] for row in loaded.fini_rows) == ("F1", "X1")
    assert tuple(member.fini_id for member in loaded.members) == ("F1",)
    assert loaded.members[0].eligible_lines == frozenset({"1", "2"})
    optimized = RunConfig(pv_mode=PVMode.OPTIMIZED)
    assert tuple(member.fini_id for member in loaded.members_for(optimized)) == (
        "F1",
        "X1",
    )


def test_demo_suite_reuses_eight_structural_pool_families_and_returns_25_results(
    tmp_path: Path,
) -> None:
    """All default outcomes and disclosed portfolio points are stable and complete."""

    progress: list[tuple[str, int, int]] = []
    result = run_scenario_suite(_inputs(tmp_path), progress=lambda *args: progress.append(args))
    assert len(result.scenarios) == 19
    assert len(result.pareto_points) == 5
    assert len(result.outcomes) == 25
    assert len(result.pool_cache) == 8
    assert result.baseline.result_class == "frozen-recalculated-baseline"
    assert all(outcome.solve_result is not None for outcome in result.scenarios)
    assert all(point.frontier_label.startswith("frontier_over_disclosed") for point in result.pareto_points)
    assert progress[-1] == ("pareto", 5, 5)


def test_one_scenario_failure_does_not_discard_later_results(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A failed scenario remains explicit while independent work continues."""

    configs = demo_configurations()[:2]
    original = scenario_module.solve_decomposed_pools

    def fail_first(*args: object, **kwargs: object) -> object:
        """Raise only for MAX and delegate every later scenario normally."""

        config = args[2]
        if config.coverage_mode is CoverageMode.MAX:  # type: ignore[union-attr]
            raise RuntimeError("deliberate fixture failure")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(scenario_module, "solve_decomposed_pools", fail_first)  # type: ignore[attr-defined]
    result = run_scenario_suite(_inputs(tmp_path), configs=configs)
    assert result.scenarios[0].status == "failed"
    assert result.scenarios[0].error_type == "RuntimeError"
    assert result.scenarios[1].solve_result is not None
    assert result.partial
