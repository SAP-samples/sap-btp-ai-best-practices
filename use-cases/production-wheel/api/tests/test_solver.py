"""Focused parity and coordinator tests for the Pyomo/HiGHS set-partition model."""

from __future__ import annotations

from dataclasses import replace

import pytest
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition

from production_wheel.candidates import (
    BlockProjection,
    Candidate,
    CandidateMember,
    CandidatePool,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.optimization import (
    BlockSolverSession,
    ObjectiveLevel,
    precompute_coefficients,
    solve_block_pool,
    solve_candidate_pools,
    solve_decomposed_pools,
)
from production_wheel.optimization.solver import (
    _baseline_warm_start_indexes,
    _combined_levels,
    _combined_primary_gap,
    _decomposed_stage_time_limits,
    _infeasible_result,
    _configure_solver,
    _solve_once,
    _SolveSnapshot,
)
from production_wheel.schemas import (
    BaselineGuardrailPolicy,
    CoverageBasis,
    CoverageMode,
    MatrixMode,
    PalletFormula,
    RunConfig,
    TargetBand,
    VersionIdentifiers,
)


def member(
    fini_id: str,
    *,
    plant: str = "P1",
    sefi: str = "S1",
    demand: float = 100.0,
    pallet: float = 1.0,
    volume: float = 0.25,
    baseline_group: str | None = None,
) -> CandidateMember:
    """Create one compact fixed-PV member for solver fixtures."""

    return CandidateMember(
        fini_id=fini_id,
        plant=plant,
        sefi=sefi,
        eligible_lines=frozenset({"L1"}),
        demand_litres=demand,
        pallet_litres=pallet,
        package_volume=volume,
        fixed_pv="PV1",
        fixed_lot_litres=100.0,
        baseline_group=baseline_group,
    )


def versions(*blocks: tuple[str, str]) -> tuple[ProductionVersion, ...]:
    """Create the shared fixed production version for each requested block."""

    return tuple(ProductionVersion(plant, sefi, "PV1", 100.0) for plant, sefi in blocks)


def manual_candidate(
    block: tuple[str, str],
    member_ids: tuple[str, ...],
    effective_batch: float,
    demand: float,
    identity: str,
    matrix_exception_pairs: tuple[tuple[str, str, float, float], ...] = (),
) -> Candidate:
    """Create a structurally valid candidate with controlled coverage."""

    return Candidate(
        block_key=block,
        member_ids=member_ids,
        pv_id="PV1",
        nominal_lot_litres=effective_batch / 0.9,
        effective_batch_litres=effective_batch,
        common_lines=("L1",),
        group_demand_litres=demand,
        matrix_exception_pairs=matrix_exception_pairs,
        equal_pck=False,
        source_tiers=("exact",),
        membership_hash=f"membership-{identity}",
        candidate_hash=f"candidate-{identity}",
    )


def manual_pool(
    block: tuple[str, str],
    candidates: tuple[Candidate, ...],
    *,
    completeness: str = "complete",
    config: RunConfig | None = None,
) -> CandidatePool:
    """Wrap controlled candidates in the production pool evidence record."""

    method = "exact" if completeness == "complete" else "restricted"
    projection = BlockProjection(block, 1, 7, len(candidates), 1, len(candidates), method)
    return CandidatePool(
        block_key=block,
        method=method,
        completeness=completeness,
        structural_fingerprint="fixture",
        projection=projection,
        candidates=candidates,
        mandatory_member_sets=0,
        mandatory_pv_configurations=0,
        size_traces=(),
        pool_hash=f"pool-{block}-{completeness}",
        config_fingerprint=(config or RunConfig()).structural_ruleset_fingerprint(),
    )


def test_exact_cover_and_max_match_direct_partition_choice() -> None:
    """MAX selects the one feasible pair instead of two higher-coverage singletons."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.MAX,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(members, versions(("P1", "S1")), config)
    result = solve_candidate_pools(pools, members, config)
    assert result.status == "optimal"
    assert result.result_class == "exact"
    assert result.warm_start_kind == "singleton"
    assert result.warm_start_group_count == 2
    assert [item.candidate.member_ids for item in result.selected] == [("F1", "F2")]
    assert result.objective_levels[0].value == pytest.approx(112.5)
    covered = [fini for item in result.selected for fini in item.candidate.member_ids]
    assert sorted(covered) == ["F1", "F2"]


def test_greenfield_all_null_baseline_uses_singleton_incumbent_and_exact_cover() -> None:
    """Blank historical groups cannot prevent a greenfield solve from starting."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.GREENFIELD_COVERAGE,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(members, versions(("P1", "S1")), config)

    result = solve_candidate_pools(pools, members, config)

    assert result.status == "optimal"
    assert result.warm_start_kind == "singleton"
    assert sorted(
        fini_id
        for selected in result.selected
        for fini_id in selected.candidate.member_ids
    ) == ["F1", "F2"]


def test_greenfield_solution_is_not_constrained_by_baseline_presence() -> None:
    """Historical labels change neither a greenfield start nor its result."""

    blank = (member("F1"), member("F2"))
    labeled = (
        member("F1", baseline_group="B1"),
        member("F2", baseline_group="B2"),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.GREENFIELD_COVERAGE,
        matrix_mode=MatrixMode.OFF,
    )
    blank_pools = generate_candidate_pools(blank, versions(("P1", "S1")), config)
    labeled_pools = generate_candidate_pools(
        labeled, versions(("P1", "S1")), config
    )

    without_baseline = solve_candidate_pools(blank_pools, blank, config)
    with_baseline = solve_candidate_pools(labeled_pools, labeled, config)

    assert [pool.pool_hash for pool in blank_pools] == [
        pool.pool_hash for pool in labeled_pools
    ]
    assert [item.candidate.candidate_hash for item in without_baseline.selected] == [
        item.candidate.candidate_hash for item in with_baseline.selected
    ]
    assert without_baseline.primary_objective == with_baseline.primary_objective
    assert without_baseline.warm_start_kind == with_baseline.warm_start_kind == "singleton"


def test_greenfield_anchors_and_epsilon_keep_coverage_and_jch_separate() -> None:
    """Pure anchors expose the expected partition trade-off without a weighted score."""

    members = (member("F1"), member("F2"))
    coverage_config = RunConfig(
        coverage_mode=CoverageMode.GREENFIELD_COVERAGE,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members, versions(("P1", "S1")), coverage_config
    )
    operations_config = coverage_config.model_copy(
        update={"coverage_mode": CoverageMode.GREENFIELD_OPERATIONS}
    )
    pareto_config = coverage_config.model_copy(
        update={"coverage_mode": CoverageMode.PARETO}
    )

    coverage = solve_candidate_pools(pools, members, coverage_config)
    operations = solve_candidate_pools(pools, members, operations_config)
    epsilon_low = solve_candidate_pools(
        pools, members, pareto_config, j_ch_epsilon=0.0
    )
    epsilon_high = solve_candidate_pools(
        pools, members, pareto_config, j_ch_epsilon=0.05
    )

    assert [item.candidate.member_ids for item in coverage.selected] == [("F1", "F2")]
    assert {item.candidate.member_ids for item in operations.selected} == {
        ("F1",),
        ("F2",),
    }
    assert {item.candidate.member_ids for item in epsilon_low.selected} == {
        ("F1",),
        ("F2",),
    }
    assert [item.candidate.member_ids for item in epsilon_high.selected] == [
        ("F1", "F2")
    ]
    assert epsilon_low.epsilon_j_ch == 0.0


def test_persistent_block_session_matches_one_shot_greenfield_solves() -> None:
    """One transferred block model preserves the existing one-shot semantics."""

    members = (member("F1"), member("F2"), member("F3"))
    base = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pool = generate_candidate_pools(
        members, versions(("P1", "S1")), base
    )[0]
    session = BlockSolverSession(pool, members, base)
    requests = (
        (CoverageMode.GREENFIELD_OPERATIONS, None),
        (CoverageMode.GREENFIELD_COVERAGE, None),
        (CoverageMode.PARETO, 0.0),
        (CoverageMode.PARETO, 0.05),
    )

    for mode, epsilon in requests:
        config = base.model_copy(update={"coverage_mode": mode})
        expected = solve_candidate_pools(
            (pool,), members, config, j_ch_epsilon=epsilon
        )
        actual = session.solve(mode, j_ch_epsilon=epsilon)
        assert actual.status == expected.status
        assert actual.result_class == expected.result_class
        assert [item.candidate.candidate_hash for item in actual.selected] == [
            item.candidate.candidate_hash for item in expected.selected
        ]
        assert [item.name for item in actual.objective_levels] == [
            item.name for item in expected.objective_levels
        ]
        assert [item.value for item in actual.objective_levels] == pytest.approx(
            [item.value for item in expected.objective_levels]
        )

    assert session.metrics.model_build_count == 1
    assert session.metrics.solver_call_count == 8


def test_persistent_block_session_removes_epsilon_and_point_locks() -> None:
    """A tight point cannot constrain a later independent coverage anchor."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    pool = generate_candidate_pools(
        members, versions(("P1", "S1")), config
    )[0]
    session = BlockSolverSession(pool, members, config)

    tight = session.solve(CoverageMode.PARETO, j_ch_epsilon=0.0)
    coverage = session.solve(CoverageMode.GREENFIELD_COVERAGE)
    operations = session.solve(CoverageMode.GREENFIELD_OPERATIONS)

    assert {item.candidate.member_ids for item in tight.selected} == {
        ("F1",),
        ("F2",),
    }
    assert [item.candidate.member_ids for item in coverage.selected] == [
        ("F1", "F2")
    ]
    assert {item.candidate.member_ids for item in operations.selected} == {
        ("F1",),
        ("F2",),
    }
    assert not session.model.pareto_epsilon.active
    assert len(session.model.lexicographic_locks) == 0


def test_greenfield_heuristic_is_deterministic_exact_and_epsilon_feasible() -> None:
    """Input order and historical labels cannot change the merge incumbent."""

    labeled = (
        member("F1", baseline_group="OLD-A"),
        member("F2", baseline_group="OLD-B"),
        member("F3", baseline_group="OLD-C"),
    )
    blank = tuple(replace(item, baseline_group=None) for item in reversed(labeled))
    config = RunConfig(
        coverage_mode=CoverageMode.PARETO,
        matrix_mode=MatrixMode.OFF,
    )
    labeled_pool = generate_candidate_pools(
        labeled, versions(("P1", "S1")), config
    )[0]
    blank_pool = generate_candidate_pools(
        blank, versions(("P1", "S1")), config
    )[0]
    epsilon = 0.05

    labeled_result = BlockSolverSession(
        replace(labeled_pool, candidates=tuple(reversed(labeled_pool.candidates))),
        tuple(reversed(labeled)),
        config,
    ).solve(CoverageMode.PARETO, j_ch_epsilon=epsilon)
    blank_result = BlockSolverSession(blank_pool, blank, config).solve(
        CoverageMode.PARETO, j_ch_epsilon=epsilon
    )

    assert labeled_result.heuristic_evidence is not None
    assert blank_result.heuristic_evidence is not None
    assert labeled_result.heuristic_evidence.exact_cover_valid
    assert labeled_result.heuristic_evidence.within_epsilon
    assert labeled_result.heuristic_evidence.j_ch <= epsilon + 1e-7
    assert labeled_result.heuristic_evidence == replace(
        blank_result.heuristic_evidence,
        construction_seconds=labeled_result.heuristic_evidence.construction_seconds,
    )
    singleton_coverage = sum(
        item.coefficients.coverage_days * item.coefficients.group_demand_litres
        for item in solve_candidate_pools(
            (blank_pool,),
            blank,
            config.model_copy(
                update={"coverage_mode": CoverageMode.GREENFIELD_OPERATIONS}
            ),
        ).selected
    )
    assert labeled_result.heuristic_evidence.coverage_numerator <= singleton_coverage


def test_coefficients_use_selected_pallet_formula_and_coverage_basis() -> None:
    """Coefficient calculation delegates to the governed KPI formulas."""

    members = (member("F1", demand=1_000, pallet=200),)
    base = RunConfig(
        coverage_basis=CoverageBasis.ADJUSTED_GROUP,
        pallet_formula=PalletFormula.MINIMUM_ONLY,
    )
    pool = generate_candidate_pools(members, versions(("P1", "S1")), base)[0]
    minimum = precompute_coefficients(pool.candidates, members, base)[0]
    whole = precompute_coefficients(
        pool.candidates,
        members,
        base.model_copy(update={"pallet_formula": PalletFormula.WHOLE_PALLET_ROUNDING}),
    )[0]
    assert minimum.coverage_days == pytest.approx(50.0)
    assert whole.coverage_days == pytest.approx(50.0)
    worst = precompute_coefficients(
        pool.candidates,
        members,
        base.model_copy(update={"coverage_basis": CoverageBasis.WORST_FINI}),
    )[0]
    assert worst.coverage_days == pytest.approx(50.0)


def test_empirical_diagnostic_zero_exception_tier_needs_no_solver_pass() -> None:
    """The default empirical matrix records its constant zero tier as not required."""

    members = (member("F1"), member("F2"))
    config = RunConfig()
    pools = generate_candidate_pools(members, versions(("P1", "S1")), config)

    result = solve_candidate_pools(pools, members, config)

    assert result.objective_levels[0].name == "target_violation_count"
    matrix_level = next(
        level
        for level in result.objective_levels
        if level.name == "matrix_exception_pairs"
    )
    assert matrix_level.value == 0
    assert matrix_level.termination_condition == "not_required"


def test_diagnostic_matrix_exception_is_the_leading_business_tier() -> None:
    """DIAGNOSTIC avoids an incompatible pair before improving MAX coverage."""

    members = (member("F1", volume=0.25), member("F2", volume=4.0))
    config = RunConfig(
        coverage_mode=CoverageMode.MAX,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.DIAGNOSTIC,
        versions=VersionIdentifiers(matrix_version="SYNTHETIC_VOLUME_MATRIX_V1"),
    )
    pools = generate_candidate_pools(members, versions(("P1", "S1")), config)
    result = solve_candidate_pools(pools, members, config)
    assert result.objective_levels[0].name == "matrix_exception_pairs"
    assert result.objective_levels[0].value == 0
    assert {item.candidate.member_ids for item in result.selected} == {("F1",), ("F2",)}


def test_operations_first_uses_governed_lexicographic_order() -> None:
    """Operations-first protects three band tiers before J_CH and weighted mean."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.OPERATIONS_FIRST,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.OFF,
        target_band=TargetBand(lower_days=100, upper_days=200),
    )
    pools = generate_candidate_pools(members, versions(("P1", "S1")), config)
    result = solve_candidate_pools(pools, members, config)
    assert [level.name for level in result.objective_levels[:5]] == [
        "target_violation_count",
        "target_worst_excess_days",
        "target_total_excess_days",
        "j_ch",
        "demand_weighted_mean_coverage_days",
    ]


def test_baseline_constrained_modes_block_both_partition_extremes() -> None:
    """Coverage cannot add J_CH and operations cannot add fragmentation."""

    separate = (
        member("F1", baseline_group="B1"),
        member("F2", baseline_group="B2"),
    )
    coverage = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
        matrix_mode=MatrixMode.OFF,
    )
    coverage_result = solve_decomposed_pools(
        generate_candidate_pools(separate, versions(("P1", "S1")), coverage),
        separate,
        coverage,
    )
    assert coverage_result.status == "optimal"
    assert coverage_result.solver_method == "pyomo_appsi_highs"
    assert coverage_result.warm_start_kind == "baseline"
    assert coverage_result.warm_start_group_count == 2
    assert coverage_result.baseline_constraint_limits is not None
    assert coverage_result.baseline_constraint_limits.j_ch_limit == 0
    assert {item.candidate.member_ids for item in coverage_result.selected} == {
        ("F1",),
        ("F2",),
    }
    assert [level.name for level in coverage_result.objective_levels[:1]] == [
        "demand_weighted_mean_coverage_days"
    ]

    together = (
        member("F1", baseline_group="B1"),
        member("F2", baseline_group="B1"),
    )
    operations = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
        matrix_mode=MatrixMode.OFF,
    )
    operations_result = solve_decomposed_pools(
        generate_candidate_pools(together, versions(("P1", "S1")), operations),
        together,
        operations,
    )
    assert operations_result.status == "optimal"
    assert operations_result.baseline_constraint_limits is not None
    assert operations_result.baseline_constraint_limits.group_count == 1
    assert operations_result.baseline_constraint_limits.singleton_group_count == 0
    assert [item.candidate.member_ids for item in operations_result.selected] == [
        ("F1", "F2")
    ]
    assert [level.name for level in operations_result.objective_levels[:2]] == [
        "j_ch",
        "demand_weighted_mean_coverage_days",
    ]


def test_baseline_constraint_limits_count_normalized_exception_pairs_once() -> None:
    """Repeated reversed volume pairs share one auditable distinct-pair limit."""

    members = (
        member("F1", baseline_group="B1"),
        member("F2", baseline_group="B1"),
        member("F3", baseline_group="B2"),
        member("F4", baseline_group="B2"),
    )
    block = ("P1", "S1")
    candidates = (
        manual_candidate(
            block,
            ("F1", "F2"),
            90,
            200,
            "12",
            (("F1", "F2", 1.0, 10.0),),
        ),
        manual_candidate(
            block,
            ("F3", "F4"),
            90,
            200,
            "34",
            (("F3", "F4", 10.0, 1.0),),
        ),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
    )

    result = solve_candidate_pools(
        (manual_pool(block, candidates, config=config),), members, config
    )

    assert result.status == "optimal"
    assert result.baseline_constraint_limits is not None
    assert result.baseline_constraint_limits.as_dict()[
        "matrix_exception_group_count"
    ] == 2
    assert result.baseline_constraint_limits.matrix_exception_pair_count == 2
    assert result.baseline_constraint_limits.matrix_exception_distinct_volume_pair_count == 1


def test_baseline_p90_guardrail_allows_one_exceedance_for_ten_groups() -> None:
    """Nearest-rank P90 uses 10H <= G when selected group count is variable."""

    block = ("P1", "S1")
    members = tuple(
        member(f"F{index}", baseline_group=f"B{index}")
        for index in range(10)
    )
    candidates = tuple(
        manual_candidate(
            block,
            (item.fini_id,),
            40 if index == 9 else 4,
            100,
            str(index),
        )
        for index, item in enumerate(members)
    )
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
    )

    result = solve_candidate_pools(
        (manual_pool(block, candidates, config=config),), members, config
    )

    assert result.status == "optimal"
    assert result.baseline_constraint_limits is not None
    limit = result.baseline_constraint_limits.p90_coverage_days
    exceedances = sum(
        item.coefficients.coverage_days > limit + 1e-9 for item in result.selected
    )
    assert len(result.selected) == 10
    assert exceedances == 1
    assert 10 * exceedances <= len(result.selected)


def test_baseline_constrained_mode_without_frozen_cover_is_not_solved() -> None:
    """A constrained comparison requires a reproducible frozen partition."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
        matrix_mode=MatrixMode.OFF,
    )
    result = solve_candidate_pools(
        generate_candidate_pools(members, versions(("P1", "S1")), config),
        members,
        config,
    )

    assert result.status == "baseline_infeasible"
    assert result.result_class == "baseline-infeasible-under-active-rules"
    assert not result.has_incumbent
    assert result.warm_start_kind == "none"


def test_solution_loader_propagates_auxiliary_values_between_tiers() -> None:
    """Loading all incumbent Vars leaves a changed auxiliary start feasible."""

    model = pyo.ConcreteModel()
    model.C = pyo.RangeSet(0, 0)
    model.x = pyo.Var(model.C, domain=pyo.Binary, initialize=0)
    model.aux = pyo.Var(domain=pyo.Binary, initialize=0)
    model.link = pyo.Constraint(expr=model.aux == model.x[0])
    model.objective = pyo.Objective(expr=-model.x[0])
    solver = _configure_solver(RunConfig(matrix_mode=MatrixMode.OFF))

    first = _solve_once(model, solver)
    assert first.feasible
    assert model.x[0].value == pytest.approx(1)
    assert model.aux.value == pytest.approx(1)

    model.lock = pyo.Constraint(expr=model.x[0] >= 1)
    model.objective.set_value(model.aux)
    second = _solve_once(model, solver)
    assert second.feasible
    assert model.aux.value == pytest.approx(1)


def test_configured_mip_gap_is_not_treated_as_an_exact_tier_proof() -> None:
    """A gap-accepted HiGHS status cannot be locked or labeled optimal."""

    snapshot = _SolveSnapshot(
        termination=TerminationCondition.optimal,
        incumbent=100.0,
        bound=99.5,
        gap=0.005,
        primals={},
        wallclock_seconds=1.0,
    )

    assert not snapshot.optimal
    assert snapshot.tier_termination == "gap_accepted"


def test_global_group_mean_uses_dinkelbach_across_blocks() -> None:
    """The coordinator minimizes the portfolio ratio, not each block's local mean."""

    members = (
        member("A1", demand=100),
        member("A2", demand=100),
        member("B1", plant="P2", demand=100),
    )
    block_a = ("P1", "S1")
    block_b = ("P2", "S1")
    pool_a = manual_pool(
        block_a,
        (
            manual_candidate(block_a, ("A1",), 3.6, 100, "a1"),  # 9 days
            manual_candidate(block_a, ("A2",), 3.6, 100, "a2"),
            manual_candidate(block_a, ("A1", "A2"), 8.0, 200, "a12"),  # 10 days
        ),
    )
    pool_b = manual_pool(
        block_b,
        (manual_candidate(block_b, ("B1",), 0.4, 100, "b1"),),  # 1 day
    )
    config = RunConfig(
        coverage_mode=CoverageMode.GROUP_MEAN,
        coverage_basis=CoverageBasis.BASE_GROUP,
    )
    local = solve_block_pool(pool_a, members, config)
    global_result = solve_candidate_pools((pool_a, pool_b), members, config)
    assert {item.candidate.member_ids for item in local.selected} == {("A1",), ("A2",)}
    assert {item.candidate.member_ids for item in global_result.selected} == {
        ("A1", "A2"),
        ("B1",),
    }
    assert next(
        level.value
        for level in global_result.objective_levels
        if level.name == "group_mean_coverage_days"
    ) == pytest.approx(5.5)


@pytest.mark.parametrize(
    ("completeness", "expected"),
    [("complete", "modeled_infeasible"), ("restricted", "pool_infeasible")],
)
def test_uncovered_precheck_has_honest_library_classification(
    completeness: str, expected: str
) -> None:
    """Missing candidate coverage is classified by disclosed pool completeness."""

    members = (member("F1"),)
    pool = manual_pool(("P1", "S1"), (), completeness=completeness)
    result = solve_candidate_pools((pool,), members, RunConfig())
    assert result.status == "uncovered_precheck"
    assert not result.has_incumbent
    assert result.result_class == expected
    assert result.warm_start_kind == "none"
    assert result.warm_start_group_count == 0


@pytest.mark.parametrize(
    ("completeness", "expected"),
    [("complete", "modeled_infeasible"), ("restricted", "pool_infeasible")],
)
def test_solver_proved_no_exact_cover_uses_pool_completeness(
    completeness: str, expected: str
) -> None:
    """An odd cycle of pairs passes incidence checks but has no exact cover."""

    members = (member("F1"), member("F2"), member("F3"))
    block = ("P1", "S1")
    candidates = tuple(
        manual_candidate(block, pair, 90.0, 200.0, "-".join(pair))
        for pair in (("F1", "F2"), ("F1", "F3"), ("F2", "F3"))
    )
    result = solve_candidate_pools(
        (manual_pool(block, candidates, completeness=completeness),),
        members,
        RunConfig(),
    )
    assert result.status == "infeasible"
    assert result.result_class == expected


def test_restricted_optimum_is_pool_optimal_and_repeated_output_is_stable() -> None:
    """A zero-gap restricted result never claims global exactness."""

    members = (member("F1"), member("F2"))
    config = RunConfig(
        coverage_mode=CoverageMode.MAX,
        coverage_basis=CoverageBasis.BASE_GROUP,
    )
    exact_pool = generate_candidate_pools(members, versions(("P1", "S1")), config)[0]
    restricted = replace(exact_pool, method="restricted", completeness="restricted")
    first = solve_candidate_pools((restricted,), members, config)
    second = solve_candidate_pools((restricted,), tuple(reversed(members)), config)
    assert first.result_class == "pool-optimal"
    assert first.relative_gap == 0
    assert [item.candidate.candidate_hash for item in first.selected] == [
        item.candidate.candidate_hash for item in second.selected
    ]


def test_decomposed_solver_combines_block_results_and_kpis() -> None:
    """The scalable path covers every block and recomputes portfolio metrics."""

    members = (
        member("A", plant="P1", demand=100),
        member("B", plant="P2", demand=200),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.DEMAND_WEIGHTED_MEAN,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        versions(("P1", "S1"), ("P2", "S1")),
        config,
    )

    result = solve_decomposed_pools(pools, members, config)

    assert result.status == "optimal"
    assert result.solver_method == "pyomo_appsi_highs_block_decomposed"
    assert {item.candidate.block_key for item in result.selected} == {
        ("P1", "S1"),
        ("P2", "S1"),
    }
    assert result.objective_levels[0].name == "demand_weighted_mean_coverage_days"
    assert {evidence.block_key for evidence in result.block_evidence} == {
        ("P1", "S1"),
        ("P2", "S1"),
    }
    assert all(evidence.objective_levels for evidence in result.block_evidence)
    for evidence in result.block_evidence:
        weighted_level = next(
            level
            for level in evidence.objective_levels
            if level.name == "demand_weighted_mean_coverage_days"
        )
        assert weighted_level.best_bound == pytest.approx(weighted_level.value)


def test_decomposed_stage_limit_is_one_scenario_wide_budget() -> None:
    """Block allocations sum to the CLI stage budget and follow library size."""

    small = manual_pool(
        ("P1", "S1"),
        (manual_candidate(("P1", "S1"), ("A",), 10, 100, "a"),),
    )
    large = manual_pool(
        ("P2", "S1"),
        tuple(
            manual_candidate(("P2", "S1"), ("B",), 10 + index, 100, f"b{index}")
            for index in range(3)
        ),
    )

    limits = _decomposed_stage_time_limits((small, large), 40.0)

    assert sum(limits.values()) == pytest.approx(40.0)
    assert limits[("P1", "S1")] == pytest.approx(10.0)
    assert limits[("P2", "S1")] == pytest.approx(30.0)


def test_decomposed_levels_disclose_unreached_later_tiers() -> None:
    """A block timeout cannot make untouched operations tiers look optimal."""

    members = (
        member("A", plant="P1", demand=100),
        member("B", plant="P2", demand=200),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.OPERATIONS_FIRST,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.OFF,
    )
    pools = generate_candidate_pools(
        members,
        versions(("P1", "S1"), ("P2", "S1")),
        config,
    )
    first = solve_block_pool(pools[0], members, config)
    second = solve_block_pool(pools[1], members, config)
    first_tier = second.objective_levels[0]
    truncated = replace(
        second,
        status="feasible_limit",
        objective_levels=(
            ObjectiveLevel(
                first_tier.name,
                first_tier.value,
                "maxTimeLimit",
                first_tier.best_bound,
                0.5,
            ),
        ),
    )

    levels = _combined_levels(
        (*first.selected, *truncated.selected),
        config,
        (first, truncated),
    )
    by_name = {level.name: level for level in levels}

    assert by_name["target_violation_count"].termination_condition == (
        "decomposed_feasible_limit"
    )
    assert by_name["target_worst_excess_days"].termination_condition == (
        "partially_reached"
    )
    assert by_name["j_ch"].termination_condition == "partially_reached"
    assert _combined_primary_gap(
        (first, replace(truncated, primary_relative_gap=None))
    ) is None


def test_solver_rejects_pool_from_wider_or_different_structural_settings() -> None:
    """A stale HARD pool cannot be mislabeled exact under matrix-off settings."""

    members = (member("F1", volume=0.25), member("F2", volume=4.0))
    hard = RunConfig(
        coverage_mode=CoverageMode.MAX,
        coverage_basis=CoverageBasis.BASE_GROUP,
        matrix_mode=MatrixMode.HARD,
    )
    pool = generate_candidate_pools(members, versions(("P1", "S1")), hard)[0]

    with pytest.raises(ValueError, match="different structural settings"):
        solve_candidate_pools(
            (pool,),
            members,
            hard.model_copy(update={"matrix_mode": MatrixMode.OFF}),
        )


def test_time_limit_without_incumbent_is_not_business_infeasibility() -> None:
    """No-incumbent status remains distinct from modeled/pool infeasibility."""

    result = _infeasible_result(
        completeness="complete",
        candidate_count=10,
        status="no_incumbent",
        termination="maxTimeLimit",
    )
    assert result.result_class == "no-incumbent-within-limit"


def test_baseline_warm_start_builds_a_complete_candidate_cover() -> None:
    """Historical memberships seed widened matrix/PV searches deterministically."""

    members = (
        CandidateMember(
            "F1", "P1", "S1", frozenset({"L1"}), 100, 10, 0.25, "PV1", 100, "G1"
        ),
        CandidateMember(
            "F2", "P1", "S1", frozenset({"L1"}), 100, 10, 0.25, "PV1", 100, "G2"
        ),
    )
    config = RunConfig(matrix_mode=MatrixMode.DIAGNOSTIC)
    pool = generate_candidate_pools(members, versions(("P1", "S1")), config)[0]

    selected = _baseline_warm_start_indexes(pool.candidates, members)

    assert len(selected) == 2
    assert {
        pool.candidates[index].member_ids for index in selected
    } == {("F1",), ("F2",)}


def test_baseline_neighbourhood_selects_one_improving_block() -> None:
    """The scalable constrained path changes only its best passing block."""

    blocks = (("P1", "S1"), ("P2", "S1"))
    members = tuple(
        member(
            fini,
            plant=plant,
            baseline_group=f"B-{plant}",
        )
        for plant, sefi in blocks
        for fini in (f"{plant}-1", f"{plant}-2")
    )
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
        baseline_guardrails=BaselineGuardrailPolicy(j_ch_absolute_tolerance=200),
    )
    pools = []
    for index, block in enumerate(blocks):
        ids = tuple(item.fini_id for item in members if item.block_key == block)
        baseline = manual_candidate(block, ids, 90, 200, f"base-{index}")
        candidates = (baseline,)
        if index == 0:
            candidates += (
                replace(
                    manual_candidate(block, ids, 45, 200, "better"),
                    pv_id="PV2",
                ),
            )
        pools.append(manual_pool(block, candidates, config=config))
    result = solve_decomposed_pools(tuple(pools), members, config)

    assert result.status == "feasible_witness"
    assert result.result_class == "baseline-neighbourhood-witness"
    assert result.baseline_constraint_scope == "one_block_neighbourhood"
    assert {
        item.candidate.block_key: item.candidate.pv_id for item in result.selected
    } == {blocks[0]: "PV2", blocks[1]: "PV1"}


def test_baseline_neighbourhood_rejects_globally_failing_trial() -> None:
    """A new exception pair cannot breach the full portfolio union limit."""

    blocks = (("P1", "S1"), ("P2", "S1"))
    members = tuple(
        member(f"{plant}-{suffix}", plant=plant, baseline_group=f"B-{plant}")
        for plant, _ in blocks
        for suffix in ("1", "2")
    )
    shared_pair = (("x", "y", 1.0, 10.0),)
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
        baseline_guardrails=BaselineGuardrailPolicy(j_ch_absolute_tolerance=200),
    )
    pools = []
    for index, block in enumerate(blocks):
        ids = tuple(item.fini_id for item in members if item.block_key == block)
        baseline = manual_candidate(
            block, ids, 90, 200, f"base-pair-{index}", shared_pair
        )
        candidates = (baseline,)
        if index == 0:
            candidates += (
                replace(
                    manual_candidate(
                        block,
                        ids,
                        45,
                        200,
                        "new-pair",
                        (("x", "y", 2.0, 20.0),),
                    ),
                    pv_id="PV2",
                ),
            )
        pools.append(manual_pool(block, candidates, config=config))
    result = solve_decomposed_pools(tuple(pools), members, config)

    assert result.result_class == "baseline-neighbourhood-nondominated"
    assert {item.candidate.pv_id for item in result.selected} == {"PV1"}
    assert result.baseline_constraint_limits is not None
    assert result.baseline_constraint_limits.matrix_exception_distinct_volume_pair_count == 1
    first_block = next(item for item in result.block_evidence if item.block_key == blocks[0])
    assert first_block.primary_objective == pytest.approx(result.primary_objective)


def test_baseline_neighbourhood_never_claims_global_or_pool_optimality() -> None:
    """A multi-block witness carries limits and evidence but no optimization bound."""

    members = (
        member("A", plant="P1", baseline_group="B1"),
        member("B", plant="P2", baseline_group="B2"),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_OPERATIONS,
    )
    pools = generate_candidate_pools(
        members, versions(("P1", "S1"), ("P2", "S1")), config
    )

    result = solve_decomposed_pools(pools, members, config)

    assert result.result_class == "baseline-neighbourhood-nondominated"
    assert result.status == "baseline_nondominated_in_evaluated_neighbourhood"
    assert result.best_bound is None
    assert result.relative_gap is None
    assert result.baseline_constraint_scope == "one_block_neighbourhood"
    assert result.baseline_constraint_limits is not None
    assert len(result.block_evidence) == 2


def test_baseline_neighbourhood_uses_global_j_ch_tolerance_residual() -> None:
    """Global relative tolerance can fund a merge in a zero-J_CH baseline block."""

    block_a = ("P1", "S1")
    block_b = ("P2", "S1")
    members = (
        member("A1", plant="P1", baseline_group="A1"),
        member("A2", plant="P1", baseline_group="A2"),
        member("B1", plant="P2", baseline_group="B"),
        member("B2", plant="P2", baseline_group="B"),
    )
    config = RunConfig(
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
        baseline_guardrails=BaselineGuardrailPolicy(j_ch_relative_tolerance=0.2),
    )
    pool_a = manual_pool(
        block_a,
        (
            manual_candidate(block_a, ("A1",), 90, 100, "a1"),
            manual_candidate(block_a, ("A2",), 90, 100, "a2"),
            manual_candidate(block_a, ("A1", "A2"), 90, 200, "a12"),
        ),
        config=config,
    )
    pool_b = manual_pool(
        block_b,
        (manual_candidate(block_b, ("B1", "B2"), 10, 200, "b12"),),
        config=config,
    )

    result = solve_decomposed_pools((pool_a, pool_b), members, config)

    assert result.baseline_constraint_limits is not None
    assert result.baseline_constraint_limits.j_ch_limit == pytest.approx(0.48)
    assert {item.candidate.member_ids for item in result.selected} == {
        ("A1", "A2"),
        ("B1", "B2"),
    }
    assert result.result_class == "baseline-neighbourhood-witness"
