"""Focused tests for deterministic exact and restricted candidate generation."""

from __future__ import annotations

from itertools import combinations

import pytest
from production_wheel.matrix import (
    BASELINE_EMPIRICAL_MATRIX_VERSION,
    CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1,
    UNKNOWN_LINE_FEASIBLE,
    YES_OBSERVED_BASELINE,
)
from production_wheel.candidates import (
    CandidateMember,
    PoolBudgetExceeded,
    ProductionVersion,
    generate_block_pool,
    generate_candidate_pools,
    line_feasible_subset_count,
    partition_blocks,
    project_block,
)
from production_wheel.schemas import (
    CoverageBasis,
    CoverageMode,
    GroupSizeMode,
    GroupSizePolicy,
    MatrixMode,
    PalletFormula,
    PoolLimits,
    PVMode,
    RunConfig,
    TargetBand,
    VersionIdentifiers,
)


def member(
    fini_id: str,
    *,
    plant: str = "P1",
    sefi: str = "S1",
    lines: tuple[str, ...] = ("1",),
    volume: float = 0.25,
    pv: str | None = "PV1",
    lot: float | None = 100.0,
    baseline: str | None = None,
) -> CandidateMember:
    """Create one compact valid synthetic candidate member."""

    ordinal = int(
        "".join(character for character in fini_id if character.isdigit()) or "1"
    )
    return CandidateMember(
        fini_id=fini_id,
        plant=plant,
        sefi=sefi,
        eligible_lines=frozenset(lines),
        demand_litres=1_000.0 + ordinal,
        pallet_litres=25.0,
        package_volume=volume,
        fixed_pv=pv,
        fixed_lot_litres=lot,
        baseline_group=baseline,
        pck_code="PCK-A",
    )


def version(
    pv_id: str = "PV1",
    lot: float | None = 100.0,
    *,
    plant: str = "P1",
    sefi: str = "S1",
) -> ProductionVersion:
    """Create one synthetic production-version row."""

    return ProductionVersion(
        plant=plant, sefi=sefi, pv_id=pv_id, nominal_lot_litres=lot
    )


def restricted_config(
    *,
    limit: int = 250_000,
    retained: int = 2_000,
    pv_mode: PVMode = PVMode.FIXED,
    matrix_mode: MatrixMode = MatrixMode.HARD,
    cap: int = 7,
    coverage_mode: CoverageMode = CoverageMode.GREENFIELD_COVERAGE,
) -> RunConfig:
    """Force restricted generation while preserving configurable pool behavior."""

    group_size = (
        GroupSizePolicy()
        if cap == 7
        else GroupSizePolicy(
            mode=GroupSizeMode.BOUNDED_RELAXATION, additional_members=cap - 7
        )
    )
    return RunConfig(
        coverage_mode=coverage_mode,
        pv_mode=pv_mode,
        matrix_mode=matrix_mode,
        group_size=group_size,
        pool_limits=PoolLimits(
            exact_member_subsets=1,
            exact_pv_configurations=1,
            restricted_pv_configurations=limit,
            retained_member_sets_per_size_scorer=retained,
        ),
    )


def memberships(pool: object) -> set[tuple[str, ...]]:
    """Return distinct member tuples from a candidate pool."""

    return {candidate.member_ids for candidate in pool.candidates}  # type: ignore[attr-defined]


def test_partition_uses_plant_and_sefi_and_sorts_members() -> None:
    """Blocks must never cross either structural boundary."""

    items = [
        member("F2"),
        member("F1"),
        member("F3", sefi="S2"),
        member("F4", plant="P2"),
    ]
    blocks = partition_blocks(items)
    assert tuple(blocks) == (("P1", "S1"), ("P1", "S2"), ("P2", "S1"))
    assert tuple(item.fini_id for item in blocks[("P1", "S1")]) == ("F1", "F2")


def test_line_only_projection_dp_counts_common_intersections() -> None:
    """The scale projection excludes subsets with an empty line intersection."""

    items = [member("F1", lines=("1", "2")), member("F2"), member("F3", lines=("2",))]
    assert line_feasible_subset_count(items, 7) == 5
    assert line_feasible_subset_count([member("F4", lines=())], 7) == 0


def test_projection_classifies_inclusively_at_both_thresholds() -> None:
    """Projected member and PV thresholds are inclusive and conservative."""

    items = [member("F1"), member("F2")]
    optimized = RunConfig(
        pv_mode=PVMode.OPTIMIZED,
        pool_limits=PoolLimits(exact_member_subsets=3, exact_pv_configurations=6),
    )
    exact = project_block(items, [version(), version("PV2", 200.0)], optimized)
    assert (exact.line_feasible_member_subsets, exact.projected_pv_configurations) == (
        3,
        6,
    )
    assert exact.method == "exact"
    restricted = project_block(
        items,
        [version(), version("PV2", 200.0)],
        optimized.model_copy(
            update={"pool_limits": PoolLimits(exact_member_subsets=2)}
        ),
    )
    assert restricted.method == "restricted"


def test_explicit_exhaustive_block_overrides_projection_thresholds() -> None:
    """A reviewed deep-run block enumerates fully without changing other blocks."""

    items = [member(f"F{index}") for index in range(1, 7)]
    config = restricted_config().model_copy(
        update={"exhaustive_blocks": (("P1", "S1"),)}
    )

    pool = generate_block_pool(items, [version()], config)

    assert pool.method == "exact"
    assert pool.completeness == "complete"
    assert len(pool.candidates) == 63


def test_forced_exact_ceiling_reverts_oversized_blocks_to_restricted() -> None:
    """A forced exhaustive block above the configuration ceiling stays restricted.

    Six line-sharing FINIs project to 63 configurations. When the forced-exact
    ceiling is below that count the block must not enumerate exhaustively (which
    on real data means never building a multi-hundred-thousand-column model);
    it falls back to the restricted beam. A generous ceiling keeps it exact.
    """

    items = [member(f"F{index}") for index in range(1, 7)]
    forced = restricted_config().model_copy(
        update={"exhaustive_blocks": (("P1", "S1"),)}
    )
    capped = forced.model_copy(
        update={
            "pool_limits": forced.pool_limits.model_copy(
                update={"forced_exact_configuration_ceiling": 10}
            )
        }
    )
    assert project_block(items, [version()], capped).method == "restricted"
    generous = forced.model_copy(
        update={
            "pool_limits": forced.pool_limits.model_copy(
                update={"forced_exact_configuration_ceiling": 250_000}
            )
        }
    )
    assert project_block(items, [version()], generous).method == "exact"


def test_fixed_mode_requires_one_common_pv_and_exact_positive_lot() -> None:
    """Mixed, missing, ambiguous, or mismatched planning PV evidence is infeasible."""

    items = [member("F1"), member("F2")]
    pool = generate_block_pool(items, [version()], RunConfig())
    assert memberships(pool) == {("F1",), ("F2",), ("F1", "F2")}

    mixed = [member("F1"), member("F2", pv="PV2", lot=200.0)]
    mixed_pool = generate_block_pool(
        mixed, [version(), version("PV2", 200.0)], RunConfig()
    )
    assert memberships(mixed_pool) == {("F1",), ("F2",)}

    mismatch = [member("F1", lot=101.0)]
    assert not generate_block_pool(mismatch, [version()], RunConfig()).candidates
    assert not generate_block_pool(
        [member("F1")], [version(lot=0.0)], RunConfig()
    ).candidates


def test_optimized_mode_expands_every_finite_unambiguous_block_pv() -> None:
    """Optimized sensitivity expands member sets without using fixed member PVs."""

    items = [member("F1", pv=None, lot=None), member("F2", pv=None, lot=None)]
    versions = [version("PV2", 200.0), version(), version("INVALID", -1.0)]
    pool = generate_block_pool(items, versions, RunConfig(pv_mode=PVMode.OPTIMIZED))
    assert len(pool.candidates) == 6
    assert {candidate.pv_id for candidate in pool.candidates} == {"PV1", "PV2"}


def test_diagnostic_matrix_mode_retains_member_pair_evidence() -> None:
    """Diagnostic sensitivity retains every incompatible FINI pair."""

    items = [member("F1", volume=0.25), member("F2", volume=4.0)]
    pool = generate_block_pool(
        items,
        [version()],
        RunConfig(
            matrix_mode=MatrixMode.DIAGNOSTIC,
            versions=VersionIdentifiers(matrix_version="SYNTHETIC_VOLUME_MATRIX_V1"),
        ),
    )
    pair = next(
        candidate for candidate in pool.candidates if len(candidate.member_ids) == 2
    )
    assert pair.matrix_exception_pairs == (("F1", "F2", 0.25, 4.0),)


def test_customer_matrix_hard_rejects_and_flexible_retains_avoid_pair() -> None:
    """AVOID is forbidden only in HARD mode and remains auditable in FLEXIBLE."""

    items = [member("F1", volume=1.0), member("F2", volume=10.0)]
    versions = VersionIdentifiers(
        matrix_version=CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
    )
    hard = generate_block_pool(
        items,
        [version()],
        RunConfig(matrix_mode=MatrixMode.HARD, versions=versions),
    )
    flexible = generate_block_pool(
        items,
        [version()],
        RunConfig(matrix_mode=MatrixMode.FLEXIBLE, versions=versions),
    )

    assert memberships(hard) == {("F1",), ("F2",)}
    pair = next(
        candidate for candidate in flexible.candidates if len(candidate.member_ids) == 2
    )
    assert pair.matrix_exception_pairs == (("F1", "F2", 1.0, 10.0),)


def test_off_matrix_mode_skips_compatibility_evidence() -> None:
    """Matrix-off sensitivity neither filters nor counts package-volume pairs."""

    items = [member("F1", volume=0.25), member("F2", volume=4.0)]
    pool = generate_block_pool(
        items, [version()], RunConfig(matrix_mode=MatrixMode.OFF)
    )
    pair = next(
        candidate for candidate in pool.candidates if len(candidate.member_ids) == 2
    )
    assert pair.matrix_exception_pairs == ()


def test_hard_matrix_filters_incompatible_pairs() -> None:
    """Hard mode rejects a candidate while retaining its feasible singletons."""

    items = [member("F1", volume=0.25), member("F2", volume=4.0)]
    pool = generate_block_pool(
        items,
        [version()],
        RunConfig(
            matrix_mode=MatrixMode.HARD,
            versions=VersionIdentifiers(matrix_version="SYNTHETIC_VOLUME_MATRIX_V1"),
        ),
    )
    assert memberships(pool) == {("F1",), ("F2",)}


def test_empirical_hard_mode_keeps_observed_and_unknown_pairs_auditable() -> None:
    """Only an explicit `NO` is hard; empirical unknown is not incompatibility."""

    config = RunConfig(
        matrix_mode=MatrixMode.HARD,
        versions=VersionIdentifiers(matrix_version=BASELINE_EMPIRICAL_MATRIX_VERSION),
    )
    items = [
        member("F1", volume=0.25, baseline="G1"),
        member("F2", volume=4.0, baseline="G1"),
        member("F3", volume=2.5, baseline="G2"),
    ]
    pool = generate_block_pool(items, [version()], config)
    observed = next(
        candidate for candidate in pool.candidates if candidate.member_ids == ("F1", "F2")
    )
    unknown = next(
        candidate for candidate in pool.candidates if candidate.member_ids == ("F1", "F3")
    )
    assert observed.matrix_exception_pairs == ()
    assert observed.matrix_evidence_pairs == (
        ("F1", "F2", 0.25, 4.0, YES_OBSERVED_BASELINE),
    )
    assert unknown.matrix_exception_pairs == ()
    assert unknown.matrix_unknown_pairs == (
        ("F1", "F3", 0.25, 2.5, UNKNOWN_LINE_FEASIBLE),
    )


def test_all_block_generation_uses_global_empirical_baseline_evidence() -> None:
    """Observed evidence in one block is available consistently to every block."""

    config = RunConfig(
        versions=VersionIdentifiers(matrix_version=BASELINE_EMPIRICAL_MATRIX_VERSION)
    )
    items = [
        member("F1", volume=0.25, baseline="A"),
        member("F2", volume=4.0, baseline="B"),
        member("F3", volume=0.25, baseline="G1", sefi="S2"),
        member("F4", volume=4.0, baseline="G1", sefi="S2"),
    ]
    pools = generate_candidate_pools(items, [version(), version(sefi="S2")], config)
    first_pair = next(
        candidate
        for candidate in pools[0].candidates
        if candidate.member_ids == ("F1", "F2")
    )
    assert first_pair.matrix_unknown_pairs == ()
    assert first_pair.matrix_evidence_pairs[0][-1] == YES_OBSERVED_BASELINE


def test_unknown_matrix_volume_is_rejected_when_matrix_is_active() -> None:
    """Active matrix modes validate volumes against the governed catalog."""

    with pytest.raises(ValueError):
        generate_block_pool(
            [member("F1", volume=1.23)],
            [version()],
            RunConfig(matrix_mode=MatrixMode.DIAGNOSTIC),
        )


def test_off_matrix_mode_allows_future_governed_volume_inputs() -> None:
    """Matrix-off can evaluate a future catalog only after upstream validation."""

    pool = generate_block_pool(
        [member("F1", volume=1.23)],
        [version()],
        RunConfig(matrix_mode=MatrixMode.OFF),
    )
    assert pool.candidates


def test_exact_candidates_match_direct_small_block_memberships() -> None:
    """Exact enumeration agrees with a direct combination oracle on a small block."""

    items = [member("F1", lines=("1", "2")), member("F2"), member("F3", lines=("2",))]
    pool = generate_block_pool(items, [version()], RunConfig())
    expected = {
        tuple(item.fini_id for item in subset)
        for size in range(1, 4)
        for subset in combinations(items, size)
        if set.intersection(*(set(item.eligible_lines) for item in subset))
    }
    assert memberships(pool) == expected
    assert pool.completeness == "complete"


def test_greenfield_restricted_mandatory_tiers_ignore_historical_groups() -> None:
    """Greenfield pools retain small sets without historical neighborhood seeds."""

    items = [member(f"F{index}", baseline="G1") for index in range(1, 5)]
    pool = generate_block_pool(items, [version()], restricted_config())
    assert pool.method == "restricted"
    assert pool.mandatory_member_sets == 14
    assert pool.mandatory_pv_configurations == 14
    assert all(
        not {"current_group", "neighborhood_add", "neighborhood_drop", "neighborhood_swap"}
        & set(candidate.source_tiers)
        for candidate in pool.candidates
    )


def test_optional_baseline_mode_retains_historical_neighborhood_provenance() -> None:
    """Explicit regression modes can still reconstruct their historical cover."""

    items = [member(f"F{index}", baseline="G1") for index in range(1, 5)]
    pool = generate_block_pool(
        items,
        [version()],
        restricted_config(
            coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE
        ),
    )
    assert pool.mandatory_member_sets == 15
    current = next(
        candidate for candidate in pool.candidates if len(candidate.member_ids) == 4
    )
    assert "current_group" in current.source_tiers


def test_restricted_swap_neighborhood_handles_members_outside_baseline_group() -> None:
    """One-for-one swap provenance is generated without iterable type errors."""

    items = [
        member("F1", baseline="G1"),
        member("F2", baseline="G1"),
        member("F3", baseline="G1"),
        member("F4", baseline="G2"),
    ]

    pool = generate_block_pool(
        items,
        [version()],
        restricted_config(
            coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE
        ),
    )
    swapped = next(
        candidate
        for candidate in pool.candidates
        if candidate.member_ids == ("F2", "F3", "F4")
    )
    assert "neighborhood_swap" in swapped.source_tiers


def test_restricted_pool_fails_instead_of_truncating_mandatory_tiers() -> None:
    """The mandatory library cannot be silently clipped to fit its ceiling."""

    items = [member(f"F{index}") for index in range(1, 5)]
    with pytest.raises(PoolBudgetExceeded) as captured:
        generate_block_pool(items, [version()], restricted_config(limit=10))
    assert captured.value.configurations == 14
    assert captured.value.limit == 10


def test_beam_expansion_and_round_robin_are_bounded_and_deterministic() -> None:
    """Size-four beam candidates use stable scorer round-robin at the ceiling."""

    items = [member(f"F{index}") for index in range(1, 7)]
    config = restricted_config(limit=43, retained=2)
    first = generate_block_pool(items, [version()], config)
    second = generate_block_pool(tuple(reversed(items)), [version()], config)
    assert len(first.candidates) == 43
    assert first.pool_hash == second.pool_hash
    assert [candidate.candidate_hash for candidate in first.candidates] == [
        candidate.candidate_hash for candidate in second.candidates
    ]
    assert first.size_traces[0].size == 4
    assert first.size_traces[0].selected_member_sets == 2
    assert first.size_traces[0].truncated


def test_objective_and_reporting_changes_reuse_the_restricted_pool() -> None:
    """Non-structural settings neither change the fingerprint nor candidate library."""

    items = [member(f"F{index}") for index in range(1, 7)]
    base = restricted_config(limit=43, retained=2)
    changed = base.model_copy(
        update={
            "coverage_mode": CoverageMode.MAX,
            "coverage_basis": CoverageBasis.WORST_FINI,
            "pallet_formula": PalletFormula.WHOLE_PALLET_ROUNDING,
            "target_band": TargetBand(minimum_days=25, maximum_days=75),
        }
    )
    first = generate_block_pool(items, [version()], base)
    second = generate_block_pool(items, [version()], changed)
    assert first.structural_fingerprint == second.structural_fingerprint
    assert first.pool_hash == second.pool_hash


@pytest.mark.parametrize(("cap", "expected"), [(8, 510), (9, 511)])
def test_bounded_caps_eight_and_nine_are_projected(cap: int, expected: int) -> None:
    """The only v1 relaxed caps are eight and nine."""

    items = [member(f"F{index}") for index in range(1, 10)]
    config = restricted_config(cap=cap)
    assert (
        project_block(items, [version()], config).line_feasible_member_subsets
        == expected
    )


def test_all_blocks_and_hashes_are_stable_under_input_order() -> None:
    """Block output, candidate identity, and pool provenance contain no randomness."""

    items = [member("F2"), member("F1"), member("F3", sefi="S2")]
    versions = [version(), version(sefi="S2")]
    first = generate_candidate_pools(items, versions, RunConfig())
    second = generate_candidate_pools(
        reversed(items), tuple(reversed(versions)), RunConfig()
    )
    assert [(pool.block_key, pool.pool_hash) for pool in first] == [
        (pool.block_key, pool.pool_hash) for pool in second
    ]


def test_greenfield_restricted_pool_is_independent_of_baseline_presence() -> None:
    """Historical labels neither alter greenfield membership nor pool hashes."""

    blank = [member(f"F{index}") for index in range(1, 7)]
    labeled = [member(f"F{index}", baseline=f"G{(index + 1) // 2}") for index in range(1, 7)]
    config = restricted_config(limit=43, retained=2)

    without_baseline = generate_block_pool(blank, [version()], config)
    with_baseline = generate_block_pool(labeled, [version()], config)

    assert without_baseline.pool_hash == with_baseline.pool_hash
    assert [item.candidate_hash for item in without_baseline.candidates] == [
        item.candidate_hash for item in with_baseline.candidates
    ]


def test_pool_fingerprint_changes_when_business_primitives_change() -> None:
    """Demand/pallet changes invalidate restricted scoring and pool provenance."""

    first = generate_block_pool([member("F1")], [version()], RunConfig())
    changed_member = member("F1")
    changed_member = CandidateMember(
        fini_id=changed_member.fini_id,
        plant=changed_member.plant,
        sefi=changed_member.sefi,
        eligible_lines=changed_member.eligible_lines,
        demand_litres=changed_member.demand_litres + 1,
        pallet_litres=changed_member.pallet_litres,
        package_volume=changed_member.package_volume,
        fixed_pv=changed_member.fixed_pv,
        fixed_lot_litres=changed_member.fixed_lot_litres,
        baseline_group=changed_member.baseline_group,
        pck_code=changed_member.pck_code,
    )
    second = generate_block_pool([changed_member], [version()], RunConfig())
    assert first.structural_fingerprint != second.structural_fingerprint
    assert first.pool_hash != second.pool_hash
