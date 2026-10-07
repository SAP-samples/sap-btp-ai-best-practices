"""Focused independent post-solve validation tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.optimization import solve_candidate_pools
from production_wheel.matrix import CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
from production_wheel.schemas import (
    CoverageBasis,
    MatrixMode,
    PVMode,
    RunConfig,
    VersionIdentifiers,
)
from production_wheel.solution_validation import validate_solution


def _member(
    fini_id: str,
    *,
    fixed_pv: str = "PV1",
    demand_litres: float = 1_000.0,
    pallet_litres: float = 100.0,
    package_volume: float = 0.25,
    baseline_group: str = "BASELINE-1",
) -> CandidateMember:
    """Create one modeled FINI whose fixed lot is resolved only by the catalog."""

    return CandidateMember(
        fini_id=fini_id,
        plant="P1",
        sefi="S1",
        eligible_lines=frozenset({"L1", "L2"}),
        demand_litres=demand_litres,
        pallet_litres=pallet_litres,
        package_volume=package_volume,
        fixed_pv=fixed_pv,
        fixed_lot_litres=None,
        baseline_group=baseline_group,
        pck_code="PCK-A",
    )


def _solved(config: RunConfig | None = None):
    """Return a small solved exact library and its canonical inputs."""

    active = config or RunConfig()
    members = (_member("F1"), _member("F2"))
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, active)
    result = solve_candidate_pools(pools, members, active)
    return active, members, versions, pools, result


def test_fixed_pv_uses_catalog_when_member_lot_is_absent() -> None:
    """Canonical extracted members need no duplicated lot when catalog is unique."""

    config, members, versions, pools, result = _solved()
    validation = validate_solution(result, members, pools, config, versions)
    assert validation.is_valid
    assert not validation.issues
    assert sum(group.group_size for group in validation.groups) == 2


def test_validation_records_every_coverage_view_and_member_adjustment() -> None:
    """Independent evidence retains all coverage views regardless of active basis."""

    config = RunConfig(coverage_basis=CoverageBasis.BASE_GROUP)
    members = (
        _member("F1", demand_litres=1_900.0, pallet_litres=100.0),
        _member("F2", demand_litres=100.0, pallet_litres=200.0),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)

    assert validation.is_valid
    assert len(validation.groups) == 1
    group = validation.groups[0]
    assert group.coverage_days == pytest.approx(112.5)
    assert group.base_group_coverage_days == pytest.approx(112.5)
    assert group.adjusted_group_coverage_days == pytest.approx(131.875)
    assert group.worst_fini_coverage_days == pytest.approx(500.0)
    assert dict(group.fini_adjusted_coverage_days) == pytest.approx(
        {"F1": 112.5, "F2": 500.0}
    )


def test_validation_preserves_empirical_unknown_without_inventing_no() -> None:
    """Independent validation agrees that an unobserved line-feasible pair is unknown."""

    config = RunConfig()
    members = (
        _member("F1", package_volume=0.25, baseline_group="BASELINE-1"),
        _member("F2", package_volume=4.0, baseline_group="BASELINE-2"),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)

    assert validation.is_valid
    paired = next(group for group in validation.groups if group.group_size == 2)
    assert paired.matrix_exception_pairs == ()
    assert paired.matrix_unknown_pairs == (
        ("F1", "F2", 0.25, 4.0, "UNKNOWN_LINE_FEASIBLE"),
    )


def test_validation_preserves_flexible_customer_avoid_evidence() -> None:
    """Independent validation accepts and recomputes every flexible AVOID pair."""

    config = RunConfig(
        matrix_mode=MatrixMode.FLEXIBLE,
        versions=VersionIdentifiers(
            matrix_version=CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1
        ),
    )
    members = (
        _member("F1", package_volume=1.0),
        _member("F2", package_volume=10.0),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)

    assert validation.is_valid
    paired = next(group for group in validation.groups if group.group_size == 2)
    assert paired.matrix_exception_pairs == (("F1", "F2", 1.0, 10.0),)


def test_candidate_and_validator_share_excluded_historical_positive_evidence() -> None:
    """All assignments inform positives while excluded rows add no line witness."""

    config = RunConfig()
    members = (
        _member("F1", package_volume=0.5, baseline_group="BASELINE-1"),
        _member("F2", package_volume=3.0, baseline_group="BASELINE-2"),
    )
    historical = (
        *members,
        replace(
            _member("X1", package_volume=0.5, baseline_group="HISTORICAL"),
            eligible_lines=frozenset(),
        ),
        replace(
            _member("X2", package_volume=3.0, baseline_group="HISTORICAL"),
            eligible_lines=frozenset(),
        ),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(
        members, versions, config, baseline_evidence_members=historical
    )
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(
        result, members, pools, config, versions, historical
    )

    assert validation.is_valid
    paired = next(group for group in validation.groups if group.group_size == 2)
    assert paired.matrix_unknown_pairs == ()
    assert paired.matrix_positive_evidence_pairs == (
        ("F1", "F2", 0.5, 3.0, "YES_OBSERVED_BASELINE"),
    )


def test_validation_recomputes_coefficients_and_exact_cover() -> None:
    """Tampered solver evidence and incomplete selection are independently rejected."""

    config, members, versions, pools, result = _solved()
    selected = result.selected[0]
    corrupt_coefficients = replace(
        selected.coefficients, coverage_days=selected.coefficients.coverage_days + 1
    )
    corrupt = replace(result, selected=(replace(selected, coefficients=corrupt_coefficients),))
    validation = validate_solution(corrupt, members, pools, config, versions)
    assert "COVERAGE_COEFFICIENT" in {
        issue.rule_id for issue in validation.issues
    }
    assert not validation.is_valid
    incomplete = validate_solution(
        replace(result, selected=()), members, pools, config, versions
    )
    assert "EXACT_COVER" in {issue.rule_id for issue in incomplete.issues}


def test_validation_rejects_missing_or_ambiguous_fixed_catalog_lot() -> None:
    """Fixed mode never silently accepts an absent or ambiguous PV lot."""

    config, members, versions, pools, result = _solved()
    ambiguous = versions + (ProductionVersion("P1", "S1", "PV1", 1_100.0),)
    validation = validate_solution(result, members, pools, config, ambiguous)
    assert "FIXED_PV_LOT_UNIQUENESS" in {
        issue.rule_id for issue in validation.issues
    }


def test_optimized_pv_requires_one_finite_active_catalog_alternative() -> None:
    """Optimized sensitivity validates its selected PV against active alternatives."""

    config = RunConfig(pv_mode=PVMode.OPTIMIZED)
    members = (_member("F1"),)
    versions = (
        ProductionVersion("P1", "S1", "PV1", 1_000.0),
        ProductionVersion("P1", "S1", "PV2", 2_000.0),
    )
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    assert validate_solution(result, members, pools, config, versions).is_valid
    assert not validate_solution(result, members, pools, config, ()).is_valid


def test_validation_detects_common_line_and_effective_batch_tampering() -> None:
    """Candidate structural evidence cannot validate itself after tampering."""

    config, members, versions, pools, result = _solved()
    selected = result.selected[0]
    corrupt_candidate = replace(
        selected.candidate,
        common_lines=("WRONG",),
        effective_batch_litres=selected.candidate.effective_batch_litres + 5,
    )
    corrupt = replace(result, selected=(replace(selected, candidate=corrupt_candidate),))
    rules = {
        issue.rule_id
        for issue in validate_solution(corrupt, members, pools, config, versions).issues
    }
    assert {"COMMON_LINE_EVIDENCE", "EFFECTIVE_BATCH_FACTOR"} <= rules
