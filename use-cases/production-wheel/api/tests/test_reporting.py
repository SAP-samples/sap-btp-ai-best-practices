"""Focused deterministic solution artifact tests."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from production_wheel.candidates import (
    CandidateMember,
    ProductionVersion,
    generate_candidate_pools,
)
from production_wheel.optimization import ObjectiveLevel, solve_candidate_pools
from production_wheel.reporting import (
    build_baseline_summary,
    build_matrix_exception_audit_rows,
    build_proposed_subgroups_rows,
    build_scenario_summary_row,
    build_solution_report,
    build_solution_groups_rows,
    write_scenario_artifacts,
)
from production_wheel.schemas import (
    CoverageBasis,
    CoverageMode,
    MatrixMode,
    RunConfig,
    VersionIdentifiers,
)
from production_wheel.solution_validation import validate_solution
from production_wheel.suite_reporting import validate_scenario_artifacts


def _member(
    fini_id: str,
    *,
    demand_litres: float = 1_000.0,
    pallet_litres: float = 100.0,
) -> CandidateMember:
    """Create one compact modeled reporting member."""

    return CandidateMember(
        fini_id=fini_id,
        plant="P1",
        sefi="S1",
        eligible_lines=frozenset({"L1"}),
        demand_litres=demand_litres,
        pallet_litres=pallet_litres,
        package_volume=0.25,
        fixed_pv="PV1",
        baseline_group="BASELINE-1",
        pck_code="PCK-A",
    )


def _bundle():
    """Return one valid solved two-FINI reporting bundle."""

    config = RunConfig(scenario_id="report-test")
    members = (_member("F1"), _member("F2"))
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)
    return config, members, pools, result, validation


def _coverage_bundle():
    """Return one pair whose pallet floor separates all coverage views."""

    config = RunConfig(
        scenario_id="coverage-evidence-test",
        coverage_basis=CoverageBasis.BASE_GROUP,
    )
    members = (
        _member("F1", demand_litres=1_900.0, pallet_litres=100.0),
        _member("F2", demand_litres=100.0, pallet_litres=200.0),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)
    return config, members, pools, result, validation


def _source_rows() -> tuple[dict[str, object], ...]:
    """Create a 1,021-row source-ordered FINI master fixture."""

    modeled = (
        {"source_row": 4, "plant": "P1", "sefi": "S1", "material": "F1", "model_status": "modeled"},
        {"source_row": 5, "plant": "P1", "sefi": "S1", "material": "F2", "model_status": "modeled"},
    )
    others = tuple(
        {
            "source_row": index + 6,
            "plant": "P1",
            "sefi": "",
            "material": f"OUT-{index:04d}",
            "model_status": "excluded" if index == 0 else "out_of_scope",
            "baseline_subgroup": "KEEP-ME",
        }
        for index in range(1_019)
    )
    return modeled + others


def test_primary_output_preserves_1021_rows_order_and_exclusions() -> None:
    """Primary CSV rows retain source order and never invent excluded assignments."""

    config, members, _, result, validation = _bundle()
    source = _source_rows()
    rows = build_proposed_subgroups_rows(
        source, members, validation.groups, result, config
    )
    assert len(rows) == 1_021
    assert [row["source_row"] for row in rows] == [row["source_row"] for row in source]
    assert rows[0]["proposed_subgroup"] == rows[1]["proposed_subgroup"]
    assert all(not row["proposed_subgroup"] for row in rows[2:])
    assert rows[2]["baseline_subgroup"] == "KEEP-ME"


def test_group_and_summary_rows_use_recomputed_validation_evidence() -> None:
    """Companion rows expose independently recomputed KPI and group evidence."""

    config, _, _, result, validation = _bundle()
    rows = build_solution_groups_rows(validation.groups, result, config)
    summary = build_scenario_summary_row(
        validation.groups, result, config, validation
    )
    assert rows[0]["members"] == "F1|F2"
    assert rows[0]["proposed_subgroup"].startswith("PROPOSED-")
    assert summary["validation_status"] == "valid"
    assert summary["modeled_fini_count"] == 2
    assert summary["singleton_group_count"] == 0
    assert summary["j_ch"] == rows[0]["j_ch_contribution"]


def test_coverage_views_and_pallet_member_diagnostics_are_exported() -> None:
    """Both CSV grains and the summary distinguish active and evidence coverage."""

    config, members, _, result, validation = _coverage_bundle()
    proposed = build_proposed_subgroups_rows(
        _source_rows(), members, validation.groups, result, config
    )
    groups = build_solution_groups_rows(validation.groups, result, config)
    summary = build_scenario_summary_row(
        validation.groups, result, config, validation
    )

    assert proposed[0]["active_coverage_basis"] == "BASE_GROUP"
    assert proposed[0]["coverage_days"] == pytest.approx(112.5)
    assert proposed[1]["fini_adjusted_coverage_days"] == pytest.approx(500.0)
    assert groups[0]["base_group_coverage_days"] == pytest.approx(112.5)
    assert groups[0]["adjusted_group_coverage_days"] == pytest.approx(131.875)
    assert groups[0]["worst_fini_coverage_days"] == pytest.approx(500.0)
    assert summary["base_group_target_violation_count"] == 0
    assert summary["adjusted_group_target_violation_count"] == 0
    assert summary["worst_fini_target_violation_count"] == 1
    assert summary["pallet_adjusted_fini_violation_count"] == 1


def test_baseline_summary_recomputes_under_active_scenario_settings() -> None:
    """Baseline deltas use the same coverage and pallet interpretation as proposal."""

    config, members, _, _, _ = _bundle()
    baseline = build_baseline_summary(
        members,
        (ProductionVersion("P1", "S1", "PV1", 1_000.0),),
        config,
    )
    assert baseline["group_count"] == 1
    assert baseline["singleton_group_count"] == 0
    assert baseline["maximum_coverage_days"] == 112.5
    assert baseline["base_group_maximum_coverage_days"] == 112.5
    assert baseline["adjusted_group_maximum_coverage_days"] == 112.5
    assert baseline["matrix_exception_pair_count"] == 0
    assert baseline["matrix_unknown_pair_count"] == 0
    assert baseline["matrix_positive_evidence_pair_count"] == 1
    assert baseline["j_ch"] > 0


def test_write_scenario_artifacts_emits_hashes_and_dependency_free_report(tmp_path) -> None:
    """Writer emits the complete per-scenario CSV/JSON/Markdown evidence set."""

    config, members, pools, result, validation = _bundle()
    paths = write_scenario_artifacts(
        tmp_path,
        _source_rows(),
        members,
        pools,
        result,
        config,
        validation,
        manifest_metadata={"input_hashes": {"snapshot": "abc"}},
        acceptance_summary={
            "acceptance_status": "PARETO_REVIEW_REQUIRED",
            "failed_baseline_guardrails": "j_ch",
        },
    )
    with paths.proposed_subgroups.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1_021
    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    assert manifest["row_counts"]["proposed_subgroups"] == 1_021
    assert manifest["row_counts"]["matrix_exception_audit"] == 0
    assert paths.matrix_exception_audit.read_text(encoding="utf-8").startswith(
        "scenario_id,configuration_id,plant,sefi,proposed_subgroup"
    )
    assert manifest["input_hashes"] == {"snapshot": "abc"}
    assert manifest["acceptance"]["acceptance_status"] == "PARETO_REVIEW_REQUIRED"
    assert "solution_report.md" in manifest["output_hashes"]
    report = paths.solution_report.read_text(encoding="utf-8")
    assert "## KPI summary" in report
    assert "## Independent validation" in report
    assert "Shared/base group coverage is the transcript-grounded default" in report
    assert "Singletons contribute zero" in report
    assert "PARETO_REVIEW_REQUIRED" in report
    with paths.scenario_summary.open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    assert summary["acceptance_status"] == "PARETO_REVIEW_REQUIRED"
    assert summary["failed_baseline_guardrails"] == "j_ch"


def test_constrained_artifacts_persist_limits_and_warm_start(tmp_path) -> None:
    """CSV, manifest, and report expose the in-model governance contract."""

    _, members, _, _, _ = _bundle()
    config = RunConfig(
        scenario_id="baseline-constrained-report-test",
        coverage_mode=CoverageMode.BASELINE_CONSTRAINED_COVERAGE,
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)

    paths = write_scenario_artifacts(
        tmp_path,
        _source_rows(),
        members,
        pools,
        result,
        config,
        validation,
        production_versions=versions,
    )

    with paths.scenario_summary.open(encoding="utf-8", newline="") as handle:
        summary = next(csv.DictReader(handle))
    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    report = paths.solution_report.read_text(encoding="utf-8")
    assert summary["baseline_constraints_applied"] == "1"
    assert summary["baseline_constraint_scope"] == "global_master"
    assert summary["in_model_limit_group_count"] == "1"
    assert summary["warm_start_kind"] == "baseline"
    assert summary["warm_start_group_count"] == "1"
    assert manifest["solver"]["baseline_constraint_limits"]["group_count"] == 1
    assert manifest["solver"]["baseline_constraint_scope"] == "global_master"
    assert manifest["solver"]["warm_start"] == {
        "kind": "baseline",
        "group_count": 1,
    }
    assert "## In-model baseline constraints and start" in report
    assert "Scope: `global_master`" in report
    assert "| group_count | 1 |" in report


def test_customer_flexible_matrix_emits_pair_level_exception_evidence() -> None:
    """Flexible customer-family groups retain every outside-family pair."""

    config = RunConfig(
        scenario_id="customer-flexible-report-test",
        matrix_mode=MatrixMode.FLEXIBLE,
        versions=VersionIdentifiers(
            matrix_version="CUSTOMER_OPERATIONAL_VOLUME_FAMILIES_V1"
        ),
    )
    members = (
        replace(_member("F1"), package_volume=0.75),
        replace(_member("F2"), package_volume=5.0),
    )
    versions = (ProductionVersion("P1", "S1", "PV1", 1_000.0),)
    pools = generate_candidate_pools(members, versions, config)
    result = solve_candidate_pools(pools, members, config)
    validation = validate_solution(result, members, pools, config, versions)

    rows = build_matrix_exception_audit_rows(
        validation.groups, config, members
    )
    summary = build_scenario_summary_row(
        validation.groups, result, config, validation
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "AVOID"
    assert rows[0]["matrix_version"] == config.versions.matrix_version
    assert rows[0]["family_evidence_a"]
    assert rows[0]["family_evidence_b"]
    assert summary["matrix_exception_group_count"] == 1
    assert summary["matrix_exception_pair_count"] == 1
    assert summary["matrix_exception_distinct_volume_pair_count"] == 1

    hard_config = config.model_copy(
        update={
            "scenario_id": "customer-hard-report-test",
            "matrix_mode": MatrixMode.HARD,
        }
    )
    hard_pools = generate_candidate_pools(members, versions, hard_config)
    hard_result = solve_candidate_pools(hard_pools, members, hard_config)
    hard_validation = validate_solution(
        hard_result, members, hard_pools, hard_config, versions
    )
    hard_summary = build_scenario_summary_row(
        hard_validation.groups, hard_result, hard_config, hard_validation
    )

    assert build_matrix_exception_audit_rows(
        hard_validation.groups, hard_config, members
    ) == ()
    assert hard_summary["matrix_exception_group_count"] == 0
    assert hard_summary["matrix_exception_pair_count"] == 0
    assert hard_summary["matrix_exception_distinct_volume_pair_count"] == 0



def test_scenario_validator_rejects_stale_acceptance_schema(tmp_path) -> None:
    """A V2 accepted label is not current V3 business-acceptance evidence."""

    config, members, pools, result, validation = _bundle()
    paths = write_scenario_artifacts(
        tmp_path,
        _source_rows(),
        members,
        pools,
        result,
        config,
        validation,
        acceptance_summary={
            "acceptance_status": "ACCEPTED_BASELINE_GUARDRAILS",
            "failed_baseline_guardrails": "",
        },
    )
    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    manifest["config"]["versions"]["schema_version"] = "PROTOTYPE_SCHEMA_V2"
    paths.run_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    stale = validate_scenario_artifacts(tmp_path)

    assert not stale["business_acceptable"]
    assert not stale["schema_current"]


def test_report_discloses_an_unreached_operations_tier() -> None:
    """Passing numeric guardrails never implies that a skipped J_CH tier was solved."""

    config, _, pools, result, validation = _bundle()
    config = config.model_copy(update={"coverage_mode": CoverageMode.OPERATIONS_FIRST})
    tiered = replace(
        result,
        objective_levels=(
            ObjectiveLevel("target_violation_count", 0.0, "decomposed_feasible_limit"),
            ObjectiveLevel("j_ch", 0.0, "partially_reached"),
        ),
    )
    acceptance = {
        "acceptance_status": "ACCEPTED_BASELINE_GUARDRAILS",
        "failed_baseline_guardrails": "",
    }
    summary = build_scenario_summary_row(
        validation.groups,
        tiered,
        config,
        validation,
        acceptance_summary=acceptance,
    )

    report = build_solution_report(
        summary,
        validation.groups,
        pools,
        tiered,
        config,
        validation,
    )

    assert summary["j_ch_objective_tier_status"] == "partially_reached"
    assert "Do not claim that `J_CH` was fully minimized" in report


def test_gap_accepted_tier_is_reported_as_a_proof_caveat() -> None:
    """A configured MIP-gap stop remains visible in the summary audit."""

    config, _, _, result, validation = _bundle()
    tiered = replace(
        result,
        objective_levels=(
            ObjectiveLevel("target_violation_count", 0.0, "gap_accepted"),
        ),
    )

    summary = build_scenario_summary_row(
        validation.groups,
        tiered,
        config,
        validation,
    )

    assert summary["objective_tier_caveats"] == "target_violation_count"
