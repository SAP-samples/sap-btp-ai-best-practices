"""Focused tests for complete suite artifact orchestration."""

from __future__ import annotations

import csv
import json

from production_wheel.candidates import CandidateMember, ProductionVersion
from production_wheel.scenarios import CanonicalInputs, ScenarioOutcome, run_scenario_suite
from production_wheel.schemas import RunConfig
from production_wheel.suite_reporting import (
    validate_scenario_artifacts,
    validate_suite_artifacts,
    write_no_incumbent_artifacts,
    write_suite_artifacts,
)


def test_suite_writer_emits_25_results_and_designated_primary(tmp_path) -> None:
    """A small complete suite writes every summary and the stable primary files."""

    members = (
        CandidateMember(
            "F1",
            "P1",
            "S1",
            frozenset({"1"}),
            1_000,
            25,
            0.25,
            "PV1",
            None,
            "B1",
            "PCK1",
        ),
        CandidateMember(
            "F2",
            "P1",
            "S1",
            frozenset({"1"}),
            1_000,
            25,
            0.25,
            "PV1",
            None,
            "B2",
            "PCK1",
        ),
    )
    inputs = CanonicalInputs(
        extracted_directory=tmp_path,
        fini_rows=(
            {"plant": "P1", "sefi": "S1", "material": "F1", "model_status": "modeled"},
            {"plant": "P1", "sefi": "S1", "material": "F2", "model_status": "modeled"},
            {"plant": "P1", "sefi": "S1", "material": "X", "model_status": "excluded"},
        ),
        members=members,
        production_versions=(ProductionVersion("P1", "S1", "PV1", 100),),
    )
    suite = run_scenario_suite(inputs)

    paths = write_suite_artifacts(tmp_path / "suite", suite)

    with paths.scenario_summary.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 25
    assert all("acceptance_status" in row for row in summaries)
    assert next(
        row for row in summaries if row["scenario_id"] == "core_target_band"
    )["acceptance_status"]
    with paths.pareto_frontier.open(newline="", encoding="utf-8") as handle:
        pareto = list(csv.DictReader(handle))
    assert len(pareto) == 5
    assert sum(
        row["result_class"] == "portfolio-degenerate-duplicate" for row in pareto
    ) >= 1
    with paths.proposed_subgroups.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["proposed_subgroup"] for row in rows[:2])
    assert not rows[2]["proposed_subgroup"]
    assert paths.matrix_exception_audit.is_file()
    manifest = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    assert manifest["counts"] == {
        "baseline": 1,
        "optimizer_configurations": 19,
        "pareto_points": 5,
        "summary_results": 25,
        "pool_fingerprints": 8,
    }
    assert manifest["primary_scenario_id"] == "core_operations_first"
    assert manifest["primary_schema_version"] == "PROTOTYPE_SCHEMA_V9"
    assert (
        manifest["primary_acceptance_status"]
        == "ACCEPTED_BASELINE_GUARDRAILS"
    )
    assert manifest["scenario_aliases"] == {
        "target_base_group_minimum_only": "core_target_band"
    }
    assert all("acceptance_status" in row for row in manifest["outcomes"])


def test_suite_validator_separates_integrity_from_partial_completion(tmp_path) -> None:
    """A partial manifest returns nonzero completeness without inventing corruption."""

    output = tmp_path / "suite"
    output.mkdir()
    required = (
        "proposed_subgroups.csv",
        "solution_groups.csv",
        "matrix_exception_audit.csv",
        "scenario_summary.csv",
        "pareto_frontier.csv",
        "constraint_audit.csv",
        "candidate_pool_summary.csv",
        "validation_issues.csv",
        "solution_report.md",
        "pool_cache.csv",
    )
    for name in required:
        (output / name).write_text("", encoding="utf-8")
    with (output / "proposed_subgroups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model_status", "proposed_subgroup"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"model_status": "modeled", "proposed_subgroup": "G"}
                for _ in range(794)
            ]
            + [
                {"model_status": "out_of_scope", "proposed_subgroup": ""}
                for _ in range(227)
            ]
        )
    with (output / "scenario_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["summary_id"])
        writer.writeheader()
        writer.writerows({"summary_id": str(index)} for index in range(25))
    (output / "validation_issues.csv").write_text(
        "severity\n", encoding="utf-8"
    )
    (output / "scenarios").mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "partial": True,
                "outputs": {},
                "primary_acceptance_status": "ACCEPTED_BASELINE_GUARDRAILS",
                    "primary_schema_version": "PROTOTYPE_SCHEMA_V9",
            }
        ),
        encoding="utf-8",
    )

    result = validate_suite_artifacts(output)

    assert not result["valid"]
    assert result["integrity_valid"]
    assert not result["complete"]
    assert result["business_acceptable"]
    assert result["errors"] == ["suite manifest is partial"]


def test_no_incumbent_bundle_is_complete_evidence_without_a_solution(
    tmp_path,
) -> None:
    """No-incumbent CLI evidence is auditable but never business-valid."""

    config = RunConfig(scenario_id="strict_timeout")
    outcome = ScenarioOutcome(
        summary_id=config.scenario_id,
        kind="scenario",
        scenario_id=config.scenario_id,
        configuration_id=config.configuration_id(),
        config=config,
        status="no_incumbent",
        result_class="no-incumbent-within-limit",
    )
    output = tmp_path / "strict-timeout"

    write_no_incumbent_artifacts(output, outcome)
    result = validate_scenario_artifacts(output)

    assert not result["valid"]
    assert result["integrity_valid"]
    assert result["complete"]
    assert not result["solution_available"]
    assert not result["business_acceptable"]
    assert result["errors"] == []
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["solver"]["baseline_constraint_scope"] == "none"
    assert manifest["solver"]["baseline_constraint_limits"] is None
    assert manifest["solver"]["warm_start"] == {
        "kind": "none",
        "group_count": 0,
    }
    with (output / "matrix_exception_audit.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_suite_validator_separates_business_acceptance_from_integrity(
    tmp_path,
) -> None:
    """An unaccepted primary is intact evidence but not a valid proposal suite."""

    output = tmp_path / "suite"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "partial": False,
                "outputs": {},
                "primary_acceptance_status": "PARETO_REVIEW_REQUIRED",
            }
        ),
        encoding="utf-8",
    )

    result = validate_suite_artifacts(output)

    assert not result["valid"]
    assert not result["business_acceptable"]
    assert "primary scenario is not baseline-guardrail accepted" in result["errors"]


def test_suite_validator_rejects_stale_primary_acceptance_schema(tmp_path) -> None:
    """An old accepted label is never current business-acceptance evidence."""

    output = tmp_path / "suite"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "partial": False,
                "outputs": {},
                "primary_acceptance_status": "ACCEPTED_BASELINE_GUARDRAILS",
                "primary_schema_version": "PROTOTYPE_SCHEMA_V2",
            }
        ),
        encoding="utf-8",
    )

    result = validate_suite_artifacts(output)

    assert not result["business_acceptable"]
    assert "primary scenario uses a stale acceptance schema" in result["errors"]
