"""Write and validate complete baseline/scenario/Pareto demo bundles."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from production_wheel.extraction.common import sha256_file, write_csv, write_json
from production_wheel.optimization import ObjectiveLevel, SolveResult
from production_wheel.reporting import (
    ScenarioArtifactPaths,
    build_baseline_summary,
    build_scenario_summary_row,
    write_scenario_artifacts,
)
from production_wheel.scenarios import (
    BaselineGuardrailAssessment,
    ScenarioOutcome,
    ScenarioSuiteResult,
    assess_baseline_guardrails,
)
from production_wheel.schemas import (
    BaselineAcceptanceStatus,
    CoverageMode,
    RunConfig,
)
from production_wheel.solution_validation import validate_solution

PRIMARY_SCENARIO_ID = "core_operations_first"


@dataclass(frozen=True, slots=True)
class SuiteArtifactPaths:
    """Top-level output paths for one complete demonstrator suite."""

    output_directory: Path
    proposed_subgroups: Path
    solution_groups: Path
    scenario_summary: Path
    pareto_frontier: Path
    matrix_exception_audit: Path
    constraint_audit: Path
    candidate_pool_summary: Path
    validation_issues: Path
    solution_report: Path
    pool_cache: Path
    run_manifest: Path


def _read_single_csv_row(path: Path) -> dict[str, str]:
    """Read the sole row from a per-scenario summary CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one scenario-summary row in {path}")
    return dict(rows[0])


def _failure_row(outcome: ScenarioOutcome) -> dict[str, Any]:
    """Build one suite summary row for a failed or timed-out scenario."""

    return {
        "summary_id": outcome.summary_id,
        "kind": outcome.kind,
        "scenario_id": outcome.scenario_id,
        "configuration_id": outcome.configuration_id or "",
        "status": outcome.status,
        "result_class": outcome.result_class,
        "error_type": outcome.error_type or "",
        "error_message": outcome.error_message or "",
        **_acceptance_fields(outcome.acceptance),
    }


def _acceptance_fields(
    assessment: BaselineGuardrailAssessment | None,
) -> dict[str, Any]:
    """Flatten one typed baseline assessment into suite-summary evidence."""

    if assessment is None:
        return {"acceptance_status": "", "failed_baseline_guardrails": ""}
    return assessment.as_dict()


def _write_failure_directory(directory: Path, outcome: ScenarioOutcome) -> None:
    """Preserve machine-readable failure evidence for a partial suite."""

    write_no_incumbent_artifacts(directory, outcome)


def write_no_incumbent_artifacts(
    directory: Path,
    outcome: ScenarioOutcome,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write complete, hash-audited evidence when no solution is available.

    Args:
        directory: New or existing scenario evidence directory.
        outcome: Failed, timed-out, or no-incumbent optimizer outcome.
        source_manifest: Optional extraction provenance.

    Returns:
        The single summary row written to ``scenario_summary.csv``.

    This records an empty exception audit and proof metadata, but deliberately
    does not create proposal files or claim independent solution validation.
    """

    directory.mkdir(parents=True, exist_ok=True)
    solve = outcome.solve_result
    config = outcome.config
    baseline_limits = (
        solve.baseline_constraint_limits.as_dict()
        if solve and solve.baseline_constraint_limits is not None
        else {}
    )
    row = {
        **_failure_row(outcome),
        "has_incumbent": 0,
        "termination_condition": solve.termination_condition if solve else "",
        "candidate_count": solve.candidate_count if solve else "",
        "candidate_pool_completeness": solve.pool_completeness if solve else "",
        "baseline_constraints_applied": int(
            bool(solve and solve.baseline_constraint_limits is not None)
        ),
        "baseline_constraint_scope": (
            solve.baseline_constraint_scope if solve else "none"
        ),
        "warm_start_kind": solve.warm_start_kind if solve else "none",
        "warm_start_group_count": solve.warm_start_group_count if solve else 0,
        "matrix_mode": config.matrix_mode.value if config else "",
        "matrix_version": config.versions.matrix_version if config else "",
        "schema_version": config.versions.schema_version if config else "",
        "matrix_exception_group_count": 0,
        "matrix_exception_pair_count": 0,
        "matrix_exception_distinct_volume_pair_count": 0,
        "validation_status": "not_applicable_no_incumbent",
        **{
            f"in_model_limit_{name}": value
            for name, value in baseline_limits.items()
        },
    }
    summary = directory / "scenario_summary.csv"
    matrix_audit = directory / "matrix_exception_audit.csv"
    write_csv(summary, (row,))
    write_csv(
        matrix_audit,
        (),
        (
            "scenario_id",
            "configuration_id",
            "plant",
            "sefi",
            "proposed_subgroup",
            "material_a",
            "material_b",
            "volume_a",
            "volume_b",
            "status",
            "matrix_version",
            "family_evidence_a",
            "family_evidence_b",
            "rationale",
        ),
    )
    manifest = {
        "evidence_class": "no_incumbent",
        "scenario_id": outcome.scenario_id,
        "configuration_id": outcome.configuration_id,
        "config": config.model_dump(mode="json") if config else {},
        "structural_fingerprint": outcome.structural_fingerprint,
        "pool_hashes": outcome.pool_hashes,
        "solver": {
            "status": outcome.status,
            "result_class": outcome.result_class,
            "has_incumbent": False,
            "termination_condition": solve.termination_condition if solve else "",
            "candidate_count": solve.candidate_count if solve else None,
            "pool_completeness": solve.pool_completeness if solve else None,
            "baseline_constraint_scope": (
                solve.baseline_constraint_scope if solve else "none"
            ),
            "baseline_constraint_limits": (
                baseline_limits or None
            ),
            "warm_start": {
                "kind": solve.warm_start_kind if solve else "none",
                "group_count": solve.warm_start_group_count if solve else 0,
            },
            "objective_levels": [
                {
                    "name": level.name,
                    "value": level.value,
                    "termination": level.termination_condition,
                    "best_bound": level.best_bound,
                    "relative_gap": level.relative_gap,
                }
                for level in (solve.objective_levels if solve else ())
            ],
            "block_evidence": [
                {
                    "block": evidence.block_key,
                    "status": evidence.status,
                    "termination_condition": evidence.termination_condition,
                    "result_class": evidence.result_class,
                    "candidate_count": evidence.candidate_count,
                    "primary_objective": evidence.primary_objective,
                    "primary_best_bound": evidence.primary_best_bound,
                    "primary_relative_gap": evidence.primary_relative_gap,
                }
                for evidence in (solve.block_evidence if solve else ())
            ],
        },
        "validation": {
            "valid": False,
            "status": "not_applicable_no_incumbent",
        },
        "acceptance": _acceptance_fields(outcome.acceptance),
        "source_extraction": dict(source_manifest or {}),
        "output_hashes": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (summary, matrix_audit)
        },
    }
    write_json(directory / "run_manifest.json", manifest)
    return row


def _write_outcome(
    directory: Path,
    outcome: ScenarioOutcome,
    suite: ScenarioSuiteResult,
    source_manifest: Mapping[str, Any] | None,
) -> tuple[ScenarioArtifactPaths | None, dict[str, Any]]:
    """Validate and write one baseline/optimizer scenario."""

    if (
        outcome.solve_result is None
        or not outcome.solve_result.has_incumbent
        or outcome.config is None
        or not outcome.pools
    ):
        write_no_incumbent_artifacts(directory, outcome, source_manifest)
        return None, _failure_row(outcome)
    validation = validate_solution(
        outcome.solve_result,
        suite.inputs.members,
        outcome.pools,
        outcome.config,
        suite.inputs.production_versions,
        suite.inputs.baseline_evidence_members or suite.inputs.members,
    )
    paths = write_scenario_artifacts(
        directory,
        suite.inputs.fini_rows,
        suite.inputs.members,
        outcome.pools,
        outcome.solve_result,
        outcome.config,
        validation,
        manifest_metadata={"source_extraction": dict(source_manifest or {})},
        production_versions=suite.inputs.production_versions,
        acceptance_summary=_acceptance_fields(outcome.acceptance),
        baseline_evidence_members=(
            suite.inputs.baseline_evidence_members or suite.inputs.members
        ),
    )
    row: dict[str, Any] = _read_single_csv_row(paths.scenario_summary)
    row["summary_id"] = outcome.summary_id
    row["kind"] = outcome.kind
    row.update(_acceptance_fields(outcome.acceptance))
    return paths, row


def _pareto_result(
    point: Any, pools: tuple[Any, ...], *, duplicate: bool = False
) -> SolveResult:
    """Adapt one disclosed portfolio point to the standard reporting contract."""

    levels = (
        ObjectiveLevel(
            "maximum_coverage_days",
            float(point.maximum_coverage_days),
            point.termination_condition,
            float(point.maximum_coverage_days) if point.status == "optimal" else None,
            point.relative_gap,
        ),
        ObjectiveLevel(
            "j_ch_epsilon",
            float(point.j_ch),
            point.termination_condition,
        ),
    )
    completeness = (
        "complete" if all(pool.completeness == "complete" for pool in pools) else "restricted"
    )
    return SolveResult(
        status="degenerate_duplicate" if duplicate else point.status,
        termination_condition=point.termination_condition,
        has_incumbent=bool(point.selected),
        result_class=(
            "portfolio-degenerate-duplicate" if duplicate else point.result_class
        ),
        selected=tuple(point.selected),
        objective_levels=levels,
        primary_objective=float(point.maximum_coverage_days),
        primary_best_bound=(
            float(point.maximum_coverage_days) if point.status == "optimal" else None
        ),
        primary_relative_gap=point.relative_gap,
        incumbent_objective=float(point.j_ch),
        best_bound=None,
        relative_gap=point.relative_gap,
        candidate_count=sum(len(pool.candidates) for pool in pools),
        pool_completeness=completeness,
        solver_method=point.solver_method,
    )


def _copy_primary(paths: ScenarioArtifactPaths, output_directory: Path) -> None:
    """Copy the designated scenario artifacts to stable top-level names."""

    for source in (
        paths.proposed_subgroups,
        paths.solution_groups,
        paths.matrix_exception_audit,
        paths.constraint_audit,
        paths.candidate_pool_summary,
        paths.validation_issues,
        paths.solution_report,
    ):
        shutil.copyfile(source, output_directory / source.name)


def write_suite_artifacts(
    output_directory: Path,
    suite: ScenarioSuiteResult,
    source_manifest: Mapping[str, Any] | None = None,
) -> SuiteArtifactPaths:
    """Write the full 25-result suite and stable primary-scenario exports.

    Args:
        output_directory: New output directory for this suite.
        suite: Baseline, 19 scenario outcomes, five portfolio points, and inputs.
        source_manifest: Optional extraction manifest retained as provenance.

    Returns:
        Paths to all stable top-level artifacts.
    """

    output_directory.mkdir(parents=True, exist_ok=False)
    scenario_root = output_directory / "scenarios"
    scenario_root.mkdir()
    summary_rows: list[dict[str, Any]] = []
    primary_paths: ScenarioArtifactPaths | None = None
    for outcome in (suite.baseline, *suite.scenarios):
        paths, row = _write_outcome(
            scenario_root / outcome.scenario_id,
            outcome,
            suite,
            source_manifest,
        )
        summary_rows.append(row)
        if outcome.scenario_id == PRIMARY_SCENARIO_ID:
            primary_paths = paths

    core = next(
        (
            outcome
            for outcome in suite.scenarios
            if outcome.scenario_id == "core_max" and outcome.config and outcome.pools
        ),
        None,
    )
    pareto_rows: list[dict[str, Any]] = []
    seen_portfolios: set[str] = set()
    for point in suite.pareto_points:
        scenario_id = f"pareto_{point.point_index:02d}"
        if core is None or not point.selected or point.maximum_coverage_days is None or point.j_ch is None:
            row = {
                "summary_id": scenario_id,
                "kind": "pareto",
                "scenario_id": scenario_id,
                "status": point.status,
                "result_class": point.result_class,
                "acceptance_status": BaselineAcceptanceStatus.NOT_ACCEPTABLE.value,
                "failed_baseline_guardrails": "pareto_point_unavailable",
                "epsilon_j_ch": point.epsilon_j_ch,
                "frontier_label": point.frontier_label,
                "error_type": point.error_type or "",
                "error_message": point.error_message or "",
            }
            write_csv(scenario_root / scenario_id / "scenario_summary.csv", (row,))
            write_json(scenario_root / scenario_id / "run_manifest.json", row)
            summary_rows.append(row)
            pareto_rows.append(row)
            continue
        duplicate = bool(
            point.portfolio_hash and point.portfolio_hash in seen_portfolios
        )
        if point.portfolio_hash:
            seen_portfolios.add(point.portfolio_hash)
        config = core.config.model_copy(
            update={"scenario_id": scenario_id, "coverage_mode": CoverageMode.PARETO}
        )
        result = _pareto_result(point, core.pools, duplicate=duplicate)
        acceptance = assess_baseline_guardrails(
            suite.inputs, core.pools, config, result
        )
        validation = validate_solution(
            result,
            suite.inputs.members,
            core.pools,
            config,
            suite.inputs.production_versions,
            suite.inputs.baseline_evidence_members or suite.inputs.members,
        )
        baseline = build_baseline_summary(
            suite.inputs.members,
            suite.inputs.production_versions,
            config,
            suite.inputs.baseline_evidence_members or suite.inputs.members,
        )
        paths = write_scenario_artifacts(
            scenario_root / scenario_id,
            suite.inputs.fini_rows,
            suite.inputs.members,
            core.pools,
            result,
            config,
            validation,
            baseline_summary=baseline,
            manifest_metadata={
                "frontier_label": point.frontier_label,
                "epsilon_j_ch": point.epsilon_j_ch,
                "source_scenarios": point.source_scenarios,
            },
            production_versions=suite.inputs.production_versions,
            acceptance_summary=_acceptance_fields(acceptance),
            baseline_evidence_members=(
                suite.inputs.baseline_evidence_members or suite.inputs.members
            ),
        )
        row = _read_single_csv_row(paths.scenario_summary)
        row.update(
            {
                "summary_id": scenario_id,
                "kind": "pareto",
                "epsilon_j_ch": point.epsilon_j_ch,
                "frontier_label": point.frontier_label,
                "portfolio_hash": point.portfolio_hash or "",
                "source_scenarios": "|".join(point.source_scenarios),
                "degenerate_duplicate": int(duplicate),
                **_acceptance_fields(acceptance),
            }
        )
        summary_rows.append(row)
        pareto_rows.append(row)

    if primary_paths is not None:
        _copy_primary(primary_paths, output_directory)

    scenario_summary = output_directory / "scenario_summary.csv"
    pareto_frontier = output_directory / "pareto_frontier.csv"
    pool_cache = output_directory / "pool_cache.csv"
    write_csv(scenario_summary, summary_rows)
    write_csv(pareto_frontier, pareto_rows)
    write_csv(
        pool_cache,
        (
            {
                "structural_fingerprint": record.structural_fingerprint,
                "block_count": record.block_count,
                "scenario_ids": "|".join(record.scenario_ids),
                "pool_hashes": "|".join(record.pool_hashes),
            }
            for record in suite.pool_cache
        ),
    )
    stable_names = (
        "proposed_subgroups.csv",
        "solution_groups.csv",
        "matrix_exception_audit.csv",
        "constraint_audit.csv",
        "candidate_pool_summary.csv",
        "validation_issues.csv",
        "solution_report.md",
        "scenario_summary.csv",
        "pareto_frontier.csv",
        "pool_cache.csv",
    )
    output_hashes = {
        name: {
            "sha256": sha256_file(output_directory / name),
            "size_bytes": (output_directory / name).stat().st_size,
        }
        for name in stable_names
        if (output_directory / name).is_file()
    }
    primary_summary = next(
        (
            row
            for row in summary_rows
            if row.get("scenario_id") == PRIMARY_SCENARIO_ID
        ),
        {},
    )
    manifest = {
        "suite": "demo",
        "primary_scenario_id": PRIMARY_SCENARIO_ID,
        "primary_schema_version": next(
            (
                outcome.config.versions.schema_version
                for outcome in suite.scenarios
                if outcome.scenario_id == PRIMARY_SCENARIO_ID
                and outcome.config is not None
            ),
            "",
        ),
        "primary_acceptance_status": primary_summary.get("acceptance_status", ""),
        "primary_failed_baseline_guardrails": primary_summary.get(
            "failed_baseline_guardrails", ""
        ),
        "partial": suite.partial,
        "elapsed_seconds": suite.elapsed_seconds,
        "source_extraction": dict(source_manifest or {}),
        "counts": {
            "baseline": 1,
            "optimizer_configurations": len(suite.scenarios),
            "pareto_points": len(suite.pareto_points),
            "summary_results": len(summary_rows),
            "pool_fingerprints": len(suite.pool_cache),
        },
        "scenario_aliases": {
            "target_base_group_minimum_only": "core_target_band"
        },
        "implementation": {
            "package_version": "0.1.0",
            "source_state": "working_tree_then_committed_checkpoint",
        },
        "outcomes": [
            {
                "summary_id": row.get("summary_id"),
                "kind": row.get("kind"),
                "status": row.get("status"),
                "result_class": row.get("result_class"),
                "acceptance_status": row.get("acceptance_status"),
            }
            for row in summary_rows
        ],
        "outputs": output_hashes,
    }
    run_manifest = output_directory / "run_manifest.json"
    write_json(run_manifest, manifest)
    return SuiteArtifactPaths(
        output_directory=output_directory,
        proposed_subgroups=output_directory / "proposed_subgroups.csv",
        solution_groups=output_directory / "solution_groups.csv",
        scenario_summary=scenario_summary,
        pareto_frontier=pareto_frontier,
        matrix_exception_audit=output_directory / "matrix_exception_audit.csv",
        constraint_audit=output_directory / "constraint_audit.csv",
        candidate_pool_summary=output_directory / "candidate_pool_summary.csv",
        validation_issues=output_directory / "validation_issues.csv",
        solution_report=output_directory / "solution_report.md",
        pool_cache=pool_cache,
        run_manifest=run_manifest,
    )


def validate_suite_artifacts(output_directory: Path) -> dict[str, Any]:
    """Validate stable suite files, hashes, row counts, and primary dispositions."""

    errors: list[str] = []
    manifest_path = output_directory / "run_manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["run_manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_status = BaselineAcceptanceStatus.ACCEPTED_BASELINE_GUARDRAILS.value
    expected_schema = RunConfig().versions.schema_version
    primary_schema = manifest.get("primary_schema_version", "")
    primary_acceptance = manifest.get("primary_acceptance_status", "")
    business_error = "primary scenario is not baseline-guardrail accepted"
    schema_error = "primary scenario uses a stale acceptance schema"
    if primary_acceptance != accepted_status:
        errors.append(business_error)
    if primary_schema != expected_schema:
        errors.append(schema_error)
    if manifest.get("partial"):
        errors.append("suite manifest is partial")
    required = {
        "proposed_subgroups.csv",
        "solution_groups.csv",
        "scenario_summary.csv",
        "pareto_frontier.csv",
        "matrix_exception_audit.csv",
        "constraint_audit.csv",
        "candidate_pool_summary.csv",
        "validation_issues.csv",
        "solution_report.md",
        "pool_cache.csv",
    }
    missing_required = sorted(
        name for name in required if not (output_directory / name).is_file()
    )
    if missing_required:
        errors.append(f"missing required suite artifacts: {missing_required}")
    for name, evidence in manifest.get("outputs", {}).items():
        path = output_directory / name
        if not path.is_file():
            errors.append(f"missing output: {name}")
        elif sha256_file(path) != evidence.get("sha256"):
            errors.append(f"hash mismatch: {name}")
    proposed = output_directory / "proposed_subgroups.csv"
    if proposed.is_file():
        with proposed.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1021:
            errors.append(f"primary proposed_subgroups row count is {len(rows)}, expected 1021")
        modeled = [row for row in rows if row.get("model_status") == "modeled"]
        nonmodeled = [row for row in rows if row.get("model_status") != "modeled"]
        if len(modeled) != 794 or any(not row.get("proposed_subgroup") for row in modeled):
            errors.append("primary modeled proposal coverage is not 794/794")
        if any(row.get("proposed_subgroup") for row in nonmodeled):
            errors.append("excluded/out-of-scope rows contain proposed subgroup values")
    else:
        errors.append("primary proposed_subgroups.csv is missing")
    summary = output_directory / "scenario_summary.csv"
    if summary.is_file():
        with summary.open(newline="", encoding="utf-8") as handle:
            if len(list(csv.DictReader(handle))) != 25:
                errors.append("scenario_summary.csv must contain 25 results")
    else:
        errors.append("scenario_summary.csv is missing")
    validation = output_directory / "validation_issues.csv"
    if validation.is_file():
        with validation.open(newline="", encoding="utf-8") as handle:
            if any(row.get("severity") == "error" for row in csv.DictReader(handle)):
                errors.append("primary independent validation contains error findings")
    else:
        errors.append("primary validation_issues.csv is missing")
    scenarios = output_directory / "scenarios"
    if scenarios.is_dir():
        for scenario_directory in sorted(path for path in scenarios.iterdir() if path.is_dir()):
            scenario_manifest = scenario_directory / "run_manifest.json"
            if not scenario_manifest.is_file():
                errors.append(f"missing scenario manifest: {scenario_directory.name}")
                continue
            evidence = json.loads(scenario_manifest.read_text(encoding="utf-8"))
            for name, output in evidence.get("output_hashes", {}).items():
                path = scenario_directory / name
                if not path.is_file() or sha256_file(path) != output.get("sha256"):
                    errors.append(
                        f"scenario artifact hash mismatch: {scenario_directory.name}/{name}"
                    )
    else:
        errors.append("scenarios directory is missing")
    completion_error = "suite manifest is partial"
    integrity_errors = [
        error
        for error in errors
        if error not in {completion_error, business_error, schema_error}
    ]
    return {
        "valid": not errors,
        "integrity_valid": not integrity_errors,
        "complete": completion_error not in errors,
        "business_acceptable": not any(
            error in errors for error in {business_error, schema_error}
        ),
        "primary_acceptance_status": primary_acceptance,
        "primary_schema_version": primary_schema,
        "expected_schema_version": expected_schema,
        "errors": errors,
    }


def validate_scenario_artifacts(output_directory: Path) -> dict[str, Any]:
    """Validate one written scenario bundle independently of suite cardinality."""

    errors: list[str] = []
    manifest_path = output_directory / "run_manifest.json"
    if not manifest_path.is_file():
        return {
            "valid": False,
            "integrity_valid": False,
            "complete": False,
            "solution_available": False,
            "errors": ["run_manifest.json is missing"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_schema = RunConfig().versions.schema_version
    observed_schema = (
        manifest.get("config", {}).get("versions", {}).get("schema_version", "")
    )
    schema_current = observed_schema == expected_schema
    acceptance_status = manifest.get("acceptance", {}).get(
        "acceptance_status", ""
    )
    if manifest.get("evidence_class") == "no_incumbent":
        for name, evidence in manifest.get("output_hashes", {}).items():
            path = output_directory / name
            if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
                errors.append(f"artifact hash mismatch: {name}")
        required = {"scenario_summary.csv", "matrix_exception_audit.csv"}
        missing = sorted(
            name for name in required if not (output_directory / name).is_file()
        )
        if missing:
            errors.append(f"missing no-incumbent evidence artifacts: {missing}")
        if manifest.get("solver", {}).get("has_incumbent") is not False:
            errors.append("no-incumbent evidence incorrectly declares a solution")
        matrix_audit = output_directory / "matrix_exception_audit.csv"
        if matrix_audit.is_file():
            with matrix_audit.open(newline="", encoding="utf-8") as handle:
                if list(csv.DictReader(handle)):
                    errors.append("no-incumbent matrix exception audit must be empty")
        if not schema_current:
            errors.append("no-incumbent evidence uses a stale schema")
        return {
            "valid": False,
            "integrity_valid": not errors,
            "complete": not errors,
            "solution_available": False,
            "business_acceptable": False,
            "acceptance_status": acceptance_status,
            "schema_current": schema_current,
            "schema_version": observed_schema,
            "expected_schema_version": expected_schema,
            "errors": errors,
        }
    business_acceptance_assessed = (
        acceptance_status
        == BaselineAcceptanceStatus.ACCEPTED_BASELINE_GUARDRAILS.value
    )
    business_acceptable = business_acceptance_assessed and schema_current
    for name, evidence in manifest.get("output_hashes", {}).items():
        path = output_directory / name
        if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
            errors.append(f"artifact hash mismatch: {name}")
    if not manifest.get("validation", {}).get("valid"):
        errors.append("independent solution validation is not valid")
    proposed = output_directory / "proposed_subgroups.csv"
    if proposed.is_file():
        with proposed.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_rows = manifest.get("row_counts", {}).get("proposed_subgroups")
        if expected_rows is not None and len(rows) != expected_rows:
            errors.append(
                f"proposed_subgroups row count is {len(rows)}, expected {expected_rows}"
            )
        modeled = [row for row in rows if row.get("model_status") == "modeled"]
        if any(not row.get("proposed_subgroup") for row in modeled):
            errors.append("one or more modeled FINIs lack a proposed subgroup")
        if any(
            row.get("proposed_subgroup")
            for row in rows
            if row.get("model_status") != "modeled"
        ):
            errors.append("excluded or out-of-scope FINIs contain proposed groups")
    else:
        errors.append("proposed_subgroups.csv is missing")
    return {
        "valid": not errors,
        "integrity_valid": not errors,
        "complete": not errors,
        "solution_available": True,
        "business_acceptable": business_acceptable,
        "business_acceptance_assessed": business_acceptance_assessed,
        "acceptance_status": acceptance_status,
        "schema_current": schema_current,
        "schema_version": observed_schema,
        "expected_schema_version": expected_schema,
        "errors": errors,
    }


def normalize_written_suite(output_directory: Path) -> dict[str, Any]:
    """Normalize degenerate Pareto and no-incumbent labels in a written bundle.

    This operates only on reporting metadata. It never changes proposed
    memberships, solver coefficients, or KPI values.
    """

    summary_path = output_directory / "scenario_summary.csv"
    pareto_path = output_directory / "pareto_frontier.csv"
    manifest_path = output_directory / "run_manifest.json"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with pareto_path.open(newline="", encoding="utf-8") as handle:
        pareto_rows = list(csv.DictReader(handle))
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for row in pareto_rows:
        portfolio_hash = row.get("portfolio_hash", "")
        duplicate = bool(portfolio_hash and portfolio_hash in seen)
        if portfolio_hash:
            seen.add(portfolio_hash)
        row["degenerate_duplicate"] = "1" if duplicate else "0"
        if duplicate:
            row["status"] = "degenerate_duplicate"
            row["result_class"] = "portfolio-degenerate-duplicate"
            duplicate_ids.add(row.get("summary_id") or row.get("scenario_id", ""))
    by_id = {row.get("summary_id") or row.get("scenario_id", ""): row for row in pareto_rows}
    for row in summary_rows:
        key = row.get("summary_id") or row.get("scenario_id", "")
        if key in by_id:
            row.update(by_id[key])
    write_csv(summary_path, summary_rows)
    write_csv(pareto_path, pareto_rows)

    for row in summary_rows:
        scenario_id = row.get("scenario_id", "")
        scenario_directory = output_directory / "scenarios" / scenario_id
        scenario_manifest_path = scenario_directory / "run_manifest.json"
        if not scenario_manifest_path.is_file():
            continue
        scenario_manifest = json.loads(
            scenario_manifest_path.read_text(encoding="utf-8")
        )
        if row.get("status") == "no_incumbent":
            scenario_manifest["artifact_status"] = "no_incumbent_evidence_only"
        if scenario_id in duplicate_ids:
            scenario_manifest["artifact_status"] = "degenerate_pareto_duplicate"
            scenario_manifest.setdefault("solver", {})["status"] = row["status"]
            scenario_manifest["solver"]["result_class"] = row["result_class"]
            scenario_summary_path = scenario_directory / "scenario_summary.csv"
            if scenario_summary_path.is_file():
                write_csv(scenario_summary_path, (row,))
                scenario_manifest.setdefault("output_hashes", {})[
                    "scenario_summary.csv"
                ] = {
                    "sha256": sha256_file(scenario_summary_path),
                    "size_bytes": scenario_summary_path.stat().st_size,
                }
        write_json(scenario_manifest_path, scenario_manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenario_aliases"] = {
        "target_base_group_minimum_only": "core_target_band"
    }
    manifest["implementation"] = {
        "package_version": "0.1.0",
        "source_state": "working_tree_then_committed_checkpoint",
    }
    manifest["outcomes"] = [
        {
            "summary_id": row.get("summary_id"),
            "kind": row.get("kind"),
            "status": row.get("status"),
            "result_class": row.get("result_class"),
        }
        for row in summary_rows
    ]
    for path in (summary_path, pareto_path):
        manifest.setdefault("outputs", {})[path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    write_json(manifest_path, manifest)
    return {
        "degenerate_pareto_duplicates": len(duplicate_ids),
        "no_incumbent_scenarios": sum(
            row.get("status") == "no_incumbent" for row in summary_rows
        ),
    }
