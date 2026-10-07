"""Write and verify baseline-independent greenfield Pareto bundles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from production_wheel.candidates import CandidateMember, ProductionVersion
from production_wheel.extraction.common import sha256_file, write_csv, write_json
from production_wheel.pareto import GreenfieldFrontierResult
from production_wheel.reporting import (
    build_baseline_summary,
    build_scenario_summary_row,
    write_scenario_artifacts,
)
from production_wheel.schemas import GreenfieldValidationStatus
from production_wheel.solution_validation import validate_solution
from production_wheel.suite_reporting import validate_scenario_artifacts


@dataclass(frozen=True, slots=True)
class GreenfieldFrontierArtifactPaths:
    """Paths emitted for one complete greenfield frontier run."""

    output_directory: Path
    frontier_summary: Path
    block_options: Path
    block_solve_audit: Path
    block_session_metrics: Path
    block_failures: Path
    global_solve_audit: Path
    solution_report: Path
    run_manifest: Path


def _acceptance(valid: bool) -> dict[str, str]:
    """Return the baseline-free independent validation status mapping."""

    return {
        "acceptance_status": (
            GreenfieldValidationStatus.VALID_PARETO_POINT.value
            if valid
            else GreenfieldValidationStatus.INVALID_PARETO_POINT.value
        ),
        "acceptance_evidence_source": "independent_greenfield_validation",
        "failed_baseline_guardrails": "",
    }


def _block_option_rows(result: GreenfieldFrontierResult) -> tuple[dict[str, Any], ...]:
    """Flatten nondominated block partitions for audit CSV output."""

    return tuple(
        {
            "plant": option.block_key[0],
            "sefi": option.block_key[1],
            "partition_hash": option.partition_hash,
            "source": option.source,
            "proof_scope": option.proof_scope,
            "status": option.status,
            "termination_condition": option.termination_condition,
            "relative_gap": option.relative_gap,
            "group_count": len(option.selected),
            "coverage_numerator": option.coverage_numerator,
            "block_demand_litres": option.demand_litres,
            "demand_weighted_mean_coverage_days": (
                option.coverage_numerator / option.demand_litres
            ),
            "j_ch": option.j_ch,
            "candidate_hashes": "|".join(
                item.candidate.candidate_hash for item in option.selected
            ),
        }
        for option in result.block_options
    )


def _block_solve_audit_rows(
    result: GreenfieldFrontierResult,
) -> tuple[dict[str, Any], ...]:
    """Flatten every anchor, epsilon request, and interval stop."""

    rows = []
    for audit in result.block_solve_audits:
        row = asdict(audit)
        row.pop("block_key")
        rows.append(
            {
                "plant": audit.block_key[0],
                "sefi": audit.block_key[1],
                **row,
            }
        )
    return tuple(rows)


def _block_session_rows(
    result: GreenfieldFrontierResult,
) -> tuple[dict[str, Any], ...]:
    """Flatten one persistent-model timing row per completed block."""

    return tuple(
        {
            "plant": audit.block_key[0],
            "sefi": audit.block_key[1],
            "candidate_count": audit.candidate_count,
            **asdict(audit.metrics),
        }
        for audit in result.block_session_audits
    )


def _block_failure_rows(
    result: GreenfieldFrontierResult,
) -> tuple[dict[str, Any], ...]:
    """Flatten explicit worker failures without discarding successful blocks."""

    return tuple(
        {
            "plant": failure.block_key[0],
            "sefi": failure.block_key[1],
            "candidate_count": failure.candidate_count,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
        }
        for failure in result.block_failures
    )


def _global_solve_audit_rows(
    result: GreenfieldFrontierResult,
) -> tuple[dict[str, Any], ...]:
    """Flatten every requested low-epsilon-biased compact-master solve."""

    return tuple(asdict(audit) for audit in result.global_solve_audits)


def _report(rows: Iterable[Mapping[str, Any]], result: GreenfieldFrontierResult) -> str:
    """Render an answer-first Markdown summary of the greenfield frontier."""

    values = tuple(rows)
    lines = [
        "# Greenfield production-wheel Pareto frontier",
        "",
        "Every listed proposal passed independent exact-cover and structural validation. "
        "Feasibility, candidate membership, objective bounds, and acceptance do not use "
        "historical groups.",
        "",
        "| Point | J_CH epsilon | Demand-weighted coverage days | J_CH | Groups | Singletons | Proof scope |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in values:
        epsilon = "anchor" if row.get('epsilon_j_ch') is None else f"{float(row['epsilon_j_ch']):.6f}"
        lines.append(
            f"| {row['point_index']} | {epsilon} | "
            f"{float(row['demand_weighted_mean_coverage_days']):.6f} | "
            f"{float(row['j_ch']):.6f} | {row['group_count']} | "
            f"{row['singleton_group_count']} | {row['proof_scope']} |"
        )
    lines.extend(
        [
            "",
            f"Block pools: {len(result.pools)}; retained block selections: "
            f"{len(result.block_options)}; elapsed seconds: {result.elapsed_seconds:.3f}.",
            f"Persistent structural model builds: "
            f"{sum(item.metrics.model_build_count for item in result.block_session_audits)}; "
            f"block HiGHS calls: "
            f"{sum(item.metrics.solver_call_count for item in result.block_session_audits)}.",
            f"Block execution: {result.block_execution_mode} with up to "
            f"{result.block_worker_count} worker(s); failures: "
            f"{len(result.block_failures)}.",
            f"Global epsilon schedule: power-law exponent "
            f"{result.global_epsilon_exponent:g}; requested "
            f"{len(result.global_solve_audits)} epsilon solves and retained "
            f"{len(result.points)} nondominated portfolios.",
            (
                f"Coupled rules used one joint candidate master with a {result.per_block_total_seconds:g}-second solver-stage budget; preprocessing is additional."
                if result.block_execution_mode == 'joint_candidate_master' and result.per_block_total_seconds is not None
                else
                f"Each block objective tier received up to "
                f"{result.per_block_stage_seconds:.0f} seconds."
                if result.per_block_stage_seconds is not None
                else f"Each block received a conservative total budget of "
                f"{result.per_block_total_seconds:.0f} seconds."
                if result.per_block_total_seconds is not None
                else "The configured scenario-wide stage budget was apportioned across blocks."
            ),
            "",
            "`J_CH = sum_g F_g(n_g - 1)` is the PoC manufacturing axis. It excludes "
            "initial and between-group setups, sequence, cleaning time, and unapproved "
            "cost coefficients. Group and singleton counts remain visible diagnostics.",
            "",
            "A `restricted-library` point is optimal only over the disclosed restricted "
            "library. A `runtime-limited` point is a feasible witness, not an "
            "optimality claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_greenfield_frontier_artifacts(
    output_directory: Path,
    fini_rows: Iterable[Mapping[str, Any]],
    production_versions: Iterable[ProductionVersion],
    result: GreenfieldFrontierResult,
    *,
    baseline_evidence_members: Iterable[CandidateMember] = (),
    manifest_metadata: Mapping[str, Any] | None = None,
) -> GreenfieldFrontierArtifactPaths:
    """Write every Pareto point, frontier audit tables, report, and manifest.

    Args:
        output_directory: New destination directory.
        fini_rows: Complete source-ordered FINI master.
        production_versions: Canonical finite PV catalog.
        result: Completed greenfield frontier result.
        baseline_evidence_members: Optional historical evidence for comparison only.
        manifest_metadata: Optional extraction provenance.

    Returns:
        Typed paths to top-level frontier artifacts.
    """

    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    versions = tuple(production_versions)
    baseline_evidence = tuple(baseline_evidence_members)
    baseline_summary = None
    baseline_error = None
    if result.baseline_available:
        try:
            baseline_summary = build_baseline_summary(
                result.members,
                versions,
                result.config,
                baseline_evidence,
            )
        except ValueError as error:
            baseline_error = str(error)
            baseline_summary = None
    summaries: list[dict[str, Any]] = []
    point_manifests: list[dict[str, Any]] = []
    points_directory = output_directory / "points"
    points_directory.mkdir()
    for point in result.points:
        if point.solve_result is None:
            raise ValueError(f"frontier point {point.point_index} has no solve result")
        validation = validate_solution(
            point.solve_result,
            result.members,
            result.pools,
            result.config,
            versions,
            baseline_evidence,
        )
        acceptance = _acceptance(validation.is_valid)
        point_config = result.config.model_copy(
            update={
                "scenario_id": f"{result.config.scenario_id}_{point.point_index:02d}"
            }
        )
        summary = build_scenario_summary_row(
            validation.groups,
            point.solve_result,
            point_config,
            validation,
            baseline_summary,
            acceptance_summary=acceptance,
        )
        summary.update(
            {
                "point_index": point.point_index,
                "epsilon_j_ch": point.epsilon_j_ch,
                "proof_scope": point.proof_scope,
                "portfolio_hash": point.portfolio_hash,
                "coverage_anchor_match": int(bool(point.coverage_anchor_match)),
                "operations_anchor_match": int(bool(point.operations_anchor_match)),
            }
        )
        summaries.append(summary)
        point_directory = points_directory / f"pareto_{point.point_index:02d}"
        write_scenario_artifacts(
            point_directory,
            fini_rows,
            result.members,
            result.pools,
            point.solve_result,
            point_config,
            validation,
            baseline_summary=baseline_summary,
            production_versions=versions,
            acceptance_summary=acceptance,
            baseline_evidence_members=baseline_evidence,
            include_baseline_comparison=baseline_summary is not None,
        )
        point_manifests.append(
            {
                "point_index": point.point_index,
                "directory": str(point_directory.relative_to(output_directory)),
                "portfolio_hash": point.portfolio_hash,
                "proof_scope": point.proof_scope,
                "validation_status": acceptance["acceptance_status"],
            }
        )

    paths = GreenfieldFrontierArtifactPaths(
        output_directory=output_directory,
        frontier_summary=output_directory / "frontier_summary.csv",
        block_options=output_directory / "block_options.csv",
        block_solve_audit=output_directory / "block_solve_audit.csv",
        block_session_metrics=output_directory / "block_session_metrics.csv",
        block_failures=output_directory / "block_failures.csv",
        global_solve_audit=output_directory / "global_solve_audit.csv",
        solution_report=output_directory / "solution_report.md",
        run_manifest=output_directory / "run_manifest.json",
    )
    write_csv(paths.frontier_summary, summaries)
    write_csv(paths.block_options, _block_option_rows(result))
    write_csv(paths.block_solve_audit, _block_solve_audit_rows(result))
    write_csv(paths.block_session_metrics, _block_session_rows(result))
    write_csv(paths.block_failures, _block_failure_rows(result))
    write_csv(paths.global_solve_audit, _global_solve_audit_rows(result))
    paths.solution_report.write_text(_report(summaries, result), encoding="utf-8")
    artifacts = (
        paths.frontier_summary,
        paths.block_options,
        paths.block_solve_audit,
        paths.block_session_metrics,
        paths.block_failures,
        paths.global_solve_audit,
        paths.solution_report,
    )
    manifest = dict(manifest_metadata or {})
    manifest.update(
        {
            "artifact_type": "greenfield_pareto_frontier",
            "scenario_id": result.config.scenario_id,
            "config": result.config.model_dump(mode="json"),
            "baseline_role": (
                "optional_comparison_only" if baseline_summary is not None else "unavailable"
            ),
            "baseline_comparison_error": baseline_error,
            "objective_axes": ["demand_weighted_mean_coverage_days", "j_ch"],
            "epsilon_axis": "j_ch",
            "global_epsilon_schedule": {
                "kind": "power_law_low_epsilon_dense",
                "exponent": result.global_epsilon_exponent,
                "requested_points": len(result.global_solve_audits),
                "retained_points": len(result.points),
            },
            "time_limit_scope": (
                "per_block_objective_tier"
                if result.per_block_stage_seconds is not None
                else "per_block_total"
                if result.per_block_total_seconds is not None
                else "scenario_wide_objective_tier"
            ),
            "per_block_stage_seconds": result.per_block_stage_seconds,
            "per_block_total_seconds": result.per_block_total_seconds,
            "block_execution_mode": result.block_execution_mode,
            "block_worker_count": result.block_worker_count,
            "large_block_candidate_threshold": (
                result.large_block_candidate_threshold
            ),
            "coverage_anchor_hash": result.coverage_anchor_hash,
            "operations_anchor_hash": result.operations_anchor_hash,
            "counts": {
                "members": len(result.members),
                "blocks": len(result.pools),
                "block_options": len(result.block_options),
                "block_solve_audit_rows": len(result.block_solve_audits),
                "block_solve_requests": sum(
                    item.status != "not_solved" for item in result.block_solve_audits
                ),
                "structural_model_builds": sum(
                    item.metrics.model_build_count
                    for item in result.block_session_audits
                ),
                "block_solver_calls": sum(
                    item.metrics.solver_call_count
                    for item in result.block_session_audits
                ),
                "block_failures": len(result.block_failures),
                "global_solve_requests": len(result.global_solve_audits),
                "pareto_points": len(result.points),
            },
            "pool_hashes": [pool.pool_hash for pool in result.pools],
            "pool_completeness": [
                {
                    "block": pool.block_key,
                    "completeness": pool.completeness,
                    "candidate_count": len(pool.candidates),
                }
                for pool in result.pools
            ],
            "points": point_manifests,
            "elapsed_seconds": result.elapsed_seconds,
            "outputs": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifacts
            },
        }
    )
    write_json(paths.run_manifest, manifest)
    return paths


def validate_greenfield_frontier_artifacts(
    output_directory: Path,
) -> dict[str, Any]:
    """Verify hashes, point bundles, proof labels, endpoints, and nondominance."""

    errors: list[str] = []
    manifest_path = output_directory / "run_manifest.json"
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["run_manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, evidence in manifest.get("outputs", {}).items():
        path = output_directory / name
        if not path.is_file() or sha256_file(path) != evidence.get("sha256"):
            errors.append(f"artifact hash mismatch: {name}")
    summary_path = output_directory / "frontier_summary.csv"
    rows: list[dict[str, str]] = []
    if summary_path.is_file():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    global_audit_path = output_directory / "global_solve_audit.csv"
    global_rows: list[dict[str, str]] = []
    if global_audit_path.is_file():
        with global_audit_path.open(newline="", encoding="utf-8") as handle:
            global_rows = list(csv.DictReader(handle))
    if len(rows) != manifest.get("counts", {}).get("pareto_points"):
        errors.append("frontier point count does not match the manifest")
    if not rows:
        errors.append("frontier contains no Pareto points")
    if manifest.get("counts", {}).get("block_failures", 0):
        errors.append("one or more block frontier workers failed")
    if len(global_rows) != manifest.get("counts", {}).get(
        "global_solve_requests", 0
    ):
        errors.append("global solve audit count does not match the manifest")
    requested_epsilons = [
        float(row["requested_epsilon_j_ch"]) for row in global_rows
    ]
    if requested_epsilons != sorted(requested_epsilons):
        errors.append("global epsilon requests are not ordered")
    schedule = manifest.get("global_epsilon_schedule", {})
    exponent = float(schedule.get("exponent", 1.0))
    if (
        exponent > 1
        and len(requested_epsilons) >= 3
        and requested_epsilons[-1] - requested_epsilons[0] > 1e-7
    ):
        steps = [
            right - left
            for left, right in zip(
                requested_epsilons, requested_epsilons[1:]
            )
        ]
        if steps[0] >= steps[-1]:
            errors.append("global epsilon schedule is not denser at low J_CH")
    retained_audit_hashes = {
        row["portfolio_hash"]
        for row in global_rows
        if row.get("retained_nondominated") == "True"
    }
    summary_hashes = {row.get("portfolio_hash", "") for row in rows}
    if retained_audit_hashes != summary_hashes:
        errors.append("retained global audit hashes do not match frontier points")
    if rows and not any(row.get("coverage_anchor_match") == "1" for row in rows):
        errors.append("coverage endpoint does not match its anchor")
    if rows and not any(row.get("operations_anchor_match") == "1" for row in rows):
        errors.append("operations endpoint does not match its anchor")
    for left in rows:
        for right in rows:
            if left is right:
                continue
            left_axes = (
                float(left["demand_weighted_mean_coverage_days"]),
                float(left["j_ch"]),
            )
            right_axes = (
                float(right["demand_weighted_mean_coverage_days"]),
                float(right["j_ch"]),
            )
            if all(a <= b + 1e-7 for a, b in zip(left_axes, right_axes, strict=True)) and any(
                a < b - 1e-7 for a, b in zip(left_axes, right_axes, strict=True)
            ):
                errors.append(
                    f"point {right['point_index']} is dominated by point {left['point_index']}"
                )
    allowed_proofs = {"exact", "restricted-library", "runtime-limited"}
    if any(row.get("proof_scope") not in allowed_proofs for row in rows):
        errors.append("frontier contains an unsupported proof label")
    point_results = []
    for point in manifest.get("points", []):
        point_result = validate_scenario_artifacts(
            output_directory / str(point["directory"])
        )
        point_results.append(point_result)
        if not point_result.get("valid"):
            errors.append(f"invalid point bundle: {point['directory']}")
    return {
        "valid": not errors,
        "integrity_valid": not errors,
        "complete": bool(rows) and len(point_results) == len(rows),
        "business_acceptable": False,
        "business_acceptance_assessed": False,
        "pareto_points": len(rows),
        "errors": errors,
    }
