"""Serialize solver evidence in memory and import verified historical bundles.

The row builders are shared with legacy CSV reporting. New runs never need a
filesystem round trip; historical CSV reading is confined to the importer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from production_wheel.greenfield_reporting import (
    _acceptance,
    _block_failure_rows,
    _block_option_rows,
    _block_session_rows,
    _block_solve_audit_rows,
    _global_solve_audit_rows,
    _report,
    validate_greenfield_frontier_artifacts,
)
from production_wheel.pareto import GreenfieldFrontierResult
from production_wheel.reporting import (
    build_baseline_summary,
    build_candidate_pool_summary_rows,
    build_constraint_audit_rows,
    build_matrix_exception_audit_rows,
    build_proposed_subgroups_rows,
    build_scenario_summary_row,
    build_solution_groups_rows,
    build_solution_report,
)
from production_wheel.scenarios import CanonicalInputs
from production_wheel.solution_validation import validate_solution

RESULT_SCHEMA_VERSION = "1"
TABLE_NAMES = (
    "solutions",
    "groups",
    "members",
    "pools",
    "constraints",
    "validation",
    "matrix_pairs",
    "block_options",
    "block_audits",
    "block_sessions",
    "block_failures",
    "global_audits",
    "option_groups",
    "option_members",
    "point_options",
)
POINT_FILES = {
    "groups": "solution_groups.csv",
    "members": "proposed_subgroups.csv",
    "pools": "candidate_pool_summary.csv",
    "constraints": "constraint_audit.csv",
    "validation": "validation_issues.csv",
    "matrix_pairs": "matrix_exception_audit.csv",
}
ROOT_FILES = {
    "solutions": "frontier_summary.csv",
    "block_options": "block_options.csv",
    "block_audits": "block_solve_audit.csv",
    "block_sessions": "block_session_metrics.csv",
    "block_failures": "block_failures.csv",
    "global_audits": "global_solve_audit.csv",
}


@dataclass(frozen=True, slots=True)
class ResultBundle:
    """Transport result metadata, named table rows, and Markdown artifact text.

    Inputs are JSON-compatible dictionaries; this container performs no storage
    operations. Tables include all canonical names even when no rows exist.
    """

    metadata: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    artifacts: dict[str, str]


def _json_safe(value: Any) -> Any:
    """Convert nested solver values to JSON primitives; nonfinite numbers are null."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _identity(*parts: Any) -> str:
    """Return an unambiguous stable identifier from ordered identity components."""
    return hashlib.sha256(
        json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _point_rows(
    rows: Iterable[Mapping[str, Any]], point_index: int
) -> list[dict[str, Any]]:
    """Attach original point identity and globally unique group identity to rows."""
    values = []
    for original in rows:
        row = dict(original)
        row["point_index"] = point_index
        if "proposed_subgroup" in row:
            code = row["proposed_subgroup"]
            row["group_id"] = (
                _identity(point_index, row.get("plant", ""), row.get("sefi", ""), code)
                if code
                else ""
            )
        values.append(row)
    return values


def _option_evidence(
    result: GreenfieldFrontierResult, tables: dict[str, list[dict[str, Any]]]
) -> None:
    """Persist retained selected definitions and exact block-selection links.

    Candidate hashes identify structural configurations. Signatures also include
    the coefficients from actual SelectedCandidate instances, so mismatched
    solver selections cannot silently link to a retained option.
    """
    signatures = {}
    member_records = {
        (*member.block_key, member.fini_id): member for member in result.members
    }
    for option, row in zip(result.block_options, tables["block_options"], strict=True):
        option_id = _identity(*option.block_key, option.partition_hash)
        row["option_id"] = option_id
        signature = tuple(
            sorted(
                json.dumps(_json_safe(asdict(item)), sort_keys=True)
                for item in option.selected
            )
        )
        signatures[(option.block_key, signature)] = option_id
        for selected in option.selected:
            candidate = selected.candidate
            group_id = _identity(option_id, candidate.candidate_hash)
            common = {
                "option_id": option_id,
                "option_group_id": group_id,
                "plant": option.block_key[0],
                "sefi": option.block_key[1],
                "candidate_hash": candidate.candidate_hash,
            }
            tables["option_groups"].append(
                {
                    **common,
                    **asdict(candidate),
                    **asdict(selected.coefficients),
                    "group_size": len(candidate.member_ids),
                    "j_ch_contribution": selected.coefficients.j_ch,
                    "coverage_violation_count": selected.coefficients.target_violation,
                    "coverage_excess_days": selected.coefficients.target_excess_days,
                    "matrix_exception_pair_count": selected.coefficients.matrix_exceptions,
                    "group_size_relaxed": selected.coefficients.relaxed_group,
                    "group_size_excess": selected.coefficients.size_excess,
                    "solver_tie_break_rank": selected.coefficients.stable_rank,
                    "coefficients": asdict(selected.coefficients),
                }
            )
            for member_id in candidate.member_ids:
                tables["option_members"].append(
                    {
                        **asdict(member_records[(*option.block_key, member_id)]),
                        **common,
                        "material": member_id,
                        "member": asdict(
                            member_records[(*option.block_key, member_id)]
                        ),
                    }
                )
    for point in result.points:
        by_block: dict[tuple[str, str], list[Any]] = {}
        if point.solve_result is None:
            raise ValueError(f"point {point.point_index} has no solve result")
        for selected in point.solve_result.selected:
            by_block.setdefault(selected.candidate.block_key, []).append(selected)
        for block, selected in sorted(by_block.items()):
            signature = tuple(
                sorted(
                    json.dumps(_json_safe(asdict(item)), sort_keys=True)
                    for item in selected
                )
            )
            option_id = signatures.get((block, signature))
            if option_id is None:
                raise ValueError(
                    f"point {point.point_index} block {block} has no retained option definition"
                )
            tables["point_options"].append(
                {
                    "point_index": point.point_index,
                    "plant": block[0],
                    "sefi": block[1],
                    "option_id": option_id,
                }
            )


def build_result_bundle(
    result: GreenfieldFrontierResult,
    inputs: CanonicalInputs,
    manifest_metadata: dict | None = None,
) -> ResultBundle:
    """Build validated result tables and reports directly from solver structures.

    Args:
        result: Completed frontier including retained options and actual selections.
        inputs: Canonical records and complete, source-ordered FINI master.
        manifest_metadata: Optional provenance and original solve_request payload.
    Returns:
        JSON-compatible bundle with explicit proof and business-assessment flags.
    """
    tables = {name: [] for name in TABLE_NAMES}
    artifacts = {}
    baseline = None
    baseline_error = None
    if result.baseline_available:
        try:
            baseline = build_baseline_summary(
                result.members,
                inputs.production_versions,
                result.config,
                inputs.baseline_evidence_members,
            )
        except ValueError as error:
            baseline_error = str(error)
    point_metadata = []
    run_member_keys = {(*member.block_key, member.fini_id) for member in result.members}
    for point in result.points:
        solved = point.solve_result
        if solved is None:
            raise ValueError(f"point {point.point_index} has no solve result")
        validation = validate_solution(
            solved,
            result.members,
            result.pools,
            result.config,
            inputs.production_versions,
            inputs.baseline_evidence_members,
        )
        config = result.config.model_copy(
            update={
                "scenario_id": f"{result.config.scenario_id}_{point.point_index:02d}"
            }
        )
        acceptance = _acceptance(validation.is_valid)
        summary = build_scenario_summary_row(
            validation.groups, solved, config, validation, baseline, acceptance
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
        tables["solutions"].append(summary)
        rows = {
            "groups": build_solution_groups_rows(validation.groups, solved, config),
            "members": build_proposed_subgroups_rows(
                inputs.fini_rows, result.members, validation.groups, solved, config
            ),
            "pools": build_candidate_pool_summary_rows(result.pools, config, solved),
            "constraints": build_constraint_audit_rows(config, ()),
            "validation": validation.issue_rows(),
            "matrix_pairs": build_matrix_exception_audit_rows(
                validation.groups,
                config,
                result.members,
                inputs.baseline_evidence_members,
            ),
        }
        for name, values in rows.items():
            scoped = _point_rows(values, point.point_index)
            if name in ("groups", "members"):
                for row in scoped:
                    row.update(
                        {
                            "acceptance_status": acceptance["acceptance_status"],
                            "failed_baseline_guardrails": "",
                        }
                    )
            if name == "members":
                for row in scoped:
                    key = (
                        str(row.get("plant", "")),
                        str(row.get("sefi", "")),
                        str(row.get("material") or row.get("fini_id", "")),
                    )
                    if key in run_member_keys:
                        row["run_model_status"] = "modeled"
                        row["run_exclusion_reason"] = ""
                    elif row.get("model_status") == "modeled":
                        # Preserve source admission; run scope is separate evidence.
                        row["run_model_status"] = "out_of_scope"
                        row["run_exclusion_reason"] = "outside_run_scope_or_disposition"
                    else:
                        row["run_model_status"] = row.get("model_status", "")
                        row["run_exclusion_reason"] = row.get("exclusion_reason", "")
            tables[name].extend(scoped)
        key = f"points/pareto_{point.point_index:02d}/solution_report.md"
        artifacts[key] = build_solution_report(
            summary,
            validation.groups,
            result.pools,
            solved,
            config,
            validation,
            rows["constraints"],
        )
        point_metadata.append(
            {
                "point_index": point.point_index,
                "proof_scope": point.proof_scope,
                "portfolio_hash": point.portfolio_hash,
                "validation_valid": validation.is_valid,
                "solver": {
                    key: value
                    for key, value in asdict(solved).items()
                    if key != "selected"
                },
            }
        )
    for name, builder in (
        ("block_options", _block_option_rows),
        ("block_audits", _block_solve_audit_rows),
        ("block_sessions", _block_session_rows),
        ("block_failures", _block_failure_rows),
        ("global_audits", _global_solve_audit_rows),
    ):
        tables[name] = list(builder(result))
    _option_evidence(result, tables)
    artifacts["solution_report.md"] = _report(tables["solutions"], result)
    metadata = dict(manifest_metadata or {})
    metadata.update(
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "source_kind": "solver_memory",
            "source_provenance": dict(manifest_metadata or {}),
            "solve_request": metadata.get("solve_request", {}),
            "config": result.config.model_dump(mode="json"),
            "scenario_id": result.config.scenario_id,
            "baseline_role": "optional_comparison_only"
            if baseline is not None
            else "unavailable",
            "baseline_comparison_error": baseline_error,
            "points": point_metadata,
            "counts": {name: len(rows) for name, rows in tables.items()},
            "proof_scopes": sorted({point.proof_scope for point in result.points}),
            "integrity_valid": all(item["validation_valid"] for item in point_metadata)
            and not result.block_failures,
            "complete": bool(result.points) and not result.block_failures,
            "business_acceptable": False,
            "business_acceptance_assessed": False,
            "option_definitions_available": True,
            "point_option_mapping_available": True,
            "elapsed_seconds": result.elapsed_seconds,
            "coverage_anchor_hash": result.coverage_anchor_hash,
            "operations_anchor_hash": result.operations_anchor_hash,
            "block_execution_mode": result.block_execution_mode,
            "block_worker_count": result.block_worker_count,
            "per_block_stage_seconds": result.per_block_stage_seconds,
            "per_block_total_seconds": result.per_block_total_seconds,
            "global_epsilon_exponent": result.global_epsilon_exponent,
            "large_block_candidate_threshold": result.large_block_candidate_threshold,
        }
    )
    return ResultBundle(_json_safe(metadata), _json_safe(tables), artifacts)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required historical CSV table, including valid empty audit files."""
    if not path.is_file():
        raise ValueError(f"missing historical artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_path(root: Path, relative: str) -> Path:
    """Resolve a manifest-relative path while rejecting paths outside its bundle."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"historical manifest path escapes bundle: {relative}")
    return path


def import_result_bundle(path: Path) -> ResultBundle:
    """Read a historical frontier only after manifest, hash, and count validation.

    Args:
        path: Root solution directory containing run_manifest.json and points/.
    Returns:
        Bundle retaining original point indices, source manifests, and reports.
        CSV values remain strings except explicit integer point_index columns.
        Historical retained option definitions and mappings are explicitly absent.
    Raises:
        ValueError: Missing, altered, incomplete, or inconsistent result evidence.
    """
    root = Path(path).resolve()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        point_manifests = []
        for name in manifest.get("outputs", {}):
            _safe_path(root, name)
        for point in manifest.get("points", []):
            directory = _safe_path(root, str(point["directory"]))
            point_manifest = json.loads(
                (directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            for name in point_manifest.get("output_hashes", {}):
                _safe_path(root, str((directory / name).relative_to(root)))
            point_manifests.append((point, directory, point_manifest))
        verification = validate_greenfield_frontier_artifacts(root)
        if not verification.get("integrity_valid") or not verification.get("complete"):
            raise ValueError(
                f"invalid historical bundle integrity: {verification.get('errors')}"
            )
        tables = {name: [] for name in TABLE_NAMES}
        for name, filename in ROOT_FILES.items():
            if filename not in manifest.get("outputs", {}):
                raise ValueError(f"missing root hash declaration: {filename}")
            tables[name] = _read_csv(root / filename)
        summaries = {}
        for row in tables["solutions"]:
            row["point_index"] = int(row["point_index"])
            if row["point_index"] in summaries:
                raise ValueError("duplicate point_index in frontier")
            summaries[row["point_index"]] = row
        indices = [int(point["point_index"]) for point, _, _ in point_manifests]
        if len(set(indices)) != len(indices) or set(indices) != set(summaries):
            raise ValueError("point manifest indices do not match frontier")
        if "solution_report.md" not in manifest.get("outputs", {}):
            raise ValueError("root report has no integrity hash")
        artifacts = {
            "solution_report.md": (root / "solution_report.md").read_text(
                encoding="utf-8"
            )
        }
        for point, directory, point_manifest in point_manifests:
            index = int(point["point_index"])
            if point.get("portfolio_hash") != summaries[index].get("portfolio_hash"):
                raise ValueError(f"point {index} portfolio hash mismatch")
            for name, filename in {
                **POINT_FILES,
                "point_summary": "scenario_summary.csv",
            }.items():
                if filename not in point_manifest.get("output_hashes", {}):
                    raise ValueError(f"missing point hash declaration: {filename}")
                rows = _read_csv(directory / filename)
                expected = point_manifest.get("row_counts", {}).get(Path(filename).stem)
                if expected is None or len(rows) != int(expected):
                    raise ValueError(f"point {index} row count mismatch: {filename}")
                if name != "point_summary":
                    tables[name].extend(_point_rows(rows, index))
                elif len(rows) != 1:
                    raise ValueError(f"point {index} must have one scenario summary")
                else:
                    mismatches = [
                        key
                        for key, value in rows[0].items()
                        if key in summaries[index] and summaries[index][key] != value
                    ]
                    if mismatches:
                        raise ValueError(
                            f"point {index} summary differs from frontier: {mismatches}"
                        )
            if "solution_report.md" not in point_manifest.get("output_hashes", {}):
                raise ValueError(f"point {index} report has no integrity hash")
            artifacts[f"{directory.relative_to(root)}/solution_report.md"] = (
                directory / "solution_report.md"
            ).read_text(encoding="utf-8")
        for table, count_key in (
            ("block_options", "block_options"),
            ("block_audits", "block_solve_audit_rows"),
            ("block_failures", "block_failures"),
            ("global_audits", "global_solve_requests"),
            ("block_sessions", "blocks"),
        ):
            if len(tables[table]) != manifest.get("counts", {}).get(count_key):
                raise ValueError(f"root row count mismatch: {table}")
        for row in tables["block_options"]:
            row["option_id"] = _identity(
                row["plant"], row["sefi"], row["partition_hash"]
            )
        metadata = {
            **manifest,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "source_kind": "historical_bundle",
            "source_path": str(root),
            "source_provenance": {"manifest": manifest, "path": str(root)},
            "source_manifest_sha256": hashlib.sha256(
                (root / "run_manifest.json").read_bytes()
            ).hexdigest(),
            "solve_request": manifest.get("solve_request", {}),
            "source_counts": manifest.get("counts", {}),
            "counts": {name: len(rows) for name, rows in tables.items()},
            "point_manifests": [
                {"point_index": int(point["point_index"]), "manifest": value}
                for point, _, value in point_manifests
            ],
            "proof_scopes": sorted({row["proof_scope"] for row in tables["solutions"]}),
            **verification,
            "option_definitions_available": False,
            "point_option_mapping_available": False,
            "historical_limitations": [
                "Retained candidate definitions and point-to-option selections were not exported; candidate hashes alone are not reconstructed definitions."
            ],
        }
        return ResultBundle(metadata, tables, artifacts)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid historical bundle: {error}") from error
