"""Run-scoped extraction pipeline for canonical optimizer inputs."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from production_wheel.extraction.common import (
    extract_power_query_m,
    infer_field_dictionary,
    query_catalog,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)
from production_wheel.extraction.enrichment_workbook import extract_pallet_candidates, resolve_pallets
from production_wheel.extraction.primary_workbook import (
    classify_fini_rows,
    extract_primary_workbook,
    refresh_scope_dependent_tables,
)
from production_wheel.progress import ProgressBar


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Describe a completed extraction run.

    Attributes:
        run_directory: Root directory containing emitted artifacts.
        manifest: Final run manifest.
    """

    run_directory: Path
    manifest: dict[str, Any]


def make_run_id(source_hash: str) -> str:
    """Return a timestamped run identifier tied to the primary source hash."""

    timestamp = utc_now().replace("-", "").replace(":", "")
    return f"{timestamp}-{source_hash[:8]}"


def extract_snapshot(
    primary_path: Path,
    run_directory: Path,
    enrichment_path: Path | None = None,
    snapshot_expectations: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Extract primary and optional enrichment data into one auditable run directory.

    Args:
        primary_path: Authoritative 12-month primary production workbook.
        run_directory: Explicit output directory for this run.
        enrichment_path: Optional enrichment workbook used only for governed enrichment/audit.
        snapshot_expectations: Optional reviewed figures of a frozen snapshot used as
            a regression guard; see ``extract_primary_workbook``. The extra key
            ``modeled_pallet_fallbacks`` checks the enrichment pallet fallback count.

    Returns:
        Completed extraction result and manifest.

    Raises:
        FileNotFoundError: If a supplied workbook does not exist.
        The returned manifest uses ``failed_validation`` when acceptance
        invariants fail; it does not raise merely for validation findings.
    """

    started = time.monotonic()
    primary_path = primary_path.resolve()
    if not primary_path.is_file():
        raise FileNotFoundError(primary_path)
    if enrichment_path is not None:
        enrichment_path = enrichment_path.resolve()
        if not enrichment_path.is_file():
            raise FileNotFoundError(enrichment_path)
    run_directory.mkdir(parents=True, exist_ok=False)
    extracted_directory = run_directory / "extracted"
    extracted_directory.mkdir()
    progress = ProgressBar(6, "Extract workbook data")

    tables, primary_metadata, issues = extract_primary_workbook(primary_path, snapshot_expectations)
    progress.update(1)
    enrichment_metadata: dict[str, Any] | None = None
    if enrichment_path is not None:
        fini_keys = {str(row["fini_plant_key"]) for row in tables["fini_master"]}
        candidates, enrichment_metadata = extract_pallet_candidates(enrichment_path, fini_keys)
        updated_fini, resolutions, pallet_issues = resolve_pallets(tables["fini_master"], candidates)
        tables["fini_master"] = updated_fini
        tables["enrichment_pallet_candidates"] = candidates
        tables["pallet_resolution"] = resolutions
        issues.extend(pallet_issues)
        modeled_fallbacks = sum(
            1
            for row in resolutions
            if row["model_status"] == "modeled" and row["resolution_status"] == "resolved_enrichment_unique"
        )
        expected_fallbacks = (snapshot_expectations or {}).get("modeled_pallet_fallbacks")
        if expected_fallbacks is not None and modeled_fallbacks != expected_fallbacks:
            issues.append(
                {
                    "rule_id": "MODELED_ENRICHMENT_PALLET_FALLBACK_COUNT",
                    "severity": "error",
                    "entity": "snapshot",
                    "entity_key": "modeled_pallet_fallbacks",
                    "observed": modeled_fallbacks,
                    "expected": expected_fallbacks,
                    "action": "block pallet-adjusted demo until source reconciliation",
                }
            )
        if enrichment_metadata.get("forecast_horizon_selected") != "12 months":
            issues.append(
                {
                    "rule_id": "ENRICHMENT_HORIZON_IS_DIAGNOSTIC",
                    "severity": "warning",
                    "entity": "source",
                    "entity_key": enrichment_path.name,
                    "observed": enrichment_metadata.get("forecast_horizon_selected"),
                    "expected": "primary workbook 12 months remains canonical",
                    "action": "do not import enrichment demand into this snapshot",
                }
            )
    else:
        tables["enrichment_pallet_candidates"] = []
        updated_fini, resolutions, pallet_issues = resolve_pallets(tables["fini_master"], [])
        tables["fini_master"] = updated_fini
        tables["pallet_resolution"] = resolutions
        issues.extend(pallet_issues)

    tables["fini_master"] = classify_fini_rows(
        tables["fini_master"],
        tables["production_versions"],
        require_resolved_pallet=True,
    )
    admission_by_key = {
        str(row["fini_plant_key"]): row for row in tables["fini_master"]
    }
    for resolution in tables["pallet_resolution"]:
        admission = admission_by_key[str(resolution["fini_plant_key"])]
        resolution["model_status"] = admission["model_status"]
        resolution["exclusion_reason"] = admission["exclusion_reason"]
        resolution["optimized_pv_model_status"] = admission[
            "optimized_pv_model_status"
        ]
    refresh_scope_dependent_tables(tables)
    pallet_exclusions = [
        row
        for row in tables["fini_master"]
        if row.get("exclusion_reason")
        in {"missing_pallet_data", "nonpositive_pallet_data"}
    ]
    if pallet_exclusions:
        issues.append(
            {
                "rule_id": "FINI_PALLET_DATA_EXCLUDED",
                "severity": "warning",
                "entity": "fini",
                "entity_key": "|".join(
                    str(row["fini_plant_key"]) for row in pallet_exclusions[:10]
                ),
                "observed": len(pallet_exclusions),
                "expected": 0,
                "action": "retain the rows as data-quality exclusions until pallet data is resolved",
            }
        )
    progress.update(2)

    query_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    for label, workbook_path in (("PRIMARY", primary_path), ("ENRICHMENT", enrichment_path)):
        if workbook_path is None:
            continue
        m_source = extract_power_query_m(workbook_path)
        if m_source is None:
            issues.append(
                {
                    "rule_id": "POWER_QUERY_SOURCE_PRESENT",
                    "severity": "error",
                    "entity": "source",
                    "entity_key": workbook_path.name,
                    "observed": "missing",
                    "expected": "embedded Section1.m",
                    "action": "retain workbook tables but mark query lineage incomplete",
                }
            )
            continue
        (extracted_directory / f"power_query_{label.lower()}.m").write_text(m_source, encoding="utf-8")
        queries, dependencies = query_catalog(m_source, label)
        query_rows.extend(queries)
        dependency_rows.extend(dependencies)
    tables["query_catalog"] = query_rows
    tables["external_dependencies"] = dependency_rows
    progress.update(3)

    field_dictionary: list[dict[str, Any]] = []
    table_sources: dict[str, str] = {}
    for table_name in tables:
        if table_name.startswith("enrichment_") or table_name == "pallet_resolution":
            source = "enrichment FINI_pallets with primary-first resolution"
        elif table_name in {"query_catalog", "external_dependencies"}:
            source = "Embedded Power Query Section1.m"
        else:
            source = "primary workbook table or derived canonical record"
        table_sources[table_name] = source
        field_dictionary.extend(infer_field_dictionary(table_name, tables[table_name], source))
    tables["field_dictionary"] = field_dictionary
    tables["validation_issues"] = issues
    progress.update(4)

    output_counts: dict[str, int] = {}
    for table_name in sorted(tables):
        output_counts[table_name] = write_csv(extracted_directory / f"{table_name}.csv", tables[table_name])
    progress.update(5)

    output_files = sorted(path for path in extracted_directory.iterdir() if path.is_file())
    output_hashes = {
        path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in output_files
    }
    package_versions = {}
    for package in ("openpyxl", "pydantic", "pyomo", "highspy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest = {
        "run_id": run_directory.name,
        "created_utc": utc_now(),
        "status": "failed_validation" if any(row["severity"] == "error" for row in issues) else "ready",
        "authoritative_horizon": {
            "source": "PRIMARY",
            "months": 12,
            "productive_weeks": 50,
            "demand_days": 250,
        },
        "inputs": {
            "primary": {
                "path": str(primary_path),
                "sha256": sha256_file(primary_path),
                "size_bytes": primary_path.stat().st_size,
            },
            "enrichment": (
                {
                    "path": str(enrichment_path),
                    "sha256": sha256_file(enrichment_path),
                    "size_bytes": enrichment_path.stat().st_size,
                }
                if enrichment_path is not None
                else None
            ),
        },
        "primary": primary_metadata,
        "enrichment": enrichment_metadata,
        "tables": output_counts,
        "outputs": output_hashes,
        "issues": {
            "error": sum(row["severity"] == "error" for row in issues),
            "warning": sum(row["severity"] == "warning" for row in issues),
            "total": len(issues),
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": package_versions,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    }
    write_json(run_directory / "run_manifest.json", manifest)
    progress.finish()
    return ExtractionResult(run_directory, manifest)
