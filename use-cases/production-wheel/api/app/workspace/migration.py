"""Explicit historical file importer; application runs never use this file path."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from .datasets_service import now
from .models import INPUT_VIEWS, ExecutionBudget
from production_wheel.result_bundle import import_result_bundle


def import_history(service, solution_directory: Path, dataset_directory: Path):
    """Validate and import a legacy extraction/frontier while preserving job identity."""
    from production_wheel.extraction.common import sha256_file

    bundle = import_result_bundle(solution_directory)
    manifest = json.loads((dataset_directory / "run_manifest.json").read_text())
    tables = {}
    extracted = dataset_directory / "extracted"
    for filename, expected in manifest.get("outputs", {}).items():
        path = extracted / filename
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"extraction checksum mismatch: {filename}")
    for path in extracted.glob("*.csv"):
        if path.stem not in INPUT_VIEWS:
            raise ValueError(f"unknown historical table: {path.stem}")
        with path.open(newline="") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
        count = manifest.get("tables", {}).get(path.stem)
        if count is not None and count != len(tables[path.stem]):
            raise ValueError(f"extraction row count mismatch: {path.stem}")
    source = bundle.metadata.get("source_extraction", {})
    if source.get("run_id") and source["run_id"] != manifest.get("run_id"):
        raise ValueError("frontier and extraction provenance do not match")
    expected_inputs = {k: v["sha256"] for k, v in source.get("inputs", {}).items() if v}
    actual_inputs = {k: v["sha256"] for k, v in manifest.get("inputs", {}).items() if v}
    if expected_inputs and expected_inputs != actual_inputs:
        raise ValueError("frontier source hashes differ from extraction")
    sources = []
    for item in manifest.get("inputs", {}).values():
        if not item:
            continue
        path = Path(item["path"])
        if path.is_file() and sha256_file(path) == item["sha256"]:
            sources.append((path.name, path.read_bytes()))
    metadata = {
        **manifest,
        "plant": next(iter(tables["fini_master"]))["plant"],
        "parser_version": "legacy-extraction-import-v1",
        "input_hashes": {k: v["sha256"] for k, v in manifest["inputs"].items() if v},
        "settings": {
            "demand_days": 250,
            "productive_weeks": 50,
            "canonical_factor": 0.9,
            "modeled_frequencies": ["01W", "02W"],
        },
        "source_artifacts_available": bool(sources),
    }
    dataset = service.register_dataset(
        "Historical extraction " + dataset_directory.name,
        {
            "metadata": metadata,
            "tables": tables,
            "issues": tables.get("validation_issues", []),
        },
        sources,
        reuse_content=True,
    )
    service.publish_dataset(dataset["dataset_id"])
    run_id = solution_directory.resolve().parent.name
    try:
        existing = service.get_run(run_id)
        if existing["dataset_id"] != dataset["dataset_id"]:
            raise ValueError("existing run has different source provenance")
        if existing.get("results_ready"):
            if existing["metadata"].get(
                "source_manifest_sha256"
            ) != bundle.metadata.get("source_manifest_sha256"):
                raise ValueError("existing run content differs from imported manifest")
            return existing
    except KeyError:
        service.repo.insert(
            "runs",
            run_id,
            {
                "run_id": run_id,
                "revision": 1,
                "dataset_id": dataset["dataset_id"],
                "status": "persisting",
                "stage": "historical_import",
                "created_at": now(),
                "results_ready": False,
                "request": {
                    "config": bundle.metadata.get("config", {}),
                    "scope": [],
                    "constraints": [],
                },
                "budget": ExecutionBudget().model_dump(),
                "metadata": {},
            },
        )
    return service.publish_results(run_id, bundle)
