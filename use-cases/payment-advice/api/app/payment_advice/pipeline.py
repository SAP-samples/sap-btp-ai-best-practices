"""
End-to-end UC-01 pipeline: one advice + a resolved schema -> canonical JSON.

Given a ``SchemaSelection`` (already resolved/confirmed by the caller), this runs:

    extract (split + Document AI + recombine)
      -> map to canonical (skipped when the schema is already canonical)
      -> validate + verify
      -> write canonical JSON + a raw-extraction sidecar + a run-log sidecar

The raw extraction keeps every Document AI column as printed (index-aligned with
the canonical line items) plus the schema's field labels, so customer rules can
read columns the canonical mapper drops (e.g. a region code).

The CLI owns client resolution, schema confirmation, and progress display; this
module owns deterministic execution and is the same entry point a future FastAPI
route would call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging

from .canonical import CANONICAL_SCHEMA, validate_canonical
from .extract import ExtractResult, extract_document
from .mapper import map_to_canonical
from .schema_select import SchemaSelection
from .verify import VerifyResult, verify_canonical


logger = logging.getLogger(__name__)
# Field definitions per (schema id, version); a version's fields never change once active.
_SCHEMA_FIELDS_CACHE: dict[tuple[str, str], list[dict[str, str]]] = {}


@dataclass
class PipelineResult:
    """Outcome of running the pipeline on one advice."""

    client_key: str
    canonical_path: Path
    run_log_path: Path
    canonical: dict[str, Any]
    validation_issues: list[str]
    verify: VerifyResult
    extract: ExtractResult
    mapping: dict[str, Any] | None = field(default=None)
    raw_extraction: dict[str, Any] = field(default_factory=dict)


def schema_fields(dox_client: Any, schema_id: str, schema_version: str, client_id: str) -> list[dict[str, str]]:
    """Return the schema's field definitions as ``[{name, label, type, scope}]``.

    Labels are the column headings as printed on the document (e.g. "Rg" for the
    field ``region``), which customer rules usually quote. A lookup failure is
    logged and yields an empty list: rules then resolve by field name only.

    Args:
        dox_client: SAP Document AI client.
        schema_id: Schema id used for the extraction.
        schema_version: Schema version used for the extraction.
        client_id: Document AI client id.

    Returns:
        Field definitions; ``scope`` is ``header`` or ``line``.
    """
    key = (str(schema_id), str(schema_version))
    if key not in _SCHEMA_FIELDS_CACHE:
        try:
            details = dox_client.get_schema_version_details(schema_id, schema_version, client_id=client_id)
        except Exception as exc:  # labels are an aid, never a reason to fail extraction
            logger.warning("Schema field labels unavailable for %s v%s: %s", schema_id, schema_version, exc)
            return []
        _SCHEMA_FIELDS_CACHE[key] = [
            {"name": str(item.get("name")), "label": str(item.get("label") or item.get("name")),
             "type": str(item.get("formattingType") or "string"), "scope": scope}
            for scope, collection in (("header", "headerFields"), ("line", "lineItemFields"))
            for item in (details.get(collection) or []) if isinstance(item, dict) and item.get("name")
        ]
    return _SCHEMA_FIELDS_CACHE[key]


def run(
    file_path: str | Path,
    client_key: str,
    selection: SchemaSelection,
    *,
    dox_client: Any,
    settings: Any,
    out_dir: str | Path | None = None,
    mapper_model: str | None = None,
) -> PipelineResult:
    """
    Execute the pipeline for one advice with a resolved schema.

    Args:
        file_path: The advice file.
        client_key: Normalized client key (for output naming and the run log).
        selection: A ready ``SchemaSelection`` (status == "ready").
        dox_client: SAP Document AI client.
        settings: PaymentAdviceSettings (client id, limits, out dir, models).
        out_dir: Output directory (defaults to settings.out_dir).
        mapper_model: Override the canonical-mapper model.

    Returns:
        A ``PipelineResult`` with the written paths, canonical payload, validation
        issues, and verification result.

    Raises:
        ValueError: If ``selection`` is not ready.
    """
    if selection.status != "ready" or not selection.schema_id:
        raise ValueError("pipeline.run requires a ready SchemaSelection with a schema_id")

    file_path = Path(file_path)
    out_dir = Path(out_dir or settings.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "_chunks"
    started = time.time()

    # Canonical schema needs number/date normalization during recombination; a custom
    # schema's raw values are normalized later by the canonical mapper.
    schema_for_norm = CANONICAL_SCHEMA if selection.is_canonical else {}
    extract = extract_document(
        file_path,
        client=dox_client,
        schema=schema_for_norm,
        dox_client_id=settings.dox_client_id,
        out_dir=chunk_dir,
        schema_id=selection.schema_id,
        schema_version=selection.schema_version,
        max_pages=settings.max_pages,
        max_line_items=settings.max_line_items,
        max_columns=settings.max_columns,
    )

    mapping: dict[str, Any] | None = None
    if selection.is_canonical:
        header = extract.aggregate.get("headers", {})
        line_items = extract.aggregate.get("line_items", [])
    else:
        mapped = map_to_canonical(
            extract.aggregate, model=mapper_model or settings.mapper_model
        )
        header = mapped["header"]
        line_items = mapped["line_items"]
        mapping = mapped["mapping"]

    validation_issues = validate_canonical(header, line_items)
    verify = verify_canonical(header, line_items)
    canonical = {"header": header, "line_items": line_items}

    raw = extract.aggregate.get("raw") or {}
    raw_extraction = {
        "schema_fields": schema_fields(dox_client, selection.schema_id, selection.schema_version,
                                       settings.dox_client_id),
        "header": raw.get("headers", {}),
        "line_items": raw.get("line_items", []),
        # Which raw column the LLM mapper sent to which canonical field; documented
        # customer mappings override it later (rule_engine.apply_documented_mappings).
        "mapping": mapping or {},
    }

    stem = f"{client_key}_{file_path.stem}"
    canonical_path = out_dir / f"{stem}_canonical.json"
    run_log_path = out_dir / f"{stem}_run.json"

    canonical_path.write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / f"{stem}_raw.json").write_text(
        json.dumps(raw_extraction, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    run_log = {
        "client_key": client_key,
        "source_file": file_path.name,
        "schema": {
            "schema_id": selection.schema_id,
            "schema_version": selection.schema_version,
            "source": selection.source,
            "is_canonical": selection.is_canonical,
        },
        "extraction": {
            "parts": extract.part_names,
            "job_ids": extract.job_ids,
            "oversized": extract.oversized,
            "size_reason": extract.size_reason,
        },
        "mapping": mapping,
        "validation_issues": validation_issues,
        "verify": {
            "confidence": verify.confidence,
            "needs_review": verify.needs_review,
            "issues": verify.issues,
            "warnings": verify.warnings,
            "suspected_duplicate_groups": verify.suspected_duplicate_groups,
        },
        "line_item_count": len(line_items),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    run_log_path.write_text(
        json.dumps(run_log, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    return PipelineResult(
        client_key=client_key,
        canonical_path=canonical_path,
        run_log_path=run_log_path,
        canonical=canonical,
        validation_issues=validation_issues,
        verify=verify,
        extract=extract,
        mapping=mapping,
        raw_extraction=raw_extraction,
    )
