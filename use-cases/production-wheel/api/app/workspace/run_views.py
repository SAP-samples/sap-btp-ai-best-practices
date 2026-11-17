"""Compact, public read models derived from immutable optimizer run records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


_RULE_TEXT_LIMIT = 4_000
_FAILURE_SAMPLE_LIMIT = 10
_RESULT_METADATA_FIELDS = (
    "result_schema_version",
    "source_kind",
    "scenario_id",
    "baseline_role",
    "baseline_comparison_error",
    "counts",
    "proof_scopes",
    "integrity_valid",
    "complete",
    "business_acceptable",
    "business_acceptance_assessed",
    "option_definitions_available",
    "point_option_mapping_available",
    "elapsed_seconds",
    "coverage_anchor_hash",
    "operations_anchor_hash",
    "block_execution_mode",
    "block_worker_count",
    "global_epsilon_exponent",
)


def _text(value: object, limit: int = _RULE_TEXT_LIMIT) -> str | None:
    """Return a bounded string suitable for an API or model-facing summary."""

    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _matrix_fingerprint(rows: list[Mapping[str, Any]]) -> str | None:
    """Return a stable digest for a frozen matrix without returning its rows."""

    if not rows:
        return None
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _count(value: object) -> int:
    """Return a non-negative diagnostic count without trusting checkpoint types."""

    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def configuration_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return fixed run configuration while excluding replay-only matrix/request data."""

    profile = record.get("plant_profile") or {}
    settings = profile.get("settings") or {}
    request_config = (record.get("request") or {}).get("config") or {}
    matrix_rows = profile.get("matrix_rows") or []
    rules = profile.get("rules") or request_config.get("business_rules") or []
    rule_ids = [str(rule["constraint_id"]) for rule in rules if rule.get("constraint_id")]
    return {
        "profile_id": profile.get("profile_id") or record.get("plant_profile_id"),
        "profile_name": profile.get("name"),
        "plant": profile.get("plant"),
        "profile_revision": profile.get("revision") or record.get("plant_profile_revision"),
        "horizon_days": settings.get("horizon_days", request_config.get("demand_days")),
        "demand_days_per_week": settings.get("demand_days_per_week"),
        "high_runner_threshold_days": settings.get(
            "high_runner_threshold_days", request_config.get("high_runner_threshold_days")
        ),
        "runner_basis": settings.get("runner_basis", request_config.get("runner_basis")),
        "coverage_mode": request_config.get("coverage_mode"),
        "coverage_basis": request_config.get("coverage_basis"),
        "matrix_mode": settings.get("matrix_mode", request_config.get("matrix_mode")),
        "matrix_version": (request_config.get("versions") or {}).get("matrix_version"),
        "matrix_pair_count": len(matrix_rows),
        "matrix_sha256": _matrix_fingerprint(matrix_rows),
        "maximum_group_size": settings.get("maximum_group_size"),
        "fixed_rule_ids": rule_ids,
        "fixed_rules_text": _text(profile.get("rules_text")),
    }


def run_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact run-list identity and lifecycle fields without nested payloads."""

    return {
        key: record.get(key)
        for key in (
            "run_id",
            "revision",
            "dataset_id",
            "draft_id",
            "title",
            "parent_run_id",
            "plant_profile_id",
            "plant_profile_revision",
            "status",
            "stage",
            "error",
            "created_at",
            "started_at",
            "solver_finished_at",
            "finished_at",
            "heartbeat_at",
            "progress",
            "results_ready",
            "cancel_requested",
            "worker_id",
        )
    }


def run_status(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a selected run's compact lifecycle and configuration summary."""

    return {**run_summary(record), "configuration": configuration_summary(record)}


def result_metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return auditable result metadata while excluding configs and evidence blobs."""

    return {key: metadata[key] for key in _RESULT_METADATA_FIELDS if key in metadata}


def run_results_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a completed result without replay-only run fields or raw metadata."""

    return {
        "run": run_status(result["run"]),
        "points": list(result.get("points") or []),
        "metadata": result_metadata_summary(result.get("metadata") or {}),
    }


def failure_diagnostics(
    record: Mapping[str, Any], checkpoint: bytes | None = None
) -> dict[str, Any]:
    """Return bounded failed-run evidence extracted from an optional checkpoint blob."""

    payload: dict[str, Any] = {}
    if checkpoint:
        try:
            decoded = json.loads(checkpoint)
            payload = decoded if isinstance(decoded, dict) else {}
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
    metadata = payload.get("metadata") or {}
    failures = (payload.get("tables") or {}).get("block_failures") or []
    affected_blocks = [
        {
            "plant": _text(failure.get("plant"), 256),
            "sefi": _text(failure.get("sefi"), 256),
            "candidate_count": failure.get("candidate_count"),
            "error_type": _text(failure.get("error_type"), 256),
            "error_message": _text(failure.get("error_message")),
        }
        for failure in failures[:_FAILURE_SAMPLE_LIMIT]
        if isinstance(failure, Mapping)
    ]
    candidate_count = sum(
        _count(failure.get("candidate_count"))
        for failure in failures
        if isinstance(failure, Mapping)
    )
    failure_types = sorted(
        {
            str(failure["error_type"])
            for failure in failures
            if isinstance(failure, Mapping) and failure.get("error_type")
        }
    )
    return {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "stage": record.get("stage"),
        "public_error": _text(record.get("error")),
        "checkpoint_available": checkpoint is not None,
        "complete": metadata.get("complete"),
        "integrity_valid": metadata.get("integrity_valid"),
        "failure_types": failure_types,
        "affected_block_count": len(failures),
        "candidate_count": candidate_count,
        "affected_blocks": affected_blocks,
        "affected_blocks_truncated": len(failures) > len(affected_blocks),
    }


def matrix_page(
    record: Mapping[str, Any],
    *,
    offset: int = 0,
    limit: int = 50,
    status: str | None = None,
) -> dict[str, Any]:
    """Page the selected run's frozen matrix without exposing its full run record."""

    if offset < 0:
        raise ValueError("matrix offset must be non-negative")
    if limit < 1 or limit > 100:
        raise ValueError("matrix limit must be between 1 and 100")
    rows = (record.get("plant_profile") or {}).get("matrix_rows") or []
    selected = [
        {key: row.get(key) for key in ("volume_a", "volume_b", "status")}
        for row in rows
        if isinstance(row, Mapping) and (status is None or row.get("status") == status)
    ]
    page = selected[offset : offset + limit]
    return {
        "run_id": record.get("run_id"),
        "total": len(selected),
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(page) < len(selected),
        "rows": page,
    }
