"""Canonical hash helpers for deterministic evidence-domain identifiers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from app.ecn.models import (
    AnalysisSectionKey,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
    JsonValue,
)


def canonical_sha256(value: JsonValue) -> str:
    """Hash a JSON-safe value with stable UTF-8 key ordering.

    Args:
        value: JSON-safe value whose semantic content determines the digest.

    Returns:
        A full lowercase SHA-256 hexadecimal digest.
    """
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_id(
    kind: str,
    affected_ids: Sequence[str],
    resulting_ids: Sequence[str],
    paths: Sequence[str],
) -> str:
    """Return a stable candidate identity without source-order accidents.

    Args:
        kind: Candidate-kind enum value.
        affected_ids: Referenced affected source-object identifiers.
        resulting_ids: Referenced resulting source-object identifiers.
        paths: All supporting source JSON paths.

    Returns:
        ``C-`` followed by the first 16 hex characters of the canonical hash.
    """
    return "C-" + canonical_sha256(
        {
            "kind": kind,
            "affected_ids": sorted(set(affected_ids)),
            "resulting_ids": sorted(set(resulting_ids)),
            "paths": sorted(set(paths)),
        }
    )[:16]


def json_safe(value: object) -> JsonValue:
    """Convert immutable domain values to canonical JSON-safe values.

    Args:
        value: A dataclass, enum, date, decimal, collection, or JSON scalar.

    Returns:
        A recursively JSON-compatible representation.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {key: json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value  # type: ignore[return-value]


def source_coordinate(source: EvidenceSource) -> tuple[str, str, str, int, str]:
    """Return the total ordering used for source provenance."""
    return (
        source.artifact_id,
        source.json_path or "",
        source.sheet_name or "",
        source.source_row_number if source.source_row_number is not None else -1,
        source.bom_row_id or "",
    )


def make_evidence_identity(
    section: AnalysisSectionKey,
    kind: EvidenceKind,
    tier: EvidenceTier | None,
    sources: Sequence[EvidenceSource],
    related_ids: Sequence[str],
    occurrence: tuple[object, ...] | None,
    discriminator: str,
) -> dict[str, JsonValue]:
    """Return the canonical identity payload behind one evidence citation."""
    ordered_sources = sorted(sources, key=source_coordinate)
    return {
        "section": section.value,
        "kind": kind.value,
        "tier": tier.value if tier is not None else None,
        "sources": [json_safe(source) for source in ordered_sources],
        "related_ids": sorted(set(related_ids)),
        "occurrence": json_safe(occurrence),
        "discriminator": discriminator,
    }


def make_evidence_id(identity: Mapping[str, JsonValue]) -> str:
    """Return the stable truncated ID for one complete identity payload."""
    return "E-" + canonical_sha256(dict(identity))[:16]


def evidence_bundle_hash(bundle_without_hash: Mapping[str, JsonValue]) -> str:
    """Hash complete ordered evidence content and provenance."""
    return canonical_sha256(dict(bundle_without_hash))
