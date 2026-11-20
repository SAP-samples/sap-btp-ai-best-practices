"""Canonical and bounded prompt projection for evidence-grounded ECN analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from app.ecn.models import (
    AnalysisSectionKey,
    EvidenceBundle,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
    JsonValue,
)

MINIMUM_PROMPT_CHARACTERS = 4_096
"""Minimum cap that can safely contain the fixed prompt schema skeleton."""

MAX_PROJECTED_TEXT_CHARACTERS = 2_000
"""Maximum characters retained from any individual free-text evidence value."""

_BLOCKED_VALUE_KEYS = {
    "content_bytes",
    "normalized_content",
    "raw_object",
    "row_json",
    "rule_source_text",
    "source_bytes",
    "source_json",
}


class PromptProjectionError(ValueError):
    """Reject a prompt cap that cannot contain the fixed safe schema skeleton."""


@dataclass(frozen=True)
class PromptEvidenceItem:
    """Hold one safe, concise evidence item included in the provider prompt.

    Args:
        evidence_id: Stable evidence identifier retained for model citations.
        section_key: The only analysis section allowed to cite the item.
        kind: Epistemic classification of the deterministic item.
        tier: Optional source-quality tier.
        summary: Bounded evidence summary.
        sources: Exact artifact locations without source payload bytes.
        concise_values: Recursively filtered and bounded structured values.
        quality_qualifiers: Original and projection-specific qualifiers.
    """

    evidence_id: str
    section_key: AnalysisSectionKey
    kind: EvidenceKind
    tier: EvidenceTier | None
    summary: str
    sources: tuple[EvidenceSource, ...]
    concise_values: dict[str, JsonValue]
    quality_qualifiers: tuple[str, ...]


@dataclass(frozen=True)
class PromptEvidenceProjection:
    """Capture the exact bounded evidence contract sent to one model call.

    Args:
        ecn_row_id: Immutable ECN occurrence identifier.
        evidence_context: Artifact and deterministic-version provenance.
        ecn_header: Safe header values copied from the header evidence item.
        concise_evidence: Included whole evidence items in canonical priority order.
        aggregate_counts: Full-bundle counts, including items omitted from the prompt.
        omitted_evidence_count: Number of full-bundle items excluded by a cap.
        omitted_evidence_count_by_section: Per-section omission counts.
        sections_requiring_gap: Sections that must report insufficient evidence.
        allowed_evidence_ids_by_section: Included IDs that each section may cite.
        projection_warnings: Safe projection diagnostics visible to the model.
    """

    ecn_row_id: str
    evidence_context: EvidenceContext
    ecn_header: Mapping[str, JsonValue]
    concise_evidence: tuple[PromptEvidenceItem, ...]
    aggregate_counts: Mapping[str, int]
    omitted_evidence_count: int
    omitted_evidence_count_by_section: Mapping[AnalysisSectionKey, int]
    sections_requiring_gap: frozenset[AnalysisSectionKey]
    allowed_evidence_ids_by_section: Mapping[AnalysisSectionKey, frozenset[str]]
    projection_warnings: tuple[str, ...]


def _safe_value(value: JsonValue) -> tuple[JsonValue, bool]:
    """Recursively bound strings and remove raw-payload keys from one JSON value."""
    if isinstance(value, str):
        if len(value) <= MAX_PROJECTED_TEXT_CHARACTERS:
            return value, False
        return value[:MAX_PROJECTED_TEXT_CHARACTERS], True
    if isinstance(value, list):
        projected: list[JsonValue] = []
        truncated = False
        for item in value:
            safe_item, item_truncated = _safe_value(item)
            projected.append(safe_item)
            truncated = truncated or item_truncated
        return projected, truncated
    if isinstance(value, dict):
        projected_mapping: dict[str, JsonValue] = {}
        truncated = False
        for key in sorted(value):
            if key.casefold() in _BLOCKED_VALUE_KEYS:
                continue
            safe_item, item_truncated = _safe_value(value[key])
            projected_mapping[key] = safe_item
            truncated = truncated or item_truncated
        return projected_mapping, truncated
    return value, False


def _project_item(item: EvidenceItem) -> PromptEvidenceItem:
    """Convert one full evidence item to its payload-safe prompt representation."""
    safe_summary, summary_truncated = _safe_value(item.summary)
    safe_values, values_truncated = _safe_value(item.values)
    qualifiers = list(item.quality_qualifiers)
    if summary_truncated or values_truncated:
        qualifiers.append("PROMPT_FREE_TEXT_TRUNCATED")
    return PromptEvidenceItem(
        evidence_id=item.evidence_id,
        section_key=item.section_key,
        kind=item.kind,
        tier=item.tier,
        summary=str(safe_summary),
        sources=item.sources,
        concise_values=dict(safe_values),
        quality_qualifiers=tuple(dict.fromkeys(qualifiers)),
    )


def _priority(item: EvidenceItem) -> tuple[int, str, str]:
    """Return the fixed evidence priority followed by stable tie-break fields."""
    if item.section_key in {
        AnalysisSectionKey.EXECUTIVE_SUMMARY,
        AnalysisSectionKey.CHANGE_INTENT,
    }:
        rank = 0
    elif item.tier is EvidenceTier.DIRECT_ECN_RAW:
        rank = 1
    elif item.tier is EvidenceTier.EXACT_WTPART_WHERE_USED:
        rank = 2
    elif (
        item.section_key is AnalysisSectionKey.CONFIGURATION_SCOPE and item.tier is None
    ):
        rank = 3
    elif item.kind in {
        EvidenceKind.WARNING,
        EvidenceKind.MISSING_INPUT,
        EvidenceKind.CONTRADICTION,
    }:
        rank = 4
    elif item.tier is EvidenceTier.CONTEXTUAL_BOM_OCCURRENCE:
        rank = 5
    elif item.tier is EvidenceTier.DOCUMENT_OWNER_CONTEXT:
        rank = 6
    else:
        rank = 7
    return rank, item.section_key.value, item.evidence_id


def _all_items(evidence: EvidenceBundle) -> tuple[EvidenceItem, ...]:
    """Return every evidence item exactly once in domain section order."""
    return tuple(
        item
        for key in AnalysisSectionKey
        for item in getattr(evidence.sections, key.value)
    )


def _aggregate_counts(items: tuple[EvidenceItem, ...]) -> dict[str, int]:
    """Build deterministic full-bundle counts without copying source content."""
    counts: dict[str, int] = {"evidence_total": len(items)}
    for key in AnalysisSectionKey:
        counts[f"section:{key.value}"] = sum(item.section_key is key for item in items)
    for kind in EvidenceKind:
        counts[f"kind:{kind.value}"] = sum(item.kind is kind for item in items)
    for tier in EvidenceTier:
        counts[f"tier:{tier.value}"] = sum(item.tier is tier for item in items)
    return counts


def _context_payload(context: EvidenceContext) -> dict[str, JsonValue]:
    """Serialize provenance context without source artifacts or full rule content."""
    return {
        "analysis_as_of_date": (
            context.analysis_as_of_date.isoformat()
            if isinstance(context.analysis_as_of_date, date)
            else None
        ),
        "bom_artifact_id": context.bom_artifact_id,
        "deterministic_logic_version": context.deterministic_logic_version,
        "ecn_artifact_id": context.ecn_artifact_id,
        "parser_version": context.parser_version,
        "rule_artifact_id": context.rule_artifact_id,
        "rule_evaluator_version": context.rule_evaluator_version,
        "upload_id": context.upload_id,
    }


def _source_payload(source: EvidenceSource) -> dict[str, JsonValue]:
    """Serialize one exact evidence pointer without any referenced payload."""
    return {
        "artifact_id": source.artifact_id,
        "artifact_kind": source.artifact_kind.value,
        "bom_row_id": source.bom_row_id,
        "json_path": source.json_path,
        "sheet_name": source.sheet_name,
        "source_row_number": source.source_row_number,
    }


def _item_payload(item: PromptEvidenceItem) -> dict[str, JsonValue]:
    """Serialize one projected evidence item to a canonical JSON-compatible mapping."""
    return {
        "concise_values": item.concise_values,
        "evidence_id": item.evidence_id,
        "kind": item.kind.value,
        "quality_qualifiers": list(item.quality_qualifiers),
        "section_key": item.section_key.value,
        "sources": [_source_payload(source) for source in item.sources],
        "summary": item.summary,
        "tier": item.tier.value if item.tier is not None else None,
    }


def _projection_payload(projection: PromptEvidenceProjection) -> dict[str, JsonValue]:
    """Return the exact provider-visible payload used for size enforcement."""
    return {
        "aggregate_counts": dict(projection.aggregate_counts),
        "allowed_evidence_ids_by_section": {
            key.value: sorted(projection.allowed_evidence_ids_by_section[key])
            for key in AnalysisSectionKey
        },
        "concise_evidence": [
            _item_payload(item) for item in projection.concise_evidence
        ],
        "ecn_header": dict(projection.ecn_header),
        "ecn_row_id": projection.ecn_row_id,
        "evidence_context": _context_payload(projection.evidence_context),
        "omitted_evidence_count": projection.omitted_evidence_count,
        "omitted_evidence_count_by_section": {
            key.value: projection.omitted_evidence_count_by_section[key]
            for key in AnalysisSectionKey
        },
        "projection_warnings": list(projection.projection_warnings),
        "sections_requiring_gap": sorted(
            key.value for key in projection.sections_requiring_gap
        ),
    }


def _canonical_json(payload: Mapping[str, JsonValue]) -> str:
    """Serialize a mapping as sorted compact JSON while preserving Unicode text."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_character_count(payload: Mapping[str, JsonValue]) -> int:
    """Count Unicode code points in the exact compact canonical JSON sent to the model."""
    return len(_canonical_json(payload))


def _build_projection(
    evidence: EvidenceBundle,
    full_items: tuple[EvidenceItem, ...],
    included: list[PromptEvidenceItem],
) -> PromptEvidenceProjection:
    """Assemble derived omission, citation, warning, and required-gap metadata."""
    included_ids = {item.evidence_id for item in included}
    omitted = tuple(item for item in full_items if item.evidence_id not in included_ids)
    omitted_by_section = {
        key: sum(item.section_key is key for item in omitted)
        for key in AnalysisSectionKey
    }
    required_gaps = {
        item.section_key
        for item in full_items
        if item.kind in {EvidenceKind.MISSING_INPUT, EvidenceKind.CONTRADICTION}
    }
    required_gaps.update(key for key, count in omitted_by_section.items() if count)
    warnings: list[str] = []
    if omitted:
        warnings.append("PROMPT_PROJECTION_TRUNCATED")
    if any(
        "PROMPT_FREE_TEXT_TRUNCATED" in item.quality_qualifiers for item in included
    ):
        warnings.append("PROMPT_FREE_TEXT_TRUNCATED")
    allowed = {
        key: frozenset(item.evidence_id for item in included if item.section_key is key)
        for key in AnalysisSectionKey
    }
    header: dict[str, JsonValue] = {}
    for item in included:
        if item.section_key is AnalysisSectionKey.EXECUTIVE_SUMMARY:
            for key in (
                "ecn_number",
                "title",
                "description_summary",
                "description_text",
                "source_ordinal",
            ):
                if key in item.concise_values:
                    header[key] = item.concise_values[key]
            if header:
                break
    return PromptEvidenceProjection(
        ecn_row_id=evidence.ecn_row_id,
        evidence_context=evidence.context,
        ecn_header=header,
        concise_evidence=tuple(included),
        aggregate_counts=_aggregate_counts(full_items),
        omitted_evidence_count=len(omitted),
        omitted_evidence_count_by_section=omitted_by_section,
        sections_requiring_gap=frozenset(required_gaps),
        allowed_evidence_ids_by_section=allowed,
        projection_warnings=tuple(warnings),
    )


def project_evidence_for_prompt(
    evidence: EvidenceBundle,
    maximum_items: int,
    maximum_characters: int,
) -> PromptEvidenceProjection:
    """Build a bounded canonical projection while retaining stable evidence IDs.

    Args:
        evidence: Complete deterministic evidence bundle for one ECN occurrence.
        maximum_items: Maximum number of whole evidence items to include.
        maximum_characters: Maximum Unicode characters in the exact JSON input.

    Returns:
        A deterministic projection with explicit omission and gap metadata.

    Raises:
        PromptProjectionError: If the cap is below the safe 4,096-character floor
            or the fixed empty projection cannot fit.
        ValueError: If ``maximum_items`` is not positive.
    """
    if maximum_characters < MINIMUM_PROMPT_CHARACTERS:
        raise PromptProjectionError("PROMPT_LIMIT_TOO_SMALL")
    if maximum_items <= 0:
        raise ValueError("maximum_items must be positive")
    full_items = _all_items(evidence)
    candidates = [_project_item(item) for item in sorted(full_items, key=_priority)]
    included = candidates[:maximum_items]
    projection = _build_projection(evidence, full_items, included)
    while (
        _canonical_character_count(_projection_payload(projection)) > maximum_characters
        and included
    ):
        included.pop()
        projection = _build_projection(evidence, full_items, included)
    if _canonical_character_count(_projection_payload(projection)) > maximum_characters:
        raise PromptProjectionError("PROMPT_LIMIT_TOO_SMALL")
    return projection


def build_impact_instructions() -> str:
    """Return the versioned system instruction for the six-section analysis."""
    return (
        "ECN impact prompt ecn-impact-v1. Return exactly the supplied strict six-section "
        "JSON schema. Cite only evidence IDs allowed for that same section. Every section "
        "listed in sections_requiring_gap must use INSUFFICIENT_EVIDENCE and name the "
        "missing evidence. NOT_APPLICABLE must remain visible with a short grounded "
        "explanation and evidence. Apply these hard inference boundaries: revision does "
        "not prove redesign; co-occurrence does not prove replacement; one-sided lists "
        "do not prove add or remove; document nesting does not prove owner-material impact; "
        "no BOM match does not prove no impact; unknown rule facts do not prove applicability; "
        "descriptions do not prove Make/Buy. Plants in plant_impact come from the brands "
        "named by the conditional rules of each VC group, mapped to plants in Settings: "
        "they state where a VC group can be built, not a confirmed configuration or take "
        "rate. Do not infer beyond the projected evidence."
    )


def build_impact_input(
    projection: PromptEvidenceProjection,
) -> list[dict[str, JsonValue]]:
    """Return one canonical Responses API user input containing projected evidence only."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": _canonical_json(_projection_payload(projection)),
                }
            ],
        }
    ]
