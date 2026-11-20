"""Project stored deterministic evidence into grouped business-readable records.

The persisted evidence bundle remains the immutable audit source.  This module
creates a version-independent read projection that keeps material, BOM, scope,
and source coordinates visible while moving internal identifiers and raw enums
into an explicit technical-details object.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from app.ecn.models import (
    AnalysisSectionKey,
    BusinessDirection,
    BusinessEvidenceSection,
    EvidenceBundle,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    JsonValue,
)


@dataclass(frozen=True)
class BusinessTechnicalDetails:
    """Retain exact audit values that the default user interface keeps collapsed.

    Args:
        internal_evidence_ids: Stable stored evidence identifiers in the group.
        raw_kinds: Stored epistemic enums represented by the group.
        raw_tiers: Stored evidence-tier enums represented by the group.
        raw_values: Original structured values for each grouped evidence item.
        quality_qualifiers: Exact machine-readable source-quality qualifiers.
        related_ids: Internal related domain identifiers.
    """

    internal_evidence_ids: tuple[str, ...]
    raw_kinds: tuple[str, ...]
    raw_tiers: tuple[str, ...]
    raw_values: tuple[dict[str, JsonValue], ...]
    quality_qualifiers: tuple[str, ...]
    related_ids: tuple[str, ...]


@dataclass(frozen=True)
class BusinessEvidenceRecord:
    """Expose one grouped fact using identifiers meaningful to business users.

    Args:
        section: Business change set containing the record.
        label: Short localized display label.
        summary: Readable explanation of the fact or issue.
        material_number: Material or component identifier when available.
        description: Material or BOM description when available.
        revision: Material revision when available.
        direction: Business change direction when applicable.
        parent: Immediate BOM parent material when available.
        bom_path: Ordered BOM ancestor path when available.
        quantity: Source quantity rendered without precision loss.
        effectivity: Concise source effectivity interval when available.
        plant: Plant identifier or bounded site name when available.
        matched_row_count: Number of raw evidence rows represented.
        sources: Exact source coordinates, capped to representative examples.
        technical_details: Full internal audit values for optional disclosure.
    """

    section: BusinessEvidenceSection
    label: str
    summary: str
    material_number: str | None
    description: str | None
    revision: str | None
    direction: BusinessDirection | None
    parent: str | None
    bom_path: tuple[str, ...]
    quantity: str | None
    effectivity: str | None
    plant: str | None
    matched_row_count: int
    sources: tuple[EvidenceSource, ...]
    technical_details: BusinessTechnicalDetails


@dataclass(frozen=True)
class BusinessEvidencePage:
    """Hold one bounded, deterministic page of a business evidence section."""

    section: BusinessEvidenceSection
    items: tuple[BusinessEvidenceRecord, ...]
    page: int
    page_size: int
    total_items: int
    total_pages: int


_QUALITY_MESSAGES: dict[str, str] = {
    "SOURCE_RELATION_ERROR_VALUE": (
        "{count} source relationship{suffix} contain an invalid value; "
        "exact matching may be incomplete."
    ),
    "TECHNICAL_IDENTITY_CONTRADICTION": (
        "{count} source object{suffix} contain conflicting technical identities; "
        "material matching requires review."
    ),
    "ENRICHED_PATH_CONTRADICTION": (
        "{count} BOM occurrence{suffix} contain contradictory hierarchy data; "
        "the affected path may be incomplete."
    ),
    "SUSPICIOUS_OR_OF_NEGATIONS": (
        "{count} configuration rule{suffix} use an ambiguous negative expression; "
        "applicability requires review."
    ),
}


def _all_items(bundle: EvidenceBundle) -> tuple[EvidenceItem, ...]:
    """Return all persisted evidence in canonical section and source order."""
    return tuple(
        item
        for section in AnalysisSectionKey
        for item in getattr(bundle.sections, section.value)
    )


def _text(value: JsonValue | None) -> str | None:
    """Return a non-empty scalar as text without inventing missing values."""
    if value is None or isinstance(value, (dict, list)):
        return None
    rendered = str(value).strip()
    return rendered or None


def _path(value: JsonValue | None) -> tuple[str, ...]:
    """Return a source path only when the stored value is a scalar list."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item).strip())


def _effectivity(values: dict[str, JsonValue]) -> str | None:
    """Render a bounded effectivity interval from normalized or raw date values."""
    start = _text(values.get("valid_from")) or _text(values.get("valid_from_raw"))
    end = _text(values.get("valid_to")) or _text(values.get("valid_to_raw"))
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"From {start}"
    if end:
        return f"Until {end}"
    return None


def _humanize_code(code: str) -> str:
    """Turn an unknown machine code into a readable label without guessing meaning."""
    words = re.sub(r"[_\-]+", " ", code).strip().lower()
    return words[:1].upper() + words[1:] if words else "Source quality issue"


def _technical(items: tuple[EvidenceItem, ...]) -> BusinessTechnicalDetails:
    """Build the collapsed technical audit object for one business group."""
    return BusinessTechnicalDetails(
        internal_evidence_ids=tuple(item.evidence_id for item in items),
        raw_kinds=tuple(dict.fromkeys(item.kind.value for item in items)),
        raw_tiers=tuple(
            dict.fromkeys(item.tier.value for item in items if item.tier is not None)
        ),
        raw_values=tuple(item.values for item in items),
        quality_qualifiers=tuple(
            sorted({value for item in items for value in item.quality_qualifiers})
        ),
        related_ids=tuple(sorted({value for item in items for value in item.related_ids})),
    )


def _representative_sources(items: tuple[EvidenceItem, ...]) -> tuple[EvidenceSource, ...]:
    """Return at most three unique source coordinates for default display."""
    sources: list[EvidenceSource] = []
    for item in items:
        for source in item.sources:
            if source not in sources:
                sources.append(source)
            if len(sources) == 3:
                return tuple(sources)
    return tuple(sources)


def _record(
    items: tuple[EvidenceItem, ...],
    *,
    section: BusinessEvidenceSection,
    label: str,
    summary: str,
    direction: BusinessDirection | None = None,
) -> BusinessEvidenceRecord:
    """Create one business record from items with compatible stored values."""
    values = items[0].values
    material_number = (
        _text(values.get("number"))
        or _text(values.get("component_number"))
        or _text(values.get("material_number"))
    )
    return BusinessEvidenceRecord(
        section=section,
        label=label,
        summary=summary,
        material_number=material_number,
        description=_text(values.get("description_text")),
        revision=_text(values.get("revision")),
        direction=direction,
        parent=_text(values.get("parent_material_number")),
        bom_path=_path(values.get("parent_path")),
        quantity=_text(values.get("quantity")),
        effectivity=_effectivity(values),
        plant=_text(values.get("plant")),
        matched_row_count=len(items),
        sources=_representative_sources(items),
        technical_details=_technical(items),
    )


def _material_section(item: EvidenceItem) -> tuple[BusinessEvidenceSection, str, BusinessDirection]:
    """Map one source material role to its non-technical business presentation."""
    role = _text(item.values.get("role"))
    if role == "RESULTING":
        return (
            BusinessEvidenceSection.ADDED_MATERIALS,
            "Resulting object reference",
            BusinessDirection.REFERENCED,
        )
    return (
        BusinessEvidenceSection.MODIFIED_MATERIALS,
        "Affected object reference",
        BusinessDirection.REFERENCED,
    )


def _ordinary_record(item: EvidenceItem) -> BusinessEvidenceRecord | None:
    """Project one non-quality stored item into a business evidence record."""
    if item.section_key is AnalysisSectionKey.AFFECTED_MATERIALS:
        section, label, direction = _material_section(item)
        number = _text(item.values.get("number")) or "an unidentified material"
        return _record(
            (item,),
            section=section,
            label=label,
            summary=f"{label} {number}.",
            direction=direction,
        )
    if item.section_key is AnalysisSectionKey.EBOM_MBOM_IMPACT:
        bom_kind = _text(item.values.get("bom_kind")) or ""
        is_mbom = "MBOM" in bom_kind
        section = (
            BusinessEvidenceSection.MBOM_MATCHES
            if is_mbom
            else BusinessEvidenceSection.EBOM_MATCHES
        )
        label = "MBOM match" if is_mbom else "EBOM match"
        return _record(
            (item,),
            section=section,
            label=label,
            summary=item.summary,
            direction=BusinessDirection.MATCHED,
        )
    if item.section_key is AnalysisSectionKey.CONFIGURATION_SCOPE:
        return _record(
            (item,),
            section=BusinessEvidenceSection.CONFIGURATION,
            label="Configuration or site applicability",
            summary=item.summary,
        )
    return None


def _group_key(record: BusinessEvidenceRecord) -> tuple[object, ...]:
    """Return exact business fields used to collapse duplicate source rows."""
    return (
        record.section,
        record.label,
        record.material_number,
        record.description,
        record.revision,
        record.direction,
        record.parent,
        record.bom_path,
        record.quantity,
        record.effectivity,
        record.plant,
        record.summary,
    )


def _merge_records(records: list[BusinessEvidenceRecord]) -> list[BusinessEvidenceRecord]:
    """Merge exact duplicate business rows while preserving complete technical audit data."""
    grouped: dict[tuple[object, ...], list[BusinessEvidenceRecord]] = {}
    for record in records:
        grouped.setdefault(_group_key(record), []).append(record)
    merged: list[BusinessEvidenceRecord] = []
    for group in grouped.values():
        first = group[0]
        # Merge directly from projected details; recreating domain evidence would
        # lose tier/value provenance and is therefore intentionally avoided.
        sources = tuple(dict.fromkeys(source for record in group for source in record.sources))[:3]
        technical = BusinessTechnicalDetails(
            internal_evidence_ids=tuple(
                value for record in group for value in record.technical_details.internal_evidence_ids
            ),
            raw_kinds=tuple(
                dict.fromkeys(value for record in group for value in record.technical_details.raw_kinds)
            ),
            raw_tiers=tuple(
                dict.fromkeys(value for record in group for value in record.technical_details.raw_tiers)
            ),
            raw_values=tuple(
                value for record in group for value in record.technical_details.raw_values
            ),
            quality_qualifiers=tuple(
                sorted({value for record in group for value in record.technical_details.quality_qualifiers})
            ),
            related_ids=tuple(
                sorted({value for record in group for value in record.technical_details.related_ids})
            ),
        )
        merged.append(
            BusinessEvidenceRecord(
                section=first.section,
                label=first.label,
                summary=first.summary,
                material_number=first.material_number,
                description=first.description,
                revision=first.revision,
                direction=first.direction,
                parent=first.parent,
                bom_path=first.bom_path,
                quantity=first.quantity,
                effectivity=first.effectivity,
                plant=first.plant,
                matched_row_count=sum(record.matched_row_count for record in group),
                sources=sources,
                technical_details=technical,
            )
        )
    return merged


def _quality_records(items: tuple[EvidenceItem, ...]) -> list[BusinessEvidenceRecord]:
    """Group source parsing and qualifier issues by code and impact."""
    groups: dict[str, list[EvidenceItem]] = {}
    for item in items:
        if item.kind not in {EvidenceKind.WARNING, EvidenceKind.CONTRADICTION}:
            continue
        code = _text(item.values.get("code")) or next(
            iter(item.quality_qualifiers), "SOURCE_QUALITY_ISSUE"
        )
        groups.setdefault(code, []).append(item)
    records: list[BusinessEvidenceRecord] = []
    for code, grouped_items in sorted(groups.items()):
        evidence = tuple(grouped_items)
        count = len(evidence)
        suffix = "" if count == 1 else "s"
        template = _QUALITY_MESSAGES.get(code)
        summary = (
            template.format(count=count, suffix=suffix)
            if template is not None
            else f"{count} source record{suffix} report: {_humanize_code(code)}."
        )
        records.append(
            _record(
                evidence,
                section=BusinessEvidenceSection.DATA_QUALITY,
                label="Data quality issue",
                summary=summary,
            )
        )
    return records


def _missing_records(items: tuple[EvidenceItem, ...]) -> list[BusinessEvidenceRecord]:
    """Return only actionable unsupported conclusions as missing evidence."""
    return [
        _record(
            (item,),
            section=BusinessEvidenceSection.MISSING_EVIDENCE,
            label="Missing evidence",
            summary=item.summary,
        )
        for item in items
        if item.kind is EvidenceKind.MISSING_INPUT
    ]


def business_evidence_records(bundle: EvidenceBundle) -> tuple[BusinessEvidenceRecord, ...]:
    """Return stable business records derived from an immutable evidence bundle.

    Args:
        bundle: Complete stored deterministic evidence for one ECN occurrence.

    Returns:
        Grouped records ordered by business section and meaningful source fields.
    """
    items = _all_items(bundle)
    ordinary = [record for item in items if (record := _ordinary_record(item))]
    records = [*_merge_records(ordinary), *_quality_records(items), *_missing_records(items)]
    section_order = {section: index for index, section in enumerate(BusinessEvidenceSection)}
    return tuple(
        sorted(
            records,
            key=lambda record: (
                section_order[record.section],
                record.material_number or "",
                record.parent or "",
                record.plant or "",
                record.summary,
                record.technical_details.internal_evidence_ids,
            ),
        )
    )


def paginate_business_evidence(
    bundle: EvidenceBundle,
    section: BusinessEvidenceSection,
    *,
    page: int,
    page_size: int,
) -> BusinessEvidencePage:
    """Return one validated page for a single business evidence change set.

    Args:
        bundle: Complete stored deterministic evidence for one ECN occurrence.
        section: Business section to retrieve.
        page: One-based page number.
        page_size: Number of rows from 1 through 100.

    Returns:
        Stable page plus total counts.

    Raises:
        ValueError: When page or page_size is outside the API bounds.
    """
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    matching = tuple(
        record for record in business_evidence_records(bundle) if record.section is section
    )
    start = (page - 1) * page_size
    total = len(matching)
    return BusinessEvidencePage(
        section=section,
        items=matching[start : start + page_size],
        page=page,
        page_size=page_size,
        total_items=total,
        total_pages=0 if total == 0 else math.ceil(total / page_size),
    )


def business_evidence_record_from_item(item: EvidenceItem) -> BusinessEvidenceRecord:
    """Project one exact stored item for the audit lookup endpoint.

    Args:
        item: Stored item already verified to belong to the requested ECN.

    Returns:
        Business-readable record with exact raw values under technical details.
    """
    if item.kind in {EvidenceKind.WARNING, EvidenceKind.CONTRADICTION}:
        return _quality_records((item,))[0]
    if item.kind is EvidenceKind.MISSING_INPUT:
        return _missing_records((item,))[0]
    ordinary = _ordinary_record(item)
    if ordinary is not None:
        return ordinary
    return _record(
        (item,),
        section=BusinessEvidenceSection.CONFIGURATION,
        label="Source evidence",
        summary=item.summary,
    )
