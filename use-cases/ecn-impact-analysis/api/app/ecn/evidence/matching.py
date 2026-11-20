"""Exact occurrence-aware ECN-to-BOM indexing and conservative matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal, TypeAlias

from app.ecn.models import (
    ArtifactKind,
    BomKind,
    BomRow,
    BomRowClassification,
    EvidenceKind,
    EvidenceSource,
    JsonValue,
    NormalizedEcn,
)


class EnrichedMatchStrength(StrEnum):
    """Describe enriched EBOM evidence reached through an exact seed."""

    COMPONENT_AND_ECN_CONFIRMED = "COMPONENT_AND_ECN_CONFIRMED"
    COMPONENT_ONLY_CONTEXT = "COMPONENT_ONLY_CONTEXT"


class MatchPath(StrEnum):
    """Describe the exact source path that created a structural seed."""

    RAW_ECN_COMPONENT = "RAW_ECN_COMPONENT"
    JSON_WTPART = "JSON_WTPART"


class StructuralSeedOrigin(StrEnum):
    """Describe the source class that permits exact component lookup."""

    RAW_MATERIAL_LIKE_OBSERVED = "RAW_MATERIAL_LIKE_OBSERVED"
    JSON_TECHNICAL_WTPART = "JSON_TECHNICAL_WTPART"


OccurrenceKey: TypeAlias = tuple[
    BomKind,
    str,
    str,
    int,
    tuple[str, ...],
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]
"""Complete immutable identity of one BOM occurrence."""


@dataclass(frozen=True)
class BomOccurrence:
    """Project one occurrence while retaining join and interpretation context."""

    bom_row_id: str
    artifact_id: str
    bom_kind: BomKind
    sheet_name: str
    source_row_number: int
    parent_bom_row_id: str | None
    parent_path: tuple[str, ...]
    item_or_position: str | None
    row_classification: BomRowClassification
    description_text: str | None
    material_number: str | None
    parent_material_number: str | None
    main_value: str | None
    sub_value: str | None
    vc_value: str | None
    component_number: str | None
    change_number: str | None
    change_number_to: str | None
    quantity: Decimal | None
    unit: str | None
    plant: str | None
    alternative: str | None
    valid_from: date | None
    valid_to: date | None
    valid_from_raw: str | None
    valid_to_raw: str | None
    is_migration_date: bool
    quality_qualifiers: tuple[str, ...]
    quality_details: dict[str, JsonValue]


@dataclass(frozen=True)
class StructuralObjectSeed:
    """Hold an exact material number authorized to seed where-used matching."""

    seed_id: str
    number: str
    origin: StructuralSeedOrigin
    source: EvidenceSource
    source_subtype: str | None
    qualifiers: tuple[str, ...]


@dataclass(frozen=True)
class OccurrenceMatch:
    """Retain one seed-to-occurrence match path and enriched strength."""

    seed: StructuralObjectSeed
    occurrence: BomOccurrence
    path: MatchPath
    enriched_strength: EnrichedMatchStrength | None


@dataclass(frozen=True)
class DocumentOwnerBridge:
    """Hold hierarchy context without claiming that the owner changed."""

    document_row_id: str
    owner_row_id: str
    owner_component: str | None
    asserts_owner_changed: Literal[False] = False


@dataclass(frozen=True)
class DeterministicDiagnostic:
    """Hold a warning, contradiction, or ordinary missing-input outcome."""

    kind: EvidenceKind
    code: str
    message: str
    sources: tuple[EvidenceSource, ...]


@dataclass(frozen=True)
class BomIndexes:
    """Hold exact trimmed indexes plus row identity for hierarchy traversal."""

    raw_by_change_number: Mapping[str, tuple[BomOccurrence, ...]]
    enriched_by_component: Mapping[str, tuple[BomOccurrence, ...]]
    mbom_by_component: Mapping[str, tuple[BomOccurrence, ...]]
    by_row_id: Mapping[str, BomOccurrence]


@dataclass(frozen=True)
class MatchResult:
    """Hold the complete conservative crosswalk result for one ECN occurrence."""

    raw_direct: tuple[BomOccurrence, ...]
    eligible_structural_seeds: tuple[StructuralObjectSeed, ...]
    enriched_matches: tuple[OccurrenceMatch, ...]
    mbom_presence: tuple[OccurrenceMatch, ...]
    document_owner_context: tuple[DocumentOwnerBridge, ...]
    context_rows_by_id: Mapping[str, BomOccurrence]
    diagnostics: tuple[DeterministicDiagnostic, ...]


def occurrence_key(occurrence: BomOccurrence) -> OccurrenceKey:
    """Return the complete identity of one BOM occurrence."""
    return (
        occurrence.bom_kind,
        occurrence.artifact_id,
        occurrence.sheet_name,
        occurrence.source_row_number,
        occurrence.parent_path,
        occurrence.item_or_position,
        occurrence.material_number,
        occurrence.parent_material_number,
        occurrence.component_number,
        occurrence.plant,
        occurrence.alternative,
        occurrence.valid_from_raw,
        occurrence.valid_to_raw,
    )


def _key(value: str | None) -> str | None:
    """Trim only external whitespace and preserve case and leading zeroes."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _source(occurrence: BomOccurrence) -> EvidenceSource:
    """Create one exact workbook coordinate for an occurrence."""
    return EvidenceSource(
        artifact_id=occurrence.artifact_id,
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        json_path=None,
        sheet_name=occurrence.sheet_name,
        bom_row_id=occurrence.bom_row_id,
        source_row_number=occurrence.source_row_number,
    )


def _occurrence_from_row(
    row: BomRow, parent_path: tuple[str, ...]
) -> BomOccurrence:
    """Project a row without losing join, scope, date, or quality context."""
    item_or_position = row.position_value
    if row.bom_kind is BomKind.RAW_EBOM_130:
        item_or_position = row.item_value or row.position_value
    return BomOccurrence(
        bom_row_id=row.bom_row_id,
        artifact_id=row.artifact_id,
        bom_kind=row.bom_kind,
        sheet_name=row.sheet_name,
        source_row_number=row.source_row_number,
        parent_bom_row_id=row.parent_bom_row_id,
        parent_path=parent_path,
        item_or_position=item_or_position,
        row_classification=row.row_classification,
        description_text=row.description_text,
        material_number=_key(row.material_number),
        parent_material_number=_key(row.parent_material_number),
        main_value=_key(row.main_value),
        sub_value=_key(row.sub_value),
        vc_value=_key(row.vc_value),
        component_number=_key(row.component_number),
        change_number=_key(row.change_number),
        change_number_to=_key(row.change_number_to),
        quantity=row.quantity,
        unit=_key(row.unit),
        plant=_key(row.plant),
        alternative=_key(row.alternative),
        valid_from=row.valid_from_date,
        valid_to=row.valid_to_date,
        valid_from_raw=_key(row.valid_from_raw),
        valid_to_raw=_key(row.valid_to_raw),
        is_migration_date=row.is_migration_date,
        quality_qualifiers=row.quality_qualifiers,
        quality_details=dict(row.quality_details),
    )


def _match_sort_key(match: OccurrenceMatch) -> tuple[str, str, int, str, str]:
    """Return stable order without collapsing distinct occurrences."""
    return (
        match.occurrence.artifact_id,
        match.occurrence.sheet_name,
        match.occurrence.source_row_number,
        match.seed.seed_id,
        match.path.value,
    )


def build_bom_indexes(rows: Sequence[BomRow]) -> BomIndexes:
    """Build exact occurrence-aware indexes from normalized HANA rows.

    Args:
        rows: Fully normalized rows from one immutable BOM artifact.

    Returns:
        Exact change/component indexes and identity-preserving occurrences.
    """
    row_map = {row.bom_row_id: row for row in rows}

    def parent_path(row: BomRow) -> tuple[str, ...]:
        """Resolve an occurrence's already-normalized structural parent path."""
        if row.bom_kind is BomKind.ENRICHED_EBOM:
            values = (
                row.main_value,
                row.sub_value,
                row.vc_value,
                row.level_4_value,
                row.level_5_value,
                row.level_6_value,
                row.parent_material_number,
            )
            return tuple(value.strip() for value in values if value and value.strip())
        if row.bom_kind is BomKind.MBOM_130:
            return tuple(
                value for value in (_key(row.parent_material_number),) if value
            )
        ancestors: list[str] = []
        seen: set[str] = set()
        parent_id = row.parent_bom_row_id
        while parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            parent = row_map.get(parent_id)
            if (
                parent is None
                or parent.artifact_id != row.artifact_id
                or parent.sheet_name != row.sheet_name
            ):
                break
            component = _key(parent.component_number)
            if component is not None:
                ancestors.append(component)
            parent_id = parent.parent_bom_row_id
        return tuple(reversed(ancestors))

    occurrences = tuple(
        _occurrence_from_row(row, parent_path(row)) for row in rows
    )
    raw: dict[str, list[BomOccurrence]] = {}
    enriched: dict[str, list[BomOccurrence]] = {}
    mbom: dict[str, list[BomOccurrence]] = {}
    for occurrence in occurrences:
        change = _key(occurrence.change_number)
        component = _key(occurrence.component_number)
        if occurrence.bom_kind is BomKind.RAW_EBOM_130 and change is not None:
            raw.setdefault(change, []).append(occurrence)
        elif occurrence.bom_kind is BomKind.ENRICHED_EBOM and component is not None:
            enriched.setdefault(component, []).append(occurrence)
        elif occurrence.bom_kind is BomKind.MBOM_130 and component is not None:
            mbom.setdefault(component, []).append(occurrence)

    def freeze(
        values: Mapping[str, list[BomOccurrence]],
    ) -> dict[str, tuple[BomOccurrence, ...]]:
        """Freeze an index into stable source-coordinate order."""
        return {
            key: tuple(
                sorted(
                    items,
                    key=lambda item: (item.sheet_name, item.source_row_number),
                )
            )
            for key, items in values.items()
        }

    return BomIndexes(
        freeze(raw),
        freeze(enriched),
        freeze(mbom),
        {item.bom_row_id: item for item in occurrences},
    )


def _nearest_material_owner(
    document: BomOccurrence, indexes: BomIndexes
) -> BomOccurrence | None:
    """Walk parent row IDs to the nearest observed material-like ancestor."""
    seen: set[str] = set()
    parent_id = document.parent_bom_row_id
    while parent_id is not None and parent_id not in seen:
        seen.add(parent_id)
        parent = indexes.by_row_id.get(parent_id)
        if parent is None:
            return None
        if parent.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED:
            return parent
        parent_id = parent.parent_bom_row_id
    return None


def match_ecn_to_bom(ecn: NormalizedEcn, indexes: BomIndexes) -> MatchResult:
    """Apply exact joins and retain every distinct occurrence path.

    Args:
        ecn: One immutable normalized ECN occurrence.
        indexes: Exact indexes built from one immutable BOM snapshot.

    Returns:
        Direct raw rows, eligible seeds, exact occurrence matches, hierarchy
        bridges, and explicit diagnostics without fuzzy or inferred links.
    """
    diagnostics: list[DeterministicDiagnostic] = []
    ecn_key = _key(ecn.ecn_number)
    raw_direct = indexes.raw_by_change_number.get(ecn_key, ()) if ecn_key else ()
    if ecn_key is None:
        diagnostics.append(
            DeterministicDiagnostic(
                EvidenceKind.MISSING_INPUT,
                "BUSINESS_ECN_MISSING",
                "No business ECN number is available for the raw Change No. join.",
                (),
            )
        )
    elif not raw_direct:
        diagnostics.append(
            DeterministicDiagnostic(
                EvidenceKind.MISSING_INPUT,
                "RAW_ECN_NO_MATCH",
                "No raw EBOM row has the exact ECN Change No.",
                (),
            )
        )
    seeds: list[StructuralObjectSeed] = []
    for row in raw_direct:
        component = _key(row.component_number)
        if (
            row.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED
            and component
        ):
            seeds.append(
                StructuralObjectSeed(
                    seed_id=f"raw:{row.bom_row_id}",
                    number=component,
                    origin=StructuralSeedOrigin.RAW_MATERIAL_LIKE_OBSERVED,
                    source=_source(row),
                    source_subtype=None,
                    qualifiers=tuple(
                        sorted(
                            set(row.quality_qualifiers)
                            | {"TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED"}
                        )
                    ),
                )
            )
        elif row.row_classification is BomRowClassification.MATERIAL_LIKE_OBSERVED:
            diagnostics.append(
                DeterministicDiagnostic(
                    EvidenceKind.MISSING_INPUT,
                    "RAW_COMPONENT_NUMBER_MISSING",
                    "A material-like raw ECN row has no component number for exact where-used matching.",
                    (_source(row),),
                )
            )
    for source_object in (*ecn.affected_objects, *ecn.resulting_objects):
        number = _key(source_object.number)
        source = EvidenceSource(
            artifact_id=ecn.artifact_id,
            artifact_kind=ArtifactKind.ECN_JSON,
            json_path=source_object.source_path,
            sheet_name=None,
            bom_row_id=None,
            source_row_number=None,
        )
        if source_object.is_technical_wtpart and number:
            seeds.append(
                StructuralObjectSeed(
                    seed_id=f"json:{source_object.source_object_id}",
                    number=number,
                    origin=StructuralSeedOrigin.JSON_TECHNICAL_WTPART,
                    source=source,
                    source_subtype=source_object.source_subtype,
                    qualifiers=source_object.quality_qualifiers,
                )
            )
        elif source_object.is_technical_wtpart:
            diagnostics.append(
                DeterministicDiagnostic(
                    EvidenceKind.MISSING_INPUT,
                    "WTPART_NUMBER_MISSING",
                    "A technical WTPart has no usable Number for exact BOM component matching.",
                    (source,),
                )
            )
    enriched_matches: dict[
        tuple[str, OccurrenceKey, MatchPath], OccurrenceMatch
    ] = {}
    mbom_matches: dict[tuple[str, OccurrenceKey, MatchPath], OccurrenceMatch] = {}
    for seed in seeds:
        path = (
            MatchPath.RAW_ECN_COMPONENT
            if seed.origin is StructuralSeedOrigin.RAW_MATERIAL_LIKE_OBSERVED
            else MatchPath.JSON_WTPART
        )
        for occurrence in indexes.enriched_by_component.get(seed.number, ()):
            strength = (
                EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
                if ecn_key is not None and _key(occurrence.change_number) == ecn_key
                else EnrichedMatchStrength.COMPONENT_ONLY_CONTEXT
            )
            match = OccurrenceMatch(seed, occurrence, path, strength)
            enriched_matches[(seed.seed_id, occurrence_key(occurrence), path)] = match
        for occurrence in indexes.mbom_by_component.get(seed.number, ()):
            match = OccurrenceMatch(seed, occurrence, path, None)
            mbom_matches[(seed.seed_id, occurrence_key(occurrence), path)] = match
    bridges: list[DocumentOwnerBridge] = []
    for document in raw_direct:
        if document.row_classification is not BomRowClassification.CONFIRMED_DOCUMENT:
            continue
        owner = _nearest_material_owner(document, indexes)
        if owner is None:
            diagnostics.append(
                DeterministicDiagnostic(
                    EvidenceKind.WARNING,
                    "DOCUMENT_OWNER_NOT_FOUND",
                    "A documentation row has no material-like ancestor.",
                    (_source(document),),
                )
            )
            continue
        bridges.append(
            DocumentOwnerBridge(
                document.bom_row_id, owner.bom_row_id, owner.component_number
            )
        )
        if _key(owner.component_number) is None:
            diagnostics.append(
                DeterministicDiagnostic(
                    EvidenceKind.WARNING,
                    "DOCUMENT_OWNER_IDENTIFIER_MISSING",
                    "The nearest material-like owner has no component identifier.",
                    (_source(document), _source(owner)),
                )
            )
    return MatchResult(
        raw_direct=tuple(raw_direct),
        eligible_structural_seeds=tuple(sorted(seeds, key=lambda seed: seed.seed_id)),
        enriched_matches=tuple(
            sorted(enriched_matches.values(), key=_match_sort_key)
        ),
        mbom_presence=tuple(sorted(mbom_matches.values(), key=_match_sort_key)),
        document_owner_context=tuple(
            sorted(bridges, key=lambda item: item.document_row_id)
        ),
        context_rows_by_id={
            row_id: indexes.by_row_id[row_id]
            for bridge in bridges
            for row_id in (bridge.document_row_id, bridge.owner_row_id)
        },
        diagnostics=tuple(diagnostics),
    )
