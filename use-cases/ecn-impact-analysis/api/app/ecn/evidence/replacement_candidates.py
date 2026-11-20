"""Explainable, local replacement candidates derived without authoritative pairing."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from app.ecn.evidence.identifiers import candidate_id


_STOP_TOKENS = frozenset(
    {"a", "an", "and", "assembly", "for", "of", "part", "the", "to"}
)
_REPLACEMENT_TOKENS = frozenset(
    {"replace", "replaced", "replaces", "replacement", "supersede", "superseded"}
)


@dataclass(frozen=True)
class ReplacementSourceObject:
    """Hold the limited source and link fields needed for candidate scoring.

    Args:
        subject_ref: Selected-ECN RDF node reference for the source object.
        role: Source collection role, never a derived add/remove interpretation.
        number: Exact source part number when present.
        name: Source name when present.
        description: Source description when present.
        object_type: Source object type when present.
        source_path: RFC 6901 pointer of the source object.
        link_path: RFC 6901 pointer of its immediate source link.
        direct_relation: Non-null direct source relationship on a resulting link.
    """

    subject_ref: str
    role: Literal["affected", "resulting"]
    number: str | None
    name: str | None
    description: str | None
    object_type: str | None
    source_path: str
    link_path: str
    direct_relation: str | int | float | bool | None


@dataclass(frozen=True)
class ReplacementCandidate:
    """Describe one ranked, non-authoritative local replacement hypothesis."""

    candidate_id: str
    affected_ref: str
    resulting_ref: str
    affected_number: str | None
    resulting_number: str | None
    rank: int
    supporting_signals: tuple[str, ...]
    conflicting_signals: tuple[str, ...]
    source_paths: tuple[str, ...]
    bom_coordinates: tuple[str, ...]
    relation_kind: Literal["replacementCandidate"] = "replacementCandidate"


@dataclass(frozen=True)
class AuthoritativeSourceRelation:
    """Expose a direct non-null source relation outside candidate ranking."""

    source_path: str
    value: str | int | float | bool
    relation_kind: Literal["authoritative_source_relation"] = "authoritative_source_relation"


def _tokens(*values: str | None) -> frozenset[str]:
    """Normalize descriptions with Unicode casefolding and a small stop-token set."""
    rendered = " ".join(value for value in values if value)
    normalized = unicodedata.normalize("NFKD", rendered).casefold()
    terms = re.sub(r"[^\w]+", " ", normalized).split()
    return frozenset(term for term in terms if term not in _STOP_TOKENS)


def _locations(
    snapshot_objects: Iterable[Mapping[str, object]],
) -> dict[str, tuple[tuple[str, str | None, str], ...]]:
    """Index only observed Q900V parent/item coordinates by exact part number."""
    indexed: dict[str, list[tuple[str, str | None, str]]] = defaultdict(list)
    for item in snapshot_objects:
        number = item.get("number")
        if not isinstance(number, str) or not number:
            continue
        locations = item.get("locations", ())
        if not isinstance(locations, list):
            continue
        for location in locations:
            if not isinstance(location, Mapping):
                continue
            if location.get("bom_scope") != "Q900V only":
                continue
            parent = location.get("parent")
            if not isinstance(parent, str) or not parent:
                continue
            item_or_position = location.get("item_or_position")
            position = item_or_position if isinstance(item_or_position, str) else None
            path = location.get("path")
            coordinate = ""
            if isinstance(path, list):
                coordinate = "/".join(
                    str(segment.get("number") or "")
                    if isinstance(segment, Mapping)
                    else str(segment)
                    for segment in path
                )
            indexed[number].append((parent, position, coordinate))
    return {
        number: tuple(
            sorted(
                set(values), key=lambda value: (value[0], value[1] or "", value[2])
            )
        )
        for number, values in indexed.items()
    }


def _shared_coordinates(
    affected: ReplacementSourceObject,
    resulting: ReplacementSourceObject,
    locations: Mapping[str, tuple[tuple[str, str | None, str], ...]],
) -> tuple[tuple[str, str | None, str], ...]:
    """Return only observed Q900V coordinates that share an immediate parent."""
    if affected.number is None or resulting.number is None:
        return ()
    candidates = []
    for left in locations.get(affected.number, ()):
        for right in locations.get(resulting.number, ()):
            if left[0] == right[0]:
                candidates.append(
                    (
                        left[0],
                        left[1] if left[1] == right[1] else None,
                        left[2] or right[2],
                    )
                )
    return tuple(
        sorted(set(candidates), key=lambda value: (value[0], value[1] or "", value[2]))
    )


def _candidate(
    affected: ReplacementSourceObject,
    resulting: ReplacementSourceObject,
    coordinates: tuple[tuple[str, str | None, str], ...],
) -> ReplacementCandidate:
    """Build one source-backed candidate with explainable boolean ranking signals."""
    supporting = {"shared_immediate_q900v_parent"}
    conflicts: set[str] = set()
    if any(position is not None for _, position, _ in coordinates):
        supporting.add("compatible_bom_item_or_position")
    affected_type = _tokens(affected.object_type)
    resulting_type = _tokens(resulting.object_type)
    if affected_type and resulting_type:
        if affected_type == resulting_type:
            supporting.add("compatible_object_type")
        else:
            conflicts.add("incompatible_object_type")
    affected_words = _tokens(affected.name, affected.description)
    resulting_words = _tokens(resulting.name, resulting.description)
    if affected_words & resulting_words:
        supporting.add("description_token_overlap")
    if _REPLACEMENT_TOKENS & (affected_words | resulting_words):
        supporting.add("replacement_wording")
    source_paths = tuple(sorted({affected.source_path, resulting.source_path}))
    coordinates_text = tuple(
        f"parent={parent};item_or_position={position or 'unknown'};path={path or 'unknown'}"
        for parent, position, path in coordinates
    )
    return ReplacementCandidate(
        candidate_id=candidate_id(
            "replacementCandidate",
            (affected.subject_ref,),
            (resulting.subject_ref,),
            source_paths,
        ),
        affected_ref=affected.subject_ref,
        resulting_ref=resulting.subject_ref,
        affected_number=affected.number,
        resulting_number=resulting.number,
        rank=0,
        supporting_signals=tuple(sorted(supporting)),
        conflicting_signals=tuple(sorted(conflicts)),
        source_paths=source_paths,
        bom_coordinates=coordinates_text,
    )


def _has_conflicting_parents(
    affected: ReplacementSourceObject,
    resulting: ReplacementSourceObject,
    locations: Mapping[str, tuple[tuple[str, str | None, str], ...]],
) -> bool:
    """Return whether both source objects have incompatible extra parent context."""
    if affected.number is None or resulting.number is None:
        return False
    affected_parents = {parent for parent, _, _ in locations.get(affected.number, ())}
    resulting_parents = {parent for parent, _, _ in locations.get(resulting.number, ())}
    return bool(affected_parents - resulting_parents) and bool(
        resulting_parents - affected_parents
    )


def _sort_key(candidate: ReplacementCandidate) -> tuple[object, ...]:
    """Sort candidates by explainable local signals before stable source references."""
    signals = set(candidate.supporting_signals)
    return (
        "compatible_bom_item_or_position" not in signals,
        "compatible_object_type" not in signals,
        "replacement_wording" not in signals,
        "description_token_overlap" not in signals,
        candidate.affected_ref,
        candidate.resulting_ref,
    )


def derive_replacement_candidates(
    source_objects: Iterable[ReplacementSourceObject],
    snapshot_objects: Iterable[Mapping[str, object]],
) -> tuple[tuple[ReplacementCandidate, ...], tuple[AuthoritativeSourceRelation, ...]]:
    """Return local candidates plus direct source relations without persisting either.

    Candidate ranking only considers an affected/resulting pair after its exact
    part numbers share an observed Q900V immediate parent. This deliberately
    avoids global Cartesian pairing and does not create an authoritative
    replacement relation.
    """
    objects = tuple(source_objects)
    locations = _locations(snapshot_objects)
    affected = tuple(item for item in objects if item.role == "affected")
    resulting = tuple(item for item in objects if item.role == "resulting")
    candidates = []
    for left in affected:
        for right in resulting:
            coordinates = _shared_coordinates(left, right, locations)
            if not coordinates:
                continue
            candidate = _candidate(left, right, coordinates)
            if _has_conflicting_parents(left, right, locations):
                candidate = replace(
                    candidate,
                    conflicting_signals=("contradictory_bom_parents",),
                )
            candidates.append(candidate)
    candidates.sort(key=_sort_key)
    signal_groups: dict[
        tuple[str, tuple[str, ...], tuple[str, ...]], list[int]
    ] = defaultdict(list)
    for index, item in enumerate(candidates):
        signal_groups[
            (item.affected_ref, item.supporting_signals, item.conflicting_signals)
        ].append(index)
    for indexes in signal_groups.values():
        if len(indexes) > 1:
            for index in indexes:
                item = candidates[index]
                candidates[index] = replace(
                    item,
                    conflicting_signals=tuple(
                        sorted(
                            (*item.conflicting_signals, "multiple_equally_plausible_results")
                        )
                    ),
                )
    ranked = tuple(replace(item, rank=index) for index, item in enumerate(candidates, 1))
    authoritative = tuple(
        sorted(
            {
                AuthoritativeSourceRelation(item.link_path, item.direct_relation)
                for item in resulting
                if item.direct_relation is not None
            },
            key=lambda item: (item.source_path, str(item.value)),
        )
    )
    return ranked, authoritative
