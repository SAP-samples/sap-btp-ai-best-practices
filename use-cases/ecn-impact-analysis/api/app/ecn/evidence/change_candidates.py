"""Conservative ECN change candidates without endpoint or action inference."""

from __future__ import annotations

from collections.abc import Sequence

from app.ecn.evidence.identifiers import candidate_id
from app.ecn.models import (
    ChangeCandidate,
    ChangeCandidateKind,
    NormalizedEcn,
    NormalizedSourceObject,
)


def _object_key(item: NormalizedSourceObject) -> tuple[str, str]:
    """Prefer exact technical identity and fall back to Number or source ID."""
    technical = item.technical_id.strip() if item.technical_id else ""
    number = item.number.strip() if item.number else ""
    if item.is_technical_wtpart and technical.startswith("OR:wt.part.WTPart:"):
        return ("TECHNICAL", technical)
    if number:
        return ("NUMBER", number)
    return ("SOURCE_OBJECT", item.source_object_id)


def _explicit_revision_difference(
    left: NormalizedSourceObject, right: NormalizedSourceObject
) -> bool:
    """Return true only for comparable explicit revision/version differences."""
    pairs = ((left.revision, right.revision), (left.version, right.version))
    return any(
        first is not None
        and second is not None
        and first.strip() != second.strip()
        for first, second in pairs
    )


def _group_by_key(
    items: Sequence[NormalizedSourceObject],
) -> dict[tuple[str, str], tuple[NormalizedSourceObject, ...]]:
    """Group duplicate source occurrences by conservative comparison identity."""
    grouped: dict[tuple[str, str], list[NormalizedSourceObject]] = {}
    for item in items:
        grouped.setdefault(_object_key(item), []).append(item)
    return {
        key: tuple(
            sorted(values, key=lambda item: (item.source_path, item.source_object_id))
        )
        for key, values in grouped.items()
    }


def _candidate(
    kind: ChangeCandidateKind,
    affected: Sequence[NormalizedSourceObject],
    resulting: Sequence[NormalizedSourceObject],
    qualifiers: tuple[str, ...],
) -> ChangeCandidate:
    """Build one stable candidate from coordinates without endpoint pairing."""
    affected_ids = tuple(sorted({item.source_object_id for item in affected}))
    resulting_ids = tuple(sorted({item.source_object_id for item in resulting}))
    paths = tuple(sorted({item.source_path for item in (*affected, *resulting)}))
    return ChangeCandidate(
        candidate_id=candidate_id(kind.value, affected_ids, resulting_ids, paths),
        kind=kind,
        affected_object_ids=affected_ids,
        resulting_object_ids=resulting_ids,
        source_paths=paths,
        explicit_relation_text=None,
        qualifiers=qualifiers,
    )


def derive_change_candidates(ecn: NormalizedEcn) -> tuple[ChangeCandidate, ...]:
    """Describe membership/continuity without add, remove, replace, or pairing.

    Args:
        ecn: One normalized source occurrence containing affected/resulting sets.

    Returns:
        Deterministically ordered set-level candidate records.
    """
    affected = _group_by_key(ecn.affected_objects)
    resulting = _group_by_key(ecn.resulting_objects)
    candidates: list[ChangeCandidate] = []
    for key in sorted(affected.keys() - resulting.keys()):
        candidates.append(
            _candidate(
                ChangeCandidateKind.AFFECTED_ONLY,
                affected[key],
                (),
                ("CANDIDATE_NOT_REMOVAL",),
            )
        )
    for key in sorted(resulting.keys() - affected.keys()):
        candidates.append(
            _candidate(
                ChangeCandidateKind.RESULTING_ONLY,
                (),
                resulting[key],
                ("CANDIDATE_NOT_ADDITION",),
            )
        )
    if ecn.affected_objects and ecn.resulting_objects:
        candidates.append(
            _candidate(
                ChangeCandidateKind.CO_OCCURRENCE,
                ecn.affected_objects,
                ecn.resulting_objects,
                ("NO_ENDPOINT_PAIRING",),
            )
        )
    for key in sorted(affected.keys() & resulting.keys()):
        left = affected[key]
        right = resulting[key]
        if any(
            _explicit_revision_difference(first, second)
            for first in left
            for second in right
        ):
            candidates.append(
                _candidate(
                    ChangeCandidateKind.REVISION_ONLY,
                    left,
                    right,
                    ("REVISION_CONTINUITY_NOT_PHYSICAL_CHANGE",),
                )
            )
    return tuple(
        sorted(candidates, key=lambda item: (item.kind.value, item.candidate_id))
    )
