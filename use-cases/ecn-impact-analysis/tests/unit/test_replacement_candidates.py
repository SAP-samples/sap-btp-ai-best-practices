"""Tests for local, explainable replacement candidates without authoritative pairing."""

from __future__ import annotations

from app.ecn.evidence.replacement_candidates import (
    ReplacementSourceObject,
    derive_replacement_candidates,
)


def _source(
    ref: str,
    role: str,
    number: str,
    description: str,
    object_type: str = "WTPart",
    direct_relation: str | None = None,
) -> ReplacementSourceObject:
    """Build one source object with deterministic local JSON coordinates."""
    path = (
        f"/CNAffectLinks/0/AffectedObjects/{ref[-1]}"
        if role == "affected"
        else f"/ImplementationPlan/0/ResultingLinks/0/ResultingObjects/{ref[-1]}"
    )
    return ReplacementSourceObject(
        subject_ref=ref,
        role=role,
        number=number,
        name=number,
        description=description,
        object_type=object_type,
        source_path=path,
        link_path=path.rsplit("/", 2)[0],
        direct_relation=direct_relation,
    )


def _snapshot(*objects: tuple[str, str, str]) -> tuple[dict[str, object], ...]:
    """Build Q900V-only snapshot locations for number, parent, and item position."""
    return tuple(
        {
            "number": number,
            "locations": [
                {
                    "parent": parent,
                    "item_or_position": position,
                    "path": [parent, number],
                    "bom_scope": "Q900V only",
                }
            ],
        }
        for number, parent, position in objects
    )


def test_candidates_require_shared_observed_q900v_parent() -> None:
    """Never create a Cartesian affected/resulting pairing across unrelated BOM parents."""
    candidates, authoritative = derive_replacement_candidates(
        (
            _source("urn:a1", "affected", "A-1", "Hydraulic pump assembly"),
            _source("urn:r1", "resulting", "R-1", "Hydraulic pump service assembly"),
            _source("urn:r2", "resulting", "R-2", "Hydraulic pump unrelated"),
        ),
        _snapshot(("A-1", "PARENT-1", "0010"), ("R-1", "PARENT-1", "0010"), ("R-2", "PARENT-2", "0020")),
    )

    assert len(candidates) == 1
    assert authoritative == ()
    candidate = candidates[0]
    assert candidate.relation_kind == "replacementCandidate"
    assert candidate.affected_number == "A-1"
    assert candidate.resulting_number == "R-1"
    assert "shared_immediate_q900v_parent" in candidate.supporting_signals
    assert "compatible_bom_item_or_position" in candidate.supporting_signals
    assert "description_token_overlap" in candidate.supporting_signals
    assert candidate.bom_coordinates


def test_candidates_explain_ambiguity_and_type_conflicts_without_confidence_scores() -> None:
    """Retain local alternatives as conflicts rather than selecting an authoritative pair."""
    candidates, _ = derive_replacement_candidates(
        (
            _source("urn:a1", "affected", "A-1", "Pump cover", "WTPart"),
            _source("urn:r1", "resulting", "R-1", "Pump cover", "WTDocument"),
            _source("urn:r2", "resulting", "R-2", "Pump cover", "WTDocument"),
        ),
        _snapshot(("A-1", "PARENT-1", "0010"), ("R-1", "PARENT-1", "0010"), ("R-2", "PARENT-1", "0010")),
    )

    assert [item.rank for item in candidates] == [1, 2]
    assert all("incompatible_object_type" in item.conflicting_signals for item in candidates)
    assert all("multiple_equally_plausible_results" in item.conflicting_signals for item in candidates)
    assert all("score" not in item.__dict__ for item in candidates)


def test_candidates_flag_divergent_extra_parent_context() -> None:
    """Retain a shared parent while disclosing incompatible additional parents."""
    candidates, _ = derive_replacement_candidates(
        (
            _source("urn:a1", "affected", "A-1", "Pump cover"),
            _source("urn:r1", "resulting", "R-1", "Pump cover"),
        ),
        _snapshot(
            ("A-1", "PARENT-1", "0010"),
            ("A-1", "PARENT-2", "0010"),
            ("R-1", "PARENT-1", "0010"),
            ("R-1", "PARENT-3", "0010"),
        ),
    )

    assert len(candidates) == 1
    assert "contradictory_bom_parents" in candidates[0].conflicting_signals


def test_direct_source_relations_are_returned_separately_from_candidates() -> None:
    """Expose a non-null source relation without promoting it into ranked output."""
    candidates, authoritative = derive_replacement_candidates(
        (
            _source("urn:a1", "affected", "A-1", "Pump cover"),
            _source(
                "urn:r1",
                "resulting",
                "R-1",
                "Pump cover replacement",
                direct_relation="OR:wt.part.WTPart:123",
            ),
        ),
        _snapshot(("A-1", "PARENT-1", "0010"), ("R-1", "PARENT-1", "0010")),
    )

    assert len(candidates) == 1
    assert len(authoritative) == 1
    assert authoritative[0].source_path.endswith("/ResultingLinks/0")
    assert authoritative[0].value == "OR:wt.part.WTPart:123"
