"""Tests for conservative non-Cartesian ECN change candidates."""

from app.ecn.evidence.change_candidates import derive_change_candidates
from app.ecn.models import ChangeCandidateKind, SourceObjectRole

from tests.fixtures.builders import make_ecn, make_source_object, synthetic_uuid


def test_candidates_never_claim_add_remove_or_pair_every_endpoint() -> None:
    """Represent source collection membership without physical-change semantics."""
    affected = tuple(
        make_source_object(
            source_object_id=synthetic_uuid(f"a-{index}"),
            role=SourceObjectRole.AFFECTED,
            technical_id=f"OR:wt.part.WTPart:{index}",
            number=f"A-{index}",
            source_path=f"$.CNAffectLinks[0].AffectedObjects[{index}]",
        )
        for index in range(200)
    )
    resulting = tuple(
        make_source_object(
            source_object_id=synthetic_uuid(f"r-{index}"),
            role=SourceObjectRole.RESULTING,
            technical_id=f"OR:wt.part.WTPart:{1000 + index}",
            number=f"R-{index}",
            source_path=(
                f"$.ImplementationPlan[0].ResultingLinks[0].ResultingObjects[{index}]"
            ),
        )
        for index in range(300)
    )
    candidates = derive_change_candidates(
        make_ecn(affected_objects=affected, resulting_objects=resulting)
    )
    co_occurrence = [
        item for item in candidates if item.kind is ChangeCandidateKind.CO_OCCURRENCE
    ]
    assert len(co_occurrence) == 1
    assert len(co_occurrence[0].affected_object_ids) == 200
    assert len(co_occurrence[0].resulting_object_ids) == 300
    assert all(item.explicit_relation_text is None for item in candidates)
    assert {item.kind.value for item in candidates}.isdisjoint(
        {"ADD", "REMOVE", "REPLACE"}
    )


def test_same_stable_identity_with_different_revision_is_continuity_only() -> None:
    """Record explicit revision movement without calling it a physical redesign."""
    affected = make_source_object(
        source_object_id=synthetic_uuid("a"),
        role=SourceObjectRole.AFFECTED,
        technical_id="OR:wt.part.WTPart:42",
        revision="5",
        version="5.2",
    )
    resulting = make_source_object(
        source_object_id=synthetic_uuid("r"),
        role=SourceObjectRole.RESULTING,
        technical_id="OR:wt.part.WTPart:42",
        revision="6",
        version="6.1",
    )
    candidates = derive_change_candidates(
        make_ecn(affected_objects=(affected,), resulting_objects=(resulting,))
    )
    revision = [
        item for item in candidates if item.kind is ChangeCandidateKind.REVISION_ONLY
    ]
    assert len(revision) == 1
    assert "REVISION_CONTINUITY_NOT_PHYSICAL_CHANGE" in revision[0].qualifiers
    assert not any(
        item.kind
        in {
            ChangeCandidateKind.AFFECTED_ONLY,
            ChangeCandidateKind.RESULTING_ONLY,
        }
        for item in candidates
    )


def test_free_text_and_boolean_replacement_fields_never_create_explicit_relation() -> None:
    """Reject free text and flags as typed endpoint relations."""
    resulting = make_source_object(
        relationship_values={
            "ReplacesAffected": True,
            "ImpactIntent": "replaces object",
        }
    )
    candidates = derive_change_candidates(make_ecn(resulting_objects=(resulting,)))
    assert ChangeCandidateKind.EXPLICIT_SOURCE_RELATION not in {
        item.kind for item in candidates
    }
