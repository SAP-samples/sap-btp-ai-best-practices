"""Focused domain tests for the immutable ECN source contracts.

These tests verify the exact structural contract of the frozen records and
enumerations defined in ``app.ecn.models`` without relying on any database,
credentials, or external files.

Run with:
    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_domain_models.py
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from hashlib import sha256
from uuid import UUID

import pytest
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    ChangeCandidate,
    ChangeCandidateKind,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    ParseState,
    SourceArtifact,
    canonical_uuid,
)


def test_canonical_uuid_accepts_uuid_and_rejects_noncanonical_text() -> None:
    """canonical_uuid returns the canonical string for a UUID object and raises
    ValueError for a braced string that differs from the canonical form."""
    value = UUID("00000000-0000-4000-8000-000000000001")
    assert canonical_uuid(value) == str(value)
    with pytest.raises(ValueError, match="canonical UUID"):
        canonical_uuid("{00000000-0000-4000-8000-000000000001}")


def test_source_artifact_is_immutable_and_keeps_original_bytes() -> None:
    """SourceArtifact stores the original bytes unchanged and refuses mutation."""
    artifact = SourceArtifact(
        artifact_id="00000000-0000-4000-8000-000000000001",
        artifact_kind=ArtifactKind.ECN_JSON,
        file_name="ecn.json",
        media_type="application/json",
        byte_size=2,
        sha256=sha256(b"[]").hexdigest(),
        content_bytes=b"[]",
        normalized_content="[]",
        parse_state=ParseState.VALID,
        parse_error_code=None,
        created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
    )
    assert artifact.content_bytes == b"[]"
    with pytest.raises(FrozenInstanceError):
        artifact.file_name = "changed.json"  # type: ignore[misc]


@pytest.mark.parametrize("invalid_digest", ["G" * 64, "A" * 64, "0" * 63])
def test_source_artifact_rejects_non_lowercase_sha256(invalid_digest: str) -> None:
    """Reject digests that are not exactly 64 lowercase hexadecimal characters."""
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        SourceArtifact(
            artifact_id="00000000-0000-4000-8000-000000000001",
            artifact_kind=ArtifactKind.ECN_JSON,
            file_name="ecn.json",
            media_type="application/json",
            byte_size=2,
            sha256=invalid_digest,
            content_bytes=b"[]",
            normalized_content="[]",
            parse_state=ParseState.VALID,
            parse_error_code=None,
            created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
        )


def test_source_artifact_rejects_digest_that_does_not_match_original_bytes() -> None:
    """Reject a well-formed digest when it was not computed from content_bytes."""
    with pytest.raises(ValueError, match="content_bytes"):
        SourceArtifact(
            artifact_id="00000000-0000-4000-8000-000000000001",
            artifact_kind=ArtifactKind.ECN_JSON,
            file_name="ecn.json",
            media_type="application/json",
            byte_size=2,
            sha256="0" * 64,
            content_bytes=b"[]",
            normalized_content="[]",
            parse_state=ParseState.VALID,
            parse_error_code=None,
            created_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
        )


def test_evidence_sections_are_explicit_not_a_dynamic_dictionary() -> None:
    """EvidenceBundleSections declares exactly the six section fields in the
    correct order matching AnalysisSectionKey definition order."""
    assert tuple(EvidenceBundleSections.__dataclass_fields__) == (
        "executive_summary",
        "change_intent",
        "affected_materials",
        "ebom_mbom_impact",
        "configuration_scope",
        "uncertainties_and_missing_evidence",
    )


def test_evidence_bundle_rejects_non_lowercase_sha256() -> None:
    """Reject an evidence digest that is not lowercase SHA-256 hexadecimal."""
    sections = EvidenceBundleSections((), (), (), (), (), ())
    context = EvidenceContext(
        upload_id="00000000-0000-4000-8000-000000000001",
        ecn_artifact_id="00000000-0000-4000-8000-000000000002",
        bom_artifact_id="00000000-0000-4000-8000-000000000003",
        rule_artifact_id=None,
        parser_version="ecn-source-v1",
        deterministic_logic_version="ecn-evidence-v1",
        rule_evaluator_version="ecn-rules-v1",
        analysis_as_of_date=None,
    )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        EvidenceBundle(
            ecn_row_id="00000000-0000-4000-8000-000000000004",
            context=context,
            sections=sections,
            evidence_hash="G" * 64,
        )


def test_evidence_context_defaults_plant_assignment_to_empty() -> None:
    """Without frozen plant settings every plant stays undetermined."""
    context = EvidenceContext(
        upload_id="00000000-0000-4000-8000-000000000001",
        ecn_artifact_id="00000000-0000-4000-8000-000000000002",
        bom_artifact_id="00000000-0000-4000-8000-000000000003",
        rule_artifact_id=None,
        parser_version="ecn-source-v1",
        deterministic_logic_version="ecn-evidence-v1",
        rule_evaluator_version="ecn-rules-v1",
        analysis_as_of_date=None,
    )

    assert context.plant_assignment == {}


def test_change_candidate_accepts_only_stable_prefixed_hash_identity() -> None:
    """Accept the Phase 2 candidate ID format without weakening its shape."""
    candidate = ChangeCandidate(
        candidate_id="C-0123456789abcdef",
        kind=ChangeCandidateKind.AFFECTED_ONLY,
        affected_object_ids=(),
        resulting_object_ids=(),
        source_paths=(),
        qualifiers=(),
        explicit_relation_text=None,
    )
    assert candidate.candidate_id == "C-0123456789abcdef"
    with pytest.raises(ValueError, match="candidate_id"):
        ChangeCandidate(
            candidate_id="C-0123456789ABCDEG",
            kind=ChangeCandidateKind.AFFECTED_ONLY,
            affected_object_ids=(),
            resulting_object_ids=(),
            source_paths=(),
            qualifiers=(),
            explicit_relation_text=None,
        )


def test_evidence_item_accepts_only_stable_prefixed_hash_identity() -> None:
    """Accept the Phase 2 evidence ID format without weakening its shape."""
    item = EvidenceItem(
        evidence_id="E-0123456789abcdef",
        section_key=AnalysisSectionKey.CHANGE_INTENT,
        kind=EvidenceKind.MISSING_INPUT,
        tier=None,
        summary="Synthetic gap",
        sources=(),
        values={},
        related_ids=(),
        quality_qualifiers=(),
    )
    assert item.evidence_id == "E-0123456789abcdef"
    with pytest.raises(ValueError, match="evidence_id"):
        EvidenceItem(
            evidence_id="E-0123456789ABCDEG",
            section_key=AnalysisSectionKey.CHANGE_INTENT,
            kind=EvidenceKind.MISSING_INPUT,
            tier=None,
            summary="Synthetic gap",
            sources=(),
            values={},
            related_ids=(),
            quality_qualifiers=(),
        )
