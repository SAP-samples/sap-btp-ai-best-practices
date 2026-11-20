"""Tests for bounded, deterministic evidence projection into LLM prompts."""

from __future__ import annotations

import json

import pytest
from app.ecn.llm.prompts import (
    PromptProjectionError,
    build_impact_input,
    project_evidence_for_prompt,
)
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
)

from tests.fixtures.builders import (
    make_evidence_bundle,
    make_evidence_bundle_sections,
    synthetic_uuid,
)


def _item(
    label: str,
    *,
    section: AnalysisSectionKey,
    kind: EvidenceKind = EvidenceKind.DETERMINISTIC_FINDING,
    tier: EvidenceTier | None = None,
    summary: str | None = None,
    values: dict[str, object] | None = None,
) -> EvidenceItem:
    """Build one valid evidence item with stable identifiers and provenance."""
    return EvidenceItem(
        evidence_id=f"E-{label:0>16}"[-18:],
        section_key=section,
        kind=kind,
        tier=tier,
        summary=summary or f"Evidence {label}",
        sources=(
            EvidenceSource(
                artifact_id=synthetic_uuid("projection-artifact"),
                artifact_kind=ArtifactKind.ECN_JSON,
                json_path="$.safe.path",
                sheet_name=None,
                bom_row_id=None,
                source_row_number=None,
            ),
        ),
        values=values or {"safe_value": label},
        related_ids=(),
        quality_qualifiers=(),
    )


@pytest.fixture
def evidence_bundle():
    """Return a bundle whose source order differs from prompt priority order."""
    header = _item(
        "0000000000000001",
        section=AnalysisSectionKey.EXECUTIVE_SUMMARY,
        kind=EvidenceKind.SOURCE_FACT,
        values={"ecn_number": "ECN-1", "description_text": "Change one part"},
    )
    direct = _item(
        "0000000000000002",
        section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
        kind=EvidenceKind.SOURCE_FACT,
        tier=EvidenceTier.DIRECT_ECN_RAW,
    )
    contextual = _item(
        "0000000000000003",
        section=AnalysisSectionKey.CONFIGURATION_SCOPE,
        tier=EvidenceTier.CONTEXTUAL_BOM_OCCURRENCE,
    )
    return make_evidence_bundle(
        sections=make_evidence_bundle_sections(
            executive_summary=(header,),
            ebom_mbom_impact=(direct,),
            configuration_scope=(contextual,),
        )
    )


def test_projection_prioritizes_direct_evidence_and_records_omissions(
    evidence_bundle,
) -> None:
    """Retain high-confidence items first and turn truncation into an explicit gap."""
    projection = project_evidence_for_prompt(
        evidence_bundle,
        maximum_items=2,
        maximum_characters=20_000,
    )
    assert [item.evidence_id for item in projection.concise_evidence] == [
        "E-0000000000000001",
        "E-0000000000000002",
    ]
    assert projection.omitted_evidence_count == 1
    assert (
        projection.omitted_evidence_count_by_section[
            AnalysisSectionKey.CONFIGURATION_SCOPE
        ]
        == 1
    )
    assert "PROMPT_PROJECTION_TRUNCATED" in projection.projection_warnings
    assert AnalysisSectionKey.CONFIGURATION_SCOPE in projection.sections_requiring_gap


def test_projection_is_canonical_when_bundle_order_changes(evidence_bundle) -> None:
    """Sort evidence by fixed priority and stable evidence ID, not source tuple order."""
    sections = evidence_bundle.sections
    reordered = make_evidence_bundle(
        sections=make_evidence_bundle_sections(
            executive_summary=tuple(reversed(sections.executive_summary)),
            ebom_mbom_impact=tuple(reversed(sections.ebom_mbom_impact)),
            configuration_scope=tuple(reversed(sections.configuration_scope)),
        )
    )
    first = project_evidence_for_prompt(evidence_bundle, 20, 20_000)
    second = project_evidence_for_prompt(reordered, 20, 20_000)
    assert build_impact_input(first) == build_impact_input(second)


def test_projection_counts_unicode_codepoints_and_omits_whole_items(
    evidence_bundle,
) -> None:
    """Apply the character cap to exact Unicode JSON without slicing an item."""
    projection = project_evidence_for_prompt(evidence_bundle, 20, 5_000)
    payload_text = build_impact_input(projection)[0]["content"][0]["text"]
    assert isinstance(payload_text, str)
    assert len(payload_text) <= 5_000
    assert json.loads(payload_text)["ecn_row_id"] == evidence_bundle.ecn_row_id


def test_projection_truncates_free_text_with_an_explicit_qualifier() -> None:
    """Bound long evidence strings and tell the model that text was truncated."""
    item = _item(
        "0000000000000004",
        section=AnalysisSectionKey.CHANGE_INTENT,
        summary="é" * 5_000,
        values={"description": "ü" * 5_000},
    )
    bundle = make_evidence_bundle(
        sections=make_evidence_bundle_sections(change_intent=(item,))
    )
    projection = project_evidence_for_prompt(bundle, 10, 20_000)
    projected = projection.concise_evidence[0]
    assert "PROMPT_FREE_TEXT_TRUNCATED" in projected.quality_qualifiers
    assert len(projected.summary) < len(item.summary)


def test_projection_removes_raw_source_payload_keys() -> None:
    """Never expose raw rows, source objects, bytes, or full rule definitions."""
    item = _item(
        "0000000000000005",
        section=AnalysisSectionKey.CHANGE_INTENT,
        values={
            "safe": "retained",
            "SOURCE_JSON": {"secret": "not-for-prompt"},
            "ROW_JSON": "not-for-prompt",
            "raw_object": {"password": "not-for-prompt"},
            "content_bytes": "not-for-prompt",
            "rule_source_text": "not-for-prompt",
        },
    )
    projection = project_evidence_for_prompt(
        make_evidence_bundle(
            sections=make_evidence_bundle_sections(change_intent=(item,))
        ),
        10,
        20_000,
    )
    serialized = json.dumps(build_impact_input(projection), ensure_ascii=False)
    assert "not-for-prompt" not in serialized
    assert projection.concise_evidence[0].concise_values == {"safe": "retained"}


def test_projection_marks_full_bundle_gaps_even_without_truncation() -> None:
    """Force a gap for sections containing missing-input or contradiction evidence."""
    gap = _item(
        "0000000000000006",
        section=AnalysisSectionKey.CONFIGURATION_SCOPE,
        kind=EvidenceKind.MISSING_INPUT,
    )
    projection = project_evidence_for_prompt(
        make_evidence_bundle(
            sections=make_evidence_bundle_sections(configuration_scope=(gap,))
        ),
        10,
        20_000,
    )
    assert AnalysisSectionKey.CONFIGURATION_SCOPE in projection.sections_requiring_gap


def test_projection_rejects_a_cap_below_the_safe_skeleton(evidence_bundle) -> None:
    """Reject a cap too small for the versioned schema and warnings envelope."""
    with pytest.raises(PromptProjectionError, match="PROMPT_LIMIT_TOO_SMALL"):
        project_evidence_for_prompt(evidence_bundle, 10, 4_095)
