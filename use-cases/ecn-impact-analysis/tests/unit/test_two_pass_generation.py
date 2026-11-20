"""Tests for bounded planner projection and deterministic evidence resolution."""

import json

import pytest
from app.ecn.evidence.business import BusinessEvidenceSection
from app.ecn.llm.two_pass import (
    FINAL_MAX_CHARACTERS,
    FINAL_MAX_RECORDS,
    PLANNER_MAX_CHARACTERS,
    EvidencePlan,
    EvidenceSlice,
    build_final_input,
    build_planner_input,
    build_planner_projection,
    resolve_evidence_plan,
)
from app.ecn.models import (
    AnalysisSectionKey,
    EvidenceItem,
    EvidenceKind,
)

from tests.fixtures.builders import (
    make_evidence_bundle,
    make_evidence_bundle_sections,
)


def _large_bundle(count: int = 150):
    """Return a stored bundle with many material facts and long descriptions."""
    base = make_evidence_bundle()
    items = tuple(
        EvidenceItem(
            evidence_id=f"E-{index:016x}",
            section_key=AnalysisSectionKey.AFFECTED_MATERIALS,
            kind=EvidenceKind.SOURCE_FACT,
            tier=None,
            summary="Stored material source fact.",
            sources=(),
            values={
                "role": "RESULTING" if index % 2 else "AFFECTED",
                "number": f"PART-{index:04d}",
                "description_text": "Long material description " + ("x" * 2_000),
            },
            related_ids=(),
            quality_qualifiers=(),
        )
        for index in range(count)
    )
    return make_evidence_bundle(
        ecn_row_id=base.ecn_row_id,
        context=base.context,
        sections=make_evidence_bundle_sections(affected_materials=items),
        evidence_hash="b" * 64,
    )


def test_planner_projection_contains_counts_keys_and_no_raw_payload() -> None:
    """Bound planning context and avoid sending the complete evidence bundle."""
    projection = build_planner_projection(_large_bundle())
    payload = build_planner_input(projection)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert len(serialized) <= PLANNER_MAX_CHARACTERS
    assert projection.section_counts[BusinessEvidenceSection.ADDED_MATERIALS] == 75
    assert "PART-0001" in projection.material_keys
    assert "internal_evidence_ids" not in serialized
    assert "raw_values" not in serialized


def test_resolver_accepts_only_allowlisted_slices_and_bounds_final_evidence() -> None:
    """Resolve requested business slices without model-authored SQL or overrun."""
    bundle = _large_bundle()
    plan = EvidencePlan(
        requested_slices=[EvidenceSlice.MATERIAL_CHANGES],
        focus_material_numbers=[],
    )
    resolved = resolve_evidence_plan(bundle, plan)
    serialized = json.dumps(build_final_input(resolved), ensure_ascii=False)
    assert 0 < len(resolved.records) <= FINAL_MAX_RECORDS
    assert len(serialized) <= FINAL_MAX_CHARACTERS
    assert resolved.omitted_record_count == 150 - len(resolved.records)
    assert resolved.allowed_evidence_ids_by_section[
        AnalysisSectionKey.AFFECTED_MATERIALS
    ]


def test_every_final_section_may_cite_the_exact_bounded_evidence_view() -> None:
    """Allow business synthesis while rejecting facts outside pass two."""
    bundle = _large_bundle(count=4)
    resolved = resolve_evidence_plan(
        bundle,
        EvidencePlan(
            requested_slices=[EvidenceSlice.MATERIAL_CHANGES],
            focus_material_numbers=[],
        ),
    )
    resolved_ids = {
        citation_id
        for record in resolved.records
        for citation_id in record.citation_ids
    }

    for section in AnalysisSectionKey:
        assert resolved_ids <= resolved.allowed_evidence_ids_by_section[section]


def test_planner_schema_rejects_arbitrary_slice_or_sql() -> None:
    """Keep planner output within the closed deterministic resolver contract."""
    with pytest.raises(Exception):
        EvidencePlan.model_validate(
            {
                "requested_slices": ["SELECT * FROM ECN_IMPACT_ECN"],
                "focus_material_numbers": [],
            }
        )
