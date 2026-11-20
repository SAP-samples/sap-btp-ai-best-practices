"""Tests for grouped, business-readable deterministic evidence projections."""

from app.ecn.evidence.business import (
    BusinessEvidenceSection,
    business_evidence_records,
    paginate_business_evidence,
)
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
)

from tests.fixtures.builders import synthetic_uuid


def _source(path: str) -> EvidenceSource:
    """Return one stable ECN JSON source coordinate for projection tests."""
    return EvidenceSource(
        artifact_id=synthetic_uuid("ecn-artifact"),
        artifact_kind=ArtifactKind.ECN_JSON,
        json_path=path,
        sheet_name=None,
        bom_row_id=None,
        source_row_number=None,
    )


def _item(
    evidence_id: str,
    *,
    section: AnalysisSectionKey,
    kind: EvidenceKind,
    summary: str,
    values: dict[str, object],
    path: str,
    tier: EvidenceTier | None = None,
    qualifiers: tuple[str, ...] = (),
) -> EvidenceItem:
    """Build one compact evidence item with exact source provenance."""
    return EvidenceItem(
        evidence_id=evidence_id,
        section_key=section,
        kind=kind,
        tier=tier,
        summary=summary,
        sources=(_source(path),),
        values=values,  # type: ignore[arg-type] - test values are JSON-compatible.
        related_ids=(),
        quality_qualifiers=qualifiers,
    )


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    """Place supplied items into their canonical stored evidence sections."""
    grouped = {
        key: tuple(item for item in items if item.section_key is key)
        for key in AnalysisSectionKey
    }
    return EvidenceBundle(
        ecn_row_id=synthetic_uuid("ecn-row"),
        context=EvidenceContext(
            upload_id=synthetic_uuid("upload"),
            ecn_artifact_id=synthetic_uuid("ecn-artifact"),
            bom_artifact_id=synthetic_uuid("bom-artifact"),
            rule_artifact_id=None,
            parser_version="ecn-source-v1",
            deterministic_logic_version="ecn-evidence-v2",
            rule_evaluator_version="ecn-rules-v1",
            analysis_as_of_date=None,
        ),
        sections=EvidenceBundleSections(
            executive_summary=grouped[AnalysisSectionKey.EXECUTIVE_SUMMARY],
            change_intent=grouped[AnalysisSectionKey.CHANGE_INTENT],
            affected_materials=grouped[AnalysisSectionKey.AFFECTED_MATERIALS],
            ebom_mbom_impact=grouped[AnalysisSectionKey.EBOM_MBOM_IMPACT],
            configuration_scope=grouped[AnalysisSectionKey.CONFIGURATION_SCOPE],
            uncertainties_and_missing_evidence=grouped[
                AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE
            ],
        ),
        evidence_hash="0" * 64,
    )


def test_materials_and_bom_occurrences_use_business_labels_and_fields() -> None:
    """Expose material numbers and BOM fields without technical enum labels."""
    source_item = _item(
        "E-0000000000000001",
        section=AnalysisSectionKey.AFFECTED_MATERIALS,
        kind=EvidenceKind.SOURCE_FACT,
        summary="ECN source object retained with its source role and identity type.",
        values={
            "role": "RESULTING",
            "number": "9465649T1",
            "revision": "B",
        },
        path="$.resulting[0]",
    )
    bom_item = _item(
        "E-0000000000000002",
        section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
        kind=EvidenceKind.DETERMINISTIC_FINDING,
        tier=EvidenceTier.EXACT_WTPART_WHERE_USED,
        summary="Enriched EBOM occurrence confirms exact component and ECN.",
        values={
            "bom_kind": "ENRICHED_EBOM",
            "component_number": "9465649T1",
            "description_text": "Safety decal",
            "parent_material_number": "PARENT-1",
            "parent_path": ["ROOT", "PARENT-1"],
            "quantity": "1",
            "plant": "PL10",
        },
        path="$.bom[0]",
    )

    records = business_evidence_records(_bundle(source_item, bom_item))

    added = next(item for item in records if item.section is BusinessEvidenceSection.ADDED_MATERIALS)
    assert added.label == "Resulting object reference"
    assert added.material_number == "9465649T1"
    assert added.revision == "B"
    assert "RESULTING" not in added.label
    ebom = next(item for item in records if item.section is BusinessEvidenceSection.EBOM_MATCHES)
    assert ebom.description == "Safety decal"
    assert ebom.parent == "PARENT-1"
    assert ebom.bom_path == ("ROOT", "PARENT-1")
    assert ebom.plant == "PL10"


def test_repeated_source_quality_problems_are_grouped_outside_missing_evidence() -> None:
    """Collapse repeated qualifiers into one actionable data-quality record."""
    warnings = tuple(
        _item(
            f"E-{index:016x}",
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            kind=EvidenceKind.WARNING,
            summary="An ECN source object carries a source-quality qualifier.",
            values={"code": "SOURCE_RELATION_ERROR_VALUE"},
            path=f"$.relationships[{index}]",
        )
        for index in range(1, 5)
    )
    missing = _item(
        "E-0000000000000010",
        section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
        kind=EvidenceKind.MISSING_INPUT,
        summary="No enriched EBOM occurrence confirms both exact component and the same ECN.",
        values={"code": "EBOM_CONFIRMED_NO_MATCH"},
        path="$.affected[0]",
    )

    records = business_evidence_records(_bundle(*warnings, missing))

    quality = [item for item in records if item.section is BusinessEvidenceSection.DATA_QUALITY]
    gaps = [item for item in records if item.section is BusinessEvidenceSection.MISSING_EVIDENCE]
    assert len(quality) == 1
    assert quality[0].matched_row_count == 4
    assert quality[0].summary == (
        "4 source relationships contain an invalid value; exact matching may be incomplete."
    )
    assert len(quality[0].sources) == 3
    assert len(gaps) == 1
    assert gaps[0].summary.startswith("No enriched EBOM occurrence")


def test_business_evidence_pagination_is_stable_and_bounded() -> None:
    """Page one section deterministically and reject unbounded page sizes."""
    items = tuple(
        _item(
            f"E-{index:016x}",
            section=AnalysisSectionKey.AFFECTED_MATERIALS,
            kind=EvidenceKind.SOURCE_FACT,
            summary="Stored material",
            values={"role": "AFFECTED", "number": f"PART-{index:03d}"},
            path=f"$.affected[{index}]",
        )
        for index in range(1, 7)
    )

    page = paginate_business_evidence(
        _bundle(*items),
        BusinessEvidenceSection.MODIFIED_MATERIALS,
        page=2,
        page_size=2,
    )

    assert page.total_items == 6
    assert page.total_pages == 3
    assert [item.material_number for item in page.items] == ["PART-003", "PART-004"]
