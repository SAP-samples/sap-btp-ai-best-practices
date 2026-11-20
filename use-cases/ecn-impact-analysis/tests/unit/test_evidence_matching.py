"""Synthetic exact-join tests for ECN-to-BOM evidence matching."""

from app.ecn.evidence.matching import (
    EnrichedMatchStrength,
    MatchPath,
    build_bom_indexes,
    match_ecn_to_bom,
)
from app.ecn.models import BomKind, BomRowClassification

from tests.fixtures.builders import (
    make_bom_row,
    make_ecn,
    make_source_object,
    synthetic_uuid,
)


def test_exact_raw_to_enriched_crosswalk_preserves_documents_and_occurrences() -> None:
    """Join exact ECN/change and component keys while retaining hierarchy context."""
    rows = (
        make_bom_row(
            bom_row_id=synthetic_uuid("raw-a"),
            bom_kind=BomKind.RAW_EBOM_130,
            sheet_name="Raw",
            source_row_number=10,
            item_value="150",
            position_value="150",
            component_number="PART-A",
            material_number="PART-A",
            change_number="ECN345274",
            row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("doc-a"),
            bom_kind=BomKind.RAW_EBOM_130,
            sheet_name="Raw",
            source_row_number=11,
            parent_bom_row_id=synthetic_uuid("raw-a"),
            dotted_level="..1",
            item_value="D01D001",
            position_value="D01D001",
            component_number="DOC ZVD 1 A",
            material_number="DOC ZVD 1 A",
            change_number="ECN345274",
            row_classification=BomRowClassification.CONFIRMED_DOCUMENT,
            is_documentation=True,
            quality_qualifiers=(),
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("raw-b"),
            bom_kind=BomKind.RAW_EBOM_130,
            sheet_name="Raw",
            source_row_number=12,
            item_value="160",
            position_value="160",
            component_number="PART-B",
            material_number="PART-B",
            change_number="ECN345274",
            row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("candidate"),
            bom_kind=BomKind.RAW_EBOM_130,
            sheet_name="Raw",
            source_row_number=13,
            item_value="170",
            component_number="FILE ZVG 8 B",
            material_number="FILE ZVG 8 B",
            change_number="ECN345274",
            row_classification=BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE,
            quality_qualifiers=(),
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("ebom-a-strong"),
            bom_kind=BomKind.ENRICHED_EBOM,
            sheet_name="Enriched",
            source_row_number=100,
            item_value=None,
            position_value="0010",
            component_number="PART-A",
            material_number="PART-A",
            change_number="ECN345274",
            main_value="MAIN",
            sub_value="SUB",
            vc_value="VC",
            parent_material_number="PARENT-A",
            row_classification=BomRowClassification.NOT_APPLICABLE,
            quality_qualifiers=(),
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("ebom-a-context"),
            bom_kind=BomKind.ENRICHED_EBOM,
            sheet_name="Enriched",
            source_row_number=101,
            item_value=None,
            position_value="0020",
            component_number="PART-A",
            material_number="PART-A",
            change_number="OTHER",
            main_value="MAIN",
            sub_value="SUB",
            vc_value="VC",
            parent_material_number="PARENT-B",
            row_classification=BomRowClassification.NOT_APPLICABLE,
            quality_qualifiers=(),
        ),
        make_bom_row(
            bom_row_id=synthetic_uuid("mbom-b"),
            bom_kind=BomKind.MBOM_130,
            sheet_name="MBOM",
            source_row_number=200,
            item_value="PARENT-M",
            position_value=None,
            component_number="PART-B",
            material_number="PART-B",
            change_number="MPXA1",
            parent_material_number="PARENT-M",
            row_classification=BomRowClassification.NOT_APPLICABLE,
            quality_qualifiers=(),
        ),
    )
    result = match_ecn_to_bom(
        make_ecn(ecn_number="ECN345274"), build_bom_indexes(rows)
    )

    assert {row.bom_row_id for row in result.raw_direct} == {
        synthetic_uuid("raw-a"),
        synthetic_uuid("doc-a"),
        synthetic_uuid("raw-b"),
        synthetic_uuid("candidate"),
    }
    assert {seed.number for seed in result.eligible_structural_seeds} == {
        "PART-A",
        "PART-B",
    }
    assert {
        match.occurrence.bom_row_id: match.enriched_strength
        for match in result.enriched_matches
    } == {
        synthetic_uuid(
            "ebom-a-strong"
        ): EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED,
        synthetic_uuid(
            "ebom-a-context"
        ): EnrichedMatchStrength.COMPONENT_ONLY_CONTEXT,
    }
    assert result.document_owner_context[0].owner_row_id == synthetic_uuid("raw-a")
    assert result.document_owner_context[0].asserts_owner_changed is False
    assert result.mbom_presence[0].path is MatchPath.RAW_ECN_COMPONENT
    assert result.mbom_presence[0].occurrence.change_number == "MPXA1"


def test_technical_wtpart_without_number_is_evidence_gap_not_a_match_seed() -> None:
    """Keep the source object while surfacing its missing exact component key."""
    source_object = make_source_object(number=None)
    result = match_ecn_to_bom(
        make_ecn(ecn_number=None, resulting_objects=(source_object,)),
        build_bom_indexes(()),
    )
    assert result.eligible_structural_seeds == ()
    assert any(item.code == "WTPART_NUMBER_MISSING" for item in result.diagnostics)
