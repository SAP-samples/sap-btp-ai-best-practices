"""Tests for relevant-rule selection and deterministic evidence bundles."""

from dataclasses import replace
from datetime import date

import pytest
from app.ecn.api_models import ImpactCounts
from app.ecn.errors import PersistenceStateError
from app.ecn.evidence import identifiers
from app.ecn.evidence.builder import (
    build_ecn_evidence,
    build_rule_index,
    derive_impact_counts,
    select_and_evaluate_relevant_rules,
)
from app.ecn.evidence.matching import (
    EnrichedMatchStrength,
    MatchPath,
    MatchResult,
    OccurrenceMatch,
    StructuralObjectSeed,
    StructuralSeedOrigin,
    build_bom_indexes,
    match_ecn_to_bom,
)
from app.ecn.models import (
    ArtifactKind,
    BomKind,
    BomRowClassification,
    EvidenceContext,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
    NormalizedEcn,
    SourceObjectRole,
)
from app.ecn.rules.evaluator import evaluate_rule
from app.ecn.rules.models import ParsedRule, RuleEvaluationState, RuleParseState
from app.ecn.rules.parser import parse_rule_definition

from tests.fixtures.builders import (
    make_bom_row,
    make_ecn,
    make_source_object,
    synthetic_uuid,
)


def _matched_path(sub_value: str = "SUB", vc_value: str = "VC") -> MatchResult:
    """Return one matched enriched occurrence with an exact adjacent path pair."""
    row = make_bom_row(
        bom_row_id=synthetic_uuid("enriched-1"),
        bom_kind=BomKind.ENRICHED_EBOM,
        sheet_name="Enriched",
        row_classification=BomRowClassification.NOT_APPLICABLE,
        source_row_number=20,
        component_number="PART-A",
        material_number="PART-A",
        change_number="ECN1",
        main_value="MAIN",
        sub_value=sub_value,
        vc_value=vc_value,
        item_value=None,
        position_value="0010",
        quality_qualifiers=(),
    )
    occurrence = build_bom_indexes((row,)).by_row_id[synthetic_uuid("enriched-1")]
    seed = StructuralObjectSeed(
        seed_id="seed-1",
        number="PART-A",
        origin=StructuralSeedOrigin.RAW_MATERIAL_LIKE_OBSERVED,
        source=EvidenceSource(
            artifact_id=synthetic_uuid("bom-1"),
            artifact_kind=ArtifactKind.BOM_WORKBOOK,
            json_path=None,
            sheet_name="Raw",
            source_row_number=10,
            bom_row_id=None,
        ),
        source_subtype=None,
        qualifiers=("TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED",),
    )
    match = OccurrenceMatch(
        seed,
        occurrence,
        MatchPath.RAW_ECN_COMPONENT,
        EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED,
    )
    return MatchResult((), (seed,), (match,), (), (), {}, ())


def _stored_rule(rule_id: str, source: str, row: int) -> ParsedRule:
    """Return one source-row-ordered parsed rule for dependency tests."""
    return replace(
        parse_rule_definition(source, rule_id, row), lifecycle_state="App For Series"
    )


def test_rule_selection_starts_at_exact_sub_vc_action_and_moves_forward() -> None:
    """Follow producer actions forward without making consumer conditions true."""
    first = _stored_rule("r1", "A = B --> Inclure (SUB = VC)", 2)
    second = _stored_rule("r2", "SUB = VC --> Inclure (X = Y)", 3)
    third = _stored_rule("r3", "X = Y --> Inclure (P = Q)", 4)
    unrelated = _stored_rule("r4", "Z = Q --> Inclure (OTHER = VALUE)", 5)
    results = select_and_evaluate_relevant_rules(
        _matched_path(), build_rule_index((third, unrelated, second, first)), {}, None
    )
    assert [result.rule_id for result in results] == ["r1", "r2", "r3"]
    assert [result.dependency_path for result in results] == [
        ("r1",),
        ("r1", "r2"),
        ("r1", "r2", "r3"),
    ]
    assert {result.state for result in results} == {
        RuleEvaluationState.NOT_EVALUATED_ACTIVATION_POLICY_MISSING
    }


def test_selection_is_exact_cycle_safe_and_does_not_reverse_infer() -> None:
    """Ignore near values and terminate a producer-consumer cycle deterministically."""
    first = _stored_rule("r1", "X = Y --> Inclure (SUB = VC)", 2)
    cycle = _stored_rule("r2", "SUB = VC --> Inclure (X = Y)", 3)
    near = _stored_rule("r3", "A = B --> Inclure (sub = VC)", 4)
    results = select_and_evaluate_relevant_rules(
        _matched_path(), build_rule_index((first, cycle, near)), {}, None
    )
    assert [result.rule_id for result in results] == ["r1", "r2"]


def test_ambiguous_relevant_rule_is_returned_as_parse_gap() -> None:
    """Surface a relevant ambiguous rule without evaluating its condition."""
    ambiguous = _stored_rule(
        "r-a", "A = B + C = D / E = F --> Inclure (SUB = VC)", 7
    )
    result = select_and_evaluate_relevant_rules(
        _matched_path(), build_rule_index((ambiguous,)), {}, None
    )
    assert len(result) == 1
    assert result[0].state is RuleEvaluationState.NOT_EVALUATED_PARSE_STATE
    assert result[0].parse_state is RuleParseState.AMBIGUOUS
    assert result[0].parse_reason_code == "MIXED_OPERATOR_PRECEDENCE"


def _context(rule_artifact_id: str | None = None) -> EvidenceContext:
    """Return deterministic provenance for evidence hashing."""
    return EvidenceContext(
        upload_id=synthetic_uuid("upload-1"),
        ecn_artifact_id=synthetic_uuid("ecn-artifact-1"),
        bom_artifact_id=synthetic_uuid("bom-1"),
        rule_artifact_id=rule_artifact_id,
        parser_version="ecn-source-v1",
        deterministic_logic_version="ecn-evidence-v1",
        rule_evaluator_version="ecn-rules-v1",
        analysis_as_of_date=None,
    )


def _evidence_scenario() -> tuple[NormalizedEcn, MatchResult]:
    """Build raw, corroborated/contextual EBOM, and no-MBOM evidence."""
    raw = make_bom_row(
        bom_row_id=synthetic_uuid("raw"),
        bom_kind=BomKind.RAW_EBOM_130,
        sheet_name="Raw",
        source_row_number=10,
        component_number="PART-A",
        material_number="PART-A",
        change_number="ECN1",
        valid_from_raw="02.01.1900",
        valid_from_date=date(1900, 1, 2),
        is_migration_date=True,
        row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
        quality_qualifiers=("TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED",),
    )
    strong = make_bom_row(
        bom_row_id=synthetic_uuid("strong"),
        bom_kind=BomKind.ENRICHED_EBOM,
        sheet_name="Enriched",
        source_row_number=20,
        row_classification=BomRowClassification.NOT_APPLICABLE,
        component_number="PART-A",
        material_number="PART-A",
        change_number="ECN1",
        item_value=None,
        position_value="0010",
        main_value="MAIN",
        sub_value="SUB",
        vc_value="VC",
        parent_material_number="PARENT-A",
        quality_qualifiers=(),
    )
    contextual = replace(
        strong,
        bom_row_id=synthetic_uuid("context"),
        source_row_number=21,
        parent_material_number="PARENT-B",
        change_number="OTHER",
    )
    affected = make_source_object(
        source_object_id=synthetic_uuid("affected-1"),
        role=SourceObjectRole.AFFECTED,
        technical_id="OR:wt.part.WTPart:42",
        number="PART-A",
        source_path="$.CNAffectLinks[0].AffectedObjects[0]",
    )
    ecn = make_ecn(affected_objects=(affected,))
    return ecn, match_ecn_to_bom(
        ecn, build_bom_indexes((contextual, raw, strong))
    )


def test_bundle_is_stable_occurrence_aware_and_preserves_1900_qualification() -> None:
    """Retain separate occurrences and raw migration dates under stable IDs."""
    ecn, match = _evidence_scenario()
    first = build_ecn_evidence(ecn, match, (), _context())
    second = build_ecn_evidence(
        ecn,
        replace(match, enriched_matches=tuple(reversed(match.enriched_matches))),
        (),
        _context(),
    )
    assert first == second
    occurrence_items = [
        item
        for item in first.sections.ebom_mbom_impact
        if item.values.get("component_number") == "PART-A"
    ]
    assert {
        item.values.get("parent_material_number") for item in occurrence_items
    } >= {"PARENT-A", "PARENT-B"}
    migration = [
        item
        for item in first.sections.ebom_mbom_impact
        if item.values.get("valid_from_raw") == "02.01.1900"
    ]
    assert migration and migration[0].values["valid_from"] == "1900-01-02"
    assert "MIGRATION_VALID_FROM_PLACEHOLDER" in migration[0].quality_qualifiers
    assert all(
        item.evidence_id.startswith("E-") and len(item.evidence_id) == 18
        for item in occurrence_items
    )


def test_counts_distinguish_confirmed_context_and_missing_mbom() -> None:
    """Count unique occurrences and represent absent MBOM as an evidence gap."""
    ecn, match = _evidence_scenario()
    bundle = build_ecn_evidence(ecn, match, (), _context())
    assert derive_impact_counts(ecn, match, bundle) == ImpactCounts(
        affected_item_count=1,
        raw_ecn_match_count=1,
        ebom_match_count=1,
        ebom_context_count=1,
        mbom_match_count=0,
        warning_count=0,
    )
    assert any(
        item.kind is EvidenceKind.MISSING_INPUT
        and item.values.get("code") == "MBOM_COMPONENT_NO_MATCH"
        for item in bundle.sections.uncertainties_and_missing_evidence
    )


def test_counts_include_unique_affected_and_resulting_materials() -> None:
    """Count added and affected technical parts without double-counting identities."""
    ecn, match = _evidence_scenario()
    added = make_source_object(
        source_object_id=synthetic_uuid("resulting-1"),
        role=SourceObjectRole.RESULTING,
        technical_id="OR:wt.part.WTPart:99",
        number="PART-B",
        source_path="$.ResultingObjects[0]",
    )
    repeated = make_source_object(
        source_object_id=synthetic_uuid("resulting-2"),
        role=SourceObjectRole.RESULTING,
        technical_id="OR:wt.part.WTPart:42",
        number="PART-A",
        source_path="$.ResultingObjects[1]",
    )
    ecn = replace(ecn, resulting_objects=(added, repeated))
    bundle = build_ecn_evidence(ecn, match, (), _context())

    assert derive_impact_counts(ecn, match, bundle).affected_item_count == 2


def test_missing_rule_artifact_is_a_configuration_gap_only() -> None:
    """Allow other sections to remain usable without rule configuration."""
    ecn, match = _evidence_scenario()
    bundle = build_ecn_evidence(ecn, match, (), _context(rule_artifact_id=None))
    assert any(
        item.kind is EvidenceKind.MISSING_INPUT
        and item.values.get("code") == "RULE_ARTIFACT_MISSING"
        for item in bundle.sections.configuration_scope
    )
    assert bundle.sections.ebom_mbom_impact


def test_plant_scope_gap_depends_on_frozen_rules_and_mapping() -> None:
    """Plants come from pinned rules plus the frozen Plant Code Mapping; lacking either keeps the gap."""
    ecn, match = _evidence_scenario()

    def has_gap(context):
        """Return whether the bundle reports plant scope as unavailable."""
        bundle = build_ecn_evidence(ecn, match, (), context)
        return any(item.values.get("code") == "PLANT_SCOPE_UNAVAILABLE" for item in bundle.sections.configuration_scope)

    base = _context(rule_artifact_id=synthetic_uuid("rules-1"))
    mapping = {"mappings": [{"suffix": "JB", "plant_code": "PL10", "plant_name": "Plant A"}], "excluded_rule_states": []}
    assert has_gap(base)
    assert has_gap(replace(base, plant_assignment=mapping))
    assert not has_gap(replace(base, plant_assignment=mapping, rule_source_artifact_ids={"rule": synthetic_uuid("rules-1")}))


def test_document_owner_item_is_tier_d_and_never_owner_change() -> None:
    """Keep documentation nesting separate from a changed-material assertion."""
    owner = make_bom_row(
        bom_row_id=synthetic_uuid("owner"),
        bom_kind=BomKind.RAW_EBOM_130,
        sheet_name="Raw",
        source_row_number=10,
        component_number="PART-A",
        change_number="ECN1",
        row_classification=BomRowClassification.MATERIAL_LIKE_OBSERVED,
    )
    document = make_bom_row(
        bom_row_id=synthetic_uuid("document"),
        bom_kind=BomKind.RAW_EBOM_130,
        sheet_name="Raw",
        source_row_number=11,
        parent_bom_row_id=synthetic_uuid("owner"),
        dotted_level="..1",
        item_value="D01D001",
        position_value="D01D001",
        component_number="DOC ZVD 1 A",
        change_number="ECN1",
        row_classification=BomRowClassification.CONFIRMED_DOCUMENT,
        is_documentation=True,
        quality_qualifiers=(),
    )
    ecn = make_ecn()
    match = match_ecn_to_bom(ecn, build_bom_indexes((owner, document)))
    bundle = build_ecn_evidence(ecn, match, (), _context())
    tier_d = [
        item
        for item in bundle.sections.ebom_mbom_impact
        if item.tier is EvidenceTier.DOCUMENT_OWNER_CONTEXT
    ]
    assert len(tier_d) == 1
    assert tier_d[0].values["asserts_owner_changed"] is False
    document_fact = next(
        item
        for item in bundle.sections.ebom_mbom_impact
        if item.values.get("row_classification") == "CONFIRMED_DOCUMENT"
    )
    assert document_fact.values["is_documentation"] is True
    assert document_fact.values["eligible_as_structural_seed"] is False


def test_truncated_evidence_hash_collision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently alias two facts to one user-visible citation ID."""
    ecn, match = _evidence_scenario()
    monkeypatch.setattr(identifiers, "canonical_sha256", lambda value: "0" * 64)
    with pytest.raises(PersistenceStateError) as caught:
        build_ecn_evidence(ecn, match, (), _context())
    assert caught.value.code == "EVIDENCE_ID_COLLISION"


def test_matched_row_quality_contradiction_is_visible_and_counted() -> None:
    """Project a relevant BOM contradiction into warnings and list count."""
    ecn, match = _evidence_scenario()
    first_match = match.enriched_matches[0]
    contradicted = replace(
        first_match,
        occurrence=replace(
            first_match.occurrence,
            quality_qualifiers=("ENRICHED_PATH_CONTRADICTION",),
            quality_details={"expected_parent_material_number": "EXPECTED"},
        ),
    )
    changed = replace(
        match, enriched_matches=(contradicted, *match.enriched_matches[1:])
    )
    bundle = build_ecn_evidence(ecn, changed, (), _context())
    assert any(
        item.kind is EvidenceKind.CONTRADICTION
        and item.values.get("code") == "ENRICHED_PATH_CONTRADICTION"
        for item in bundle.sections.uncertainties_and_missing_evidence
    )
    assert derive_impact_counts(ecn, changed, bundle).warning_count == 1


def test_effectivity_stays_undetermined_without_analysis_date() -> None:
    """Never classify a BOM interval active when no analysis date is supplied."""
    ecn, match = _evidence_scenario()
    bundle = build_ecn_evidence(ecn, match, (), _context())
    assert all(
        item.values.get("is_active") is None
        for item in bundle.sections.ebom_mbom_impact
        if "is_active" in item.values
    )


def test_bundle_retains_full_normalized_ecn_prose_as_citable_source_evidence() -> None:
    """Carry problem and solution prose into the evidence-only model boundary."""
    ecn, match = _evidence_scenario()
    prose = (
        "DescriptionSummary:\nShort summary\n\n"
        "Description:\nDetailed problem statement\n\n"
        "SOLUTION_DESCRIPTION:\nApproved solution detail"
    )
    bundle = build_ecn_evidence(
        replace(
            ecn,
            description_summary="Short summary",
            description_text=prose,
        ),
        match,
        (),
        _context(),
    )
    header = bundle.sections.executive_summary[0]
    assert header.values["description_text"] == prose
    assert header.sources[0].json_path == f"$[{ecn.source_ordinal}]"


def test_full_prose_is_not_replaced_by_the_bounded_display_summary() -> None:
    """Preserve normalized full prose when the display summary is truncated."""
    ecn, match = _evidence_scenario()
    full_summary = "S" * 2_100
    bounded_summary = full_summary[:2_000]
    bundle = build_ecn_evidence(
        replace(
            ecn,
            description_summary=bounded_summary,
            description_text=f"DescriptionSummary:\n{full_summary}",
        ),
        match,
        (),
        _context(),
    )
    values = bundle.sections.executive_summary[0].values
    assert values["description_summary"] == bounded_summary
    assert values["description_text"] == f"DescriptionSummary:\n{full_summary}"


def test_prose_only_ecn_remains_available_to_later_generation() -> None:
    """Retain useful ECN text even when header display fields are unavailable."""
    ecn, match = _evidence_scenario()
    bundle = build_ecn_evidence(
        replace(
            ecn,
            ecn_number=None,
            title=None,
            description_summary=None,
            description_text="Description:\nOnly available problem context",
        ),
        match,
        (),
        _context(),
    )
    values = bundle.sections.executive_summary[0].values
    assert values["ecn_number"] is None
    assert values["description_summary"] is None
    assert values["description_text"] == "Description:\nOnly available problem context"


def test_scoped_rule_citations_keep_the_actual_source_artifact():
    """A rule from the second scoped workbook must not cite the primary workbook."""
    ecn, matches = _evidence_scenario()
    rule = _stored_rule("scope-rule", "A = B --> Inclure (SUB = VC)", 2)
    actual = synthetic_uuid("second-rule-artifact")
    context = replace(_context(synthetic_uuid("first-rule-artifact")), rule_source_artifact_ids={rule.rule_id: actual})
    results = select_and_evaluate_relevant_rules(matches, build_rule_index((rule,)), {}, None)
    bundle = build_ecn_evidence(ecn, matches, results, context)
    findings = [item for item in bundle.sections.configuration_scope if item.values.get("rule_id") == rule.rule_id]
    assert findings and findings[0].sources[0].artifact_id == actual
