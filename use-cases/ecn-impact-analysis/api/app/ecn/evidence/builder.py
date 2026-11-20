"""Select path-relevant rules and build deterministic evidence projections."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from app.ecn.api_models import ImpactCounts
from app.ecn.errors import PersistenceStateError
from app.ecn.evidence import identifiers
from app.ecn.evidence.change_candidates import derive_change_candidates
from app.ecn.evidence.matching import (
    BomOccurrence,
    EnrichedMatchStrength,
    MatchPath,
    MatchResult,
    occurrence_key,
)
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    BomRowClassification,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
    JsonValue,
    NormalizedEcn,
)
from app.ecn.rules.evaluator import evaluate_rule
from app.ecn.rules.models import (
    AndExpression,
    EqualsExpression,
    FactSet,
    IncludeAssignment,
    NotExpression,
    OrExpression,
    ParsedRule,
    RuleActivationPolicy,
    RuleEvaluation,
    RuleEvaluationState,
    RuleExpression,
)


@dataclass(frozen=True)
class RuleDependencyEdge:
    """Hold one exact producer-action to consumer-condition dependency."""

    producer_rule_id: str
    assignment: IncludeAssignment
    consumer_rule_id: str
    condition: EqualsExpression


@dataclass(frozen=True)
class RuleIndex:
    """Index rules by exact actions and forward dependency edges."""

    by_id: Mapping[str, ParsedRule]
    by_action_assignment: Mapping[tuple[str, str], tuple[ParsedRule, ...]]
    forward_edges_by_rule_id: Mapping[str, tuple[RuleDependencyEdge, ...]]


def _equalities(
    expression: RuleExpression | None,
) -> tuple[EqualsExpression, ...]:
    """Return exact equality leaves in source order from a known AST."""
    if expression is None:
        return ()
    if isinstance(expression, EqualsExpression):
        return (expression,)
    if isinstance(expression, NotExpression):
        return _equalities(expression.operand)
    if isinstance(expression, (AndExpression, OrExpression)):
        return tuple(
            item
            for operand in expression.operands
            for item in _equalities(operand)
        )
    raise TypeError(f"Unexpected rule expression: {type(expression).__name__}")


def _rule_order(rule: ParsedRule) -> tuple[int, str]:
    """Return deterministic source-row order with missing rows last."""
    return (
        rule.source_row_number if rule.source_row_number is not None else 2**31,
        rule.rule_id,
    )


def build_rule_index(rules: Sequence[ParsedRule]) -> RuleIndex:
    """Build exact action keys and producer-to-consumer dependency edges.

    Args:
        rules: Stored parsed, ambiguous, or unsupported rule records.

    Returns:
        Immutable-by-contract mappings for identities, action seeds, and
        forward dependency traversal.
    """
    by_id = {rule.rule_id: rule for rule in rules}
    actions: dict[tuple[str, str], list[ParsedRule]] = {}
    consumers: dict[
        tuple[str, str], list[tuple[ParsedRule, EqualsExpression]]
    ] = {}
    for rule in rules:
        if rule.action is not None:
            for assignment in rule.action.assignments:
                actions.setdefault(
                    (assignment.characteristic, assignment.value), []
                ).append(rule)
        for condition in _equalities(rule.condition):
            consumers.setdefault((condition.left, condition.right), []).append(
                (rule, condition)
            )
    edges: dict[str, list[RuleDependencyEdge]] = {
        rule.rule_id: [] for rule in rules
    }
    for key, producers in actions.items():
        for producer in producers:
            assignment = IncludeAssignment(*key)
            for consumer, condition in consumers.get(key, ()):
                edges[producer.rule_id].append(
                    RuleDependencyEdge(
                        producer.rule_id,
                        assignment,
                        consumer.rule_id,
                        condition,
                    )
                )

    def edge_order(edge: RuleDependencyEdge) -> tuple[int, str]:
        """Sort an edge by the consumer's stable source order."""
        return _rule_order(by_id[edge.consumer_rule_id])

    return RuleIndex(
        by_id=by_id,
        by_action_assignment={
            key: tuple(sorted(value, key=_rule_order))
            for key, value in actions.items()
        },
        forward_edges_by_rule_id={
            key: tuple(sorted(value, key=edge_order)) for key, value in edges.items()
        },
    )


def select_and_evaluate_relevant_rules(
    match_result: MatchResult,
    rule_index: RuleIndex,
    facts: Mapping[str, FactSet],
    activation_policy: RuleActivationPolicy | None,
) -> tuple[RuleEvaluation, ...]:
    """Select exact path actions, traverse forward, and evaluate independently.

    Args:
        match_result: Exact occurrence-aware crosswalk for one ECN.
        rule_index: Exact action/dependency index over one rule artifact.
        facts: Independent configuration facts; selected actions never mutate it.
        activation_policy: Explicit evaluator policy or ``None`` for MVP v1.

    Returns:
        Source-ordered evaluations with shortest stable dependency paths.
    """
    path_by_rule: dict[str, tuple[str, ...]] = {}
    queue: deque[str] = deque()
    for match in match_result.enriched_matches:
        sub_value = match.occurrence.sub_value
        vc_value = match.occurrence.vc_value
        if not sub_value or not vc_value:
            continue
        for rule in rule_index.by_action_assignment.get((sub_value, vc_value), ()):
            candidate = (rule.rule_id,)
            current = path_by_rule.get(rule.rule_id)
            if current is None or (len(candidate), candidate) < (len(current), current):
                path_by_rule[rule.rule_id] = candidate
                queue.append(rule.rule_id)
    while queue:
        producer_id = queue.popleft()
        source_path = path_by_rule[producer_id]
        for edge in rule_index.forward_edges_by_rule_id.get(producer_id, ()):
            if edge.consumer_rule_id in source_path:
                continue
            candidate = (*source_path, edge.consumer_rule_id)
            current = path_by_rule.get(edge.consumer_rule_id)
            if current is None or (len(candidate), candidate) < (len(current), current):
                path_by_rule[edge.consumer_rule_id] = candidate
                queue.append(edge.consumer_rule_id)
    selected = sorted(
        (rule_index.by_id[rule_id] for rule_id in path_by_rule),
        key=_rule_order,
    )
    return tuple(
        replace(
            evaluate_rule(rule, facts, activation_policy),
            dependency_path=path_by_rule[rule.rule_id],
        )
        for rule in selected
    )


class _EvidenceCollector:
    """Create, collision-check, sort, and group deterministic evidence items."""

    def __init__(self) -> None:
        """Initialize empty item and full-identity indexes."""
        self._items: list[EvidenceItem] = []
        self._identity_by_id: dict[str, dict[str, JsonValue]] = {}

    def add(
        self,
        *,
        section: AnalysisSectionKey,
        kind: EvidenceKind,
        tier: EvidenceTier | None,
        summary: str,
        sources: Sequence[EvidenceSource] = (),
        values: Mapping[str, JsonValue] | None = None,
        related_ids: Sequence[str] = (),
        qualifiers: Sequence[str] = (),
        occurrence: tuple[object, ...] | None = None,
        discriminator: str,
    ) -> EvidenceItem:
        """Add one fact and reject any truncated-ID collision."""
        ordered_sources = tuple(
            sorted(sources, key=identifiers.source_coordinate)
        )
        identity = identifiers.make_evidence_identity(
            section,
            kind,
            tier,
            ordered_sources,
            related_ids,
            occurrence,
            discriminator,
        )
        evidence_id = identifiers.make_evidence_id(identity)
        prior = self._identity_by_id.get(evidence_id)
        if prior is not None and prior != identity:
            raise PersistenceStateError(
                "EVIDENCE_ID_COLLISION",
                "Two deterministic evidence facts produced the same citation ID.",
                status_code=500,
            )
        self._identity_by_id[evidence_id] = identity
        item = EvidenceItem(
            evidence_id=evidence_id,
            section_key=section,
            kind=kind,
            tier=tier,
            summary=summary,
            sources=ordered_sources,
            values=dict(values or {}),
            related_ids=tuple(sorted(set(related_ids))),
            quality_qualifiers=tuple(sorted(set(qualifiers))),
        )
        if item not in self._items:
            self._items.append(item)
        return item

    def grouped(self) -> EvidenceBundleSections:
        """Return all six sections in deterministic source/ID order."""
        grouped: dict[AnalysisSectionKey, list[EvidenceItem]] = {
            section: [] for section in AnalysisSectionKey
        }
        for item in self._items:
            grouped[item.section_key].append(item)

        def order(item: EvidenceItem) -> tuple[tuple[str, str, str, int, str], str]:
            """Sort source-backed items by first coordinate, then stable ID."""
            coordinate = (
                identifiers.source_coordinate(item.sources[0])
                if item.sources
                else ("", "", "", -1, "")
            )
            return coordinate, item.evidence_id

        return EvidenceBundleSections(
            executive_summary=tuple(
                sorted(grouped[AnalysisSectionKey.EXECUTIVE_SUMMARY], key=order)
            ),
            change_intent=tuple(
                sorted(grouped[AnalysisSectionKey.CHANGE_INTENT], key=order)
            ),
            affected_materials=tuple(
                sorted(grouped[AnalysisSectionKey.AFFECTED_MATERIALS], key=order)
            ),
            ebom_mbom_impact=tuple(
                sorted(grouped[AnalysisSectionKey.EBOM_MBOM_IMPACT], key=order)
            ),
            configuration_scope=tuple(
                sorted(grouped[AnalysisSectionKey.CONFIGURATION_SCOPE], key=order)
            ),
            uncertainties_and_missing_evidence=tuple(
                sorted(
                    grouped[
                        AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE
                    ],
                    key=order,
                )
            ),
        )


def _bom_source(occurrence: BomOccurrence) -> EvidenceSource:
    """Return the exact workbook coordinate for one normalized occurrence."""
    return EvidenceSource(
        artifact_id=occurrence.artifact_id,
        artifact_kind=ArtifactKind.BOM_WORKBOOK,
        json_path=None,
        sheet_name=occurrence.sheet_name,
        source_row_number=occurrence.source_row_number,
        bom_row_id=occurrence.bom_row_id,
    )


def _json_source(artifact_id: str, path: str) -> EvidenceSource:
    """Return one exact ECN JSON coordinate."""
    return EvidenceSource(
        artifact_id=artifact_id,
        artifact_kind=ArtifactKind.ECN_JSON,
        json_path=path,
        sheet_name=None,
        source_row_number=None,
        bom_row_id=None,
    )


def _rule_source(artifact_id: str, row_number: int | None) -> EvidenceSource:
    """Return one exact conditional-rule workbook coordinate."""
    return EvidenceSource(
        artifact_id=artifact_id,
        artifact_kind=ArtifactKind.CONDITIONAL_RULE_WORKBOOK,
        json_path=None,
        sheet_name="Default",
        source_row_number=row_number,
        bom_row_id=None,
    )


def _bom_values(occurrence: BomOccurrence) -> dict[str, JsonValue]:
    """Return concise fields without losing occurrence semantics."""
    return {
        "bom_kind": occurrence.bom_kind.value,
        "component_number": occurrence.component_number,
        "material_number": occurrence.material_number,
        "parent_material_number": occurrence.parent_material_number,
        "parent_path": list(occurrence.parent_path),
        "item_or_position": occurrence.item_or_position,
        "row_classification": occurrence.row_classification.value,
        "is_documentation": (
            occurrence.row_classification
            is BomRowClassification.CONFIRMED_DOCUMENT
        ),
        "eligible_as_structural_seed": (
            occurrence.row_classification
            is BomRowClassification.MATERIAL_LIKE_OBSERVED
        ),
        "description_text": occurrence.description_text,
        "change_number": occurrence.change_number,
        "change_number_to": occurrence.change_number_to,
        "quantity": (
            str(occurrence.quantity) if occurrence.quantity is not None else None
        ),
        "unit": occurrence.unit,
        "plant": occurrence.plant,
        "alternative": occurrence.alternative,
        "valid_from_raw": occurrence.valid_from_raw,
        "valid_from": occurrence.valid_from.isoformat() if occurrence.valid_from else None,
        "valid_to_raw": occurrence.valid_to_raw,
        "valid_to": occurrence.valid_to.isoformat() if occurrence.valid_to else None,
        "is_migration_date": occurrence.is_migration_date,
        "is_active": None,
        "quality_details": dict(occurrence.quality_details),
    }


def _occurrence_qualifiers(occurrence: BomOccurrence) -> tuple[str, ...]:
    """Project the migration boolean into the named evidence qualifier."""
    values = list(occurrence.quality_qualifiers)
    if occurrence.is_migration_date:
        values.append("MIGRATION_VALID_FROM_PLACEHOLDER")
    return tuple(dict.fromkeys(values))


def _required_row(match_result: MatchResult, row_id: str) -> BomOccurrence:
    """Resolve a document bridge endpoint retained by the matching result."""
    try:
        return match_result.context_rows_by_id[row_id]
    except KeyError as exc:
        raise PersistenceStateError(
            "EVIDENCE_SOURCE_ROW_MISSING",
            "A document-owner bridge references an unavailable BOM row.",
            status_code=500,
        ) from exc


def _add_gap(
    collector: _EvidenceCollector,
    *,
    section: AnalysisSectionKey,
    code: str,
    message: str,
    sources: Sequence[EvidenceSource] = (),
) -> None:
    """Add one ordinary missing-input item with stable code identity."""
    collector.add(
        section=section,
        kind=EvidenceKind.MISSING_INPUT,
        tier=None,
        summary=message,
        sources=sources,
        values={"code": code},
        discriminator=f"missing:{code}",
    )


def _add_diagnostics_and_missing_inputs(
    collector: _EvidenceCollector,
    ecn: NormalizedEcn,
    match_result: MatchResult,
    rule_results: Sequence[RuleEvaluation],
    context: EvidenceContext,
) -> None:
    """Add diagnostics and only gaps required by unavailable conclusions."""
    diagnostic_codes: set[str] = set()
    for diagnostic in match_result.diagnostics:
        diagnostic_codes.add(diagnostic.code)
        collector.add(
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            kind=diagnostic.kind,
            tier=None,
            summary=diagnostic.message,
            sources=diagnostic.sources,
            values={"code": diagnostic.code},
            discriminator=f"diagnostic:{diagnostic.code}",
        )
    for qualifier in sorted(set(ecn.normalization_qualifiers)):
        kind = (
            EvidenceKind.CONTRADICTION
            if "IDENTITY" in qualifier
            and ("DISAGREE" in qualifier or "CONFLICT" in qualifier)
            else EvidenceKind.WARNING
        )
        collector.add(
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            kind=kind,
            tier=None,
            summary="ECN normalization produced a source-quality qualifier.",
            sources=(_json_source(ecn.artifact_id, f"$[{ecn.source_ordinal}]"),),
            values={"code": qualifier},
            discriminator=f"normalization:{qualifier}",
        )
    for source_object in (*ecn.affected_objects, *ecn.resulting_objects):
        for qualifier in sorted(set(source_object.quality_qualifiers)):
            collector.add(
                section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
                kind=(
                    EvidenceKind.CONTRADICTION
                    if qualifier == "TECHNICAL_IDENTITY_CONTRADICTION"
                    else EvidenceKind.WARNING
                ),
                tier=None,
                summary="An ECN source object carries a source-quality qualifier.",
                sources=(_json_source(ecn.artifact_id, source_object.source_path),),
                values={
                    "code": qualifier,
                    "source_object_id": source_object.source_object_id,
                },
                related_ids=(source_object.source_object_id,),
                discriminator=(
                    f"source-object-quality:{source_object.source_object_id}:{qualifier}"
                ),
            )
    for result in rule_results:
        if "SUSPICIOUS_OR_OF_NEGATIONS" not in result.warnings:
            continue
        sources = (
            ()
            if context.rule_artifact_id is None
            else (_rule_source(context.rule_source_artifact_ids.get(result.rule_id, context.rule_artifact_id), result.source_row_number),)
        )
        collector.add(
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            kind=EvidenceKind.WARNING,
            tier=None,
            summary="A relevant OR-of-negations rule requires semantic review.",
            sources=sources,
            values={
                "code": "SUSPICIOUS_OR_OF_NEGATIONS",
                "rule_id": result.rule_id,
            },
            related_ids=(result.rule_id,),
            discriminator=f"rule-quality:{result.rule_id}:SUSPICIOUS_OR_OF_NEGATIONS",
        )
    relevant_occurrences = (
        *match_result.context_rows_by_id.values(),
        *match_result.raw_direct,
        *(match.occurrence for match in match_result.enriched_matches),
        *(match.occurrence for match in match_result.mbom_presence),
    )
    informational_qualifiers = {
        "TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED",
        "MIGRATION_VALID_FROM_PLACEHOLDER",
    }
    contradiction_qualifiers = {"ENRICHED_PATH_CONTRADICTION"}
    for occurrence in relevant_occurrences:
        for qualifier in sorted(
            set(occurrence.quality_qualifiers) - informational_qualifiers
        ):
            collector.add(
                section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
                kind=(
                    EvidenceKind.CONTRADICTION
                    if qualifier in contradiction_qualifiers
                    else EvidenceKind.WARNING
                ),
                tier=None,
                summary="A matched BOM occurrence carries a source-quality qualifier.",
                sources=(_bom_source(occurrence),),
                values={
                    "code": qualifier,
                    "quality_details": dict(occurrence.quality_details),
                },
                occurrence=occurrence_key(occurrence),
                discriminator=f"bom-quality:{occurrence.bom_row_id}:{qualifier}",
            )
    source_objects = (*ecn.affected_objects, *ecn.resulting_objects)
    if not source_objects:
        _add_gap(
            collector,
            section=AnalysisSectionKey.AFFECTED_MATERIALS,
            code="ECN_SOURCE_OBJECTS_MISSING",
            message="The ECN contains no affected or resulting source-object occurrence.",
        )
    elif not any(item.is_technical_wtpart for item in source_objects):
        _add_gap(
            collector,
            section=AnalysisSectionKey.AFFECTED_MATERIALS,
            code="TECHNICAL_WTPART_IDENTITY_MISSING",
            message="No source object has a validated technical WTPart identity for exact BOM matching.",
        )
    if not match_result.raw_direct and not diagnostic_codes.intersection(
        {"RAW_ECN_NO_MATCH", "BUSINESS_ECN_MISSING"}
    ):
        _add_gap(
            collector,
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            code="RAW_ECN_NO_MATCH",
            message="No exact raw EBOM Change No. occurrence is available.",
        )
    confirmed = [
        item
        for item in match_result.enriched_matches
        if item.path is MatchPath.RAW_ECN_COMPONENT
        and item.enriched_strength
        is EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
    ]
    if not confirmed:
        _add_gap(
            collector,
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            code="EBOM_CONFIRMED_NO_MATCH",
            message="No enriched EBOM occurrence confirms both exact component and the same ECN.",
        )
    matched_mbom_components = {
        item.seed.number for item in match_result.mbom_presence
    }
    unmatched_mbom_seeds = {
        seed.number: seed
        for seed in match_result.eligible_structural_seeds
        if seed.number not in matched_mbom_components
    }
    for component_number, seed in sorted(unmatched_mbom_seeds.items()):
        collector.add(
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            kind=EvidenceKind.MISSING_INPUT,
            tier=None,
            summary=(
                "No exact eligible occurrence is available in the MBOM snapshot "
                f"for component {component_number}."
            ),
            sources=(seed.source,),
            values={
                "code": "MBOM_COMPONENT_NO_MATCH",
                "component_number": component_number,
            },
            qualifiers=seed.qualifiers,
            discriminator=f"missing:mbom-component:{component_number}",
        )
    if (
        not match_result.mbom_presence
        and not match_result.eligible_structural_seeds
    ):
        _add_gap(
            collector,
            section=AnalysisSectionKey.UNCERTAINTIES_AND_MISSING_EVIDENCE,
            code="MBOM_COMPONENT_NO_MATCH",
            message="No exact eligible component occurrence is available in the MBOM snapshot.",
        )
    path_is_known = any(
        item.occurrence.sub_value and item.occurrence.vc_value
        for item in match_result.enriched_matches
    )
    if context.rule_artifact_id is None:
        _add_gap(
            collector,
            section=AnalysisSectionKey.CONFIGURATION_SCOPE,
            code="RULE_ARTIFACT_MISSING",
            message="No imported conditional-rule artifact is available.",
        )
    elif not path_is_known:
        _add_gap(
            collector,
            section=AnalysisSectionKey.CONFIGURATION_SCOPE,
            code="RULE_PATH_NOT_ESTABLISHED",
            message="Matched EBOM evidence does not establish an adjacent Sub/VC pair for rule selection.",
        )
    elif not rule_results:
        collector.add(
            section=AnalysisSectionKey.CONFIGURATION_SCOPE,
            kind=EvidenceKind.DETERMINISTIC_FINDING,
            tier=None,
            summary="No stored rule Include action exactly matches the established Sub/VC pair.",
            sources=(_rule_source(context.rule_artifact_id, None),),
            values={"code": "NO_RELEVANT_RULE_MATCH"},
            discriminator="rule-selection:no-relevant-match",
        )
    matched_occurrences = {
        occurrence_key(item.occurrence): item.occurrence
        for item in (*match_result.enriched_matches, *match_result.mbom_presence)
    }
    if not matched_occurrences:
        _add_gap(
            collector,
            section=AnalysisSectionKey.CONFIGURATION_SCOPE,
            code="BOM_CONFIGURATION_SCOPE_MISSING",
            message="No matched BOM occurrence provides plant, alternative, or effectivity context.",
        )
    else:
        scope_sources = tuple(
            _bom_source(item) for item in matched_occurrences.values()
        )
        scope_fields = {
            # Plants come from conditional rules and the Plant Code Mapping
            # frozen at import, never from BOM rows.
            "PLANT_SCOPE_UNAVAILABLE": not (
                context.rule_source_artifact_ids and context.plant_assignment.get("mappings")
            ),
            "ALTERNATIVE_SCOPE_UNAVAILABLE": all(
                item.alternative is None for item in matched_occurrences.values()
            ),
            "EFFECTIVITY_SCOPE_UNAVAILABLE": all(
                item.valid_from_raw is None and item.valid_to_raw is None
                for item in matched_occurrences.values()
            ),
        }
        for code, unavailable in scope_fields.items():
            if unavailable:
                _add_gap(
                    collector,
                    section=AnalysisSectionKey.CONFIGURATION_SCOPE,
                    code=code,
                    message="Production-valid scope cannot be established from the available BOM fields.",
                    sources=scope_sources,
                )
        if context.analysis_as_of_date is None:
            _add_gap(
                collector,
                section=AnalysisSectionKey.CONFIGURATION_SCOPE,
                code="ANALYSIS_AS_OF_DATE_MISSING",
                message="No analysis-as-of date is available, so effectivity cannot be classified active.",
                sources=scope_sources,
            )


def build_ecn_evidence(
    ecn: NormalizedEcn,
    match_result: MatchResult,
    rule_results: Sequence[RuleEvaluation],
    context: EvidenceContext,
) -> EvidenceBundle:
    """Build six deterministic sections with partial findings and explicit gaps.

    Args:
        ecn: One immutable normalized ECN occurrence.
        match_result: Exact BOM crosswalk for the same occurrence.
        rule_results: Relevant independently evaluated rule results.
        context: Artifact and deterministic-version provenance.

    Returns:
        Collision-checked, canonically hashed six-section evidence bundle.
    """
    collector = _EvidenceCollector()
    root_source = _json_source(ecn.artifact_id, f"$[{ecn.source_ordinal}]")
    collector.add(
        section=AnalysisSectionKey.EXECUTIVE_SUMMARY,
        kind=EvidenceKind.SOURCE_FACT,
        tier=None,
        summary="Uploaded ECN occurrence header.",
        sources=(root_source,),
        values={
            "ecn_number": ecn.ecn_number,
            "title": ecn.title,
            "description_summary": ecn.description_summary,
            "description_text": ecn.description_text,
            "source_ordinal": ecn.source_ordinal,
        },
        discriminator="ecn-header",
    )
    for candidate in derive_change_candidates(ecn):
        sources = tuple(
            _json_source(ecn.artifact_id, path) for path in candidate.source_paths
        )
        collector.add(
            section=AnalysisSectionKey.CHANGE_INTENT,
            kind=EvidenceKind.DETERMINISTIC_FINDING,
            tier=None,
            summary=f"Source objects form a {candidate.kind.value} candidate fact.",
            sources=sources,
            values={
                "candidate_kind": candidate.kind.value,
                "affected_object_ids": list(candidate.affected_object_ids),
                "resulting_object_ids": list(candidate.resulting_object_ids),
                "is_add_remove_replace_decision": False,
            },
            related_ids=(candidate.candidate_id,),
            qualifiers=candidate.qualifiers,
            discriminator=f"change-candidate:{candidate.candidate_id}",
        )
    for source_object in (*ecn.affected_objects, *ecn.resulting_objects):
        collector.add(
            section=AnalysisSectionKey.AFFECTED_MATERIALS,
            kind=EvidenceKind.SOURCE_FACT,
            tier=None,
            summary="ECN source object retained with its source role and identity type.",
            sources=(_json_source(ecn.artifact_id, source_object.source_path),),
            values={
                "role": source_object.role.value,
                "number": source_object.number,
                "technical_id": source_object.technical_id,
                "version_id": source_object.version_id,
                "revision": source_object.revision,
                "version": source_object.version,
                "is_technical_wtpart": source_object.is_technical_wtpart,
                "source_subtype": source_object.source_subtype,
            },
            related_ids=(source_object.source_object_id,),
            qualifiers=source_object.quality_qualifiers,
            discriminator=f"source-object:{source_object.source_object_id}",
        )
    for raw in match_result.raw_direct:
        collector.add(
            section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
            kind=EvidenceKind.SOURCE_FACT,
            tier=EvidenceTier.DIRECT_ECN_RAW,
            summary="Raw EBOM row has the exact ECN Change No.",
            sources=(_bom_source(raw),),
            values=_bom_values(raw),
            occurrence=occurrence_key(raw),
            qualifiers=_occurrence_qualifiers(raw),
            discriminator=f"raw-direct:{raw.bom_row_id}",
        )
    for match in match_result.enriched_matches:
        strong = (
            match.enriched_strength
            is EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
        )
        collector.add(
            section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
            kind=EvidenceKind.DETERMINISTIC_FINDING,
            tier=(
                EvidenceTier.EXACT_WTPART_WHERE_USED
                if strong
                else EvidenceTier.CONTEXTUAL_BOM_OCCURRENCE
            ),
            summary=(
                "Enriched EBOM occurrence confirms exact component and ECN."
                if strong
                else "Enriched EBOM occurrence is exact component-only context."
            ),
            sources=(match.seed.source, _bom_source(match.occurrence)),
            values={
                **_bom_values(match.occurrence),
                "match_path": match.path.value,
                "match_strength": (
                    match.enriched_strength.value
                    if match.enriched_strength is not None
                    else None
                ),
            },
            occurrence=occurrence_key(match.occurrence),
            qualifiers=(
                *match.seed.qualifiers,
                *_occurrence_qualifiers(match.occurrence),
            ),
            discriminator=(
                f"enriched:{match.seed.seed_id}:{match.occurrence.bom_row_id}:"
                f"{match.path.value}"
            ),
        )
    for match in match_result.mbom_presence:
        tier = (
            EvidenceTier.EXACT_WTPART_WHERE_USED
            if match.path is MatchPath.JSON_WTPART
            else EvidenceTier.CONTEXTUAL_BOM_OCCURRENCE
        )
        collector.add(
            section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
            kind=EvidenceKind.DETERMINISTIC_FINDING,
            tier=tier,
            summary=(
                "Exact component is present in one MBOM occurrence; "
                "this is not a direct ECN link."
            ),
            sources=(match.seed.source, _bom_source(match.occurrence)),
            values={
                **_bom_values(match.occurrence),
                "match_path": match.path.value,
                "is_direct_ecn_mbom_link": False,
            },
            occurrence=occurrence_key(match.occurrence),
            qualifiers=(
                *match.seed.qualifiers,
                *_occurrence_qualifiers(match.occurrence),
            ),
            discriminator=(
                f"mbom-presence:{match.seed.seed_id}:{match.occurrence.bom_row_id}:"
                f"{match.path.value}"
            ),
        )
    for bridge in match_result.document_owner_context:
        document = _required_row(match_result, bridge.document_row_id)
        owner = _required_row(match_result, bridge.owner_row_id)
        collector.add(
            section=AnalysisSectionKey.EBOM_MBOM_IMPACT,
            kind=EvidenceKind.DETERMINISTIC_FINDING,
            tier=EvidenceTier.DOCUMENT_OWNER_CONTEXT,
            summary="Documentation is nested under the nearest material-like ancestor.",
            sources=(_bom_source(document), _bom_source(owner)),
            values={
                "document_row_id": bridge.document_row_id,
                "owner_row_id": bridge.owner_row_id,
                "owner_component": bridge.owner_component,
                "asserts_owner_changed": False,
            },
            related_ids=(bridge.document_row_id, bridge.owner_row_id),
            discriminator=f"document-owner:{bridge.document_row_id}:{bridge.owner_row_id}",
        )
    for result in rule_results:
        kind = (
            EvidenceKind.DETERMINISTIC_FINDING
            if result.state is RuleEvaluationState.EVALUATED
            else EvidenceKind.MISSING_INPUT
        )
        sources = (
            ()
            if context.rule_artifact_id is None
            else (_rule_source(context.rule_source_artifact_ids.get(result.rule_id, context.rule_artifact_id), result.source_row_number),)
        )
        collector.add(
            section=AnalysisSectionKey.CONFIGURATION_SCOPE,
            kind=kind,
            tier=None,
            summary=(
                "Relevant conditional rule was evaluated."
                if kind is EvidenceKind.DETERMINISTIC_FINDING
                else "Relevant conditional rule could not be evaluated under the available policy/evidence."
            ),
            sources=sources,
            values={
                "rule_id": result.rule_id,
                "rule_number": result.rule_number,
                "parse_state": result.parse_state.value,
                "parse_reason_code": result.parse_reason_code,
                "parse_reason_offset": result.parse_reason_offset,
                "state": result.state.value,
                "truth_value": (
                    result.truth_value.value if result.truth_value else None
                ),
                "applied_assignments": [
                    {
                        "characteristic": item.characteristic,
                        "value": item.value,
                    }
                    for item in result.applied_assignments
                ],
                "missing_variables": list(result.missing_variables),
                "incomplete_variables": list(result.incomplete_variables),
                "activation_policy_version": result.activation_policy_version,
                "dependency_path": list(result.dependency_path),
            },
            related_ids=(result.rule_id,),
            qualifiers=result.warnings,
            discriminator=f"rule-result:{result.rule_id}",
        )
    _add_diagnostics_and_missing_inputs(
        collector, ecn, match_result, rule_results, context
    )
    if context.deterministic_logic_version == "ecn-evidence-v3" and context.analysis_as_of_date is not None:
        from app.ecn.evidence.material_changes import interpret_changes
        for change in interpret_changes(ecn):
            collector.add(section=AnalysisSectionKey.CHANGE_INTENT, kind=EvidenceKind.DETERMINISTIC_FINDING,
                tier=None, summary=f"{change['number']}: {change['action'].replace('_', ' ').lower()}.",
                sources=tuple(_json_source(ecn.artifact_id, path) for path in change["source_paths"]),
                values={key: change[key] for key in ("number", "action", "revision_from", "revision_to", "lifecycle_evidence")},
                qualifiers=change["qualifiers"], discriminator="lifecycle:" + change["number"])
    sections = collector.grouped()
    hash_input: dict[str, JsonValue] = {
        "ecn_row_id": ecn.ecn_row_id,
        "context": identifiers.json_safe(context),
        "sections": identifiers.json_safe(sections),
    }
    return EvidenceBundle(
        ecn.ecn_row_id,
        context,
        sections,
        identifiers.evidence_bundle_hash(hash_input),
    )


def derive_impact_counts(
    ecn: NormalizedEcn,
    match_result: MatchResult,
    evidence: EvidenceBundle,
) -> ImpactCounts:
    """Return list counts with confirmed/context/presence semantics separated.

    Args:
        ecn: One normalized ECN occurrence.
        match_result: Exact crosswalk used to create the evidence.
        evidence: Complete six-section evidence bundle used for warning counts.

    Returns:
        Stable list-page counts without double-counting occurrence paths.
    """
    changed = {
        item.technical_id or item.source_object_id
        for item in (*ecn.affected_objects, *ecn.resulting_objects)
        if item.is_technical_wtpart
    }
    raw = {occurrence_key(item) for item in match_result.raw_direct}
    confirmed = {
        occurrence_key(item.occurrence)
        for item in match_result.enriched_matches
        if item.path is MatchPath.RAW_ECN_COMPONENT
        and item.enriched_strength
        is EnrichedMatchStrength.COMPONENT_AND_ECN_CONFIRMED
    }
    all_enriched = {
        occurrence_key(item.occurrence) for item in match_result.enriched_matches
    }
    contextual = all_enriched - confirmed
    mbom = {
        occurrence_key(item.occurrence) for item in match_result.mbom_presence
    }
    all_items = (
        *evidence.sections.executive_summary,
        *evidence.sections.change_intent,
        *evidence.sections.affected_materials,
        *evidence.sections.ebom_mbom_impact,
        *evidence.sections.configuration_scope,
        *evidence.sections.uncertainties_and_missing_evidence,
    )
    warning_count = sum(
        item.kind in {EvidenceKind.WARNING, EvidenceKind.CONTRADICTION}
        for item in all_items
    )
    return ImpactCounts(
        affected_item_count=len(changed),
        raw_ecn_match_count=len(raw),
        ebom_match_count=len(confirmed),
        ebom_context_count=len(contextual),
        mbom_match_count=len(mbom),
        warning_count=warning_count,
    )
