"""Lazy public evidence contracts without persistence import cycles."""

from importlib import import_module
from typing import Any

from app.ecn.evidence import identifiers

_EXPORT_MODULES = {
    "BomIndexes": "matching",
    "BomOccurrence": "matching",
    "BusinessDirection": "business",
    "BusinessEvidencePage": "business",
    "BusinessEvidenceRecord": "business",
    "BusinessEvidenceSection": "business",
    "BusinessTechnicalDetails": "business",
    "DeterministicDiagnostic": "matching",
    "DocumentOwnerBridge": "matching",
    "EnrichedMatchStrength": "matching",
    "MatchPath": "matching",
    "MatchResult": "matching",
    "OccurrenceKey": "matching",
    "OccurrenceMatch": "matching",
    "PlantFootprintSignal": "builder",
    "RuleDependencyEdge": "builder",
    "RuleIndex": "builder",
    "StructuralObjectSeed": "matching",
    "StructuralSeedOrigin": "matching",
    "build_bom_indexes": "matching",
    "business_evidence_record_from_item": "business",
    "business_evidence_records": "business",
    "build_ecn_evidence": "builder",
    "build_rule_index": "builder",
    "derive_change_candidates": "change_candidates",
    "derive_impact_counts": "builder",
    "match_ecn_to_bom": "matching",
    "occurrence_key": "matching",
    "paginate_business_evidence": "business",
    "select_and_evaluate_relevant_rules": "builder",
}

for _identifier_name in (
    "candidate_id",
    "canonical_sha256",
    "evidence_bundle_hash",
    "json_safe",
    "make_evidence_id",
    "make_evidence_identity",
    "source_coordinate",
):
    _EXPORT_MODULES[_identifier_name] = "identifiers"

__all__ = ["identifiers", *_EXPORT_MODULES]


def __getattr__(name: str) -> Any:
    """Load a public evidence symbol from its defining module on first access."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"app.ecn.evidence.{module_name}"), name)
    globals()[name] = value
    return value
