"""Strict application validation for provider-generated ECN impact analyses."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from app.ecn.analysis_models import (
    AnalysisCompleteness,
    AnalysisResponse,
    ImpactFactorAssessment,
    ImpactFactors,
    ImpactSeverity,
    SectionStatus,
    derive_completeness,
    derive_overall_impact,
    validate_analysis_response,
)
from app.ecn.llm.prompts import PromptEvidenceProjection
from app.ecn.llm.provider import ProviderCallResult, ProviderCallStatus
from app.ecn.llm.two_pass import FinalEvidenceProjection
from app.ecn.models import AnalysisSectionKey, BusinessEvidenceSection


class AnalysisValidationError(ValueError):
    """Represent fixed-code schema, status, or projected-evidence validation failure."""


@dataclass(frozen=True)
class ValidatedAnalysis:
    """Return a validated response with its derived completeness and evidence hash.

    Args:
        response: Strict six-section, evidence-scoped analysis response.
        completeness: Aggregate status derived from the six sections.
        evidence_hash: Immutable digest of the evidence bundle used for generation.
    """

    response: AnalysisResponse
    completeness: AnalysisCompleteness
    evidence_hash: str


def _deterministic_readiness(
    projection: FinalEvidenceProjection,
) -> ImpactFactorAssessment:
    """Derive data readiness from resolved evidence rather than model output.

    Args:
        projection: Exact bounded business evidence supplied to the final call.

    Returns:
        A valid evidence-cited readiness assessment independent of the LLM.
    """
    issue_sections = {
        BusinessEvidenceSection.DATA_QUALITY,
        BusinessEvidenceSection.MISSING_EVIDENCE,
    }
    issue_ids = [
        citation
        for record in projection.records
        if record.section in issue_sections
        for citation in record.citation_ids
    ]
    fallback_ids = issue_ids or list(projection.header_citation_ids)
    if projection.snapshot_context:
        counts = projection.snapshot_context["counts"]
        questions = projection.snapshot_context.get("questions", [])
        missing = next((q for q in questions if q["code"] == "BOM_LOCATION_UNAVAILABLE"), {})
        names = [s.get("number") or "Unnumbered object" for s in missing.get("subjects", [])]
        locations = ", ".join(names[:5])
        remaining = missing.get("affected_count", len(names)) - min(5, len(names))
        if locations and remaining > 0:
            locations += f" and {remaining} more"
        return ImpactFactorAssessment(
            severity=ImpactSeverity.MEDIUM if fallback_ids else ImpactSeverity.UNKNOWN,
            assessment=(f'- {counts["located_objects"]} of {counts["technical_objects"]} referenced objects located in supplied BOMs.\n'
                        f'- {"Confirm BOM placement for " + locations if locations else "Review named follow-ups and validity before concluding impact"}.'
                        if projection.snapshot_context.get("version") == "ecn-snapshot-v2" else f'{counts["located_objects"]} of {counts["technical_objects"]} referenced technical objects '
                        f'are located in the supplied BOM; {counts["unlocated_objects"]} have no observed location. '
                        'The snapshot is available for review. Open questions describe what cannot yet be concluded.'),
            evidence_ids=list(dict.fromkeys(fallback_ids))[:10],
            missing_evidence=[q["question"] + (": " + ", ".join(s.get("number") or "Unnumbered object" for s in q.get("subjects", [])[:5])
                if q.get("subjects") else "") for q in questions]
                or (["Supply source evidence before assessing readiness."] if not fallback_ids else []),
        )
    if projection.omitted_record_count or any(
        record.section is BusinessEvidenceSection.MISSING_EVIDENCE
        for record in projection.records
    ):
        severity = ImpactSeverity.HIGH if fallback_ids else ImpactSeverity.UNKNOWN
        assessment = "Required evidence is missing or the bounded view is incomplete."
    elif any(
        record.section is BusinessEvidenceSection.DATA_QUALITY
        for record in projection.records
    ):
        severity = ImpactSeverity.MEDIUM if fallback_ids else ImpactSeverity.UNKNOWN
        assessment = "Source-quality issues may limit exact matching."
    elif projection.records or projection.header_citation_ids:
        severity = ImpactSeverity.LOW
        fallback_ids = fallback_ids or list(projection.records[0].citation_ids)
        assessment = "The resolved evidence is within the bounded analysis limits."
    else:
        severity = ImpactSeverity.UNKNOWN
        assessment = "No deterministic evidence was available to assess readiness."
    return ImpactFactorAssessment(
        severity=severity,
        assessment=assessment,
        evidence_ids=(
            fallback_ids[:10] if severity is not ImpactSeverity.UNKNOWN else []
        ),
        missing_evidence=(
            ["Resolve source-quality or missing-evidence gaps."]
            if severity is ImpactSeverity.UNKNOWN
            else []
        ),
    )


def _normalize_final_payload(
    top_level: dict[str, object],
    projection: FinalEvidenceProjection,
) -> str:
    """Replace model-authored deterministic fields before strict validation.

    Business factor assessments remain model output. Data readiness is replaced
    from the resolved evidence, and overall impact is then derived from the
    validated business factors according to the fixed severity rule.

    Args:
        top_level: Parsed provider JSON object.
        projection: Exact bounded evidence view used for the final call.

    Returns:
        JSON containing deterministic readiness and, when business factors are
        structurally valid, a deterministically derived overall impact.
    """
    raw_factors = top_level.get("impact_factors")
    if not isinstance(raw_factors, dict):
        return json.dumps(top_level, ensure_ascii=False)
    normalized_factors = dict(raw_factors)
    if projection.snapshot_context.get("version") == "ecn-snapshot-v2":
        # A plain paragraph can be presented as one bullet without removing any
        # qualification. Never truncate content to satisfy presentation limits.
        for name in ("manufacturing", "procurement", "configuration"):
            factor = normalized_factors.get(name)
            if isinstance(factor, dict) and isinstance(factor.get("assessment"), str):
                passage = factor["assessment"].strip()
                if passage and not any(line.lstrip().startswith(("- ", "* ")) for line in passage.splitlines()):
                    normalized_factors[name] = {**factor, "assessment": "- " + " ".join(passage.split())}
    normalized_factors["data_readiness"] = _deterministic_readiness(
        projection
    ).model_dump(mode="json")
    top_level["impact_factors"] = normalized_factors
    try:
        # Strict Pydantic models intentionally distinguish Python strings from
        # enum instances, while strict JSON validation accepts the provider's
        # serialized enum values. Preserve that JSON boundary here.
        factors = ImpactFactors.model_validate_json(
            json.dumps(normalized_factors, ensure_ascii=False)
        )
    except ValidationError:
        return json.dumps(top_level, ensure_ascii=False)
    top_level["overall_impact"] = derive_overall_impact(factors).value
    return json.dumps(top_level, ensure_ascii=False)


def parse_and_validate_analysis(
    provider_result: ProviderCallResult,
    projection: PromptEvidenceProjection | FinalEvidenceProjection,
    evidence_hash: str,
) -> ValidatedAnalysis:
    """Validate response completion, six sections, gaps, and projected IDs.

    Args:
        provider_result: Already-persistable outcome from one actual provider call.
        projection: Exact bounded evidence projection sent to that call.
        evidence_hash: Full deterministic evidence digest for immutable persistence.

    Returns:
        The parsed response, derived completeness, and verified evidence hash.

    Raises:
        AnalysisValidationError: For provider errors, incomplete/refused/missing
            output, invalid schema/citations, required-gap violations, or bad hash.
    """
    if provider_result.status is ProviderCallStatus.ERROR:
        raise AnalysisValidationError("provider call did not return a response")
    if provider_result.response_status != "completed":
        raise AnalysisValidationError("provider response is incomplete")
    if provider_result.refusal_present:
        raise AnalysisValidationError("provider response contains a refusal")
    if provider_result.raw_json is None:
        raise AnalysisValidationError("provider response is missing output")
    raw_json = provider_result.raw_json
    if isinstance(projection, FinalEvidenceProjection):
        try:
            top_level = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise AnalysisValidationError("structured output is not valid JSON") from exc
        if not isinstance(top_level, dict) or not {
            "impact_factors",
            "overall_impact",
        }.issubset(top_level):
            raise AnalysisValidationError(
                "current analysis output is missing severity factors"
            )
        raw_json = _normalize_final_payload(top_level, projection)
    if len(evidence_hash) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_hash
    ):
        raise AnalysisValidationError("evidence hash is invalid")
    allowed = {
        key: set(projection.allowed_evidence_ids_by_section.get(key, frozenset()))
        for key in AnalysisSectionKey
    }
    try:
        response = validate_analysis_response(raw_json, allowed)
    except ValidationError as exc:
        if "cross-section" in str(exc):
            raise AnalysisValidationError(
                "structured output contains a cross-section evidence citation"
            ) from None
        raise AnalysisValidationError(
            "structured output schema validation failed"
        ) from None
    for key in projection.sections_requiring_gap:
        section = getattr(response.sections, key.value)
        if section.status is not SectionStatus.INSUFFICIENT_EVIDENCE:
            raise AnalysisValidationError(
                f"section {key.value} must use INSUFFICIENT_EVIDENCE"
            )
    if isinstance(projection, FinalEvidenceProjection) and projection.snapshot_context.get("version") == "ecn-snapshot-v2":
        passages = [(response.sections.executive_summary.assessment_markdown, 5)]
        passages.extend((getattr(response.impact_factors, name).assessment, 3)
                        for name in ("manufacturing", "procurement", "configuration", "data_readiness"))
        for passage, maximum in passages:
            lines = [line.strip() for line in passage.splitlines() if line.strip()]
            if not lines or len(lines) > maximum or any(not line.startswith(("- ", "* ")) for line in lines):
                raise AnalysisValidationError("Executive summary and factors must use bounded Markdown bullets")
    if isinstance(projection, FinalEvidenceProjection):
        permitted = set(projection.header_citation_ids)
        permitted.update(
            citation
            for record in projection.records
            for citation in record.citation_ids
        )
        permitted.update(fact.evidence_id for fact in projection.discovered_facts)
        for factor_name in ("manufacturing", "procurement", "configuration"):
            factor = getattr(response.impact_factors, factor_name)
            if set(factor.evidence_ids) - permitted:
                raise AnalysisValidationError(
                    f"{factor_name} factor contains an unknown evidence citation"
                )
    return ValidatedAnalysis(
        response=response,
        completeness=derive_completeness(response),
        evidence_hash=evidence_hash,
    )
