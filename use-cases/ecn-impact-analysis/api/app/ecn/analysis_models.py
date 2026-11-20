"""Strict Pydantic models for the six-section LLM analysis output.

This module defines the contracts that govern what the analysis agent may
return and what the repository layer may store.  All section/response models
use ``strict=True`` so Pydantic does not silently coerce types across the
LLM/API boundary.

Imports from ``app.ecn.models`` (Task 1) are the single source of truth for
shared enumerations and type aliases.  Nothing in this file redeclares a type
from Task 1.

Typical usage::

    from app.ecn.analysis_models import (
        AnalysisResponse,
        derive_completeness,
        validate_analysis_response,
    )

    response = validate_analysis_response(llm_json_output, allowed_ids_by_section)
    completeness = derive_completeness(response)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from app.ecn.models import AnalysisSectionKey


# ---------------------------------------------------------------------------
# Status and completeness enumerations
# ---------------------------------------------------------------------------


class SectionStatus(StrEnum):
    """Classify whether a section's evidence is sufficient, insufficient, or not applicable.

    These values drive which invariants are enforced by ``AnalysisSection``'s
    model validator:

    - ``SUFFICIENT``: evidence supports the assessment; evidence_ids required,
      missing_evidence must be empty.
    - ``INSUFFICIENT_EVIDENCE``: the assessment is incomplete; missing_evidence
      is required, evidence_ids are optional.
    - ``NOT_APPLICABLE``: the section is irrelevant for this ECN; evidence_ids
      required (to explain why), missing_evidence must be empty.
    """

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AnalysisCompleteness(StrEnum):
    """Classify the overall completeness of a six-section analysis response.

    ``COMPLETED`` means every section is either SUFFICIENT or NOT_APPLICABLE.
    ``COMPLETED_WITH_GAPS`` means at least one section is INSUFFICIENT_EVIDENCE.

    This is an API/result projection derived from the six section statuses.
    The mutable HANA job table records only ``JobState.COMPLETED``; this enum
    is computed when reading analysis results and is never stored as a job state.
    """

    COMPLETED = "COMPLETED"
    COMPLETED_WITH_GAPS = "COMPLETED_WITH_GAPS"


class ImpactSeverity(StrEnum):
    """Classify supported business impact and deterministic data readiness."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Section model
# ---------------------------------------------------------------------------


class AnalysisSection(BaseModel):
    """Store one evidence-scoped Markdown assessment for a single analysis section.

    Status invariants are enforced by the model-level validator and cannot be
    disabled by Python optimizations (no bare ``assert``):

    - ``SUFFICIENT`` and ``NOT_APPLICABLE``: ``evidence_ids`` must be non-empty;
      ``missing_evidence`` must be empty.
    - ``INSUFFICIENT_EVIDENCE``: ``missing_evidence`` must be non-empty;
      ``evidence_ids`` are optional.

    All list entries must be externally trimmed, nonblank, and unique.

    Args:
        status: The section-level sufficiency verdict.
        assessment_markdown: Human-readable Markdown assessment text (non-empty
            after stripping).
        evidence_ids: IDs of evidence items that support this section.
            Must be unique; entries are trimmed to nonblank.
        missing_evidence: Descriptions of data needed to complete the
            assessment.  Must be unique and non-empty for INSUFFICIENT_EVIDENCE;
            must be empty for SUFFICIENT and NOT_APPLICABLE.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    status: SectionStatus
    assessment_markdown: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    evidence_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ]
    missing_evidence: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ]

    @model_validator(mode="after")
    def validate_status_contract(self) -> "AnalysisSection":
        """Enforce status-specific evidence and missing-input invariants.

        Checks performed in order:
        1. ``evidence_ids`` entries are unique.
        2. ``missing_evidence`` entries are unique.
        3. SUFFICIENT/NOT_APPLICABLE require at least one evidence_id and
           forbid any missing_evidence entries.
        4. INSUFFICIENT_EVIDENCE requires at least one missing_evidence entry.

        Returns:
            The validated ``AnalysisSection`` instance (self).

        Raises:
            ValueError: If any invariant is violated.
        """
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("evidence_ids must be unique")
        if len(set(self.missing_evidence)) != len(self.missing_evidence):
            raise ValueError("missing_evidence must be unique")
        if self.status in {SectionStatus.SUFFICIENT, SectionStatus.NOT_APPLICABLE}:
            if not self.evidence_ids or self.missing_evidence:
                raise ValueError(
                    "status requires evidence_ids and forbids missing_evidence"
                )
        if self.status is SectionStatus.INSUFFICIENT_EVIDENCE and not self.missing_evidence:
            raise ValueError("insufficient evidence requires missing_evidence")
        return self


# ---------------------------------------------------------------------------
# Six-section container and top-level response
# ---------------------------------------------------------------------------


class AnalysisSections(BaseModel):
    """Aggregate all six mandatory analysis sections in field order.

    Field names match ``AnalysisSectionKey`` string values so that
    ``getattr(sections, key.value)`` retrieves the correct section.
    The model is strict (``strict=True``) and forbids extra keys.

    Args:
        executive_summary: High-level summary section.
        change_intent: Change-intent and rationale section.
        affected_materials: Affected-materials section.
        ebom_mbom_impact: EBOM/MBOM impact section.
        configuration_scope: Configuration-scope section.
        uncertainties_and_missing_evidence: Uncertainties and gaps section.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    executive_summary: AnalysisSection
    change_intent: AnalysisSection
    affected_materials: AnalysisSection
    ebom_mbom_impact: AnalysisSection
    configuration_scope: AnalysisSection
    uncertainties_and_missing_evidence: AnalysisSection


class ImpactFactorAssessment(BaseModel):
    """Store one evidence-cited severity factor or an explicit unknown result.

    Args:
        severity: High, medium, low, or unknown classification.
        assessment: Concise business-readable rationale.
        evidence_ids: Supporting deterministic citations for known severities.
        missing_evidence: Required evidence when severity is unknown.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    severity: ImpactSeverity
    assessment: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
    ]
    evidence_ids: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ]
    missing_evidence: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ]

    @model_validator(mode="after")
    def validate_factor_evidence(self) -> "ImpactFactorAssessment":
        """Require citations for known severity and gaps for unknown severity."""
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("factor evidence_ids must be unique")
        if len(set(self.missing_evidence)) != len(self.missing_evidence):
            raise ValueError("factor missing_evidence must be unique")
        if self.severity is ImpactSeverity.UNKNOWN:
            if not self.missing_evidence:
                raise ValueError("unknown factor requires missing_evidence")
        elif not self.evidence_ids:
            raise ValueError(
                "known factor severity requires evidence_ids"
            )
        return self


def _legacy_unknown_factor() -> ImpactFactorAssessment:
    """Return a compatibility value for analyses stored before factor scoring."""
    return ImpactFactorAssessment(
        severity=ImpactSeverity.UNKNOWN,
        assessment="This factor was not assessed by the legacy analysis schema.",
        evidence_ids=[],
        missing_evidence=["Run the current two-pass analysis workflow."],
    )


class ImpactFactors(BaseModel):
    """Aggregate business impact factors and separate deterministic readiness."""

    model_config = ConfigDict(extra="forbid", strict=True)

    manufacturing: ImpactFactorAssessment = Field(default_factory=_legacy_unknown_factor)
    procurement: ImpactFactorAssessment = Field(default_factory=_legacy_unknown_factor)
    configuration: ImpactFactorAssessment = Field(default_factory=_legacy_unknown_factor)
    data_readiness: ImpactFactorAssessment = Field(default_factory=_legacy_unknown_factor)


def derive_overall_impact(factors: ImpactFactors) -> ImpactSeverity:
    """Derive overall impact from business factors, excluding data readiness.

    Args:
        factors: Manufacturing, procurement, configuration, and readiness results.

    Returns:
        Fixed high/medium/low/unknown result from the three business factors.
    """
    business = (
        factors.manufacturing.severity,
        factors.procurement.severity,
        factors.configuration.severity,
    )
    if ImpactSeverity.HIGH in business:
        return ImpactSeverity.HIGH
    if ImpactSeverity.MEDIUM in business:
        return ImpactSeverity.MEDIUM
    if ImpactSeverity.LOW in business:
        return ImpactSeverity.LOW
    if all(value is ImpactSeverity.NOT_APPLICABLE for value in business):
        return ImpactSeverity.NOT_APPLICABLE
    return ImpactSeverity.UNKNOWN


class DataReadinessDescription(BaseModel):
    """Explain one deterministic issue; code identifies the input and description is AI prose."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ImpactNextSteps(BaseModel):
    """Keep complementary next actions attributed to their impact perspective."""

    model_config = ConfigDict(extra="forbid", strict=True)

    manufacturing: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(default_factory=list)
    procurement: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(default_factory=list)
    configuration: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Hold the complete six-section LLM analysis output.

    This is the top-level model that is parsed from the raw LLM JSON output
    and stored verbatim in the persistence layer.  The model is strict and
    forbids extra keys.

    Args:
        change_brief: One AI-generated decision headline; ``None`` for legacy analyses.
        sections: All six mandatory analysis sections.
        impact_factors: Manufacturing, procurement, configuration, and readiness.
        overall_impact: Fixed business severity derived without data readiness.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    change_brief: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240, pattern=r"^[^\r\n]+$")
    ] | None = None
    sections: AnalysisSections
    impact_factors: ImpactFactors = Field(default_factory=ImpactFactors)
    overall_impact: ImpactSeverity = ImpactSeverity.UNKNOWN
    data_readiness_descriptions: list[DataReadinessDescription] = Field(default_factory=list)
    additional_next_steps: ImpactNextSteps = Field(default_factory=ImpactNextSteps)

    @model_validator(mode="after")
    def validate_overall_impact(self, info) -> "AnalysisResponse":
        """Reject model output that contradicts the deterministic severity rule."""
        expected = derive_overall_impact(self.impact_factors)
        if self.overall_impact is not expected and not (info.context or {}).get("historical"):
            raise ValueError("overall_impact conflicts with business factor severities")
        return self


# ---------------------------------------------------------------------------
# Cross-section evidence validation
# ---------------------------------------------------------------------------


def validate_analysis_response(
    raw_json: str,
    allowed_evidence_ids_by_section: Mapping[AnalysisSectionKey, set[str]],
) -> AnalysisResponse:
    """Parse strict JSON and reject unknown or cross-section evidence references.

    After structural Pydantic validation, every evidence ID cited in each
    section is checked against that section's allowed set.  An ID that is
    unknown to the ECN evidence bundle, belongs to a different section, or
    belongs to a different ECN bundle entirely is rejected with a
    ``ValidationError`` whose location path identifies the offending section.

    Args:
        raw_json: Raw JSON string emitted by the LLM.
        allowed_evidence_ids_by_section: Mapping from each ``AnalysisSectionKey``
            to the set of evidence IDs that are valid for that section in the
            current ECN bundle.  An ID valid for one section is invalid if cited
            from another section.

    Returns:
        A structurally valid ``AnalysisResponse`` with all cross-section
        evidence checks passing.

    Raises:
        ValidationError: If the JSON is structurally invalid, or if any
            evidence ID in a section is not in that section's allowed set.
    """
    response = AnalysisResponse.model_validate_json(raw_json)
    for key in AnalysisSectionKey:
        section = getattr(response.sections, key.value)
        invalid = set(section.evidence_ids) - allowed_evidence_ids_by_section.get(key, set())
        if invalid:
            raise ValidationError.from_exception_data(
                "AnalysisResponse",
                [
                    {
                        "type": "value_error",
                        "loc": ("sections", key.value, "evidence_ids"),
                        "input": section.evidence_ids,
                        "ctx": {
                            "error": ValueError("unknown or cross-section evidence ID")
                        },
                    }
                ],
            )
    return response


# ---------------------------------------------------------------------------
# Completeness derivation
# ---------------------------------------------------------------------------


def derive_completeness(response: AnalysisResponse) -> AnalysisCompleteness:
    """Derive the overall completeness classification from the six section statuses.

    Iterates each section in ``AnalysisSectionKey`` order.  Returns
    ``COMPLETED_WITH_GAPS`` immediately upon finding the first
    ``INSUFFICIENT_EVIDENCE`` section.  A section with ``NOT_APPLICABLE``
    status alone does not create a gap.

    Args:
        response: A validated ``AnalysisResponse`` (all six sections present).

    Returns:
        ``AnalysisCompleteness.COMPLETED_WITH_GAPS`` if at least one section
        has ``INSUFFICIENT_EVIDENCE`` status; ``AnalysisCompleteness.COMPLETED``
        otherwise.
    """
    for key in AnalysisSectionKey:
        section = getattr(response.sections, key.value)
        if section.status is SectionStatus.INSUFFICIENT_EVIDENCE:
            return AnalysisCompleteness.COMPLETED_WITH_GAPS
    return AnalysisCompleteness.COMPLETED
