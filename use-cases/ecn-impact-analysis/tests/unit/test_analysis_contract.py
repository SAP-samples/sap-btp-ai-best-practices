"""Unit tests for the six-section strict analysis contract.

Tests cover:
- Status-specific invariants (SUFFICIENT, NOT_APPLICABLE, INSUFFICIENT_EVIDENCE).
- Uniqueness and nonblank enforcement on evidence_ids and missing_evidence.
- Extra/missing fields rejected by strict=True + extra="forbid".
- Cross-section and unknown evidence ID rejection by validate_analysis_response.
- derive_completeness producing COMPLETED vs COMPLETED_WITH_GAPS.

Run these tests in isolation with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_analysis_contract.py
"""

import json

import pytest
from pydantic import ValidationError

from app.ecn.analysis_models import (
    AnalysisCompleteness,
    AnalysisResponse,
    AnalysisSection,
    SectionStatus,
    ImpactFactors,
    ImpactFactorAssessment,
    ImpactSeverity,
    derive_completeness,
    validate_analysis_response,
)
from app.ecn.models import AnalysisSectionKey
from tests.fixtures.builders import analysis_payload


# ---------------------------------------------------------------------------
# Cross-section evidence ID rejection
# ---------------------------------------------------------------------------


def test_validation_rejects_cross_section_evidence_id() -> None:
    """validate_analysis_response must reject an ID from the wrong section."""
    payload = analysis_payload()
    payload["sections"]["change_intent"]["evidence_ids"] = ["EXEC-1"]
    allowed = {key: {f"{key.value.upper()}-1"} for key in AnalysisSectionKey}
    with pytest.raises(ValidationError, match="change_intent"):
        validate_analysis_response(json.dumps(payload), allowed)


# ---------------------------------------------------------------------------
# NOT_APPLICABLE completeness projection
# ---------------------------------------------------------------------------


def test_not_applicable_alone_does_not_create_a_gap() -> None:
    """NOT_APPLICABLE sections must not cause COMPLETED_WITH_GAPS."""
    payload = analysis_payload(status="NOT_APPLICABLE")
    response = AnalysisResponse.model_validate(payload)
    assert derive_completeness(response) is AnalysisCompleteness.COMPLETED


def test_overall_impact_is_derived_only_from_business_factors() -> None:
    """Ignore data-readiness severity when deriving the business impact level."""
    payload = analysis_payload()
    payload["impact_factors"] = ImpactFactors(
        manufacturing=ImpactFactorAssessment(
            severity=ImpactSeverity.MEDIUM,
            assessment="Manufacturing paths require coordinated updates.",
            evidence_ids=["EBOM_MBOM_IMPACT-1"],
            missing_evidence=[],
        ),
        procurement=ImpactFactorAssessment(
            severity=ImpactSeverity.LOW,
            assessment="No supported procurement escalation was found.",
            evidence_ids=["AFFECTED_MATERIALS-1"],
            missing_evidence=[],
        ),
        configuration=ImpactFactorAssessment(
            severity=ImpactSeverity.LOW,
            assessment="Configuration scope is bounded.",
            evidence_ids=["CONFIGURATION_SCOPE-1"],
            missing_evidence=[],
        ),
        data_readiness=ImpactFactorAssessment(
            severity=ImpactSeverity.HIGH,
            assessment="Many source records require review.",
            evidence_ids=["UNCERTAINTIES_AND_MISSING_EVIDENCE-1"],
            missing_evidence=[],
        ),
    )
    payload["overall_impact"] = ImpactSeverity.MEDIUM
    response = AnalysisResponse.model_validate(payload)
    assert response.overall_impact is ImpactSeverity.MEDIUM

    payload["overall_impact"] = ImpactSeverity.HIGH
    with pytest.raises(ValidationError, match="overall_impact"):
        AnalysisResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# INSUFFICIENT_EVIDENCE requires non-empty missing_evidence
# ---------------------------------------------------------------------------


def test_insufficient_section_requires_specific_missing_evidence() -> None:
    """INSUFFICIENT_EVIDENCE with an empty missing_evidence list must be rejected.

    Note: ``AnalysisSection`` uses ``strict=True``, so ``status`` must be a
    ``SectionStatus`` enum instance when using ``model_validate``.  Plain
    strings are only accepted via ``model_validate_json`` (the LLM path).
    """
    payload = analysis_payload()
    section = payload["sections"]["configuration_scope"]
    section.update(
        status=SectionStatus.INSUFFICIENT_EVIDENCE,
        evidence_ids=[],
        missing_evidence=[],
    )
    with pytest.raises(ValidationError, match="missing_evidence"):
        AnalysisResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# COMPLETED_WITH_GAPS: at least one INSUFFICIENT_EVIDENCE section
# ---------------------------------------------------------------------------


def test_at_least_one_insufficient_section_produces_completed_with_gaps() -> None:
    """derive_completeness returns COMPLETED_WITH_GAPS when any section is insufficient."""
    payload = analysis_payload()
    payload["sections"]["ebom_mbom_impact"].update(
        status=SectionStatus.INSUFFICIENT_EVIDENCE,
        evidence_ids=[],
        missing_evidence=["BOM impact data is not available"],
    )
    response = AnalysisResponse.model_validate(payload)
    assert derive_completeness(response) is AnalysisCompleteness.COMPLETED_WITH_GAPS


# ---------------------------------------------------------------------------
# SUFFICIENT / NOT_APPLICABLE must have evidence_ids and no missing_evidence
# ---------------------------------------------------------------------------


def test_sufficient_section_must_have_evidence_ids() -> None:
    """SUFFICIENT with empty evidence_ids must be rejected."""
    with pytest.raises(ValidationError, match="evidence_ids"):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.SUFFICIENT,
                "assessment_markdown": "ok",
                "evidence_ids": [],
                "missing_evidence": [],
            }
        )


def test_sufficient_section_must_not_have_missing_evidence() -> None:
    """SUFFICIENT with non-empty missing_evidence must be rejected."""
    with pytest.raises(ValidationError):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.SUFFICIENT,
                "assessment_markdown": "ok",
                "evidence_ids": ["E-1"],
                "missing_evidence": ["something missing"],
            }
        )


def test_not_applicable_must_not_have_missing_evidence() -> None:
    """NOT_APPLICABLE with non-empty missing_evidence must be rejected."""
    with pytest.raises(ValidationError):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.NOT_APPLICABLE,
                "assessment_markdown": "not relevant",
                "evidence_ids": ["E-1"],
                "missing_evidence": ["not empty"],
            }
        )


# ---------------------------------------------------------------------------
# Uniqueness enforcement
# ---------------------------------------------------------------------------


def test_duplicate_evidence_ids_are_rejected() -> None:
    """Duplicate entries in evidence_ids must be rejected."""
    with pytest.raises(ValidationError, match="evidence_ids"):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.SUFFICIENT,
                "assessment_markdown": "assessment",
                "evidence_ids": ["E-1", "E-1"],
                "missing_evidence": [],
            }
        )


def test_duplicate_missing_evidence_entries_are_rejected() -> None:
    """Duplicate entries in missing_evidence must be rejected."""
    with pytest.raises(ValidationError, match="missing_evidence"):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.INSUFFICIENT_EVIDENCE,
                "assessment_markdown": "assessment",
                "evidence_ids": [],
                "missing_evidence": ["same", "same"],
            }
        )


# ---------------------------------------------------------------------------
# Extra and missing field enforcement (strict=True + extra="forbid")
# ---------------------------------------------------------------------------


def test_extra_field_in_analysis_section_is_rejected() -> None:
    """extra='forbid' must reject any unexpected field in AnalysisSection."""
    with pytest.raises(ValidationError):
        AnalysisSection.model_validate(
            {
                "status": SectionStatus.SUFFICIENT,
                "assessment_markdown": "ok",
                "evidence_ids": ["E-1"],
                "missing_evidence": [],
                "unexpected_field": "surprise",
            }
        )


def test_extra_section_in_analysis_response_is_rejected() -> None:
    """extra='forbid' must reject any unexpected section key in AnalysisResponse."""
    payload = analysis_payload()
    payload["sections"]["extra_section"] = {
        "status": "SUFFICIENT",
        "assessment_markdown": "extra",
        "evidence_ids": ["E-1"],
        "missing_evidence": [],
    }
    with pytest.raises(ValidationError):
        AnalysisResponse.model_validate(payload)


def test_change_brief_must_be_one_short_line() -> None:
    """The stored decision headline cannot become another multi-line source description."""
    payload = analysis_payload()
    payload["change_brief"] = "The flexible shim replaces the rigid shim."
    assert AnalysisResponse.model_validate(payload).change_brief == payload["change_brief"]

    payload["change_brief"] = "First line.\nSecond line."
    with pytest.raises(ValidationError, match="change_brief"):
        AnalysisResponse.model_validate(payload)


def test_missing_section_is_rejected() -> None:
    """A payload missing any of the six required sections must be rejected."""
    payload = analysis_payload()
    del payload["sections"]["change_intent"]
    with pytest.raises(ValidationError):
        AnalysisResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# Unknown evidence IDs (not in any bundle section)
# ---------------------------------------------------------------------------


def test_validation_rejects_unknown_evidence_id() -> None:
    """validate_analysis_response must reject an ID not in any allowed set."""
    payload = analysis_payload()
    payload["sections"]["executive_summary"]["evidence_ids"] = ["TOTALLY-UNKNOWN-99"]
    allowed = {key: {f"{key.value.upper()}-1"} for key in AnalysisSectionKey}
    with pytest.raises(ValidationError, match="executive_summary"):
        validate_analysis_response(json.dumps(payload), allowed)


def test_validation_accepts_correct_evidence_ids() -> None:
    """validate_analysis_response must succeed when all evidence IDs match their section."""
    payload = analysis_payload()
    # Each section's evidence_ids matches key.value.upper() + "-1"
    allowed = {key: {f"{key.value.upper()}-1"} for key in AnalysisSectionKey}
    response = validate_analysis_response(json.dumps(payload), allowed)
    assert isinstance(response, AnalysisResponse)
