"""Tests for strict six-section provider response validation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from app.ecn.analysis_models import AnalysisCompleteness, SectionStatus
from app.ecn.llm.prompts import build_impact_instructions, project_evidence_for_prompt
from app.ecn.llm.provider import ProviderCallResult, ProviderCallStatus
from app.ecn.llm.validation import AnalysisValidationError, parse_and_validate_analysis
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
)
from app.observability import TokenUsage

from tests.fixtures.builders import (
    make_evidence_bundle,
    make_evidence_bundle_sections,
    synthetic_uuid,
)


def _evidence(key: AnalysisSectionKey, index: int) -> EvidenceItem:
    """Return one safe evidence item for a specific analysis section."""
    return EvidenceItem(
        evidence_id=f"E-{index:016x}",
        section_key=key,
        kind=EvidenceKind.SOURCE_FACT,
        tier=None,
        summary=f"Evidence for {key.value}",
        sources=(
            EvidenceSource(
                artifact_id=synthetic_uuid("validation-artifact"),
                artifact_kind=ArtifactKind.ECN_JSON,
                json_path=f"$.{key.value}",
                sheet_name=None,
                bom_row_id=None,
                source_row_number=None,
            ),
        ),
        values={"value": key.value},
        related_ids=(),
        quality_qualifiers=(),
    )


@pytest.fixture
def projection():
    """Return a complete projection with one allowed ID per section."""
    items = {
        key: _evidence(key, index) for index, key in enumerate(AnalysisSectionKey, 1)
    }
    sections = make_evidence_bundle_sections(
        **{key.value: (items[key],) for key in AnalysisSectionKey}
    )
    return project_evidence_for_prompt(
        make_evidence_bundle(sections=sections), 20, 20_000
    )


def _payload(projection, *, status: str = "SUFFICIENT") -> dict[str, object]:
    """Build one literal six-section provider payload using projected evidence IDs."""
    sections: dict[str, object] = {}
    for key in AnalysisSectionKey:
        evidence_id = next(iter(projection.allowed_evidence_ids_by_section[key]))
        sections[key.value] = {
            "status": status,
            "assessment_markdown": f"Grounded assessment for {key.value}.",
            "evidence_ids": [evidence_id],
            "missing_evidence": [],
        }
    return {"sections": sections}


def _result(
    payload: dict[str, object] | None, **overrides: object
) -> ProviderCallResult:
    """Return a received provider result for validation tests."""
    base: dict[str, object] = {
        "status": ProviderCallStatus.RECEIVED,
        "raw_json": None if payload is None else json.dumps(payload),
        "response_status": "completed",
        "refusal_present": False,
        "usage": TokenUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            input_total_tokens=10,
            usage_available=True,
        ),
        "model_name": "gpt-5.4",
        "model_deployment": None,
        "provider_request_id": "response-1",
        "latency_ms": 42,
        "error_code": None,
    }
    base.update(overrides)
    return ProviderCallResult(**base)  # type: ignore[arg-type]


def test_validation_accepts_all_six_grounded_sections(projection) -> None:
    """Return a validated immutable result when every section cites its own evidence."""
    validated = parse_and_validate_analysis(
        _result(_payload(projection)), projection, evidence_hash="a" * 64
    )
    assert validated.completeness is AnalysisCompleteness.COMPLETED
    assert validated.evidence_hash == "a" * 64


def test_validation_rejects_cross_section_evidence(projection) -> None:
    """A valid bundle ID is still invalid when cited from the wrong section."""
    payload = _payload(projection)
    sections = payload["sections"]
    source_ids = sections["affected_materials"]["evidence_ids"]
    source_id = source_ids[0]
    sections["configuration_scope"]["evidence_ids"].append(source_id)
    invalid_result = _result(payload)
    with pytest.raises(AnalysisValidationError, match="cross-section"):
        parse_and_validate_analysis(invalid_result, projection, evidence_hash="a" * 64)


def test_projection_forces_insufficient_status(projection) -> None:
    """A section with omitted evidence cannot claim full sufficiency."""
    truncated = replace(
        projection,
        sections_requiring_gap=frozenset({AnalysisSectionKey.CONFIGURATION_SCOPE}),
    )
    with pytest.raises(AnalysisValidationError, match="INSUFFICIENT_EVIDENCE"):
        parse_and_validate_analysis(
            _result(_payload(projection)), truncated, evidence_hash="a" * 64
        )


def test_insufficient_section_derives_completed_with_gaps(projection) -> None:
    """Accept a visible gap and derive the aggregate completeness state."""
    payload = _payload(projection)
    payload["sections"]["configuration_scope"] = {
        "status": "INSUFFICIENT_EVIDENCE",
        "assessment_markdown": "The available configuration facts are incomplete.",
        "evidence_ids": [],
        "missing_evidence": ["Active configuration facts are unavailable."],
    }
    validated = parse_and_validate_analysis(
        _result(payload), projection, evidence_hash="b" * 64
    )
    assert validated.completeness is AnalysisCompleteness.COMPLETED_WITH_GAPS


def test_not_applicable_is_visible_and_grounded(projection) -> None:
    """Accept a grounded NOT_APPLICABLE section without turning it into a gap."""
    payload = _payload(projection)
    evidence_id = next(
        iter(
            projection.allowed_evidence_ids_by_section[
                AnalysisSectionKey.EBOM_MBOM_IMPACT
            ]
        )
    )
    payload["sections"]["ebom_mbom_impact"] = {
        "status": "NOT_APPLICABLE",
        "assessment_markdown": "The evidence establishes why this section is not applicable.",
        "evidence_ids": [evidence_id],
        "missing_evidence": [],
    }
    validated = parse_and_validate_analysis(
        _result(payload), projection, evidence_hash="c" * 64
    )
    assert (
        validated.response.sections.ebom_mbom_impact.status
        is SectionStatus.NOT_APPLICABLE
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"response_status": "incomplete"}, "incomplete"),
        ({"refusal_present": True}, "refusal"),
        ({"raw_json": None}, "missing output"),
        (
            {
                "status": ProviderCallStatus.ERROR,
                "error_code": "PROVIDER_RESPONSE_INVALID",
                "raw_json": None,
            },
            "provider",
        ),
    ],
)
def test_validation_rejects_nonvalidatable_provider_results(
    projection, overrides: dict[str, object], message: str
) -> None:
    """Reject provider errors, incomplete responses, refusals, and missing output."""
    result = _result(_payload(projection), **overrides)
    with pytest.raises(AnalysisValidationError, match=message):
        parse_and_validate_analysis(result, projection, evidence_hash="d" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra="forbidden"),
        lambda payload: payload["sections"].pop("change_intent"),
        lambda payload: payload["sections"].update(extra_section={}),
    ],
)
def test_validation_rejects_extra_or_missing_schema_fields(
    projection, mutation
) -> None:
    """Enforce the exact top-level and six-section structured-output schema."""
    payload = _payload(projection)
    mutation(payload)
    with pytest.raises(AnalysisValidationError, match="schema"):
        parse_and_validate_analysis(
            _result(payload), projection, evidence_hash="e" * 64
        )


def test_validation_rejects_malformed_json(projection) -> None:
    """Return a fixed schema error without exposing malformed provider content."""
    result = replace(_result(_payload(projection)), raw_json="{private payload")
    with pytest.raises(AnalysisValidationError, match="schema") as error:
        parse_and_validate_analysis(result, projection, evidence_hash="f" * 64)
    assert "private payload" not in str(error.value)


def test_instructions_prohibit_unsupported_inferences() -> None:
    """State every evidence boundary that prevents plausible but unsafe conclusions."""
    instructions = build_impact_instructions()
    for phrase in (
        "revision does not prove redesign",
        "co-occurrence does not prove replacement",
        "one-sided lists do not prove add or remove",
        "document nesting does not prove owner-material impact",
        "no BOM match does not prove no impact",
        "unknown rule facts do not prove applicability",
        "descriptions do not prove Make/Buy",
        "Plants in plant_impact come from the brands",
        "not a confirmed configuration or take rate",
    ):
        assert phrase in instructions
