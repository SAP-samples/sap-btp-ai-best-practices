"""One explicitly opt-in and potentially billable provider request.

Run only after accepting one model call:
    RUN_LLM_INTEGRATION_TESTS=1 PYTHONPATH=api python -m pytest -q -m llm_live

This test does not require HANA and does not compute monetary cost.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from app.ecn.llm.prompts import project_evidence_for_prompt
from app.ecn.llm.provider import ProviderCallStatus, create_impact_provider
from app.ecn.llm.validation import parse_and_validate_analysis
from app.ecn.models import OUTPUT_SCHEMA_VERSION, PROMPT_VERSION
from app.ecn.settings import load_ecn_settings
from app.observability import emit_llm_usage_event
from app.persistence.records import PinnedGenerationConfig
from tests.fixtures.builders import minimal_evidence_bundle


pytestmark = pytest.mark.llm_live


def test_one_live_strict_response_exposes_usage_and_logger_plumbing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate one live response plus complete usage logger plumbing."""
    if os.getenv("RUN_LLM_INTEGRATION_TESTS") != "1":
        pytest.skip("set RUN_LLM_INTEGRATION_TESTS=1 to accept one provider call")
    settings = load_ecn_settings()
    provider = create_impact_provider(settings)

    async def run_once() -> tuple[object, object]:
        """Make exactly one provider request and always close its client."""
        try:
            bundle = minimal_evidence_bundle()
            projection = project_evidence_for_prompt(
                bundle,
                settings.max_prompt_evidence_items,
                settings.max_prompt_characters,
            )
            config = PinnedGenerationConfig(
                prompt_version=PROMPT_VERSION,
                output_schema_version=OUTPUT_SCHEMA_VERSION,
                model_name=settings.llm_model_name,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
                max_prompt_evidence_items=settings.max_prompt_evidence_items,
                max_prompt_characters=settings.max_prompt_characters,
            )
            response = await provider.generate(projection, config)
            validated = parse_and_validate_analysis(
                response, projection, bundle.evidence_hash
            )
            return response, validated
        finally:
            await provider.close()

    response, validated = asyncio.run(run_once())
    assert len(validated.response.sections.model_fields) == 6  # type: ignore[attr-defined]
    assert response.status is ProviderCallStatus.RECEIVED  # type: ignore[attr-defined]
    assert response.usage.usage_available is True  # type: ignore[attr-defined]
    emit_llm_usage_event(
        route="ecn-impact-analysis",
        method="INTERNAL",
        actor_type="batch",
        provider="sap-ai-core",
        model=response.model_name,  # type: ignore[attr-defined]
        model_deployment=response.model_deployment,  # type: ignore[attr-defined]
        llm_endpoint="responses",
        usage_available=response.usage.usage_available,  # type: ignore[attr-defined]
        input_tokens=response.usage.input_tokens,  # type: ignore[attr-defined]
        cached_input_tokens=response.usage.cached_input_tokens,  # type: ignore[attr-defined]
        cache_read_input_tokens=response.usage.cache_read_input_tokens,  # type: ignore[attr-defined]
        cache_write_input_tokens=response.usage.cache_write_input_tokens,  # type: ignore[attr-defined]
        input_total_tokens=response.usage.input_total_tokens,  # type: ignore[attr-defined]
        output_tokens=response.usage.output_tokens,  # type: ignore[attr-defined]
        total_tokens=response.usage.total_tokens,  # type: ignore[attr-defined]
        outcome="success",
        latency_ms=response.latency_ms,  # type: ignore[attr-defined]
        correlation_id="live-provider-test",
        upload_id="live-upload",
        job_id="live-job",
        attempt_id="live-attempt",
    )
    assert "btp.llm_usage.v1" in capsys.readouterr().out
