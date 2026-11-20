"""Bounded model-adapter acceptance using synthetic facts only, with no HANA writes.

Run: RUN_SNAPSHOT_MODEL_LIVE=1 PYTHONPATH=api .venv/bin/python -m pytest -q -s --tb=short tests/integration/test_snapshot_models_live.py
Makes one summary request and one assistant turn (up to four calls) on the configured
SAP Gen AI Hub model. Customer source exports are never loaded by this test.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest
from app.ecn.evidence.snapshot import build_snapshot, snapshot_page
from app.ecn.llm.provider import create_impact_provider
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.llm.validation import parse_and_validate_analysis
from app.ecn.models import OUTPUT_SCHEMA_VERSION, PROMPT_VERSION
from app.ecn.settings import load_ecn_settings
from app.persistence.records import PinnedGenerationConfig

from tests.fixtures.builders import (
    make_bom_row,
    make_normalized_ecn,
    make_source_object,
    minimal_evidence_bundle,
)

pytestmark = [
    pytest.mark.llm_live,
    pytest.mark.skipif(
        os.getenv("RUN_SNAPSHOT_MODEL_LIVE") != "1",
        reason="Explicit bounded synthetic model test only",
    ),
]


@pytest.mark.asyncio
async def test_single_synthesis_and_read_only_agent_with_synthetic_sources():
    """Validate actual SAP Responses transports for both supported model entry points."""
    from app.ecn.assistant.runtime import AgentRuntime

    settings = load_ecn_settings()
    provider = create_impact_provider(settings)
    source = make_normalized_ecn(
        affected_objects=(make_source_object(number="TEST-PART"),), resulting_objects=()
    )
    snapshot = build_snapshot(
        source, (make_bom_row(component_number="TEST-PART", dotted_level="4"),)
    )

    def overview(key):
        """Return synthetic counts only for the selected test assessment."""
        assert key == source.ecn_row_id
        return {k: v for k, v in snapshot.items() if k != "objects"}

    agent = AgentRuntime(
        SimpleNamespace(
            overview=overview,
            objects=lambda key, **filters: snapshot_page(snapshot, **filters),
        ),
        None,
        settings.llm_model_name,
        asyncio.Semaphore(2),
    )
    try:
        bundle = minimal_evidence_bundle()
        projection = build_synthesis_projection(bundle)
        config = PinnedGenerationConfig(
            prompt_version=PROMPT_VERSION,
            output_schema_version=OUTPUT_SCHEMA_VERSION,
            model_name=settings.llm_model_name,
            request_timeout_seconds=180,
            max_prompt_evidence_items=100,
            max_prompt_characters=120000,
        )
        result = await provider.generate(projection, config)
        validated = parse_and_validate_analysis(
            result, projection, bundle.evidence_hash
        )
        assert validated.response.impact_factors and result.usage.usage_available
        print(
            f"Synthetic summary validated; input={result.usage.input_tokens}, output={result.usage.output_tokens}",
            flush=True,
        )
        answer = await agent.ainvoke(
            "Use the overview tool and tell me how many technical objects are located. This is a synthetic adapter test; keep the answer to one sentence.",
            source.ecn_row_id,
            [],
        )
        assert answer["answer"] and 1 <= answer["model_calls"] <= 10
        print(
            f"Synthetic assistant completed in {answer['model_calls']} model calls: {answer['answer']}",
            flush=True,
        )
    finally:
        await agent.close()
        await provider.close()
