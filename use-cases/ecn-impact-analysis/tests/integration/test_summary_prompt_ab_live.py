"""Opt-in A/B acceptance test for the ECN executive-summary wording policy.

Run only after accepting two SAP Generative AI Hub calls:
    RUN_SUMMARY_AB_TEST=1 PYTHONPATH=api python -m pytest -q -s -m llm_live \
        tests/integration/test_summary_prompt_ab_live.py

The fixture is synthetic. It sends no uploaded ECN, BOM, HANA, or customer data.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.ecn.llm.provider import ProviderCallStage, ProviderCallStatus, create_impact_provider
from app.ecn.llm.staged import (
    EXECUTIVE_SUMMARY_PATTERN,
    SummaryNarrativeStyle,
    SummaryStageResult,
    build_stage_prompt,
)
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.models import OUTPUT_SCHEMA_VERSION, PROMPT_VERSION
from app.ecn.settings import load_ecn_settings
from app.persistence.records import PinnedGenerationConfig
from tests.fixtures.builders import minimal_evidence_bundle


pytestmark = [
    pytest.mark.llm_live,
    pytest.mark.skipif(
        os.getenv("RUN_SUMMARY_AB_TEST") != "1",
        reason="set RUN_SUMMARY_AB_TEST=1 to accept two synthetic SAP Gen AI Hub calls",
    ),
]

_SYNTHETIC_IDENTIFIERS = ("OLD-LINE-001", "NEW-LINE-002", "ASSY-HYD-001", "VC-GROUP-009")


def _synthetic_snapshot() -> dict:
    """Return a representative, non-customer BOM scenario for A/B comparison."""
    return {
        "assessment_date": "2026-05-01",
        "changes": [
            {
                "number": "OLD-LINE-001",
                "description": "Load-sensing line tube",
                "action": "PHASE_OUT",
            },
            {
                "number": "NEW-LINE-002",
                "description": "Load-sensing line tube",
                "action": "PHASE_IN",
            },
        ],
        "objects": [],
        "bom_changes": {
            "lines": [
                {
                    "scope_id": "SYNTHETIC-SCOPE",
                    "assembly": "ASSY-HYD-001",
                    "assembly_description": "CVT hydraulic supply",
                    "parent": "ASSY-HYD-001",
                    "alternative": "01",
                    "sequence": None,
                    "plants": [{"plant_code": "PL10", "plant_name": "Plant A"}],
                    "vc_groups": ["VC-GROUP-009"],
                    "model_range": "Q900V",
                    "change": "REPLACED",
                    "change_label": "Replaced",
                    "outgoing": {
                        "number": "OLD-LINE-001",
                        "description": "Load-sensing line tube",
                        "quantity": 1,
                        "unit": "PC",
                        "date": "2026-05-18",
                    },
                    "incoming": {
                        "number": "NEW-LINE-002",
                        "description": "Load-sensing line tube",
                        "quantity": 1,
                        "unit": "PC",
                        "date": "2026-05-19",
                    },
                    "engineering_date": "2026-05-01",
                    "manufacturing_date": "2026-05-19",
                    "levels": ["EBOM", "MBOM"],
                    "qualifiers": [],
                }
            ],
            "materials": [
                {
                    "number": "NEW-LINE-002",
                    "description": "Load-sensing line tube",
                    "role": "INCOMING",
                    "status": "NEW_PART",
                    "status_label": "New part",
                    "date": "2026-05-19",
                    "still_used_in": [],
                }
            ],
            "counts": {"REPLACED": 1, "ADDED": 0, "REMOVED": 0, "QUANTITY_CHANGED": 0},
        },
        "plant_impact": {
            "plants": [
                {
                    "plant_code": "PL10",
                    "plant_name": "Plant A",
                    "status": "CONFIRMED",
                    "cutover_from": "2026-05-19",
                    "old_part_stock": [{"number": "OLD-LINE-001", "quantity": 5}],
                    "new_parts_with_purchasing": [],
                    "new_parts_without_purchasing": [],
                    "materials": ["OLD-LINE-001", "NEW-LINE-002"],
                }
            ]
        },
        "questions": [],
    }


def _print_variant(label: str, summary: SummaryStageResult, usage: object) -> None:
    """Print one reviewer-readable A/B result without logging source payloads."""
    print(f"\n=== {label} ===")
    print(f"Headline: {summary.change_brief}")
    print(summary.sections.executive_summary.assessment_markdown)
    print(
        "Usage: "
        f"input={getattr(usage, 'input_tokens', None)}, "
        f"output={getattr(usage, 'output_tokens', None)}, "
        f"total={getattr(usage, 'total_tokens', None)}",
    )


@pytest.mark.asyncio
async def test_description_first_summary_is_a_grounded_identifier_free_candidate() -> None:
    """Compare two summary prompts with synthetic data before changing production wording."""
    load_dotenv(Path(__file__).resolve().parents[2] / "api" / ".env")
    settings = load_ecn_settings()
    projection = build_synthesis_projection(minimal_evidence_bundle(), snapshot=_synthetic_snapshot())
    config = PinnedGenerationConfig(
        prompt_version=PROMPT_VERSION,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        model_name=settings.llm_model_name,
        request_timeout_seconds=settings.llm_request_timeout_seconds,
        max_prompt_evidence_items=settings.max_prompt_evidence_items,
        max_prompt_characters=settings.max_prompt_characters,
    )
    provider = create_impact_provider(settings)

    async def generate(style: SummaryNarrativeStyle) -> tuple[SummaryStageResult, object]:
        """Generate and structurally validate exactly one A/B summary variant."""
        prompt = build_stage_prompt(
            projection,
            "SUMMARY",
            completed={},
            summary_narrative_style=style,
        )
        result = await provider.generate(prompt, config, ProviderCallStage.SUMMARY)
        assert result.status is ProviderCallStatus.RECEIVED
        assert result.response_status == "completed"
        assert not result.refusal_present
        assert result.raw_json
        summary = SummaryStageResult.model_validate_json(result.raw_json)
        assert re.fullmatch(
            EXECUTIVE_SUMMARY_PATTERN,
            summary.sections.executive_summary.assessment_markdown,
        )
        return summary, result.usage

    try:
        current, current_usage = await generate(SummaryNarrativeStyle.IDENTIFIER_LED)
        candidate, candidate_usage = await generate(SummaryNarrativeStyle.DESCRIPTION_FIRST)
    finally:
        await provider.close()

    _print_variant("A — current identifier-led summary", current, current_usage)
    _print_variant("B — description-first candidate", candidate, candidate_usage)

    candidate_text = candidate.sections.executive_summary.assessment_markdown
    assert all(identifier not in candidate_text for identifier in _SYNTHETIC_IDENTIFIERS)
    assert "hydraulic" in candidate_text.lower() or "load-sensing" in candidate_text.lower()
    assert "PL10" in candidate_text or "Plant A" in candidate_text
    assert "2026-05-19" in candidate_text or "19 May 2026" in candidate_text
