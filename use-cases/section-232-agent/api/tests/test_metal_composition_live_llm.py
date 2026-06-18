from __future__ import annotations

import json
import os

import pytest

from app.services.metal_composition.config import get_settings
from app.services.metal_composition.hts_catalog import HanaHTSCatalogResolver
from app.services.metal_composition.workflow.hana_tree_search import run_hana_tree_search
from app.services.metal_composition.workflow.hts_fact_profile import (
    build_hts_fact_profile_input,
    synthesize_hts_fact_profile,
)
from app.services.metal_composition.workflow.llm import LLMClient
from app.services.metal_composition.workflow.token_usage import TokenUsageRecorder


pytestmark = pytest.mark.skipif(
    os.getenv("METAL_COMPOSITION_RUN_LIVE_LLM_TESTS") != "1",
    reason="Set METAL_COMPOSITION_RUN_LIVE_LLM_TESTS=1 to run live LLM/HANA integration tests.",
)


def test_live_hts_profile_and_hana_tree_handle_expanded_catalog_context() -> None:
    """Run live HTS profile and HANA tree calls for the expanded catalog prompt."""

    settings = get_settings()
    llm = LLMClient(settings)
    usage_recorder = TokenUsageRecorder()
    state = {
        "product_code": "4006107",
        "source_summary": {
            "source_row_id": 310,
            "source_kind": "gcc",
            "pn_revised_standardized": "4006107",
            "part_description": "DIFFUSER COAT.",
            "new_part_description": None,
            "priority_detail": "PUMP SPARES",
            "total_weight_gram": 84000.0,
        },
        "source_row": {
            "Product code": "4006107",
            "PN Revised/ Standardized": "4006107",
            "Part description": "DIFFUSER COAT.",
            "New Part Description": None,
            "Priority.1": "PUMP SPARES",
            "Material Content Method": "Engineering Drawings",
            "MaterialIdentified": "1.0",
            "Total Weight (Gram)": 84000.0,
        },
        "final_composition": {
            "is_metal_item": True,
            "estimated_total_metal_grams": 78000.0,
            "top_level_grams": {
                "steel": 0.0,
                "aluminum": 0.0,
                "copper": 0.0,
                "cast_iron": 78000.0,
            },
            "steel_subtype_grams": {
                "electrical_steel": 0.0,
                "cold_rolled_coil_steel": 0.0,
                "hot_rolled_coil_steel": 0.0,
                "stainless_steel_304": 0.0,
                "stainless_steel_316": 0.0,
                "stainless_steel_bar": 0.0,
                "duplex_steel": 0.0,
                "cast_steel": 0.0,
            },
            "confidence": 1.0,
            "reasoning": "GCC tracker reports a predominantly cast iron article.",
        },
        "diagram_output": {
            "status": "omitted",
            "extracted_codes": [],
            "material_cues": ["cast iron (78000 g)"],
            "material_properties": [],
            "context_of_use": "",
            "hts_hints": [],
            "uncertainty_notes": [],
        },
    }

    profile_input = build_hts_fact_profile_input(state)
    profile, profile_details = synthesize_hts_fact_profile(
        state,
        settings,
        llm,
        usage_recorder=usage_recorder,
    )
    tree_state = {
        **state,
        "hts_fact_profile": profile,
    }
    tree_output = run_hana_tree_search(
        tree_state,
        settings,
        llm,
        HanaHTSCatalogResolver(settings=settings),
        usage_recorder=usage_recorder,
    )
    summary = usage_recorder.build_summary()
    recorded_tasks = {entry.task for entry in summary.entries}
    allowed_codes = {
        str(item.get("code"))
        for item in tree_output.get("family_options_sent", [])
    }
    for children in (tree_output.get("child_options_sent", {}) or {}).values():
        allowed_codes.update(str(item.get("code")) for item in children)
    combined_profile_text = json.dumps(profile, ensure_ascii=False).lower()

    assert profile_input["source_summary"]["extra_item_context"] == "PUMP SPARES"
    assert profile["status"] in {"ok", "completed"}
    assert profile_details["fallback_used"] is False
    assert "pump" in combined_profile_text or "spare" in combined_profile_text
    assert tree_output["status"] == "completed"
    assert tree_output["errors"] == []
    assert "heading_router" not in tree_output
    assert "heading_router" not in recorded_tasks
    assert "synthesis" in recorded_tasks
    assert "family_router" in recorded_tasks
    assert tree_output["routing_diagnostics"]["heading_options_considered_count"] >= 1
    assert tree_output["routing_diagnostics"]["family_option_count"] >= tree_output["routing_diagnostics"][
        "heading_options_considered_count"
    ]
    assert tree_output["routing_diagnostics"]["family_option_count"] == len(
        tree_output.get("family_options_sent", [])
    )
    assert tree_output["routing_diagnostics"]["family_prompt_bytes"] < 120000
    assert tree_output["candidate_suggestions"]
    assert {
        str(item.get("hts_code"))
        for item in tree_output["candidate_suggestions"]
    } <= allowed_codes
