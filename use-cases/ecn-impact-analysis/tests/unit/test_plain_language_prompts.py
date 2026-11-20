"""Prompt v6: every stage carries plain-language rules; the summary has a fixed shape."""

import re

from app.ecn.llm.staged import EXECUTIVE_SUMMARY_PATTERN, STYLE


def test_executive_summary_shape():
    """Three labelled bullets are required; 'Still open' is optional; old free bullets fail."""
    base = "- **What changes:** A is replaced by B.\n- **What is impacted:** Assembly G.\n- **Where:** Plant PL10."
    assert re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, base)
    assert re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, base + "\n- **Still open:** Stock unit.")
    assert not re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, "- SIM-ECN-001 is a synthetic engineering impact demonstration.")
    assert not re.fullmatch(EXECUTIVE_SUMMARY_PATTERN, base + "\n- **Still open:** x\n- **Still open:** y")


def test_summary_prompt_requires_an_extremely_brief_ai_change_headline():
    """The detail-page headline is a stored model output, not a second source description."""
    from app.ecn.llm.staged import build_stage_prompt
    from app.ecn.llm.synthesis import build_synthesis_projection
    from tests.fixtures.builders import make_evidence_bundle

    prompt = build_stage_prompt(build_synthesis_projection(make_evidence_bundle()), "SUMMARY", completed={})

    assert "change_brief" in prompt.instructions
    assert "at most 16 words" in prompt.instructions
    assert "Do not list parts, assemblies, labels" in prompt.instructions
    assert "rules override the general identifier-naming rule" in prompt.instructions


def test_style_names_bom_facts_and_forbidden_jargon():
    """The shared style points to BOM changes and bans the jargon users flagged."""
    assert "snapshot_context.bom_changes" in STYLE and "plant_impact" in STYLE
    for word in ("resulting object", "occurrence", "source-backed", "supplied scope"):
        assert word in STYLE.split("do not use these words:")[1]


def test_chat_prompt_uses_shared_style():
    """The chat assistant follows the same plain-language rules."""
    from app.ecn.assistant.runtime import BASE_PROMPT
    assert STYLE in BASE_PROMPT


def test_chat_prompt_names_parts_and_grounds_next_steps():
    """Chat must name exact parts and check the review decision before listing next steps."""
    from app.ecn.assistant.runtime import BASE_PROMPT
    assert "exact number" in BASE_PROMPT and "the two new parts" in BASE_PROMPT
    assert "get_review_status" in BASE_PROMPT and "get_data_quality_issues" in BASE_PROMPT




def test_chat_prompt_formats_changes_dates_and_omits_noise():
    """Chat answers use numbered changes and one-line dates, and skip unchanged facts and gaps."""
    from app.ecn.assistant.runtime import BASE_PROMPT
    fmt = BASE_PROMPT.split("Chat answer format.")[1]
    assert "What changed:" in fmt and "1. QKX4638810 is replaced by QKX463881A" in fmt
    assert "EBOM Release Date: 1/16/2026" in fmt and "MBOM Cutover Date: 6/30/2026" in fmt
    assert "Impact: Medium" in fmt and "Never mention the stored" in fmt
    assert "unchanged quantity" in fmt and "Do not add gap" in fmt
    assert "Mention a gap once, next to its finding" not in BASE_PROMPT


def test_summary_prompt_ab_variants_keep_data_identical_but_offer_description_first_candidate():
    """The opt-in A/B runner changes only SUMMARY instructions, never evidence input."""
    from app.ecn.llm.staged import SummaryNarrativeStyle, build_stage_prompt
    from app.ecn.llm.synthesis import build_synthesis_projection
    from tests.fixtures.builders import make_evidence_bundle

    projection = build_synthesis_projection(make_evidence_bundle())
    current = build_stage_prompt(
        projection,
        "SUMMARY",
        completed={},
        summary_narrative_style=SummaryNarrativeStyle.IDENTIFIER_LED,
    )
    default = build_stage_prompt(projection, "SUMMARY", completed={})
    candidate = build_stage_prompt(
        projection,
        "SUMMARY",
        completed={},
        summary_narrative_style=SummaryNarrativeStyle.DESCRIPTION_FIRST,
    )

    assert current.payload == candidate.payload == default.payload
    assert "parts replaced, added, removed or changed in quantity, with numbers and dates" in current.instructions
    assert "OVERRIDE for executive_summary.assessment_markdown only" not in current.instructions
    assert "OVERRIDE for executive_summary.assessment_markdown only" in candidate.instructions
    assert "Do not output a material, assembly, or VC-group identifier at all" in candidate.instructions
    assert "meaningful supplied description" in candidate.instructions
    assert "truthful generic count or category" in candidate.instructions
    assert "Do not apply this override to change_brief" in candidate.instructions
    assert candidate.instructions.rindex("Do not output a material, assembly, or VC-group identifier at all") > candidate.instructions.rindex(STYLE)
    assert default.instructions == candidate.instructions
