"""Prompt scenarios for quote extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .extraction_schema import schema_as_pretty_json


@dataclass(frozen=True)
class PromptSpec:
    """Resolved prompt data for a scenario."""

    scenario_key: str
    version: str
    system_prompt: str
    user_prompt: str
    summary: str


def simple_prompt() -> PromptSpec:
    """A deliberately simple baseline prompt."""

    return PromptSpec(
        scenario_key="simple_prompt",
        version="v1",
        system_prompt="You are an expert business document extraction assistant.",
        user_prompt=(
            "Extract all relevant fields from this vendor quote document. "
            "Return valid JSON only. Include vendor, quote dates, totals, terms, "
            "and all line items you can identify."
        ),
        summary="Simple baseline: asks for quote fields without prescribing the full schema.",
    )


def detailed_static_prompt() -> PromptSpec:
    """A stable schema-first extraction prompt."""

    return PromptSpec(
        scenario_key="detailed_static_prompt",
        version="v1",
        system_prompt=(
            "You are a precise extraction engine for vendor quotation documents. "
            "You extract data for purchase requisition preparation. "
            "Never invent values. Use null when a field is missing or unclear."
        ),
        user_prompt=(
            "Extract this vendor quote into the exact JSON schema below.\n\n"
            "Rules:\n"
            "- Return only valid JSON and no markdown.\n"
            "- Preserve exact quote numbers, customer numbers, part numbers, and line numbers.\n"
            "- Normalize dates to YYYY-MM-DD when possible.\n"
            "- Monetary values must be numbers without currency symbols.\n"
            "- Put currency in ISO-like code, for example USD.\n"
            "- Extract every visible line item row.\n"
            "- Include short evidence snippets or page hints in the evidence object when possible.\n"
            "- Add warnings for ambiguous, missing, or inferred values.\n\n"
            f"JSON schema:\n{schema_as_pretty_json()}"
        ),
        summary="Detailed static prompt: fixed quote/PR schema with strict normalization rules.",
    )


def dynamic_prompt_seed() -> PromptSpec:
    """Prompt used to ask an LLM to generate a document-specific extraction prompt."""

    return PromptSpec(
        scenario_key="dynamic_prompt",
        version="v1",
        system_prompt=(
            "You design extraction prompts for business documents. "
            "You inspect layout clues and produce a better extraction prompt."
        ),
        user_prompt=(
            "Analyze this vendor quote document and create a document-specific extraction prompt. "
            "The generated prompt must still require the canonical JSON schema below and must be in English. "
            "Return JSON with keys: detected_vendor, layout_notes, risks, generated_prompt.\n\n"
            f"Canonical schema:\n{schema_as_pretty_json()}"
        ),
        summary="Dynamic prompt seed: asks a model to generate a tuned extraction prompt first.",
    )


def judge_prompt() -> PromptSpec:
    """Prompt used to judge and repair extraction quality."""

    return PromptSpec(
        scenario_key="dynamic_prompt_judge_loop",
        version="v1",
        system_prompt=(
            "You are a strict extraction quality judge for purchase requisition source documents. "
            "Find omissions, unsupported values, malformed line items, and schema violations."
        ),
        user_prompt=(
            "Judge the extraction result against the source vendor quote. "
            "Return valid JSON with keys: score_0_to_100, critical_errors, missing_fields, "
            "suspicious_values, repair_instructions, pass. The pass key must be true only if "
            "the result is reliable enough for downstream PR preparation."
        ),
        summary="Judge prompt: evaluates extraction quality and proposes repair instructions.",
    )


def get_prompt(scenario_key: str) -> PromptSpec:
    """Return prompt specification by scenario key."""

    prompts = {
        "simple_prompt": simple_prompt,
        "detailed_static_prompt": detailed_static_prompt,
        "dynamic_prompt": dynamic_prompt_seed,
        "dynamic_prompt_judge_loop": detailed_static_prompt,
    }
    try:
        return prompts[scenario_key]()
    except KeyError as exc:
        raise KeyError(f"Unsupported prompt scenario: {scenario_key}") from exc
