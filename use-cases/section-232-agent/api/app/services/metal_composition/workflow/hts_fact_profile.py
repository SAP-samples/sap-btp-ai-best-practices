"""HTS fact-profile synthesis for downstream retrieval and ranking."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import MetalCompositionSettings
from .llm import LLMClient
from .normalize import (
    _STANDARD_CUE_RE,
    _extract_json_payload,
    _is_generic_function_label,
    _normalize_confidence,
    _normalize_float,
    _normalize_heading_hypotheses,
    _normalize_text,
    _unique_strings,
)
from .original_data import prompt_safe_source_row, prompt_safe_source_summary
from .token_usage import TokenUsageRecorder
from .types import MetalCompositionState

logger = logging.getLogger(__name__)

_TREE_SEARCH_DIRECTIVE_BRANCH_FOCUS = {
    "parts",
    "complete_articles",
    "material_articles",
    "residual_other",
}


def build_hts_fact_profile_input(state: MetalCompositionState) -> Dict[str, Any]:
    final_composition = dict(state.get("final_composition") or {})
    diagram_output = dict(state.get("diagram_output") or {})

    return {
        "source_summary": prompt_safe_source_summary(state.get("source_summary", {}) or {}),
        "feature_context": prompt_safe_source_row(state.get("source_row", {}) or {}),
        "final_composition": {
            "is_metal_item": final_composition.get("is_metal_item"),
            "estimated_total_metal_grams": final_composition.get("estimated_total_metal_grams"),
            "top_level_grams": final_composition.get("top_level_grams", {}),
            "steel_subtype_grams": final_composition.get("steel_subtype_grams", {}),
            "confidence": final_composition.get("confidence"),
            "reasoning": final_composition.get("reasoning"),
        },
        "diagram": {
            "status": diagram_output.get("status"),
            "extracted_codes": diagram_output.get("extracted_codes", []),
            "material_cues": diagram_output.get("material_cues", []),
            "material_properties": diagram_output.get("material_properties", []),
            "context_of_use": diagram_output.get("context_of_use", ""),
            "hts_hints": diagram_output.get("hts_hints", []),
            "uncertainty_notes": diagram_output.get("uncertainty_notes", []),
        },
    }


def fallback_hts_fact_profile(state: MetalCompositionState) -> Dict[str, Any]:
    source_summary = state.get("source_summary", {}) or {}
    source_row = state.get("source_row", {}) or {}
    safe_summary = prompt_safe_source_summary(source_summary)
    safe_row = prompt_safe_source_row(source_row)
    extra_item_context = str(
        safe_summary.get("extra_item_context") or safe_row.get("extra_item_context") or ""
    ).strip()
    final_composition = dict(state.get("final_composition") or {})
    diagram_output = dict(state.get("diagram_output") or {})

    diagram_context_of_use = str(diagram_output.get("context_of_use") or "").strip()
    article_candidates = _unique_strings(
        [
            str(source_summary.get("new_part_description") or ""),
            str(source_summary.get("part_description") or ""),
            str(source_row.get("New Part Description") or ""),
            str(source_row.get("Part description") or ""),
        ]
        + ([extra_item_context] if extra_item_context else [])
        + ([diagram_context_of_use] if diagram_context_of_use else [])
    )
    article_summary = article_candidates[0] if article_candidates else "Industrial part"
    function_candidates = _unique_strings(
        [
            str(source_summary.get("part_description") or ""),
            str(source_summary.get("new_part_description") or ""),
            str(source_row.get("Part description") or ""),
            str(source_row.get("New Part Description") or ""),
        ]
        + ([extra_item_context] if extra_item_context else [])
        + ([diagram_context_of_use] if diagram_context_of_use else [])
    )
    function_summary = article_summary
    for candidate in function_candidates:
        if not _is_generic_function_label(candidate):
            function_summary = candidate
            break

    material_profile = {
        "is_metal_item": bool(final_composition.get("is_metal_item", False)),
        "estimated_total_metal_grams": _normalize_float(
            final_composition.get("estimated_total_metal_grams", 0.0)
        ),
        "top_level_grams": dict(final_composition.get("top_level_grams", {}) or {}),
        "steel_subtype_grams": dict(final_composition.get("steel_subtype_grams", {}) or {}),
        "confidence": _normalize_confidence(final_composition.get("confidence", 0.0)),
        "reasoning": str(final_composition.get("reasoning", "") or "").strip(),
    }

    diagram_clues = _unique_strings(
        [str(value) for value in diagram_output.get("material_cues", [])]
        + [str(value) for value in diagram_output.get("extracted_codes", [])]
        + [str(value) for value in diagram_output.get("uncertainty_notes", [])]
        + ([f"Extra item context: {extra_item_context}"] if extra_item_context else [])
    )[:8]

    discriminator_notes: List[str] = []
    if not material_profile["is_metal_item"]:
        discriminator_notes.append(
            "Confirm whether the article is primarily metal before applying metal-specific HTS branches."
        )
    if material_profile["confidence"] < 0.75:
        discriminator_notes.append(
            "Material profile confidence is moderate; validate material-driven heading splits."
        )
    if diagram_output.get("status") == "omitted":
        discriminator_notes.append("No diagram analysis was available for article-form clues.")
    confidence = round(material_profile["confidence"], 4)

    reasoning_parts = [
        "Fallback HTS fact profile synthesized from part descriptions and final material reasoning."
    ]
    if diagram_clues:
        reasoning_parts.append("Diagram cues were included for article-form hints.")

    # Derive heading_hypotheses from diagram hts_hints (soft suggestions)
    heading_hypotheses: List[str] = []
    for hint in diagram_output.get("hts_hints", []):
        if isinstance(hint, dict) and hint.get("chapter"):
            chapter_str = str(hint["chapter"])
            if chapter_str not in heading_hypotheses:
                heading_hypotheses.append(chapter_str)

    return {
        "status": "completed",
        "article_summary": article_summary,
        "function_summary": function_summary,
        "material_profile": material_profile,
        "diagram_clues": diagram_clues,
        "heading_hypotheses": heading_hypotheses,
        "tree_search_directives": [],
        "discriminator_notes": discriminator_notes,
        "reasoning": " ".join(reasoning_parts),
        "confidence": max(0.0, min(1.0, confidence)),
        "raw_model_response": "",
        "fallback_reason": "",
    }


def normalize_tree_search_directives(values: Any) -> List[Dict[str, Any]]:
    """Return safe HTS tree-search directives from model or fallback output.

    Args:
        values: Raw directive payload, normally a list of dictionaries emitted by
            the HTS fact-profile model.

    Returns:
        A list of directives containing only valid 4-digit headings, known branch
        focus values, normalized booleans, compact rationales, and bounded
        confidence values.
    """

    if not isinstance(values, list):
        return []

    directives: List[Dict[str, Any]] = []
    seen = set()
    for raw_item in values:
        if not isinstance(raw_item, dict):
            continue
        heading = _normalize_text(raw_item.get("heading"))
        if not re.fullmatch(r"\d{4}", heading):
            continue

        raw_focus = raw_item.get("branch_focus", [])
        if isinstance(raw_focus, str):
            raw_focus_values = [raw_focus]
        elif isinstance(raw_focus, list):
            raw_focus_values = raw_focus
        else:
            raw_focus_values = []
        branch_focus = _unique_strings(
            [
                _normalize_text(value).lower()
                for value in raw_focus_values
                if _normalize_text(value).lower() in _TREE_SEARCH_DIRECTIVE_BRANCH_FOCUS
            ]
        )
        if not branch_focus:
            continue

        key = (heading, tuple(branch_focus))
        if key in seen:
            continue
        seen.add(key)
        directives.append(
            {
                "heading": heading,
                "branch_focus": branch_focus,
                "include_residual_children": bool(raw_item.get("include_residual_children", False)),
                "rationale": _normalize_text(raw_item.get("rationale"))[:240],
                "confidence": _normalize_confidence(raw_item.get("confidence")),
            }
        )
    return directives


def normalize_hts_fact_profile(
    payload: Dict[str, Any],
    *,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    fallback_material = dict(fallback.get("material_profile") or {})
    raw_material = payload.get("material_profile", {})
    if not isinstance(raw_material, dict):
        raw_material = {}

    top_level_grams = dict(fallback_material.get("top_level_grams", {}) or {})
    for metal in ("steel", "aluminum", "copper", "cast_iron"):
        if metal in raw_material.get("top_level_grams", {}):
            top_level_grams[metal] = max(
                0.0,
                _normalize_float((raw_material.get("top_level_grams", {}) or {}).get(metal, 0.0)),
            )

    steel_subtype_grams = dict(fallback_material.get("steel_subtype_grams", {}) or {})
    for subtype in (
        "electrical_steel",
        "cold_rolled_coil_steel",
        "hot_rolled_coil_steel",
        "stainless_steel_304",
        "stainless_steel_316",
        "stainless_steel_bar",
        "duplex_steel",
        "cast_steel",
    ):
        if subtype in raw_material.get("steel_subtype_grams", {}):
            steel_subtype_grams[subtype] = max(
                0.0,
                _normalize_float((raw_material.get("steel_subtype_grams", {}) or {}).get(subtype, 0.0)),
            )

    normalized = {
        "status": str(payload.get("status") or fallback.get("status") or "completed").strip()
        or "completed",
        "article_summary": str(payload.get("article_summary") or fallback.get("article_summary") or "").strip(),
        "function_summary": str(
            payload.get("function_summary") or fallback.get("function_summary") or ""
        ).strip(),
        "material_profile": {
            "is_metal_item": bool(
                raw_material.get("is_metal_item", fallback_material.get("is_metal_item", False))
            ),
            "estimated_total_metal_grams": max(
                0.0,
                _normalize_float(
                    raw_material.get(
                        "estimated_total_metal_grams",
                        fallback_material.get("estimated_total_metal_grams", 0.0),
                    )
                ),
            ),
            "top_level_grams": top_level_grams,
            "steel_subtype_grams": steel_subtype_grams,
            "confidence": max(
                0.0,
                min(
                    1.0,
                    _normalize_confidence(raw_material.get("confidence", fallback_material.get("confidence", 0.0))),
                ),
            ),
            "reasoning": str(
                raw_material.get("reasoning") or fallback_material.get("reasoning") or ""
            ).strip(),
        },
        "diagram_clues": _unique_strings(
            [str(value) for value in payload.get("diagram_clues", []) or fallback.get("diagram_clues", [])]
        ),
        "heading_hypotheses": _normalize_heading_hypotheses(
            list(payload.get("heading_hypotheses", []) or fallback.get("heading_hypotheses", []))
        ),
        "tree_search_directives": normalize_tree_search_directives(
            payload.get("tree_search_directives", fallback.get("tree_search_directives", []))
        ),
        "discriminator_notes": _unique_strings(
            [
                str(value)
                for value in payload.get("discriminator_notes", [])
                or fallback.get("discriminator_notes", [])
            ]
        ),
        "reasoning": str(payload.get("reasoning") or fallback.get("reasoning") or "").strip(),
        "confidence": _normalize_confidence(payload.get("confidence", fallback.get("confidence", 0.0))),
        "raw_model_response": str(
            payload.get("raw_model_response") or fallback.get("raw_model_response") or ""
        ),
        "fallback_reason": str(payload.get("fallback_reason") or fallback.get("fallback_reason") or "").strip(),
    }
    if not normalized["article_summary"]:
        normalized["article_summary"] = str(fallback.get("article_summary") or "Industrial part")
    if not normalized["function_summary"]:
        normalized["function_summary"] = normalized["article_summary"]
    if not normalized["reasoning"]:
        normalized["reasoning"] = str(fallback.get("reasoning") or "").strip()
    return normalized


def synthesize_hts_fact_profile(
    state: MetalCompositionState,
    settings: MetalCompositionSettings,
    llm: LLMClient,
    *,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    fallback = fallback_hts_fact_profile(state)
    prompt_input = build_hts_fact_profile_input(state)
    raw_text = ""
    system_prompt = (
        "You are preparing an HTS-oriented fact profile for a downstream retrieval and ranking agent. "
        "Return STRICT JSON only with keys: "
        '"status", "article_summary", "function_summary", "material_profile", '
        '"diagram_clues", "heading_hypotheses", "tree_search_directives", '
        '"discriminator_notes", "reasoning", "confidence". '
        '"material_profile" must contain keys: "is_metal_item", "estimated_total_metal_grams", '
        '"top_level_grams", "steel_subtype_grams", "confidence", "reasoning". '
        '"diagram_clues" and "discriminator_notes" must be arrays of short strings. '
        '"heading_hypotheses" must be an array of plausible 4-digit HTS headings only when justified. '
        '"tree_search_directives" must be an array of objects with keys "heading", "branch_focus", '
        '"include_residual_children", "rationale", "confidence"; use branch_focus values only from '
        '"parts", "complete_articles", "material_articles", "residual_other". '
        "Use directives to tell downstream HANA catalog recall which real HTS branches to explore, "
        "not to make the final classification. "
        'Every "confidence" field must be numeric between 0.0 and 1.0, for example "confidence": 0.82, not words like "high" or "medium". '
        "Focus on article identity, function, material clues, and unresolved discriminators that help HTS selection."
    )
    prompt = (
        "Source summary: "
        f"{json.dumps(prompt_safe_source_summary(state.get('source_summary', {}) or {}), default=str)}\n"
        f"Source context: {json.dumps(prompt_safe_source_row(state.get('source_row', {}) or {}), default=str)}\n"
        f"Signals: {json.dumps(prompt_input, default=str)}"
    )
    try:
        response = llm.invoke_native_chat_completion(
            model_name=settings.hts_fact_profile_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            thinking_type=settings.hts_fact_profile_reasoning_type,
            thinking_budget=settings.hts_fact_profile_reasoning_budget,
            phase="hts_fact_profile",
            task="synthesis",
            usage_recorder=usage_recorder,
        )
        message = response.choices[0].message.content
        raw_text = message if isinstance(message, str) else json.dumps(message)
        payload = _extract_json_payload(raw_text)
        return normalize_hts_fact_profile(payload, fallback=fallback), {
            "model": settings.hts_fact_profile_model_name,
            "prompt_bytes": len(prompt.encode("utf-8")),
            "fallback_used": False,
        }
    except Exception as exc:  # noqa: BLE001 - keep HTS alive with a deterministic fallback
        logger.warning("HTS fact-profile synthesis fell back to deterministic output: %s", exc)
        fallback_reason = str(exc)
        fallback["reasoning"] = f"{fallback['reasoning']} Fallback reason: {fallback_reason}".strip()
        fallback["raw_model_response"] = raw_text
        fallback["fallback_reason"] = fallback_reason
        return fallback, {
            "model": settings.hts_fact_profile_model_name,
            "prompt_bytes": len(prompt.encode("utf-8")),
            "fallback_used": True,
            "error": fallback_reason,
        }


def extract_standard_cues(state: MetalCompositionState) -> List[str]:
    candidates = _unique_strings(
        [
            state.get("hts_fact_profile", {}).get("article_summary", ""),
            state.get("hts_fact_profile", {}).get("function_summary", ""),
            state.get("source_summary", {}).get("part_description", ""),
            state.get("source_summary", {}).get("new_part_description", ""),
            state.get("source_row", {}).get("Part description", ""),
            state.get("source_row", {}).get("New Part Description", ""),
        ]
    )
    matches: List[str] = []
    for candidate in candidates:
        matches.extend(_STANDARD_CUE_RE.findall(candidate))
    return _unique_strings(matches)


def material_search_clues(material_profile: Dict[str, Any]) -> List[str]:
    clues: List[str] = []
    top_level = dict(material_profile.get("top_level_grams", {}) or {})
    for metal, grams in top_level.items():
        if float(grams or 0.0) <= 0.0:
            continue
        clues.append(f"{metal.replace('_', ' ')} {float(grams):.1f} g")

    subtypes = dict(material_profile.get("steel_subtype_grams", {}) or {})
    subtype_terms = [
        subtype.replace("_", " ")
        for subtype, grams in subtypes.items()
        if float(grams or 0.0) > 0.0
    ][:2]
    clues.extend(subtype_terms)
    return _unique_strings(clues)
