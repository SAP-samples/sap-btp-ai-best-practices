"""Trade-decision logic: candidate pool, LLM selector, fallback, and Section 232."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from app.models.metal_composition import (
    HTSCandidate,
    HTSClassification,
    HTSCitation,
    Section232Assessment,
)

from ..config import MetalCompositionSettings
from ..hts_catalog import HanaHTSCatalogResolver, HTSCodeResolution
from ..section_232_rules_engine import (
    apply_section_232_weight_override,
    build_skipped_weight_override,
    build_section_232_ruleset_assessment,
    build_section_232_ruleset_reasoner_output,
    evaluate_section_232_ruleset,
    legal_coverage_established,
)
from .llm import LLMClient
from .normalize import (
    _code_digit_count,
    _extract_json_payload,
    _normalize_bool,
    _normalize_candidate_code,
    _normalize_confidence,
    _normalize_float,
    _normalize_int,
    _normalize_reasoning_key,
    _normalize_text,
    _response_text,
    _unique_strings,
)
from .original_data import prompt_safe_source_row, prompt_safe_source_summary
from .token_usage import TokenUsageRecorder
from .url_helpers import (
    _normalize_chapter_99_codes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no external dependencies beyond normalize/url_helpers)
# ---------------------------------------------------------------------------


def normalize_candidate_entry(raw_candidate: Any) -> Dict[str, Any] | None:
    """Normalize a single raw candidate dict into an HTSCandidate model_dump, or None."""
    if not isinstance(raw_candidate, dict):
        return None
    code = _normalize_candidate_code(raw_candidate.get("code"))
    if not code:
        return None
    citations: List[HTSCitation] = []
    for citation in raw_candidate.get("citations", []) or []:
        try:
            citations.append(HTSCitation.model_validate(citation))
        except Exception:  # noqa: BLE001 - tolerate partial citation payloads
            continue
    candidate = HTSCandidate(
        code=code,
        description=_normalize_text(raw_candidate.get("description") or "HTS candidate"),
        digits=int(raw_candidate.get("digits") or _code_digit_count(code)),
        confidence=_normalize_confidence(raw_candidate.get("confidence")),
        reasoning=_normalize_text(raw_candidate.get("reasoning")),
        validation_status=_normalize_text(raw_candidate.get("validation_status")) or None,
        normalized_from=_normalize_candidate_code(raw_candidate.get("normalized_from")) or None,
        resolution_basis=_normalize_text(raw_candidate.get("resolution_basis")) or None,
        citations=citations,
        origins=_unique_strings(raw_candidate.get("origins", []) or []),
        hana_router_stage=_normalize_text(raw_candidate.get("hana_router_stage")) or None,
        specificity_supported=_normalize_int(raw_candidate.get("specificity_supported")) or None,
        matched_path=_normalize_text(raw_candidate.get("matched_path")) or None,
        matched_terms=_unique_strings(raw_candidate.get("matched_terms", []) or []),
        matched_phrases=_unique_strings(raw_candidate.get("matched_phrases", []) or []),
        retrieval_score=_normalize_float(raw_candidate.get("retrieval_score")),
        missing_discriminators=_unique_strings(raw_candidate.get("missing_discriminators", []) or []),
    )
    return candidate.model_dump()


def merge_candidate_entries(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two candidate dicts for the same code, keeping the higher-confidence side as primary."""
    if _normalize_confidence(right.get("confidence")) > _normalize_confidence(left.get("confidence")):
        primary, secondary = right, left
    else:
        primary, secondary = left, right

    seen_citations = set()
    citations: List[Dict[str, Any]] = []
    for candidate in (primary, secondary):
        for citation in candidate.get("citations", []) or []:
            key = (
                citation.get("page_number"),
                citation.get("page_label"),
                citation.get("chapter_number"),
                citation.get("heading_code"),
                citation.get("source_url"),
                citation.get("ruling_number"),
            )
            if key in seen_citations:
                continue
            seen_citations.add(key)
            citations.append(dict(citation))

    reasoning_parts: List[str] = []
    seen_reasoning = set()
    for part in (
        _normalize_text(primary.get("reasoning")),
        _normalize_text(secondary.get("reasoning")),
    ):
        key = _normalize_reasoning_key(part)
        if not part or not key or key in seen_reasoning:
            continue
        seen_reasoning.add(key)
        reasoning_parts.append(part)
    reasoning = " ".join(reasoning_parts).strip()

    return {
        "code": primary.get("code") or secondary.get("code"),
        "description": primary.get("description") or secondary.get("description") or "HTS candidate",
        "digits": int(primary.get("digits") or secondary.get("digits") or 0),
        "confidence": max(
            _normalize_confidence(primary.get("confidence")),
            _normalize_confidence(secondary.get("confidence")),
        ),
        "reasoning": reasoning,
        "validation_status": primary.get("validation_status") or secondary.get("validation_status"),
        "normalized_from": primary.get("normalized_from") or secondary.get("normalized_from"),
        "resolution_basis": primary.get("resolution_basis") or secondary.get("resolution_basis"),
        "citations": citations,
        "origins": _unique_strings(
            list(primary.get("origins", []) or [])
            + list(secondary.get("origins", []) or [])
        ),
        "hana_router_stage": _normalize_text(primary.get("hana_router_stage") or secondary.get("hana_router_stage")) or None,
        "specificity_supported": max(
            _normalize_int(primary.get("specificity_supported")),
            _normalize_int(secondary.get("specificity_supported")),
        )
        or None,
        "matched_path": _normalize_text(primary.get("matched_path") or secondary.get("matched_path")) or None,
        "matched_terms": _unique_strings(
            list(primary.get("matched_terms", []) or [])
            + list(secondary.get("matched_terms", []) or [])
        ),
        "matched_phrases": _unique_strings(
            list(primary.get("matched_phrases", []) or [])
            + list(secondary.get("matched_phrases", []) or [])
        ),
        "retrieval_score": max(
            _normalize_float(primary.get("retrieval_score")),
            _normalize_float(secondary.get("retrieval_score")),
        ),
        "missing_discriminators": _unique_strings(
            list(primary.get("missing_discriminators", []) or [])
            + list(secondary.get("missing_discriminators", []) or [])
        ),
    }


def source_rank(*, source_type: str, origin: str) -> int:
    """Return an integer priority for a given signal source/origin pair."""
    if origin == "hana_tree":
        return 5
    if source_type == "hana":
        return 4
    if source_type == "hts":
        return 1
    return 0


def validation_rank(validation_status: str) -> int:
    """Return an integer priority for a given validation status."""
    return {
        "current_exact": 4,
        "legacy_mapped_exact": 3,
        "downgraded_to_8": 2,
        "downgraded_to_6": 1,
    }.get(validation_status, 0)


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------


def collect_hana_tree_candidate_signals(hana_tree_search_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect candidate signals from HANA tree search output."""
    signals: List[Dict[str, Any]] = []
    seen = set()
    for item in hana_tree_search_output.get("candidate_suggestions", []) or []:
        if not isinstance(item, dict):
            continue
        code = _normalize_candidate_code(item.get("hts_code") or item.get("code"))
        if not code:
            continue
        key = (code, "hana_tree")
        if key in seen:
            continue
        seen.add(key)
        signals.append(
            {
                "origin": "hana_tree",
                "source_type": "hana",
                "hts_code": code,
                "confidence": _normalize_confidence(item.get("confidence")),
                "ruling_numbers": [],
                "similarity_summary": _normalize_text(
                    " ".join(
                        _unique_strings(
                            [
                                item.get("rationale"),
                                item.get("matched_path"),
                            ]
                        )
                    )
                ),
                "source_url": "",
                "hana_router_stage": _normalize_text(item.get("hana_router_stage")) or None,
                "specificity_supported": _normalize_int(item.get("specificity_supported")) or None,
                "matched_path": _normalize_text(item.get("matched_path")),
                "matched_terms": _unique_strings(item.get("matched_terms", []) or []),
                "matched_phrases": _unique_strings(item.get("matched_phrases", []) or []),
                "retrieval_score": _normalize_float(item.get("retrieval_score")),
                "recall_source": _normalize_text(item.get("recall_source")),
                "recall_reason": _normalize_text(item.get("recall_reason")),
                "recall_rank": _normalize_int(item.get("recall_rank")),
                "missing_discriminators": _unique_strings(item.get("missing_discriminators", []) or []),
            }
        )
    return signals


# ---------------------------------------------------------------------------
# Pool building
# ---------------------------------------------------------------------------


def build_catalog_candidate_pool(
    state: Dict[str, Any],
    settings: MetalCompositionSettings,
    resolver: HanaHTSCatalogResolver,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Build a validated candidate pool from HANA tree search signals."""
    signals = collect_hana_tree_candidate_signals(state.get("hana_tree_search_output", {}) or {})
    groups: Dict[str, Dict[str, Any]] = {}
    rejected_candidates: List[Dict[str, Any]] = []

    for signal in signals:
        resolution = resolver.resolve_code(signal.get("hts_code"))
        resolution_entry = {
            "requested_code": resolution.requested_code,
            "resolved_code": resolution.resolved_code,
            "validation_status": resolution.validation_status,
            "normalized_from": resolution.normalized_from,
            "resolution_basis": resolution.resolution_basis,
            "source_type": signal.get("source_type"),
            "origin": signal.get("origin"),
            "source_url": signal.get("source_url"),
            "ruling_numbers": list(signal.get("ruling_numbers", []) or []),
            "confidence": _normalize_confidence(signal.get("confidence")),
        }
        if resolution.validation_status == "invalid" or not resolution.resolved_code:
            rejected_candidates.append(resolution_entry)
            continue

        group = groups.setdefault(
            resolution.resolved_code,
            {
                "resolution": resolution,
                "validation_rank": validation_rank(resolution.validation_status),
                "source_rank": 0,
                "signal_count": 0,
                "confidence_total": 0.0,
                "best_signal_confidence": 0.0,
                "ruling_numbers": [],
                "summaries": [],
                "citations": [],
                "requested_codes": [],
                "origins": [],
                "hana_router_stages": [],
                "specificity_supported": [],
                "matched_paths": [],
                "matched_terms": [],
                "matched_phrases": [],
                "retrieval_scores": [],
                "missing_discriminators": [],
            },
        )
        group["signal_count"] += 1
        signal_confidence = _normalize_confidence(signal.get("confidence"))
        group["confidence_total"] += signal_confidence
        group["best_signal_confidence"] = max(group["best_signal_confidence"], signal_confidence)
        group["source_rank"] = max(
            group["source_rank"],
            source_rank(
                source_type=str(signal.get("source_type") or ""),
                origin=str(signal.get("origin") or "raw"),
            ),
        )
        group["ruling_numbers"].extend(_unique_strings(signal.get("ruling_numbers", []) or []))
        summary = _normalize_text(signal.get("similarity_summary"))
        if summary:
            group["summaries"].append(summary)
        group["origins"].append(str(signal.get("origin") or ""))
        if signal.get("hana_router_stage"):
            group["hana_router_stages"].append(str(signal.get("hana_router_stage")))
        if signal.get("specificity_supported"):
            group["specificity_supported"].append(_normalize_int(signal.get("specificity_supported")))
        if signal.get("matched_path"):
            group["matched_paths"].append(_normalize_text(signal.get("matched_path")))
        group["matched_terms"].extend(_unique_strings(signal.get("matched_terms", []) or []))
        group["matched_phrases"].extend(_unique_strings(signal.get("matched_phrases", []) or []))
        if signal.get("retrieval_score") is not None:
            group["retrieval_scores"].append(_normalize_float(signal.get("retrieval_score")))
        group["missing_discriminators"].extend(_unique_strings(signal.get("missing_discriminators", []) or []))
        if resolution.requested_code:
            group["requested_codes"].append(resolution.requested_code)

    candidate_pool: Dict[str, Dict[str, Any]] = {}
    validated_trace: List[Dict[str, Any]] = []
    for code, group in groups.items():
        resolution: HTSCodeResolution = group["resolution"]
        catalog_row = resolution.catalog_row
        avg_signal_confidence = group["confidence_total"] / max(group["signal_count"], 1)
        candidate_confidence = _normalize_confidence(avg_signal_confidence)

        deduped_citations: List[Dict[str, Any]] = []
        seen_citations = set()
        for citation in group["citations"]:
            key = (citation.get("source_url"), citation.get("ruling_number"))
            if key in seen_citations:
                continue
            seen_citations.add(key)
            deduped_citations.append(citation)

        missing_discriminators: List[str] = []
        if _code_digit_count(code) < 10:
            missing_discriminators.append(
                f"Current evidence supports only the validated {_code_digit_count(code)}-digit HTS family."
            )
        if resolution.validation_status != "current_exact":
            missing_discriminators.append(resolution.resolution_basis)
        missing_discriminators.extend(_unique_strings(group["missing_discriminators"]))

        reasoning = " ".join(
            _unique_strings(
                [
                    resolution.resolution_basis,
                    *group["summaries"][:2],
                ]
            )
        ).strip()
        candidate = HTSCandidate(
            code=code,
            description=_normalize_text(catalog_row.get("path_description") or catalog_row.get("description") or "Validated HTS candidate"),
            digits=_code_digit_count(code),
            confidence=candidate_confidence,
            reasoning=reasoning,
            validation_status=resolution.validation_status,
            normalized_from=resolution.normalized_from,
            resolution_basis=resolution.resolution_basis,
            citations=[HTSCitation.model_validate(citation) for citation in deduped_citations],
            origins=_unique_strings(group["origins"]),
            hana_router_stage=(
                _unique_strings(group["hana_router_stages"])[0]
                if _unique_strings(group["hana_router_stages"])
                else None
            ),
            specificity_supported=max(group["specificity_supported"]) if group["specificity_supported"] else None,
            matched_path=_unique_strings(group["matched_paths"])[0] if _unique_strings(group["matched_paths"]) else None,
            matched_terms=_unique_strings(group["matched_terms"]),
            matched_phrases=_unique_strings(group["matched_phrases"]),
            retrieval_score=max(group["retrieval_scores"]) if group["retrieval_scores"] else 0.0,
            missing_discriminators=_unique_strings(missing_discriminators),
        ).model_dump()
        candidate["source_rank"] = int(group["source_rank"])
        candidate["validation_rank"] = int(group["validation_rank"])
        candidate_pool[code] = candidate
        validated_trace.append(
            {
                **candidate,
                "requested_codes": _unique_strings(group["requested_codes"]),
                "signal_count": int(group["signal_count"]),
                "source_rank": int(group["source_rank"]),
                "validation_rank": int(group["validation_rank"]),
            }
        )

    return candidate_pool, {
        "status": "completed",
        "strategy": "hana_catalog",
        "catalog_table": settings.hts_catalog_hana_table,
        "code_map_table": settings.hts_code_map_hana_table,
        "hana_tree_search_status": str((state.get("hana_tree_search_output", {}) or {}).get("status") or "omitted"),
        "validated_candidates": validated_trace,
        "rejected_candidates": rejected_candidates,
    }


# ---------------------------------------------------------------------------
# Trade decision payload normalization
# ---------------------------------------------------------------------------


def normalize_trade_decision_payload(
    payload: Dict[str, Any],
    *,
    candidate_pool: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize LLM trade decision response against the candidate pool."""
    selected_code = _normalize_candidate_code(payload.get("selected_code"))
    if not selected_code or selected_code not in candidate_pool:
        raise ValueError("Trade decision model did not return a selectable validated HTS code.")

    rankings: List[Dict[str, Any]] = []
    seen_codes = set()
    for raw_item in payload.get("candidate_rankings", []) or []:
        if not isinstance(raw_item, dict):
            continue
        code = _normalize_candidate_code(raw_item.get("code"))
        if not code or code not in candidate_pool or code in seen_codes:
            continue
        seen_codes.add(code)
        rankings.append(
            {
                "code": code,
                "rationale": _normalize_text(raw_item.get("rationale") or raw_item.get("reasoning")),
                "confidence": _normalize_confidence(raw_item.get("confidence")),
                "missing_discriminators": _unique_strings(raw_item.get("missing_discriminators", []) or []),
            }
        )

    if selected_code not in seen_codes:
        rankings.insert(
            0,
            {
                "code": selected_code,
                "rationale": _normalize_text(payload.get("reasoning")),
                "confidence": _normalize_confidence(payload.get("confidence")),
                "missing_discriminators": [],
            },
        )

    return {
        "selected_code": selected_code,
        "reasoning": _normalize_text(payload.get("reasoning")),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "needs_human_review": _normalize_bool(payload.get("needs_human_review")),
        "candidate_rankings": rankings,
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_trade_decision_prompt(
    *,
    state: Dict[str, Any],
    resolution_output: Dict[str, Any],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the trade decision LLM call."""
    source_summary = dict(state.get("source_summary", {}) or {})
    source_row = dict(state.get("source_row", {}) or {})
    final_composition = dict(state.get("final_composition", {}) or {})
    hts_fact_profile = dict(state.get("hts_fact_profile", {}) or {})
    hana_tree_search_output = dict(state.get("hana_tree_search_output", {}) or {})

    prompt_payload = {
        "product_context": {
            "product_code": state.get("product_code"),
            "source_summary": prompt_safe_source_summary(source_summary),
            "source_row": prompt_safe_source_row(source_row),
        },
        "composition_findings": final_composition,
        "hts_fact_profile": {
            "article_summary": hts_fact_profile.get("article_summary"),
            "function_summary": hts_fact_profile.get("function_summary"),
            "material_profile": hts_fact_profile.get("material_profile"),
            "heading_hypotheses": hts_fact_profile.get("heading_hypotheses"),
            "tree_search_directives": hts_fact_profile.get("tree_search_directives"),
            "discriminator_notes": hts_fact_profile.get("discriminator_notes"),
            "reasoning": hts_fact_profile.get("reasoning"),
        },
        "hana_tree_search_findings": {
            "status": hana_tree_search_output.get("status"),
            "context": hana_tree_search_output.get("context", {}) or {},
            "chapter_router": {
                "status": (hana_tree_search_output.get("chapter_router", {}) or {}).get("status"),
                "selected_chapters": (hana_tree_search_output.get("chapter_router", {}) or {}).get(
                    "selected_chapters", []
                ),
                "reasoning": (hana_tree_search_output.get("chapter_router", {}) or {}).get("reasoning"),
            },
            "routing_diagnostics": hana_tree_search_output.get("routing_diagnostics", {}) or {},
            "recall_candidates": hana_tree_search_output.get("recall_candidates", []) or [],
            "family_options_sent": hana_tree_search_output.get("family_options_sent", []) or [],
            "child_options_sent": hana_tree_search_output.get("child_options_sent", {}) or {},
            "family_router": {
                "status": (hana_tree_search_output.get("family_router", {}) or {}).get("status"),
                "candidate_suggestions": (hana_tree_search_output.get("family_router", {}) or {}).get(
                    "candidate_suggestions", []
                ),
                "reasoning": (hana_tree_search_output.get("family_router", {}) or {}).get("reasoning"),
            },
            "candidate_suggestions": hana_tree_search_output.get("candidate_suggestions", []) or [],
        },
        "validated_candidates": resolution_output.get("validated_candidates", []) or [],
        "rejected_candidates": resolution_output.get("rejected_candidates", []) or [],
    }

    system_prompt = (
        "You are selecting the best-supported HTS code from a prevalidated set of current HTS candidates. "
        "Choose exactly one code from validated_candidates. "
        "Do not invent HTS codes. Do not choose rejected candidates. "
        "Prefer a broader 6-digit or 8-digit family when the evidence does not justify a narrower 10-digit line. "
        "Use the product facts, material/function clues, HANA tree-search findings, and each candidate's "
        "catalog description and validation status. The reasoning and rationale fields are user-facing: "
        "Do not mention internal field names or raw enum values such as validation_status, current_exact, "
        "legacy_mapped_exact, downgraded_to_8, or downgraded_to_6. Translate them into plain language, "
        "for example: the code appears as an exact active entry in the current HTS catalog. "
        "Return STRICT JSON only with keys: "
        '"selected_code", "reasoning", "confidence", "needs_human_review", "candidate_rankings". '
        '"candidate_rankings" must be an array of objects with keys: '
        '"code", "rationale", "confidence", "missing_discriminators". '
        'Every "confidence" value must be numeric between 0.0 and 1.0, for example "confidence": 0.82, not "confidence": "high". '
        "Use only codes already present in validated_candidates."
    )
    user_prompt = f"Decision input:\n{json.dumps(prompt_payload, default=str)}"
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# LLM selector and fallback
# ---------------------------------------------------------------------------


def run_trade_decision_selector(
    *,
    state: Dict[str, Any],
    candidate_pool: Dict[str, Dict[str, Any]],
    resolution_output: Dict[str, Any],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    """Run the LLM-based trade decision selector."""
    system_prompt, user_prompt = build_trade_decision_prompt(
        state=state,
        resolution_output=resolution_output,
    )
    response = llm.invoke_native_chat_completion(
        model_name=settings.trade_decision_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="medium",
        thinking_type=settings.trade_decision_reasoning_type,
        thinking_budget=settings.trade_decision_reasoning_budget,
        phase="trade_decision",
        task="selector",
        usage_recorder=usage_recorder,
    )
    raw_text = _response_text(response.choices[0].message.content)
    payload = _extract_json_payload(raw_text)
    normalized = normalize_trade_decision_payload(payload, candidate_pool=candidate_pool)
    normalized["raw_model_response"] = raw_text
    normalized["prompt_bytes"] = len(user_prompt.encode("utf-8"))
    return normalized


def fallback_trade_decision_codes(
    candidate_pool: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Deterministic fallback ordering of candidates when the LLM selector fails."""
    return [
        code
        for code, candidate in sorted(
            candidate_pool.items(),
            key=lambda item: (
                validation_rank(str(item[1].get("validation_status") or "")),
                int(item[1].get("source_rank") or 0),
                _normalize_float(item[1].get("retrieval_score")),
                _code_digit_count(item[0]),
                _normalize_confidence(item[1].get("confidence")),
            ),
            reverse=True,
        )
    ]


# ---------------------------------------------------------------------------
# Finalization and Section 232
# ---------------------------------------------------------------------------


def _resolve_section_232_on_date(raw_value: Any) -> date:
    if isinstance(raw_value, date):
        return raw_value
    if isinstance(raw_value, str):
        try:
            return date.fromisoformat(raw_value)
        except ValueError:
            pass
    return date.today()


def finalize_trade_decision(
    *,
    state: Dict[str, Any],
    candidate_pool: Dict[str, Dict[str, Any]],
    resolution_output: Dict[str, Any],
    ordered_codes: List[str],
    selected_code: str,
    selected_reasoning: str,
    selected_confidence: float,
    needs_human_review_hint: bool,
    selector_status: str,
    settings: MetalCompositionSettings,
    llm: LLMClient,
    resolver: HanaHTSCatalogResolver,
    section_232_source_store: Optional[object] = None,
    usage_recorder: Optional[TokenUsageRecorder] = None,
    selector_payload: Optional[Dict[str, Any]] = None,
    selector_error: str = "",
) -> Dict[str, Any]:
    """Build the final trade decision output including classification and Section 232."""
    unique_codes = _unique_strings([selected_code, *ordered_codes])
    ordered_candidates = [
        HTSCandidate.model_validate(candidate_pool[code])
        for code in unique_codes
        if code in candidate_pool
    ][:5]
    best_candidate = ordered_candidates[0] if ordered_candidates else None

    if not selected_reasoning and best_candidate is not None:
        selected_reasoning = _normalize_text(best_candidate.reasoning)
    if not selected_reasoning:
        selected_reasoning = "No stable current HTS candidate was available from official search evidence."

    needs_human_review = bool(
        needs_human_review_hint
        or best_candidate is None
        or best_candidate.digits < 10
        or (best_candidate.validation_status or "") != "current_exact"
        or best_candidate.missing_discriminators
    )

    candidate_codes = [candidate.code for candidate in ordered_candidates]
    legal_result = evaluate_section_232_ruleset(
        selected_code=best_candidate.code if best_candidate is not None else "",
        on_date=_resolve_section_232_on_date(state.get("section_232_on_date")),
        ruleset_store=getattr(section_232_source_store, "ruleset_store", None),
        top_level_grams=dict((state.get("final_composition", {}) or {}).get("top_level_grams") or {}),
    )
    section_payload = build_section_232_ruleset_assessment(
        selected_code=best_candidate.code if best_candidate is not None else "",
        candidate_codes=candidate_codes,
        legal_result=legal_result,
    )
    section_232_reasoner_output = build_section_232_ruleset_reasoner_output(legal_result)
    if legal_coverage_established(legal_result):
        section_payload, metal_weight_override = apply_section_232_weight_override(
            final_composition=dict(state.get("final_composition", {}) or {}),
            diagram_output=dict(state.get("diagram_output", {}) or {}),
            section_payload=section_payload,
        )
    else:
        metal_weight_override = build_skipped_weight_override(
            final_composition=dict(state.get("final_composition", {}) or {}),
            diagram_output=dict(state.get("diagram_output", {}) or {}),
            section_payload=section_payload,
        )
    section_232_reasoner_output["metal_weight_override"] = metal_weight_override

    classification = HTSClassification(
        status="completed" if best_candidate is not None else "unavailable",
        best_candidate=best_candidate,
        candidates=ordered_candidates,
        confidence=_normalize_confidence(selected_confidence),
        reasoning=selected_reasoning,
        needs_human_review=needs_human_review,
    )
    section = Section232Assessment(
        status=str(section_payload.get("status") or "unavailable"),
        decision=str(section_payload.get("decision") or "needs_review"),
        confidence=_normalize_confidence(section_payload.get("confidence")),
        basis_summary=_normalize_text(section_payload.get("basis_summary")),
        needs_human_review=_normalize_bool(section_payload.get("needs_human_review")),
        weight_rule_applied=_normalize_bool(section_payload.get("weight_rule_applied")),
        supporting_hts_candidates=list(section_payload.get("supporting_hts_candidates", []) or []),
        chapter_99_candidates=_normalize_chapter_99_codes(
            list(section_payload.get("chapter_99_candidates", []) or [])
        ),
        evidence=[item for item in section_payload.get("evidence", []) or [] if isinstance(item, dict)],
    )
    resolution_payload = {
        **resolution_output,
        "strategy": "hana_catalog_llm_selector",
        "selector_status": selector_status,
        "selector_model": settings.trade_decision_model_name,
        "selector_reasoning_effort": "medium",
        "selector_payload": selector_payload or {},
        "selector_error": selector_error,
        "selected_code": best_candidate.code if best_candidate is not None else "",
        "selected_confidence": _normalize_confidence(selected_confidence),
        "selected_reasoning": selected_reasoning,
        "needs_human_review": needs_human_review,
    }

    return {
        "hts_classification": classification.model_dump(),
        "section_232_assessment": section.model_dump(),
        "section_232_reasoner_output": section_232_reasoner_output,
        "hts_resolution_output": resolution_payload,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_trade_decision(
    state: Dict[str, Any],
    settings: MetalCompositionSettings,
    llm: LLMClient,
    resolver: HanaHTSCatalogResolver,
    *,
    section_232_source_store: Optional[object] = None,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute the full trade decision pipeline: pool, select, finalize."""
    try:
        candidate_pool, resolution_output = build_catalog_candidate_pool(
            state,
            settings,
            resolver,
        )
    except Exception as exc:  # noqa: BLE001 - keep runtime failure visible in the response
        logger.warning("HTS catalog resolution failed: %s", exc)
        return (
            {
                "hts_classification": HTSClassification(
                    status="unavailable",
                    best_candidate=None,
                    candidates=[],
                    confidence=0.0,
                    reasoning=f"HTS catalog resolution failed: {exc}",
                    needs_human_review=True,
                ).model_dump(),
                "section_232_assessment": Section232Assessment(
                    status="unavailable",
                    decision="needs_review",
                    confidence=0.0,
                    basis_summary="HTS catalog resolution failed before Section 232 screening.",
                    supporting_hts_candidates=[],
                    chapter_99_candidates=[],
                    evidence=[],
                ).model_dump(),
                "section_232_reasoner_output": {
                    "status": "omitted",
                    "strategy": "heuristic_fallback",
                    "fallback_used": True,
                    "reason": "catalog_resolution_failed",
                },
                "hts_resolution_output": {
                    "status": "failed",
                    "strategy": "hana_catalog",
                    "catalog_table": settings.hts_catalog_hana_table,
                    "code_map_table": settings.hts_code_map_hana_table,
                    "error": str(exc),
                    "validated_candidates": [],
                    "rejected_candidates": [],
                },
            },
            {
                "model": settings.trade_decision_model_name,
                "strategy": "llm_catalog_reasoning",
                "fallback_used": True,
                "error": str(exc),
            },
        )
    if not candidate_pool:
        return (
            finalize_trade_decision(
                state=state,
                candidate_pool=candidate_pool,
                resolution_output=resolution_output,
                ordered_codes=[],
                selected_code="",
                selected_reasoning="No validated current HTS candidates were produced from the HANA tree-search evidence.",
                selected_confidence=0.0,
                needs_human_review_hint=True,
                selector_status="omitted",
                settings=settings,
                llm=llm,
                resolver=resolver,
                section_232_source_store=section_232_source_store,
                usage_recorder=usage_recorder,
            ),
            {
                "model": settings.trade_decision_model_name,
                "strategy": "llm_catalog_reasoning",
                "fallback_used": False,
                "candidate_count": 0,
            },
        )

    fb_codes = fallback_trade_decision_codes(candidate_pool)

    try:
        selector_payload = run_trade_decision_selector(
            state=state,
            candidate_pool=candidate_pool,
            resolution_output=resolution_output,
            settings=settings,
            llm=llm,
            usage_recorder=usage_recorder,
        )
        ranking_map = {
            item["code"]: item
            for item in selector_payload.get("candidate_rankings", []) or []
            if item.get("code")
        }
        ordered_codes = _unique_strings(
            [selector_payload["selected_code"]]
            + [item["code"] for item in selector_payload.get("candidate_rankings", []) or [] if item.get("code")]
            + fb_codes
        )

        updated_pool: Dict[str, Dict[str, Any]] = {}
        for code, candidate in candidate_pool.items():
            updated = dict(candidate)
            assessment = ranking_map.get(code)
            if assessment:
                rationale = _normalize_text(assessment.get("rationale"))
                if rationale:
                    updated["reasoning"] = " ".join(
                        _unique_strings([rationale, _normalize_text(updated.get("reasoning"))])
                    ).strip()
                updated["confidence"] = _normalize_confidence(
                    assessment.get("confidence", updated.get("confidence"))
                )
                updated["missing_discriminators"] = _unique_strings(
                    list(updated.get("missing_discriminators", []) or [])
                    + list(assessment.get("missing_discriminators", []) or [])
                )
            updated_pool[code] = updated

        return (
            finalize_trade_decision(
                state=state,
                candidate_pool=updated_pool,
                resolution_output=resolution_output,
                ordered_codes=ordered_codes,
                selected_code=selector_payload["selected_code"],
                selected_reasoning=selector_payload.get("reasoning", ""),
                selected_confidence=selector_payload.get(
                    "confidence",
                    updated_pool.get(selector_payload["selected_code"], {}).get("confidence", 0.0),
                ),
                needs_human_review_hint=selector_payload.get("needs_human_review", False),
                selector_status="completed",
                selector_payload=selector_payload,
                settings=settings,
                llm=llm,
                resolver=resolver,
                section_232_source_store=section_232_source_store,
                usage_recorder=usage_recorder,
            ),
            {
                "model": settings.trade_decision_model_name,
                "strategy": "llm_catalog_reasoning",
                "fallback_used": False,
                "candidate_count": len(candidate_pool),
            },
        )
    except Exception as exc:  # noqa: BLE001 - keep service continuity on selector failures
        logger.warning("Trade decision selector fell back to deterministic output: %s", exc)
        fallback_code = fb_codes[0] if fb_codes else ""
        fallback_reasoning = _normalize_text(
            candidate_pool.get(fallback_code, {}).get("reasoning") if fallback_code else ""
        )
        return (
            finalize_trade_decision(
                state=state,
                candidate_pool=candidate_pool,
                resolution_output=resolution_output,
                ordered_codes=fb_codes,
                selected_code=fallback_code,
                selected_reasoning=fallback_reasoning,
                selected_confidence=_normalize_confidence(
                    candidate_pool.get(fallback_code, {}).get("confidence", 0.0)
                ),
                needs_human_review_hint=True,
                selector_status="failed",
                selector_error=str(exc),
                settings=settings,
                llm=llm,
                resolver=resolver,
                section_232_source_store=section_232_source_store,
                usage_recorder=usage_recorder,
            ),
            {
                "model": settings.trade_decision_model_name,
                "strategy": "llm_catalog_reasoning",
                "fallback_used": True,
                "candidate_count": len(candidate_pool),
                "error": str(exc),
            },
        )
