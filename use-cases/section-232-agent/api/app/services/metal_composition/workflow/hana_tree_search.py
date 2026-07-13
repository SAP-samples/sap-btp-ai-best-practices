"""HANA tree search: catalog-based HTS candidate routing via LLM."""

from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, Dict, List, Optional

from ..config import MetalCompositionSettings
from ..hts_catalog import HanaHTSCatalogResolver
from ..timing import finish_timing, utc_now_iso
from .hts_fact_profile import normalize_tree_search_directives
from .llm import LLMClient
from .normalize import (
    _STANDARD_CUE_RE,
    _extract_json_payload,
    _normalize_candidate_code,
    _normalize_confidence,
    _normalize_float,
    _normalize_heading_hypotheses,
    _normalize_int,
    _normalize_text,
    _response_text,
    _unique_strings,
)
from .original_data import prompt_safe_source_row, prompt_safe_source_summary
from .token_usage import TokenUsageRecorder
from .types import MetalCompositionState

logger = logging.getLogger(__name__)

_MIN_LIKELY_CHAPTERS = 3
_MAX_LIKELY_CHAPTERS = 6
_MAX_HEADINGS_PER_SELECTED_CHAPTER = 10
_MAX_FAMILIES_PER_CONSIDERED_HEADING = 2
_MAX_CHILDREN_PER_EXPLORED_FAMILY = 3
_SMALL_HEADING_FULL_EXPANSION_THRESHOLD = 15
_FULL_RECALL_PER_HEADING = 1000
_FULL_CHILD_RECALL_PER_FAMILY = 1000
_MAX_FAMILY_ROUTER_PROMPT_BYTES = 100_000
_FALLBACK_CANDIDATE_LIMIT = 8


def omitted_hana_tree_search_output(*, reason: str) -> Dict[str, Any]:
    return {
        "status": "omitted",
        "reason": reason,
        "strategy": "hana_tree_llm_router",
        "context": {},
        "chapter_router": {},
        "family_router": {},
        "recall_candidates": [],
        "candidate_suggestions": [],
        "errors": [],
    }


def _heading_hypothesis_chapters(context: Dict[str, Any]) -> List[int]:
    """Return chapter numbers implied by model heading hypotheses and directives.

    Args:
        context: HANA tree search context containing heading hypotheses and
            optional tree-search directives.

    Returns:
        Ordered unique HTS chapter numbers implied by 4-digit headings.
    """

    chapters: List[int] = []
    seen = set()
    directive_headings = [
        directive.get("heading")
        for directive in normalize_tree_search_directives(context.get("tree_search_directives", []))
    ]
    for heading in [*(context.get("heading_hypotheses", []) or []), *directive_headings]:
        heading_text = _normalize_text(heading)
        if not re.fullmatch(r"\d{4}", heading_text):
            continue
        chapter_number = int(heading_text[:2])
        if chapter_number <= 0 or chapter_number in seen:
            continue
        seen.add(chapter_number)
        chapters.append(chapter_number)
    return chapters


def _select_likely_chapter_options(
    *,
    context: Dict[str, Any],
    chapter_options: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    option_by_chapter: Dict[int, Dict[str, Any]] = {}
    for option in chapter_options:
        chapter_number = _normalize_int(option.get("chapter_number"))
        if chapter_number <= 0 or chapter_number in option_by_chapter:
            continue
        option_by_chapter[chapter_number] = option

    selected: List[Dict[str, Any]] = []
    selected_chapters = set()
    for chapter_number in _heading_hypothesis_chapters(context):
        option = option_by_chapter.get(chapter_number)
        if option is None or chapter_number in selected_chapters:
            continue
        selected.append(option)
        selected_chapters.add(chapter_number)
        if len(selected) >= _MAX_LIKELY_CHAPTERS:
            return selected
    if selected:
        return selected[:_MAX_LIKELY_CHAPTERS]

    available_count = len(option_by_chapter)
    target_count = min(_MAX_LIKELY_CHAPTERS, available_count)

    ranked_chapters = sorted(
        enumerate(chapter_options),
        key=lambda indexed: (
            _normalize_float(indexed[1].get("prefilter_score")),
            -indexed[0],
        ),
        reverse=True,
    )
    positive_ranked_chapters = [
        indexed
        for indexed in ranked_chapters
        if _normalize_float(indexed[1].get("prefilter_score")) > 0.0
    ]
    ranked_additions = positive_ranked_chapters
    if not selected and not ranked_additions:
        ranked_additions = ranked_chapters[: min(_MIN_LIKELY_CHAPTERS, available_count)]

    for _index, option in ranked_additions:
        if len(selected) >= target_count:
            break
        chapter_number = _normalize_int(option.get("chapter_number"))
        if chapter_number <= 0 or chapter_number in selected_chapters:
            continue
        selected.append(option)
        selected_chapters.add(chapter_number)

    return selected[:_MAX_LIKELY_CHAPTERS]


def _tree_extra_item_context(
    source_summary: Dict[str, Any],
    source_row: Dict[str, Any],
) -> str:
    """Return extra Material Master item context for HANA tree retrieval prompts.

    Inputs:
        source_summary: Normalized selected-source summary.
        source_row: Original Material Master source row context.

    Expected output:
        The BY Priority duplicate value exposed under a neutral label, or an
        empty string when unavailable.
    """

    safe_summary = prompt_safe_source_summary(source_summary)
    safe_row = prompt_safe_source_row(source_row)
    return _normalize_text(
        safe_summary.get("extra_item_context") or safe_row.get("extra_item_context")
    )


def _prompt_catalog_text(value: Any, *, max_chars: int = 240) -> str:
    """Return compact catalog text for LLM prompts.

    Inputs:
        value: Raw catalog text.
        max_chars: Maximum characters to include.

    Expected output:
        Normalized and length-bounded text suitable for prompt payloads.
    """

    text = _normalize_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _prompt_chapter_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact chapter fields for the family-router prompt.

    Inputs:
        option: HANA chapter catalog option selected by deterministic routing.

    Expected output:
        A compact chapter payload without embedded heading rows, because the
        compact heading list is sent separately.
    """

    return {
        "chapter_number": option.get("chapter_number"),
        "title": _prompt_catalog_text(option.get("title"), max_chars=120),
        "summary": _prompt_catalog_text(option.get("summary"), max_chars=220),
        "prefilter_score": round(_normalize_float(option.get("prefilter_score")), 4),
    }


def _prompt_catalog_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """Return the compact HANA catalog fields needed by the family router.

    Inputs:
        option: Heading, family, or child catalog option.

    Expected output:
        A small JSON-serializable dictionary that keeps code identity,
        path meaning, deterministic score, and matched retrieval terms.
    """

    compact: Dict[str, Any] = {}
    for key in ("code", "digits", "chapter_number", "heading_code", "family_code"):
        if option.get(key) not in (None, ""):
            compact[key] = option.get(key)
    if option.get("description"):
        compact["description"] = _prompt_catalog_text(option.get("description"), max_chars=110)
    if option.get("path_description"):
        compact["path_description"] = _prompt_catalog_text(option.get("path_description"), max_chars=150)
    if option.get("prefilter_score") is not None:
        compact["prefilter_score"] = round(_normalize_float(option.get("prefilter_score")), 4)
    if option.get("recall_source"):
        compact["recall_source"] = _prompt_catalog_text(option.get("recall_source"), max_chars=80)
    if option.get("recall_rank"):
        compact["recall_rank"] = _normalize_int(option.get("recall_rank"))
    matched_terms = [str(value) for value in option.get("matched_terms", []) or [] if str(value).strip()]
    matched_phrases = [str(value) for value in option.get("matched_phrases", []) or [] if str(value).strip()]
    if matched_terms:
        compact["matched_terms"] = matched_terms[:8]
    if matched_phrases:
        compact["matched_phrases"] = matched_phrases[:5]
    return compact


def _prompt_child_options(child_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Return compact child options keyed by family code for LLM prompts.

    Inputs:
        child_map: Catalog-backed children returned by the HANA resolver.

    Expected output:
        A JSON-serializable mapping with only the code, path, score, and match
        evidence required by the family router.
    """

    return {
        family_code: [_prompt_catalog_option(child_option) for child_option in child_options]
        for family_code, child_options in child_map.items()
    }


def _family_router_system_prompt(*, candidate_suggestion_target: int) -> str:
    """Return the shared instruction for HANA family-router calls.

    Args:
        candidate_suggestion_target: Minimum number of candidates the router
            should return when enough selectable catalog options are present.

    Returns:
        System prompt text used for both single-call and packeted family routing.
    """

    target = max(1, int(candidate_suggestion_target or 1))
    return (
        "You are selecting the strongest HANA-backed HTS candidate branches from a real catalog tree. "
        "Choose the best candidate set from family_options and child_options. "
        f"Return at least min({target}, available selectable options) candidate_suggestions. "
        "When fewer than that many options are strong matches, include the closest catalog-backed alternatives "
        "with lower confidence so the final selector can compare them. "
        "Return STRICT JSON only with keys: "
        '"candidate_suggestions", "reasoning", "confidence". '
        '"candidate_suggestions" must be an array of objects with keys: '
        '"code", "rationale", "confidence", "specificity_supported", "missing_discriminators". '
        'Each "confidence" value must be numeric between 0.0 and 1.0, for example "confidence": 0.82, not "confidence": "high". '
        '"specificity_supported" must be 6, 8, or 10. '
        "Use only codes that appear in family_options or child_options. "
        "Select the most specific code (8 or 10 digit) whose description clearly matches the product. "
        "Give preference to candidates with those 8-10 digits unless the 6-digit family is the only clear match. "
        "Fall back to the 6-digit family only when no child description adds meaningful discriminatory value."
    )


def _build_family_router_prompt(
    *,
    product_context: Dict[str, Any],
    selected_chapter_options: List[Dict[str, Any]],
    heading_options: List[Dict[str, Any]],
    family_options: List[Dict[str, Any]],
    child_map: Dict[str, List[Dict[str, Any]]],
    candidate_suggestion_target: int,
) -> Dict[str, Any]:
    """Build a compact family-router prompt payload.

    Args:
        product_context: Product facts and profile directives used by the router.
        selected_chapter_options: HANA chapter options represented in this prompt.
        heading_options: HANA heading options represented in this prompt.
        family_options: Recalled family options represented in this prompt.
        child_map: Recalled child options keyed by family code.
        candidate_suggestion_target: Minimum number of candidate suggestions
            requested from the router when enough options are available.

    Returns:
        JSON-serializable prompt payload for one family-router call.
    """

    return {
        "product_context": product_context,
        "candidate_suggestion_target": max(1, int(candidate_suggestion_target or 1)),
        "considered_chapters": [_prompt_chapter_option(item) for item in selected_chapter_options],
        "heading_options": [_prompt_catalog_option(option) for option in heading_options],
        "family_options": [_prompt_catalog_option(option) for option in family_options],
        "child_options": _prompt_child_options(child_map),
    }


def _json_size_bytes(payload: Dict[str, Any]) -> int:
    """Return the UTF-8 JSON size of a prompt payload.

    Args:
        payload: JSON-serializable prompt payload.

    Returns:
        Byte size of the serialized prompt payload.
    """

    return len(json.dumps(payload, default=str).encode("utf-8"))


def _fallback_confidence_from_retrieval_score(value: Any) -> float:
    """Return conservative confidence for non-model fallback candidates.

    Args:
        value: Raw deterministic retrieval score from HANA prefiltering.

    Returns:
        A bounded confidence in [0.0, 0.4]. Retrieval scores are not calibrated
        probabilities, so fallback candidates must not display as high-confidence
        model decisions.
    """

    score = _normalize_float(value)
    if score <= 0.0:
        return 0.0
    return min(0.4, score / 20.0)


def _candidate_suggestion_from_recall(
    item: Dict[str, Any],
    *,
    rationale: str,
    missing_discriminators: List[str],
) -> Dict[str, Any]:
    """Convert a recall candidate into a fallback candidate suggestion.

    Args:
        item: Catalog-backed recall candidate.
        rationale: Fallback rationale attached to the suggestion.
        missing_discriminators: Missing-evidence notes explaining why fallback was used.

    Returns:
        Candidate suggestion shaped like the normalized family-router output.
    """

    return {
        "hts_code": item["code"],
        "confidence": _fallback_confidence_from_retrieval_score(item.get("prefilter_score", 0.0)),
        "rationale": rationale,
        "specificity_supported": _normalize_int(item.get("digits")) or 6,
        "hana_router_stage": _normalize_text(item.get("hana_router_stage")) or "family",
        "matched_path": _normalize_text(item.get("path_description")),
        "matched_terms": _unique_strings(item.get("matched_terms", []) or []),
        "matched_phrases": _unique_strings(item.get("matched_phrases", []) or []),
        "retrieval_score": _normalize_float(item.get("prefilter_score", 0.0)),
        "recall_source": _normalize_text(item.get("recall_source")),
        "recall_reason": _normalize_text(item.get("recall_reason")),
        "recall_rank": _normalize_int(item.get("recall_rank")),
        "missing_discriminators": list(missing_discriminators),
    }


def _fallback_candidate_suggestions(
    recall_candidates: List[Dict[str, Any]],
    *,
    rationale: str,
    missing_discriminators: List[str],
) -> List[Dict[str, Any]]:
    """Return ranked fallback suggestions from recall candidates.

    Args:
        recall_candidates: Ordered broad recall candidates.
        rationale: Fallback rationale for each suggestion.
        missing_discriminators: Missing-evidence notes attached to each suggestion.

    Returns:
        At most the configured fallback limit of candidate suggestions.
    """

    return [
        _candidate_suggestion_from_recall(
            item,
            rationale=rationale,
            missing_discriminators=missing_discriminators,
        )
        for item in recall_candidates[:_FALLBACK_CANDIDATE_LIMIT]
    ]


def _ensure_candidate_suggestion_floor(
    candidate_suggestions: List[Dict[str, Any]],
    recall_candidates: List[Dict[str, Any]],
    *,
    target_count: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Extend router suggestions with top recall candidates up to target count.

    Args:
        candidate_suggestions: Router-selected candidate suggestions.
        recall_candidates: Ordered catalog-backed recall candidates.
        target_count: Minimum candidate suggestion count after extension.

    Returns:
        Tuple of the extended suggestion list and the number of recall
        candidates added.
    """

    target = max(0, int(target_count or 0))
    if target <= 0:
        return list(candidate_suggestions), 0

    extended = [dict(item) for item in candidate_suggestions]
    seen_codes = {
        _normalize_candidate_code(item.get("hts_code") or item.get("code"))
        for item in extended
        if isinstance(item, dict)
    }
    added_count = 0
    for recall_candidate in recall_candidates:
        if len(extended) >= target:
            break
        code = _normalize_candidate_code(recall_candidate.get("code") or recall_candidate.get("hts_code"))
        if not code or code in seen_codes:
            continue
        extended.append(
            _candidate_suggestion_from_recall(
                recall_candidate,
                rationale="",
                missing_discriminators=[],
            )
        )
        seen_codes.add(code)
        added_count += 1
    return extended, added_count


def _family_router_packets(
    *,
    product_context: Dict[str, Any],
    selected_chapter_options: List[Dict[str, Any]],
    heading_options: List[Dict[str, Any]],
    family_options: List[Dict[str, Any]],
    child_map: Dict[str, List[Dict[str, Any]]],
    recall_candidates: List[Dict[str, Any]],
    full_prompt: Dict[str, Any],
    full_prompt_bytes: int,
    candidate_suggestion_target: int,
) -> List[Dict[str, Any]]:
    """Split large family-router payloads into deterministic same-level packets.

    Args:
        product_context: Product facts and directives shared by all packets.
        selected_chapter_options: Selected HANA chapters in exploration order.
        heading_options: Selected HANA headings in exploration order.
        family_options: Recalled family options in exploration order.
        child_map: Recalled child options keyed by family code.
        recall_candidates: Ordered recall candidates used as packet allow lists.
        full_prompt: Single-call prompt payload.
        full_prompt_bytes: Serialized size of the single-call prompt payload.
        candidate_suggestion_target: Minimum number of candidates requested
            from each router packet when enough selectable options exist.

    Returns:
        One packet for small prompts, or chapter/heading packets for oversized
        prompts. Each packet carries its prompt payload, allowed options, and
        recall candidates for fallback.
    """

    if full_prompt_bytes <= _MAX_FAMILY_ROUTER_PROMPT_BYTES:
        return [
            {
                "packet_id": "all",
                "chapter_number": None,
                "heading_code": None,
                "prompt": full_prompt,
                "prompt_bytes": full_prompt_bytes,
                "allowed_options": recall_candidates,
                "recall_candidates": recall_candidates,
                "candidate_suggestion_target": min(candidate_suggestion_target, len(recall_candidates)),
            }
        ]

    chapter_option_by_number = {
        _normalize_int(option.get("chapter_number")): option
        for option in selected_chapter_options
        if _normalize_int(option.get("chapter_number")) > 0
    }
    chapter_order = [
        number
        for number in (_normalize_int(option.get("chapter_number")) for option in selected_chapter_options)
        if number in chapter_option_by_number
    ]

    headings_by_chapter: Dict[int, List[Dict[str, Any]]] = {number: [] for number in chapter_order}
    for heading in heading_options:
        chapter_number = _normalize_int(heading.get("chapter_number"))
        if chapter_number in headings_by_chapter:
            headings_by_chapter[chapter_number].append(heading)

    families_by_chapter: Dict[int, List[Dict[str, Any]]] = {number: [] for number in chapter_order}
    for family in family_options:
        chapter_number = _normalize_int(family.get("chapter_number"))
        if chapter_number in families_by_chapter:
            families_by_chapter[chapter_number].append(family)

    recall_by_chapter: Dict[int, List[Dict[str, Any]]] = {number: [] for number in chapter_order}
    for candidate in recall_candidates:
        chapter_number = _normalize_int(candidate.get("chapter_number"))
        if chapter_number in recall_by_chapter:
            recall_by_chapter[chapter_number].append(candidate)

    packets: List[Dict[str, Any]] = []
    for chapter_number in chapter_order:
        chapter_family_options = families_by_chapter.get(chapter_number, [])
        chapter_recall_candidates = recall_by_chapter.get(chapter_number, [])
        if not chapter_family_options or not chapter_recall_candidates:
            continue

        family_codes = {str(item.get("code")) for item in chapter_family_options if item.get("code")}
        chapter_child_map = {
            family_code: child_map.get(family_code, [])
            for family_code in family_codes
            if child_map.get(family_code)
        }
        chapter_prompt = _build_family_router_prompt(
            product_context=product_context,
            selected_chapter_options=[chapter_option_by_number[chapter_number]],
            heading_options=headings_by_chapter.get(chapter_number, []),
            family_options=chapter_family_options,
            child_map=chapter_child_map,
            candidate_suggestion_target=min(candidate_suggestion_target, len(chapter_recall_candidates)),
        )
        chapter_prompt_bytes = _json_size_bytes(chapter_prompt)
        if chapter_prompt_bytes <= _MAX_FAMILY_ROUTER_PROMPT_BYTES:
            packets.append(
                {
                    "packet_id": f"chapter-{chapter_number}",
                    "chapter_number": chapter_number,
                    "heading_code": None,
                    "prompt": chapter_prompt,
                    "prompt_bytes": chapter_prompt_bytes,
                    "allowed_options": chapter_recall_candidates,
                    "recall_candidates": chapter_recall_candidates,
                    "candidate_suggestion_target": min(candidate_suggestion_target, len(chapter_recall_candidates)),
                }
            )
            continue

        for heading in headings_by_chapter.get(chapter_number, []):
            heading_code = str(heading.get("heading_code") or "")
            heading_family_options = [
                item for item in chapter_family_options if str(item.get("heading_code") or "") == heading_code
            ]
            heading_recall_candidates = [
                item for item in chapter_recall_candidates if str(item.get("heading_code") or "") == heading_code
            ]
            if not heading_family_options or not heading_recall_candidates:
                continue

            heading_family_codes = {str(item.get("code")) for item in heading_family_options if item.get("code")}
            heading_child_map = {
                family_code: child_map.get(family_code, [])
                for family_code in heading_family_codes
                if child_map.get(family_code)
            }
            heading_prompt = _build_family_router_prompt(
                product_context=product_context,
                selected_chapter_options=[chapter_option_by_number[chapter_number]],
                heading_options=[heading],
                family_options=heading_family_options,
                child_map=heading_child_map,
                candidate_suggestion_target=min(candidate_suggestion_target, len(heading_recall_candidates)),
            )
            packets.append(
                {
                    "packet_id": f"chapter-{chapter_number}-heading-{heading_code}",
                    "chapter_number": chapter_number,
                    "heading_code": heading_code,
                    "prompt": heading_prompt,
                    "prompt_bytes": _json_size_bytes(heading_prompt),
                    "allowed_options": heading_recall_candidates,
                    "recall_candidates": heading_recall_candidates,
                    "candidate_suggestion_target": min(candidate_suggestion_target, len(heading_recall_candidates)),
                }
            )

    return packets or [
        {
            "packet_id": "all",
            "chapter_number": None,
            "heading_code": None,
            "prompt": full_prompt,
            "prompt_bytes": full_prompt_bytes,
            "allowed_options": recall_candidates,
            "recall_candidates": recall_candidates,
            "candidate_suggestion_target": min(candidate_suggestion_target, len(recall_candidates)),
        }
    ]


def _catalog_path_segments(option: Dict[str, Any]) -> List[str]:
    """Return normalized catalog path segments for branch-shape checks.

    Args:
        option: Catalog row option with an optional path description.

    Returns:
        Lowercase path segments with punctuation trimmed from each branch label.
    """

    path = _normalize_text(option.get("path_description"))
    return [segment.strip(" :").lower() for segment in path.split(">") if segment.strip(" :")]


def _is_parts_branch(option: Dict[str, Any]) -> bool:
    """Return whether a catalog option represents a parts branch below a heading.

    Args:
        option: Family or child catalog option.

    Returns:
        True when the option's own branch label, not only the parent heading,
        indicates parts/parts thereof/parts and accessories.
    """

    description = _normalize_text(option.get("description")).strip(" :").lower()
    if description in {"parts", "parts thereof", "parts and accessories", "other parts"}:
        return True

    segments = _catalog_path_segments(option)
    branch_segments = segments[1:] if len(segments) > 1 else segments
    return any(
        segment in {"parts", "parts thereof", "parts and accessories", "other parts"}
        or segment.startswith("parts ")
        for segment in branch_segments
    )


def _is_residual_other_option(option: Dict[str, Any]) -> bool:
    """Return whether an option is an HTS residual Other/Other parts leaf.

    Args:
        option: Family or child catalog option.

    Returns:
        True when the option ends in a residual other branch, such as
        "Other", "Other parts", or "Other > Other".
    """

    description = _normalize_text(option.get("description")).strip(" :").lower()
    segments = _catalog_path_segments(option)
    branch_segments = segments[1:] if len(segments) > 1 else segments
    last_segment = branch_segments[-1] if branch_segments else ""
    if description in {"other", "other parts"}:
        return True
    if last_segment in {"other", "other parts"}:
        return True
    return len(branch_segments) >= 2 and branch_segments[-2:] == ["other", "other"]


def _directives_for_heading(context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Group normalized tree-search directives by heading.

    Args:
        context: HANA tree search context built from the workflow state.

    Returns:
        Mapping of 4-digit heading to normalized directives.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for directive in normalize_tree_search_directives(context.get("tree_search_directives", [])):
        grouped.setdefault(str(directive["heading"]), []).append(directive)
    return grouped


def _matching_directive(
    option: Dict[str, Any],
    directives_by_heading: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Return the first directive that asks recall to include this option.

    Args:
        option: Catalog option under consideration.
        directives_by_heading: Normalized directives keyed by heading.

    Returns:
        The matching directive, or None when no directive applies.
    """

    heading = str(option.get("heading_code") or "")
    for directive in directives_by_heading.get(heading, []):
        focus = set(directive.get("branch_focus", []) or [])
        if "parts" in focus and _is_parts_branch(option):
            return directive
        if "residual_other" in focus and _is_residual_other_option(option):
            return directive
        if "complete_articles" in focus and not _is_parts_branch(option):
            return directive
        if "material_articles" in focus and int(option.get("chapter_number") or 0) in {72, 73, 76, 74, 75}:
            return directive
    return None


def _with_recall_metadata(
    option: Dict[str, Any],
    *,
    source: str,
    reason: str,
    stage: str,
) -> Dict[str, Any]:
    """Return a catalog option annotated for recall diagnostics.

    Args:
        option: Catalog option selected for recall.
        source: Machine-readable source explaining why recall kept the option.
        reason: Human-readable explanation for diagnostics and prompts.
        stage: HANA router stage, either family or child.

    Returns:
        A shallow copy of the option with recall metadata attached.
    """

    return {
        **option,
        "hana_router_stage": stage,
        "recall_source": source,
        "recall_reason": reason,
        "retrieval_score": _normalize_float(option.get("prefilter_score")),
    }


def _select_family_options_for_recall(
    *,
    all_family_options: List[Dict[str, Any]],
    selected_headings: List[str],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Select family options for broad recall before model decision ranking.

    Args:
        all_family_options: All resolver-returned family options for selected headings.
        selected_headings: Heading order chosen by deterministic chapter/heading recall.
        context: HANA tree search context with optional model directives.

    Returns:
        Family options ordered by selected heading order and annotated with recall
        metadata. Small headings are fully retained, while larger headings keep
        lexical top options plus directive-matching branches.
    """

    directives_by_heading = _directives_for_heading(context)
    grouped: Dict[str, List[Dict[str, Any]]] = {heading: [] for heading in selected_headings}
    for option in all_family_options:
        heading = str(option.get("heading_code") or "")
        if heading in grouped:
            grouped[heading].append(option)

    selected: List[Dict[str, Any]] = []
    for heading in selected_headings:
        heading_options = grouped.get(heading, [])
        if not heading_options:
            continue

        selected_by_code: Dict[str, Dict[str, Any]] = {}
        heading_is_small = len(heading_options) < _SMALL_HEADING_FULL_EXPANSION_THRESHOLD
        for index, option in enumerate(heading_options):
            directive = _matching_directive(option, directives_by_heading)
            include_for_lexical_rank = index < _MAX_FAMILIES_PER_CONSIDERED_HEADING
            if not (heading_is_small or include_for_lexical_rank or directive):
                continue

            if directive:
                source = "profile_directive"
                reason = _normalize_text(directive.get("rationale")) or (
                    f"Included because HTS profile directed exploration under heading {heading}."
                )
            elif heading_is_small:
                source = "small_heading_full_expansion"
                reason = (
                    f"Included because heading {heading} has {len(heading_options)} family options, "
                    f"below the {_SMALL_HEADING_FULL_EXPANSION_THRESHOLD} option full-expansion threshold."
                )
            else:
                source = "lexical_rank"
                reason = (
                    f"Included as a top lexical family under heading {heading} before family-router selection."
                )

            code = str(option.get("code"))
            selected_by_code[code] = _with_recall_metadata(
                option,
                source=source,
                reason=reason,
                stage="family",
            )

        selected.extend(selected_by_code.values())
    return selected


def _select_child_options_for_recall(
    *,
    resolver: HanaHTSCatalogResolver,
    family_options: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Expand child options while preserving residual children below parts branches.

    Args:
        resolver: HANA catalog resolver.
        family_options: Recalled family options.
        context: HANA tree search context.

    Returns:
        Child options keyed by family code, with top lexical children retained for
        high-priority lexical/directive families and residual Other/Other parts
        children additionally retained under explored parts branches.
    """

    family_codes = [
        str(item["code"])
        for item in family_options
        if str(item.get("recall_source") or "") in {"lexical_rank", "profile_directive"}
    ]
    base_child_map = resolver.expand_children(
        family_codes,
        context,
        per_family=_MAX_CHILDREN_PER_EXPLORED_FAMILY,
    )
    parts_family_codes = [str(item["code"]) for item in family_options if _is_parts_branch(item)]
    residual_child_map = (
        resolver.expand_children(
            parts_family_codes,
            context,
            per_family=_FULL_CHILD_RECALL_PER_FAMILY,
        )
        if parts_family_codes
        else {}
    )

    selected_child_map: Dict[str, List[Dict[str, Any]]] = {}
    for family_option in family_options:
        family_code = str(family_option["code"])
        selected_by_code: Dict[str, Dict[str, Any]] = {}

        for child in base_child_map.get(family_code, []) or []:
            selected_by_code[str(child["code"])] = _with_recall_metadata(
                child,
                source="lexical_rank",
                reason=f"Included as a top lexical child under family {family_code}.",
                stage="child",
            )

        if _is_parts_branch(family_option):
            for child in residual_child_map.get(family_code, []) or []:
                if not _is_residual_other_option(child):
                    continue
                selected_by_code[str(child["code"])] = _with_recall_metadata(
                    child,
                    source="parts_residual_child",
                    reason=(
                        f"Included because family {family_code} is a parts branch and this child is a "
                        "residual Other/Other parts provision."
                    ),
                    stage="child",
                )

        selected_child_map[family_code] = list(selected_by_code.values())
    return selected_child_map


def _build_recall_candidates(
    *,
    family_options: List[Dict[str, Any]],
    child_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build the broad catalog-backed candidate list used for diagnostics/fallback.

    Args:
        family_options: Recalled family-level catalog options.
        child_map: Recalled child catalog options keyed by family code.

    Returns:
        Ordered recall candidates with global recall ranks and diagnostic fields.
    """

    candidates: List[Dict[str, Any]] = []
    seen_codes = set()
    for family in family_options:
        code = str(family.get("code") or "")
        if code and code not in seen_codes:
            seen_codes.add(code)
            candidates.append(dict(family))
        for child in child_map.get(code, []) or []:
            child_code = str(child.get("code") or "")
            if not child_code or child_code in seen_codes:
                continue
            seen_codes.add(child_code)
            candidates.append(dict(child))

    for index, candidate in enumerate(candidates, start=1):
        candidate["recall_rank"] = index
    return candidates


def build_hana_tree_search_context(state: MetalCompositionState) -> Dict[str, Any]:
    source_summary = dict(state.get("source_summary", {}) or {})
    source_row = dict(state.get("source_row", {}) or {})
    extra_item_context = _tree_extra_item_context(source_summary, source_row)
    hts_fact_profile = dict(state.get("hts_fact_profile", {}) or {})
    final_composition = dict(state.get("final_composition", {}) or {})
    material_profile = dict(hts_fact_profile.get("material_profile", {}) or final_composition)

    raw_text_values = [
        source_summary.get("part_description"),
        source_summary.get("new_part_description"),
        source_summary.get("pn_revised_standardized"),
        source_row.get("Part description"),
        source_row.get("New Part Description"),
        extra_item_context,
        hts_fact_profile.get("article_summary"),
        hts_fact_profile.get("function_summary"),
    ]
    phrases = _unique_strings(raw_text_values)
    standard_cues = _unique_strings(
        re.findall(
            _STANDARD_CUE_RE,
            " ".join(_normalize_text(value) for value in raw_text_values if _normalize_text(value)),
        )
    )
    material_clues: List[str] = []
    top_level = dict(material_profile.get("top_level_grams", {}) or {})
    steel_subtypes = dict(material_profile.get("steel_subtype_grams", {}) or {})
    for label, grams in top_level.items():
        if float(grams or 0.0) > 0.0:
            material_clues.append(label.replace("_", " "))
    for label, grams in steel_subtypes.items():
        if float(grams or 0.0) > 0.0:
            material_clues.append(label.replace("_", " "))

    tree_search_directives = normalize_tree_search_directives(
        hts_fact_profile.get("tree_search_directives", []) or []
    )
    heading_hypotheses = _normalize_heading_hypotheses(
        [
            *(hts_fact_profile.get("heading_hypotheses", []) or []),
            *(directive.get("heading") for directive in tree_search_directives),
        ]
    )
    tokens = _unique_strings(
        [
            token
            for value in [*phrases, *material_clues, *standard_cues]
            for token in re.findall(r"[A-Za-z0-9#./-]+", _normalize_text(value).lower())
            if len(token) >= 2
        ]
    )

    return {
        "product_code": state.get("product_code"),
        "article_summary": _normalize_text(hts_fact_profile.get("article_summary")),
        "function_summary": _normalize_text(hts_fact_profile.get("function_summary")),
        "part_description": _normalize_text(
            source_summary.get("part_description") or source_row.get("Part description")
        ),
        "new_part_description": _normalize_text(
            source_summary.get("new_part_description") or source_row.get("New Part Description")
        ),
        "extra_item_context": extra_item_context,
        "material_clues": material_clues,
        "standard_cues": standard_cues,
        "heading_hypotheses": heading_hypotheses,
        "tree_search_directives": tree_search_directives,
        "discriminator_notes": _unique_strings(hts_fact_profile.get("discriminator_notes", []) or []),
        "phrases": phrases,
        "tokens": tokens,
    }


def normalize_hana_chapter_router_payload(
    payload: Dict[str, Any],
    *,
    chapter_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    option_map = {int(item["chapter_number"]): item for item in chapter_options}
    selected: List[Dict[str, Any]] = []
    seen = set()
    for raw_item in payload.get("selected_chapters", []) or []:
        chapter_number = _normalize_int(
            raw_item.get("chapter_number") if isinstance(raw_item, dict) else raw_item
        )
        if chapter_number <= 0 or chapter_number not in option_map or chapter_number in seen:
            continue
        seen.add(chapter_number)
        rationale = _normalize_text(raw_item.get("rationale") if isinstance(raw_item, dict) else "")
        selected.append(
            {
                "chapter_number": chapter_number,
                "title": option_map[chapter_number]["title"],
                "summary": option_map[chapter_number]["summary"],
                "rationale": rationale,
                "confidence": _normalize_confidence(raw_item.get("confidence") if isinstance(raw_item, dict) else 0.0),
                "prefilter_score": option_map[chapter_number]["prefilter_score"],
            }
        )
    if not selected:
        raise ValueError("Chapter router did not return any selectable chapters.")

    exclusions: List[Dict[str, Any]] = []
    excluded_map = payload.get("excluded_chapters", []) or []
    for raw_item in excluded_map[:4]:
        chapter_number = _normalize_int(
            raw_item.get("chapter_number") if isinstance(raw_item, dict) else raw_item
        )
        if chapter_number <= 0 or chapter_number not in option_map:
            continue
        exclusions.append(
            {
                "chapter_number": chapter_number,
                "title": option_map[chapter_number]["title"],
                "rationale": _normalize_text(raw_item.get("rationale") if isinstance(raw_item, dict) else ""),
            }
        )

    return {
        "selected_chapters": selected,
        "excluded_chapters": exclusions,
        "reasoning": _normalize_text(payload.get("reasoning")),
        "confidence": _normalize_confidence(payload.get("confidence")),
    }


def normalize_hana_family_router_payload(
    payload: Dict[str, Any],
    *,
    allowed_options: List[Dict[str, Any]],
) -> Dict[str, Any]:
    option_map = {str(item["code"]): item for item in allowed_options}
    selected: List[Dict[str, Any]] = []
    seen = set()
    for raw_item in payload.get("candidate_suggestions", []) or []:
        code = _normalize_candidate_code(raw_item.get("code") if isinstance(raw_item, dict) else raw_item)
        if not code or code not in option_map or code in seen:
            continue
        seen.add(code)
        option = option_map[code]
        specificity_supported = _normalize_int(
            raw_item.get("specificity_supported") if isinstance(raw_item, dict) else option["digits"]
        )
        if specificity_supported not in {6, 8, 10}:
            specificity_supported = int(option["digits"])
        specificity_supported = max(specificity_supported, int(option["digits"]))
        selected.append(
            {
                "hts_code": code,
                "confidence": _normalize_confidence(raw_item.get("confidence") if isinstance(raw_item, dict) else 0.0),
                "rationale": _normalize_text(raw_item.get("rationale") if isinstance(raw_item, dict) else ""),
                "specificity_supported": specificity_supported,
                "hana_router_stage": str(option.get("hana_router_stage") or "family"),
                "matched_path": _normalize_text(option.get("path_description")),
                "matched_terms": _unique_strings(option.get("matched_terms", []) or []),
                "matched_phrases": _unique_strings(option.get("matched_phrases", []) or []),
                "retrieval_score": _normalize_float(option.get("prefilter_score")),
                "recall_source": _normalize_text(option.get("recall_source")),
                "recall_reason": _normalize_text(option.get("recall_reason")),
                "recall_rank": _normalize_int(option.get("recall_rank")),
                "missing_discriminators": _unique_strings(
                    raw_item.get("missing_discriminators", []) if isinstance(raw_item, dict) else []
                ),
            }
        )
    if not selected:
        raise ValueError("Family router did not return any selectable HANA candidates.")

    return {
        "candidate_suggestions": selected,
        "reasoning": _normalize_text(payload.get("reasoning")),
        "confidence": _normalize_confidence(payload.get("confidence")),
    }


def run_hana_tree_search(
    state: MetalCompositionState,
    settings: MetalCompositionSettings,
    llm: LLMClient,
    resolver: HanaHTSCatalogResolver,
    *,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    started_perf = perf_counter()
    started_at = utc_now_iso()
    context = build_hana_tree_search_context(state)
    model_name = settings.hana_tree_router_model_name
    errors: List[str] = []

    chapter_options = resolver.list_chapter_options(context)
    selected_chapter_options = _select_likely_chapter_options(
        context=context,
        chapter_options=chapter_options,
    )
    selected_chapter_numbers = [item["chapter_number"] for item in selected_chapter_options]
    heading_options = resolver.list_heading_options(
        selected_chapter_numbers,
        context,
        per_chapter=_MAX_HEADINGS_PER_SELECTED_CHAPTER,
    )

    product_context = {
        "product_code": state.get("product_code"),
        "article_summary": context.get("article_summary"),
        "function_summary": context.get("function_summary"),
        "part_description": context.get("part_description"),
        "new_part_description": context.get("new_part_description"),
        "extra_item_context": context.get("extra_item_context"),
        "material_clues": context.get("material_clues", []),
        "standard_cues": context.get("standard_cues", []),
        "heading_hypotheses": context.get("heading_hypotheses", []),
        "tree_search_directives": context.get("tree_search_directives", []),
        "discriminator_notes": context.get("discriminator_notes", []),
    }

    heading_by_chapter: Dict[int, List[Dict[str, Any]]] = {}
    for heading in heading_options:
        heading_by_chapter.setdefault(heading["chapter_number"], []).append(heading)

    chapter_groups = [
        {
            "chapter_number": ch["chapter_number"],
            "title": ch["title"],
            "summary": ch["summary"],
            "prefilter_score": ch["prefilter_score"],
            "headings": heading_by_chapter.get(ch["chapter_number"], []),
        }
        for ch in selected_chapter_options
        if heading_by_chapter.get(ch["chapter_number"])
    ]
    sent_chapter_numbers = [item["chapter_number"] for item in chapter_groups]
    routing_diagnostics = {
        "available_chapter_count": len(chapter_options),
        "selected_chapters": selected_chapter_numbers,
        "sent_chapters": sent_chapter_numbers,
        "chapter_options_sent_count": len(chapter_groups),
        "heading_options_sent_count": len(heading_options),
        "max_chapters": _MAX_LIKELY_CHAPTERS,
        "max_headings_per_chapter": _MAX_HEADINGS_PER_SELECTED_CHAPTER,
    }

    selected_headings = list(dict.fromkeys(str(item["heading_code"]) for item in heading_options))
    chapter_option_map = {int(ch["chapter_number"]): ch for ch in selected_chapter_options}
    chapter_router = {
        "status": "completed",
        "fallback_used": False,
        "selected_chapters": [
            {
                "chapter_number": cn,
                "title": chapter_option_map[cn]["title"],
                "summary": chapter_option_map[cn]["summary"],
                "rationale": "Deterministically selected from HANA chapter prefilter; heading selection is deferred to family expansion.",
                "confidence": _normalize_confidence(chapter_option_map[cn]["prefilter_score"]),
                "prefilter_score": chapter_option_map[cn]["prefilter_score"],
            }
            for cn in sent_chapter_numbers
            if cn in chapter_option_map
        ],
        "reasoning": "Heading-router LLM selection was skipped; all top prefiltered headings in the considered chapters were expanded.",
        "confidence": _normalize_confidence(
            max((float(chapter_option_map[cn]["prefilter_score"]) for cn in sent_chapter_numbers), default=0.0)
        ),
        "model": model_name,
    }

    all_family_options = resolver.list_family_options(
        selected_headings,
        context,
        per_heading=_FULL_RECALL_PER_HEADING,
    )
    family_options = _select_family_options_for_recall(
        all_family_options=all_family_options,
        selected_headings=selected_headings,
        context=context,
    )
    child_map = _select_child_options_for_recall(
        resolver=resolver,
        family_options=family_options,
        context=context,
    )
    recall_candidates = _build_recall_candidates(
        family_options=family_options,
        child_map=child_map,
    )
    candidate_suggestion_target = min(
        max(1, int(settings.hts_k_candidates or 1)),
        len(recall_candidates),
    )
    family_prompt = _build_family_router_prompt(
        product_context=product_context,
        selected_chapter_options=selected_chapter_options,
        heading_options=heading_options,
        family_options=family_options,
        child_map=child_map,
        candidate_suggestion_target=candidate_suggestion_target,
    )
    family_prompt_bytes = _json_size_bytes(family_prompt)
    family_router_packets = _family_router_packets(
        product_context=product_context,
        selected_chapter_options=selected_chapter_options,
        heading_options=heading_options,
        family_options=family_options,
        child_map=child_map,
        recall_candidates=recall_candidates,
        full_prompt=family_prompt,
        full_prompt_bytes=family_prompt_bytes,
        candidate_suggestion_target=candidate_suggestion_target,
    )

    family_router_call_results: List[Dict[str, Any]] = []
    raw_family_router_responses: List[str] = []
    family_router_reasoning: List[str] = []
    family_router_confidences: List[float] = []
    candidate_suggestions: List[Dict[str, Any]] = []
    seen_candidate_codes = set()
    fallback_used = False
    packet_errors: List[str] = []
    failed_packet_fallbacks: List[List[Dict[str, Any]]] = []

    for packet in family_router_packets:
        packet_id = str(packet["packet_id"])
        try:
            response = llm.invoke_native_chat_completion(
                model_name=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": _family_router_system_prompt(
                            candidate_suggestion_target=int(packet.get("candidate_suggestion_target") or 1)
                        ),
                    },
                    {"role": "user", "content": f"Decision input:\n{json.dumps(packet['prompt'], default=str)}"},
                ],
                reasoning_effort="low",
                thinking_type=settings.hana_tree_router_reasoning_type,
                thinking_budget=settings.hana_tree_router_reasoning_budget,
                phase="hana_tree_search",
                task="family_router",
                usage_recorder=usage_recorder,
            )
            raw_text = _response_text(response.choices[0].message.content)
            packet_router = normalize_hana_family_router_payload(
                _extract_json_payload(raw_text),
                allowed_options=list(packet["allowed_options"]),
            )
            raw_family_router_responses.append(raw_text)
            family_router_reasoning.append(packet_router["reasoning"])
            family_router_confidences.append(_normalize_confidence(packet_router["confidence"]))
            for suggestion in packet_router["candidate_suggestions"]:
                code = str(suggestion.get("hts_code") or "")
                if not code or code in seen_candidate_codes:
                    continue
                seen_candidate_codes.add(code)
                candidate_suggestions.append(suggestion)
            family_router_call_results.append(
                {
                    "status": "completed",
                    "fallback_used": False,
                    "packet_id": packet_id,
                    "chapter_number": packet.get("chapter_number"),
                    "heading_code": packet.get("heading_code"),
                    "prompt_bytes": packet["prompt_bytes"],
                    "candidate_count": len(packet_router["candidate_suggestions"]),
                    "reasoning": packet_router["reasoning"],
                    "confidence": packet_router["confidence"],
                    "raw_model_response": raw_text,
                }
            )
        except Exception as exc:  # noqa: BLE001
            error_text = f"family_router[{packet_id}]: {exc}"
            errors.append(error_text)
            packet_errors.append(error_text)
            packet_fallback = _fallback_candidate_suggestions(
                list(packet["recall_candidates"]),
                rationale="Fallback retained a ranked HANA recall candidate after family-router failure.",
                missing_discriminators=[
                    "Family router failed; continuing with ranked HANA recall candidates."
                ],
            )
            failed_packet_fallbacks.append(packet_fallback)
            family_router_call_results.append(
                {
                    "status": "failed",
                    "fallback_used": False,
                    "packet_id": packet_id,
                    "chapter_number": packet.get("chapter_number"),
                    "heading_code": packet.get("heading_code"),
                    "prompt_bytes": packet["prompt_bytes"],
                    "candidate_count": 0,
                    "fallback_candidate_count": len(packet_fallback),
                    "error": str(exc),
                }
            )

    all_packets_failed = bool(family_router_packets) and len(packet_errors) == len(family_router_packets)
    if all_packets_failed:
        fallback_used = True
        for packet_fallback in failed_packet_fallbacks:
            for suggestion in packet_fallback:
                code = str(suggestion.get("hts_code") or "")
                if not code or code in seen_candidate_codes:
                    continue
                seen_candidate_codes.add(code)
                candidate_suggestions.append(suggestion)
        for call_result in family_router_call_results:
            if call_result.get("status") == "failed":
                call_result["fallback_used"] = True
                call_result["candidate_count"] = call_result.get("fallback_candidate_count", 0)

    candidate_suggestions, candidate_floor_added_count = _ensure_candidate_suggestion_floor(
        candidate_suggestions,
        recall_candidates,
        target_count=candidate_suggestion_target,
    )
    router_status = "failed" if all_packets_failed else "partial" if packet_errors else "completed"
    router_confidence = max(family_router_confidences, default=0.0)
    router_reasoning = (
        " ".join(reason for reason in family_router_reasoning if reason).strip()
        or "Family router failed; continuing with ranked HANA recall candidates."
    )
    if len(family_router_packets) > 1:
        router_reasoning = (
            f"Merged {len(family_router_packets)} packeted family-router calls. "
            f"{router_reasoning}"
        )

    family_router = {
        "status": router_status,
        "fallback_used": fallback_used,
        "candidate_suggestions": candidate_suggestions,
        "reasoning": router_reasoning,
        "confidence": router_confidence,
        "model": model_name,
        "prompt_bytes": family_prompt_bytes,
        "packeted": len(family_router_packets) > 1,
        "packet_count": len(family_router_packets),
        "max_packet_prompt_bytes": max((int(packet["prompt_bytes"]) for packet in family_router_packets), default=0),
        "candidate_suggestion_target": candidate_suggestion_target,
        "candidate_suggestion_floor_added_count": candidate_floor_added_count,
        "router_calls": family_router_call_results,
    }
    if raw_family_router_responses and len(family_router_packets) == 1:
        family_router["raw_model_response"] = raw_family_router_responses[0]
    if packet_errors:
        family_router["error"] = "; ".join(packet_errors)

    status = "completed" if candidate_suggestions else "failed"
    routing_diagnostics.update(
        {
            "heading_options_considered_count": len(heading_options),
            "family_option_count": len(family_options),
            "all_family_option_count": len(all_family_options),
            "child_family_count": len(child_map),
            "recall_candidate_count": len(recall_candidates),
            "candidate_count": len(candidate_suggestions),
            "candidate_suggestion_target": candidate_suggestion_target,
            "candidate_suggestion_floor_added_count": candidate_floor_added_count,
            "max_families_per_considered_heading": _MAX_FAMILIES_PER_CONSIDERED_HEADING,
            "max_children_per_explored_family": _MAX_CHILDREN_PER_EXPLORED_FAMILY,
            "small_heading_full_expansion_threshold": _SMALL_HEADING_FULL_EXPANSION_THRESHOLD,
            "family_prompt_bytes": family_prompt_bytes,
            "family_router_packet_count": len(family_router_packets),
            "family_router_max_prompt_bytes": family_router["max_packet_prompt_bytes"],
            "family_router_prompt_byte_budget": _MAX_FAMILY_ROUTER_PROMPT_BYTES,
        }
    )
    output = {
        "status": status,
        "strategy": "hana_tree_llm_router",
        "context": context,
        "chapter_router": chapter_router,
        "family_router": family_router,
        "candidate_suggestions": candidate_suggestions,
        "routing_diagnostics": routing_diagnostics,
        "recall_candidates": recall_candidates,
        "family_options_sent": family_options,
        "child_options_sent": child_map,
        "errors": errors,
        "model": model_name,
        "timing": finish_timing(
            started_perf,
            started_at,
            status=status,
            details={
                "model": model_name,
                "heading_router_skipped": True,
                "available_chapter_count": len(chapter_options),
                "chapter_options_sent_count": len(chapter_groups),
                "heading_options_considered_count": len(heading_options),
                "family_option_count": len(family_options),
                "all_family_option_count": len(all_family_options),
                "child_family_count": len(child_map),
                "recall_candidate_count": len(recall_candidates),
                "candidate_count": len(candidate_suggestions),
                "candidate_suggestion_target": candidate_suggestion_target,
                "candidate_suggestion_floor_added_count": candidate_floor_added_count,
                "max_families_per_considered_heading": _MAX_FAMILIES_PER_CONSIDERED_HEADING,
                "max_children_per_explored_family": _MAX_CHILDREN_PER_EXPLORED_FAMILY,
                "small_heading_full_expansion_threshold": _SMALL_HEADING_FULL_EXPANSION_THRESHOLD,
                "family_prompt_bytes": family_prompt_bytes,
                "family_router_packet_count": len(family_router_packets),
                "family_router_max_prompt_bytes": family_router["max_packet_prompt_bytes"],
                "family_router_prompt_byte_budget": _MAX_FAMILY_ROUTER_PROMPT_BYTES,
            },
        ),
    }
    if not candidate_suggestions:
        output["reason"] = "HANA tree routing did not produce any candidate suggestions."
    return output
