"""Section 232 notice interpretation into draft rule candidates."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .hts_catalog import canonicalize_hts_code
from .section_232_rulesets import Section232DraftRuleCandidate, _METAL_SCOPE_TOKEN_RE
from .section_232_sources import PersistedSection232Source, extract_hts_mentions
from .workflow.normalize import (
    _extract_json_payload,
    _normalize_bool,
    _normalize_confidence,
    _normalize_text,
    _response_text,
)
from .workflow.token_usage import TokenUsageRecorder

_RULE_TYPE_ALIASES = {
    "include": "include",
    "included": "include",
    "add": "include",
    "added": "include",
    "addition": "include",
    "eligible": "include",
    "remove": "remove",
    "removed": "remove",
    "removal": "remove",
    "exclude": "remove",
    "excluded": "remove",
    "no_longer_eligible": "remove",
    "not_eligible": "remove",
    "rate_schedule": "rate_schedule",
    "reduced_rate": "rate_schedule",
    "temporary_reduced_rate": "rate_schedule",
    "temporary_rate": "rate_schedule",
    "tariff": "rate_schedule",
    "tariff_schedule": "rate_schedule",
    "duty": "rate_schedule",
    "additional_duty": "rate_schedule",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
)
_EXCLUSIONARY_SCOPE_RE = re.compile(
    r"\b(?:not|except|excluding|excluded|without|other\s+than|minus|but\s+not|save)\b",
    re.IGNORECASE,
)
_ADMIN_CONTEXT_RE = re.compile(
    r"\b(?:docket|xrin|fr doc|billing code|federal register|vol\.|department of commerce|bureau of industry)\b",
    re.IGNORECASE,
)
_DIGITS_ONLY_RE = re.compile(r"^\d{6,10}$")
_SUBDIVISION_BATCH_SIZE = 40
_MAX_EXAMPLES_PER_SEED = 5
_MAX_MATCHED_TEXTS_PER_SEED = 3
_MAX_CONTEXT_SNIPPET_CHARS = 160


@dataclass(frozen=True)
class Section232CandidateSeed:
    candidate_key: str
    source_id: str
    filename: str
    normalized_hts_code: str
    source_pages: List[int]
    matched_texts: List[str] = field(default_factory=list)
    context_snippets: List[str] = field(default_factory=list)
    representative_excerpt: str = ""
    candidate_quality: str = "normal"
    candidate_flags: List[str] = field(default_factory=list)


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return _normalize_text(value)


def _normalize_interpreted_metal_scope(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return "unspecified"
    if _EXCLUSIONARY_SCOPE_RE.search(text):
        return "unspecified"
    if _METAL_SCOPE_TOKEN_RE.search(text) is None:
        return "unspecified"
    return text


def _normalize_rule_type(value: Any) -> str:
    normalized = _normalize_text(value).lower().replace(" ", "_").replace("-", "_")
    return _RULE_TYPE_ALIASES.get(normalized, "")


def _normalize_date(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    if not text or text.lower() in {"null", "none", "n/a", "na", "undefined", "unknown"}:
        return None
    candidates = [text]
    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso_match:
        candidates.append(iso_match.group(0))
    long_date_match = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    if long_date_match:
        candidates.append(long_date_match.group(0))
    short_date_match = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    if short_date_match:
        candidates.append(short_date_match.group(0).replace("Sept", "Sep"))
    for candidate in candidates:
        for date_format in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, date_format).date().isoformat()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_pages(values: Iterable[Any]) -> List[int]:
    pages: List[int] = []
    seen = set()
    for value in values:
        text = _normalize_text(value)
        if not text:
            continue
        match = re.search(r"\d+", text)
        if not match:
            continue
        page_number = int(match.group(0))
        if page_number <= 0 or page_number in seen:
            continue
        seen.add(page_number)
        pages.append(page_number)
    return sorted(pages)


def _candidate_id(
    *,
    batch_id: str,
    hts_code: str,
    rule_type: str,
    metal_scope: str,
    effective_from: Optional[str],
    effective_to: Optional[str],
    source_document_ids: Sequence[str],
    source_pages: Sequence[int],
    candidate_quality: str,
    candidate_flags: Sequence[str],
) -> str:
    key = json.dumps(
        {
            "batch_id": batch_id,
            "hts_code": hts_code,
            "rule_type": rule_type,
            "metal_scope": _normalize_text(metal_scope),
            "effective_from": effective_from,
            "effective_to": effective_to,
            "source_document_ids": list(source_document_ids),
            "source_pages": list(source_pages),
            "candidate_quality": candidate_quality,
            "candidate_flags": list(candidate_flags),
        },
        sort_keys=True,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _append_unique(target: List[str], value: str, *, max_items: int = _MAX_EXAMPLES_PER_SEED) -> None:
    normalized = _normalize_text(value)
    if not normalized or normalized in target:
        return
    if len(target) >= max_items:
        return
    target.append(normalized)


def _truncate_prompt_text(value: Any, *, max_chars: int = _MAX_CONTEXT_SNIPPET_CHARS) -> str:
    normalized = _optional_text(value)
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _candidate_quality_for_occurrences(
    *,
    matched_texts: Sequence[str],
    context_snippets: Sequence[str],
) -> tuple[str, List[str]]:
    flags: List[str] = []
    if matched_texts and all(_DIGITS_ONLY_RE.fullmatch(text or "") for text in matched_texts):
        flags.append("digits_only_token")
    if any(_ADMIN_CONTEXT_RE.search(snippet or "") for snippet in context_snippets):
        flags.append("administrative_reference_context")
    quality = "suspect" if flags else "normal"
    return quality, flags


def _iter_seed_occurrences(source: PersistedSection232Source) -> Iterable[Dict[str, Any]]:
    for page in source.page_texts:
        page_number = int(page.get("page_number") or 0)
        page_excerpt = _optional_text(page.get("page_excerpt") or page.get("layout_aware_text") or page.get("text"))
        raw_occurrences = list(page.get("hts_occurrences") or [])
        if raw_occurrences:
            for occurrence in raw_occurrences:
                yield {
                    "page_number": page_number,
                    "matched_text": _optional_text(occurrence.get("matched_text") or occurrence.get("normalized_hts_code")),
                    "normalized_hts_code": canonicalize_hts_code(
                        occurrence.get("normalized_hts_code") or occurrence.get("matched_text")
                    ),
                    "context_text": _optional_text(occurrence.get("context_text")) or page_excerpt,
                }
            continue

        fallback_text = _optional_text(page.get("layout_aware_text") or page.get("text") or page_excerpt)
        for mention in extract_hts_mentions(fallback_text):
            yield {
                "page_number": page_number,
                "matched_text": mention,
                "normalized_hts_code": canonicalize_hts_code(mention),
                "context_text": page_excerpt or fallback_text,
            }


def _build_source_candidate_seeds(source: PersistedSection232Source) -> List[Section232CandidateSeed]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for occurrence in _iter_seed_occurrences(source):
        normalized_hts_code = canonicalize_hts_code(occurrence.get("normalized_hts_code"))
        if not normalized_hts_code:
            continue
        current = grouped.setdefault(
            normalized_hts_code,
            {
                "pages": [],
                "matched_texts": [],
                "context_snippets": [],
                "representative_excerpt": "",
            },
        )
        page_number = int(occurrence.get("page_number") or 0)
        if page_number > 0 and page_number not in current["pages"]:
            current["pages"].append(page_number)
        _append_unique(current["matched_texts"], str(occurrence.get("matched_text") or normalized_hts_code))
        _append_unique(current["context_snippets"], str(occurrence.get("context_text") or ""))
        if not current["representative_excerpt"]:
            current["representative_excerpt"] = _optional_text(occurrence.get("context_text"))

    seeds: List[Section232CandidateSeed] = []
    for normalized_hts_code, payload in grouped.items():
        source_pages = sorted(int(value) for value in payload["pages"] if int(value) > 0)
        matched_texts = list(payload["matched_texts"])
        context_snippets = list(payload["context_snippets"])
        candidate_quality, candidate_flags = _candidate_quality_for_occurrences(
            matched_texts=matched_texts,
            context_snippets=context_snippets,
        )
        seeds.append(
            Section232CandidateSeed(
                candidate_key=f"{source.source_id}:{normalized_hts_code}",
                source_id=source.source_id,
                filename=source.filename,
                normalized_hts_code=normalized_hts_code,
                source_pages=source_pages,
                matched_texts=matched_texts,
                context_snippets=context_snippets,
                representative_excerpt=str(payload.get("representative_excerpt") or ""),
                candidate_quality=candidate_quality,
                candidate_flags=candidate_flags,
            )
        )
    seeds.sort(key=lambda item: (item.source_pages[0] if item.source_pages else 0, item.normalized_hts_code))
    return seeds


def _serialize_seed(seed: Section232CandidateSeed) -> Dict[str, Any]:
    return {
        "candidate_key": seed.candidate_key,
        "normalized_hts_code": seed.normalized_hts_code,
        "source_id": seed.source_id,
        "source_filename": seed.filename,
        "source_pages": list(seed.source_pages),
        "matched_texts": list(seed.matched_texts[:1]),
        "representative_excerpt": _truncate_prompt_text(seed.representative_excerpt),
        "candidate_quality": seed.candidate_quality,
        "candidate_flags": list(seed.candidate_flags),
    }


def _build_prompt_payload_for_source(
    source: PersistedSection232Source,
    *,
    seeds: Sequence[Section232CandidateSeed],
    subdivision_label: Optional[str] = None,
) -> Dict[str, Any]:
    page_outline = [
        {
            "page_number": int(page.get("page_number") or 0),
            "page_excerpt": _truncate_prompt_text(page.get("page_excerpt"), max_chars=220),
            "layout_aware_text_preview": _truncate_prompt_text(
                page.get("layout_aware_text") or page.get("text"),
                max_chars=220,
            ),
            "hts_mention_count": len(page.get("hts_occurrences") or page.get("hts_mentions") or []),
        }
        for page in source.page_texts
    ]
    serialized_source = {
        "source_id": source.source_id,
        "filename": source.filename,
        "page_count": source.page_count,
        "extraction_status": source.extraction_status,
        "warnings": list(source.warnings),
        "full_document_text": source.full_text,
        "page_outline": page_outline,
    }
    return {
        "source": serialized_source,
        "sources": [serialized_source],
        "task": {
            "subdivision_label": subdivision_label,
            "candidate_count": len(seeds),
            "instructions": {
                "full_document_context_required": True,
                "dates_must_be_explicit_or_null": True,
                "rate_text_must_be_explicit_or_null": True,
                "keep_candidate_keys_exact": True,
            },
        },
        "candidates": [_serialize_seed(seed) for seed in seeds],
    }


def _invoke_payload(
    *,
    llm: Any,
    system_prompt: str,
    prompt_payload: Dict[str, Any],
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> Dict[str, Any]:
    invoke_kwargs: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Section 232 source batch:\n{json.dumps(prompt_payload, default=str)}"},
        ],
        "temperature": 0,
        "max_tokens": 16000,
        "phase": "section_232_ruleset",
        "task": "section_232_batch_interpreter",
        "usage_recorder": usage_recorder,
    }
    model_name = getattr(getattr(llm, "settings", None), "section_232_model_name", None)
    if model_name:
        invoke_kwargs["model_name"] = model_name

    response = llm.invoke_native_chat_completion(**invoke_kwargs)
    message_content = response.choices[0].message.content if getattr(response, "choices", None) else response
    return _extract_json_payload(_response_text(message_content))


def _payload_char_count(payload: Dict[str, Any]) -> int:
    return len(json.dumps(payload, default=str))


def _max_prompt_chars_for_llm(llm: Any) -> int:
    settings = getattr(llm, "settings", None)
    try:
        value = int(getattr(settings, "section_232_max_prompt_chars", 120000) or 120000)
    except (TypeError, ValueError):
        value = 120000
    return max(value, 10000)


def _candidate_result_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_items = payload.get("results") or payload.get("candidates") or payload.get("rules") or payload.get("items") or []
    return [item for item in raw_items if isinstance(item, dict)]


def _seed_key_for_result(
    result: Dict[str, Any],
    *,
    seeds: Sequence[Section232CandidateSeed],
) -> str:
    candidate_key = _optional_text(result.get("candidate_key"))
    if candidate_key:
        return candidate_key

    normalized_result_code = canonicalize_hts_code(
        result.get("hts_code") or result.get("normalized_hts_code") or result.get("code")
    )
    if not normalized_result_code:
        return ""

    source_identifiers = {
        _optional_text(value)
        for value in [
            *(result.get("source_document_ids") or []),
            *(result.get("source_ids") or []),
            result.get("source_id"),
            result.get("source_filename"),
            result.get("filename"),
        ]
        if _optional_text(value)
    }
    result_pages = set(_normalize_pages(result.get("source_pages") or result.get("page_numbers") or []))

    matches: List[str] = []
    for seed in seeds:
        if seed.normalized_hts_code != normalized_result_code:
            continue
        if source_identifiers and seed.source_id not in source_identifiers and seed.filename not in source_identifiers:
            continue
        if result_pages and not result_pages.intersection(set(seed.source_pages)):
            continue
        matches.append(seed.candidate_key)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        return ""

    if source_identifiers:
        filename_matches = [
            match
            for match in matches
            if any(
                seed.candidate_key == match and seed.filename in source_identifiers
                for seed in seeds
            )
        ]
        if len(filename_matches) == 1:
            return filename_matches[0]
    return ""


def _missing_candidate_keys(
    results: Sequence[Dict[str, Any]],
    *,
    seeds: Sequence[Section232CandidateSeed],
) -> List[str]:
    returned_keys = {
        _seed_key_for_result(item, seeds=seeds)
        for item in results
        if _seed_key_for_result(item, seeds=seeds)
    }
    return [seed.candidate_key for seed in seeds if seed.candidate_key not in returned_keys]


def _chunked(iterable: Sequence[Section232CandidateSeed], size: int) -> Iterable[List[Section232CandidateSeed]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _subdivision_batches(seeds: Sequence[Section232CandidateSeed]) -> List[tuple[str, List[Section232CandidateSeed]]]:
    grouped: Dict[int, List[Section232CandidateSeed]] = {}
    for seed in seeds:
        page = seed.source_pages[0] if seed.source_pages else 0
        grouped.setdefault(page, []).append(seed)

    batches: List[tuple[str, List[Section232CandidateSeed]]] = []
    for page_number in sorted(grouped):
        page_group = grouped[page_number]
        if len(page_group) <= _SUBDIVISION_BATCH_SIZE:
            batches.append((f"page-{page_number}", page_group))
            continue
        for index, chunk in enumerate(_chunked(page_group, _SUBDIVISION_BATCH_SIZE), start=1):
            batches.append((f"page-{page_number}-chunk-{index}", chunk))
    if len(batches) == 1 and len(batches[0][1]) == len(seeds):
        return [
            (f"candidate-chunk-{index}", chunk)
            for index, chunk in enumerate(_chunked(list(seeds), _SUBDIVISION_BATCH_SIZE), start=1)
        ]
    return batches


def _interpret_source_candidates(
    *,
    source: PersistedSection232Source,
    seeds: Sequence[Section232CandidateSeed],
    llm: Any,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are interpreting a Section 232 legal notice. "
        "The candidate HTS codes were extracted deterministically from the notice. "
        "Use the FULL DOCUMENT TEXT as the primary context, even when the code mention and legal basis appear on different pages. "
        "For every candidate_key in the request, return exactly one result object. "
        "Use only the provided notice text. Do not invent HTS codes or dates. "
        "If the candidate looks like an administrative reference such as a docket number, set is_real_hts_candidate to false. "
        "rule_type must be one of include, remove, or rate_schedule. "
        "Use rate_schedule only when the notice explicitly describes duty or rate treatment for a covered code. "
        "If a start date or end date is not explicit, return null. Do not infer a date. "
        "If rate wording is not explicit, return null for rate_text. "
        "Return STRICT JSON only with an object containing key results. "
        "Each result object must use keys: candidate_key, is_real_hts_candidate, rule_type, metal_scope, "
        "effective_from, effective_to, rate_text, source_excerpt, source_pages, interpreter_confidence."
    )

    primary_prompt_payload = _build_prompt_payload_for_source(source, seeds=seeds)
    max_prompt_chars = _max_prompt_chars_for_llm(llm)
    try:
        if _payload_char_count(primary_prompt_payload) <= max_prompt_chars:
            primary_payload = _invoke_payload(
                llm=llm,
                system_prompt=system_prompt,
                prompt_payload=primary_prompt_payload,
                usage_recorder=usage_recorder,
            )
            primary_results = _candidate_result_payload(primary_payload)
            if not _missing_candidate_keys(primary_results, seeds=seeds):
                return primary_results
        else:
            primary_results = []
    except Exception:
        primary_results = []

    stitched_results: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for subdivision_label, seed_batch in _subdivision_batches(seeds):
        try:
            payload = _invoke_payload(
                llm=llm,
                system_prompt=system_prompt,
                prompt_payload=_build_prompt_payload_for_source(
                    source,
                    seeds=seed_batch,
                    subdivision_label=subdivision_label,
                ),
                usage_recorder=usage_recorder,
            )
        except Exception:
            continue
        for item in _candidate_result_payload(payload):
            candidate_key = _seed_key_for_result(item, seeds=seed_batch)
            if not candidate_key or candidate_key in seen_keys:
                continue
            stitched_results.append(item)
            seen_keys.add(candidate_key)
    return stitched_results


def _sanitize_effective_window(
    *,
    effective_from: Optional[str],
    effective_to: Optional[str],
    candidate_flags: List[str],
) -> tuple[Optional[str], Optional[str]]:
    if effective_from and effective_to:
        try:
            if date.fromisoformat(effective_to) < date.fromisoformat(effective_from):
                candidate_flags.append("invalid_effective_window")
                return effective_from, None
        except ValueError:
            candidate_flags.append("invalid_effective_window")
            return effective_from, None
    return effective_from, effective_to


def _seed_excerpt(seed: Section232CandidateSeed) -> str:
    return seed.representative_excerpt or (seed.context_snippets[0] if seed.context_snippets else "")


def _candidate_from_result(
    *,
    batch_id: str,
    seed: Section232CandidateSeed,
    result: Dict[str, Any] | None,
) -> Section232DraftRuleCandidate:
    candidate_flags = list(seed.candidate_flags)
    if result is None:
        candidate_flags.append("llm_result_missing")
    if result is None:
        is_real_hts_candidate = False
    elif "is_real_hts_candidate" in result:
        is_real_hts_candidate = _normalize_bool(result.get("is_real_hts_candidate"))
    else:
        is_real_hts_candidate = True
    if not is_real_hts_candidate:
        candidate_flags.append("llm_marked_not_real_hts_candidate")

    rule_type = _normalize_rule_type(result.get("rule_type")) if isinstance(result, dict) else ""
    if not rule_type:
        candidate_flags.append("rule_type_undefined")
        rule_type = "include"

    metal_scope = _normalize_interpreted_metal_scope(result.get("metal_scope")) if isinstance(result, dict) else "unspecified"
    effective_from = _normalize_date(result.get("effective_from")) if isinstance(result, dict) else None
    effective_to = _normalize_date(result.get("effective_to")) if isinstance(result, dict) else None
    effective_from, effective_to = _sanitize_effective_window(
        effective_from=effective_from,
        effective_to=effective_to,
        candidate_flags=candidate_flags,
    )
    rate_text = _optional_text(result.get("rate_text")) if isinstance(result, dict) else ""
    source_pages = _normalize_pages(result.get("source_pages") or result.get("page_numbers") or []) if isinstance(result, dict) else []
    if not source_pages:
        source_pages = list(seed.source_pages)
    source_excerpt = (
        _optional_text(result.get("source_excerpt") or result.get("excerpt"))
        if isinstance(result, dict)
        else ""
    ) or _seed_excerpt(seed)

    candidate_quality = seed.candidate_quality
    if any(
        flag in {"llm_marked_not_real_hts_candidate", "llm_result_missing", "rule_type_undefined", "invalid_effective_window"}
        for flag in candidate_flags
    ):
        candidate_quality = "suspect"

    return Section232DraftRuleCandidate(
        candidate_id=_candidate_id(
            batch_id=batch_id,
            hts_code=seed.normalized_hts_code,
            rule_type=rule_type,
            metal_scope=metal_scope,
            effective_from=effective_from,
            effective_to=effective_to,
            source_document_ids=[seed.source_id],
            source_pages=source_pages,
            candidate_quality=candidate_quality,
            candidate_flags=candidate_flags,
        ),
        batch_id=batch_id,
        hts_code=seed.normalized_hts_code,
        rule_type=rule_type,  # type: ignore[arg-type]
        coverage_effect="remove" if rule_type == "remove" else "include",
        effective_from=effective_from,
        effective_to=effective_to,
        metal_scope=metal_scope,
        source_document_ids=[seed.source_id],
        source_pages=source_pages,
        source_excerpt=source_excerpt,
        interpreter_confidence=_normalize_confidence(result.get("interpreter_confidence")) if isinstance(result, dict) else 0.0,
        catalog_match_found=False,
        review_decision="pending",
        rate_text=rate_text or None,
        candidate_quality=candidate_quality,  # type: ignore[arg-type]
        candidate_flags=candidate_flags,
        processed_at=None,
    )


def interpret_section_232_batch(
    batch_id: str,
    extracted_sources: Sequence[PersistedSection232Source],
    llm: Any,
    usage_recorder: Optional[TokenUsageRecorder] = None,
) -> List[Section232DraftRuleCandidate]:
    candidates: List[Section232DraftRuleCandidate] = []
    for source in extracted_sources:
        seeds = _build_source_candidate_seeds(source)
        if not seeds:
            continue
        results = _interpret_source_candidates(
            source=source,
            seeds=seeds,
            llm=llm,
            usage_recorder=usage_recorder,
        )
        results_by_key = {
            _seed_key_for_result(item, seeds=seeds): item
            for item in results
            if _seed_key_for_result(item, seeds=seeds)
        }
        for seed in seeds:
            candidates.append(
                _candidate_from_result(
                    batch_id=batch_id,
                    seed=seed,
                    result=results_by_key.get(seed.candidate_key),
                )
            )
    return candidates
