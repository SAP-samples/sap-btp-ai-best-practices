"""
Raw-to-canonical field mapper for the Payment Advice Extractor (UC-01).

When a client is extracted with a *custom* schema (client-native field names, which
Document AI extracts more reliably), the raw aggregate must be normalized to the
canonical ``payment_advice_canonical`` shape.

Strategy (scalable and cheap): make ONE LLM call to derive a field-name mapping
(raw -> canonical) from the raw field names plus a few sample rows, then apply that
mapping deterministically to every row. Cost is independent of row count.

Schemas already flagged canonical skip this module entirely (identity mapping).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .aggregation import _normalize_value
from .canonical import (
    CANONICAL_SCHEMA,
    HEADER_FIELD_META,
    HEADER_FIELD_NAMES,
    LINE_FIELD_META,
    LINE_FIELD_NAMES,
)
from .config import DEFAULT_MODEL
from .llm import complete_json

_HEADER_DEFS = {f["name"]: f for f in CANONICAL_SCHEMA["headerFields"]}
_LINE_DEFS = {f["name"]: f for f in CANONICAL_SCHEMA["lineItemFields"]}

_SYSTEM_PROMPT = (
    "You map source payment-advice field names to a fixed canonical schema. "
    "You reply with a single JSON object and nothing else."
)


def _canonical_spec_text() -> str:
    """Render the canonical fields as a compact 'name (type): description' list."""
    header = "\n".join(f"- {n} ({t}): {d}" for n, t, d in HEADER_FIELD_META)
    line = "\n".join(f"- {n} ({t}): {d}" for n, t, d in LINE_FIELD_META)
    return f"HEADER FIELDS:\n{header}\n\nLINE-ITEM FIELDS:\n{line}"


def _build_user_prompt(
    raw_header_keys: list[str], raw_line_keys: list[str], sample_lines: list[dict[str, Any]]
) -> str:
    """Build the mapping prompt from raw field names and a few sample rows."""
    return (
        "Map each SOURCE field name to the best CANONICAL field name by meaning, or null "
        "if none fits. Use each canonical name at most once per section.\n\n"
        f"CANONICAL SCHEMA:\n{_canonical_spec_text()}\n\n"
        f"SOURCE HEADER FIELD NAMES:\n{json.dumps(raw_header_keys, ensure_ascii=False)}\n\n"
        f"SOURCE LINE-ITEM FIELD NAMES:\n{json.dumps(raw_line_keys, ensure_ascii=False)}\n\n"
        f"SAMPLE LINE ITEMS (source names -> values):\n"
        f"{json.dumps(sample_lines, ensure_ascii=False, default=str)}\n\n"
        'Return JSON exactly as: {"header": {"<source>": "<canonical or null>"}, '
        '"line_items": {"<source>": "<canonical or null>"}}'
    )


def propose_field_mapping(
    raw_header_keys: list[str],
    raw_line_keys: list[str],
    sample_lines: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    openai_create: Callable[..., Any] | None = None,
    gemini_generate: Callable[..., Any] | None = None,
) -> dict[str, dict[str, str]]:
    """
    Ask the LLM for a raw->canonical field mapping (one call).

    Returns:
        ``{"header": {raw: canonical}, "line_items": {raw: canonical}}`` containing
        only entries whose canonical target is a real canonical field name.
    """
    result = complete_json(
        _SYSTEM_PROMPT,
        _build_user_prompt(raw_header_keys, raw_line_keys, sample_lines),
        model=model,
        openai_create=openai_create,
        gemini_generate=gemini_generate,
        route_label="mapper:map_to_canonical",
    )
    return {
        "header": _clean_mapping(result.get("header"), HEADER_FIELD_NAMES),
        "line_items": _clean_mapping(result.get("line_items"), LINE_FIELD_NAMES),
    }


def _clean_mapping(raw_mapping: Any, allowed_targets: tuple[str, ...]) -> dict[str, str]:
    """Keep only entries mapping a source name to a valid canonical target."""
    if not isinstance(raw_mapping, dict):
        return {}
    allowed = set(allowed_targets)
    cleaned: dict[str, str] = {}
    for source, target in raw_mapping.items():
        if isinstance(source, str) and isinstance(target, str) and target in allowed:
            cleaned[source] = target
    return cleaned


def apply_mapping(
    aggregate: dict[str, Any], mapping: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """
    Apply a raw->canonical mapping to an aggregate, coercing canonical types.

    Header and line-item values are renamed to canonical fields (first source wins
    if two map to the same target) and coerced with the canonical field type
    (numbers/dates). Row count and order are preserved.

    Returns:
        ``{"header": {...}, "line_items": [...]}`` in canonical shape.
    """
    header_map = mapping.get("header", {})
    line_map = mapping.get("line_items", {})

    header_out: dict[str, Any] = {}
    for raw_key, value in aggregate.get("headers", {}).items():
        canon = header_map.get(raw_key)
        if canon and canon not in header_out:
            header_out[canon] = _normalize_value(value, _HEADER_DEFS.get(canon, {}))

    lines_out: list[dict[str, Any]] = []
    for row in aggregate.get("line_items", []):
        new_row: dict[str, Any] = {}
        for raw_key, value in row.items():
            canon = line_map.get(raw_key)
            if canon and canon not in new_row:
                new_row[canon] = _normalize_value(value, _LINE_DEFS.get(canon, {}))
        lines_out.append(new_row)

    return {"header": header_out, "line_items": lines_out}


def map_to_canonical(
    aggregate: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    openai_create: Callable[..., Any] | None = None,
    gemini_generate: Callable[..., Any] | None = None,
    sample_size: int = 5,
) -> dict[str, Any]:
    """
    Map a raw extraction aggregate to the canonical shape.

    Args:
        aggregate: Output of ``aggregate_documents`` (raw ``headers`` + ``line_items``).
        model: LLM model for the single mapping call.
        openai_create / gemini_generate: Injectable provider callables (tests).
        sample_size: Number of sample line items shown to the model.

    Returns:
        ``{"header": {...}, "line_items": [...], "mapping": {...}}``.
    """
    line_items = aggregate.get("line_items", [])
    raw_header_keys = list(aggregate.get("headers", {}).keys())
    raw_line_keys = sorted({key for row in line_items for key in row})
    sample_lines = line_items[:sample_size]

    mapping = propose_field_mapping(
        raw_header_keys, raw_line_keys, sample_lines,
        model=model, openai_create=openai_create, gemini_generate=gemini_generate,
    )
    mapped = apply_mapping(aggregate, mapping)
    mapped["mapping"] = mapping
    return mapped
