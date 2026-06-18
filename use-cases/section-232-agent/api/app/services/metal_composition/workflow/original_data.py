"""Helpers for limiting original product data exposed to LLM prompts."""

from __future__ import annotations

from typing import Any, Dict


SOURCE_SUMMARY_PROMPT_KEYS = (
    "source_row_id",
    "source_kind",
    "pn_revised_standardized",
    "part_description",
    "new_part_description",
    "total_weight_gram",
)

SOURCE_ROW_PROMPT_KEYS = (
    "Product code",
    "PN Revised/ Standardized",
    "Part description",
    "New Part Description",
    "Material Content Method",
    "MaterialIdentified",
    "Total Weight (Gram)",
)


def _extra_item_context(source: Dict[str, Any]) -> Any:
    """Return neutral item-context text without exposing GCC priority labels.

    Inputs:
        source: Source summary or source row dictionary.

    Expected output:
        The BY Priority duplicate value when present, otherwise any already
        normalized extra item context value.
    """

    return source.get("extra_item_context") or source.get("priority_detail") or source.get("Priority.1")


def prompt_safe_source_summary(source_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return only original source-summary fields that help identify/classify the item.

    Inputs:
        source_summary: API-facing source summary derived from the GCC tracker.

    Expected output:
        A dictionary excluding operational fields such as priority, business segment, site, and dates.
        The GCC BY Priority duplicate is included only under ``extra_item_context``.
    """

    source = dict(source_summary or {})
    prompt_source = {
        key: source.get(key)
        for key in SOURCE_SUMMARY_PROMPT_KEYS
        if source.get(key) is not None
    }
    extra_context = _extra_item_context(source)
    if extra_context is not None:
        prompt_source["extra_item_context"] = extra_context
    return prompt_source


def prompt_safe_source_row(source_row: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return only original source-row fields that should be included in LLM prompts.

    Inputs:
        source_row: Business-safe source row from GCC tracker resolution.

    Expected output:
        A dictionary limited to item identity, description, material labels, and total weight.
        The GCC BY Priority duplicate is included only under ``extra_item_context``.
    """

    source = dict(source_row or {})
    prompt_source = {
        key: source.get(key)
        for key in SOURCE_ROW_PROMPT_KEYS
        if source.get(key) is not None
    }
    extra_context = _extra_item_context(source)
    if extra_context is not None:
        prompt_source["extra_item_context"] = extra_context
    return prompt_source
