"""
Helpers for normalising LLM outputs and merging page-level extractions.

Image-based extractions collect partial payloads per page. This module merges
them into a single header and line item list while also coercing values into
consistent formats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Tuple

from .schemas import (
    DocumentHeader,
    HEADER_FIELDS,
    LINE_ITEM_FIELDS,
    LineItem,
    PageExtraction,
)


def _clean_string(value: object) -> str | None:
    """
    Normalise string values returned by the LLM.

    Empty strings and placeholder values such as \\\"n/a\\\" are treated as
    missing to simplify downstream handling.
    """

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lower() not in {"n/a", "none", "null"}:
            return stripped
        return None
    return str(value)


def _clean_currency(value: object) -> str | None:
    """Ensure currency codes are uppercase three-letter strings."""

    cleaned = _clean_string(value)
    if not cleaned:
        return None
    return cleaned.upper()[:3]


def _clean_number(value: object) -> float | None:
    """Convert numeric strings to floats while handling thousand separators."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.replace(",", "").replace(" ", "")
        candidate = candidate.replace("€", "").replace("$", "")
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
]


def _clean_date(value: object) -> str | None:
    """
    Parse a date string into ISO format (YYYY-MM-DD).

    If the date cannot be parsed, the original value is discarded (None) to
    avoid introducing incorrect information.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
        # Try raw ISO strings with time components
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            return None
    return None


def normalise_header(raw_header: dict | None) -> DocumentHeader:
    """
    Convert a partial header dictionary into a fully populated DocumentHeader.

    Missing fields are filled with None so the downstream code can rely on the
    presence of every key.
    """

    raw_header = raw_header or {}
    header: DocumentHeader = {}
    for field in HEADER_FIELDS:
        value = raw_header.get(field)
        if field in {"documentDate", "deliveryDate"}:
            header[field] = _clean_date(value)
        elif field == "currencyCode":
            header[field] = _clean_currency(value)
        elif field == "netAmount":
            header[field] = _clean_number(value)
        else:
            header[field] = _clean_string(value)
    return header


def normalise_line_item(raw_item: dict | None) -> LineItem:
    """
    Convert a partial line item dictionary into a LineItem TypedDict.

    Similar to the header normalisation, all fields are present to simplify the
    creation of the final tabular representation.
    """

    raw_item = raw_item or {}
    item: LineItem = {}
    for field in LINE_ITEM_FIELDS:
        value = raw_item.get(field)
        if field in {"netAmount", "quantity", "unitPrice"}:
            item[field] = _clean_number(value)
        elif field == "currencyCode":
            item[field] = _clean_currency(value)
        else:
            item[field] = _clean_string(value)
    return item


def merge_page_extractions(pages: Iterable[PageExtraction]) -> Tuple[DocumentHeader, List[LineItem]]:
    """
    Merge the outputs of multiple page-level extractions.

    - For header fields we keep the first non-null value encountered across
      pages.
    - All line items are concatenated in the order the pages were processed.
    """

    merged_header: DocumentHeader = normalise_header({})
    line_items: List[LineItem] = []

    for page in pages:
        page_header = normalise_header(page.get("header"))
        for key in HEADER_FIELDS:
            if merged_header.get(key) in (None, "") and page_header.get(key) not in (None, ""):
                merged_header[key] = page_header[key]
        for item in page.get("lineItems", []):
            line_items.append(normalise_line_item(item))

    return merged_header, line_items

