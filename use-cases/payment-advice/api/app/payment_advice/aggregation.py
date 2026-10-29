"""
Pure deterministic aggregation of SAP Document AI extraction payloads.

Copied from the reused DocumentAI-Agent and adapted for UC-01 with one change: a
``dedupe_line_items`` flag.

- ``dedupe_line_items=True`` (default) preserves the original multi-file behavior:
  exact normalized-duplicate line items are merged and complementary rows stay
  separate. Used when aggregating genuinely independent files.
- ``dedupe_line_items=False`` is **concat mode** for a single oversized document
  split into disjoint parts. Every extracted line is appended in order with no
  merging, because the parts are non-overlapping slices of one ordered list, so a
  "duplicate" would be a legitimate repeated row, not redundancy.

Header handling is unchanged in both modes: identical repeated header values (as a
split produces) collapse to a single value with no conflict.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


def _strip_separators_heuristic(text: str) -> str:
    """Best-effort thousands/decimal cleanup when the schema declares no separators.

    Rules: with both '.' and ',', the rightmost is the decimal separator and the
    other is removed as grouping. With only ',', a single comma whose trailing group
    is not three digits is treated as a decimal point; otherwise commas are grouping.
    A lone '.' is left as the decimal point.
    """
    text = text.replace(" ", "")
    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")  # European: 1.234,56
        return text.replace(",", "")  # US: 1,234.56
    if has_comma:
        fraction = text.split(",")[-1]
        if text.count(",") == 1 and len(fraction) != 3:
            return text.replace(",", ".")  # decimal comma: 5,3 -> 5.3
        return text.replace(",", "")  # grouping: 1,234 -> 1234
    return text


def _normalize_number(value: Any, formatting: dict[str, Any]) -> int | float | Any:
    """Normalize a schema-formatted number while preserving invalid input."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = Decimal(str(value))
    else:
        text = str(value).strip()
        # Trailing-sign and accounting negatives (e.g. European "123,45-" or
        # "(123.45)") put the minus on the right; detect the sign, parse the
        # magnitude, then re-apply it.
        negative = False
        if text.endswith("-"):
            negative = True
            text = text[:-1].strip()
        elif text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1].strip()
        thousands = str(formatting.get("thousandSeparator") or "")
        decimal = str(formatting.get("decimalSeparator") or "")
        if thousands or decimal:
            # Explicit schema formatting wins.
            if thousands:
                text = text.replace(thousands, "")
            if decimal and decimal != ".":
                text = text.replace(decimal, ".")
        else:
            # No declared separators (our canonical/generated schemas): guess safely.
            text = _strip_separators_heuristic(text)
        try:
            number = Decimal(text)
        except InvalidOperation:
            return copy.deepcopy(value)
        if negative:
            number = -abs(number)
    return int(number) if number == number.to_integral() else float(number)


def _normalize_date(value: Any) -> Any:
    """Normalize common SAP and human date strings to an ISO date."""
    if not isinstance(value, str):
        return copy.deepcopy(value)
    text = value.strip()
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return text


def _normalize_value(value: Any, definition: dict[str, Any]) -> Any:
    """Normalize one value according to its SAP schema field definition."""
    if value is None:
        return None
    formatting_type = str(definition.get("formattingType") or "string").lower()
    if formatting_type in {"number", "currency", "amount"}:
        return _normalize_number(value, definition.get("formatting") or {})
    if formatting_type in {"date", "datetime"}:
        return _normalize_date(value)
    if isinstance(value, str):
        return " ".join(value.split())
    return copy.deepcopy(value)


def _comparison_key(value: Any) -> str:
    """Return a stable comparison key without changing the displayed value."""
    comparable = value.casefold() if isinstance(value, str) else value
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True, default=str)


def _field_definitions(schema: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    """Index one schema field collection by its field name."""
    return {
        str(item["name"]): item
        for item in schema.get(key, [])
        if isinstance(item, dict) and item.get("name")
    }


def _source(document: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    """Build the source pointer retained for one aggregate cell."""
    return {
        "document_id": document.get("id"),
        "file_name": document.get("file_name"),
        "value": copy.deepcopy(field.get("value")),
        "raw_value": copy.deepcopy(field.get("rawValue")),
        "page": field.get("page"),
        "coordinates": copy.deepcopy(field.get("coordinates")),
    }


def _raw_text(field: dict[str, Any], normalized: Any) -> Any:
    """Return the cell as the customer printed it: normalized value, else Document AI's rawValue.

    Customer rules (e.g. a region letter "A") must survive even when Document AI
    cannot normalize the cell under its formatting type and returns value=null.
    """
    if normalized is not None:
        return normalized
    raw = field.get("rawValue")
    if isinstance(raw, str):
        raw = " ".join(raw.split())
    return raw if raw not in (None, "") else None


def _extracted_fields(values: Iterable[Any]) -> Iterable[dict[str, Any]]:
    """Yield only named field dictionaries from an SAP extraction collection."""
    for value in values:
        if isinstance(value, dict) and value.get("name"):
            yield value


def aggregate_documents(
    schema: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    dedupe_line_items: bool = True,
) -> dict[str, Any]:
    """Aggregate source extractions without mutating or replacing source truth.

    Args:
        schema: SAP schema containing header and line-item field definitions.
        documents: Source records each with an ``extraction`` payload
            (``{"headerFields": [...], "lineItems": [[...], ...]}``).
        dedupe_line_items: When True (multi-file mode), merge exact normalized
            duplicate lines. When False (concat mode for a split document), append
            every line in order with no merging.

    Returns:
        A detached aggregate with normalized headers, ordered/deduplicated line
        items, cell-level provenance, unresolved header conflicts, and ``raw``:
        ``{"headers": {...}, "line_items": [...]}`` holding every cell as printed
        (normalized value, else Document AI's rawValue), index-aligned with
        ``line_items``, for customer rules that read columns the mapper drops.
    """
    header_definitions = _field_definitions(schema, "headerFields")
    line_definitions = _field_definitions(schema, "lineItemFields")
    headers: dict[str, Any] = {}
    header_values: dict[str, list[Any]] = {}
    header_provenance: dict[str, list[dict[str, Any]]] = {}
    line_items: list[dict[str, Any]] = []
    line_provenance: dict[str, dict[str, list[dict[str, Any]]]] = {}
    line_indexes: dict[str, int] = {}
    # Raw view for customer rules: every cell as printed, index-aligned with line_items.
    raw_headers: dict[str, Any] = {}
    raw_line_items: list[dict[str, Any]] = []

    for document in documents:
        extraction = document.get("extraction") or {}
        for field in _extracted_fields(extraction.get("headerFields") or []):
            name = str(field["name"])
            normalized = _normalize_value(field.get("value"), header_definitions.get(name, {}))
            raw = _raw_text(field, normalized)
            if raw is not None:
                raw_headers.setdefault(name, raw)
            if normalized is None:
                continue
            key = _comparison_key(normalized)
            header_provenance.setdefault(name, []).append(_source(document, field))
            if name not in headers:
                headers[name] = normalized
                header_values[name] = [normalized]
            elif key not in {_comparison_key(value) for value in header_values[name]}:
                header_values[name].append(normalized)

        for source_line in extraction.get("lineItems") or []:
            normalized_line: dict[str, Any] = {}
            raw_line: dict[str, Any] = {}
            cell_sources: dict[str, list[dict[str, Any]]] = {}
            for field in _extracted_fields(source_line if isinstance(source_line, list) else []):
                name = str(field["name"])
                normalized = _normalize_value(field.get("value"), line_definitions.get(name, {}))
                raw = _raw_text(field, normalized)
                if raw is not None:
                    raw_line[name] = raw
                if normalized is None:
                    continue
                normalized_line[name] = normalized
                cell_sources[name] = [_source(document, field)]
            if not normalized_line:
                continue

            if not dedupe_line_items:
                # Concat mode: append every line in order, never merge.
                index = len(line_items)
                line_items.append(normalized_line)
                raw_line_items.append(raw_line)
                line_provenance[str(index)] = cell_sources
                continue

            # Multi-file mode: exact normalized equality is the only automatic merge.
            # Complementary rows stay separate for human/agent review.
            line_key = json.dumps(
                {name: _comparison_key(value) for name, value in normalized_line.items()},
                ensure_ascii=False,
                sort_keys=True,
            )
            existing_index = line_indexes.get(line_key)
            if existing_index is None:
                existing_index = len(line_items)
                line_indexes[line_key] = existing_index
                line_items.append(normalized_line)
                raw_line_items.append(raw_line)
                line_provenance[str(existing_index)] = cell_sources
            else:
                for name, sources in cell_sources.items():
                    line_provenance[str(existing_index)].setdefault(name, []).extend(sources)

    conflicts = [
        {"scope": "header", "field": name, "values": values}
        for name, values in header_values.items()
        if len(values) > 1
    ]
    return {
        "headers": headers,
        "line_items": line_items,
        "provenance": {
            "headers": header_provenance,
            "line_items": line_provenance,
        },
        "conflicts": conflicts,
        "raw": {"headers": raw_headers, "line_items": raw_line_items},
    }
