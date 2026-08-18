"""
po_detector.py
--------------
Detects PO numbers from SAP Document AI extraction results and LLM outputs.

PO number field names across extraction methods:
  - SAP DocAI headerFields: "purchaseOrderNumber", "poNumber", "purchaseOrder"
  - LLM structured/prompting: "purchaseOrderNumber", "poNumber"
  - Generic pattern: starts with 45 and is 10 digits (standard SAP PO format)
"""

from __future__ import annotations

import re
from typing import Any

_PO_FIELD_NAMES = {
    "purchaseOrderNumber",
    "purchaseordernumber",
    "ponumber",
    "poNumber",
    "purchaseOrder",
    "purchaseorder",
    "po_number",
    "po",
}

# Standard SAP PO: 10-digit number starting with 45
_SAP_PO_PATTERN = re.compile(r"\b45\d{8}\b")


def _looks_like_po(value: str) -> bool:
    """Return True if the value matches a standard SAP PO number format."""
    cleaned = value.strip().replace("-", "").replace(" ", "")
    return bool(_SAP_PO_PATTERN.fullmatch(cleaned)) or (
        cleaned.isdigit() and 7 <= len(cleaned) <= 12
    )


def extract_po_from_sap_result(sap_result: dict[str, Any]) -> str | None:
    """
    Extract PO number from a SAP Document AI extraction result.

    Handles all nesting structures the pipeline may return:
      - sap_result["extraction"]["headerFields"]   (API v1 pipeline)
      - sap_result["document"]["headerFields"]     (some SAP DocAI responses)
      - sap_result["headerFields"]                 (flat fallback)
    """
    header_fields: list[dict] = (
        (sap_result.get("extraction") or {}).get("headerFields")
        or (sap_result.get("document") or {}).get("headerFields")
        or sap_result.get("headerFields")
        or []
    )

    for field in header_fields:
        name = str(field.get("name") or "").lower()
        if name in {n.lower() for n in _PO_FIELD_NAMES}:
            value = str(field.get("value") or "").strip()
            if value and _looks_like_po(value):
                return value

    return None


def extract_po_from_llm_result(llm_result: dict[str, Any]) -> str | None:
    """
    Extract PO number from an LLM extraction result (structured or prompting).
    LLM results use camelCase direct keys.
    """
    for key in _PO_FIELD_NAMES:
        value = llm_result.get(key) or llm_result.get(key.lower())
        if value:
            value_str = str(value).strip()
            if _looks_like_po(value_str):
                return value_str
    return None


def detect_po_number(
    sap_result: dict[str, Any] | None = None,
    llm_structured: dict[str, Any] | None = None,
    llm_prompting: dict[str, Any] | None = None,
) -> str | None:
    """
    Detect a PO number from any available extraction result.

    Priority: SAP DocAI (primary source) → LLM structured → LLM prompting.
    """
    if sap_result:
        po = extract_po_from_sap_result(sap_result)
        if po:
            return po

    if llm_structured:
        po = extract_po_from_llm_result(llm_structured)
        if po:
            return po

    if llm_prompting:
        po = extract_po_from_llm_result(llm_prompting)
        if po:
            return po

    return None
