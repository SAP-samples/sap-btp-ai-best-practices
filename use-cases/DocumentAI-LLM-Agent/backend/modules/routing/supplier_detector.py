"""
supplier_detector.py
--------------------
Extracts and normalizes the supplier name from a SAP Document AI result.

Responsibilities:
  - Extract supplier name from SAP headerFields
  - Normalize supplier names (lowercase, strip, remove legal suffixes)
  - Validate empty supplier
  - Confidence validation
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Possible field names for supplier in SAP Document AI (checked in order)
SUPPLIER_FIELD_NAMES: list[str] = [
    "senderName",
    "supplierName",
    "vendorName",
    "sender",
    "supplier",
    "vendor",
    "companyName",
    "sellerName",
    "issuerName",
]

# Minimum confidence to consider a detected supplier valid
CONFIDENCE_THRESHOLD: float = 0.5

# Legal entity suffixes to strip for better matching
_LEGAL_SUFFIXES: list[str] = [
    r"\bs\.a\.?\b",
    r"\bs\.r\.l\.?\b",
    r"\bltda\.?\b",
    r"\binc\.?\b",
    r"\bcorp\.?\b",
    r"\bllc\.?\b",
    r"\bgmbh\.?\b",
    r"\bag\b",
    r"\bplc\.?\b",
    r"\bspa\.?\b",
    r"\bsrl\.?\b",
    r"\bsa\b",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SupplierDetectionError(Exception):
    """Raised when supplier detection fails critically."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_supplier_name(name: str) -> str:
    """
    Normalize a supplier name for comparison:
      - Lowercase
      - Strip leading/trailing whitespace
      - Remove legal entity suffixes (S.A., Inc., GmbH, etc.)
      - Collapse multiple spaces

    Args:
        name: Raw supplier name string.

    Returns:
        Normalized string.
    """
    if not name:
        return ""

    result = name.strip().lower()

    # Remove legal suffixes
    for suffix_pattern in _LEGAL_SUFFIXES:
        result = re.sub(suffix_pattern, "", result, flags=re.IGNORECASE).strip()

    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result


def _extract_from_header_fields(
    header_fields: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Search header fields for a supplier name field.

    Returns the first match found (in SUPPLIER_FIELD_NAMES priority order),
    or None if not found.
    """
    # Build a lookup by field name (case-insensitive)
    fields_by_name: dict[str, dict] = {}
    for field in header_fields:
        field_name = (field.get("name") or "").lower()
        if field_name:
            fields_by_name[field_name] = field

    for candidate in SUPPLIER_FIELD_NAMES:
        field = fields_by_name.get(candidate.lower())
        if field is None:
            continue

        raw_value = field.get("value") or field.get("rawValue") or ""
        if not raw_value:
            logger.debug("Field '%s' found but value is empty.", candidate)
            continue

        confidence = field.get("confidence")
        if confidence is None:
            confidence = 1.0  # assume full confidence if not provided

        return {
            "supplier_name": raw_value,
            "confidence": float(confidence),
            "field_name": candidate,
        }

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_supplier_name(sap_result: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the supplier name from a SAP Document AI job result.

    Args:
        sap_result: Full SAP Document AI job result dict.

    Returns:
        Detection result dict:
            - supplier_name (str): raw extracted name, or "" if not found
            - supplier_name_normalized (str): normalized version
            - confidence (float): extraction confidence 0.0–1.0
            - field_name (str | None): which SAP field was used
            - detected (bool): True if a supplier name was found
    """
    logger.info("Starting supplier detection...")

    extraction: dict = (
        sap_result.get("extraction")
        or sap_result.get("document")
        or {}
    )
    header_fields: list[dict] = extraction.get("headerFields") or []

    if not header_fields:
        logger.warning(
            "No headerFields found in SAP result. Cannot detect supplier."
        )
        return _empty_detection("No headerFields in SAP result")

    match = _extract_from_header_fields(header_fields)

    if match is None:
        logger.warning(
            "No supplier field found in SAP result. Checked: %s",
            SUPPLIER_FIELD_NAMES,
        )
        return _empty_detection("No supplier field found in headerFields")

    raw_name = match["supplier_name"]
    confidence = match["confidence"]
    field_name = match["field_name"]

    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            "Supplier '%s' detected with low confidence: %.2f (threshold: %.2f). "
            "Proceeding anyway.",
            raw_name,
            confidence,
            CONFIDENCE_THRESHOLD,
        )

    normalized = normalize_supplier_name(raw_name)

    logger.info(
        "Supplier detected: '%s' → normalized: '%s' (field: %s, confidence: %.2f)",
        raw_name,
        normalized,
        field_name,
        confidence,
    )

    return {
        "supplier_name": raw_name,
        "supplier_name_normalized": normalized,
        "confidence": confidence,
        "field_name": field_name,
        "detected": True,
    }


def _empty_detection(reason: str = "") -> dict[str, Any]:
    """Return an empty detection result."""
    return {
        "supplier_name": "",
        "supplier_name_normalized": "",
        "confidence": 0.0,
        "field_name": None,
        "detected": False,
        "reason": reason,
    }