"""
document_type_detector.py
--------------------------
Detects whether a SAP Document AI extraction result is an invoice or
a customer Purchase Order, based on the fields present.

Logic:
  - If key PO fields present (purchaseOrderNumber, customerName, buyerName)
    AND no typical invoice-only fields → purchase_order
  - Otherwise → invoice

This allows the pipeline to transparently route to the correct flow
without the user having to select a document type manually.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Fields that strongly indicate a customer Purchase Order
_PO_INDICATOR_FIELDS = {
    "purchaseordernumber",
    "buyername",
    "customername",
    "shiptoname",
    "requesteddeliverydate",
    "deliverydate",
}

# Fields that strongly indicate an invoice
_INVOICE_INDICATOR_FIELDS = {
    "sendername",
    "invoicenumber",
    "documentnumber",
    "grossamount",
    "taxamount",
    "taxrate",
    "iban",
    "senderbankaccount",
    "invoicingparty",
}


def _get_field_names(sap_result: dict[str, Any]) -> set[str]:
    """Extract all field names from a SAP Document AI result (lowercase)."""
    extraction = (
        sap_result.get("extraction")
        or sap_result.get("document")
        or {}
    )
    header_fields: list[dict] = extraction.get("headerFields") or []
    return {str(f.get("name", "")).lower() for f in header_fields if f.get("name")}


def detect_document_type(sap_result: dict[str, Any]) -> str:
    """
    Detect whether the document is an invoice or a customer Purchase Order.

    Returns:
        "purchase_order" — if PO indicators dominate
        "invoice"        — default (invoice indicators dominate or ambiguous)
    """
    field_names = _get_field_names(sap_result)

    po_score     = len(field_names & _PO_INDICATOR_FIELDS)
    invoice_score = len(field_names & _INVOICE_INDICATOR_FIELDS)

    logger.info(
        "Document type detection | po_score=%d | invoice_score=%d | fields=%s",
        po_score,
        invoice_score,
        sorted(field_names),
    )

    # Clear PO signal: at least 1 PO indicator AND more PO than invoice indicators
    if po_score >= 1 and po_score >= invoice_score:
        logger.info("Document type detected: purchase_order")
        return "purchase_order"

    logger.info("Document type detected: invoice")
    return "invoice"
