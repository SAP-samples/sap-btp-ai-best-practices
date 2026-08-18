"""
so_extractor.py
---------------
Extracts structured Purchase Order data from a PDF using SAP Document AI
with the "SAP_purchaseOrder_schema" schema.

Reuses InvoiceProcessor from modules/invoice/process_invoice.py — only the
schema name and field mapping differ from invoice extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from modules.invoice.process_invoice import InvoiceProcessor, InvoiceProcessingError
from SalesOrderProcess.so_models import ExtractedPurchaseOrder, SalesOrderLineItem

logger = logging.getLogger(__name__)

# Schema constants
PO_SCHEMA_NAME = "SAP_purchaseOrder_schema"
PO_DOCUMENT_TYPE = "purchaseOrder"


# ---------------------------------------------------------------------------
# Field name aliases — SAP Document AI may use different names depending on
# the schema version / template.
# ---------------------------------------------------------------------------

_HEADER_ALIASES: dict[str, list[str]] = {
    # DocAI purchaseOrder schema puts buyer name in senderExtraAddressPart or senderAddress
    # (the "sender" in a PO is the buyer/customer sending the order to AI4U)
    "customer_name": [
        "senderExtraAddressPart",   # ← DocAI purchaseOrder schema primary field
        "customerName",
        "buyerName",
        "shipToName",
        "receiverName",
        "senderName",
    ],
    "purchase_order_number":    ["purchaseOrderNumber", "documentNumber"],
    "order_date":               ["orderDate", "documentDate"],
    "requested_delivery_date":  ["deliveryDate", "requestedDeliveryDate"],
    "currency":                 ["currencyCode", "currency"],
    "total_amount":             ["grossAmount", "totalAmount", "netAmount"],
    "special_instructions":     ["specialInstructions", "notes"],
}

_LINE_ALIASES: dict[str, list[str]] = {
    "material_code":  ["materialNumber", "productCode", "itemCode"],
    "description":    ["description", "productDescription"],
    "quantity":       ["quantity", "qty"],
    "uom":            ["unitOfMeasure", "unit"],
    "unit_price":     ["unitPrice", "price"],
    "total_price":    ["totalPrice", "amount"],
    "currency":       ["currencyCode"],
}


def _first_value(fields_dict: dict[str, Any], aliases: list[str]) -> Any:
    """Return the first non-empty value found in fields_dict for any alias."""
    for alias in aliases:
        val = fields_dict.get(alias)
        if val is not None and val != "":
            return val
    return None


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# SalesOrderExtractor
# ---------------------------------------------------------------------------


class SalesOrderExtractor:
    """
    Extracts Purchase Order data from a PDF using SAP Document AI.

    Uses InvoiceProcessor internally, but submits with the
    purchaseOrder schema and maps fields to ExtractedPurchaseOrder.
    """

    def __init__(self) -> None:
        self._processor = InvoiceProcessor()

    def extract_with_llm(
        self,
        pdf_path: Path,
        client_id: str = "default",
    ) -> tuple[ExtractedPurchaseOrder, dict]:
        """DocAI extraction only — no LLM needed when schema is correct."""
        po = self.extract(pdf_path, client_id=client_id)
        logger.info(
            "PO extracted | customer=%r | po=%r | items=%d",
            po.customer_name, po.purchase_order_number, len(po.line_items),
        )
        return po, {}

    def extract(
        self,
        pdf_path: Path,
        client_id: str = "default",
    ) -> ExtractedPurchaseOrder:
        """
        Submit the PDF to SAP Document AI and return a structured PO.

        Args:
            pdf_path: Path to the customer PO PDF.
            client_id: SAP Document AI client ID.

        Returns:
            ExtractedPurchaseOrder with all available fields populated.

        Raises:
            InvoiceProcessingError: On any Document AI error.
        """
        logger.info(
            "Submitting PO for extraction | file=%s | schema=%s | client=%s",
            pdf_path.name,
            PO_SCHEMA_NAME,
            client_id,
        )

        job_id = self._processor.submit_document(
            pdf_path,
            schema_name=PO_SCHEMA_NAME,
            client_id=client_id,
            document_type=PO_DOCUMENT_TYPE,
        )
        logger.info("Document AI job created | job_id=%s", job_id)

        result = self._processor.poll_until_done(job_id)

        # Persist raw output for audit trail
        self._processor.save_result(job_id, result)

        return self._map_result(result)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _map_result(self, result: dict[str, Any]) -> ExtractedPurchaseOrder:
        """Map the raw SAP Document AI result to ExtractedPurchaseOrder."""
        extraction = (
            result.get("extraction")
            or result.get("document")
            or {}
        )

        header_fields_raw: list[dict] = extraction.get("headerFields") or []
        line_items_raw: list[Any] = extraction.get("lineItems") or []

        # Build a flat dict: field_name → value for header fields
        header_dict: dict[str, Any] = {
            f.get("name", ""): (
                f.get("value") if f.get("value") is not None else f.get("rawValue")
            )
            for f in header_fields_raw
            if f.get("name")
        }

        raw_sap_fields = {k: v for k, v in header_dict.items()}

        # Map header fields
        customer_name = _first_value(header_dict, _HEADER_ALIASES["customer_name"]) or ""
        po_number = _first_value(header_dict, _HEADER_ALIASES["purchase_order_number"]) or ""
        order_date = _first_value(header_dict, _HEADER_ALIASES["order_date"]) or ""
        delivery_date = _first_value(header_dict, _HEADER_ALIASES["requested_delivery_date"]) or ""
        currency = _first_value(header_dict, _HEADER_ALIASES["currency"]) or ""
        total_amount = _safe_float(_first_value(header_dict, _HEADER_ALIASES["total_amount"]))
        special_instructions = _first_value(header_dict, _HEADER_ALIASES["special_instructions"])

        # Map line items
        line_items = self._map_line_items(line_items_raw, currency)

        logger.info(
            "PO extracted | customer=%r | po_number=%r | items=%d",
            customer_name,
            po_number,
            len(line_items),
        )

        return ExtractedPurchaseOrder(
            customer_name=str(customer_name),
            purchase_order_number=str(po_number),
            order_date=str(order_date),
            requested_delivery_date=str(delivery_date),
            currency=str(currency),
            total_amount=total_amount,
            line_items=line_items,
            special_instructions=str(special_instructions) if special_instructions else None,
            raw_sap_fields=raw_sap_fields,
        )

    def _map_line_items(
        self,
        line_items_raw: list[Any],
        fallback_currency: str,
    ) -> list[SalesOrderLineItem]:
        """
        Convert SAP Document AI lineItems (array-of-arrays or array-of-dicts)
        to a list of SalesOrderLineItem.

        SAP Document AI returns lineItems as an array of arrays of field objects:
          [ [ {name, value, ...}, ... ], [ {name, value, ...}, ... ] ]
        """
        items: list[SalesOrderLineItem] = []

        for idx, raw_item in enumerate(line_items_raw):
            # Normalise: each entry may be a list of fields or a dict
            if isinstance(raw_item, list):
                fields: list[dict] = raw_item
            elif isinstance(raw_item, dict):
                # Already a flat dict — wrap in list for uniform handling
                fields = [{"name": k, "value": v} for k, v in raw_item.items()]
            else:
                logger.warning("Unexpected line item format at index %d: %s", idx, type(raw_item))
                continue

            field_dict: dict[str, Any] = {
                f.get("name", ""): (
                    f.get("value") if f.get("value") is not None else f.get("rawValue")
                )
                for f in fields
                if isinstance(f, dict) and f.get("name")
            }

            material_code = str(_first_value(field_dict, _LINE_ALIASES["material_code"]) or "")
            description   = str(_first_value(field_dict, _LINE_ALIASES["description"]) or "")
            quantity       = _safe_float(_first_value(field_dict, _LINE_ALIASES["quantity"])) or 1.0
            uom            = str(_first_value(field_dict, _LINE_ALIASES["uom"]) or "EA")
            unit_price     = _safe_float(_first_value(field_dict, _LINE_ALIASES["unit_price"]))
            total_price    = _safe_float(_first_value(field_dict, _LINE_ALIASES["total_price"]))
            item_currency  = str(_first_value(field_dict, _LINE_ALIASES["currency"]) or fallback_currency)

            # Derive unit_price from total if missing
            if unit_price is None and total_price is not None and quantity:
                try:
                    unit_price = total_price / quantity
                except ZeroDivisionError:
                    unit_price = None

            items.append(
                SalesOrderLineItem(
                    material_code=material_code,
                    description=description,
                    quantity=quantity,
                    uom=uom,
                    unit_price=unit_price,
                    currency=item_currency or None,
                )
            )

        return items
