"""Build the writeback payload and post an order to the (mock) ECC system exactly once.

Payload shape follows the "Writeback API - required fields" of the requirements
(orderType TA, distributionChannel 10, customerPurchaseOrderType WWW, SAP customer and material
numbers, sales unit quantities). ``post_order`` locks the order row (``SELECT ... FOR UPDATE``),
checks its status, creates the sales document and marks the order POSTED in one transaction, so
automatic posting and a human "Confirm" can never create two documents for the same order.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

from ..db.engine import get_engine
from ..db.sql import fetch_all, fetch_one
from ..ecc.sales_documents import create_sales_order
from . import repository
from .matching import alpha_out

POSTABLE_STATUSES = {"PROCESSING", "NEEDS_REVIEW", "FAILED"}


class PostingError(Exception):
    """Posting was refused (wrong state) or the ECC call returned error messages."""

    def __init__(self, message: str, status_code: int = 422, messages: list[dict] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.messages = messages or []


def _yyyymmdd(value: Any) -> str | None:
    """Format a date as YYYYMMDD (payload format)."""
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime("%Y%m%d") if isinstance(value, date) else None


def build_posting_payload(header: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the writeback payload from a validated order (pure).

    Args:
        header: POAGENT_ORDER_HEADER row.
        items: POAGENT_ORDER_ITEM rows; only MATERIAL lines are transferred.

    Returns:
        Payload dict (see module docstring).
    """
    material_items = [i for i in items if i.get("LINE_TYPE", "MATERIAL") == "MATERIAL"]
    return {
        "orderId": header["ORDER_ID"],
        "orderDate": _yyyymmdd(header.get("PO_DATE")),
        "orderReference": header.get("PO_NUMBER"),
        "orderType": header.get("ORDER_TYPE"),
        "soldToId": alpha_out(header.get("SOLD_TO")),
        "shipToId": alpha_out(header.get("SHIP_TO")),
        "salesOrgId": header.get("SALES_ORG"),
        "distributionChannel": header.get("DISTR_CHANNEL"),
        "division": header.get("DIVISION"),
        "customerPurchaseOrderType": header.get("PO_TYPE"),
        "requestedDeliveryDate": _yyyymmdd(header.get("REQ_DELIVERY_DATE")),
        "currency": header.get("CURRENCY"),
        "paymentTerms": header.get("PAYMENT_TERMS"),
        "incoterms": header.get("INCOTERMS"),
        "incotermsPlace": header.get("INCOTERMS_PLACE"),
        "netValue": header.get("NET_AMOUNT"),
        "items": [{
            "lineItemNumber": int(item["ITEM_NO"]),
            "sapMaterialNumber": alpha_out(item.get("MATERIAL")),
            "quantity": item.get("QUANTITY"),
            "unitOfMeasure": item.get("UOM"),
            "plant": item.get("PLANT"),
            "customerMaterialNumber": item.get("CUSTOMER_MATERIAL"),
            "tintedMaterialColourCode": item.get("COLOUR_CODE"),
            "description": item.get("DESCRIPTION"),
            "netPrice": round(item["LINE_AMOUNT"] / item["QUANTITY"], 2)
            if item.get("LINE_AMOUNT") and item.get("QUANTITY") else None,
        } for item in material_items],
    }


def post_order(order_id: str) -> str:
    """Post one validated order to ECC and mark it POSTED (single transaction).

    The caller is responsible for validation (gate or Confirm); this function only guards
    against double posting and records the result.

    Args:
        order_id: Order id.

    Returns:
        The sales document number.

    Raises:
        PostingError: 404 unknown order, 409 already posted / not postable, 422 ECC errors.
    """
    with get_engine().begin() as conn:
        header = fetch_one('SELECT * FROM "POAGENT_ORDER_HEADER" WHERE "ORDER_ID" = :o FOR UPDATE', conn, o=order_id)
        if header is None:
            raise PostingError("Order not found", 404)
        if header.get("SALES_DOC") or header["STATUS"] == "POSTED":
            raise PostingError(f"Order is already posted as {header.get('SALES_DOC')}", 409)
        if header["STATUS"] not in POSTABLE_STATUSES:
            raise PostingError(f"Order in status {header['STATUS']} cannot be posted", 409)
        items = fetch_all('SELECT * FROM "POAGENT_ORDER_ITEM" WHERE "ORDER_ID" = :o ORDER BY "ITEM_NO"', conn, o=order_id)
        payload = build_posting_payload(header, items)
        result = create_sales_order(payload, conn)
        errors = [m for m in result["messages"] if m["type"] in ("E", "A")]
        if errors or not result.get("vbeln"):  # raising inside begin() rolls the document back
            raise PostingError("ECC rejected the order: " + "; ".join(m["message"] for m in errors), 422, errors)
        repository.update_header(conn, order_id, {
            "STATUS": "POSTED", "SALES_DOC": result["vbeln"],
            "STATUS_MESSAGE": "; ".join(m["message"] for m in result["messages"]),
            "POSTING_PAYLOAD": json.dumps(payload, ensure_ascii=False, default=str),
            "POSTED_AT": datetime.now(timezone.utc).replace(tzinfo=None), "LOCKED_BY": None, "LOCKED_AT": None,
        })
    return result["vbeln"]

