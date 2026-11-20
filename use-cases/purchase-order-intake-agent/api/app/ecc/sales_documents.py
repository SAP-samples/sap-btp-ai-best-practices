"""Write and read (mock) ECC sales documents: VBAK / VBKD / VBPA / VBAP.

``create_sales_order`` is the single swap point for posting:

* Mock (this module): inserts the document into the ``*_POAGENT`` tables in the caller's
  transaction, numbering it from ``VBELN_SEQ_POAGENT`` (9100000001, ...).
* Production: POST the same payload to a SAP Integration Suite iFlow (BTP Destination, OAuth2
  client credentials) that maps it to ``BAPI_SALESORDER_CREATEFROMDAT2`` + ``BAPI_TRANSACTION_COMMIT``
  (RFC adapter) or an ``ORDERS05`` IDoc via Cloud Connector, and returns the document number or the
  BAPIRET2 messages. See docs/ecc-integration.md for the field mapping.

Return value in both cases::

    {"vbeln": "9100000001" | None,
     "messages": [{"type": "S"|"W"|"E"|"A", "id": "V1", "number": "311", "message": "..."}]}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..db.schema import clip_row
from ..db.sql import fetch_all, insert_row
from ..orders.matching import alpha_in


def _message(kind: str, number: str, message: str) -> dict[str, str]:
    """Build one BAPIRET2-like message."""
    return {"type": kind, "id": "V1", "number": number, "message": message}


def _date(value: str | None):
    """Parse a YYYYMMDD payload date."""
    return datetime.strptime(value, "%Y%m%d").date() if value else None


def create_sales_order(payload: dict[str, Any], conn: Connection) -> dict[str, Any]:
    """Create a sales order from the writeback payload (mock: insert into HANA).

    Args:
        payload: Writeback payload built by ``orders.posting.build_posting_payload``.
        conn: Connection inside the caller's transaction (rolled back if an error is returned).

    Returns:
        {"vbeln", "messages"} as described in the module docstring.
    """
    items = payload.get("items") or []
    errors = []
    if not payload.get("soldToId") or not payload.get("shipToId"):
        errors.append(_message("E", "219", "Sold-to and ship-to party are required"))
    if not items:
        errors.append(_message("E", "554", "No items were transferred"))
    if errors:
        return {"vbeln": None, "messages": errors}

    vbeln = str(conn.execute(text('SELECT "VBELN_SEQ_POAGENT".NEXTVAL FROM DUMMY')).scalar())
    now = datetime.now(timezone.utc)
    reference = payload.get("orderReference") or ""
    insert_row(conn, "VBAK_POAGENT", clip_row("VBAK_POAGENT", {
        "VBELN": vbeln, "AUART": payload["orderType"], "VKORG": payload["salesOrgId"],
        "VTWEG": payload["distributionChannel"], "SPART": payload["division"],
        "KUNNR": alpha_in(payload["soldToId"], 10), "BSTNK": reference[:20], "BSTDK": _date(payload.get("orderDate")),
        "BSARK": payload.get("customerPurchaseOrderType"), "AUDAT": now.date(),
        "VDATU": _date(payload.get("requestedDeliveryDate")), "WAERK": payload.get("currency"),
        "NETWR": payload.get("netValue"), "ERDAT": now.date(), "ERZET": now.time().replace(microsecond=0),
        "ERNAM": "POAGENT",
    }))
    insert_row(conn, "VBKD_POAGENT", clip_row("VBKD_POAGENT", {
        "VBELN": vbeln, "POSNR": "000000", "BSTKD": reference, "BSTDK": _date(payload.get("orderDate")),
        "BSARK": payload.get("customerPurchaseOrderType"), "ZTERM": payload.get("paymentTerms"),
        "INCO1": payload.get("incoterms"), "INCO2": payload.get("incotermsPlace"),
    }))
    for parvw, key in (("AG", "soldToId"), ("WE", "shipToId")):
        insert_row(conn, "VBPA_POAGENT", {"VBELN": vbeln, "POSNR": "000000", "PARVW": parvw,
                                          "KUNNR": alpha_in(payload[key], 10)})
    for item in items:
        insert_row(conn, "VBAP_POAGENT", clip_row("VBAP_POAGENT", {
            "VBELN": vbeln, "POSNR": f"{int(item['lineItemNumber']):06d}",
            "MATNR": alpha_in(item["sapMaterialNumber"], 18), "ARKTX": item.get("description"),
            "KDMAT": item.get("customerMaterialNumber"), "KWMENG": item["quantity"], "VRKME": item["unitOfMeasure"],
            "WERKS": item.get("plant"), "NETPR": item.get("netPrice"), "WAERK": payload.get("currency"),
            "ZZCOLOR": item.get("tintedMaterialColourCode"),
        }))
    return {"vbeln": vbeln, "messages": [_message("S", "311", f"Standard Order {vbeln} has been saved")]}


def list_sales_orders(limit: int = 100) -> list[dict[str, Any]]:
    """Return posted sales documents for the order history, newest first.

    Args:
        limit: Maximum rows.

    Returns:
        Rows with VBELN, AUART, VKORG, KUNNR, sold-to/ship-to names, BSTKD, BSTDK, WAERK, NETWR,
        ERDAT, ERZET, ITEM_COUNT and the app ORDER_ID that produced the document.
    """
    return fetch_all(
        """
        SELECT a."VBELN", a."AUART", a."VKORG", a."VTWEG", a."SPART", a."KUNNR", k."NAME1" AS "SOLD_TO_NAME",
               we."KUNNR" AS "SHIP_TO", s."NAME1" AS "SHIP_TO_NAME", d."BSTKD", a."BSTDK", a."WAERK", a."NETWR",
               a."ERDAT", a."ERZET",
               (SELECT COUNT(*) FROM "VBAP_POAGENT" p WHERE p."VBELN" = a."VBELN") AS "ITEM_COUNT",
               h."ORDER_ID"
          FROM "VBAK_POAGENT" a
          LEFT JOIN "VBKD_POAGENT" d ON d."VBELN" = a."VBELN" AND d."POSNR" = '000000'
          LEFT JOIN "KNA1_POAGENT" k ON k."KUNNR" = a."KUNNR"
          LEFT JOIN "VBPA_POAGENT" we ON we."VBELN" = a."VBELN" AND we."POSNR" = '000000' AND we."PARVW" = 'WE'
          LEFT JOIN "KNA1_POAGENT" s ON s."KUNNR" = we."KUNNR"
          LEFT JOIN "POAGENT_ORDER_HEADER" h ON h."SALES_DOC" = a."VBELN"
         ORDER BY a."ERDAT" DESC, a."ERZET" DESC, a."VBELN" DESC
         LIMIT :limit
        """,
        limit=int(limit),
    )
