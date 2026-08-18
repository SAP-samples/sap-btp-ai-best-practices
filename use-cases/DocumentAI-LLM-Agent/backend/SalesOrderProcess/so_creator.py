"""
so_creator.py
-------------
Creates a Sales Order in S/4HANA via API_SALES_ORDER_SRV/A_SalesOrder.

Flow:
  1. Fetch CSRF token from SAP metadata endpoint
  2. POST A_SalesOrder with first item via deep-insert
  3. POST each additional item to A_SalesOrderItem
  4. POST special_instructions as header text (TX01) — non-fatal if it fails
  5. Return CreateSOResponse

Logic adapted from:
  sales_order_process/backend/app/api/sales_order_create_routes.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Optional

from fastapi import Request

from S4.s4_client import get_s4_base_url, get_s4_session
from SalesOrderProcess.so_models import CreateSORequest, CreateSOResponse, SalesOrderLineItem
from config import settings

logger = logging.getLogger(__name__)

_SO_API = "/sap/opu/odata/sap/API_SALES_ORDER_SRV"
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 120


# ---------------------------------------------------------------------------
# SAP date helpers
# ---------------------------------------------------------------------------


def _sap_date(d: dt.date) -> str:
    """Format a Python date as SAP OData /Date(ms)/ string."""
    ms = int(
        dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    return f"/Date({ms})/"


def _item_number(position: int) -> str:
    """
    Convert 1-based position to SAP zero-padded 6-digit item number.
    position=1 → '000010', position=2 → '000020', etc.
    """
    return str(position * 10).zfill(6)


# ---------------------------------------------------------------------------
# CSRF token
# ---------------------------------------------------------------------------


def _fetch_csrf(
    session: Any,
    base_url: str,
    client: str,
) -> tuple[str, Any]:
    """
    Fetch X-CSRF-Token from SAP.

    Strategy:
      1. GET $metadata (always responds 200 and returns the token)
      2. Fallback: GET A_SalesOrder?$top=1
    """
    url = f"{base_url}{_SO_API}/$metadata"
    resp = session.get(
        url,
        headers={"X-CSRF-Token": "Fetch", "Accept": "*/*"},
        params={"sap-client": client},
        timeout=_READ_TIMEOUT,
    )
    token = resp.headers.get("X-CSRF-Token") or resp.headers.get("x-csrf-token")

    if not token:
        url_fallback = f"{base_url}{_SO_API}/A_SalesOrder"
        resp = session.get(
            url_fallback,
            headers={"X-CSRF-Token": "Fetch"},
            params={"sap-client": client, "$top": "1"},
            timeout=60,
        )
        token = resp.headers.get("X-CSRF-Token") or resp.headers.get("x-csrf-token")

    if not token:
        raise RuntimeError(
            f"CSRF token not returned by SAP (status={resp.status_code})"
        )

    logger.info("CSRF token obtained")
    return token, resp.cookies


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def _build_initial_payload(request_data: CreateSORequest) -> dict:
    """Build the deep-insert payload for creating a SO with the first item."""
    today = dt.date.today()
    delivery_date = today + dt.timedelta(days=7)
    first_item = request_data.items[0]

    payload: dict = {
        "SalesOrderType": "OR",
        "SalesOrganization": request_data.sales_organization,
        "DistributionChannel": request_data.distribution_channel,
        "OrganizationDivision": request_data.division,
        "SoldToParty": request_data.customer_bp,
        "TransactionCurrency": request_data.currency,
        "RequestedDeliveryDate": _sap_date(delivery_date),
        "PricingDate": _sap_date(today),
        "to_Item": {
            "results": [
                {
                    "SalesOrderItem": _item_number(1),
                    "Material": first_item.sap_material or first_item.material_code,
                    "RequestedQuantity": str(first_item.quantity),
                    "RequestedQuantityUnit": first_item.uom,
                    "PricingDate": _sap_date(today),
                }
            ]
        },
    }

    if request_data.purchase_order_number:
        payload["PurchaseOrderByCustomer"] = request_data.purchase_order_number

    return payload


# ---------------------------------------------------------------------------
# SAP HTTP helpers
# ---------------------------------------------------------------------------


def _post_sales_order(
    session: Any,
    base_url: str,
    csrf_token: str,
    cookies: Any,
    payload: dict,
    client: str,
) -> dict:
    """POST the initial SO payload to A_SalesOrder and return the parsed response."""
    url = f"{base_url}{_SO_API}/A_SalesOrder"
    resp = session.post(
        url,
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation",
        },
        cookies=cookies,
        params={"sap-client": client},
        data=json.dumps(payload),
        timeout=_READ_TIMEOUT,
    )

    if not resp.ok:
        error_text = resp.text[:1000]
        logger.error("SAP error %s creating SO: %s", resp.status_code, error_text)
        try:
            sap_msg = (
                resp.json()
                .get("error", {})
                .get("message", {})
                .get("value", error_text)
            )
        except Exception:
            sap_msg = error_text
        raise RuntimeError(f"SAP {resp.status_code}: {sap_msg}")

    return resp.json()


def _add_item(
    session: Any,
    base_url: str,
    csrf_token: str,
    cookies: Any,
    so_number: str,
    position: int,
    item: SalesOrderLineItem,
    client: str,
) -> None:
    """
    POST a single item to an existing Sales Order via A_SalesOrderItem.

    Dates are inherited from the SO header — do NOT include them here.
    204 No Content and 201 Created are both treated as success.
    """
    item_no = _item_number(position)
    url = f"{base_url}{_SO_API}/A_SalesOrderItem"

    item_payload = {
        "SalesOrder": so_number,
        "SalesOrderItem": item_no,
        "Material": item.sap_material or item.material_code,
        "RequestedQuantity": str(item.quantity),
        "RequestedQuantityUnit": item.uom,
    }

    logger.info(
        "Adding item %s | material=%s | qty=%s to SO %s",
        item_no,
        item_payload["Material"],
        item.quantity,
        so_number,
    )

    resp = session.post(
        url,
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        cookies=cookies,
        params={"sap-client": client},
        data=json.dumps(item_payload),
        timeout=60,
    )

    if resp.status_code in (200, 201, 204):
        logger.info(
            "Item %s added to SO %s (HTTP %s)", item_no, so_number, resp.status_code
        )
        return

    error_text = resp.text[:500]
    try:
        sap_msg = (
            resp.json()
            .get("error", {})
            .get("message", {})
            .get("value", error_text)
        )
    except Exception:
        sap_msg = error_text
    raise RuntimeError(
        f"Failed to add item {item_no} (material={item_payload['Material']}) "
        f"to SO {so_number}: SAP {resp.status_code}: {sap_msg}"
    )


def _add_header_text(
    session: Any,
    base_url: str,
    csrf_token: str,
    cookies: Any,
    so_number: str,
    text_id: str,
    text: str,
    client: str,
) -> None:
    """POST a header text entry to /A_SalesOrder('{so}')/to_Text."""
    url = f"{base_url}{_SO_API}/A_SalesOrder('{so_number}')/to_Text"

    payload = {
        "SalesOrder": so_number,
        "Language": "EN",
        "LongTextID": text_id,
        "LongText": text,
    }

    logger.info(
        "Adding header text to SO %s | TextID=%s | length=%d",
        so_number,
        text_id,
        len(text),
    )

    resp = session.post(
        url,
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        cookies=cookies,
        params={"sap-client": client},
        data=json.dumps(payload),
        timeout=60,
    )

    if resp.status_code in (200, 201, 204):
        logger.info(
            "Header text added to SO %s (HTTP %s)", so_number, resp.status_code
        )
        return

    error_text = resp.text[:500]
    try:
        sap_msg = (
            resp.json()
            .get("error", {})
            .get("message", {})
            .get("value", error_text)
        )
    except Exception:
        sap_msg = error_text
    raise RuntimeError(
        f"Failed to add header text to SO {so_number}: SAP {resp.status_code}: {sap_msg}"
    )


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------


def create_sales_order(
    data: CreateSORequest,
    request: Optional[Request] = None,
) -> CreateSOResponse:
    """
    Create a Sales Order in S/4HANA with all line items.

    Strategy:
      - POST first item via deep-insert (A_SalesOrder + to_Item)
      - POST each additional item individually to A_SalesOrderItem
      - If special_instructions present, POST header text (TX01) — non-fatal

    Returns:
        CreateSOResponse (success or failure with error message).
    """
    if not data.items:
        return CreateSOResponse(
            success=False,
            error="At least one item is required",
        )

    if not data.customer_bp:
        return CreateSOResponse(
            success=False,
            error="customer_bp (Business Partner number) is required",
        )

    session = get_s4_session(request)
    base_url = get_s4_base_url(request)
    client = settings.S4_CLIENT

    if not base_url:
        return CreateSOResponse(
            success=False,
            error="S/4HANA base URL not configured. Check .env or session headers.",
        )

    logger.info(
        "Creating Sales Order | customer_bp=%s | items=%d | sales_org=%s",
        data.customer_bp,
        len(data.items),
        data.sales_organization,
    )

    try:
        # Step 1: Fetch CSRF token
        csrf_token, cookies = _fetch_csrf(session, base_url, client)

        # Step 2: Create SO with first item (deep-insert)
        payload = _build_initial_payload(data)
        logger.info(
            "Creating initial SO | customer=%s | material=%s",
            data.customer_bp,
            data.items[0].sap_material or data.items[0].material_code,
        )

        result = _post_sales_order(session, base_url, csrf_token, cookies, payload, client)
        result_data = result.get("d") or result
        so_number = result_data.get("SalesOrder") or result_data.get("SalesOrderNumber", "")
        customer = result_data.get("SoldToParty") or data.customer_bp

        if not so_number:
            return CreateSOResponse(
                success=False,
                error="SAP did not return a Sales Order number. Response: " + str(result_data),
            )

        logger.info("Sales Order created: %s | customer: %s", so_number, customer)
        items_created = 1

        # Step 3: Add remaining items
        for idx, item in enumerate(data.items[1:], start=2):
            _add_item(
                session=session,
                base_url=base_url,
                csrf_token=csrf_token,
                cookies=cookies,
                so_number=so_number,
                position=idx,
                item=item,
                client=client,
            )
            items_created += 1
            logger.info(
                "Item %s added to SO %s (material=%s)",
                _item_number(idx),
                so_number,
                item.sap_material or item.material_code,
            )

        logger.info(
            "Sales Order %s completed | items_created=%d",
            so_number,
            items_created,
        )

        # Step 4: Add header text for special instructions (non-fatal)
        if data.special_instructions and data.special_instructions.strip():
            try:
                _add_header_text(
                    session=session,
                    base_url=base_url,
                    csrf_token=csrf_token,
                    cookies=cookies,
                    so_number=so_number,
                    text_id="TX01",
                    text=data.special_instructions.strip(),
                    client=client,
                )
                logger.info(
                    "Special instructions added to SO %s as header text (TX01)", so_number
                )
            except Exception as exc:
                logger.warning(
                    "Could not add header text to SO %s (non-fatal): %s", so_number, exc
                )

        return CreateSOResponse(
            success=True,
            sales_order=so_number,
            customer=customer,
            items_created=items_created,
            message=(
                f"Sales Order {so_number} created successfully "
                f"with {items_created} item(s)"
            ),
        )

    except RuntimeError as exc:
        logger.error("SAP error creating Sales Order: %s", exc)
        return CreateSOResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error creating Sales Order")
        return CreateSOResponse(success=False, error=str(exc))
