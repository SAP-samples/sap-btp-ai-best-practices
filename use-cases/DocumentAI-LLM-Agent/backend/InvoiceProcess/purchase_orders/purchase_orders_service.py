"""
purchase_orders_service.py
--------------------------
Searches Purchase Orders by vendor (Supplier) from S/4HANA.

API: API_PURCHASEORDER_PROCESS_SRV (OData V2) — /A_PurchaseOrder
     Tries V4 (API_PURCHASEORDER_2) first on S/4HANA 2023+; falls back to V2.

Filter: Supplier eq '<bp_number>'
Returns list of open POs with header + item summary.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import requests
import urllib3
from fastapi import Request

from S4.sap_credentials import get_sap_config
from config import settings

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_PAGE_SIZE = 50

# OData V2 (universal, on-premise OP2022+)
_API_V2 = "API_PURCHASEORDER_PROCESS_SRV"
_ENTITY_V2 = "A_PurchaseOrder"

# OData V4 (S/4HANA Cloud / OP2023+) — tried first, graceful fallback
_API_V4 = "API_PURCHASEORDER_2"
_ENTITY_V4 = "PurchaseOrder"


def _parse_odata_date(raw: str | None) -> str | None:
    """Convert /Date(ms)/ → YYYY-MM-DD, or return ISO string as-is."""
    if not raw:
        return None
    match = re.search(r"/Date\((-?\d+)(?:[+-]\d+)?\)/", str(raw))
    if match:
        from datetime import datetime, timezone
        ms = int(match.group(1))
        try:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, ValueError):
            return None
    return str(raw) if raw else None


def _map_po(raw: dict) -> dict:
    """Map raw OData PO record to a clean dict."""
    return {
        "purchase_order":         raw.get("PurchaseOrder", ""),
        "supplier":               raw.get("Supplier", ""),
        "company_code":           raw.get("CompanyCode", ""),
        "purchasing_organization": raw.get("PurchasingOrganization", ""),
        "purchasing_group":       raw.get("PurchasingGroup", ""),
        "document_date":          _parse_odata_date(raw.get("PurchaseOrderDate")),
        "currency":               raw.get("DocumentCurrency", ""),
        "status":                 raw.get("PurchasingProcessingStatus", ""),
        "supplier_name":          raw.get("AddressName", ""),
    }


def _fetch_pos_v2(
    session: requests.Session,
    base_url: str,
    client: str,
    supplier: str,
    top: int,
) -> list[dict]:
    """Fetch POs via OData V2 API_PURCHASEORDER_PROCESS_SRV."""
    url = f"{base_url}/sap/opu/odata/sap/{_API_V2}/{_ENTITY_V2}"
    safe = supplier.replace("'", "''")
    params = {
        "sap-client": client,
        "$format":    "json",
        "$top":       str(top),
        "$filter":    f"Supplier eq '{safe}'",
        "$select":    (
            "PurchaseOrder,Supplier,CompanyCode,PurchasingOrganization,"
            "PurchasingGroup,PurchaseOrderDate,DocumentCurrency,"
            "PurchasingProcessingStatus,AddressName"
        ),
        "$orderby": "PurchaseOrderDate desc",
    }
    logger.info("PO search V2 | supplier=%s | url=%s", supplier, url)
    resp = session.get(url, params=params, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    resp.raise_for_status()
    raw_list: list[dict] = resp.json().get("d", {}).get("results", [])
    return [_map_po(r) for r in raw_list]


def _fetch_pos_v4(
    session: requests.Session,
    base_url: str,
    client: str,
    supplier: str,
    top: int,
) -> list[dict]:
    """Fetch POs via OData V4 API_PURCHASEORDER_2 (S/4HANA 2023+ Cloud)."""
    url = f"{base_url}/sap/odata/sap/{_API_V4}/{_ENTITY_V4}"
    safe = supplier.replace("'", "''")
    params = {
        "sap-client": client,
        "$format":    "json",
        "$top":       str(top),
        "$filter":    f"Supplier eq '{safe}'",
        "$select":    (
            "PurchaseOrder,Supplier,CompanyCode,PurchasingOrganization,"
            "PurchasingGroup,PurchaseOrderDate,DocumentCurrency,AddressName"
        ),
        "$orderby": "PurchaseOrderDate desc",
    }
    logger.info("PO search V4 | supplier=%s | url=%s", supplier, url)
    resp = session.get(url, params=params, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    resp.raise_for_status()
    # V4 returns value array directly
    data = resp.json()
    raw_list: list[dict] = data.get("value") or data.get("d", {}).get("results", [])
    return [_map_po(r) for r in raw_list]


def search_purchase_orders_sync(
    supplier: str,
    top: int = 20,
    request: Optional[Request] = None,
) -> list[dict]:
    """
    Search open Purchase Orders by vendor/supplier number.

    Tries OData V4 first (S/4HANA 2023+), falls back to V2.
    Returns list of PO dicts sorted by date descending.
    """
    config = get_sap_config(request)
    base_url = config.base_url or settings.S4_BASE_URL.rstrip("/")
    client   = config.client or settings.S4_CLIENT

    if not base_url:
        raise RuntimeError("S4_BASE_URL is not configured")
    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = config.build_session()
    safe_top = max(1, min(top, 200))

    # Try V4 first (graceful fallback to V2 on 404/405/403)
    try:
        pos = _fetch_pos_v4(session, base_url, client, supplier, safe_top)
        logger.info("PO search V4 OK | supplier=%s | found=%d", supplier, len(pos))
        return pos
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in (404, 405, 403, 501):
            logger.info("V4 not available (HTTP %d), falling back to V2", status)
        else:
            logger.warning("V4 error HTTP %d, falling back to V2", status)
    except Exception as exc:
        logger.info("V4 unavailable (%s), falling back to V2", type(exc).__name__)

    # V2 fallback
    try:
        pos = _fetch_pos_v2(session, base_url, client, supplier, safe_top)
        logger.info("PO search V2 OK | supplier=%s | found=%d", supplier, len(pos))
        return pos
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status == 401:
            raise RuntimeError("Authentication failed — check S4 credentials") from exc
        if status == 403:
            raise RuntimeError("Access denied — user lacks authorization for API_PURCHASEORDER_PROCESS_SRV") from exc
        if status == 404:
            raise RuntimeError("Purchase Order API endpoint not found — check S4_BASE_URL") from exc
        raise RuntimeError(f"S/4HANA returned HTTP {status} for PO search") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("S/4HANA PO search timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc


async def search_purchase_orders(
    supplier: str,
    top: int = 20,
    request: Optional[Request] = None,
) -> list[dict]:
    """Async wrapper — runs blocking HTTP in a thread pool."""
    return await asyncio.to_thread(
        search_purchase_orders_sync, supplier, top, request
    )
