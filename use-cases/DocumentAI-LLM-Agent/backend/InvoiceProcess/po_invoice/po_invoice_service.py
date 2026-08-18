"""
po_invoice_service.py
---------------------
Posts a PO-based Supplier Invoice to S/4HANA via MIRO-equivalent deep create.

API: API_SUPPLIERINVOICE_PROCESS_SRV / A_SupplierInvoice
PO reference node: to_SuplrInvcItemPurOrdRef

Flow:
1. Resolve Business Partner (BP override or auto-match by supplier name)
2. Fetch CSRF token from the service metadata endpoint
3. Build deep-create payload with PO reference line item
4. POST to A_SupplierInvoice
5. Return FI document number, company code, fiscal year
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Optional

import requests
import urllib3
from fastapi import Request

from S4.sap_credentials import get_sap_config
from InvoiceProcess.po_invoice.po_invoice_models import PostPOInvoiceRequest
from config import settings
from matching.customer_api_matcher import search_customer_odata

logger = logging.getLogger(__name__)

API_SERVICE = "API_SUPPLIERINVOICE_PROCESS_SRV"

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _sap_datetime(d: dt.date) -> str:
    """Return SAP OData datetime string for date-only values (time part 00:00)."""
    return d.strftime("%Y-%m-%dT00:00:00")


# ---------------------------------------------------------------------------
# Invoice number sanitizer
# ---------------------------------------------------------------------------

def _sanitize_invoice_number(raw: str) -> str:
    """
    Sanitize invoice number for SAP SupplierInvoiceIDByInvcgParty.
    SAP allows max 16 alphanumeric chars (letters, digits, hyphen, slash).
    """
    import re
    if not raw:
        return "UNKNOWN"
    clean = re.sub(r"[^A-Za-z0-9\-/]", "", raw.strip())
    return clean[:16] if clean else "UNKNOWN"


def _parse_iso_date(date_str: str) -> dt.date:
    """Parse ISO date string YYYY-MM-DD. Falls back to today if empty or unparseable."""
    if not date_str or not date_str.strip():
        logger.warning("invoice_date is empty — using today as fallback")
        return dt.date.today()
    # Handle /Date(ms)/ from SAP OData in case the frontend passes it raw
    if date_str.startswith("/Date("):
        import re
        m = re.search(r"/Date\((-?\d+)", date_str)
        if m:
            return dt.datetime.fromtimestamp(int(m.group(1)) / 1000, tz=dt.timezone.utc).date()
    try:
        return dt.date.fromisoformat(date_str.strip()[:10])
    except ValueError:
        logger.warning("invoice_date %r is not YYYY-MM-DD — using today as fallback", date_str)
        return dt.date.today()


# ---------------------------------------------------------------------------
# CSRF token fetch
# ---------------------------------------------------------------------------

def _fetch_csrf(
    session: requests.Session,
    base_url: str,
    client: str,
) -> tuple[str, dict]:
    """
    Fetch X-CSRF-Token from the service $metadata endpoint.
    Falls back to $top=0 on A_SupplierInvoice if metadata doesn't return a token.
    Returns (token, cookies).
    """
    meta_url = f"{base_url}/sap/opu/odata/sap/{API_SERVICE}/$metadata"
    params = {"sap-client": client}

    logger.info("Fetching CSRF token | url=%s", meta_url)

    resp = session.get(
        meta_url,
        headers={"X-CSRF-Token": "Fetch"},
        params=params,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    )
    token = resp.headers.get("X-CSRF-Token")

    if not token or token.lower() == "required":
        fallback_url = f"{base_url}/sap/opu/odata/sap/{API_SERVICE}/A_SupplierInvoice"
        resp = session.get(
            fallback_url,
            headers={"X-CSRF-Token": "Fetch"},
            params={**params, "$top": "0"},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        token = resp.headers.get("X-CSRF-Token")

    if not token:
        raise RuntimeError(
            f"Could not obtain CSRF token (status={resp.status_code})"
        )

    logger.info("CSRF token obtained | status=%d", resp.status_code)
    return token, dict(resp.cookies)


# ---------------------------------------------------------------------------
# Vendor BP resolution
# ---------------------------------------------------------------------------

def _find_vendor_bp(
    supplier_name: str,
    request: Optional[Request] = None,
) -> tuple[str, str]:
    """
    Resolve Business Partner number for a supplier name.
    Returns (bp_number, matched_display_name).
    Raises RuntimeError if no confident match is found.
    """
    logger.info("Searching BP for supplier=%r", supplier_name)
    results = search_customer_odata(supplier_name, top=5, request=request)

    if not results:
        raise RuntimeError(
            f"No Business Partner found in S/4HANA for supplier: {supplier_name!r}"
        )

    best = results[0]
    if best["score"] < 0.5:
        raise RuntimeError(
            f"No confident BP match for {supplier_name!r} "
            f"(best: {best['customer_name']!r}, score={best['score']:.2f})"
        )

    logger.info(
        "BP matched | bp=%s | name=%r | score=%.2f",
        best["business_partner"],
        best["customer_name"],
        best["score"],
    )
    return best["business_partner"], best["customer_name"]


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def post_po_invoice(
    data: PostPOInvoiceRequest,
    request: Optional[Request] = None,
) -> dict:
    """
    Post a PO-based Supplier Invoice to S/4HANA FI (MIRO equivalent).

    Uses a deep-create POST to A_SupplierInvoice with the
    to_SuplrInvcItemPurOrdRef nested node for PO line item reference.

    Returns a dict compatible with PostPOInvoiceResponse.
    """
    config = get_sap_config(request)
    base_url = config.base_url or settings.S4_BASE_URL.rstrip("/")
    client = config.client or settings.S4_CLIENT

    if not base_url:
        raise RuntimeError("S4_BASE_URL is not configured")

    # Validate required fields before calling SAP
    if not data.currency or not data.currency.strip():
        raise ValueError(
            "DocumentCurrency is required but was not extracted from the invoice. "
            "Please check the extraction result and ensure the currency field is populated."
        )
    if not data.purchase_order or not data.purchase_order.strip():
        raise ValueError("PurchaseOrder number is required for PO-based invoice posting.")

    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 1. Resolve Business Partner
    if data.business_partner:
        bp = data.business_partner
        bp_display_name = data.supplier_name
        logger.info("Using provided business_partner=%s", bp)
    else:
        bp, bp_display_name = _find_vendor_bp(data.supplier_name, request=request)

    # 2. Resolve tax code
    tax_code = data.tax_code or settings.FI_PO_TAX_CODE

    # 3. Build session + CSRF token
    session = config.build_session()
    csrf_token, csrf_cookies = _fetch_csrf(session, base_url, client)

    # 4. Build payload — SAP PO invoice uses datetime strings, NOT /Date(ms)/
    invoice_date = _parse_iso_date(data.invoice_date)
    today = dt.date.today()
    line_amount = data.total_amount  # single PO line carries full amount

    payload = {
        "d": {
            "CompanyCode": settings.FI_COMPANY_CODE,
            "DocumentDate": _sap_datetime(invoice_date),
            "PostingDate": _sap_datetime(today),
            "SupplierInvoiceIDByInvcgParty": _sanitize_invoice_number(data.invoice_number),
            "InvoicingParty": bp,
            "DocumentCurrency": data.currency,
            "InvoiceGrossAmount": str(data.total_amount),
            "TaxIsCalculatedAutomatically": True,
            "to_SuplrInvcItemPurOrdRef": {
                "results": [
                    {
                        "SupplierInvoiceItem": "00001",
                        "PurchaseOrder": data.purchase_order,
                        "PurchaseOrderItem": data.purchase_order_item or "00010",
                        "DocumentCurrency": data.currency,
                        "SupplierInvoiceItemAmount": str(line_amount),
                        "TaxCode": tax_code,
                    }
                ]
            },
        }
    }

    logger.info(
        "Posting PO SupplierInvoice | company=%s | bp=%s | po=%s | amount=%s %s",
        settings.FI_COMPANY_CODE,
        bp,
        data.purchase_order,
        data.total_amount,
        data.currency,
    )

    # 5. POST
    post_url = (
        f"{base_url}/sap/opu/odata/sap/{API_SERVICE}/A_SupplierInvoice"
    )
    headers = {
        "X-CSRF-Token": csrf_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    params = {"sap-client": client}

    try:
        resp = session.post(
            post_url,
            headers=headers,
            cookies=csrf_cookies,
            params=params,
            data=json.dumps(payload),
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"S/4HANA did not respond within {_READ_TIMEOUT}s when posting PO invoice"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc

    logger.info("A_SupplierInvoice PO POST | status=%d", resp.status_code)

    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            sap_code = err.get("code", "")
            sap_msg  = err.get("message", {}).get("value", "")
        except Exception:
            sap_code = ""
            sap_msg  = resp.text[:300]

        # M8/375 = GR-Based IV: Goods Receipt not posted yet
        if "M8/375" in sap_code or "ReferenceDocument" in sap_msg:
            raise RuntimeError(
                f"PO {data.purchase_order} requires a Goods Receipt before invoicing. "
                "No GR found for this PO. Please post the Goods Receipt first in MIGO (transaction MIGO / movement type 101), "
                "then retry the invoice posting."
            )

        raise RuntimeError(
            f"S/4HANA returned HTTP {resp.status_code} posting PO invoice. "
            f"Details: {sap_msg or resp.text[:300]}"
        )

    # 6. Parse response
    try:
        result_data = resp.json().get("d", {})
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse S/4HANA PO invoice response: {exc}"
        ) from exc

    fi_document = result_data.get("SupplierInvoice", "")
    company_code = result_data.get("CompanyCode", settings.FI_COMPANY_CODE)
    fiscal_year = result_data.get("FiscalYear", "")

    logger.info(
        "PO invoice posted | fi_document=%s | company=%s | fy=%s",
        fi_document,
        company_code,
        fiscal_year,
    )

    return {
        "success": True,
        "fi_document": fi_document,
        "company_code": company_code,
        "fiscal_year": fiscal_year,
        "business_partner_used": bp,
        "supplier_name_matched": bp_display_name,
        "purchase_order": data.purchase_order,
        "purchase_order_item": data.purchase_order_item or "00010",
        "error": "",
    }
