"""
supplier_invoice_service.py
---------------------------
Service layer for posting Supplier Invoices to S/4HANA FI via
the API_SUPPLIERINVOICE_PROCESS_SRV OData API (A_SupplierInvoice).

Flow:
1. Resolve Business Partner: use override if provided, else auto-match by name
2. Fetch CSRF token (same pattern as backend/post/create/post.py)
3. Build payload with header + GL account line item
4. POST to A_SupplierInvoice with deep-insert
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
from InvoiceProcess.supplier_invoice.supplier_invoice_models import PostInvoiceRequest
from config import settings
from matching.customer_api_matcher import search_customer_odata

logger = logging.getLogger(__name__)

API_SUPPLIER_INVOICE = "API_SUPPLIERINVOICE_PROCESS_SRV"

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Date helper (replicates backend/post/create/post.py::sap_date)
# ---------------------------------------------------------------------------

def _sap_date(d: dt.date) -> str:
    """Convert a date to SAP OData /Date(ms)/ format."""
    ms = int(
        dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    )
    return f"/Date({ms})/"


# ---------------------------------------------------------------------------
# Invoice number sanitizer
# ---------------------------------------------------------------------------

def _sanitize_invoice_number(raw: str) -> str:
    """
    Sanitize invoice number for SAP SupplierInvoiceIDByInvcgParty.
    SAP allows max 16 alphanumeric chars (letters, digits, hyphen, slash).
    Strips leading/trailing spaces, removes invalid chars, truncates to 16.
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
# CSRF token fetch (replicates backend/post/create/post.py::fetch_csrf)
# ---------------------------------------------------------------------------

def _fetch_csrf(session: requests.Session, base_url: str, client: str) -> tuple[str, dict]:
    """
    Fetch a CSRF token from the Supplier Invoice metadata endpoint.
    Returns (token, cookies).
    """
    url = f"{base_url}/sap/opu/odata/sap/{API_SUPPLIER_INVOICE}/$metadata"
    params = {"sap-client": client}

    logger.info("Fetching CSRF token | url=%s", url)

    resp = session.get(
        url,
        headers={"X-CSRF-Token": "Fetch"},
        params=params,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    )

    token = resp.headers.get("X-CSRF-Token")
    if not token or token.lower() == "required":
        # Fallback: try $top=0 on A_SupplierInvoice
        fallback_url = (
            f"{base_url}/sap/opu/odata/sap/{API_SUPPLIER_INVOICE}/A_SupplierInvoice"
        )
        resp = session.get(
            fallback_url,
            headers={"X-CSRF-Token": "Fetch"},
            params={**params, "$top": "0"},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        token = resp.headers.get("X-CSRF-Token")

    if not token:
        raise RuntimeError(
            f"Could not obtain CSRF token from S/4HANA (status={resp.status_code})"
        )

    logger.info("CSRF token obtained | status=%d", resp.status_code)
    return token, dict(resp.cookies)


# ---------------------------------------------------------------------------
# Vendor BP resolution
# ---------------------------------------------------------------------------

def find_vendor_bp(supplier_name: str, request: Optional[Request] = None) -> tuple[str, str]:
    """
    Find the Business Partner number for a supplier name.

    Returns (business_partner_number, matched_display_name).
    Raises RuntimeError if no match is found with score > 0.5.
    """
    logger.info("Searching BP for supplier_name=%r", supplier_name)

    results = search_customer_odata(supplier_name, top=5, request=request)

    if not results:
        raise RuntimeError(
            f"No Business Partner found in S/4HANA for supplier: {supplier_name!r}"
        )

    best = results[0]
    if best["score"] < 0.5:
        raise RuntimeError(
            f"No confident BP match for supplier {supplier_name!r} "
            f"(best: {best['customer_name']!r}, score={best['score']:.2f})"
        )

    logger.info(
        "BP matched | supplier=%r | bp=%s | name=%r | score=%.2f",
        supplier_name,
        best["business_partner"],
        best["customer_name"],
        best["score"],
    )
    return best["business_partner"], best["customer_name"]


# ---------------------------------------------------------------------------
# Main service function
# ---------------------------------------------------------------------------

def post_supplier_invoice(
    data: PostInvoiceRequest,
    request: Optional[Request] = None,
) -> dict:
    """
    Post a Supplier Invoice to S/4HANA FI via A_SupplierInvoice.

    Returns a dict compatible with PostInvoiceResponse.
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
    if not data.supplier_name and not data.business_partner:
        raise ValueError("Either supplier_name or business_partner must be provided.")

    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 1. Resolve Business Partner
    if data.business_partner:
        bp = data.business_partner
        bp_display_name = data.supplier_name
        logger.info("Using provided business_partner=%s", bp)
    else:
        bp, bp_display_name = find_vendor_bp(data.supplier_name, request=request)

    # 2. Build session
    session = config.build_session()

    # 3. Fetch CSRF token
    csrf_token, csrf_cookies = _fetch_csrf(session, base_url, client)

    # 4. Build payload
    invoice_date = _parse_iso_date(data.invoice_date)
    today = dt.date.today()

    payload = {
        "CompanyCode": settings.FI_COMPANY_CODE,
        "DocumentDate": _sap_date(invoice_date),
        "PostingDate": _sap_date(today),
        "SupplierInvoiceIDByInvcgParty": _sanitize_invoice_number(data.invoice_number),
        "InvoicingParty": bp,
        "DocumentCurrency": data.currency,
        "InvoiceGrossAmount": str(data.total_amount),
        "TaxIsCalculatedAutomatically": True,
        "to_SupplierInvoiceItemGLAcct": {
            "results": [
                {
                    "SupplierInvoiceItem": "1",
                    "GLAccount": data.gl_account.strip() or settings.FI_EXPENSE_GL_ACCOUNT,
                    "DocumentCurrency": data.currency,
                    "SupplierInvoiceItemAmount": str(data.total_amount),
                    "DebitCreditCode": "S",
                    "TaxCode": settings.FI_PO_TAX_CODE,
                }
            ]
        },
    }

    logger.info(
        "Posting SupplierInvoice | company=%s | bp=%s | amount=%s %s | invoice_no=%s",
        settings.FI_COMPANY_CODE,
        bp,
        data.total_amount,
        data.currency,
        data.invoice_number,
    )

    # 5. POST
    post_url = (
        f"{base_url}/sap/opu/odata/sap/{API_SUPPLIER_INVOICE}/A_SupplierInvoice"
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
            f"S/4HANA did not respond within {_READ_TIMEOUT}s when posting invoice"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc

    logger.info("A_SupplierInvoice POST | status=%d | url=%s", resp.status_code, post_url)

    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            sap_code = err.get("code", "")
            sap_msg  = err.get("message", {}).get("value", "")
            # Also collect any detail messages
            details  = err.get("innererror", {}).get("errordetails", [])
            detail_msgs = " | ".join(d.get("message", "") for d in details if d.get("message"))
        except Exception:
            sap_code, sap_msg, detail_msgs = "", resp.text[:300], ""

        # F5/201 = Posting period not open
        if "F5/201" in sap_code or "Posting period" in sap_msg:
            raise RuntimeError(
                f"Posting period not open in S/4HANA. {sap_msg} — "
                "Please open the posting period in transaction OB52 or use a date within an open period."
            )

        human_msg = sap_msg or detail_msgs or f"HTTP {resp.status_code}"
        raise RuntimeError(f"S/4HANA rejected the invoice: {human_msg}")

    # 6. Parse response
    try:
        result_data = resp.json().get("d", {})
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse S/4HANA response after posting invoice: {exc}"
        ) from exc

    fi_document = result_data.get("SupplierInvoice", "")
    company_code = result_data.get("CompanyCode", settings.FI_COMPANY_CODE)
    fiscal_year = result_data.get("FiscalYear", "")

    logger.info(
        "SupplierInvoice posted | fi_document=%s | company_code=%s | fiscal_year=%s",
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
        "error": "",
    }
