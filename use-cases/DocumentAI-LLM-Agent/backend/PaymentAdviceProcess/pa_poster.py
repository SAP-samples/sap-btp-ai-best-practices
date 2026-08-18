"""
pa_poster.py
------------
Posts a Payment Advice to S/4HANA FI via API_PAYMENT_ADVICE_SRV.

Flow:
1. Resolve payer Business Partner (BP override or auto-match by name)
2. Fetch CSRF token
3. Build deep-create payload: header + to_PaymentAdviceItem
4. POST to A_PaymentAdvice
5. Return PaymentAdvice document number
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
from PaymentAdviceProcess.pa_models import PostPaymentAdviceRequest
from config import settings
from matching.customer_api_matcher import search_customer_odata

logger = logging.getLogger(__name__)

API_PA = "API_PAYMENT_ADVICE_SRV"
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 60


def _parse_iso_date(date_str: str) -> dt.date:
    if not date_str or not date_str.strip():
        return dt.date.today()
    if date_str.startswith("/Date("):
        import re
        m = re.search(r"/Date\((-?\d+)", date_str)
        if m:
            return dt.datetime.fromtimestamp(int(m.group(1)) / 1000, tz=dt.timezone.utc).date()
    try:
        return dt.date.fromisoformat(date_str.strip()[:10])
    except ValueError:
        return dt.date.today()


def _fetch_csrf(session: requests.Session, base_url: str, client: str) -> tuple[str, dict]:
    url = f"{base_url}/sap/opu/odata/sap/{API_PA}/$metadata"
    resp = session.get(
        url,
        headers={"X-CSRF-Token": "Fetch"},
        params={"sap-client": client},
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    )
    token = resp.headers.get("X-CSRF-Token")
    if not token or token.lower() == "required":
        fallback = f"{base_url}/sap/opu/odata/sap/{API_PA}/A_PaymentAdvice"
        resp = session.get(
            fallback,
            headers={"X-CSRF-Token": "Fetch"},
            params={"sap-client": client, "$top": "0"},
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        token = resp.headers.get("X-CSRF-Token")
    if not token:
        raise RuntimeError(f"Could not obtain CSRF token (status={resp.status_code})")
    logger.info("CSRF token obtained for PA | status=%d", resp.status_code)
    return token, dict(resp.cookies)


def _find_payer_bp(payer_name: str, request: Optional[Request] = None) -> tuple[str, str]:
    logger.info("Searching BP for payer_name=%r", payer_name)
    results = search_customer_odata(payer_name, top=5, request=request)
    if not results:
        raise RuntimeError(f"No Business Partner found for payer: {payer_name!r}")
    best = results[0]
    if best["score"] < 0.5:
        raise RuntimeError(
            f"No confident BP match for payer {payer_name!r} "
            f"(best: {best['customer_name']!r}, score={best['score']:.2f})"
        )
    logger.info("Payer BP matched | bp=%s | name=%r | score=%.2f",
                best["business_partner"], best["customer_name"], best["score"])
    return best["business_partner"], best["customer_name"]


def post_payment_advice(
    data: PostPaymentAdviceRequest,
    request: Optional[Request] = None,
) -> dict:
    """
    Post a Payment Advice to S/4HANA FI via API_PAYMENT_ADVICE_SRV/A_PaymentAdvice.
    Returns a dict compatible with PostPaymentAdviceResponse.
    """
    config = get_sap_config(request)
    base_url = config.base_url or settings.S4_BASE_URL.rstrip("/")
    client   = config.client or settings.S4_CLIENT

    if not base_url:
        raise RuntimeError("S4_BASE_URL is not configured")
    if not data.currency or not data.currency.strip():
        raise ValueError("DocumentCurrency is required.")
    if not data.payer_name and not data.payer_bp:
        raise ValueError("Either payer_name or payer_bp must be provided.")

    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 1. Resolve BP
    if data.payer_bp:
        bp = data.payer_bp
        bp_display = data.payer_name
    else:
        bp, bp_display = _find_payer_bp(data.payer_name, request=request)

    # 2. Session + CSRF
    session = config.build_session()
    csrf_token, csrf_cookies = _fetch_csrf(session, base_url, client)

    # 3. Build payload
    payment_date = _parse_iso_date(data.payment_date)

    items_results = []
    for idx, item in enumerate(data.line_items, start=1):
        item_entry = {
            "PaymentAdviceItem":        str(idx).zfill(5),
            "AssignmentReference":      (item.invoice_number or item.payment_reference or "")[:18],
            "Currency":                 item.currency or data.currency,
            "GrossAmountInPaymentCurrency": str(item.gross_amount or item.net_payment_amount or 0),
            "NetPaymentAmountInPaytCurrency": str(item.net_payment_amount or item.gross_amount or 0),
            "DocumentItemText":         (f"Invoice {item.invoice_number}" if item.invoice_number else "Payment Advice item")[:50],
        }
        if item.discount_amount:
            item_entry["CashDiscountAmountInPaytCrcy"] = str(item.discount_amount)
        items_results.append(item_entry)

    # If no line items extracted, create one summary item
    if not items_results:
        items_results = [{
            "PaymentAdviceItem":        "00001",
            "AssignmentReference":      (data.bank_reference or "")[:18],
            "Currency":                 data.currency,
            "GrossAmountInPaymentCurrency": str(data.total_amount),
            "NetPaymentAmountInPaytCurrency": str(data.total_amount),
            "DocumentItemText":         (data.payment_advice_note or "Payment Advice from Document AI")[:50],
        }]

    payload = {
        "d": {
            "CompanyCode":               settings.FI_COMPANY_CODE,
            "PaymentAdviceAccountType":  "K",
            "PaymentAdviceAccount":      bp,
            "PaymentAdviceType":         "10",    # standard payment advice type
            "PaymentCurrency":           data.currency,
            "PaymentAdviceHeaderText":   (data.payment_advice_note or "Payment Advice from Document AI")[:25],
            "BankReference":             (data.bank_reference or "")[:35],
            "to_PaymentAdviceItem": {"results": items_results},
        }
    }

    logger.info(
        "Posting PaymentAdvice | company=%s | bp=%s | amount=%s %s | items=%d",
        settings.FI_COMPANY_CODE, bp, data.total_amount, data.currency, len(items_results),
    )

    # 4. POST
    post_url = f"{base_url}/sap/opu/odata/sap/{API_PA}/A_PaymentAdvice"
    headers = {
        "X-CSRF-Token": csrf_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    params = {"sap-client": client}

    try:
        resp = session.post(
            post_url, headers=headers, cookies=csrf_cookies, params=params,
            data=json.dumps(payload), timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"S/4HANA timed out posting Payment Advice") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc

    logger.info("A_PaymentAdvice POST | status=%d", resp.status_code)

    if not resp.ok:
        try:
            err = resp.json().get("error", {})
            sap_code = err.get("code", "")
            sap_msg  = err.get("message", {}).get("value", "")
        except Exception:
            sap_code, sap_msg = "", resp.text[:300]

        if "F5/201" in sap_code or "Posting period" in sap_msg:
            raise RuntimeError(
                f"Posting period not open. {sap_msg} — Open period in OB52 or use a date within an open period."
            )
        raise RuntimeError(f"S/4HANA rejected the Payment Advice: {sap_msg or f'HTTP {resp.status_code}'}")

    # 5. Parse response
    try:
        result_data = resp.json().get("d", {})
    except Exception as exc:
        raise RuntimeError(f"Failed to parse S/4HANA PA response: {exc}") from exc

    pa_number    = result_data.get("PaymentAdvice", "")
    company_code = result_data.get("CompanyCode", settings.FI_COMPANY_CODE)
    logger.info("PaymentAdvice posted | pa=%s | company=%s", pa_number, company_code)

    return {
        "success": True,
        "payment_advice": pa_number,
        "company_code": company_code,
        "business_partner_used": bp,
        "payer_name_matched": bp_display,
        "error": "",
    }
