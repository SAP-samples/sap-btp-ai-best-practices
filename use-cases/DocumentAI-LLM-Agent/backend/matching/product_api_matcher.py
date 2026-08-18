"""
product_api_matcher.py
----------------------
Search Materials/Products in S/4HANA by material code or description.

OData V2 on-premise uses substringof() — NOT contains() (V4 syntax).

Strategy:
1. If query looks like a SAP code (letters+digits+separators):
   a. Exact GET A_Product('{code}') — score 1.0 if found
   b. substringof on Product code
2. If query looks like a description:
   a. substringof on ProductDescription via A_ProductDescription
   b. Fetch full product details for each code found
3. Fallback (both paths): fetch top-200 A_Product with to_Description,
   score client-side with rapidfuzz
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests
import urllib3
from fastapi import Request
from rapidfuzz import fuzz

from S4.sap_credentials import get_sap_config
from config import settings

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_FALLBACK_TOP = 200
_MIN_SCORE = 0.60

_CODE_RE = re.compile(r"^[A-Z0-9=\-_/\.]+$")


def _looks_like_code(text: str) -> bool:
    """Return True if query looks like a SAP material code (letters+digits+separators)."""
    t = text.upper().strip()
    if len(t) < 3:
        return False
    if not _CODE_RE.fullmatch(t):
        return False
    has_letter = any(c.isalpha() for c in t)
    has_digit = any(c.isdigit() for c in t)
    return has_letter and has_digit


def _compact(s: str) -> str:
    """Remove all whitespace, uppercase."""
    return re.sub(r"\s+", "", s.upper())


def _canonical(s: str) -> str:
    """Remove whitespace and separators, uppercase — for normalized code matching."""
    return re.sub(r"[\s\-_/\.=]+", "", s.upper())


def _get_description(item: dict) -> str:
    """Extract best available description from a product record (prefer EN, then DE)."""
    descriptions = (item.get("to_Description") or {}).get("results", [])
    for lang in ("EN", "DE"):
        for d in descriptions:
            if d.get("Language", "") == lang:
                desc = d.get("ProductDescription", "").strip()
                if desc:
                    return desc
    if descriptions:
        return descriptions[0].get("ProductDescription", "").strip()
    return ""


def _score(
    code_query: str,
    desc_query: str,
    material_code: str,
    material_desc: str,
) -> tuple[float, str]:
    """Score a candidate product against the query using rapidfuzz."""
    cq_norm = _compact(code_query)
    mc_norm = _compact(material_code)

    # Exact code match (spaces collapsed, uppercase)
    if cq_norm and mc_norm and cq_norm == mc_norm:
        return 1.0, "EXACT_CODE_MATCH"

    # Normalized code match (separators removed)
    cq_can = _canonical(code_query)
    mc_can = _canonical(material_code)
    if cq_can and mc_can and cq_can == mc_can:
        return 0.99, "NORMALIZED_CODE_MATCH"

    # Partial code match (one contains the other after canonicalization)
    # Require at least 50% length ratio to avoid short codes matching inside long ones
    # e.g. "2" should NOT match inside "MXA920WS60CM"
    if cq_can and mc_can and (cq_can in mc_can or mc_can in cq_can):
        ratio = min(len(cq_can), len(mc_can)) / max(len(cq_can), len(mc_can))
        if ratio >= 0.5:
            return 0.95, "PARTIAL_CODE_MATCH"

    # Description matching
    dq = desc_query.strip().upper()
    md = material_desc.strip().upper()
    if dq and md:
        if dq == md:
            return 0.99, "EXACT_DESC_MATCH"
        if dq in md or md in dq:
            return 0.95, "CONTAINS_DESC_MATCH"

    # Fuzzy match using combined code+description
    combined_q = f"{code_query} {desc_query}".strip()
    combined_r = f"{material_code} {material_desc}".strip()
    score = fuzz.token_set_ratio(combined_q, combined_r) / 100.0
    return score, "FUZZY_DESC_MATCH"


def _fetch(session, endpoint: str, params: dict) -> list[dict]:
    resp = session.get(endpoint, params=params, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    resp.raise_for_status()
    return resp.json().get("d", {}).get("results", [])


def search_material_odata(
    q: str,
    top: int = 10,
    request: Optional[Request] = None,
) -> list[dict]:
    """
    Search Products/Materials in S/4HANA by code or description.

    Returns list of dicts: product, description, score, confidence.
    Sorted by score descending.
    """
    config = get_sap_config(request)
    base_url = config.base_url or settings.S4_BASE_URL.rstrip("/")

    if not base_url:
        raise RuntimeError("S4_BASE_URL is not configured")
    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = config.build_session()
    client = config.client or settings.S4_CLIENT
    endpoint = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product"
    desc_endpoint = f"{base_url}/sap/opu/odata/sap/API_PRODUCT_SRV/A_ProductDescription"
    q_escaped = q.replace("'", "''")

    is_code = _looks_like_code(q)
    code_query = q if is_code else ""
    desc_query = q if not is_code else ""

    def build_results(raw_list: list[dict]) -> list[dict]:
        out = []
        for item in raw_list:
            code = item.get("Product", "")
            desc = _get_description(item)
            score, conf = _score(code_query, desc_query, code, desc)
            if score >= _MIN_SCORE:
                out.append({
                    "product": code,
                    "description": desc,
                    "score": round(score, 4),
                    "confidence": conf,
                })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top]

    base_params = {
        "sap-client": client,
        "$format": "json",
        "$top": str(top),
        "$expand": "to_Description",
    }

    if is_code:
        # ── Strategy 1a: exact Product code GET ─────────────────────────────
        try:
            resp = session.get(
                f"{endpoint}('{q_escaped}')",
                params={"sap-client": client, "$format": "json", "$expand": "to_Description"},
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if resp.ok:
                raw = resp.json().get("d", {})
                if raw.get("Product"):
                    desc = _get_description(raw)
                    logger.info("material_search exact_code | q=%r | found=1", q)
                    return [{"product": raw["Product"], "description": desc, "score": 1.0, "confidence": "EXACT_CODE_MATCH"}]
        except Exception:
            pass

        # ── Strategy 1b: substringof on Product code ─────────────────────────
        try:
            params = {**base_params, "$filter": f"substringof('{q_escaped}',Product)"}
            raw = _fetch(session, endpoint, params)
            if raw:
                results = build_results(raw)
                if results:
                    logger.info("material_search substringof(Product) | q=%r | found=%d", q, len(results))
                    return results
        except requests.exceptions.HTTPError:
            pass
        except Exception:
            pass

        # ── Strategy 1c: code-like query may also be an OEM model / description ─
        # Try description search as well (e.g. "MXA920W-S-60CM" stored as description)
        try:
            desc_params = {
                "sap-client": client,
                "$format": "json",
                "$top": str(top),
                "$filter": f"substringof('{q_escaped}',ProductDescription)",
                "$select": "Product,ProductDescription,Language",
            }
            raw_desc = _fetch(session, desc_endpoint, desc_params)
            if raw_desc:
                codes = list({r.get("Product", "") for r in raw_desc if r.get("Product")})[:top]
                results = []
                for code in codes:
                    code_escaped = code.replace("'", "''")
                    try:
                        resp = session.get(
                            f"{endpoint}('{code_escaped}')",
                            params={"sap-client": client, "$format": "json", "$expand": "to_Description"},
                            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                        )
                        if resp.ok:
                            p_raw = resp.json().get("d", {})
                            if p_raw.get("Product"):
                                desc = _get_description(p_raw)
                                # Score using description query path so the desc match logic fires
                                s, c = _score("", q, p_raw["Product"], desc)
                                if s >= _MIN_SCORE:
                                    results.append({
                                        "product": p_raw["Product"],
                                        "description": desc,
                                        "score": round(s, 4),
                                        "confidence": c,
                                    })
                    except Exception:
                        pass
                if results:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    logger.info("material_search code_as_desc | q=%r | found=%d", q, len(results))
                    return results[:top]
        except Exception:
            pass

    else:
        # ── Strategy 2: substringof on ProductDescription ────────────────────
        try:
            params = {
                "sap-client": client,
                "$format": "json",
                "$top": str(top),
                "$filter": f"substringof('{q_escaped}',ProductDescription)",
                "$select": "Product,ProductDescription,Language",
            }
            raw_desc = _fetch(session, desc_endpoint, params)
            if raw_desc:
                codes = list({r.get("Product", "") for r in raw_desc if r.get("Product")})[:top]
                results = []
                for code in codes:
                    code_escaped = code.replace("'", "''")
                    try:
                        resp = session.get(
                            f"{endpoint}('{code_escaped}')",
                            params={"sap-client": client, "$format": "json", "$expand": "to_Description"},
                            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                        )
                        if resp.ok:
                            p_raw = resp.json().get("d", {})
                            if p_raw.get("Product"):
                                results.extend(build_results([p_raw]))
                    except Exception:
                        pass
                if results:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    logger.info("material_search description | q=%r | found=%d", q, len(results))
                    return results[:top]
        except Exception:
            pass

    # ── Fallback: fetch top-200 and score client-side with rapidfuzz ─────────
    try:
        params = {**base_params, "$top": str(_FALLBACK_TOP)}
        raw = _fetch(session, endpoint, params)
        results = build_results(raw)
        logger.info("material_search fallback | q=%r | found=%d", q, len(results))
        return results[:top]
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 0
        raise RuntimeError(f"S/4HANA returned HTTP {status_code} for material search") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("S/4HANA material search timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc
