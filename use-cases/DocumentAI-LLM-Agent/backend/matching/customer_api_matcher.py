"""
customer_api_matcher.py
-----------------------
Search Business Partners in S/4HANA by name.

OData V2 on-premise uses substringof() — NOT contains() (that is V4 syntax).
Strategy:
  1. Try substringof() on BusinessPartnerName (V2 compatible)
  2. If that returns HTTP 400 (some systems reject substringof on that field),
     fall back to fetching by OrganizationBPName1 substringof
  3. If both fail, fetch $top=200 without filter and do client-side matching

All returned results are scored client-side with rapidfuzz for consistent ranking.
Returns a ranked list of matches with score and confidence level.
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
_FALLBACK_TOP = 200  # fetch without filter and match client-side

# Tier ordering for final sort
_TIER_ORDER = {
    "EXACT_CODE": 0,
    "EXACT_NAME": 1,
    "PARTIAL_NAME": 2,
    "FUZZY": 3,
}


def _norm(text: str) -> str:
    """Normalize: uppercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.upper().strip())


def _fetch_bp(
    session: requests.Session,
    endpoint: str,
    params: dict,
) -> list[dict]:
    """GET helper — returns raw results list or raises."""
    resp = session.get(endpoint, params=params, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
    resp.raise_for_status()
    return resp.json().get("d", {}).get("results", [])


def _token_overlap_scores(query_norm: str, name_norm: str) -> tuple[float, float, float]:
    """
    Returns (jaccard, query_coverage, name_coverage) based on word token sets.
    Ignores tokens shorter than 2 characters.
    """
    q_tokens = {t for t in query_norm.split() if len(t) >= 2}
    n_tokens = {t for t in name_norm.split() if len(t) >= 2}
    if not q_tokens or not n_tokens:
        return 0.0, 0.0, 0.0
    intersection = q_tokens & n_tokens
    union = q_tokens | n_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    query_coverage = len(intersection) / len(q_tokens) if q_tokens else 0.0
    name_coverage = len(intersection) / len(n_tokens) if n_tokens else 0.0
    return jaccard, query_coverage, name_coverage


def _fuzzy_name_score(query_norm: str, name_norm: str) -> float:
    """Weighted combination of rapidfuzz metrics for name matching."""
    jaccard, query_coverage, _ = _token_overlap_scores(query_norm, name_norm)
    token_sort = fuzz.token_sort_ratio(query_norm, name_norm) / 100.0
    char_ratio = fuzz.ratio(query_norm, name_norm) / 100.0
    score = (
        0.45 * token_sort
        + 0.30 * char_ratio
        + 0.15 * jaccard
        + 0.10 * query_coverage
    )
    return score


def _score_candidate(query_norm: str, cand: dict) -> tuple[float, str] | None:
    """
    Score a single BP candidate against the normalized query.
    Returns (score, tier) or None if score is too low to include.
    """
    bp = cand.get("BusinessPartner", "")
    name_full = (cand.get("BusinessPartnerFullName") or "").strip()
    name_org = (cand.get("OrganizationBPName1") or "").strip()
    name_bp = (cand.get("BusinessPartnerName") or "").strip()
    display_name = name_full or name_org or name_bp or bp

    bp_norm = _norm(bp)
    name_norm = _norm(display_name)

    # Exact code match
    if query_norm == bp_norm:
        return 1.0, "EXACT_CODE"

    # Exact name match
    if query_norm == name_norm:
        return 1.0, "EXACT_NAME"

    # Partial name match: one contains the other (guard against very short strings)
    if len(query_norm) >= 3 and len(name_norm) >= 3:
        length_ratio = min(len(query_norm), len(name_norm)) / max(len(query_norm), len(name_norm))
        if length_ratio >= 0.4 and (query_norm in name_norm or name_norm in query_norm):
            score = 0.85 + 0.10 * length_ratio
            return round(score, 4), "PARTIAL_NAME"

    # Fuzzy match
    score = _fuzzy_name_score(query_norm, name_norm)
    if score >= 0.50:
        return round(score, 4), "FUZZY"

    return None


def _score_results(raw_list: list[dict], query_norm: str) -> list[dict]:
    results = []
    for item in raw_list:
        bp = item.get("BusinessPartner", "")
        name_full = (item.get("BusinessPartnerFullName") or "").strip()
        name_org = (item.get("OrganizationBPName1") or "").strip()
        name_bp = (item.get("BusinessPartnerName") or "").strip()
        display_name = name_full or name_org or name_bp or bp

        scored = _score_candidate(query_norm, item)
        if scored is None:
            continue
        score, tier = scored

        results.append({
            "business_partner": bp,
            "customer_name": display_name,
            "score": score,
            "confidence": tier,
        })

    # Sort: tier order first, then score DESC within tier
    results.sort(key=lambda x: (_TIER_ORDER.get(x["confidence"], 99), -x["score"]))
    return results


def search_customer_odata(
    q: str,
    top: int = 10,
    request: Optional[Request] = None,
) -> list[dict]:
    """
    Search Business Partners in S/4HANA by name.

    Uses OData V2 substringof() (on-premise compatible).
    Falls back to client-side filtering if the filter is rejected.
    All results are scored with rapidfuzz for consistent ranking.

    Returns a list of dicts: business_partner, customer_name, score, confidence
    """
    config = get_sap_config(request)
    base_url = config.base_url or settings.S4_BASE_URL.rstrip("/")

    if not base_url:
        raise RuntimeError("S4_BASE_URL is not configured")

    if not config.verify:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = config.build_session()
    endpoint = f"{base_url}/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"
    client = config.client or settings.S4_CLIENT
    q_escaped = q.replace("'", "''")
    query_norm = _norm(q)

    select = "BusinessPartner,OrganizationBPName1,BusinessPartnerFullName,BusinessPartnerName"

    # ── Strategy 1: substringof on BusinessPartnerName (V2 standard) ──────────
    try:
        params = {
            "sap-client": client,
            "$format": "json",
            "$top": str(top),
            "$filter": f"substringof('{q_escaped}',BusinessPartnerName)",
            "$select": select,
        }
        logger.info("BP search strategy=substringof(BusinessPartnerName) | q=%r", q)
        raw = _fetch_bp(session, endpoint, params)
        if raw:
            results = _score_results(raw, query_norm)
            logger.info("BP search found=%d via substringof(BusinessPartnerName)", len(results))
            return results[:top]
        logger.info("substringof(BusinessPartnerName) returned 0 results, trying OrganizationBPName1")
    except requests.exceptions.HTTPError as exc:
        logger.warning(
            "substringof(BusinessPartnerName) failed HTTP %s, trying OrganizationBPName1",
            exc.response.status_code if exc.response else "?",
        )

    # ── Strategy 2: substringof on OrganizationBPName1 ────────────────────────
    try:
        params = {
            "sap-client": client,
            "$format": "json",
            "$top": str(top),
            "$filter": f"substringof('{q_escaped}',OrganizationBPName1)",
            "$select": select,
        }
        logger.info("BP search strategy=substringof(OrganizationBPName1) | q=%r", q)
        raw = _fetch_bp(session, endpoint, params)
        if raw:
            results = _score_results(raw, query_norm)
            logger.info("BP search found=%d via substringof(OrganizationBPName1)", len(results))
            return results[:top]
        logger.info("substringof(OrganizationBPName1) returned 0 results, falling back to full scan")
    except requests.exceptions.HTTPError as exc:
        logger.warning(
            "substringof(OrganizationBPName1) failed HTTP %s, falling back to full scan",
            exc.response.status_code if exc.response else "?",
        )

    # ── Strategy 3: fetch up to _FALLBACK_TOP and match client-side ───────────
    logger.info("BP search strategy=full scan top=%d | q=%r", _FALLBACK_TOP, q)
    try:
        params = {
            "sap-client": client,
            "$format": "json",
            "$top": str(_FALLBACK_TOP),
            "$select": select,
        }
        raw = _fetch_bp(session, endpoint, params)
        results = _score_results(raw, query_norm)
        logger.info("BP search found=%d via full scan", len(results))
        return results[:top]
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 0
        raise RuntimeError(f"S/4HANA returned HTTP {status_code} for customer search") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("S/4HANA customer search timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach S/4HANA at {base_url}") from exc
