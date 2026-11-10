"""CSE (Company Search Engine) async client for background company enrichment.

Ported from ideation_deep_agent_extra/tools/company_search.py to async httpx.
This module is NOT a LangChain tool — it runs in the background, not invoked by the LLM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

DEFAULT_NUM_MATCHES = 10
DEFAULT_HTTP_TIMEOUT = 30
DEFAULT_MIN_MATCH_SCORE = 0.45


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _read_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _read_env_int(name: str, default: int) -> int:
    raw = _read_env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_env_float(name: str, default: float) -> float:
    raw = _read_env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def cse_enabled() -> bool:
    """Returns True when both CSE_URL and CSE_KEY are configured."""
    return bool(_read_env("CSE_URL")) and bool(_read_env("CSE_KEY"))


def _cse_base_url() -> str:
    value = _read_env("CSE_URL")
    if not value:
        raise ValueError("Missing required environment variable: CSE_URL")
    return value.rstrip("/")


def _cse_api_key() -> str:
    value = _read_env("CSE_KEY")
    if not value:
        raise ValueError("Missing required environment variable: CSE_KEY")
    return value


def _cse_http_timeout() -> int:
    return _read_env_int("CSE_HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

async def _cse_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticated async GET to the CSE API."""
    base_url = _cse_base_url()
    api_key = _cse_api_key()
    timeout = _cse_http_timeout()

    url = f"{base_url}{endpoint}"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    logger.info("[CSE] GET %s params=%s", endpoint, params)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()

    logger.info("[CSE] response status=%s endpoint=%s", response.status_code, endpoint)
    logger.debug("[CSE] full response: %s", payload)
    return payload


# ---------------------------------------------------------------------------
# Scoring & matching helpers (ported from source — pure functions)
# ---------------------------------------------------------------------------

def _tokenize_name(name: str) -> List[str]:
    text = str(name or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _candidate_match_score(query: str, candidate_name: str) -> float:
    query_tokens = set(_tokenize_name(query))
    candidate_tokens = set(_tokenize_name(candidate_name))
    if not query_tokens or not candidate_tokens:
        return 0.0
    intersection = query_tokens & candidate_tokens
    union = query_tokens | candidate_tokens
    jaccard = len(intersection) / len(union) if union else 0.0
    # For short queries, use containment (query subset of candidate) to avoid
    # penalizing single-word lookups like "volvo" → "Volvo Car Corporation"
    containment = len(intersection) / len(query_tokens)
    return max(jaccard, containment * 0.7)


def _candidate_with_score(query: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(candidate)
    enriched["match_score"] = round(
        _candidate_match_score(query, str(candidate.get("name", ""))), 3
    )
    return enriched


def _sort_cse_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = [_candidate_with_score(query, c) for c in candidates]
    exchange_rank = {"NYSE": 3, "NASDAQ": 2, "OTC": 1}
    enriched.sort(
        key=lambda item: (
            float(item.get("match_score", 0.0) or 0.0),
            exchange_rank.get(str(item.get("exchange", "")).strip().upper(), 0),
            1 if item.get("ticker") else 0,
        ),
        reverse=True,
    )
    return enriched


def _is_cse_name_ambiguous(query: str, candidates: List[Dict[str, Any]]) -> bool:
    if len(candidates) < 2:
        return False
    top_score = float(candidates[0].get("match_score", 0.0) or 0.0)
    second_score = float(candidates[1].get("match_score", 0.0) or 0.0)
    return abs(top_score - second_score) <= 0.08


def _normalize_cse_company_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = payload.get("column_names")
    rows = payload.get("company_info")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return []

    normalized_columns = [str(col or "").strip().lower() for col in columns]
    out: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, list):
            continue
        values: Dict[str, Any] = {}
        for idx, col_name in enumerate(normalized_columns):
            if idx < len(row):
                values[col_name] = row[idx]

        name = str(values.get("name", "") or "").strip()
        if not name:
            continue

        ticker_raw = values.get("ticker", "")
        ticker = str(ticker_raw or "").strip() or None
        exchange_raw = values.get("exchange", "")
        exchange = str(exchange_raw or "").strip() or None

        out.append({"name": name, "ticker": ticker, "exchange": exchange})

    return out


# ---------------------------------------------------------------------------
# Markdown formatters (ported from source)
# ---------------------------------------------------------------------------

def _format_cse_report_analysis_markdown(agent_dict: Mapping[str, Any]) -> str:
    if not isinstance(agent_dict, Mapping):
        return ""

    sections: List[str] = []
    for _key, section_value in agent_dict.items():
        if not isinstance(section_value, Mapping):
            continue
        section_name = str(section_value.get("name", "") or "").strip()
        section_summary = str(section_value.get("summary", "") or "").strip()
        evidence = section_value.get("evidence", [])

        if not section_name and not section_summary:
            continue

        lines: List[str] = []
        if section_name:
            lines.append(f"## {section_name}")
            lines.append("")
        if section_summary:
            lines.append(section_summary)
            lines.append("")
        if isinstance(evidence, list) and evidence:
            lines.append("### Supporting Evidence")
            lines.append("")
            for item in evidence[:8]:
                if not isinstance(item, Mapping):
                    continue
                item_name = str(item.get("name", "") or "").strip()
                item_content = str(item.get("content", "") or "").strip()
                item_citation = str(item.get("citation", "") or "").strip()
                bullet = item_content
                if item_name:
                    bullet = f"**{item_name}**: {bullet}" if bullet else f"**{item_name}**"
                if item_citation:
                    bullet = f"{bullet} ({item_citation})" if bullet else item_citation
                if bullet:
                    lines.append(f"- {bullet}")
            lines.append("")

        section_md = "\n".join(lines).strip()
        if section_md:
            sections.append(section_md)

    return "\n\n".join(sections).strip()


def _format_cse_customer_stories_markdown(companies: Sequence[Any]) -> str:
    sections: List[str] = []
    for story in companies[:6]:
        if not isinstance(story, Mapping):
            continue
        company_name = str(story.get("company_name", "") or "").strip()
        connection = str(story.get("connection", "") or "").strip()
        solutions = story.get("sap_solutions_used", [])
        challenges = story.get("similar_challenges", [])
        outcomes = story.get("benefits_and_outcomes", [])
        source_path = str(story.get("source_path", "") or "").strip()

        lines: List[str] = []
        if company_name:
            lines.append(f"## {company_name}")
            lines.append("")
        if connection:
            lines.append(connection)
            lines.append("")
        if isinstance(solutions, list) and solutions:
            lines.append(f"- SAP solutions used: {', '.join(str(s).strip() for s in solutions if str(s).strip())}")
        if isinstance(challenges, list) and challenges:
            lines.append(f"- Similar challenges: {'; '.join(str(c).strip() for c in challenges if str(c).strip())}")
        if isinstance(outcomes, list) and outcomes:
            lines.append(f"- Benefits and outcomes: {'; '.join(str(o).strip() for o in outcomes if str(o).strip())}")
        if source_path:
            lines.append(f"- Story link: {source_path}")

        story_md = "\n".join(lines).strip()
        if story_md:
            sections.append(story_md)

    return "\n\n".join(sections).strip()


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

async def cse_find_company(
    company_name: str,
    num_matches: int = DEFAULT_NUM_MATCHES,
) -> Dict[str, Any]:
    """Search for company identity via CSE API with Jaccard scoring."""
    company_input = str(company_name or "").strip()
    if not company_input:
        return {"status": "not_found", "message": "Empty company name.", "company": None}

    num_matches_clamped = max(1, min(int(num_matches), 10))

    payload = await _cse_get(
        "/v3/cse/find-company",
        {"company_name": company_input, "num_matches": num_matches_clamped},
    )

    candidates = _normalize_cse_company_candidates(payload)
    if not candidates:
        return {"status": "not_found", "message": f"No companies found for '{company_input}'.", "company": None}

    # Filter out non-company entries (reports, uploads, etc.)
    company_candidates = [
        c for c in candidates
        if not any(kw in str(c.get("name", "")).lower() for kw in ("report", "upload", "annual"))
    ]
    if not company_candidates:
        company_candidates = candidates

    sorted_candidates = _sort_cse_candidates(company_input, company_candidates)
    min_score = _read_env_float("CSE_FIND_COMPANY_MIN_MATCH_SCORE", DEFAULT_MIN_MATCH_SCORE)
    confident = [c for c in sorted_candidates if float(c.get("match_score", 0.0) or 0.0) >= min_score]

    if not confident:
        return {
            "status": "not_found",
            "message": f"No confident match for '{company_input}'.",
            "company": None,
            "candidates": sorted_candidates[:num_matches_clamped],
        }

    # For background enrichment: always select top candidate (no disambiguation)
    top = confident[0]
    is_ambiguous = _is_cse_name_ambiguous(company_input, confident)
    if is_ambiguous:
        logger.info("[CSE] Ambiguous result for '%s' — auto-selecting top candidate: %s", company_input, top.get("name"))

    return {
        "status": "ok",
        "message": f"Found company: {top.get('name')}",
        "company": top,
    }


async def cse_report_analysis(company_name: str) -> Dict[str, Any]:
    """Fetch structured EDGAR report analysis for a company."""
    company_input = str(company_name or "").strip()
    if not company_input:
        return {"status": "error", "message": "Empty company name.", "analysis_markdown": ""}

    payload = await _cse_get("/v3/cse/get-report-analysis", {"company_name": company_input})
    agent_dict = payload.get("agent_dict", {})
    analysis_md = _format_cse_report_analysis_markdown(agent_dict if isinstance(agent_dict, Mapping) else {})

    if not analysis_md:
        return {"status": "error", "message": f"Empty report analysis for '{company_input}'.", "analysis_markdown": ""}

    return {
        "status": "ok",
        "message": f"Report analysis for '{company_input}'.",
        "analysis_markdown": analysis_md,
    }


async def cse_customer_stories(company_name: str) -> Dict[str, Any]:
    """Fetch related SAP customer stories for a company."""
    company_input = str(company_name or "").strip()
    if not company_input:
        return {"status": "error", "message": "Empty company name.", "stories_markdown": "", "story_links": []}

    payload = await _cse_get("/v3/cse/get-report-customer-stories", {"company_name": company_input})
    cve_agent_dict = payload.get("cve_agent_dict", {})
    companies = cve_agent_dict.get("companies", []) if isinstance(cve_agent_dict, Mapping) else []
    stories_md = _format_cse_customer_stories_markdown(companies if isinstance(companies, list) else [])
    story_links = [
        str(item.get("source_path", "")).strip()
        for item in (companies if isinstance(companies, list) else [])
        if isinstance(item, Mapping) and str(item.get("source_path", "")).strip()
    ]

    if not stories_md:
        return {"status": "error", "message": f"No customer stories for '{company_input}'.", "stories_markdown": "", "story_links": []}

    return {
        "status": "ok",
        "message": f"Customer stories for '{company_input}'.",
        "stories_markdown": stories_md,
        "story_links": story_links,
    }


async def cse_report_link(company_name: str) -> Dict[str, Any]:
    """Fetch the EDGAR report source URL for a company."""
    company_input = str(company_name or "").strip()
    if not company_input:
        return {"status": "error", "message": "Empty company name.", "report_link": ""}

    payload = await _cse_get("/v3/cse/get-report-link", {"company_name": company_input})
    link = str(payload.get("link", "") or "").strip()

    if not link:
        return {"status": "error", "message": f"No report link for '{company_input}'.", "report_link": ""}

    return {"status": "ok", "message": f"Report link for '{company_input}'.", "report_link": link}


async def cse_suggest_companies(query: str, num_matches: int = DEFAULT_NUM_MATCHES) -> List[Dict[str, Any]]:
    """Return scored company candidates for autocomplete (no confidence threshold)."""
    query_input = str(query or "").strip()
    if not query_input or len(query_input) < 2:
        return []

    num_matches_clamped = max(1, min(int(num_matches), 10))

    try:
        payload = await _cse_get(
            "/v3/cse/find-company",
            {"company_name": query_input, "num_matches": num_matches_clamped},
        )
    except Exception as exc:
        logger.warning("[CSE] suggest_companies failed: %s", exc)
        return []

    candidates = _normalize_cse_company_candidates(payload)
    if not candidates:
        return []

    company_candidates = [
        c for c in candidates
        if not any(kw in str(c.get("name", "")).lower() for kw in ("report", "upload", "annual"))
    ]
    if not company_candidates:
        company_candidates = candidates

    sorted_candidates = _sort_cse_candidates(query_input, company_candidates)
    return sorted_candidates[:num_matches_clamped]


async def run_cse_enrichment(company_name: str) -> Dict[str, Any]:
    """Orchestrate full CSE enrichment: find company → parallel (analysis + stories + link).

    The downstream endpoints (report-analysis, customer-stories, report-link) resolve
    company names internally, so we proceed even if find-company fails — using the raw
    user input as the lookup name.

    Returns a unified enrichment dict. Never raises — catches all exceptions.
    """
    try:
        find_result = await cse_find_company(company_name)

        company: Optional[Dict[str, Any]] = None
        if find_result.get("status") == "ok":
            company = find_result["company"]
            resolved_name = company.get("name", company_name)
        else:
            logger.info(
                "[CSE] find-company returned %s for '%s' — proceeding with raw name",
                find_result.get("status"), company_name,
            )
            resolved_name = company_name

        analysis_result, stories_result, link_result = await asyncio.gather(
            cse_report_analysis(resolved_name),
            cse_customer_stories(resolved_name),
            cse_report_link(resolved_name),
            return_exceptions=True,
        )

        analysis_md = ""
        if isinstance(analysis_result, dict) and analysis_result.get("status") == "ok":
            analysis_md = analysis_result.get("analysis_markdown", "")
        elif isinstance(analysis_result, Exception):
            logger.warning("[CSE] report_analysis failed: %s", analysis_result)

        stories_md = ""
        story_links: List[str] = []
        if isinstance(stories_result, dict) and stories_result.get("status") == "ok":
            stories_md = stories_result.get("stories_markdown", "")
            story_links = stories_result.get("story_links", [])
        elif isinstance(stories_result, Exception):
            logger.warning("[CSE] customer_stories failed: %s", stories_result)

        report_link = ""
        if isinstance(link_result, dict) and link_result.get("status") == "ok":
            report_link = link_result.get("report_link", "")
        elif isinstance(link_result, Exception):
            logger.warning("[CSE] report_link failed: %s", link_result)

        has_any_data = bool(analysis_md or stories_md or report_link)

        if not has_any_data and company is None:
            return {
                "status": "not_found",
                "company": None,
                "analysis_markdown": "",
                "stories_markdown": "",
                "story_links": [],
                "report_link": "",
                "message": f"No data found for '{company_name}'.",
            }

        return {
            "status": "ok",
            "company": company or {"name": resolved_name, "ticker": None, "exchange": None},
            "analysis_markdown": analysis_md,
            "stories_markdown": stories_md,
            "story_links": story_links,
            "report_link": report_link,
        }

    except Exception as exc:
        logger.error("[CSE] enrichment pipeline failed: %s", exc)
        return {
            "status": "failed",
            "company": None,
            "analysis_markdown": "",
            "stories_markdown": "",
            "story_links": [],
            "report_link": "",
            "error": str(exc),
        }
