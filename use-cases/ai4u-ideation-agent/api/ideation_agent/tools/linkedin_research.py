"""LinkedIn position analysis for background enrichment.

Self-contained module: scrapes LinkedIn public guest APIs for job listings,
then uses an LLM to analyze technology signals, AI maturity, business roles,
and operational bottlenecks. No external dependencies on other agent implementations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote as url_encode

import requests

from app.observability.llm_usage_logging import (
    LLMUsageMeta,
    emit_llm_usage_event,
    emit_llm_usage_for_response,
)

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # pragma: no cover
    BeautifulSoup = None

logger = logging.getLogger(__name__)


def linkedin_enabled() -> bool:
    """Returns True when LinkedIn research is explicitly enabled via env var."""
    return os.getenv("IDEATION_LINKEDIN_SKILL_ENABLED", "").strip().lower() in ("true", "1", "yes")

_MAX_JOBS = 200
_LOCATION = "Worldwide"
_TIME_RANGE = "r2592000"  # 30 days
_SLEEP_MS = 500
_DISTANCE = 1000
_RETRY_DELAYS = (2.0, 5.0)  # exponential backoff for 429s

INIT_URL = "https://www.linkedin.com/jobs/search"
PAGE_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
POST_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Scraping helpers (unchanged)
# ---------------------------------------------------------------------------


def _require_bs4():
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is required. Install via: pip install beautifulsoup4")
    return BeautifulSoup


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_relative_days(value: str) -> Optional[float]:
    text = _clean_text(value).lower()
    if not text:
        return None
    if "just now" in text or "today" in text:
        return 0.0
    for pattern, factor in ((r"(\d+)\s*hour", 1.0 / 24.0), (r"(\d+)\s*day", 1.0), (r"(\d+)\s*week", 7.0), (r"(\d+)\s*month", 30.0)):
        m = re.search(pattern, text)
        if m:
            return float(m.group(1)) * factor
    return None


def _params(*, keywords: str, time_range: str, location: str, distance: int, start: int) -> str:
    return (
        f"?keywords={url_encode(keywords)}"
        f"&f_TPR={time_range}"
        f"&location={url_encode(location)}"
        f"&distance={distance}"
        f"&position=1&pageNum=0&start={start}&sortBy=DD"
    )


def _safe_get(url: str, timeout_sec: int = 30) -> requests.Response:
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
        if delay:
            logger.debug("[LinkedIn] 429 on attempt %d, backing off %.1fs", attempt, delay)
            time.sleep(delay)
        response = requests.get(url, headers=HEADERS, timeout=timeout_sec)
        if response.status_code == 429:
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()
    return response


def _parse_job_count(html: str) -> int:
    soup = _require_bs4()(html, "html.parser")
    node = soup.find("span", {"class": "results-context-header__job-count"})
    if node and node.text:
        digits = re.sub(r"[^\d]", "", node.text)
        if digits:
            return int(digits)
    return 0


def _job_ids_for_page(html: str) -> List[str]:
    soup = _require_bs4()(html, "html.parser")
    ids: List[str] = []
    for li in soup.find_all("li"):
        card = li.find("div", {"class": "base-card"}) or li.find("a", {"class": "base-card"})
        if not card:
            continue
        urn = card.get("data-entity-urn") or ""
        parts = str(urn).split(":")
        if len(parts) >= 4 and parts[3]:
            ids.append(parts[3])
    return ids


def _job_detail(job_id: str) -> Dict[str, Any]:
    uri = POST_URL + str(job_id)
    response = _safe_get(uri)
    soup = _require_bs4()(response.text, "html.parser")

    detail: Dict[str, Any] = {
        "title": "", "company": "", "location": "", "id": str(job_id),
        "link": uri, "description": "", "time": "", "level": "",
        "industry": "", "type": "", "function": "",
    }

    anchor = soup.select_one("div.top-card-layout__entity-info a")
    if anchor:
        detail["link"] = anchor.get("href") or uri
        detail["title"] = _clean_text(anchor.text)

    company = soup.select_one("a.topcard__org-name-link")
    if company and company.text:
        detail["company"] = _clean_text(company.text)
    else:
        fallback = soup.select_one(".topcard__flavor-row > span.topcard__flavor")
        if fallback and fallback.text:
            detail["company"] = _clean_text(fallback.text)

    loc = soup.select_one(
        "h4.top-card-layout__second-subline > div.topcard__flavor-row > span.topcard__flavor.topcard__flavor--bullet"
    )
    if loc and loc.text:
        detail["location"] = _clean_text(loc.text)

    posted = soup.select_one(".posted-time-ago__text")
    if posted and posted.text:
        detail["time"] = _clean_text(posted.text)

    desc = soup.select_one(".show-more-less-html__markup.show-more-less-html__markup--clamp-after-5")
    if desc and desc.text:
        detail["description"] = _clean_text(desc.text)

    keys = {"Seniority level": "level", "Job function": "function", "Employment type": "type", "Industries": "industry"}
    for item in soup.select("li.description__job-criteria-item"):
        key_node = item.select_one(".description__job-criteria-subheader")
        val_node = item.select_one(".description__job-criteria-text")
        mapped = keys.get(_clean_text(key_node.text if key_node else ""))
        if mapped:
            detail[mapped] = _clean_text(val_node.text if val_node else "")

    return detail


def _generate_keyword_variations(company_name: str) -> List[str]:
    """Generate keyword variations to bypass LinkedIn's per-query pagination cap."""
    base = company_name.strip()
    variations = [base]
    if "group" not in base.lower():
        variations.append(f"{base} Group")
    variations.append(f"{base} jobs")
    return variations[:3]


def _fetch_job_ids_for_keyword(keyword: str) -> List[str]:
    """Fetch job IDs for a single keyword (pagination loop). Thread-safe."""
    try:
        init_uri = INIT_URL + _params(keywords=keyword, time_range=_TIME_RANGE, location=_LOCATION, distance=_DISTANCE, start=0)
        init_response = _safe_get(init_uri)
        total_available = _parse_job_count(init_response.text)
        target = min(_MAX_JOBS, total_available) if total_available else _MAX_JOBS
        target = min(target, 1000)
        pages = max(1, math.ceil(target / 25))
    except Exception:
        return []

    job_ids: List[str] = []
    for page_idx in range(pages):
        start = page_idx * 25
        page_uri = PAGE_URL + _params(keywords=keyword, time_range=_TIME_RANGE, location=_LOCATION, distance=_DISTANCE, start=start)
        response = None
        for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                response = requests.get(page_uri, headers=HEADERS, timeout=30)
            except Exception:
                break
            if response.status_code == 429:
                continue
            break
        if response is None or response.status_code == 429:
            continue
        if response.status_code == 400:
            break
        if not response.ok:
            continue
        page_ids = _job_ids_for_page(response.text)
        if not page_ids:
            continue
        job_ids.extend(page_ids)
        if len(job_ids) >= target:
            job_ids = job_ids[:target]
            break
        if _SLEEP_MS > 0:
            time.sleep(_SLEEP_MS / 1000.0)

    return job_ids


def _fetch_details_batch(job_ids: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
    """Fetch job details in parallel. Returns successfully fetched rows."""
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_job_detail, jid): jid for jid in job_ids}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                continue
    return rows


def _norm_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {k: _clean_text(row.get(k)) for k in ("title", "company", "location", "id", "link", "description", "time", "level", "industry", "type", "function")}


def _infer_time_window(rows: Sequence[Dict[str, str]]) -> str:
    ages = [_parse_relative_days(r.get("time", "")) for r in rows]
    clean = [a for a in ages if a is not None]
    if not clean:
        return "unspecified"
    return f"{int(min(clean))}-{int(max(clean))} days ago"


# ---------------------------------------------------------------------------
# LLM-based analysis
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert at analyzing job postings to extract hiring intelligence "
    "for enterprise AI strategy recommendations. Return ONLY valid JSON."
)

_ANALYSIS_USER_TEMPLATE = """Company: {company_name}
Total open positions analyzed: {total_positions}

{descriptions_text}

Return JSON with this exact structure:
{{
  "top_technologies": [
    {{"name": "<technology/platform>", "count": <actual mentions across all postings>, "insight": "<how they use it>"}}
  ],
  "ai_signals": {{
    "ai_signal_found": <true/false>,
    "ai_maturity_level": "<none|emerging|scaling|advanced>",
    "ai_maturity_reasoning": "<2-3 sentences with evidence from the postings>"
  }},
  "top_business_vacancies": [
    {{"role_family": "<business area>", "count": <number of roles>, "insight": "<what this reveals>"}}
  ],
  "ideation_insights": {{
    "bottlenecks": ["<specific operational bottleneck visible in these postings>"],
    "automation_opportunities": ["<specific, actionable opportunity derived from actual job description pain points>"]
  }}
}}

Rules:
- These results come from a LinkedIn search for "{company_name}". Some postings may be from subsidiaries, partners, resellers, or third parties rather than the company directly. Weight direct corporate roles more heavily for strategic insights, but note ecosystem patterns where they reveal broader challenges.
- top_technologies: max 5, ordered by prevalence. Include platforms, languages, cloud, enterprise systems.
- ai_maturity_level: "none" if no AI/ML mentions, "emerging" if few roles mention AI, "scaling" if moderate, "advanced" if pervasive.
- top_business_vacancies: max 5 non-technical roles grouped by business function. Note if roles are from the company directly vs its ecosystem.
- bottlenecks: max 4. Must be SPECIFIC to {company_name} based on evidence in these postings.
- automation_opportunities: max 4. Must be SPECIFIC and ACTIONABLE, derived from actual pain points visible in descriptions. Never use generic suggestions.
"""


def _build_descriptions_text(rows: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for r in rows:
        parts.append(
            f"Title: {r.get('title', '')}\n"
            f"Function: {r.get('function', '')}\n"
            f"Level: {r.get('level', '')}\n"
            f"Description: {r.get('description', '')[:600]}"
        )
    return "\n---\n".join(parts)


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


async def _analyze_with_llm(
    rows: List[Dict[str, str]],
    company_name: str,
    *,
    context_id: str = "",
) -> Dict[str, Any]:
    """Analyze all job descriptions using gemini-2.5-pro."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from ideation_agent.common import make_llm

    model_name = "gemini-2.5-pro"
    descriptions_text = _build_descriptions_text(rows)

    llm = make_llm(provider="google", model_name=model_name, temperature=0.1, json_mode=True)

    user_prompt = _ANALYSIS_USER_TEMPLATE.format(
        company_name=company_name,
        total_positions=len(rows),
        descriptions_text=descriptions_text,
    )

    meta = LLMUsageMeta(
        route="/background/ideation-agent/linkedin",
        method="BACKGROUND",
        correlation_id=context_id or None,
    )
    started = time.perf_counter()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception:
        emit_llm_usage_event(
            route=meta.route,
            method=meta.method,
            client_host=meta.client_host,
            provider="sap-ai-core",
            model=model_name,
            llm_endpoint="chat.completions",
            input_tokens=0,
            output_tokens=0,
            outcome="error",
            latency_ms=int((time.perf_counter() - started) * 1000),
            correlation_id=meta.correlation_id,
            step_name="ideation_agent.linkedin_analysis",
            feature="linkedin",
        )
        raise
    emit_llm_usage_for_response(
        response,
        meta=meta,
        provider="sap-ai-core",
        model=model_name,
        llm_endpoint="chat.completions",
        latency_ms=int((time.perf_counter() - started) * 1000),
        step_name="ideation_agent.linkedin_analysis",
        feature="linkedin",
    )

    raw = response.content if isinstance(response.content, str) else str(response.content or "")
    return _parse_llm_json(raw)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _scrape_jobs(company_name: str) -> tuple:
    """Scrape LinkedIn with parallel keyword searches and parallel detail fetching."""
    _require_bs4()

    keywords = _generate_keyword_variations(company_name)
    all_job_ids: List[str] = []

    for kw in keywords:
        all_job_ids.extend(_fetch_job_ids_for_keyword(kw))

    seen: set = set()
    unique_ids: List[str] = []
    for jid in all_job_ids:
        if jid not in seen:
            seen.add(jid)
            unique_ids.append(jid)

    if not unique_ids:
        return [], 0

    rows = _fetch_details_batch(unique_ids, max_workers=3)

    if not rows:
        return [], 0

    normalized = [_norm_row(r) for r in rows]
    return normalized, len(unique_ids)


async def run_linkedin_research(company_name: str, *, context_id: str = "") -> Dict[str, Any]:
    """Async entry point. Fires on turn 1 alongside CSE and Perplexity."""
    normalized = company_name.strip()
    if not normalized:
        return {"status": "error", "message": "company_name is required"}

    logger.info("[LinkedIn] Starting position analysis company='%s' max_jobs=%d", normalized, _MAX_JOBS)

    try:
        # Phase 1: Scrape (synchronous, in thread)
        filtered, total_hint = await asyncio.to_thread(_scrape_jobs, normalized)

        if not filtered:
            return {"status": "no_data", "message": f"No LinkedIn postings found for '{normalized}'."}

        # Phase 2: LLM analysis (async)
        logger.info("[LinkedIn] Analyzing %d positions with LLM company='%s'", len(filtered), normalized)
        analysis = await _analyze_with_llm(filtered, normalized, context_id=context_id)

        # Collect metadata
        links = []
        for row in filtered:
            link = _clean_text(row.get("link", ""))
            if link and link not in links:
                links.append(link)

        result = {
            "status": "ok",
            "top_technologies": analysis.get("top_technologies", []),
            "top_business_vacancies": analysis.get("top_business_vacancies", []),
            "ai_signals": analysis.get("ai_signals", {}),
            "ideation_insights": analysis.get("ideation_insights", {}),
            "confidence": round(min(0.35 + len(filtered) / 400.0 + 0.2, 0.95), 2),
            "rows_analyzed": len(filtered),
            "time_window": _infer_time_window(filtered),
            "source_links": links,
        }

        logger.info(
            "[LinkedIn] Analysis complete company='%s' rows=%d",
            normalized, len(filtered),
        )
        return result

    except Exception as exc:
        logger.exception("[LinkedIn] Analysis failed company='%s'", normalized)
        return {"status": "error", "message": str(exc)}
