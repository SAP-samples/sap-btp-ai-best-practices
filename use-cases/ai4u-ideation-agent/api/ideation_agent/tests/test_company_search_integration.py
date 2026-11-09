"""Integration tests for CSE — hits the real API.

Run explicitly with:
    pytest -m integration ideation_agent/tests/test_company_search_integration.py -v

Requires CSE_URL and CSE_KEY environment variables (loaded from .env).
Skipped automatically when credentials are missing.
"""
from __future__ import annotations

import os
import time

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except Exception:
    pass

from ideation_agent.tools.company_search import (
    cse_enabled,
    cse_find_company,
    cse_report_analysis,
    cse_customer_stories,
    run_cse_enrichment,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not cse_enabled(), reason="CSE not configured (missing CSE_URL/CSE_KEY)"),
]


# ---------------------------------------------------------------------------
# Company resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_volvo():
    result = await cse_find_company("volvo")
    assert result["status"] == "ok", f"Failed: {result.get('message')}"
    assert "volvo" in result["company"]["name"].lower()


@pytest.mark.asyncio
async def test_find_sap():
    result = await cse_find_company("SAP")
    assert result["status"] == "ok", f"Failed: {result.get('message')}"
    assert "sap" in result["company"]["name"].lower()


@pytest.mark.asyncio
async def test_find_bmw():
    """BMW may not be in CSE database — verify graceful handling."""
    result = await cse_find_company("BMW")
    assert result["status"] in ("ok", "not_found")
    if result["status"] == "ok":
        assert "bmw" in result["company"]["name"].lower()


@pytest.mark.asyncio
async def test_find_siemens():
    """Siemens may not be in CSE database — verify graceful handling."""
    result = await cse_find_company("siemens")
    assert result["status"] in ("ok", "not_found")
    if result["status"] == "ok":
        assert "siemens" in result["company"]["name"].lower()


@pytest.mark.asyncio
async def test_find_deutsche_telekom():
    """Deutsche Telekom may not be in CSE database — verify graceful handling."""
    result = await cse_find_company("Deutsche Telekom")
    assert result["status"] in ("ok", "not_found")
    if result["status"] == "ok":
        name_lower = result["company"]["name"].lower()
        assert "telekom" in name_lower or "deutsche" in name_lower


@pytest.mark.asyncio
async def test_find_nonexistent():
    result = await cse_find_company("xyznonexist99999")
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_no_false_positives():
    """Ensure unrelated companies don't match common queries."""
    result = await cse_find_company("BMW")
    if result["status"] == "ok":
        # If something matched, it must actually contain 'bmw'
        assert "bmw" in result["company"]["name"].lower()
    # If not_found, that's correct — the CSE DB shouldn't return false matches


# ---------------------------------------------------------------------------
# Report analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_analysis_volvo():
    result = await cse_report_analysis("Volvo Car AB")
    assert result["status"] == "ok", f"Failed: {result.get('message')}"
    assert len(result["analysis_markdown"]) > 100
    assert "##" in result["analysis_markdown"]


# ---------------------------------------------------------------------------
# Customer stories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_customer_stories_volvo():
    result = await cse_customer_stories("Volvo Car AB")
    assert result["status"] == "ok", f"Failed: {result.get('message')}"
    assert len(result["stories_markdown"]) > 50


# ---------------------------------------------------------------------------
# Full enrichment pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_enrichment_volvo():
    start = time.time()
    result = await run_cse_enrichment("volvo")
    elapsed = time.time() - start

    assert result["status"] == "ok", f"Failed: {result.get('error', result.get('message'))}"
    assert result["company"]["name"]
    assert len(result.get("analysis_markdown", "")) > 100
    assert len(result.get("stories_markdown", "")) > 50
    assert result.get("report_link")
    # Should complete within reasonable time (parallel calls)
    assert elapsed < 30, f"Enrichment took too long: {elapsed:.1f}s"
    print(f"\n  Enrichment for 'volvo': {elapsed:.1f}s, company={result['company']['name']}")


@pytest.mark.asyncio
async def test_full_enrichment_sap():
    result = await run_cse_enrichment("SAP")
    assert result["status"] == "ok", f"Failed: {result.get('error', result.get('message'))}"
    assert "sap" in result["company"]["name"].lower()
    assert result.get("analysis_markdown")
    assert result.get("report_link")


@pytest.mark.asyncio
async def test_full_enrichment_siemens():
    """Siemens may not resolve via find-company but downstream endpoints resolve it."""
    result = await run_cse_enrichment("siemens")
    assert result["status"] == "ok", f"Failed: {result.get('error', result.get('message'))}"
    assert "siemens" in result["company"]["name"].lower()
    assert result.get("analysis_markdown") or result.get("stories_markdown")


@pytest.mark.asyncio
async def test_full_enrichment_bmw():
    """BMW may not resolve via find-company but downstream endpoints resolve it."""
    result = await run_cse_enrichment("BMW")
    assert result["status"] in ("ok", "not_found")
    if result["status"] == "ok":
        assert "bmw" in result["company"]["name"].lower()
        assert result.get("analysis_markdown") or result.get("stories_markdown")
