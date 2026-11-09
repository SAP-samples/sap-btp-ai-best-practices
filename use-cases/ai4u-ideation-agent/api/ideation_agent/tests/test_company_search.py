"""Tests for CSE (Company Search Engine) background enrichment."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

# Patch env vars before importing
os.environ.setdefault("CSE_URL", "https://cse.example.com")
os.environ.setdefault("CSE_KEY", "test-api-key")

from ideation_agent.tools.company_search import (
    _candidate_match_score,
    _format_cse_customer_stories_markdown,
    _format_cse_report_analysis_markdown,
    _is_cse_name_ambiguous,
    _normalize_cse_company_candidates,
    _sort_cse_candidates,
    _tokenize_name,
    cse_enabled,
    cse_find_company,
    cse_report_analysis,
    cse_customer_stories,
    run_cse_enrichment,
)
from ideation_agent import cse_background


# ---------------------------------------------------------------------------
# Unit tests: scoring helpers
# ---------------------------------------------------------------------------


def test_tokenize_name():
    assert _tokenize_name("SAP SE") == ["sap", "se"]
    assert _tokenize_name("Deutsche Telekom AG") == ["deutsche", "telekom", "ag"]
    assert _tokenize_name("") == []


def test_candidate_match_score_exact():
    score = _candidate_match_score("SAP", "SAP SE")
    assert score > 0.4


def test_candidate_match_score_no_overlap():
    score = _candidate_match_score("BMW", "Apple Inc")
    assert score == 0.0


def test_sort_cse_candidates():
    candidates = [
        {"name": "SAP America Inc", "ticker": None, "exchange": None},
        {"name": "SAP SE", "ticker": "SAP", "exchange": "NYSE"},
        {"name": "SAP India", "ticker": "SAP", "exchange": "OTC"},
    ]
    sorted_candidates = _sort_cse_candidates("SAP", candidates)
    assert sorted_candidates[0]["name"] == "SAP SE"


def test_is_cse_name_ambiguous_close_scores():
    candidates = [
        {"name": "BMW AG", "match_score": 0.67},
        {"name": "BMW Group", "match_score": 0.65},
    ]
    assert _is_cse_name_ambiguous("BMW", candidates) is True


def test_is_cse_name_ambiguous_clear_winner():
    candidates = [
        {"name": "SAP SE", "match_score": 0.85},
        {"name": "SAP America", "match_score": 0.45},
    ]
    assert _is_cse_name_ambiguous("SAP", candidates) is False


# ---------------------------------------------------------------------------
# Unit tests: response parsing
# ---------------------------------------------------------------------------


def test_normalize_cse_company_candidates():
    payload = {
        "column_names": ["name", "ticker", "exchange"],
        "company_info": [
            ["SAP SE", "SAP", "NYSE"],
            ["SAP America Inc", "", ""],
        ],
    }
    candidates = _normalize_cse_company_candidates(payload)
    assert len(candidates) == 2
    assert candidates[0]["name"] == "SAP SE"
    assert candidates[0]["ticker"] == "SAP"
    assert candidates[1]["ticker"] is None


def test_normalize_cse_company_candidates_invalid():
    assert _normalize_cse_company_candidates({}) == []
    assert _normalize_cse_company_candidates({"column_names": [], "company_info": "bad"}) == []


# ---------------------------------------------------------------------------
# Unit tests: markdown formatters
# ---------------------------------------------------------------------------


def test_format_cse_report_analysis_markdown():
    agent_dict = {
        "section1": {
            "name": "Company Overview",
            "summary": "SAP is a global enterprise software company.",
            "evidence": [
                {"name": "Revenue", "content": "$30B annual revenue", "citation": "10-K 2024"},
            ],
        }
    }
    md = _format_cse_report_analysis_markdown(agent_dict)
    assert "## Company Overview" in md
    assert "SAP is a global enterprise software company." in md
    assert "**Revenue**" in md
    assert "10-K 2024" in md


def test_format_cse_customer_stories_markdown():
    companies = [
        {
            "company_name": "Acme Corp",
            "connection": "Similar industry and scale",
            "sap_solutions_used": ["SAP S/4HANA", "SAP BTP"],
            "similar_challenges": ["Legacy modernization"],
            "benefits_and_outcomes": ["30% cost reduction"],
            "source_path": "https://example.com/story",
        }
    ]
    md = _format_cse_customer_stories_markdown(companies)
    assert "## Acme Corp" in md
    assert "SAP S/4HANA" in md
    assert "30% cost reduction" in md


# ---------------------------------------------------------------------------
# Async tests: CSE API functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cse_find_company_single_confident_match():
    mock_response = {
        "column_names": ["name", "ticker", "exchange"],
        "company_info": [["BMW AG", "BMW", "NYSE"]],
    }
    with patch(
        "ideation_agent.tools.company_search._cse_get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await cse_find_company("BMW")
        assert result["status"] == "ok"
        assert result["company"]["name"] == "BMW AG"
        assert result["company"]["ticker"] == "BMW"


@pytest.mark.asyncio
async def test_cse_find_company_ambiguous_auto_selects_top():
    mock_response = {
        "column_names": ["name", "ticker", "exchange"],
        "company_info": [
            ["BMW AG", "BMW", "NYSE"],
            ["BMW Group Holdings", "BMWG", "NASDAQ"],
        ],
    }
    with patch(
        "ideation_agent.tools.company_search._cse_get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await cse_find_company("BMW")
        assert result["status"] == "ok"
        assert result["company"]["name"] == "BMW AG"


@pytest.mark.asyncio
async def test_cse_find_company_not_found():
    mock_response = {"column_names": ["name", "ticker", "exchange"], "company_info": []}
    with patch(
        "ideation_agent.tools.company_search._cse_get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await cse_find_company("NonExistentCompanyXYZ123")
        assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_cse_report_analysis_formats_markdown():
    mock_response = {
        "company": "SAP SE",
        "agent_dict": {
            "overview": {
                "name": "Overview",
                "summary": "Enterprise software leader.",
                "evidence": [],
            }
        },
    }
    with patch(
        "ideation_agent.tools.company_search._cse_get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await cse_report_analysis("SAP")
        assert result["status"] == "ok"
        assert "## Overview" in result["analysis_markdown"]


@pytest.mark.asyncio
async def test_cse_customer_stories_formats_markdown():
    mock_response = {
        "company": "SAP SE",
        "cve_agent_dict": {
            "companies": [
                {
                    "company_name": "Test Corp",
                    "connection": "Same industry",
                    "sap_solutions_used": ["BTP"],
                    "similar_challenges": ["AI adoption"],
                    "benefits_and_outcomes": ["Improved efficiency"],
                    "source_path": "https://example.com",
                }
            ],
            "query": "SAP",
        },
    }
    with patch(
        "ideation_agent.tools.company_search._cse_get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await cse_customer_stories("SAP")
        assert result["status"] == "ok"
        assert "## Test Corp" in result["stories_markdown"]
        assert "https://example.com" in result["story_links"]


# ---------------------------------------------------------------------------
# Async tests: enrichment pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cse_enrichment_full_pipeline():
    find_response = {
        "column_names": ["name", "ticker", "exchange"],
        "company_info": [["SAP SE", "SAP", "NYSE"]],
    }
    analysis_response = {
        "company": "SAP SE",
        "agent_dict": {"s1": {"name": "Overview", "summary": "Leader.", "evidence": []}},
    }
    stories_response = {
        "company": "SAP SE",
        "cve_agent_dict": {
            "companies": [{"company_name": "Peer", "connection": "Same", "sap_solutions_used": [], "similar_challenges": [], "benefits_and_outcomes": [], "source_path": ""}],
            "query": "",
        },
    }
    link_response = {"company": "SAP SE", "link": "https://sec.gov/sap"}

    call_count = {"n": 0}

    async def mock_cse_get(endpoint, params):
        call_count["n"] += 1
        if "find-company" in endpoint:
            return find_response
        elif "report-analysis" in endpoint:
            return analysis_response
        elif "customer-stories" in endpoint:
            return stories_response
        elif "report-link" in endpoint:
            return link_response
        return {}

    with patch(
        "ideation_agent.tools.company_search._cse_get",
        side_effect=mock_cse_get,
    ):
        result = await run_cse_enrichment("SAP")
        assert result["status"] == "ok"
        assert result["company"]["name"] == "SAP SE"
        assert "## Overview" in result["analysis_markdown"]
        assert result["report_link"] == "https://sec.gov/sap"
        assert call_count["n"] == 4  # find + analysis + stories + link


# ---------------------------------------------------------------------------
# Tests: graceful degradation
# ---------------------------------------------------------------------------


def test_cse_disabled_when_env_missing():
    with patch.dict(os.environ, {"CSE_URL": "", "CSE_KEY": ""}, clear=False):
        from ideation_agent.tools import company_search

        assert company_search.cse_enabled() is False


# ---------------------------------------------------------------------------
# Async tests: background task manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_task_lifecycle():
    async def mock_enrichment(name):
        return {"status": "ok", "company": {"name": name}}

    with patch(
        "ideation_agent.tools.company_search.run_cse_enrichment",
        side_effect=mock_enrichment,
    ):
        ctx = "test-ctx-123"
        cse_background.cleanup(ctx)

        assert cse_background.status(ctx) == "idle"
        assert cse_background.get_result(ctx) is None

        await cse_background.start_enrichment(ctx, "TestCorp")
        assert cse_background.status(ctx) == "pending"

        # Let the background task complete
        await asyncio.sleep(0.1)

        result = cse_background.get_result(ctx)
        assert result is not None
        assert result["status"] == "ok"
        assert cse_background.status(ctx) == "complete"

        cse_background.cleanup(ctx)
        assert cse_background.status(ctx) == "idle"
