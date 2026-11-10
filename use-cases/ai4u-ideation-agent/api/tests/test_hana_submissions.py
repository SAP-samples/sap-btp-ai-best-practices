"""Tests for HANA submissions store — exercises real SQL logic with mocked connection."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from app.services.hana_submissions import (
    HanaSubmissionStore,
    _compute_content_hash,
    get_submission_store,
)


# ── Unit tests for helper functions ──────────────────────────────────────────

def test_compute_content_hash_deterministic():
    """Same input always produces the same hash."""
    h1 = _compute_content_hash("Healthcare", "AI Triage", "Long waits", "AI solution")
    h2 = _compute_content_hash("Healthcare", "AI Triage", "Long waits", "AI solution")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_content_hash_case_insensitive():
    """Hash is case-insensitive (lowercase normalization)."""
    h1 = _compute_content_hash("Healthcare", "AI Triage", "Long waits", "AI solution")
    h2 = _compute_content_hash("HEALTHCARE", "ai triage", "LONG WAITS", "ai SOLUTION")
    assert h1 == h2


def test_compute_content_hash_different_inputs():
    """Different inputs produce different hashes."""
    h1 = _compute_content_hash("Healthcare", "AI Triage", "Long waits", "AI solution")
    h2 = _compute_content_hash("Finance", "Fraud Detection", "Undetected fraud", "Monitoring")
    assert h1 != h2


# ── Integration tests exercising real store methods via mocked HANA ──────────

@pytest.fixture
def mock_hana():
    """Patch the hana_store module functions used by HanaSubmissionStore."""
    with patch("ideation_agent.hana_store._execute", new_callable=AsyncMock) as mock_exec, \
         patch("ideation_agent.hana_store._fetch_all", new_callable=AsyncMock) as mock_fetch_all, \
         patch("ideation_agent.hana_store._fetch_one", new_callable=AsyncMock) as mock_fetch_one:
        yield {
            "execute": mock_exec,
            "fetch_all": mock_fetch_all,
            "fetch_one": mock_fetch_one,
        }


@pytest.mark.asyncio
async def test_init_schema_creates_table(mock_hana):
    """init_schema calls CREATE TABLE and CREATE INDEX statements."""
    store = HanaSubmissionStore()
    await store.init_schema()

    # Should have been called once for table + 1 ALTER + 4 indexes = 6 total
    assert mock_hana["execute"].call_count == 6

    # First call should be CREATE TABLE
    first_call_sql = mock_hana["execute"].call_args_list[0][0][0]
    assert "CREATE COLUMN TABLE IDEATION_SUBMISSIONS" in first_call_sql
    assert "SUBMISSION_ID" in first_call_sql
    assert "SCORE" in first_call_sql


@pytest.mark.asyncio
async def test_init_schema_idempotent(mock_hana):
    """init_schema doesn't crash if table already exists."""
    mock_hana["execute"].side_effect = Exception("table already exists (error 288)")

    store = HanaSubmissionStore()
    # Should not raise
    await store.init_schema()


@pytest.mark.asyncio
async def test_upsert_submission_insert_new(mock_hana):
    """upsert_submission inserts when no existing entry found."""
    mock_hana["fetch_one"].return_value = None  # No existing submission

    store = HanaSubmissionStore()
    submission_id = await store.upsert_submission({
        "uid": "user-001",
        "email": "john@sap.com",
        "company_domain": "sap.com",
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER waits",
        "solution": "AI classifies urgency",
        "app_type": "Agent",
        "company_context": "",
        "sources": [],
        "evaluator_feedback": "Great!",
        "evaluator_detailed_analysis": "Solid approach.",
        "evaluator_suggestions": ["Add NLP"],
        "appropriateness_level": "excellent",
        "score": 850,
        "rating": 3,
        "context_id": "ctx-1",
        "source": "ideation-agent",
    })

    # Should have checked for existing by context_id (fetch_one) + content_hash (fetch_one) then inserted (execute)
    assert mock_hana["fetch_one"].call_count == 2
    assert mock_hana["execute"].call_count == 1

    # Verify the INSERT SQL
    insert_sql = mock_hana["execute"].call_args[0][0]
    assert "INSERT INTO IDEATION_SUBMISSIONS" in insert_sql

    # Verify params include the data
    insert_params = mock_hana["execute"].call_args[0][1]
    assert "user-001" in insert_params
    assert "john@sap.com" in insert_params
    assert "sap.com" in insert_params
    assert 850 in insert_params

    # Returns a submission_id (hash-based)
    assert len(submission_id) == 64


@pytest.mark.asyncio
async def test_upsert_submission_update_existing(mock_hana):
    """upsert_submission updates when content hash matches existing entry."""
    mock_hana["fetch_one"].return_value = {"submission_id": "existing-sub-123"}

    store = HanaSubmissionStore()
    submission_id = await store.upsert_submission({
        "uid": "user-001",
        "email": "john@sap.com",
        "company_domain": "sap.com",
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER waits",
        "solution": "AI classifies urgency",
        "app_type": "Agent",
        "evaluator_feedback": "Updated feedback",
        "evaluator_detailed_analysis": "Better analysis",
        "evaluator_suggestions": [],
        "appropriateness_level": "good",
        "score": 900,
        "rating": 2,
        "context_id": "ctx-2",
        "source": "ideation-agent",
    })

    assert submission_id == "existing-sub-123"

    # Should have done UPDATE not INSERT
    update_sql = mock_hana["execute"].call_args[0][0]
    assert "UPDATE IDEATION_SUBMISSIONS" in update_sql
    assert "WHERE SUBMISSION_ID = ?" in update_sql


@pytest.mark.asyncio
async def test_get_user_history(mock_hana):
    """get_user_history returns formatted results."""
    mock_hana["fetch_all"].return_value = [
        {
            "submission_id": "sub-1",
            "industry": "Healthcare",
            "title": "AI Triage",
            "problem": "Long waits",
            "solution": "AI classifies",
            "app_type": "Agent",
            "evaluator_feedback": "Great",
            "evaluator_detailed_analysis": "Nice",
            "evaluator_suggestions_json": '["Add NLP"]',
            "appropriateness_level": "excellent",
            "score": 850,
            "rating": 3,
            "submitted_at": datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
            "context_id": "ctx-1",
        },
    ]

    store = HanaSubmissionStore()
    rows = await store.get_user_history("user-001", limit=10, offset=0)

    assert len(rows) == 1
    assert rows[0]["score"] == 850
    assert rows[0]["evaluator_suggestions"] == ["Add NLP"]
    assert rows[0]["submitted_at"] == "2025-07-01T10:00:00+00:00"

    # Verify SQL has correct ordering and pagination
    sql = mock_hana["fetch_all"].call_args[0][0]
    assert "WHERE UID = ?" in sql
    assert "ORDER BY SUBMITTED_AT DESC" in sql
    assert "LIMIT 10 OFFSET 0" in sql


@pytest.mark.asyncio
async def test_get_company_leaderboard(mock_hana):
    """get_company_leaderboard returns ranked entries for a domain."""
    mock_hana["fetch_all"].return_value = [
        {
            "submission_id": "sub-a",
            "uid": "user-a",
            "email": "alice@sap.com",
            "industry": "Healthcare",
            "title": "AI Triage",
            "problem": "Long waits",
            "solution": "AI solution",
            "app_type": "Agent",
            "evaluator_feedback": "Excellent!",
            "evaluator_detailed_analysis": "Great.",
            "evaluator_suggestions_json": "[]",
            "appropriateness_level": "excellent",
            "score": 920,
            "rating": 3,
            "submitted_at": datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
            "context_id": "ctx-1",
        },
        {
            "submission_id": "sub-b",
            "uid": "user-b",
            "email": "bob@sap.com",
            "industry": "Finance",
            "title": "Fraud",
            "problem": "Fraud",
            "solution": "Monitor",
            "app_type": "Agent",
            "evaluator_feedback": "OK",
            "evaluator_detailed_analysis": "Decent.",
            "evaluator_suggestions_json": None,
            "appropriateness_level": "good",
            "score": 700,
            "rating": 2,
            "submitted_at": datetime(2025, 6, 30, 9, 0, tzinfo=timezone.utc),
            "context_id": "ctx-2",
        },
    ]

    store = HanaSubmissionStore()
    rows = await store.get_company_leaderboard("sap.com", limit=50)

    assert len(rows) == 2
    assert rows[0]["score"] == 920  # Highest first (ORDER BY in SQL)
    assert rows[1]["score"] == 700
    assert rows[1]["evaluator_suggestions"] == []  # None → []

    # Verify SQL
    sql = mock_hana["fetch_all"].call_args[0][0]
    assert "WHERE COMPANY_DOMAIN = ?" in sql
    assert "ORDER BY SCORE DESC, SUBMITTED_AT DESC" in sql
    assert "LIMIT 50" in sql


@pytest.mark.asyncio
async def test_count_company_entries(mock_hana):
    """count_company_entries returns integer count."""
    mock_hana["fetch_one"].return_value = {"cnt": 42}

    store = HanaSubmissionStore()
    count = await store.count_company_entries("sap.com")

    assert count == 42
    sql = mock_hana["fetch_one"].call_args[0][0]
    assert "COUNT(*)" in sql
    assert "WHERE COMPANY_DOMAIN = ?" in sql


@pytest.mark.asyncio
async def test_count_user_history(mock_hana):
    """count_user_history returns integer count."""
    mock_hana["fetch_one"].return_value = {"cnt": 5}

    store = HanaSubmissionStore()
    count = await store.count_user_history("user-001")

    assert count == 5


@pytest.mark.asyncio
async def test_get_submission_store_singleton():
    """get_submission_store returns a singleton and initializes schema once."""
    import app.services.hana_submissions as module

    # Reset singleton state
    module._store = None

    with patch.object(HanaSubmissionStore, "init_schema", new_callable=AsyncMock) as mock_init:
        store1 = await get_submission_store()
        store2 = await get_submission_store()

        assert store1 is store2
        mock_init.assert_called_once()  # Only initialized once

    # Reset for other tests
    module._store = None


@pytest.mark.asyncio
async def test_get_by_context_found(mock_hana):
    """get_by_context returns submission data when found."""
    mock_hana["fetch_one"].return_value = {
        "submission_id": "sub-ctx-1",
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long waits",
        "solution": "AI classifies",
        "app_type": "Agent",
        "evaluator_feedback": "Nailed it!",
        "evaluator_detailed_analysis": "Strong approach.",
        "evaluator_suggestions_json": '["Add NLP"]',
        "appropriateness_level": "excellent",
        "score": 870,
        "rating": 0,
        "submitted_at": datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
        "context_id": "ctx-abc",
    }

    store = HanaSubmissionStore()
    row = await store.get_by_context("user-001", "ctx-abc")

    assert row is not None
    assert row["score"] == 870
    assert row["evaluator_feedback"] == "Nailed it!"
    assert row["evaluator_suggestions"] == ["Add NLP"]
    assert row["submitted_at"] == "2025-07-01T10:00:00+00:00"

    # Verify correct SQL
    sql = mock_hana["fetch_one"].call_args[0][0]
    assert "WHERE UID = ? AND CONTEXT_ID = ?" in sql


@pytest.mark.asyncio
async def test_get_by_context_not_found(mock_hana):
    """get_by_context returns None when no match."""
    mock_hana["fetch_one"].return_value = None

    store = HanaSubmissionStore()
    row = await store.get_by_context("user-001", "ctx-missing")

    assert row is None


@pytest.mark.asyncio
async def test_get_by_context_empty_context_id(mock_hana):
    """get_by_context returns None for empty context_id without querying."""
    store = HanaSubmissionStore()
    row = await store.get_by_context("user-001", "")

    assert row is None
    mock_hana["fetch_one"].assert_not_called()


# ── get_all_by_context tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_all_by_context_multiple(mock_hana):
    """get_all_by_context returns all evals for a conversation ordered by message_index."""
    mock_hana["fetch_all"].return_value = [
        {
            "submission_id": "sub-1",
            "industry": "Healthcare",
            "title": "AI Triage v1",
            "problem": "Long waits",
            "solution": "AI classifies",
            "app_type": "Agent",
            "evaluator_feedback": "Good start",
            "evaluator_detailed_analysis": "Decent.",
            "evaluator_suggestions_json": '["Add NLP"]',
            "appropriateness_level": "good",
            "score": 700,
            "rating": 0,
            "submitted_at": datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
            "context_id": "ctx-multi",
            "message_index": 5,
        },
        {
            "submission_id": "sub-2",
            "industry": "Healthcare",
            "title": "AI Triage v2",
            "problem": "Long waits",
            "solution": "AI classifies + NLP",
            "app_type": "Agent",
            "evaluator_feedback": "Great improvement",
            "evaluator_detailed_analysis": "Strong.",
            "evaluator_suggestions_json": '["Add monitoring"]',
            "appropriateness_level": "excellent",
            "score": 900,
            "rating": 0,
            "submitted_at": datetime(2025, 7, 1, 11, 0, tzinfo=timezone.utc),
            "context_id": "ctx-multi",
            "message_index": 9,
        },
    ]

    store = HanaSubmissionStore()
    rows = await store.get_all_by_context("user-001", "ctx-multi")

    assert len(rows) == 2
    assert rows[0]["score"] == 700
    assert rows[0]["message_index"] == 5
    assert rows[1]["score"] == 900
    assert rows[1]["message_index"] == 9
    assert rows[1]["evaluator_suggestions"] == ["Add monitoring"]

    sql = mock_hana["fetch_all"].call_args[0][0]
    assert "WHERE UID = ? AND CONTEXT_ID = ?" in sql
    assert "ORDER BY MESSAGE_INDEX ASC" in sql


@pytest.mark.asyncio
async def test_get_all_by_context_empty(mock_hana):
    """get_all_by_context returns empty list when no match."""
    mock_hana["fetch_all"].return_value = []

    store = HanaSubmissionStore()
    rows = await store.get_all_by_context("user-001", "ctx-none")
    assert rows == []


@pytest.mark.asyncio
async def test_get_all_by_context_empty_context_id(mock_hana):
    """get_all_by_context returns empty list for empty context_id without querying."""
    store = HanaSubmissionStore()
    rows = await store.get_all_by_context("user-001", "")
    assert rows == []
    mock_hana["fetch_all"].assert_not_called()


@pytest.mark.asyncio
async def test_upsert_with_message_index_dedup(mock_hana):
    """upsert with message_index queries by (context_id, uid, message_index)."""
    mock_hana["fetch_one"].return_value = None  # No existing

    store = HanaSubmissionStore()
    await store.upsert_submission({
        "uid": "user-001", "email": "a@sap.com", "company_domain": "sap.com",
        "industry": "Tech", "title": "v1", "problem": "P", "solution": "S",
        "app_type": "Agent", "evaluator_feedback": "OK",
        "evaluator_detailed_analysis": "", "evaluator_suggestions": [],
        "appropriateness_level": "good", "score": 700, "rating": 0,
        "context_id": "ctx-1", "source": "ideation-agent", "message_index": 5,
    })

    # First fetch_one call should include MESSAGE_INDEX in the query
    first_call_sql = mock_hana["fetch_one"].call_args_list[0][0][0]
    assert "MESSAGE_INDEX = ?" in first_call_sql

    # INSERT should include MESSAGE_INDEX
    insert_sql = mock_hana["execute"].call_args[0][0]
    assert "MESSAGE_INDEX" in insert_sql
    insert_params = mock_hana["execute"].call_args[0][1]
    assert 5 in insert_params


@pytest.mark.asyncio
async def test_upsert_same_context_different_message_index_no_overwrite(mock_hana):
    """Two submissions with same context but different message_index create separate rows.

    Reproduces the exact reload bug scenario: eval1 (msg 5, score 920) and
    eval2 (msg 9, score 850) must not interfere with each other during upsert.
    Each gets its own INSERT because the dedup query includes MESSAGE_INDEX.
    """
    store = HanaSubmissionStore()

    # --- First submission (message_index=5, score=920) ---
    mock_hana["fetch_one"].return_value = None  # No existing row
    await store.upsert_submission({
        "uid": "user-001", "email": "a@sap.com", "company_domain": "sap.com",
        "industry": "Procurement", "title": "v1", "problem": "PDF matching",
        "solution": "Document AI", "app_type": "Application",
        "evaluator_feedback": "Great!", "evaluator_detailed_analysis": "Sharp.",
        "evaluator_suggestions": ["Add learning"], "appropriateness_level": "excellent",
        "score": 920, "rating": 0, "context_id": "ctx-reload",
        "source": "ideation-agent", "message_index": 5,
    })

    # Verify the dedup query used (context_id, uid, message_index=5)
    dedup_sql = mock_hana["fetch_one"].call_args_list[0][0][0]
    dedup_params = mock_hana["fetch_one"].call_args_list[0][0][1]
    assert "MESSAGE_INDEX = ?" in dedup_sql
    assert dedup_params == ("ctx-reload", "user-001", 5)

    # Verify INSERT with score=920
    insert_params = mock_hana["execute"].call_args_list[0][0][1]
    assert 920 in insert_params
    assert 5 in insert_params  # message_index

    # --- Second submission (message_index=9, score=850) ---
    mock_hana["fetch_one"].reset_mock()
    mock_hana["execute"].reset_mock()
    mock_hana["fetch_one"].return_value = None  # Different message_index → no existing row

    await store.upsert_submission({
        "uid": "user-001", "email": "a@sap.com", "company_domain": "sap.com",
        "industry": "Procurement", "title": "v2", "problem": "PDF matching",
        "solution": "Document AI + HANA", "app_type": "Application",
        "evaluator_feedback": "Solid!", "evaluator_detailed_analysis": "Good.",
        "evaluator_suggestions": ["Feedback loop"], "appropriateness_level": "excellent",
        "score": 850, "rating": 0, "context_id": "ctx-reload",
        "source": "ideation-agent", "message_index": 9,
    })

    # Verify dedup query used message_index=9 (NOT 5)
    dedup_sql2 = mock_hana["fetch_one"].call_args_list[0][0][0]
    dedup_params2 = mock_hana["fetch_one"].call_args_list[0][0][1]
    assert "MESSAGE_INDEX = ?" in dedup_sql2
    assert dedup_params2 == ("ctx-reload", "user-001", 9)

    # Verify INSERT with score=850 (NOT 920)
    insert_params2 = mock_hana["execute"].call_args_list[0][0][1]
    assert 850 in insert_params2
    assert 9 in insert_params2  # message_index
