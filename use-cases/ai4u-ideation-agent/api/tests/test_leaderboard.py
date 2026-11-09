"""Tests for the leaderboard and submissions endpoints."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from app.auth import get_current_user, UserIdentity
from app.main import app


@pytest.fixture
def corporate_client():
    """TestClient with a corporate email user (sap.com)."""
    def mock_user():
        return UserIdentity(
            user_id="corp-user-001",
            email="john.doe@sap.com",
            given_name="John",
            family_name="Doe",
            user_uuid="corp-uuid-001",
            auth_provider="xsuaa",
        )

    app.dependency_overrides[get_current_user] = mock_user
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def public_email_client():
    """TestClient with a public email user (gmail.com)."""
    def mock_user():
        return UserIdentity(
            user_id="public-user-001",
            email="someone@gmail.com",
            given_name="Someone",
            family_name="Public",
            user_uuid="public-uuid-001",
            auth_provider="cdc",
        )

    app.dependency_overrides[get_current_user] = mock_user
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Leaderboard Tests ────────────────────────────────────────────────────────

def test_leaderboard_rejects_public_email(public_email_client):
    """Leaderboard returns 403 for public email domains."""
    response = public_email_client.get("/api/ideation-agent/leaderboard/company")
    assert response.status_code == 403
    assert "corporate email" in response.json()["detail"]


@patch("app.routers.leaderboard.get_submission_store")
def test_leaderboard_returns_empty_for_no_submissions(mock_get_store, corporate_client):
    """Leaderboard returns empty list when no submissions exist."""
    mock_store = AsyncMock()
    mock_store.count_company_entries.return_value = 0
    mock_store.get_company_leaderboard.return_value = []
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/leaderboard/company")
    assert response.status_code == 200
    data = response.json()
    assert data["companyDomain"] == "sap.com"
    assert data["totalEntries"] == 0
    assert data["entries"] == []


@patch("app.routers.leaderboard.get_submission_store")
def test_leaderboard_returns_ranked_entries(mock_get_store, corporate_client):
    """Leaderboard returns entries ranked by score DESC."""
    mock_store = AsyncMock()
    mock_store.count_company_entries.return_value = 3
    mock_store.get_company_leaderboard.return_value = [
        {
            "submission_id": "sub-1",
            "uid": "user-a",
            "email": "alice@sap.com",
            "industry": "Healthcare",
            "title": "AI Triage",
            "problem": "Long ER waits",
            "solution": "AI classifies urgency",
            "app_type": "Agent",
            "evaluator_feedback": "Excellent!",
            "evaluator_detailed_analysis": "Great work.",
            "evaluator_suggestions": ["Add NLP"],
            "appropriateness_level": "excellent",
            "score": 920,
            "rating": 3,
            "submitted_at": "2025-07-01T10:00:00",
            "context_id": "ctx-1",
        },
        {
            "submission_id": "sub-2",
            "uid": "corp-user-001",  # current user
            "email": "john.doe@sap.com",
            "industry": "Finance",
            "title": "Fraud Detection",
            "problem": "Undetected fraud",
            "solution": "Real-time monitoring agent",
            "app_type": "Agent",
            "evaluator_feedback": "Solid!",
            "evaluator_detailed_analysis": "Good approach.",
            "evaluator_suggestions": [],
            "appropriateness_level": "good",
            "score": 780,
            "rating": 2,
            "submitted_at": "2025-07-02T14:00:00",
            "context_id": "ctx-2",
        },
        {
            "submission_id": "sub-3",
            "uid": "user-b",
            "email": "bob@sap.com",
            "industry": "Retail",
            "title": "Demand Forecast",
            "problem": "Stockouts",
            "solution": "ML prediction model",
            "app_type": "Application",
            "evaluator_feedback": "Decent",
            "evaluator_detailed_analysis": "OK.",
            "evaluator_suggestions": ["Try time series"],
            "appropriateness_level": "fair",
            "score": 550,
            "rating": 1,
            "submitted_at": "2025-06-30T09:00:00",
            "context_id": "ctx-3",
        },
    ]
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/leaderboard/company")
    assert response.status_code == 200
    data = response.json()

    assert data["companyDomain"] == "sap.com"
    assert data["totalEntries"] == 3
    assert len(data["entries"]) == 3

    # Verify ranking order (score DESC)
    assert data["entries"][0]["rank"] == 1
    assert data["entries"][0]["score"] == 920
    assert data["entries"][0]["title"] == "AI Triage"

    assert data["entries"][1]["rank"] == 2
    assert data["entries"][1]["score"] == 780
    assert data["entries"][1]["isCurrentUser"] is True  # john.doe@sap.com

    assert data["entries"][2]["rank"] == 3
    assert data["entries"][2]["score"] == 550

    # Verify current user is identified
    current_user_entries = [e for e in data["entries"] if e["isCurrentUser"]]
    assert len(current_user_entries) == 1
    assert current_user_entries[0]["email"] == "john.doe@sap.com"


@patch("app.routers.leaderboard.get_submission_store")
def test_leaderboard_display_name_formatting(mock_get_store, corporate_client):
    """Leaderboard formats display names as full name."""
    mock_store = AsyncMock()
    mock_store.count_company_entries.return_value = 1
    mock_store.get_company_leaderboard.return_value = [
        {
            "submission_id": "sub-x",
            "uid": "corp-user-001",
            "email": "john.doe@sap.com",
            "industry": "Tech",
            "title": "Test",
            "problem": "Test",
            "solution": "Test",
            "app_type": "Agent",
            "evaluator_feedback": "OK",
            "evaluator_detailed_analysis": "",
            "evaluator_suggestions": [],
            "appropriateness_level": "good",
            "score": 600,
            "rating": 0,
            "submitted_at": "2025-07-01T10:00:00",
            "context_id": None,
        },
    ]
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/leaderboard/company")
    data = response.json()

    # Current user should use given_name + family_name
    assert data["entries"][0]["displayName"] == "John Doe"


@patch("app.routers.leaderboard.get_submission_store")
def test_leaderboard_respects_limit(mock_get_store, corporate_client):
    """Leaderboard passes the limit parameter to the store."""
    mock_store = AsyncMock()
    mock_store.count_company_entries.return_value = 0
    mock_store.get_company_leaderboard.return_value = []
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/leaderboard/company?limit=10")
    assert response.status_code == 200

    mock_store.get_company_leaderboard.assert_called_once_with("sap.com", limit=10)


# ── Submissions Tests ────────────────────────────────────────────────────────

@patch("app.routers.submissions.get_submission_store")
def test_submit_solution_success(mock_get_store, corporate_client):
    """POST /submissions saves and returns success."""
    mock_store = AsyncMock()
    mock_store.upsert_submission.return_value = "submission-id-abc"
    mock_get_store.return_value = mock_store

    response = corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER wait times",
        "solution": "AI agent triages by symptom severity",
        "app_type": "Agent",
        "evaluator_feedback": "Nailed it!",
        "evaluator_detailed_analysis": "Great approach",
        "evaluator_suggestions": ["Add NLP"],
        "appropriateness_level": "excellent",
        "score": 850,
        "rating": 3,
        "context_id": "ctx-test-1",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["submission_id"] == "submission-id-abc"

    # Verify store was called with correct data
    call_args = mock_store.upsert_submission.call_args[0][0]
    assert call_args["uid"] == "corp-user-001"
    assert call_args["email"] == "john.doe@sap.com"
    assert call_args["company_domain"] == "sap.com"
    assert call_args["industry"] == "Healthcare"
    assert call_args["score"] == 850
    assert call_args["source"] == "ideation-agent"


@patch("app.routers.submissions.get_submission_store")
def test_submit_solution_public_email_no_domain(mock_get_store, public_email_client):
    """Submission from public email user stores empty company_domain."""
    mock_store = AsyncMock()
    mock_store.upsert_submission.return_value = "sub-public"
    mock_get_store.return_value = mock_store

    response = public_email_client.post("/api/ideation-agent/submissions", json={
        "industry": "Tech",
        "title": "Test",
        "problem": "Test problem",
        "solution": "Test solution",
        "score": 500,
    })

    assert response.status_code == 200
    call_args = mock_store.upsert_submission.call_args[0][0]
    assert call_args["company_domain"] == ""  # public domain excluded


@patch("app.routers.submissions.get_submission_store")
def test_submission_history(mock_get_store, corporate_client):
    """GET /submissions/history returns user's past submissions."""
    mock_store = AsyncMock()
    mock_store.count_user_history.return_value = 2
    mock_store.get_user_history.return_value = [
        {
            "submission_id": "sub-1",
            "industry": "Healthcare",
            "title": "AI Triage",
            "problem": "Long waits",
            "solution": "AI classifies urgency",
            "app_type": "Agent",
            "evaluator_feedback": "Great!",
            "evaluator_detailed_analysis": "Nice work",
            "evaluator_suggestions": [],
            "appropriateness_level": "excellent",
            "score": 900,
            "rating": 3,
            "submitted_at": "2025-07-01T10:00:00",
            "context_id": "ctx-1",
        },
        {
            "submission_id": "sub-2",
            "industry": "Finance",
            "title": "Fraud Detection",
            "problem": "Undetected fraud",
            "solution": "Monitoring agent",
            "app_type": "Agent",
            "evaluator_feedback": "OK",
            "evaluator_detailed_analysis": "Decent",
            "evaluator_suggestions": ["Add alerts"],
            "appropriateness_level": "good",
            "score": 700,
            "rating": 2,
            "submitted_at": "2025-06-30T08:00:00",
            "context_id": "ctx-2",
        },
    ]
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/history?limit=10&cursor=0")
    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 2
    assert data["limit"] == 10
    assert data["cursor"] == 0
    assert len(data["entries"]) == 2
    assert data["entries"][0]["score"] == 900
    assert data["entries"][1]["title"] == "Fraud Detection"


def test_submit_solution_validates_request(corporate_client):
    """POST /submissions rejects invalid body (missing required fields)."""
    response = corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Healthcare",
        # missing title, problem, solution
    })
    assert response.status_code == 422


# ── Submission by Context Tests ─────────────────────────────────────────────

@patch("app.routers.submissions.get_submission_store")
def test_get_submission_by_context_found(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id} returns stored evaluation."""
    mock_store = AsyncMock()
    mock_store.get_by_context.return_value = {
        "submission_id": "sub-ctx-1",
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long waits",
        "solution": "AI classifies",
        "app_type": "Agent",
        "evaluator_feedback": "Nailed it!",
        "evaluator_detailed_analysis": "Strong approach.",
        "evaluator_suggestions": ["Add NLP"],
        "appropriateness_level": "excellent",
        "score": 870,
        "rating": 0,
        "submitted_at": "2025-07-01T10:00:00+00:00",
        "context_id": "ctx-abc",
    }
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-abc")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["score"] == 870
    assert data["feedback"] == "Nailed it!"
    assert data["detailedAnalysis"] == "Strong approach."
    assert data["suggestions"] == ["Add NLP"]
    assert data["appropriatenessLevel"] == "excellent"
    assert data["success"] is True


@patch("app.routers.submissions.get_submission_store")
def test_get_submission_by_context_not_found(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id} returns found=false when no match."""
    mock_store = AsyncMock()
    mock_store.get_by_context.return_value = None
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-missing")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False


@patch("app.routers.submissions.get_submission_store")
def test_get_submission_by_context_ignores_error_results(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id} treats stored error evaluations as not found."""
    mock_store = AsyncMock()
    mock_store.get_by_context.return_value = {
        "submission_id": "sub-err",
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long waits",
        "solution": "AI classifies",
        "app_type": "Agent",
        "evaluator_feedback": "Evaluation Error",
        "evaluator_detailed_analysis": "We couldn't process the AI evaluation.",
        "evaluator_suggestions": [],
        "appropriateness_level": "fair",
        "score": 500,
        "rating": 0,
        "submitted_at": "2025-07-01T10:00:00+00:00",
        "context_id": "ctx-with-error",
    }
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-with-error")
    assert response.status_code == 200
    data = response.json()
    # Should be treated as not found, triggering a fresh re-evaluation
    assert data["found"] is False


# ── Bulk Submission Retrieval Tests ────────────────────────────────────────────

@patch("app.routers.submissions.get_submission_store")
def test_get_all_submissions_by_context(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id}/all returns multiple evals."""
    mock_store = AsyncMock()
    mock_store.get_all_by_context.return_value = [
        {
            "submission_id": "sub-1", "score": 700, "message_index": 5,
            "evaluator_feedback": "Good", "evaluator_detailed_analysis": "OK",
            "evaluator_suggestions": ["Add NLP"], "appropriateness_level": "good",
        },
        {
            "submission_id": "sub-2", "score": 900, "message_index": 9,
            "evaluator_feedback": "Great", "evaluator_detailed_analysis": "Strong",
            "evaluator_suggestions": [], "appropriateness_level": "excellent",
        },
    ]
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-multi/all")
    assert response.status_code == 200
    data = response.json()
    assert len(data["entries"]) == 2
    assert data["entries"][0]["messageIndex"] == 5
    assert data["entries"][0]["score"] == 700
    assert data["entries"][1]["messageIndex"] == 9
    assert data["entries"][1]["score"] == 900


@patch("app.routers.submissions.get_submission_store")
def test_get_all_submissions_filters_errors(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id}/all filters out error evaluations."""
    mock_store = AsyncMock()
    mock_store.get_all_by_context.return_value = [
        {
            "submission_id": "sub-err", "score": 500, "message_index": 3,
            "evaluator_feedback": "Evaluation Error",
            "evaluator_detailed_analysis": "We couldn't process.",
            "evaluator_suggestions": [], "appropriateness_level": "fair",
        },
        {
            "submission_id": "sub-ok", "score": 800, "message_index": 7,
            "evaluator_feedback": "Solid", "evaluator_detailed_analysis": "Nice",
            "evaluator_suggestions": ["Improve X"], "appropriateness_level": "good",
        },
    ]
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-mixed/all")
    assert response.status_code == 200
    data = response.json()
    assert len(data["entries"]) == 1
    assert data["entries"][0]["score"] == 800


@patch("app.routers.submissions.get_submission_store")
def test_get_all_submissions_empty(mock_get_store, corporate_client):
    """GET /submissions/by-context/{id}/all returns empty for unknown context."""
    mock_store = AsyncMock()
    mock_store.get_all_by_context.return_value = []
    mock_get_store.return_value = mock_store

    response = corporate_client.get("/api/ideation-agent/submissions/by-context/ctx-unknown/all")
    assert response.status_code == 200
    data = response.json()
    assert data["entries"] == []


@patch("app.routers.submissions.get_submission_store")
def test_submit_with_message_index(mock_get_store, corporate_client):
    """POST /submissions passes message_index to store."""
    mock_store = AsyncMock()
    mock_store.upsert_submission.return_value = "sub-idx"
    mock_get_store.return_value = mock_store

    response = corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER wait times",
        "solution": "AI agent triages by symptom severity",
        "score": 850,
        "message_index": 7,
        "context_id": "ctx-test",
    })

    assert response.status_code == 200
    call_args = mock_store.upsert_submission.call_args[0][0]
    assert call_args["message_index"] == 7


# ── Full Reload Scenario Tests ────────────────────────────────────────────────

@patch("app.routers.submissions.get_submission_store")
def test_reload_returns_both_evals_without_reevaluation(mock_get_store, corporate_client):
    """Simulates reload: submit 2 evals with different scores and message_index,
    then GET /all returns both with original scores intact (no LLM re-call).

    This is the exact bug scenario: user gets eval1=920 and eval2=850,
    reloads, and should see both original scores — NOT re-evaluated ones.
    """
    mock_store = AsyncMock()

    # --- Phase 1: Submit two evaluations (simulates initial conversation) ---
    mock_store.upsert_submission.side_effect = ["sub-msg5", "sub-msg9"]
    mock_get_store.return_value = mock_store

    # First eval (message_index=5, score=920)
    resp1 = corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Procurement",
        "title": "Material Group Assignment v1",
        "problem": "Manual PDF matching",
        "solution": "Document AI + semantic search",
        "app_type": "Application",
        "evaluator_feedback": "You've hit the procurement jackpot!",
        "evaluator_detailed_analysis": "Sharp approach",
        "evaluator_suggestions": ["Add active learning", "Migrate Excel to DB"],
        "appropriateness_level": "excellent",
        "score": 920,
        "rating": 0,
        "context_id": "ctx-reload-test",
        "message_index": 5,
    })
    assert resp1.status_code == 200
    assert resp1.json()["success"] is True

    # Second eval (message_index=9, score=850 — lower after refinement)
    resp2 = corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Procurement",
        "title": "Material Group Assignment v2",
        "problem": "Manual PDF matching",
        "solution": "Document AI + semantic search + HANA Cloud DB",
        "app_type": "Application",
        "evaluator_feedback": "Solid improvement",
        "evaluator_detailed_analysis": "Good addition of DB",
        "evaluator_suggestions": ["Add feedback loop"],
        "appropriateness_level": "excellent",
        "score": 850,
        "rating": 0,
        "context_id": "ctx-reload-test",
        "message_index": 9,
    })
    assert resp2.status_code == 200
    assert resp2.json()["success"] is True

    # --- Phase 2: Simulate page reload — fetch all evals for this context ---
    mock_store.get_all_by_context.return_value = [
        {
            "submission_id": "sub-msg5", "score": 920, "message_index": 5,
            "evaluator_feedback": "You've hit the procurement jackpot!",
            "evaluator_detailed_analysis": "Sharp approach",
            "evaluator_suggestions": ["Add active learning", "Migrate Excel to DB"],
            "appropriateness_level": "excellent",
        },
        {
            "submission_id": "sub-msg9", "score": 850, "message_index": 9,
            "evaluator_feedback": "Solid improvement",
            "evaluator_detailed_analysis": "Good addition of DB",
            "evaluator_suggestions": ["Add feedback loop"],
            "appropriateness_level": "excellent",
        },
    ]

    resp_all = corporate_client.get(
        "/api/ideation-agent/submissions/by-context/ctx-reload-test/all"
    )
    assert resp_all.status_code == 200
    data = resp_all.json()

    # Both evals returned with ORIGINAL scores
    assert len(data["entries"]) == 2
    assert data["entries"][0]["messageIndex"] == 5
    assert data["entries"][0]["score"] == 920  # NOT re-evaluated
    assert data["entries"][1]["messageIndex"] == 9
    assert data["entries"][1]["score"] == 850  # NOT re-evaluated

    # Scores are different (proves no overwrite happened)
    assert data["entries"][0]["score"] != data["entries"][1]["score"]


@patch("app.routers.submissions.get_submission_store")
def test_upsert_preserves_separate_message_indexes(mock_get_store, corporate_client):
    """Verify that submitting eval for message_index=5 does NOT overwrite message_index=9.

    This tests the dedup guarantee: (context_id, uid, message_index) is a unique key.
    """
    mock_store = AsyncMock()
    mock_get_store.return_value = mock_store

    # First submission
    mock_store.upsert_submission.return_value = "sub-first"
    corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Tech", "title": "V1", "problem": "P", "solution": "S",
        "score": 920, "context_id": "ctx-dedup", "message_index": 5,
    })

    # Second submission — different message_index
    mock_store.upsert_submission.return_value = "sub-second"
    corporate_client.post("/api/ideation-agent/submissions", json={
        "industry": "Tech", "title": "V2", "problem": "P", "solution": "S+",
        "score": 850, "context_id": "ctx-dedup", "message_index": 9,
    })

    # Verify both calls passed distinct message_index values
    calls = mock_store.upsert_submission.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0]["message_index"] == 5
    assert calls[0][0][0]["score"] == 920
    assert calls[1][0][0]["message_index"] == 9
    assert calls[1][0][0]["score"] == 850
