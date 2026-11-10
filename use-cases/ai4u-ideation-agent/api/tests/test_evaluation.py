"""Tests for the evaluation service and router."""

import json
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


def test_evaluation_service_builds_user_prompt():
    """Test that _build_user_prompt formats correctly."""
    from app.services.evaluation import _build_user_prompt

    prompt = _build_user_prompt(
        industry="Healthcare",
        title="AI Triage Assistant",
        problem="Patients wait too long in emergency rooms.",
        solution="An AI agent that triages patients based on symptoms.",
        app_type="Agent",
        company_context="Large hospital network",
    )

    assert "Healthcare" in prompt
    assert "AI Triage Assistant" in prompt
    assert "Patients wait too long" in prompt
    assert "AI agent that triages patients" in prompt
    assert "Agent" in prompt
    assert "Large hospital network" in prompt


def test_evaluation_service_builds_prompt_without_company_context():
    """Test prompt without optional company_context."""
    from app.services.evaluation import _build_user_prompt

    prompt = _build_user_prompt(
        industry="Retail",
        title="Demand Forecasting",
        problem="Stockouts during peak seasons.",
        solution="ML model predicting demand.",
        app_type="Application",
    )

    assert "Retail" in prompt
    assert "Company Context" not in prompt


def test_extract_message_content_from_orchestration_result():
    """Test _extract_message_content handles orchestration v2 response shape."""
    from app.services.evaluation import _extract_message_content

    # Mock orchestration result structure
    mock_message = MagicMock()
    mock_message.content = '{"score": 750, "feedback": "Nice!"}'

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_final_result = MagicMock()
    mock_final_result.choices = [mock_choice]

    mock_result = MagicMock()
    mock_result.final_result = mock_final_result

    content = _extract_message_content(mock_result)
    assert content == '{"score": 750, "feedback": "Nice!"}'


def test_extract_message_content_empty_result():
    """Test _extract_message_content with empty result."""
    from app.services.evaluation import _extract_message_content

    mock_result = MagicMock()
    mock_result.final_result = None

    content = _extract_message_content(mock_result)
    assert content == ""


@patch("app.services.evaluation._orchestration_imports")
@patch("app.services.evaluation._extract_message_content")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_solution_success(mock_usage_resp, mock_extract, mock_imports):
    """Test evaluate_solution returns correct structure on success."""
    from app.services.evaluation import evaluate_solution

    # Mock the orchestration service
    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = MagicMock()

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    # Mock the extracted content
    mock_extract.return_value = json.dumps({
        "score": 820,
        "feedback": "Nailed It!",
        "detailedAnalysis": "**Strong** approach with clear reasoning.",
        "suggestions": ["Consider adding RAG"],
        "appropriatenessLevel": "excellent",
    })

    result = evaluate_solution(
        industry="Finance",
        title="Fraud Detection Agent",
        problem="Fraudulent transactions go undetected.",
        solution="An AI agent that monitors transactions in real-time.",
        app_type="Agent",
    )

    assert result["success"] is True
    assert result["score"] == 820
    assert result["feedback"] == "Nailed It!"
    assert result["appropriatenessLevel"] == "excellent"
    assert "Consider adding RAG" in result["suggestions"]


@patch("app.services.evaluation._orchestration_imports")
@patch("app.observability.llm_usage_logging.emit_llm_usage_event")
def test_evaluate_solution_orchestration_failure(mock_usage_event, mock_imports):
    """Test evaluate_solution returns fallback on orchestration failure."""
    from app.services.evaluation import evaluate_solution

    mock_service_instance = MagicMock()
    mock_service_instance.run.side_effect = RuntimeError("Connection timeout")

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    result = evaluate_solution(
        industry="Tech",
        title="Test",
        problem="Test problem",
        solution="Test solution",
    )

    assert result["success"] is False
    assert result["score"] == 500
    assert "Evaluation Error" in result["feedback"]
    assert "Connection timeout" in result["error"]


@patch("app.services.evaluation._orchestration_imports")
@patch("app.services.evaluation._extract_message_content")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_solution_invalid_json(mock_usage_resp, mock_extract, mock_imports):
    """Test evaluate_solution handles invalid JSON from LLM."""
    from app.services.evaluation import evaluate_solution

    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = MagicMock()

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    # Return invalid JSON
    mock_extract.return_value = "This is not JSON at all"

    result = evaluate_solution(
        industry="Tech",
        title="Test",
        problem="Test",
        solution="Test",
    )

    assert result["success"] is False
    assert result["score"] == 500
    assert "Invalid JSON" in result["error"]


@patch("app.services.evaluation._orchestration_imports")
@patch("app.services.evaluation._extract_message_content")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_solution_truncated_json(mock_usage_resp, mock_extract, mock_imports):
    """Test evaluate_solution handles truncated JSON from hitting token limit.

    This simulates the real failure mode: the LLM hits max_completion_tokens
    and returns an incomplete JSON string like the content got cut off mid-field.
    """
    from app.services.evaluation import evaluate_solution

    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = MagicMock()

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    # Simulates what actually happened: LLM output cut off at token limit
    mock_extract.return_value = (
        '{\n'
        '    "score": 930,\n'
        '    "feedback": "A well-oiled machine in the making.",\n'
        '    "detailed'
    )

    result = evaluate_solution(
        industry="Healthcare",
        title="AI Triage",
        problem="Long waits",
        solution="AI agent triages patients",
    )

    assert result["success"] is False
    assert result["score"] == 500
    assert "Invalid JSON" in result["error"]


# ── Integration tests with FastAPI TestClient ────────────────────────────────

@pytest.fixture
def client():
    """Create a TestClient with auth dependency overridden."""
    from app.main import app
    from app.auth import get_current_user, UserIdentity

    def mock_user():
        return UserIdentity(
            user_id="test-user-123",
            email="testuser@sap.com",
            given_name="Test",
            family_name="User",
            user_uuid="test-uuid",
            auth_provider="xsuaa",
        )

    app.dependency_overrides[get_current_user] = mock_user
    yield TestClient(app)
    app.dependency_overrides.clear()


@patch("app.services.evaluation._orchestration_imports")
@patch("app.services.evaluation._extract_message_content")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_endpoint_success(mock_usage, mock_extract, mock_imports, client):
    """Integration test: POST /api/ideation-agent/evaluate returns valid response."""
    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = MagicMock()

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    mock_extract.return_value = json.dumps({
        "score": 750,
        "feedback": "Solid work!",
        "detailedAnalysis": "- Good use of AI\n- Clear problem statement",
        "suggestions": ["Add monitoring"],
        "appropriatenessLevel": "good",
    })

    response = client.post("/api/ideation-agent/evaluate", json={
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER wait times",
        "solution": "AI agent triages patients by symptom severity",
        "app_type": "Agent",
        "context_id": "ctx-test-1",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["score"] == 750
    assert data["feedback"] == "Solid work!"
    assert data["appropriatenessLevel"] == "good"
    assert "Add monitoring" in data["suggestions"]


@patch("app.services.evaluation._orchestration_imports")
@patch("app.observability.llm_usage_logging.emit_llm_usage_event")
def test_evaluate_endpoint_handles_failure(mock_usage, mock_imports, client):
    """Integration test: POST /api/ideation-agent/evaluate handles LLM failure gracefully."""
    mock_service_instance = MagicMock()
    mock_service_instance.run.side_effect = Exception("AI Core unavailable")

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    response = client.post("/api/ideation-agent/evaluate", json={
        "industry": "Retail",
        "title": "Demand Forecast",
        "problem": "Stockouts",
        "solution": "ML model predicts demand",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["score"] == 500
    assert "Evaluation Error" in data["feedback"]


def test_evaluate_endpoint_validates_request(client):
    """Integration test: POST /api/ideation-agent/evaluate rejects invalid body."""
    response = client.post("/api/ideation-agent/evaluate", json={
        "industry": "Healthcare",
        # missing required fields: title, problem, solution
    })
    assert response.status_code == 422  # Validation error


def test_leaderboard_endpoint_rejects_public_email(client):
    """Integration test: GET /api/ideation-agent/leaderboard/company rejects public domains."""
    from app.main import app
    from app.auth import get_current_user, UserIdentity

    # Override with a gmail user
    def gmail_user():
        return UserIdentity(
            user_id="gmail-user",
            email="someone@gmail.com",
            given_name="Gmail",
            family_name="User",
            user_uuid="gmail-uuid",
            auth_provider="cdc",
        )

    app.dependency_overrides[get_current_user] = gmail_user
    response = client.get("/api/ideation-agent/leaderboard/company")
    assert response.status_code == 403
    assert "corporate email" in response.json()["detail"]


# ── Realistic integration test (no _extract_message_content mock) ────────────

@patch("app.services.evaluation._orchestration_imports")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_endpoint_realistic_orchestration_response(mock_usage, mock_imports, client):
    """Integration test that exercises real _extract_message_content with a realistic
    orchestration response object. This test would have caught the original token-limit
    truncation bug because it doesn't mock the extraction layer.
    """
    valid_json = json.dumps({
        "score": 850,
        "feedback": "Rock solid!",
        "detailedAnalysis": "- **Strong** NLP integration\n- Clear routing logic\n- Good EHR fit",
        "suggestions": ["Add multilingual support"],
        "appropriatenessLevel": "excellent",
    })

    # Build a realistic response object matching orchestration v2 shape
    mock_message = MagicMock()
    mock_message.content = valid_json  # string content

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_final_result = MagicMock()
    mock_final_result.choices = [mock_choice]

    mock_result = MagicMock()
    mock_result.final_result = mock_final_result

    # Mock only the orchestration service (not _extract_message_content)
    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = mock_result

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    response = client.post("/api/ideation-agent/evaluate", json={
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER wait times",
        "solution": "AI agent triages patients by symptom severity",
        "app_type": "Agent",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["score"] == 850
    assert data["feedback"] == "Rock solid!"


@patch("app.services.evaluation._orchestration_imports")
@patch("app.observability.llm_usage_logging.emit_llm_usage_for_response")
def test_evaluate_endpoint_truncated_response_no_extract_mock(mock_usage, mock_imports, client):
    """Simulates real token-limit truncation WITHOUT mocking _extract_message_content.
    The orchestration returns truncated content, and the endpoint should return a
    graceful fallback instead of crashing.
    """
    # This is exactly what the LLM returns when it hits max_completion_tokens
    truncated_json = '{\n    "score": 930,\n    "feedback": "A well-oiled machine in the making.",\n    "detailed'

    mock_message = MagicMock()
    mock_message.content = truncated_json

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_final_result = MagicMock()
    mock_final_result.choices = [mock_choice]

    mock_result = MagicMock()
    mock_result.final_result = mock_final_result

    mock_service_instance = MagicMock()
    mock_service_instance.run.return_value = mock_result

    mock_orchestration_service = MagicMock(return_value=mock_service_instance)
    mock_imports.return_value = {
        "Template": MagicMock(),
        "SystemMessage": MagicMock(),
        "UserMessage": MagicMock(),
        "LLMModelDetails": MagicMock(),
        "PromptTemplatingModuleConfig": MagicMock(),
        "ModuleConfig": MagicMock(),
        "OrchestrationConfig": MagicMock(),
        "OrchestrationService": mock_orchestration_service,
    }

    response = client.post("/api/ideation-agent/evaluate", json={
        "industry": "Healthcare",
        "title": "AI Triage",
        "problem": "Long ER wait times",
        "solution": "AI agent triages patients",
        "app_type": "Agent",
    })

    assert response.status_code == 200
    data = response.json()
    # Should gracefully fail, not crash with 500
    assert data["success"] is False
    assert data["score"] == 500
    assert "Invalid JSON" in data["error"]

