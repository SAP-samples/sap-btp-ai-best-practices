"""Tests for A2A API key protection."""

import pytest
from fastapi.testclient import TestClient

from app.a2a_app import validate_a2a_api_key


def test_validate_a2a_api_key_accepts_matching_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify A2A requests accept the configured API key.

    Inputs:
        monkeypatch: Pytest helper used to configure ``API_KEY``.

    Outputs:
        None. Assertions confirm no error is returned for a valid header.
    """

    monkeypatch.setenv("API_KEY", "secret-key")

    assert validate_a2a_api_key("secret-key") is None


def test_validate_a2a_api_key_rejects_missing_or_wrong_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify A2A requests reject missing or invalid API keys.

    Inputs:
        monkeypatch: Pytest helper used to configure ``API_KEY``.

    Outputs:
        None. Assertions confirm invalid caller credentials return 403.
    """

    monkeypatch.setenv("API_KEY", "secret-key")

    assert validate_a2a_api_key(None) == (403, "Could not validate credentials")
    assert validate_a2a_api_key("wrong") == (403, "Could not validate credentials")


def test_validate_a2a_api_key_requires_server_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify A2A requests fail closed when the server API key is absent.

    Inputs:
        monkeypatch: Pytest helper used to clear ``API_KEY``.

    Outputs:
        None. Assertions confirm missing server configuration returns 500.
    """

    monkeypatch.delenv("API_KEY", raising=False)

    assert validate_a2a_api_key("secret-key") == (
        500,
        "API key not configured on server",
    )


def test_legacy_joule_router_requires_api_key(api_client: TestClient) -> None:
    """Verify legacy Joule-facing API routes also require ``X-API-Key``.

    Inputs:
        api_client: FastAPI test client.

    Outputs:
        None. Assertions confirm the route rejects missing credentials and
        accepts the configured test API key.
    """

    unauthorized_response = api_client.get("/api/joule/summary")
    authorized_response = api_client.get(
        "/api/joule/summary",
        headers={"X-API-Key": "test-api-key"},
    )

    assert unauthorized_response.status_code == 403
    assert authorized_response.status_code == 200
