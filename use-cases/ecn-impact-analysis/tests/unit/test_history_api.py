"""Protected history-reset endpoint tests."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.ecn.errors import PersistenceStateError
from app.main import create_app
from app.persistence.records import HistoryDeletionCounts
from fastapi.testclient import TestClient


class HistoryServiceFake:
    """Return counts or an active-job conflict."""

    def __init__(self, active: bool = False) -> None:
        """Configure whether reset is blocked by active jobs."""
        self.active = active

    def reset_history(self, confirmation: str) -> HistoryDeletionCounts:
        """Enforce the phrase and return exact counts for a committed reset."""
        if self.active:
            raise PersistenceStateError(
                "HISTORY_RESET_ACTIVE_JOBS", "History reset is blocked", status_code=409
            )
        if confirmation != "RESET HISTORY":
            raise AssertionError("Pydantic should reject the wrong phrase")
        return HistoryDeletionCounts(1, 2, 3, 4, 5, 6, 7)


def _client(monkeypatch, active: bool = False) -> TestClient:
    """Build a test app with a configurable reset service."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    services = SimpleNamespace(history_service=HistoryServiceFake(active))

    @asynccontextmanager
    async def lifespan(_app):
        """Yield test services."""
        yield services

    return TestClient(create_app(service_lifespan_factory=lifespan))


def test_history_reset_requires_exact_phrase_and_maps_counts(monkeypatch) -> None:
    """Reject accidental reset bodies and expose every committed deletion count."""
    client = _client(monkeypatch)
    with client:
        rejected = client.request(
            "DELETE",
            "/api/history",
            headers={"X-API-Key": "test-api-key"},
            json={"confirmation": "reset history"},
        )
        response = client.request(
            "DELETE",
            "/api/history",
            headers={"X-API-Key": "test-api-key"},
            json={"confirmation": "RESET HISTORY"},
        )
    assert rejected.status_code == 422
    assert "reset history" not in rejected.text
    assert response.status_code == 200
    assert response.json() == {
        "deleted_attempt_count": 1,
        "deleted_analysis_count": 2,
        "deleted_job_count": 3,
        "deleted_ecn_count": 4,
        "deleted_upload_count": 5,
        "deleted_bom_row_count": 6,
        "deleted_source_artifact_count": 7,
    }


def test_history_reset_returns_conflict_immediately_for_active_work(
    monkeypatch,
) -> None:
    """Return a safe 409 instead of waiting for active jobs."""
    client = _client(monkeypatch, active=True)
    with client:
        response = client.request(
            "DELETE",
            "/api/history",
            headers={"X-API-Key": "test-api-key"},
            json={"confirmation": "RESET HISTORY"},
        )
    assert response.status_code == 409
    assert response.json()["code"] == "HISTORY_RESET_ACTIVE_JOBS"
