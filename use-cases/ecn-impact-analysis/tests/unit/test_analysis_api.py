"""Progress and manual-retry HTTP workflow tests."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.main import create_app
from app.persistence.records import (
    AnalysisProgressResponse,
    RetryQueueResult,
    RetryQueueStatus,
)
from fastapi.testclient import TestClient

UPLOAD_ID = "018f54fe-7b2d-7d7d-aef4-81940d349501"
ECN_ID = "018f54fe-7b2d-7d7d-aef4-81940d349502"
JOB_ID = "018f54fe-7b2d-7d7d-aef4-81940d349503"


class AnalysisRepositoryFake:
    """Return a stable progress snapshot."""

    def get_progress(self, upload_id: str | None = None) -> AnalysisProgressResponse:
        """Return mutually exclusive aggregate counts."""
        return AnalysisProgressResponse(
            queued_count=1,
            running_count=1,
            completed_count=2,
            completed_with_gaps_count=3,
            failed_count=4,
            unfinished_count=2,
            total_count=11,
        )


class RetryServiceFake:
    """Return a configurable retry result."""

    def __init__(self, result: RetryQueueResult) -> None:
        """Store the next transition result."""
        self.result = result

    async def queue_failed(self, _ecn_row_id: str) -> RetryQueueResult:
        """Return the configured result."""
        return self.result


class RuntimeFake:
    """Record post-commit retry scheduling."""

    def __init__(self) -> None:
        """Initialize an empty scheduled list."""
        self.retries: list[str] = []

    def schedule_retry(self, job_id: str) -> None:
        """Record one scheduled committed job."""
        self.retries.append(job_id)


def _client(monkeypatch, retry_result: RetryQueueResult):
    """Build a test app around progress, retry, and runtime fakes."""
    monkeypatch.setenv("API_KEY", "test-api-key")
    runtime = RuntimeFake()
    services = SimpleNamespace(
        analysis_repository=AnalysisRepositoryFake(),
        retry_service=RetryServiceFake(retry_result),
        runtime=runtime,
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield test services."""
        yield services

    return TestClient(create_app(service_lifespan_factory=lifespan)), runtime


def test_progress_includes_scope_and_all_non_overlapping_counts(monkeypatch) -> None:
    """Expose the optional upload identity alongside repository counts."""
    client, _runtime = _client(
        monkeypatch, RetryQueueResult(RetryQueueStatus.QUEUED, JOB_ID)
    )
    with client:
        response = client.get(
            f"/api/analysis-progress?upload_id={UPLOAD_ID}",
            headers={"X-API-Key": "test-api-key"},
        )
    assert response.status_code == 200
    assert response.json() == {
        "upload_id": UPLOAD_ID,
        "queued_count": 1,
        "blocked_count": 0,
        "ready_count": 0,
        "running_count": 1,
        "completed_count": 2,
        "completed_with_gaps_count": 3,
        "failed_count": 4,
        "unfinished_count": 2,
        "total_count": 11,
    }


def test_retry_schedules_only_after_successful_queue_commit(monkeypatch) -> None:
    """Schedule the committed job exactly once after a QUEUED result."""
    client, runtime = _client(
        monkeypatch, RetryQueueResult(RetryQueueStatus.QUEUED, JOB_ID)
    )
    with client:
        response = client.post(
            f"/api/ecns/{ECN_ID}/analysis/retry",
            headers={"X-API-Key": "test-api-key"},
        )
    assert response.status_code == 202
    assert response.json() == {
        "ecn_row_id": ECN_ID,
        "job_id": JOB_ID,
        "state": "QUEUED",
    }
    assert runtime.retries == [JOB_ID]


def test_retry_maps_missing_and_non_retryable_without_scheduling(monkeypatch) -> None:
    """Translate the repository discriminator to 404/409 safe domain errors."""
    for result, expected_status in (
        (RetryQueueResult(RetryQueueStatus.ECN_NOT_FOUND, None), 404),
        (RetryQueueResult(RetryQueueStatus.NOT_RETRYABLE, JOB_ID), 409),
    ):
        client, runtime = _client(monkeypatch, result)
        with client:
            response = client.post(
                f"/api/ecns/{ECN_ID}/analysis/retry",
                headers={"X-API-Key": "test-api-key"},
            )
        assert response.status_code == expected_status
        assert runtime.retries == []
        assert set(response.json()) == {"code", "message", "details"}
