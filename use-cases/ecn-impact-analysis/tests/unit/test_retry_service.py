"""Tests for guarded asynchronous manual retry transitions."""

import pytest
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.ecn.services.retry import RetryService
from app.persistence.records import RetryQueueResult, RetryQueueStatus

ECN_ID = "018f54fe-7b2d-7d7d-aef4-81940d349201"
JOB_ID = "018f54fe-7b2d-7d7d-aef4-81940d349202"


class RecordingAnalysisRepository:
    """Record retry calls and whether the shared guard was held."""

    def __init__(self, guard: HistoryMutationGuard) -> None:
        """Store the guard and one successful repository response."""
        self.guard = guard
        self.calls: list[str] = []

    def queue_failed_job(self, ecn_row_id: str) -> RetryQueueResult:
        """Return a successful retry while recording the requested ECN."""
        self.calls.append(ecn_row_id)
        return RetryQueueResult(RetryQueueStatus.QUEUED, JOB_ID)


@pytest.mark.asyncio
async def test_retry_service_runs_repository_transition_and_returns_result() -> None:
    """Execute the synchronous retry mutation through the async service boundary."""
    guard = HistoryMutationGuard()
    repository = RecordingAnalysisRepository(guard)
    service = RetryService(repository, guard)  # type: ignore[arg-type]

    result = await service.queue_failed(ECN_ID)

    assert result == RetryQueueResult(RetryQueueStatus.QUEUED, JOB_ID)
    assert repository.calls == [ECN_ID]
