"""Shutdown and restart-safety tests for in-process ECN generation."""

from __future__ import annotations

import asyncio
import threading

import pytest
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.ecn.services.runtime import GenerationRuntime

from tests.unit.test_generation_service import (
    BlockingGenerationService,
    FixedClock,
    RecordingClaimRepository,
    StableUuidFactory,
    _job,
)


@pytest.mark.asyncio
async def test_close_cancels_and_gathers_tasks_before_provider_close() -> None:
    """Finish task cancellation before closing the shared provider client."""
    job = _job("shutdown")
    repository = RecordingClaimRepository((job,))
    service = BlockingGenerationService(repository)
    runtime = GenerationRuntime(
        service,
        repository,
        HistoryMutationGuard(),
        1,
        FixedClock(),
        StableUuidFactory(),
    )
    runtime.schedule_retry(job)
    await service.wait_until_active(1)
    await runtime.close()
    assert service.active == 0
    assert service.closed is True


class BlockingClaimRepository(RecordingClaimRepository):
    """Block the synchronous claim transaction until a test-controlled release."""

    def __init__(self, job: str) -> None:
        """Initialize claim barriers and queued state."""
        super().__init__((job,))
        self.entered = threading.Event()
        self.release_claim = threading.Event()

    def claim_job_and_start_attempt(self, job_id, seed):
        """Hold the fake transaction before completing the normal atomic claim."""
        self.entered.set()
        self.release_claim.wait(timeout=5)
        return super().claim_job_and_start_attempt(job_id, seed)


@pytest.mark.asyncio
async def test_shutdown_waits_for_blocking_claim_before_provider_and_engine_close() -> (
    None
):
    """Do not close runtime resources while a synchronous claim transaction runs."""
    job = _job("blocking-claim")
    repository = BlockingClaimRepository(job)
    service = BlockingGenerationService(repository)
    runtime = GenerationRuntime(
        service,
        repository,
        HistoryMutationGuard(),
        1,
        FixedClock(),
        StableUuidFactory(),
    )
    runtime.schedule_retry(job)
    await asyncio.to_thread(repository.entered.wait, 2)
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert closing.done() is False
    assert service.closed is False
    repository.release_claim.set()
    await closing
    assert service.closed is True
