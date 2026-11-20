"""Bounded process-wide scheduling for independent ECN analysis jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal
from uuid import UUID

from app.ecn.llm.preflight import ProviderStatusService
from app.ecn.services.generation import (
    AnalysisGenerationService,
    Clock,
    JobProcessingResult,
)
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.records import AttemptSeed, ClaimJobResult, ClaimJobStatus

from .blocking_io import run_blocking


class GenerationRuntime:
    """Own concurrency, attempt identities/context, and every tracked task."""

    def __init__(
        self,
        service: AnalysisGenerationService,
        analysis_repository: AnalysisRepository,
        mutation_guard: HistoryMutationGuard,
        maximum_concurrency: int,
        clock: Clock,
        uuid_factory: Callable[[], UUID],
        provider_name: str = "sap-ai-core",
        llm_endpoint: Literal["responses"] = "responses",
        provider_status_service: ProviderStatusService | None = None,
        provider_max_concurrency: int | None = None,
    ) -> None:
        """Initialize job and shared provider limits with the task registry.

        Args:
            service: Attempt-level generation service called after a successful claim.
            analysis_repository: Synchronous HANA job-claim repository.
            mutation_guard: Guard shared with publication, retry, and history reset.
            maximum_concurrency: Positive concurrent-job limit.
            clock: UTC attempt timestamp source.
            uuid_factory: Factory for attempt and correlation UUIDs.
            provider_name: Persisted provider label for each new attempt.
            llm_endpoint: Persisted provider endpoint family; Responses only.
            provider_status_service: Cached preflight used to hold unavailable jobs.
            provider_max_concurrency: Shared provider-call limit, including chat.

        Raises:
            ValueError: If concurrency or provider context is invalid.
        """
        if maximum_concurrency <= 0:
            raise ValueError("maximum_concurrency must be positive")
        if not provider_name or len(provider_name) > 100:
            raise ValueError("provider_name is blank or exceeds its persistence bound")
        self._service = service
        self._repository = analysis_repository
        self._guard = mutation_guard
        self._semaphore = asyncio.Semaphore(maximum_concurrency)
        self._provider_semaphore = asyncio.Semaphore(provider_max_concurrency or maximum_concurrency)
        if isinstance(self._service, AnalysisGenerationService):
            self._service.set_provider_capacity(self._provider_semaphore)
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._provider_name = provider_name
        self._llm_endpoint = llm_endpoint
        self._provider_status_service = provider_status_service
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def provider_capacity(self):
        """Share the provider-call semaphore with user-triggered sidebar investigations."""
        return self._provider_semaphore

    def schedule_jobs(self, job_ids: tuple[str, ...]) -> None:
        """Create and retain one batch task for already-published job IDs.

        Args:
            job_ids: Explicit immutable IDs returned by the committed publication.

        Raises:
            RuntimeError: If scheduling is attempted after shutdown starts.
        """
        if not job_ids:
            return
        self._track(asyncio.create_task(self.process_jobs(job_ids)))

    def schedule_retry(self, job_id: str) -> None:
        """Create and retain one retry task through the same bounded runtime.

        Args:
            job_id: Explicit job ID returned by the committed retry transition.

        Raises:
            RuntimeError: If scheduling is attempted after shutdown starts.
        """

        async def process_retry() -> None:
            """Process one retry while containing its terminal result in the task."""
            await self.process_job(job_id)

        self._track(asyncio.create_task(process_retry()))

    def _track(self, task: asyncio.Task[None]) -> None:
        """Retain a task and consume its exception when it reaches a terminal state."""
        if self._closing:
            task.cancel()
            raise RuntimeError("generation runtime is closing")
        self._tasks.add(task)

        def discard(completed: asyncio.Task[None]) -> None:
            """Remove terminal work and retrieve contained exceptions."""
            self._tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)

    async def process_jobs(self, job_ids: tuple[str, ...]) -> None:
        """Fan out explicit jobs and await all without sibling cancellation.

        Args:
            job_ids: IDs committed by one upload publication.
        """
        await asyncio.gather(
            *(self.process_job(job_id) for job_id in job_ids),
            return_exceptions=True,
        )

    async def process_job(self, job_id: str) -> JobProcessingResult:
        """Wait for capacity, atomically claim, then process one job.

        Args:
            job_id: Immutable analysis job identifier.

        Returns:
            Completed/failed attempt result or ``SKIPPED`` for a lost claim race.
        """
        blocker = (
            self._provider_status_service.generation_blocker()
            if self._provider_status_service is not None
            else None
        )
        if blocker is not None:
            blocked = await run_blocking(
                self._repository.block_queued_job,
                job_id,
                blocker,
            )
            return JobProcessingResult(
                job_id=job_id,
                state="BLOCKED" if blocked else "SKIPPED",
                error_code=blocker if blocked else None,
            )
        async with self._semaphore:
            seed = AttemptSeed(
                attempt_id=str(self._uuid_factory()),
                correlation_id=str(self._uuid_factory()),
                started_at=self._clock.now(),
                provider=self._provider_name,
                model_deployment=None,
                llm_endpoint=self._llm_endpoint,
            )
            claim = await run_blocking(self._claim_under_guard, job_id, seed)
            if claim.status is ClaimJobStatus.PINNED_CONFIG_INVALID:
                return JobProcessingResult(
                    job_id=job_id,
                    state="FAILED",
                    error_code="PINNED_CONFIG_INVALID",
                )
            if claim.status is not ClaimJobStatus.CLAIMED:
                return JobProcessingResult(job_id=job_id, state="SKIPPED")
            if claim.claimed_job is None:
                raise RuntimeError("claimed status did not include claimed job data")
            return await self._service.run_claimed_job(claim.claimed_job)

    def _claim_under_guard(
        self,
        job_id: str,
        seed: AttemptSeed,
    ) -> ClaimJobResult:
        """Serialize the short claim mutation with retry, reset, and publication.

        Args:
            job_id: Job to claim after provider capacity becomes available.
            seed: New attempt identity and immutable provider context.

        Returns:
            The repository's discriminated atomic claim result.
        """
        with self._guard.hold():
            return self._repository.claim_job_and_start_attempt(job_id, seed)

    async def wait_until_idle(self) -> None:
        """Wait for the current tracked-task set to become empty in tests/shutdown."""
        while self._tasks:
            snapshot = tuple(self._tasks)
            await asyncio.gather(*snapshot, return_exceptions=True)

    async def close(self) -> None:
        """Cancel and gather tracked tasks before closing the provider service."""
        self._closing = True
        while self._tasks:
            snapshot = tuple(self._tasks)
            for task in snapshot:
                task.cancel()
            await asyncio.gather(*snapshot, return_exceptions=True)
        await self._service.close()
