"""Deterministic concurrency tests for the process-wide generation runtime."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from app.ecn.models import JobState
from app.ecn.services.generation import JobProcessingResult
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.ecn.services.runtime import GenerationRuntime
from app.persistence.records import (
    AttemptSeed,
    AttemptStart,
    ClaimedJob,
    ClaimJobResult,
    ClaimJobStatus,
    PinnedGenerationConfig,
)

from tests.fixtures.builders import make_evidence_bundle


class FixedClock:
    """Return one deterministic timezone-aware timestamp."""

    def now(self) -> datetime:
        """Return the fixed current time."""
        return datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


class StableUuidFactory:
    """Return a distinct deterministic UUID on every call."""

    def __init__(self) -> None:
        """Initialize the UUID counter."""
        self.counter = 0

    def __call__(self) -> UUID:
        """Return the next stable UUID."""
        self.counter += 1
        return uuid5(NAMESPACE_URL, f"generation-runtime:{self.counter}")


class RecordingClaimRepository:
    """Implement atomic in-memory claim state for runtime boundary tests."""

    def __init__(self, job_ids: tuple[str, ...]) -> None:
        """Store queued jobs and initialize synchronized claim observations."""
        self.states = {job_id: JobState.QUEUED for job_id in job_ids}
        self.claimed: list[tuple[str, AttemptSeed]] = []
        self._lock = threading.Lock()

    def claim_job_and_start_attempt(
        self, job_id: str, seed: AttemptSeed
    ) -> ClaimJobResult:
        """Claim a queued job once and derive its model from pinned configuration."""
        with self._lock:
            state = self.states.get(job_id)
            if state is None:
                return ClaimJobResult(ClaimJobStatus.NOT_FOUND, None)
            if state is not JobState.QUEUED:
                return ClaimJobResult(ClaimJobStatus.NOT_QUEUED, None)
            if job_id.endswith("bad0"):
                self.states[job_id] = JobState.FAILED
                return ClaimJobResult(ClaimJobStatus.PINNED_CONFIG_INVALID, None)
            self.states[job_id] = JobState.RUNNING
            self.claimed.append((job_id, seed))
        config = PinnedGenerationConfig(
            prompt_version="ecn-impact-v1",
            output_schema_version="ecn-impact-sections-v1",
            model_name="pinned-model",
            request_timeout_seconds=30,
            max_prompt_evidence_items=20,
            max_prompt_characters=20_000,
            strict_structured_output=True,
        )
        attempt = AttemptStart(
            attempt_id=seed.attempt_id,
            correlation_id=seed.correlation_id,
            started_at=seed.started_at,
            provider=seed.provider,
            model_name=config.model_name,
            model_deployment=seed.model_deployment,
            llm_endpoint=seed.llm_endpoint,
        )
        evidence = make_evidence_bundle()
        claimed = ClaimedJob(
            job_id=job_id,
            upload_id=evidence.context.upload_id,
            ecn_row_id=evidence.ecn_row_id,
            attempt_number=1,
            attempt=attempt,
            evidence=evidence,
            generation_config=config,
        )
        return ClaimJobResult(ClaimJobStatus.CLAIMED, claimed)

    def waiting_job_states(self) -> set[JobState]:
        """Return states for jobs that have not yet been claimed."""
        with self._lock:
            return {state for state in self.states.values() if state is JobState.QUEUED}

    def block_queued_job(self, job_id: str, error_code: str) -> bool:
        """Transition one queued job to blocked without creating an attempt."""
        del error_code
        with self._lock:
            if self.states.get(job_id) is not JobState.QUEUED:
                return False
            self.states[job_id] = JobState.BLOCKED
            return True

    def finish(self, job_id: str, state: JobState) -> None:
        """Record the terminal state produced by the fake generation service."""
        with self._lock:
            self.states[job_id] = state


class BlockingGenerationService:
    """Act like provider-backed generation with deterministic event barriers."""

    def __init__(self, repository: RecordingClaimRepository) -> None:
        """Bind job state and initialize concurrency counters."""
        self.repository = repository
        self.active = 0
        self.maximum_active = 0
        self.calls: list[str] = []
        self.fail_jobs: set[str] = set()
        self.release = asyncio.Event()
        self._condition = asyncio.Condition()
        self.closed = False

    async def run_claimed_job(self, claimed: ClaimedJob) -> JobProcessingResult:
        """Block a claimed call until released, then complete or raise independently."""
        async with self._condition:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.calls.append(claimed.job_id)
            self._condition.notify_all()
        try:
            await self.release.wait()
            if claimed.job_id in self.fail_jobs:
                self.repository.finish(claimed.job_id, JobState.FAILED)
                raise RuntimeError("contained sibling failure")
            self.repository.finish(claimed.job_id, JobState.COMPLETED)
            return JobProcessingResult(claimed.job_id, "COMPLETED")
        finally:
            async with self._condition:
                self.active -= 1
                self._condition.notify_all()

    async def wait_until_active(self, count: int) -> None:
        """Wait deterministically until exactly ``count`` calls are active."""
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(lambda: self.active >= count), timeout=2
            )

    def release_all(self) -> None:
        """Release current and subsequent provider calls."""
        self.release.set()

    async def close(self) -> None:
        """Record provider-service shutdown."""
        self.closed = True


def _job(label: str) -> str:
    """Return a canonical deterministic UUID for a logical job label."""
    return str(uuid5(NAMESPACE_URL, f"runtime-job:{label}"))


class BlockedProviderStatus:
    """Expose one cached authentication blocker to the runtime."""

    def generation_blocker(self) -> str:
        """Return the stable cached preflight failure."""
        return "PROVIDER_AUTH_FAILED"


@pytest.fixture
def runtime_fixture():
    """Return runtime collaborators for seven queued jobs."""
    jobs = tuple(_job(str(index)) for index in range(7))
    repository = RecordingClaimRepository(jobs)
    service = BlockingGenerationService(repository)
    runtime = GenerationRuntime(
        service=service,
        analysis_repository=repository,
        mutation_guard=HistoryMutationGuard(),
        maximum_concurrency=4,
        clock=FixedClock(),
        uuid_factory=StableUuidFactory(),
    )
    return runtime, service, repository, jobs


@pytest.mark.asyncio
async def test_uploads_and_retries_share_a_four_call_cap(runtime_fixture) -> None:
    """Never exceed four active calls across every scheduling source."""
    runtime, service, repository, jobs = runtime_fixture
    runtime.schedule_jobs(jobs[:3])
    runtime.schedule_jobs(jobs[3:6])
    runtime.schedule_retry(jobs[6])
    await service.wait_until_active(4)
    assert service.maximum_active == 4
    assert repository.waiting_job_states() == {JobState.QUEUED}
    service.release_all()
    await runtime.wait_until_idle()
    assert service.maximum_active == 4
    assert set(repository.states.values()) == {JobState.COMPLETED}


@pytest.mark.asyncio
async def test_waiting_jobs_remain_queued_until_a_slot_is_available(
    runtime_fixture,
) -> None:
    """Acquire provider capacity before atomically claiming a queued job."""
    runtime, service, repository, jobs = runtime_fixture
    runtime.schedule_jobs(jobs)
    await service.wait_until_active(4)
    assert sum(state is JobState.RUNNING for state in repository.states.values()) == 4
    assert sum(state is JobState.QUEUED for state in repository.states.values()) == 3
    service.release_all()
    await runtime.wait_until_idle()


@pytest.mark.asyncio
async def test_duplicate_scheduling_calls_provider_once(runtime_fixture) -> None:
    """Let the repository's atomic claim make duplicate scheduling idempotent."""
    runtime, service, repository, jobs = runtime_fixture
    runtime.schedule_retry(jobs[0])
    runtime.schedule_retry(jobs[0])
    await service.wait_until_active(1)
    service.release_all()
    await runtime.wait_until_idle()
    assert service.calls == [jobs[0]]
    assert len(repository.claimed) == 1


@pytest.mark.asyncio
async def test_sibling_exception_does_not_cancel_other_jobs(runtime_fixture) -> None:
    """Gather job failures independently so every sibling reaches its own outcome."""
    runtime, service, repository, jobs = runtime_fixture
    service.fail_jobs.add(jobs[1])
    runtime.schedule_jobs(jobs[:3])
    await service.wait_until_active(3)
    service.release_all()
    await runtime.wait_until_idle()
    assert repository.states[jobs[0]] is JobState.COMPLETED
    assert repository.states[jobs[1]] is JobState.FAILED
    assert repository.states[jobs[2]] is JobState.COMPLETED


@pytest.mark.asyncio
async def test_event_loop_remains_responsive_while_jobs_run(runtime_fixture) -> None:
    """Keep background processing asynchronous while provider calls are blocked."""
    runtime, service, _repository, jobs = runtime_fixture
    runtime.schedule_retry(jobs[0])
    await service.wait_until_active(1)
    observed: list[str] = []
    await asyncio.sleep(0)
    observed.append("responsive")
    assert observed == ["responsive"]
    service.release_all()
    await runtime.wait_until_idle()


@pytest.mark.asyncio
async def test_wait_until_idle_includes_tasks_added_while_waiting(
    runtime_fixture,
) -> None:
    """Loop over task snapshots until later scheduling has also completed."""
    runtime, service, _repository, jobs = runtime_fixture
    runtime.schedule_retry(jobs[0])
    await service.wait_until_active(1)
    waiter = asyncio.create_task(runtime.wait_until_idle())
    runtime.schedule_retry(jobs[1])
    service.release_all()
    await waiter
    assert set(service.calls) == {jobs[0], jobs[1]}


@pytest.mark.asyncio
async def test_pinned_config_failure_never_calls_provider(runtime_fixture) -> None:
    """Return a stable failure when repository claim rejects malformed pinned config."""
    runtime, service, repository, _jobs = runtime_fixture
    bad_job = "00000000-0000-4000-8000-00000000bad0"
    repository.states[bad_job] = JobState.QUEUED
    result = await runtime.process_job(bad_job)
    assert result == JobProcessingResult(bad_job, "FAILED", "PINNED_CONFIG_INVALID")
    assert service.calls == []


@pytest.mark.asyncio
async def test_failed_preflight_blocks_job_without_claim_or_provider_call() -> None:
    """Keep deterministic ingestion visible while preventing repeated AI failures."""
    job_id = _job("provider-blocked")
    repository = RecordingClaimRepository((job_id,))
    service = BlockingGenerationService(repository)
    runtime = GenerationRuntime(
        service=service,
        analysis_repository=repository,
        mutation_guard=HistoryMutationGuard(),
        maximum_concurrency=4,
        clock=FixedClock(),
        uuid_factory=StableUuidFactory(),
        provider_status_service=BlockedProviderStatus(),  # type: ignore[arg-type]
    )
    result = await runtime.process_job(job_id)
    assert result == JobProcessingResult(job_id, "BLOCKED", "PROVIDER_AUTH_FAILED")
    assert repository.states[job_id] is JobState.BLOCKED
    assert repository.claimed == []
    assert service.calls == []
