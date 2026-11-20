"""Attempt-level orchestration for evidence-grounded ECN impact generation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import UUID, uuid5

from app.ecn.llm.provider import (
    EcnAnalysisProvider,
    ProviderCallResult,
    ProviderCallStage,
    ProviderCallStatus,
)
from app.ecn.assistant.discovery import DiscoveryResult
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.llm.two_pass import (
    FinalEvidenceProjection,
)
from app.ecn.llm.validation import (
    AnalysisValidationError,
    ValidatedAnalysis,
    parse_and_validate_analysis,
)
from app.ecn.models import STAGED_PROMPT_VERSIONS, canonical_uuid
from app.observability import TokenUsage
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.records import (
    AnalysisCompletion,
    ClaimedJob,
    ProviderCallOutcome,
    ProviderOutcome,
    ProviderStageRecord,
    SafeJobError,
    ValidationOutcome,
)

from .blocking_io import run_blocking
from .staged_generation import run_staged_analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobProcessingResult:
    """Return the contained terminal result of processing one analysis job.

    Args:
        job_id: Immutable analysis job identifier.
        state: Whether processing skipped, completed, or failed the job.
        error_code: Stable safe failure code, otherwise ``None``.
    """

    job_id: str
    state: Literal["SKIPPED", "BLOCKED", "COMPLETED", "FAILED"]
    error_code: str | None = None


class Clock(Protocol):
    """Provide injectable timezone-aware UTC timestamps."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""


class SystemClock:
    """Return wall-clock UTC timestamps for production attempt persistence."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc)


def build_provider_outcome(
    result: ProviderCallResult,
    provider_outcome: ProviderCallOutcome,
    finished_at: datetime,
) -> ProviderOutcome:
    """Validate and bound provider metadata while preserving nullable usage.

    Args:
        result: Safe adapter result from one actual provider call.
        provider_outcome: Persistence lifecycle value derived from ``result``.
        finished_at: Timezone-aware completion timestamp.

    Returns:
        A persistence-safe provider outcome with nullable usage retained.

    Raises:
        ValueError: If status/outcome, usage availability, time, or bounds conflict.
    """
    if finished_at.tzinfo is None:
        raise ValueError("finished_at must be timezone-aware")
    expected = (
        ProviderCallOutcome.SUCCESS
        if result.status is ProviderCallStatus.RECEIVED
        else ProviderCallOutcome.ERROR
    )
    if provider_outcome is not expected:
        raise ValueError("provider status conflicts with persistence outcome")
    if result.usage.usage_available:
        usage: TokenUsage | None = result.usage
    else:
        counts = (
            result.usage.input_tokens,
            result.usage.cached_input_tokens,
            result.usage.cache_write_input_tokens,
            result.usage.input_total_tokens,
            result.usage.output_tokens,
            result.usage.total_tokens,
        )
        if any(value is not None for value in counts):
            raise ValueError("unavailable provider usage contains token counts")
        usage = None
    return ProviderOutcome(
        provider_outcome=provider_outcome,
        finished_at=finished_at,
        latency_ms=result.latency_ms,
        usage=usage,
        usage_available=result.usage.usage_available,
        model_deployment=result.model_deployment,
        error_code=None
        if provider_outcome is ProviderCallOutcome.SUCCESS
        else result.error_code,
    )


def build_analysis_completion(
    claimed: ClaimedJob,
    validated: ValidatedAnalysis,
    analysis_id: UUID,
    created_at: datetime | None = None,
    model_deployment: str | None = None,
    discovery: DiscoveryResult | None = None,
) -> AnalysisCompletion:
    """Build a persistence-safe immutable completion tied to the claimed attempt.

    Args:
        claimed: Claimed job and exact persisted attempt that produced the result.
        validated: Strict response tied to the full evidence digest.
        analysis_id: New application-generated immutable analysis UUID.
        created_at: Optional injected completion time; defaults to current UTC.
        model_deployment: Actual bounded deployment reported by the provider.
        discovery: Source-only RDF facts made available to final synthesis.

    Returns:
        A fully cross-checked ``AnalysisCompletion`` command.

    Raises:
        ValueError: If identifiers, evidence, model, versions, or timestamp conflict.
    """
    identifiers = (
        claimed.job_id,
        claimed.upload_id,
        claimed.ecn_row_id,
        claimed.attempt.attempt_id,
    )
    for identifier in identifiers:
        canonical_uuid(identifier)
    canonical_uuid(analysis_id)
    if claimed.ecn_row_id != claimed.evidence.ecn_row_id:
        raise ValueError("claimed ECN row does not match its evidence bundle")
    if validated.evidence_hash != claimed.evidence.evidence_hash:
        raise ValueError("validated evidence hash does not match the claimed bundle")
    if claimed.attempt.model_name != claimed.generation_config.model_name:
        raise ValueError("attempt model does not match the pinned generation model")
    bounded = {
        "prompt_version": (claimed.generation_config.prompt_version, 40),
        "output_schema_version": (
            claimed.generation_config.output_schema_version,
            40,
        ),
        "model_name": (claimed.attempt.model_name, 200),
        "deterministic_logic_version": (
            claimed.evidence.context.deterministic_logic_version,
            40,
        ),
        "rule_evaluator_version": (
            claimed.evidence.context.rule_evaluator_version,
            40,
        ),
    }
    if any(not value or len(value) > maximum for value, maximum in bounded.values()):
        raise ValueError("completion version or model field exceeds its bound")
    deployment = model_deployment or claimed.attempt.model_deployment
    if deployment is not None and (not deployment or len(deployment) > 200):
        raise ValueError("completion model deployment exceeds its bound")
    completed_at = created_at or datetime.now(timezone.utc)
    if completed_at.tzinfo is None:
        raise ValueError("completion timestamp must be timezone-aware")
    discovery = discovery or DiscoveryResult.failed("RDF_DISCOVERY_UNAVAILABLE")
    if len(discovery.discovery_hash) != 64:
        raise ValueError("discovery hash must be a SHA-256 digest")
    if discovery.error_code is not None and len(discovery.error_code) > 80:
        raise ValueError("discovery error code exceeds its persistence bound")
    return AnalysisCompletion(
        analysis_id=str(analysis_id),
        job_id=claimed.job_id,
        ecn_row_id=claimed.ecn_row_id,
        successful_attempt_id=claimed.attempt.attempt_id,
        response=validated.response,
        completeness=validated.completeness,
        evidence_hash=validated.evidence_hash,
        prompt_version=claimed.generation_config.prompt_version,
        model_name=claimed.attempt.model_name,
        model_deployment=deployment,
        deterministic_logic_version=(
            claimed.evidence.context.deterministic_logic_version
        ),
        rule_evaluator_version=claimed.evidence.context.rule_evaluator_version,
        output_schema_version=claimed.generation_config.output_schema_version,
        created_at=completed_at,
        discovery_status=discovery.status.value,
        discovery_error_code=discovery.error_code,
        discovery_hash=discovery.discovery_hash,
        discovered_evidence_json=discovery.evidence_json(),
    )


_SAFE_MESSAGES = {
    "PROMPT_PROJECTION_ERROR": "The evidence prompt could not be prepared",
    "PROVIDER_TIMEOUT": "The analysis provider timed out",
    "PROVIDER_INTERRUPTED": "The analysis was interrupted during shutdown",
    "PROVIDER_AUTH_FAILED": "The analysis provider credentials were rejected",
    "PROVIDER_FORBIDDEN": "The analysis provider denied access",
    "MODEL_NOT_AVAILABLE": "The configured analysis model is not available",
    "PROVIDER_RATE_LIMITED": "The analysis provider rate limit was reached",
    "PROVIDER_SERVER_ERROR": "The analysis provider returned a server error",
    "PROVIDER_RESPONSE_INVALID": "The analysis provider response was invalid",
    "ANALYSIS_VALIDATION_ERROR": "The provider response failed analysis validation",
    "DB_PERSISTENCE_ERROR": "The analysis result could not be persisted",
    "PROVIDER_ADAPTER_CONTRACT_ERROR": "The analysis provider adapter failed",
}


def _stage_call_number(stage: ProviderCallStage, prior_count: int = 0) -> int:
    """Return the next stable audit position for a generation model stage."""
    return {
        ProviderCallStage.DISCOVERY_1: 1,
        ProviderCallStage.DISCOVERY_2: 2,
        ProviderCallStage.PLANNER: 1,
        ProviderCallStage.FINAL: prior_count + 1,
    }[stage]


class AnalysisGenerationService:
    """Process one already-claimed job with isolated provider and HANA outcomes."""

    def __init__(
        self,
        analysis_repository: AnalysisRepository,
        provider: EcnAnalysisProvider,
        usage_emitter: Callable[..., None],
        clock: Clock,
        uuid_factory: Callable[[], UUID],
        snapshot_service=None,
        discovery_service=None,
        rdf_projection_service=None,
    ) -> None:
        """Store collaborators used to process one already-claimed job.

        Args:
            analysis_repository: Synchronous HANA attempt/job repository.
            provider: Long-lived asynchronous synthesis provider adapter.
            usage_emitter: Payload-free structured stdout telemetry function.
            clock: Injectable UTC clock.
            uuid_factory: Callable that creates immutable analysis UUIDs.
            snapshot_service: Optional relational snapshot context source.
            discovery_service: Optional bounded RDF discovery worker.
            rdf_projection_service: Optional currentness gate rechecked before final synthesis.
        """
        self._repository = analysis_repository
        self._provider = provider
        self._usage_emitter = usage_emitter
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._snapshots = snapshot_service
        self._discovery = discovery_service
        self._rdf = rdf_projection_service
        self._provider_capacity = None

    def set_provider_capacity(self, capacity: asyncio.Semaphore) -> None:
        """Share a process-wide provider-call limit with the assistant runtime."""
        self._provider_capacity = capacity

    async def run_claimed_job(self, claimed: ClaimedJob) -> JobProcessingResult:
        """Synthesize one explicitly queued ECN from complete counts and bounded evidence."""
        try:
            snapshot = (
                await run_blocking(self._snapshots.get, claimed.evidence.ecn_row_id)
                if self._snapshots
                else None
            )
            projection = build_synthesis_projection(claimed.evidence, snapshot)
        except Exception:  # noqa: BLE001 - contain snapshot/projection failure before billing.
            return await self._finish_without_provider_call(
                claimed, "PROMPT_PROJECTION_ERROR"
            )
        discovery = await self._run_discovery(claimed, projection)
        records = await self._persist_discovery_calls(claimed, discovery)
        if isinstance(records, JobProcessingResult):
            return records
        if self._rdf is not None:
            try:
                await run_blocking(self._rdf.require_current, claimed.ecn_row_id)
            except Exception:  # noqa: BLE001 - discard stale graph evidence before final call.
                discovery = DiscoveryResult.failed(
                    "RDF_PROJECTION_UNAVAILABLE", discovery.calls
                )
        try:
            projection = build_synthesis_projection(
                claimed.evidence, snapshot, discovery=discovery
            )
        except Exception:  # noqa: BLE001 - retain completed discovery call audit rows.
            if records:
                return await self._finish_after_calls(
                    claimed,
                    records,
                    "PROMPT_PROJECTION_ERROR",
                    ValidationOutcome.NOT_RUN,
                )
            return await self._finish_without_provider_call(
                claimed, "PROMPT_PROJECTION_ERROR"
            )
        if claimed.generation_config.prompt_version in STAGED_PROMPT_VERSIONS:
            return await self._run_staged_analysis(claimed, projection, snapshot or {}, records, discovery)
        final = await self._run_stage(
            claimed,
            projection,
            ProviderCallStage.FINAL,
            records,
        )
        if isinstance(final, JobProcessingResult):
            return final
        result, record = final
        if result.status is ProviderCallStatus.ERROR:
            return await self._finish_after_calls(
                claimed,
                (*records, record),
                result.error_code or "PROVIDER_RESPONSE_INVALID",
                ValidationOutcome.NOT_RUN,
            )
        return await self._validate_and_complete(
            claimed,
            projection,
            result,
            (*records, record),
            discovery,
        )

    async def _run_staged_analysis(
        self, claimed: ClaimedJob, projection: FinalEvidenceProjection,
        snapshot: dict, discovery_records: tuple[ProviderStageRecord, ...],
        discovery: DiscoveryResult,
    ) -> JobProcessingResult:
        """Run the parallel perspectives and sequential summary for a v3 job."""
        return await run_staged_analysis(
            self, claimed, projection, snapshot, discovery_records, discovery,
        )

    async def _run_discovery(
        self,
        claimed: ClaimedJob,
        projection: FinalEvidenceProjection,
    ) -> DiscoveryResult:
        """Run optional graph discovery without letting a graph gap block baseline synthesis."""
        if self._discovery is None:
            return DiscoveryResult.failed("RDF_DISCOVERY_UNAVAILABLE")
        try:
            return await self._discovery.discover(projection, claimed.generation_config)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - discovery remains non-authoritative.
            return DiscoveryResult.failed("RDF_DISCOVERY_FAILED")

    async def _persist_discovery_calls(
        self,
        claimed: ClaimedJob,
        discovery: DiscoveryResult,
    ) -> tuple[ProviderStageRecord, ...] | JobProcessingResult:
        """Persist and meter each attempted discovery call before final synthesis."""
        records: list[ProviderStageRecord] = []
        for call in discovery.calls:
            record = self._stage_record_from_result(
                claimed,
                call.result,
                call.stage,
                call.started_at,
                call.finished_at,
                call_number=call.sequence,
            )
            self._emit_usage(
                claimed,
                call.result.usage,
                call.result.model_deployment,
                call.result.latency_ms,
                "success"
                if call.result.status is ProviderCallStatus.RECEIVED
                else "error",
                call.stage,
                model_name=call.result.model_name,
            )
            try:
                await run_blocking(self._repository.record_provider_call, record)
            except Exception:  # noqa: BLE001 - provider audit is authoritative.
                return await self._finish_after_calls(
                    claimed,
                    (*records, record),
                    "DB_PERSISTENCE_ERROR",
                    ValidationOutcome.NOT_RUN,
                )
            records.append(record)
        return tuple(records)

    async def _finish_without_provider_call(
        self,
        claimed: ClaimedJob,
        error_code: str,
    ) -> JobProcessingResult:
        """Persist NOT_CALLED and fail a job without emitting token telemetry."""
        finished = self._clock.now()
        outcome = ProviderOutcome(
            provider_outcome=ProviderCallOutcome.NOT_CALLED,
            finished_at=finished,
            latency_ms=None,
            usage=None,
            usage_available=False,
            model_deployment=None,
            error_code=error_code,
        )
        try:
            await run_blocking(
                self._repository.record_provider_outcome,
                claimed.attempt.attempt_id,
                outcome,
            )
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._best_effort_database_failure(claimed, finished)
        await self._fail_job(
            claimed,
            error_code,
            ValidationOutcome.NOT_RUN,
            finished,
        )
        return JobProcessingResult(claimed.job_id, "FAILED", error_code)

    async def _run_stage(
        self,
        claimed: ClaimedJob,
        projection: object,
        stage: ProviderCallStage,
        prior_records: tuple[ProviderStageRecord, ...] = (),
    ) -> tuple[ProviderCallResult, ProviderStageRecord] | JobProcessingResult:
        """Execute, audit, and emit telemetry for exactly one bounded model stage."""
        started = self._clock.now()
        try:
            # The model adapter owns the per-request timeout and may make one
            # fallback request after the primary deployment times out.
            async with asyncio.timeout(
                claimed.generation_config.request_timeout_seconds * 2 + 5
            ):
                result = await self._provider.generate(
                    projection,  # type: ignore[arg-type]
                    claimed.generation_config,
                    stage,
                )
        except TimeoutError:
            record = self._failed_stage_record(
                claimed,
                stage,
                started,
                ProviderCallOutcome.TIMEOUT,
                "PROVIDER_TIMEOUT",
                call_number=len(prior_records) + 1,
            )
            return await self._persist_failed_stage(claimed, prior_records, record)
        except asyncio.CancelledError:
            record = self._failed_stage_record(
                claimed,
                stage,
                started,
                ProviderCallOutcome.INTERRUPTED,
                "PROVIDER_INTERRUPTED",
                call_number=len(prior_records) + 1,
            )
            await self._persist_failed_stage(claimed, prior_records, record)
            raise
        except Exception:  # noqa: BLE001 - adapter contract boundary is contained.
            record = self._failed_stage_record(
                claimed,
                stage,
                started,
                ProviderCallOutcome.ERROR,
                "PROVIDER_ADAPTER_CONTRACT_ERROR",
                call_number=len(prior_records) + 1,
            )
            return await self._persist_failed_stage(claimed, prior_records, record)

        finished = self._clock.now()
        record = self._stage_record_from_result(
            claimed,
            result,
            stage,
            started,
            finished,
            call_number=len(prior_records) + 1,
        )
        self._emit_usage(
            claimed,
            result.usage,
            result.model_deployment,
            result.latency_ms,
            "success" if result.status is ProviderCallStatus.RECEIVED else "error",
            stage,
            model_name=result.model_name,
        )
        try:
            await run_blocking(self._repository.record_provider_call, record)
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._finish_after_calls(
                claimed,
                (*prior_records, record),
                "DB_PERSISTENCE_ERROR",
                ValidationOutcome.NOT_RUN,
            )
        return result, record

    def _failed_stage_record(
        self,
        claimed: ClaimedJob,
        stage: ProviderCallStage,
        started: datetime,
        outcome: ProviderCallOutcome,
        error_code: str,
        call_number: int | None = None,
    ) -> ProviderStageRecord:
        """Build a safe child record when the adapter did not return a result."""
        finished = self._clock.now()
        latency_ms = max(0, round((finished - started).total_seconds() * 1_000))
        return ProviderStageRecord(
            call_id=str(uuid5(UUID(claimed.attempt.attempt_id), stage.value)),
            attempt_id=claimed.attempt.attempt_id,
            stage=stage.value,
            call_number=call_number or _stage_call_number(stage),
            started_at=started,
            finished_at=finished,
            outcome=outcome,
            http_status=None,
            provider_code=None,
            provider_request_id=None,
            retry_after_seconds=None,
            latency_ms=latency_ms,
            model_deployment=None,
            error_code=error_code,
            usage=TokenUsage(),
        )

    @staticmethod
    def _stage_record_from_result(
        claimed: ClaimedJob,
        result: ProviderCallResult,
        stage: ProviderCallStage | str,
        started: datetime,
        finished: datetime,
        call_number: int | None = None,
    ) -> ProviderStageRecord:
        """Convert one returned adapter result into an immutable child record."""
        if isinstance(stage, ProviderCallStage) and result.stage is not stage:
            raise ValueError("provider returned a result for a different stage")
        stage_label = stage.value if isinstance(stage, ProviderCallStage) else stage
        return ProviderStageRecord(
            call_id=str(uuid5(UUID(claimed.attempt.attempt_id), stage_label)),
            attempt_id=claimed.attempt.attempt_id,
            stage=stage_label,
            call_number=(
                call_number
                if call_number is not None
                else _stage_call_number(stage)
            ),
            started_at=started,
            finished_at=finished,
            outcome=(
                ProviderCallOutcome.SUCCESS
                if result.status is ProviderCallStatus.RECEIVED
                else ProviderCallOutcome.ERROR
            ),
            http_status=result.http_status,
            provider_code=result.provider_code,
            provider_request_id=result.provider_request_id,
            retry_after_seconds=result.retry_after_seconds,
            latency_ms=result.latency_ms,
            model_deployment=result.model_deployment,
            error_code=result.error_code,
            usage=result.usage,
        )

    async def _persist_failed_stage(
        self,
        claimed: ClaimedJob,
        prior_records: tuple[ProviderStageRecord, ...],
        record: ProviderStageRecord,
    ) -> JobProcessingResult:
        """Persist one timeout/interruption/adapter error and close the parent attempt."""
        self._emit_usage(
            claimed,
            record.usage,
            record.model_deployment,
            record.latency_ms,
            "error",
            ProviderCallStage(record.stage),
        )
        try:
            await run_blocking(self._repository.record_provider_call, record)
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._best_effort_database_failure(claimed, record.finished_at)
        return await self._finish_after_calls(
            claimed,
            (*prior_records, record),
            record.error_code or "PROVIDER_RESPONSE_INVALID",
            ValidationOutcome.NOT_RUN,
        )

    async def _finish_after_calls(
        self,
        claimed: ClaimedJob,
        records: tuple[ProviderStageRecord, ...],
        error_code: str,
        validation_outcome: ValidationOutcome,
    ) -> JobProcessingResult:
        """Aggregate completed stages, finalize the attempt, and fail the job safely."""
        outcome = self._aggregate_outcome(records)
        try:
            await run_blocking(
                self._repository.record_provider_outcome,
                claimed.attempt.attempt_id,
                outcome,
            )
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._best_effort_database_failure(
                claimed, outcome.finished_at
            )
        await self._fail_job(
            claimed, error_code, validation_outcome, outcome.finished_at
        )
        return JobProcessingResult(claimed.job_id, "FAILED", error_code)

    async def _validate_and_complete(
        self,
        claimed: ClaimedJob,
        projection: FinalEvidenceProjection,
        result: ProviderCallResult,
        records: tuple[ProviderStageRecord, ...],
        discovery: DiscoveryResult,
    ) -> JobProcessingResult:
        """Finalize aggregate telemetry, validate the final response, and complete."""
        outcome = self._aggregate_outcome(records)
        try:
            await run_blocking(
                self._repository.record_provider_outcome,
                claimed.attempt.attempt_id,
                outcome,
            )
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._best_effort_database_failure(
                claimed, outcome.finished_at
            )
        try:
            validated = parse_and_validate_analysis(
                result,
                projection,
                claimed.evidence.evidence_hash,
            )
        except AnalysisValidationError as exc:
            logger.warning("Final analysis validation failed for job %s: %s", claimed.job_id, exc)
            await self._fail_job(
                claimed,
                "ANALYSIS_VALIDATION_ERROR",
                ValidationOutcome.INVALID,
                outcome.finished_at,
            )
            return JobProcessingResult(
                claimed.job_id, "FAILED", "ANALYSIS_VALIDATION_ERROR"
            )
        completion = build_analysis_completion(
            claimed,
            validated,
            self._uuid_factory(),
            created_at=outcome.finished_at,
            model_deployment=result.model_deployment,
            discovery=discovery,
        )
        try:
            await run_blocking(
                self._repository.complete_job,
                claimed.job_id,
                completion,
            )
        except Exception:  # noqa: BLE001 - translate persistence failures safely.
            return await self._best_effort_database_failure(
                claimed, outcome.finished_at
            )
        return JobProcessingResult(claimed.job_id, "COMPLETED")

    @staticmethod
    def _aggregate_outcome(records: tuple[ProviderStageRecord, ...]) -> ProviderOutcome:
        """Aggregate per-stage usage and lifecycle into the existing attempt row."""
        if not records:
            raise ValueError("at least one provider stage is required")
        final_record = next(
            (record for record in reversed(records) if record.stage in {"FINAL", "SUMMARY"}),
            None,
        )
        outcome = (
            final_record.outcome
            if final_record is not None
            else ProviderCallOutcome.SUCCESS
        )
        if final_record is None:
            for candidate in (
                ProviderCallOutcome.INTERRUPTED,
                ProviderCallOutcome.TIMEOUT,
                ProviderCallOutcome.ERROR,
            ):
                if any(record.outcome is candidate for record in records):
                    outcome = candidate
                    break

        def total(attribute: str) -> int | None:
            values = [getattr(record.usage, attribute) for record in records]
            known = [value for value in values if value is not None]
            return sum(known) if known else None

        usage_available = any(record.usage.usage_available for record in records)
        usage = (
            TokenUsage(
                input_tokens=total("input_tokens"),
                cached_input_tokens=total("cached_input_tokens"),
                output_tokens=total("output_tokens"),
                total_tokens=total("total_tokens"),
                cache_write_input_tokens=total("cache_write_input_tokens"),
                input_total_tokens=total("input_total_tokens"),
                usage_available=True,
            )
            if usage_available
            else None
        )
        return ProviderOutcome(
            provider_outcome=outcome,
            finished_at=max(record.finished_at for record in records),
            latency_ms=sum(record.latency_ms for record in records),
            usage=usage,
            usage_available=usage_available,
            model_deployment=next(
                (
                    record.model_deployment
                    for record in reversed(records)
                    if record.model_deployment
                ),
                None,
            ),
            error_code=(
                final_record.error_code
                if final_record is not None
                else next(
                    (
                        record.error_code
                        for record in reversed(records)
                        if record.error_code
                    ),
                    None,
                )
            ),
        )

    async def _fail_job(
        self,
        claimed: ClaimedJob,
        error_code: str,
        validation_outcome: ValidationOutcome,
        finished_at: datetime,
    ) -> None:
        """Persist one safe terminal failure after provider outcome storage."""
        error = SafeJobError(
            code=error_code,
            message=_SAFE_MESSAGES[error_code],
            validation_outcome=validation_outcome,
        )
        try:
            await run_blocking(
                self._repository.fail_job,
                claimed.job_id,
                claimed.attempt.attempt_id,
                error,
                finished_at,
            )
        except Exception:  # noqa: BLE001 - terminal conflict stays contained.
            return

    async def _best_effort_database_failure(
        self,
        claimed: ClaimedJob,
        finished_at: datetime,
    ) -> JobProcessingResult:
        """Try to expose a safe DB failure without masking the contained result."""
        await self._fail_job(
            claimed,
            "DB_PERSISTENCE_ERROR",
            ValidationOutcome.NOT_RUN,
            finished_at,
        )
        return JobProcessingResult(claimed.job_id, "FAILED", "DB_PERSISTENCE_ERROR")

    def _emit_usage(
        self,
        claimed: ClaimedJob,
        usage: TokenUsage,
        model_deployment: str | None,
        latency_ms: int,
        outcome: str,
        stage: ProviderCallStage | str,
        model_name: str | None = None,
    ) -> None:
        """Best-effort emit payload-free telemetry for one provider stage.

        Telemetry is deliberately outside the authoritative persistence path.
        A logging sink contract error must never strand a claimed job in the
        ``RUNNING`` state or discard provider diagnostics.

        Args:
            claimed: Job and immutable attempt metadata for correlation.
            usage: Availability-preserving usage from the model response.
            model_deployment: Actual provider deployment label, when available.
            latency_ms: Elapsed provider-stage latency.
            outcome: Safe outcome category for telemetry.
            stage: Provider stage label.
            model_name: Deployment that returned or last attempted the response.
        """
        try:
            self._usage_emitter(
                route="ecn-impact-analysis",
                method="INTERNAL",
                actor_type="batch",
                provider=claimed.attempt.provider,
                model=model_name or claimed.attempt.model_name,
                model_deployment=model_deployment,
                llm_endpoint=claimed.attempt.llm_endpoint,
                usage_available=usage.usage_available,
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                input_total_tokens=usage.input_total_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                outcome=outcome,
                latency_ms=latency_ms,
                correlation_id=claimed.attempt.correlation_id,
                upload_id=claimed.upload_id,
                job_id=claimed.job_id,
                attempt_id=claimed.attempt.attempt_id,
                stage=stage.value if isinstance(stage, ProviderCallStage) else stage,
            )
        except Exception:  # noqa: BLE001 - observability cannot own job state.
            return

    async def close(self) -> None:
        """Close the provider client after all runtime tasks have stopped."""
        if self._discovery is not None:
            await self._discovery.close()
        await self._provider.close()
