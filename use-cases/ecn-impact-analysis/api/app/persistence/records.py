"""Immutable persistence command and result records for the ECN impact pipeline.

Every record defined here is a frozen dataclass that represents a unit of work
crossing the repository boundary — either a command (what the caller wants to
store) or a result (what the repository returns after a mutation or query).

These records are the only types that may cross the HTTP/repository boundary in
both directions.  No SQLAlchemy model, no ORM row, and no raw dict escapes the
repository layer.

Imports from ``app.ecn.models`` (Task 1) and ``app.ecn.api_models`` (Task 3)
provide the shared enumerations, domain records, and API view models.
``app.observability.llm_usage_logging.TokenUsage`` is imported with a
compatibility contract: the current four-field class is never modified here,
and any new attributes are accessed via ``getattr`` in the codec layer so that
this module remains forward-compatible with future richer usage shapes.

Standard library and first-party only; no SQLAlchemy imports here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from app.ecn.analysis_models import AnalysisCompleteness, AnalysisResponse
from app.ecn.api_models import AnalysisJobView, ImpactCounts, StoredAnalysisView
from app.ecn.models import (
    ArtifactKind,
    BomRow,
    ChangeCandidate,
    EvidenceBundle,
    JsonValue,
    NormalizedEcn,
    SourceArtifact,
    canonical_uuid,
)
from app.observability.llm_usage_logging import TokenUsage
from app.ecn.reference_data.models import ReferenceDatasetPin


# ---------------------------------------------------------------------------
# Upload and publication records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UploadRecord:
    """Represent one persisted upload batch as a read-only snapshot.

    Mirrors the columns in ``ECN_IMPACT_UPLOAD`` that are relevant to the
    read path; the ``CREATED_BY``, ``PARSER_VERSION``, and logic-version
    columns are omitted because they are not required for any read projection.

    Args:
        upload_id: Canonical UUID of the upload batch.
        request_id: Canonical UUID of the idempotency request.
        ecn_artifact_id: Canonical UUID of the ECN JSON artifact.
        bom_artifact_id: Canonical UUID of the BOM workbook artifact.
        rule_artifact_id: Canonical UUID of the conditional-rule workbook;
            ``None`` when not supplied.
        bom_reused: ``True`` when the BOM artifact already existed.
        imported_ecn_count: Total ECN rows imported from this upload.
        publication_order: Monotonically increasing sequence for this batch.
        created_at: Timezone-aware timestamp when the batch was stored.
    """

    upload_id: str
    request_id: str
    ecn_artifact_id: str
    bom_artifact_id: str
    rule_artifact_id: str | None
    bom_reused: bool
    imported_ecn_count: int
    publication_order: int
    created_at: datetime


@dataclass(frozen=True)
class UploadPublishResult:
    """Return the created-or-retrieved upload record and an idempotency flag.

    ``created=True`` means a new upload was stored; ``created=False`` means
    the idempotency request ID matched an existing record that was returned
    unchanged.

    Args:
        upload: The upload batch record, whether newly created or existing.
        created: ``True`` when the record was inserted in this call.
        projection_ecn_ids: Exact ECN IDs whose durable ``PENDING`` projection
            rows committed in this new publication; empty for an idempotent
            replay or race-loss result.
    """

    upload: UploadRecord
    created: bool
    projection_ecn_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EcnPublication:
    """Carry one normalized ECN and all its derived evidence for a write operation.

    Every field is produced before the repository is called; the repository
    stores them atomically and never re-derives them.

    Args:
        ecn: The fully normalized ECN record.
        change_candidates: Derived change candidates for this ECN.
        evidence: The deterministic evidence bundle for this ECN.
        counts: Pre-computed deterministic impact and evidence counts.
        job_id: Canonical UUID of the analysis job to create.
        prompt_version: Version tag of the LLM prompt template to use.
        output_schema_version: Version tag of the structured output schema.
        model_name: Gen AI Hub model name to pin to this job.
        model_config_json: Model configuration serialized as a JSON string.
    """

    ecn: NormalizedEcn
    change_candidates: tuple[ChangeCandidate, ...]
    evidence: EvidenceBundle
    counts: ImpactCounts
    job_id: str
    prompt_version: str
    output_schema_version: str
    model_name: str
    model_config_json: str


@dataclass(frozen=True)
class UploadPublication:
    """Carry the full input to one atomic upload write.

    Every sub-record is pre-computed and validated by the caller; the
    repository writes them in dependency order without re-deriving anything.

    Args:
        upload_id: Canonical UUID of the new upload batch.
        request_id: Canonical UUID of the idempotency key.
        ecn_artifact: Source artifact for the uploaded ECN JSON.
        bom_artifact: Source artifact for the uploaded BOM workbook.
        new_bom_rows: BOM rows to insert (empty when the BOM is reused).
        rule_artifact: Conditional-rule workbook artifact; ``None`` when not
            supplied.
        bom_reused: ``True`` when the BOM artifact already existed.
        ecns: Tuple of ECN publications to store, one per normalized ECN.
        actor_id: Opaque hashed actor identifier; ``None`` for batch uploads.
        parser_version: Version tag of the parser that produced the ECNs.
        deterministic_logic_version: Version tag of the evidence engine.
        rule_evaluator_version: Version tag of the rule evaluator.
        created_at: Timezone-aware timestamp for this publication.
    """

    upload_id: str
    request_id: str
    ecn_artifact: SourceArtifact
    bom_artifact: SourceArtifact
    new_bom_rows: tuple[BomRow, ...]
    rule_artifact: SourceArtifact | None
    bom_reused: bool
    ecns: tuple[EcnPublication, ...]
    actor_id: str | None
    parser_version: str
    deterministic_logic_version: str
    rule_evaluator_version: str
    created_at: datetime
    dataset_links: tuple[ReferenceDatasetPin, ...] = ()
    ecn_reused: bool = False


# ---------------------------------------------------------------------------
# ECN and evidence read records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredEcnRecord:
    """Hold one persisted ECN row with its derived fields and evidence hash.

    This record deliberately does not carry the full ``NormalizedEcn`` affected/
    resulting object graph because ``ECN_IMPACT_ECN`` stores only the display
    fields, source JSON, candidates, counts, and evidence hash.  The full
    evidence bundle is reconstructed separately from ``EVIDENCE_JSON``.

    Args:
        ecn_row_id: Canonical UUID of this ECN row.
        upload_id: Canonical UUID of the owning upload batch.
        source_ordinal: Zero-based index of this ECN in the source file.
        ecn_number: Source-system ECN identifier; ``None`` when absent.
        title: ECN title; ``None`` when absent.
        description_summary: Short description summary; ``None`` when absent.
        description_text: Full description text; ``None`` when absent.
        counts: Deterministic impact and evidence counts.
        source_json: Original raw JSON object for traceability.
        change_candidates: Tuple of derived change candidates.
        evidence_hash: 64-char hex digest of the evidence bundle.
        parser_version: Version tag of the parser that produced this ECN.
        deterministic_logic_version: Version tag of the evidence engine.
        rule_evaluator_version: Version tag of the rule evaluator.
        created_at: Timezone-aware timestamp when this ECN was stored.
        source_status: Source-system release or approval state; ``None`` when
            absent.
        requestor_name: Name in the source requestor field; ``None`` when absent.
        affected_products: Declared affected product/model labels from the source.
        resolution_date: Source-system ResolutionDate (the date the ECN was
            released), kept as the raw ISO-8601 string exactly as it appears in
            ``source_json``; ``None`` when the source carries no value. Parsed to
            a datetime only at the API boundary.
    """

    ecn_row_id: str
    upload_id: str
    source_ordinal: int
    ecn_number: str | None
    title: str | None
    description_summary: str | None
    description_text: str | None
    counts: ImpactCounts
    source_json: dict[str, JsonValue]
    change_candidates: tuple[ChangeCandidate, ...]
    evidence_hash: str
    parser_version: str
    deterministic_logic_version: str
    rule_evaluator_version: str
    created_at: datetime
    source_status: str | None = None
    requestor_name: str | None = None
    affected_products: tuple[str, ...] = ()
    resolution_date: str | None = None


@dataclass(frozen=True)
class ArtifactMetadataRecord:
    """Carry immutable artifact identity without hydrating stored content.

    Args:
        artifact_id: Canonical artifact UUID.
        artifact_kind: Source artifact type.
        file_name: Original uploaded file name.
        sha256: Immutable content digest.
    """

    artifact_id: str
    artifact_kind: ArtifactKind
    file_name: str
    sha256: str


@dataclass(frozen=True)
class EcnDetailRecord:
    """Aggregate the full ECN detail page data across multiple tables.

    Combines the normalized ECN fields, upload batch, artifact provenance,
    evidence bundle, current analysis job state, and optional stored analysis
    into a single read result so that the detail handler avoids N+1 reads.

    Args:
        ecn: The stored ECN row.
        upload: The owning upload batch.
        ecn_artifact: Provenance of the ECN JSON artifact.
        bom_artifact: Provenance of the BOM workbook artifact.
        rule_artifact: Provenance of the rule workbook; ``None`` when absent.
        job: Current analysis job state.
        analysis: Stored validated analysis; ``None`` when not yet completed.
    """

    ecn: StoredEcnRecord
    upload: UploadRecord
    ecn_artifact: ArtifactMetadataRecord | SourceArtifact
    bom_artifact: ArtifactMetadataRecord | SourceArtifact
    rule_artifact: ArtifactMetadataRecord | SourceArtifact | None
    job: AnalysisJobView
    analysis: StoredAnalysisView | None


# ---------------------------------------------------------------------------
# Job claiming records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptSeed:
    """Carry caller-supplied attempt identity and provider context into a claim.

    ``AttemptSeed`` is the input to ``claim_job``; the repository pins the
    actual ``model_name`` from the job configuration and returns an
    ``AttemptStart``.  The seed deliberately omits ``model_name`` so that the
    worker cannot override the pinned model.

    Args:
        attempt_id: Canonical UUID for this attempt (caller-generated).
        correlation_id: Logging correlation identifier (≤100 chars).
        started_at: Timezone-aware start timestamp.
        provider: Provider label (≤100 chars).
        model_deployment: Gen AI Hub deployment name; ``None`` when not pinned.
        llm_endpoint: Provider endpoint family (≤100 chars).
    """

    attempt_id: str
    correlation_id: str
    started_at: datetime
    provider: str
    model_deployment: str | None
    llm_endpoint: str


@dataclass(frozen=True)
class AttemptStart:
    """Represent the exact STARTED attempt persisted from a seed and pinned model.

    Returned by the repository after the attempt row is inserted.  Differs
    from ``AttemptSeed`` in having ``model_name`` filled in from the job's
    pinned configuration.

    Args:
        attempt_id: Canonical UUID for this attempt.
        correlation_id: Logging correlation identifier.
        started_at: Timezone-aware start timestamp.
        provider: Provider label.
        model_name: Gen AI Hub model name from the pinned job config.
        model_deployment: Gen AI Hub deployment name; ``None`` when not pinned.
        llm_endpoint: Provider endpoint family.
    """

    attempt_id: str
    correlation_id: str
    started_at: datetime
    provider: str
    model_name: str
    model_deployment: str | None
    llm_endpoint: str


@dataclass(frozen=True)
class PinnedGenerationConfig:
    """Hold the immutable generation configuration pinned to a job at creation time.

    The ``strict_structured_output`` field is always ``True`` in this phase;
    the ``Literal[True]`` annotation enforces this at the type level.

    Args:
        prompt_version: Version tag of the LLM prompt template (≤40 chars).
        output_schema_version: Version tag of the structured output schema
            (≤40 chars).
        model_name: Gen AI Hub model name (≤200 chars).
        request_timeout_seconds: Maximum time to wait for a provider response.
        max_prompt_evidence_items: Maximum evidence items to include in prompt.
        max_prompt_characters: Maximum total prompt character count.
        strict_structured_output: Always ``True``; structured output is required.
    """

    prompt_version: str
    output_schema_version: str
    model_name: str
    request_timeout_seconds: int
    max_prompt_evidence_items: int
    max_prompt_characters: int
    strict_structured_output: Literal[True] = True


@dataclass(frozen=True)
class ClaimedJob:
    """Return claimed evidence, pinned config, and the exact persisted attempt.

    Args:
        job_id: Canonical UUID of the claimed analysis job.
        upload_id: Canonical UUID of the owning upload.
        ecn_row_id: Canonical UUID of the ECN to analyse.
        attempt_number: 1-based attempt index for this claim.
        attempt: The persisted attempt record, including the pinned model name.
        evidence: The full deterministic evidence bundle for this ECN.
        generation_config: The immutable generation configuration for this job.
    """

    job_id: str
    upload_id: str
    ecn_row_id: str
    attempt_number: int
    attempt: AttemptStart
    evidence: EvidenceBundle
    generation_config: PinnedGenerationConfig


class ClaimJobStatus(StrEnum):
    """Describe the mutually exclusive result of trying to claim one job.

    ``CLAIMED`` requires a non-null ``claimed_job``; every other status
    requires a null ``claimed_job``.  The runtime maps non-claimed statuses
    to the stable error codes ``ANALYSIS_JOB_NOT_FOUND``,
    ``ANALYSIS_JOB_NOT_QUEUED``, and ``PINNED_CONFIG_INVALID``.
    """

    CLAIMED = "CLAIMED"
    NOT_FOUND = "NOT_FOUND"
    NOT_QUEUED = "NOT_QUEUED"
    PINNED_CONFIG_INVALID = "PINNED_CONFIG_INVALID"


@dataclass(frozen=True)
class ClaimJobResult:
    """Return one explicit claim outcome without overloading ``None``.

    ``status`` is the discriminator: ``CLAIMED`` requires a non-null
    ``claimed_job``; every other status requires ``claimed_job is None``.

    Args:
        status: The mutually exclusive claim outcome.
        claimed_job: The claimed job details; ``None`` when not claimed.

    Raises:
        ValueError: If ``CLAIMED`` status has a null ``claimed_job``, or if
            any non-CLAIMED status has a non-null ``claimed_job``.
    """

    status: ClaimJobStatus
    claimed_job: ClaimedJob | None

    def __post_init__(self) -> None:
        """Enforce the discriminator invariant for claimed vs non-claimed."""
        if self.status is ClaimJobStatus.CLAIMED:
            if self.claimed_job is None:
                raise ValueError("a claimed result requires the claimed job")
        elif self.claimed_job is not None:
            raise ValueError("a non-claimed result cannot contain a claimed job")


# ---------------------------------------------------------------------------
# Retry queue records
# ---------------------------------------------------------------------------


class RetryQueueStatus(StrEnum):
    """Describe the mutually exclusive result of requesting a manual retry.

    ``QUEUED`` requires a canonical non-null ``job_id``.  ``ECN_NOT_FOUND``
    requires a null ``job_id``.  ``NOT_RETRYABLE`` may retain the existing
    ``job_id`` when one is known.
    """

    QUEUED = "QUEUED"
    ECN_NOT_FOUND = "ECN_NOT_FOUND"
    NOT_RETRYABLE = "NOT_RETRYABLE"


@dataclass(frozen=True)
class RetryQueueResult:
    """Return one explicit retry outcome for later HTTP 404/409 mapping.

    Args:
        status: The mutually exclusive retry outcome.
        job_id: Canonical UUID of the (re-)queued job; may be ``None`` for
            ``ECN_NOT_FOUND``.

    Raises:
        ValueError: If ``job_id`` is set but not a canonical UUID, if
            ``QUEUED`` status lacks a ``job_id``, or if ``ECN_NOT_FOUND``
            status has a non-null ``job_id``.
    """

    status: RetryQueueStatus
    job_id: str | None

    def __post_init__(self) -> None:
        """Validate UUID format and enforce discriminator invariants."""
        if self.job_id is not None:
            canonical_uuid(self.job_id)
        if self.status is RetryQueueStatus.QUEUED and self.job_id is None:
            raise ValueError("a queued retry result requires a job ID")
        if self.status is RetryQueueStatus.ECN_NOT_FOUND and self.job_id is not None:
            raise ValueError("a missing ECN cannot contain a job ID")


# ---------------------------------------------------------------------------
# Provider outcome records
# ---------------------------------------------------------------------------


class ProviderCallOutcome(StrEnum):
    """Separate provider lifecycle states, including a pre-call terminal failure.

    ``NOT_CALLED`` means the attempt was terminated before any provider
    call was made.  All other values correspond to states after a call was
    initiated.
    """

    NOT_CALLED = "NOT_CALLED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    INTERRUPTED = "INTERRUPTED"


class ValidationOutcome(StrEnum):
    """Classify whether a provider response was validated against the schema.

    ``NOT_RUN`` means validation was never attempted (e.g. the provider call
    failed before a response was available).
    """

    NOT_RUN = "NOT_RUN"
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class AttemptUsageColumns:
    """Hold the exact token-count values to store in ``ECN_IMPACT_LLM_ATTEMPT``.

    Every integer field is nullable because usage may be genuinely absent
    (``usage_available=False``) or because certain future attributes have not
    yet been surfaced by the provider.  ``usage_available`` is stored
    separately so that zero counts can be distinguished from unavailable counts.

    Args:
        input_tokens: Prompt/input tokens; ``None`` when unavailable.
        cache_read_input_tokens: Cache-read tokens; ``None`` when unavailable.
        cache_write_input_tokens: Cache-write tokens; ``None`` when the
            provider does not yet expose this field.
        input_total_tokens: Sum of raw + cached input; ``None`` when the
            provider does not yet expose this field.
        output_tokens: Output tokens; ``None`` when unavailable.
        total_tokens: Grand total tokens; ``None`` when unavailable.
        usage_available: ``True`` when the provider returned usage data.
    """

    input_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_input_tokens: int | None
    input_total_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    usage_available: bool


@dataclass(frozen=True)
class ProviderOutcome:
    """Carry the complete post-call provider result for one attempt.

    Args:
        provider_outcome: The lifecycle state after the call resolved.
        finished_at: Timezone-aware timestamp when the call finished.
        latency_ms: Wall-clock latency in milliseconds; ``None`` when the
            call did not complete normally.
        usage: Normalized token usage object; ``None`` when not available.
        usage_available: ``True`` when ``usage`` carries real data.
        model_deployment: The deployment that actually served the call;
            may differ from the requested deployment.  ``None`` when unknown.
        error_code: Stable machine-readable error code; ``None`` on success.
    """

    provider_outcome: ProviderCallOutcome
    finished_at: datetime
    latency_ms: int | None
    usage: TokenUsage | None
    usage_available: bool
    model_deployment: str | None
    error_code: str | None


@dataclass(frozen=True)
class ProviderStageRecord:
    """Carry one discovery, legacy planner, or final provider-call child record.

    Args:
        call_id: Deterministic UUID for this attempt and stage.
        attempt_id: Owning aggregate attempt UUID.
        stage: Sequential ``DISCOVERY_N``, legacy planner, or final label.
        call_number: Stable one-based position within the generation attempt.
        started_at: Timezone-aware stage start.
        finished_at: Timezone-aware stage finish.
        outcome: Terminal provider lifecycle result.
        http_status: Safe HTTP status when available.
        provider_code: Bounded provider error code when available.
        provider_request_id: Bounded provider request identifier when available.
        retry_after_seconds: Safe retry hint when available.
        latency_ms: Nonnegative stage latency.
        model_deployment: Actual model or deployment label when available.
        error_code: Stable application error category when failed.
        usage: Normalized per-stage token usage.
    """

    call_id: str
    attempt_id: str
    stage: str
    call_number: int
    started_at: datetime
    finished_at: datetime
    outcome: ProviderCallOutcome
    http_status: int | None
    provider_code: str | None
    provider_request_id: str | None
    retry_after_seconds: int | None
    latency_ms: int
    model_deployment: str | None
    error_code: str | None
    usage: TokenUsage


# ---------------------------------------------------------------------------
# Error and completion records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafeJobError:
    """Hold a safe error classification for a failed analysis job.

    Args:
        code: Stable machine-readable error code (≤80 chars).
        message: Safe human-readable message for API responses (≤2000 chars).
        validation_outcome: The outcome of any schema validation that was
            attempted before the error was raised.
    """

    code: str
    message: str
    validation_outcome: ValidationOutcome


@dataclass(frozen=True)
class AnalysisCompletion:
    """Carry the full validated analysis result for the completion write.

    ``job_id`` is duplicated here even though the caller passes it to
    ``complete_job``.  The duplicate is cross-checked before insert to prevent
    a completion from silently borrowing another job's row.

    Args:
        analysis_id: Canonical UUID of the new analysis record.
        job_id: Canonical UUID of the job this analysis completes.
        ecn_row_id: Canonical UUID of the ECN this analysis covers.
        successful_attempt_id: Canonical UUID of the attempt that produced
            the accepted response.
        response: The fully validated ``AnalysisResponse``.
        completeness: The derived completeness classification.
        evidence_hash: 64-char hex digest of the evidence bundle used.
        prompt_version: Version tag of the prompt template (≤40 chars).
        model_name: Gen AI Hub model name (≤200 chars).
        model_deployment: Deployment name; ``None`` when not recorded.
        deterministic_logic_version: Version tag of the evidence engine.
        rule_evaluator_version: Version tag of the rule evaluator.
        output_schema_version: Version tag of the output schema.
        created_at: Timezone-aware timestamp when the analysis was stored.
        discovery_status: ``COMPLETE`` only when all graph facts were captured.
        discovery_error_code: Safe graph discovery gap classification, if any.
        discovery_hash: SHA-256 of exactly the discovered facts retained.
        discovered_evidence_json: Canonical source-only graph evidence array.
    """

    analysis_id: str
    job_id: str
    ecn_row_id: str
    successful_attempt_id: str
    response: AnalysisResponse
    completeness: AnalysisCompleteness
    evidence_hash: str
    prompt_version: str
    model_name: str
    model_deployment: str | None
    deterministic_logic_version: str
    rule_evaluator_version: str
    output_schema_version: str
    created_at: datetime
    discovery_status: Literal["COMPLETE", "FAILED"] = "FAILED"
    discovery_error_code: str | None = "RDF_DISCOVERY_UNAVAILABLE"
    discovery_hash: str = "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    discovered_evidence_json: str = "[]"


# ---------------------------------------------------------------------------
# History deletion counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryDeletionCounts:
    """Report exact dependency-ordered deletion row counts after a history reset.

    Field order matches the FK deletion cascade: attempts first, source
    artifacts last.  All counts are nonnegative integers.

    Args:
        attempts: Number of ``ECN_IMPACT_LLM_ATTEMPT`` rows deleted.
        analyses: Number of ``ECN_IMPACT_ANALYSIS`` rows deleted.
        jobs: Number of ``ECN_IMPACT_ANALYSIS_JOB`` rows deleted.
        ecns: Number of ``ECN_IMPACT_ECN`` rows deleted.
        uploads: Number of ``ECN_IMPACT_UPLOAD`` rows deleted.
        bom_rows: Number of ``ECN_IMPACT_BOM_ROW`` rows deleted.
        source_artifacts: Number of ``ECN_IMPACT_SOURCE_ARTIFACT`` rows deleted.
    """

    attempts: int
    analyses: int
    jobs: int
    ecns: int
    uploads: int
    bom_rows: int
    source_artifacts: int


@dataclass(frozen=True)
class EcnDeletionCounts:
    """Report exact row counts after deleting selected ECN line items.

    A line item is every assessment version of one original ECN source
    position.  Reference data (BOM, material master, purchase info, stock,
    rules) is never part of this deletion.

    Args:
        ecn_row_ids: Every deleted ``ECN_ROW_ID`` (selected rows plus their
            assessment versions), used for knowledge-graph cleanup.
        ecns: Number of ``ECN_IMPACT_ECN`` rows deleted.
        analyses: Number of ``ECN_IMPACT_ANALYSIS`` rows deleted.
        attempts: Number of ``ECN_IMPACT_LLM_ATTEMPT`` rows deleted.
        jobs: Number of ``ECN_IMPACT_ANALYSIS_JOB`` rows deleted.
        reviews: Number of ``ECN_IMPACT_REVIEW`` rows deleted.
        uploads: Number of ``ECN_IMPACT_UPLOAD`` rows left without any ECN
            and therefore deleted.
        source_artifacts: Number of ``ECN_JSON`` artifacts left without any
            upload and therefore deleted.
        feasibility_assessments: Number of
            ``ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT`` rows deleted.
            Defaults to ``0`` so positional construction stays compatible.
    """

    ecn_row_ids: tuple[str, ...]
    ecns: int
    analyses: int
    attempts: int
    jobs: int
    reviews: int
    uploads: int
    source_artifacts: int
    feasibility_assessments: int = 0


# ---------------------------------------------------------------------------
# Analysis progress response
# ---------------------------------------------------------------------------


class AnalysisProgressResponse(BaseModel):
    """Aggregate job-state counts for all analysis jobs, optionally scoped to one upload.

    The six state counts are mutually exclusive and exhaustive.  ``unfinished_count``
    and ``total_count`` are derived values validated to be consistent with the
    six state counts; the model validator enforces this invariant so that
    callers cannot accidentally pass inconsistent derived values.

    Args:
        queued_count: Number of jobs waiting to be claimed.
        blocked_count: Number of jobs held for provider availability.
        running_count: Number of jobs currently being processed.
        completed_count: Number of jobs completed with a ``COMPLETED`` analysis.
        completed_with_gaps_count: Number of jobs completed with a
            ``COMPLETED_WITH_GAPS`` analysis (at least one section is
            ``INSUFFICIENT_EVIDENCE``).
        failed_count: Number of jobs that have failed.
        unfinished_count: Derived = ``queued_count + blocked_count + running_count``.
        total_count: Derived = sum of all six state counts.
    """

    model_config = ConfigDict(frozen=True)

    ready_count: int = 0
    queued_count: int
    blocked_count: int = 0
    running_count: int
    completed_count: int
    completed_with_gaps_count: int
    failed_count: int
    unfinished_count: int
    total_count: int

    @model_validator(mode="after")
    def _check_derived_counts(self) -> "AnalysisProgressResponse":
        """Validate that unfinished_count and total_count match the six state counts.

        Returns:
            Self after validation.

        Raises:
            ValueError: When either derived count is inconsistent.
        """
        expected_unfinished = (
            self.queued_count + self.blocked_count + self.running_count
        )
        expected_total = (
            self.queued_count
            + self.blocked_count
            + self.running_count
            + self.completed_count
            + self.completed_with_gaps_count
            + self.failed_count
            + self.ready_count
        )
        if self.unfinished_count != expected_unfinished:
            raise ValueError(
                f"unfinished_count {self.unfinished_count} must equal "
                f"queued_count + blocked_count + running_count = {expected_unfinished}"
            )
        if self.total_count != expected_total:
            raise ValueError(
                f"total_count {self.total_count} must equal "
                f"sum of all six state counts = {expected_total}"
            )
        return self
