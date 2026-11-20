"""Complete lifecycle management for ECN analysis jobs and provider attempts.

This module implements ``AnalysisRepository``, which owns all write paths for
the LLM analysis worker:

1. ``claim_job_and_start_attempt`` — atomically transitions a QUEUED job to
   RUNNING, creates the matching STARTED attempt, and returns the pinned
   evidence and generation configuration the worker needs to call the model.

2. ``record_provider_outcome`` — conditionally writes one terminal provider
   result to an attempt whose owning job is still RUNNING.

3. ``complete_job`` — one atomic transaction that inserts the immutable analysis
   record, marks the successful attempt VALID, and closes the job as COMPLETED.
   Succeeds only when all version, hash, state, and uniqueness preconditions pass.

4. ``fail_job`` — conditionally writes the safe error classification to the
   attempt and transitions the job to FAILED in a single transaction.

5. ``queue_failed_job`` — atomically retries a FAILED job,
   returning a typed result that distinguishes ECN_NOT_FOUND, NOT_RETRYABLE,
   and QUEUED without ambiguity.

6. ``get_progress`` — returns mutually exclusive per-state counts for all jobs,
   optionally scoped to one upload batch.

7. ``fail_interrupted_jobs`` — startup recovery that transitions QUEUED/RUNNING
   jobs to FAILED and marks STARTED/unfinished attempts as INTERRUPTED while
   preserving all already-finished provider telemetry and usage counts.

Design rules enforced here
--------------------------
* One-winner claim: the UPDATE ``WHERE STATE='QUEUED'`` gate stops two workers
  from claiming the same job concurrently.  Only a rowcount of 1 permits the
  attempt INSERT.  The race loser receives ``NOT_QUEUED`` with no side-effects.
* Pinned model: ``AttemptSeed`` intentionally has no ``model_name`` field.  The
  repository derives ``model_name`` exclusively from the stored
  ``MODEL_CONFIG_JSON`` and ``MODEL_NAME`` columns so that the worker cannot
  override the config that was set at ingestion time.
* Typed outcomes: ``ClaimJobResult.status`` distinguishes ``CLAIMED``,
  ``NOT_FOUND``, ``NOT_QUEUED``, and ``PINNED_CONFIG_INVALID`` so that callers
  can map each to a precise HTTP status or retry decision without overloading
  ``None``.
* Pre-provider terminal failure: ``NOT_CALLED`` is the outcome when a
  deterministic step (e.g. prompt projection) fails before any LLM request is
  made.  It requires a non-null error code and prohibits usage or latency so
  that the audit trail never implies an API call occurred.
* Atomic completion: ``complete_job`` requires matching version, hash, and state
  before replacing any previous accepted analysis with the validated result.
  Any mismatch rolls back the replacement and job transition together.
* Safe failure: ``fail_job`` strips control characters from error messages and
  caps them to prevent oversized values reaching the HANA columns.
* Restart safety: ``fail_interrupted_jobs`` targets only attempts still in the
  STARTED state with a null FINISHED_AT, preserving any token usage or provider
  telemetry that was recorded before the restart.
* Short transactions: no ``Connection`` is retained across methods.
* No SDK calls inside transactions: provider calls, serialization, and evidence
  loading happen entirely outside the database transaction window.

SQL safety rules
----------------
* Every user or caller-supplied value is a bound parameter.
* The only text interpolated into SQL is a table identifier validated against
  ``^ECN_IMPACT_[A-Z0-9_]{1,110}$`` by ``quote_identifier``.
* No hand-built string concatenation outside the allowlisted SQL fragment
  builders defined in this module.
* Bind-parameter names use UPPERCASE for INSERT/UPDATE row-dict parameters and
  lowercase for standalone scalar parameters, matching the dict keys passed to
  ``execute()`` exactly.

Usage::

    from app.persistence.analysis_repository import AnalysisRepository
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    repo = AnalysisRepository(engine)
    result = repo.claim_job_and_start_attempt(job_id, seed)
    if result.status is ClaimJobStatus.CLAIMED:
        # call the model, then:
        repo.record_provider_outcome(seed.attempt_id, provider_outcome)
        repo.complete_job(job_id, completion)
"""

from __future__ import annotations

import re
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from app.ecn.analysis_models import derive_completeness
from app.ecn.errors import PersistenceStateError
from app.ecn.models import canonical_uuid
from app.persistence.codec import (
    analysis_completion_to_row,
    evidence_bundle_from_json,
    normalize_row_keys,
    read_lob_text,
    usage_to_columns,
)
from app.persistence.hana import to_hana_timestamp
from app.persistence.records import (
    AnalysisCompletion,
    AnalysisProgressResponse,
    AttemptSeed,
    AttemptStart,
    AttemptUsageColumns,
    ClaimedJob,
    ClaimJobResult,
    ClaimJobStatus,
    PinnedGenerationConfig,
    ProviderCallOutcome,
    ProviderOutcome,
    ProviderStageRecord,
    RetryQueueResult,
    RetryQueueStatus,
    SafeJobError,
    ValidationOutcome,
)
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

# ---------------------------------------------------------------------------
# Error text sanitization helpers
# ---------------------------------------------------------------------------

# Pattern covers C0 controls (except \t \n \r), DEL, and the C1 block.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _strip_error_text(value: str, max_length: int) -> str:
    """Strip control characters and cap the string at max_length.

    Strips all C0 control characters except horizontal tab, line feed, and
    carriage return, plus DEL and the C1 block, then truncates to max_length.

    Args:
        value: Raw error code or message string.
        max_length: Maximum permitted character length.

    Returns:
        A sanitized, length-capped string safe for HANA NVARCHAR storage.
    """
    return _CONTROL_CHAR_PATTERN.sub("", value)[:max_length]


# ---------------------------------------------------------------------------
# Pinned configuration Pydantic model
# ---------------------------------------------------------------------------


class _PinnedPayload(BaseModel):
    """Validate the canonical job-local generation limits stored at ingestion.

    ``extra="forbid"`` and ``strict=True`` ensure that any key added or removed
    between the ingestion version and the current worker version is detected
    as a ``PINNED_CONFIG_INVALID`` error, preventing silent misconfiguration.

    Attributes:
        max_prompt_characters: Maximum total prompt character budget (≥ 4096).
        max_prompt_evidence_items: Maximum evidence items per prompt.
        request_timeout_seconds: Provider call timeout in seconds.
        strict_structured_output: Always ``True`` — structured output is required.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    max_prompt_characters: int = Field(ge=4_096)
    max_prompt_evidence_items: PositiveInt
    request_timeout_seconds: PositiveInt
    strict_structured_output: Literal[True]


# ---------------------------------------------------------------------------
# Pinned config parser
# ---------------------------------------------------------------------------


def parse_pinned_generation_config(
    prompt_version: str,
    output_schema_version: str,
    model_name: str,
    model_config_json: str,
) -> PinnedGenerationConfig:
    """Combine separately persisted versions and model with one strict config payload.

    Validates the three scalar version/model strings against their HANA column
    widths, then delegates to ``_PinnedPayload`` for strict JSON validation.
    Any parse failure raises ``ValueError`` so the caller can detect
    ``PINNED_CONFIG_INVALID`` without catching multiple exception types.

    Args:
        prompt_version: Version tag of the LLM prompt template (≤40 chars).
        output_schema_version: Version tag of the output schema (≤40 chars).
        model_name: Gen AI Hub model name from the job row (≤200 chars).
        model_config_json: The ``MODEL_CONFIG_JSON`` column value.

    Returns:
        A fully validated ``PinnedGenerationConfig`` record.

    Raises:
        ValueError: When any version/model field is blank or too long, when
            ``model_config_json`` is not parseable as JSON, or when any field
            violates its bound.
        pydantic.ValidationError: When the JSON structure is incompatible.
    """
    bounded_fields = {
        "prompt_version": (prompt_version, 40),
        "output_schema_version": (output_schema_version, 40),
        "model_name": (model_name, 200),
    }
    if any(
        not value or len(value) > maximum
        for value, maximum in bounded_fields.values()
    ):
        raise ValueError("a pinned generation field is blank or exceeds its HANA width")
    payload = _PinnedPayload.model_validate_json(model_config_json)
    return PinnedGenerationConfig(
        prompt_version=prompt_version,
        output_schema_version=output_schema_version,
        model_name=model_name,
        request_timeout_seconds=payload.request_timeout_seconds,
        max_prompt_evidence_items=payload.max_prompt_evidence_items,
        max_prompt_characters=payload.max_prompt_characters,
        strict_structured_output=True,
    )


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------


def attempt_start_to_row(
    job_id: str,
    attempt_number: int,
    attempt: AttemptStart,
) -> dict[str, object]:
    """Map the repository-derived STARTED attempt to all 21 HANA column parameters.

    Validates the attempt_id and job_id as canonical UUIDs, ensures
    attempt_number is positive, and checks every required string against its
    HANA column width.  The returned dict uses UPPERCASE column-name keys so
    that the INSERT ``text()`` bind parameters match directly.

    Args:
        job_id: Canonical UUID of the owning analysis job.
        attempt_number: 1-based attempt index (derived from ATTEMPT_COUNT + 1).
        attempt: The ``AttemptStart`` record with all required attempt fields.

    Returns:
        A dict with 21 entries keyed by HANA column names ready for INSERT.

    Raises:
        ValueError: When ``job_id`` or ``attempt.attempt_id`` are not canonical
            UUIDs, when ``attempt_number`` is below 1, or when any required text
            field is blank or exceeds its column width.
    """
    canonical_uuid(job_id)
    canonical_uuid(attempt.attempt_id)
    if attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    required_widths = {
        "correlation_id": (attempt.correlation_id, 100),
        "provider": (attempt.provider, 100),
        "model_name": (attempt.model_name, 200),
        "llm_endpoint": (attempt.llm_endpoint, 100),
    }
    if any(not value or len(value) > width for value, width in required_widths.values()):
        raise ValueError("required attempt text is blank or exceeds its HANA width")
    if attempt.model_deployment is not None and (
        not attempt.model_deployment or len(attempt.model_deployment) > 200
    ):
        raise ValueError("model_deployment exceeds its HANA width")
    return {
        "ATTEMPT_ID": attempt.attempt_id,
        "JOB_ID": job_id,
        "ATTEMPT_NUMBER": attempt_number,
        "CORRELATION_ID": attempt.correlation_id,
        "PROVIDER": attempt.provider,
        "MODEL_NAME": attempt.model_name,
        "MODEL_DEPLOYMENT": attempt.model_deployment,
        "LLM_ENDPOINT": attempt.llm_endpoint,
        "STARTED_AT": to_hana_timestamp(attempt.started_at),
        "FINISHED_AT": None,
        "PROVIDER_OUTCOME": ProviderCallOutcome.STARTED.value,
        "VALIDATION_OUTCOME": ValidationOutcome.NOT_RUN.value,
        "ERROR_CODE": None,
        "LATENCY_MS": None,
        "INPUT_TOKENS": None,
        "CACHE_READ_INPUT_TOKENS": None,
        "CACHE_WRITE_INPUT_TOKENS": None,
        "INPUT_TOTAL_TOKENS": None,
        "OUTPUT_TOKENS": None,
        "TOTAL_TOKENS": None,
        "USAGE_AVAILABLE": 0,
    }


def claimed_job_from_row(
    row: Mapping[str, object],
    attempt: AttemptStart,
    attempt_number: int,
    config: PinnedGenerationConfig,
) -> ClaimedJob:
    """Return the claimed work item with the same attempt that was persisted in HANA.

    Cross-checks that ``attempt.model_name`` equals ``config.model_name`` so
    that a coding error in the caller can be detected immediately.  Evidence is
    decoded from the ``EVIDENCE_JSON`` column using the codec's full deserializer.

    Args:
        row: The joined analysis-job + ecn row returned by the claim SELECT.
        attempt: The ``AttemptStart`` record just inserted into HANA.
        attempt_number: The 1-based attempt count used for the INSERT.
        config: The validated ``PinnedGenerationConfig`` for this job.

    Returns:
        A fully populated ``ClaimedJob`` record ready to be returned to the worker.

    Raises:
        ValueError: When ``attempt.model_name != config.model_name``.
        PersistenceStateError: When ``EVIDENCE_JSON`` cannot be decoded.
    """
    row = normalize_row_keys(row)
    if attempt.model_name != config.model_name:
        raise ValueError("persisted attempt model must equal the pinned job model")
    return ClaimedJob(
        job_id=canonical_uuid(str(row["JOB_ID"])),
        upload_id=canonical_uuid(str(row["UPLOAD_ID"])),
        ecn_row_id=canonical_uuid(str(row["ECN_ROW_ID"])),
        attempt_number=attempt_number,
        attempt=attempt,
        evidence=evidence_bundle_from_json(read_lob_text(row["EVIDENCE_JSON"])),
        generation_config=config,
    )


def provider_outcome_to_row(
    attempt_id: str,
    outcome: ProviderOutcome,
    usage: AttemptUsageColumns,
) -> dict[str, object]:
    """Map a terminal provider outcome to UPDATE column parameters for the attempt table.

    All keys in the returned dict use UPPERCASE HANA column names, matching the
    UPPERCASE bind parameter names used by ``_provider_outcome_sql`` on every
    column.  This ensures SQLAlchemy's case-sensitive bind-name matching works
    correctly for both unit tests and live HANA execution.

    Args:
        attempt_id: Canonical UUID of the attempt being updated.
        outcome: Terminal provider outcome record.
        usage: Pre-computed column values from ``usage_to_columns``.

    Returns:
        A dict of parameters to pass to the conditional UPDATE statement,
        all keyed by UPPERCASE HANA column names matching the SQL bind params.

    Raises:
        ValueError: When ``outcome.latency_ms`` is negative, when
            ``outcome.model_deployment`` exceeds 200 characters, or when
            ``outcome.error_code`` exceeds 80 characters.
    """
    if outcome.latency_ms is not None and outcome.latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    if outcome.model_deployment is not None and len(outcome.model_deployment) > 200:
        raise ValueError("model_deployment exceeds its HANA width (200)")
    if outcome.error_code is not None and len(outcome.error_code) > 80:
        raise ValueError("error_code exceeds its HANA width (80)")
    return {
        # All parameter names match their UPPERCASE HANA column names so that
        # SQLAlchemy text() bind resolution succeeds for both unit and integration tests.
        "ATTEMPT_ID": attempt_id,
        "FINISHED_AT": to_hana_timestamp(outcome.finished_at),
        "PROVIDER_OUTCOME": outcome.provider_outcome.value,
        "ERROR_CODE": outcome.error_code,
        "LATENCY_MS": outcome.latency_ms,
        "MODEL_DEPLOYMENT": outcome.model_deployment,
        "INPUT_TOKENS": usage.input_tokens,
        "CACHE_READ_INPUT_TOKENS": usage.cache_read_input_tokens,
        "CACHE_WRITE_INPUT_TOKENS": usage.cache_write_input_tokens,
        "INPUT_TOTAL_TOKENS": usage.input_total_tokens,
        "OUTPUT_TOKENS": usage.output_tokens,
        "TOTAL_TOKENS": usage.total_tokens,
        "USAGE_AVAILABLE": 1 if usage.usage_available else 0,
    }


def provider_stage_record_to_row(record: ProviderStageRecord) -> dict[str, object]:
    """Map a discovery, legacy, or staged provider call to HANA columns.

    Args:
        record: Validated per-stage provider diagnostic record.

    Returns:
        Uppercase HANA bind parameters, including nullable usage columns.

    Raises:
        ValueError: When identifiers, timestamps, bounds, or stage ordering conflict.
    """
    canonical_uuid(record.call_id)
    canonical_uuid(record.attempt_id)
    if record.stage.startswith("DISCOVERY_"):
        suffix = record.stage.removeprefix("DISCOVERY_")
        expected_number = int(suffix) if suffix.isdigit() else None
    elif record.stage == "PLANNER":
        expected_number = 1
    elif record.stage in {
        "FINAL", "MANUFACTURING", "PROCUREMENT", "CONFIGURATION",
        "DATA_READINESS", "SUMMARY",
    }:
        expected_number = record.call_number
    else:
        expected_number = None
    if (
        expected_number is None
        or not 1 <= record.call_number
        or len(record.stage) > 20
        or record.call_number != expected_number
    ):
        raise ValueError("provider stage call_number conflicts with stage")
    if record.started_at.tzinfo is None or record.finished_at.tzinfo is None:
        raise ValueError("provider stage timestamps must be timezone-aware")
    if record.finished_at < record.started_at:
        raise ValueError("provider stage cannot finish before it starts")
    if record.outcome in {ProviderCallOutcome.STARTED, ProviderCallOutcome.NOT_CALLED}:
        raise ValueError("provider stage outcome must represent an initiated terminal call")
    if record.latency_ms < 0:
        raise ValueError("provider stage latency must be non-negative")
    if record.http_status is not None and not 100 <= record.http_status <= 599:
        raise ValueError("provider stage HTTP status is invalid")
    if record.retry_after_seconds is not None and not 0 <= record.retry_after_seconds <= 86_400:
        raise ValueError("provider stage retry-after is invalid")
    widths = {
        "provider_code": (record.provider_code, 120),
        "provider_request_id": (record.provider_request_id, 200),
        "model_deployment": (record.model_deployment, 200),
        "error_code": (record.error_code, 80),
    }
    if any(value is not None and (not value or len(value) > width) for value, width in widths.values()):
        raise ValueError("provider stage metadata exceeds its HANA width")
    if record.outcome is ProviderCallOutcome.SUCCESS and record.error_code is not None:
        raise ValueError("successful provider stage cannot contain an error code")
    if record.outcome is not ProviderCallOutcome.SUCCESS and record.error_code is None:
        raise ValueError("failed provider stage requires an error code")
    usage = usage_to_columns(record.usage, usage_available=record.usage.usage_available)
    return {
        "CALL_ID": record.call_id,
        "ATTEMPT_ID": record.attempt_id,
        "STAGE": record.stage,
        "CALL_NUMBER": record.call_number,
        "STARTED_AT": to_hana_timestamp(record.started_at),
        "FINISHED_AT": to_hana_timestamp(record.finished_at),
        "OUTCOME": record.outcome.value,
        "HTTP_STATUS": record.http_status,
        "PROVIDER_CODE": record.provider_code,
        "PROVIDER_REQUEST_ID": record.provider_request_id,
        "RETRY_AFTER_SECONDS": record.retry_after_seconds,
        "LATENCY_MS": record.latency_ms,
        "MODEL_DEPLOYMENT": record.model_deployment,
        "ERROR_CODE": record.error_code,
        "INPUT_TOKENS": usage.input_tokens,
        "CACHE_READ_INPUT_TOKENS": usage.cache_read_input_tokens,
        "CACHE_WRITE_INPUT_TOKENS": usage.cache_write_input_tokens,
        "INPUT_TOTAL_TOKENS": usage.input_total_tokens,
        "OUTPUT_TOKENS": usage.output_tokens,
        "TOTAL_TOKENS": usage.total_tokens,
        "USAGE_AVAILABLE": 1 if usage.usage_available else 0,
    }


# ---------------------------------------------------------------------------
# Seed validation
# ---------------------------------------------------------------------------


def _validate_attempt_seed(seed: AttemptSeed) -> None:
    """Validate all AttemptSeed fields against their HANA column widths.

    Called before opening any transaction so that a bad seed never starts a
    database round-trip.  UUID format is validated by ``canonical_uuid``.

    Args:
        seed: The caller-supplied attempt seed to validate.

    Raises:
        ValueError: When ``attempt_id`` is not a canonical UUID, or when any
            required text field is blank or exceeds its column width.
    """
    canonical_uuid(seed.attempt_id)
    required_widths = {
        "correlation_id": (seed.correlation_id, 100),
        "provider": (seed.provider, 100),
        "llm_endpoint": (seed.llm_endpoint, 100),
    }
    if any(not value or len(value) > width for value, width in required_widths.values()):
        raise ValueError("required AttemptSeed field is blank or exceeds its HANA width")
    if seed.model_deployment is not None and (
        not seed.model_deployment or len(seed.model_deployment) > 200
    ):
        raise ValueError("model_deployment exceeds its HANA width")


# ---------------------------------------------------------------------------
# SQL fragment builders
# ---------------------------------------------------------------------------


def _job_claim_source_sql(tables: PersistenceTableNames) -> str:
    """Return the SELECT that loads the full job and ECN context for a claim attempt.

    Filters only by JOB_ID so the caller can distinguish NOT_FOUND (no row) from
    NOT_QUEUED (row exists but STATE != 'QUEUED') using explicit Python checks
    rather than SQL-level state filtering.

    Joins ``ECN_IMPACT_ANALYSIS_JOB`` with ``ECN_IMPACT_ECN`` to include the
    ``UPLOAD_ID`` and ``EVIDENCE_JSON`` columns that are stored on the ECN row
    rather than the job row.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL string with a single ``:job_id`` bind parameter.
    """
    j = quote_identifier(tables.analysis_job)
    e = quote_identifier(tables.ecn)
    return (
        f'SELECT J."JOB_ID", J."ECN_ROW_ID", J."STATE", J."ATTEMPT_COUNT", '
        f'J."PROMPT_VERSION", J."OUTPUT_SCHEMA_VERSION", J."MODEL_NAME", J."MODEL_CONFIG_JSON", '
        f'E."UPLOAD_ID", E."EVIDENCE_JSON" '
        f'FROM {j} J '
        f'JOIN {e} E ON E."ECN_ROW_ID" = J."ECN_ROW_ID" '
        f'WHERE J."JOB_ID" = :job_id'
    )


def _fail_invalid_config_sql(analysis_job_table: str) -> str:
    """Return the conditional UPDATE that terminates a job with an invalid pinned config.

    The ``WHERE STATE='QUEUED'`` predicate prevents overwriting a job that was
    already claimed or completed by another worker between the SELECT and this
    UPDATE.  The safe error message never includes the raw malformed JSON.

    Args:
        analysis_job_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL string with ``:job_id`` and ``:finished_at`` binds.
    """
    t = quote_identifier(analysis_job_table)
    return (
        f'UPDATE {t} '
        f'SET "STATE" = \'FAILED\', '
        f'"FINISHED_AT" = :finished_at, '
        f'"ERROR_CODE" = \'PINNED_CONFIG_INVALID\', '
        f'"ERROR_MESSAGE" = \'The pinned generation configuration is invalid\' '
        f'WHERE "JOB_ID" = :job_id AND "STATE" = \'QUEUED\''
    )


def _claim_sql(analysis_job_table: str) -> str:
    """Return the atomic conditional UPDATE that claims a QUEUED job.

    Only a rowcount of 1 means the claim succeeded.  Rowcount 0 means another
    worker won the race (the job is no longer QUEUED).  The UPDATE resets
    ``FINISHED_AT``, ``ERROR_CODE``, and ``ERROR_MESSAGE`` to NULL so that
    retry attempts start with a clean slate.

    Args:
        analysis_job_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL string with ``:job_id`` and ``:started_at`` binds.
    """
    t = quote_identifier(analysis_job_table)
    return (
        f'UPDATE {t} '
        f'SET "STATE" = \'RUNNING\', '
        f'"ATTEMPT_COUNT" = "ATTEMPT_COUNT" + 1, '
        f'"STARTED_AT" = :started_at, '
        f'"FINISHED_AT" = NULL, '
        f'"ERROR_CODE" = NULL, '
        f'"ERROR_MESSAGE" = NULL '
        f'WHERE "JOB_ID" = :job_id AND "STATE" = \'QUEUED\''
    )


def _attempt_insert_sql(llm_attempt_table: str) -> str:
    """Return the INSERT for a new STARTED attempt row.

    All 21 columns are included with bind parameters matching the UPPERCASE keys
    returned by ``attempt_start_to_row``.  Token columns are inserted as NULL and
    ``USAGE_AVAILABLE`` as 0 because no provider call has occurred yet.

    Args:
        llm_attempt_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL string with 21 bind parameters (all uppercase).
    """
    t = quote_identifier(llm_attempt_table)
    return (
        f'INSERT INTO {t} ('
        f'"ATTEMPT_ID", "JOB_ID", "ATTEMPT_NUMBER", "CORRELATION_ID", '
        f'"PROVIDER", "MODEL_NAME", "MODEL_DEPLOYMENT", "LLM_ENDPOINT", '
        f'"STARTED_AT", "FINISHED_AT", "PROVIDER_OUTCOME", "VALIDATION_OUTCOME", '
        f'"ERROR_CODE", "LATENCY_MS", "INPUT_TOKENS", "CACHE_READ_INPUT_TOKENS", '
        f'"CACHE_WRITE_INPUT_TOKENS", "INPUT_TOTAL_TOKENS", "OUTPUT_TOKENS", "TOTAL_TOKENS", '
        f'"USAGE_AVAILABLE"'
        f') VALUES ('
        f':ATTEMPT_ID, :JOB_ID, :ATTEMPT_NUMBER, :CORRELATION_ID, '
        f':PROVIDER, :MODEL_NAME, :MODEL_DEPLOYMENT, :LLM_ENDPOINT, '
        f':STARTED_AT, :FINISHED_AT, :PROVIDER_OUTCOME, :VALIDATION_OUTCOME, '
        f':ERROR_CODE, :LATENCY_MS, :INPUT_TOKENS, :CACHE_READ_INPUT_TOKENS, '
        f':CACHE_WRITE_INPUT_TOKENS, :INPUT_TOTAL_TOKENS, :OUTPUT_TOKENS, :TOTAL_TOKENS, '
        f':USAGE_AVAILABLE'
        f')'
    )


def _completion_precondition_sql(tables: PersistenceTableNames) -> str:
    """Return the SELECT that loads all fields needed to validate a completion attempt.

    Joins the analysis-job, ECN, and attempt tables.  A LEFT JOIN on the attempt
    ensures the query returns a row even when the attempt does not belong to this
    job (the mismatch is then caught in Python).  A correlated EXISTS sub-select
    checks whether any analysis already exists for this job (UQ on JOB_ID).

    All alias names are UPPERCASE so that the result mapping is consistent with
    the rest of the repository and the test helpers.

    Bind parameters:
    - ``:job_id``    — canonical UUID of the job to complete.
    - ``:attempt_id`` — canonical UUID of the successful attempt.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL SELECT string with ``job_id`` and ``attempt_id`` binds.
    """
    j = quote_identifier(tables.analysis_job)
    e = quote_identifier(tables.ecn)
    la = quote_identifier(tables.llm_attempt)
    an = quote_identifier(tables.analysis)
    return (
        f'SELECT '
        f'J."STATE" AS "JOB_STATE", '
        f'J."ECN_ROW_ID" AS "JOB_ECN_ROW_ID", '
        f'J."PROMPT_VERSION", '
        f'J."OUTPUT_SCHEMA_VERSION", '
        f'J."MODEL_NAME", '
        f'LA."JOB_ID" AS "ATTEMPT_JOB_ID", '
        f'LA."PROVIDER_OUTCOME", '
        f'LA."VALIDATION_OUTCOME", '
        f'LA."MODEL_DEPLOYMENT" AS "ATTEMPT_MODEL_DEPLOYMENT", '
        f'E."EVIDENCE_HASH", '
        f'E."DETERMINISTIC_LOGIC_VERSION", '
        f'E."RULE_EVALUATOR_VERSION", '
        f'CASE WHEN EXISTS ('
        f'SELECT 1 FROM {an} AN WHERE AN."JOB_ID" = :job_id'
        f') THEN 1 ELSE 0 END AS "ANALYSIS_EXISTS" '
        f'FROM {j} J '
        f'JOIN {e} E ON E."ECN_ROW_ID" = J."ECN_ROW_ID" '
        f'LEFT JOIN {la} LA ON LA."ATTEMPT_ID" = :attempt_id '
        f'WHERE J."JOB_ID" = :job_id'
    )


def _require_completion_preconditions(
    state: Mapping[str, object] | None,
    completion: AnalysisCompletion,
) -> None:
    """Validate all completion preconditions against the precondition row.

    Raises ``PersistenceStateError("ANALYSIS_STATE_CONFLICT")`` for any
    mismatch, missing row, wrong state, or duplicate analysis.

    Args:
        state: The row returned by ``_completion_precondition_sql``, or
            ``None`` when the job was not found.
        completion: The caller-supplied completion record.

    Raises:
        PersistenceStateError: On any precondition failure.
    """
    def _fail(reason: str) -> None:
        raise PersistenceStateError(
            "ANALYSIS_STATE_CONFLICT",
            f"The analysis state no longer permits completion: {reason}",
            status_code=409,
        )

    if state is None:
        _fail("job not found")

    if str(state["JOB_STATE"]) != "RUNNING":
        _fail(f"job state is {state['JOB_STATE']!r}, expected RUNNING")

    if str(state["JOB_ECN_ROW_ID"]) != completion.ecn_row_id:
        _fail("ECN row ID mismatch between job and completion")

    # Attempt must belong to this job (LEFT JOIN may return NULL for cross-job attempt).
    if state["ATTEMPT_JOB_ID"] is None or str(state["ATTEMPT_JOB_ID"]) != completion.job_id:
        _fail("attempt not found or belongs to a different job")

    if str(state["PROVIDER_OUTCOME"]) != "SUCCESS":
        _fail(f"provider outcome is {state['PROVIDER_OUTCOME']!r}, expected SUCCESS")

    if str(state["VALIDATION_OUTCOME"]) != "NOT_RUN":
        _fail(f"validation outcome is {state['VALIDATION_OUTCOME']!r}, expected NOT_RUN")

    if str(state["EVIDENCE_HASH"]) != completion.evidence_hash:
        _fail("evidence hash mismatch")

    if str(state["PROMPT_VERSION"]) != completion.prompt_version:
        _fail("prompt version mismatch")

    if str(state["OUTPUT_SCHEMA_VERSION"]) != completion.output_schema_version:
        _fail("output schema version mismatch")

    if str(state["MODEL_NAME"]) != completion.model_name:
        _fail("model name mismatch")

    if str(state["DETERMINISTIC_LOGIC_VERSION"]) != completion.deterministic_logic_version:
        _fail("deterministic logic version mismatch")

    if str(state["RULE_EVALUATOR_VERSION"]) != completion.rule_evaluator_version:
        _fail("rule evaluator version mismatch")

    # Model deployment: NULL stored vs None in completion must both be None;
    # non-NULL stored must match completion exactly.
    stored_deployment = state["ATTEMPT_MODEL_DEPLOYMENT"]
    if stored_deployment is None:
        if completion.model_deployment is not None:
            _fail("stored attempt model deployment is NULL but completion carries one")
    else:
        if completion.model_deployment is None or str(stored_deployment) != completion.model_deployment:
            _fail("model deployment mismatch")

    # Completeness must be derivable from the response carried in the completion.
    expected = derive_completeness(completion.response)
    if completion.completeness != expected:
        _fail(
            f"completeness {completion.completeness!r} does not match "
            f"derived value {expected!r}"
        )


def _validate_completion_ids(job_id: str, completion: AnalysisCompletion) -> None:
    """Cross-check the completion's job_id against the method's job_id argument.

    Called before opening any transaction so that a mismatched caller can be
    rejected early without a database round-trip.

    Args:
        job_id: The canonical UUID passed to ``complete_job``.
        completion: The completion record whose ``job_id`` must match.

    Raises:
        ValueError: When ``job_id`` is not a canonical UUID.
        PersistenceStateError: When ``completion.job_id != job_id``.
    """
    canonical_uuid(job_id)
    canonical_uuid(completion.analysis_id)
    if completion.job_id != job_id:
        raise PersistenceStateError(
            "ANALYSIS_STATE_CONFLICT",
            "The analysis state conflict: completion job_id does not match the supplied job_id",
            status_code=409,
        )


def _analysis_insert_sql(analysis_table: str) -> str:
    """Return the INSERT for one immutable analysis record.

    All 18 columns are included with UPPERCASE bind parameter names matching
    the keys returned by ``analysis_completion_to_row``.

    Args:
        analysis_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL INSERT string with all 18 UPPERCASE bind params.
    """
    t = quote_identifier(analysis_table)
    return (
        f'INSERT INTO {t} ('
        f'"ANALYSIS_ID", "ECN_ROW_ID", "JOB_ID", "SUCCESSFUL_ATTEMPT_ID", '
        f'"COMPLETENESS_STATE", "CONTENT_JSON", "EVIDENCE_HASH", "PROMPT_VERSION", '
        f'"MODEL_NAME", "MODEL_DEPLOYMENT", "DETERMINISTIC_LOGIC_VERSION", '
        f'"RULE_EVALUATOR_VERSION", "OUTPUT_SCHEMA_VERSION", "CREATED_AT", '
        f'"DISCOVERY_STATUS", "DISCOVERY_ERROR_CODE", "DISCOVERY_HASH", '
        f'"DISCOVERED_EVIDENCE_JSON"'
        f') VALUES ('
        f':ANALYSIS_ID, :ECN_ROW_ID, :JOB_ID, :SUCCESSFUL_ATTEMPT_ID, '
        f':COMPLETENESS_STATE, :CONTENT_JSON, :EVIDENCE_HASH, :PROMPT_VERSION, '
        f':MODEL_NAME, :MODEL_DEPLOYMENT, :DETERMINISTIC_LOGIC_VERSION, '
        f':RULE_EVALUATOR_VERSION, :OUTPUT_SCHEMA_VERSION, :CREATED_AT, '
        f':DISCOVERY_STATUS, :DISCOVERY_ERROR_CODE, :DISCOVERY_HASH, '
        f':DISCOVERED_EVIDENCE_JSON'
        f')'
    )


def _validate_attempt_sql(llm_attempt_table: str) -> str:
    """Return the conditional UPDATE that marks a successful attempt VALID.

    Only updates an attempt that is still in NOT_RUN validation state, belongs
    to the specified job, and had a SUCCESS provider outcome.  The WHERE clause
    guards against duplicate validation writes.

    Bind parameters use lowercase names (``attempt_id``, ``job_id``) to match
    the scalar dict passed at call time.

    Args:
        llm_attempt_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL UPDATE string with ``attempt_id`` and ``job_id`` binds.
    """
    t = quote_identifier(llm_attempt_table)
    return (
        f'UPDATE {t} '
        f'SET "VALIDATION_OUTCOME" = \'VALID\' '
        f'WHERE "ATTEMPT_ID" = :attempt_id '
        f'AND "JOB_ID" = :job_id '
        f"AND \"PROVIDER_OUTCOME\" = 'SUCCESS' "
        f"AND \"VALIDATION_OUTCOME\" = 'NOT_RUN'"
    )


def _complete_job_sql(analysis_job_table: str) -> str:
    """Return the conditional UPDATE that marks a RUNNING job as COMPLETED.

    Clears the error columns alongside the state transition so that any
    ephemeral error state left from a prior attempt is removed.  The
    ``WHERE STATE='RUNNING'`` predicate prevents a duplicate completion write.

    Bind parameters use lowercase names (``job_id``, ``finished_at``).

    Args:
        analysis_job_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL UPDATE string with ``job_id`` and ``finished_at`` binds.
    """
    t = quote_identifier(analysis_job_table)
    return (
        f'UPDATE {t} '
        f'SET "STATE" = \'COMPLETED\', '
        f'"FINISHED_AT" = :finished_at, '
        f'"ERROR_CODE" = NULL, '
        f'"ERROR_MESSAGE" = NULL '
        f'WHERE "JOB_ID" = :job_id '
        f"AND \"STATE\" = 'RUNNING'"
    )


def _fail_attempt_sql(llm_attempt_table: str) -> str:
    """Return the conditional UPDATE that records a validation outcome on a failed attempt.

    Targets only an attempt that has already reached a terminal provider state
    (PROVIDER_OUTCOME != 'STARTED') and belongs to the specified job.

    Bind parameters use UPPERCASE names (``ATTEMPT_ID``, ``JOB_ID``,
    ``VALIDATION_OUTCOME``) to match the row-dict convention.

    Args:
        llm_attempt_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL UPDATE string with UPPERCASE bind params.
    """
    t = quote_identifier(llm_attempt_table)
    return (
        f'UPDATE {t} '
        f'SET "VALIDATION_OUTCOME" = :VALIDATION_OUTCOME '
        f'WHERE "ATTEMPT_ID" = :ATTEMPT_ID '
        f'AND "JOB_ID" = :JOB_ID '
        f"AND \"PROVIDER_OUTCOME\" != 'STARTED'"
    )


def _fail_job_sql(analysis_job_table: str) -> str:
    """Return the conditional UPDATE that transitions a RUNNING job to FAILED.

    Records the sanitized safe error code and message alongside the state
    change.  The ``WHERE STATE='RUNNING'`` predicate prevents double-failing.

    Bind parameters use UPPERCASE names (``JOB_ID``, ``FINISHED_AT``,
    ``ERROR_CODE``, ``ERROR_MESSAGE``) to match the row-dict convention.

    Args:
        analysis_job_table: Physical table name validated by ``quote_identifier``.

    Returns:
        A parameterized SQL UPDATE string with UPPERCASE bind params.
    """
    t = quote_identifier(analysis_job_table)
    return (
        f'UPDATE {t} '
        f'SET "STATE" = \'FAILED\', '
        f'"FINISHED_AT" = :FINISHED_AT, '
        f'"ERROR_CODE" = :ERROR_CODE, '
        f'"ERROR_MESSAGE" = :ERROR_MESSAGE '
        f'WHERE "JOB_ID" = :JOB_ID '
        f"AND \"STATE\" = 'RUNNING'"
    )


def _retry_source_sql(tables: PersistenceTableNames) -> str:
    """Return the SELECT that loads ECN + job + analysis context for a retry request.

    Starts at the ECN table and LEFT JOINs its canonical job and any existing
    analysis so that all three combinations (no job, job exists, analysis
    exists) are distinguishable:

    - ``row is None``: no ECN → ``ECN_NOT_FOUND``
    - ``row["JOB_ID"] is None``: ECN but no job → ``NOT_RETRYABLE``
    - ``row["STATE"] != 'FAILED'``: job exists but is not retryable
    - otherwise: ``QUEUED`` after a successful UPDATE.

    Bind parameters:
    - ``:ecn_row_id`` — canonical UUID of the ECN.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL SELECT string with one ``ecn_row_id`` bind param.
    """
    e = quote_identifier(tables.ecn)
    j = quote_identifier(tables.analysis_job)
    an = quote_identifier(tables.analysis)
    return (
        f'SELECT '
        f'E."ECN_ROW_ID", '
        f'J."JOB_ID", '
        f'J."STATE", '
        f'AN."ANALYSIS_ID" '
        f'FROM {e} E '
        f'LEFT JOIN {j} J ON J."ECN_ROW_ID" = E."ECN_ROW_ID" '
        f'LEFT JOIN {an} AN ON AN."JOB_ID" = J."JOB_ID" '
        f'WHERE E."ECN_ROW_ID" = :ecn_row_id'
    )


def _queue_failed_sql(tables: PersistenceTableNames) -> str:
    """Return the conditional UPDATE that atomically retries a FAILED job.

    Requires ``STATE='FAILED'``. A previously accepted analysis can remain
    visible while the retry runs. Preserves all
    pinned fields (PROMPT_VERSION, OUTPUT_SCHEMA_VERSION, MODEL_NAME,
    MODEL_CONFIG_JSON) and ATTEMPT_COUNT so that the next claim uses the
    same pinned generation config.

    Bind parameters:
    - ``:job_id`` — canonical UUID of the job to re-queue.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL UPDATE string with one ``job_id`` bind param.
    """
    j = quote_identifier(tables.analysis_job)
    return (
        f'UPDATE {j} '
        f'SET "STATE" = \'QUEUED\', '
        f'"STARTED_AT" = NULL, '
        f'"FINISHED_AT" = NULL, '
        f'"ERROR_CODE" = NULL, '
        f'"ERROR_MESSAGE" = NULL '
        f'WHERE "JOB_ID" = :job_id '
        f"AND \"STATE\" = 'FAILED'"
    )


def _progress_sql(tables: PersistenceTableNames, upload_id: str | None) -> str:
    """Return a single-row aggregate SELECT that counts jobs by state.

    Uses mutually exclusive CASE expressions so each job contributes to exactly
    one of the six counts.  COMPLETED jobs are sub-divided by analysis
    completeness state (``COMPLETED`` vs ``COMPLETED_WITH_GAPS``).

    When ``upload_id`` is supplied the ECN table is INNER-JOINed and a WHERE
    clause filters by UPLOAD_ID; otherwise only the job and analysis tables
    are accessed.

    Bind parameters (only when upload_id is not None):
    - ``:upload_id`` — canonical UUID of the upload to scope results to.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.
        upload_id: Optional upload UUID; ``None`` for a global count.

    Returns:
        A parameterized SQL SELECT string returning one row with six count columns.
    """
    j = quote_identifier(tables.analysis_job)
    a = quote_identifier(tables.analysis)

    ecn_join = ""
    where_clause = ""
    if upload_id is not None:
        e = quote_identifier(tables.ecn)
        ecn_join = f"INNER JOIN {e} E ON E.\"ECN_ROW_ID\" = J.\"ECN_ROW_ID\" "
        where_clause = "WHERE E.\"UPLOAD_ID\" = :upload_id"

    return (
        f'SELECT '
        f'COUNT(CASE WHEN J."STATE" = \'READY\' THEN 1 END) AS "READY_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'QUEUED\' THEN 1 END) AS "QUEUED_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'BLOCKED\' THEN 1 END) AS "BLOCKED_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'RUNNING\' THEN 1 END) AS "RUNNING_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'COMPLETED\' AND A."COMPLETENESS_STATE" = \'COMPLETED\' '
        f'THEN 1 END) AS "COMPLETED_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'COMPLETED\' AND A."COMPLETENESS_STATE" = \'COMPLETED_WITH_GAPS\' '
        f'THEN 1 END) AS "COMPLETED_WITH_GAPS_COUNT", '
        f'COUNT(CASE WHEN J."STATE" = \'FAILED\' THEN 1 END) AS "FAILED_COUNT" '
        f'FROM {j} J '
        f'LEFT JOIN {a} A ON A."JOB_ID" = J."JOB_ID" '
        f'{ecn_join}'
        f'{where_clause}'
    )


def _block_queued_job_sql(tables: PersistenceTableNames) -> str:
    """Return the conditional transition from queued to provider-blocked."""
    table = quote_identifier(tables.analysis_job)
    return (
        f"UPDATE {table} "
        f"SET \"STATE\" = 'BLOCKED', "
        f'"ERROR_CODE" = :error_code, '
        f'"ERROR_MESSAGE" = :error_message '
        f'WHERE "JOB_ID" = :job_id '
        f"AND \"STATE\" = 'QUEUED'"
    )


def _blocked_job_ids_sql(tables: PersistenceTableNames) -> str:
    """Return IDs of jobs blocked specifically by provider preflight."""
    table = quote_identifier(tables.analysis_job)
    return (
        f'SELECT "JOB_ID" FROM {table} '
        f"WHERE \"STATE\" = 'BLOCKED' "
        f'AND "ERROR_CODE" IN ('
        f"'PROVIDER_AUTH_FAILED', 'PROVIDER_FORBIDDEN', 'MODEL_NOT_AVAILABLE', "
        f"'PROVIDER_RATE_LIMITED', 'PROVIDER_TIMEOUT', 'PROVIDER_SERVER_ERROR') "
        f'ORDER BY "JOB_ID"'
    )


def _queue_blocked_job_sql(tables: PersistenceTableNames) -> str:
    """Return the conditional transition from provider-blocked back to queued."""
    table = quote_identifier(tables.analysis_job)
    return (
        f"UPDATE {table} "
        f"SET \"STATE\" = 'QUEUED', "
        f'"ERROR_CODE" = NULL, '
        f'"ERROR_MESSAGE" = NULL '
        f'WHERE "JOB_ID" = :job_id '
        f"AND \"STATE\" = 'BLOCKED'"
    )


def _interrupt_started_attempts_sql(tables: PersistenceTableNames) -> str:
    """Return the UPDATE that marks STARTED/unfinished attempts as INTERRUPTED.

    Targets exclusively attempts whose PROVIDER_OUTCOME is still 'STARTED'
    AND whose FINISHED_AT is NULL (i.e. the worker was interrupted before it
    could record a terminal outcome).  Attempts that reached a terminal provider
    outcome (SUCCESS, ERROR, TIMEOUT, NOT_CALLED) are intentionally excluded
    by the FINISHED_AT IS NULL predicate because they have a non-null FINISHED_AT.

    The SET clause deliberately omits all token/usage columns so that any
    partial telemetry written before the restart is preserved in the row.

    Bind parameters:
    - ``:finished_at`` — the restart timestamp to record as FINISHED_AT.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL UPDATE string with one ``finished_at`` bind param.
    """
    t = quote_identifier(tables.llm_attempt)
    return (
        f'UPDATE {t} '
        f'SET "PROVIDER_OUTCOME" = \'INTERRUPTED\', '
        f'"VALIDATION_OUTCOME" = \'NOT_RUN\', '
        f'"FINISHED_AT" = :finished_at '
        f'WHERE "PROVIDER_OUTCOME" = \'STARTED\' '
        f'AND "FINISHED_AT" IS NULL'
    )


def _fail_unfinished_jobs_sql(tables: PersistenceTableNames) -> str:
    """Return the UPDATE that fails all QUEUED or RUNNING jobs after a restart.

    Targets any job in QUEUED or RUNNING state regardless of attempt history,
    because both states imply the worker was interrupted before completing
    or gracefully failing the job.

    Bind parameters:
    - ``:finished_at`` — the restart timestamp to record as FINISHED_AT.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL UPDATE string with one ``finished_at`` bind param.
    """
    t = quote_identifier(tables.analysis_job)
    return (
        f'UPDATE {t} '
        f'SET "STATE" = \'FAILED\', '
        f'"FINISHED_AT" = :finished_at, '
        f'"ERROR_CODE" = \'PROCESS_RESTARTED\', '
        f'"ERROR_MESSAGE" = \'The analysis worker process was restarted '
        f'before this job completed\' '
        f'WHERE "STATE" IN (\'QUEUED\', \'RUNNING\')'
    )


def _provider_outcome_sql(tables: PersistenceTableNames) -> str:
    """Return the conditional UPDATE that records one terminal provider outcome.

    The WHERE clause requires:
    * ``PROVIDER_OUTCOME = 'STARTED'``: prevents duplicate terminal writes.
    * ``EXISTS (...STATE = 'RUNNING')``: prevents writing to an attempt whose
      job was already completed or failed by another code path.

    ``VALIDATION_OUTCOME`` is intentionally excluded from the SET clause; it is
    updated separately by the validation write path (Task 9).

    All bind parameter names use UPPERCASE so they match the UPPERCASE dict keys
    returned by ``provider_outcome_to_row``.  SQLAlchemy bind-name matching is
    case-sensitive, so every ``:NAME`` token in this SQL must be an exact
    case-match of its corresponding key in the parameters dict.

    Args:
        tables: Active ``PersistenceTableNames`` for identifier allowlisting.

    Returns:
        A parameterized SQL string with all bind params in UPPERCASE.
    """
    a = quote_identifier(tables.llm_attempt)
    j = quote_identifier(tables.analysis_job)
    return (
        f'UPDATE {a} '
        f'SET "FINISHED_AT" = :FINISHED_AT, '
        f'"PROVIDER_OUTCOME" = :PROVIDER_OUTCOME, '
        f'"ERROR_CODE" = :ERROR_CODE, '
        f'"LATENCY_MS" = :LATENCY_MS, '
        f'"MODEL_DEPLOYMENT" = :MODEL_DEPLOYMENT, '
        f'"INPUT_TOKENS" = :INPUT_TOKENS, '
        f'"CACHE_READ_INPUT_TOKENS" = :CACHE_READ_INPUT_TOKENS, '
        f'"CACHE_WRITE_INPUT_TOKENS" = :CACHE_WRITE_INPUT_TOKENS, '
        f'"INPUT_TOTAL_TOKENS" = :INPUT_TOTAL_TOKENS, '
        f'"OUTPUT_TOKENS" = :OUTPUT_TOKENS, '
        f'"TOTAL_TOKENS" = :TOTAL_TOKENS, '
        f'"USAGE_AVAILABLE" = :USAGE_AVAILABLE '
        f'WHERE "ATTEMPT_ID" = :ATTEMPT_ID '
        f"AND \"PROVIDER_OUTCOME\" = 'STARTED' "
        f'AND EXISTS ('
        f'SELECT 1 FROM {j} '
        f'WHERE "JOB_ID" = {a}."JOB_ID" '
        f"AND \"STATE\" = 'RUNNING'"
        f')'
    )


def _provider_call_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the immutable INSERT for one bounded discovery or final call.

    Args:
        tables: Active persistence table names.

    Returns:
        Parameterized INSERT covering all provider-call child columns.
    """
    table = quote_identifier(tables.provider_call)
    columns = (
        "CALL_ID",
        "ATTEMPT_ID",
        "STAGE",
        "CALL_NUMBER",
        "STARTED_AT",
        "FINISHED_AT",
        "OUTCOME",
        "HTTP_STATUS",
        "PROVIDER_CODE",
        "PROVIDER_REQUEST_ID",
        "RETRY_AFTER_SECONDS",
        "LATENCY_MS",
        "MODEL_DEPLOYMENT",
        "ERROR_CODE",
        "INPUT_TOKENS",
        "CACHE_READ_INPUT_TOKENS",
        "CACHE_WRITE_INPUT_TOKENS",
        "INPUT_TOTAL_TOKENS",
        "OUTPUT_TOKENS",
        "TOTAL_TOKENS",
        "USAGE_AVAILABLE",
    )
    quoted = ", ".join(f'"{column}"' for column in columns)
    parameters = ", ".join(f":{column}" for column in columns)
    return f"INSERT INTO {table} ({quoted}) VALUES ({parameters})"


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class AnalysisRepository:
    """Claim queued analysis jobs and record provider attempt outcomes.

    All public methods open a short-lived ``engine.begin()`` context per call.
    No ``Connection`` is retained as instance state between calls.  No model
    SDK, serialization, or I/O is performed inside a transaction.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the HANA instance.
        table_names: Optional ``PersistenceTableNames`` override; defaults to
            the production names.  Pass a suffixed instance for isolated live
            tests.

    Example::

        repo = AnalysisRepository(engine)
        result = repo.claim_job_and_start_attempt(job_id, seed)
        if result.status is ClaimJobStatus.CLAIMED:
            # ... call model ...
            repo.record_provider_outcome(seed.attempt_id, outcome)
    """

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames = PersistenceTableNames(),
    ) -> None:
        """Bind the engine and table-name set.

        Args:
            engine: SQLAlchemy ``Engine``; never stored beyond this reference.
            table_names: Physical table identifiers already validated by
                ``PersistenceTableNames.__post_init__``.
        """
        self._engine = engine
        self._tables = table_names

    def save_stage_result(
        self, result_id: str, job_id: str, attempt_id: str, stage: str,
        evidence_hash: str, prompt_version: str, schema_version: str,
        content: dict, created_at: datetime,
    ) -> None:
        """Persist one validated stage output tied to the exact attempt and evidence."""
        for identifier in (result_id, job_id, attempt_id):
            canonical_uuid(identifier)
        if stage not in {"MANUFACTURING", "PROCUREMENT", "CONFIGURATION", "DATA_READINESS", "SUMMARY"}:
            raise ValueError("unsupported analysis stage")
        with self._engine.begin() as connection:
            connection.execute(text(
                f'INSERT INTO {quote_identifier(self._tables.analysis_stage)} '
                '("RESULT_ID", "JOB_ID", "ATTEMPT_ID", "STAGE", "EVIDENCE_HASH", '
                '"PROMPT_VERSION", "OUTPUT_SCHEMA_VERSION", "CONTENT_JSON", "CREATED_AT") '
                'VALUES (:result_id, :job_id, :attempt_id, :stage, :evidence_hash, '
                ':prompt_version, :schema_version, :content, :created_at)'
            ), {
                "result_id": result_id, "job_id": job_id, "attempt_id": attempt_id,
                "stage": stage, "evidence_hash": evidence_hash,
                "prompt_version": prompt_version, "schema_version": schema_version,
                "content": json.dumps(content, ensure_ascii=False, sort_keys=True),
                "created_at": to_hana_timestamp(created_at),
            })

    def list_stage_results(
        self, job_id: str, evidence_hash: str, prompt_version: str,
        schema_version: str,
    ) -> dict[str, dict]:
        """Return compatible stage outputs newer than the last accepted analysis."""
        canonical_uuid(job_id)
        with self._engine.connect() as connection:
            rows = connection.execute(text(
                f'SELECT S."STAGE", S."CONTENT_JSON" FROM {quote_identifier(self._tables.analysis_stage)} S '
                f'JOIN {quote_identifier(self._tables.llm_attempt)} LA ON LA."ATTEMPT_ID" = S."ATTEMPT_ID" '
                f'LEFT JOIN {quote_identifier(self._tables.analysis)} AN ON AN."JOB_ID" = S."JOB_ID" '
                f'LEFT JOIN {quote_identifier(self._tables.llm_attempt)} BA ON BA."ATTEMPT_ID" = AN."SUCCESSFUL_ATTEMPT_ID" '
                'WHERE S."JOB_ID" = :job_id AND S."EVIDENCE_HASH" = :evidence_hash '
                'AND S."PROMPT_VERSION" = :prompt_version AND S."OUTPUT_SCHEMA_VERSION" = :schema_version '
                'AND LA."ATTEMPT_NUMBER" > COALESCE(BA."ATTEMPT_NUMBER", 0) '
                'ORDER BY LA."ATTEMPT_NUMBER" DESC, S."CREATED_AT" DESC'
            ), {"job_id": job_id, "evidence_hash": evidence_hash,
                "prompt_version": prompt_version, "schema_version": schema_version}).mappings().all()
        outputs = {}
        for raw_row in rows:
            row = normalize_row_keys(raw_row)
            stage = str(row["STAGE"])
            if stage not in outputs:
                outputs[stage] = json.loads(read_lob_text(row["CONTENT_JSON"]))
        return outputs

    def stage_results_for_ecn(self, ecn_row_id: str) -> dict[str, dict]:
        """Read compatible validated stage results for the selected ECN occurrence."""
        canonical_uuid(ecn_row_id)
        with self._engine.connect() as connection:
            row = connection.execute(text(
                f'SELECT J."JOB_ID", J."PROMPT_VERSION", J."OUTPUT_SCHEMA_VERSION", '
                f'E."EVIDENCE_HASH" FROM {quote_identifier(self._tables.analysis_job)} J '
                f'JOIN {quote_identifier(self._tables.ecn)} E ON E."ECN_ROW_ID" = J."ECN_ROW_ID" '
                'WHERE J."ECN_ROW_ID" = :ecn_row_id'
            ), {"ecn_row_id": ecn_row_id}).mappings().one_or_none()
        if row is None:
            return {}
        values = normalize_row_keys(row)
        return self.list_stage_results(
            str(values["JOB_ID"]), str(values["EVIDENCE_HASH"]),
            str(values["PROMPT_VERSION"]), str(values["OUTPUT_SCHEMA_VERSION"]),
        )

    def claim_job_and_start_attempt(
        self,
        job_id: str,
        seed: AttemptSeed,
    ) -> ClaimJobResult:
        """Atomically claim one QUEUED job and create its sole in-progress attempt.

        Executes inside a single ``engine.begin()`` transaction:

        1. SELECT the job + ECN row regardless of state.
        2. If no row → return ``NOT_FOUND``.
        3. If ``STATE != 'QUEUED'`` → return ``NOT_QUEUED``.
        4. Parse ``MODEL_CONFIG_JSON``; if invalid → conditional UPDATE to FAILED
           (``WHERE STATE='QUEUED'``); if that UPDATE wins → return
           ``PINNED_CONFIG_INVALID``; if it loses → return ``NOT_QUEUED``.
        5. UPDATE job to RUNNING (``WHERE STATE='QUEUED'``); if rowcount != 1 →
           return ``NOT_QUEUED`` (race loss).
        6. INSERT attempt with ``PROVIDER_OUTCOME='STARTED'`` and all token columns
           NULL; derive ``model_name`` from the parsed config, never from the seed.
        7. Return ``CLAIMED`` with the fully populated ``ClaimedJob``.

        Args:
            job_id: Canonical UUID of the job to claim.
            seed: Caller-supplied attempt identity and provider context.

        Returns:
            A ``ClaimJobResult`` with a discriminated ``ClaimJobStatus``.

        Raises:
            ValueError: When ``job_id`` is not a canonical UUID or when any
                ``seed`` field fails its width validation.
        """
        canonical_uuid(job_id)
        _validate_attempt_seed(seed)

        with self._engine.begin() as connection:
            # --- Step 1: load job + ECN row (state-agnostic) ---
            _raw = connection.execute(
                text(_job_claim_source_sql(self._tables)),
                {"job_id": job_id},
            ).mappings().one_or_none()

            if _raw is None:
                return ClaimJobResult(ClaimJobStatus.NOT_FOUND, None)

            # Normalize here so all subsequent row["..."] accesses are case-agnostic.
            job_row = normalize_row_keys(_raw)

            if str(job_row["STATE"]) != "QUEUED":
                return ClaimJobResult(ClaimJobStatus.NOT_QUEUED, None)

            # --- Step 2: validate pinned config (never expose raw JSON in error) ---
            try:
                config = parse_pinned_generation_config(
                    str(job_row["PROMPT_VERSION"]),
                    str(job_row["OUTPUT_SCHEMA_VERSION"]),
                    str(job_row["MODEL_NAME"]),
                    read_lob_text(job_row["MODEL_CONFIG_JSON"]),
                )
            except (ValueError, ValidationError):
                # Fail the job in HANA, but only if it is still QUEUED.
                failed = connection.execute(
                    text(_fail_invalid_config_sql(self._tables.analysis_job)),
                    {
                        "job_id": job_id,
                        "finished_at": to_hana_timestamp(seed.started_at),
                    },
                )
                if failed.rowcount != 1:
                    return ClaimJobResult(ClaimJobStatus.NOT_QUEUED, None)
                return ClaimJobResult(ClaimJobStatus.PINNED_CONFIG_INVALID, None)

            # --- Step 3: atomic claim UPDATE (one-winner gate) ---
            updated = connection.execute(
                text(_claim_sql(self._tables.analysis_job)),
                {
                    "job_id": job_id,
                    "started_at": to_hana_timestamp(seed.started_at),
                },
            )
            if updated.rowcount != 1:
                return ClaimJobResult(ClaimJobStatus.NOT_QUEUED, None)

            # --- Step 4: derive attempt (model from config, not from seed) ---
            attempt_number = int(job_row["ATTEMPT_COUNT"]) + 1
            attempt = AttemptStart(
                attempt_id=seed.attempt_id,
                correlation_id=seed.correlation_id,
                started_at=seed.started_at,
                provider=seed.provider,
                model_name=config.model_name,
                model_deployment=seed.model_deployment,
                llm_endpoint=seed.llm_endpoint,
            )

            # --- Step 5: insert STARTED attempt ---
            connection.execute(
                text(_attempt_insert_sql(self._tables.llm_attempt)),
                attempt_start_to_row(job_id, attempt_number, attempt),
            )

            claimed = claimed_job_from_row(job_row, attempt, attempt_number, config)
            return ClaimJobResult(ClaimJobStatus.CLAIMED, claimed)

    def complete_job(
        self,
        job_id: str,
        completion: AnalysisCompletion,
    ) -> None:
        """Persist one immutable validated analysis and close its exact job and attempt.

        Executes exactly one ``engine.begin()`` transaction containing three
        ordered writes:

        1. Replace any previous accepted analysis with the new validated record.
        2. Conditionally UPDATE the successful attempt: ``NOT_RUN`` → ``VALID``.
        3. Conditionally UPDATE the job: ``RUNNING`` → ``COMPLETED``.

        All three writes succeed together or roll back together.  Before the
        writes, a precondition SELECT validates that the job is RUNNING with the
        correct ECN, the attempt belongs to this job with a SUCCESS provider
        outcome and NOT_RUN validation outcome, all pinned version fields match,
        the evidence hash matches, and the attempt model deployment matches.
        An earlier accepted result remains readable until this transaction commits.

        Any mismatch, duplicate analysis, or late conditional UPDATE (rowcount 0)
        rolls back all three writes and raises ``PersistenceStateError`` with
        code ``ANALYSIS_STATE_CONFLICT``.

        Args:
            job_id: Canonical UUID of the job to complete.
            completion: The validated analysis result with all pinned fields.

        Raises:
            ValueError: When ``job_id`` is not a canonical UUID.
            PersistenceStateError: When ``completion.job_id != job_id``, when any
                precondition check fails, when the analysis is a duplicate, or
                when either conditional UPDATE matches no row.
        """
        _validate_completion_ids(job_id, completion)
        with self._engine.begin() as connection:
            _raw_state = connection.execute(
                text(_completion_precondition_sql(self._tables)),
                {"job_id": job_id, "attempt_id": completion.successful_attempt_id},
            ).mappings().one_or_none()
            # Normalize before passing to the precondition checker so that all
            # row["KEY"] accesses inside it are dialect-case-agnostic.
            state = normalize_row_keys(_raw_state) if _raw_state is not None else None
            _require_completion_preconditions(state, completion)
            if int(state["ANALYSIS_EXISTS"]):
                connection.execute(
                    text(f'DELETE FROM {quote_identifier(self._tables.analysis)} '
                         'WHERE "JOB_ID" = :job_id'),
                    {"job_id": job_id},
                )
            try:
                connection.execute(
                    text(_analysis_insert_sql(self._tables.analysis)),
                    analysis_completion_to_row(completion),
                )
            except IntegrityError as exc:
                # Convert duplicate-analysis unique violation to the same safe error
                # so callers see ANALYSIS_STATE_CONFLICT regardless of whether the
                # conflict was caught by the precondition SELECT or by the constraint.
                raise PersistenceStateError(
                    "ANALYSIS_STATE_CONFLICT",
                    "The analysis state conflict: a record already exists for this job or ECN",
                    status_code=409,
                ) from exc
            attempt_update = connection.execute(
                text(_validate_attempt_sql(self._tables.llm_attempt)),
                {"attempt_id": completion.successful_attempt_id, "job_id": job_id},
            )
            job_update = connection.execute(
                text(_complete_job_sql(self._tables.analysis_job)),
                {"job_id": job_id, "finished_at": to_hana_timestamp(completion.created_at)},
            )
            if attempt_update.rowcount != 1 or job_update.rowcount != 1:
                raise PersistenceStateError(
                    "ANALYSIS_STATE_CONFLICT",
                    "The analysis state no longer permits completion: conditional update matched no row",
                    status_code=409,
                )

    def fail_job(
        self,
        job_id: str,
        attempt_id: str,
        error: SafeJobError,
        finished_at: datetime,
    ) -> None:
        """Record a safe error classification and transition the job to FAILED.

        Executes one transaction with two conditional writes:

        1. UPDATE the attempt's ``VALIDATION_OUTCOME`` to ``error.validation_outcome``
           (``INVALID`` when the provider returned SUCCESS but validation failed;
           ``NOT_RUN`` when the provider did not return a response).  The WHERE
           clause requires ``PROVIDER_OUTCOME != 'STARTED'`` to guard against
           writing to an attempt that has not yet finished its provider call.
        2. UPDATE the job to ``FAILED`` with the sanitized error code and message.
           The WHERE clause requires ``STATE='RUNNING'``.

        Error code and message are stripped of control characters and capped at
        80 and 2000 characters respectively before storage.

        Args:
            job_id: Canonical UUID of the RUNNING job to fail.
            attempt_id: Canonical UUID of the terminal attempt.
            error: Safe error classification with a stable code, public message,
                and validation outcome.
            finished_at: Timezone-aware timestamp when the failure occurred.

        Raises:
            ValueError: When ``job_id`` or ``attempt_id`` are not canonical UUIDs.
            PersistenceStateError: When either conditional UPDATE matches no row,
                indicating a concurrent state change.
        """
        canonical_uuid(job_id)
        canonical_uuid(attempt_id)
        safe_code = _strip_error_text(error.code, 80)
        safe_message = _strip_error_text(error.message, 2000)
        ts = to_hana_timestamp(finished_at)
        with self._engine.begin() as connection:
            attempt_update = connection.execute(
                text(_fail_attempt_sql(self._tables.llm_attempt)),
                {
                    "ATTEMPT_ID": attempt_id,
                    "JOB_ID": job_id,
                    "VALIDATION_OUTCOME": error.validation_outcome.value,
                },
            )
            job_update = connection.execute(
                text(_fail_job_sql(self._tables.analysis_job)),
                {
                    "JOB_ID": job_id,
                    "FINISHED_AT": ts,
                    "ERROR_CODE": safe_code,
                    "ERROR_MESSAGE": safe_message,
                },
            )
            if attempt_update.rowcount != 1 or job_update.rowcount != 1:
                raise PersistenceStateError(
                    "ANALYSIS_STATE_CONFLICT",
                    "The analysis state no longer permits a failure write: conditional update matched no row",
                    status_code=409,
                )

    def queue_failed_job(self, ecn_row_id: str) -> RetryQueueResult:
        """Atomically re-queue a FAILED job while retaining any prior analysis.

        Executes inside a single ``engine.begin()`` transaction.  The SELECT
        covers three distinguishable outcomes:

        - ``ECN_NOT_FOUND``: no row for ``ecn_row_id`` — the ECN never existed.
        - ``NOT_RETRYABLE``: the ECN exists but has no FAILED job.
        - ``QUEUED``: the FAILED→QUEUED UPDATE succeeded (rowcount=1).

        The UPDATE preserves all pinned fields (prompt version, output schema
        version, model name, model config JSON) and attempt count so that the
        next claim uses the same generation configuration.

        Args:
            ecn_row_id: Canonical UUID of the ECN whose job should be re-queued.

        Returns:
            A ``RetryQueueResult`` with a discriminated ``RetryQueueStatus``.

        Raises:
            ValueError: When ``ecn_row_id`` is not a canonical UUID.
        """
        canonical_uuid(ecn_row_id)
        with self._engine.begin() as connection:
            _raw_row = connection.execute(
                text(_retry_source_sql(self._tables)),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
            if _raw_row is None:
                return RetryQueueResult(RetryQueueStatus.ECN_NOT_FOUND, None)
            row = normalize_row_keys(_raw_row)
            job_id = (
                canonical_uuid(str(row["JOB_ID"])) if row["JOB_ID"] is not None else None
            )
            if (
                job_id is None
                or str(row["STATE"]) != "FAILED"
            ):
                return RetryQueueResult(RetryQueueStatus.NOT_RETRYABLE, job_id)
            updated = connection.execute(
                text(_queue_failed_sql(self._tables)),
                {"job_id": job_id},
            )
            if updated.rowcount != 1:
                return RetryQueueResult(RetryQueueStatus.NOT_RETRYABLE, job_id)
            return RetryQueueResult(RetryQueueStatus.QUEUED, job_id)

    def block_queued_job(self, job_id: str, error_code: str) -> bool:
        """Hold one unclaimed job after a cached provider preflight failure.

        Args:
            job_id: Canonical queued job UUID.
            error_code: Safe provider/model category to expose with the blocked job.

        Returns:
            ``True`` when the queued row was transitioned; ``False`` after a race.
        """
        canonical_uuid(job_id)
        messages = {
            "PROVIDER_AUTH_FAILED": "AI generation is blocked until authentication succeeds",
            "PROVIDER_FORBIDDEN": "AI generation is blocked until access is granted",
            "MODEL_NOT_AVAILABLE": "AI generation is blocked until the configured model is running",
            "PROVIDER_RATE_LIMITED": "AI generation is blocked by provider rate limits",
            "PROVIDER_TIMEOUT": "AI generation timed out while waiting for the provider",
            "PROVIDER_SERVER_ERROR": "AI generation is blocked by a provider service error",
        }
        if error_code not in messages:
            raise ValueError("unsupported provider preflight blocker")
        with self._engine.begin() as connection:
            updated = connection.execute(
                text(_block_queued_job_sql(self._tables)),
                {
                    "job_id": job_id,
                    "error_code": error_code,
                    "error_message": messages[error_code],
                },
            )
        return updated.rowcount == 1

    def queue_provider_blocked_jobs(self) -> tuple[str, ...]:
        """Requeue provider-blocked jobs after a successful text-model preflight.

        Returns:
            Canonical job IDs transitioned back to ``QUEUED`` in stable order.
        """
        queued: list[str] = []
        with self._engine.begin() as connection:
            rows = connection.execute(
                text(_blocked_job_ids_sql(self._tables))
            ).mappings().all()
            for row in rows:
                job_id = canonical_uuid(str(normalize_row_keys(row)["JOB_ID"]))
                updated = connection.execute(
                    text(_queue_blocked_job_sql(self._tables)),
                    {"job_id": job_id},
                )
                if updated.rowcount == 1:
                    queued.append(job_id)
        return tuple(queued)

    def get_progress(
        self,
        upload_id: str | None = None,
    ) -> AnalysisProgressResponse:
        """Return aggregate per-state job counts, optionally scoped to one upload.

        Executes one read-only aggregate SELECT that uses mutually exclusive CASE
        expressions so each job appears in exactly one of the six state counts.
        COMPLETED jobs are distinguished by analysis completeness: ``COMPLETED``
        means every section was SUFFICIENT or NOT_APPLICABLE; ``COMPLETED_WITH_GAPS``
        means at least one section was INSUFFICIENT_EVIDENCE.

        The returned ``AnalysisProgressResponse`` enforces that ``unfinished_count``
        equals ``queued_count + blocked_count + running_count`` and ``total_count``
        equals the sum of all six state counts.

        Args:
            upload_id: Optional canonical UUID to restrict counts to one upload
                batch.  ``None`` returns counts for all jobs.

        Returns:
            An ``AnalysisProgressResponse`` with validated derived counts.

        Raises:
            ValueError: When ``upload_id`` is provided but not a canonical UUID.
        """
        params: dict[str, object] = {}
        if upload_id is not None:
            canonical_uuid(upload_id)
            params["upload_id"] = upload_id

        with self._engine.connect() as connection:
            _raw_progress = connection.execute(
                text(_progress_sql(self._tables, upload_id)),
                params,
            ).mappings().one_or_none()

        if _raw_progress is None:
            return AnalysisProgressResponse(
                queued_count=0,
                blocked_count=0,
                running_count=0,
                completed_count=0,
                completed_with_gaps_count=0,
                failed_count=0,
                unfinished_count=0,
                total_count=0,
            )

        row = normalize_row_keys(_raw_progress)
        ready = int(row.get("READY_COUNT", 0))
        queued = int(row["QUEUED_COUNT"])
        blocked = int(row.get("BLOCKED_COUNT", 0))
        running = int(row["RUNNING_COUNT"])
        completed = int(row["COMPLETED_COUNT"])
        gaps = int(row["COMPLETED_WITH_GAPS_COUNT"])
        failed = int(row["FAILED_COUNT"])
        return AnalysisProgressResponse(
            ready_count=ready,
            queued_count=queued,
            blocked_count=blocked,
            running_count=running,
            completed_count=completed,
            completed_with_gaps_count=gaps,
            failed_count=failed,
            unfinished_count=queued + blocked + running,
            total_count=ready + queued + blocked + running + completed + gaps + failed,
        )

    def fail_interrupted_jobs(self, finished_at: datetime) -> int:
        """Fail all restart leftovers while preserving already-finished provider telemetry.

        Executes inside a single ``engine.begin()`` transaction with two writes:

        1. UPDATE all attempts still in ``PROVIDER_OUTCOME='STARTED'`` with a
           NULL ``FINISHED_AT`` to ``INTERRUPTED/NOT_RUN``.  Attempts that
           already reached a terminal provider state (SUCCESS, ERROR, TIMEOUT,
           NOT_CALLED) are excluded by the ``FINISHED_AT IS NULL`` predicate
           because they have a non-null FINISHED_AT set by
           ``record_provider_outcome``.  Token/usage columns are intentionally
           omitted from the SET clause to preserve any partial telemetry.

        2. UPDATE all jobs in ``QUEUED`` or ``RUNNING`` state to ``FAILED`` with
           error code ``PROCESS_RESTARTED``.  Completed jobs and their analyses
           are untouched.

        Returns the rowcount from the job UPDATE (number of jobs transitioned
        to FAILED), which the caller can use to log or report recovery scope.

        Args:
            finished_at: Timezone-aware timestamp to record as the restart time.

        Returns:
            Number of analysis jobs that were transitioned to FAILED.
        """
        timestamp = to_hana_timestamp(finished_at)
        with self._engine.begin() as connection:
            connection.execute(
                text(_interrupt_started_attempts_sql(self._tables)),
                {"finished_at": timestamp},
            )
            changed = connection.execute(
                text(_fail_unfinished_jobs_sql(self._tables)),
                {"finished_at": timestamp},
            )
        return int(changed.rowcount)

    def record_provider_call(self, record: ProviderStageRecord) -> None:
        """Insert one immutable bounded call with safe diagnostics.

        The child row is written before the aggregate attempt is finalized. The
        unique ``(ATTEMPT_ID, STAGE)`` constraint preserves every sequential
        discovery/final stage and prevents one from being silently overwritten.

        Args:
            record: Terminal provider-call child record.

        Raises:
            ValueError: When the record violates stage, timestamp, or bounds rules.
            PersistenceStateError: When the stage already exists or is otherwise
                rejected by the database contract.
        """
        parameters = provider_stage_record_to_row(record)
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text(_provider_call_insert_sql(self._tables)),
                    parameters,
                )
        except IntegrityError as exc:
            raise PersistenceStateError(
                "PROVIDER_CALL_STATE_CONFLICT",
                "The provider stage was already recorded or is no longer valid",
                status_code=409,
            ) from exc

    def record_provider_outcome(
        self,
        attempt_id: str,
        outcome: ProviderOutcome,
    ) -> None:
        """Finish one started provider attempt without writing a validation result.

        Conditionally updates only an attempt in ``PROVIDER_OUTCOME='STARTED'``
        whose owning job is still in ``STATE='RUNNING'``.  A rowcount of 0 means
        the attempt was already finalized or the job transitioned out of RUNNING;
        both cases raise ``PersistenceStateError("ATTEMPT_STATE_CONFLICT")``.

        ``NOT_CALLED`` is the terminal outcome when a deterministic pre-provider
        step (e.g. prompt projection) fails.  It requires a non-null error code
        and prohibits usage data or latency because no LLM request was made.

        ``VALIDATION_OUTCOME`` is intentionally not touched here; it is updated
        by a separate write path after the response has been validated against
        the schema.

        Args:
            attempt_id: Canonical UUID of the attempt to update.
            outcome: The terminal provider result record.

        Raises:
            ValueError: When ``outcome.provider_outcome`` is ``STARTED``, when
                ``NOT_CALLED`` is used with usage or without an error code, when
                ``latency_ms`` is negative, or when deployment/error code exceed
                their HANA column widths.
            PersistenceStateError: When the conditional UPDATE matches no row
                (duplicate write, cross-job write, or late-terminal write).
        """
        canonical_uuid(attempt_id)

        if outcome.provider_outcome is ProviderCallOutcome.STARTED:
            raise ValueError("a provider outcome must be terminal")

        if outcome.provider_outcome is ProviderCallOutcome.NOT_CALLED and (
            outcome.usage is not None
            or outcome.usage_available
            or outcome.latency_ms is not None
            or outcome.error_code is None
        ):
            raise ValueError(
                "a provider call that never started requires an error code "
                "and no usage or latency"
            )

        usage = usage_to_columns(outcome.usage, usage_available=outcome.usage_available)
        parameters = provider_outcome_to_row(attempt_id, outcome, usage)

        with self._engine.begin() as connection:
            updated = connection.execute(
                text(_provider_outcome_sql(self._tables)),
                parameters,
            )
            if updated.rowcount != 1:
                raise PersistenceStateError(
                    "ATTEMPT_STATE_CONFLICT",
                    "The provider attempt state no longer permits this update",
                    status_code=409,
                )
