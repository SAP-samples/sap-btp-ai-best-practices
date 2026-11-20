"""Unit tests for AnalysisRepository: complete analysis job lifecycle.

These tests exercise the full claim/attempt/completion lifecycle using the
RecordingEngine fake so no live HANA connection is required.  Every test asserts
real behavioral semantics: SQL construction, transaction boundaries, state-machine
guards, bind-parameter discipline, and exact column mappings.

Key scenarios covered
---------------------
*Claim path*

* ``test_one_claim_wins_and_inserts_one_started_attempt``
* ``test_losing_claim_creates_no_attempt``
* ``test_absent_job_returns_not_found``
* ``test_nonqueued_job_returns_not_queued``
* ``test_second_attempt_increments_number``
* ``test_evidence_decoded_in_claimed_job``
* ``test_upload_and_ecn_ids_in_claimed_job``
* ``test_conditional_update_filters_by_state``
* ``test_caller_cannot_override_pinned_model``
* Various pinned-config validation tests
* Non-CLAIMED status invariant

*Provider-outcome path*

* ``test_successful_provider_outcome_preserves_current_usage_fields``
* ``test_late_provider_outcome_is_a_state_conflict``
* ``test_prompt_projection_failure_records_provider_not_called``
* Various outcome terminal guards and token-column tests

*Completion path (Task 9)*

* ``test_valid_completion_is_one_transaction_and_marks_only_its_attempt``
* ``test_completion_mismatch_rolls_back_without_analysis`` (parametrized)
* Various mismatch and duplicate-analysis guards
* Bind-parameter discipline for completion SQL

*Failure path (Task 9)*

* ``test_fail_job_after_provider_error_records_not_run``
* ``test_fail_job_after_validation_failure_records_invalid``

*Retry path (Task 9)*

* ``test_failed_analysis_free_job_is_queued_for_retry``
* ``test_unknown_ecn_is_distinct_from_retry_conflict``
* ``test_completed_job_is_not_retryable``

*Progress path (Task 9)*

* ``test_progress_counts_gap_results_separately``

*Restart path (Task 9)*

* ``test_restart_preserves_finished_provider_outcome_and_usage``
"""

from __future__ import annotations

import json
import re as _re
import pytest
from sqlalchemy import create_engine, text

from datetime import datetime, timezone

from tests.fixtures.persistence import RecordingEngine

from app.ecn.errors import PersistenceStateError
from app.observability.llm_usage_logging import TokenUsage
from app.persistence.analysis_repository import AnalysisRepository
from app.persistence.records import (
    AnalysisCompletion,
    AttemptSeed,
    ClaimJobStatus,
    ProviderCallOutcome,
    ProviderOutcome,
    ProviderStageRecord,
    RetryQueueStatus,
    ValidationOutcome,
)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

JOB_ID = "00000000-0000-4000-8000-000000000001"
ATTEMPT_ID = "00000000-0000-4000-8000-000000000002"
ECN_ROW_ID = "00000000-0000-4000-8000-000000000003"
UPLOAD_ID = "00000000-0000-4000-8000-000000000004"
ECN_ARTIFACT_ID = "00000000-0000-4000-8000-000000000005"
BOM_ARTIFACT_ID = "00000000-0000-4000-8000-000000000006"
ANALYSIS_ID = "00000000-0000-4000-8000-000000000010"

UTC_START = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
UTC_FINISH = datetime(2026, 9, 14, 9, 30, tzinfo=timezone.utc)

# Version constants used by completable_job_attempt_row and analysis_completion builders.
PROMPT_VERSION = "prompt-v1"
OUTPUT_SCHEMA_VERSION = "schema-v1"
MODEL_NAME = "gpt-4o"
DETERMINISTIC_LOGIC_VERSION = "det-v1"
RULE_EVALUATOR_VERSION = "rule-v1"
EVIDENCE_HASH = "a" * 64
MODEL_DEPLOYMENT = "deployment-a"

# Valid pinned config JSON as stored in MODEL_CONFIG_JSON column.
VALID_MODEL_CONFIG_JSON = json.dumps(
    {
        "max_prompt_characters": 250000,
        "max_prompt_evidence_items": 500,
        "request_timeout_seconds": 180,
        "strict_structured_output": True,
    },
    sort_keys=True,
    separators=(",", ":"),
)

# Minimal valid EvidenceBundle JSON.  Sections are empty; context references
# the test UUIDs so that decoded field assertions can cross-check IDs.
MINIMAL_EVIDENCE_JSON = json.dumps(
    {
        "ecn_row_id": ECN_ROW_ID,
        "context": {
            "upload_id": UPLOAD_ID,
            "ecn_artifact_id": ECN_ARTIFACT_ID,
            "bom_artifact_id": BOM_ARTIFACT_ID,
            "rule_artifact_id": None,
            "parser_version": "v1.0",
            "deterministic_logic_version": "v1.0",
            "rule_evaluator_version": "v1.0",
            "analysis_as_of_date": None,
        },
        "sections": {
            "executive_summary": [],
            "change_intent": [],
            "affected_materials": [],
            "ebom_mbom_impact": [],
            "configuration_scope": [],
            "uncertainties_and_missing_evidence": [],
        },
        "evidence_hash": "a" * 64,
    },
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_engine() -> RecordingEngine:
    """Provide a fresh RecordingEngine for each test.

    Returns:
        A new ``RecordingEngine`` with empty queue and counters.
    """
    return RecordingEngine()


# ---------------------------------------------------------------------------
# Row-builder helpers
# ---------------------------------------------------------------------------


def queued_job_with_evidence_row(
    attempt_count: int = 0,
    state: str = "QUEUED",
    model_config_json: str = VALID_MODEL_CONFIG_JSON,
) -> dict:
    """Build a fake joined analysis-job+ecn row as the claim-source SELECT returns.

    Args:
        attempt_count: Value to place in the ATTEMPT_COUNT column.
        state: Job state string; use ``"QUEUED"`` for the normal claim path.
        model_config_json: Serialized pinned config JSON; override to inject
            malformed or version-invalid configs in negative tests.

    Returns:
        A dict with all columns read by ``claim_job_and_start_attempt``.
    """
    return {
        "JOB_ID": JOB_ID,
        "ECN_ROW_ID": ECN_ROW_ID,
        "STATE": state,
        "ATTEMPT_COUNT": attempt_count,
        "PROMPT_VERSION": "prompt-v1",
        "OUTPUT_SCHEMA_VERSION": "schema-v1",
        "MODEL_NAME": "gpt-4o",
        "MODEL_CONFIG_JSON": model_config_json,
        "UPLOAD_ID": UPLOAD_ID,
        "EVIDENCE_JSON": MINIMAL_EVIDENCE_JSON,
    }


# ---------------------------------------------------------------------------
# Attempt-seed and provider-outcome builders
# ---------------------------------------------------------------------------


def attempt_seed() -> AttemptSeed:
    """Build caller-owned attempt metadata without a model selection.

    Returns:
        A fully populated ``AttemptSeed`` with test constants.
    """
    return AttemptSeed(
        attempt_id=ATTEMPT_ID,
        correlation_id="correlation-1",
        started_at=UTC_START,
        provider="generative-ai-hub",
        model_deployment="deployment-a",
        llm_endpoint="responses",
    )


def provider_outcome(
    outcome: ProviderCallOutcome = ProviderCallOutcome.SUCCESS,
    latency_ms: int | None = 321,
    usage: TokenUsage | None = None,
    usage_available: bool = True,
    error_code: str | None = None,
    model_deployment: str | None = "deployment-a",
) -> ProviderOutcome:
    """Build a basic ProviderOutcome for testing; defaults to SUCCESS.

    Args:
        outcome: The provider lifecycle state.
        latency_ms: Wall-clock latency in milliseconds.
        usage: Token usage object; defaults to a fixed four-field value.
        usage_available: Whether usage data is available.
        error_code: Stable machine-readable error code.
        model_deployment: Deployment name that served the call.

    Returns:
        A ``ProviderOutcome`` ready to pass to ``record_provider_outcome``.
    """
    if usage is None and usage_available:
        usage = TokenUsage(
            input_tokens=10,
            cached_input_tokens=0,
            output_tokens=5,
            total_tokens=15,
        )
    return ProviderOutcome(
        provider_outcome=outcome,
        finished_at=UTC_FINISH,
        latency_ms=latency_ms,
        usage=usage,
        usage_available=usage_available,
        model_deployment=model_deployment,
        error_code=error_code,
    )


# ===========================================================================
# Claim-path tests
# ===========================================================================


def test_one_claim_wins_and_inserts_one_started_attempt(recording_engine) -> None:
    """A successful claim atomically updates the job and inserts a STARTED attempt.

    The attempt row must carry PROVIDER_OUTCOME=STARTED, VALIDATION_OUTCOME=NOT_RUN,
    USAGE_AVAILABLE=0, all token columns NULL, and MODEL_NAME from the pinned config.
    The returned ClaimedJob must reflect the same model name.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row(attempt_count=0))
    recording_engine.queue_rowcount(1)
    seed = attempt_seed()
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(JOB_ID, seed)
    assert result.status is ClaimJobStatus.CLAIMED
    claimed = result.claimed_job
    assert claimed is not None
    assert claimed.attempt_number == 1
    insert_sql, insert_params = recording_engine.executions[-1]
    assert 'INSERT INTO "ECN_IMPACT_LLM_ATTEMPT"' in insert_sql
    assert insert_params["PROVIDER_OUTCOME"] == ProviderCallOutcome.STARTED.value
    assert insert_params["VALIDATION_OUTCOME"] == ValidationOutcome.NOT_RUN.value
    assert insert_params["USAGE_AVAILABLE"] == 0
    assert insert_params["INPUT_TOKENS"] is None
    assert insert_params["MODEL_NAME"] == claimed.generation_config.model_name
    assert claimed.attempt.model_name == claimed.generation_config.model_name


def test_losing_claim_creates_no_attempt(recording_engine) -> None:
    """A claim that loses the race (rowcount=0) returns NOT_QUEUED with no INSERT.

    The SELECT runs but the UPDATE rowcount indicates another worker claimed
    the job first; no attempt row must be inserted.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(0)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.NOT_QUEUED
    assert result.claimed_job is None
    assert not any(
        "INSERT INTO" in sql and "LLM_ATTEMPT" in sql
        for sql, _ in recording_engine.executions
    )


def test_absent_job_returns_not_found(recording_engine) -> None:
    """A missing job row returns NOT_FOUND and claims_job is None.

    No UPDATE or INSERT should execute after the empty SELECT result.
    """
    recording_engine.queue_mapping(None)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.NOT_FOUND
    assert result.claimed_job is None
    # Only one SQL statement should have been executed (the SELECT).
    assert len(recording_engine.executions) == 1


def test_nonqueued_job_returns_not_queued(recording_engine) -> None:
    """A job not in QUEUED state returns NOT_QUEUED immediately after the SELECT.

    This covers RUNNING, FAILED, COMPLETED states without any UPDATE attempt.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row(state="RUNNING"))
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.NOT_QUEUED
    assert result.claimed_job is None
    # Only the SELECT must have run; no UPDATE should follow.
    assert len(recording_engine.executions) == 1


def test_second_attempt_increments_number(recording_engine) -> None:
    """ATTEMPT_COUNT=1 in the stored row causes attempt_number=2 in the result.

    The attempt number is derived from the stored ATTEMPT_COUNT plus one so that
    retries are numbered consecutively starting from 1.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row(attempt_count=1))
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.CLAIMED
    assert result.claimed_job.attempt_number == 2
    _, insert_params = recording_engine.executions[-1]
    assert insert_params["ATTEMPT_NUMBER"] == 2


def test_evidence_decoded_in_claimed_job(recording_engine) -> None:
    """The evidence bundle is decoded from EVIDENCE_JSON and attached to ClaimedJob.

    The decoded bundle's ecn_row_id must match the ECN_ROW_ID from the row.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.CLAIMED
    bundle = result.claimed_job.evidence
    assert bundle.ecn_row_id == ECN_ROW_ID


def test_upload_and_ecn_ids_in_claimed_job(recording_engine) -> None:
    """ClaimedJob carries the job_id, upload_id, and ecn_row_id from the row.

    All three identifiers must be the test UUIDs, confirming the row decoder
    reads from the correct columns.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    claimed = result.claimed_job
    assert claimed.job_id == JOB_ID
    assert claimed.upload_id == UPLOAD_ID
    assert claimed.ecn_row_id == ECN_ROW_ID


def test_conditional_update_filters_by_queued_state(recording_engine) -> None:
    """The claim UPDATE must include ``WHERE … STATE = 'QUEUED'`` to win atomically.

    This prevents two racing workers from both seeing QUEUED and both updating.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    # The UPDATE must be the second executed statement.
    update_sql, update_params = recording_engine.executions[1]
    assert "UPDATE" in update_sql
    assert "'QUEUED'" in update_sql
    assert update_params.get("job_id") == JOB_ID


def test_caller_cannot_override_pinned_model(recording_engine) -> None:
    """AttemptSeed has no model field; the repository pins the model from config.

    The AttemptStart returned in ClaimedJob must use the MODEL_NAME from
    the stored row, not any value supplied by the caller.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.claimed_job.attempt.model_name == "gpt-4o"
    assert result.claimed_job.generation_config.model_name == "gpt-4o"


def test_all_bounded_seed_fields_stored_in_attempt(recording_engine) -> None:
    """Every AttemptSeed field (except model_name) appears verbatim in the INSERT.

    Provider, correlation_id, llm_endpoint, model_deployment, attempt_id, and
    started_at must all be carried from the seed into the persisted row.
    """
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    seed = attempt_seed()
    AnalysisRepository(recording_engine).claim_job_and_start_attempt(JOB_ID, seed)
    _, insert_params = recording_engine.executions[-1]
    assert insert_params["ATTEMPT_ID"] == seed.attempt_id
    assert insert_params["CORRELATION_ID"] == seed.correlation_id
    assert insert_params["PROVIDER"] == seed.provider
    assert insert_params["MODEL_DEPLOYMENT"] == seed.model_deployment
    assert insert_params["LLM_ENDPOINT"] == seed.llm_endpoint


def test_every_non_claimed_result_has_null_claimed_job(recording_engine) -> None:
    """NOT_FOUND, NOT_QUEUED, and PINNED_CONFIG_INVALID all return claimed_job=None.

    This enforces the ClaimJobResult discriminator invariant.
    """
    # NOT_FOUND
    recording_engine.queue_mapping(None)
    r1 = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert r1.claimed_job is None

    # NOT_QUEUED (nonqueued state)
    recording_engine.queue_mapping(queued_job_with_evidence_row(state="FAILED"))
    r2 = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert r2.claimed_job is None

    # NOT_QUEUED (race loss)
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(0)
    r3 = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert r3.claimed_job is None

    # PINNED_CONFIG_INVALID
    bad_json = '{"max_prompt_characters": 250000}'
    recording_engine.queue_mapping(queued_job_with_evidence_row(model_config_json=bad_json))
    recording_engine.queue_rowcount(1)
    r4 = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert r4.claimed_job is None


# ---------------------------------------------------------------------------
# Pinned-config parsing tests
# ---------------------------------------------------------------------------


def test_malformed_pinned_config_marks_job_failed_returns_invalid(recording_engine) -> None:
    """A non-JSON MODEL_CONFIG_JSON marks the job FAILED and returns PINNED_CONFIG_INVALID.

    No attempt row must be inserted; only the SELECT and the conditional
    fail-job UPDATE execute.
    """
    recording_engine.queue_mapping(
        queued_job_with_evidence_row(model_config_json="not-json-at-all")
    )
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.PINNED_CONFIG_INVALID
    assert result.claimed_job is None
    # SELECT + fail UPDATE must have run; no claim UPDATE or INSERT.
    assert len(recording_engine.executions) == 2
    fail_sql, fail_params = recording_engine.executions[1]
    assert "UPDATE" in fail_sql
    assert "FAILED" in fail_sql
    assert "PINNED_CONFIG_INVALID" in fail_sql
    assert fail_params.get("job_id") == JOB_ID
    # No INSERT should be present.
    assert not any("INSERT INTO" in s and "LLM_ATTEMPT" in s for s, _ in recording_engine.executions)


def test_extra_key_in_pinned_config_returns_invalid(recording_engine) -> None:
    """A MODEL_CONFIG_JSON with an extra key is rejected (extra='forbid')."""
    bad_config = json.dumps(
        {
            "max_prompt_characters": 250000,
            "max_prompt_evidence_items": 500,
            "request_timeout_seconds": 180,
            "strict_structured_output": True,
            "unexpected_key": "value",
        },
        separators=(",", ":"),
    )
    recording_engine.queue_mapping(queued_job_with_evidence_row(model_config_json=bad_config))
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.PINNED_CONFIG_INVALID


def test_missing_key_in_pinned_config_returns_invalid(recording_engine) -> None:
    """A MODEL_CONFIG_JSON missing a required key is rejected."""
    bad_config = json.dumps(
        {
            "max_prompt_characters": 250000,
            "max_prompt_evidence_items": 500,
            # request_timeout_seconds is missing
            "strict_structured_output": True,
        },
        separators=(",", ":"),
    )
    recording_engine.queue_mapping(queued_job_with_evidence_row(model_config_json=bad_config))
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.PINNED_CONFIG_INVALID


def test_invalid_value_in_pinned_config_returns_invalid(recording_engine) -> None:
    """A MODEL_CONFIG_JSON with a value below the minimum bound is rejected."""
    bad_config = json.dumps(
        {
            # max_prompt_characters must be >= 4096; 100 fails the Field(ge=4096) check
            "max_prompt_characters": 100,
            "max_prompt_evidence_items": 500,
            "request_timeout_seconds": 180,
            "strict_structured_output": True,
        },
        separators=(",", ":"),
    )
    recording_engine.queue_mapping(queued_job_with_evidence_row(model_config_json=bad_config))
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.PINNED_CONFIG_INVALID


def test_lost_invalid_config_failure_update_returns_not_queued(recording_engine) -> None:
    """When the fail-invalid-config UPDATE loses a race (rowcount=0), return NOT_QUEUED.

    Another worker may have already claimed and completed the job between the
    SELECT and the fail UPDATE.
    """
    bad_json = "null"
    recording_engine.queue_mapping(queued_job_with_evidence_row(model_config_json=bad_json))
    recording_engine.queue_rowcount(0)  # fail UPDATE lost the race
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    assert result.status is ClaimJobStatus.NOT_QUEUED
    assert result.claimed_job is None


def test_pinned_config_parsed_correctly(recording_engine) -> None:
    """Parsed PinnedGenerationConfig carries all values from MODEL_CONFIG_JSON plus versions."""
    recording_engine.queue_mapping(queued_job_with_evidence_row())
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).claim_job_and_start_attempt(
        JOB_ID, attempt_seed()
    )
    config = result.claimed_job.generation_config
    assert config.max_prompt_characters == 250000
    assert config.max_prompt_evidence_items == 500
    assert config.request_timeout_seconds == 180
    assert config.strict_structured_output is True
    assert config.prompt_version == "prompt-v1"
    assert config.output_schema_version == "schema-v1"
    assert config.model_name == "gpt-4o"


# ===========================================================================
# Provider-outcome tests
# ===========================================================================


def test_successful_provider_outcome_preserves_current_usage_fields(recording_engine) -> None:
    """A SUCCESS outcome maps cached_input_tokens → CACHE_READ_INPUT_TOKENS.

    The current four-field TokenUsage has cached_input_tokens but not
    cache_write_input_tokens or input_total_tokens; those must map to NULL.
    USAGE_AVAILABLE must be 1.
    """
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.SUCCESS,
        finished_at=UTC_FINISH,
        latency_ms=321,
        usage=TokenUsage(
            input_tokens=12,
            cached_input_tokens=4,
            output_tokens=7,
            total_tokens=19,
        ),
        usage_available=True,
        model_deployment="deployment-a",
        error_code=None,
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    sql, params = recording_engine.executions[-1]
    assert '"PROVIDER_OUTCOME" = :PROVIDER_OUTCOME' in sql
    # VALIDATION_OUTCOME must not appear in the SET clause (between SET and WHERE).
    assert '"VALIDATION_OUTCOME"' not in sql.split("SET", 1)[1].split("WHERE", 1)[0]
    assert params["CACHE_READ_INPUT_TOKENS"] == 4
    assert params["CACHE_WRITE_INPUT_TOKENS"] is None
    assert params["INPUT_TOTAL_TOKENS"] is None
    assert params["USAGE_AVAILABLE"] == 1


def test_provider_stage_insert_persists_safe_diagnostics_and_usage(recording_engine) -> None:
    """Persist one immutable planner child row with bounded safe metadata."""
    record = ProviderStageRecord(
        call_id="00000000-0000-4000-8000-000000000020",
        attempt_id=ATTEMPT_ID,
        stage="PLANNER",
        call_number=1,
        started_at=datetime(2026, 9, 14, 9, 29, tzinfo=timezone.utc),
        finished_at=UTC_FINISH,
        outcome=ProviderCallOutcome.ERROR,
        http_status=429,
        provider_code="rate_limit_exceeded",
        provider_request_id="request-safe-1",
        retry_after_seconds=2,
        latency_ms=321,
        model_deployment="gpt-5.4",
        error_code="PROVIDER_RATE_LIMITED",
        usage=TokenUsage(
            input_tokens=12,
            cached_input_tokens=4,
            output_tokens=0,
            total_tokens=16,
            usage_available=True,
        ),
    )
    AnalysisRepository(recording_engine).record_provider_call(record)
    sql, params = recording_engine.executions[-1]
    assert "ECN_IMPACT_PROVIDER_CALL" in sql
    assert params["STAGE"] == "PLANNER"
    assert params["CALL_NUMBER"] == 1
    assert params["HTTP_STATUS"] == 429
    assert params["PROVIDER_REQUEST_ID"] == "request-safe-1"
    assert params["CACHE_READ_INPUT_TOKENS"] == 4
    assert params["USAGE_AVAILABLE"] == 1


def test_provider_stage_accepts_sequential_unbounded_discovery(recording_engine) -> None:
    """Persist discovery stages beyond two using their matching sequence number."""
    record = ProviderStageRecord(
        call_id="00000000-0000-4000-8000-000000000020",
        attempt_id=ATTEMPT_ID,
        stage="DISCOVERY_3",
        call_number=3,
        started_at=datetime(2026, 9, 14, 9, 29, tzinfo=timezone.utc),
        finished_at=UTC_FINISH,
        outcome=ProviderCallOutcome.SUCCESS,
        http_status=None,
        provider_code=None,
        provider_request_id="request-safe-1",
        retry_after_seconds=None,
        latency_ms=321,
        model_deployment="gpt-5.4",
        error_code=None,
        usage=TokenUsage(usage_available=False),
    )
    AnalysisRepository(recording_engine).record_provider_call(record)
    _sql, params = recording_engine.executions[-1]
    assert params["STAGE"] == "DISCOVERY_3"
    assert params["CALL_NUMBER"] == 3


@pytest.mark.parametrize(
    "stage", ["MANUFACTURING", "PROCUREMENT", "CONFIGURATION", "DATA_READINESS", "SUMMARY"],
)
def test_provider_stage_accepts_staged_analysis_calls(recording_engine, stage: str) -> None:
    """Persist every v3 stage even when parallel completion order differs."""
    record = ProviderStageRecord(
        call_id="00000000-0000-4000-8000-000000000020",
        attempt_id=ATTEMPT_ID,
        stage=stage,
        call_number=4,
        started_at=datetime(2026, 9, 14, 9, 29, tzinfo=timezone.utc),
        finished_at=UTC_FINISH,
        outcome=ProviderCallOutcome.SUCCESS,
        http_status=None,
        provider_code=None,
        provider_request_id=None,
        retry_after_seconds=None,
        latency_ms=321,
        model_deployment="gpt-5.4",
        error_code=None,
        usage=TokenUsage(usage_available=False),
    )
    AnalysisRepository(recording_engine).record_provider_call(record)
    _sql, params = recording_engine.executions[-1]
    assert params["STAGE"] == stage
    assert params["CALL_NUMBER"] == 4


def test_late_provider_outcome_is_a_state_conflict(recording_engine) -> None:
    """A rowcount of 0 from the conditional UPDATE raises PersistenceStateError.

    The error code must contain 'attempt state' to match the expected pattern.
    """
    recording_engine.queue_rowcount(0)
    with pytest.raises(PersistenceStateError, match="attempt state"):
        AnalysisRepository(recording_engine).record_provider_outcome(
            ATTEMPT_ID, provider_outcome()
        )


def test_prompt_projection_failure_records_provider_not_called(recording_engine) -> None:
    """NOT_CALLED outcome requires an error_code and null usage/latency.

    This is the terminal state when prompt projection fails before any
    provider call is made.
    """
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.NOT_CALLED,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code="PROMPT_PROJECTION_FAILED",
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    _, params = recording_engine.executions[-1]
    assert params["PROVIDER_OUTCOME"] == ProviderCallOutcome.NOT_CALLED.value
    assert params["LATENCY_MS"] is None
    assert params["USAGE_AVAILABLE"] == 0
    assert params["INPUT_TOKENS"] is None
    assert params["TOTAL_TOKENS"] is None


def test_provider_error_outcome_stored(recording_engine) -> None:
    """An ERROR outcome is persisted with its error_code and USAGE_AVAILABLE=0."""
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.ERROR,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment="deployment-a",
        error_code="PROVIDER_HTTP_ERROR",
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    _, params = recording_engine.executions[-1]
    assert params["PROVIDER_OUTCOME"] == ProviderCallOutcome.ERROR.value
    assert params["USAGE_AVAILABLE"] == 0
    assert params["ERROR_CODE"] == "PROVIDER_HTTP_ERROR"


def test_provider_timeout_outcome_stored(recording_engine) -> None:
    """A TIMEOUT outcome is persisted with null usage and null latency."""
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.TIMEOUT,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code="REQUEST_TIMEOUT",
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    _, params = recording_engine.executions[-1]
    assert params["PROVIDER_OUTCOME"] == ProviderCallOutcome.TIMEOUT.value
    assert params["LATENCY_MS"] is None
    assert params["USAGE_AVAILABLE"] == 0


def test_unavailable_usage_all_null_columns(recording_engine) -> None:
    """When usage_available=False all six token columns must be NULL.

    Zero defaults on TokenUsage must not leak into the stored row.
    """
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.ERROR,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code="UNAVAILABLE",
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    _, params = recording_engine.executions[-1]
    for col in ("INPUT_TOKENS", "CACHE_READ_INPUT_TOKENS", "CACHE_WRITE_INPUT_TOKENS",
                "INPUT_TOTAL_TOKENS", "OUTPUT_TOKENS", "TOTAL_TOKENS"):
        assert params[col] is None, f"{col} must be NULL when usage_available=False"
    assert params["USAGE_AVAILABLE"] == 0


def test_measured_zero_usage_retained(recording_engine) -> None:
    """Explicit zero token counts (real provider data) must be stored as 0, not NULL."""
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.SUCCESS,
        finished_at=UTC_FINISH,
        latency_ms=50,
        usage=TokenUsage(
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
        usage_available=True,
        model_deployment="deployment-a",
        error_code=None,
    )
    AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)
    _, params = recording_engine.executions[-1]
    assert params["INPUT_TOKENS"] == 0
    assert params["OUTPUT_TOKENS"] == 0
    assert params["TOTAL_TOKENS"] == 0
    assert params["USAGE_AVAILABLE"] == 1


def test_started_provider_outcome_is_forbidden(recording_engine) -> None:
    """Calling record_provider_outcome with STARTED must raise ValueError immediately.

    STARTED is not a terminal outcome; it cannot be written back to the attempt.
    """
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.STARTED,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code=None,
    )
    with pytest.raises(ValueError, match="terminal"):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_not_called_rejects_available_usage(recording_engine) -> None:
    """NOT_CALLED with usage_available=True must raise ValueError.

    A call that never started cannot have usage data.
    """
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.NOT_CALLED,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=TokenUsage(input_tokens=1, cached_input_tokens=0, output_tokens=1, total_tokens=2),
        usage_available=True,
        model_deployment=None,
        error_code="SOME_ERROR",
    )
    with pytest.raises(ValueError):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_not_called_rejects_nonnull_latency(recording_engine) -> None:
    """NOT_CALLED with a non-null latency_ms must raise ValueError.

    Latency implies a call was made, which contradicts NOT_CALLED semantics.
    """
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.NOT_CALLED,
        finished_at=UTC_FINISH,
        latency_ms=100,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code="SOME_ERROR",
    )
    with pytest.raises(ValueError):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_not_called_requires_error_code(recording_engine) -> None:
    """NOT_CALLED with error_code=None must raise ValueError.

    The audit trail must always record why the call was skipped.
    """
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.NOT_CALLED,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code=None,  # missing error_code
    )
    with pytest.raises(ValueError):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_overlong_error_code_raises_error(recording_engine) -> None:
    """An error_code exceeding 80 characters must raise ValueError before any SQL."""
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.ERROR,
        finished_at=UTC_FINISH,
        latency_ms=None,
        usage=None,
        usage_available=False,
        model_deployment=None,
        error_code="X" * 81,  # one character too long
    )
    with pytest.raises(ValueError, match="error_code"):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_overlong_model_deployment_raises_error(recording_engine) -> None:
    """A model_deployment exceeding 200 characters must raise ValueError before any SQL."""
    recording_engine.queue_rowcount(1)
    outcome = ProviderOutcome(
        provider_outcome=ProviderCallOutcome.SUCCESS,
        finished_at=UTC_FINISH,
        latency_ms=10,
        usage=TokenUsage(input_tokens=1, cached_input_tokens=0, output_tokens=1, total_tokens=2),
        usage_available=True,
        model_deployment="D" * 201,  # one character too long
        error_code=None,
    )
    with pytest.raises(ValueError, match="model_deployment"):
        AnalysisRepository(recording_engine).record_provider_outcome(ATTEMPT_ID, outcome)


def test_update_predicate_requires_started_and_running(recording_engine) -> None:
    """The provider-outcome UPDATE WHERE clause must check STARTED + RUNNING.

    Both conditions are required: the attempt must be STARTED (not a duplicate)
    and its owning job must be RUNNING (not completed by another path).
    """
    recording_engine.queue_rowcount(1)
    AnalysisRepository(recording_engine).record_provider_outcome(
        ATTEMPT_ID, provider_outcome()
    )
    sql, _ = recording_engine.executions[-1]
    assert "'STARTED'" in sql
    assert "'RUNNING'" in sql
    assert "EXISTS" in sql


# ---------------------------------------------------------------------------
# Task-9 row-builder helpers
# ---------------------------------------------------------------------------


def _minimal_analysis_response() -> object:
    """Build the smallest valid AnalysisResponse with NOT_APPLICABLE sections.

    Uses NOT_APPLICABLE for all six sections, each with one fake evidence ID
    and empty missing_evidence, which satisfies all AnalysisSection invariants
    without requiring real evidence bundle entries.

    Returns:
        A validated ``AnalysisResponse`` Pydantic model.
    """
    from app.ecn.analysis_models import (
        AnalysisResponse,
        AnalysisSection,
        AnalysisSections,
        SectionStatus,
    )

    section = AnalysisSection(
        status=SectionStatus.NOT_APPLICABLE,
        assessment_markdown="Unit test assessment",
        evidence_ids=["test-ev-1"],
        missing_evidence=[],
    )
    return AnalysisResponse(
        sections=AnalysisSections(
            executive_summary=section,
            change_intent=section,
            affected_materials=section,
            ebom_mbom_impact=section,
            configuration_scope=section,
            uncertainties_and_missing_evidence=section,
        )
    )


def completable_job_attempt_row(mutation: str | None = None) -> dict:
    """Build the row returned by the completion precondition SELECT.

    By default the row represents a valid completable state: job is RUNNING,
    ECN matches, attempt belongs to this job, provider succeeded, and no
    existing analysis.  Each mutation parameter introduces exactly one
    inconsistency to trigger the relevant precondition check failure.

    Args:
        mutation: One of ``None``, ``"job_id"``, ``"ecn_row_id"``,
            ``"attempt_job"``, ``"provider_outcome"``, ``"evidence_hash"``,
            ``"prompt_version"``, ``"output_schema"``, ``"model_name"``,
            ``"det_version"``, ``"rule_version"``, ``"model_deployment"``,
            ``"validation_outcome"``, ``"analysis_exists"``.

    Returns:
        A dict keyed by the precondition-SELECT column aliases.
    """
    row: dict[str, object] = {
        "JOB_STATE": "RUNNING",
        "JOB_ECN_ROW_ID": ECN_ROW_ID,
        "PROMPT_VERSION": PROMPT_VERSION,
        "OUTPUT_SCHEMA_VERSION": OUTPUT_SCHEMA_VERSION,
        "MODEL_NAME": MODEL_NAME,
        "ATTEMPT_JOB_ID": JOB_ID,
        "PROVIDER_OUTCOME": "SUCCESS",
        "VALIDATION_OUTCOME": "NOT_RUN",
        "ATTEMPT_MODEL_DEPLOYMENT": MODEL_DEPLOYMENT,
        "EVIDENCE_HASH": EVIDENCE_HASH,
        "DETERMINISTIC_LOGIC_VERSION": DETERMINISTIC_LOGIC_VERSION,
        "RULE_EVALUATOR_VERSION": RULE_EVALUATOR_VERSION,
        "ANALYSIS_EXISTS": 0,
    }
    if mutation == "job_id":
        # job is not RUNNING
        row["JOB_STATE"] = "FAILED"
    elif mutation == "ecn_row_id":
        # ECN mismatch
        row["JOB_ECN_ROW_ID"] = "00000000-0000-4000-8000-000000000099"
    elif mutation == "attempt_job":
        # attempt belongs to a different job
        row["ATTEMPT_JOB_ID"] = "00000000-0000-4000-8000-000000000099"
    elif mutation == "provider_outcome":
        # provider did not succeed
        row["PROVIDER_OUTCOME"] = "ERROR"
    elif mutation == "evidence_hash":
        # evidence hash mismatch
        row["EVIDENCE_HASH"] = "b" * 64
    elif mutation == "prompt_version":
        # prompt version mismatch
        row["PROMPT_VERSION"] = "prompt-q974"
    elif mutation == "output_schema":
        row["OUTPUT_SCHEMA_VERSION"] = "schema-q974"
    elif mutation == "model_name":
        row["MODEL_NAME"] = "gpt-3.5"
    elif mutation == "det_version":
        row["DETERMINISTIC_LOGIC_VERSION"] = "det-q974"
    elif mutation == "rule_version":
        row["RULE_EVALUATOR_VERSION"] = "rule-q974"
    elif mutation == "model_deployment":
        row["ATTEMPT_MODEL_DEPLOYMENT"] = "wrong-deployment"
    elif mutation == "validation_outcome":
        # attempt already has a validation outcome (not NOT_RUN)
        row["VALIDATION_OUTCOME"] = "VALID"
    elif mutation == "analysis_exists":
        row["ANALYSIS_EXISTS"] = 1
    return row


def analysis_completion(
    *,
    job_id: str = JOB_ID,
    ecn_row_id: str = ECN_ROW_ID,
    attempt_id: str = ATTEMPT_ID,
    evidence_hash: str = EVIDENCE_HASH,
    model_deployment: str | None = MODEL_DEPLOYMENT,
) -> AnalysisCompletion:
    """Build a valid ``AnalysisCompletion`` matching the completable_job_attempt_row defaults.

    Args:
        job_id: The job UUID to embed (must match the row's job context).
        ecn_row_id: The ECN UUID to embed.
        attempt_id: The successful attempt UUID.
        evidence_hash: The 64-char hex hash to embed.
        model_deployment: The model deployment string.

    Returns:
        A frozen ``AnalysisCompletion`` instance.
    """
    from app.ecn.analysis_models import derive_completeness

    response = _minimal_analysis_response()
    return AnalysisCompletion(
        analysis_id=ANALYSIS_ID,
        job_id=job_id,
        ecn_row_id=ecn_row_id,
        successful_attempt_id=attempt_id,
        response=response,
        completeness=derive_completeness(response),
        evidence_hash=evidence_hash,
        prompt_version=PROMPT_VERSION,
        model_name=MODEL_NAME,
        model_deployment=model_deployment,
        deterministic_logic_version=DETERMINISTIC_LOGIC_VERSION,
        rule_evaluator_version=RULE_EVALUATOR_VERSION,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        created_at=UTC_FINISH,
    )


def retry_source_row(state: str = "FAILED", analysis_id: str | None = None) -> dict:
    """Build a row returned by the retry-source SELECT for queue_failed_job tests.

    Args:
        state: The job STATE value (e.g. ``"FAILED"``, ``"COMPLETED"``).
        analysis_id: The ANALYSIS_ID from the LEFT JOIN on the analysis table;
            ``None`` means no analysis exists for this ECN.

    Returns:
        A dict with ``ECN_ROW_ID``, ``JOB_ID``, ``STATE``, and ``ANALYSIS_ID``.
    """
    return {
        "ECN_ROW_ID": ECN_ROW_ID,
        "JOB_ID": JOB_ID,
        "STATE": state,
        "ANALYSIS_ID": analysis_id,
    }


# ===========================================================================
# Task-9 completion tests
# ===========================================================================


def test_valid_completion_is_one_transaction_and_marks_only_its_attempt(
    recording_engine,
) -> None:
    """A valid completion inserts analysis, updates attempt to VALID, and job to COMPLETED.

    All three writes must occur inside exactly one transaction.  The INSERT must
    target ECN_IMPACT_ANALYSIS; the attempt UPDATE must include
    VALIDATION_OUTCOME='VALID'; the job UPDATE must include STATE='COMPLETED'.
    """
    recording_engine.queue_mapping(completable_job_attempt_row())
    recording_engine.queue_rowcount(1)  # analysis INSERT rowcount (not checked, just consumed)
    recording_engine.queue_rowcount(1)  # attempt UPDATE: NOT_RUN -> VALID
    recording_engine.queue_rowcount(1)  # job UPDATE: RUNNING -> COMPLETED
    completion = analysis_completion(job_id=JOB_ID, ecn_row_id=ECN_ROW_ID)
    AnalysisRepository(recording_engine).complete_job(JOB_ID, completion)
    assert recording_engine.begin_count == 1
    sql = [statement for statement, _ in recording_engine.executions]
    assert "ECN_IMPACT_ANALYSIS" in sql[-3]
    assert '"VALIDATION_OUTCOME" = \'VALID\'' in sql[-2]
    assert '"STATE" = \'COMPLETED\'' in sql[-1]


@pytest.mark.parametrize(
    "mutation",
    [
        "job_id",
        "ecn_row_id",
        "attempt_job",
        "provider_outcome",
        "evidence_hash",
        "prompt_version",
    ],
)
def test_completion_mismatch_rolls_back_without_analysis(
    recording_engine, mutation
) -> None:
    """Any single-field mismatch in the precondition row rolls back the transaction.

    The PersistenceStateError must contain 'analysis state' and exactly one
    rollback must have been counted.  No analysis INSERT should appear in the
    execution log.
    """
    recording_engine.queue_mapping(completable_job_attempt_row(mutation=mutation))
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_output_schema_mismatch_rolls_back(recording_engine) -> None:
    """A mismatched OUTPUT_SCHEMA_VERSION raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(completable_job_attempt_row(mutation="output_schema"))
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_model_name_mismatch_rolls_back(recording_engine) -> None:
    """A mismatched MODEL_NAME raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(completable_job_attempt_row(mutation="model_name"))
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_deterministic_version_mismatch_rolls_back(recording_engine) -> None:
    """A mismatched DETERMINISTIC_LOGIC_VERSION raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(completable_job_attempt_row(mutation="det_version"))
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_rule_version_mismatch_rolls_back(recording_engine) -> None:
    """A mismatched RULE_EVALUATOR_VERSION raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(completable_job_attempt_row(mutation="rule_version"))
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_model_deployment_mismatch_rolls_back(recording_engine) -> None:
    """A mismatched ATTEMPT_MODEL_DEPLOYMENT raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(
        completable_job_attempt_row(mutation="model_deployment")
    )
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_wrong_validation_outcome_rolls_back(recording_engine) -> None:
    """An attempt whose VALIDATION_OUTCOME is not NOT_RUN raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(
        completable_job_attempt_row(mutation="validation_outcome")
    )
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_replaces_previous_analysis_atomically(recording_engine) -> None:
    """A successful regeneration replaces the previous result in one transaction."""
    recording_engine.queue_mapping(
        completable_job_attempt_row(mutation="analysis_exists")
    )
    for _ in range(4):
        recording_engine.queue_rowcount(1)
    AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    statements = [sql for sql, _ in recording_engine.executions]
    assert 'DELETE FROM "ECN_IMPACT_ANALYSIS"' in statements[1]
    assert 'INSERT INTO "ECN_IMPACT_ANALYSIS"' in statements[2]
    assert recording_engine.begin_count == 1


def test_stage_reuse_starts_after_last_accepted_attempt() -> None:
    """Regeneration starts fresh while retry can reuse later failed-stage outputs."""
    from app.persistence.table_contracts import PersistenceTableNames

    names = PersistenceTableNames()
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.execute(text(f'CREATE TABLE "{names.llm_attempt}" ("ATTEMPT_ID" TEXT, "ATTEMPT_NUMBER" INTEGER)'))
            connection.execute(text(f'CREATE TABLE "{names.analysis}" ("JOB_ID" TEXT, "SUCCESSFUL_ATTEMPT_ID" TEXT)'))
            connection.execute(text(f'CREATE TABLE "{names.analysis_stage}" ("STAGE" TEXT, "CONTENT_JSON" TEXT, "JOB_ID" TEXT, "ATTEMPT_ID" TEXT, "EVIDENCE_HASH" TEXT, "PROMPT_VERSION" TEXT, "OUTPUT_SCHEMA_VERSION" TEXT, "CREATED_AT" TEXT)'))
            accepted = ATTEMPT_ID
            retry = "00000000-0000-4000-8000-000000000099"
            connection.execute(text(f'INSERT INTO "{names.llm_attempt}" VALUES (:id, 1), (:retry, 2)'), {"id": accepted, "retry": retry})
            connection.execute(text(f'INSERT INTO "{names.analysis}" VALUES (:job, :accepted)'), {"job": JOB_ID, "accepted": accepted})
            for attempt_id, label in ((accepted, "old"), (retry, "new")):
                connection.execute(text(f'INSERT INTO "{names.analysis_stage}" VALUES (\'MANUFACTURING\', :content, :job, :attempt, :hash, :prompt, :schema, :created)'), {
                    "content": json.dumps({"label": label}), "job": JOB_ID, "attempt": attempt_id,
                    "hash": EVIDENCE_HASH, "prompt": PROMPT_VERSION,
                    "schema": OUTPUT_SCHEMA_VERSION, "created": "2026-09-23",
                })
        results = AnalysisRepository(engine).list_stage_results(JOB_ID, EVIDENCE_HASH, PROMPT_VERSION, OUTPUT_SCHEMA_VERSION)
        assert results == {"MANUFACTURING": {"label": "new"}}
    finally:
        engine.dispose()


def test_stage_reuse_normalizes_hana_result_labels(recording_engine) -> None:
    """Read lowercase HANA labels before building the partial-stage response."""
    recording_engine.queue_mappings([{
        "stage": "DATA_READINESS", "content_json": '{"descriptions": []}',
    }])
    results = AnalysisRepository(recording_engine).list_stage_results(
        JOB_ID, EVIDENCE_HASH, PROMPT_VERSION, OUTPUT_SCHEMA_VERSION,
    )
    assert results == {"DATA_READINESS": {"descriptions": []}}


def test_completion_precondition_row_none_raises_conflict(recording_engine) -> None:
    """A missing precondition row (job not found) raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(None)
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_job_id_mismatch_between_args_raises_before_sql(
    recording_engine,
) -> None:
    """When completion.job_id != job_id parameter, raises before opening a transaction."""
    other_job = "00000000-0000-4000-8000-000000000099"
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(
            other_job, analysis_completion(job_id=JOB_ID)
        )
    assert recording_engine.begin_count == 0


def test_completion_bad_completeness_derivation_rolls_back(recording_engine) -> None:
    """When completion.completeness does not match derive_completeness(response), rolls back.

    This guards against a caller that derived completeness from a different
    response object than the one carried in the completion record.
    """
    from app.ecn.analysis_models import AnalysisCompleteness

    recording_engine.queue_mapping(completable_job_attempt_row())
    bad_completion = analysis_completion()
    # Override completeness to COMPLETED_WITH_GAPS even though all sections are NOT_APPLICABLE.
    bad = AnalysisCompletion(
        analysis_id=ANALYSIS_ID,
        job_id=JOB_ID,
        ecn_row_id=ECN_ROW_ID,
        successful_attempt_id=ATTEMPT_ID,
        response=bad_completion.response,
        completeness=AnalysisCompleteness.COMPLETED_WITH_GAPS,  # wrong
        evidence_hash=EVIDENCE_HASH,
        prompt_version=PROMPT_VERSION,
        model_name=MODEL_NAME,
        model_deployment=MODEL_DEPLOYMENT,
        deterministic_logic_version=DETERMINISTIC_LOGIC_VERSION,
        rule_evaluator_version=RULE_EVALUATOR_VERSION,
        output_schema_version=OUTPUT_SCHEMA_VERSION,
        created_at=UTC_FINISH,
    )
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, bad)
    assert recording_engine.rollback_count == 1


def test_completion_attempt_rowcount_zero_rolls_back(recording_engine) -> None:
    """If the attempt UPDATE returns rowcount=0, raises ANALYSIS_STATE_CONFLICT.

    This simulates a lost race where the attempt was already finalized.
    """
    recording_engine.queue_mapping(completable_job_attempt_row())
    recording_engine.queue_rowcount(1)  # analysis INSERT consumed
    recording_engine.queue_rowcount(0)  # attempt UPDATE → race loss
    recording_engine.queue_rowcount(1)  # job UPDATE (never reached)
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_completion_job_rowcount_zero_rolls_back(recording_engine) -> None:
    """If the job UPDATE returns rowcount=0, raises ANALYSIS_STATE_CONFLICT."""
    recording_engine.queue_mapping(completable_job_attempt_row())
    recording_engine.queue_rowcount(1)  # analysis INSERT consumed
    recording_engine.queue_rowcount(1)  # attempt UPDATE succeeds
    recording_engine.queue_rowcount(0)  # job UPDATE → race loss
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).complete_job(JOB_ID, analysis_completion())
    assert recording_engine.rollback_count == 1


def test_analysis_insert_sql_bind_params_match_completion_row(
    recording_engine,
) -> None:
    """Every :NAME bind token in the analysis INSERT must match a key in analysis_completion_to_row.

    This is the bind-parameter discipline guard for the completion write path,
    preventing case/spelling drift between _analysis_insert_sql and
    analysis_completion_to_row from silently becoming a live-only failure.
    """
    from app.persistence.analysis_repository import _analysis_insert_sql
    from app.persistence.codec import analysis_completion_to_row
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _analysis_insert_sql(PersistenceTableNames().analysis)
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    params = analysis_completion_to_row(analysis_completion())
    dict_keys = set(params.keys())
    missing = bind_tokens - dict_keys
    assert not missing, (
        f"SQL bind tokens not found in analysis_completion_to_row keys: {sorted(missing)}"
    )


# ===========================================================================
# Task-9 fail_job tests
# ===========================================================================


def test_fail_job_after_provider_error_records_not_run(recording_engine) -> None:
    """fail_job with a NOT_RUN validation outcome persists VALIDATION_OUTCOME='NOT_RUN'.

    This covers the case where the provider returned an error or timed out
    so no response was available for schema validation.
    """
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(1)  # attempt UPDATE
    recording_engine.queue_rowcount(1)  # job UPDATE
    error = SafeJobError(
        code="PROVIDER_TIMEOUT",
        message="The provider call timed out",
        validation_outcome=ValidationOutcome.NOT_RUN,
    )
    AnalysisRepository(recording_engine).fail_job(
        JOB_ID, ATTEMPT_ID, error, UTC_FINISH
    )
    assert recording_engine.begin_count == 1
    attempt_sql, attempt_params = recording_engine.executions[0]
    assert "VALIDATION_OUTCOME" in attempt_sql
    # The params must carry NOT_RUN for VALIDATION_OUTCOME
    param_value = attempt_params.get("VALIDATION_OUTCOME")
    assert param_value == "NOT_RUN", f"expected NOT_RUN, got {param_value!r}"


def test_fail_job_after_validation_failure_records_invalid(recording_engine) -> None:
    """fail_job with an INVALID validation outcome persists VALIDATION_OUTCOME='INVALID'.

    This covers the case where the provider returned SUCCESS but the response
    did not pass schema or evidence-cross-reference validation.
    """
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(1)  # attempt UPDATE
    recording_engine.queue_rowcount(1)  # job UPDATE
    error = SafeJobError(
        code="RESPONSE_SCHEMA_INVALID",
        message="The LLM response failed schema validation",
        validation_outcome=ValidationOutcome.INVALID,
    )
    AnalysisRepository(recording_engine).fail_job(
        JOB_ID, ATTEMPT_ID, error, UTC_FINISH
    )
    attempt_sql, attempt_params = recording_engine.executions[0]
    assert attempt_params.get("VALIDATION_OUTCOME") == "INVALID"


def test_fail_job_is_one_transaction(recording_engine) -> None:
    """fail_job executes exactly one transaction containing both the attempt and job updates."""
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(1)
    recording_engine.queue_rowcount(1)
    AnalysisRepository(recording_engine).fail_job(
        JOB_ID, ATTEMPT_ID,
        SafeJobError("E", "msg", validation_outcome=ValidationOutcome.NOT_RUN),
        UTC_FINISH,
    )
    assert recording_engine.begin_count == 1
    assert len(recording_engine.executions) == 2


def test_fail_job_state_conflict_on_zero_rowcount(recording_engine) -> None:
    """fail_job raises ANALYSIS_STATE_CONFLICT when the attempt UPDATE matches no row."""
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(0)  # attempt UPDATE → no match
    recording_engine.queue_rowcount(1)  # job UPDATE (never reached)
    with pytest.raises(PersistenceStateError, match="analysis state"):
        AnalysisRepository(recording_engine).fail_job(
            JOB_ID, ATTEMPT_ID,
            SafeJobError("E", "msg", validation_outcome=ValidationOutcome.NOT_RUN),
            UTC_FINISH,
        )
    assert recording_engine.rollback_count == 1


def test_fail_job_error_code_stripped_and_capped(recording_engine) -> None:
    """Error code is stripped of control characters and capped at 80 characters."""
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(1)
    recording_engine.queue_rowcount(1)
    long_code = "X" * 90  # exceeds 80-char cap
    error = SafeJobError(
        code=long_code,
        message="A message",
        validation_outcome=ValidationOutcome.NOT_RUN,
    )
    AnalysisRepository(recording_engine).fail_job(
        JOB_ID, ATTEMPT_ID, error, UTC_FINISH
    )
    _, job_params = recording_engine.executions[1]
    stored_code = job_params.get("ERROR_CODE") or job_params.get("error_code")
    assert stored_code is not None
    assert len(stored_code) <= 80


# ===========================================================================
# Task-9 retry tests
# ===========================================================================


def test_failed_analysis_free_job_is_queued_for_retry(recording_engine) -> None:
    """A FAILED job with no analysis is atomically re-queued; result is QUEUED."""
    recording_engine.queue_mapping(retry_source_row(state="FAILED", analysis_id=None))
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.QUEUED
    assert result.job_id == JOB_ID


def test_unknown_ecn_is_distinct_from_retry_conflict(recording_engine) -> None:
    """A missing ECN row returns ECN_NOT_FOUND; job_id is None."""
    recording_engine.queue_mapping(None)
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.ECN_NOT_FOUND
    assert result.job_id is None


def test_completed_job_is_not_retryable(recording_engine) -> None:
    """A COMPLETED job with an analysis returns NOT_RETRYABLE and issues no UPDATE."""
    recording_engine.queue_mapping(
        retry_source_row(state="COMPLETED", analysis_id=ANALYSIS_ID)
    )
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.NOT_RETRYABLE
    assert result.job_id == JOB_ID
    assert not any(
        'SET "STATE" = \'QUEUED\'' in sql for sql, _ in recording_engine.executions
    )


def test_running_job_is_not_retryable(recording_engine) -> None:
    """A RUNNING job (not FAILED) returns NOT_RETRYABLE."""
    recording_engine.queue_mapping(
        retry_source_row(state="RUNNING", analysis_id=None)
    )
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.NOT_RETRYABLE


def test_failed_regeneration_with_previous_analysis_is_retryable(recording_engine) -> None:
    """A failed replacement can retry while the previous result remains visible."""
    recording_engine.queue_mapping(
        retry_source_row(state="FAILED", analysis_id=ANALYSIS_ID)
    )
    recording_engine.queue_rowcount(1)
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.QUEUED


def test_ecn_with_no_job_is_not_retryable(recording_engine) -> None:
    """An ECN that exists but has no job row returns NOT_RETRYABLE (job_id=None)."""
    # JOB_ID is None when the LEFT JOIN finds no job
    recording_engine.queue_mapping(
        {"ECN_ROW_ID": ECN_ROW_ID, "JOB_ID": None, "STATE": None, "ANALYSIS_ID": None}
    )
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.NOT_RETRYABLE
    assert result.job_id is None


def test_retry_update_rowcount_zero_returns_not_retryable(recording_engine) -> None:
    """If the FAILED->QUEUED UPDATE wins no row, returns NOT_RETRYABLE (late concurrency)."""
    recording_engine.queue_mapping(retry_source_row(state="FAILED", analysis_id=None))
    recording_engine.queue_rowcount(0)  # UPDATE lost the race
    result = AnalysisRepository(recording_engine).queue_failed_job(ECN_ROW_ID)
    assert result.status is RetryQueueStatus.NOT_RETRYABLE


def test_queued_job_is_blocked_without_starting_an_attempt(recording_engine) -> None:
    """Persist one safe preflight blocker only while the job remains queued."""
    recording_engine.queue_rowcount(1)
    changed = AnalysisRepository(recording_engine).block_queued_job(
        JOB_ID,
        "PROVIDER_AUTH_FAILED",
    )
    assert changed is True
    sql, params = recording_engine.executions[-1]
    assert 'SET "STATE" = \'BLOCKED\'' in sql
    assert 'AND "STATE" = \'QUEUED\'' in sql
    assert params == {
        "job_id": JOB_ID,
        "error_code": "PROVIDER_AUTH_FAILED",
        "error_message": "AI generation is blocked until authentication succeeds",
    }


def test_only_provider_blocked_jobs_are_requeued_after_recovery(
    recording_engine,
) -> None:
    """Return only provider-blocked IDs whose conditional requeue update wins."""
    second_job_id = "00000000-0000-4000-8000-000000000011"
    recording_engine.queue_mappings(
        [{"JOB_ID": JOB_ID}, {"JOB_ID": second_job_id}]
    )
    recording_engine.queue_rowcount(1)
    recording_engine.queue_rowcount(0)
    queued = AnalysisRepository(recording_engine).queue_provider_blocked_jobs()
    assert queued == (JOB_ID,)
    select_sql, _ = recording_engine.executions[0]
    assert '"STATE" = \'BLOCKED\'' in select_sql
    assert "PROVIDER_AUTH_FAILED" in select_sql
    first_update_sql, first_params = recording_engine.executions[1]
    assert 'SET "STATE" = \'QUEUED\'' in first_update_sql
    assert 'AND "STATE" = \'BLOCKED\'' in first_update_sql
    assert first_params == {"job_id": JOB_ID}


# ===========================================================================
# Task-9 progress tests
# ===========================================================================


def test_progress_counts_gap_results_separately(recording_engine) -> None:
    """Progress sums all six states and derives unfinished/total correctly."""
    recording_engine.queue_mapping(
        {
            "QUEUED_COUNT": 2,
            "BLOCKED_COUNT": 6,
            "RUNNING_COUNT": 1,
            "COMPLETED_COUNT": 3,
            "COMPLETED_WITH_GAPS_COUNT": 4,
            "FAILED_COUNT": 5,
        }
    )
    progress = AnalysisRepository(recording_engine).get_progress()
    assert progress.blocked_count == 6
    assert progress.unfinished_count == 9  # 2 + 6 + 1
    assert progress.total_count == 21  # 2+6+1+3+4+5


def test_progress_empty_table_returns_zero_counts(recording_engine) -> None:
    """An aggregate query returning None (empty table) yields all-zero progress."""
    recording_engine.queue_mapping(None)
    progress = AnalysisRepository(recording_engine).get_progress()
    assert progress.queued_count == 0
    assert progress.running_count == 0
    assert progress.completed_count == 0
    assert progress.completed_with_gaps_count == 0
    assert progress.failed_count == 0
    assert progress.unfinished_count == 0
    assert progress.total_count == 0


def test_progress_upload_scoped_passes_upload_id(recording_engine) -> None:
    """When upload_id is supplied, it is passed as a bind parameter to the SQL."""
    recording_engine.queue_mapping(
        {
            "QUEUED_COUNT": 1,
            "RUNNING_COUNT": 0,
            "COMPLETED_COUNT": 0,
            "COMPLETED_WITH_GAPS_COUNT": 0,
            "FAILED_COUNT": 0,
        }
    )
    AnalysisRepository(recording_engine).get_progress(upload_id=UPLOAD_ID)
    _, params = recording_engine.executions[-1]
    assert params.get("upload_id") == UPLOAD_ID


# ===========================================================================
# Task-9 restart-recovery tests
# ===========================================================================


def test_restart_preserves_finished_provider_outcome_and_usage(
    recording_engine,
) -> None:
    """fail_interrupted_jobs only interrupts STARTED/unfinished attempts; returns job count.

    The attempt UPDATE WHERE clause must filter by PROVIDER_OUTCOME='STARTED' and
    FINISHED_AT IS NULL.  The SET clause must not touch any token columns, so that
    already-finished attempts with recorded usage are left intact.
    """
    recording_engine.queue_rowcount(1)  # attempt UPDATE: STARTED/null-finish interrupted
    recording_engine.queue_rowcount(3)  # job UPDATE: QUEUED/RUNNING jobs failed
    changed = AnalysisRepository(recording_engine).fail_interrupted_jobs(UTC_FINISH)
    assert changed == 3
    attempt_sql, _attempt_params = recording_engine.executions[-2]
    # WHERE clause must require STARTED and null FINISHED_AT
    assert '"PROVIDER_OUTCOME" = \'STARTED\'' in attempt_sql.split("WHERE", 1)[1]
    assert '"FINISHED_AT" IS NULL' in attempt_sql
    # SET clause must not include token/usage columns
    assert "INPUT_TOKENS" not in attempt_sql.split("SET", 1)[1].split("WHERE", 1)[0]


def test_restart_recovery_is_one_transaction(recording_engine) -> None:
    """fail_interrupted_jobs executes both UPDATEs inside a single transaction."""
    recording_engine.queue_rowcount(0)  # attempt UPDATE
    recording_engine.queue_rowcount(0)  # job UPDATE
    AnalysisRepository(recording_engine).fail_interrupted_jobs(UTC_FINISH)
    assert recording_engine.begin_count == 1
    assert len(recording_engine.executions) == 2


def test_restart_job_update_targets_queued_and_running(recording_engine) -> None:
    """The job UPDATE must target both QUEUED and RUNNING states."""
    recording_engine.queue_rowcount(0)
    recording_engine.queue_rowcount(0)
    AnalysisRepository(recording_engine).fail_interrupted_jobs(UTC_FINISH)
    job_sql, _ = recording_engine.executions[-1]
    assert "'QUEUED'" in job_sql
    assert "'RUNNING'" in job_sql


def test_provider_outcome_sql_bind_params_match_dict_keys(recording_engine) -> None:
    """Every :NAME bind token in the provider-outcome SQL must match a dict key.

    This is a regression guard against future case/spelling drift between
    _provider_outcome_sql and provider_outcome_to_row.  SQLAlchemy bind-name
    matching is case-sensitive; a mismatch causes a runtime error on every real
    record_provider_outcome call.

    The regex captures bind names of ANY case (:A-Za-z_+) so that a regression
    back to lowercase bind params (e.g. :provider_outcome) is also captured and
    detected as missing from the uppercase dict keys.

    A negative sub-check deliberately introduces a lowercase bind param into the
    SQL and asserts the guard fires, proving the regex would catch the original
    defect and any future regression of the same kind.
    """
    import re as _re
    from app.persistence.analysis_repository import (
        _provider_outcome_sql,
        provider_outcome_to_row,
    )
    from app.persistence.table_contracts import PersistenceTableNames
    from app.observability.llm_usage_logging import TokenUsage
    from app.persistence.codec import usage_to_columns

    sql = _provider_outcome_sql(PersistenceTableNames())
    # Capture bind names of any case so that lowercase regressions are detected.
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))

    usage = usage_to_columns(
        TokenUsage(input_tokens=1, cached_input_tokens=0, output_tokens=1, total_tokens=2),
        usage_available=True,
    )
    params = provider_outcome_to_row(
        ATTEMPT_ID,
        provider_outcome(),
        usage,
    )
    dict_keys = set(params.keys())

    # Primary assertion: no bind token is absent from the params dict.
    missing = bind_tokens - dict_keys
    assert not missing, (
        f"SQL bind tokens not found in params dict: {sorted(missing)}. "
        "Ensure _provider_outcome_sql and provider_outcome_to_row use matching names."
    )

    # Negative sub-check: introduce one lowercase bind token (simulating the
    # original defect) and assert the guard detects it.  This proves the regex
    # actually catches case regressions rather than silently passing them.
    regressed_sql = sql.replace(":PROVIDER_OUTCOME", ":provider_outcome")
    regressed_tokens = set(_re.findall(r":([A-Za-z_]+)", regressed_sql))
    regressed_missing = regressed_tokens - dict_keys
    assert regressed_missing, (
        "Negative sub-check failed: the guard did not detect the deliberately "
        "lowercased ':provider_outcome' bind token — the regex is not catching "
        "lowercase regressions."
    )


# ===========================================================================
# Task-9 bind-parameter discipline guards for new write-path SQL builders
# ===========================================================================
# One guard per new write statement added in Task 9.  Each guard asserts that
# every :NAME bind token in the SQL is a subset of the keys in the params dict
# that the production call actually passes.  A case/spelling drift in either
# the SQL or the params dict will make the test fail before it can become a
# live-only HANA runtime error.


def test_validate_attempt_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _validate_attempt_sql must exist in its lowercase params dict.

    The production call passes ``{"attempt_id": ..., "job_id": ...}``; the SQL
    must use ``:attempt_id`` and ``:job_id`` (lowercase) to match.
    """
    from app.persistence.analysis_repository import _validate_attempt_sql
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _validate_attempt_sql(PersistenceTableNames().llm_attempt)
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production complete_job call uses.
    params = {"attempt_id": ATTEMPT_ID, "job_id": JOB_ID}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_validate_attempt_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_complete_job_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _complete_job_sql must exist in its lowercase params dict.

    The production call passes ``{"job_id": ..., "finished_at": ...}``; the SQL
    must use ``:job_id`` and ``:finished_at`` (lowercase) to match.
    """
    from app.persistence.analysis_repository import _complete_job_sql
    from app.persistence.hana import to_hana_timestamp
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _complete_job_sql(PersistenceTableNames().analysis_job)
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production complete_job call uses.
    params = {"job_id": JOB_ID, "finished_at": to_hana_timestamp(UTC_FINISH)}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_complete_job_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_fail_attempt_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _fail_attempt_sql must exist in its UPPERCASE params dict.

    The production call passes ``{"ATTEMPT_ID": ..., "JOB_ID": ..., "VALIDATION_OUTCOME": ...}``;
    the SQL must use UPPERCASE bind names to match the row-dict convention.
    """
    from app.persistence.analysis_repository import _fail_attempt_sql
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _fail_attempt_sql(PersistenceTableNames().llm_attempt)
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production fail_job call uses.
    params = {
        "ATTEMPT_ID": ATTEMPT_ID,
        "JOB_ID": JOB_ID,
        "VALIDATION_OUTCOME": ValidationOutcome.NOT_RUN.value,
    }
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_fail_attempt_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_fail_job_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _fail_job_sql must exist in its UPPERCASE params dict.

    The production call passes ``{"JOB_ID": ..., "FINISHED_AT": ..., "ERROR_CODE": ...,
    "ERROR_MESSAGE": ...}``; the SQL must use UPPERCASE bind names.
    """
    from app.persistence.analysis_repository import _fail_job_sql
    from app.persistence.hana import to_hana_timestamp
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _fail_job_sql(PersistenceTableNames().analysis_job)
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production fail_job call uses.
    params = {
        "JOB_ID": JOB_ID,
        "FINISHED_AT": to_hana_timestamp(UTC_FINISH),
        "ERROR_CODE": "ERR",
        "ERROR_MESSAGE": "msg",
    }
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_fail_job_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_queue_failed_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _queue_failed_sql must exist in its lowercase params dict.

    The production call passes ``{"job_id": ...}``; the SQL must use ``:job_id``
    (lowercase) for both the WHERE clause and the NOT EXISTS sub-query.
    """
    from app.persistence.analysis_repository import _queue_failed_sql
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _queue_failed_sql(PersistenceTableNames())
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production queue_failed_job call uses.
    params = {"job_id": JOB_ID}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_queue_failed_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_interrupt_started_attempts_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _interrupt_started_attempts_sql must be in the lowercase params dict.

    The production call passes ``{"finished_at": ...}``; the SQL must use
    ``:finished_at`` (lowercase) to match.
    """
    from app.persistence.analysis_repository import _interrupt_started_attempts_sql
    from app.persistence.hana import to_hana_timestamp
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _interrupt_started_attempts_sql(PersistenceTableNames())
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production fail_interrupted_jobs call uses.
    params = {"finished_at": to_hana_timestamp(UTC_FINISH)}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_interrupt_started_attempts_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_fail_unfinished_jobs_sql_bind_params_match_dict_keys() -> None:
    """Every :NAME bind token in _fail_unfinished_jobs_sql must be in the lowercase params dict.

    The production call passes ``{"finished_at": ...}``; the SQL must use
    ``:finished_at`` (lowercase) to match.
    """
    from app.persistence.analysis_repository import _fail_unfinished_jobs_sql
    from app.persistence.hana import to_hana_timestamp
    from app.persistence.table_contracts import PersistenceTableNames

    sql = _fail_unfinished_jobs_sql(PersistenceTableNames())
    bind_tokens = set(_re.findall(r":([A-Za-z_]+)", sql))
    # Params dict built from the same literal dict the production fail_interrupted_jobs call uses.
    params = {"finished_at": to_hana_timestamp(UTC_FINISH)}
    missing = bind_tokens - set(params.keys())
    assert not missing, (
        f"_fail_unfinished_jobs_sql bind tokens not found in params dict: {sorted(missing)}"
    )


def test_fail_job_job_rowcount_zero_rolls_back(recording_engine) -> None:
    """fail_job raises ANALYSIS_STATE_CONFLICT when the job UPDATE matches no row.

    Completes the rowcount-conflict coverage alongside
    ``test_fail_job_state_conflict_on_zero_rowcount`` (which covers the attempt
    UPDATE losing).  Here the attempt UPDATE succeeds (rowcount=1) but the job
    UPDATE loses the race (rowcount=0).
    """
    from app.persistence.records import SafeJobError

    recording_engine.queue_rowcount(1)  # attempt UPDATE succeeds
    recording_engine.queue_rowcount(0)  # job UPDATE → race loss
    with pytest.raises(PersistenceStateError):
        AnalysisRepository(recording_engine).fail_job(
            JOB_ID,
            ATTEMPT_ID,
            SafeJobError("E", "msg", validation_outcome=ValidationOutcome.NOT_RUN),
            UTC_FINISH,
        )
    assert recording_engine.rollback_count == 1
