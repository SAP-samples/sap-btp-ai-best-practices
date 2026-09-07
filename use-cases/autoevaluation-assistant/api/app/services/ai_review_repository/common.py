"""Shared helpers for AI review repository implementations."""

import hashlib
import json
import os
from typing import Any

from app.models.ai_review import QuestionReviewResult

ACTIVE_REVIEW_TASK_STATUSES = {
    "in_progress",
    "extracting_documents",
    "embedding_documents",
    "retrieving_evidence",
    "finalizing_answer",
    "retrying",
}
"""Task statuses that represent active manual question processing."""

LOCAL_PENDING_TASK_STATUS = "pending_local"
"""Pending status used by local workers so deployed legacy workers ignore jobs."""

HANA_CHUNK_INSERT_BATCH_SIZE = 250
"""Maximum number of evidence chunk rows sent per HANA insert batch."""

ADMIN_DOCUMENT_CORPUS_ID = "admin"
"""Synthetic corpus identifier used in shared document API response models."""

LEGACY_DESELECT_RESULT_WARNING = (
    "Legacy AI review result used the retired 'deselect' decision; "
    "the decision is exposed as 'unsupported'. Rerun the review before "
    "applying AI marks."
)
"""Warning attached when persisted pre-guarantee results are normalized."""


def parse_persisted_question_review_result(
    result_json: str | bytes | bytearray,
) -> QuestionReviewResult:
    """Validate stored review JSON with narrow legacy decision compatibility.

    Inputs:
        result_json: JSON payload read from a HANA question-result row.

    Outputs:
        QuestionReviewResult: Current-schema result. Persisted ``deselect``
        decisions from the retired schema are exposed as neutral
        ``unsupported`` evidence statuses and receive an explicit warning.

    Raises:
        json.JSONDecodeError: Raised when the stored value is not valid JSON.
        pydantic.ValidationError: Raised for schema problems other than the
        specifically supported legacy ``deselect`` decision.
    """

    payload = json.loads(result_json)
    normalized_legacy_decision = False
    for level_result in payload.get("level_results", []):
        if not isinstance(level_result, dict):
            continue
        for decision in level_result.get("answer_item_decisions", []):
            if isinstance(decision, dict) and decision.get("decision") == "deselect":
                decision["decision"] = "unsupported"
                normalized_legacy_decision = True

    if normalized_legacy_decision:
        payload["verified_selected_answer_item_ids"] = []
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list) and LEGACY_DESELECT_RESULT_WARNING not in warnings:
            warnings.append(LEGACY_DESELECT_RESULT_WARNING)

    return QuestionReviewResult.model_validate(payload)


def _current_app_env() -> str:
    """Return the normalized runtime environment for queue scoping.

    Inputs:
        None. The value is read from ``APP_ENV``.

    Outputs:
        str: Lowercase application environment, defaulting to ``local``.
    """

    return (os.getenv("APP_ENV") or "local").strip().lower() or "local"


def _pending_task_status_for_runtime() -> str:
    """Return the pending task status this runtime should create and lease.

    Inputs:
        None. The status is based on ``APP_ENV``.

    Outputs:
        str: ``pending`` for production, ``pending_local`` otherwise.
    """

    return "pending" if _current_app_env() == "production" else LOCAL_PENDING_TASK_STATUS


def _normalize_task_status_for_response(status: str) -> str:
    """Normalize internal queue statuses before returning polling payloads.

    Inputs:
        status: Persisted task status.

    Outputs:
        str: Client-facing task status.
    """

    return "pending" if status == LOCAL_PENDING_TASK_STATUS else status


def _row_batches(
    rows: list[dict[str, Any]],
    batch_size: int = HANA_CHUNK_INSERT_BATCH_SIZE,
) -> list[list[dict[str, Any]]]:
    """Split prepared DML rows into bounded HANA executemany batches.

    Inputs:
        rows: Prepared row dictionaries for one SQL statement.
        batch_size: Positive maximum number of rows per batch.

    Outputs:
        list[list[dict[str, Any]]]: Ordered non-empty row batches.

    Raises:
        ValueError: Raised when ``batch_size`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        rows[index : index + batch_size]
        for index in range(0, len(rows), batch_size)
    ]


def _derive_job_status(task_statuses: list[str], fallback_status: str) -> str:
    """Derive a job status from its persisted question task statuses.

    Inputs:
        task_statuses: Status values for all tasks belonging to one job.
        fallback_status: Stored job status to return when the job has no tasks.

    Outputs:
        str: One of the persisted task-derived statuses used by polling clients.
    """

    task_statuses = [_normalize_task_status_for_response(status) for status in task_statuses]
    fallback_status = _normalize_task_status_for_response(fallback_status)
    if not task_statuses:
        return fallback_status
    if all(status == "completed" for status in task_statuses):
        return "completed"
    if any(status in ACTIVE_REVIEW_TASK_STATUSES for status in task_statuses):
        return "in_progress"
    if any(status == "failed" for status in task_statuses):
        if all(status == "failed" for status in task_statuses):
            return "failed"
        if all(status in {"completed", "failed"} for status in task_statuses):
            return "partial_failed"
    return "pending"


def _derive_finished_batch_status(task_statuses: list[str]) -> str | None:
    """Return the terminal parent status when every batch task is finished.

    Inputs:
        task_statuses: Status values for all question tasks in one batch job.

    Outputs:
        str | None: ``completed``, ``failed``, or ``partial_failed`` when all
        tasks are terminal; otherwise ``None``.
    """

    if not task_statuses:
        return None
    if any(status not in {"completed", "failed"} for status in task_statuses):
        return None
    if all(status == "completed" for status in task_statuses):
        return "completed"
    if all(status == "failed" for status in task_statuses):
        return "failed"
    return "partial_failed"


def _document_phase_processed_count(
    job_status: str,
    document_count: int,
    extracted_document_count: int,
    embedded_document_count: int,
) -> int:
    """Return document progress for the active high-level batch phase.

    Inputs:
        job_status: Current batch job status.
        document_count: Total deduplicated documents linked to the batch job.
        extracted_document_count: Documents with persisted extraction rows.
        embedded_document_count: Documents with persisted embedded chunks.

    Outputs:
        int: Processed document count appropriate for the active phase.
    """

    if job_status == "extracting_documents":
        return extracted_document_count
    if job_status == "embedding_documents":
        return embedded_document_count
    if job_status in {"reviewing_questions", "completed", "failed", "partial_failed"}:
        return document_count
    return 0


def document_content_hash(content: bytes) -> str:
    """Return a stable SHA-256 content hash for an uploaded document.

    Inputs:
        content: Raw uploaded document bytes.

    Outputs:
        str: Hex-encoded SHA-256 digest used for assessment-scoped
        deduplication.
    """

    return hashlib.sha256(content).hexdigest()


def _document_payload_bytes(content: Any) -> bytes:
    """Normalize uploaded document content to bytes for hashing and storage.

    Inputs:
        content: Uploaded document content from FastAPI, HANA, or tests.

    Outputs:
        bytes: Binary payload used by repository persistence methods.
    """

    if isinstance(content, bytes):
        return content
    if isinstance(content, memoryview):
        return content.tobytes()
    return bytes(content)
