"""Pure persistence-to-HTTP mappings for ECN workflow mutations."""

from __future__ import annotations

from app.ecn.api_models import HistoryResetResponse, RetryResponse, UploadResponse
from app.ecn.services.ingestion import IngestionResult
from app.persistence.records import (
    AnalysisProgressResponse,
    HistoryDeletionCounts,
    RetryQueueResult,
    RetryQueueStatus,
)


def upload_response_from_result(
    result: IngestionResult,
    progress: AnalysisProgressResponse,
) -> UploadResponse:
    """Map committed upload identity and its current progress without I/O.

    Args:
        result: Authoritative ingestion result returned after publication.
        progress: Upload-scoped state counts read after that publication.

    Returns:
        Complete public upload response with stable follow-up URLs.
    """
    upload = result.upload
    return UploadResponse(
        upload_id=upload.upload_id,
        request_id=upload.request_id,
        created=result.created,
        imported_ecn_count=upload.imported_ecn_count,
        bom_artifact_id=upload.bom_artifact_id,
        bom_reused=upload.bom_reused,
        ready_count=progress.ready_count,
        queued_count=progress.queued_count,
        blocked_count=progress.blocked_count,
        running_count=progress.running_count,
        completed_count=progress.completed_count,
        completed_with_gaps_count=progress.completed_with_gaps_count,
        failed_count=progress.failed_count,
        list_url="/api/ecns",
        progress_url=f"/api/analysis-progress?upload_id={upload.upload_id}",
    )


def retry_response_from_result(
    ecn_row_id: str,
    result: RetryQueueResult,
) -> RetryResponse:
    """Map only a successful, committed failed-to-queued transition.

    Args:
        ecn_row_id: Canonical identity of the retried ECN occurrence.
        result: Discriminated repository transition result.

    Returns:
        Public queued response carrying the committed job identifier.

    Raises:
        ValueError: If the repository result is not a successful queue outcome.
    """
    if result.status is not RetryQueueStatus.QUEUED or result.job_id is None:
        raise ValueError("retry response requires a QUEUED repository result")
    return RetryResponse(ecn_row_id=ecn_row_id, job_id=result.job_id, state="QUEUED")


def history_reset_response_from_counts(
    counts: HistoryDeletionCounts,
) -> HistoryResetResponse:
    """Map all seven dependency-ordered deletion counts without I/O.

    Args:
        counts: Exact row counts returned by the committed history transaction.

    Returns:
        Complete public history-reset response.
    """
    return HistoryResetResponse(
        deleted_attempt_count=counts.attempts,
        deleted_analysis_count=counts.analyses,
        deleted_job_count=counts.jobs,
        deleted_ecn_count=counts.ecns,
        deleted_upload_count=counts.uploads,
        deleted_bom_row_count=counts.bom_rows,
        deleted_source_artifact_count=counts.source_artifacts,
    )
