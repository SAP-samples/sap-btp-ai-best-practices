"""Field-complete tests for pure ECN workflow response mapping."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from app.ecn.api_mapping import (
    history_reset_response_from_counts,
    retry_response_from_result,
    upload_response_from_result,
)
from app.ecn.services.ingestion import IngestionResult
from app.persistence.records import (
    AnalysisProgressResponse,
    HistoryDeletionCounts,
    RetryQueueResult,
    RetryQueueStatus,
    UploadRecord,
)

UPLOAD_ID = "018f54fe-7b2d-7d7d-aef4-81940d349101"
REQUEST_ID = "018f54fe-7b2d-7d7d-aef4-81940d349102"
BOM_ID = "018f54fe-7b2d-7d7d-aef4-81940d349103"
JOB_ID = "018f54fe-7b2d-7d7d-aef4-81940d349104"
ECN_ID = "018f54fe-7b2d-7d7d-aef4-81940d349105"


def _upload_result() -> IngestionResult:
    """Return one complete committed ingestion result for mapper tests."""
    upload = UploadRecord(
        upload_id=UPLOAD_ID,
        request_id=REQUEST_ID,
        ecn_artifact_id="018f54fe-7b2d-7d7d-aef4-81940d349106",
        bom_artifact_id=BOM_ID,
        rule_artifact_id=None,
        bom_reused=True,
        imported_ecn_count=3,
        publication_order=7,
        created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )
    return IngestionResult(upload=upload, created=True, queued_job_ids=(JOB_ID,))


def test_upload_mapper_copies_every_committed_field() -> None:
    """Expose exact IDs, state counts, and stable navigation URLs."""
    progress = AnalysisProgressResponse(
        queued_count=1,
        running_count=0,
        completed_count=1,
        completed_with_gaps_count=1,
        failed_count=0,
        unfinished_count=1,
        total_count=3,
    )

    response = upload_response_from_result(_upload_result(), progress)

    assert response.model_dump() == {
        "upload_id": UUID(UPLOAD_ID),
        "request_id": UUID(REQUEST_ID),
        "created": True,
        "imported_ecn_count": 3,
        "bom_artifact_id": UUID(BOM_ID),
        "bom_reused": True,
        "queued_count": 1,
        "blocked_count": 0,
        "ready_count": 0,
        "running_count": 0,
        "completed_count": 1,
        "completed_with_gaps_count": 1,
        "failed_count": 0,
        "list_url": "/api/ecns",
        "progress_url": f"/api/analysis-progress?upload_id={UPLOAD_ID}",
    }


def test_retry_mapper_accepts_only_a_committed_queued_result() -> None:
    """Reject missing and non-retryable repository results at the pure boundary."""
    response = retry_response_from_result(
        ECN_ID, RetryQueueResult(RetryQueueStatus.QUEUED, JOB_ID)
    )
    assert response.model_dump() == {
        "ecn_row_id": UUID(ECN_ID),
        "job_id": UUID(JOB_ID),
        "state": "QUEUED",
    }

    with pytest.raises(ValueError, match="QUEUED"):
        retry_response_from_result(
            ECN_ID, RetryQueueResult(RetryQueueStatus.NOT_RETRYABLE, JOB_ID)
        )


def test_history_mapper_copies_all_seven_deletion_counts() -> None:
    """Keep the public count names aligned with the dependency-ordered record."""
    response = history_reset_response_from_counts(
        HistoryDeletionCounts(
            attempts=1,
            analyses=2,
            jobs=3,
            ecns=4,
            uploads=5,
            bom_rows=6,
            source_artifacts=7,
        )
    )
    assert response.model_dump() == {
        "deleted_attempt_count": 1,
        "deleted_analysis_count": 2,
        "deleted_job_count": 3,
        "deleted_ecn_count": 4,
        "deleted_upload_count": 5,
        "deleted_bom_row_count": 6,
        "deleted_source_artifact_count": 7,
    }
