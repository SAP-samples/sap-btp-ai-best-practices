from __future__ import annotations

from app.services.metal_composition.classification_jobs import (
    ClassificationJobItemSeed,
    ClassificationJobStore,
)


class _CaptureCursor:
    """Capture HANA cursor executions for queue-status contract tests."""

    rowcount = 1

    def __init__(self) -> None:
        self.executions: list[tuple[str, list[object]]] = []

    def __enter__(self) -> "_CaptureCursor":
        """Return this cursor from a context manager."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Close the fake cursor context."""

    def execute(self, sql: str, params: list[object] | None = None) -> None:
        """Capture SQL and bound parameters."""

        self.executions.append((sql, list(params or [])))


class _CaptureConnection:
    """Provide a cursor API compatible with HANAConnection for tests."""

    def __init__(self) -> None:
        self.cursor_instance = _CaptureCursor()

    def cursor(self) -> _CaptureCursor:
        """Return the reusable fake cursor."""

        return self.cursor_instance


def test_hana_job_store_writes_mm_queue_status_but_returns_public_queued() -> None:
    """New HANA jobs should use an MM-specific queue value while API models stay stable."""

    connection = _CaptureConnection()
    store = object.__new__(ClassificationJobStore)
    store.connection = connection
    store.schema = None
    store.jobs_table = "JOBS"
    store.job_items_table = "JOB_ITEMS"
    store.ownership_table = "OWNERSHIP"

    def fake_get_job(job_id: str):
        """Return the persisted job row using the internal MM queue status."""

        return ClassificationJobStore._row_to_job(
            {
                "job_id": job_id,
                "job_type": "single",
                "status": "queued_mm",
                "submitted_at": "2026-06-18T00:00:00Z",
                "started_at": None,
                "finished_at": None,
                "total_count": 1,
                "completed_count": 0,
                "failed_count": 0,
                "error_message": None,
                "claimed_by": None,
                "claimed_at": None,
            }
        )

    store.get_job = fake_get_job

    job = store.submit_job(
        job_type="single",
        items=[
            ClassificationJobItemSeed(
                item_id="mm:0",
                dataset_scope="scope-a",
                position=0,
            )
        ],
    )

    job_insert = connection.cursor_instance.executions[0]
    item_insert = connection.cursor_instance.executions[1]
    assert job_insert[1][2] == "queued_mm"
    assert item_insert[1][7] == "queued_mm"
    assert job.status == "queued"


def test_hana_job_store_claims_only_mm_queue_status() -> None:
    """The MM worker should not claim legacy queued rows that old workers also poll."""

    store = object.__new__(ClassificationJobStore)
    store.connection = _CaptureConnection()
    store.schema = None
    store.jobs_table = "JOBS"

    def fake_fetch_rows(sql: str, params: list[object] | None = None) -> list[dict[str, object]]:
        """Verify the claim SELECT uses the internal MM queue status."""

        del params
        assert '''WHERE "STATUS" = 'queued_mm' ''' in " ".join(sql.split())
        return []

    store._fetch_rows = fake_fetch_rows

    assert store.claim_next_queued_job(worker_id="worker-a") is None


def test_hana_job_row_mapping_converts_mm_queue_status_to_public_status() -> None:
    """Persisted MM queue status should not leak through API response models."""

    job = ClassificationJobStore._row_to_job(
        {
            "job_id": "job-1",
            "job_type": "single",
            "status": "queued_mm",
            "submitted_at": "2026-06-18T00:00:00Z",
            "started_at": None,
            "finished_at": None,
            "total_count": 1,
            "completed_count": 0,
            "failed_count": 0,
            "error_message": None,
            "claimed_by": None,
            "claimed_at": None,
        }
    )
    item = ClassificationJobStore._row_to_job_item(
        {
            "job_item_id": "item-1",
            "job_id": "job-1",
            "item_id": "mm:0",
            "dataset_scope": "scope-a",
            "position": 0,
            "document_mode": "text_only",
            "include_token_usage": 0,
            "status": "queued_mm",
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            "last_classified_at": None,
        }
    )

    assert job.status == "queued"
    assert item.status == "queued"
