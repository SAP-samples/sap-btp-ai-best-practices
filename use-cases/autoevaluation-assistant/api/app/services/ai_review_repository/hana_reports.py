"""SAP HANA persistence for asynchronous assessment report jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from app.models.reports import (
    AssessmentReportDownload,
    AssessmentReportJobResponse,
    AssessmentReportJobStatus,
    AssessmentReportJobStatusResponse,
    AssessmentReportSource,
)

ACTIVE_REPORT_STATUSES = {"generating", "rendering"}
_REPORT_SCHEMA_READY = False
_REPORT_SCHEMA_LOCK = Lock()


def _decode_report_source_json(value: Any) -> Any:
    """Decode one HANA NCLOB value without validating the newest schema.

    Inputs:
        value: String, bytes, or memoryview returned for ``source_json``.

    Outputs:
        Any: Raw decoded JSON value. Invalid JSON becomes ``None`` so the leased
        worker can persist a safe terminal snapshot failure.
    """

    if isinstance(value, memoryview):
        value = value.tobytes()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return None


class HanaAssessmentReportsMixin:
    """Persist report queue snapshots, leases, results, and expiry in HANA."""

    session: Any

    def create_assessment_report_job(
        self,
        source: AssessmentReportSource,
        now: datetime | None = None,
    ) -> AssessmentReportJobResponse:
        """Insert one pending report job with immutable source JSON.

        Inputs:
            source: Trusted frozen report-v2 or report-v3 snapshot.
            now: Optional deterministic creation time for tests.

        Outputs:
            AssessmentReportJobResponse: Created job identity and initial state.
        """

        self._ensure_assessment_report_schema()
        created_at = now or datetime.now(timezone.utc)
        job_id = f"report-{uuid4()}"
        self.session.execute(
            text(
                "insert into assessment_report_jobs "
                "(job_id, assessment_id, customer_class, sector, language, "
                "status, source_json, retry_count, created_at, updated_at) "
                "values (:job_id, :assessment_id, :customer_class, :sector, "
                ":language, :status, :source_json, 0, :created_at, :updated_at)"
            ),
            {
                "job_id": job_id,
                "assessment_id": source.assessment_id,
                "customer_class": source.customer_class,
                # Preserve the physical legacy column without making it a
                # runtime source of truth for class/NACE cohort selection.
                "sector": source.nace1,
                "language": source.language,
                "status": "pending",
                "source_json": source.model_dump_json(),
                "created_at": created_at,
                "updated_at": created_at,
            },
        )
        return AssessmentReportJobResponse(
            job_id=job_id,
            status="pending",
            language=source.language,
        )

    def get_assessment_report_job_status(
        self,
        job_id: str,
        now: datetime | None = None,
    ) -> AssessmentReportJobStatusResponse:
        """Return one unexpired report job for browser polling.

        Inputs:
            job_id: Report job identifier.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            AssessmentReportJobStatusResponse: Persisted lifecycle and metadata.

        Raises:
            KeyError: If the job is unknown or expired.
        """

        _ = now
        row = self.session.execute(
            text(
                "select job_id, assessment_id, language, status, progress_message, "
                "file_name, error_code, error_message, retry_count, created_at, "
                "updated_at, expires_at, "
                "case when status = 'completed' and pdf_blob is not null "
                "then 1 else 0 end as download_ready "
                "from assessment_report_jobs "
                "where job_id = :job_id "
                "and (expires_at is null or expires_at > current_utctimestamp)"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if row is None:
            raise KeyError(f"Unknown assessment report job ID: {job_id}")
        return AssessmentReportJobStatusResponse(
            job_id=row["job_id"],
            assessment_id=row["assessment_id"],
            language=row["language"],
            status=row["status"],
            progress_message=row.get("progress_message"),
            download_ready=bool(row.get("download_ready")),
            file_name=row.get("file_name"),
            error_code=row.get("error_code"),
            error_message=row.get("error_message"),
            retry_count=int(row.get("retry_count", 0) or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row.get("expires_at"),
        )

    def lease_next_assessment_report_job(
        self,
        worker_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Atomically lease the oldest pending or expired active report job.

        Inputs:
            worker_id: Worker process taking the report job.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            dict[str, Any] | None: Leased job and raw decoded source mapping.
        """

        _ = now
        row = self.session.execute(
            text(
                "select top 1 job_id, source_json, status, retry_count "
                "from assessment_report_jobs "
                "where status = 'pending' "
                "or (status in ('generating', 'rendering') "
                "and lease_expires_at <= current_utctimestamp) "
                "order by created_at"
            )
        ).mappings().first()
        if row is None:
            return None
        lease_result = self.session.execute(
            text(
                "update assessment_report_jobs "
                "set status = 'generating', "
                "progress_message = 'Generating report narrative', "
                "lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_utctimestamp, 900), "
                "retry_count = retry_count + "
                "case when status = 'pending' then 0 else 1 end, "
                "updated_at = current_utctimestamp "
                "where job_id = :job_id "
                "and (status = 'pending' "
                "or (status in ('generating', 'rendering') "
                "and lease_expires_at <= current_utctimestamp))"
            ),
            {"job_id": row["job_id"], "worker_id": worker_id},
        )
        if lease_result.rowcount == 0:
            return None
        retry_count = int(row.get("retry_count", 0) or 0)
        if row["status"] != "pending":
            retry_count += 1
        return {
            "job_id": row["job_id"],
            "source": _decode_report_source_json(row["source_json"]),
            "status": "generating",
            "progress_message": "Generating report narrative",
            "lease_owner": worker_id,
            "retry_count": retry_count,
        }

    def update_assessment_report_job_progress(
        self,
        job_id: str,
        worker_id: str,
        status: AssessmentReportJobStatus,
        progress_message: str,
        now: datetime | None = None,
    ) -> None:
        """Persist an active report generation phase for the lease owner.

        Inputs:
            job_id: Report job being processed.
            worker_id: Current lease owner.
            status: Active ``generating`` or ``rendering`` phase.
            progress_message: Human-readable progress text.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            None. A guarded update changes the active row.
        """

        _ = now
        if status not in ACTIVE_REPORT_STATUSES:
            raise ValueError(f"Unsupported active report status: {status}")
        result = self.session.execute(
            text(
                "update assessment_report_jobs "
                "set status = :status, progress_message = :progress_message, "
                "updated_at = current_utctimestamp "
                "where job_id = :job_id and lease_owner = :worker_id "
                "and status in ('generating', 'rendering') "
                "and lease_expires_at > current_utctimestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "status": status,
                "progress_message": progress_message,
            },
        )
        self._require_report_job_update(result.rowcount, job_id, worker_id)

    def complete_assessment_report_job(
        self,
        job_id: str,
        worker_id: str,
        file_name: str,
        pdf_content: bytes,
        now: datetime | None = None,
    ) -> None:
        """Store generated PDF bytes and start the 24-hour retention window.

        Inputs:
            job_id: Report job being completed.
            worker_id: Current lease owner.
            file_name: Safe browser download filename.
            pdf_content: Generated PDF bytes.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            None. The guarded row becomes completed and downloadable.
        """

        _ = now
        result = self.session.execute(
            text(
                "update assessment_report_jobs "
                "set status = 'completed', progress_message = 'Report ready', "
                "file_name = :file_name, pdf_blob = :pdf_blob, "
                "lease_owner = null, lease_expires_at = null, "
                "expires_at = add_seconds(current_utctimestamp, 86400), "
                "updated_at = current_utctimestamp "
                "where job_id = :job_id and lease_owner = :worker_id "
                "and status in ('generating', 'rendering') "
                "and lease_expires_at > current_utctimestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "file_name": file_name,
                "pdf_blob": bytes(pdf_content),
            },
        )
        self._require_report_job_update(result.rowcount, job_id, worker_id)

    def fail_assessment_report_job(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        now: datetime | None = None,
    ) -> None:
        """Persist a safe terminal failure for the active report lease.

        Inputs:
            job_id: Report job that failed.
            worker_id: Current lease owner.
            error_code: Stable machine-readable failure category.
            error_message: Safe user-facing error text.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            None. The row remains pollable for 24 hours without PDF content.
        """

        _ = now
        result = self.session.execute(
            text(
                "update assessment_report_jobs "
                "set status = 'failed', progress_message = 'Report generation failed', "
                "error_code = :error_code, error_message = :error_message, "
                "lease_owner = null, lease_expires_at = null, "
                "expires_at = add_seconds(current_utctimestamp, 86400), "
                "updated_at = current_utctimestamp "
                "where job_id = :job_id and lease_owner = :worker_id "
                "and status in ('generating', 'rendering') "
                "and lease_expires_at > current_utctimestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "error_code": error_code,
                "error_message": error_message,
            },
        )
        self._require_report_job_update(result.rowcount, job_id, worker_id)

    def get_assessment_report_download(
        self,
        job_id: str,
        now: datetime | None = None,
    ) -> AssessmentReportDownload:
        """Return PDF bytes for one completed, unexpired report job.

        Inputs:
            job_id: Report job identifier.
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            AssessmentReportDownload: Safe filename and binary PDF content.

        Raises:
            KeyError: If the job is unknown or expired.
            ValueError: If the job is known but not ready for download.
        """

        row = self.session.execute(
            text(
                "select job_id, status, file_name, pdf_blob, expires_at "
                "from assessment_report_jobs where job_id = :job_id "
                "and (expires_at is null or expires_at > current_utctimestamp)"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if row is None:
            raise KeyError(f"Unknown assessment report job ID: {job_id}")
        expires_at = row.get("expires_at")
        if expires_at is not None and now is not None and expires_at <= now:
            raise KeyError(f"Unknown assessment report job ID: {job_id}")
        if row["status"] != "completed" or row.get("pdf_blob") is None:
            raise ValueError(f"Assessment report job {job_id} is not ready for download")
        return AssessmentReportDownload(
            job_id=row["job_id"],
            file_name=row["file_name"],
            content=bytes(row["pdf_blob"]),
        )

    def purge_expired_assessment_report_jobs(
        self,
        now: datetime | None = None,
    ) -> int:
        """Delete terminal report rows past their 24-hour retention deadline.

        Inputs:
            now: Optional interface-parity timestamp; HANA UTC time is canonical.

        Outputs:
            int: Number of deleted rows when reported by the HANA driver.
        """

        _ = now
        result = self.session.execute(
            text(
                "delete from assessment_report_jobs "
                "where expires_at is not null "
                "and expires_at <= current_utctimestamp"
            )
        )
        return max(int(result.rowcount or 0), 0)

    @staticmethod
    def _require_report_job_update(
        rowcount: int | None,
        job_id: str,
        worker_id: str,
    ) -> None:
        """Raise when a guarded HANA lease update changed no row."""

        if rowcount == 0:
            raise ValueError(
                "Assessment report job update requires a non-expired lease for "
                f"worker {worker_id}: {job_id}"
            )

    def _ensure_assessment_report_schema(self) -> None:
        """Initialize and validate HANA schema once before the first enqueue.

        Inputs:
            None. The concrete repository's ``create_schema`` method is used.

        Outputs:
            None. Successful initialization is cached for this API process.
        """

        global _REPORT_SCHEMA_READY
        if _REPORT_SCHEMA_READY:
            return
        with _REPORT_SCHEMA_LOCK:
            if _REPORT_SCHEMA_READY:
                return
            create_schema = getattr(self, "create_schema", None)
            if callable(create_schema):
                create_schema()
            _REPORT_SCHEMA_READY = True
