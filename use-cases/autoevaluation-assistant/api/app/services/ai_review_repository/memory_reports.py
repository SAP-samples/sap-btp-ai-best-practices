"""In-memory persistence for asynchronous assessment report jobs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from app.models.reports import (
    AssessmentReportDownload,
    AssessmentReportJobResponse,
    AssessmentReportJobStatus,
    AssessmentReportJobStatusResponse,
    AssessmentReportSource,
)

REPORT_LEASE_MINUTES = 15
REPORT_RETENTION_HOURS = 24
ACTIVE_REPORT_STATUSES = {"generating", "rendering"}


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class MemoryAssessmentReportsMixin:
    """Store report queue state and generated PDF bytes in memory for tests."""

    assessment_report_jobs: dict[str, dict[str, Any]]

    def create_assessment_report_job(
        self,
        source: AssessmentReportSource,
        now: datetime | None = None,
    ) -> AssessmentReportJobResponse:
        """Persist a pending report job from an immutable source snapshot.

        Inputs:
            source: Trusted frozen report-v2 or report-v3 snapshot.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            AssessmentReportJobResponse: Created job identity and initial status.
        """

        current_time = now or _utc_now()
        job_id = f"report-{uuid4()}"
        self.assessment_report_jobs[job_id] = {
            "job_id": job_id,
            "assessment_id": source.assessment_id,
            "customer_class": source.customer_class,
            # Keep the legacy-named metadata slot populated for repository
            # parity; source JSON provenance remains the only cohort authority.
            "sector": source.nace1,
            "language": source.language,
            "status": "pending",
            "source": deepcopy(source.model_dump(mode="json")),
            "progress_message": None,
            "lease_owner": None,
            "lease_expires_at": None,
            "retry_count": 0,
            "file_name": None,
            "pdf_content": None,
            "error_code": None,
            "error_message": None,
            "expires_at": None,
            "created_at": current_time,
            "updated_at": current_time,
        }
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
        """Return one report job, treating expired rows as unknown.

        Inputs:
            job_id: Report job identifier returned by the enqueue API.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            AssessmentReportJobStatusResponse: Current polling state.

        Raises:
            KeyError: If the job is unknown or past its retention deadline.
        """

        current_time = now or _utc_now()
        job = self._assessment_report_job(job_id, current_time)
        return AssessmentReportJobStatusResponse(
            job_id=job_id,
            assessment_id=job["assessment_id"],
            language=job["language"],
            status=job["status"],
            progress_message=job["progress_message"],
            download_ready=(
                job["status"] == "completed" and bool(job["pdf_content"])
            ),
            file_name=job["file_name"],
            error_code=job["error_code"],
            error_message=job["error_message"],
            retry_count=job["retry_count"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            expires_at=job["expires_at"],
        )

    def lease_next_assessment_report_job(
        self,
        worker_id: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Lease the first pending or expired active report job.

        Inputs:
            worker_id: Worker process taking ownership of the generation call.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            dict[str, Any] | None: Deep-copied leased job including its source.
        """

        current_time = now or _utc_now()
        for job in self.assessment_report_jobs.values():
            lease_expired = (
                job["status"] in ACTIVE_REPORT_STATUSES
                and job["lease_expires_at"] is not None
                and job["lease_expires_at"] <= current_time
            )
            if job["status"] != "pending" and not lease_expired:
                continue
            if lease_expired:
                job["retry_count"] += 1
            job["status"] = "generating"
            job["progress_message"] = "Generating report narrative"
            job["lease_owner"] = worker_id
            job["lease_expires_at"] = current_time + timedelta(
                minutes=REPORT_LEASE_MINUTES
            )
            job["updated_at"] = current_time
            return deepcopy(job)
        return None

    def update_assessment_report_job_progress(
        self,
        job_id: str,
        worker_id: str,
        status: AssessmentReportJobStatus,
        progress_message: str,
        now: datetime | None = None,
    ) -> None:
        """Persist an active worker phase while preserving its lease.

        Inputs:
            job_id: Report job being processed.
            worker_id: Current lease owner.
            status: Active phase, either ``generating`` or ``rendering``.
            progress_message: Human-readable worker progress.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            None. The in-memory row is updated in place.
        """

        if status not in ACTIVE_REPORT_STATUSES:
            raise ValueError(f"Unsupported active report status: {status}")
        current_time = now or _utc_now()
        job = self._active_assessment_report_job(job_id, worker_id, current_time)
        job["status"] = status
        job["progress_message"] = progress_message
        job["updated_at"] = current_time

    def complete_assessment_report_job(
        self,
        job_id: str,
        worker_id: str,
        file_name: str,
        pdf_content: bytes,
        now: datetime | None = None,
    ) -> None:
        """Persist generated PDF bytes and mark the job completed.

        Inputs:
            job_id: Report job being completed.
            worker_id: Current lease owner.
            file_name: Safe browser download filename.
            pdf_content: Generated PDF bytes.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            None. The row becomes downloadable for 24 hours.
        """

        current_time = now or _utc_now()
        job = self._active_assessment_report_job(job_id, worker_id, current_time)
        job.update(
            {
                "status": "completed",
                "progress_message": "Report ready",
                "file_name": file_name,
                "pdf_content": bytes(pdf_content),
                "lease_owner": None,
                "lease_expires_at": None,
                "expires_at": current_time
                + timedelta(hours=REPORT_RETENTION_HOURS),
                "updated_at": current_time,
            }
        )

    def fail_assessment_report_job(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        now: datetime | None = None,
    ) -> None:
        """Persist a safe terminal generation failure.

        Inputs:
            job_id: Report job that failed.
            worker_id: Current lease owner.
            error_code: Stable machine-readable failure category.
            error_message: Safe user-facing failure explanation.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            None. The job remains pollable for 24 hours and has no file.
        """

        current_time = now or _utc_now()
        job = self._active_assessment_report_job(job_id, worker_id, current_time)
        job.update(
            {
                "status": "failed",
                "progress_message": "Report generation failed",
                "lease_owner": None,
                "lease_expires_at": None,
                "error_code": error_code,
                "error_message": error_message,
                "expires_at": current_time
                + timedelta(hours=REPORT_RETENTION_HOURS),
                "updated_at": current_time,
            }
        )

    def get_assessment_report_download(
        self,
        job_id: str,
        now: datetime | None = None,
    ) -> AssessmentReportDownload:
        """Return generated bytes for a completed, unexpired job.

        Inputs:
            job_id: Report job to download.
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            AssessmentReportDownload: Filename and immutable PDF bytes.

        Raises:
            KeyError: If the job is unknown or expired.
            ValueError: If generation has not completed successfully.
        """

        job = self._assessment_report_job(job_id, now or _utc_now())
        if job["status"] != "completed" or not job["pdf_content"]:
            raise ValueError(f"Assessment report job {job_id} is not ready for download")
        return AssessmentReportDownload(
            job_id=job_id,
            file_name=job["file_name"],
            content=bytes(job["pdf_content"]),
        )

    def purge_expired_assessment_report_jobs(
        self,
        now: datetime | None = None,
    ) -> int:
        """Delete terminal report jobs past their retention deadline.

        Inputs:
            now: Optional deterministic UTC timestamp for tests.

        Outputs:
            int: Number of removed in-memory job rows.
        """

        current_time = now or _utc_now()
        expired_ids = [
            job_id
            for job_id, job in self.assessment_report_jobs.items()
            if job["expires_at"] is not None and job["expires_at"] <= current_time
        ]
        for job_id in expired_ids:
            self.assessment_report_jobs.pop(job_id, None)
        return len(expired_ids)

    def _assessment_report_job(
        self,
        job_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Return an existing unexpired job row or raise ``KeyError``."""

        job = self.assessment_report_jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown assessment report job ID: {job_id}")
        if job["expires_at"] is not None and job["expires_at"] <= now:
            self.assessment_report_jobs.pop(job_id, None)
            raise KeyError(f"Unknown assessment report job ID: {job_id}")
        return job

    def _active_assessment_report_job(
        self,
        job_id: str,
        worker_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Return a job only when the supplied worker owns its active lease."""

        job = self._assessment_report_job(job_id, now)
        lease_is_active = (
            job["status"] in ACTIVE_REPORT_STATUSES
            and job["lease_owner"] == worker_id
            and job["lease_expires_at"] is not None
            and job["lease_expires_at"] > now
        )
        if not lease_is_active:
            raise ValueError(
                "Assessment report job update requires a non-expired lease for "
                f"worker {worker_id}: {job_id}"
            )
        return job
