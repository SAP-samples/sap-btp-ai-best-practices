from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.models.extraction import ExtractionJobStatusResponse, ExtractionJobSubmitResponse, JobResultFile
from app.services.extraction_jobs import JobNotFoundError, JobResultNotReadyError


class ApiRouteTests(unittest.TestCase):
    """Route-level tests for the extraction FastAPI endpoints."""

    def setUp(self) -> None:
        """Create a fresh TestClient for each route test."""

        self.client = TestClient(app)

    def test_defaults_hide_filesystem_reference_paths(self) -> None:
        """The defaults endpoint does not expose internal reference-data paths."""

        response = self.client.get("/api/extraction/defaults")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("community_codes_path", payload)
        self.assertNotIn("unspsc_context_path", payload)
        self.assertEqual(payload["top_k"], 5)

    def test_run_returns_job_id_without_waiting_for_classification(self) -> None:
        """Submitting PDFs returns an accepted job instead of a final result."""

        manager = Mock()
        manager.submit.return_value = ExtractionJobSubmitResponse(
            job_id="job-123",
            status="QUEUED",
            status_url="/api/extraction/jobs/job-123",
            download_url="/api/extraction/jobs/job-123/download",
            created_at="2026-05-26T13:00:00Z",
        )

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.post(
                "/api/extraction/run",
                files={"files": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
                data={"llm_verify": "true"},
            )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["job_id"], "job-123")
        self.assertEqual(payload["status"], "QUEUED")
        self.assertEqual(payload["status_url"], "/api/extraction/jobs/job-123")
        manager.submit.assert_called_once()

    def test_job_status_returns_job_metadata(self) -> None:
        """The job status route returns polling metadata from the manager."""

        manager = Mock()
        manager.get_status.return_value = ExtractionJobStatusResponse(
            job_id="job-123",
            status="SUCCEEDED",
            progress=100,
            stage="completed",
            message="Extraction complete.",
            created_at="2026-05-26T13:00:00Z",
            updated_at="2026-05-26T13:01:00Z",
            started_at="2026-05-26T13:00:05Z",
            finished_at="2026-05-26T13:01:00Z",
            file_count=1,
            llm_verify=True,
            top_k=5,
            runtime_seconds=55.0,
            reference_data_version="synthetic-v1",
            output_filename="commodity_codes.xlsx",
            output_size=12,
            download_url="/api/extraction/jobs/job-123/download",
            headers_preview=[],
            line_items_preview=[],
            errors=[],
            warnings=[],
        )

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.get("/api/extraction/jobs/job-123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "SUCCEEDED")
        self.assertEqual(response.json()["download_url"], "/api/extraction/jobs/job-123/download")

    def test_job_download_returns_excel_bytes_after_success(self) -> None:
        """Completed job downloads return the generated Excel bytes."""

        manager = Mock()
        manager.get_result_file.return_value = JobResultFile(
            filename="commodity_codes.xlsx",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            content=b"xlsx-bytes",
        )

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.get("/api/extraction/jobs/job-123/download")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"xlsx-bytes")
        self.assertEqual(
            response.headers["content-type"],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    def test_job_download_reports_not_ready_and_missing_jobs(self) -> None:
        """Download route maps pending and unknown jobs to correct HTTP status codes."""

        manager = Mock()
        manager.get_result_file.side_effect = JobResultNotReadyError("Job is not complete.")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            not_ready = self.client.get("/api/extraction/jobs/job-123/download")
        self.assertEqual(not_ready.status_code, 409)

        manager.get_result_file.side_effect = JobNotFoundError("Job not found.")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            missing = self.client.get("/api/extraction/jobs/job-404/download")
        self.assertEqual(missing.status_code, 404)

    def test_legacy_path_based_download_endpoint_is_removed(self) -> None:
        """The old path-based download route is no longer registered."""

        response = self.client.get("/api/extraction/download?path=result.xlsx")
        self.assertEqual(response.status_code, 404)

    def test_extraction_endpoints_require_api_key_when_configured(self) -> None:
        """Configured API keys are required on protected extraction endpoints."""

        manager = Mock()
        manager.submit.return_value = ExtractionJobSubmitResponse(
            job_id="job-123",
            status="QUEUED",
            status_url="/api/extraction/jobs/job-123",
            download_url="/api/extraction/jobs/job-123/download",
            created_at="2026-05-26T13:00:00Z",
        )

        with patch.dict("os.environ", {"API_KEY": "secret"}, clear=False), patch(
            "app.routers.extraction.get_job_manager", return_value=manager
        ):
            missing = self.client.post(
                "/api/extraction/run",
                files={"files": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )
            wrong = self.client.post(
                "/api/extraction/run",
                files={"files": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
                headers={"X-API-Key": "wrong"},
            )
            accepted = self.client.post(
                "/api/extraction/run",
                files={"files": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
                headers={"X-API-Key": "secret"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(accepted.status_code, 202)

    def test_extraction_endpoints_fail_closed_in_production_without_api_key(self) -> None:
        """Production extraction routes return 503 when API_KEY is not configured."""

        with patch.dict("os.environ", {"APP_ENV": "production"}, clear=False):
            os.environ.pop("API_KEY", None)
            response = self.client.get("/api/extraction/defaults")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
