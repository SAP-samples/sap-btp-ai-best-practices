from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.models.extraction import (
    ExtractionJobResultResponse,
    ExtractionJobStatusResponse,
    ExtractionJobSubmitResponse,
    JobResultFile,
    JouleLineItem,
)
from app.services.extraction_jobs import (
    JobNotFoundError,
    JobResultNotReadyError,
    JobStorageError,
    QueueFullError,
)


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

    def test_raw_pdf_submission_uses_fixed_server_owned_options(self) -> None:
        """Raw Joule uploads submit one PDF with LLM verification enabled."""

        manager = Mock()
        manager.submit.return_value = ExtractionJobSubmitResponse(
            job_id="job-raw",
            status="QUEUED",
            status_url="/api/extraction/jobs/job-raw",
            download_url="/api/extraction/jobs/job-raw/download",
            created_at="2026-07-18T08:00:00Z",
        )

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.post(
                "/api/extraction/jobs",
                content=b"%PDF-1.7\nraw-pdf",
                headers={"Content-Type": "application/octet-stream"},
            )

        self.assertEqual(response.status_code, 202)
        submitted = manager.submit.call_args.kwargs
        self.assertEqual(len(submitted["files"]), 1)
        self.assertEqual(submitted["files"][0].filename, "joule_document.pdf")
        self.assertEqual(submitted["files"][0].content, b"%PDF-1.7\nraw-pdf")
        self.assertTrue(submitted["config"].llm_verify)
        self.assertEqual(submitted["config"].top_k, 5)

    def test_raw_pdf_submission_accepts_exact_limit_and_media_type_parameters(self) -> None:
        """Exactly 10 MiB and a parameterized octet-stream media type are accepted."""

        manager = Mock()
        manager.submit.return_value = ExtractionJobSubmitResponse(
            job_id="job-limit",
            status="QUEUED",
            status_url="/api/extraction/jobs/job-limit",
            download_url="/api/extraction/jobs/job-limit/download",
            created_at="2026-07-18T08:00:00Z",
        )
        exact_limit = b"%PDF-" + b"x" * (10 * 1024 * 1024 - 5)

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.post(
                "/api/extraction/jobs",
                content=exact_limit,
                headers={"Content-Type": "application/octet-stream; charset=binary"},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(len(manager.submit.call_args.kwargs["files"][0].content), 10 * 1024 * 1024)

    def test_raw_pdf_submission_rejects_invalid_bodies(self) -> None:
        """Raw submissions reject wrong media types, empty bodies, oversized files, and non-PDF bytes."""

        wrong_media = self.client.post(
            "/api/extraction/jobs",
            content=b"%PDF-1.7",
            headers={"Content-Type": "application/pdf"},
        )
        empty = self.client.post(
            "/api/extraction/jobs",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        oversized = self.client.post(
            "/api/extraction/jobs",
            content=b"%PDF-" + b"x" * (10 * 1024 * 1024),
            headers={"Content-Type": "application/octet-stream"},
        )
        invalid_pdf = self.client.post(
            "/api/extraction/jobs",
            content=b"not-a-pdf",
            headers={"Content-Type": "application/octet-stream"},
        )

        self.assertEqual(wrong_media.status_code, 415)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(invalid_pdf.status_code, 400)

    def test_raw_pdf_submission_preserves_queue_full_behavior(self) -> None:
        """Raw submissions map the existing queue-capacity error to HTTP 429."""

        manager = Mock()
        manager.submit.side_effect = QueueFullError("Too many jobs.")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.post(
                "/api/extraction/jobs",
                content=b"%PDF-1.7",
                headers={"Content-Type": "application/octet-stream"},
            )

        self.assertEqual(response.status_code, 429)

    def test_raw_pdf_submission_maps_manager_failures(self) -> None:
        """Raw submissions preserve validation, storage, and runtime error status codes."""

        cases = (
            (ValueError("invalid job"), 400),
            (JobStorageError("offline"), 503),
            (RuntimeError("worker failed"), 500),
        )
        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__):
                manager = Mock()
                manager.submit.side_effect = error
                with patch("app.routers.extraction.get_job_manager", return_value=manager):
                    response = self.client.post(
                        "/api/extraction/jobs",
                        content=b"%PDF-1.7",
                        headers={"Content-Type": "application/octet-stream"},
                    )
                self.assertEqual(response.status_code, expected_status)

    def test_raw_pdf_submission_requires_configured_api_key(self) -> None:
        """The Joule upload route is protected by the shared extraction API key."""

        manager = Mock()
        manager.submit.return_value = ExtractionJobSubmitResponse(
            job_id="job-auth",
            status="QUEUED",
            status_url="/api/extraction/jobs/job-auth",
            download_url="/api/extraction/jobs/job-auth/download",
            created_at="2026-07-18T08:00:00Z",
        )
        with patch.dict("os.environ", {"API_KEY": "secret"}, clear=False), patch(
            "app.routers.extraction.get_job_manager", return_value=manager
        ):
            missing = self.client.post(
                "/api/extraction/jobs",
                content=b"%PDF-1.7",
                headers={"Content-Type": "application/octet-stream"},
            )
            accepted = self.client.post(
                "/api/extraction/jobs",
                content=b"%PDF-1.7",
                headers={"Content-Type": "application/octet-stream", "X-API-Key": "secret"},
            )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(accepted.status_code, 202)

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

    def test_job_result_returns_one_paginated_joule_page(self) -> None:
        """Completed result requests return the manager's deterministic Joule payload."""

        manager = Mock()
        manager.get_result_page.return_value = ExtractionJobResultResponse(
            job_id="job-123",
            status="SUCCEEDED",
            pagination={
                "current_page": 2,
                "page_size": 30,
                "total_items": 31,
                "total_pages": 2,
                "previous_page": 1,
                "next_page": None,
            },
            line_items=[
                JouleLineItem(
                    description="Brake pads",
                    net_amount=100,
                    quantity=2,
                    unit_price=50.0,
                    ai_suggested_commodity_code="RC0001",
                    ai_confidence_score="91%",
                    ai_reasoning="Best semantic match.",
                )
            ],
        )

        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            response = self.client.get("/api/extraction/jobs/job-123/result?page=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"job_id", "status", "line_items", "pagination"})
        self.assertEqual(payload["line_items"][0]["quantity"], 2)
        self.assertEqual(payload["pagination"]["current_page"], 2)
        self.assertEqual(payload["pagination"]["previous_page"], 1)
        self.assertIsNone(payload["pagination"]["next_page"])
        manager.get_result_page.assert_called_once_with("job-123", page=2)

    def test_job_result_maps_invalid_missing_pending_and_storage_errors(self) -> None:
        """The result route maps pagination and repository failures to its public status codes."""

        manager = Mock()
        manager.get_result_page.side_effect = ValueError("Page must be at least 1.")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            invalid = self.client.get("/api/extraction/jobs/job-123/result?page=0")
            non_numeric = self.client.get("/api/extraction/jobs/job-123/result?page=abc")

        manager.get_result_page.side_effect = JobNotFoundError("missing")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            missing = self.client.get("/api/extraction/jobs/job-404/result")

        manager.get_result_page.side_effect = JobResultNotReadyError("pending")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            pending = self.client.get("/api/extraction/jobs/job-123/result")

        manager.get_result_page.side_effect = JobStorageError("offline")
        with patch("app.routers.extraction.get_job_manager", return_value=manager):
            unavailable = self.client.get("/api/extraction/jobs/job-123/result")

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(non_numeric.status_code, 400)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(pending.status_code, 409)
        self.assertEqual(unavailable.status_code, 503)

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
