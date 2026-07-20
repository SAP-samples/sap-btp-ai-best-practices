from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.extraction_service import ExtractionConfig, ExtractionResult, run_extraction_for_paths
from app.services.extraction_jobs import (
    ExtractionJobManager,
    HanaExtractionJobRepository,
    InMemoryExtractionJobRepository,
    JobFilePayload,
    JobNotFoundError,
    JobResultNotReadyError,
    JobStorageError,
    QueueFullError,
)


class ExtractionJobManagerTests(unittest.TestCase):
    """Unit tests for the in-process extraction job manager."""

    def _hana_repository_with_row(self, row):
        """Build a HANA repository backed by one minimal fake query row.

        Args:
            row: Tuple returned by the fake cursor, or ``None`` for a missing job.

        Returns:
            The configured repository, connection, and cursor mocks.
        """

        repository = HanaExtractionJobRepository(schema="TEST_SCHEMA")
        connection = Mock()
        cursor = Mock()
        cursor.fetchone.return_value = row
        cursor.description = [("JOB_ID",), ("STATUS",), ("RESULT_METADATA_JSON",)]
        connection.cursor.return_value = cursor
        repository._connect = Mock(return_value=connection)
        repository._ensure_ready = Mock()
        return repository, connection, cursor

    def test_successful_job_stores_status_previews_and_result_blob(self) -> None:
        """Successful jobs persist status metadata and downloadable result bytes."""

        repository = InMemoryExtractionJobRepository()

        def run_pipeline(pdf_paths, _config):
            """Fake successful extraction pipeline for manager tests."""

            output_path = Path(pdf_paths[0]).with_name("result.xlsx")
            output_path.write_bytes(b"xlsx-bytes")
            return {
                "output_path": str(output_path),
                "output_exists": True,
                "file_count": len(pdf_paths),
                "llm_verify": True,
                "top_k": 5,
                "runtime_seconds": 1.5,
                "reference_data_version": "synthetic-v1",
                "headers_preview": [{"file": "sample.pdf"}],
                "line_items_preview": [{"description": "Brake pads"}],
                "errors": [],
                "warnings": [],
            }

        manager = ExtractionJobManager(
            repository=repository,
            run_pipeline=run_pipeline,
            max_workers=1,
            max_queued_jobs=2,
        )

        submitted = manager.submit(
            files=[JobFilePayload(filename="sample.pdf", content_type="application/pdf", content=b"%PDF-1.4")],
            config=ExtractionConfig(llm_verify=True, top_k=5),
        )
        manager.wait_for_job(submitted.job_id, timeout_seconds=5)

        status = manager.get_status(submitted.job_id)
        result = manager.get_result_file(submitted.job_id)

        self.assertEqual(status.status, "SUCCEEDED")
        self.assertEqual(status.progress, 100)
        self.assertEqual(status.reference_data_version, "synthetic-v1")
        self.assertEqual(status.line_items_preview, [{"description": "Brake pads"}])
        self.assertEqual(result.content, b"xlsx-bytes")
        self.assertEqual(result.filename, "result.xlsx")

    def test_failed_job_records_error_and_does_not_return_download(self) -> None:
        """Failed jobs store their error and block result downloads."""

        repository = InMemoryExtractionJobRepository()

        def run_pipeline(_pdf_paths, _config):
            """Fake failing extraction pipeline for manager tests."""

            raise RuntimeError("classification failed")

        manager = ExtractionJobManager(
            repository=repository,
            run_pipeline=run_pipeline,
            max_workers=1,
            max_queued_jobs=2,
        )

        submitted = manager.submit(
            files=[JobFilePayload(filename="sample.pdf", content_type="application/pdf", content=b"%PDF-1.4")],
            config=ExtractionConfig(),
        )
        manager.wait_for_job(submitted.job_id, timeout_seconds=5)

        status = manager.get_status(submitted.job_id)

        self.assertEqual(status.status, "FAILED")
        self.assertIn("classification failed", status.errors)
        with self.assertRaises(JobResultNotReadyError):
            manager.get_result_file(submitted.job_id)

    def test_joule_results_are_paginated_in_fixed_pages_of_thirty(self) -> None:
        """More than 20 enriched rows serialize, persist, and paginate end to end."""

        repository = InMemoryExtractionJobRepository()
        enriched = pd.DataFrame(
            [
                {
                    "description": f"Item {index}",
                    "netAmount": index,
                    "quantity": 1,
                    "unitPrice": float(index),
                    "LLM_Suggestion_Desc": f"RC{index:04d}",
                    "LLM_Confidence_Desc": 0.9,
                    "LLM_Reason_Desc": "Best semantic match.",
                }
                for index in range(31)
            ]
        )

        def fake_embedding_pipeline(pdf_paths, _config):
            """Return 31 enriched matcher rows and the required workbook artifact."""

            output_path = Path(pdf_paths[0]).with_name("result.xlsx")
            output_path.write_bytes(b"xlsx-bytes")
            return ExtractionResult(
                output_path=output_path,
                headers_df=pd.DataFrame(),
                line_items_df=enriched,
                runtime_seconds=1.0,
                reference_data_version="v1",
            )

        with patch("app.services.extraction_service._run_embedding_pipeline", side_effect=fake_embedding_pipeline):
            manager = ExtractionJobManager(repository=repository, run_pipeline=run_extraction_for_paths)
            submitted = manager.submit(
                files=[JobFilePayload(filename="sample.pdf", content_type="application/pdf", content=b"%PDF-1.4")],
                config=ExtractionConfig(llm_verify=True),
            )
            manager.wait_for_job(submitted.job_id, timeout_seconds=5)

        first_page = manager.get_result_page(submitted.job_id, page=1)
        second_page = manager.get_result_page(submitted.job_id, page=2)
        stored_items = repository.get_result_metadata(submitted.job_id)["joule_line_items"]

        self.assertEqual(len(stored_items), 31)
        self.assertEqual(len(first_page.line_items), 30)
        self.assertEqual(len(second_page.line_items), 1)
        self.assertEqual(first_page.pagination.total_items, 31)
        self.assertEqual(first_page.pagination.total_pages, 2)
        self.assertIsNone(first_page.pagination.previous_page)
        self.assertEqual(first_page.pagination.next_page, 2)
        self.assertEqual(second_page.pagination.previous_page, 1)
        self.assertIsNone(second_page.pagination.next_page)
        self.assertEqual(second_page.line_items[0].description, "Item 30")
        with self.assertRaises(ValueError):
            manager.get_result_page(submitted.job_id, page=0)
        with self.assertRaises(ValueError):
            manager.get_result_page(submitted.job_id, page=3)

    def test_hana_result_metadata_returns_succeeded_json(self) -> None:
        """Succeeded HANA rows decode their structured result NCLOB."""

        repository, connection, _cursor = self._hana_repository_with_row(
            ("job-1", "SUCCEEDED", '{"joule_line_items":[{"description":"Item"}]}')
        )

        metadata = repository.get_result_metadata("job-1")

        self.assertEqual(metadata["joule_line_items"][0]["description"], "Item")
        connection.close.assert_called_once()

    def test_hana_result_metadata_preserves_missing_and_nonterminal_errors(self) -> None:
        """Missing and unfinished HANA jobs retain their public repository errors."""

        missing_repository, _connection, _cursor = self._hana_repository_with_row(None)
        with self.assertRaises(JobNotFoundError):
            missing_repository.get_result_metadata("missing")

        pending_repository, _connection, _cursor = self._hana_repository_with_row(
            ("job-1", "RUNNING", "{}")
        )
        with self.assertRaises(JobResultNotReadyError):
            pending_repository.get_result_metadata("job-1")

    def test_hana_result_metadata_wraps_malformed_json_and_storage_failures(self) -> None:
        """Malformed NCLOB JSON and SQL failures are normalized as storage errors."""

        malformed_repository, _connection, _cursor = self._hana_repository_with_row(
            ("job-1", "SUCCEEDED", "{not-json")
        )
        with self.assertRaises(JobStorageError):
            malformed_repository.get_result_metadata("job-1")

        failing_repository, _connection, cursor = self._hana_repository_with_row(
            ("job-1", "SUCCEEDED", "{}")
        )
        cursor.execute.side_effect = RuntimeError("database offline")
        with self.assertRaises(JobStorageError):
            failing_repository.get_result_metadata("job-1")

    def test_queue_limit_counts_running_jobs(self) -> None:
        """The queue limit includes jobs that are already running."""

        repository = InMemoryExtractionJobRepository()
        started = threading.Event()
        release = threading.Event()

        def run_pipeline(pdf_paths, _config):
            """Fake blocking extraction pipeline for queue-limit tests."""

            started.set()
            release.wait(timeout=5)
            output_path = Path(pdf_paths[0]).with_name("result.xlsx")
            output_path.write_bytes(b"xlsx-bytes")
            return {
                "output_path": str(output_path),
                "output_exists": True,
                "file_count": 1,
                "llm_verify": False,
                "top_k": 5,
                "runtime_seconds": 0.1,
                "reference_data_version": "synthetic-v1",
                "headers_preview": [],
                "line_items_preview": [],
                "errors": [],
                "warnings": [],
            }

        manager = ExtractionJobManager(
            repository=repository,
            run_pipeline=run_pipeline,
            max_workers=1,
            max_queued_jobs=1,
        )

        first = manager.submit(
            files=[JobFilePayload(filename="first.pdf", content_type="application/pdf", content=b"%PDF-1.4")],
            config=ExtractionConfig(),
        )
        self.assertTrue(started.wait(timeout=5))

        with self.assertRaises(QueueFullError):
            manager.submit(
                files=[JobFilePayload(filename="second.pdf", content_type="application/pdf", content=b"%PDF-1.4")],
                config=ExtractionConfig(),
            )

        release.set()
        manager.wait_for_job(first.job_id, timeout_seconds=5)


if __name__ == "__main__":
    unittest.main()
