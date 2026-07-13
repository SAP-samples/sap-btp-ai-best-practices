from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.extraction_service import ExtractionConfig
from app.services.extraction_jobs import (
    ExtractionJobManager,
    InMemoryExtractionJobRepository,
    JobFilePayload,
    JobResultNotReadyError,
    QueueFullError,
)


class ExtractionJobManagerTests(unittest.TestCase):
    """Unit tests for the in-process extraction job manager."""

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
