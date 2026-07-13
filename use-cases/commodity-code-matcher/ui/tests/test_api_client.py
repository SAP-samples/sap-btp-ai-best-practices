from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import api_client


class _Response:
    """Minimal requests response stub for UI client tests."""

    def __init__(self, payload=None, content: bytes = b"", status_code: int = 200) -> None:
        """Create a fake response with JSON payload and raw content."""

        self._payload = payload or {}
        self.content = content
        self.status_code = status_code

    def json(self):
        """Return the configured JSON payload."""

        return self._payload

    def raise_for_status(self) -> None:
        """Pretend the response status is successful."""

        return None


class _Upload:
    """Minimal Streamlit upload stub for UI client tests."""

    name = "sample.pdf"

    def getvalue(self) -> bytes:
        """Return a fake PDF byte stream."""

        return b"%PDF-1.4"


class ApiClientTests(unittest.TestCase):
    """Unit tests for the Streamlit API client wrappers."""

    def test_run_extraction_submits_job_and_polls_status_with_api_key(self) -> None:
        """run_extraction submits once and polls job status with auth headers."""

        post = Mock(
            return_value=_Response(
                {
                    "job_id": "job-123",
                    "status": "QUEUED",
                    "status_url": "/api/extraction/jobs/job-123",
                    "download_url": "/api/extraction/jobs/job-123/download",
                }
            )
        )
        get = Mock(return_value=_Response({"job_id": "job-123", "status": "SUCCEEDED", "progress": 100}))

        with patch.dict("os.environ", {"API_BASE_URL": "https://api.example", "API_KEY": "secret"}, clear=False), patch(
            "src.api_client.requests.post", post
        ), patch("src.api_client.requests.get", get):
            result = api_client.run_extraction(
                [_Upload()],
                poll_interval_seconds=0,
                slow_poll_interval_seconds=0,
                slow_after_seconds=999,
            )

        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertEqual(post.call_args.kwargs["headers"], {"X-API-Key": "secret"})
        self.assertEqual(get.call_args.args[0], "https://api.example/api/extraction/jobs/job-123")
        self.assertEqual(get.call_args.kwargs["headers"], {"X-API-Key": "secret"})

    def test_download_output_uses_job_download_endpoint(self) -> None:
        """download_output fetches the job-scoped workbook endpoint."""

        get = Mock(return_value=_Response(content=b"xlsx-bytes"))

        with patch.dict("os.environ", {"API_BASE_URL": "https://api.example"}, clear=False), patch(
            "src.api_client.requests.get", get
        ):
            content = api_client.download_output("job-123")

        self.assertEqual(content, b"xlsx-bytes")
        self.assertEqual(get.call_args.args[0], "https://api.example/api/extraction/jobs/job-123/download")


if __name__ == "__main__":
    unittest.main()
