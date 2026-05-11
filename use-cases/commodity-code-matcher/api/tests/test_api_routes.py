from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app


class ApiRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_defaults_hide_filesystem_reference_paths(self) -> None:
        response = self.client.get("/api/extraction/defaults")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("community_codes_path", payload)
        self.assertNotIn("unspsc_context_path", payload)
        self.assertEqual(payload["top_k"], 5)

    def test_run_accepts_minimal_payload(self) -> None:
        result_payload = {
            "output_path": "/tmp/result.xlsx",
            "download_path": "result.xlsx",
            "output_exists": True,
            "file_count": 1,
            "llm_verify": True,
            "top_k": 5,
            "runtime_seconds": 1.2,
            "reference_data_version": "synthetic-v1",
            "headers_preview": [],
            "line_items_preview": [],
            "errors": [],
            "warnings": [],
        }
        with patch("app.routers.extraction.run_extraction_pipeline", new=AsyncMock(return_value=result_payload)):
            response = self.client.post(
                "/api/extraction/run",
                files={"files": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
                data={"llm_verify": "true"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reference_data_version"], "synthetic-v1")


if __name__ == "__main__":
    unittest.main()
