from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.optimizer import router as optimizer_router
from app.security import get_api_key


class _ManagerStub:
    def __init__(self, record):
        self._record = record

    def get_process(self, process_id: str):
        if process_id != "proc-1":
            return None
        return self._record


class TestOptimizerStatusProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = FastAPI()
        cls.app.include_router(optimizer_router, prefix="/api", tags=["optimizer"])
        cls.app.dependency_overrides[get_api_key] = lambda: "test"
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.dependency_overrides.pop(get_api_key, None)

    @patch("app.routers.optimizer.get_process_manager")
    def test_status_includes_progress_object(self, mock_get_mgr) -> None:
        progress = {
            "step_index": 7,
            "step_total": 10,
            "step_code": "estimate_lifetime_rpt1",
            "step_label": "Estimating invoice lifetime with RPT-1",
            "phase_status": "running",
            "elapsed_seconds": 123.0,
            "updated_at": "2026-02-25T10:18:00",
            "details": {"batches_total": 50, "batches_completed": 10},
        }
        mock_get_mgr.return_value = _ManagerStub(
            {
                "id": "proc-1",
                "status": "running",
                "started_at": "2026-02-25T10:45:00",
                "error_message": None,
                "candidate_count": None,
                "selected_count": None,
                "excluded_count": None,
                "progress_json": json.dumps(progress),
            }
        )

        response = self.client.get("/api/optimizer/processes/proc-1/status")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "running")
        self.assertEqual(body["started_at"], "2026-02-25T10:45:00")
        self.assertEqual(body["progress"]["step_index"], 7)
        self.assertEqual(body["progress"]["step_total"], 10)


if __name__ == "__main__":
    unittest.main()
