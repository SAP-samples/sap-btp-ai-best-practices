"""HTTP tests for the lifecycle history dataset endpoints.

Run from the repository root:
    cd api && PYTHONPATH=. ../.venv/bin/python -m unittest tests.test_lifecycle_api
"""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.lifecycle import get_lifecycle_store, router
from app.services.database.backend import BackendType, DatabaseBackend
from app.services.lifecycle.store import LifecycleStore


def _workbook(invoices):
    """Return .xlsx bytes with one observed lifetime per invoice reference."""
    frame = pd.DataFrame({
        'Company Code': ['F1'] * len(invoices),
        'Customer': ['C1'] * len(invoices),
        'Invoice Reference': invoices,
        'Summary File Date (UTC)': ['2026-01-01'] * len(invoices),
        'Reconciliation File Date (UTC)': ['2026-01-31'] * len(invoices),
        'Purchase Price': [100.0] * len(invoices),
    })
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


class LifecycleApiTests(unittest.TestCase):
    """Upload, read, page and activate datasets through the authenticated router."""

    def setUp(self):
        """Serve the router against an isolated SQLite-backed lifecycle store."""
        self.directory = tempfile.TemporaryDirectory()
        store = LifecycleStore(DatabaseBackend(BackendType.SQLITE), Path(self.directory.name) / 'history.db')
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_lifecycle_store] = lambda: store
        self.key_patch = patch("app.security.API_KEY", "test-only")
        self.key_patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        """Release the key patch and temporary database."""
        self.key_patch.stop()
        self.directory.cleanup()

    def upload(self, dataset_id, content, **form):
        """POST one workbook with the API key."""
        return self.client.post("/api/workspace/lifecycle/datasets", headers={"X-API-Key": "test-only"},
                                files={"file": ("history.xlsx", content)},
                                data={"dataset_id": dataset_id, **form})

    def test_requires_api_key(self):
        """Every lifecycle route is behind the shared X-API-Key dependency."""
        self.assertEqual(self.client.get("/api/workspace/lifecycle/datasets").status_code, 403)

    def test_upload_list_detail_rows_and_activation(self):
        """A saved dataset is listed, readable page by page and activatable."""
        headers = {"X-API-Key": "test-only"}
        self.assertEqual(self.client.get("/api/workspace/lifecycle/active", headers=headers).status_code, 404)
        created = self.upload("reference-v1", _workbook(["A", "B", "C"]), activate="true")
        self.assertEqual(created.status_code, 200, created.text)
        body = created.json()
        self.assertEqual((body["row_count"], body["reference_purpose"], body["active"]), (3, "fixed_reference", True))

        second = self.upload("reference-v2", _workbook(["D"]))
        self.assertFalse(second.json()["active"])
        listed = self.client.get("/api/workspace/lifecycle/datasets", headers=headers).json()["items"]
        self.assertEqual({item["dataset_id"]: item["active"] for item in listed},
                         {"reference-v1": True, "reference-v2": False})

        page = self.client.get("/api/workspace/lifecycle/datasets/reference-v1/rows",
                               params={"limit": 2, "offset": 0}, headers=headers).json()
        self.assertEqual((page["total"], len(page["items"])), (3, 2))
        rest = self.client.get("/api/workspace/lifecycle/datasets/reference-v1/rows",
                               params={"limit": 2, "offset": 2}, headers=headers).json()
        self.assertEqual(len(rest["items"]), 1)
        self.assertEqual(page["items"][0]["credit_duration_days"], 30)

        switched = self.client.put("/api/workspace/lifecycle/active", json={"dataset_id": "reference-v2"}, headers=headers)
        self.assertEqual(switched.status_code, 200, switched.text)
        active = self.client.get("/api/workspace/lifecycle/active", headers=headers).json()
        self.assertEqual(active["dataset_id"], "reference-v2")
        detail = self.client.get("/api/workspace/lifecycle/datasets/reference-v1", headers=headers).json()
        self.assertFalse(detail["active"])

    def test_same_id_is_idempotent_and_changed_content_conflicts(self):
        """Identical bytes return the saved dataset; different bytes under the same ID return 409."""
        content = _workbook(["A"])
        self.assertEqual(self.upload("v1", content).status_code, 200)
        self.assertEqual(self.upload("v1", content).status_code, 200)
        self.assertEqual(self.upload("v1", _workbook(["A", "B"])).status_code, 409)

    def test_unknown_dataset_and_invalid_workbook(self):
        """Unknown IDs return 404; workbooks without the target columns return 422."""
        headers = {"X-API-Key": "test-only"}
        self.assertEqual(self.client.get("/api/workspace/lifecycle/datasets/missing", headers=headers).status_code, 404)
        self.assertEqual(self.client.put("/api/workspace/lifecycle/active", json={"dataset_id": "missing"},
                                         headers=headers).status_code, 404)
        buffer = io.BytesIO()
        pd.DataFrame({"Customer": ["C1"]}).to_excel(buffer, index=False)
        self.assertEqual(self.upload("bad", buffer.getvalue()).status_code, 422)
        self.assertEqual(self.upload("empty", b"").status_code, 422)


if __name__ == "__main__":
    unittest.main()
