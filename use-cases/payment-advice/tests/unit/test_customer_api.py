"""
Unit tests for the customer-management API (offline).

Uses FastAPI's TestClient with the API-key dependency overridden and the HANA-
backed customer functions monkeypatched, so no SAP HANA is touched.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.payment_advice.customers import Customer  # noqa: E402
from app.routers import payment_advice as pa  # noqa: E402
from app.security import get_api_key  # noqa: E402


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(pa.router, prefix="/api/payment-advice")
    app.dependency_overrides[get_api_key] = lambda: "test-key"
    pa._engine = object()  # bypass get_engine()/bootstrap in _engine_ready()
    return TestClient(app)


class CustomerApi(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_list_customers(self) -> None:
        fake = [
            Customer("globex", "Globex", True, "active"),
            Customer("newco", "NewCo", False, "active"),
        ]
        with mock.patch.object(pa, "list_customers", return_value=fake):
            resp = self.client.get("/api/payment-advice/customers")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(len(body), 2)
        self.assertEqual(body[0], {"client_key": "globex", "display_name": "Globex", "is_critical": True, "status": "active"})
        self.assertFalse(body[1]["is_critical"])

    def test_promote(self) -> None:
        updated = Customer("newco", "NewCo", True, "active")
        with mock.patch.object(pa, "set_critical", return_value=updated) as setter:
            resp = self.client.post("/api/payment-advice/customers/NewCo/promote")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["is_critical"])
        # path is normalized to the stored key and promoted (True)
        setter.assert_called_once()
        self.assertEqual(setter.call_args.args[1:], ("newco", True))

    def test_demote(self) -> None:
        updated = Customer("globex", "Globex", False, "active")
        with mock.patch.object(pa, "set_critical", return_value=updated) as setter:
            resp = self.client.post("/api/payment-advice/customers/globex/demote")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["is_critical"])
        self.assertEqual(setter.call_args.args[1:], ("globex", False))

    def test_unknown_client_404(self) -> None:
        with mock.patch.object(pa, "set_critical", return_value=None):
            resp = self.client.post("/api/payment-advice/customers/ghost/promote")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
