"""
Unit tests for the client-management methods added to dox_client (offline).

Uses a fake requests.Session (no SAP credentials or network), per the
sap-document-ai-client skill. Verifies endpoint paths, query params, and JSON
bodies for list_clients, create_clients, and delete_schemas.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from dox_client import ServiceKey  # noqa: E402
from dox_client.sap_dox_client import SapDoxClient  # noqa: E402


class _FakeResp:
    def __init__(self, status: int = 200, payload=None) -> None:
        self.status_code = status
        self._payload = payload
        self.text = ""
        self.content = b"x" if payload is not None else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Returns queued responses for .request and a canned token for .post."""

    def __init__(self, responses: list[_FakeResp]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url, **kwargs):  # token endpoint
        return _FakeResp(200, {"access_token": "tok", "expires_in": 3600})

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def _client(responses: list[_FakeResp]) -> tuple[SapDoxClient, _FakeSession]:
    key = ServiceKey(
        token_base_url="https://token.example",
        client_id="cid",
        client_secret="sec",
        dox_base_url="https://dox.example",
        swagger_path="/document-information-extraction/v1/",
    )
    session = _FakeSession(responses)
    return SapDoxClient(key, session=session), session


class ClientManagement(unittest.TestCase):
    def test_create_clients_body(self) -> None:
        client, session = _client([_FakeResp(200, {"inserted": 1})])
        client.create_clients([{"clientId": "ai4u_payment_advice", "clientName": "AI4U Payment Advice"}])
        method, url, kwargs = session.calls[-1]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/clients"))
        self.assertEqual(kwargs["json"], {"value": [{"clientId": "ai4u_payment_advice", "clientName": "AI4U Payment Advice"}]})

    def test_create_clients_rejects_empty(self) -> None:
        client, _ = _client([])
        with self.assertRaises(ValueError):
            client.create_clients([])

    def test_list_clients_normalizes_wrapper(self) -> None:
        client, session = _client([_FakeResp(200, {"payload": [{"clientId": "c_00"}]})])
        result = client.list_clients(limit=50)
        self.assertEqual(result, [{"clientId": "c_00"}])
        method, url, kwargs = session.calls[-1]
        self.assertEqual(method, "GET")
        self.assertTrue(url.endswith("/clients"))
        self.assertEqual(kwargs["params"], {"limit": 50, "offset": 0})

    def test_delete_schemas_body_and_params(self) -> None:
        client, session = _client([_FakeResp(204, None)])
        client.delete_schemas("schema-1", client_id="ai4u_payment_advice")
        method, url, kwargs = session.calls[-1]
        self.assertEqual(method, "DELETE")
        self.assertTrue(url.endswith("/schemas"))
        self.assertEqual(kwargs["params"], {"clientId": "ai4u_payment_advice"})
        self.assertEqual(kwargs["json"], {"value": ["schema-1"]})

    def test_delete_schemas_rejects_empty(self) -> None:
        client, _ = _client([])
        with self.assertRaises(ValueError):
            client.delete_schemas([])

    def test_partial_field_dicts_get_setup_defaults(self) -> None:
        """A dict with only name/label/formattingType must still carry setupType/setup, or SAP returns 500."""
        client, session = _client([_FakeResp(200, {})])
        client.add_fields_to_schema_version("schema-1", 2, client_id="ai4u_payment_advice", replace=True, full_definitions=True,
                                            line_item_fields=[{"name": "region", "label": "Rg", "formattingType": "string"}])
        field = session.calls[-1][2]["json"]["lineItemFields"][0]
        self.assertEqual((field["setupType"], field["setup"]["type"], field["formatting"]), ("static", "auto", {}))
        self.assertEqual((field["label"], field["formattingType"]), ("Rg", "string"))


if __name__ == "__main__":
    unittest.main()
