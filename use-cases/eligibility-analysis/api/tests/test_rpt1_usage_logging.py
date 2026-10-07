"""Tests for RPT-1 request usage telemetry."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


def _load_rpt1_client_class():
    """Load the direct RPT-1 client from its repo-local module path."""
    module_path = Path(__file__).resolve().parents[1] / "rpt1" / "rpt1_client.py"
    spec = importlib.util.spec_from_file_location("rpt1_client_for_tests", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RPT1Client


class _Response:
    """Small fake requests response object."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        """Return the fake JSON payload."""
        return self._payload


class _Session:
    """Fake requests session capturing the prediction payload."""

    def __init__(self):
        self.last_request_json = None

    def request(self, **kwargs):
        """Return a fake RPT-1 prediction response."""
        self.last_request_json = kwargs.get("json")
        return _Response(
            {
                "id": "rpt-request-1",
                "status": {"code": 1, "message": "ok"},
                "metadata": {"provider": "fake"},
                "predictions": [
                    {
                        "ID": "q-1",
                        "TARGET": [{"prediction": 42, "confidence": 0.9}],
                    },
                    {
                        "ID": "q-2",
                        "TARGET": [{"prediction": 84, "confidence": 0.8}],
                    },
                ],
            }
        )


class TestRPT1UsageLogging(unittest.TestCase):
    """Verify RPT-1 request telemetry counts cells and predictions."""

    def test_predict_emits_usage_event_and_returns_metadata(self) -> None:
        """RPT-1 predict logs input cells and prediction count per request."""
        RPT1Client = _load_rpt1_client_class()
        session = _Session()
        client = RPT1Client(
            aicore_base_url="https://example.invalid",
            aicore_auth_url="https://auth.example.invalid",
            client_id="client",
            client_secret="secret",
            model_name="sap-rpt-1-small",
            session=session,
            max_retries=0,
        )
        # Preset the resolved deployment so this test covers telemetry, not discovery.
        client.deployment_url = "https://deployment.example.invalid"
        client.deployment_id = "deployment-1"
        client._access_token = "token"
        client._access_token_expiry = time.time() + 3600
        context_df = pd.DataFrame(
            {
                "ID": ["c-1", "c-2", "c-3"],
                "FEATURE": [1, 2, 3],
                "TARGET": [10, 20, 30],
            }
        )
        query_df = pd.DataFrame(
            {
                "ID": ["q-1", "q-2"],
                "FEATURE": [4, 5],
            }
        )

        client.fit(
            context_df,
            target_columns=["TARGET"],
            index_column="ID",
            task_types={"TARGET": "regression"},
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            result = client.predict(query_df)

        event = json.loads(stream.getvalue())
        self.assertEqual(event["schema_version"], "btp.rpt1_usage.v1")
        self.assertEqual(event["event_type"], "rpt1_usage")
        self.assertEqual(event["provider"], "sap-ai-core")
        self.assertEqual(event["rpt_endpoint"], "predict")
        self.assertEqual(event["context_rows"], 3)
        self.assertEqual(event["query_rows"], 2)
        self.assertEqual(event["input_rows"], 5)
        self.assertEqual(event["input_columns"], 3)
        self.assertEqual(event["input_cells"], 15)
        self.assertEqual(event["prediction_count"], 2)
        self.assertEqual(event["target_columns"], ["TARGET"])
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["request_id"], "rpt-request-1")
        self.assertEqual(event["model_name"], "sap-rpt-1-small")
        self.assertIsNone(event["model_version"])
        self.assertEqual(event["deployment_id"], "deployment-1")

        usage = result.metadata["rpt1_usage"]
        self.assertEqual(usage["input_cells"], 15)
        self.assertEqual(usage["prediction_count"], 2)
        self.assertEqual(usage["context_rows"], 3)
        self.assertEqual(usage["query_rows"], 2)


if __name__ == "__main__":
    unittest.main()
