"""Offline checks for the SAP-RPT-TS client asset (no network calls).

Covers response flattening, hierarchy flattening, and HTTP error redaction.
The shared OAuth/discovery client comes from the sibling ``rpt-1`` skill.

Run from the skill folder or anywhere else:
    python tests/test_rpt_ts_client.py
    python -I tests/test_rpt_ts_client.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "assets"))
sys.path.insert(0, str(SKILL_ROOT.parent / "rpt-1" / "assets"))  # shared rpt_client_api

from rpt_client_api import RPTAPIError  # noqa: E402
from rpt_ts_client import RPTTSClientAPI, forecast_to_frame, hierarchy_to_frame  # noqa: E402


def test_forecast_to_frame():
    """Nested [series][variate][step] arrays become one row per step."""
    response = {
        "result": [[[10.0, 11.0]], [[5.0, 6.0]]],
        "confidence_interval_output": [[[[9.0, 10.0], [11.0, 12.0]]], [[[4.0, 5.0], [6.0, 7.0]]]],
        "result_timestamp": [["2024-05-01", "2024-06-01"], None],
    }
    frame = forecast_to_frame(response, series_ids=["a", "b"])
    assert len(frame) == 4
    first = frame.iloc[0].to_dict()
    assert first == {"series": "a", "variate": 0, "step": 1, "timestamp": "2024-05-01",
                     "point": 10.0, "lower": 9.0, "upper": 11.0}
    assert pd.isna(frame.iloc[3]["timestamp"]) and frame.iloc[3]["upper"] == 7.0
    no_interval = forecast_to_frame({"result": [[[1.0]]], "confidence_interval_output": None})
    assert pd.isna(no_interval.iloc[0]["lower"])


def test_hierarchy_to_frame():
    """The reconciled tree is flattened with levels and joined paths."""
    response = {
        "result": [],
        "hierarchical_result": [
            {"": [[3.0]], "__children__": [
                {"north": [[2.0]], "__children__": [{"north | x": [[2.0]]}]},
                {"south": [[1.0]]},
            ]}
        ],
    }
    frame = hierarchy_to_frame(response)
    assert frame["path"].tolist() == ["total", "north", "north / north | x", "south"]
    assert frame["level"].tolist() == [0, 1, 2, 1]


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Session whose ``post`` returns a prepared response and records the call."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def offline_ts_client(response):
    """Build an RPTTSClientAPI without OAuth or discovery, using a fake session."""
    client = object.__new__(RPTTSClientAPI)
    client.model_name, client.timeout_s, client.resource_group = "sap-rpt-ts", 5, "default"
    client.deployment_url = "https://example.invalid/deployments/d1"
    client._client_id, client._client_secret = "client-id-123", "secret-456"
    client._token, client._token_expires_at = "token-789", time.monotonic() + 3600
    client._session = FakeSession(response)
    return client


def test_ts_forecast_success_and_redacted_error():
    """Success returns the body; HTTP errors raise with credentials redacted."""
    ok = offline_ts_client(FakeResponse(200, {"result": [[[1.0]]], "usage": {}}))
    assert ok.forecast({"horizon": 1, "data": [{"target": [1, 2]}]})["result"] == [[[1.0]]]
    url, kwargs = ok._session.calls[0]
    assert url.endswith("/predict") and kwargs["headers"]["Authorization"] == "Bearer token-789"

    failing = offline_ts_client(FakeResponse(422, {"detail": "bad horizon, token-789 secret-456"}))
    try:
        failing.forecast({"data": []})
    except RPTAPIError as error:
        text = str(error)
        assert error.http_status == 422 and "bad horizon" in text
        assert "token-789" not in text and "secret-456" not in text
        assert "secret-456" not in str(error.response_payload)
    else:
        raise AssertionError("RPTAPIError not raised")



if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} offline checks passed")
