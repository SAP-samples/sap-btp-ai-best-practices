"""Offline checks for the RPT direct-REST support modules (no network calls).

Covers rpt_ts_client (response flattening, hierarchy flattening, error
redaction) and rpt_decisions (validation, table translation, answer parsing).

Run from the rest/ folder or anywhere else:
    python tests/test_rest_offline.py
    python -I tests/test_rest_offline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpt_client_api import RPTAPIError  # noqa: E402
from rpt_decisions import (  # noqa: E402
    PLACEHOLDER,
    build_decision_payload,
    parse_decision_answers,
    validate_decision_request,
)
from rpt_ts_client import RPTTSClientAPI, forecast_to_frame, hierarchy_to_frame  # noqa: E402

TRIAGE = {
    "state": "Charged twice, fix it!",
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the tone?",
            "criteria": {"calm": "Polite", "angry": "Hostile"},
        },
        "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    },
    "examples": [
        {"state": "Great product", "answers": {"billing": False, "tone": "calm", "priority": "0"}},
        {"state": "Refund me now", "answers": {"billing": True, "tone": "angry", "priority": "1"}},
    ],
}


def expect_error(error_type, call, fragment):
    """Assert that ``call()`` raises ``error_type`` whose message contains ``fragment``."""
    try:
        call()
    except error_type as error:
        assert fragment in str(error), f"{fragment!r} not in {error!r}"
        return
    raise AssertionError(f"{error_type.__name__} not raised")


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


def test_decision_validation():
    """The documented decision rules fail early with readable messages."""
    validate_decision_request(TRIAGE)
    missing_label = {**TRIAGE, "examples": [
        {"state": "a", "answers": {"billing": False, "tone": "calm", "priority": "0"}},
        {"state": "b", "answers": {"billing": True, "tone": "calm", "priority": "1"}},
    ]}
    expect_error(ValueError, lambda: validate_decision_request(missing_label), "never appear")
    expect_error(ValueError, lambda: validate_decision_request({**TRIAGE, "examples": TRIAGE["examples"][:1]}),
                 "2-2,048 examples")
    unknown = {**TRIAGE, "questions": {"x": {"type": "rank", "instructions": "?"}}}
    expect_error(ValueError, lambda: validate_decision_request(unknown), "unsupported type")


def test_decision_payload_described_and_plain():
    """Examples become context rows, states become [PREDICT] query rows."""
    payload = build_decision_payload(TRIAGE)
    rows = payload["rows"]
    assert [row["row_id"] for row in rows] == ["example-0", "example-1", "query-0"]
    assert rows[1]["What is the tone?"] == "angry: Hostile"
    assert rows[1]["Is this about billing?"] == "yes" and rows[0]["How urgent?"] == "low"
    assert rows[2]["What is the tone?"] == PLACEHOLDER
    targets = {t["name"]: t["top_k"] for t in payload["prediction_config"]["target_columns"]}
    assert targets == {"Is this about billing?": 2, "What is the tone?": 2, "How urgent?": 2}

    plain = build_decision_payload(TRIAGE, encoding="plain")
    assert plain["rows"][1] == {"row_id": "example-1", "state": "Refund me now",
                                "billing": "true", "tone": "angry", "priority": "1"}


def test_decision_structured_and_chat_states():
    """Dict states become columns, list states become JSON text."""
    request = {**TRIAGE, "examples": [
        {"state": {"channel": "mail", "text": "Great"}, "answers": TRIAGE["examples"][0]["answers"]},
        {"state": {"channel": "chat", "text": "Refund"}, "answers": TRIAGE["examples"][1]["answers"]},
    ]}
    rows = build_decision_payload(request, [{"channel": "mail", "text": "?"}])["rows"]
    assert rows[2]["channel"] == "mail" and "state" not in rows[2]
    chat = build_decision_payload(TRIAGE, [[{"role": "user", "text": "hi"}]])["rows"][2]
    assert chat["state"] == '[{"role": "user", "text": "hi"}]'
    clash = {**request, "examples": [{"state": {"row_id": "x"}, "answers": e["answers"]} for e in request["examples"]]}
    expect_error(ValueError, lambda: build_decision_payload(clash), "clash")


def test_decision_answer_parsing():
    """RPT candidates map back to Playground answers; missing options get 0."""
    response = {"predictions": [
        {"row_id": "query-1",
         "Is this about billing?": [{"prediction": "no", "confidence": 0.8}],
         "What is the tone?": [{"prediction": "calm: Polite", "confidence": 1.0}],
         "How urgent?": [{"prediction": "low", "confidence": 1.0}]},
        {"row_id": "query-0",
         "Is this about billing?": [{"prediction": "yes", "confidence": 0.7}, {"prediction": "no", "confidence": 0.3}],
         "What is the tone?": [{"prediction": "angry: Hostile", "confidence": 0.6},
                               {"prediction": "calm: Polite", "confidence": 0.4}],
         "How urgent?": [{"prediction": "high", "confidence": 0.75}, {"prediction": "low", "confidence": 0.25}]},
    ]}
    first, second = parse_decision_answers(response, TRIAGE)
    assert first["billing"] == {"type": "noul", "noul": 0.7}
    assert first["tone"]["choice"] == "angry" and first["tone"]["probabilities"] == {"calm": 0.4, "angry": 0.6}
    assert first["priority"]["score"] == 0.75 and first["priority"]["legend"] == {"0": "low", "1": "high"}
    assert second["billing"]["noul"] == 0.0  # "yes" not returned
    assert second["tone"]["confidence"] == 1.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} offline checks passed")
