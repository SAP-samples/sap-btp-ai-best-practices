"""Offline checks for the RPT direct-REST support modules (no network calls).

Covers rpt_ts_client (response flattening, hierarchy flattening, error
redaction) and rpt_decisions_client (route, 429 retry, error redaction, result
order, answer flattening).

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
import rpt_decisions_client  # noqa: E402
from rpt_decisions_client import RPTDecisionsClientAPI, answers_to_frame, most_likely  # noqa: E402
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

    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = str(payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    """Session whose ``post`` returns prepared responses and records the calls.

    A single response is returned for every call; a list is returned in order,
    and its last entry is repeated once the list is used up. A callable is
    called with the request JSON and its return value is used.
    """

    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if callable(self.response):
            return self.response(kwargs.get("json"))
        if isinstance(self.response, list):
            return self.response[min(len(self.calls), len(self.response)) - 1]
        return self.response


def offline_ts_client(response, client_class=RPTTSClientAPI):
    """Build a client without OAuth or discovery, using a fake session."""
    client = object.__new__(client_class)
    client.model_name, client.timeout_s, client.resource_group = "offline-model", 5, "default"
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


ANSWERS = {
    "billing": {"type": "noul", "noul": 0.85},
    "tone": {"type": "choice", "choice": "calm", "confidence": 0.415,
             "probabilities": {"angry": 0.28, "calm": 0.61, "neutral": 0.11}},
    "priority": {"type": "score", "score": 0.77, "confidence": 0.0,
                 "legend": {"0": "can wait", "1": "this week", "2": "today"},
                 "probabilities": {"0": 0.54, "1": 0.15, "2": 0.31}},
}


def test_decide_route_and_body():
    """decide posts the request unchanged to /systemone and returns the body."""
    client = offline_ts_client(FakeResponse(200, {"answers": ANSWERS, "usage": {}}), RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    url, kwargs = client._session.calls[0]
    assert url.endswith("/systemone") and kwargs["json"] == TRIAGE
    assert kwargs["headers"]["Authorization"] == "Bearer token-789"


def test_decide_retries_rate_limit():
    """HTTP 429 is retried, waiting Retry-After seconds."""
    responses = [FakeResponse(429, {"error": "TooManyRequest"}, {"Retry-After": "0"}),
                 FakeResponse(200, {"answers": ANSWERS})]
    client = offline_ts_client(responses, RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    assert len(client._session.calls) == 2


def test_decide_retries_transient_upstream_error():
    """A transient upstream HTTP 500 is retried without changing the request."""
    responses = [FakeResponse(500, {"message": "Temporary upstream failure"}, {"Retry-After": "0"}),
                 FakeResponse(200, {"answers": ANSWERS})]
    client = offline_ts_client(responses, RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    assert len(client._session.calls) == 2
    assert all(kwargs["json"] == TRIAGE for _, kwargs in client._session.calls)


def test_decide_error_is_redacted():
    """Validation errors raise with the server detail and credentials redacted."""
    detail = [{"loc": ["examples"], "msg": "examples must contain at least 2 entries token-789"}]
    client = offline_ts_client(FakeResponse(422, {"detail": detail, "status": {"code": 2}}), RPTDecisionsClientAPI)
    try:
        client.decide(TRIAGE)
    except rpt_decisions_client.RPTAPIError as error:
        assert error.http_status == 422 and "at least 2 entries" in str(error)
        assert "token-789" not in str(error) and "token-789" not in str(error.response_payload)
    else:
        raise AssertionError("RPTAPIError not raised")
    gateway = offline_ts_client(FakeResponse(400, {"error": "BadRequest", "message": "Subpath 'predict' is not allowed"}),
                                RPTDecisionsClientAPI)
    expect_error(RPTAPIError, lambda: gateway.decide(TRIAGE), "is not allowed")


def test_decide_many_keeps_order():
    """decide_many sends one request per state and returns results in input order."""
    def echo(body):
        """Answer with the state's length so each result can be traced back."""
        return FakeResponse(200, {"answers": {"billing": {"type": "noul", "noul": len(body["state"]) / 100}}})

    client = offline_ts_client(echo, RPTDecisionsClientAPI)
    states = ["a" * n for n in range(1, 21)]
    results = client.decide_many(TRIAGE, states, max_concurrency=4, progress=False)
    assert [r["answers"]["billing"]["noul"] for r in results] == [n / 100 for n in range(1, 21)]
    assert len(client._session.calls) == 20
    assert all(set(kwargs["json"]) == {"state", "questions", "examples"} for _, kwargs in client._session.calls)


def test_answers_to_frame_and_most_likely():
    """Answers are flattened to one row per state; most_likely picks one label."""
    frame = answers_to_frame([{"answers": ANSWERS}], index=["msg-1"])
    row = frame.loc["msg-1"].to_dict()
    assert row == {"billing": 0.85, "tone": "calm", "tone_confidence": 0.415,
                   "priority": 0.77, "priority_level": "0", "priority_confidence": 0.0}
    assert most_likely(ANSWERS["billing"]) is True and most_likely(ANSWERS["tone"]) == "calm"


def test_missing_confidence_is_not_derived():
    """Absent server confidence remains missing even with full probabilities."""
    answers = {key: {k: v for k, v in answer.items() if k != "confidence"}
               for key, answer in ANSWERS.items()}
    frame = answers_to_frame([{"answers": answers}])
    assert pd.isna(frame.loc[0, "tone_confidence"])
    assert pd.isna(frame.loc[0, "priority_confidence"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} offline checks passed")
