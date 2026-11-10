"""Offline contract checks for the direct decisions client, without credentials.

Run from the skill folder or from any directory:
    python tests/test_rpt_decisions.py
    python -I tests/test_rpt_decisions.py
"""
from __future__ import annotations

import sys
import json
import importlib.util
import time
from pathlib import Path
import pandas as pd

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "assets"))
sys.path.insert(0, str(SKILL_ROOT.parent / "rpt-1" / "assets"))
from rpt_client_api import RPTAPIError
import rpt_decisions_client
from rpt_decisions_client import RPTDecisionsClientAPI, answers_to_frame, most_likely

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


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code, payload, headers=None):
        """Initialize HTTP status, payload and optional fake response headers."""
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload
        self.text = str(payload)
        self.headers = headers or {}

    def json(self):
        """Return the prepared JSON payload with no transport or parsing."""
        return self._payload


class FakeSession:
    """Session whose ``post`` returns prepared responses and records the calls.

    A single response is returned for every call; a list is returned in order,
    and its last entry is repeated once the list is used up. A callable is
    called with the request JSON and its return value is used.
    """

    def __init__(self, response):
        """Initialize a session with a response, sequence or JSON callback."""
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        """Record URL/options and return the next configured fake response."""
        self.calls.append((url, kwargs))
        if callable(self.response):
            return self.response(kwargs.get("json"))
        if isinstance(self.response, list):
            return self.response[min(len(self.calls), len(self.response)) - 1]
        return self.response


def offline_client(response, client_class=RPTDecisionsClientAPI):
    """Build a client without OAuth or discovery, using a fake session."""
    client = object.__new__(client_class)
    client.model_name, client.timeout_s, client.resource_group = "offline-model", 5, "default"
    client.deployment_url = "https://example.invalid/deployments/d1"
    client._client_id, client._client_secret = "client-id-123", "secret-456"
    client._token, client._token_expires_at = "token-789", time.monotonic() + 3600
    client._session = FakeSession(response)
    return client


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
    client = offline_client(FakeResponse(200, {"answers": ANSWERS, "usage": {}}), RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    url, kwargs = client._session.calls[0]
    assert url.endswith("/systemone") and kwargs["json"] == TRIAGE
    assert kwargs["headers"]["Authorization"] == "Bearer token-789"


def test_decide_retries_rate_limit():
    """HTTP 429 is retried, waiting Retry-After seconds."""
    responses = [FakeResponse(429, {"error": "TooManyRequest"}, {"Retry-After": "0"}),
                 FakeResponse(200, {"answers": ANSWERS})]
    client = offline_client(responses, RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    assert len(client._session.calls) == 2


def test_decide_retries_transient_upstream_error():
    """A transient upstream HTTP 500 is retried without changing the request."""
    responses = [FakeResponse(500, {"message": "Temporary upstream failure"}, {"Retry-After": "0"}),
                 FakeResponse(200, {"answers": ANSWERS})]
    client = offline_client(responses, RPTDecisionsClientAPI)
    assert client.decide(TRIAGE)["answers"] == ANSWERS
    assert len(client._session.calls) == 2
    assert all(kwargs["json"] == TRIAGE for _, kwargs in client._session.calls)


def test_decide_error_is_redacted():
    """Validation errors raise with the server detail and credentials redacted."""
    detail = [{"loc": ["examples"], "msg": "examples must contain at least 2 entries token-789"}]
    client = offline_client(FakeResponse(422, {"detail": detail, "status": {"code": 2}}), RPTDecisionsClientAPI)
    try:
        client.decide(TRIAGE)
    except rpt_decisions_client.RPTAPIError as error:
        assert error.http_status == 422 and "at least 2 entries" in str(error)
        assert "token-789" not in str(error) and "token-789" not in str(error.response_payload)
    else:
        raise AssertionError("RPTAPIError not raised")
    gateway = offline_client(FakeResponse(400, {"error": "BadRequest", "message": "Subpath 'predict' is not allowed"}),
                                RPTDecisionsClientAPI)
    expect_error(RPTAPIError, lambda: gateway.decide(TRIAGE), "is not allowed")


def test_decide_many_keeps_order():
    """decide_many sends one request per state and returns results in input order."""
    def echo(body):
        """Answer with the state's length so each result can be traced back."""
        return FakeResponse(200, {"answers": {"billing": {"type": "noul", "noul": len(body["state"]) / 100}}})

    client = offline_client(echo, RPTDecisionsClientAPI)
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


def test_bundled_examples_cover_declared_labels():
    """All bundled synthetic requests have complete labels and valid type shapes."""
    fixtures = json.loads((SKILL_ROOT / "assets" / "decision_examples.json").read_text())
    for name in ["triage", "ticket_routing", "chat_triage"]:
        request = fixtures[name]
        assert 2 <= len(request["examples"]) <= 2048
        assert 1 <= len(request["questions"]) <= 10
        for key, question in request["questions"].items():
            assert question["instructions"]
            labels = [example["answers"][key] for example in request["examples"]]
            if question["type"] == "noul":
                assert all(isinstance(label, bool) for label in labels)
                assert set(labels) == {True, False}
            elif question["type"] == "choice":
                assert set(labels) == set(question["criteria"])
            else:
                assert set(labels) == {str(i) for i in range(len(question["criteria"]))}
    assert len(fixtures["triage"]["examples"]) == 18


def test_live_summary_preserves_confidence():
    """The live reporter separates zero/missing confidence from winning probability."""
    spec = importlib.util.spec_from_file_location("live_decisions_probe", SKILL_ROOT / "scripts" / "live_test_rpt_decisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize({"answers": ANSWERS})
    assert summary["priority"]["endpoint_confidence"] == 0.0
    assert summary["priority"]["winning_probability"] == 0.54
    assert abs(summary["priority"]["weighted_score"] - 0.77) < 1e-9
    assert "endpoint_confidence" not in summary["billing"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} offline checks passed")
