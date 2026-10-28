"""Offline checks for the RPT decisions asset (no network calls).

Covers request validation, table translation (described and plain encodings,
structured and chat states), and answer parsing. The shared client module comes
from the sibling ``rpt-1`` skill.

Run from the skill folder or anywhere else:
    python tests/test_rpt_decisions.py
    python -I tests/test_rpt_decisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "assets"))
sys.path.insert(0, str(SKILL_ROOT.parent / "rpt-1" / "assets"))  # shared rpt_client_api

from rpt_decisions import (  # noqa: E402
    PLACEHOLDER,
    build_decision_payload,
    parse_decision_answers,
    validate_decision_request,
)

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


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} offline checks passed")
