"""
Unit tests for the LLM helper and the raw->canonical mapper (offline).

Fake OpenAI-chat and Gemini callables replace SAP Gen AI Hub, so no network or
credentials are used. Covers JSON parsing/retry, transient backoff, model dispatch
(gpt vs gemini), token-usage event emission, mapping cleaning, and deterministic
apply (rename, type coercion, order/row-count preservation, money separators).

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import llm as llm_mod  # noqa: E402
from app.payment_advice.llm import LLMError, complete_json  # noqa: E402
from app.payment_advice.mapper import apply_mapping, map_to_canonical  # noqa: E402


# --- fake OpenAI-chat response/callable ---
class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [type("C", (), {"message": _Msg(content)})()]


class FakeCreate:
    """Fake OpenAI chat.completions.create returning queued contents."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.contents.pop(0))


# --- fake Gemini response/callable ---
class _GemUsage:
    prompt_token_count = 5
    candidates_token_count = 3
    total_token_count = 8


class _GemResp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = _GemUsage()


class FakeGemini:
    """Fake google-native generate_content returning queued contents."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _GemResp(self.contents.pop(0))


def _last_event(buffer: io.StringIO) -> dict:
    lines = [line for line in buffer.getvalue().splitlines() if line.strip().startswith("{")]
    return json.loads(lines[-1])


class CompleteJson(unittest.TestCase):
    def test_parses_json(self) -> None:
        create = FakeCreate(['{"a": 1}'])
        self.assertEqual(complete_json("s", "u", openai_create=create), {"a": 1})
        self.assertIn("max_completion_tokens", create.calls[0])
        self.assertEqual(create.calls[0]["response_format"], {"type": "json_object"})

    def test_retries_on_bad_json(self) -> None:
        create = FakeCreate(["not json", '{"ok": true}'])
        self.assertEqual(complete_json("s", "u", openai_create=create, retries=1), {"ok": True})
        self.assertEqual(len(create.calls), 2)

    def test_raises_after_exhausting_retries(self) -> None:
        with self.assertRaises(LLMError):
            complete_json("s", "u", openai_create=FakeCreate(["nope", "still nope"]), retries=1)

    def test_transient_error_retried_with_backoff(self) -> None:
        class _RaiseThenOk:
            def __init__(self) -> None:
                self.n = 0

            def __call__(self, **kwargs):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("Error code: 429 - rate_limit_exceeded")
                return _Resp('{"ok": true}')

        create = _RaiseThenOk()
        slept: list[float] = []
        out = complete_json("s", "u", openai_create=create, sleep=slept.append)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(create.n, 2)
        self.assertEqual(len(slept), 1)

    def test_non_transient_error_raises_immediately(self) -> None:
        def _boom(**kwargs):
            raise RuntimeError("Error code: 400 - bad request")

        slept: list[float] = []
        with self.assertRaises(LLMError):
            complete_json("s", "u", openai_create=_boom, sleep=slept.append)
        self.assertEqual(slept, [])

    def test_unsupported_model_rejected(self) -> None:
        with self.assertRaises(LLMError):
            complete_json("s", "u", model="not-a-model", openai_create=FakeCreate(["{}"]))

    def test_emits_usage_event_openai(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            complete_json("s", "u", openai_create=FakeCreate(['{"a": 1}']), route_label="mapper:test")
        event = _last_event(buffer)
        self.assertEqual(event["schema_version"], "btp.llm_usage.v1")
        self.assertEqual(event["event_type"], "llm_usage")
        self.assertEqual(event["model"], "gpt-5.6-luna")
        self.assertEqual(event["llm_endpoint"], "chat.completions")
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["route"], "mapper:test")

    def test_gemini_path_and_usage_event(self) -> None:
        self.assertIn("gemini-3.1-flash-lite", llm_mod.ALLOWED_MODELS)
        gem = FakeGemini(['{"ok": true}'])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            out = complete_json("s", "u", model="gemini-3.1-flash-lite", gemini_generate=gem)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(len(gem.calls), 1)  # routed to the gemini path
        self.assertEqual(gem.calls[0]["config"]["response_mime_type"], "application/json")
        event = _last_event(buffer)
        self.assertEqual(event["model"], "gemini-3.1-flash-lite")
        self.assertEqual(event["llm_endpoint"], "generateContent")
        self.assertEqual(event["input_tokens"], 5)
        self.assertEqual(event["output_tokens"], 3)

    def test_error_emits_error_event(self) -> None:
        def _boom(**kwargs):
            raise RuntimeError("Error code: 400 - bad request")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with self.assertRaises(LLMError):
                complete_json("s", "u", openai_create=_boom)
        self.assertEqual(_last_event(buffer)["outcome"], "error")


class ApplyMapping(unittest.TestCase):
    def test_rename_coerce_and_preserve_order(self) -> None:
        aggregate = {
            "headers": {"Payer": "Globex", "Paid": "1990.17", "Junk": "x"},
            "line_items": [
                {"Inv": "A1", "Amt": "10.5", "When": "12/15/2025"},
                {"Inv": "A2", "Amt": "20"},
            ],
        }
        mapping = {
            "header": {"Payer": "payer_name", "Paid": "payment_amount"},
            "line_items": {"Inv": "invoice_reference", "Amt": "net_amount", "When": "invoice_date"},
        }
        out = apply_mapping(aggregate, mapping)
        self.assertEqual(out["header"], {"payer_name": "Globex", "payment_amount": 1990.17})
        self.assertEqual(len(out["line_items"]), 2)
        self.assertEqual(out["line_items"][0]["net_amount"], 10.5)
        self.assertEqual(out["line_items"][0]["invoice_date"], "2025-12-15")
        self.assertEqual(out["line_items"][1]["net_amount"], 20)

    def test_money_separator_coercion(self) -> None:
        cases = {
            "1,990.17": 1990.17,
            "16,589.72": 16589.72,
            "1.234,56": 1234.56,
            "1,234": 1234,
            "-30.8": -30.8,
            "5,3": 5.3,
            # Trailing-sign and accounting negatives:
            "30.8-": -30.8,
            "1.234,56-": -1234.56,
            "1,234.56-": -1234.56,
            "(30.80)": -30.8,
        }
        for raw, expected in cases.items():
            aggregate = {"headers": {"Total": raw}, "line_items": []}
            mapping = {"header": {"Total": "payment_amount"}, "line_items": {}}
            out = apply_mapping(aggregate, mapping)
            self.assertEqual(out["header"]["payment_amount"], expected, f"{raw!r} -> {expected}")

    def test_first_source_wins_on_collision(self) -> None:
        aggregate = {"headers": {"A": "first", "B": "second"}, "line_items": []}
        mapping = {"header": {"A": "payer_name", "B": "payer_name"}, "line_items": {}}
        out = apply_mapping(aggregate, mapping)
        self.assertEqual(out["header"], {"payer_name": "first"})


class MapToCanonical(unittest.TestCase):
    def test_end_to_end_with_fake(self) -> None:
        aggregate = {
            "headers": {"Payer Co": "Fabrikam", "Check Total": "16589.72"},
            "line_items": [{"Invoice Number": "8916712122", "Amount Paid($)": "-5.3"}],
        }
        mapping_json = json.dumps(
            {
                "header": {"Payer Co": "payer_name", "Check Total": "payment_amount"},
                "line_items": {"Invoice Number": "invoice_reference", "Amount Paid($)": "net_amount"},
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = map_to_canonical(aggregate, openai_create=FakeCreate([mapping_json]))
        self.assertEqual(result["header"], {"payer_name": "Fabrikam", "payment_amount": 16589.72})
        self.assertEqual(result["line_items"][0], {"invoice_reference": "8916712122", "net_amount": -5.3})
        self.assertIn("mapping", result)

    def test_invalid_targets_dropped(self) -> None:
        aggregate = {"headers": {"X": "v"}, "line_items": []}
        mapping_json = json.dumps({"header": {"X": "not_a_field"}, "line_items": {}})
        with contextlib.redirect_stdout(io.StringIO()):
            result = map_to_canonical(aggregate, openai_create=FakeCreate([mapping_json]))
        self.assertEqual(result["header"], {})


if __name__ == "__main__":
    unittest.main()
