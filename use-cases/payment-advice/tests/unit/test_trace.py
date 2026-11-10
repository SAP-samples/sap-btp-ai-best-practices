"""
Unit tests for extract_tool_calls in api/app/deduction_agent/trace.py.

Run with:
    PYTHONPATH=api .venv/bin/python -m unittest tests/unit/test_trace.py -v
"""

import unittest
from types import SimpleNamespace

from app.deduction_agent.trace import extract_tool_calls, _SUMMARY_MAX


class TestExtractToolCalls(unittest.TestCase):
    """Tests for extract_tool_calls() covering object-style and dict-style messages."""

    def test_extracts_calls_and_result_summaries(self):
        """A single AI tool call matched to its ToolMessage by tool_call_id (object-style)."""
        ai = SimpleNamespace(
            tool_calls=[{"name": "get_deduction_rules", "args": {"client_key": "c"}, "id": "call_1"}],
            content="",
        )
        tool = SimpleNamespace(content="null", tool_call_id="call_1")
        out = extract_tool_calls([ai, tool])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool"], "get_deduction_rules")
        self.assertEqual(out[0]["args"], {"client_key": "c"})
        self.assertEqual(out[0]["result_summary"], "null")

    def test_no_tool_calls_returns_empty(self):
        """An AI message with an empty tool_calls list returns []."""
        ai = SimpleNamespace(tool_calls=[], content="done")
        self.assertEqual(extract_tool_calls([ai]), [])

    def test_dict_style_messages(self):
        """Dict-style messages (plain dicts) are handled by the dict branch of _get_attr."""
        ai = {"tool_calls": [{"name": "foo", "args": {}, "id": "x"}], "content": ""}
        tool = {"content": "bar", "tool_call_id": "x"}
        out = extract_tool_calls([ai, tool])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool"], "foo")
        self.assertEqual(out[0]["result_summary"], "bar")

    def test_result_summary_truncated_to_cap(self):
        """Tool result content longer than the cap is truncated to exactly _SUMMARY_MAX chars."""
        ai = SimpleNamespace(
            tool_calls=[{"name": "big_tool", "args": {}, "id": "c1"}],
            content="",
        )
        tool = SimpleNamespace(content="x" * (_SUMMARY_MAX + 100), tool_call_id="c1")
        out = extract_tool_calls([ai, tool])
        self.assertEqual(len(out[0]["result_summary"]), _SUMMARY_MAX)
