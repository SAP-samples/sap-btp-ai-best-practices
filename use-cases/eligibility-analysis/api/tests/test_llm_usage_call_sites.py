"""Tests for LLM usage logging at application call sites."""

from __future__ import annotations

import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage

from app.observability.llm_usage_logging import LlmUsageContext


class _FakeSyncLlm:
    """Synchronous fake LLM returning LangChain-style usage metadata."""

    def invoke(self, _messages):
        """Return a fake model response with usage metadata."""
        return SimpleNamespace(
            content="Section A: Summary\nSection B: Recommendations",
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 9,
                "total_tokens": 39,
                "input_token_details": {"cache_read": 4},
            },
        )


class _FakeAsyncLlm:
    """Asynchronous fake LLM returning LangChain-style usage metadata."""

    async def ainvoke(self, _messages):
        """Return a fake model response with usage metadata."""
        return SimpleNamespace(
            content="Agent answer",
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 12,
                "total_tokens": 62,
                "input_token_details": {"cache_read": 10},
            },
        )


class _FailingAsyncLlm:
    """Asynchronous fake LLM that raises after a provider attempt."""

    async def ainvoke(self, _messages):
        """Raise a provider failure."""
        raise RuntimeError("provider unavailable")


def _read_event(stdout: str) -> dict:
    """Return the first stdout JSON event."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    return json.loads(lines[0])


class TestA2aLlmUsageLogging(unittest.TestCase):
    """Verify the A2A assistant call emits usage events."""

    def test_a2a_assistant_success_logs_usage_event(self) -> None:
        """The async assistant LLM call logs tokens from the model response."""
        from app.a2a.agent import _invoke_assistant_llm

        context = LlmUsageContext(
            route="/api/a2a",
            method="POST",
            user_id="agent-user@example.com",
            client_host="pytest",
            actor_type="human",
            correlation_id="agent-corr",
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            response = asyncio.run(
                _invoke_assistant_llm(
                    _FakeAsyncLlm(),
                    SystemMessage(content="system"),
                    {"messages": [HumanMessage(content="hello")]},
                    model_name="gpt-4.1",
                    usage_context=context,
                )
            )

        self.assertEqual(response.content, "Agent answer")
        event = _read_event(stream.getvalue())
        self.assertEqual(event["route"], "/api/a2a")
        self.assertEqual(event["model"], "gpt-4.1")
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["input_tokens"], 50)
        self.assertEqual(event["cached_input_tokens"], 10)
        self.assertEqual(event["output_tokens"], 12)
        self.assertEqual(event["correlation_id"], "agent-corr")

    def test_a2a_assistant_error_logs_usage_event(self) -> None:
        """The async assistant LLM call logs an error event before reraising."""
        from app.a2a.agent import _invoke_assistant_llm

        context = LlmUsageContext(route="/api/a2a", method="POST", actor_type="human")

        stream = io.StringIO()
        with self.assertRaises(RuntimeError), redirect_stdout(stream):
            asyncio.run(
                _invoke_assistant_llm(
                    _FailingAsyncLlm(),
                    SystemMessage(content="system"),
                    {"messages": [HumanMessage(content="hello")]},
                    model_name="gpt-4.1",
                    usage_context=context,
                )
            )

        event = _read_event(stream.getvalue())
        self.assertEqual(event["route"], "/api/a2a")
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["input_tokens"], 0)
        self.assertEqual(event["cached_input_tokens"], 0)
        self.assertEqual(event["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
