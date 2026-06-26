"""Tests for LLM demo and LangGraph agent router usage logging."""

import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.models.llm import LLMRequest
from app.routers import agent_demo, llm_demo


class FakeRequest(SimpleNamespace):
    """Small request-like object exposing the fields used by logging helpers."""


def make_fake_request(path: str) -> FakeRequest:
    """Create a FastAPI request stand-in for direct router unit tests.

    Args:
        path: Route path to expose through the fake request URL.

    Returns:
        FakeRequest: Request-like object with URL, method, headers, and client.
    """
    return FakeRequest(
        url=SimpleNamespace(path=path),
        method="POST",
        headers={
            "x-user-id": "unit-test-user",
            "x-client-host": "unit-test-browser",
            "x-correlation-id": "corr-unit-test",
        },
        client=SimpleNamespace(host="127.0.0.1"),
    )


class LLMRouterTests(unittest.TestCase):
    """Validate router-level LLM usage logging and response usage values."""

    def test_llm_demo_uses_native_chat_completions_and_logs_usage(self) -> None:
        """Call the simple demo with a fake chat completion response."""
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello from gpt-4.1"))],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4, total_tokens=13),
        )

        with patch.object(llm_demo.chat.completions, "create", return_value=fake_response) as create:
            output = io.StringIO()
            with redirect_stdout(output):
                result = asyncio.run(
                    llm_demo.simple_chat(
                        make_fake_request("/api/llm-demo/chat"),
                        LLMRequest(message="Hello", temperature=0.1, max_tokens=64),
                    )
                )

        create.assert_called_once()
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4.1")
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(kwargs["messages"][1], {"role": "user", "content": "Hello"})

        self.assertTrue(result.success)
        self.assertEqual(result.text, "Hello from gpt-4.1")
        self.assertEqual(result.usage, {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13})

        event = json.loads(output.getvalue().strip())
        self.assertEqual(event["event_type"], "llm_usage")
        self.assertEqual(event["route"], "/api/llm-demo/chat")
        self.assertEqual(event["model"], "gpt-4.1")
        self.assertEqual(event["llm_endpoint"], "chat.completions")
        self.assertEqual(event["input_tokens"], 9)
        self.assertEqual(event["output_tokens"], 4)
        self.assertEqual(event["total_tokens"], 13)
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["correlation_id"], "corr-unit-test")

    def test_agent_demo_logs_each_langgraph_llm_call_and_returns_usage(self) -> None:
        """Call the agent demo with a fake LangChain model response."""

        class FakeLLM:
            """Fake LangChain chat model used by the compiled LangGraph app."""

            def bind_tools(self, tools):
                """Return this fake model after recording no tool-specific behavior."""
                return self

            async def ainvoke(self, messages):
                """Return one AI message with LangChain-style usage metadata."""
                return AIMessage(
                    content="No calculator needed.",
                    usage_metadata={"input_tokens": 6, "output_tokens": 5, "total_tokens": 11},
                )

        with patch.object(agent_demo, "make_llm", return_value=FakeLLM()):
            output = io.StringIO()
            with redirect_stdout(output):
                result = asyncio.run(
                    agent_demo.agent_chat(
                        make_fake_request("/api/agent-demo/chat"),
                        LLMRequest(message="Say hello"),
                    )
                )

        self.assertTrue(result.success)
        self.assertEqual(result.text, "No calculator needed.")
        self.assertEqual(result.usage, {"prompt_tokens": 6, "completion_tokens": 5, "total_tokens": 11})

        events = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "llm_usage")
        self.assertEqual(events[0]["route"], "/api/agent-demo/chat")
        self.assertEqual(events[0]["input_tokens"], 6)
        self.assertEqual(events[0]["output_tokens"], 5)
        self.assertEqual(events[0]["total_tokens"], 11)
        self.assertEqual(events[0]["outcome"], "success")


if __name__ == "__main__":
    unittest.main()
