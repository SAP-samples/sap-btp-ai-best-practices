"""Regressions for waiting without spending graph steps or replaying a launch."""

import asyncio
from pathlib import Path

from app.agent.config import load_config
from app.agent.tools.workspace_tools import workspace_tools


def test_default_model_is_gpt_54_with_high_effort():
    """The packaged fallback follows the requested GPT-5.4 default."""
    config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
    assert (config.model.provider, config.model.name, config.model.reasoning_effort) == (
        "openai", "gpt-5.4", "high")


def test_wait_tool_returns_only_after_completion_and_emits_progress():
    """One awaited tool spans several lifecycle reads without a model invocation."""

    class Service:
        """Simulate a job advancing under an independent worker."""

        def __init__(self):
            """Start with a queued job and zero reads."""
            self.reads = 0

        def get_run(self, run_id):
            """Advance the fake persisted status on each read."""
            self.reads += 1
            return {
                "run_id": run_id,
                "status": "completed" if self.reads == 3 else "running",
            }

    async def exercise():
        """Wait once for a three-read job and retain emitted public events."""
        service, events = Service(), []
        tools = {t.name: t for t in workspace_tools(service, "session", events.append)}
        result = await tools["wait_for_run"].ainvoke({"run_id": "r"})
        assert result["status"] == "completed"
        assert service.reads == 3
        assert events[0]["type"] == "run_progress"

    asyncio.run(exercise())


def test_reset_cancels_active_chat_and_closes_runtime():
    """Reset interrupts an awaiting model/tool turn without calling any job cancellation."""
    from app.routers.workspace_chat import ChatMessage, chat_message, reset_chat

    async def exercise():
        """Start one blocked chat, reset it and verify owned resources are released."""
        started, closed = asyncio.Event(), asyncio.Event()

        class Service:
            """Only conversation selection is accessible; no optimizer mutations exist."""

            def context(self, context_id, patch=None):
                """Return an empty selection for the chat."""
                return {}

        class Runtime:
            """Keep a turn waiting until reset cancels it."""

            async def ainvoke(self, *args, **kwargs):
                """Signal readiness then block cooperatively."""
                started.set()
                await asyncio.Event().wait()

            async def aclose(self):
                """Record guaranteed runtime cleanup."""
                closed.set()

        async def factory(*args):
            """Construct the no-provider runtime."""
            return Runtime()

        response = await chat_message(
            ChatMessage(message="wait", context_id="reset-test"), Service(), factory
        )
        await started.wait()
        assert await reset_chat("reset-test") == {"reset": True}
        assert closed.is_set()
        assert [item async for item in response.body_iterator] == []

    asyncio.run(exercise())


def test_rate_limit_and_recursion_errors_explain_job_independence():
    """Chat recovery errors identify rate limits without implying optimizer failure."""
    from app.routers.workspace_chat import chat_error_message

    class RateLimited(Exception):
        """Expose the public provider status without response secrets."""

        status_code = 429

    assert "rate limit" in chat_error_message(RateLimited()).lower()
    assert "continue" in chat_error_message(RateLimited()).lower()
    assert "New conversation" in chat_error_message(
        type("GraphRecursionError", (Exception,), {})()
    )


def test_graph_waits_through_sixty_job_reads_without_recursion_failure(monkeypatch):
    """A full graph turn can wait longer than its 50-step budget using two model calls."""
    import importlib
    from types import SimpleNamespace

    from app.agent.memory import NullConversationStore
    from app.agent.runtime import AgentRuntime
    from app.agent.skills import SkillLoader
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    module = importlib.import_module("app.agent.tools.workspace_tools")
    monkeypatch.setattr(module, "WAIT_POLL_SECONDS", 0.001)

    class Model(FakeMessagesListChatModel):
        """Return a wait tool call followed by an ordinary final answer."""

        def bind_tools(self, tools, **kwargs):
            """Bind no external provider; preserve the predetermined response sequence."""
            return self

    class Service:
        """Advance one independent job through sixty persisted status reads."""

        reads = 0

        def get_run(self, run_id):
            """Return terminal status only after more reads than the graph step limit."""
            self.reads += 1
            return {
                "run_id": run_id,
                "status": "completed" if self.reads == 60 else "running",
            }

    async def exercise():
        """Execute the real graph with the production waiting tool and a fake LLM."""
        config = load_config(Path(__file__).parents[1] / "app/agent/config/agent.yaml")
        service, events = Service(), []
        model = Model(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "wait-1",
                            "name": "wait_for_run",
                            "args": {"run_id": "r"},
                        }
                    ],
                ),
                AIMessage(content="Completed"),
            ]
        )
        runtime = AgentRuntime(
            config,
            model,
            SkillLoader(config.skills.directory),
            SimpleNamespace(tools=[]),
            NullConversationStore(config.memory.max_messages),
            workspace_tools(service, "graph-test"),
        )
        result = await runtime.ainvoke(
            "Wait for run r", "graph-test", session_history=[], on_event=events.append
        )
        assert result.output_text == "Completed"
        assert service.reads == 60
        assert sum(event["type"] == "tool_call" for event in events) == 1

    asyncio.run(exercise())


def test_wait_returns_actionable_queue_delay_instead_of_hanging(monkeypatch):
    """An unclaimed job releases chat with its existing ID, without another submission."""
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module("app.agent.tools.workspace_tools")
    monkeypatch.setattr(module, "WAIT_QUEUE_TIMEOUT_SECONDS", 0, raising=False)
    service = SimpleNamespace(get_run=lambda run_id: {"run_id":run_id,"status":"queued","worker_id":None})
    tool = next(t for t in workspace_tools(service,"queue-test") if t.name=="wait_for_run")
    result = asyncio.run(asyncio.wait_for(tool.ainvoke({"run_id":"existing"}), timeout=1))
    assert result["wait_status"] == "queued_not_started"
    assert result["run_id"] == "existing"
    assert "worker" in result["message"]
