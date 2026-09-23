"""Verify the graph topology, ReAct loop, structured pass, and concise memory."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from template_agent.config import AgentConfig
from template_agent.mcp import MCPManager
from template_agent.memory import Conversation
from template_agent.runtime import AgentRuntime
from template_agent.skills import SkillLoader


class StructuredAnswer(BaseModel):
    """Represent the schema requested in the test structured pass."""

    answer: str


class RecordingModel(BaseChatModel):
    """Request two skills, observe their ToolMessage, then answer."""

    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        """Return the fake model identifier required by BaseChatModel."""

        return "recording-test-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RecordingModel":
        """Return this fake while retaining the normal bind-tools interface."""

        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        """Return a deterministic provider-native formatting stand-in."""

        if isinstance(schema, dict):
            return RunnableLambda(lambda messages: {"answer": "structured-json"})
        return RunnableLambda(lambda messages: StructuredAnswer(answer="structured"))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Emit a tool call first and a final answer after tool execution."""

        self.calls.append(messages)
        if isinstance(messages[-1], ToolMessage):
            assert "SKILL: alpha" in messages[-1].content
            assert "SKILL: beta" in messages[-1].content
            answer = AIMessage(content="skills loaded")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_names": ["alpha", "beta"]},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


class RecordingStore:
    """Provide an in-memory ConversationStore test double."""

    max_messages = 40

    def __init__(self) -> None:
        """Initialize empty storage."""

        self.values: dict[str, Conversation] = {}

    def ensure(self) -> None:
        """Perform no setup."""

    def load(self, context_id: str) -> Conversation:
        """Return a defensive copy of stored turns."""

        return list(self.values.get(context_id, []))

    def save(self, context_id: str, messages: Conversation) -> None:
        """Record a defensive copy of stored turns."""

        self.values[context_id] = list(messages)

    def clear(self, context_id: str) -> None:
        """Delete one context."""

        self.values.pop(context_id, None)

    def close(self) -> None:
        """Close nothing."""


class ConcurrentModel(BaseChatModel):
    """Expose whether model calls for one context overlap in time."""

    active_calls: int = 0
    max_active_calls: int = 0

    @property
    def _llm_type(self) -> str:
        """Return the fake model identifier required by BaseChatModel."""

        return "concurrent-test-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ConcurrentModel":
        """Return this fake while retaining the normal bind-tools interface."""

        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Provide the required synchronous interface for the async-only test fake."""

        raise AssertionError("ConcurrentModel must be invoked asynchronously")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Delay a deterministic answer while recording concurrent entry."""

        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            await asyncio.sleep(0.02)
            request = next(
                message.content
                for message in reversed(messages)
                if message.type == "human"
            )
            answer = AIMessage(content=f"answer:{request}")
            return ChatResult(generations=[ChatGeneration(message=answer)])
        finally:
            self.active_calls -= 1


class BlockingSaveStore(RecordingStore):
    """Hold the first turn's save so cancellation ordering can be tested."""

    def __init__(self, failure: Exception | None = None) -> None:
        """Create thread-safe signals and an optional delayed storage failure."""

        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()
        self.failure = failure

    def save(self, context_id: str, messages: Conversation) -> None:
        """Block only the first turn's write until the test releases it."""

        if messages[-2]["content"] == "first":
            self.save_started.set()
            self.release_save.wait(timeout=5)
            if self.failure:
                raise self.failure
        super().save(context_id, messages)


def _skill(root: Path, name: str) -> None:
    """Create a minimal valid skill fixture."""

    directory = root / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n\n# {name}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_runtime_executes_batch_skill_react_loop_and_structured_pass(
    tmp_path: Path,
) -> None:
    """Exercise the complete graph while persisting only user/final turns."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    _skill(skill_root, "alpha")
    _skill(skill_root, "beta")
    config = AgentConfig.model_validate(
        {
            "base_prompt": "Base prompt",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skill_root)},
            "memory": {"enabled": False},
        }
    )
    model = RecordingModel()
    store = RecordingStore()
    runtime = AgentRuntime(
        config,
        model,
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    result = await runtime.ainvoke(
        "load what you need",
        "ctx",
        response_model=StructuredAnswer,
    )

    assert result.output_text == "skills loaded"
    assert result.output_parsed == StructuredAnswer(answer="structured")
    assert any(isinstance(message, ToolMessage) for message in result.messages)
    assert store.values["ctx"] == [
        {"role": "user", "content": "load what you need"},
        {"role": "assistant", "content": "skills loaded"},
    ]
    first_system = model.calls[0][0].content
    assert "alpha: alpha description" in first_system
    assert "beta: beta description" in first_system
    graph = runtime.graph.get_graph()
    assert {"load_skill_node", "agent", "tools"}.issubset(graph.nodes)

    json_result = await runtime.ainvoke(
        "load for JSON output",
        "ctx-json",
        response_model={
            "title": "Answer",
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    assert json_result.output_parsed == {"answer": "structured-json"}


@pytest.mark.asyncio
async def test_runtime_serializes_concurrent_invocations_for_one_context(
    tmp_path: Path,
) -> None:
    """Concurrent turns for one context retain both ordered memory updates."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    config = AgentConfig.model_validate(
        {
            "base_prompt": "Base prompt",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skill_root)},
            "memory": {"enabled": False},
        }
    )
    model = ConcurrentModel()
    store = RecordingStore()
    runtime = AgentRuntime(
        config,
        model,
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    first, second = await asyncio.gather(
        runtime.ainvoke("first", "shared-context"),
        runtime.ainvoke("second", "shared-context"),
    )

    assert {first.output_text, second.output_text} == {"answer:first", "answer:second"}
    assert model.max_active_calls == 1
    assert store.values["shared-context"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer:first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer:second"},
    ]


@pytest.mark.asyncio
async def test_runtime_holds_context_lock_until_cancelled_save_finishes(
    tmp_path: Path,
) -> None:
    """A cancelled write cannot later overwrite a newer same-context turn."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    config = AgentConfig.model_validate(
        {
            "base_prompt": "Base prompt",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skill_root)},
            "memory": {"enabled": False},
        }
    )
    store = BlockingSaveStore()
    runtime = AgentRuntime(
        config,
        ConcurrentModel(),
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    first_task = asyncio.create_task(runtime.ainvoke("first", "shared-context"))
    assert await asyncio.to_thread(store.save_started.wait, 2)
    first_task.cancel()
    second_task = asyncio.create_task(runtime.ainvoke("second", "shared-context"))
    try:
        await asyncio.sleep(0.05)
        assert not first_task.done()
        assert not second_task.done()
    finally:
        store.release_save.set()

    first_result, second_result = await asyncio.gather(
        first_task,
        second_task,
        return_exceptions=True,
    )
    assert isinstance(first_result, asyncio.CancelledError)
    assert second_result.output_text == "answer:second"
    assert store.values["shared-context"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer:first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer:second"},
    ]


@pytest.mark.asyncio
async def test_runtime_wraps_storage_failure_while_cancellation_is_pending(
    tmp_path: Path,
) -> None:
    """A delayed storage error uses the documented generic runtime failure."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    config = AgentConfig.model_validate(
        {
            "base_prompt": "Base prompt",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skill_root)},
            "memory": {"enabled": False},
        }
    )
    store = BlockingSaveStore(failure=ValueError("storage-failed"))
    runtime = AgentRuntime(
        config,
        ConcurrentModel(),
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    invocation = asyncio.create_task(runtime.ainvoke("first", "shared-context"))
    assert await asyncio.to_thread(store.save_started.wait, 2)
    invocation.cancel()
    await asyncio.sleep(0.01)
    store.release_save.set()

    with pytest.raises(
        RuntimeError,
        match="Conversation storage failed while cancellation was pending",
    ) as caught:
        await invocation

    assert isinstance(caught.value.__cause__, ValueError)
