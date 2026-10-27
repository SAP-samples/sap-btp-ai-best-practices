"""Verify the graph topology, ReAct loop, structured pass, memory, and leasing."""

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
from template_agent.models import Attachment
from template_agent.memory import (
    ConcurrentUpdateError,
    ContextBusyError,
    StoredConversation,
)
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
    """Provide an in-memory ConversationStore double with HANA lease semantics.

    One instance shared by two runtimes behaves like one HANA table shared by
    two Cloud Foundry instances: leases and versions are enforced across them.
    """

    def __init__(self) -> None:
        """Initialize empty storage, versions, pending actions, and leases."""

        self.values: dict[str, list[BaseMessage]] = {}
        self.versions: dict[str, int] = {}
        self.pending: dict[str, dict[str, Any] | None] = {}
        self.leases: dict[str, str] = {}
        self._guard = threading.Lock()

    def ensure(self) -> None:
        """Perform no setup."""

    def acquire(self, context_id: str, owner: str, lease_seconds: int) -> bool:
        """Grant the lease when nobody holds it."""

        with self._guard:
            if context_id in self.leases:
                return False
            self.leases[context_id] = owner
            return True

    def load(self, context_id: str) -> StoredConversation:
        """Return a defensive copy of stored state."""

        return StoredConversation(
            messages=list(self.values.get(context_id, [])),
            version=self.versions.get(context_id, 0),
            pending_action=self.pending.get(context_id),
        )

    def save(
        self,
        context_id: str,
        owner: str,
        expected_version: int,
        messages: list[BaseMessage],
        pending_action: dict[str, Any] | None,
    ) -> None:
        """Store state only for the lease holder at the expected version."""

        with self._guard:
            if (
                self.leases.get(context_id) != owner
                or self.versions.get(context_id, 0) != expected_version
            ):
                raise ConcurrentUpdateError("stale turn")
            self.values[context_id] = list(messages)
            self.pending[context_id] = pending_action
            self.versions[context_id] = expected_version + 1
            del self.leases[context_id]

    def release(self, context_id: str, owner: str) -> None:
        """Drop the lease only if ``owner`` holds it."""

        with self._guard:
            if self.leases.get(context_id) == owner:
                del self.leases[context_id]

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

    def save(
        self,
        context_id: str,
        owner: str,
        expected_version: int,
        messages: list[BaseMessage],
        pending_action: dict[str, Any] | None,
    ) -> None:
        """Block only the first turn's write until the test releases it."""

        if messages[-2].content == "first":
            self.save_started.set()
            self.release_save.wait(timeout=5)
            if self.failure:
                raise self.failure
        super().save(context_id, owner, expected_version, messages, pending_action)


def _turns(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    """Return (type, text) for user and final assistant messages only."""

    return [
        (message.type, message.content)
        for message in messages
        if message.type == "human"
        or (message.type == "ai" and not getattr(message, "tool_calls", None))
    ]


def _config(skill_root: Path, **overrides: Any) -> AgentConfig:
    """Build a minimal valid configuration with optional section overrides."""

    return AgentConfig.model_validate(
        {
            "base_prompt": "Base prompt",
            "model": {"provider": "openai", "name": "fake"},
            "skills": {"directory": str(skill_root)},
            "memory": {"enabled": False},
            **overrides,
        }
    )


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
    """Exercise the complete graph and persist the full tool-call history."""

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
    stored = store.values["ctx"]
    assert [message.type for message in stored] == ["human", "ai", "tool", "ai"]
    assert stored[1].tool_calls[0]["name"] == "load_skill"
    assert stored[2].tool_call_id == "call-1"
    assert store.leases == {}
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
    assert _turns(store.values["shared-context"]) == [
        ("human", "first"),
        ("ai", "answer:first"),
        ("human", "second"),
        ("ai", "answer:second"),
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
    assert _turns(store.values["shared-context"]) == [
        ("human", "first"),
        ("ai", "answer:first"),
        ("human", "second"),
        ("ai", "answer:second"),
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


@pytest.mark.asyncio
async def test_two_runtimes_sharing_a_store_serialize_through_the_lease(
    tmp_path: Path,
) -> None:
    """Two instances (separate asyncio locks) still never run one context at once."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    config = _config(skill_root, memory={"enabled": False, "lock_wait_seconds": 5})
    store = RecordingStore()
    model = ConcurrentModel()
    instances = [
        AgentRuntime(config, model, SkillLoader(skill_root), MCPManager(), store)
        for _ in range(2)
    ]

    await asyncio.gather(
        instances[0].ainvoke("first", "shared-context"),
        instances[1].ainvoke("second", "shared-context"),
    )

    assert model.max_active_calls == 1
    assert [text for kind, text in _turns(store.values["shared-context"]) if kind == "human"] in (
        ["first", "second"],
        ["second", "first"],
    )
    assert store.versions["shared-context"] == 2
    assert store.leases == {}


@pytest.mark.asyncio
async def test_busy_context_fails_after_lock_wait(tmp_path: Path) -> None:
    """A lease held elsewhere raises ContextBusyError once the wait elapses."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    store = RecordingStore()
    store.leases["shared-context"] = "other-instance"
    runtime = AgentRuntime(
        _config(skill_root, memory={"enabled": False, "lock_wait_seconds": 0.3}),
        ConcurrentModel(),
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    with pytest.raises(ContextBusyError):
        await runtime.ainvoke("hello", "shared-context")
    assert store.leases == {"shared-context": "other-instance"}


@pytest.mark.asyncio
async def test_stale_turn_is_rejected_and_lease_released(tmp_path: Path) -> None:
    """If another turn saved first, the result is not written and the lease is freed."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()

    class ExpiringStore(RecordingStore):
        """Simulate another instance saving after this turn's lease expired."""

        def load(self, context_id: str) -> StoredConversation:
            """Return state, then advance the version as a concurrent writer would."""

            stored = super().load(context_id)
            self.versions[context_id] = stored.version + 1
            return stored

    store = ExpiringStore()
    runtime = AgentRuntime(
        _config(skill_root),
        ConcurrentModel(),
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    with pytest.raises(ConcurrentUpdateError):
        await runtime.ainvoke("hello", "ctx")
    assert "ctx" not in store.values
    assert store.leases == {}


@pytest.mark.asyncio
async def test_history_drops_attachment_bytes_and_respects_cap(tmp_path: Path) -> None:
    """Stored user messages keep text only, and the cap restarts on a user turn."""

    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    image = tmp_path / "pixel.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    store = RecordingStore()
    runtime = AgentRuntime(
        _config(skill_root, memory={"enabled": False, "max_messages": 3}),
        ConcurrentModel(),
        SkillLoader(skill_root),
        MCPManager(),
        store,
    )

    await runtime.ainvoke("one", "ctx")
    result = await runtime.ainvoke("two", "ctx", attachments=[Attachment(path=image)])

    assert any(
        isinstance(block, dict) and block.get("type") == "image"
        for block in result.messages[-2].content
    )
    stored = store.values["ctx"]
    assert len(stored) == 2
    assert stored[0].type == "human"
    blocks = stored[0].content
    assert all(block.get("type") == "text" for block in blocks)
    assert "attachment omitted from history: image" in blocks[1]["text"]
