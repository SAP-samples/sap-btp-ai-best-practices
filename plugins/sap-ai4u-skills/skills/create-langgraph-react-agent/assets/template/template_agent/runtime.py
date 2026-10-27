"""Assemble and run the minimal skill-aware LangGraph ReAct agent."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4
from weakref import WeakValueDictionary

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .config import AgentConfig, load_config
from .mcp import MCPManager
from .memory import ContextBusyError, ConversationStore, create_conversation_store
from .models import AgentResult, Attachment
from .providers import create_chat_model
from .skills import SkillLoader, catalogue_prompt


class AgentState(MessagesState, total=False):
    """Hold messages plus invocation-specific prompt and skill metadata."""

    context_id: str
    system_prompt: str
    available_skills: list[dict[str, str]]


class AgentRuntime:
    """Own the configured model, tools, graph, MCP clients, and memory store."""

    def __init__(
        self,
        config: AgentConfig,
        model: BaseChatModel,
        skill_loader: SkillLoader,
        mcp_manager: MCPManager,
        memory: ConversationStore,
        extra_tools: Sequence[BaseTool] = (),
    ) -> None:
        """Initialize and compile one reusable agent runtime."""

        self.config = config
        self.model = model
        self.skill_loader = skill_loader
        self.mcp_manager = mcp_manager
        self.memory = memory
        self.tools = [skill_loader.tool(), *extra_tools, *mcp_manager.tools]
        self.graph = self._build_graph()
        self._context_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )

    @classmethod
    async def create(
        cls,
        config_path: str | Path,
        extra_tools: Sequence[BaseTool] = (),
    ) -> "AgentRuntime":
        """Create a runtime from YAML and initialize enabled integrations.

        Args:
            config_path: Agent YAML configuration path.
            extra_tools: Optional use-case-specific LangChain tools.

        Returns:
            An initialized runtime ready for ``ainvoke``.
        """

        config = load_config(config_path)
        skill_loader = SkillLoader(
            config.skills.directory,
            config.skills.max_loaded_characters,
        )
        mcp_manager = await MCPManager.create(config.mcp)
        memory = create_conversation_store(config.memory)
        try:
            await asyncio.to_thread(memory.ensure)
            model = create_chat_model(config.model)
            return cls(config, model, skill_loader, mcp_manager, memory, extra_tools)
        except Exception:
            await mcp_manager.aclose()
            memory.close()
            raise

    async def __aenter__(self) -> "AgentRuntime":
        """Return this runtime for ``async with`` usage."""

        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close runtime-owned integration resources."""

        await self.aclose()

    async def ainvoke(
        self,
        text: str,
        context_id: str,
        attachments: Sequence[Attachment] = (),
        response_model: type[Any] | dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run one multi-turn ReAct invocation.

        Args:
            text: Current user request.
            context_id: HANA conversation key, up to 256 characters.
            attachments: Optional local image/PDF inputs.
            response_model: Optional Pydantic class or JSON Schema for a second
                provider-native structured-output pass.

        Returns:
            Final text, optional parsed output, and this invocation's graph trace.

        Raises:
            ContextBusyError: Another instance kept the context busy for longer
                than ``memory.lock_wait_seconds``.
            ConcurrentUpdateError: The lease expired mid-turn and another turn
                saved first; this turn's result was not persisted.
        """

        normalized_context = _validate_context_id(context_id)
        if not text.strip():
            raise ValueError("User text cannot be empty")

        # Two layers keep the read-run-save cycle atomic for one conversation:
        # the asyncio lock queues overlapping turns inside this process without
        # touching HANA, and the HANA lease does the same across instances.
        async with self._context_lock(normalized_context):
            owner = uuid4().hex
            await self._acquire_lease(normalized_context, owner)
            try:
                return await self._ainvoke_serialized(
                    text,
                    normalized_context,
                    owner,
                    attachments,
                    response_model,
                )
            except BaseException:
                # A successful save already released the lease; on any failure
                # (including cancellation) free it so the next turn need not wait
                # for expiry. Releasing a lease we no longer hold is a no-op, and a
                # failed release must not hide the original error: the lease then
                # simply expires after memory.lease_seconds.
                with contextlib.suppress(Exception):
                    await self._run_memory_operation(
                        self.memory.release, normalized_context, owner
                    )
                raise

    async def _acquire_lease(self, context_id: str, owner: str) -> None:
        """Poll the store until ``owner`` holds the context lease.

        Raises:
            ContextBusyError: When ``memory.lock_wait_seconds`` elapses first.
        """

        settings = self.config.memory
        loop = asyncio.get_running_loop()
        deadline = loop.time() + settings.lock_wait_seconds
        delay = 0.2
        while True:
            acquired = await self._run_memory_operation(
                self.memory.acquire, context_id, owner, settings.lease_seconds
            )
            if acquired:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ContextBusyError(
                    f"Another request for conversation {context_id!r} is still running"
                )
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, 2.0)

    async def _ainvoke_serialized(
        self,
        text: str,
        context_id: str,
        owner: str,
        attachments: Sequence[Attachment],
        response_model: type[Any] | dict[str, Any] | None,
    ) -> AgentResult:
        """Run one invocation while the caller holds the context lock and lease."""

        stored = await self._run_memory_operation(self.memory.load, context_id)
        current = _human_message(text, attachments, self.config.model.provider)
        initial: AgentState = {
            "messages": [*stored.messages, current],
            "context_id": context_id,
            "system_prompt": self.config.base_prompt,
            "available_skills": [],
        }
        final = await self.graph.ainvoke(
            initial,
            {"recursion_limit": self.config.recursion_limit},
        )
        messages = list(final["messages"])
        last = _last_final_ai_message(messages)
        output_text = _message_text(last)
        parsed: Any | None = None
        if response_model is not None:
            formatter = self.model.with_structured_output(response_model)
            parsed = await formatter.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Return the requested structured object using only the original "
                            "request and the completed agent answer. Do not add facts."
                        )
                    ),
                    HumanMessage(
                        content=f"Original request:\n{text}\n\nAgent answer:\n{output_text}"
                    ),
                ]
            )
        await self._run_memory_operation(
            self.memory.save,
            context_id,
            owner,
            stored.version,
            self._history_to_store(messages),
            None,
        )
        return AgentResult(
            output_text=output_text,
            output_parsed=parsed,
            messages=messages,
        )

    def _history_to_store(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """Return the bounded, attachment-free history persisted after a turn.

        The full graph state is kept, including tool calls and tool results.
        ``trim_messages`` with ``start_on="human"`` enforces
        ``memory.max_messages`` without leaving a tool result orphaned from the
        AI message that requested it.
        """

        cleaned = [_without_attachment_bytes(message) for message in messages]
        return trim_messages(
            cleaned,
            max_tokens=self.config.memory.max_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )

    def _context_lock(self, context_id: str) -> asyncio.Lock:
        """Return the live in-process lock for one normalized context ID."""

        lock = self._context_locks.get(context_id)
        if lock is None:
            lock = asyncio.Lock()
            self._context_locks[context_id] = lock
        return lock

    async def _run_memory_operation(
        self,
        operation: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run blocking storage to completion even if its caller is cancelled.

        Args:
            operation: Synchronous conversation-store method to execute.
            *args: Positional arguments passed to the storage method.

        Returns:
            The storage method's return value.

        Raises:
            asyncio.CancelledError: After an already-started storage call finishes.
            RuntimeError: If storage fails while cancellation is being propagated.
        """

        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # `to_thread` cannot stop its worker. Keep awaiting it so the caller's
            # context lock remains held and an older mutation cannot race a newer one.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                task.result()
            except Exception as exc:
                raise RuntimeError(
                    "Conversation storage failed while cancellation was pending"
                ) from exc
            raise

    async def clear_context(self, context_id: str) -> None:
        """Delete one context's persisted HANA conversation.

        Only the in-process lock is taken: clearing is an administrative action,
        and a turn still running elsewhere fails safely at save time because
        its row no longer matches.
        """

        normalized_context = _validate_context_id(context_id)
        async with self._context_lock(normalized_context):
            await self._run_memory_operation(self.memory.clear, normalized_context)

    async def aclose(self) -> None:
        """Close MCP clients and persistence resources."""

        await self.mcp_manager.aclose()
        await asyncio.to_thread(self.memory.close)

    def _build_graph(self) -> Any:
        """Compile START -> load_skill_node -> agent -> tools loop -> END."""

        model_with_tools = self.model.bind_tools(self.tools)

        def load_skill_node(state: AgentState) -> dict[str, Any]:
            """Rescan skill metadata and construct this invocation's system prompt."""

            summaries = self.skill_loader.scan()
            return {
                "system_prompt": catalogue_prompt(self.config.base_prompt, summaries),
                "available_skills": [
                    {"name": item.name, "description": item.description}
                    for item in summaries
                ],
            }

        async def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
            """Invoke the bound model with the current dynamic system prompt."""

            response = await model_with_tools.ainvoke(
                [SystemMessage(content=state["system_prompt"]), *state["messages"]]
            )
            return {"messages": [response]}

        builder = StateGraph(AgentState)
        builder.add_node("load_skill_node", load_skill_node)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        builder.add_edge(START, "load_skill_node")
        builder.add_edge("load_skill_node", "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        return builder.compile()


def _human_message(
    text: str,
    attachments: Sequence[Attachment],
    provider: str,
) -> HumanMessage:
    """Build a plain or multimodal HumanMessage without persisting file bytes."""

    if not attachments:
        return HumanMessage(content=text)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(attachment.to_content_block(provider) for attachment in attachments)
    return HumanMessage(content=content)


def _without_attachment_bytes(message: BaseMessage) -> BaseMessage:
    """Replace image/file blocks in a user message with a short text placeholder.

    Attachment bytes are sent to the model for the current turn only; storing
    them would bloat every later turn and persist user files in HANA.
    """

    if not isinstance(message, HumanMessage) or isinstance(message.content, str):
        return message
    blocks: list[Any] = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") in {"image", "file"}:
            name = block.get("filename") or "image"
            blocks.append(
                {
                    "type": "text",
                    "text": f"[attachment omitted from history: {name}, {block.get('mime_type')}]",
                }
            )
        else:
            blocks.append(block)
    return message.model_copy(update={"content": blocks})


def _last_final_ai_message(messages: Sequence[BaseMessage]) -> AIMessage:
    """Return the most recent AI message that is not an intermediate tool call."""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message
    raise RuntimeError("Agent finished without a final AI message")


def _message_text(message: BaseMessage) -> str:
    """Normalize string or standard-block message content into plain text."""

    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def _validate_context_id(context_id: str) -> str:
    """Return a non-empty HANA-safe context value of at most 256 characters."""

    normalized = context_id.strip()
    if not normalized or len(normalized) > 256:
        raise ValueError("context_id must contain 1 to 256 characters")
    return normalized
