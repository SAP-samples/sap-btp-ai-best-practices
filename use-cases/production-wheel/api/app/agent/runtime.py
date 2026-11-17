"""Assemble and run the minimal skill-aware LangGraph ReAct agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from .config import AgentConfig, load_config
from .context_budget import (
    ContextBudgetExceeded,
    ContextBudgetMiddleware,
    _summary_message,
    emit_context_telemetry,
)
from .mcp import MCPManager
from .memory import ConversationStore, create_conversation_store
from .models import AgentResult, Attachment
from .providers import create_chat_model
from .skills import SkillLoader, catalogue_prompt


class AgentState(MessagesState, total=False):
    """Hold messages plus invocation-specific prompt and skill metadata."""

    context_id: str
    system_prompt: str
    available_skills: list[dict[str, str]]
    context_summary: str | None
    history_compacted: bool


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
        self.context_budget = ContextBudgetMiddleware(
            config.context_budget,
            self.tools,
            summarizer_factory=lambda: create_chat_model(
                config.model.model_copy(
                    update={
                        "max_tokens": config.context_budget.summary_max_tokens,
                        "reasoning_effort": (
                            "none" if config.model.provider == "claude" else None
                        ),
                    }
                )
            ),
        )
        self.graph = self._build_graph()

    @classmethod
    async def create(
        cls,
        config_path: str | Path,
        extra_tools: Sequence[BaseTool] = (),
        model_name: str | None = None,
    ) -> "AgentRuntime":
        """Create a runtime from YAML and initialize enabled integrations.

        Args:
            config_path: Agent YAML configuration path.
            extra_tools: Optional use-case-specific LangChain tools.
            model_name: Optional saved AI Core deployment choice for this runtime.

        Returns:
            An initialized runtime ready for ``ainvoke``.
        """

        config = load_config(config_path)
        if model_name is not None:
            from app.workspace.ai_model_settings import model_configuration

            config = config.model_copy(update={"model": model_configuration(model_name)})
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
        on_event: Callable[[dict[str, Any]], None] | None = None,
        session_history: Sequence[dict[str, str]] | None = None,
        context_summary: str | None = None,
    ) -> AgentResult:
        """Run one multi-turn ReAct invocation.

        Args:
            text: Current user request.
            context_id: Invocation/session key, up to 256 characters.
            attachments: Optional local image/PDF inputs.
            response_model: Optional Pydantic class or JSON Schema for a second
                provider-native structured-output pass.
            on_event: Optional callback invoked with a small dict per graph step
                (skills loaded, each tool call, each tool result, final answer) so a
                caller can surface live progress. Streaming happens regardless; this
                only controls whether step events are emitted.
            session_history: Optional browser-session turns. When supplied, these
                replace persistent loading and are never saved by the runtime.
            context_summary: Optional browser-only summary of older page turns.
                It is never sent to HANA or durable conversation memory.

        Returns:
            Final text, optional parsed output, and this invocation's graph trace.
        """

        normalized_context = _validate_context_id(context_id)
        if not text.strip():
            raise ValueError("User text cannot be empty")
        stored = (
            [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in session_history
                if item.get("role") in {"user", "assistant"}
            ][-self.memory.max_messages :]
            if session_history is not None
            else await asyncio.to_thread(self.memory.load, normalized_context)
        )
        browser_summary = _validated_context_summary(
            context_summary, self.config.context_budget.max_summary_characters
        )
        history = [
            *([_summary_message(browser_summary)] if browser_summary else []),
            *[_stored_message(item) for item in stored],
        ]
        current = _human_message(text, attachments, self.config.model.provider)
        initial: AgentState = {
            "messages": [*history, current],
            "context_id": normalized_context,
            "system_prompt": self.config.base_prompt,
            "available_skills": [],
            "context_summary": browser_summary,
            "history_compacted": False,
        }
        config = {"recursion_limit": self.config.recursion_limit}
        final_state: dict[str, Any] | None = None
        # Stream super-step updates so a caller can surface each tool call and result
        # live; the parallel "values" stream carries the authoritative final state
        # (identical to what ainvoke would have returned).
        async for stream_mode, chunk in self.graph.astream(
            initial, config, stream_mode=["updates", "values"]
        ):
            if stream_mode == "values":
                final_state = chunk
            elif stream_mode == "updates" and on_event is not None:
                for node, delta in chunk.items():
                    if isinstance(delta, dict):
                        _emit_step_events(on_event, node, delta)
        messages = list(final_state["messages"]) if final_state else list(initial["messages"])
        last = _last_final_ai_message(messages)
        output_text = _message_text(last)
        parsed: Any | None = None
        if response_model is not None:
            formatter = self.model.with_structured_output(
                response_model,
                **_structured_output_kwargs(self.config.model.provider),
            )
            structured_system = (
                (final_state or initial)["system_prompt"]
                + "\n\nReturn the requested structured object using only the original "
                "request and the completed agent answer. Do not add facts."
            )
            try:
                prepared = await self.context_budget.prepare(
                    structured_system,
                    [HumanMessage(content=f"Original request:\n{text}\n\nAgent answer:\n{output_text}")],
                    normalized_context,
                )
                parsed = await formatter.ainvoke(
                    [SystemMessage(content=structured_system), *prepared.messages]
                )
                emit_context_telemetry(
                    normalized_context, prepared, getattr(parsed, "usage_metadata", None)
                )
            except ContextBudgetExceeded:
                parsed = None
        updated = [
            *stored,
            {"role": "user", "content": text},
            {"role": "assistant", "content": output_text},
        ][-self.memory.max_messages :]
        if session_history is None:
            await asyncio.to_thread(self.memory.save, normalized_context, updated)
        return AgentResult(
            output_text=output_text,
            output_parsed=parsed,
            messages=messages,
            context_summary=(final_state or initial).get("context_summary"),
            history_compacted=bool((final_state or initial).get("history_compacted")),
        )

    async def clear_context(self, context_id: str) -> None:
        """Delete one context's persisted HANA conversation."""

        await asyncio.to_thread(self.memory.clear, _validate_context_id(context_id))

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
            prompt = self.config.base_prompt
            if self.config.skills.required:
                prompt += "\n\nRequired domain guidance:\n" + self.skill_loader.load(self.config.skills.required)
            return {
                "system_prompt": catalogue_prompt(prompt, summaries),
                "available_skills": [
                    {"name": item.name, "description": item.description}
                    for item in summaries
                ],
            }

        async def agent_node(state: AgentState) -> dict[str, Any]:
            """Invoke the bound model with the current dynamic system prompt."""

            try:
                prepared = await self.context_budget.prepare(
                    state["system_prompt"], state["messages"], state["context_id"]
                )
            except ContextBudgetExceeded:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I could not safely send this chat context to the model. "
                                "Start a new conversation or narrow the request, then retry."
                            )
                        )
                    ]
                }
            response = await model_with_tools.ainvoke(
                [SystemMessage(content=state["system_prompt"]), *prepared.messages]
            )
            emit_context_telemetry(
                state["context_id"], prepared, getattr(response, "usage_metadata", None)
            )
            update: dict[str, Any] = {"messages": [response]}
            if prepared.context_summary is not None:
                update["context_summary"] = prepared.context_summary
                update["history_compacted"] = True
            return update

        builder = StateGraph(AgentState)
        builder.add_node("load_skill_node", load_skill_node)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        builder.add_edge(START, "load_skill_node")
        builder.add_edge("load_skill_node", "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        return builder.compile()


def _structured_output_kwargs(provider: str) -> dict[str, str]:
    """Choose the structured-output transport a provider accepts for our schemas.

    OpenAI's native ``json_schema`` response format (LangChain's default) rejects
    ``oneOf``, which Pydantic emits for discriminated unions such as
    ``BusinessConstraint``. Tool-call arguments accept it, and the caller still
    validates the parsed object with Pydantic. Other providers keep their default.

    Args:
        provider: Configured model provider name (``openai``, ``claude``, ``gemini``).

    Returns:
        Keyword arguments for ``with_structured_output``; empty means library default.
    """

    return {"method": "function_calling"} if provider == "openai" else {}


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


def _stored_message(item: dict[str, str]) -> BaseMessage:
    """Convert one validated persisted role/content mapping to a message."""

    if item["role"] == "user":
        return HumanMessage(content=item["content"])
    return AIMessage(content=item["content"])


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


def _validated_context_summary(value: str | None, limit: int) -> str | None:
    """Normalize a browser-only compact summary before it enters model context."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > limit:
        raise ValueError("context_summary exceeds the configured browser-session limit")
    return normalized


def _preview(text: object, limit: int = 160) -> str:
    """Return a single-line, length-bounded preview of message or tool content."""

    flattened = " ".join(str(text).split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def _emit_step_events(
    on_event: Callable[[dict[str, Any]], None], node: str, delta: dict[str, Any]
) -> None:
    """Translate one graph-node update into concise progress events for a caller.

    Emits ``skills`` (once, when skills are loaded), ``tool_call`` per tool the
    model invokes, ``tool_result`` per returned tool message, and ``final`` when the
    model produces its answer with no further tool calls.
    """

    if node == "load_skill_node":
        names = [item.get("name") for item in (delta.get("available_skills") or [])]
        on_event({"type": "skills", "names": [name for name in names if name]})
        return
    for message in delta.get("messages", []) or []:
        if isinstance(message, AIMessage):
            calls = message.tool_calls or []
            for call in calls:
                on_event(
                    {"type": "tool_call", "name": call.get("name"), "args": call.get("args") or {}}
                )
            if not calls:
                on_event({"type": "final"})
        elif isinstance(message, ToolMessage):
            on_event(
                {
                    "type": "tool_result",
                    "name": getattr(message, "name", None),
                    "preview": _preview(_message_text(message)),
                }
            )
