"""Tests for LangGraph-backed Joule knowledge agent runtime behavior."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.services.joule_knowledge_agent import (
    JouleKnowledgeGraphAgent,
    SYSTEM_PROMPT,
    build_joule_knowledge_graph,
    make_chat_llm,
)


class _FakeCompiledGraph:
    """Fake compiled graph that records LangGraph invocation config.

    Inputs:
        None. Calls are stored for assertions.

    Outputs:
        Object exposing ``ainvoke`` with the subset used by the runtime agent.
    """

    def __init__(self) -> None:
        """Initialize an empty call log.

        Inputs:
            None.

        Outputs:
            None.
        """

        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def ainvoke(
        self,
        payload: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, object]:
        """Record the invocation and return a fake final message.

        Inputs:
            payload: LangGraph input payload.
            config: LangGraph runtime config.

        Outputs:
            dict[str, object]: Fake result containing an assistant message.
        """

        self.calls.append((payload, config))
        return {"messages": [_FakeMessage("The answer from HANA.")]}


class _FakeMessage:
    """Fake LangChain message carrying text content."""

    def __init__(self, content: str) -> None:
        """Store message content.

        Inputs:
            content: Text content returned by the fake graph.

        Outputs:
            None.
        """

        self.content = content


class _FakeMessageRepository:
    """Fake HANA message repository used to test conversation persistence."""

    def __init__(self) -> None:
        """Initialize an empty persisted message list.

        Inputs:
            None.

        Outputs:
            None.
        """

        self.messages: list[tuple[str, str, str]] = []

    def append_agent_message(self, context_id: str, role: str, content: str) -> None:
        """Record one persisted conversation message.

        Inputs:
            context_id: A2A/LangGraph context identifier.
            role: Conversation role.
            content: Message text.

        Outputs:
            None.
        """

        self.messages.append((context_id, role, content))


def test_system_prompt_requires_bidirectional_tool_fallback() -> None:
    """Verify tool-routing instructions preserve fuzzy and RAG fallback paths."""

    normalized_prompt = " ".join(SYSTEM_PROMPT.lower().split())

    assert "glossary_search" in normalized_prompt
    assert "question_explanation" in normalized_prompt
    assert "dimension_explanation" in normalized_prompt
    assert "admin_document_search" in normalized_prompt
    assert "if glossary_search, question_explanation, or dimension_explanation" in normalized_prompt
    assert "try admin_document_search before saying the answer is unavailable" in normalized_prompt
    assert "if admin_document_search" in normalized_prompt
    assert "try glossary_search, question_explanation, and dimension_explanation" in normalized_prompt


def test_joule_graph_binds_admin_document_search_tool() -> None:
    """Verify the LangGraph agent exposes the admin RAG retrieval tool."""

    class FakeToolBoundLlm:
        """Fake chat model that records tool binding names."""

        proxy_model_name = "joule-model"

        def __init__(self) -> None:
            """Initialize the bound tool name log."""

            self.tool_names: list[str] = []

        def bind_tools(self, tools: list[object]) -> "FakeToolBoundLlm":
            """Record LangChain tool names and return self."""

            self.tool_names = [tool.name for tool in tools]
            return self

    llm = FakeToolBoundLlm()

    build_joule_knowledge_graph(
        service_factory=lambda: object(),
        llm=llm,
    )

    assert "admin_document_search" in llm.tool_names


def test_joule_knowledge_agent_uses_context_id_as_langgraph_thread() -> None:
    """Verify A2A context IDs are mapped to LangGraph thread IDs.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm graph config and HANA message persistence use
        the same context identifier across user and assistant messages.
    """

    graph = _FakeCompiledGraph()
    repository = _FakeMessageRepository()
    agent = JouleKnowledgeGraphAgent(
        graph_factory=lambda: graph,
        message_repository=repository,
    )

    response = asyncio.run(agent.answer("What does AI mean?", "ctx-123"))

    assert response.message == "The answer from HANA."
    assert graph.calls[0][1]["configurable"] == {"thread_id": "ctx-123"}
    assert graph.calls[0][1]["metadata"]["llm_usage"]["route"] == "/a2a"
    assert graph.calls[0][1]["metadata"]["llm_usage"]["correlation_id"] == "ctx-123"
    assert repository.messages == [
        ("ctx-123", "user", "What does AI mean?"),
        ("ctx-123", "assistant", "The answer from HANA."),
    ]


def test_joule_knowledge_agent_reuses_same_graph_between_turns() -> None:
    """Verify the agent keeps one compiled graph while changing thread config.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm repeated calls reuse the same graph and pass
        the supplied context ID on every invocation.
    """

    graph = _FakeCompiledGraph()
    agent = JouleKnowledgeGraphAgent(graph_factory=lambda: graph)

    asyncio.run(agent.answer("First", "ctx-1"))
    asyncio.run(agent.answer("Second", "ctx-1"))

    assert len(graph.calls) == 2
    assert all(call[1]["configurable"] == {"thread_id": "ctx-1"} for call in graph.calls)


def test_joule_graph_assistant_emits_usage_event(capsys) -> None:
    """Verify the Joule LangGraph assistant node emits token usage telemetry.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions confirm LangChain usage metadata is logged.
    """

    class FakeToolBoundLlm:
        """Fake tool-bound chat model returning LangChain usage metadata."""

        proxy_model_name = "joule-model"

        def bind_tools(self, tools: list[object]) -> "FakeToolBoundLlm":
            """Return self after receiving graph tools.

            Inputs:
                tools: Tool definitions supplied by the graph builder.

            Outputs:
                FakeToolBoundLlm: This fake model instance.
            """

            return self

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            """Return a fake assistant message with token usage metadata.

            Inputs:
                messages: Prompt messages sent by the graph assistant node.

            Outputs:
                AIMessage: Assistant response with normalized usage metadata.
            """

            return AIMessage(
                content="Answer from Joule graph.",
                usage_metadata={
                    "input_tokens": 44,
                    "output_tokens": 11,
                    "total_tokens": 55,
                },
            )

    graph = build_joule_knowledge_graph(
        service_factory=lambda: object(),
        llm=FakeToolBoundLlm(),
    )

    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="What does AI mean?")]},
            config={
                "configurable": {"thread_id": "ctx-usage"},
                "metadata": {
                    "llm_usage": {
                        "route": "/a2a",
                        "method": "POST",
                        "correlation_id": "ctx-usage",
                        "actor_type": "human",
                    }
                },
            },
        )
    )

    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "/a2a"
    assert event["method"] == "POST"
    assert event["model"] == "joule-model"
    assert event["llm_endpoint"] == "chat.completions"
    assert event["input_tokens"] == 44
    assert event["output_tokens"] == 11
    assert event["outcome"] == "success"
    assert event["correlation_id"] == "ctx-usage"


def test_joule_graph_assistant_emits_error_usage_event(capsys) -> None:
    """Verify failed Joule assistant calls emit error usage telemetry.

    Inputs:
        capsys: Pytest stdout capture fixture.

    Outputs:
        None. Assertions confirm failed LangChain calls are logged.
    """

    class FailingToolBoundLlm:
        """Fake chat model that raises from async invocation."""

        proxy_model_name = "joule-model"

        def bind_tools(self, tools: list[object]) -> "FailingToolBoundLlm":
            """Return self after receiving graph tools."""

            return self

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            """Raise a deterministic provider-style error."""

            raise ValueError("provider unavailable")

    graph = build_joule_knowledge_graph(
        service_factory=lambda: object(),
        llm=FailingToolBoundLlm(),
    )

    with pytest.raises(ValueError):
        asyncio.run(
            graph.ainvoke(
                {"messages": [HumanMessage(content="What does AI mean?")]},
                config={
                    "configurable": {"thread_id": "ctx-error"},
                    "metadata": {
                        "llm_usage": {
                            "route": "/a2a",
                            "method": "POST",
                            "correlation_id": "ctx-error",
                        }
                    },
                },
            )
        )

    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["route"] == "/a2a"
    assert event["model"] == "joule-model"
    assert event["input_tokens"] == 0
    assert event["output_tokens"] == 0
    assert event["outcome"] == "error"


def test_make_chat_llm_uses_joule_specific_model_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Joule A2A model config does not reuse generic GenAI naming.

    Inputs:
        monkeypatch: Pytest helper used to replace environment values and the
        SAP Gen AI Hub proxy class.

    Outputs:
        None. Assertions confirm ``JOULE_A2A_MODEL_NAME`` controls the A2A
        agent LLM model selection.
    """

    calls: list[dict[str, object]] = []

    class FakeChatOpenAI:
        """Capture ChatOpenAI constructor arguments for env-var assertions."""

        def __init__(self, **kwargs: object) -> None:
            """Record constructor keyword arguments.

            Inputs:
                kwargs: Constructor arguments passed by ``make_chat_llm``.

            Outputs:
                None.
            """

            calls.append(kwargs)

    monkeypatch.setenv("GENAI_MODEL_NAME", "document-review-model")
    monkeypatch.setenv("JOULE_A2A_MODEL_NAME", "joule-a2a-model")
    monkeypatch.setitem(
        sys.modules,
        "gen_ai_hub.proxy.langchain.openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )

    make_chat_llm()

    assert calls[0]["proxy_model_name"] == "joule-a2a-model"
