"""LangGraph runtime for the Assessment knowledge A2A agent."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.orm import sessionmaker

from app.db import create_hana_engine
from app.observability.llm_usage_logging import (
    TokenUsage,
    emit_llm_usage_event,
    extract_token_usage,
    model_name_from_llm,
    usage_context_from_config,
)
from app.services.joule_knowledge import (
    GlossarySearchResult,
    JouleKnowledgeService,
)
from app.services.ai_review_repository.hana import HanaAiReviewRepository
from app.services.joule_knowledge_importer import GenAiHubEmbeddingClient
from app.services.joule_knowledge_repository import HanaJouleKnowledgeRepository

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Assessment Knowledge Agent.

Answer questions in English or Italian according to the user's language.

You have two complementary knowledge paths:

1. Fuzzy/structured assessment resource tools:
   - glossary_search: terms, acronyms, regulations, concepts, definitions, and
     short named topics such as "AI Act".
   - question_explanation: assessment questions, question IDs, or why a
     questionnaire topic matters.
   - dimension_explanation: assessment dimensions and their aliases.
2. RAG admin document tool:
   - admin_document_search: broader policy, process, methodology, procedure,
     and document-grounded questions from admin-managed documents.

Use the tool path that best matches the user request, but do not treat either
path as exclusive. If glossary_search, question_explanation, or
dimension_explanation returns empty or insufficient results, try
admin_document_search before saying the answer is unavailable. If
admin_document_search returns empty or insufficient results, try
glossary_search, question_explanation, and dimension_explanation before saying
the answer is unavailable.

Base answers only on returned tool content. Cite admin document answers with
file_name and chunk_id when those fields are available. When all relevant tool
paths are empty or not relevant, state that the answer is not available in the
loaded assessment resources or admin documents.
"""


@dataclass(frozen=True)
class AgentResponse:
    """Response returned by the Joule knowledge runtime.

    Attributes:
        status: Runtime status for the A2A executor.
        message: Final text to return to Joule.
    """

    status: Literal["completed", "input_required", "error"]
    message: str


class GenAiHubRuntimeTranslator:
    """Translate queries and grounded answers through SAP Gen AI Hub.

    Inputs:
        llm: LangChain-compatible chat model. If omitted, a GPT-4.1 proxy model
            is created from SAP Gen AI Hub configuration.

    Outputs:
        Translator object used by ``JouleKnowledgeService`` for Italian
        compatibility around English source data.
    """

    def __init__(self, llm: Any | None = None) -> None:
        """Initialize the translator LLM.

        Inputs:
            llm: Optional chat model test double or SAP Gen AI Hub proxy model.

        Outputs:
            None.
        """

        self.llm = llm or make_chat_llm(max_tokens=512, temperature=0.0)

    def translate_query_to_english(self, text: str, language: str) -> str:
        """Translate a retrieval query into English.

        Inputs:
            text: User query.
            language: Source language.

        Outputs:
            str: English query suitable for semantic retrieval.
        """

        if language == "en":
            return text
        started_at = time.perf_counter()
        usage = TokenUsage()
        model = model_name_from_llm(self.llm)
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Translate the user query to concise English for search. "
                            "Return only the translated query."
                        )
                    ),
                    HumanMessage(content=text),
                ]
            )
            usage = extract_token_usage(response)
            emit_llm_usage_event(
                route="joule:translator:translate_query_to_english",
                actor_type="system",
                model=model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception:
            emit_llm_usage_event(
                route="joule:translator:translate_query_to_english",
                actor_type="system",
                model=model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise
        return _render_content(response.content)

    def translate_answer_from_english(self, text: str, language: str) -> str:
        """Translate grounded English content into the requested language.

        Inputs:
            text: English source content.
            language: Target language.

        Outputs:
            str: Translated content, or the original text for English.
        """

        if language == "en":
            return text
        started_at = time.perf_counter()
        usage = TokenUsage()
        model = model_name_from_llm(self.llm)
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Translate the following grounded business text to Italian. "
                            "Keep codes and dimension names unchanged."
                        )
                    ),
                    HumanMessage(content=text),
                ]
            )
            usage = extract_token_usage(response)
            emit_llm_usage_event(
                route="joule:translator:translate_answer_from_english",
                actor_type="system",
                model=model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        except Exception:
            emit_llm_usage_event(
                route="joule:translator:translate_answer_from_english",
                actor_type="system",
                model=model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise
        return _render_content(response.content)


def make_chat_llm(max_tokens: int | None = None, temperature: float | None = None) -> Any:
    """Create the SAP Gen AI Hub GPT chat model used by LangGraph.

    Inputs:
        max_tokens: Optional response token limit.
        temperature: Optional model temperature.

    Outputs:
        Any: LangChain-compatible SAP Gen AI Hub chat model.
    """

    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

    return ChatOpenAI(
        proxy_model_name=os.getenv("JOULE_A2A_MODEL_NAME", "gpt-4.1"),
        temperature=(
            float(os.getenv("GENAI_TEMPERATURE", "0.2"))
            if temperature is None
            else temperature
        ),
        max_tokens=(
            int(os.getenv("GENAI_MAX_TOKENS", "2048"))
            if max_tokens is None
            else max_tokens
        ),
    )


def _render_content(content: object) -> str:
    """Render LangChain or A2A content into plain text.

    Inputs:
        content: String, structured content list, or any object returned by the
            model/tool runtime.

    Outputs:
        str: Text representation safe to return to Joule.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content)


def _format_glossary_results(results: list[GlossarySearchResult]) -> str:
    """Format glossary rows into compact tool output for the LLM.

    Inputs:
        results: Glossary search results.

    Outputs:
        str: JSON string containing result rows.
    """

    return json.dumps(
        [
            {
                "term": result.term,
                "definition": result.definition,
                "language": result.language,
                "score": result.score,
                "source_row": result.source_row,
            }
            for result in results
        ],
        ensure_ascii=False,
    )


def build_joule_knowledge_service() -> JouleKnowledgeService:
    """Create the HANA-backed retrieval service for the agent.

    Inputs:
        None. HANA and SAP AI Core credentials are read from the environment.

    Outputs:
        JouleKnowledgeService: Service wired to HANA, embeddings, and runtime
        translation.
    """

    Session = sessionmaker(bind=create_hana_engine())
    session = Session()
    repository = HanaJouleKnowledgeRepository(session)
    repository.create_schema()
    HanaAiReviewRepository(session).create_schema()
    repository.commit()
    return JouleKnowledgeService(
        repository=repository,
        embedding_client=GenAiHubEmbeddingClient(),
        translator=GenAiHubRuntimeTranslator(),
    )


def build_joule_message_repository() -> HanaJouleKnowledgeRepository:
    """Create a HANA repository for persisted agent conversation messages.

    Inputs:
        None. HANA credentials are read from the environment.

    Outputs:
        HanaJouleKnowledgeRepository: Repository with schema initialized.
    """

    Session = sessionmaker(bind=create_hana_engine())
    session = Session()
    repository = HanaJouleKnowledgeRepository(session)
    repository.create_schema()
    repository.commit()
    return repository


def build_joule_knowledge_graph(
    service_factory: Callable[[], JouleKnowledgeService] = build_joule_knowledge_service,
    llm: Any | None = None,
) -> Any:
    """Build the LangGraph tool-calling graph for Joule knowledge answers.

    Inputs:
        service_factory: Factory that creates the retrieval service used by
            tool functions.
        llm: Optional chat model for tests; defaults to SAP Gen AI Hub GPT-4.1.

    Outputs:
        Any: Compiled LangGraph application.
    """

    service = service_factory()

    @tool
    def question_explanation(query: str, language: str = "en") -> str:
        """Find why an assessment question is important."""

        results = service.question_explanation(query=query, language=language)
        return json.dumps(
            [
                {
                    "question_id": result.question_id,
                    "dimension": result.dimension,
                    "question": result.question,
                    "explanation": result.explanation,
                    "similarity_score": result.similarity_score,
                }
                for result in results
            ],
            ensure_ascii=False,
        )

    @tool
    def dimension_explanation(dimension: str, language: str = "en") -> str:
        """Explain one of the seven assessment dimensions."""

        result = service.dimension_explanation(dimension=dimension, language=language)
        if result is None:
            return json.dumps({"error": "dimension_not_found"}, ensure_ascii=False)
        return json.dumps(
            {
                "dimension_key": result.dimension_key,
                "name": result.name,
                "explanation": result.explanation,
                "aliases": result.aliases,
            },
            ensure_ascii=False,
        )

    @tool
    def glossary_search(term: str, language: str = "en", top_k: int = 5) -> str:
        """Search the bilingual assessment glossary using exact and fuzzy matching."""

        return _format_glossary_results(
            service.glossary_search(term=term, language=language, top_k=top_k)
        )

    @tool
    def admin_document_search(query: str, language: str = "en", top_k: int = 5) -> str:
        """Search admin-managed documents for general grounded Joule answers."""

        return json.dumps(
            service.admin_document_search(
                query=query,
                language=language,
                top_k=top_k,
            ),
            ensure_ascii=False,
        )

    chat_llm = llm or make_chat_llm()
    graph_model = model_name_from_llm(chat_llm)
    bound_llm = chat_llm.bind_tools(
        [
            question_explanation,
            dimension_explanation,
            glossary_search,
            admin_document_search,
        ]
    )

    async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Call the LLM with system instructions and accumulated messages."""

        started_at = time.perf_counter()
        usage = TokenUsage()
        context = usage_context_from_config(
            ensure_config(),
            default_route="/a2a",
            default_actor_type="human",
        )
        try:
            response = await bound_llm.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
            usage = extract_token_usage(response)
            emit_llm_usage_event(
                route=context.route,
                method=context.method,
                user_id=context.user_id,
                actor_type=context.actor_type,
                client_host=context.client_host,
                model=graph_model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="success",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                correlation_id=context.correlation_id,
            )
        except Exception:
            emit_llm_usage_event(
                route=context.route,
                method=context.method,
                user_id=context.user_id,
                actor_type=context.actor_type,
                client_host=context.client_host,
                model=graph_model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                correlation_id=context.correlation_id,
            )
            raise
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", call_model)
    builder.add_node(
        "tools",
        ToolNode(
            [
                question_explanation,
                dimension_explanation,
                glossary_search,
                admin_document_search,
            ]
        ),
    )
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()


class JouleKnowledgeGraphAgent:
    """Coordinate HANA message persistence and LangGraph invocation.

    Inputs:
        graph_factory: Optional factory returning a compiled LangGraph app.
        message_repository: Optional repository exposing conversation message
            persistence methods.

    Outputs:
        Runtime object used by the A2A executor.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
        self,
        graph_factory: Callable[[], Any] = build_joule_knowledge_graph,
        message_repository: Any | None = None,
    ) -> None:
        """Initialize the graph runtime and optional message repository.

        Inputs:
            graph_factory: Factory called lazily to build the compiled graph.
            message_repository: Optional repository used to persist messages.

        Outputs:
            None.
        """

        self._graph_factory = graph_factory
        self._graph: Any | None = None
        self.message_repository = message_repository

    async def answer(self, query: str, context_id: str) -> AgentResponse:
        """Answer one user turn using the supplied A2A context ID.

        Inputs:
            query: User utterance from Joule.
            context_id: A2A context ID mapped to LangGraph ``thread_id``.

        Outputs:
            AgentResponse: Completed or error response for the A2A executor.
        """

        graph = self._ensure_graph()
        prior_messages = self._load_prior_messages(context_id)
        messages: list[BaseMessage] = [
            *prior_messages,
            HumanMessage(content=query),
        ]
        self._append_message(context_id, "user", query)
        try:
            result = await graph.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {"thread_id": context_id},
                    "metadata": {
                        "llm_usage": {
                            "route": "/a2a",
                            "method": "POST",
                            "correlation_id": context_id,
                            "actor_type": "human",
                        }
                    },
                },
            )
            message = _render_content(result["messages"][-1].content)
            self._append_message(context_id, "assistant", message)
            return AgentResponse(status="completed", message=message)
        except Exception as exc:
            logger.exception("Joule knowledge graph invocation failed")
            return AgentResponse(status="error", message=f"Error: {exc}")

    def _ensure_graph(self) -> Any:
        """Return the lazily compiled graph.

        Inputs:
            None.

        Outputs:
            Any: Compiled LangGraph app.
        """

        if self._graph is None:
            self._graph = self._graph_factory()
        return self._graph

    def _load_prior_messages(self, context_id: str) -> list[BaseMessage]:
        """Load persisted prior messages for a context when available.

        Inputs:
            context_id: A2A context ID.

        Outputs:
            list[BaseMessage]: Prior conversation messages converted to
            LangChain message objects.
        """

        if self.message_repository is None or not hasattr(
            self.message_repository,
            "list_agent_messages",
        ):
            return []
        messages: list[BaseMessage] = []
        for row in self.message_repository.list_agent_messages(context_id):
            if row["role"] == "assistant":
                messages.append(AIMessage(content=row["content"]))
            else:
                messages.append(HumanMessage(content=row["content"]))
        return messages

    def _append_message(self, context_id: str, role: str, content: str) -> None:
        """Persist a message if a repository is configured.

        Inputs:
            context_id: A2A context ID.
            role: Message role.
            content: Message content.

        Outputs:
            None.
        """

        if self.message_repository is None:
            return
        self.message_repository.append_agent_message(context_id, role, content)
        if hasattr(self.message_repository, "commit"):
            self.message_repository.commit()
