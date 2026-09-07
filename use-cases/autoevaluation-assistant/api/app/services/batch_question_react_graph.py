"""LangGraph ReAct retrieval agent for one assessment batch question review.

The graph uses one assistant node and one rag_search tool node. The assistant
may call the tool at most twice; callers can attach a final structured review
node so free-form chat text is never parsed as ``QuestionReviewResult``.
"""

from __future__ import annotations

import json
import logging
import time
import hashlib
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.observability.llm_usage_logging import (
    TokenUsage,
    emit_llm_usage_event,
    extract_token_usage,
    model_name_from_llm,
    usage_context_from_config,
)
from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AssessmentQuestion
from app.models.language import DEFAULT_LANGUAGE
from app.services.ai_review_graph import (
    QuestionReviewInput,
    _format_retrieved_chunk_for_final_review,
    _normalize_question_review_result,
)
from app.services.document_extractors import ExtractedDocument
from app.services.evidence_indexing import (
    EvidenceChunk,
    chunk_extracted_documents,
    content_hash,
    embed_evidence_chunks,
)
from app.services.evidence_retrieval import (
    RagQuery,
    RagRetrievedChunk,
    build_question_rag_queries,
    retrieve_batch_rag_candidates,
    retrieve_document_rag_candidates,
    retrieve_task_rag_candidates,
)
from app.services.evidence_routing import estimate_text_tokens
from app.services.genai_responses import GenAiReviewClient, ReviewClientError
from app.services.joule_knowledge_importer import (
    DEFAULT_EMBEDDING_MODEL,
    GenAiHubEmbeddingClient,
)

MAX_RAG_TOOL_CALLS = 2
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You validate one assessment questionnaire question using only evidence returned by the rag_search tool.
Call rag_search once with queries that cover the question and all answer items.
Call rag_search a second time only if the first results are off-topic or insufficient for important decisions.
After at most two rag_search calls, briefly summarize whether the retrieved evidence is enough for a final structured review.
Do not choose questionnaire answers in chat. Do not include hidden reasoning or unsupported facts."""


def _commit_if_supported(repository: Any) -> None:
    """Commit repository progress when the repository exposes a commit hook.

    Inputs:
        repository: Repository object passed into the ReAct graph.

    Outputs:
        None. Durable repositories commit progress telemetry; test repositories
        can record the call or ignore it.
    """

    commit = getattr(repository, "commit", None)
    if callable(commit):
        commit()


def _format_timing_log_value(value: Any) -> str:
    """Format one timing-log value for default Python logging output.

    Inputs:
        value: Metric value to render in the log message.

    Outputs:
        str: Compact value text. Simple scalar values are unquoted; strings
        containing whitespace are JSON-quoted for readability.
    """

    if value is None:
        return "null"
    if isinstance(value, str):
        return value if value and not any(char.isspace() for char in value) else json.dumps(value)
    return str(value)


def _timing_log_message(stage: str, duration_ms: int, metrics: dict[str, Any]) -> str:
    """Build a readable timing message for default log formatters.

    Inputs:
        stage: Short machine-readable stage name.
        duration_ms: Stage duration in milliseconds.
        metrics: Additional key/value fields to append to the message.

    Outputs:
        str: Log message containing event name plus key/value timing fields.
    """

    fields = {"stage": stage, "duration_ms": duration_ms, **metrics}
    parts = [
        f"{key}={_format_timing_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    ]
    return "manual_ai_review_stage_timing " + " ".join(parts)


def _log_manual_stage_timing(
    review_input: QuestionReviewInput,
    stage: str,
    started_at: float,
    **metrics: Any,
) -> int:
    """Log one manual ReAct RAG stage duration with structured metadata.

    Inputs:
        review_input: Manual question review input that identifies the task and
            framework question.
        stage: Short machine-readable stage name.
        started_at: ``time.perf_counter`` timestamp captured before the stage.
        **metrics: Stage-specific counts, model names, or outcome values.

    Outputs:
        int: Stage duration in milliseconds.
    """

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log_metrics = {
        "task_id": review_input.task_id,
        "question_id": review_input.question.question_id,
        "dimension": review_input.question.dimension,
        **metrics,
    }
    logger.info(
        _timing_log_message(stage, duration_ms, log_metrics),
        extra={
            "task_id": review_input.task_id,
            "question_id": review_input.question.question_id,
            "dimension": review_input.question.dimension,
            "stage": stage,
            "duration_ms": duration_ms,
            **metrics,
        },
    )
    return duration_ms


def _format_chunk_count(chunk_count: int) -> str:
    """Return a human-readable evidence chunk count phrase.

    Inputs:
        chunk_count: Number of chunks being processed.

    Outputs:
        str: Count plus singular or plural noun for progress messages.
    """

    return f"{chunk_count} evidence {'chunk' if chunk_count == 1 else 'chunks'}"


def _question_prompt(
    question: AssessmentQuestion,
    current_selected_answer_item_ids: list[str],
    language: str,
) -> str:
    """Build the user prompt for one batch question review."""
    answer_lines = [
        f"- {item.answer_item_id} | level={item.level} | item_index={item.item_index} | {item.text}"
        for item in question.answer_items
    ]
    return "\n".join(
        [
            f"Language: {language}",
            f"Question ID: {question.question_id}",
            f"Dimension: {question.dimension}",
            f"Section: {question.section}",
            f"Question: {question.question}",
            f"Explanation: {question.explanation or 'None'}",
            "Current selected answer item IDs:",
            json.dumps(current_selected_answer_item_ids, ensure_ascii=False),
            "Answer items:",
            *answer_lines,
            "Use rag_search to gather evidence, then summarize retrieval sufficiency.",
        ]
    )


def _retrieved_chunks_from_graph_state(state: dict[str, Any]) -> list[RagRetrievedChunk]:
    """Collect retrieved evidence chunks from ReAct tool messages.

    Inputs:
        state: Final LangGraph state containing assistant and tool messages.

    Outputs:
        list[RagRetrievedChunk]: Deduplicated retrieved chunks in first-seen
        order. Malformed non-JSON tool content is ignored because the final
        structured review can still report insufficient evidence.
    """

    chunks: list[RagRetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    for message in state.get("messages", []):
        if getattr(message, "type", "") != "tool":
            continue
        try:
            payload = json.loads(str(getattr(message, "content", "")))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for chunk_payload in payload.get("chunks", []):
            try:
                chunk = RagRetrievedChunk.model_validate(chunk_payload)
            except ValueError:
                continue
            if chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def _build_structured_review_prompt(
    question: AssessmentQuestion,
    task: dict[str, Any],
    retrieved_chunks: list[RagRetrievedChunk],
    language: str,
) -> list[dict[str, Any]]:
    """Build the final structured review prompt from retrieved batch evidence.

    Inputs:
        question: Assessment framework question under review.
        task: Leased batch question task with current answer selections.
        retrieved_chunks: Evidence chunks returned by ReAct ``rag_search`` calls.
        language: Supported language for user-facing review text.

    Outputs:
        list[dict[str, Any]]: Responses API input messages for
        ``GenAiReviewClient.review_question``.
    """

    current_answers = list(task.get("current_selected_answer_item_ids", []))
    answer_lines = [
        f"- {item.answer_item_id} | level={item.level} | item_index={item.item_index} | {item.text}"
        for item in question.answer_items
    ]
    retrieved_chunk_lines = [
        _format_retrieved_chunk_for_final_review(chunk)
        for chunk in retrieved_chunks
    ]
    prompt_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Language: {language}",
                    f"Question ID: {question.question_id}",
                    f"Dimension: {question.dimension}",
                    f"Section: {question.section}",
                    f"Question: {question.question}",
                    f"Explanation: {question.explanation or 'None'}",
                    "Current selected answer item IDs:",
                    json.dumps(current_answers, ensure_ascii=False),
                    "Answer items:",
                    *answer_lines,
                ]
            ),
            (
                "Use only the retrieved evidence chunks below for the final "
                "QuestionReviewResult. If the chunks do not support an answer "
                "item, mark it unsupported or unclear; do not infer facts from "
                "the question text alone."
            ),
            "Retrieved evidence chunks:",
            *(retrieved_chunk_lines or ["No evidence chunks were retrieved."]),
        ]
    )
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}]


RagRetriever = Callable[[list[RagQuery], int], list[RagRetrievedChunk]]
"""Callable shape used by the reusable ReAct graph to retrieve chunks."""

RagToolCallSaver = Callable[
    [int, list[RagQuery], list[RagRetrievedChunk], int],
    None,
]
"""Callable shape used by the reusable ReAct graph to persist tool calls."""

RagProgressUpdater = Callable[
    [str, str, int | None, int | None, int | None],
    None,
]
"""Callable shape used by the reusable ReAct graph to publish progress."""

QuestionReviewRunner = Callable[[list[RagRetrievedChunk]], QuestionReviewResult]
"""Callable shape used by the ReAct graph final structured review node."""


class QuestionReactState(MessagesState, total=False):
    """LangGraph state for retrieval messages plus the optional final result.

    Inputs:
        messages: Conversation messages managed by ``MessagesState``.
        review_result: Optional structured review produced by the final graph
            node when a ``QuestionReviewRunner`` is supplied.

    Outputs:
        Typed state dictionary accepted by ``StateGraph``.
    """

    review_result: QuestionReviewResult


def build_question_react_graph(
    question: AssessmentQuestion,
    task: dict[str, Any],
    worker_id: str,
    llm: Any,
    retrieve_candidates: RagRetriever,
    save_tool_call: RagToolCallSaver,
    update_progress: RagProgressUpdater | None = None,
    review_question: QuestionReviewRunner | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> Any:
    """Build a compiled ReAct graph for one question and one vector corpus.

    Inputs:
        question: Framework question being reviewed.
        task: Task payload carrying task ID and current answer selections.
        worker_id: Worker lease owner used by progress callbacks.
        llm: Tool-capable chat model used for the ReAct retrieval step.
        retrieve_candidates: Callback that retrieves chunks from either a
            batch job corpus or a manual task corpus.
        save_tool_call: Callback that persists one ``rag_search`` audit row.
        update_progress: Optional callback for task progress telemetry.
        review_question: Optional callback that receives all retrieved chunks
            and returns the final structured review result. When omitted, the
            graph stops after the retrieval assistant completes.
        language: Output language for retrieval planning prompts.

    Outputs:
        Any: Compiled LangGraph application using ``QuestionReactState``.
    """
    _ = worker_id
    _ = language
    rag_call_counter = {"count": 0}

    @tool
    def rag_search(queries: list[dict[str, Any]], top_k_per_query: int = 8) -> str:
        """Search evidence chunks for one question using batched queries."""
        started_at = time.perf_counter()
        rag_call_counter["count"] += 1
        rag_call_number = rag_call_counter["count"]
        rag_queries = [RagQuery.model_validate(query) for query in queries]
        if update_progress is not None:
            update_progress(
                "retrieving_evidence",
                f"Retrieving evidence for {len(rag_queries)} queries.",
                rag_call_number,
                len(rag_queries),
                None,
            )
        chunks = retrieve_candidates(rag_queries, top_k_per_query)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        payload = {
            "question_id": question.question_id,
            "query_count": len(rag_queries),
            "retrieved_count": len(chunks),
            "duration_ms": duration_ms,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        save_tool_call(
            rag_call_number=rag_call_number,
            queries=rag_queries,
            chunks=chunks,
            duration_ms=duration_ms,
        )
        if update_progress is not None:
            update_progress(
                "finalizing_answer",
                f"RAG call {rag_call_number}/2 retrieved {len(chunks)} chunks "
                f"from {len(rag_queries)} queries.",
                rag_call_number,
                len(rag_queries),
                len(chunks),
            )
        return json.dumps(payload, ensure_ascii=False)

    bound_llm = llm.bind_tools([rag_search])

    react_model = model_name_from_llm(llm)

    def assistant(state: QuestionReactState) -> dict[str, list[BaseMessage]]:
        """Call the tool-bound LLM with system instructions and messages."""

        started_at = time.perf_counter()
        usage = TokenUsage()
        context = usage_context_from_config(
            ensure_config(),
            default_route="worker:document-processing:react_assistant",
            default_actor_type="batch",
        )
        try:
            response = bound_llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            )
            usage = extract_token_usage(response)
            emit_llm_usage_event(
                route=context.route,
                method=context.method,
                user_id=context.user_id,
                actor_type=context.actor_type,
                client_host=context.client_host,
                model=react_model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                input_total_tokens=usage.input_total_tokens,
                total_tokens=usage.total_tokens,
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
                model=react_model,
                llm_endpoint="chat.completions",
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                input_total_tokens=usage.input_total_tokens,
                total_tokens=usage.total_tokens,
                output_tokens=usage.output_tokens,
                outcome="error",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                correlation_id=context.correlation_id,
            )
            raise
        return {"messages": [response]}

    def final_destination() -> str:
        """Return the next terminal node for the configured graph shape."""

        return "final_review" if review_question is not None else END

    def route(state: QuestionReactState) -> str:
        """Route to tools while enforcing the maximum RAG tool call count."""
        route_name = tools_condition(state)
        if route_name != "tools":
            return final_destination()
        tool_message_count = sum(
            1 for message in state["messages"] if getattr(message, "type", "") == "tool"
        )
        return "tools" if tool_message_count < MAX_RAG_TOOL_CALLS else final_destination()

    def run_final_review(
        state: QuestionReactState,
    ) -> dict[str, QuestionReviewResult]:
        """Run the optional structured final review from retrieved chunks.

        Inputs:
            state: Graph state after the retrieval loop.

        Outputs:
            dict[str, QuestionReviewResult]: ``review_result`` update when the
            graph was configured with a final review callback.
        """

        if review_question is None:
            return {}
        retrieved_chunks = _retrieved_chunks_from_graph_state(state)
        if not retrieved_chunks and rag_call_counter["count"] == 0:
            started_at = time.perf_counter()
            fallback_queries = build_question_rag_queries(question)
            if update_progress is not None:
                update_progress(
                    "retrieving_evidence",
                    (
                        "Running fallback evidence retrieval because the "
                        "assistant did not call rag_search."
                    ),
                    1,
                    len(fallback_queries),
                    None,
                )
            retrieved_chunks = retrieve_candidates(fallback_queries, 8)
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            save_tool_call(
                rag_call_number=1,
                queries=fallback_queries,
                chunks=retrieved_chunks,
                duration_ms=duration_ms,
            )
            if update_progress is not None:
                update_progress(
                    "finalizing_answer",
                    (
                        "Fallback RAG retrieved "
                        f"{len(retrieved_chunks)} chunks from "
                        f"{len(fallback_queries)} queries."
                    ),
                    1,
                    len(fallback_queries),
                    len(retrieved_chunks),
                )
        return {"review_result": review_question(retrieved_chunks)}

    graph = StateGraph(QuestionReactState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode([rag_search]))
    graph.add_node("final_review", run_final_review)
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        route,
        {"tools": "tools", "final_review": "final_review", END: END},
    )
    graph.add_edge("tools", "assistant")
    graph.add_edge("final_review", END)
    return graph.compile()


def build_batch_question_react_graph(
    question: AssessmentQuestion,
    task: dict[str, Any],
    repository: Any,
    embedding_client: Any,
    worker_id: str,
    llm: Any,
    review_question: QuestionReviewRunner | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> Any:
    """Build a compiled ReAct graph for one leased batch question task.

    Inputs:
        question: Framework question being reviewed.
        task: Batch question task row with job ID and task ID.
        repository: Repository exposing batch retrieval and audit methods.
        embedding_client: Client used to embed RAG tool queries.
        worker_id: Worker that holds the task lease.
        llm: Tool-capable chat model used for retrieval planning.
        review_question: Optional final structured review callback.
        language: Output language for retrieval planning prompts.

    Outputs:
        Any: Compiled LangGraph application for batch document RAG.
    """

    def retrieve_candidates(
        queries: list[RagQuery],
        top_k_per_query: int,
    ) -> list[RagRetrievedChunk]:
        """Retrieve chunks from the shared batch document corpus."""
        return retrieve_batch_rag_candidates(
            repository=repository,
            embedding_client=embedding_client,
            job_id=task["job_id"],
            queries=queries,
            top_k_per_query=top_k_per_query,
        )

    def save_tool_call(
        rag_call_number: int,
        queries: list[RagQuery],
        chunks: list[RagRetrievedChunk],
        duration_ms: int,
    ) -> None:
        """Persist one batch ``rag_search`` tool-call audit row."""
        repository.save_batch_rag_tool_call(
            job_id=task["job_id"],
            task_id=task["task_id"],
            rag_call_number=rag_call_number,
            queries_json=[query.model_dump(mode="json") for query in queries],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            matched_queries_json={
                chunk.chunk_id: [
                    query.model_dump(mode="json") for query in chunk.matched_queries
                ]
                for chunk in chunks
            },
            query_count=len(queries),
            retrieved_chunk_count=len(chunks),
            duration_ms=duration_ms,
            stop_reason="tool_result",
        )

    def update_progress(
        status: str,
        progress_message: str,
        rag_call_count: int | None,
        query_count: int | None,
        retrieved_chunk_count: int | None,
    ) -> None:
        """Persist batch question progress for polling clients."""
        repository.update_batch_question_progress(
            task_id=task["task_id"],
            worker_id=worker_id,
            status=status,
            progress_message=progress_message,
            rag_call_count=rag_call_count,
            query_count=query_count,
            retrieved_chunk_count=retrieved_chunk_count,
        )
        _commit_if_supported(repository)

    return build_question_react_graph(
        question=question,
        task=task,
        worker_id=worker_id,
        llm=llm,
        retrieve_candidates=retrieve_candidates,
        save_tool_call=save_tool_call,
        update_progress=update_progress,
        review_question=review_question,
        language=language,
    )


def _attachment_ids_for_documents(
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    attachment_ids_by_document: list[str] | None,
) -> list[str]:
    """Return source attachment IDs aligned with extracted documents.

    Inputs:
        extracted_documents: Ordered extracted documents.
        attachment_id_by_file: Filename to attachment ID fallback mapping.
        attachment_ids_by_document: Optional ordered IDs from the worker.

    Outputs:
        list[str]: Attachment IDs aligned one-to-one with
        ``extracted_documents``.

    Raises:
        ValueError: Raised when provided ordered IDs do not align with the
        extracted document list.
    """
    if attachment_ids_by_document is not None:
        if len(attachment_ids_by_document) != len(extracted_documents):
            raise ValueError(
                "attachment_ids_by_document must align with extracted_documents"
            )
        return list(attachment_ids_by_document)
    return [attachment_id_by_file[document.file_name] for document in extracted_documents]


def _manual_summary_chunk_id(
    task_id: str,
    attachment_id: str,
    summary_text: str,
) -> str:
    """Build a stable chunk ID for one manual spreadsheet summary.

    Inputs:
        task_id: Manual question task identifier.
        attachment_id: Source attachment identifier.
        summary_text: Summary text included in the retrieval corpus.

    Outputs:
        str: Stable summary chunk identifier.
    """
    digest = hashlib.sha256(
        "\n".join([task_id, attachment_id, content_hash(summary_text)]).encode("utf-8")
    ).hexdigest()
    return f"chunk-{digest[:32]}"


def _manual_excel_summary_chunks(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
    attachment_ids_by_document: list[str],
    review_client: Any,
) -> list[EvidenceChunk]:
    """Create spreadsheet summary chunks for a manual question corpus.

    Inputs:
        review_input: Manual question review input.
        extracted_documents: Ordered extracted documents.
        attachment_ids_by_document: Attachment IDs aligned with the extracted
            documents.
        review_client: Review client used for LLM summaries when supported.

    Outputs:
        list[EvidenceChunk]: Summary chunks ready for embedding.
    """
    from app.services.batch_review_graph import _excel_summary_text

    summary_chunks: list[EvidenceChunk] = []
    for document, attachment_id in zip(
        extracted_documents,
        attachment_ids_by_document,
        strict=True,
    ):
        if document.document_type not in {"xlsx", "xlsm"}:
            continue
        summary_text = _excel_summary_text(
            document=document,
            review_client=review_client,
            language=review_input.language,
        )
        if not summary_text:
            continue
        source_block_ids = [block.block_id for block in document.blocks]
        summary_chunks.append(
            EvidenceChunk(
                chunk_id=_manual_summary_chunk_id(
                    task_id=review_input.task_id,
                    attachment_id=attachment_id,
                    summary_text=summary_text,
                ),
                task_id=review_input.task_id,
                attachment_id=attachment_id,
                file_name=document.file_name,
                document_type=document.document_type,
                source_block_ids=source_block_ids,
                location_json={"summary": True},
                chunk_text=summary_text,
                estimated_tokens=estimate_text_tokens(summary_text),
                embedding_model="",
                embedding=[],
                content_hash=content_hash(summary_text),
            )
        )
    return summary_chunks


def _manual_document_chunks(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    attachment_ids_by_document: list[str] | None,
    review_client: Any,
) -> list[EvidenceChunk]:
    """Create raw and spreadsheet-summary chunks for manual task RAG.

    Inputs:
        review_input: Manual question review input.
        extracted_documents: Ordered extracted evidence documents.
        attachment_id_by_file: Filename to attachment ID fallback mapping.
        attachment_ids_by_document: Optional ordered IDs for duplicate basename
            safety.
        review_client: Review client used for spreadsheet summaries.

    Outputs:
        list[EvidenceChunk]: Raw and summary chunks ready for embedding.
    """
    ordered_attachment_ids = _attachment_ids_for_documents(
        extracted_documents=extracted_documents,
        attachment_id_by_file=attachment_id_by_file,
        attachment_ids_by_document=attachment_ids_by_document,
    )
    chunks: list[EvidenceChunk] = []
    for document, attachment_id in zip(
        extracted_documents,
        ordered_attachment_ids,
        strict=True,
    ):
        chunks.extend(
            chunk_extracted_documents(
                task_id=review_input.task_id,
                attachment_id_by_file={document.file_name: attachment_id},
                extracted_documents=[document],
            )
        )
    chunks.extend(
        _manual_excel_summary_chunks(
            review_input=review_input,
            extracted_documents=extracted_documents,
            attachment_ids_by_document=ordered_attachment_ids,
            review_client=review_client,
        )
    )
    return chunks


def _insufficient_extractable_text_result(
    review_input: QuestionReviewInput,
    model: str,
) -> QuestionReviewResult:
    """Build a deterministic result for evidence with no indexable text.

    Inputs:
        review_input: Manual question review input.
        model: Model name to record in the result payload.

    Outputs:
        QuestionReviewResult: Explicit limitation result with no verified
        answer selections.
    """
    return QuestionReviewResult(
        question_id=review_input.question.question_id,
        model=model,
        overall_status="insufficient_extractable_text",
        highest_supported_level=None,
        current_selected_answer_item_ids=list(
            review_input.current_selected_answer_item_ids
        ),
        verified_selected_answer_item_ids=[],
        warnings=[
            (
                "The uploaded evidence did not produce extractable text for "
                "RAG indexing. Scan-only PDFs require OCR or text-bearing "
                "documents before AI verification can select answers."
            )
        ],
    )


def _default_react_llm() -> Any:
    """Create the default non-mini chat model for ReAct retrieval planning.

    Inputs:
        None. SAP Gen AI Hub configuration is loaded from the environment.

    Outputs:
        Any: LangChain chat model bound later to the ``rag_search`` tool.
    """
    from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

    return ChatOpenAI(
        proxy_model_name="gpt-5.4",
        temperature=0,
        max_tokens=4096,
    )


def run_manual_question_react_review(
    review_input: QuestionReviewInput,
    extracted_documents: list[ExtractedDocument],
    attachment_id_by_file: dict[str, str],
    repository: Any,
    worker_id: str = "",
    review_client: Any | None = None,
    embedding_client: Any | None = None,
    llm: Any | None = None,
    attachment_ids_by_document: list[str] | None = None,
) -> QuestionReviewResult:
    """Run the ReAct RAG review flow for one manual question task.

    Inputs:
        review_input: Validated manual question review input.
        extracted_documents: Text-bearing extracted evidence documents.
        attachment_id_by_file: Mapping from extracted document filename to the
            source attachment ID.
        repository: Repository exposing task chunk persistence, vector search,
            retrieval audit, and optional attempt persistence.
        worker_id: Worker lease owner for parity with batch execution.
        review_client: Optional injected review client for tests.
        embedding_client: Optional injected embedding client for tests.
        llm: Optional injected tool-capable chat model for tests.
        attachment_ids_by_document: Optional source attachment IDs aligned with
            ``extracted_documents`` for duplicate filename safety.

    Outputs:
        QuestionReviewResult: Normalized structured review result, or an
        explicit insufficient-text result when no chunks can be indexed.
    """
    load_dotenv()
    attempt_saver = getattr(repository, "save_question_attempt", None)
    if callable(attempt_saver):
        attempt_saver(
            task_id=review_input.task_id,
            mode="rag",
            status="routed",
            model=getattr(review_client, "model", None),
            usage_json={"strategy": "react_rag"},
            error_code=None,
            error_message=None,
        )

    client = review_client or GenAiReviewClient()
    progress_updater = getattr(repository, "update_question_progress", None)

    def update_progress(
        status: str,
        progress_message: str,
        rag_call_count: int | None = None,
        query_count: int | None = None,
        retrieved_chunk_count: int | None = None,
    ) -> None:
        """Persist manual task progress when the repository supports it."""

        if not callable(progress_updater) or not worker_id:
            return
        progress_updater(
            task_id=review_input.task_id,
            worker_id=worker_id,
            status=status,
            progress_message=progress_message,
            rag_call_count=rag_call_count,
            query_count=query_count,
            retrieved_chunk_count=retrieved_chunk_count,
        )
        _commit_if_supported(repository)

    chunk_started_at = time.perf_counter()
    chunks = _manual_document_chunks(
        review_input=review_input,
        extracted_documents=extracted_documents,
        attachment_id_by_file=attachment_id_by_file,
        attachment_ids_by_document=attachment_ids_by_document,
        review_client=client,
    )
    _log_manual_stage_timing(
        review_input,
        "chunk_documents",
        chunk_started_at,
        document_count=len(extracted_documents),
        chunk_count=len(chunks),
        block_count=sum(len(document.blocks) for document in extracted_documents),
    )
    if not chunks:
        save_started_at = time.perf_counter()
        repository.save_evidence_chunks(task_id=review_input.task_id, chunks=[])
        _log_manual_stage_timing(
            review_input,
            "save_chunks",
            save_started_at,
            chunk_count=0,
        )
        update_progress(
            "finalizing_answer",
            "No extractable evidence chunks were found in the uploaded documents.",
            0,
            0,
            0,
        )
        return _insufficient_extractable_text_result(
            review_input=review_input,
            model=getattr(client, "model", None) or "gpt-5.4",
        )

    embedder = embedding_client or GenAiHubEmbeddingClient(DEFAULT_EMBEDDING_MODEL)
    embedding_model = getattr(embedder, "model_name", None) or DEFAULT_EMBEDDING_MODEL
    update_progress(
        "embedding_documents",
        f"Creating embeddings for {_format_chunk_count(len(chunks))}.",
        0,
        0,
        0,
    )
    embed_started_at = time.perf_counter()
    embedded_chunks = embed_evidence_chunks(
        chunks=chunks,
        embedding_client=embedder,
        embedding_model=embedding_model,
    )
    _log_manual_stage_timing(
        review_input,
        "embed_chunks",
        embed_started_at,
        chunk_count=len(chunks),
        embedding_model=embedding_model,
    )
    save_started_at = time.perf_counter()
    repository.save_evidence_chunks(
        task_id=review_input.task_id,
        chunks=[chunk.model_dump(mode="json") for chunk in embedded_chunks],
    )
    _log_manual_stage_timing(
        review_input,
        "save_chunks",
        save_started_at,
        chunk_count=len(embedded_chunks),
    )

    task = {
        "task_id": review_input.task_id,
        "question_id": review_input.question.question_id,
        "dimension": review_input.question.dimension,
        "current_selected_answer_item_ids": list(
            review_input.current_selected_answer_item_ids
        ),
    }
    update_progress(
        "in_progress",
        (
            f"Planning retrieval for {review_input.question.question_id} "
            f"({review_input.question.dimension})."
        ),
        0,
        0,
        0,
    )

    def retrieve_candidates(
        queries: list[RagQuery],
        top_k_per_query: int,
    ) -> list[RagRetrievedChunk]:
        """Retrieve chunks from the manual task vector corpus."""
        return retrieve_task_rag_candidates(
            repository=repository,
            embedding_client=embedder,
            task_id=review_input.task_id,
            queries=queries,
            top_k_per_query=top_k_per_query,
        )

    def save_tool_call(
        rag_call_number: int,
        queries: list[RagQuery],
        chunks: list[RagRetrievedChunk],
        duration_ms: int,
    ) -> None:
        """Persist one manual ``rag_search`` tool-call audit row."""
        repository.save_retrieval_round(
            task_id=review_input.task_id,
            round_number=rag_call_number,
            queries=[query.query for query in queries],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            accepted_evidence_json={
                "matched_queries": {
                    chunk.chunk_id: [
                        query.model_dump(mode="json")
                        for query in chunk.matched_queries
                    ]
                    for chunk in chunks
                },
                "query_count": len(queries),
                "retrieved_chunk_count": len(chunks),
                "duration_ms": duration_ms,
            },
            evidence_gaps=[],
            refined_queries=[],
            stop_reason="tool_result",
        )
        _commit_if_supported(repository)

    react_started_at: float | None = None

    def final_review(retrieved_chunks: list[RagRetrievedChunk]) -> QuestionReviewResult:
        """Run and time the manual structured review as the graph final node."""

        _log_manual_stage_timing(
            review_input,
            "react_rag",
            react_started_at or time.perf_counter(),
            retrieved_chunk_count=len(retrieved_chunks),
            rag_call_count=len(repository.retrieval_rounds.get(review_input.task_id, []))
            if isinstance(getattr(repository, "retrieval_rounds", None), dict)
            else None,
        )
        update_progress(
            "finalizing_answer",
            (
                "Preparing structured review from "
                f"{len(retrieved_chunks)} retrieved evidence chunks."
            ),
            None,
            None,
            len(retrieved_chunks),
        )
        review_started_at = time.perf_counter()
        try:
            result = client.review_question(
                _build_structured_review_prompt(
                    question=review_input.question,
                    task=task,
                    retrieved_chunks=retrieved_chunks,
                    language=review_input.language,
                ),
                language=review_input.language,
            )
        except ReviewClientError:
            _log_manual_stage_timing(
                review_input,
                "structured_review",
                review_started_at,
                outcome="error",
                error_type="ReviewClientError",
                model=getattr(client, "model", None),
            )
            raise
        except Exception as exc:
            _log_manual_stage_timing(
                review_input,
                "structured_review",
                review_started_at,
                outcome="error",
                error_type=type(exc).__name__,
                model=getattr(client, "model", None),
            )
            raise ReviewClientError(
                "Manual question structured review failed.",
                error_code="review_model_failed",
                original_error=exc,
            ) from exc
        _log_manual_stage_timing(
            review_input,
            "structured_review",
            review_started_at,
            outcome="completed",
            model=getattr(client, "model", None),
        )
        return _normalize_question_review_result(review_input.question, result)

    graph = build_question_react_graph(
        question=review_input.question,
        task=task,
        worker_id=worker_id,
        llm=llm or _default_react_llm(),
        retrieve_candidates=retrieve_candidates,
        save_tool_call=save_tool_call,
        update_progress=update_progress,
        review_question=final_review,
        language=review_input.language,
    )
    react_started_at = time.perf_counter()
    state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=_question_prompt(
                        question=review_input.question,
                        current_selected_answer_item_ids=list(
                            review_input.current_selected_answer_item_ids
                        ),
                        language=review_input.language,
                    )
                )
            ]
        }
    )
    result = state.get("review_result")
    if result is None:
        raise ReviewClientError(
            "Manual question structured review did not produce a result.",
            error_code="review_model_failed",
        )
    return result


def run_corpus_question_react_review(
    question: AssessmentQuestion,
    task: dict[str, Any],
    repository: Any,
    embedding_client: Any | None = None,
    review_client: Any | None = None,
    llm: Any | None = None,
    worker_id: str = "",
    language: str = DEFAULT_LANGUAGE,
) -> QuestionReviewResult:
    """Run the ReAct RAG review graph against the assessment document corpus.

    Inputs:
        question: Framework question being reviewed.
        task: Unified question task row containing assessment_id and task_id.
        repository: Repository exposing corpus retrieval and progress methods.
        embedding_client: Optional embedding client override for tests.
        review_client: Optional review client override for tests.
        llm: Optional tool-capable chat model override for tests.
        worker_id: Worker that holds the active task lease.
        language: Output language for planning and final review.

    Outputs:
        QuestionReviewResult: Normalized structured review result.
    """
    load_dotenv()
    if embedding_client is None:
        embedding_client = GenAiHubEmbeddingClient()
    client = review_client or GenAiReviewClient()
    if llm is None:
        llm = _default_react_llm()
    current_answers = list(task.get("current_selected_answer_item_ids", []))
    task_language = task.get("language") or language

    repository.update_question_progress(
        task_id=task["task_id"],
        worker_id=worker_id,
        status="in_progress",
        progress_message=(
            f"Planning retrieval for {question.question_id} "
            f"({question.dimension})."
        ),
        rag_call_count=0,
        query_count=0,
        retrieved_chunk_count=0,
    )
    _commit_if_supported(repository)

    def retrieve_candidates(
        queries: list[RagQuery],
        top_k_per_query: int,
    ) -> list[RagRetrievedChunk]:
        """Retrieve chunks from the assessment-scoped document corpus."""
        return retrieve_document_rag_candidates(
            repository=repository,
            embedding_client=embedding_client,
            assessment_id=task["assessment_id"],
            queries=queries,
            top_k_per_query=top_k_per_query,
        )

    def save_tool_call(
        rag_call_number: int,
        queries: list[RagQuery],
        chunks: list[RagRetrievedChunk],
        duration_ms: int,
    ) -> None:
        """Persist one corpus ``rag_search`` tool-call audit row."""
        repository.save_retrieval_round(
            task_id=task["task_id"],
            round_number=rag_call_number,
            queries=[query.query for query in queries],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            accepted_evidence_json={
                "matched_queries": {
                    chunk.chunk_id: [
                        query.model_dump(mode="json")
                        for query in chunk.matched_queries
                    ]
                    for chunk in chunks
                },
                "query_count": len(queries),
                "retrieved_chunk_count": len(chunks),
                "duration_ms": duration_ms,
            },
            evidence_gaps=[],
            refined_queries=[],
            stop_reason="tool_result",
        )
        _commit_if_supported(repository)

    def update_progress(
        status: str,
        progress_message: str,
        rag_call_count: int | None,
        query_count: int | None,
        retrieved_chunk_count: int | None,
    ) -> None:
        """Update unified question task progress for polling clients."""
        repository.update_question_progress(
            task_id=task["task_id"],
            worker_id=worker_id,
            status=status,
            progress_message=progress_message,
            rag_call_count=rag_call_count,
            query_count=query_count,
            retrieved_chunk_count=retrieved_chunk_count,
        )
        _commit_if_supported(repository)

    def final_review(retrieved_chunks: list[RagRetrievedChunk]) -> QuestionReviewResult:
        """Run the structured final review from retrieved corpus chunks."""
        update_progress(
            "finalizing_answer",
            (
                "Preparing structured review from "
                f"{len(retrieved_chunks)} retrieved evidence chunks."
            ),
            None,
            None,
            len(retrieved_chunks),
        )
        try:
            result = client.review_question(
                _build_structured_review_prompt(
                    question=question,
                    task=task,
                    retrieved_chunks=retrieved_chunks,
                    language=task_language,
                ),
                language=task_language,
            )
        except ReviewClientError:
            raise
        except Exception as exc:
            raise ReviewClientError(
                "Corpus question structured review failed.",
                error_code="review_model_failed",
                original_error=exc,
            ) from exc
        return _normalize_question_review_result(question, result)

    graph = build_question_react_graph(
        question=question,
        task=task,
        worker_id=worker_id,
        llm=llm,
        retrieve_candidates=retrieve_candidates,
        save_tool_call=save_tool_call,
        update_progress=update_progress,
        review_question=final_review,
        language=task_language,
    )
    state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=_question_prompt(
                        question=question,
                        current_selected_answer_item_ids=current_answers,
                        language=task_language,
                    )
                )
            ]
        }
    )
    result = state.get("review_result")
    if result is None:
        raise ReviewClientError(
            "Corpus question structured review did not produce a result.",
            error_code="review_model_failed",
        )
    return result


def run_batch_question_react_review(
    question: AssessmentQuestion,
    task: dict[str, Any],
    repository: Any,
    embedding_client: Any | None = None,
    review_client: Any | None = None,
    llm: Any | None = None,
    worker_id: str = "",
    language: str = DEFAULT_LANGUAGE,
) -> QuestionReviewResult:
    """Run the ReAct RAG review graph for one leased batch question task."""
    load_dotenv()
    if embedding_client is None:
        embedding_client = GenAiHubEmbeddingClient()
    client = review_client or GenAiReviewClient()
    if llm is None:
        llm = _default_react_llm()
    current_answers = list(task.get("current_selected_answer_item_ids", []))
    progress_updater = getattr(repository, "update_batch_question_progress", None)
    if callable(progress_updater):
        progress_updater(
            task_id=task["task_id"],
            worker_id=worker_id,
            status="in_progress",
            progress_message=(
                f"Planning retrieval for {question.question_id} "
                f"({question.dimension})."
            ),
        )
        _commit_if_supported(repository)

    def final_review(retrieved_chunks: list[RagRetrievedChunk]) -> QuestionReviewResult:
        """Run the batch structured review as the graph final node."""

        if callable(progress_updater):
            progress_updater(
                task_id=task["task_id"],
                worker_id=worker_id,
                status="finalizing_answer",
                progress_message=(
                    "Preparing structured review from "
                    f"{len(retrieved_chunks)} retrieved evidence chunks."
                ),
            )
            _commit_if_supported(repository)
        try:
            result = client.review_question(
                _build_structured_review_prompt(
                    question=question,
                    task=task,
                    retrieved_chunks=retrieved_chunks,
                    language=language,
                ),
                language=language,
            )
        except ReviewClientError:
            raise
        except Exception as exc:
            raise ReviewClientError(
                "Batch question structured review failed.",
                error_code="review_model_failed",
                original_error=exc,
            ) from exc
        return _normalize_question_review_result(question, result)

    graph = build_batch_question_react_graph(
        question=question,
        task=task,
        repository=repository,
        embedding_client=embedding_client,
        worker_id=worker_id,
        llm=llm,
        review_question=final_review,
        language=language,
    )
    state = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=_question_prompt(
                        question=question,
                        current_selected_answer_item_ids=current_answers,
                        language=language,
                    )
                )
            ]
        }
    )
    result = state.get("review_result")
    if result is None:
        raise ReviewClientError(
            "Batch question structured review did not produce a result.",
            error_code="review_model_failed",
        )
    return result
