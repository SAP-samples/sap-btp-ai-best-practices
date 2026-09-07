"""Tests for the batch upload LangGraph fanout review workflow."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from app.models.ai_review import QuestionReviewResult, RetrievalRoundAssessment
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.batch_question_react_graph import (
    build_question_react_graph,
    run_batch_question_react_review,
)
from app.services.batch_review_graph import (
    BatchIndexingResult,
    BatchReviewInput,
    run_document_ingestion_from_hana,
    run_batch_indexing_from_hana,
    run_batch_review_graph,
)
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository
from app.services.document_extractors import ExtractedBlock, ExtractedDocument
from app.services.evidence_retrieval import RagQuery, RagRetrievedChunk


class FakeEmbeddingClient:
    """Fake embedding client for batch graph tests."""

    def __init__(self) -> None:
        """Initialize call logs for batch and query embeddings."""
        self.batch_texts: list[str] = []
        self.query_texts: list[str] = []
        self.model_name = "fake-embedding"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings while recording batch texts.

        Inputs:
            texts: Chunk texts to embed.

        Outputs:
            list[list[float]]: One vector per input text.
        """
        self.batch_texts.extend(texts)
        return [[float(index + 1)] for index, _text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic query embedding.

        Inputs:
            text: Retrieval query.

        Outputs:
            list[float]: Query vector.
        """
        self.query_texts.append(text)
        return [float(len(text))]


class FakeReviewClient:
    """Fake review client that approves retrieved evidence and returns results."""

    model = "fake-gpt-5.4"

    def __init__(self, overall_status: str = "supported") -> None:
        """Initialize review call counters."""
        self.reviewed_questions: list[str] = []
        self.overall_status = overall_status

    def assess_retrieved_evidence(
        self,
        prompt: str,
        language: str = "en",
    ) -> RetrievalRoundAssessment:
        """Return a sufficient evidence assessment for the first retrieved chunk.

        Inputs:
            prompt: RAG assessment prompt.
            language: Requested output language.

        Outputs:
            RetrievalRoundAssessment: Sufficient assessment to trigger final
            question review.
        """
        _ = prompt, language
        return RetrievalRoundAssessment(
            accepted_evidence=[],
            rejected_evidence=[],
            evidence_gaps=[],
            refined_queries=[],
            sufficient_for_final_review=True,
            stop_reason="sufficient_evidence",
        )

    def review_question(
        self,
        input_messages: list[dict[str, object]],
        language: str = "en",
    ) -> QuestionReviewResult:
        """Return a result for the question ID embedded in the prompt.

        Inputs:
            input_messages: Final review prompt messages.
            language: Requested output language.

        Outputs:
            QuestionReviewResult: Minimal successful review result.
        """
        _ = language
        prompt_text = str(input_messages)
        question_id = (
            "Q.RCG.01.01" if "Q.RCG.01.01" in prompt_text else "Q.STR.01.01"
        )
        self.reviewed_questions.append(question_id)
        return QuestionReviewResult(
            question_id=question_id,
            model=self.model,
            overall_status=self.overall_status,
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        )


class FakeBatchRepository:
    """In-memory repository seam used by the batch graph test."""

    def __init__(self) -> None:
        """Initialize chunk and result stores."""
        self.saved_chunks: list[dict[str, object]] = []
        self.results: dict[str, QuestionReviewResult] = {}
        self.rounds: list[dict[str, object]] = []
        self.progress_updates: list[dict[str, object]] = []
        self.commit_count = 0

    def save_batch_document_chunks(
        self,
        job_id: str,
        chunks: list[dict[str, object]],
        replace_existing: bool = True,
    ) -> None:
        """Persist embedded chunks for assertions.

        Inputs:
            job_id: Batch job identifier.
            chunks: Embedded chunk dictionaries.
            replace_existing: Whether to replace previous job chunks.

        Outputs:
            None. Chunks are recorded.
        """
        _ = job_id
        if replace_existing:
            self.saved_chunks = []
        self.saved_chunks.extend(chunks)

    def search_batch_document_chunks(
        self,
        job_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return saved chunks as vector search candidates.

        Inputs:
            job_id: Batch job identifier.
            query_embedding: Query vector.
            top_k: Maximum candidate count.

        Outputs:
            list[dict[str, object]]: Retrieved chunk rows.
        """
        _ = job_id, query_embedding
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "attachment_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "document_type": chunk["document_type"],
                "chunk_text": chunk["chunk_text"],
                "location_json": chunk["location_json"],
                "similarity_score": 0.99,
            }
            for chunk in self.saved_chunks[:top_k]
        ]

    def save_batch_retrieval_round(
        self,
        job_id: str,
        task_id: str,
        round_number: int,
        queries: list[str],
        retrieved_chunk_ids: list[str],
        accepted_evidence_json: dict[str, object],
        evidence_gaps: list[str],
        refined_queries: list[str],
        stop_reason: str | None,
    ) -> str:
        """Record one retrieval round.

        Inputs:
            job_id: Batch job identifier.
            task_id: Question task identifier.
            round_number: One-based round number.
            queries: Retrieval queries.
            retrieved_chunk_ids: Retrieved chunk IDs.
            accepted_evidence_json: Accepted evidence metadata.
            evidence_gaps: Remaining evidence gaps.
            refined_queries: Follow-up queries.
            stop_reason: Loop stop reason.

        Outputs:
            str: Retrieval round identifier.
        """
        self.rounds.append(
            {
                "job_id": job_id,
                "task_id": task_id,
                "round_number": round_number,
                "queries": queries,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "accepted_evidence_json": accepted_evidence_json,
                "evidence_gaps": evidence_gaps,
                "refined_queries": refined_queries,
                "stop_reason": stop_reason,
            }
        )
        return f"round-{len(self.rounds)}"

    def save_batch_question_result(
        self,
        task_id: str,
        result: QuestionReviewResult,
    ) -> None:
        """Persist one batch question result for assertions."""
        self.results[task_id] = result

    def save_batch_rag_tool_call(
        self,
        job_id: str,
        task_id: str,
        rag_call_number: int,
        queries_json: list[dict[str, object]],
        retrieved_chunk_ids: list[str],
        matched_queries_json: dict[str, object],
        query_count: int,
        retrieved_chunk_count: int,
        duration_ms: int,
        stop_reason: str,
    ) -> str:
        """Record one RAG tool call for assertions."""
        self.rounds.append(
            {
                "job_id": job_id,
                "task_id": task_id,
                "rag_call_number": rag_call_number,
                "query_count": query_count,
                "retrieved_chunk_count": retrieved_chunk_count,
            }
        )
        return f"batch-rag-tool-call-{len(self.rounds)}"

    def update_batch_question_progress(
        self,
        task_id: str,
        worker_id: str,
        status: str,
        progress_message: str | None = None,
        rag_call_count: int | None = None,
        query_count: int | None = None,
        retrieved_chunk_count: int | None = None,
    ) -> None:
        """Record one question progress update for assertions."""
        self.progress_updates.append(
            {
                "task_id": task_id,
                "worker_id": worker_id,
                "status": status,
                "progress_message": progress_message,
                "rag_call_count": rag_call_count,
                "query_count": query_count,
                "retrieved_chunk_count": retrieved_chunk_count,
            }
        )

    def commit(self) -> None:
        """Record one explicit progress commit."""
        self.commit_count += 1


class FakeToolCallingLlm:
    """Fake tool-calling chat model for ReAct graph tests."""

    def __init__(
        self,
        final_question_id: str,
        final_status: str,
        requested_tool_calls: int = 1,
    ) -> None:
        """Store final response settings and initialize counters."""
        self.final_question_id = final_question_id
        self.final_status = final_status
        self.requested_tool_calls = requested_tool_calls
        self.tool_call_requests = 0

    def bind_tools(self, tools: list[object]) -> "FakeToolCallingLlm":
        """Return self while recording bound tools for interface compatibility."""
        self.tools = tools
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        """Return tool calls first, then a strict JSON final answer."""
        tool_messages = [
            message for message in messages if getattr(message, "type", "") == "tool"
        ]
        if len(tool_messages) < min(self.requested_tool_calls, 2):
            self.tool_call_requests += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_search",
                        "id": f"rag-call-{self.tool_call_requests}",
                        "args": {
                            "queries": [
                                {
                                    "query": "strategic plan formalized approved",
                                    "answer_item_id": "Q.STR.01.01-L2-001",
                                    "level": 2,
                                }
                            ],
                            "top_k_per_query": 8,
                        },
                    }
                ],
            )
        return AIMessage(
            content=json.dumps(
                {
                    "question_id": self.final_question_id,
                    "model": "gpt-5.4",
                    "overall_status": self.final_status,
                    "current_selected_answer_item_ids": [],
                    "verified_selected_answer_item_ids": [],
                }
            )
        )


class FakeNoToolLlm:
    """Fake chat model that never calls tools.

    Inputs:
        None. The fake always returns a plain assistant message.

    Outputs:
        A deterministic model double used to verify fallback retrieval.
    """

    def bind_tools(self, tools: list[object]) -> "FakeNoToolLlm":
        """Return self while recording bound tools for interface compatibility.

        Inputs:
            tools: Tool definitions supplied by the graph.

        Outputs:
            FakeNoToolLlm: The same fake instance.
        """
        self.tools = tools
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        """Return a non-tool assistant message.

        Inputs:
            messages: Conversation messages supplied by the graph.

        Outputs:
            AIMessage: Plain assistant message without tool calls.
        """
        _ = messages
        return AIMessage(content="I can finalize without tool calls.")


class FakeRepairLlm:
    """Fake chat model that returns invalid JSON once and repaired JSON once."""

    def __init__(self, invalid_content: str, repaired_content: str) -> None:
        """Store invalid and repaired content for deterministic repair tests."""
        self.invalid_content = invalid_content
        self.repaired_content = repaired_content
        self.final_calls = 0
        self.repair_calls = 0

    def bind_tools(self, tools: list[object]) -> "FakeRepairLlm":
        """Return self for tool-binding interface compatibility."""
        self.tools = tools
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        """Return invalid content for graph final answer, then repaired content."""
        joined = "\n".join(str(getattr(message, "content", "")) for message in messages)
        if "Repair this response" in joined:
            self.repair_calls += 1
            return AIMessage(content=self.repaired_content)
        self.final_calls += 1
        return AIMessage(content=self.invalid_content)


def _question(question_id: str, text: str) -> AssessmentQuestion:
    """Create a minimal question fixture.

    Inputs:
        question_id: Question identifier.
        text: Question text.

    Outputs:
        AssessmentQuestion: Valid question with one answer item.
    """
    return AssessmentQuestion(
        question_id=question_id,
        dimension="Strategy",
        section="Governance",
        question=text,
        explanation="Review evidence.",
        answer_items=[
            AnswerItem(
                answer_item_id=f"{question_id}-L1-001",
                question_id=question_id,
                level=1,
                item_index=1,
                text="Evidence exists.",
            )
        ],
    )


def test_batch_review_graph_indexes_once_and_fans_out_questions() -> None:
    """Verify batch graph indexes shared documents once and reviews each question.

    Inputs:
        None. The test uses fakes for embedding, retrieval persistence, and model
        review calls.

    Outputs:
        None. Assertions confirm one indexing pass and two persisted question
        results.
    """
    repository = FakeBatchRepository()
    embedding_client = FakeEmbeddingClient()
    review_client = FakeReviewClient()

    result = run_batch_review_graph(
        BatchReviewInput(
            job_id="batch-job-1",
            language="en",
            questions=[
                _question("Q.STR.01.01", "Is strategy documented?"),
                _question("Q.RCG.01.01", "Are controls defined?"),
            ],
            current_answers={"Q.STR.01.01": []},
            documents=[
                ExtractedDocument(
                    file_name="strategy.pdf",
                    document_type="pdf",
                    metadata={"document_id": "document-1"},
                    blocks=[
                        ExtractedBlock(
                            block_id="pdf-page-0001",
                            block_type="page",
                            text="The strategy and controls are documented.",
                            page=1,
                        )
                    ],
                )
            ],
            task_ids_by_question={
                "Q.STR.01.01": "batch-task-1",
                "Q.RCG.01.01": "batch-task-2",
            },
        ),
        repository=repository,
        embedding_client=embedding_client,
        review_client=review_client,
        max_concurrency=2,
    )

    assert len(repository.saved_chunks) == 1
    assert len(embedding_client.batch_texts) == 1
    assert sorted(review_client.reviewed_questions) == ["Q.RCG.01.01", "Q.STR.01.01"]
    assert sorted(repository.results) == ["batch-task-1", "batch-task-2"]
    assert result.completed_count == 2


def test_batch_review_graph_indexes_excel_raw_and_summary_chunks() -> None:
    """Verify Excel batch documents produce raw chunks plus summary chunks.

    Inputs:
        None. The test supplies one extracted workbook with a fake summary
        method on the review client.

    Outputs:
        None. Assertions confirm both traceable raw sheet content and a summary
        retrieval chunk are embedded and persisted.
    """

    class SummaryReviewClient(FakeReviewClient):
        """Fake review client that provides an Excel summary hook."""

        def summarize_excel_document(self, document, language: str = "en") -> str:
            """Return a deterministic workbook summary.

            Inputs:
                document: Extracted workbook document.
                language: Requested language.

            Outputs:
                str: Summary text used as a retrieval chunk.
            """
            _ = language
            return f"Summary of {document.file_name}: objectives are tracked."

    repository = FakeBatchRepository()
    embedding_client = FakeEmbeddingClient()
    review_client = SummaryReviewClient()

    run_batch_review_graph(
        BatchReviewInput(
            job_id="batch-job-2",
            language="en",
            questions=[_question("Q.STR.01.01", "Are objectives tracked?")],
            current_answers={"Q.STR.01.01": []},
            documents=[
                ExtractedDocument(
                    file_name="objectives.xlsx",
                    document_type="xlsx",
                    metadata={"document_id": "document-xlsx"},
                    blocks=[
                        ExtractedBlock(
                            block_id="xlsx-block-0001",
                            block_type="sheet",
                            text="A1=Objective | B1=Owner\nA2=Reduce emissions | B2=HSE",
                            sheet_name="Objectives",
                            table_name="Objectives!A1:B2",
                            row_start=1,
                            row_end=2,
                        )
                    ],
                )
            ],
            task_ids_by_question={"Q.STR.01.01": "batch-task-xlsx"},
        ),
        repository=repository,
        embedding_client=embedding_client,
        review_client=review_client,
        max_concurrency=1,
    )

    assert [chunk["chunk_kind"] for chunk in repository.saved_chunks] == [
        "raw",
        "summary",
    ]
    assert len(embedding_client.batch_texts) == 2
    assert "Objectives!A1:B2" in repository.saved_chunks[0]["location_json"]["tables"][0]
    assert "objectives are tracked" in repository.saved_chunks[1]["chunk_text"]


def test_react_question_graph_calls_rag_tool_and_returns_result() -> None:
    """Verify one question graph invocation retrieves evidence and returns JSON."""
    repository = FakeBatchRepository()
    repository.saved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "chunk_text": "The strategic plan is formalized and approved.",
            "location_json": {"pages": [2]},
            "similarity_score": 0.91,
        }
    ]
    embedding_client = FakeEmbeddingClient()
    llm = FakeToolCallingLlm(
        final_question_id="Q.STR.01.01",
        final_status="supported",
    )

    result = run_batch_question_react_review(
        question=_question("Q.STR.01.01", "Is strategy documented?"),
        task={
            "task_id": "batch-task-1",
            "job_id": "batch-job-1",
            "question_id": "Q.STR.01.01",
            "current_selected_answer_item_ids": [],
        },
        repository=repository,
        embedding_client=embedding_client,
        llm=llm,
        review_client=FakeReviewClient(),
        worker_id="worker-1",
    )

    assert result.question_id == "Q.STR.01.01"
    assert result.overall_status == "supported"


def test_react_graph_runs_final_structured_review_node() -> None:
    """Verify the reusable ReAct graph owns final structured review execution.

    Inputs:
        None. The test builds the reusable graph with fake retrieval and final
        review callbacks.

    Outputs:
        None. Assertions confirm graph invocation returns ``review_result`` in
        state after the retrieval loop stops calling tools.
    """

    question = _question("Q.STR.01.01", "Is strategy documented?")
    task = {
        "task_id": "task-graph-final",
        "question_id": "Q.STR.01.01",
        "dimension": "Strategy",
        "current_selected_answer_item_ids": [],
    }
    retrieved = [
        RagRetrievedChunk(
            chunk_id="chunk-1",
            file_name="strategy.pdf",
            document_type="pdf",
            chunk_text="The strategic plan is formalized and approved.",
            location_json={"pages": [2]},
            similarity_score=0.91,
            matched_queries=[
                RagQuery(
                    query="strategic plan formalized approved",
                    answer_item_id="Q.STR.01.01-L2-001",
                    level=2,
                )
            ],
        )
    ]
    saved_tool_calls: list[dict[str, object]] = []
    final_review_calls: list[list[RagRetrievedChunk]] = []

    def retrieve_candidates(
        queries: list[RagQuery],
        top_k_per_query: int,
    ) -> list[RagRetrievedChunk]:
        """Return one retrieved chunk after validating the tool query."""

        assert queries[0].query == "strategic plan formalized approved"
        assert top_k_per_query == 8
        return retrieved

    def save_tool_call(
        rag_call_number: int,
        queries: list[RagQuery],
        chunks: list[RagRetrievedChunk],
        duration_ms: int,
    ) -> None:
        """Record one retrieval audit callback."""

        saved_tool_calls.append(
            {
                "rag_call_number": rag_call_number,
                "query_count": len(queries),
                "chunk_count": len(chunks),
                "duration_ms": duration_ms,
            }
        )

    def final_review(chunks: list[RagRetrievedChunk]) -> QuestionReviewResult:
        """Return a final structured review result from graph state chunks."""

        final_review_calls.append(chunks)
        return QuestionReviewResult(
            question_id=question.question_id,
            model="fake-gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        )

    graph = build_question_react_graph(
        question=question,
        task=task,
        worker_id="worker-graph",
        llm=FakeToolCallingLlm(
            final_question_id="Q.STR.01.01",
            final_status="supported",
        ),
        retrieve_candidates=retrieve_candidates,
        save_tool_call=save_tool_call,
        review_question=final_review,
    )

    state = graph.invoke({"messages": [HumanMessage(content="Review the question.")]})

    assert state["review_result"].overall_status == "supported"
    assert saved_tool_calls[0]["chunk_count"] == 1
    assert final_review_calls == [retrieved]


def test_react_graph_falls_back_when_llm_skips_rag_tool() -> None:
    """Verify final review cannot run before at least one retrieval attempt.

    Inputs:
        None. The graph is invoked with an LLM fake that never calls tools.

    Outputs:
        None. Assertions confirm fallback RAG retrieves and audits chunks.
    """

    question = _question("Q.STR.03.01", "Are objectives monitored?")
    task = {
        "task_id": "task-graph-fallback",
        "question_id": "Q.STR.03.01",
        "dimension": "Strategy",
        "current_selected_answer_item_ids": [],
    }
    retrieved = [
        RagRetrievedChunk(
            chunk_id="chunk-objectives",
            file_name="objectives.pdf",
            document_type="pdf",
            chunk_text="Annual objectives are monitored quarterly by management.",
            location_json={"pages": [4]},
            similarity_score=0.93,
            matched_queries=[],
        )
    ]
    saved_tool_calls: list[dict[str, object]] = []
    progress_updates: list[dict[str, object]] = []
    final_review_calls: list[list[RagRetrievedChunk]] = []

    def retrieve_candidates(
        queries: list[RagQuery],
        top_k_per_query: int,
    ) -> list[RagRetrievedChunk]:
        """Return one retrieved chunk after validating fallback queries.

        Inputs:
            queries: Fallback query metadata generated from the question.
            top_k_per_query: Per-query candidate limit.

        Outputs:
            list[RagRetrievedChunk]: Deterministic retrieved evidence.
        """
        assert top_k_per_query == 8
        assert queries[0].query == "Are objectives monitored? Review evidence."
        return retrieved

    def save_tool_call(
        rag_call_number: int,
        queries: list[RagQuery],
        chunks: list[RagRetrievedChunk],
        duration_ms: int,
    ) -> None:
        """Record the fallback retrieval audit callback."""

        saved_tool_calls.append(
            {
                "rag_call_number": rag_call_number,
                "query_count": len(queries),
                "chunk_count": len(chunks),
                "duration_ms": duration_ms,
            }
        )

    def update_progress(
        status: str,
        progress_message: str,
        rag_call_count: int | None,
        query_count: int | None,
        retrieved_chunk_count: int | None,
    ) -> None:
        """Record graph progress updates for assertions."""

        progress_updates.append(
            {
                "status": status,
                "progress_message": progress_message,
                "rag_call_count": rag_call_count,
                "query_count": query_count,
                "retrieved_chunk_count": retrieved_chunk_count,
            }
        )

    def final_review(chunks: list[RagRetrievedChunk]) -> QuestionReviewResult:
        """Return a final structured review result from fallback chunks."""

        final_review_calls.append(chunks)
        return QuestionReviewResult(
            question_id=question.question_id,
            model="fake-gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        )

    graph = build_question_react_graph(
        question=question,
        task=task,
        worker_id="worker-graph",
        llm=FakeNoToolLlm(),
        retrieve_candidates=retrieve_candidates,
        save_tool_call=save_tool_call,
        update_progress=update_progress,
        review_question=final_review,
    )

    state = graph.invoke({"messages": [HumanMessage(content="Review the question.")]})

    assert state["review_result"].overall_status == "supported"
    assert saved_tool_calls[0]["rag_call_number"] == 1
    assert saved_tool_calls[0]["chunk_count"] == 1
    assert progress_updates[-1]["retrieved_chunk_count"] == 1
    assert final_review_calls == [retrieved]


def test_react_question_graph_publishes_progress_before_first_llm_call() -> None:
    """Verify polling sees question work before the first model call returns.

    Inputs:
        None. The test runs the ReAct graph with a fake LLM and captures
        repository progress updates.

    Outputs:
        None. Assertions confirm the first progress update marks the question
        in progress and is committed before tool-specific progress events.
    """
    repository = FakeBatchRepository()
    repository.saved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "chunk_text": "The strategic plan is formalized and approved.",
            "location_json": {"pages": [2]},
            "similarity_score": 0.91,
        }
    ]
    llm = FakeToolCallingLlm(
        final_question_id="Q.STR.01.01",
        final_status="supported",
    )

    run_batch_question_react_review(
        question=_question("Q.STR.01.01", "Is strategy documented?"),
        task={
            "task_id": "batch-task-1",
            "job_id": "batch-job-1",
            "question_id": "Q.STR.01.01",
            "current_selected_answer_item_ids": [],
        },
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        llm=llm,
        review_client=FakeReviewClient(),
        worker_id="worker-1",
    )

    assert repository.progress_updates[0]["status"] == "in_progress"
    assert "Q.STR.01.01" in str(repository.progress_updates[0]["progress_message"])
    assert repository.commit_count >= 1


def test_react_question_graph_allows_at_most_two_rag_calls() -> None:
    """Verify the graph forces a final answer after two RAG tool calls."""
    repository = FakeBatchRepository()
    repository.saved_chunks = [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "file_name": "strategy.pdf",
            "document_type": "pdf",
            "chunk_text": "The strategic plan is formalized.",
            "location_json": {"pages": [1]},
            "similarity_score": 0.9,
        }
    ]
    llm = FakeToolCallingLlm(
        final_question_id="Q.STR.01.01",
        final_status="insufficient_evidence",
        requested_tool_calls=3,
    )

    result = run_batch_question_react_review(
        question=_question("Q.STR.01.01", "Is strategy documented?"),
        task={
            "task_id": "batch-task-1",
            "job_id": "batch-job-1",
            "question_id": "Q.STR.01.01",
            "current_selected_answer_item_ids": [],
        },
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        llm=llm,
        review_client=FakeReviewClient(overall_status="insufficient_evidence"),
        worker_id="worker-1",
    )

    assert result.overall_status == "insufficient_evidence"
    assert llm.tool_call_requests == 2


def test_react_question_graph_uses_structured_review_when_final_chat_is_invalid() -> None:
    """Verify invalid final chat content is not parsed as the final result."""
    llm = FakeRepairLlm(
        invalid_content='{"question_id":"Q.STR.01.01","answers":[]}',
        repaired_content="should not be used",
    )
    review_client = FakeReviewClient()

    result = run_batch_question_react_review(
        question=_question("Q.STR.01.01", "Is strategy documented?"),
        task={
            "task_id": "batch-task-1",
            "job_id": "batch-job-1",
            "question_id": "Q.STR.01.01",
            "current_selected_answer_item_ids": [],
        },
        repository=FakeBatchRepository(),
        embedding_client=FakeEmbeddingClient(),
        llm=llm,
        review_client=review_client,
        worker_id="worker-1",
    )

    assert result.overall_status == "supported"
    assert llm.repair_calls == 0
    assert review_client.reviewed_questions == ["Q.STR.01.01"]


class FakeBatchRepositoryWithDocuments(FakeBatchRepository):
    """Fake repository with one uploaded batch document for indexing tests."""

    def __init__(self) -> None:
        """Initialize base stores and one document row."""
        super().__init__()
        self.extractions: list[dict[str, object]] = []
        self.indexing_progress: list[dict[str, object]] = []

    def get_batch_job_documents(self, job_id: str) -> list[dict[str, object]]:
        """Return one deterministic uploaded document."""
        assert job_id == "batch-job-1"
        return [
            {
                "document_id": "document-1",
                "file_name": "strategy.pdf",
                "content": b"%PDF deterministic fixture",
            },
            {
                "document_id": "document-2",
                "file_name": "controls.pdf",
                "content": b"%PDF deterministic fixture 2",
            },
        ]

    def update_batch_indexing_progress(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        indexed_chunk_count: int | None = None,
    ) -> None:
        """Record indexing progress for assertions."""
        assert job_id == "batch-job-1"
        assert worker_id == "worker-index"
        self.indexing_progress.append(
            {
                "status": status,
                "indexed_chunk_count": indexed_chunk_count,
            }
        )

    def save_batch_document_extraction(
        self,
        job_id: str,
        document_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Record extraction rows for assertions."""
        self.extractions.append(
            {
                "job_id": job_id,
                "document_id": document_id,
                "file_name": extracted.file_name,
                "estimated_tokens": estimated_tokens,
                "warnings": warnings,
            }
        )
        return "batch-extraction-1"


def test_run_batch_indexing_from_hana_extracts_and_indexes_without_reviewing_questions(monkeypatch) -> None:
    """Verify batch indexing stores chunks but does not run question review."""
    repository = FakeBatchRepositoryWithDocuments()
    embedding_client = FakeEmbeddingClient()
    review_client = FakeReviewClient()
    monkeypatch.setattr(
        "app.services.batch_review_graph.extract_evidence_files",
        lambda paths: [
            ExtractedDocument(
                file_name=str(paths[0].name),
                document_type="pdf",
                metadata={},
                blocks=[
                    ExtractedBlock(
                        block_id="pdf-page-0001",
                        block_type="page",
                        text=f"{paths[0].name} evidence is formalized.",
                        page=1,
                    )
                ],
            )
        ],
    )

    result = run_batch_indexing_from_hana(
        repository=repository,
        job={"job_id": "batch-job-1", "language": "en", "lease_owner": "worker-index"},
        review_client=review_client,
        embedding_client=embedding_client,
    )

    assert result.job_id == "batch-job-1"
    assert result.indexed_chunk_count >= 2
    assert repository.results == {}
    embedding_progress = [
        row
        for row in repository.indexing_progress
        if row["status"] == "embedding_documents"
    ]
    assert [row["indexed_chunk_count"] for row in embedding_progress] == [0, 1, 2]
    assert {chunk["document_id"] for chunk in repository.saved_chunks} == {
        "document-1",
        "document-2",
    }


def test_run_document_ingestion_from_hana_indexes_eml_documents() -> None:
    """Verify Document Manager ingestion extracts, chunks, and embeds EML files.

    Inputs:
        None. The test creates an in-memory document ingestion job.

    Outputs:
        None. Assertions confirm persisted extraction and chunk rows.
    """

    repository = InMemoryAiReviewRepository()
    email_bytes = (
        b"Subject: Governance evidence\r\n"
        b"From: chair@example.com\r\n"
        b"To: audit@example.com\r\n"
        b"\r\n"
        b"Board minutes confirm annual oversight of strategy and risk."
    )
    job = repository.create_document_ingestion_job(
        assessment_id="assessment-1",
        documents=[
            {
                "file_name": "governance.eml",
                "content_type": "message/rfc822",
                "content": email_bytes,
            }
        ],
    )
    leased_job = repository.lease_next_document_ingestion_job("worker-eml")

    result = run_document_ingestion_from_hana(
        repository=repository,
        job=leased_job,
        embedding_client=FakeEmbeddingClient(),
    )

    document_id = repository.document_ingestion_job_documents[job.job_id][0]
    assert result.indexed_chunk_count == 1
    assert repository.document_extractions[document_id][0]["document_type"] == "eml"
    assert repository.document_chunks["assessment-1"][0]["document_type"] == "eml"
    assert (
        "annual oversight"
        in repository.document_chunks["assessment-1"][0]["chunk_text"]
    )
