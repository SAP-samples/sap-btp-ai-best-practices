"""LangGraph fanout workflow for global batch upload AI review jobs."""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AssessmentQuestion
from app.models.documents import DocumentIngestionJobResponse
from app.models.language import DEFAULT_LANGUAGE
from app.services.ai_review_graph import (
    QuestionReviewInput,
    _build_rag_final_prompt,
    _client_model_name,
    _normalize_question_review_result,
    _rag_insufficient_result,
    extract_evidence_files,
)
from app.services.document_extractors import ExtractedBlock, ExtractedDocument
from app.services.evidence_indexing import (
    EvidenceChunk,
    chunk_extracted_documents,
    content_hash,
    embed_evidence_chunks,
)
from app.services.evidence_rag import run_retrieval_assessment_loop
from app.services.evidence_retrieval import (
    build_initial_retrieval_queries,
    retrieve_evidence_candidates,
)
from app.services.evidence_routing import estimate_text_tokens
from app.services.genai_responses import GenAiReviewClient
from app.services.joule_knowledge_importer import (
    DEFAULT_EMBEDDING_MODEL,
    GenAiHubEmbeddingClient,
)


DEFAULT_BATCH_MAX_CONCURRENCY = 4
"""Default number of question review branches allowed to run concurrently."""


def _commit_if_supported(repository: Any) -> None:
    """Commit repository progress when supported by the repository boundary.

    Inputs:
        repository: Repository object passed into batch graph helpers.

    Outputs:
        None. HANA-backed repositories commit progress; in-memory test seams may
        record or ignore the call.
    """

    commit = getattr(repository, "commit", None)
    if callable(commit):
        commit()


class BatchReviewInput(BaseModel):
    """Input required to run one global batch review graph.

    Inputs:
        Field values describing the batch job, localized questions, extracted
        documents, and task IDs.

    Outputs:
        A validated input object used by the batch graph.
    """

    job_id: str
    language: str = DEFAULT_LANGUAGE
    questions: list[AssessmentQuestion] = Field(default_factory=list)
    current_answers: dict[str, list[str]] = Field(default_factory=dict)
    documents: list[ExtractedDocument] = Field(default_factory=list)
    task_ids_by_question: dict[str, str] = Field(default_factory=dict)


class BatchReviewGraphResult(BaseModel):
    """Result returned after running the batch review graph.

    Inputs:
        Field values describing the completed batch graph run.

    Outputs:
        A validated result with a completed question count.
    """

    job_id: str
    completed_count: int


class BatchIndexingResult(BaseModel):
    """Result returned after shared batch document indexing completes."""

    job_id: str
    indexed_chunk_count: int


class _BatchGraphState(TypedDict, total=False):
    """LangGraph state for batch indexing and question fanout."""

    review_input: BatchReviewInput
    indexed_chunk_count: int
    completed_questions: Annotated[list[str], operator.add]


class _BatchQuestionState(TypedDict, total=False):
    """Per-question state sent into a fanout branch."""

    review_input: BatchReviewInput
    question: AssessmentQuestion


class _BatchRetrievalRepository:
    """Adapter exposing task-shaped retrieval methods over a batch corpus.

    Inputs:
        repository: Batch repository exposing search and round persistence.
        job_id: Batch job identifier for the shared document corpus.

    Outputs:
        Adapter object accepted by existing RAG retrieval helpers.
    """

    def __init__(self, repository: Any, job_id: str) -> None:
        """Initialize the batch retrieval adapter.

        Inputs:
            repository: Repository with batch search/persistence methods.
            job_id: Batch job identifier.

        Outputs:
            None. Adapter stores repository and job ID for later calls.
        """

        self.repository = repository
        self.job_id = job_id

    def search_evidence_chunks(
        self,
        task_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search batch document chunks for a question task.

        Inputs:
            task_id: Batch question task identifier, accepted for interface
                compatibility.
            query_embedding: Query embedding vector.
            top_k: Maximum number of chunks to retrieve.

        Outputs:
            list[dict[str, object]]: Retrieved chunk rows.
        """

        _ = task_id
        return self.repository.search_batch_document_chunks(
            job_id=self.job_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def save_retrieval_round(
        self,
        task_id: str,
        round_number: int,
        queries: list[str],
        retrieved_chunk_ids: list[str],
        accepted_evidence_json: dict[str, Any],
        evidence_gaps: list[str],
        refined_queries: list[str],
        stop_reason: str | None,
    ) -> str:
        """Persist one batch retrieval round through the underlying repository.

        Inputs:
            task_id: Batch question task identifier.
            round_number: One-based retrieval round number.
            queries: Queries used during this round.
            retrieved_chunk_ids: Retrieved chunk identifiers.
            accepted_evidence_json: Accepted/rejected evidence metadata.
            evidence_gaps: Remaining evidence gaps.
            refined_queries: Follow-up queries.
            stop_reason: Loop stop reason.

        Outputs:
            str: Retrieval round identifier returned by the repository.
        """

        return self.repository.save_batch_retrieval_round(
            job_id=self.job_id,
            task_id=task_id,
            round_number=round_number,
            queries=queries,
            retrieved_chunk_ids=retrieved_chunk_ids,
            accepted_evidence_json=accepted_evidence_json,
            evidence_gaps=evidence_gaps,
            refined_queries=refined_queries,
            stop_reason=stop_reason,
        )


def _document_id(document: ExtractedDocument) -> str:
    """Return the batch document ID carried in extraction metadata.

    Inputs:
        document: Extracted batch document.

    Outputs:
        str: Stable document ID from metadata, falling back to filename.
    """

    return str(document.metadata.get("document_id") or document.file_name)


def _chunk_document_for_batch(
    job_id: str,
    document: ExtractedDocument,
) -> list[EvidenceChunk]:
    """Chunk one extracted document using the batch document ID as source ID.

    Inputs:
        job_id: Batch job identifier.
        document: Extracted document to chunk.

    Outputs:
        list[EvidenceChunk]: Chunks ready for embedding.
    """

    return chunk_extracted_documents(
        task_id=job_id,
        attachment_id_by_file={document.file_name: _document_id(document)},
        extracted_documents=[document],
    )


def _summary_chunk_id(job_id: str, document_id: str, summary_text: str) -> str:
    """Build a stable chunk ID for an Excel summary chunk.

    Inputs:
        job_id: Batch job identifier.
        document_id: Source batch document identifier.
        summary_text: Summary text included in the chunk.

    Outputs:
        str: Stable summary chunk identifier.
    """

    digest = hashlib.sha256(
        "\n".join([job_id, document_id, content_hash(summary_text)]).encode("utf-8")
    ).hexdigest()
    return f"chunk-{digest[:32]}"


def _fallback_excel_summary(document: ExtractedDocument) -> str:
    """Build a deterministic fallback summary for tests and offline runs.

    Inputs:
        document: Extracted spreadsheet document.

    Outputs:
        str: Compact workbook summary text.
    """

    sheet_names = [
        block.sheet_name for block in document.blocks if block.sheet_name is not None
    ]
    unique_sheets = list(dict.fromkeys(sheet_names))
    sample_lines = []
    for block in document.blocks[:3]:
        first_line = block.text.splitlines()[0] if block.text else ""
        if first_line:
            sample_lines.append(first_line)
    return "\n".join(
        [
            f"Workbook summary for {document.file_name}.",
            f"Sheets: {', '.join(unique_sheets) if unique_sheets else 'unknown'}.",
            "Representative content:",
            *(sample_lines or ["No non-empty cells extracted."]),
        ]
    )


def _excel_summary_text(
    document: ExtractedDocument,
    review_client: Any,
    language: str,
) -> str:
    """Return an LLM spreadsheet summary when the client supports it.

    Inputs:
        document: Extracted spreadsheet document.
        review_client: Review client or fake client.
        language: Requested output language.

    Outputs:
        str: Summary text suitable for an additional retrieval chunk.
    """

    summarizer = getattr(review_client, "summarize_excel_document", None)
    if callable(summarizer):
        return str(summarizer(document=document, language=language)).strip()
    return _fallback_excel_summary(document)


def _excel_summary_chunks(
    job_id: str,
    documents: list[ExtractedDocument],
    review_client: Any,
    language: str,
) -> list[EvidenceChunk]:
    """Create one summary chunk for each extracted Excel workbook.

    Inputs:
        job_id: Batch job identifier.
        documents: Extracted batch documents.
        review_client: Review client used for LLM summaries.
        language: Requested summary language.

    Outputs:
        list[EvidenceChunk]: Summary chunks without embeddings.
    """

    summary_chunks: list[EvidenceChunk] = []
    for document in documents:
        if document.document_type not in {"xlsx", "xlsm"}:
            continue
        summary_text = _excel_summary_text(
            document=document,
            review_client=review_client,
            language=language,
        )
        if not summary_text:
            continue
        document_id = _document_id(document)
        source_block_ids = [block.block_id for block in document.blocks]
        summary_chunks.append(
            EvidenceChunk(
                chunk_id=_summary_chunk_id(job_id, document_id, summary_text),
                task_id=job_id,
                attachment_id=document_id,
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


def _batch_chunk_dict(chunk: EvidenceChunk, chunk_kind: str) -> dict[str, Any]:
    """Convert an evidence chunk into the batch persistence shape.

    Inputs:
        chunk: Embedded evidence chunk.
        chunk_kind: ``raw`` or ``summary``.

    Outputs:
        dict[str, Any]: Chunk dictionary with ``document_id`` and ``chunk_kind``.
    """

    payload = chunk.model_dump(mode="json")
    payload["document_id"] = chunk.attachment_id
    payload["chunk_kind"] = chunk_kind
    return payload


def _index_batch_documents(
    review_input: BatchReviewInput,
    repository: Any,
    embedding_client: Any,
    review_client: Any,
    worker_id: str | None = None,
) -> int:
    """Chunk, summarize, embed, and persist shared batch document chunks.

    Inputs:
        review_input: Batch graph input containing extracted documents.
        repository: Repository exposing ``save_batch_document_chunks``.
        embedding_client: Embedding client exposing ``embed_batch``.
        review_client: Review client used for Excel summaries.
        worker_id: Optional batch indexing lease owner. When supplied, indexing
            progress is persisted after each document.

    Outputs:
        int: Number of persisted batch chunks.
    """

    embedding_model = (
        getattr(embedding_client, "model_name", None) or DEFAULT_EMBEDDING_MODEL
    )

    repository.save_batch_document_chunks(
        job_id=review_input.job_id,
        chunks=[],
        replace_existing=True,
    )
    _commit_if_supported(repository)

    indexed_count = 0
    for document in review_input.documents:
        raw_chunks = _chunk_document_for_batch(
            job_id=review_input.job_id,
            document=document,
        )
        summary_chunks = _excel_summary_chunks(
            job_id=review_input.job_id,
            documents=[document],
            review_client=review_client,
            language=review_input.language,
        )
        embedded_raw_chunks = embed_evidence_chunks(
            chunks=raw_chunks,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
        )
        embedded_summary_chunks = embed_evidence_chunks(
            chunks=summary_chunks,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
        )
        document_chunks = [
            *[_batch_chunk_dict(chunk, "raw") for chunk in embedded_raw_chunks],
            *[
                _batch_chunk_dict(chunk, "summary")
                for chunk in embedded_summary_chunks
            ],
        ]
        repository.save_batch_document_chunks(
            job_id=review_input.job_id,
            chunks=document_chunks,
            replace_existing=False,
        )
        indexed_count += len(document_chunks)
        if worker_id:
            repository.update_batch_indexing_progress(
                job_id=review_input.job_id,
                worker_id=worker_id,
                status="embedding_documents",
                indexed_chunk_count=indexed_count,
            )
        _commit_if_supported(repository)

    return indexed_count


def run_batch_indexing_graph(
    indexing_input: BatchReviewInput,
    repository: Any,
    embedding_client: Any | None = None,
    review_client: Any | None = None,
    worker_id: str | None = None,
) -> BatchIndexingResult:
    """Index shared batch documents once without reviewing questions.

    Inputs:
        indexing_input: Batch input containing extracted documents.
        repository: Repository exposing ``save_batch_document_chunks``.
        embedding_client: Optional embedding client.
        review_client: Optional review client for Excel summaries.
        worker_id: Optional batch indexing lease owner for progress updates.

    Outputs:
        BatchIndexingResult: Indexed chunk count for the batch job.
    """

    embedder = embedding_client or GenAiHubEmbeddingClient(DEFAULT_EMBEDDING_MODEL)
    client = review_client or GenAiReviewClient()
    indexed_count = _index_batch_documents(
        review_input=indexing_input,
        repository=repository,
        embedding_client=embedder,
        review_client=client,
        worker_id=worker_id,
    )
    return BatchIndexingResult(
        job_id=indexing_input.job_id,
        indexed_chunk_count=indexed_count,
    )


def _review_batch_question(
    review_input: BatchReviewInput,
    question: AssessmentQuestion,
    repository: Any,
    embedding_client: Any,
    review_client: Any,
) -> QuestionReviewResult:
    """Run RAG review for one question against the shared batch corpus.

    Inputs:
        review_input: Batch graph input.
        question: Framework question reviewed by this branch.
        repository: Repository exposing batch result and retrieval methods.
        embedding_client: Embedding client for retrieval queries.
        review_client: Review client for RAG assessment and final review.

    Outputs:
        QuestionReviewResult: Normalized question review result.
    """

    task_id = review_input.task_ids_by_question[question.question_id]
    question_input = QuestionReviewInput(
        task_id=task_id,
        language=review_input.language,
        question=question,
        current_selected_answer_item_ids=review_input.current_answers.get(
            question.question_id,
            [],
        ),
        attachment_paths=[],
    )
    retrieval_repository = _BatchRetrievalRepository(
        repository=repository,
        job_id=review_input.job_id,
    )
    rag_result = run_retrieval_assessment_loop(
        task_id=task_id,
        question=question,
        language=review_input.language,
        initial_queries=build_initial_retrieval_queries(question),
        retrieve_candidates=lambda queries: retrieve_evidence_candidates(
            repository=retrieval_repository,
            embedding_client=embedding_client,
            task_id=task_id,
            queries=queries,
        ),
        review_client=review_client,
        repository=retrieval_repository,
    )
    if not rag_result.sufficient_for_final_review:
        return _rag_insufficient_result(
            review_input=question_input,
            model=_client_model_name(review_client),
            evidence_gaps=rag_result.evidence_gaps,
            stop_reason=rag_result.stop_reason,
        )
    result = review_client.review_question(
        _build_rag_final_prompt(
            review_input=question_input,
            accepted_chunks=rag_result.accepted_chunks,
            evidence_gaps=rag_result.evidence_gaps,
            stop_reason=rag_result.stop_reason,
        ),
        language=review_input.language,
    )
    return _normalize_question_review_result(question, result)


def build_batch_review_graph(
    repository: Any,
    embedding_client: Any,
    review_client: Any,
) -> Any:
    """Build and compile the batch indexing plus question fanout graph.

    Inputs:
        repository: Repository exposing batch persistence methods.
        embedding_client: Embedding client for chunk and query embeddings.
        review_client: Review client for Excel summaries and RAG review.

    Outputs:
        Compiled LangGraph application using ``Send`` for question fanout.
    """

    def index_documents(state: _BatchGraphState) -> dict[str, Any]:
        """Index shared batch documents once before question fanout.

        Inputs:
            state: Batch graph state containing ``review_input``.

        Outputs:
            dict[str, Any]: Indexed chunk count update.
        """

        return {
            "indexed_chunk_count": _index_batch_documents(
                review_input=state["review_input"],
                repository=repository,
                embedding_client=embedding_client,
                review_client=review_client,
            )
        }

    def fanout_questions(state: _BatchGraphState) -> list[Send]:
        """Create one parallel send per framework question.

        Inputs:
            state: Batch graph state containing ``review_input``.

        Outputs:
            list[Send]: Dynamic fanout instructions for LangGraph.
        """

        review_input = state["review_input"]
        return [
            Send(
                "review_question",
                {
                    "review_input": review_input,
                    "question": question,
                },
            )
            for question in review_input.questions
        ]

    def review_question(state: _BatchQuestionState) -> dict[str, Any]:
        """Review one fanned-out question and persist its result.

        Inputs:
            state: Per-question branch state.

        Outputs:
            dict[str, Any]: Completed question ID merged through a reducer.
        """

        review_input = state["review_input"]
        question = state["question"]
        result = _review_batch_question(
            review_input=review_input,
            question=question,
            repository=repository,
            embedding_client=embedding_client,
            review_client=review_client,
        )
        task_id = review_input.task_ids_by_question[question.question_id]
        repository.save_batch_question_result(task_id=task_id, result=result)
        return {"completed_questions": [question.question_id]}

    graph = StateGraph(_BatchGraphState)
    graph.add_node("index_documents", index_documents)
    graph.add_node("review_question", review_question)
    graph.add_edge(START, "index_documents")
    graph.add_conditional_edges("index_documents", fanout_questions)
    graph.add_edge("review_question", END)
    return graph.compile()


def run_batch_review_graph(
    review_input: BatchReviewInput,
    repository: Any,
    embedding_client: Any | None = None,
    review_client: Any | None = None,
    max_concurrency: int = DEFAULT_BATCH_MAX_CONCURRENCY,
) -> BatchReviewGraphResult:
    """Run a global batch review graph.

    Inputs:
        review_input: Batch job input with extracted documents and question
            tasks.
        repository: Repository exposing batch persistence methods.
        embedding_client: Optional embedding client. When omitted, SAP Gen AI
            Hub embeddings are used.
        review_client: Optional review client. When omitted, ``gpt-5.4`` review
            client is used.
        max_concurrency: Maximum number of parallel LangGraph branches.

    Outputs:
        BatchReviewGraphResult: Completed question count for the graph run.
    """

    embedder = embedding_client or GenAiHubEmbeddingClient(DEFAULT_EMBEDDING_MODEL)
    client = review_client or GenAiReviewClient()
    graph = build_batch_review_graph(
        repository=repository,
        embedding_client=embedder,
        review_client=client,
    )
    final_state = graph.invoke(
        {"review_input": review_input},
        config={"max_concurrency": max(1, int(max_concurrency))},
    )
    return BatchReviewGraphResult(
        job_id=review_input.job_id,
        completed_count=len(final_state.get("completed_questions", [])),
    )


def _document_bytes(document: dict[str, Any]) -> bytes:
    """Normalize repository document content to bytes.

    Inputs:
        document: Repository document row containing ``content``.

    Outputs:
        bytes: Binary document content.
    """

    content = document["content"]
    if isinstance(content, bytes):
        return content
    if isinstance(content, memoryview):
        return content.tobytes()
    return bytes(content)


def run_document_ingestion_from_hana(
    repository: Any,
    job: dict[str, Any],
    review_client: Any | None = None,
    embedding_client: Any | None = None,
) -> DocumentIngestionJobResponse:
    """Extract and index uploaded documents for the assessment corpus.

    Inputs:
        repository: Repository exposing document ingestion persistence methods.
        job: Leased ingestion job with assessment_id, lease_owner, and documents.
        review_client: Optional client reserved for future spreadsheet summaries.
        embedding_client: Optional embedding client override for tests.

    Outputs:
        DocumentIngestionJobResponse: Completed ingestion progress summary.
    """
    _ = review_client
    if embedding_client is None:
        embedding_client = GenAiHubEmbeddingClient()
    documents = list(job.get("documents", []))
    indexed_chunk_count = 0
    with TemporaryDirectory(prefix="document-assessment-manager-") as temporary_directory:
        for index, document in enumerate(documents, start=1):
            repository.update_document_ingestion_progress(
                job_id=job["job_id"],
                worker_id=job["lease_owner"],
                status="extracting_documents",
                processed_document_count=index - 1,
            )
            _commit_if_supported(repository)
            document_directory = Path(temporary_directory) / f"{index:03d}"
            document_directory.mkdir()
            file_name = Path(document["file_name"]).name
            path = document_directory / file_name
            path.write_bytes(_document_bytes(document))
            document_id = str(document["document_id"])
            extracted_documents: list[ExtractedDocument] = []
            for extracted_document in extract_evidence_files([path]):
                extracted_with_id = extracted_document.model_copy(
                    update={
                        "metadata": {
                            **extracted_document.metadata,
                            "document_id": document_id,
                        }
                    }
                )
                repository.save_document_extraction(
                    document_id=document_id,
                    extracted=extracted_with_id,
                    estimated_tokens=sum(
                        estimate_text_tokens(block.text)
                        for block in extracted_with_id.blocks
                    ),
                    warnings=list(extracted_with_id.metadata.get("warnings", [])),
                )
                extracted_documents.append(extracted_with_id)

            repository.update_document_ingestion_progress(
                job_id=job["job_id"],
                worker_id=job["lease_owner"],
                status="embedding_documents",
                processed_document_count=index,
                indexed_chunk_count=indexed_chunk_count,
            )
            _commit_if_supported(repository)
            attachment_id_by_file = {
                extracted.file_name: document_id for extracted in extracted_documents
            }
            chunks = chunk_extracted_documents(
                task_id=f"document-corpus:{job['assessment_id']}",
                attachment_id_by_file=attachment_id_by_file,
                extracted_documents=extracted_documents,
            )
            embedded_chunks = embed_evidence_chunks(
                chunks=chunks,
                embedding_client=embedding_client,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
            )
            repository.save_document_chunks(
                assessment_id=job["assessment_id"],
                document_id=document_id,
                chunks=[chunk.model_dump(mode="json") for chunk in embedded_chunks],
            )
            indexed_chunk_count += len(embedded_chunks)
            repository.update_document_ingestion_progress(
                job_id=job["job_id"],
                worker_id=job["lease_owner"],
                status="embedding_documents",
                processed_document_count=index,
                indexed_chunk_count=indexed_chunk_count,
            )
            _commit_if_supported(repository)
    return DocumentIngestionJobResponse(
        job_id=job["job_id"],
        assessment_id=job["assessment_id"],
        status="completed",
        document_count=len(documents),
        processed_document_count=len(documents),
        indexed_chunk_count=indexed_chunk_count,
    )


def run_batch_job_from_hana(
    repository: Any,
    job: dict[str, Any],
    review_client: Any | None = None,
    embedding_client: Any | None = None,
    max_concurrency: int = DEFAULT_BATCH_MAX_CONCURRENCY,
) -> BatchReviewGraphResult:
    """Build and run a batch graph from repository job state.

    Inputs:
        repository: Repository exposing framework lookup, batch document, and
            batch task methods.
        job: Leased batch job row.
        review_client: Optional injected review client.
        embedding_client: Optional injected embedding client.
        max_concurrency: Maximum number of parallel question branches.

    Outputs:
        BatchReviewGraphResult: Completed question count.
    """

    documents = repository.get_batch_job_documents(job["job_id"])
    tasks = repository.get_batch_question_tasks(job["job_id"])
    task_ids_by_question = {
        task["question_id"]: task["task_id"]
        for task in tasks
    }
    current_answers = {
        task["question_id"]: list(task.get("current_selected_answer_item_ids", []))
        for task in tasks
    }
    questions = [
        question
        for question in repository.list_all_questions(job.get("language", DEFAULT_LANGUAGE))
        if question.question_id in task_ids_by_question
    ]

    with TemporaryDirectory(prefix="document-assessment-batch-review-") as temporary_directory:
        extracted_documents: list[ExtractedDocument] = []
        for index, document in enumerate(documents, start=1):
            document_directory = Path(temporary_directory) / f"{index:03d}"
            document_directory.mkdir()
            file_name = Path(document["file_name"]).name
            path = document_directory / file_name
            path.write_bytes(_document_bytes(document))
            document_id = str(document["document_id"])
            for extracted_document in extract_evidence_files([path]):
                extracted_with_id = extracted_document.model_copy(
                    update={
                        "metadata": {
                            **extracted_document.metadata,
                            "document_id": document_id,
                        }
                    }
                )
                repository.save_batch_document_extraction(
                    job_id=job["job_id"],
                    document_id=document_id,
                    extracted=extracted_with_id,
                    estimated_tokens=sum(
                        estimate_text_tokens(block.text)
                        for block in extracted_with_id.blocks
                    ),
                    warnings=list(extracted_with_id.metadata.get("warnings", [])),
                )
                _commit_if_supported(repository)
                extracted_documents.append(extracted_with_id)

        return run_batch_review_graph(
            BatchReviewInput(
                job_id=job["job_id"],
                language=job.get("language", DEFAULT_LANGUAGE),
                questions=questions,
                current_answers=current_answers,
                documents=extracted_documents,
                task_ids_by_question=task_ids_by_question,
            ),
            repository=repository,
            embedding_client=embedding_client,
            review_client=review_client,
            max_concurrency=max_concurrency,
        )


def run_batch_indexing_from_hana(
    repository: Any,
    job: dict[str, Any],
    review_client: Any | None = None,
    embedding_client: Any | None = None,
) -> BatchIndexingResult:
    """Extract and index documents for one leased batch job.

    Inputs:
        repository: Repository exposing batch document, extraction, and chunk
            persistence methods.
        job: Leased batch job row containing ``job_id``, ``language``, and
            ``lease_owner``.
        review_client: Optional injected review client for Excel summaries.
        embedding_client: Optional injected embedding client.

    Outputs:
        BatchIndexingResult: Indexed chunk count without question review.
    """

    documents = repository.get_batch_job_documents(job["job_id"])
    with TemporaryDirectory(prefix="document-assessment-batch-review-") as temporary_directory:
        extracted_documents: list[ExtractedDocument] = []
        for index, document in enumerate(documents, start=1):
            repository.update_batch_indexing_progress(
                job_id=job["job_id"],
                worker_id=job["lease_owner"],
                status="extracting_documents",
            )
            _commit_if_supported(repository)
            document_directory = Path(temporary_directory) / f"{index:03d}"
            document_directory.mkdir()
            file_name = Path(document["file_name"]).name
            path = document_directory / file_name
            path.write_bytes(_document_bytes(document))
            document_id = str(document["document_id"])
            for extracted_document in extract_evidence_files([path]):
                extracted_with_id = extracted_document.model_copy(
                    update={
                        "metadata": {
                            **extracted_document.metadata,
                            "document_id": document_id,
                        }
                    }
                )
                repository.save_batch_document_extraction(
                    job_id=job["job_id"],
                    document_id=document_id,
                    extracted=extracted_with_id,
                    estimated_tokens=sum(
                        estimate_text_tokens(block.text)
                        for block in extracted_with_id.blocks
                    ),
                    warnings=list(extracted_with_id.metadata.get("warnings", [])),
                )
                _commit_if_supported(repository)
                extracted_documents.append(extracted_with_id)
        repository.update_batch_indexing_progress(
            job_id=job["job_id"],
            worker_id=job["lease_owner"],
            status="embedding_documents",
            indexed_chunk_count=0,
        )
        _commit_if_supported(repository)
        return run_batch_indexing_graph(
            BatchReviewInput(
                job_id=job["job_id"],
                language=job.get("language", DEFAULT_LANGUAGE),
                documents=extracted_documents,
            ),
            repository=repository,
            embedding_client=embedding_client,
            review_client=review_client,
            worker_id=job["lease_owner"],
        )
