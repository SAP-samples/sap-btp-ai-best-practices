"""Evidence retrieval query generation and vector-search orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.assessment import AssessmentQuestion


class RetrievedEvidenceChunk(BaseModel):
    """Evidence chunk returned by vector retrieval.

    Inputs:
        Field values describing retrieved chunk text and source metadata.

    Outputs:
        Validated candidate chunk for RAG assessment.
    """

    chunk_id: str
    attachment_id: str | None = None
    file_name: str
    document_type: str
    chunk_text: str
    location_json: dict[str, object] = Field(default_factory=dict)
    similarity_score: float


def build_initial_retrieval_queries(question: AssessmentQuestion) -> list[str]:
    """Build first-round retrieval queries from question and answer items.

    Inputs:
        question: Assessment framework question with explanation and answer items.

    Outputs:
        list[str]: Ordered retrieval queries for semantic chunk search.
    """

    queries = [
        " ".join(
            part for part in [question.question, question.explanation or ""] if part
        ).strip()
    ]
    items_by_level: dict[int, list[str]] = {}
    for item in question.answer_items:
        items_by_level.setdefault(item.level, []).append(item.text)
    for level in sorted(items_by_level):
        query_text = " ".join(items_by_level[level])
        queries.append(f"Level {level} evidence for {question.section}: {query_text}")
    return [query for query in queries if query]


def retrieve_evidence_candidates(
    repository: object,
    embedding_client: object,
    task_id: str,
    queries: list[str],
    top_k_per_query: int = 8,
    max_chunks_per_file: int = 4,
) -> list[RetrievedEvidenceChunk]:
    """Retrieve candidate evidence chunks for one or more semantic queries.

    Inputs:
        repository: Repository exposing ``search_evidence_chunks``.
        embedding_client: Client exposing ``embed_query``.
        task_id: Review task identifier.
        queries: Retrieval query strings.
        top_k_per_query: Maximum chunks requested per query.
        max_chunks_per_file: Per-source cap used to keep one long attachment
            from crowding out shorter evidence sources. When repositories return
            ``attachment_id``, duplicate filenames are counted separately.

    Outputs:
        list[RetrievedEvidenceChunk]: Deduplicated chunks in retrieval order.
    """

    candidates: list[RetrievedEvidenceChunk] = []
    seen_chunk_ids: set[str] = set()
    file_counts: dict[str, int] = {}
    for query in queries:
        query_embedding = embedding_client.embed_query(query)
        rows = repository.search_evidence_chunks(
            task_id=task_id,
            query_embedding=query_embedding,
            top_k=top_k_per_query,
        )
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue
            file_name = str(row["file_name"])
            source_key = str(row.get("attachment_id") or file_name)
            if file_counts.get(source_key, 0) >= max_chunks_per_file:
                continue
            seen_chunk_ids.add(chunk_id)
            file_counts[source_key] = file_counts.get(source_key, 0) + 1
            candidates.append(RetrievedEvidenceChunk.model_validate(row))
    return candidates


class RagQuery(BaseModel):
    """One semantic retrieval query linked to optional answer metadata.

    Inputs:
        query: Search text sent to the embedding model.
        answer_item_id: Optional answer item this query targets.
        level: Optional maturity level this query targets.

    Outputs:
        Validated query metadata used by batch RAG retrieval.
    """

    query: str
    answer_item_id: str | None = None
    level: int | None = None


class RagRetrievedChunk(RetrievedEvidenceChunk):
    """Retrieved chunk annotated with the queries that matched it.

    Inputs:
        Retrieved evidence fields plus matched query metadata.

    Outputs:
        Validated chunk payload returned by the rag_search tool.
    """

    matched_queries: list[RagQuery] = Field(default_factory=list)


def build_question_rag_queries(question: AssessmentQuestion) -> list[RagQuery]:
    """Build one question-level query plus one query per answer item.

    Inputs:
        question: Assessment framework question with explanation and answer items.

    Outputs:
        list[RagQuery]: Ordered retrieval queries with answer metadata.
    """
    query_text = " ".join(
        part for part in [question.question, question.explanation or ""] if part
    ).strip()
    queries: list[RagQuery] = [RagQuery(query=query_text)] if query_text else []
    for item in question.answer_items:
        queries.append(
            RagQuery(
                query=f"{question.section}: {item.text}",
                answer_item_id=item.answer_item_id,
                level=item.level,
            )
        )
    return [query for query in queries if query.query.strip()]


def retrieve_batch_rag_candidates(
    repository: object,
    embedding_client: object,
    job_id: str,
    queries: list[RagQuery],
    top_k_per_query: int = 8,
    max_chunks_per_document: int = 4,
    max_total_chunks: int = 30,
) -> list[RagRetrievedChunk]:
    """Retrieve deduplicated batch document chunks for a set of query metadata.

    Inputs:
        repository: Repository exposing ``search_batch_document_chunks``.
        embedding_client: Client exposing ``embed_batch``.
        job_id: Batch job identifier owning the shared vector corpus.
        queries: RAG query metadata objects.
        top_k_per_query: Maximum chunks requested per query.
        max_chunks_per_document: Per-document cap to ensure diversity.
        max_total_chunks: Hard cap on total returned chunks.

    Outputs:
        list[RagRetrievedChunk]: Deduplicated chunks sorted by similarity,
            each annotated with the queries that matched it.
    """
    if not queries:
        return []
    embeddings = embedding_client.embed_batch([query.query for query in queries])
    if len(embeddings) != len(queries):
        raise ValueError(
            f"Embedding client returned {len(embeddings)} vectors for {len(queries)} queries"
        )

    chunks_by_id: dict[str, RagRetrievedChunk] = {}
    document_counts: dict[str, int] = {}
    for query, embedding in zip(queries, embeddings, strict=True):
        rows = repository.search_batch_document_chunks(
            job_id=job_id,
            query_embedding=embedding,
            top_k=top_k_per_query,
        )
        for row in rows:
            chunk_id = str(row["chunk_id"])
            document_id = str(
                row.get("document_id") or row.get("attachment_id") or row["file_name"]
            )
            if chunk_id not in chunks_by_id:
                if document_counts.get(document_id, 0) >= max_chunks_per_document:
                    continue
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
                chunks_by_id[chunk_id] = RagRetrievedChunk.model_validate(
                    {**row, "matched_queries": [query]}
                )
            else:
                existing = chunks_by_id[chunk_id]
                existing.matched_queries.append(query)

    return sorted(
        chunks_by_id.values(),
        key=lambda chunk: chunk.similarity_score,
        reverse=True,
    )[:max_total_chunks]


def retrieve_document_rag_candidates(
    repository: object,
    embedding_client: object,
    assessment_id: str,
    queries: list[RagQuery],
    top_k_per_query: int = 8,
    max_chunks_per_document: int = 4,
    max_total_chunks: int = 30,
) -> list[RagRetrievedChunk]:
    """Retrieve deduplicated chunks from an assessment-scoped document corpus.

    Inputs:
        repository: Repository exposing ``search_document_chunks``.
        embedding_client: Client exposing ``embed_batch``.
        assessment_id: Assessment whose indexed corpus should be searched.
        queries: RAG query metadata objects.
        top_k_per_query: Maximum chunks requested per query.
        max_chunks_per_document: Per-document cap to ensure source diversity.
        max_total_chunks: Hard cap on total returned chunks.

    Outputs:
        list[RagRetrievedChunk]: Deduplicated chunks sorted by similarity,
        each annotated with the queries that matched it.
    """
    if not queries:
        return []
    embeddings = embedding_client.embed_batch([query.query for query in queries])
    if len(embeddings) != len(queries):
        raise ValueError(
            f"Embedding client returned {len(embeddings)} vectors for {len(queries)} queries"
        )

    chunks_by_id: dict[str, RagRetrievedChunk] = {}
    document_counts: dict[str, int] = {}
    for query, embedding in zip(queries, embeddings, strict=True):
        rows = repository.search_document_chunks(
            assessment_id=assessment_id,
            query_embedding=embedding,
            top_k=top_k_per_query,
        )
        for row in rows:
            chunk_id = str(row["chunk_id"])
            document_id = str(
                row.get("document_id") or row.get("attachment_id") or row["file_name"]
            )
            if chunk_id not in chunks_by_id:
                if document_counts.get(document_id, 0) >= max_chunks_per_document:
                    continue
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
                chunks_by_id[chunk_id] = RagRetrievedChunk.model_validate(
                    {**row, "matched_queries": [query]}
                )
            else:
                chunks_by_id[chunk_id].matched_queries.append(query)

    return sorted(
        chunks_by_id.values(),
        key=lambda chunk: chunk.similarity_score,
        reverse=True,
    )[:max_total_chunks]


def retrieve_task_rag_candidates(
    repository: object,
    embedding_client: object,
    task_id: str,
    queries: list[RagQuery],
    top_k_per_query: int = 8,
    max_chunks_per_attachment: int = 4,
    max_total_chunks: int = 30,
) -> list[RagRetrievedChunk]:
    """Retrieve deduplicated chunks for one manual question task.

    Inputs:
        repository: Repository exposing ``search_evidence_chunks``.
        embedding_client: Client exposing ``embed_batch``.
        task_id: Manual AI review task identifier owning the vector corpus.
        queries: Structured RAG queries generated by the ReAct tool.
        top_k_per_query: Maximum chunks requested per query.
        max_chunks_per_attachment: Per-attachment cap to preserve source
            diversity when one file dominates similarity results.
        max_total_chunks: Hard cap on total chunks returned to the graph.

    Outputs:
        list[RagRetrievedChunk]: Deduplicated chunks sorted by similarity, each
        annotated with the queries that matched it.
    """
    if not queries:
        return []
    embeddings = embedding_client.embed_batch([query.query for query in queries])
    if len(embeddings) != len(queries):
        raise ValueError(
            f"Embedding client returned {len(embeddings)} vectors for {len(queries)} queries"
        )

    chunks_by_id: dict[str, RagRetrievedChunk] = {}
    attachment_counts: dict[str, int] = {}
    for query, embedding in zip(queries, embeddings, strict=True):
        rows = repository.search_evidence_chunks(
            task_id=task_id,
            query_embedding=embedding,
            top_k=top_k_per_query,
        )
        for row in rows:
            chunk_id = str(row["chunk_id"])
            attachment_id = str(row.get("attachment_id") or row["file_name"])
            if chunk_id not in chunks_by_id:
                if attachment_counts.get(attachment_id, 0) >= max_chunks_per_attachment:
                    continue
                attachment_counts[attachment_id] = (
                    attachment_counts.get(attachment_id, 0) + 1
                )
                chunks_by_id[chunk_id] = RagRetrievedChunk.model_validate(
                    {**row, "matched_queries": [query]}
                )
            else:
                existing = chunks_by_id[chunk_id]
                existing.matched_queries.append(query)

    return sorted(
        chunks_by_id.values(),
        key=lambda chunk: chunk.similarity_score,
        reverse=True,
    )[:max_total_chunks]
