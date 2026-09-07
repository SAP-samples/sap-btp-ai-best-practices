"""Tests for evidence retrieval query generation and ranking."""

from __future__ import annotations

from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.evidence_retrieval import (
    RagQuery,
    RetrievedEvidenceChunk,
    build_initial_retrieval_queries,
    build_question_rag_queries,
    retrieve_batch_rag_candidates,
    retrieve_evidence_candidates,
)


class FakeEmbeddingClient:
    """Fake embedding client returning query-length vectors."""

    def __init__(self) -> None:
        """Initialize an empty query text log.

        Inputs:
            None.

        Outputs:
            None. The fake is ready to record embedded query text.
        """
        self.texts: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        """Embed one query as a deterministic vector.

        Inputs:
            text: Retrieval query text.

        Outputs:
            list[float]: Simple query vector.
        """
        self.texts.append(text)
        return [float(len(text))]


class FakeRepository:
    """Fake repository exposing vector evidence search."""

    def __init__(self) -> None:
        """Initialize an empty query log."""
        self.queries: list[list[float]] = []

    def search_evidence_chunks(
        self,
        task_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return deterministic chunks for each vector query.

        Inputs:
            task_id: Review task identifier.
            query_embedding: Query vector produced by the fake embedding client.
            top_k: Maximum result count.

        Outputs:
            list[dict[str, object]]: Repository-shaped chunk rows.
        """
        self.queries.append(query_embedding)
        return [
            {
                "chunk_id": f"chunk-{len(self.queries)}",
                "attachment_id": "attachment-report",
                "file_name": "report.pdf",
                "document_type": "pdf",
                "chunk_text": "Board approves strategy.",
                "location_json": {"pages": [1]},
                "similarity_score": 0.91,
            }
        ][:top_k]


def _question() -> AssessmentQuestion:
    """Build a test assessment question with two maturity levels."""
    return AssessmentQuestion(
        question_id="Q.STR.02.01",
        dimension="Strategy",
        section="Objectives",
        question="Does the organization set strategic objectives?",
        explanation="Review evidence of strategic targets and monitoring.",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.02.01-L1-001",
                question_id="Q.STR.02.01",
                level=1,
                item_index=1,
                text="Objectives are defined.",
            ),
            AnswerItem(
                answer_item_id="Q.STR.02.01-L2-001",
                question_id="Q.STR.02.01",
                level=2,
                item_index=1,
                text="Objectives are monitored.",
            ),
        ],
    )


def test_build_initial_retrieval_queries_includes_question_and_levels() -> None:
    """Verify retrieval queries cover question, explanation, and answer levels."""
    queries = build_initial_retrieval_queries(_question())

    assert queries[0].startswith("Does the organization set strategic objectives?")
    assert any("Level 1" in query and "Objectives are defined" in query for query in queries)
    assert any("Level 2" in query and "Objectives are monitored" in query for query in queries)


def test_retrieve_evidence_candidates_embeds_each_query() -> None:
    """Verify retrieval embeds queries and returns typed candidate chunks."""
    repository = FakeRepository()
    embedding_client = FakeEmbeddingClient()

    candidates = retrieve_evidence_candidates(
        repository=repository,
        embedding_client=embedding_client,
        task_id="task-1",
        queries=["strategic objectives", "monitoring evidence"],
        top_k_per_query=1,
    )

    assert embedding_client.texts == ["strategic objectives", "monitoring evidence"]
    assert len(repository.queries) == 2
    assert all(isinstance(candidate, RetrievedEvidenceChunk) for candidate in candidates)
    assert candidates[0].chunk_id == "chunk-1"
    assert candidates[0].similarity_score == 0.91


def test_retrieve_evidence_candidates_caps_chunks_per_file() -> None:
    """Verify retrieval diversity prevents one file from crowding all results."""

    class SameFileRepository:
        """Fake repository returning multiple chunks from one file."""

        def search_evidence_chunks(
            self,
            task_id: str,
            query_embedding: list[float],
            top_k: int,
        ) -> list[dict[str, object]]:
            """Return same-file chunk rows.

            Inputs:
                task_id: Review task identifier.
                query_embedding: Query vector.
                top_k: Maximum result count.

            Outputs:
                list[dict[str, object]]: Three same-file chunks.
            """
            return [
                {
                    "chunk_id": f"chunk-{index}",
                    "attachment_id": "attachment-annual-report",
                    "file_name": "annual-report.pdf",
                    "document_type": "pdf",
                    "chunk_text": f"Evidence {index}",
                    "location_json": {"pages": [index]},
                    "similarity_score": 1.0 - index / 10,
                }
                for index in range(1, 4)
            ]

    candidates = retrieve_evidence_candidates(
        repository=SameFileRepository(),
        embedding_client=FakeEmbeddingClient(),
        task_id="task-1",
        queries=["strategy"],
        top_k_per_query=3,
        max_chunks_per_file=2,
    )

    assert [candidate.chunk_id for candidate in candidates] == ["chunk-1", "chunk-2"]


def test_retrieve_evidence_candidates_counts_duplicate_names_by_attachment() -> None:
    """Verify duplicate filenames do not share one retrieval diversity budget.

    Inputs:
        None. A fake repository returns chunks from two attachments with the
        same basename.

    Outputs:
        None. Assertions confirm retrieval allows one candidate from each
        attachment when per-source chunk caps are strict.
    """

    class DuplicateNameRepository:
        """Fake repository returning same-basename chunks from two attachments."""

        def search_evidence_chunks(
            self,
            task_id: str,
            query_embedding: list[float],
            top_k: int,
        ) -> list[dict[str, object]]:
            """Return chunks with the same filename and different attachments.

            Inputs:
                task_id: Review task identifier.
                query_embedding: Query vector.
                top_k: Maximum result count.

            Outputs:
                list[dict[str, object]]: Three chunks across two attachments.
            """

            return [
                {
                    "chunk_id": "chunk-a1",
                    "attachment_id": "attachment-a",
                    "file_name": "strategy-evidence.pdf",
                    "document_type": "pdf",
                    "chunk_text": "First upload evidence one.",
                    "location_json": {"pages": [1]},
                    "similarity_score": 0.99,
                },
                {
                    "chunk_id": "chunk-a2",
                    "attachment_id": "attachment-a",
                    "file_name": "strategy-evidence.pdf",
                    "document_type": "pdf",
                    "chunk_text": "First upload evidence two.",
                    "location_json": {"pages": [2]},
                    "similarity_score": 0.98,
                },
                {
                    "chunk_id": "chunk-b1",
                    "attachment_id": "attachment-b",
                    "file_name": "strategy-evidence.pdf",
                    "document_type": "pdf",
                    "chunk_text": "Second upload evidence one.",
                    "location_json": {"pages": [1]},
                    "similarity_score": 0.97,
                },
            ][:top_k]

    candidates = retrieve_evidence_candidates(
        repository=DuplicateNameRepository(),
        embedding_client=FakeEmbeddingClient(),
        task_id="task-1",
        queries=["strategy"],
        top_k_per_query=3,
        max_chunks_per_file=1,
    )

    assert [candidate.chunk_id for candidate in candidates] == [
        "chunk-a1",
        "chunk-b1",
    ]
    assert [candidate.attachment_id for candidate in candidates] == [
        "attachment-a",
        "attachment-b",
    ]


def test_build_question_rag_queries_includes_question_and_answer_items() -> None:
    """Verify one question produces query metadata for every answer item."""
    question = AssessmentQuestion(
        question_id="Q.STR.01.01",
        dimension="Strategy",
        section="Strategic Planning",
        question="Does the organization have strategic planning?",
        explanation="Look for planning process governance.",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.01.01-L1-001",
                question_id="Q.STR.01.01",
                level=1,
                item_index=1,
                text="Planning process is identified.",
            ),
            AnswerItem(
                answer_item_id="Q.STR.01.01-L2-001",
                question_id="Q.STR.01.01",
                level=2,
                item_index=1,
                text="Strategic plan is formalized.",
            ),
        ],
    )

    queries = build_question_rag_queries(question)

    assert queries[0].answer_item_id is None
    assert queries[0].level is None
    assert queries[0].query.startswith("Does the organization")
    assert [query.answer_item_id for query in queries[1:]] == [
        "Q.STR.01.01-L1-001",
        "Q.STR.01.01-L2-001",
    ]


def test_retrieve_batch_rag_candidates_batch_embeds_queries_and_deduplicates() -> None:
    """Verify batch RAG retrieval embeds all query texts in one request."""

    class FakeBatchEmbeddingClient:
        def __init__(self) -> None:
            self.batch_calls: list[list[str]] = []

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            self.batch_calls.append(texts)
            return [[float(index)] for index, _text in enumerate(texts, start=1)]

    class FakeBatchRepository:
        def search_batch_document_chunks(self, job_id, query_embedding, top_k):
            return [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "attachment_id": "document-1",
                    "file_name": "strategy.pdf",
                    "document_type": "pdf",
                    "chunk_text": "The strategic plan is approved by the board.",
                    "location_json": {"pages": [4]},
                    "similarity_score": 0.9,
                }
            ]

    queries = [
        RagQuery(query="strategic plan", answer_item_id="A1", level=2),
        RagQuery(query="board approval", answer_item_id="A2", level=3),
    ]
    embedding_client = FakeBatchEmbeddingClient()

    chunks = retrieve_batch_rag_candidates(
        repository=FakeBatchRepository(),
        embedding_client=embedding_client,
        job_id="batch-job-1",
        queries=queries,
        top_k_per_query=8,
    )

    assert embedding_client.batch_calls == [["strategic plan", "board approval"]]
    assert len(chunks) == 1
    assert [query.answer_item_id for query in chunks[0].matched_queries] == ["A1", "A2"]
