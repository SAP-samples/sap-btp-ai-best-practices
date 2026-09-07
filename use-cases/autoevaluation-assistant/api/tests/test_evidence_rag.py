"""Tests for the bounded agentic evidence RAG loop."""

import pytest

from app.models.ai_review import (
    RetrievedEvidenceDecision,
    RetrievalRoundAssessment,
)
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.evidence_retrieval import RetrievedEvidenceChunk
from app.services.evidence_rag import (
    _assessment_prompt,
    run_retrieval_assessment_loop,
)


class FakeReviewClient:
    """Fake review client with scripted retrieval assessments."""

    def __init__(self, assessments: list[RetrievalRoundAssessment]) -> None:
        """Store scripted assessments and initialize call log.

        Inputs:
            assessments: Scripted retrieval-round assessments returned in order.

        Outputs:
            None. The fake is ready to record prompt calls.
        """
        self.assessments = assessments
        self.calls: list[list[dict[str, object]]] = []

    def assess_retrieved_evidence(
        self,
        input_messages: list[dict[str, object]],
        language: str = "en",
    ) -> RetrievalRoundAssessment:
        """Return the next scripted assessment.

        Inputs:
            input_messages: Prompt messages assembled for the current round.
            language: Review language.

        Outputs:
            RetrievalRoundAssessment: Next scripted assessment.
        """
        self.calls.append(input_messages)
        return self.assessments.pop(0)


class FakeRetriever:
    """Fake retriever returning one candidate per round."""

    def __init__(self) -> None:
        """Initialize query log.

        Inputs:
            None.

        Outputs:
            None. The fake is ready to record retrieval queries.
        """
        self.queries: list[list[str]] = []

    def __call__(self, queries: list[str]) -> list[RetrievedEvidenceChunk]:
        """Return deterministic candidates for supplied queries.

        Inputs:
            queries: Retrieval queries for the round.

        Outputs:
            list[RetrievedEvidenceChunk]: One candidate chunk.
        """
        self.queries.append(queries)
        return [
            RetrievedEvidenceChunk(
                chunk_id=f"chunk-{len(self.queries)}",
                file_name="report.pdf",
                document_type="pdf",
                chunk_text="Board approves objectives.",
                location_json={"pages": [len(self.queries)]},
                similarity_score=0.9,
            )
        ]


class FakeRepository:
    """Fake repository recording retrieval round persistence."""

    def __init__(self) -> None:
        """Initialize empty retrieval round log.

        Inputs:
            None.

        Outputs:
            None. The fake is ready to collect persisted rounds.
        """
        self.rounds: list[dict[str, object]] = []

    def save_retrieval_round(self, **kwargs: object) -> str:
        """Record one retrieval round.

        Inputs:
            **kwargs: Retrieval round fields.

        Outputs:
            str: Fake retrieval round ID.
        """
        self.rounds.append(dict(kwargs))
        return f"retrieval-round-{len(self.rounds)}"


class RepeatingChunkRetriever:
    """Fake retriever that always returns the same chunk ID."""

    def __init__(self) -> None:
        """Initialize query log for repeated retrieval assertions.

        Inputs:
            None.

        Outputs:
            None. The fake is ready to replay one chunk across rounds.
        """
        self.queries: list[list[str]] = []

    def __call__(self, queries: list[str]) -> list[RetrievedEvidenceChunk]:
        """Return the same chunk ID on every retrieval round.

        Inputs:
            queries: Retrieval queries for the current round.

        Outputs:
            list[RetrievedEvidenceChunk]: Candidate list containing ``chunk-1``.
        """
        self.queries.append(queries)
        return [
            RetrievedEvidenceChunk(
                chunk_id="chunk-1",
                file_name="report.pdf",
                document_type="pdf",
                chunk_text="Board approves objectives.",
                location_json={"pages": [len(self.queries)]},
                similarity_score=0.9,
            )
        ]


def _question() -> AssessmentQuestion:
    """Build a test Strategy question."""
    return AssessmentQuestion(
        question_id="Q.STR.02.01",
        dimension="Strategy",
        section="Objectives",
        question="Does the organization set strategic objectives?",
        answer_items=[
            AnswerItem(
                answer_item_id="Q.STR.02.01-L1-001",
                question_id="Q.STR.02.01",
                level=1,
                item_index=1,
                text="Objectives are defined.",
            )
        ],
    )


def test_rag_loop_refines_queries_until_sufficient() -> None:
    """Verify the loop uses refined queries when first evidence is incomplete."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[],
                evidence_gaps=["Need board approval evidence."],
                refined_queries=["board approval objectives"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            ),
            RetrievalRoundAssessment(
                accepted_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-2",
                        relevance="support",
                        related_answer_item_ids=["Q.STR.02.01-L1-001"],
                        rationale="The chunk supports objective approval.",
                        confidence=0.9,
                    )
                ],
                evidence_gaps=[],
                refined_queries=[],
                sufficient_for_final_review=True,
                stop_reason="sufficient_evidence",
            ),
        ]
    )
    retriever = FakeRetriever()
    repository = FakeRepository()

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=retriever,
        review_client=review_client,
        repository=repository,
        max_rounds=3,
    )

    assert len(retriever.queries) == 2
    assert retriever.queries[1] == ["board approval objectives"]
    assert result.stop_reason == "sufficient_evidence"
    assert result.sufficient_for_final_review is True
    assert result.accepted_evidence[0].source_chunk_id == "chunk-2"
    assert len(repository.rounds) == 2


def test_rag_loop_persists_normalized_stop_reason_code() -> None:
    """Verify prose model stop reasons are persisted as compact codes.

    Inputs:
        None. The test scripts one sufficient assessment with a long prose stop
        reason matching the HANA overflow failure mode.

    Outputs:
        None. Assertions confirm the persisted round and returned loop result
        use the short ``sufficient_evidence`` code.
    """
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="support",
                        related_answer_item_ids=["Q.STR.02.01-L1-001"],
                        rationale="The chunk supports objective approval.",
                        confidence=0.9,
                    )
                ],
                evidence_gaps=[],
                refined_queries=[],
                sufficient_for_final_review=True,
                stop_reason=(
                    "Accepted evidence is sufficient to ground a final review "
                    "on whether strategy definition takes risks into account, "
                    "especially for basic through advanced formal integration."
                ),
            )
        ]
    )
    retriever = FakeRetriever()
    repository = FakeRepository()

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=retriever,
        review_client=review_client,
        repository=repository,
        max_rounds=3,
    )

    assert repository.rounds[0]["stop_reason"] == "sufficient_evidence"
    assert result.stop_reason == "sufficient_evidence"


def test_rag_loop_stops_at_max_rounds() -> None:
    """Verify the loop stops predictably when evidence stays insufficient."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[],
                evidence_gaps=["Need target evidence."],
                refined_queries=["target evidence"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            ),
            RetrievalRoundAssessment(
                accepted_evidence=[],
                evidence_gaps=["Need monitoring evidence."],
                refined_queries=["monitoring evidence"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            ),
        ]
    )
    retriever = FakeRetriever()
    repository = FakeRepository()

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=retriever,
        review_client=review_client,
        repository=repository,
        max_rounds=2,
    )

    assert result.stop_reason == "max_rounds"
    assert result.sufficient_for_final_review is False
    assert result.evidence_gaps == ["Need monitoring evidence."]


def test_rag_loop_stops_when_no_refined_queries_are_returned() -> None:
    """Verify the loop stops when evidence is insufficient and no new queries exist."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[],
                evidence_gaps=["Need board approval evidence."],
                refined_queries=[],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            )
        ]
    )

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=FakeRetriever(),
        review_client=review_client,
        repository=FakeRepository(),
        max_rounds=3,
    )

    assert result.stop_reason == "no_refined_queries"
    assert result.sufficient_for_final_review is False
    assert result.rounds_executed == 1


def test_rag_loop_persists_rejected_evidence_and_filters_accepted_chunks() -> None:
    """Verify rejected evidence is persisted while accepted chunks stay filtered."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="support",
                        related_answer_item_ids=["Q.STR.02.01-L1-001"],
                        rationale="This chunk directly supports the objective.",
                        confidence=0.95,
                    )
                ],
                rejected_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-x",
                        relevance="background_only",
                        related_answer_item_ids=[],
                        rationale="This chunk is background context only.",
                        confidence=0.55,
                    )
                ],
                evidence_gaps=[],
                refined_queries=[],
                sufficient_for_final_review=True,
                stop_reason="sufficient_evidence",
            )
        ]
    )
    retriever = FakeRetriever()
    repository = FakeRepository()

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=retriever,
        review_client=review_client,
        repository=repository,
        max_rounds=1,
    )

    persisted_payload = repository.rounds[0]["accepted_evidence_json"]
    assert persisted_payload["accepted_evidence"][0]["source_chunk_id"] == "chunk-1"
    assert persisted_payload["rejected_evidence"][0]["source_chunk_id"] == "chunk-x"
    assert result.rejected_evidence[0].source_chunk_id == "chunk-x"
    assert [chunk.chunk_id for chunk in result.accepted_chunks] == ["chunk-1"]


def test_rag_loop_removes_prior_accepted_chunk_when_later_rejected() -> None:
    """Verify later rejection supersedes earlier accepted evidence for one chunk."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="support",
                        related_answer_item_ids=["Q.STR.02.01-L1-001"],
                        rationale="The first round supports the objective.",
                        confidence=0.92,
                    )
                ],
                evidence_gaps=["Need confirmation of board approval."],
                refined_queries=["board approval objectives"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            ),
            RetrievalRoundAssessment(
                accepted_evidence=[],
                rejected_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="background_only",
                        related_answer_item_ids=[],
                        rationale="The later round shows the chunk is only context.",
                        confidence=0.61,
                    )
                ],
                evidence_gaps=[],
                refined_queries=[],
                sufficient_for_final_review=True,
                stop_reason="sufficient_evidence",
            ),
        ]
    )

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=RepeatingChunkRetriever(),
        review_client=review_client,
        repository=FakeRepository(),
        max_rounds=2,
    )

    assert result.accepted_chunks == []
    assert result.accepted_evidence == []
    assert [decision.source_chunk_id for decision in result.rejected_evidence] == [
        "chunk-1"
    ]


def test_rag_loop_removes_prior_rejection_when_later_accepted() -> None:
    """Verify later acceptance supersedes earlier rejected evidence for one chunk."""
    review_client = FakeReviewClient(
        assessments=[
            RetrievalRoundAssessment(
                accepted_evidence=[],
                rejected_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="background_only",
                        related_answer_item_ids=[],
                        rationale="The first round treats the chunk as context.",
                        confidence=0.62,
                    )
                ],
                evidence_gaps=["Need stronger objective approval evidence."],
                refined_queries=["objective approval board"],
                sufficient_for_final_review=False,
                stop_reason="needs_more_evidence",
            ),
            RetrievalRoundAssessment(
                accepted_evidence=[
                    RetrievedEvidenceDecision(
                        source_chunk_id="chunk-1",
                        relevance="support",
                        related_answer_item_ids=["Q.STR.02.01-L1-001"],
                        rationale="The later round finds the chunk supports approval.",
                        confidence=0.91,
                    )
                ],
                rejected_evidence=[],
                evidence_gaps=[],
                refined_queries=[],
                sufficient_for_final_review=True,
                stop_reason="sufficient_evidence",
            ),
        ]
    )

    result = run_retrieval_assessment_loop(
        task_id="task-1",
        question=_question(),
        language="en",
        initial_queries=["strategic objectives"],
        retrieve_candidates=RepeatingChunkRetriever(),
        review_client=review_client,
        repository=FakeRepository(),
        max_rounds=2,
    )

    assert result.rejected_evidence == []
    assert [decision.source_chunk_id for decision in result.accepted_evidence] == [
        "chunk-1"
    ]
    assert [chunk.chunk_id for chunk in result.accepted_chunks] == ["chunk-1"]


def test_rag_loop_rejects_non_positive_max_rounds() -> None:
    """Verify loop configuration rejects non-positive maximum rounds."""
    with pytest.raises(ValueError, match="max_rounds must be positive"):
        run_retrieval_assessment_loop(
            task_id="task-1",
            question=_question(),
            language="en",
            initial_queries=["strategic objectives"],
            retrieve_candidates=FakeRetriever(),
            review_client=FakeReviewClient(assessments=[]),
            repository=FakeRepository(),
            max_rounds=0,
        )


def test_assessment_prompt_formats_previous_gaps_as_bullets() -> None:
    """Verify prompt text renders previous gaps without Python list repr output."""
    prompt = _assessment_prompt(
        question=_question(),
        previous_gaps=["Need board approval evidence.", "Need monitoring evidence."],
        retrieved_chunks=[],
    )

    content = prompt[0]["content"]
    assert isinstance(content, list)
    text = content[0]["text"]
    assert "Previous evidence gaps:" in text
    assert "- Need board approval evidence." in text
    assert "- Need monitoring evidence." in text
    assert "['Need board approval evidence.'" not in text
