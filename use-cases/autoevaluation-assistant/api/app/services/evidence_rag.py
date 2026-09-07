"""Agentic evidence RAG loop for assessment question review tasks."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from app.models.ai_review import (
    RetrievedEvidenceDecision,
    RetrievalRoundAssessment,
    normalize_retrieval_stop_reason,
)
from app.models.assessment import AssessmentQuestion
from app.services.evidence_retrieval import RetrievedEvidenceChunk


class RagLoopResult(BaseModel):
    """Accumulated output from the bounded retrieval assessment loop.

    Inputs:
        Field values collected across RAG retrieval rounds.

    Outputs:
        Validated loop result used to build the final review prompt.
    """

    accepted_evidence: list[RetrievedEvidenceDecision] = Field(default_factory=list)
    rejected_evidence: list[RetrievedEvidenceDecision] = Field(default_factory=list)
    accepted_chunks: list[RetrievedEvidenceChunk] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    sufficient_for_final_review: bool = False
    stop_reason: str
    rounds_executed: int


def _assessment_prompt(
    question: AssessmentQuestion,
    previous_gaps: list[str],
    retrieved_chunks: list[RetrievedEvidenceChunk],
) -> list[dict[str, object]]:
    """Build one Responses-style user message for retrieval assessment.

    Inputs:
        question: Assessment question being reviewed.
        previous_gaps: Evidence gaps carried from the prior retrieval round.
        retrieved_chunks: Candidate chunks retrieved for this round.

    Outputs:
        list[dict[str, object]]: One user message with a single ``input_text``
        content block describing the question, answer items, prior gaps, and
        current candidate chunks.
    """

    answer_lines = [
        (
            f"- {item.answer_item_id} | level={item.level} | "
            f"item_index={item.item_index} | {item.text}"
        )
        for item in question.answer_items
    ]
    chunk_lines = [
        "\n".join(
            [
                f"- Chunk ID: {chunk.chunk_id}",
                *(
                    [f"  Attachment ID: {chunk.attachment_id}"]
                    if chunk.attachment_id
                    else []
                ),
                f"  File: {chunk.file_name}",
                f"  Type: {chunk.document_type}",
                f"  Location: {chunk.location_json}",
                f"  Text: {chunk.chunk_text}",
            ]
        )
        for chunk in retrieved_chunks
    ]
    previous_gap_lines = previous_gaps or ["None"]
    prompt = "\n".join(
        [
            "Assess retrieved evidence chunks for this assessment question.",
            f"Question ID: {question.question_id}",
            f"Question: {question.question}",
            f"Explanation: {question.explanation or 'None'}",
            "Answer items:",
            *answer_lines,
            "Previous evidence gaps:",
            *[f"- {gap}" for gap in previous_gap_lines],
            "Retrieved chunks:",
            *(chunk_lines or ["- None"]),
        ]
    )
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]


def run_retrieval_assessment_loop(
    task_id: str,
    question: AssessmentQuestion,
    language: str,
    initial_queries: list[str],
    retrieve_candidates: Callable[[list[str]], list[RetrievedEvidenceChunk]],
    review_client: object,
    repository: object,
    max_rounds: int = 3,
) -> RagLoopResult:
    """Run a bounded retrieve-assess-refine loop for one question task.

    Inputs:
        task_id: Review task identifier used for persistence.
        question: Assessment question that drives retrieval and assessment.
        language: Requested review language passed to the review client.
        initial_queries: First-round retrieval queries.
        retrieve_candidates: Callable that returns candidate chunks per round.
        review_client: Client exposing ``assess_retrieved_evidence``.
        repository: Persistence boundary exposing ``save_retrieval_round``.
        max_rounds: Maximum number of retrieval rounds to execute.

    Outputs:
        RagLoopResult: Accumulated accepted evidence, rejected evidence,
        accepted chunks, latest evidence gaps, stop reason, and rounds executed.

    Raises:
        ValueError: Raised when ``max_rounds`` is not positive.
    """

    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")

    queries = list(initial_queries)
    accepted_decisions_by_id: dict[str, RetrievedEvidenceDecision] = {}
    rejected_decisions_by_id: dict[str, RetrievedEvidenceDecision] = {}
    accepted_chunks_by_id: dict[str, RetrievedEvidenceChunk] = {}
    evidence_gaps: list[str] = []

    for round_number in range(1, max_rounds + 1):
        candidates = retrieve_candidates(queries)
        assessment = review_client.assess_retrieved_evidence(
            _assessment_prompt(
                question=question,
                previous_gaps=evidence_gaps,
                retrieved_chunks=candidates,
            ),
            language=language,
        )

        # Keep only chunks the model explicitly accepted into the final review set.
        candidate_chunks_by_id = {chunk.chunk_id: chunk for chunk in candidates}
        for decision in assessment.accepted_evidence:
            rejected_decisions_by_id.pop(decision.source_chunk_id, None)
            accepted_decisions_by_id[decision.source_chunk_id] = decision
            chunk = candidate_chunks_by_id.get(decision.source_chunk_id)
            if chunk is not None:
                accepted_chunks_by_id[chunk.chunk_id] = chunk
        for decision in assessment.rejected_evidence:
            accepted_decisions_by_id.pop(decision.source_chunk_id, None)
            accepted_chunks_by_id.pop(decision.source_chunk_id, None)
            rejected_decisions_by_id[decision.source_chunk_id] = decision

        round_stop_reason = normalize_retrieval_stop_reason(
            stop_reason=assessment.stop_reason,
            sufficient_for_final_review=assessment.sufficient_for_final_review,
        )
        repository.save_retrieval_round(
            task_id=task_id,
            round_number=round_number,
            queries=queries,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in candidates],
            accepted_evidence_json={
                "accepted_evidence": [
                    decision.model_dump() for decision in assessment.accepted_evidence
                ],
                "rejected_evidence": [
                    decision.model_dump() for decision in assessment.rejected_evidence
                ],
            },
            evidence_gaps=assessment.evidence_gaps,
            refined_queries=assessment.refined_queries,
            stop_reason=round_stop_reason,
        )

        evidence_gaps = list(assessment.evidence_gaps)
        if assessment.sufficient_for_final_review:
            return RagLoopResult(
                accepted_evidence=list(accepted_decisions_by_id.values()),
                rejected_evidence=list(rejected_decisions_by_id.values()),
                accepted_chunks=list(accepted_chunks_by_id.values()),
                evidence_gaps=evidence_gaps,
                sufficient_for_final_review=True,
                stop_reason=round_stop_reason or "sufficient_evidence",
                rounds_executed=round_number,
            )
        if not assessment.refined_queries:
            return RagLoopResult(
                accepted_evidence=list(accepted_decisions_by_id.values()),
                rejected_evidence=list(rejected_decisions_by_id.values()),
                accepted_chunks=list(accepted_chunks_by_id.values()),
                evidence_gaps=evidence_gaps,
                sufficient_for_final_review=False,
                stop_reason="no_refined_queries",
                rounds_executed=round_number,
            )
        queries = list(assessment.refined_queries)

    return RagLoopResult(
        accepted_evidence=list(accepted_decisions_by_id.values()),
        rejected_evidence=list(rejected_decisions_by_id.values()),
        accepted_chunks=list(accepted_chunks_by_id.values()),
        evidence_gaps=evidence_gaps,
        sufficient_for_final_review=False,
        stop_reason="max_rounds",
        rounds_executed=max_rounds,
    )
