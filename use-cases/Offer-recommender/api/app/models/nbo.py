from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.nbo.models import (
    DecisionExplanation,
    DecisionQuestion,
    DecisionResult,
    FactValue,
    OfferDecision,
    RecommendationExplanation,
    WorkflowStage,
)


class AnswerOptionResponse(BaseModel):
    label: str
    value: Any
    description: str | None = None


class QuestionResponse(BaseModel):
    question_id: str
    prompt: str
    expected_fact: str
    candidate_programs: list[str] = Field(default_factory=list)
    priority: int
    source: str
    answer_options: list[AnswerOptionResponse] = Field(default_factory=list)
    explanation: "DecisionExplanationResponse | None" = None


class OfferResponse(BaseModel):
    program_id: str
    display_name: str
    status: str
    rank: int | None = None
    confidence: str
    reason_codes: list[str] = Field(default_factory=list)
    blocking_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    explanation: "DecisionExplanationResponse | None" = None


class FactResponse(BaseModel):
    fact_id: str
    value: Any = None
    value_type: str
    source: str
    confidence: str
    evidence: list[str] = Field(default_factory=list)
    missing_reason: str | None = None


class RecommendationExplanationResponse(BaseModel):
    summary: str
    facts_used: list[str] = Field(default_factory=list)
    missing_core_facts: list[str] = Field(default_factory=list)
    next_step: str | None = None


class DecisionExplanationResponse(BaseModel):
    """Serializable explanation attached to offers and questions."""

    summary: str
    details: list[str] = Field(default_factory=list)
    facts_used: list[str] = Field(default_factory=list)
    rules_used: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    polish_status: str = "fallback"


class RecommendationResponse(BaseModel):
    billing_account: str
    customer_type: str
    final_offer: OfferResponse | None = None
    eligible_offers: list[OfferResponse] = Field(default_factory=list)
    blocked_offers: list[OfferResponse] = Field(default_factory=list)
    questions: list[QuestionResponse] = Field(default_factory=list)
    facts: dict[str, FactResponse] = Field(default_factory=dict)
    decision_trace: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    workflow_stage: WorkflowStage | None = None
    explanation: RecommendationExplanationResponse | None = None
    source_documents: list[str] = Field(default_factory=list)
    suppression_reasons: list[str] = Field(default_factory=list)
    routing_stage: str | None = None
    current_enrollment_detected: bool = False


class EvaluateAccountRequest(BaseModel):
    billing_account: str
    user_answers: dict[str, Any] = Field(default_factory=dict)
    declined_programs: list[str] = Field(default_factory=list)


class ChatMessageResponse(BaseModel):
    role: str
    content: str
    timestamp: str


class CreateThreadResponse(BaseModel):
    thread_id: str
    state: "ChatThreadStateResponse"


class ChatMessageRequest(BaseModel):
    message: str


class DeclineProgramRequest(BaseModel):
    program_id: str


class SavedAnswersResponse(BaseModel):
    """Saved customer answer overlays for one billing account."""

    billing_account: str
    answers: dict[str, Any] = Field(default_factory=dict)


class ResetAnswersResponse(BaseModel):
    """Result of a saved-answer reset operation."""

    billing_account: str | None = None
    deleted_count: int


class ChatThreadStateResponse(BaseModel):
    thread_id: str
    billing_account: str | None = None
    customer_type: str | None = None
    messages: list[ChatMessageResponse] = Field(default_factory=list)
    user_answers: dict[str, Any] = Field(default_factory=dict)
    pending_questions: list[QuestionResponse] = Field(default_factory=list)
    current_question: QuestionResponse | None = None
    decision_result: RecommendationResponse | None = None
    declined_programs: list[str] = Field(default_factory=list)
    status_phase: str = "awaiting_account"


class BatchArtifactsResponse(BaseModel):
    excel_path: str
    json_path: str


class BatchSummaryResponse(BaseModel):
    total_accounts: int
    residential_accounts: int
    commercial_accounts: int
    accounts_with_final_offer: int


class BatchRunResponse(BaseModel):
    run_id: str
    created_at: str
    summary: BatchSummaryResponse
    artifacts: BatchArtifactsResponse


def serialize_fact(fact: FactValue) -> FactResponse:
    value = fact.value
    if isinstance(value, datetime):
        value = value.isoformat()
    return FactResponse(
        fact_id=fact.fact_id,
        value=value,
        value_type=fact.value_type,
        source=fact.source.value,
        confidence=fact.confidence.value,
        evidence=list(fact.evidence),
        missing_reason=fact.missing_reason,
    )


def serialize_offer(offer: OfferDecision) -> OfferResponse:
    return OfferResponse(
        program_id=offer.program_id,
        display_name=offer.display_name,
        status=offer.status.value,
        rank=offer.rank,
        confidence=offer.confidence.value,
        reason_codes=list(offer.reason_codes),
        blocking_facts=list(offer.blocking_facts),
        missing_facts=list(offer.missing_facts),
        evidence=list(offer.evidence),
        metadata=dict(offer.metadata),
        explanation=serialize_decision_explanation(offer.explanation)
        if offer.explanation
        else None,
    )


def serialize_question(question: DecisionQuestion) -> QuestionResponse:
    return QuestionResponse(
        question_id=question.question_id,
        prompt=question.prompt,
        expected_fact=question.expected_fact,
        candidate_programs=list(question.candidate_programs),
        priority=question.priority,
        source=question.source,
        answer_options=[
            AnswerOptionResponse(
                label=option.label,
                value=option.value,
                description=option.description,
            )
            for option in question.answer_options
        ],
        explanation=serialize_decision_explanation(question.explanation)
        if question.explanation
        else None,
    )


def serialize_decision_explanation(
    explanation: DecisionExplanation,
) -> DecisionExplanationResponse:
    """Convert a domain explanation dataclass into the API response shape."""
    return DecisionExplanationResponse(
        summary=explanation.summary,
        details=list(explanation.details),
        facts_used=list(explanation.facts_used),
        rules_used=list(explanation.rules_used),
        blockers=list(explanation.blockers),
        source_documents=list(explanation.source_documents),
        polish_status=explanation.polish_status,
    )


def serialize_explanation(
    explanation: RecommendationExplanation,
) -> RecommendationExplanationResponse:
    return RecommendationExplanationResponse(
        summary=explanation.summary,
        facts_used=list(explanation.facts_used),
        missing_core_facts=list(explanation.missing_core_facts),
        next_step=explanation.next_step,
    )


def serialize_recommendation(result: DecisionResult) -> RecommendationResponse:
    return RecommendationResponse(
        billing_account=result.billing_account,
        customer_type=result.customer_type,
        final_offer=serialize_offer(result.final_offer) if result.final_offer else None,
        eligible_offers=[serialize_offer(offer) for offer in result.eligible_offers],
        blocked_offers=[serialize_offer(offer) for offer in result.blocked_offers],
        questions=[serialize_question(question) for question in result.questions],
        facts={
            fact_id: serialize_fact(fact)
            for fact_id, fact in result.facts.items()
            if not fact_id.startswith("_")
        },
        decision_trace=list(result.decision_trace),
        flags=list(result.flags),
        workflow_stage=result.workflow_stage,
        explanation=serialize_explanation(result.explanation) if result.explanation else None,
        source_documents=list(result.source_documents),
        suppression_reasons=list(result.suppression_reasons),
        routing_stage=result.routing_stage,
        current_enrollment_detected=result.current_enrollment_detected,
    )
