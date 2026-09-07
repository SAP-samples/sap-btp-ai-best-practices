"""AI review models for question-level assessment verification results."""

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from app.models.language import DEFAULT_LANGUAGE
from app.services.customer_class_scope import CustomerClass, DEFAULT_CUSTOMER_CLASS


AnswerDecision = Literal[
    "keep_selected",
    "select",
    "low_confidence",
    "unsupported",
    "unclear",
]
"""Allowed AI decisions for an individual answer item.

``low_confidence`` is a selected recommendation used only when stronger
higher-level evidence minimally implies a prerequisite answer, but the
prerequisite is not directly evidenced by the uploaded documents.
"""


class EvidenceRef(BaseModel):
    """Evidence reference extracted from a reviewed source document.

    Inputs:
        Field values describing a source snippet and optional location metadata.

    Outputs:
        A validated evidence reference with confidence bounds enforced and empty
        support or contradiction lists when omitted.

    Attributes:
        evidence_ref_id: Stable identifier used to link decisions to this evidence.
        file_name: Name of the source file where the evidence was found.
        document_type: Type or category of source document reviewed.
        snippet: Short evidence excerpt used by the model.
        supports_answer_item_ids: Answer item IDs supported by the evidence.
        contradicts_answer_item_ids: Answer item IDs contradicted by the evidence.
        confidence: Model confidence in the evidence interpretation, from 0 to 1.
        page: Optional page number for paginated documents.
        sheet_name: Optional sheet name for spreadsheet documents.
        table_name: Optional table name or identifier containing the evidence.
        row_number: Optional row number containing the evidence.
        column_name: Optional column name or index containing the evidence.
        section_label: Optional document section containing the evidence.
    """

    evidence_ref_id: str
    file_name: str
    document_type: str
    snippet: str
    supports_answer_item_ids: list[str] = Field(default_factory=list)
    contradicts_answer_item_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    page: int | None = None
    sheet_name: str | None = None
    table_name: str | None = None
    row_number: int | None = None
    column_name: str | int | None = None
    section_label: str | None = None


class AnswerItemDecision(BaseModel):
    """AI decision for a single answer item within a question review.

    Inputs:
        Field values describing the model decision for one answer item.

    Outputs:
        A validated answer item decision with constrained decision values and
        confidence bounds enforced.

    Attributes:
        answer_item_id: Identifier of the answer item being evaluated.
        decision: Model decision describing whether the item should be kept,
            selected, selected as a low-confidence prerequisite, marked
            unsupported, or marked unclear.
        confidence: Model confidence in the decision, from 0 to 1.
        rationale: Short explanation for the decision.
        evidence_ref_ids: Evidence reference IDs used to justify the decision.
    """

    answer_item_id: str
    decision: AnswerDecision
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_ref_ids: list[str] = Field(default_factory=list)


class LevelReviewResult(BaseModel):
    """Review result for all answer items belonging to one maturity level.

    Inputs:
        Field values describing the review status, reasoning, decisions, and
        evidence for one level.

    Outputs:
        A validated level review result with level bounds enforced.

    Attributes:
        level: Maturity level being reviewed, constrained to values 1-5.
        level_status: Summary status for this level after reviewing all items.
        level_reasoning: Explanation of why the level has the reported status.
        answer_item_decisions: Decisions for answer items in this level.
        level_evidence_refs: Evidence reference IDs relevant to this level.
    """

    level: int = Field(ge=1, le=5)
    level_status: str
    level_reasoning: str
    answer_item_decisions: list[AnswerItemDecision] = Field(default_factory=list)
    level_evidence_refs: list[str] = Field(default_factory=list)


class UsageMetadata(BaseModel):
    """Token usage metadata returned with a question review.

    Inputs:
        Optional token counters and provider detail values when they are
        available from the model response.

    Outputs:
        A strict usage object accepted by OpenAI Responses structured-output
        JSON schema validation.

    Attributes:
        input_tokens: Number of input tokens when reported.
        output_tokens: Number of output tokens when reported.
        total_tokens: Total token count when reported.
        provider: Optional provider or deployment label.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider: str | None = None


EvidenceRelevance = Literal[
    "support",
    "contradiction",
    "background_only",
    "unrelated",
    "incomplete_but_useful",
]
"""Allowed relevance labels for retrieved evidence chunks."""

RetrievalStopReason = Literal[
    "needs_more_evidence",
    "sufficient_evidence",
    "no_refined_queries",
    "max_rounds",
]
"""Allowed compact stop-reason codes for RAG retrieval rounds."""

_RETRIEVAL_STOP_REASON_CODES: set[str] = {
    "needs_more_evidence",
    "sufficient_evidence",
    "no_refined_queries",
    "max_rounds",
}


def normalize_retrieval_stop_reason(
    stop_reason: object,
    sufficient_for_final_review: bool | None = None,
) -> RetrievalStopReason | None:
    """Normalize model stop-reason text into a compact operational code.

    Inputs:
        stop_reason: Raw model stop reason, expected to be one of the supported
            short codes but tolerated as legacy/free-text output.
        sufficient_for_final_review: Optional sufficiency flag from the same
            RAG assessment, used to disambiguate prose reasons.

    Outputs:
        RetrievalStopReason | None: HANA-safe short code, or ``None`` when no
        stop reason was supplied and no sufficiency signal is available.
    """
    if stop_reason is None:
        return "sufficient_evidence" if sufficient_for_final_review is True else None

    reason_text = str(stop_reason).strip()
    if not reason_text:
        return "sufficient_evidence" if sufficient_for_final_review is True else None

    normalized_code = reason_text.lower().replace("-", "_").replace(" ", "_")
    if normalized_code in _RETRIEVAL_STOP_REASON_CODES:
        return cast(RetrievalStopReason, normalized_code)

    reason_lower = reason_text.lower()
    if "sufficient" in reason_lower or (
        sufficient_for_final_review is True and "final review" in reason_lower
    ):
        return "sufficient_evidence"
    if "no" in reason_lower and (
        "refined" in reason_lower or "query" in reason_lower
    ):
        return "no_refined_queries"
    if "max" in reason_lower and "round" in reason_lower:
        return "max_rounds"
    if sufficient_for_final_review is False or any(
        marker in reason_lower
        for marker in ("need", "gap", "more evidence", "retrieve")
    ):
        return "needs_more_evidence"
    return None


class RetrievedEvidenceDecision(BaseModel):
    """Model assessment of one retrieved evidence chunk.

    Inputs:
        Field values describing relevance, related answer items, rationale, and
        confidence for a retrieved chunk.

    Outputs:
        Validated evidence assessment used by the agentic RAG loop.
    """

    model_config = ConfigDict(extra="forbid")

    source_chunk_id: str
    relevance: EvidenceRelevance
    related_answer_item_ids: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class RetrievalRoundAssessment(BaseModel):
    """Structured assessment returned after one RAG retrieval round.

    Inputs:
        Field values describing accepted evidence, remaining gaps, refined
        queries, and whether final review can proceed.

    Outputs:
        Validated assessment used to continue or stop the RAG loop.
    """

    model_config = ConfigDict(extra="forbid")

    accepted_evidence: list[RetrievedEvidenceDecision] = Field(default_factory=list)
    rejected_evidence: list[RetrievedEvidenceDecision] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    refined_queries: list[str] = Field(default_factory=list)
    sufficient_for_final_review: bool
    stop_reason: RetrievalStopReason | None = None

    @field_validator("stop_reason", mode="before")
    @classmethod
    def normalize_stop_reason(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> RetrievalStopReason | None:
        """Normalize ``stop_reason`` before enum validation.

        Inputs:
            value: Raw stop reason supplied by the model or tests.
            info: Pydantic validation context containing prior field values.

        Outputs:
            RetrievalStopReason | None: Compact stop-reason code accepted by the
            structured schema and HANA persistence boundary.
        """

        sufficient_for_final_review = info.data.get("sufficient_for_final_review")
        if not isinstance(sufficient_for_final_review, bool):
            sufficient_for_final_review = None
        return normalize_retrieval_stop_reason(
            stop_reason=value,
            sufficient_for_final_review=sufficient_for_final_review,
        )


class QuestionReviewResult(BaseModel):
    """Complete AI review output for one assessment question.

    Inputs:
        Field values describing question-level model output, level results,
        evidence, usage metadata, and warnings.

    Outputs:
        A validated question review result preserving level-grouped decisions and
        evidence references.

    Attributes:
        question_id: Identifier of the reviewed assessment question.
        model: Name of the non-mini AI model that generated the review.
        overall_status: Question-level status after evaluating every level.
        highest_supported_level: Optional highest maturity level supported by
            evidence, constrained to values 1-5 when present.
        current_selected_answer_item_ids: Answer item IDs selected before review.
        verified_selected_answer_item_ids: Answer item IDs verified after review.
        level_results: Level-grouped review results and answer item decisions.
        evidence_refs: Evidence references used across the question review.
        usage: Token or provider usage metadata returned by the model.
        warnings: Non-fatal warnings produced during review.
    """

    question_id: str
    model: str
    overall_status: str
    highest_supported_level: int | None = Field(default=None, ge=1, le=5)
    current_selected_answer_item_ids: list[str] = Field(default_factory=list)
    verified_selected_answer_item_ids: list[str] = Field(default_factory=list)
    level_results: list[LevelReviewResult] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    usage: UsageMetadata = Field(default_factory=UsageMetadata)
    warnings: list[str] = Field(default_factory=list)


class ReviewJobResponse(BaseModel):
    """Response returned when an AI review job is created or queried.

    Inputs:
        Field values describing the job identifier, current status, and task count.

    Outputs:
        A validated review job response.

    Attributes:
        job_id: Stable identifier for the review job.
        language: Language used by the background worker for generated
            user-facing review text.
        status: Current job status.
        task_count: Number of review tasks included in the job.
    """

    job_id: str
    language: str = "en"
    status: str
    task_count: int


class BatchReviewJobResponse(ReviewJobResponse):
    """Response returned when a global batch AI review job is created.

    Inputs:
        Field values describing the batch job identifier, language, status,
        number of question tasks, and number of deduplicated documents.

    Outputs:
        A validated batch review job response for API clients.

    Attributes:
        document_count: Number of unique documents linked to the batch job after
            content-hash deduplication.
    """

    document_count: int


class CorpusReviewJobRequest(BaseModel):
    """Request to analyze selected questions against the document corpus.

    Inputs:
        API JSON body from the Assessment page.

    Outputs:
        Validated request used to create question review tasks.
    """

    assessment_id: str
    dimension: str
    question_ids: list[str]
    current_answers: dict[str, list[str]] = Field(default_factory=dict)
    language: str = DEFAULT_LANGUAGE
    customer_class: CustomerClass = DEFAULT_CUSTOMER_CLASS


class CorpusBatchReviewJobRequest(BaseModel):
    """Request to analyze every framework question against the document corpus.

    Inputs:
        API JSON body from the all-question Assessment action.

    Outputs:
        Validated request used to create one review task per framework question.
    """

    assessment_id: str
    current_answers: dict[str, list[str]] = Field(default_factory=dict)
    language: str = DEFAULT_LANGUAGE
    customer_class: CustomerClass = DEFAULT_CUSTOMER_CLASS


class ReviewResetResponse(BaseModel):
    """Response returned after clearing stored AI review state for a dimension.

    Inputs:
        Field values identifying the cleared assessment/dimension scope and the
        number of persisted job/task rows removed.

    Outputs:
        A validated reset response suitable for UI reset feedback.

    Attributes:
        assessment_id: Assessment instance whose review state was cleared.
        dimension: Assessment dimension cleared by the reset operation.
        deleted_job_count: Number of dimension review jobs removed.
        deleted_task_count: Number of question review tasks removed.
    """

    assessment_id: str
    dimension: str
    deleted_job_count: int
    deleted_task_count: int


class ReviewTaskStatus(BaseModel):
    """Status, progress telemetry, and optional result or failure detail for one question review task.

    Inputs:
        Field values describing the persisted worker state for one task.

    Outputs:
        A validated task status payload suitable for polling clients.

    Attributes:
        task_id: Stable identifier for the question task.
        question_id: Framework question reviewed by this task.
        dimension: Optional framework dimension for global batch status rows.
        status: Current worker status for the task.
        result: Structured AI review result when the task has completed.
        result_requires_rerun: Whether the stored result uses a retired contract
            and must not be applied to questionnaire answers.
        error_message: Optional error details when task processing failed.
        error_code: Optional machine-readable error code for categorized failures.
        progress_message: Human-readable progress message from the active worker.
        rag_call_count: Number of RAG retrieval rounds completed so far.
        query_count: Total queries issued across all retrieval rounds.
        retrieved_chunk_count: Total evidence chunks retrieved so far.
        lease_owner: Identifier of the worker instance holding the task lease.
        lease_expires_at: ISO timestamp when the current lease expires.
        updated_at: ISO timestamp of the last task status update.
    """

    task_id: str
    question_id: str
    dimension: str | None = None
    status: str
    result: QuestionReviewResult | None = None
    result_requires_rerun: bool = False
    error_message: str | None = None
    error_code: str | None = None
    progress_message: str | None = None
    rag_call_count: int = 0
    query_count: int = 0
    retrieved_chunk_count: int = 0
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    updated_at: str | None = None


class ReviewJobStatusResponse(BaseModel):
    """Status payload returned when polling an AI review job, with optional batch phase.

    Inputs:
        Field values describing aggregate job progress and per-question task
        statuses.

    Outputs:
        A validated job status payload with completed results attached.

    Attributes:
        job_id: Stable identifier for the review job.
        language: Language used by the background worker for generated
            user-facing review text.
        status: Derived job status based on its task states.
        task_count: Number of question tasks created for the job.
        completed_count: Number of tasks with completed AI review results.
        failed_count: Number of tasks that failed processing.
        batch_phase: Current high-level batch phase (indexing, reviewing, etc.).
        indexed_chunk_count: Number of evidence chunks indexed so far.
        document_count: Number of uploaded batch documents linked to this job.
        processed_document_count: Number of documents processed for the current
            document phase.
        active_question_id: Question currently being reviewed by the batch worker.
        active_dimension: Dimension of the question currently being reviewed.
        tasks: Per-question task statuses and completed results.
    """

    job_id: str
    language: str = "en"
    status: str
    task_count: int
    completed_count: int
    failed_count: int = 0
    batch_phase: str | None = None
    indexed_chunk_count: int = 0
    document_count: int = 0
    processed_document_count: int = 0
    active_question_id: str | None = None
    active_dimension: str | None = None
    tasks: list[ReviewTaskStatus] = Field(default_factory=list)
