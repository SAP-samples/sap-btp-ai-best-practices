"""Tests for assessment and AI review Pydantic contracts."""

from app.models.ai_review import (
    AnswerItemDecision,
    EvidenceRef,
    LevelReviewResult,
    QuestionReviewResult,
    RetrievalRoundAssessment,
    ReviewJobStatusResponse,
    ReviewTaskStatus,
)
from app.models.assessment import AnswerItem, AssessmentQuestion
from pydantic import ValidationError
import pytest


def test_question_review_result_preserves_level_grouped_output_and_evidence_refs() -> None:
    """Verify review output keeps level decisions and evidence references grouped.

    Inputs:
        None. The test creates a review result with two levels and two evidence
        references.

    Outputs:
        None. The assertions confirm that level-grouped decisions, evidence
        reference IDs, supporting answer item IDs, and usage metadata survive
        Pydantic validation.
    """
    review_result = QuestionReviewResult(
        question_id="Q-001",
        model="gpt-5.4",
        overall_status="partially_supported",
        highest_supported_level=2,
        current_selected_answer_item_ids=["Q-001-L2-1"],
        verified_selected_answer_item_ids=["Q-001-L1-1"],
        level_results=[
            LevelReviewResult(
                level=1,
                level_status="supported",
                level_reasoning="The submitted policy clearly supports level 1.",
                answer_item_decisions=[
                    AnswerItemDecision(
                        answer_item_id="Q-001-L1-1",
                        decision="select",
                        confidence=0.92,
                        rationale="The cited policy explicitly supports this item.",
                        evidence_ref_ids=["EV-1"],
                    )
                ],
                level_evidence_refs=["EV-1"],
            ),
            LevelReviewResult(
                level=2,
                level_status="unsupported",
                level_reasoning="Level 2 requires a measured process that is absent.",
                answer_item_decisions=[
                    AnswerItemDecision(
                        answer_item_id="Q-001-L2-1",
                        decision="unsupported",
                        confidence=0.81,
                        rationale="The selected item is not supported by evidence.",
                        evidence_ref_ids=["EV-2"],
                    )
                ],
                level_evidence_refs=["EV-2"],
            ),
        ],
        evidence_refs=[
            EvidenceRef(
                evidence_ref_id="EV-1",
                file_name="policy.pdf",
                document_type="policy",
                snippet="The company maintains a documented policy.",
                supports_answer_item_ids=["Q-001-L1-1"],
                confidence=0.93,
                page=4,
                section_label="Governance",
            ),
            EvidenceRef(
                evidence_ref_id="EV-2",
                file_name="procedure.xlsx",
                document_type="procedure",
                snippet="No measured process is documented in the procedure table.",
                contradicts_answer_item_ids=["Q-001-L2-1"],
                confidence=0.78,
                sheet_name="Controls",
                table_name="Maturity evidence",
                row_number=7,
                column_name="Finding",
            ),
        ],
        usage={"input_tokens": 1200, "output_tokens": 350},
        warnings=["Level 2 evidence was incomplete."],
    )

    assert [level.level for level in review_result.level_results] == [1, 2]
    assert review_result.level_results[0].level_evidence_refs == ["EV-1"]
    assert review_result.level_results[1].answer_item_decisions[0].decision == (
        "unsupported"
    )
    assert review_result.evidence_refs[0].supports_answer_item_ids == ["Q-001-L1-1"]
    assert review_result.evidence_refs[1].contradicts_answer_item_ids == ["Q-001-L2-1"]
    assert review_result.evidence_refs[0].section_label == "Governance"
    assert review_result.evidence_refs[1].sheet_name == "Controls"
    assert review_result.evidence_refs[1].table_name == "Maturity evidence"
    assert review_result.evidence_refs[1].row_number == 7
    assert review_result.evidence_refs[1].column_name == "Finding"
    assert review_result.usage.input_tokens == 1200


def test_deselect_decision_is_rejected() -> None:
    """Verify new AI review results cannot use legacy deselect semantics.

    Inputs:
        None. The test builds a single answer decision with the retired
        ``deselect`` value.

    Outputs:
        None. Assertions confirm Pydantic rejects the value so new model output
        cannot instruct the UI to unselect an answer item.
    """
    with pytest.raises(ValidationError):
        AnswerItemDecision(
            answer_item_id="Q-001-L1-1",
            decision="deselect",
            confidence=0.75,
            rationale="Legacy output attempted to unselect this answer.",
        )


def test_low_confidence_decision_is_valid_and_selected() -> None:
    """Verify low-confidence prerequisite recommendations serialize as decisions.

    Inputs:
        None. The test creates a review result with one low-confidence decision.

    Outputs:
        None. Assertions confirm the new decision value survives validation and
        JSON serialization as a selectable recommendation.
    """
    review_result = QuestionReviewResult(
        question_id="Q-004",
        model="gpt-5.4",
        overall_status="partially_supported",
        highest_supported_level=3,
        current_selected_answer_item_ids=[],
        verified_selected_answer_item_ids=["Q-004-L1-1"],
        level_results=[
            LevelReviewResult(
                level=1,
                level_status="low_confidence",
                level_reasoning=(
                    "Reasoning and evidence: strategy-plan.pdf section Governance "
                    "supports a level 3 Board approval item, so this prerequisite "
                    "level 1 owner item is minimally implied but not directly stated."
                ),
                answer_item_decisions=[
                    AnswerItemDecision(
                        answer_item_id="Q-004-L1-1",
                        decision="low_confidence",
                        confidence=0.54,
                        rationale=(
                            "The higher-level evidence implies this prerequisite, "
                            "but no document states it directly."
                        ),
                        evidence_ref_ids=["EV-3"],
                    )
                ],
                level_evidence_refs=["EV-3"],
            )
        ],
    )

    serialized = review_result.model_dump()

    assert serialized["level_results"][0]["answer_item_decisions"][0]["decision"] == (
        "low_confidence"
    )


def test_question_review_result_usage_schema_is_strict_for_responses_api() -> None:
    """Verify usage metadata has a strict JSON schema for Responses parsing.

    Inputs:
        None. The test inspects the generated Pydantic JSON schema used by the
        native OpenAI Responses parser.

    Outputs:
        None. Assertions confirm the ``usage`` object forbids arbitrary
        additional properties, which is required by the live Responses API.
    """
    schema = QuestionReviewResult.model_json_schema()
    usage_ref = schema["properties"]["usage"]["$ref"].removeprefix("#/$defs/")
    usage_schema = schema["$defs"][usage_ref]

    assert usage_schema["type"] == "object"
    assert usage_schema["additionalProperties"] is False
    assert {"input_tokens", "output_tokens", "total_tokens", "provider"} <= set(
        usage_schema["properties"]
    )


def test_retrieval_round_assessment_normalizes_stop_reason_to_short_code() -> None:
    """Verify RAG assessment stop reasons are stored as compact codes.

    Inputs:
        None. The test creates one RAG assessment using a prose stop reason like
        the live model returned.

    Outputs:
        None. Assertions confirm the model contract normalizes the prose to the
        HANA-safe operational code.
    """
    assessment = RetrievalRoundAssessment(
        accepted_evidence=[],
        rejected_evidence=[],
        evidence_gaps=["Highest maturity evidence is still incomplete."],
        refined_queries=[],
        sufficient_for_final_review=True,
        stop_reason=(
            "Accepted evidence is sufficient to ground a final review on whether "
            "strategy definition takes risks into account, especially for basic "
            "through advanced formal integration."
        ),
    )

    assert assessment.stop_reason == "sufficient_evidence"


def test_retrieval_round_assessment_schema_is_strict() -> None:
    """Verify RAG evidence assessment has a structured strict schema.

    Inputs:
        None. The test reads the generated Pydantic schema.

    Outputs:
        None. Assertions confirm required RAG assessment fields exist.
    """
    schema = RetrievalRoundAssessment.model_json_schema()
    accepted_items_ref = schema["properties"]["accepted_evidence"]["items"][
        "$ref"
    ].removeprefix("#/$defs/")
    accepted_items_schema = schema["$defs"][accepted_items_ref]
    stop_reason_schema = schema["properties"]["stop_reason"]
    stop_reason_variants = stop_reason_schema.get("anyOf", [stop_reason_schema])
    stop_reason_values = {
        value
        for variant in stop_reason_variants
        for value in variant.get("enum", [])
        if value is not None
    }

    assert schema["additionalProperties"] is False
    assert "accepted_evidence" in schema["properties"]
    assert "rejected_evidence" in schema["properties"]
    assert "evidence_gaps" in schema["properties"]
    assert "refined_queries" in schema["properties"]
    assert "sufficient_for_final_review" in schema["properties"]
    assert {
        "needs_more_evidence",
        "sufficient_evidence",
        "no_refined_queries",
        "max_rounds",
    } <= stop_reason_values
    assert accepted_items_schema["additionalProperties"] is False


def test_assessment_question_carries_answer_items_grouped_by_level() -> None:
    """Verify assessment questions can carry catalog answer items by level.

    Inputs:
        None. The test creates an assessment question with answer items from
        levels 1 and 2.

    Outputs:
        None. The assertions confirm that answer items keep their level,
        ordering, question ID, and text after validation.
    """
    assessment_question = AssessmentQuestion(
        question_id="Q-002",
        dimension="Climate governance",
        section="Policy",
        question="Does the organization maintain climate governance policies?",
        explanation="Review formal policy documents and governance procedures.",
        answer_items=[
            AnswerItem(
                answer_item_id="Q-002-L1-1",
                question_id="Q-002",
                level=1,
                item_index=1,
                text="A basic climate governance policy exists.",
            ),
            AnswerItem(
                answer_item_id="Q-002-L2-1",
                question_id="Q-002",
                level=2,
                item_index=1,
                text="The policy is assigned to accountable business owners.",
            ),
            AnswerItem(
                answer_item_id="Q-002-L2-2",
                question_id="Q-002",
                level=2,
                item_index=2,
                text="The policy is reviewed on a documented cadence.",
            ),
        ],
    )

    answer_items_by_level = {
        level: [
            item.text
            for item in assessment_question.answer_items
            if item.level == level
        ]
        for level in {item.level for item in assessment_question.answer_items}
    }

    assert assessment_question.explanation == (
        "Review formal policy documents and governance procedures."
    )
    assert answer_items_by_level == {
        1: ["A basic climate governance policy exists."],
        2: [
            "The policy is assigned to accountable business owners.",
            "The policy is reviewed on a documented cadence.",
        ],
    }
    assert assessment_question.answer_items[2].item_index == 2


def test_answer_item_rejects_exported_ecomarket_columns() -> None:
    """Verify exported answer columns cannot enter normalized answer items.

    Inputs:
        None. The test passes framework export-only columns into ``AnswerItem``.

    Outputs:
        None. The assertions confirm Pydantic rejects ``default_selected`` and
        ``optional`` as extra fields.
    """
    with pytest.raises(ValidationError) as validation_error:
        AnswerItem(
            answer_item_id="Q-003-L1-1",
            question_id="Q-003",
            level=1,
            item_index=1,
            text="A basic control exists.",
            default_selected=True,
            optional=False,
        )

    rejected_fields = {
        error["loc"][0]
        for error in validation_error.value.errors()
        if error["type"] == "extra_forbidden"
    }

    assert rejected_fields == {"default_selected", "optional"}


def test_review_job_status_accepts_batch_progress_fields() -> None:
    """Verify batch polling can expose progress without breaking old clients."""
    payload = ReviewJobStatusResponse(
        job_id="batch-job-1",
        language="en",
        status="reviewing_questions",
        task_count=2,
        completed_count=1,
        failed_count=0,
        batch_phase="reviewing_questions",
        indexed_chunk_count=42,
        active_question_id="Q.STR.01.01",
        active_dimension="Strategy",
        document_count=18,
        processed_document_count=4,
        tasks=[
            ReviewTaskStatus(
                task_id="batch-task-1",
                question_id="Q.STR.01.01",
                dimension="Strategy",
                status="retrieving_evidence",
                progress_message="RAG call 1/2 retrieved 12 chunks.",
                rag_call_count=1,
                query_count=18,
                retrieved_chunk_count=12,
                error_code=None,
            )
        ],
    )

    assert payload.batch_phase == "reviewing_questions"
    assert payload.tasks[0].rag_call_count == 1
    assert payload.tasks[0].retrieved_chunk_count == 12
    assert payload.document_count == 18
    assert payload.processed_document_count == 4
