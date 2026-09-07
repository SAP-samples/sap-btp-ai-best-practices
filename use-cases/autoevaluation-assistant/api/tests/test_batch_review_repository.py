"""Tests for batch upload AI review repository behavior."""

from __future__ import annotations

from app.models.ai_review import QuestionReviewResult
from app.models.assessment import AnswerItem, AssessmentQuestion
from app.services.ai_review_repository.memory import InMemoryAiReviewRepository


def _question(
    question_id: str,
    dimension: str,
    text: str,
) -> AssessmentQuestion:
    """Create a minimal assessment question fixture for batch tests.

    Inputs:
        question_id: Question identifier persisted on the batch task.
        dimension: Canonical framework dimension.
        text: Question text used by review prompts.

    Outputs:
        AssessmentQuestion: Valid framework question with one answer item.
    """
    return AssessmentQuestion(
        question_id=question_id,
        dimension=dimension,
        section="Governance",
        question=text,
        explanation="Review uploaded evidence.",
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


def test_create_batch_job_deduplicates_documents_and_creates_all_question_tasks() -> None:
    """Verify batch uploads are content-deduplicated and fan out to all questions.

    Inputs:
        None. The test seeds two framework questions and uploads duplicate
        document content under different names.

    Outputs:
        None. Assertions confirm one stored document, two pending batch question
        tasks, and no legacy question task leakage.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
        _question("Q.RCG.01.01", "Risk & Control Governance", "Are controls defined?"),
    ]

    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={"Q.STR.01.01": ["Q.STR.01.01-L1-001"]},
        documents=[
            {
                "file_name": "strategy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF same evidence",
            },
            {
                "file_name": "strategy-copy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF same evidence",
            },
        ],
        language="it",
    )

    batch_status = repository.get_batch_job_status(job.job_id)

    assert job.task_count == 2
    assert job.document_count == 1
    assert batch_status.language == "it"
    assert batch_status.task_count == 2
    assert sorted(task.question_id for task in batch_status.tasks) == [
        "Q.RCG.01.01",
        "Q.STR.01.01",
    ]
    assert len(repository.batch_documents) == 1
    assert repository.lease_next_task(worker_id="legacy-worker") is None


def test_batch_job_leasing_and_results_are_separate_from_dimension_jobs() -> None:
    """Verify batch jobs have their own lease and result lifecycle.

    Inputs:
        None. The test creates one batch job, leases it, then completes one
        question task.

    Outputs:
        None. Assertions confirm batch status polling sees the result and the
        batch job lease is not exposed as a legacy question task.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={"Q.STR.01.01": []},
        documents=[
            {
                "file_name": "strategy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF strategy evidence",
            }
        ],
    )

    leased_job = repository.lease_next_batch_job(worker_id="batch-worker")
    assert leased_job is not None
    assert leased_job["job_id"] == job.job_id
    assert repository.lease_next_task(worker_id="legacy-worker") is None

    task_id = next(iter(repository.batch_question_tasks))
    repository.save_batch_question_result(
        task_id=task_id,
        result=QuestionReviewResult(
            question_id="Q.STR.01.01",
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=["Q.STR.01.01-L1-001"],
        ),
    )
    repository.complete_batch_job(job_id=job.job_id, worker_id="batch-worker")

    status = repository.get_batch_job_status(job.job_id)

    assert status.status == "completed"
    assert status.completed_count == 1
    assert status.tasks[0].result is not None
    assert status.tasks[0].result.overall_status == "supported"


def test_batch_indexing_completion_enables_question_leases() -> None:
    """Verify question tasks are not leased until shared indexing completes."""
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={"Q.STR.01.01": []},
        documents=[{"file_name": "strategy.pdf", "content": b"%PDF"}],
    )

    assert repository.lease_next_batch_question_task("worker-q") is None
    leased_job = repository.lease_next_batch_indexing_job("worker-index")
    assert leased_job is not None

    repository.complete_batch_indexing(
        job_id=job.job_id,
        worker_id="worker-index",
        indexed_chunk_count=3,
    )
    leased_task = repository.lease_next_batch_question_task("worker-q")

    assert leased_task is not None
    assert leased_task["question_id"] == "Q.STR.01.01"
    assert repository.batch_jobs[job.job_id]["status"] == "reviewing_questions"
    assert repository.batch_jobs[job.job_id]["indexed_chunk_count"] == 3


def test_batch_question_progress_and_failure_are_isolated() -> None:
    """Verify one failed batch question does not fail sibling tasks."""
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
        _question("Q.RCG.01.01", "Risk & Control Governance", "Are controls defined?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={},
        documents=[{"file_name": "strategy.pdf", "content": b"%PDF"}],
    )
    leased_job = repository.lease_next_batch_indexing_job("worker-index")
    assert leased_job is not None
    repository.complete_batch_indexing(job.job_id, "worker-index", indexed_chunk_count=4)

    first = repository.lease_next_batch_question_task("worker-q")
    assert first is not None
    repository.update_batch_question_progress(
        task_id=first["task_id"],
        worker_id="worker-q",
        status="retrieving_evidence",
        progress_message="RAG call 1/2 retrieved 8 chunks.",
        rag_call_count=1,
        query_count=6,
        retrieved_chunk_count=8,
    )
    repository.save_batch_question_failure(
        task_id=first["task_id"],
        worker_id="worker-q",
        error_code="invalid_structured_output",
        error_message="Model returned invalid JSON.",
    )

    status = repository.get_batch_job_status(job.job_id)
    failed = [task for task in status.tasks if task.status == "failed"]
    pending = [task for task in status.tasks if task.status == "pending"]

    assert len(failed) == 1
    assert len(pending) == 1
    assert status.status == "reviewing_questions"
    assert status.tasks[0].error_code in {None, "invalid_structured_output"}


def test_batch_job_completes_when_last_question_task_finishes() -> None:
    """Verify per-question processing closes the parent batch job.

    Inputs:
        None. The test creates two review-ready batch question tasks and saves a
        result for each one through independent leases.

    Outputs:
        None. Assertions confirm polling sees a terminal parent job status so
        stale UI pollers can stop.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
        _question("Q.RCG.01.01", "Risk & Control Governance", "Are controls defined?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={},
        documents=[{"file_name": "strategy.pdf", "content": b"%PDF"}],
    )
    leased_job = repository.lease_next_batch_indexing_job("worker-index")
    assert leased_job is not None
    repository.complete_batch_indexing(job.job_id, "worker-index", indexed_chunk_count=4)

    first = repository.lease_next_batch_question_task("worker-q-1")
    assert first is not None
    repository.save_batch_question_result(
        task_id=first["task_id"],
        worker_id="worker-q-1",
        result=QuestionReviewResult(
            question_id=first["question_id"],
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        ),
    )
    assert repository.get_batch_job_status(job.job_id).status == "reviewing_questions"

    second = repository.lease_next_batch_question_task("worker-q-2")
    assert second is not None
    repository.save_batch_question_result(
        task_id=second["task_id"],
        worker_id="worker-q-2",
        result=QuestionReviewResult(
            question_id=second["question_id"],
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        ),
    )

    status = repository.get_batch_job_status(job.job_id)

    assert status.status == "completed"
    assert status.batch_phase == "completed"
    assert status.completed_count == 2
    assert repository.batch_jobs[job.job_id]["status"] == "completed"


def test_batch_job_partially_fails_when_last_question_task_fails() -> None:
    """Verify mixed task outcomes close a batch job as partial_failed.

    Inputs:
        None. The test completes one question and fails the remaining leased
        question task.

    Outputs:
        None. Assertions confirm polling gets a terminal partial failure status.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
        _question("Q.RCG.01.01", "Risk & Control Governance", "Are controls defined?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={},
        documents=[{"file_name": "strategy.pdf", "content": b"%PDF"}],
    )
    leased_job = repository.lease_next_batch_indexing_job("worker-index")
    assert leased_job is not None
    repository.complete_batch_indexing(job.job_id, "worker-index", indexed_chunk_count=4)

    first = repository.lease_next_batch_question_task("worker-q-1")
    assert first is not None
    repository.save_batch_question_result(
        task_id=first["task_id"],
        worker_id="worker-q-1",
        result=QuestionReviewResult(
            question_id=first["question_id"],
            model="gpt-5.4",
            overall_status="supported",
            current_selected_answer_item_ids=[],
            verified_selected_answer_item_ids=[],
        ),
    )
    second = repository.lease_next_batch_question_task("worker-q-2")
    assert second is not None
    repository.save_batch_question_failure(
        task_id=second["task_id"],
        worker_id="worker-q-2",
        error_code="invalid_structured_output",
        error_message="Model returned invalid JSON.",
    )

    status = repository.get_batch_job_status(job.job_id)

    assert status.status == "partial_failed"
    assert status.batch_phase == "partial_failed"
    assert status.completed_count == 1
    assert status.failed_count == 1


def test_clear_dimension_review_state_removes_batch_rows_and_orphan_documents() -> None:
    """Verify dimension reset cleans completed batch rows for that dimension.

    Inputs:
        None. The test creates a single-dimension batch job and adds shared
        indexing artifacts that should be removed when the dimension is reset.

    Outputs:
        None. Assertions confirm batch job, task, chunks, retrieval rounds, and
        now-orphaned document metadata are removed.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={"Q.STR.01.01": []},
        documents=[
            {
                "file_name": "strategy.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF strategy evidence",
            }
        ],
    )
    task_id = next(iter(repository.batch_question_tasks))
    repository.batch_document_chunks[job.job_id] = [{"chunk_id": "chunk-1"}]
    repository.batch_retrieval_rounds[job.job_id] = [
        {"task_id": task_id, "round_number": 1}
    ]
    repository.batch_extractions[job.job_id] = [{"extraction_id": "extraction-1"}]
    repository.batch_extracted_blocks[job.job_id] = [{"block_row_id": "block-1"}]

    reset = repository.clear_dimension_review_state(
        assessment_id="assessment-1",
        dimension="Strategy",
    )

    assert reset.deleted_job_count == 1
    assert reset.deleted_task_count == 1
    assert repository.batch_jobs == {}
    assert repository.batch_question_tasks == {}
    assert repository.batch_job_documents == {}
    assert repository.batch_document_chunks == {}
    assert repository.batch_retrieval_rounds == {}
    assert repository.batch_extractions == {}
    assert repository.batch_extracted_blocks == {}
    assert repository.batch_documents == {}


def test_clear_assessment_review_state_removes_multi_dimension_batch_job() -> None:
    """Verify complete reset removes a global batch job across dimensions.

    Inputs:
        None. The test creates one batch job with Strategy and Risk tasks plus
        shared indexing artifacts.

    Outputs:
        None. Assertions confirm the parent batch job and every child process
        artifact are removed, so the worker cannot resume stale batch work.
    """
    repository = InMemoryAiReviewRepository()
    repository.framework_questions = [
        _question("Q.STR.01.01", "Strategy", "Is strategy documented?"),
        _question("Q.RCG.01.01", "Risk & Control Governance", "Are controls defined?"),
    ]
    job = repository.create_batch_job(
        assessment_id="assessment-1",
        current_answers={"Q.STR.01.01": [], "Q.RCG.01.01": []},
        documents=[
            {
                "file_name": "global.pdf",
                "content_type": "application/pdf",
                "content": b"%PDF global evidence",
            }
        ],
    )
    repository.batch_document_chunks[job.job_id] = [{"chunk_id": "chunk-1"}]
    repository.batch_retrieval_rounds[job.job_id] = [
        {"task_id": task_id, "round_number": 1}
        for task_id in repository.batch_question_tasks
    ]
    repository.batch_extractions[job.job_id] = [{"extraction_id": "extraction-1"}]
    repository.batch_extracted_blocks[job.job_id] = [{"block_row_id": "block-1"}]

    reset = repository.clear_assessment_review_state(assessment_id="assessment-1")

    assert reset.deleted_job_count == 1
    assert reset.deleted_task_count == 2
    assert repository.batch_jobs == {}
    assert repository.batch_question_tasks == {}
    assert repository.batch_question_results == {}
    assert repository.batch_job_documents == {}
    assert repository.batch_document_chunks == {}
    assert repository.batch_retrieval_rounds == {}
    assert repository.batch_extractions == {}
    assert repository.batch_extracted_blocks == {}
    assert repository.batch_documents == {}


def test_save_batch_rag_tool_call_records_query_and_chunk_counts() -> None:
    """Verify RAG tool calls are auditable without a model assessment step."""
    repository = InMemoryAiReviewRepository()
    tool_call_id = repository.save_batch_rag_tool_call(
        job_id="batch-job-1",
        task_id="batch-task-1",
        rag_call_number=1,
        queries_json=[{"query": "strategy", "answer_item_id": "A1", "level": 1}],
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        matched_queries_json={"chunk-1": [{"query": "strategy"}]},
        query_count=1,
        retrieved_chunk_count=2,
        duration_ms=120,
        stop_reason="tool_result",
    )

    assert tool_call_id.startswith("batch-retrieval-round-")
    assert repository.batch_retrieval_rounds["batch-job-1"][0]["accepted_evidence_json"]["query_count"] == 1
