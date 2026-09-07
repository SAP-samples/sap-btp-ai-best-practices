"""Manual and corpus question review persistence for the in-memory repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from app.models.ai_review import (
    BatchReviewJobResponse,
    QuestionReviewResult,
    ReviewJobResponse,
    ReviewJobStatusResponse,
    ReviewResetResponse,
    ReviewTaskStatus,
)
from app.models.assessment import AnswerItem, AssessmentDimension, AssessmentQuestion
from app.models.documents import (
    DocumentDownload,
    DocumentIngestionJobResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.models.language import DEFAULT_LANGUAGE, normalize_language
from app.services.document_extractors import ExtractedDocument
from app.services.customer_class_scope import (
    CustomerClass,
    DEFAULT_CUSTOMER_CLASS,
    filter_current_answer_ids,
    max_allowed_level,
)

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    document_content_hash,
)

class MemoryReviewsMixin:
    """Manage in-memory manual review jobs, leases, progress, evidence, and results."""

    def create_dimension_job(
        self,
        assessment_id: str,
        dimension: str,
        current_answers: dict[str, list[str]],
        attachments: dict[str, list[dict[str, Any]]],
        language: str = DEFAULT_LANGUAGE,
    ) -> ReviewJobResponse:
        """Create one review job and pending question tasks for attached files.

        Inputs:
            assessment_id: Identifier of the assessment being reviewed.
            dimension: Assessment framework dimension selected for review.
            current_answers: Current selected answer item IDs keyed by question
                ID.
            attachments: Uploaded file payloads keyed by question ID.
            language: Supported language code used by the worker prompt.

        Outputs:
            ReviewJobResponse: The created job ID, initial status, and number
            of question tasks created.
        """
        language = normalize_language(language)
        job_id = f"job-{uuid4()}"
        task_count = 0

        self.jobs[job_id] = {
            "job_id": job_id,
            "assessment_id": assessment_id,
            "dimension": dimension,
            "language": language,
            "status": "pending",
        }

        for question_id, files in attachments.items():
            if not files:
                continue

            task_id = f"task-{uuid4()}"
            self.tasks[task_id] = {
                "task_id": task_id,
                "job_id": job_id,
                "assessment_id": assessment_id,
                "dimension": dimension,
                "language": language,
                "question_id": question_id,
                "status": "pending",
                "current_selected_answer_item_ids": deepcopy(
                    current_answers.get(question_id, [])
                ),
                "max_allowed_level": 5,
                "lease_owner": None,
                "lease_expires_at": None,
                "progress_message": None,
                "rag_call_count": 0,
                "query_count": 0,
                "retrieved_chunk_count": 0,
                "error_code": None,
                "error_message": None,
                "updated_at": time.time(),
            }
            attachment_rows = []
            for file_payload in files:
                attachment_rows.append(
                    {
                        "attachment_id": f"attachment-{uuid4()}",
                        "file_name": file_payload["file_name"],
                        "content_type": file_payload.get(
                            "content_type",
                            "application/octet-stream",
                        ),
                        "content": file_payload.get("content", b""),
                    }
                )
            self.attachments[task_id] = attachment_rows
            task_count += 1

        self.jobs[job_id]["task_count"] = task_count
        return ReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending",
            task_count=task_count,
        )

    def create_corpus_question_job(
        self,
        assessment_id: str,
        dimension: str,
        question_ids: list[str],
        current_answers: dict[str, list[str]],
        language: str = DEFAULT_LANGUAGE,
        customer_class: CustomerClass = DEFAULT_CUSTOMER_CLASS,
    ) -> ReviewJobResponse:
        """Create selected question tasks that retrieve from the document corpus."""
        language = normalize_language(language)
        allowed_question_ids = set(question_ids)
        questions = [
            question
            for question in self.list_questions(dimension=dimension, language=language)
            if question.question_id in allowed_question_ids
        ]
        if not questions:
            raise ValueError("Question analysis requires at least one valid question ID.")
        scoped_questions = [
            (question, level)
            for question in questions
            if (level := max_allowed_level(question.question_id, customer_class)) > 0
        ]
        if not scoped_questions:
            raise ValueError("Selected questions are not available at this customer level.")
        return self._create_corpus_job(
            assessment_id=assessment_id,
            dimension=dimension,
            questions=scoped_questions,
            current_answers=current_answers,
            language=language,
        )

    def create_corpus_all_questions_job(
        self,
        assessment_id: str,
        current_answers: dict[str, list[str]],
        language: str = DEFAULT_LANGUAGE,
        customer_class: CustomerClass = DEFAULT_CUSTOMER_CLASS,
    ) -> ReviewJobResponse:
        """Create one corpus-backed task for every framework question."""
        language = normalize_language(language)
        questions = self.list_all_questions(language=language)
        if not questions:
            raise ValueError("All-question analysis requires imported framework questions.")
        scoped_questions = [
            (question, level)
            for question in questions
            if (level := max_allowed_level(question.question_id, customer_class)) > 0
        ]
        if not scoped_questions:
            raise ValueError("All-question analysis has no questions available at this customer level.")
        return self._create_corpus_job(
            assessment_id=assessment_id,
            dimension="__ALL__",
            questions=scoped_questions,
            current_answers=current_answers,
            language=language,
        )

    def _create_corpus_job(
        self,
        assessment_id: str,
        dimension: str,
        questions: list[tuple[AssessmentQuestion, int]],
        current_answers: dict[str, list[str]],
        language: str,
    ) -> ReviewJobResponse:
        """Create a unified in-memory review job for supplied questions."""
        job_id = f"job-{uuid4()}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "assessment_id": assessment_id,
            "dimension": dimension,
            "language": language,
            "status": "pending",
        }
        for question, max_level in questions:
            task_id = f"task-{uuid4()}"
            self.tasks[task_id] = {
                "task_id": task_id,
                "job_id": job_id,
                "assessment_id": assessment_id,
                "question_id": question.question_id,
                "dimension": question.dimension,
                "current_selected_answer_item_ids": filter_current_answer_ids(
                    question,
                    deepcopy(current_answers.get(question.question_id, [])),
                    max_level,
                ),
                "max_allowed_level": max_level,
                "status": "pending",
                "lease_owner": None,
                "lease_expires_at": None,
                "retry_count": 0,
                "error_code": None,
                "error_message": None,
                "progress_message": None,
                "rag_call_count": 0,
                "query_count": 0,
                "retrieved_chunk_count": 0,
                "updated_at": time.time(),
            }
        return ReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending",
            task_count=len(questions),
        )

    def get_job_status(self, job_id: str) -> ReviewJobStatusResponse:
        """Return derived progress and completed results for one review job.

        Inputs:
            job_id: Identifier of the in-memory review job to poll.

        Outputs:
            ReviewJobStatusResponse: Aggregate job progress plus per-question
            task statuses and completed AI review results.

        Raises:
            KeyError: Raised when no in-memory job matches ``job_id``.
        """
        if job_id not in self.jobs:
            raise KeyError(f"Unknown AI review job ID: {job_id}")

        task_rows = [
            task for task in self.tasks.values() if task["job_id"] == job_id
        ]
        tasks = [
            ReviewTaskStatus(
                task_id=task["task_id"],
                question_id=task["question_id"],
                dimension=task.get("dimension"),
                status=task["status"],
                result=deepcopy(self.results.get(task["task_id"])),
                error_code=task.get("error_code"),
                error_message=task.get("error_message"),
                progress_message=task.get("progress_message"),
                rag_call_count=int(task.get("rag_call_count", 0) or 0),
                query_count=int(task.get("query_count", 0) or 0),
                retrieved_chunk_count=int(task.get("retrieved_chunk_count", 0) or 0),
                lease_owner=task.get("lease_owner"),
                lease_expires_at=(
                    str(task["lease_expires_at"]) if task.get("lease_expires_at") else None
                ),
                updated_at=str(task["updated_at"]) if task.get("updated_at") else None,
            )
            for task in task_rows
        ]
        task_statuses = [task.status for task in tasks]
        return ReviewJobStatusResponse(
            job_id=job_id,
            language=self.jobs[job_id].get("language", DEFAULT_LANGUAGE),
            status=_derive_job_status(task_statuses, self.jobs[job_id]["status"]),
            task_count=len(tasks),
            completed_count=sum(1 for status in task_statuses if status == "completed"),
            failed_count=sum(1 for status in task_statuses if status == "failed"),
            tasks=tasks,
        )

    def clear_dimension_review_state(
        self,
        assessment_id: str,
        dimension: str,
    ) -> ReviewResetResponse:
        """Delete in-memory AI review jobs and child rows for one dimension.

        Inputs:
            assessment_id: Assessment instance whose AI review state should be
                cleared.
            dimension: Framework dimension selected by the user reset action.

        Outputs:
            ReviewResetResponse: Counts of removed legacy and batch jobs/tasks.
        """
        job_ids = {
            job_id
            for job_id, job in self.jobs.items()
            if job["assessment_id"] == assessment_id and job["dimension"] == dimension
        }
        task_ids = {
            task_id
            for task_id, task in self.tasks.items()
            if task["job_id"] in job_ids
        }

        for task_id in task_ids:
            self.tasks.pop(task_id, None)
            self.attachments.pop(task_id, None)
            self.results.pop(task_id, None)
            self.extractions.pop(task_id, None)
            self.extracted_blocks.pop(task_id, None)
            self.evidence_chunks.pop(task_id, None)
            self.retrieval_rounds.pop(task_id, None)
            self.attempts.pop(task_id, None)
        for job_id in job_ids:
            self.jobs.pop(job_id, None)

        batch_task_ids = {
            task_id
            for task_id, task in self.batch_question_tasks.items()
            if self.batch_jobs.get(task["job_id"], {}).get("assessment_id")
            == assessment_id
            and task["dimension"] == dimension
        }
        affected_batch_job_ids = {
            self.batch_question_tasks[task_id]["job_id"]
            for task_id in batch_task_ids
        }
        for task_id in batch_task_ids:
            self.batch_question_tasks.pop(task_id, None)
            self.batch_question_results.pop(task_id, None)
        for job_id, rounds in list(self.batch_retrieval_rounds.items()):
            remaining_rounds = [
                round_row
                for round_row in rounds
                if round_row.get("task_id") not in batch_task_ids
            ]
            if remaining_rounds:
                self.batch_retrieval_rounds[job_id] = remaining_rounds
            else:
                self.batch_retrieval_rounds.pop(job_id, None)

        empty_batch_job_ids = {
            job_id
            for job_id in affected_batch_job_ids
            if not any(
                task["job_id"] == job_id
                for task in self.batch_question_tasks.values()
            )
        }
        for job_id in empty_batch_job_ids:
            self.batch_jobs.pop(job_id, None)
            self.batch_job_documents.pop(job_id, None)
            self.batch_extractions.pop(job_id, None)
            self.batch_extracted_blocks.pop(job_id, None)
            self.batch_document_chunks.pop(job_id, None)
            self.batch_retrieval_rounds.pop(job_id, None)

        referenced_document_ids = {
            document_id
            for document_ids in self.batch_job_documents.values()
            for document_id in document_ids
        }
        for document_id, document in list(self.batch_documents.items()):
            if (
                document["assessment_id"] == assessment_id
                and document_id not in referenced_document_ids
            ):
                self.batch_documents.pop(document_id, None)

        return ReviewResetResponse(
            assessment_id=assessment_id,
            dimension=dimension,
            deleted_job_count=len(job_ids) + len(empty_batch_job_ids),
            deleted_task_count=len(task_ids) + len(batch_task_ids),
        )

    def clear_assessment_review_state(self, assessment_id: str) -> ReviewResetResponse:
        """Delete all in-memory AI review process state for one assessment.

        Inputs:
            assessment_id: Assessment whose legacy jobs, global batch jobs,
                uploaded documents, extraction rows, chunks, retrieval audit
                rows, tasks, and results should be cleared.

        Outputs:
            ReviewResetResponse: Removed job/task counts across all dimensions.
        """
        job_ids = {
            job_id
            for job_id, job in self.jobs.items()
            if job["assessment_id"] == assessment_id
        }
        task_ids = {
            task_id
            for task_id, task in self.tasks.items()
            if task["job_id"] in job_ids
        }

        for task_id in task_ids:
            self.tasks.pop(task_id, None)
            self.attachments.pop(task_id, None)
            self.results.pop(task_id, None)
            self.extractions.pop(task_id, None)
            self.extracted_blocks.pop(task_id, None)
            self.evidence_chunks.pop(task_id, None)
            self.retrieval_rounds.pop(task_id, None)
            self.attempts.pop(task_id, None)
        for job_id in job_ids:
            self.jobs.pop(job_id, None)

        batch_job_ids = {
            job_id
            for job_id, job in self.batch_jobs.items()
            if job["assessment_id"] == assessment_id
        }
        batch_task_ids = {
            task_id
            for task_id, task in self.batch_question_tasks.items()
            if task["job_id"] in batch_job_ids
        }
        for task_id in batch_task_ids:
            self.batch_question_tasks.pop(task_id, None)
            self.batch_question_results.pop(task_id, None)

        for job_id in batch_job_ids:
            self.batch_jobs.pop(job_id, None)
            self.batch_job_documents.pop(job_id, None)
            self.batch_extractions.pop(job_id, None)
            self.batch_extracted_blocks.pop(job_id, None)
            self.batch_document_chunks.pop(job_id, None)
            self.batch_retrieval_rounds.pop(job_id, None)

        for document_id, document in list(self.batch_documents.items()):
            if document["assessment_id"] == assessment_id:
                self.batch_documents.pop(document_id, None)

        return ReviewResetResponse(
            assessment_id=assessment_id,
            dimension="all",
            deleted_job_count=len(job_ids) + len(batch_job_ids),
            deleted_task_count=len(task_ids) + len(batch_task_ids),
        )

    def list_pending_tasks(self, limit: int) -> list[dict[str, Any]]:
        """Return pending question tasks up to the requested limit.

        Inputs:
            limit: Maximum number of pending tasks to return.

        Outputs:
            list[dict[str, Any]]: Deep-copied pending task rows in insertion
            order so callers cannot mutate repository state directly.
        """
        pending_tasks = [
            task for task in self.tasks.values() if task["status"] == "pending"
        ]
        return deepcopy(pending_tasks[:limit])

    def lease_next_task(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the first pending task for a worker.

        Inputs:
            worker_id: Identifier of the worker process taking the task.

        Outputs:
            dict[str, Any] | None: A deep copy of the leased task, or ``None``
            when no pending task is available. Leased tasks are marked
            ``in_progress`` with a 15-minute expiration timestamp.
        """
        now = time.time()
        for task in self.tasks.values():
            lease_expired = (
                task["status"] in ACTIVE_REVIEW_TASK_STATUSES
                and task["lease_expires_at"] is not None
                and task["lease_expires_at"] <= now
            )
            if task["status"] != "pending" and not lease_expired:
                continue

            task["status"] = "in_progress"
            task["lease_owner"] = worker_id
            task["lease_expires_at"] = now + 900
            task["updated_at"] = now
            return deepcopy(task)

        return None

    def update_question_progress(
        self,
        task_id: str,
        worker_id: str,
        status: str,
        progress_message: str | None = None,
        rag_call_count: int | None = None,
        query_count: int | None = None,
        retrieved_chunk_count: int | None = None,
    ) -> None:
        """Persist progress for one leased in-memory manual question task.

        Inputs:
            task_id: Manual question task identifier.
            worker_id: Worker that holds the task lease.
            status: New sub-phase status.
            progress_message: Human-readable status for polling clients.
            rag_call_count: Optional running RAG tool-call count.
            query_count: Optional running retrieval query count.
            retrieved_chunk_count: Optional running retrieved chunk count.

        Outputs:
            None. The in-memory task row is updated.

        Raises:
            KeyError: Raised when ``task_id`` is unknown.
            ValueError: Raised when the task is leased by another worker.
        """

        if task_id not in self.tasks:
            raise KeyError(f"Unknown AI review task ID: {task_id}")
        task = self.tasks[task_id]
        if task.get("lease_owner") != worker_id:
            raise ValueError(f"AI review task {task_id} is not leased by {worker_id}")
        task["status"] = status
        task["progress_message"] = progress_message
        if rag_call_count is not None:
            task["rag_call_count"] = rag_call_count
        if query_count is not None:
            task["query_count"] = query_count
        if retrieved_chunk_count is not None:
            task["retrieved_chunk_count"] = retrieved_chunk_count
        task["updated_at"] = time.time()

    def get_task_attachments(self, task_id: str) -> list[dict[str, Any]]:
        """Return uploaded attachments for a question task.

        Inputs:
            task_id: Identifier of the question task whose files are needed.

        Outputs:
            list[dict[str, Any]]: Deep-copied attachment rows, or an empty list
            when the task has no stored attachments.
        """
        return deepcopy(self.attachments.get(task_id, []))

    def save_question_attempt(
        self,
        task_id: str,
        mode: str,
        status: str,
        model: str | None,
        usage_json: dict[str, Any] | None,
        error_code: str | None,
        error_message: str | None,
    ) -> str:
        """Persist one in-memory question review route or model attempt.

        Inputs:
            task_id: Question task that owns the attempt.
            mode: Processing mode, such as ``direct`` or ``rag``.
            status: Attempt outcome or route state.
            model: Optional model identifier used by the attempt.
            usage_json: Optional token or provider usage payload.
            error_code: Optional structured error code.
            error_message: Optional diagnostic error message.

        Outputs:
            str: Generated attempt ID for assertions and audit linkage.
        """
        attempt_id = f"attempt-{uuid4()}"
        self.attempts.setdefault(task_id, []).append(
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "mode": mode,
                "status": status,
                "model": model,
                "usage_json": deepcopy(usage_json),
                "error_code": error_code,
                "error_message": error_message,
            }
        )
        return attempt_id

    def save_attachment_extraction(
        self,
        task_id: str,
        attachment_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Persist an in-memory extraction summary.

        Inputs:
            task_id: Question task that owns the uploaded evidence.
            attachment_id: Uploaded attachment row linked to the extraction.
            extracted: Validated extracted document payload.
            estimated_tokens: Estimated token count for extracted text.
            warnings: Extraction warnings to store for audit review.

        Outputs:
            str: Generated extraction ID for the extraction summary row.
        """
        extraction_id = f"extraction-{uuid4()}"
        total_characters = int(
            extracted.metadata.get(
                "total_characters",
                sum(len(block.text) for block in extracted.blocks),
            )
        )
        page_count_value = extracted.metadata.get("page_count")
        page_count = int(page_count_value) if page_count_value is not None else None
        quality_json = extracted.metadata.get(
            "quality",
            {
                key: value
                for key, value in extracted.metadata.items()
                if key != "warnings"
            },
        )
        self.extractions.setdefault(task_id, []).append(
            {
                "extraction_id": extraction_id,
                "task_id": task_id,
                "attachment_id": attachment_id,
                "file_name": extracted.file_name,
                "document_type": extracted.document_type,
                "page_count": page_count,
                "extracted_block_count": len(extracted.blocks),
                "total_characters": total_characters,
                "estimated_tokens": estimated_tokens,
                "quality_json": deepcopy(quality_json),
                "warnings_json": deepcopy(warnings),
            }
        )
        return extraction_id

    def save_evidence_chunks(
        self,
        task_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Persist evidence chunks in memory with copy isolation.

        Inputs:
            task_id: Question task that owns the chunks.
            chunks: Chunk dictionaries with source metadata, text, hash, and
                optional embedding vectors.

        Outputs:
            None. Existing chunks and retrieval rounds for the task are replaced
            so retrying a reclaimed task starts from a clean RAG state.
        """
        self.retrieval_rounds.pop(task_id, None)
        self.evidence_chunks[task_id] = deepcopy(chunks)

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
        """Persist one in-memory RAG retrieval round.

        Inputs:
            task_id: Question task that owns the retrieval round.
            round_number: One-based retrieval round number.
            queries: Queries used during this retrieval round.
            retrieved_chunk_ids: Chunk IDs returned by vector retrieval.
            accepted_evidence_json: Model-selected evidence metadata.
            evidence_gaps: Remaining evidence gaps after assessment.
            refined_queries: Follow-up queries suggested for the next round.
            stop_reason: Optional reason the loop stopped or continued.

        Outputs:
            str: Generated retrieval round ID.
        """
        retrieval_round_id = f"retrieval-round-{uuid4()}"
        self.retrieval_rounds.setdefault(task_id, []).append(
            {
                "retrieval_round_id": retrieval_round_id,
                "task_id": task_id,
                "round_number": round_number,
                "queries": deepcopy(queries),
                "retrieved_chunk_ids": deepcopy(retrieved_chunk_ids),
                "accepted_evidence_json": deepcopy(accepted_evidence_json),
                "evidence_gaps": deepcopy(evidence_gaps),
                "refined_queries": deepcopy(refined_queries),
                "stop_reason": stop_reason,
            }
        )
        return retrieval_round_id

    def search_evidence_chunks(
        self,
        task_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return stored chunks with deterministic test similarity scores.

        Inputs:
            task_id: Review task identifier.
            query_embedding: Query vector, accepted for interface parity.
            top_k: Maximum result count.

        Outputs:
            list[dict[str, object]]: Stored chunks shaped like HANA search rows.
        """

        _ = query_embedding
        rows = []
        for index, chunk in enumerate(self.evidence_chunks.get(task_id, []), start=1):
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "attachment_id": chunk.get("attachment_id"),
                    "file_name": chunk["file_name"],
                    "document_type": chunk["document_type"],
                    "chunk_text": chunk["chunk_text"],
                    "location_json": chunk["location_json"],
                    "similarity_score": 1.0 / index,
                }
            )
        return rows[: max(1, min(int(top_k), 50))]

    def save_question_result(
        self,
        task_id: str,
        worker_id: str,
        result: QuestionReviewResult,
    ) -> None:
        """Persist a question review result and mark the task completed.

        Inputs:
            task_id: Identifier of the task that produced the result.
            worker_id: Identifier of the active worker lease holder.
            result: Validated ``QuestionReviewResult`` to associate with the
                task.

        Outputs:
            None. The canonical result is stored by task ID and the task status
            becomes ``completed`` when the worker owns a non-expired active lease.

        Raises:
            KeyError: Raised when ``task_id`` is not known to the repository.
            TypeError: Raised when ``result`` is not a ``QuestionReviewResult``.
            ValueError: Raised when ``result.question_id`` does not match the
                task question ID, or the worker does not hold a non-expired
                active lease for the task.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Unknown AI review task ID: {task_id}")
        if not isinstance(result, QuestionReviewResult):
            raise TypeError("result must be a QuestionReviewResult")
        if result.question_id != self.tasks[task_id]["question_id"]:
            raise ValueError(
                "Question result does not match task question ID: "
                f"{result.question_id} != {self.tasks[task_id]['question_id']}"
            )

        task = self.tasks[task_id]
        lease_is_active = (
            task["status"] in ACTIVE_REVIEW_TASK_STATUSES
            and task["lease_owner"] == worker_id
            and task["lease_expires_at"] is not None
            and task["lease_expires_at"] > time.time()
        )
        if not lease_is_active:
            raise ValueError(
                "Task result cannot be saved without a non-expired active "
                f"lease for worker {worker_id}: {task_id}"
            )

        self.results[task_id] = deepcopy(result)
        task["status"] = "completed"
        task["lease_owner"] = None
        task["lease_expires_at"] = None
        task["updated_at"] = time.time()

    def save_question_failure(
        self,
        task_id: str,
        worker_id: str,
        error_message: str,
    ) -> None:
        """Persist a question review task failure in memory.

        Inputs:
            task_id: Identifier of the task that failed.
            worker_id: Identifier of the active worker lease holder.
            error_message: Clear failure message exposed to polling clients.

        Outputs:
            None. The task status becomes ``failed`` and stores
            ``error_message`` when the active lease is valid.

        Raises:
            KeyError: Raised when ``task_id`` is not known to the repository.
            ValueError: Raised when the worker does not hold a non-expired
            active lease for the task.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Unknown AI review task ID: {task_id}")

        task = self.tasks[task_id]
        lease_is_active = (
            task["status"] in ACTIVE_REVIEW_TASK_STATUSES
            and task["lease_owner"] == worker_id
            and task["lease_expires_at"] is not None
            and task["lease_expires_at"] > time.time()
        )
        if not lease_is_active:
            raise ValueError(
                "Task failure cannot be saved without a non-expired active "
                f"lease for worker {worker_id}: {task_id}"
            )

        task["status"] = "failed"
        task["error_message"] = error_message
        task["lease_owner"] = None
        task["lease_expires_at"] = None
        task["updated_at"] = time.time()
