"""Batch review and indexing persistence for the in-memory repository."""

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

class MemoryBatchesMixin:
    """Manage in-memory batch jobs, indexing, retrieval rounds, and results."""

    def create_batch_job(
        self,
        assessment_id: str,
        current_answers: dict[str, list[str]],
        documents: list[dict[str, Any]],
        language: str = DEFAULT_LANGUAGE,
    ) -> BatchReviewJobResponse:
        """Create one in-memory global batch review job.

        Inputs:
            assessment_id: Assessment instance receiving AI review support.
            current_answers: Current selected answer item IDs keyed by question
                ID.
            documents: Uploaded global document payloads.
            language: Supported language code used by the worker prompt.

        Outputs:
            BatchReviewJobResponse: Created job ID and task/document counts.

        Raises:
            ValueError: Raised when no documents or framework questions exist.
        """

        if not documents:
            raise ValueError("Batch review jobs require at least one document.")
        language = normalize_language(language)
        questions = self.list_all_questions(language=language)
        if not questions:
            raise ValueError("Batch review jobs require imported framework questions.")

        job_id = f"batch-job-{uuid4()}"
        linked_document_ids: list[str] = []
        existing_by_hash = {
            (
                document["assessment_id"],
                document["content_hash"],
            ): document_id
            for document_id, document in self.batch_documents.items()
        }
        for payload in documents:
            content = _document_payload_bytes(payload.get("content", b""))
            content_hash = document_content_hash(content)
            key = (assessment_id, content_hash)
            document_id = existing_by_hash.get(key)
            if document_id is None:
                document_id = f"document-{uuid4()}"
                self.batch_documents[document_id] = {
                    "document_id": document_id,
                    "assessment_id": assessment_id,
                    "content_hash": content_hash,
                    "file_name": payload["file_name"],
                    "content_type": payload.get(
                        "content_type",
                        "application/octet-stream",
                    ),
                    "content": content,
                }
                existing_by_hash[key] = document_id
            if document_id not in linked_document_ids:
                linked_document_ids.append(document_id)

        self.batch_jobs[job_id] = {
            "job_id": job_id,
            "assessment_id": assessment_id,
            "language": language,
            "status": "pending_indexing",
            "lease_owner": None,
            "lease_expires_at": None,
            "error_message": None,
            "retry_count": 0,
            "indexed_chunk_count": 0,
            "active_question_id": None,
            "active_dimension": None,
        }
        self.batch_job_documents[job_id] = linked_document_ids

        for question in questions:
            task_id = f"batch-task-{uuid4()}"
            self.batch_question_tasks[task_id] = {
                "task_id": task_id,
                "job_id": job_id,
                "question_id": question.question_id,
                "dimension": question.dimension,
                "status": "pending",
                "current_selected_answer_item_ids": deepcopy(
                    current_answers.get(question.question_id, [])
                ),
                "error_message": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "retry_count": 0,
                "progress_message": None,
                "rag_call_count": 0,
                "query_count": 0,
                "retrieved_chunk_count": 0,
                "error_code": None,
            }

        return BatchReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending_indexing",
            task_count=len(questions),
            document_count=len(linked_document_ids),
        )

    def get_batch_job_status(self, job_id: str) -> ReviewJobStatusResponse:
        """Return derived progress and completed results for a batch job.

        Inputs:
            job_id: Identifier of the in-memory batch review job to poll.

        Outputs:
            ReviewJobStatusResponse: Aggregate job progress plus per-question
            batch task statuses.

        Raises:
            KeyError: Raised when no in-memory batch job matches ``job_id``.
        """
        if job_id not in self.batch_jobs:
            raise KeyError(f"Unknown batch AI review job ID: {job_id}")

        task_rows = [
            task
            for task in self.batch_question_tasks.values()
            if task["job_id"] == job_id
        ]
        tasks = [
            ReviewTaskStatus(
                task_id=task["task_id"],
                question_id=task["question_id"],
                dimension=task["dimension"],
                status=task["status"],
                result=deepcopy(self.batch_question_results.get(task["task_id"])),
                error_code=task.get("error_code"),
                error_message=task.get("error_message"),
                progress_message=task.get("progress_message"),
                rag_call_count=int(task.get("rag_call_count", 0)),
                query_count=int(task.get("query_count", 0)),
                retrieved_chunk_count=int(task.get("retrieved_chunk_count", 0)),
                lease_owner=task.get("lease_owner"),
                lease_expires_at=(
                    str(task.get("lease_expires_at")) if task.get("lease_expires_at") else None
                ),
            )
            for task in task_rows
        ]
        task_statuses = [task.status for task in tasks]
        job_status = self.batch_jobs[job_id]["status"]
        finished_status = _derive_finished_batch_status(task_statuses)
        if job_status == "reviewing_questions" and finished_status is not None:
            derived_status = finished_status
        elif job_status in {"pending_indexing", "extracting_documents", "embedding_documents", "reviewing_questions"}:
            derived_status = job_status
        else:
            derived_status = _derive_job_status(task_statuses, job_status)
        document_count = len(self.batch_job_documents.get(job_id, []))
        extracted_document_count = len(
            {
                row.get("document_id")
                for row in self.batch_extractions.get(job_id, [])
                if row.get("document_id")
            }
        )
        embedded_document_count = len(
            {
                row.get("document_id")
                for row in self.batch_document_chunks.get(job_id, [])
                if row.get("document_id")
            }
        )
        return ReviewJobStatusResponse(
            job_id=job_id,
            language=self.batch_jobs[job_id].get("language", DEFAULT_LANGUAGE),
            status=derived_status,
            task_count=len(tasks),
            completed_count=sum(1 for status in task_statuses if status == "completed"),
            failed_count=sum(1 for status in task_statuses if status == "failed"),
            batch_phase=self.batch_jobs[job_id].get("status"),
            indexed_chunk_count=int(self.batch_jobs[job_id].get("indexed_chunk_count", 0)),
            document_count=document_count,
            processed_document_count=_document_phase_processed_count(
                job_status=job_status,
                document_count=document_count,
                extracted_document_count=extracted_document_count,
                embedded_document_count=embedded_document_count,
            ),
            active_question_id=self.batch_jobs[job_id].get("active_question_id"),
            active_dimension=self.batch_jobs[job_id].get("active_dimension"),
            tasks=tasks,
        )

    def lease_next_batch_job(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the first pending or expired batch job for a worker.

        Inputs:
            worker_id: Identifier of the worker process taking the job.

        Outputs:
            dict[str, Any] | None: Deep-copied leased batch job row, or
            ``None`` when no batch job can be leased.
        """
        now = time.time()
        for job in self.batch_jobs.values():
            lease_expired = (
                job["status"] == "in_progress"
                and job["lease_expires_at"] is not None
                and job["lease_expires_at"] <= now
            )
            if job["status"] not in {"pending", "pending_indexing"} and not lease_expired:
                continue
            job["status"] = "in_progress"
            job["lease_owner"] = worker_id
            job["lease_expires_at"] = now + 3600
            return deepcopy(job)
        return None

    def lease_next_batch_indexing_job(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next batch job that still needs shared document indexing."""
        now = time.time()
        for job in self.batch_jobs.values():
            lease_expired = (
                job["status"] in {"extracting_documents", "embedding_documents"}
                and job.get("lease_expires_at") is not None
                and job["lease_expires_at"] <= now
            )
            if job["status"] != "pending_indexing" and not lease_expired:
                continue
            job["status"] = "extracting_documents"
            job["lease_owner"] = worker_id
            job["lease_expires_at"] = now + 3600
            job["retry_count"] = int(job.get("retry_count", 0)) + (1 if lease_expired else 0)
            return deepcopy(job)
        return None

    def update_batch_indexing_progress(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        indexed_chunk_count: int | None = None,
    ) -> None:
        """Persist shared batch indexing progress for polling clients."""
        job = self.batch_jobs[job_id]
        if job.get("lease_owner") != worker_id:
            raise ValueError(f"Batch job {job_id} is not leased by {worker_id}")
        job["status"] = status
        if indexed_chunk_count is not None:
            job["indexed_chunk_count"] = indexed_chunk_count

    def complete_batch_indexing(
        self,
        job_id: str,
        worker_id: str,
        indexed_chunk_count: int,
    ) -> None:
        """Mark shared batch indexing complete and enable question leasing."""
        job = self.batch_jobs[job_id]
        if job.get("lease_owner") != worker_id:
            raise ValueError(f"Batch job {job_id} is not leased by {worker_id}")
        job["status"] = "reviewing_questions"
        job["indexed_chunk_count"] = indexed_chunk_count
        job["lease_owner"] = None
        job["lease_expires_at"] = None

    def lease_next_batch_question_task(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next pending or expired batch question task."""
        now = time.time()
        ready_job_ids = {
            job_id
            for job_id, job in self.batch_jobs.items()
            if job["status"] == "reviewing_questions"
        }
        for task in self.batch_question_tasks.values():
            lease_expired = (
                task["status"] in {"in_progress", "retrieving_evidence", "finalizing_answer", "retrying"}
                and task.get("lease_expires_at") is not None
                and task["lease_expires_at"] <= now
            )
            if task["job_id"] not in ready_job_ids:
                continue
            if task["status"] != "pending" and not lease_expired:
                continue
            task["status"] = "in_progress"
            task["lease_owner"] = worker_id
            task["lease_expires_at"] = now + 900
            if lease_expired:
                task["retry_count"] = int(task.get("retry_count", 0)) + 1
            return deepcopy(task)
        return None

    def update_batch_question_progress(
        self,
        task_id: str,
        worker_id: str,
        status: str,
        progress_message: str | None = None,
        rag_call_count: int | None = None,
        query_count: int | None = None,
        retrieved_chunk_count: int | None = None,
    ) -> None:
        """Persist progress for one leased batch question task."""
        task = self.batch_question_tasks[task_id]
        if task.get("lease_owner") != worker_id:
            raise ValueError(f"Batch question task {task_id} is not leased by {worker_id}")
        task["status"] = status
        task["progress_message"] = progress_message
        if rag_call_count is not None:
            task["rag_call_count"] = rag_call_count
        if query_count is not None:
            task["query_count"] = query_count
        if retrieved_chunk_count is not None:
            task["retrieved_chunk_count"] = retrieved_chunk_count
        job = self.batch_jobs.get(task["job_id"])
        if job is not None:
            job["active_question_id"] = task["question_id"]
            job["active_dimension"] = task["dimension"]

    def save_batch_question_failure(
        self,
        task_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark one leased batch question task failed without touching siblings."""
        task = self.batch_question_tasks[task_id]
        if task.get("lease_owner") != worker_id:
            raise ValueError(f"Batch question task {task_id} is not leased by {worker_id}")
        task["status"] = "failed"
        task["error_code"] = error_code
        task["error_message"] = error_message
        task["lease_owner"] = None
        task["lease_expires_at"] = None
        self._finalize_batch_job_if_ready(task["job_id"])

    def get_batch_job_documents(self, job_id: str) -> list[dict[str, Any]]:
        """Return documents linked to one batch job.

        Inputs:
            job_id: Batch job identifier.

        Outputs:
            list[dict[str, Any]]: Deep-copied document rows.
        """
        return deepcopy(
            [
                self.batch_documents[document_id]
                for document_id in self.batch_job_documents.get(job_id, [])
            ]
        )

    def get_batch_question_tasks(self, job_id: str) -> list[dict[str, Any]]:
        """Return batch question tasks for one batch job.

        Inputs:
            job_id: Batch job identifier.

        Outputs:
            list[dict[str, Any]]: Deep-copied task rows.
        """
        return deepcopy(
            [
                task
                for task in self.batch_question_tasks.values()
                if task["job_id"] == job_id
            ]
        )

    def save_batch_document_extraction(
        self,
        job_id: str,
        document_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Persist an in-memory batch extraction summary.

        Inputs:
            job_id: Batch job that owns this indexing run.
            document_id: Deduplicated document identifier.
            extracted: Validated extracted document payload.
            estimated_tokens: Estimated token count for extracted text.
            warnings: Extraction warnings to store for audit review.

        Outputs:
            str: Generated extraction ID.
        """
        extraction_id = f"batch-extraction-{uuid4()}"
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
                if key not in {"warnings", "document_id"}
            },
        )
        self.batch_extractions.setdefault(job_id, []).append(
            {
                "extraction_id": extraction_id,
                "job_id": job_id,
                "document_id": document_id,
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

    def save_batch_document_chunks(
        self,
        job_id: str,
        chunks: list[dict[str, Any]],
        replace_existing: bool = True,
    ) -> None:
        """Persist shared batch document chunks in memory.

        Inputs:
            job_id: Batch job that owns the chunks.
            chunks: Chunk dictionaries with document metadata and embeddings.
            replace_existing: Whether to replace previous chunks for the job.

        Outputs:
            None. Existing chunks and retrieval rounds for the job are replaced
            only when ``replace_existing`` is true.
        """
        if replace_existing:
            self.batch_retrieval_rounds.pop(job_id, None)
            self.batch_document_chunks[job_id] = []
        self.batch_document_chunks.setdefault(job_id, []).extend(deepcopy(chunks))

    def search_batch_document_chunks(
        self,
        job_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return stored batch chunks with deterministic similarity scores.

        Inputs:
            job_id: Batch job identifier.
            query_embedding: Query vector, accepted for interface parity.
            top_k: Maximum result count.

        Outputs:
            list[dict[str, object]]: Stored chunks shaped like vector search rows.
        """
        _ = query_embedding
        rows = []
        for index, chunk in enumerate(
            self.batch_document_chunks.get(job_id, []),
            start=1,
        ):
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "attachment_id": chunk["document_id"],
                    "file_name": chunk["file_name"],
                    "document_type": chunk["document_type"],
                    "chunk_text": chunk["chunk_text"],
                    "location_json": chunk["location_json"],
                    "chunk_kind": chunk.get("chunk_kind", "raw"),
                    "similarity_score": 1.0 / index,
                }
            )
        return rows[: max(1, min(int(top_k), 50))]

    def save_batch_retrieval_round(
        self,
        job_id: str,
        task_id: str,
        round_number: int,
        queries: list[str],
        retrieved_chunk_ids: list[str],
        accepted_evidence_json: dict[str, Any],
        evidence_gaps: list[str],
        refined_queries: list[str],
        stop_reason: str | None,
    ) -> str:
        """Persist one in-memory batch retrieval round.

        Inputs:
            job_id: Batch job that owns the shared vector corpus.
            task_id: Batch question task being reviewed.
            round_number: One-based retrieval round number.
            queries: Queries used during this retrieval round.
            retrieved_chunk_ids: Retrieved chunk IDs.
            accepted_evidence_json: Accepted/rejected evidence metadata.
            evidence_gaps: Remaining evidence gaps.
            refined_queries: Follow-up queries.
            stop_reason: Loop stop reason.

        Outputs:
            str: Generated retrieval round ID.
        """
        retrieval_round_id = f"batch-retrieval-round-{uuid4()}"
        self.batch_retrieval_rounds.setdefault(job_id, []).append(
            {
                "retrieval_round_id": retrieval_round_id,
                "job_id": job_id,
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

    def save_batch_rag_tool_call(
        self,
        job_id: str,
        task_id: str,
        rag_call_number: int,
        queries_json: list[dict[str, Any]],
        retrieved_chunk_ids: list[str],
        matched_queries_json: dict[str, Any],
        query_count: int,
        retrieved_chunk_count: int,
        duration_ms: int,
        stop_reason: str,
    ) -> str:
        """Persist one RAG tool call audit row for a batch question.

        Inputs:
            job_id: Batch job that owns the shared vector corpus.
            task_id: Batch question task being reviewed.
            rag_call_number: One-based RAG call counter within the task.
            queries_json: Serialized query metadata sent to embedding.
            retrieved_chunk_ids: IDs of chunks returned by vector search.
            matched_queries_json: Mapping of chunk IDs to matched queries.
            query_count: Number of queries embedded.
            retrieved_chunk_count: Number of distinct chunks returned.
            duration_ms: Elapsed time in milliseconds.
            stop_reason: Reason the tool call concluded.

        Outputs:
            str: Generated retrieval round ID.
        """
        return self.save_batch_retrieval_round(
            job_id=job_id,
            task_id=task_id,
            round_number=rag_call_number,
            queries=[query["query"] for query in queries_json],
            retrieved_chunk_ids=retrieved_chunk_ids,
            accepted_evidence_json={
                "matched_queries": matched_queries_json,
                "query_count": query_count,
                "retrieved_chunk_count": retrieved_chunk_count,
                "duration_ms": duration_ms,
            },
            evidence_gaps=[],
            refined_queries=[],
            stop_reason=stop_reason,
        )

    def save_batch_question_result(
        self,
        task_id: str,
        result: QuestionReviewResult,
        worker_id: str | None = None,
    ) -> None:
        """Persist a batch question result and mark the leased task completed.

        Inputs:
            task_id: Batch question task identifier.
            result: Validated question review result.
            worker_id: Optional worker that holds the task lease.

        Outputs:
            None. The task status becomes ``completed``.

        Raises:
            KeyError: Raised when ``task_id`` is unknown.
            ValueError: Raised when the result question does not match the task
                or the worker does not hold the lease.
        """
        if task_id not in self.batch_question_tasks:
            raise KeyError(f"Unknown batch AI review task ID: {task_id}")
        task = self.batch_question_tasks[task_id]
        if worker_id is not None and task.get("lease_owner") != worker_id:
            raise ValueError(f"Batch question task {task_id} is not leased by {worker_id}")
        if result.question_id != task["question_id"]:
            raise ValueError(
                "Question result does not match batch task question ID: "
                f"{result.question_id} != {task['question_id']}"
            )
        self.batch_question_results[task_id] = deepcopy(result)
        task["status"] = "completed"
        task["lease_owner"] = None
        task["lease_expires_at"] = None
        self._finalize_batch_job_if_ready(task["job_id"])

    def _finalize_batch_job_if_ready(self, job_id: str) -> None:
        """Close a batch job once every child question task is terminal.

        Inputs:
            job_id: Batch job whose in-memory child task rows should be
            inspected.

        Outputs:
            None. The job status becomes ``completed``, ``failed``, or
            ``partial_failed`` only when no child task is still active.
        """
        job = self.batch_jobs.get(job_id)
        if job is None:
            return
        task_statuses = [
            task["status"]
            for task in self.batch_question_tasks.values()
            if task["job_id"] == job_id
        ]
        finished_status = _derive_finished_batch_status(task_statuses)
        if finished_status is None:
            return
        job["status"] = finished_status
        job["lease_owner"] = None
        job["lease_expires_at"] = None
        job["active_question_id"] = None
        job["active_dimension"] = None

    def complete_batch_job(self, job_id: str, worker_id: str) -> None:
        """Mark an in-memory leased batch job completed.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that currently holds the lease.

        Outputs:
            None. Matching leased jobs are marked completed.
        """
        job = self.batch_jobs[job_id]
        if job["status"] in {"completed", "failed", "partial_failed"}:
            return
        if job["lease_owner"] != worker_id:
            raise ValueError(
                "Batch job cannot be completed without an active lease for "
                f"worker {worker_id}: {job_id}"
            )
        job["status"] = "completed"

    def save_batch_job_failure(
        self,
        job_id: str,
        worker_id: str,
        error_message: str,
    ) -> None:
        """Persist an in-memory batch job failure.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that currently holds the lease.
            error_message: Failure detail exposed through polling.

        Outputs:
            None. Incomplete batch question tasks are marked failed.
        """
        job = self.batch_jobs[job_id]
        if job["lease_owner"] != worker_id:
            raise ValueError(
                "Batch job failure cannot be saved without an active lease for "
                f"worker {worker_id}: {job_id}"
            )
        job["status"] = "failed"
        job["error_message"] = error_message
        for task in self.batch_question_tasks.values():
            if task["job_id"] == job_id and task["status"] != "completed":
                task["status"] = "failed"
                task["error_message"] = error_message
