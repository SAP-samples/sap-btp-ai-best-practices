"""Batch review and indexing persistence for the HANA repository."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

from sqlalchemy import text

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
from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, table_name_from_create_statement
from app.services.joule_knowledge_importer import vector_to_json

from .common import (
    ACTIVE_REVIEW_TASK_STATUSES,
    ADMIN_DOCUMENT_CORPUS_ID,
    LEGACY_DESELECT_RESULT_WARNING,
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    _row_batches,
    document_content_hash,
    parse_persisted_question_review_result,
)

class HanaBatchesMixin:
    """Manage batch jobs, indexing leases, retrieval rounds, and question results."""

    def create_batch_job(
        self,
        assessment_id: str,
        current_answers: dict[str, list[str]],
        documents: list[dict[str, Any]],
        language: str = DEFAULT_LANGUAGE,
    ) -> BatchReviewJobResponse:
        """Create a global batch review job and all framework question tasks.

        Inputs:
            assessment_id: Assessment instance receiving AI review support.
            current_answers: Current selected answer item IDs keyed by question ID.
            documents: Uploaded global document payloads.
            language: Supported language code used by the worker prompt.

        Outputs:
            BatchReviewJobResponse: Created batch job with task and unique
            document counts.

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
        self.session.execute(
            text(
                "insert into ai_batch_jobs "
                "(job_id, assessment_id, language, status) "
                "values (:job_id, :assessment_id, :language, :status)"
            ),
            {
                "job_id": job_id,
                "assessment_id": assessment_id,
                "language": language,
                "status": "pending_indexing",
            },
        )

        linked_document_ids: set[str] = set()
        for document in documents:
            content = _document_payload_bytes(document.get("content", b""))
            content_hash = document_content_hash(content)
            existing_row = self.session.execute(
                text(
                    "select top 1 document_id "
                    "from ai_batch_documents "
                    "where assessment_id = :assessment_id "
                    "and content_hash = :content_hash "
                    "order by created_at"
                ),
                {
                    "assessment_id": assessment_id,
                    "content_hash": content_hash,
                },
            ).mappings().first()
            if existing_row is None:
                document_id = f"document-{uuid4()}"
                self.session.execute(
                    text(
                        "insert into ai_batch_documents "
                        "(document_id, assessment_id, content_hash, file_name, "
                        "content_type, content_blob) "
                        "values (:document_id, :assessment_id, :content_hash, "
                        ":file_name, :content_type, :content_blob)"
                    ),
                    {
                        "document_id": document_id,
                        "assessment_id": assessment_id,
                        "content_hash": content_hash,
                        "file_name": document["file_name"],
                        "content_type": document.get(
                            "content_type",
                            "application/octet-stream",
                        ),
                        "content_blob": content,
                    },
                )
            else:
                document_id = str(existing_row["document_id"])

            if document_id in linked_document_ids:
                continue
            linked_document_ids.add(document_id)
            self.session.execute(
                text(
                    "insert into ai_batch_job_documents "
                    "(link_id, job_id, document_id) "
                    "values (:link_id, :job_id, :document_id)"
                ),
                {
                    "link_id": f"batch-document-link-{uuid4()}",
                    "job_id": job_id,
                    "document_id": document_id,
                },
            )

        for question in questions:
            task_id = f"batch-task-{uuid4()}"
            self.session.execute(
                text(
                    "insert into ai_batch_question_tasks "
                    "(task_id, job_id, question_id, dimension, "
                    "current_answers_json, status) "
                    "values (:task_id, :job_id, :question_id, :dimension, "
                    ":current_answers_json, :status)"
                ),
                {
                    "task_id": task_id,
                    "job_id": job_id,
                    "question_id": question.question_id,
                    "dimension": question.dimension,
                    "current_answers_json": json.dumps(
                        current_answers.get(question.question_id, []),
                        ensure_ascii=False,
                    ),
                    "status": "pending",
                },
            )

        return BatchReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending",
            task_count=len(questions),
            document_count=len(linked_document_ids),
        )

    def get_batch_job_status(self, job_id: str) -> ReviewJobStatusResponse:
        """Return progress and completed results for one global batch job.

        Inputs:
            job_id: Batch AI review job identifier.

        Outputs:
            ReviewJobStatusResponse: Aggregate progress plus per-question
            result payloads.

        Raises:
            KeyError: Raised when no persisted batch job matches ``job_id``.
        """

        job_row = self.session.execute(
            text(
                "select job_id, language, status, "
                "indexed_chunk_count, active_question_id, active_dimension "
                "from ai_batch_jobs "
                "where job_id = :job_id"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if job_row is None:
            raise KeyError(f"Unknown batch AI review job ID: {job_id}")

        document_progress_row = self.session.execute(
            text(
                "select "
                "(select count(*) from ai_batch_job_documents "
                "where job_id = :job_id) as document_count, "
                "(select count(distinct document_id) "
                "from ai_batch_document_extractions "
                "where job_id = :job_id) as extracted_document_count, "
                "(select count(distinct document_id) "
                "from ai_batch_document_chunks "
                "where job_id = :job_id) as embedded_document_count "
                "from dummy"
            ),
            {"job_id": job_id},
        ).mappings().first()
        document_count = int((document_progress_row or {}).get("document_count", 0) or 0)
        extracted_document_count = int(
            (document_progress_row or {}).get("extracted_document_count", 0) or 0
        )
        embedded_document_count = int(
            (document_progress_row or {}).get("embedded_document_count", 0) or 0
        )

        rows = self.session.execute(
            text(
                "select t.task_id, t.question_id, t.dimension, t.status, "
                "t.error_code, t.error_message, t.progress_message, "
                "t.rag_call_count, t.query_count, t.retrieved_chunk_count, "
                "t.lease_owner, t.lease_expires_at, t.updated_at, "
                "r.result_json "
                "from ai_batch_question_tasks t "
                "left join ai_batch_question_results r on r.task_id = t.task_id "
                "where t.job_id = :job_id "
                "order by t.created_at"
            ),
            {"job_id": job_id},
        ).mappings().all()

        tasks: list[ReviewTaskStatus] = []
        for row in rows:
            result = None
            result_requires_rerun = False
            if row["result_json"]:
                result = parse_persisted_question_review_result(row["result_json"])
                result_requires_rerun = LEGACY_DESELECT_RESULT_WARNING in result.warnings
            tasks.append(
                ReviewTaskStatus(
                    task_id=row["task_id"],
                    question_id=row["question_id"],
                    dimension=row["dimension"],
                    status=row["status"],
                    result=result,
                    result_requires_rerun=result_requires_rerun,
                    error_code=row["error_code"],
                    error_message=row["error_message"],
                    progress_message=row["progress_message"],
                    rag_call_count=int(row["rag_call_count"] or 0),
                    query_count=int(row["query_count"] or 0),
                    retrieved_chunk_count=int(row["retrieved_chunk_count"] or 0),
                    lease_owner=row["lease_owner"],
                    lease_expires_at=(
                        str(row["lease_expires_at"]) if row["lease_expires_at"] else None
                    ),
                    updated_at=(
                        str(row["updated_at"]) if row["updated_at"] else None
                    ),
                )
            )

        task_statuses = [task.status for task in tasks]
        job_status = job_row["status"]
        finished_status = _derive_finished_batch_status(task_statuses)
        if job_status == "reviewing_questions" and finished_status is not None:
            derived_status = finished_status
        elif job_status in {
            "pending_indexing",
            "extracting_documents",
            "embedding_documents",
            "reviewing_questions",
        }:
            derived_status = job_status
        else:
            derived_status = _derive_job_status(task_statuses, job_status)
        return ReviewJobStatusResponse(
            job_id=job_id,
            language=job_row["language"] or DEFAULT_LANGUAGE,
            status=derived_status,
            task_count=len(tasks),
            completed_count=sum(1 for status in task_statuses if status == "completed"),
            failed_count=sum(1 for status in task_statuses if status == "failed"),
            batch_phase=job_status,
            indexed_chunk_count=int(job_row["indexed_chunk_count"] or 0),
            document_count=document_count,
            processed_document_count=_document_phase_processed_count(
                job_status=job_status,
                document_count=document_count,
                extracted_document_count=extracted_document_count,
                embedded_document_count=embedded_document_count,
            ),
            active_question_id=job_row["active_question_id"],
            active_dimension=job_row["active_dimension"],
            tasks=tasks,
        )

    def lease_next_batch_job(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next pending global batch job for a worker.

        Inputs:
            worker_id: Identifier of the worker process taking the batch job.

        Outputs:
            dict[str, Any] | None: Leased batch job row, or ``None`` when no
            pending or expired batch job is available.
        """

        row = self.session.execute(
            text(
                "select top 1 job_id, assessment_id, language, status "
                "from ai_batch_jobs "
                "where status = 'pending' "
                "or (status = 'in_progress' "
                "and lease_expires_at <= current_timestamp) "
                "order by created_at"
            )
        ).mappings().first()
        if row is None:
            return None

        lease_result = self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = 'in_progress', lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_timestamp, 3600), "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and (status = 'pending' "
                "or (status = 'in_progress' and lease_expires_at <= current_timestamp))"
            ),
            {"worker_id": worker_id, "job_id": row["job_id"]},
        )
        if lease_result.rowcount == 0:
            return None
        return {
            "job_id": row["job_id"],
            "assessment_id": row["assessment_id"],
            "language": row["language"] or DEFAULT_LANGUAGE,
            "status": "in_progress",
            "lease_owner": worker_id,
        }

    def get_batch_job_documents(self, job_id: str) -> list[dict[str, Any]]:
        """Return deduplicated uploaded documents linked to a batch job.

        Inputs:
            job_id: Batch job whose documents should be indexed.

        Outputs:
            list[dict[str, Any]]: Document dictionaries containing identifiers,
            filenames, content types, hashes, and binary content.
        """

        rows = self.session.execute(
            text(
                "select d.document_id, d.file_name, d.content_type, "
                "d.content_hash, d.content_blob as content "
                "from ai_batch_job_documents l "
                "join ai_batch_documents d on d.document_id = l.document_id "
                "where l.job_id = :job_id "
                "order by l.created_at"
            ),
            {"job_id": job_id},
        ).mappings().all()
        return [dict(row) for row in rows]

    def get_batch_question_tasks(self, job_id: str) -> list[dict[str, Any]]:
        """Return question task rows for one batch job.

        Inputs:
            job_id: Batch job identifier.

        Outputs:
            list[dict[str, Any]]: Batch question task dictionaries with parsed
            current answer selections.
        """

        rows = self.session.execute(
            text(
                "select task_id, job_id, question_id, dimension, "
                "current_answers_json, status "
                "from ai_batch_question_tasks "
                "where job_id = :job_id "
                "order by created_at"
            ),
            {"job_id": job_id},
        ).mappings().all()
        return [
            {
                "task_id": row["task_id"],
                "job_id": row["job_id"],
                "question_id": row["question_id"],
                "dimension": row["dimension"],
                "current_selected_answer_item_ids": json.loads(
                    row["current_answers_json"]
                ),
                "status": row["status"],
            }
            for row in rows
        ]

    def save_batch_document_extraction(
        self,
        job_id: str,
        document_id: str,
        extracted: ExtractedDocument,
        estimated_tokens: int,
        warnings: list[str],
    ) -> str:
        """Persist one batch document extraction summary.

        Inputs:
            job_id: Batch job that owns this indexing run.
            document_id: Deduplicated batch document identifier.
            extracted: Extracted text payload for the document.
            estimated_tokens: Estimated token count for extracted text.
            warnings: Extraction warnings to store for audit review.

        Outputs:
            str: Generated extraction ID for the extraction summary row.
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
        self.session.execute(
            text(
                "insert into ai_batch_document_extractions "
                "(extraction_id, job_id, document_id, file_name, document_type, "
                "page_count, extracted_block_count, total_characters, "
                "estimated_tokens, quality_json, warnings_json) "
                "values (:extraction_id, :job_id, :document_id, :file_name, "
                ":document_type, :page_count, :extracted_block_count, "
                ":total_characters, :estimated_tokens, :quality_json, "
                ":warnings_json)"
            ),
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
                "quality_json": json.dumps(quality_json, ensure_ascii=False),
                "warnings_json": json.dumps(warnings, ensure_ascii=False),
            },
        )
        return extraction_id

    def save_batch_document_chunks(
        self,
        job_id: str,
        chunks: list[dict[str, Any]],
        replace_existing: bool = True,
    ) -> None:
        """Persist embedded chunks for one global batch job.

        Inputs:
            job_id: Batch job that owns the shared document corpus.
            chunks: Chunk dictionaries with source metadata, vectors, and kind.
            replace_existing: Whether to clear existing chunks and retrieval
                rounds for the job before inserting the supplied chunks.

        Outputs:
            None. Existing chunks are cleared only when ``replace_existing`` is
            true, which lets indexing persist per-document progress.
        """

        if replace_existing:
            self.session.execute(
                text("delete from ai_batch_retrieval_rounds where job_id = :job_id"),
                {"job_id": job_id},
            )
            self.session.execute(
                text("delete from ai_batch_document_chunks where job_id = :job_id"),
                {"job_id": job_id},
            )
        insert_rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "job_id": job_id,
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "document_type": chunk["document_type"],
                "source_block_ids_json": json.dumps(
                    chunk["source_block_ids"], ensure_ascii=False
                ),
                "location_json": json.dumps(
                    chunk["location_json"], ensure_ascii=False
                ),
                "chunk_text": chunk["chunk_text"],
                "estimated_tokens": chunk["estimated_tokens"],
                "embedding_model": chunk["embedding_model"],
                "embedding": vector_to_json(chunk.get("embedding")),
                "content_hash": chunk["content_hash"],
                "chunk_kind": chunk.get("chunk_kind", "raw"),
            }
            for chunk in chunks
        ]
        insert_statement = text(
            "insert into ai_batch_document_chunks "
            "(chunk_id, job_id, document_id, file_name, document_type, "
            "source_block_ids_json, location_json, chunk_text, "
            "estimated_tokens, embedding_model, embedding, "
            "content_hash, chunk_kind) "
            "values (:chunk_id, :job_id, :document_id, :file_name, "
            ":document_type, :source_block_ids_json, :location_json, "
            ":chunk_text, :estimated_tokens, :embedding_model, "
            "to_real_vector(:embedding), :content_hash, :chunk_kind)"
        )
        for batch in _row_batches(insert_rows):
            self.session.execute(
                insert_statement,
                batch,
            )

    def search_batch_document_chunks(
        self,
        job_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search embedded batch document chunks by cosine similarity.

        Inputs:
            job_id: Batch job identifier.
            query_embedding: Query embedding vector.
            top_k: Maximum number of chunks to return.

        Outputs:
            list[dict[str, object]]: Retrieved chunk rows with similarity scores.
        """

        limit = max(1, min(int(top_k), 50))
        query_vector = vector_to_json(query_embedding)
        rows = self.session.execute(
            text(
                f"select top {limit} chunk_id, document_id, file_name, "
                "document_type, chunk_text, location_json, chunk_kind, "
                "cosine_similarity(embedding, to_real_vector(:query_vector)) "
                "as similarity_score "
                "from ai_batch_document_chunks "
                "where job_id = :job_id and embedding is not null "
                "order by similarity_score desc"
            ),
            {"job_id": job_id, "query_vector": query_vector},
        ).mappings().all()
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "attachment_id": row["document_id"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "chunk_text": row["chunk_text"],
                "location_json": (
                    json.loads(row["location_json"])
                    if isinstance(row["location_json"], str)
                    else dict(row["location_json"])
                ),
                "chunk_kind": row["chunk_kind"],
                "similarity_score": float(row["similarity_score"]),
            }
            for row in rows
        ]

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
        """Persist one batch RAG retrieval round.

        Inputs:
            job_id: Batch job that owns the shared vector corpus.
            task_id: Batch question task being reviewed.
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

        retrieval_round_id = f"batch-retrieval-round-{uuid4()}"
        self.session.execute(
            text(
                "insert into ai_batch_retrieval_rounds "
                "(retrieval_round_id, job_id, task_id, round_number, "
                "queries_json, retrieved_chunk_ids_json, accepted_evidence_json, "
                "evidence_gaps_json, refined_queries_json, stop_reason) "
                "values (:retrieval_round_id, :job_id, :task_id, :round_number, "
                ":queries_json, :retrieved_chunk_ids_json, "
                ":accepted_evidence_json, :evidence_gaps_json, "
                ":refined_queries_json, :stop_reason)"
            ),
            {
                "retrieval_round_id": retrieval_round_id,
                "job_id": job_id,
                "task_id": task_id,
                "round_number": round_number,
                "queries_json": json.dumps(queries, ensure_ascii=False),
                "retrieved_chunk_ids_json": json.dumps(
                    retrieved_chunk_ids, ensure_ascii=False
                ),
                "accepted_evidence_json": json.dumps(
                    accepted_evidence_json, ensure_ascii=False
                ),
                "evidence_gaps_json": json.dumps(evidence_gaps, ensure_ascii=False),
                "refined_queries_json": json.dumps(
                    refined_queries, ensure_ascii=False
                ),
                "stop_reason": stop_reason,
            },
        )
        return retrieval_round_id

    def save_batch_question_result(
        self,
        task_id: str,
        result: QuestionReviewResult,
        worker_id: str | None = None,
    ) -> None:
        """Persist one batch question result and mark its task completed.

        Inputs:
            task_id: Batch question task identifier.
            result: Validated question review result.
            worker_id: Optional worker that holds the task lease.

        Outputs:
            None. The batch question task receives one canonical result row.

        Raises:
            KeyError: Raised when no known batch task matches ``task_id``.
            ValueError: Raised when the result question does not match the task.
        """

        task_row = self.session.execute(
            text(
                "select job_id, question_id "
                "from ai_batch_question_tasks "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        ).mappings().first()
        if task_row is None:
            raise KeyError(f"Unknown batch AI review task ID: {task_id}")
        if result.question_id != task_row["question_id"]:
            raise ValueError(
                "Question result does not match batch task question ID: "
                f"{result.question_id} != {task_row['question_id']}"
            )

        self.session.execute(
            text(
                "update ai_batch_question_tasks "
                "set status = 'completed', "
                "lease_owner = null, lease_expires_at = null, "
                "updated_at = current_timestamp "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        )
        self.session.execute(
            text(
                "delete from ai_batch_question_results "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        )
        self.session.execute(
            text(
                "insert into ai_batch_question_results "
                "(result_id, task_id, job_id, question_id, result_json) "
                "values (:result_id, :task_id, :job_id, :question_id, :result_json)"
            ),
            {
                "result_id": f"batch-result-{uuid4()}",
                "task_id": task_id,
                "job_id": task_row["job_id"],
                "question_id": result.question_id,
                "result_json": result.model_dump_json(),
            },
        )
        self._finalize_batch_job_if_ready(str(task_row["job_id"]))

    def _finalize_batch_job_if_ready(self, job_id: str) -> None:
        """Close a batch job once every child question task is terminal.

        Inputs:
            job_id: Batch job whose child question task statuses should be
            inspected.

        Outputs:
            None. The parent batch job is updated to a terminal status only
            when all child tasks are ``completed`` or ``failed``.
        """

        row = self.session.execute(
            text(
                "select count(*) as task_count, "
                "sum(case when status = 'completed' then 1 else 0 end) "
                "as completed_count, "
                "sum(case when status = 'failed' then 1 else 0 end) "
                "as failed_count, "
                "sum(case when status not in ('completed', 'failed') then 1 else 0 end) "
                "as unfinished_count "
                "from ai_batch_question_tasks "
                "where job_id = :job_id"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if row is None:
            return

        task_count = int(row["task_count"] or 0)
        unfinished_count = int(row["unfinished_count"] or 0)
        if task_count == 0 or unfinished_count > 0:
            return

        completed_count = int(row["completed_count"] or 0)
        failed_count = int(row["failed_count"] or 0)
        if failed_count == 0:
            status = "completed"
        elif completed_count == 0:
            status = "failed"
        else:
            status = "partial_failed"

        self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = :status, lease_owner = null, "
                "lease_expires_at = null, active_question_id = null, "
                "active_dimension = null, updated_at = current_timestamp "
                "where job_id = :job_id"
            ),
            {"job_id": job_id, "status": status},
        )

    def complete_batch_job(self, job_id: str, worker_id: str) -> None:
        """Mark a leased batch job completed.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that currently holds the batch lease.

        Outputs:
            None. The batch job status is set to ``completed`` when the active
            lease matches.
        """

        self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = 'completed', updated_at = current_timestamp "
                "where job_id = :job_id "
                "and status = 'in_progress' "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {"job_id": job_id, "worker_id": worker_id},
        )

    def save_batch_job_failure(
        self,
        job_id: str,
        worker_id: str,
        error_message: str,
    ) -> None:
        """Persist a batch job failure for polling clients.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that currently holds the batch lease.
            error_message: Failure detail to expose through polling.

        Outputs:
            None. The batch job and incomplete question tasks are marked failed.
        """

        self.session.execute(
            text(
                "update ai_batch_question_tasks "
                "set status = 'failed', error_message = :error_message, "
                "updated_at = current_timestamp "
                "where job_id = :job_id and status != 'completed'"
            ),
            {"job_id": job_id, "error_message": error_message},
        )
        self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = 'failed', error_message = :error_message, "
                "lease_owner = null, lease_expires_at = null, "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "error_message": error_message,
            },
        )

    def lease_next_batch_indexing_job(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next pending batch job awaiting document indexing.

        Inputs:
            worker_id: Identifier of the worker process taking the indexing lease.

        Outputs:
            dict[str, Any] | None: Leased batch job row, or None when no job
            awaits indexing.
        """
        row = self.session.execute(
            text(
                "select top 1 job_id, assessment_id, language, status "
                "from ai_batch_jobs "
                "where status = 'pending_indexing' "
                "or (status in ('extracting_documents', 'embedding_documents') "
                "and lease_expires_at <= current_timestamp) "
                "order by created_at"
            )
        ).mappings().first()
        if row is None:
            return None

        self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = 'extracting_documents', lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_timestamp, 3600), "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and (status = 'pending_indexing' "
                "or (status in ('extracting_documents', 'embedding_documents') "
                "and lease_expires_at <= current_timestamp))"
            ),
            {"worker_id": worker_id, "job_id": row["job_id"]},
        )
        return {
            **dict(row),
            "status": "extracting_documents",
            "lease_owner": worker_id,
        }

    def update_batch_indexing_progress(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        indexed_chunk_count: int | None = None,
    ) -> None:
        """Persist batch indexing progress for polling clients.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that holds the indexing lease.
            status: New indexing sub-phase status.
            indexed_chunk_count: Running count of indexed chunks.

        Outputs:
            None. The batch job status and optional chunk count are updated.
        """
        params: dict[str, Any] = {
            "job_id": job_id,
            "worker_id": worker_id,
            "status": status,
        }
        chunk_clause = ""
        if indexed_chunk_count is not None:
            chunk_clause = ", indexed_chunk_count = :indexed_chunk_count"
            params["indexed_chunk_count"] = indexed_chunk_count

        self.session.execute(
            text(
                "update ai_batch_jobs "
                f"set status = :status{chunk_clause}, "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            params,
        )

    def complete_batch_indexing(
        self,
        job_id: str,
        worker_id: str,
        indexed_chunk_count: int,
    ) -> None:
        """Mark shared batch indexing complete and enable question leasing.

        Inputs:
            job_id: Batch job identifier.
            worker_id: Worker that holds the indexing lease.
            indexed_chunk_count: Final count of indexed document chunks.

        Outputs:
            None. The batch job transitions to reviewing_questions and the
            lease is released.
        """
        self.session.execute(
            text(
                "update ai_batch_jobs "
                "set status = 'reviewing_questions', "
                "indexed_chunk_count = :indexed_chunk_count, "
                "lease_owner = null, lease_expires_at = null, "
                "updated_at = current_timestamp "
                "where job_id = :job_id "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "indexed_chunk_count": indexed_chunk_count,
            },
        )

    def lease_next_batch_question_task(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next pending batch question task for review.

        Inputs:
            worker_id: Identifier of the worker process taking the task lease.

        Outputs:
            dict[str, Any] | None: Leased task row, or None when no question
            task is ready for review.
        """
        row = self.session.execute(
            text(
                "select top 1 t.task_id, t.job_id, t.question_id, t.dimension, "
                "t.current_answers_json, t.status "
                "from ai_batch_question_tasks t "
                "inner join ai_batch_jobs j on j.job_id = t.job_id "
                "where j.status = 'reviewing_questions' "
                "and (t.status = 'pending' "
                "or (t.status in ('in_progress', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and t.lease_expires_at <= current_timestamp)) "
                "order by t.created_at"
            )
        ).mappings().first()
        if row is None:
            return None

        self.session.execute(
            text(
                "update ai_batch_question_tasks "
                "set status = 'in_progress', lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_timestamp, 900), "
                "updated_at = current_timestamp "
                "where task_id = :task_id "
                "and (status = 'pending' "
                "or (status in ('in_progress', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_expires_at <= current_timestamp))"
            ),
            {"worker_id": worker_id, "task_id": row["task_id"]},
        )
        return {
            **dict(row),
            "status": "in_progress",
            "lease_owner": worker_id,
        }

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
        """Persist progress for one leased batch question task.

        Inputs:
            task_id: Batch question task identifier.
            worker_id: Worker that holds the task lease.
            status: New sub-phase status for the question task.
            progress_message: Human-readable progress for polling clients.
            rag_call_count: Running count of RAG tool calls.
            query_count: Running total of queries issued.
            retrieved_chunk_count: Running total of evidence chunks retrieved.

        Outputs:
            None. The task progress fields are updated.
        """
        self.session.execute(
            text(
                "update ai_batch_question_tasks "
                "set status = :status, progress_message = :progress_message, "
                "rag_call_count = coalesce(:rag_call_count, rag_call_count), "
                "query_count = coalesce(:query_count, query_count), "
                "retrieved_chunk_count = coalesce(:retrieved_chunk_count, retrieved_chunk_count), "
                "updated_at = current_timestamp "
                "where task_id = :task_id "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "task_id": task_id,
                "worker_id": worker_id,
                "status": status,
                "progress_message": progress_message,
                "rag_call_count": rag_call_count,
                "query_count": query_count,
                "retrieved_chunk_count": retrieved_chunk_count,
            },
        )
        task_row = self.session.execute(
            text(
                "select question_id, dimension from ai_batch_question_tasks "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        ).mappings().first()
        if task_row:
            self.session.execute(
                text(
                    "update ai_batch_jobs "
                    "set active_question_id = :question_id, "
                    "active_dimension = :dimension, "
                    "updated_at = current_timestamp "
                    "where job_id = ("
                    "select job_id from ai_batch_question_tasks "
                    "where task_id = :task_id)"
                ),
                {
                    "task_id": task_id,
                    "question_id": task_row["question_id"],
                    "dimension": task_row["dimension"],
                },
            )

    def save_batch_question_failure(
        self,
        task_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Mark one batch question task as failed.

        Inputs:
            task_id: Batch question task identifier.
            worker_id: Worker that holds the task lease.
            error_code: Machine-readable failure category.
            error_message: Human-readable failure detail.

        Outputs:
            None. The task transitions to failed status with error details.
        """
        task_row = self.session.execute(
            text(
                "select job_id "
                "from ai_batch_question_tasks "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        ).mappings().first()
        self.session.execute(
            text(
                "update ai_batch_question_tasks "
                "set status = 'failed', error_code = :error_code, "
                "error_message = :error_message, "
                "lease_owner = null, lease_expires_at = null, "
                "updated_at = current_timestamp "
                "where task_id = :task_id "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "task_id": task_id,
                "worker_id": worker_id,
                "error_code": error_code,
                "error_message": error_message,
            },
        )
        if task_row is not None:
            self._finalize_batch_job_if_ready(str(task_row["job_id"]))

    def save_batch_rag_tool_call(
        self,
        job_id: str,
        task_id: str,
        rag_call_number: int,
        queries_json: list[dict[str, Any]],
        retrieved_chunk_ids: list[str],
        matched_queries_json: dict[str, list[dict[str, Any]]],
        query_count: int,
        retrieved_chunk_count: int,
        duration_ms: int,
        stop_reason: str,
    ) -> str:
        """Persist one RAG tool call audit row for a batch question task.

        Inputs:
            job_id: Batch job identifier.
            task_id: Batch question task identifier.
            rag_call_number: Sequential tool call number within the task.
            queries_json: Structured query payloads sent to the vector search.
            retrieved_chunk_ids: Chunk IDs returned by the vector search.
            matched_queries_json: Per-chunk matched query metadata.
            query_count: Number of queries in this tool call.
            retrieved_chunk_count: Number of chunks retrieved.
            duration_ms: Tool call wall-clock duration in milliseconds.
            stop_reason: Reason the tool call loop continued or stopped.

        Outputs:
            str: Generated retrieval round ID for audit references.
        """
        retrieval_round_id = f"batch-retrieval-round-{uuid4()}"
        accepted_evidence_json = {
            "matched_queries": matched_queries_json,
            "query_count": query_count,
            "retrieved_chunk_count": retrieved_chunk_count,
            "duration_ms": duration_ms,
        }
        self.session.execute(
            text(
                "insert into ai_batch_retrieval_rounds "
                "(retrieval_round_id, job_id, task_id, round_number, "
                "queries_json, retrieved_chunk_ids_json, accepted_evidence_json, "
                "evidence_gaps_json, refined_queries_json, stop_reason) "
                "values (:retrieval_round_id, :job_id, :task_id, :round_number, "
                ":queries_json, :retrieved_chunk_ids_json, "
                ":accepted_evidence_json, :evidence_gaps_json, "
                ":refined_queries_json, :stop_reason)"
            ),
            {
                "retrieval_round_id": retrieval_round_id,
                "job_id": job_id,
                "task_id": task_id,
                "round_number": rag_call_number,
                "queries_json": json.dumps(queries_json, ensure_ascii=False),
                "retrieved_chunk_ids_json": json.dumps(
                    retrieved_chunk_ids,
                    ensure_ascii=False,
                ),
                "accepted_evidence_json": json.dumps(
                    accepted_evidence_json,
                    ensure_ascii=False,
                ),
                "evidence_gaps_json": json.dumps([], ensure_ascii=False),
                "refined_queries_json": json.dumps([], ensure_ascii=False),
                "stop_reason": stop_reason,
            },
        )
        return retrieval_round_id
