"""Manual and corpus question review persistence for the HANA repository."""

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
from app.services.customer_class_scope import (
    CustomerClass,
    DEFAULT_CUSTOMER_CLASS,
    filter_current_answer_ids,
    max_allowed_level,
)
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

class HanaReviewsMixin:
    """Manage manual review jobs, leases, progress, evidence, and results."""

    def create_dimension_job(
        self,
        assessment_id: str,
        dimension: str,
        current_answers: dict[str, list[str]],
        attachments: dict[str, list[dict[str, Any]]],
        language: str = DEFAULT_LANGUAGE,
    ) -> ReviewJobResponse:
        """Create a dimension review job with pending question tasks and files.

        Inputs:
            assessment_id: Identifier of the assessment being reviewed.
            dimension: Framework dimension selected for AI review.
            language: Supported language code used by the worker prompt.
            current_answers: Current selected answer item IDs keyed by question ID.
            attachments: Uploaded file payloads keyed by question ID.

        Outputs:
            ReviewJobResponse: Created job ID, initial status, and task count.
        """

        job_id = f"job-{uuid4()}"
        language = normalize_language(language)
        self.session.execute(
            text(
                "insert into ai_dimension_jobs "
                "(job_id, assessment_id, dimension, language, status) "
                "values (:job_id, :assessment_id, :dimension, :language, :status)"
            ),
            {
                "job_id": job_id,
                "assessment_id": assessment_id,
                "dimension": dimension,
                "language": language,
                "status": "pending",
            },
        )

        task_count = 0
        for question_id, files in attachments.items():
            if not files:
                continue

            task_id = f"task-{uuid4()}"
            self.session.execute(
                text(
                    "insert into ai_question_tasks "
                    "(task_id, job_id, question_id, dimension, "
                    "current_answers_json, status) "
                    "values (:task_id, :job_id, :question_id, :dimension, "
                    ":current_answers_json, :status)"
                ),
                {
                    "task_id": task_id,
                    "job_id": job_id,
                    "question_id": question_id,
                    "dimension": dimension,
                    "current_answers_json": json.dumps(
                        current_answers.get(question_id, [])
                    ),
                    "status": "pending",
                },
            )

            for file_payload in files:
                attachment_id = f"attachment-{uuid4()}"
                self.session.execute(
                    text(
                        "insert into ai_task_attachments "
                        "(attachment_id, task_id, file_name, content_type, "
                        "content_blob) "
                        "values (:attachment_id, :task_id, :file_name, "
                        ":content_type, :content_blob)"
                    ),
                    {
                        "attachment_id": attachment_id,
                        "task_id": task_id,
                        "file_name": file_payload["file_name"],
                        "content_type": file_payload["content_type"],
                        "content_blob": file_payload["content"],
                    },
                )
            task_count += 1

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
        """Create selected question tasks that retrieve from the document corpus.

        Inputs:
            assessment_id: Assessment instance receiving AI review support.
            dimension: Dimension containing the selected questions.
            question_ids: Question IDs to analyze.
            current_answers: Selected answer item IDs keyed by question ID.
            language: Supported language code used by review prompts.

        Outputs:
            ReviewJobResponse: Created job ID and task count.
        """
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

        job_id = f"job-{uuid4()}"
        pending_task_status = _pending_task_status_for_runtime()
        self.session.execute(
            text(
                "insert into ai_dimension_jobs "
                "(job_id, assessment_id, dimension, language, status) "
                "values (:job_id, :assessment_id, :dimension, :language, 'pending')"
            ),
            {
                "job_id": job_id,
                "assessment_id": assessment_id,
                "dimension": dimension,
                "language": language,
            },
        )
        for question, max_level in scoped_questions:
            self.session.execute(
                text(
                    "insert into ai_question_tasks "
                    "(task_id, job_id, question_id, dimension, "
                    "current_answers_json, max_allowed_level, status) "
                    "values (:task_id, :job_id, :question_id, :dimension, "
                    ":current_answers_json, :max_allowed_level, :status)"
                ),
                {
                    "task_id": f"task-{uuid4()}",
                    "job_id": job_id,
                    "question_id": question.question_id,
                    "dimension": question.dimension,
                    "current_answers_json": json.dumps(
                        filter_current_answer_ids(
                            question,
                            current_answers.get(question.question_id, []),
                            max_level,
                        ),
                        ensure_ascii=False,
                    ),
                    "max_allowed_level": max_level,
                    "status": pending_task_status,
                },
            )
        return ReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending",
            task_count=len(scoped_questions),
        )

    def create_corpus_all_questions_job(
        self,
        assessment_id: str,
        current_answers: dict[str, list[str]],
        language: str = DEFAULT_LANGUAGE,
        customer_class: CustomerClass = DEFAULT_CUSTOMER_CLASS,
    ) -> ReviewJobResponse:
        """Create one corpus-backed task for every framework question.

        Inputs:
            assessment_id: Assessment instance receiving AI review support.
            current_answers: Selected answer item IDs keyed by question ID.
            language: Supported language code used by review prompts.

        Outputs:
            ReviewJobResponse: Created all-question job ID and task count.
        """
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
        job_id = f"job-{uuid4()}"
        pending_task_status = _pending_task_status_for_runtime()
        self.session.execute(
            text(
                "insert into ai_dimension_jobs "
                "(job_id, assessment_id, dimension, language, status) "
                "values (:job_id, :assessment_id, :dimension, :language, 'pending')"
            ),
            {
                "job_id": job_id,
                "assessment_id": assessment_id,
                "dimension": "__ALL__",
                "language": language,
            },
        )
        for question, max_level in scoped_questions:
            self.session.execute(
                text(
                    "insert into ai_question_tasks "
                    "(task_id, job_id, question_id, dimension, "
                    "current_answers_json, max_allowed_level, status) "
                    "values (:task_id, :job_id, :question_id, :dimension, "
                    ":current_answers_json, :max_allowed_level, :status)"
                ),
                {
                    "task_id": f"task-{uuid4()}",
                    "job_id": job_id,
                    "question_id": question.question_id,
                    "dimension": question.dimension,
                    "current_answers_json": json.dumps(
                        filter_current_answer_ids(
                            question,
                            current_answers.get(question.question_id, []),
                            max_level,
                        ),
                        ensure_ascii=False,
                    ),
                    "max_allowed_level": max_level,
                    "status": pending_task_status,
                },
            )
        return ReviewJobResponse(
            job_id=job_id,
            language=language,
            status="pending",
            task_count=len(scoped_questions),
        )

    def get_job_status(self, job_id: str) -> ReviewJobStatusResponse:
        """Return derived progress and completed results for one review job.

        Inputs:
            job_id: Identifier of the dimension review job to poll.

        Outputs:
            ReviewJobStatusResponse: Aggregate job progress plus per-question
            task statuses and completed AI review results.

        Raises:
            KeyError: Raised when no persisted job matches ``job_id``.
        """

        job_row = self.session.execute(
            text(
                "select job_id, language, status "
                "from ai_dimension_jobs "
                "where job_id = :job_id"
            ),
            {"job_id": job_id},
        ).mappings().first()
        if job_row is None:
            raise KeyError(f"Unknown AI review job ID: {job_id}")

        rows = self.session.execute(
            text(
                "select t.task_id, t.question_id, t.dimension, t.status, "
                "t.error_code, t.error_message, t.progress_message, "
                "t.rag_call_count, t.query_count, t.retrieved_chunk_count, "
                "t.lease_owner, t.lease_expires_at, t.updated_at, "
                "r.result_json "
                "from ai_question_tasks t "
                "left join ai_question_results r on r.task_id = t.task_id "
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
            task_status = _normalize_task_status_for_response(str(row["status"]))
            tasks.append(
                ReviewTaskStatus(
                    task_id=row["task_id"],
                    question_id=row["question_id"],
                    dimension=row["dimension"],
                    status=task_status,
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
        return ReviewJobStatusResponse(
            job_id=job_id,
            language=job_row["language"] or DEFAULT_LANGUAGE,
            status=_derive_job_status(task_statuses, job_row["status"]),
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
        """Delete persisted AI review jobs and child rows for one dimension.

        Inputs:
            assessment_id: Assessment instance whose AI review state should be
                cleared.
            dimension: Framework dimension selected by the user reset action.

        Outputs:
            ReviewResetResponse: Counts of removed legacy and batch jobs/tasks.
        """

        parameters = {"assessment_id": assessment_id, "dimension": dimension}
        job_filter = (
            "select job_id from ai_dimension_jobs "
            "where assessment_id = :assessment_id and dimension = :dimension"
        )
        task_filter = (
            "select task_id from ai_question_tasks "
            f"where job_id in ({job_filter})"
        )
        task_count_row = self.session.execute(
            text(
                "select count(*) as task_count "
                "from ai_question_tasks "
                f"where job_id in ({job_filter})"
            ),
            parameters,
        ).mappings().first()
        job_count_row = self.session.execute(
            text(
                "select count(*) as job_count "
                "from ai_dimension_jobs "
                "where assessment_id = :assessment_id and dimension = :dimension"
            ),
            parameters,
        ).mappings().first()

        for table_name in (
            "ai_retrieval_rounds",
            "ai_evidence_chunks",
            "ai_extracted_blocks",
            "ai_attachment_extractions",
            "ai_applied_suggestions",
            "ai_evidence_refs",
            "ai_question_chunk_runs",
            "ai_question_attempts",
            "ai_question_results",
            "ai_task_attachments",
        ):
            self._delete_task_scoped_rows_if_table_exists(
                table_name=table_name,
                task_filter=task_filter,
                parameters=parameters,
            )

        self.session.execute(
            text(f"delete from ai_question_tasks where job_id in ({job_filter})"),
            parameters,
        )
        self.session.execute(
            text(
                "delete from ai_dimension_jobs "
                "where assessment_id = :assessment_id and dimension = :dimension"
            ),
            parameters,
        )

        batch_task_count = 0
        deleted_batch_job_count = 0
        batch_tables_exist = self._table_exists("ai_batch_jobs") and self._table_exists(
            "ai_batch_question_tasks"
        )
        if batch_tables_exist:
            batch_job_filter = (
                "select job_id from ai_batch_jobs "
                "where assessment_id = :assessment_id"
            )
            batch_task_filter = (
                "select task_id from ai_batch_question_tasks "
                "where dimension = :dimension "
                f"and job_id in ({batch_job_filter})"
            )
            batch_task_count_row = self.session.execute(
                text(
                    "select count(*) as task_count "
                    "from ai_batch_question_tasks "
                    f"where task_id in ({batch_task_filter})"
                ),
                parameters,
            ).mappings().first()
            batch_task_count = int(
                (batch_task_count_row or {}).get("task_count", 0)
            )

            for table_name in (
                "ai_batch_retrieval_rounds",
                "ai_batch_question_results",
            ):
                self._delete_task_scoped_rows_if_table_exists(
                    table_name=table_name,
                    task_filter=batch_task_filter,
                    parameters=parameters,
                )

            self.session.execute(
                text(
                    "delete from ai_batch_question_tasks "
                    f"where task_id in ({batch_task_filter})"
                ),
                parameters,
            )

            empty_batch_job_rows = self.session.execute(
                text(
                    "select j.job_id "
                    "from ai_batch_jobs j "
                    "where j.assessment_id = :assessment_id "
                    "and not exists ("
                    "select 1 from ai_batch_question_tasks t "
                    "where t.job_id = j.job_id"
                    ")"
                ),
                {"assessment_id": assessment_id},
            ).mappings().all()
            empty_batch_job_ids = [
                str(row["job_id"]) for row in empty_batch_job_rows
            ]
            deleted_batch_job_count = len(empty_batch_job_ids)

            for batch_job_id in empty_batch_job_ids:
                for table_name in (
                    "ai_batch_retrieval_rounds",
                    "ai_batch_document_chunks",
                    "ai_batch_extracted_blocks",
                    "ai_batch_document_extractions",
                    "ai_batch_job_documents",
                    "ai_batch_question_results",
                    "ai_batch_question_tasks",
                ):
                    self._delete_job_scoped_rows_if_table_exists(
                        table_name=table_name,
                        job_id=batch_job_id,
                    )
                self.session.execute(
                    text("delete from ai_batch_jobs where job_id = :job_id"),
                    {"job_id": batch_job_id},
                )

            if self._table_exists("ai_batch_documents") and self._table_exists(
                "ai_batch_job_documents"
            ):
                orphan_document_rows = self.session.execute(
                    text(
                        "select d.document_id "
                        "from ai_batch_documents d "
                        "where d.assessment_id = :assessment_id "
                        "and not exists ("
                        "select 1 from ai_batch_job_documents l "
                        "where l.document_id = d.document_id"
                        ")"
                    ),
                    {"assessment_id": assessment_id},
                ).mappings().all()
                for row in orphan_document_rows:
                    self.session.execute(
                        text(
                            "delete from ai_batch_documents "
                            "where document_id = :document_id"
                        ),
                        {"document_id": row["document_id"]},
                    )

        return ReviewResetResponse(
            assessment_id=assessment_id,
            dimension=dimension,
            deleted_job_count=int((job_count_row or {}).get("job_count", 0))
            + deleted_batch_job_count,
            deleted_task_count=int((task_count_row or {}).get("task_count", 0))
            + batch_task_count,
        )

    def clear_assessment_review_state(self, assessment_id: str) -> ReviewResetResponse:
        """Delete all persisted AI review process state for one assessment.

        Inputs:
            assessment_id: Assessment instance whose legacy dimension jobs,
                global batch jobs, uploaded batch documents, extraction rows,
                chunks, retrieval audit rows, question tasks, and results should
                be cleared.

        Outputs:
            ReviewResetResponse: Counts of removed jobs and tasks across all
            dimensions. ``dimension`` is returned as ``all`` because this method
            intentionally resets the whole assessment processing state.
        """

        parameters = {"assessment_id": assessment_id}
        job_filter = (
            "select job_id from ai_dimension_jobs "
            "where assessment_id = :assessment_id"
        )
        task_filter = (
            "select task_id from ai_question_tasks "
            f"where job_id in ({job_filter})"
        )
        task_count_row = self.session.execute(
            text(
                "select count(*) as task_count "
                "from ai_question_tasks "
                f"where job_id in ({job_filter})"
            ),
            parameters,
        ).mappings().first()
        job_count_row = self.session.execute(
            text(
                "select count(*) as job_count "
                "from ai_dimension_jobs "
                "where assessment_id = :assessment_id"
            ),
            parameters,
        ).mappings().first()

        for table_name in (
            "ai_retrieval_rounds",
            "ai_evidence_chunks",
            "ai_extracted_blocks",
            "ai_attachment_extractions",
            "ai_applied_suggestions",
            "ai_evidence_refs",
            "ai_question_chunk_runs",
            "ai_question_attempts",
            "ai_question_results",
            "ai_task_attachments",
        ):
            self._delete_task_scoped_rows_if_table_exists(
                table_name=table_name,
                task_filter=task_filter,
                parameters=parameters,
            )

        self.session.execute(
            text(f"delete from ai_question_tasks where job_id in ({job_filter})"),
            parameters,
        )
        self.session.execute(
            text(
                "delete from ai_dimension_jobs "
                "where assessment_id = :assessment_id"
            ),
            parameters,
        )

        batch_task_count = 0
        batch_job_count = 0
        batch_job_filter = (
            "select job_id from ai_batch_jobs "
            "where assessment_id = :assessment_id"
        )
        if self._table_exists("ai_batch_jobs"):
            batch_job_count_row = self.session.execute(
                text(
                    "select count(*) as job_count "
                    "from ai_batch_jobs "
                    "where assessment_id = :assessment_id"
                ),
                parameters,
            ).mappings().first()
            batch_job_count = int((batch_job_count_row or {}).get("job_count", 0))

        if self._table_exists("ai_batch_question_tasks"):
            batch_task_filter = (
                "select task_id from ai_batch_question_tasks "
                f"where job_id in ({batch_job_filter})"
            )
            batch_task_count_row = self.session.execute(
                text(
                    "select count(*) as task_count "
                    "from ai_batch_question_tasks "
                    f"where job_id in ({batch_job_filter})"
                ),
                parameters,
            ).mappings().first()
            batch_task_count = int(
                (batch_task_count_row or {}).get("task_count", 0)
            )
            for table_name in (
                "ai_batch_retrieval_rounds",
                "ai_batch_question_results",
            ):
                self._delete_task_scoped_rows_if_table_exists(
                    table_name=table_name,
                    task_filter=batch_task_filter,
                    parameters=parameters,
                )

        for table_name in (
            "ai_batch_retrieval_rounds",
            "ai_batch_document_chunks",
            "ai_batch_extracted_blocks",
            "ai_batch_document_extractions",
            "ai_batch_job_documents",
            "ai_batch_question_results",
            "ai_batch_question_tasks",
        ):
            self._delete_job_scoped_rows_for_filter_if_table_exists(
                table_name=table_name,
                job_filter=batch_job_filter,
                parameters=parameters,
            )

        if self._table_exists("ai_batch_documents"):
            self.session.execute(
                text(
                    "delete from ai_batch_documents "
                    "where assessment_id = :assessment_id"
                ),
                parameters,
            )
        if self._table_exists("ai_batch_jobs"):
            self.session.execute(
                text(
                    "delete from ai_batch_jobs "
                    "where assessment_id = :assessment_id"
                ),
                parameters,
            )

        return ReviewResetResponse(
            assessment_id=assessment_id,
            dimension="all",
            deleted_job_count=int((job_count_row or {}).get("job_count", 0))
            + batch_job_count,
            deleted_task_count=int((task_count_row or {}).get("task_count", 0))
            + batch_task_count,
        )

    def lease_next_task(self, worker_id: str) -> dict[str, Any] | None:
        """Lease the next pending or expired in-progress task for a worker.

        Inputs:
            worker_id: Identifier of the worker process taking the task.

        Outputs:
            dict[str, Any] | None: Leased task data with parsed current answers,
            or ``None`` when no task can be leased.
        """

        pending_task_status = _pending_task_status_for_runtime()
        row = self.session.execute(
            text(
                "select top 1 t.task_id, t.job_id, j.assessment_id, t.question_id, "
                "t.dimension, j.language, t.current_answers_json, "
                "t.max_allowed_level, t.status "
                "from ai_question_tasks t "
                "join ai_dimension_jobs j on j.job_id = t.job_id "
                "where t.status = :pending_status "
                "or (t.status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and t.lease_expires_at <= current_timestamp) "
                "order by t.created_at"
            ),
            {"pending_status": pending_task_status},
        ).mappings().first()
        if row is None:
            return None

        lease_result = self.session.execute(
            text(
                "update ai_question_tasks "
                "set status = 'in_progress', lease_owner = :worker_id, "
                "lease_expires_at = add_seconds(current_timestamp, 900), "
                "updated_at = current_timestamp "
                "where task_id = :task_id "
                "and (status = :pending_status "
                "or (status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_expires_at <= current_timestamp))"
            ),
            {
                "worker_id": worker_id,
                "task_id": row["task_id"],
                "pending_status": pending_task_status,
            },
        )
        if lease_result.rowcount == 0:
            return None

        return {
            "task_id": row["task_id"],
            "job_id": row["job_id"],
            "assessment_id": row["assessment_id"],
            "question_id": row["question_id"],
            "dimension": row["dimension"],
            "language": row["language"] or DEFAULT_LANGUAGE,
            "status": "in_progress",
            "current_selected_answer_item_ids": json.loads(
                row["current_answers_json"]
            ),
            "max_allowed_level": int(row.get("max_allowed_level", 5) or 5),
            "lease_owner": worker_id,
        }

    def get_task_attachments(self, task_id: str) -> list[dict[str, Any]]:
        """Return uploaded attachment payloads for one task.

        Inputs:
            task_id: Identifier of the task whose files are needed.

        Outputs:
            list[dict[str, Any]]: Attachment dictionaries containing file name,
            content type, and binary content.
        """
        if not self._table_exists("ai_task_attachments"):
            return []

        rows = self.session.execute(
            text(
                "select attachment_id, file_name, content_type, content_blob as content "
                "from ai_task_attachments "
                "where task_id = :task_id "
                "order by created_at"
            ),
            {"task_id": task_id},
        ).mappings()
        return [dict(row) for row in rows]

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
        """Persist progress for one leased manual question task.

        Inputs:
            task_id: Manual question task identifier.
            worker_id: Worker that holds the task lease.
            status: New sub-phase status for the question task.
            progress_message: Human-readable progress for polling clients.
            rag_call_count: Running count of RAG tool calls.
            query_count: Running total of queries issued.
            retrieved_chunk_count: Running total of evidence chunks retrieved.

        Outputs:
            None. The task progress fields are updated when the active lease
            still belongs to ``worker_id``.
        """

        self.session.execute(
            text(
                "update ai_question_tasks "
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
        """Persist one question review route or model attempt in HANA.

        Inputs:
            task_id: Question task that owns the attempt.
            mode: Processing mode, such as ``direct`` or ``rag``.
            status: Attempt outcome or route state.
            model: Optional model identifier used by the attempt.
            usage_json: Optional token or provider usage payload.
            error_code: Optional structured error code.
            error_message: Optional diagnostic error message.

        Outputs:
            str: Generated attempt ID for downstream audit linkage.
        """
        attempt_id = f"attempt-{uuid4()}"
        self.session.execute(
            text(
                "insert into ai_question_attempts "
                "(attempt_id, task_id, mode, status, model, usage_json, "
                "error_code, error_message) "
                "values (:attempt_id, :task_id, :mode, :status, :model, "
                ":usage_json, :error_code, :error_message)"
            ),
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "mode": mode,
                "status": status,
                "model": model,
                "usage_json": json.dumps(usage_json, ensure_ascii=False)
                if usage_json is not None
                else None,
                "error_code": error_code,
                "error_message": error_message,
            },
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
        """Persist an attachment extraction summary.

        Inputs:
            task_id: Question task that owns the uploaded evidence.
            attachment_id: Uploaded attachment row linked to the extraction.
            extracted: Validated extracted document payload.
            estimated_tokens: Estimated token count for the extracted text.
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
        self.session.execute(
            text(
                "insert into ai_attachment_extractions "
                "(extraction_id, task_id, attachment_id, file_name, "
                "document_type, page_count, extracted_block_count, "
                "total_characters, estimated_tokens, quality_json, warnings_json) "
                "values (:extraction_id, :task_id, :attachment_id, :file_name, "
                ":document_type, :page_count, :extracted_block_count, "
                ":total_characters, :estimated_tokens, :quality_json, "
                ":warnings_json)"
            ),
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
                "quality_json": json.dumps(quality_json, ensure_ascii=False),
                "warnings_json": json.dumps(warnings, ensure_ascii=False),
            },
        )
        return extraction_id

    def save_evidence_chunks(
        self,
        task_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Persist embedded evidence chunks for later HANA vector retrieval.

        Inputs:
            task_id: Question task that owns the chunks.
            chunks: Chunk dictionaries with source metadata, text, hash, and
                optional embedding vectors.

        Outputs:
            None. Existing chunks and retrieval rounds for the task are cleared
            first so retrying a reclaimed task does not fail on deterministic
            chunk primary keys.
        """
        self.session.execute(
            text("delete from ai_retrieval_rounds where task_id = :task_id"),
            {"task_id": task_id},
        )
        self.session.execute(
            text("delete from ai_evidence_chunks where task_id = :task_id"),
            {"task_id": task_id},
        )
        insert_rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "task_id": task_id,
                "attachment_id": chunk["attachment_id"],
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
            }
            for chunk in chunks
        ]
        insert_statement = text(
            "insert into ai_evidence_chunks "
            "(chunk_id, task_id, attachment_id, file_name, document_type, "
            "source_block_ids_json, location_json, chunk_text, "
            "estimated_tokens, embedding_model, embedding, content_hash) "
            "values (:chunk_id, :task_id, :attachment_id, :file_name, "
            ":document_type, :source_block_ids_json, :location_json, "
            ":chunk_text, :estimated_tokens, :embedding_model, "
            "to_real_vector(:embedding), :content_hash)"
        )
        for batch in _row_batches(insert_rows):
            self.session.execute(
                insert_statement,
                batch,
            )

    def search_evidence_chunks(
        self,
        task_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Search embedded evidence chunks for a task by cosine similarity.

        Inputs:
            task_id: Review task identifier.
            query_embedding: Query embedding vector.
            top_k: Maximum number of chunks to return.

        Outputs:
            list[dict[str, object]]: Retrieved chunk rows with similarity scores.
        """

        limit = max(1, min(int(top_k), 50))
        query_vector = vector_to_json(query_embedding)
        rows = self.session.execute(
            text(
                f"select top {limit} chunk_id, attachment_id, file_name, "
                "document_type, chunk_text, location_json, "
                "cosine_similarity(embedding, to_real_vector(:query_vector)) "
                "as similarity_score "
                "from ai_evidence_chunks "
                "where task_id = :task_id and embedding is not null "
                "order by similarity_score desc"
            ),
            {"task_id": task_id, "query_vector": query_vector},
        ).mappings().all()
        return [
            {
                "chunk_id": row["chunk_id"],
                "attachment_id": row["attachment_id"],
                "file_name": row["file_name"],
                "document_type": row["document_type"],
                "chunk_text": row["chunk_text"],
                "location_json": (
                    json.loads(row["location_json"])
                    if isinstance(row["location_json"], str)
                    else dict(row["location_json"])
                ),
                "similarity_score": float(row["similarity_score"]),
            }
            for row in rows
        ]

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
        """Persist one RAG retrieval round and its query refinement metadata.

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
        self.session.execute(
            text(
                "insert into ai_retrieval_rounds "
                "(retrieval_round_id, task_id, round_number, queries_json, "
                "retrieved_chunk_ids_json, accepted_evidence_json, "
                "evidence_gaps_json, refined_queries_json, stop_reason) "
                "values (:retrieval_round_id, :task_id, :round_number, "
                ":queries_json, :retrieved_chunk_ids_json, "
                ":accepted_evidence_json, :evidence_gaps_json, "
                ":refined_queries_json, :stop_reason)"
            ),
            {
                "retrieval_round_id": retrieval_round_id,
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
            result: Validated ``QuestionReviewResult`` to store.

        Outputs:
            None. The task receives one result row and its status becomes
            ``completed``.

        Raises:
            KeyError: Raised when no known task matches ``task_id``.
            TypeError: Raised when ``result`` is not a ``QuestionReviewResult``.
            ValueError: Raised when ``result.question_id`` does not match the
                task question ID, or the worker does not hold a non-expired
                active lease for the task.
        """

        if not isinstance(result, QuestionReviewResult):
            raise TypeError("result must be a QuestionReviewResult")

        task_row = self.session.execute(
            text(
                "select question_id from ai_question_tasks "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        ).mappings().first()
        if task_row is None:
            raise KeyError(f"Unknown AI review task ID: {task_id}")
        if result.question_id != task_row["question_id"]:
            raise ValueError(
                "Question result does not match task question ID: "
                f"{result.question_id} != {task_row['question_id']}"
            )

        lease_row = self.session.execute(
            text(
                "select question_id from ai_question_tasks "
                "where task_id = :task_id "
                "and status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {"task_id": task_id, "worker_id": worker_id},
        ).mappings().first()
        if lease_row is None:
            raise ValueError(
                "Task result cannot be saved without a non-expired active "
                f"lease for worker {worker_id}: {task_id}"
            )

        update_result = self.session.execute(
            text(
                "update ai_question_tasks "
                "set status = 'completed', updated_at = current_timestamp "
                "where task_id = :task_id "
                "and status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {"task_id": task_id, "worker_id": worker_id},
        )
        if update_result.rowcount == 0:
            raise ValueError(
                "Task result cannot be completed because the active lease is "
                f"no longer held by worker {worker_id}: {task_id}"
            )
        self.session.execute(
            text(
                "delete from ai_question_results "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        )
        self.session.execute(
            text(
                "insert into ai_question_results "
                "(result_id, task_id, question_id, result_json) "
                "values (:result_id, :task_id, :question_id, :result_json)"
            ),
            {
                "result_id": f"result-{uuid4()}",
                "task_id": task_id,
                "question_id": result.question_id,
                "result_json": result.model_dump_json(),
            },
        )

    def save_question_failure(
        self,
        task_id: str,
        worker_id: str,
        error_message: str,
    ) -> None:
        """Persist a task failure for polling clients.

        Inputs:
            task_id: Identifier of the task that failed.
            worker_id: Identifier of the active worker lease holder.
            error_message: Clear failure message to expose through job polling.

        Outputs:
            None. The task status becomes ``failed`` when the worker owns a
            non-expired active lease for the task.

        Raises:
            KeyError: Raised when no known task matches ``task_id``.
            ValueError: Raised when the worker does not hold a non-expired
            active lease for the task.
        """

        task_row = self.session.execute(
            text(
                "select question_id from ai_question_tasks "
                "where task_id = :task_id"
            ),
            {"task_id": task_id},
        ).mappings().first()
        if task_row is None:
            raise KeyError(f"Unknown AI review task ID: {task_id}")

        lease_row = self.session.execute(
            text(
                "select question_id from ai_question_tasks "
                "where task_id = :task_id "
                "and status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {"task_id": task_id, "worker_id": worker_id},
        ).mappings().first()
        if lease_row is None:
            raise ValueError(
                "Task failure cannot be saved without a non-expired active "
                f"lease for worker {worker_id}: {task_id}"
            )

        update_result = self.session.execute(
            text(
                "update ai_question_tasks "
                "set status = 'failed', error_message = :error_message, "
                "lease_owner = null, lease_expires_at = null, "
                "updated_at = current_timestamp "
                "where task_id = :task_id "
                "and status in ('in_progress', 'extracting_documents', "
                "'embedding_documents', 'retrieving_evidence', "
                "'finalizing_answer', 'retrying') "
                "and lease_owner = :worker_id "
                "and lease_expires_at > current_timestamp"
            ),
            {
                "task_id": task_id,
                "worker_id": worker_id,
                "error_message": error_message,
            },
        )
        if update_result.rowcount == 0:
            raise ValueError(
                "Task failure cannot be saved because the active lease is no "
                f"longer held by worker {worker_id}: {task_id}"
            )
