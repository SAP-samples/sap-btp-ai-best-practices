"""Schema creation and migration helpers for the HANA AI review repository."""

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
    _derive_finished_batch_status,
    _derive_job_status,
    _document_payload_bytes,
    _document_phase_processed_count,
    _normalize_task_status_for_response,
    _pending_task_status_for_runtime,
    _row_batches,
    document_content_hash,
)

class HanaSchemaOpsMixin:
    """Provide schema setup and optional-table cleanup helpers."""

    def create_schema(self) -> None:
        """Create all HANA tables required by the repository if absent.

        Inputs:
            None. DDL is read from ``HANA_SCHEMA_STATEMENTS``.

        Outputs:
            None. The active session receives one DDL statement per schema item.
        """

        for statement in HANA_SCHEMA_STATEMENTS:
            table_name = table_name_from_create_statement(statement)
            if self._table_exists(table_name):
                continue
            self.session.execute(text(statement))
        self._ensure_ai_dimension_jobs_language_column()
        self._ensure_question_scope_columns()
        self._ensure_question_progress_columns()
        self._ensure_batch_progress_columns()
        self._validate_assessment_report_jobs_contract()

    def _table_exists(self, table_name: str) -> bool:
        """Return whether a table already exists in the current HANA schema.

        Inputs:
            table_name: Logical table name from a schema DDL statement.

        Outputs:
            bool: ``True`` when HANA reports the table in the current schema,
            otherwise ``False``.
        """

        row = self.session.execute(
            text(
                "select table_name "
                "from sys.tables "
                "where schema_name = current_schema "
                "and table_name = :table_name"
            ),
            {"table_name": table_name.upper()},
        ).first()
        return row is not None

    def _delete_task_scoped_rows_if_table_exists(
        self,
        table_name: str,
        task_filter: str,
        parameters: dict[str, str],
    ) -> None:
        """Delete task-scoped child rows when the child table exists.

        Inputs:
            table_name: HANA table name that contains a ``task_id`` column.
            task_filter: SQL subquery that selects the task IDs to clear.
            parameters: Bind parameters used by the task filter.

        Outputs:
            None. Missing optional child tables are skipped so older deployed
            schemas can still reset parent review jobs and tasks.
        """

        if not self._table_exists(table_name):
            return
        self.session.execute(
            text(f"delete from {table_name} where task_id in ({task_filter})"),
            parameters,
        )

    def _delete_job_scoped_rows_if_table_exists(
        self,
        table_name: str,
        job_id: str,
    ) -> None:
        """Delete rows keyed by ``job_id`` when a batch child table exists.

        Inputs:
            table_name: HANA table name that contains a ``job_id`` column.
            job_id: Batch job identifier whose child rows should be removed.

        Outputs:
            None. Missing optional child tables are skipped so older deployed
            schemas can still reset parent review state.
        """

        if not self._table_exists(table_name):
            return
        self.session.execute(
            text(f"delete from {table_name} where job_id = :job_id"),
            {"job_id": job_id},
        )

    def _delete_job_scoped_rows_for_filter_if_table_exists(
        self,
        table_name: str,
        job_filter: str,
        parameters: dict[str, str],
    ) -> None:
        """Delete rows keyed by jobs selected by a fixed SQL filter.

        Inputs:
            table_name: HANA table name that contains a ``job_id`` column.
            job_filter: SQL subquery selecting job IDs to clear.
            parameters: Bind parameters used by the job filter.

        Outputs:
            None. Missing optional batch tables are skipped so older deployed
            schemas can still reset parent state.
        """

        if not self._table_exists(table_name):
            return
        self.session.execute(
            text(f"delete from {table_name} where job_id in ({job_filter})"),
            parameters,
        )

    def _column_exists(self, table_name: str, column_name: str) -> bool:
        """Return whether a column exists in the current HANA schema.

        Inputs:
            table_name: Logical table name to inspect.
            column_name: Logical column name to inspect.

        Outputs:
            bool: ``True`` when HANA reports the column in the current schema,
            otherwise ``False``.
        """

        row = self.session.execute(
            text(
                "select column_name "
                "from sys.table_columns "
                "where schema_name = current_schema "
                "and table_name = :table_name "
                "and column_name = :column_name"
            ),
            {
                "table_name": table_name.upper(),
                "column_name": column_name.upper(),
            },
        ).first()
        return row is not None

    def _validate_assessment_report_jobs_contract(self) -> None:
        """Verify the existing report queue table matches its required contract.

        Inputs:
            None. Column metadata is read from HANA ``SYS.TABLE_COLUMNS``.

        Outputs:
            None. A compatible table returns without mutation.

        Raises:
            RuntimeError: If required columns, HANA types, or nullability differ.
        """

        expected = {
            "JOB_ID": ("NVARCHAR", "FALSE"),
            "ASSESSMENT_ID": ("NVARCHAR", "FALSE"),
            "CUSTOMER_CLASS": ("NVARCHAR", "FALSE"),
            "LANGUAGE": ("NVARCHAR", "FALSE"),
            "STATUS": ("NVARCHAR", "FALSE"),
            "SOURCE_JSON": ("NCLOB", "FALSE"),
            "LEASE_OWNER": ("NVARCHAR", "TRUE"),
            "LEASE_EXPIRES_AT": ("TIMESTAMP", "TRUE"),
            "PDF_BLOB": ("BLOB", "TRUE"),
            "EXPIRES_AT": ("TIMESTAMP", "TRUE"),
        }
        rows = self.session.execute(
            text(
                "select column_name, data_type_name, is_nullable "
                "from sys.table_columns "
                "where schema_name = current_schema "
                "and table_name = :table_name"
            ),
            {"table_name": "ASSESSMENT_REPORT_JOBS"},
        ).mappings().all()
        actual = {
            str(row["column_name"]).upper(): (
                str(row["data_type_name"]).upper(),
                str(row["is_nullable"]).upper(),
            )
            for row in rows
        }
        mismatches = {
            column: {"expected": contract, "actual": actual.get(column)}
            for column, contract in expected.items()
            if actual.get(column) != contract
        }
        if mismatches:
            raise RuntimeError(
                "assessment_report_jobs has an incompatible contract: "
                f"{mismatches}"
            )

    def _ensure_ai_dimension_jobs_language_column(self) -> None:
        """Add the job language column for existing HANA deployments.

        Inputs:
            None. The method checks the current schema for ``ai_dimension_jobs``.

        Outputs:
            None. A HANA ``ALTER TABLE ... ADD (...)`` statement is executed
            only when the table exists without the ``language`` column.
        """

        if not self._table_exists("ai_dimension_jobs"):
            return
        if self._column_exists("ai_dimension_jobs", "language"):
            return
        self.session.execute(
            text(
                "alter table ai_dimension_jobs add "
                "(language nvarchar(8) default 'en' not null)"
            )
        )

    def _ensure_question_progress_columns(self) -> None:
        """Add manual question progress columns for existing HANA deployments.

        Inputs:
            None. The method checks ``ai_question_tasks`` for the progress
            fields needed by manual upload polling.

        Outputs:
            None. Missing HANA columns are added without changing existing rows.
        """

        if not self._table_exists("ai_question_tasks"):
            return
        columns = [
            ("progress_message", "nclob"),
            ("rag_call_count", "integer default 0"),
            ("query_count", "integer default 0"),
            ("retrieved_chunk_count", "integer default 0"),
        ]
        for column_name, column_definition in columns:
            if self._column_exists("ai_question_tasks", column_name):
                continue
            self.session.execute(
                text(
                    "alter table ai_question_tasks add "
                    f"({column_name} {column_definition})"
                )
            )

    def _ensure_question_scope_columns(self) -> None:
        """Add customer-class scope columns for existing HANA deployments.

        Inputs:
            None. The method checks ``ai_question_tasks`` for the persisted max
            allowed answer level.

        Outputs:
            None. Missing HANA columns are added without changing existing rows.
        """

        if not self._table_exists("ai_question_tasks"):
            return
        if self._column_exists("ai_question_tasks", "max_allowed_level"):
            return
        self.session.execute(
            text(
                "alter table ai_question_tasks add "
                "(max_allowed_level integer default 5 not null)"
            )
        )

    def _ensure_batch_progress_columns(self) -> None:
        """Add batch progress columns for existing HANA deployments.

        Inputs:
            None. The method checks existing batch tables for missing columns.

        Outputs:
            None. ALTER TABLE statements are executed only for missing columns.
        """
        if self._table_exists("ai_batch_jobs"):
            if not self._column_exists("ai_batch_jobs", "indexed_chunk_count"):
                self.session.execute(
                    text(
                        "alter table ai_batch_jobs add ("
                        "indexed_chunk_count integer default 0, "
                        "active_question_id nvarchar(64), "
                        "active_dimension nvarchar(255))"
                    )
                )

        if self._table_exists("ai_batch_question_tasks"):
            if not self._column_exists("ai_batch_question_tasks", "lease_owner"):
                self.session.execute(
                    text(
                        "alter table ai_batch_question_tasks add ("
                        "lease_owner nvarchar(128), "
                        "lease_expires_at timestamp, "
                        "retry_count integer default 0, "
                        "progress_message nclob, "
                        "rag_call_count integer default 0, "
                        "query_count integer default 0, "
                        "retrieved_chunk_count integer default 0)"
                    )
                )
