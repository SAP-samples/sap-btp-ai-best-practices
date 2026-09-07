"""Tests for HANA schema definitions used by the assessment document review PoC."""

import pytest

from app.services.hana_schema import (
    HANA_SCHEMA_STATEMENTS,
    required_table_names,
    table_name_from_create_statement,
)
from app.services.ai_review_repository.hana_schema_ops import HanaSchemaOpsMixin


def test_schema_contains_framework_and_ai_review_tables() -> None:
    """Verify all required logical tables are represented in schema DDL.

    Inputs:
        None. The test imports the schema constants directly.

    Outputs:
        None. Assertions confirm the exact table set, table creation statements,
        and intentionally omitted framework export-only answer item columns.
    """

    sql = "\n".join(HANA_SCHEMA_STATEMENTS).lower()

    assert required_table_names() == {
        "assessment_dimensions",
        "assessment_dimension_translations",
        "assessment_questions",
        "assessment_question_translations",
        "assessment_question_topic_translations",
        "assessment_answer_items",
        "assessment_answer_item_translations",
        "assessment_question_explanations",
        "assessment_framework_imports",
        "assessment_user_answers",
        "assessment_score_benchmarks",
        "assessment_benchmark_imports",
        "assessment_benchmark_companies",
        "assessment_benchmark_submissions",
        "assessment_benchmark_responses",
        "assessment_benchmark_scores",
        "assessment_profiles",
        "assessment_report_jobs",
        "ai_dimension_jobs",
        "ai_question_tasks",
        "ai_question_attempts",
        "ai_question_results",
        "ai_retrieval_rounds",
        "ai_evidence_refs",
        "ai_applied_suggestions",
        "ai_documents",
        "ai_document_ingestion_jobs",
        "ai_document_ingestion_job_documents",
        "ai_document_extractions",
        "ai_document_extracted_blocks",
        "ai_document_chunks",
        "joule_admin_documents",
        "joule_admin_document_ingestion_jobs",
        "joule_admin_document_ingestion_job_documents",
        "joule_admin_document_extractions",
        "joule_admin_document_extracted_blocks",
        "joule_admin_document_chunks",
    }
    assert "if not exists" not in sql
    for table_name in required_table_names():
        assert f"create table {table_name}" in sql

    dimension_jobs_sql = sql.split(
        "create table ai_dimension_jobs", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "language nvarchar(8) default 'en' not null" in dimension_jobs_sql

    question_translations_sql = sql.split(
        "create table assessment_question_translations", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "primary key (question_id, language)" in question_translations_sql

    answer_items_sql = sql.split(
        "create table assessment_answer_items", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "default_selected" not in answer_items_sql
    assert "optional" not in answer_items_sql

    question_results_sql = sql.split(
        "create table ai_question_results", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "task_id nvarchar(64) not null unique" in question_results_sql

    document_chunks_sql = sql.split(
        "create table ai_document_chunks", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "assessment_id nvarchar(128) not null" in document_chunks_sql
    assert "embedding real_vector" in document_chunks_sql
    assert "source_block_ids_json nclob not null" in document_chunks_sql

    retrieval_rounds_sql = sql.split(
        "create table ai_retrieval_rounds", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "round_number integer not null" in retrieval_rounds_sql
    assert "stop_reason nvarchar(128)" in retrieval_rounds_sql

    documents_sql = sql.split(
        "create table ai_documents", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "assessment_id nvarchar(128) not null" in documents_sql
    assert "content_hash nvarchar(128) not null" in documents_sql
    assert "content_blob blob not null" in documents_sql
    assert "status nvarchar(32) not null" in documents_sql
    assert "unique (assessment_id, content_hash)" in documents_sql

    ingestion_jobs_sql = sql.split(
        "create table ai_document_ingestion_jobs", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "processed_document_count integer default 0" in ingestion_jobs_sql
    assert "indexed_chunk_count integer default 0" in ingestion_jobs_sql

    admin_documents_sql = sql.split(
        "create table joule_admin_documents", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "assessment_id" not in admin_documents_sql
    assert "content_hash nvarchar(128) not null" in admin_documents_sql
    assert "content_blob blob not null" in admin_documents_sql
    assert "unique (content_hash)" in admin_documents_sql

    admin_chunks_sql = sql.split(
        "create table joule_admin_document_chunks", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "assessment_id" not in admin_chunks_sql
    assert "embedding real_vector" in admin_chunks_sql
    assert "source_block_ids_json nclob not null" in admin_chunks_sql


def test_table_name_from_create_statement_extracts_hana_table_name() -> None:
    """Verify schema table names can be parsed for idempotent HANA creation.

    Inputs:
        None. The test uses a representative schema statement.

    Outputs:
        None. Assertions confirm the logical table name is extracted without
        depending on HANA's unsupported ``IF NOT EXISTS`` syntax.
    """

    assert (
        table_name_from_create_statement(
            """
            create table ai_question_results (
              result_id nvarchar(64) primary key
            )
            """
        )
        == "ai_question_results"
    )


def test_question_tasks_include_progress_and_lease_columns() -> None:
    """Verify unified question task DDL supports independent worker leases."""
    schema = "\n".join(HANA_SCHEMA_STATEMENTS).lower()

    assert "create table ai_question_tasks" in schema
    assert "lease_owner nvarchar(128)" in schema
    assert "lease_expires_at timestamp" in schema
    assert "max_allowed_level integer default 5 not null" in schema
    assert "retry_count integer default 0" in schema
    assert "progress_message nclob" in schema
    assert "rag_call_count integer default 0" in schema
    assert "query_count integer default 0" in schema
    assert "retrieved_chunk_count integer default 0" in schema


def test_final_document_manager_schema_excludes_legacy_review_tables() -> None:
    """Verify the final schema removes legacy upload-backed review tables."""
    removed_tables = {
        "ai_task_attachments",
        "ai_attachment_extractions",
        "ai_extracted_blocks",
        "ai_evidence_chunks",
        "ai_question_chunk_runs",
        "ai_batch_jobs",
        "ai_batch_documents",
        "ai_batch_job_documents",
        "ai_batch_document_extractions",
        "ai_batch_extracted_blocks",
        "ai_batch_document_chunks",
        "ai_batch_retrieval_rounds",
        "ai_batch_question_tasks",
        "ai_batch_question_results",
    }
    schema = "\n".join(HANA_SCHEMA_STATEMENTS).lower()

    assert required_table_names().isdisjoint(removed_tables)
    for table_name in removed_tables:
        assert f"create table {table_name}" not in schema


def test_schema_contains_scoring_tables() -> None:
    """Verify scoring and benchmark state are HANA-owned."""
    schema = "\n".join(HANA_SCHEMA_STATEMENTS).lower()

    user_answers_sql = schema.split(
        "create table assessment_user_answers", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "assessment_id nvarchar(128) not null" in user_answers_sql
    assert "question_id nvarchar(64) not null" in user_answers_sql
    assert "answer_item_id nvarchar(128) not null" in user_answers_sql
    assert "customer_class nvarchar(32) not null" in user_answers_sql
    assert "source nvarchar(32) default 'manual' not null" in user_answers_sql
    assert "primary key (assessment_id, question_id, answer_item_id)" in user_answers_sql

    benchmarks_sql = schema.split(
        "create table assessment_score_benchmarks", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "sector nvarchar(128)" in benchmarks_sql
    assert "dimension nvarchar(255)" in benchmarks_sql
    assert "question_id nvarchar(64)" in benchmarks_sql
    assert "benchmark_score decimal(9,4) not null" in benchmarks_sql

    report_jobs_sql = schema.split(
        "create table assessment_report_jobs", maxsplit=1
    )[1].split("create table", maxsplit=1)[0]
    assert "source_json nclob not null" in report_jobs_sql
    assert "lease_owner nvarchar(128)" in report_jobs_sql
    assert "lease_expires_at timestamp" in report_jobs_sql
    assert "pdf_blob blob" in report_jobs_sql
    assert "expires_at timestamp" in report_jobs_sql


class _ColumnRows:
    """Return configured HANA catalog column mappings."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        """Store column rows returned by ``SYS.TABLE_COLUMNS``."""

        self.rows = rows

    def mappings(self) -> "_ColumnRows":
        """Return this object for SQLAlchemy mapping-chain compatibility."""

        return self

    def all(self) -> list[dict[str, str]]:
        """Return the configured catalog mappings."""

        return self.rows


class _ColumnSession:
    """Serve HANA column-contract rows to schema validation."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        """Store rows for the next catalog query."""

        self.rows = rows

    def execute(self, _statement: object, _parameters: dict) -> _ColumnRows:
        """Return configured column metadata."""

        return _ColumnRows(self.rows)


class _SchemaRepository(HanaSchemaOpsMixin):
    """Expose report table contract validation for tests."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        """Initialize a fake catalog session."""

        self.session = _ColumnSession(rows)


def test_report_job_table_contract_rejects_missing_or_wrong_columns() -> None:
    """Verify startup fails when an existing report table is incompatible."""

    repository = _SchemaRepository(
        [
            {
                "column_name": "JOB_ID",
                "data_type_name": "NVARCHAR",
                "is_nullable": "FALSE",
            },
            {
                "column_name": "SOURCE_JSON",
                "data_type_name": "NVARCHAR",
                "is_nullable": "FALSE",
            },
        ]
    )

    with pytest.raises(RuntimeError, match="incompatible contract"):
        repository._validate_assessment_report_jobs_contract()
