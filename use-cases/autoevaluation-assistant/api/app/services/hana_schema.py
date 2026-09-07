"""HANA schema DDL for assessment framework and AI review state."""

from __future__ import annotations


HANA_SCHEMA_STATEMENTS = [
    """
    create table assessment_dimensions (
      dimension nvarchar(255) primary key,
      display_order integer not null,
      question_count integer not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table assessment_dimension_translations (
      dimension nvarchar(255) not null,
      language nvarchar(8) not null,
      display_name nvarchar(255) not null,
      primary key (dimension, language)
    )
    """,
    """
    create table assessment_questions (
      question_id nvarchar(64) primary key,
      dimension nvarchar(255) not null,
      section nvarchar(255) not null,
      question_text nclob not null,
      display_order integer not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table assessment_question_translations (
      question_id nvarchar(64) not null,
      language nvarchar(8) not null,
      section nvarchar(255) not null,
      question_text nclob not null,
      primary key (question_id, language)
    )
    """,
    """
    create table assessment_question_topic_translations (
      question_id nvarchar(64) not null,
      language nvarchar(8) not null,
      topic_title nvarchar(255) not null,
      primary key (question_id, language)
    )
    """,
    """
    create table assessment_answer_items (
      answer_item_id nvarchar(128) primary key,
      question_id nvarchar(64) not null,
      level integer not null,
      item_index integer not null,
      answer_text nclob not null
    )
    """,
    """
    create table assessment_answer_item_translations (
      answer_item_id nvarchar(128) not null,
      language nvarchar(8) not null,
      answer_text nclob not null,
      primary key (answer_item_id, language)
    )
    """,
    """
    create table assessment_question_explanations (
      question_id nvarchar(64) primary key,
      dimension nvarchar(255) not null,
      question_text nclob not null,
      explanation nclob not null
    )
    """,
    """
    create table assessment_framework_imports (
      import_id nvarchar(64) primary key,
      source_workbook nvarchar(512) not null,
      source_explanations nvarchar(512) not null,
      imported_at timestamp default current_timestamp,
      dimension_count integer not null,
      question_count integer not null,
      answer_item_count integer not null,
      explanation_count integer not null,
      status nvarchar(32) not null
    )
    """,
    """
    create table assessment_user_answers (
      assessment_id nvarchar(128) not null,
      question_id nvarchar(64) not null,
      answer_item_id nvarchar(128) not null,
      customer_class nvarchar(32) not null,
      source nvarchar(32) default 'manual' not null,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp,
      primary key (assessment_id, question_id, answer_item_id)
    )
    """,
    """
    create table assessment_score_benchmarks (
      benchmark_id nvarchar(64) primary key,
      customer_class nvarchar(32) not null,
      sector nvarchar(128),
      dimension nvarchar(255),
      question_id nvarchar(64),
      benchmark_score decimal(9,4) not null,
      sample_size integer default 0 not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table assessment_benchmark_imports (
      import_id nvarchar(64) primary key,
      source_workbook blob not null,
      source_filename nvarchar(512) not null,
      source_sha256 nvarchar(64) not null,
      scoring_version nvarchar(64) not null,
      row_count integer not null,
      company_count integer not null,
      questionnaire_count integer not null,
      question_count integer not null,
      accepted_count integer not null,
      rejected_count integer not null,
      warnings_json nclob not null,
      status nvarchar(32) not null,
      is_active tinyint default 0 not null,
      created_at timestamp default current_utctimestamp not null,
      activated_at timestamp
    )
    """,
    """
    create table assessment_benchmark_companies (
      import_id nvarchar(64) not null,
      source_company_id nvarchar(128) not null,
      company_name nvarchar(512),
      customer_class nvarchar(32) not null,
      revenue decimal(19,4),
      employees integer,
      nace1 nvarchar(512),
      nace2 nvarchar(512),
      nace3 nvarchar(512),
      company_size nvarchar(512),
      legal_form nvarchar(1024),
      geographic_presence nvarchar(255),
      is_listed tinyint,
      is_public_contracting_client tinyint,
      uses_self_governance_code tinyint,
      primary key (import_id, source_company_id)
    )
    """,
    """
    create table assessment_benchmark_submissions (
      import_id nvarchar(64) not null,
      questionnaire_id nvarchar(128) not null,
      source_company_id nvarchar(128) not null,
      submission_date date,
      extraction_date date,
      release_status nvarchar(64),
      raw_assessment_category nvarchar(255),
      primary key (import_id, questionnaire_id)
    )
    """,
    """
    create table assessment_benchmark_responses (
      import_id nvarchar(64) not null,
      questionnaire_id nvarchar(128) not null,
      question_id nvarchar(64) not null,
      canonical_answer_id nvarchar(128) not null,
      source_answer_id nvarchar(128) not null,
      source_answer_text nclob not null,
      source_answer_value nvarchar(64),
      level integer not null,
      selected tinyint not null,
      optional tinyint not null,
      managed tinyint not null,
      validation_status nvarchar(64),
      primary key (import_id, questionnaire_id, question_id, canonical_answer_id)
    )
    """,
    """
    create table assessment_benchmark_scores (
      score_id nvarchar(64) primary key,
      import_id nvarchar(64) not null,
      questionnaire_id nvarchar(128) not null,
      scope_type nvarchar(32) not null,
      dimension nvarchar(255),
      topic nvarchar(255),
      question_id nvarchar(64),
      calculated_score decimal(9,4) not null,
      supplied_score decimal(9,4)
    )
    """,
    """
    create table assessment_profiles (
      assessment_id nvarchar(128) primary key,
      display_name nvarchar(512) not null,
      source_company_id nvarchar(128),
      customer_class nvarchar(32) not null,
      nace1 nvarchar(512),
      created_at timestamp default current_utctimestamp not null,
      updated_at timestamp default current_utctimestamp not null
    )
    """,
    """
    create table assessment_report_jobs (
      job_id nvarchar(64) primary key,
      assessment_id nvarchar(128) not null,
      customer_class nvarchar(32) not null,
      sector nvarchar(128),
      language nvarchar(8) not null,
      status nvarchar(32) not null,
      source_json nclob not null,
      progress_message nclob,
      lease_owner nvarchar(128),
      lease_expires_at timestamp,
      retry_count integer default 0 not null,
      file_name nvarchar(512),
      pdf_blob blob,
      error_code nvarchar(128),
      error_message nclob,
      expires_at timestamp,
      created_at timestamp default current_utctimestamp not null,
      updated_at timestamp default current_utctimestamp not null
    )
    """,
    """
    create table ai_dimension_jobs (
      job_id nvarchar(64) primary key,
      assessment_id nvarchar(128) not null,
      dimension nvarchar(255) not null,
      language nvarchar(8) default 'en' not null,
      status nvarchar(32) not null,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_question_tasks (
      task_id nvarchar(64) primary key,
      job_id nvarchar(64) not null,
      question_id nvarchar(64) not null,
      dimension nvarchar(255) not null,
      current_answers_json nclob not null,
      max_allowed_level integer default 5 not null,
      status nvarchar(32) not null,
      lease_owner nvarchar(128),
      lease_expires_at timestamp,
      retry_count integer default 0,
      error_code nvarchar(128),
      error_message nclob,
      progress_message nclob,
      rag_call_count integer default 0,
      query_count integer default 0,
      retrieved_chunk_count integer default 0,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_question_attempts (
      attempt_id nvarchar(64) primary key,
      task_id nvarchar(64) not null,
      mode nvarchar(32) not null,
      status nvarchar(32) not null,
      model nvarchar(128),
      usage_json nclob,
      error_code nvarchar(128),
      error_message nclob,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_question_results (
      result_id nvarchar(64) primary key,
      task_id nvarchar(64) not null unique,
      question_id nvarchar(64) not null,
      result_json nclob not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_retrieval_rounds (
      retrieval_round_id nvarchar(64) primary key,
      task_id nvarchar(64) not null,
      round_number integer not null,
      queries_json nclob not null,
      retrieved_chunk_ids_json nclob not null,
      accepted_evidence_json nclob not null,
      evidence_gaps_json nclob not null,
      refined_queries_json nclob not null,
      stop_reason nvarchar(128),
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_evidence_refs (
      evidence_ref_id nvarchar(64) primary key,
      task_id nvarchar(64) not null,
      answer_item_id nvarchar(128),
      evidence_json nclob not null
    )
    """,
    """
    create table ai_applied_suggestions (
      applied_id nvarchar(64) primary key,
      task_id nvarchar(64) not null,
      question_id nvarchar(64) not null,
      applied_answer_item_ids_json nclob not null,
      applied_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_documents (
      document_id nvarchar(64) primary key,
      assessment_id nvarchar(128) not null,
      content_hash nvarchar(128) not null,
      file_name nvarchar(512) not null,
      content_type nvarchar(255) not null,
      content_blob blob not null,
      status nvarchar(32) not null,
      error_message nclob,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp,
      unique (assessment_id, content_hash)
    )
    """,
    """
    create table ai_document_ingestion_jobs (
      job_id nvarchar(64) primary key,
      assessment_id nvarchar(128) not null,
      status nvarchar(32) not null,
      lease_owner nvarchar(128),
      lease_expires_at timestamp,
      retry_count integer default 0,
      document_count integer default 0,
      processed_document_count integer default 0,
      indexed_chunk_count integer default 0,
      error_code nvarchar(128),
      error_message nclob,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_document_ingestion_job_documents (
      link_id nvarchar(64) primary key,
      job_id nvarchar(64) not null,
      document_id nvarchar(64) not null,
      created_at timestamp default current_timestamp,
      unique (job_id, document_id)
    )
    """,
    """
    create table ai_document_extractions (
      extraction_id nvarchar(64) primary key,
      document_id nvarchar(64) not null,
      file_name nvarchar(512) not null,
      document_type nvarchar(64) not null,
      page_count integer,
      extracted_block_count integer not null,
      total_characters integer not null,
      estimated_tokens integer not null,
      quality_json nclob not null,
      warnings_json nclob not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_document_extracted_blocks (
      block_row_id nvarchar(64) primary key,
      extraction_id nvarchar(64) not null,
      document_id nvarchar(64) not null,
      block_id nvarchar(128) not null,
      block_type nvarchar(64) not null,
      text nclob not null,
      location_json nclob not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table ai_document_chunks (
      chunk_id nvarchar(64) primary key,
      assessment_id nvarchar(128) not null,
      document_id nvarchar(64) not null,
      file_name nvarchar(512) not null,
      document_type nvarchar(64) not null,
      source_block_ids_json nclob not null,
      location_json nclob not null,
      chunk_text nclob not null,
      estimated_tokens integer not null,
      embedding_model nvarchar(128) not null,
      embedding real_vector,
      content_hash nvarchar(128) not null,
      chunk_kind nvarchar(32) default 'raw' not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table joule_admin_documents (
      document_id nvarchar(64) primary key,
      content_hash nvarchar(128) not null,
      file_name nvarchar(512) not null,
      content_type nvarchar(255) not null,
      content_blob blob not null,
      status nvarchar(32) not null,
      error_message nclob,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp,
      unique (content_hash)
    )
    """,
    """
    create table joule_admin_document_ingestion_jobs (
      job_id nvarchar(64) primary key,
      status nvarchar(32) not null,
      lease_owner nvarchar(128),
      lease_expires_at timestamp,
      retry_count integer default 0,
      document_count integer default 0,
      processed_document_count integer default 0,
      indexed_chunk_count integer default 0,
      error_code nvarchar(128),
      error_message nclob,
      created_at timestamp default current_timestamp,
      updated_at timestamp default current_timestamp
    )
    """,
    """
    create table joule_admin_document_ingestion_job_documents (
      link_id nvarchar(64) primary key,
      job_id nvarchar(64) not null,
      document_id nvarchar(64) not null,
      created_at timestamp default current_timestamp,
      unique (job_id, document_id)
    )
    """,
    """
    create table joule_admin_document_extractions (
      extraction_id nvarchar(64) primary key,
      document_id nvarchar(64) not null,
      file_name nvarchar(512) not null,
      document_type nvarchar(64) not null,
      page_count integer,
      extracted_block_count integer not null,
      total_characters integer not null,
      estimated_tokens integer not null,
      quality_json nclob not null,
      warnings_json nclob not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table joule_admin_document_extracted_blocks (
      block_row_id nvarchar(64) primary key,
      extraction_id nvarchar(64) not null,
      document_id nvarchar(64) not null,
      block_id nvarchar(128) not null,
      block_type nvarchar(64) not null,
      text nclob not null,
      location_json nclob not null,
      created_at timestamp default current_timestamp
    )
    """,
    """
    create table joule_admin_document_chunks (
      chunk_id nvarchar(64) primary key,
      document_id nvarchar(64) not null,
      file_name nvarchar(512) not null,
      document_type nvarchar(64) not null,
      source_block_ids_json nclob not null,
      location_json nclob not null,
      chunk_text nclob not null,
      estimated_tokens integer not null,
      embedding_model nvarchar(128) not null,
      embedding real_vector,
      content_hash nvarchar(128) not null,
      chunk_kind nvarchar(32) default 'raw' not null,
      created_at timestamp default current_timestamp
    )
    """,
]
"""DDL statements that create the logical framework and AI review tables."""


def table_name_from_create_statement(statement: str) -> str:
    """Extract the unquoted table name from a HANA ``CREATE TABLE`` statement.

    Inputs:
        statement: One DDL statement from ``HANA_SCHEMA_STATEMENTS``.

    Outputs:
        str: Lowercase logical table name created by the statement.

    Raises:
        ValueError: Raised when the statement does not match the expected
        ``CREATE TABLE <name>`` shape used by this module.
    """

    normalized = " ".join(statement.strip().split())
    prefix = "create table "
    if not normalized.lower().startswith(prefix):
        raise ValueError(f"Unsupported schema statement: {statement}")
    return normalized[len(prefix) :].split("(", maxsplit=1)[0].strip().lower()


def required_table_names() -> set[str]:
    """Return required table names for the PoC schema.

    Inputs:
        None.

    Outputs:
        set[str]: Logical table names created by ``HANA_SCHEMA_STATEMENTS``.
    """

    return {
        table_name_from_create_statement(statement)
        for statement in HANA_SCHEMA_STATEMENTS
    }
