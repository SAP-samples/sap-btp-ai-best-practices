"""HANA schema DDL for the Assessment knowledge agent."""

from __future__ import annotations


JOULE_KNOWLEDGE_SCHEMA_STATEMENTS = [
    """
    create table joule_knowledge_imports (
      import_id nvarchar(64) primary key,
      source_name nvarchar(128) not null,
      source_path nvarchar(1024) not null,
      source_sha256 nvarchar(64) not null,
      source_sheet nvarchar(255) not null,
      row_count integer not null,
      imported_at timestamp default current_utctimestamp not null,
      status nvarchar(32) not null
    )
    """,
    """
    create table joule_glossary_terms (
      term_id nvarchar(128) primary key,
      language nvarchar(8) not null,
      term nvarchar(512) not null,
      normalized_term nvarchar(512) not null,
      definition nclob not null,
      source_workbook nvarchar(512) not null,
      source_sheet nvarchar(255) not null,
      source_row integer not null,
      created_at timestamp default current_utctimestamp not null
    )
    """,
    """
    create table joule_question_explanations (
      question_id nvarchar(64) primary key,
      dimension_key nvarchar(255) not null,
      source_dimension nvarchar(255) not null,
      question_text nclob not null,
      explanation nclob not null,
      source_workbook nvarchar(512) not null,
      source_sheet nvarchar(255) not null,
      source_row integer not null,
      embedding_model nvarchar(128),
      question_embedding real_vector,
      explanation_embedding real_vector,
      created_at timestamp default current_utctimestamp not null
    )
    """,
    """
    create table joule_dimensions (
      dimension_key nvarchar(255) primary key,
      display_order integer not null,
      name nvarchar(255) not null,
      aliases_json nclob not null,
      explanation nclob not null,
      source_workbook nvarchar(512) not null,
      source_sheet nvarchar(255) not null,
      source_row integer not null,
      created_at timestamp default current_utctimestamp not null
    )
    """,
    """
    create table joule_agent_messages (
      message_id nvarchar(64) primary key,
      context_id nvarchar(128) not null,
      turn_number integer not null,
      role nvarchar(32) not null,
      content nclob not null,
      metadata_json nclob,
      created_at timestamp default current_utctimestamp not null
    )
    """,
]
"""DDL statements that create the Joule knowledge agent HANA tables."""


JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS = [
    """
    create fuzzy search index joule_glossary_term_fzy
    on joule_glossary_terms (term)
    search mode text
    """,
    """
    create fuzzy search index joule_glossary_norm_fzy
    on joule_glossary_terms (normalized_term)
    search mode text
    """,
]
"""Fuzzy text indexes used by glossary term lookup."""


def table_name_from_joule_create_statement(statement: str) -> str:
    """Extract the logical table name from a Joule knowledge DDL statement.

    Inputs:
        statement: One DDL statement from
            ``JOULE_KNOWLEDGE_SCHEMA_STATEMENTS``.

    Outputs:
        str: Lowercase logical table name created by the statement.

    Raises:
        ValueError: Raised when the statement does not match the expected
        ``CREATE TABLE <name>`` shape used by this module.
    """

    normalized = " ".join(statement.strip().split())
    prefix = "create table "
    if not normalized.lower().startswith(prefix):
        raise ValueError(f"Unsupported Joule knowledge schema statement: {statement}")
    return normalized[len(prefix) :].split("(", maxsplit=1)[0].strip().lower()


def index_name_from_joule_index_statement(statement: str) -> str:
    """Extract the logical index name from a HANA fuzzy index statement.

    Inputs:
        statement: One DDL statement from
            ``JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS``.

    Outputs:
        str: Lowercase logical index name created by the statement.

    Raises:
        ValueError: Raised when the statement does not match the expected
        ``CREATE FUZZY SEARCH INDEX <name>`` shape.
    """

    normalized = " ".join(statement.strip().split())
    prefix = "create fuzzy search index "
    if not normalized.lower().startswith(prefix):
        raise ValueError(f"Unsupported Joule knowledge index statement: {statement}")
    return normalized[len(prefix) :].split(" ", maxsplit=1)[0].strip().lower()


def required_joule_knowledge_table_names() -> set[str]:
    """Return all required Joule knowledge HANA table names.

    Inputs:
        None.

    Outputs:
        set[str]: Logical table names created by
        ``JOULE_KNOWLEDGE_SCHEMA_STATEMENTS``.
    """

    return {
        "joule_knowledge_imports",
        "joule_glossary_terms",
        "joule_question_explanations",
        "joule_dimensions",
        "joule_agent_messages",
    }
