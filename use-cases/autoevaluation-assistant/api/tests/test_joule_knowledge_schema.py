"""Tests for HANA schema definitions used by the Joule knowledge agent."""

from app.services.joule_knowledge_schema import (
    JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS,
    JOULE_KNOWLEDGE_SCHEMA_STATEMENTS,
    required_joule_knowledge_table_names,
    table_name_from_joule_create_statement,
)


def test_joule_knowledge_schema_contains_required_hana_tables() -> None:
    """Verify the Joule knowledge schema declares all required HANA tables.

    Inputs:
        None. The test imports DDL constants directly.

    Outputs:
        None. Assertions validate expected table names and HANA-safe DDL shape.
    """

    sql = "\n".join(JOULE_KNOWLEDGE_SCHEMA_STATEMENTS).lower()

    assert required_joule_knowledge_table_names() == {
        "joule_knowledge_imports",
        "joule_glossary_terms",
        "joule_question_explanations",
        "joule_dimensions",
        "joule_agent_messages",
    }
    assert "if not exists" not in sql
    assert "real_vector" in sql
    assert "create table joule_question_explanations" in sql
    assert "question_embedding real_vector" in sql
    assert "explanation_embedding real_vector" in sql
    for table_name in required_joule_knowledge_table_names():
        assert f"create table {table_name}" in sql


def test_joule_knowledge_schema_declares_fuzzy_search_indexes() -> None:
    """Verify glossary lookup can use HANA fuzzy text search indexes.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm HANA fuzzy index syntax is used for the term
        columns that back glossary search.
    """

    index_sql = "\n".join(JOULE_KNOWLEDGE_FUZZY_INDEX_STATEMENTS).lower()

    assert "create fuzzy search index" in index_sql
    assert "on joule_glossary_terms" in index_sql
    assert "search mode text" in index_sql
    assert "term" in index_sql


def test_table_name_from_joule_create_statement_extracts_table_name() -> None:
    """Verify idempotent schema creation can parse Joule table DDL names.

    Inputs:
        None.

    Outputs:
        None. Assertions confirm table name extraction without unsupported
        HANA ``IF NOT EXISTS`` syntax.
    """

    assert (
        table_name_from_joule_create_statement(
            """
            create table joule_agent_messages (
              message_id nvarchar(64) primary key
            )
            """
        )
        == "joule_agent_messages"
    )
