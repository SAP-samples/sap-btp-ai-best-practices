"""Contract tests for the versioned NACE benchmark import surface."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, required_table_names


BENCHMARK_TABLE_COLUMNS = {
    "assessment_benchmark_imports": {
        "import_id",
        "source_workbook",
        "source_filename",
        "source_sha256",
        "scoring_version",
        "row_count",
        "company_count",
        "questionnaire_count",
        "question_count",
        "accepted_count",
        "rejected_count",
        "warnings_json",
        "status",
        "is_active",
        "created_at",
        "activated_at",
    },
    "assessment_benchmark_companies": {
        "import_id",
        "source_company_id",
        "company_name",
        "customer_class",
        "revenue",
        "employees",
        "nace1",
        "nace2",
        "nace3",
        "company_size",
        "legal_form",
        "geographic_presence",
        "is_listed",
        "is_public_contracting_client",
        "uses_self_governance_code",
    },
    "assessment_benchmark_submissions": {
        "import_id",
        "questionnaire_id",
        "source_company_id",
        "submission_date",
        "extraction_date",
        "release_status",
        "raw_assessment_category",
    },
    "assessment_benchmark_responses": {
        "import_id",
        "questionnaire_id",
        "question_id",
        "canonical_answer_id",
        "source_answer_id",
        "source_answer_text",
        "source_answer_value",
        "level",
        "selected",
        "optional",
        "managed",
        "validation_status",
    },
    "assessment_benchmark_scores": {
        "score_id",
        "import_id",
        "questionnaire_id",
        "scope_type",
        "dimension",
        "topic",
        "question_id",
        "calculated_score",
        "supplied_score",
    },
    "assessment_profiles": {
        "assessment_id",
        "display_name",
        "source_company_id",
        "customer_class",
        "nace1",
        "created_at",
        "updated_at",
    },
}
"""Exact logical columns required for each benchmark-owned HANA table."""


def _table_statement(table_name: str) -> str:
    """Return normalized DDL for one table from the shared schema list.

    Inputs:
        table_name: Lowercase HANA table name to locate.

    Outputs:
        str: Lowercase whitespace-normalized ``CREATE TABLE`` statement.
    """

    prefix = f"create table {table_name} "
    return next(
        " ".join(statement.lower().split())
        for statement in HANA_SCHEMA_STATEMENTS
        if " ".join(statement.lower().split()).startswith(prefix)
    )


@pytest.mark.parametrize(
    ("module_name", "attributes"),
    [
        (
            "app.services.benchmark_import",
            {
                "parse_benchmark_workbook",
                "import_benchmark_workbook",
                "BenchmarkValidationError",
            },
        ),
        (
            "app.services.benchmark_import.scoring",
            {"calculate_benchmark_scores"},
        ),
        (
            "app.services.benchmark_import.repository",
            {"HanaBenchmarkRepository", "write_benchmark_dataset"},
        ),
        ("scripts.import_assessment_benchmarks", {"main"}),
        ("scripts.generate_synthetic_benchmark_workbook", {"generate_workbook"}),
    ],
)
def test_benchmark_import_modules_expose_decomposed_api(
    module_name: str,
    attributes: set[str],
) -> None:
    """Verify the importer is split into parser, scoring, repository, and CLIs.

    Inputs:
        module_name: Import path for one production module.
        attributes: Public callables or classes that module must expose.

    Outputs:
        None. Assertions fail until the requested package surface exists.
    """

    try:
        module: ModuleType | None = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None

    assert module is not None, f"Missing benchmark import module: {module_name}"
    assert attributes <= set(dir(module))


def test_schema_contains_exact_benchmark_table_contracts() -> None:
    """Verify versioned benchmark imports and assessment profiles are HANA-owned.

    Inputs:
        None. The test reads the repository's shared HANA DDL declarations.

    Outputs:
        None. Assertions validate required tables, columns, keys, and types.
    """

    assert set(BENCHMARK_TABLE_COLUMNS) <= required_table_names()
    for table_name, columns in BENCHMARK_TABLE_COLUMNS.items():
        statement = _table_statement(table_name)
        for column in columns:
            assert f"{column} " in statement

    imports_sql = _table_statement("assessment_benchmark_imports")
    assert "source_workbook blob not null" in imports_sql
    assert "source_sha256 nvarchar(64) not null" in imports_sql
    assert "warnings_json nclob not null" in imports_sql
    assert "is_active tinyint default 0 not null" in imports_sql

    companies_sql = _table_statement("assessment_benchmark_companies")
    assert "primary key (import_id, source_company_id)" in companies_sql

    submissions_sql = _table_statement("assessment_benchmark_submissions")
    assert "primary key (import_id, questionnaire_id)" in submissions_sql

    responses_sql = _table_statement("assessment_benchmark_responses")
    assert (
        "primary key (import_id, questionnaire_id, question_id, canonical_answer_id)"
        in responses_sql
    )

    scores_sql = _table_statement("assessment_benchmark_scores")
    assert "scope_type nvarchar(32) not null" in scores_sql
    assert "calculated_score decimal(9,4) not null" in scores_sql

    profiles_sql = _table_statement("assessment_profiles")
    assert "assessment_id nvarchar(128) primary key" in profiles_sql
    assert "updated_at timestamp default current_utctimestamp not null" in profiles_sql
