"""Transactional HANA persistence for validated benchmark datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from sqlalchemy import text

from app.services.hana_schema import HANA_SCHEMA_STATEMENTS, table_name_from_create_statement

from .models import (
    BenchmarkCompany,
    BenchmarkDataset,
    BenchmarkResponse,
    BenchmarkScore,
    BenchmarkSubmission,
    BenchmarkWriteResult,
)


BENCHMARK_REQUIRED_COLUMNS = {
    "ASSESSMENT_BENCHMARK_IMPORTS": {
        "IMPORT_ID",
        "SOURCE_WORKBOOK",
        "SOURCE_FILENAME",
        "SOURCE_SHA256",
        "SCORING_VERSION",
        "ROW_COUNT",
        "COMPANY_COUNT",
        "QUESTIONNAIRE_COUNT",
        "QUESTION_COUNT",
        "ACCEPTED_COUNT",
        "REJECTED_COUNT",
        "WARNINGS_JSON",
        "STATUS",
        "IS_ACTIVE",
        "CREATED_AT",
        "ACTIVATED_AT",
    },
    "ASSESSMENT_BENCHMARK_COMPANIES": {
        "IMPORT_ID",
        "SOURCE_COMPANY_ID",
        "COMPANY_NAME",
        "CUSTOMER_CLASS",
        "REVENUE",
        "EMPLOYEES",
        "NACE1",
        "NACE2",
        "NACE3",
        "COMPANY_SIZE",
        "LEGAL_FORM",
        "GEOGRAPHIC_PRESENCE",
        "IS_LISTED",
        "IS_PUBLIC_CONTRACTING_CLIENT",
        "USES_SELF_GOVERNANCE_CODE",
    },
    "ASSESSMENT_BENCHMARK_SUBMISSIONS": {
        "IMPORT_ID",
        "QUESTIONNAIRE_ID",
        "SOURCE_COMPANY_ID",
        "SUBMISSION_DATE",
        "EXTRACTION_DATE",
        "RELEASE_STATUS",
        "RAW_ASSESSMENT_CATEGORY",
    },
    "ASSESSMENT_BENCHMARK_RESPONSES": {
        "IMPORT_ID",
        "QUESTIONNAIRE_ID",
        "QUESTION_ID",
        "CANONICAL_ANSWER_ID",
        "SOURCE_ANSWER_ID",
        "SOURCE_ANSWER_TEXT",
        "SOURCE_ANSWER_VALUE",
        "LEVEL",
        "SELECTED",
        "OPTIONAL",
        "MANAGED",
        "VALIDATION_STATUS",
    },
    "ASSESSMENT_BENCHMARK_SCORES": {
        "SCORE_ID",
        "IMPORT_ID",
        "QUESTIONNAIRE_ID",
        "SCOPE_TYPE",
        "DIMENSION",
        "TOPIC",
        "QUESTION_ID",
        "CALCULATED_SCORE",
        "SUPPLIED_SCORE",
    },
    "ASSESSMENT_PROFILES": {
        "ASSESSMENT_ID",
        "DISPLAY_NAME",
        "SOURCE_COMPANY_ID",
        "CUSTOMER_CLASS",
        "NACE1",
        "CREATED_AT",
        "UPDATED_AT",
    },
}
"""Exact uppercase HANA column names required by benchmark-owned tables."""

BENCHMARK_TABLE_NAMES = tuple(
    table_name.lower() for table_name in BENCHMARK_REQUIRED_COLUMNS
)
"""Logical benchmark tables created and validated by the import write path."""


def _benchmark_schema_statements() -> list[str]:
    """Return shared HANA DDL statements for benchmark-owned tables only.

    Inputs:
        None. Statements are selected from the application's canonical DDL list.

    Outputs:
        list[str]: Ordered ``CREATE TABLE`` statements used by this repository.
    """

    return [
        statement
        for statement in HANA_SCHEMA_STATEMENTS
        if table_name_from_create_statement(statement) in BENCHMARK_TABLE_NAMES
    ]


def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield ordered, bounded non-empty HANA executemany batches.

    Inputs:
        rows: Prepared DML parameter mappings.
        batch_size: Positive maximum rows per batch.

    Outputs:
        Iterable[list[dict[str, Any]]]: Ordered slices no larger than batch size.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for index in range(0, len(rows), batch_size):
        yield rows[index : index + batch_size]


def _optional_flag(value: bool | None) -> int | None:
    """Convert an optional Python boolean to a HANA tinyint value.

    Inputs:
        value: Optional normalized profile flag.

    Outputs:
        int | None: ``1``, ``0``, or ``None`` for SQL binding.
    """

    return None if value is None else int(value)


def _company_parameters(company: BenchmarkCompany) -> dict[str, Any]:
    """Convert a normalized company model to HANA insert parameters.

    Inputs:
        company: Validated immutable company profile.

    Outputs:
        dict[str, Any]: Named SQL parameters matching company table columns.
    """

    return {
        "import_id": company.import_id,
        "source_company_id": company.source_company_id,
        "company_name": company.company_name,
        "customer_class": company.customer_class,
        "revenue": company.revenue,
        "employees": company.employees,
        "nace1": company.nace1,
        "nace2": company.nace2,
        "nace3": company.nace3,
        "company_size": company.company_size,
        "legal_form": company.legal_form,
        "geographic_presence": company.geographic_presence,
        "is_listed": _optional_flag(company.is_listed),
        "is_public_contracting_client": _optional_flag(
            company.is_public_contracting_client
        ),
        "uses_self_governance_code": _optional_flag(
            company.uses_self_governance_code
        ),
    }


def _submission_parameters(submission: BenchmarkSubmission) -> dict[str, Any]:
    """Convert a normalized submission model to HANA insert parameters.

    Inputs:
        submission: Validated immutable questionnaire metadata.

    Outputs:
        dict[str, Any]: Named SQL parameters matching submission columns.
    """

    return {
        "import_id": submission.import_id,
        "questionnaire_id": submission.questionnaire_id,
        "source_company_id": submission.source_company_id,
        "submission_date": submission.submission_date,
        "extraction_date": submission.extraction_date,
        "release_status": submission.release_status,
        "raw_assessment_category": submission.raw_assessment_category,
    }


def _response_parameters(response: BenchmarkResponse) -> dict[str, Any]:
    """Convert a normalized response model to HANA insert parameters.

    Inputs:
        response: Canonically mapped immutable source response.

    Outputs:
        dict[str, Any]: Named SQL parameters matching response columns.
    """

    return {
        "import_id": response.import_id,
        "questionnaire_id": response.questionnaire_id,
        "question_id": response.question_id,
        "canonical_answer_id": response.canonical_answer_id,
        "source_answer_id": response.source_answer_id,
        "source_answer_text": response.source_answer_text,
        "source_answer_value": response.source_answer_value,
        "level": response.level,
        "selected": int(response.selected),
        "optional": int(response.optional),
        "managed": int(response.managed),
        "validation_status": response.validation_status,
    }


def _score_parameters(score: BenchmarkScore) -> dict[str, Any]:
    """Convert a calculated score model to HANA insert parameters.

    Inputs:
        score: Immutable calculated benchmark score.

    Outputs:
        dict[str, Any]: Named SQL parameters matching score columns.
    """

    return {
        "score_id": score.score_id,
        "import_id": score.import_id,
        "questionnaire_id": score.questionnaire_id,
        "scope_type": score.scope_type,
        "dimension": score.dimension,
        "topic": score.topic,
        "question_id": score.question_id,
        "calculated_score": score.calculated_score,
        "supplied_score": score.supplied_score,
    }


class HanaBenchmarkRepository:
    """Persist one validated benchmark import through a SQLAlchemy session.

    Inputs:
        session: Active SQLAlchemy-compatible HANA session.

    Outputs:
        Repository exposing schema and versioned import operations.
    """

    def __init__(self, session: Any) -> None:
        """Store the HANA session used by later repository operations.

        Inputs:
            session: SQLAlchemy-compatible session supplied by the caller.

        Outputs:
            None. The session is retained on the repository instance.
        """

        self.session = session

    def _table_exists(self, table_name: str) -> bool:
        """Return whether the current HANA schema contains one table.

        Inputs:
            table_name: Lowercase logical HANA table name.

        Outputs:
            bool: ``True`` when ``SYS.TABLES`` reports the table.
        """

        row = self.session.execute(
            text(
                "select table_name from sys.tables "
                "where schema_name = current_schema and table_name = :table_name"
            ),
            {"table_name": table_name.upper()},
        ).first()
        return row is not None

    def _validate_table_contract(self, table_name: str) -> None:
        """Reject an existing benchmark table with incompatible columns.

        Inputs:
            table_name: Lowercase logical HANA table name.

        Outputs:
            None. Exact contracts return without mutation.

        Raises:
            RuntimeError: If required and actual column-name sets differ.
        """

        rows = self.session.execute(
            text(
                "select column_name from sys.table_columns "
                "where schema_name = current_schema and table_name = :table_name"
            ),
            {"table_name": table_name.upper()},
        ).mappings().all()
        actual = {str(row["column_name"]).upper() for row in rows}
        expected = BENCHMARK_REQUIRED_COLUMNS[table_name.upper()]
        if actual != expected:
            raise RuntimeError(
                f"{table_name} has an incompatible contract: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    def ensure_schema(self) -> None:
        """Idempotently create and validate all benchmark-owned HANA tables.

        Inputs:
            None. Canonical DDL comes from ``HANA_SCHEMA_STATEMENTS``.

        Outputs:
            None. Missing tables are created and every table is column-validated.
        """

        for statement in _benchmark_schema_statements():
            table_name = table_name_from_create_statement(statement)
            if not self._table_exists(table_name):
                self.session.execute(text(statement))
            self._validate_table_contract(table_name)

    def _active_import_for_sha(self, source_sha256: str) -> str | None:
        """Return the active import ID for an identical source hash.

        Inputs:
            source_sha256: Hex-encoded incoming workbook hash.

        Outputs:
            str | None: Existing active import identity, or ``None``.
        """

        row = self.session.execute(
            text(
                "select import_id from assessment_benchmark_imports "
                "where source_sha256 = :source_sha256 and is_active = 1"
            ),
            {"source_sha256": source_sha256},
        ).first()
        return None if row is None else str(row[0])

    def _insert_batches(
        self,
        table_name: str,
        statement: str,
        rows: list[dict[str, Any]],
        batch_size: int,
    ) -> None:
        """Insert prepared rows using bounded SQLAlchemy executemany batches.

        Inputs:
            table_name: Logical child table used in error context.
            statement: Parameterized HANA ``INSERT`` statement.
            rows: Prepared named-parameter mappings.
            batch_size: Positive maximum mappings per execute call.

        Outputs:
            None. Every row is sent to the active transaction in order.
        """

        for batch in _batches(rows, batch_size):
            self.session.execute(text(statement), batch)

    def _validate_insert_count(
        self,
        table_name: str,
        import_id: str,
        expected_count: int,
    ) -> None:
        """Verify one child table contains the expected new-import row count.

        Inputs:
            table_name: Logical child table to count.
            import_id: New inactive import identity.
            expected_count: Number of normalized rows sent to that table.

        Outputs:
            None. Matching counts return before activation.

        Raises:
            RuntimeError: If HANA readback count differs from expected.
        """

        actual_count = self.session.execute(
            text(f"select count(*) from {table_name} where import_id = :import_id"),
            {"import_id": import_id},
        ).scalar_one()
        if int(actual_count) != expected_count:
            raise RuntimeError(
                f"{table_name} insert validation failed: "
                f"expected {expected_count}, found {actual_count}"
            )

    def store_dataset(
        self,
        dataset: BenchmarkDataset,
        *,
        batch_size: int = 250,
    ) -> BenchmarkWriteResult:
        """Insert an inactive version, validate it, then switch activation.

        Inputs:
            dataset: Fully validated benchmark dataset with zero rejected rows.
            batch_size: Positive maximum child rows per executemany call.

        Outputs:
            BenchmarkWriteResult: Active import or identical-SHA no-op outcome.

        Raises:
            ValueError: If the dataset is not successful or has rejected rows.
            RuntimeError: If inserted row counts fail validation.
        """

        if not dataset.summary.success or dataset.summary.rejected_count:
            raise ValueError("Only fully validated benchmark datasets can be written")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.ensure_schema()
        existing_import_id = self._active_import_for_sha(
            dataset.summary.source_sha256
        )
        if existing_import_id is not None:
            return BenchmarkWriteResult(
                import_id=existing_import_id,
                status="no_op",
                no_op=True,
            )

        import_id = dataset.summary.import_id
        if import_id is None:
            raise ValueError("Validated benchmark dataset is missing import_id")
        self.session.execute(
            text(
                "insert into assessment_benchmark_imports ("
                "import_id, source_workbook, source_filename, source_sha256, "
                "scoring_version, row_count, company_count, questionnaire_count, "
                "question_count, accepted_count, rejected_count, warnings_json, "
                "status, is_active"
                ") values ("
                ":import_id, :source_workbook, :source_filename, :source_sha256, "
                ":scoring_version, :row_count, :company_count, :questionnaire_count, "
                ":question_count, :accepted_count, :rejected_count, :warnings_json, "
                ":status, :is_active"
                ")"
            ),
            {
                "import_id": import_id,
                "source_workbook": dataset.workbook_bytes,
                "source_filename": dataset.summary.source_filename,
                "source_sha256": dataset.summary.source_sha256,
                "scoring_version": dataset.summary.scoring_version,
                "row_count": dataset.summary.row_count,
                "company_count": dataset.summary.company_count,
                "questionnaire_count": dataset.summary.questionnaire_count,
                "question_count": dataset.summary.question_count,
                "accepted_count": dataset.summary.accepted_count,
                "rejected_count": dataset.summary.rejected_count,
                "warnings_json": json.dumps(
                    [
                        warning.model_dump(mode="json")
                        for warning in dataset.summary.warnings
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "status": "loading",
                "is_active": 0,
            },
        )

        prepared_rows = {
            "assessment_benchmark_companies": [
                _company_parameters(company) for company in dataset.companies
            ],
            "assessment_benchmark_submissions": [
                _submission_parameters(submission)
                for submission in dataset.submissions
            ],
            "assessment_benchmark_responses": [
                _response_parameters(response) for response in dataset.responses
            ],
            "assessment_benchmark_scores": [
                _score_parameters(score) for score in dataset.scores
            ],
        }
        insert_statements = {
            "assessment_benchmark_companies": (
                "insert into assessment_benchmark_companies ("
                "import_id, source_company_id, company_name, customer_class, "
                "revenue, employees, nace1, nace2, nace3, company_size, legal_form, "
                "geographic_presence, is_listed, is_public_contracting_client, "
                "uses_self_governance_code"
                ") values ("
                ":import_id, :source_company_id, :company_name, :customer_class, "
                ":revenue, :employees, :nace1, :nace2, :nace3, :company_size, "
                ":legal_form, :geographic_presence, :is_listed, "
                ":is_public_contracting_client, :uses_self_governance_code)"
            ),
            "assessment_benchmark_submissions": (
                "insert into assessment_benchmark_submissions ("
                "import_id, questionnaire_id, source_company_id, submission_date, "
                "extraction_date, release_status, raw_assessment_category"
                ") values ("
                ":import_id, :questionnaire_id, :source_company_id, :submission_date, "
                ":extraction_date, :release_status, :raw_assessment_category)"
            ),
            "assessment_benchmark_responses": (
                "insert into assessment_benchmark_responses ("
                "import_id, questionnaire_id, question_id, canonical_answer_id, "
                "source_answer_id, source_answer_text, source_answer_value, level, "
                "selected, optional, managed, validation_status"
                ") values ("
                ":import_id, :questionnaire_id, :question_id, :canonical_answer_id, "
                ":source_answer_id, :source_answer_text, :source_answer_value, :level, "
                ":selected, :optional, :managed, :validation_status)"
            ),
            "assessment_benchmark_scores": (
                "insert into assessment_benchmark_scores ("
                "score_id, import_id, questionnaire_id, scope_type, dimension, topic, "
                "question_id, calculated_score, supplied_score"
                ") values ("
                ":score_id, :import_id, :questionnaire_id, :scope_type, :dimension, "
                ":topic, :question_id, :calculated_score, :supplied_score)"
            ),
        }
        for table_name, rows in prepared_rows.items():
            self._insert_batches(
                table_name,
                insert_statements[table_name],
                rows,
                batch_size,
            )
            self._validate_insert_count(table_name, import_id, len(rows))

        self.session.execute(
            text(
                "update assessment_benchmark_imports "
                "set is_active = 0, status = 'inactive' "
                "where is_active = 1"
            )
        )
        self.session.execute(
            text(
                "update assessment_benchmark_imports "
                "set is_active = 1, status = 'active', "
                "activated_at = current_utctimestamp "
                "where import_id = :import_id"
            ),
            {"import_id": import_id},
        )
        return BenchmarkWriteResult(
            import_id=import_id,
            status="active",
            no_op=False,
        )

    def purge_inactive_imports(self, active_import_id: str) -> int:
        """Delete inactive benchmark versions while preserving the active import.

        Inputs:
            active_import_id: Import ID that must remain active and untouched.

        Outputs:
            int: Number of inactive import records deleted. Child rows and the
            stored workbook BLOBs are removed in dependency-safe order.

        Raises:
            RuntimeError: If ``active_import_id`` is not the active import.
        """

        active_row = self.session.execute(
            text(
                "select import_id from assessment_benchmark_imports "
                "where import_id = :import_id and is_active = 1 and status = 'active'"
            ),
            {"import_id": active_import_id},
        ).first()
        if active_row is None:
            raise RuntimeError(
                f"Benchmark history cleanup refused: {active_import_id} is not active"
            )

        inactive_rows = self.session.execute(
            text(
                "select import_id from assessment_benchmark_imports "
                "where is_active = 0 and import_id <> :import_id"
            ),
            {"import_id": active_import_id},
        ).mappings().all()
        inactive_ids = [str(row["import_id"]) for row in inactive_rows]
        for import_id in inactive_ids:
            parameters = {"import_id": import_id}
            for table_name in (
                "assessment_benchmark_scores",
                "assessment_benchmark_responses",
                "assessment_benchmark_submissions",
                "assessment_benchmark_companies",
            ):
                self.session.execute(
                    text(f"delete from {table_name} where import_id = :import_id"),
                    parameters,
                )
            self.session.execute(
                text(
                    "delete from assessment_benchmark_imports "
                    "where import_id = :import_id and is_active = 0"
                ),
                parameters,
            )
        return len(inactive_ids)


def write_benchmark_dataset(
    session: Any,
    dataset: BenchmarkDataset,
    *,
    batch_size: int = 250,
    replace_history: bool = False,
) -> BenchmarkWriteResult:
    """Persist and atomically activate one validated benchmark dataset.

    Inputs:
        session: SQLAlchemy-compatible HANA session owned by the caller.
        dataset: Fully validated benchmark dataset.
        batch_size: Positive maximum child rows per executemany call.
        replace_history: Whether to delete every inactive import after the
            validated dataset is active.

    Outputs:
        BenchmarkWriteResult: Active import or identical active-SHA no-op.

    Raises:
        Exception: Any schema, insert, validation, or activation failure is
        re-raised after rolling back the complete DML transaction.
    """

    repository = HanaBenchmarkRepository(session)
    try:
        result = repository.store_dataset(dataset, batch_size=batch_size)
        if replace_history:
            repository.purge_inactive_imports(result.import_id)
        session.commit()
        return result
    except Exception:
        session.rollback()
        raise
