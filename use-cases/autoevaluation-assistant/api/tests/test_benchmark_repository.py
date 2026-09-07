"""HANA-boundary tests for atomic versioned benchmark persistence."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from app.services.benchmark_import.models import (
    BenchmarkCompany,
    BenchmarkDataset,
    BenchmarkResponse,
    BenchmarkScore,
    BenchmarkSubmission,
    BenchmarkValidationSummary,
)
from app.services.benchmark_import.repository import (
    HanaBenchmarkRepository,
    write_benchmark_dataset,
)


EXPECTED_COLUMNS = {
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
"""Complete expected columns returned by the fake HANA catalog."""


class _Result:
    """Provide the SQLAlchemy result methods used by the repository."""

    def __init__(
        self,
        *,
        first: Any = None,
        rows: list[dict[str, Any]] | None = None,
        scalar: int | None = None,
    ) -> None:
        """Store configured first-row, mappings, and scalar results.

        Inputs:
            first: Value returned by ``first()``.
            rows: Mapping rows returned by ``mappings().all()``.
            scalar: Integer returned by ``scalar_one()``.

        Outputs:
            None. The fake result is ready for repository calls.
        """

        self._first = first
        self._rows = rows or []
        self._scalar = scalar

    def first(self) -> Any:
        """Return the configured first result row.

        Inputs:
            None.

        Outputs:
            Any: Configured row or ``None``.
        """

        return self._first

    def mappings(self) -> "_Result":
        """Return this object for SQLAlchemy mapping-chain compatibility.

        Inputs:
            None.

        Outputs:
            _Result: This result instance.
        """

        return self

    def all(self) -> list[dict[str, Any]]:
        """Return configured mapping rows.

        Inputs:
            None.

        Outputs:
            list[dict[str, Any]]: Catalog rows configured by the fake session.
        """

        return self._rows

    def scalar_one(self) -> int:
        """Return the configured required scalar count.

        Inputs:
            None.

        Outputs:
            int: Configured scalar value.
        """

        assert self._scalar is not None
        return self._scalar


class _FakeHanaSession:
    """Model the repository's HANA catalog and transactional row state."""

    def __init__(
        self,
        *,
        tables: set[str] | None = None,
        columns: dict[str, set[str]] | None = None,
        imports: list[dict[str, Any]] | None = None,
        fail_on: str | None = None,
        count_adjustments: dict[str, int] | None = None,
    ) -> None:
        """Initialize fake catalog, persisted rows, and optional failure hooks.

        Inputs:
            tables: Existing uppercase HANA table names.
            columns: Catalog columns keyed by uppercase table name.
            imports: Existing import-version rows.
            fail_on: Optional SQL/event substring that raises ``RuntimeError``.
            count_adjustments: Optional table count offsets for validation tests.

        Outputs:
            None. Complete fake state is ready for SQL execution.
        """

        self.tables = set(EXPECTED_COLUMNS) if tables is None else set(tables)
        self.columns = deepcopy(columns or EXPECTED_COLUMNS)
        self.imports = deepcopy(imports or [])
        self.children: dict[str, list[dict[str, Any]]] = {
            "assessment_benchmark_companies": [],
            "assessment_benchmark_submissions": [],
            "assessment_benchmark_responses": [],
            "assessment_benchmark_scores": [],
        }
        self.fail_on = fail_on
        self.count_adjustments = count_adjustments or {}
        self.events: list[str] = []
        self.batch_lengths: list[tuple[str, int]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self._snapshot: tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]] | None = None

    def _start_transaction(self) -> None:
        """Capture row state before the first DML mutation.

        Inputs:
            None.

        Outputs:
            None. A deep snapshot is retained until commit or rollback.
        """

        if self._snapshot is None:
            self._snapshot = (deepcopy(self.imports), deepcopy(self.children))

    def _maybe_fail(self, event: str) -> None:
        """Raise the configured database failure for a matching event.

        Inputs:
            event: Normalized SQL/event description.

        Outputs:
            None. Non-matching events continue; matching events raise.
        """

        if self.fail_on and self.fail_on in event:
            raise RuntimeError(f"injected database failure: {event}")

    def execute(self, statement: Any, parameters: Any = None) -> _Result:
        """Execute the bounded subset of SQL used by benchmark persistence.

        Inputs:
            statement: SQLAlchemy text clause issued by the repository.
            parameters: Optional scalar mapping or executemany mapping list.

        Outputs:
            _Result: Catalog, row, or count result appropriate for the SQL.
        """

        sql = " ".join(str(statement).lower().split())
        if "from sys.tables" in sql:
            table_name = str(parameters["table_name"]).upper()
            return _Result(first=(table_name,) if table_name in self.tables else None)
        if sql.startswith("create table "):
            table_name = sql.split()[2].split("(", maxsplit=1)[0].upper()
            self.tables.add(table_name)
            self.events.append(f"create:{table_name.lower()}")
            return _Result()
        if "from sys.table_columns" in sql:
            table_name = str(parameters["table_name"]).upper()
            return _Result(
                rows=[
                    {"column_name": column}
                    for column in sorted(self.columns.get(table_name, set()))
                ]
            )
        if (
            "select import_id from assessment_benchmark_imports" in sql
            and "source_sha256 = :source_sha256" in sql
        ):
            matching = next(
                (
                    row
                    for row in self.imports
                    if row["source_sha256"] == parameters["source_sha256"]
                    and row["is_active"] == 1
                ),
                None,
            )
            return _Result(first=(matching["import_id"],) if matching else None)
        if (
            "select import_id from assessment_benchmark_imports" in sql
            and "is_active = 1 and status = 'active'" in sql
        ):
            matching = next(
                (
                    row
                    for row in self.imports
                    if row["import_id"] == parameters["import_id"]
                    and row["is_active"] == 1
                    and row["status"] == "active"
                ),
                None,
            )
            return _Result(first=(matching["import_id"],) if matching else None)
        if (
            "select import_id from assessment_benchmark_imports" in sql
            and "is_active = 0" in sql
        ):
            return _Result(
                rows=[
                    {"import_id": row["import_id"]}
                    for row in self.imports
                    if row["is_active"] == 0
                    and row["import_id"] != parameters["import_id"]
                ]
            )
        if sql.startswith("insert into assessment_benchmark_imports"):
            self._start_transaction()
            self._maybe_fail("insert:assessment_benchmark_imports")
            self.imports.append(dict(parameters))
            self.events.append("insert:assessment_benchmark_imports")
            return _Result()
        for table_name in self.children:
            if sql.startswith(f"delete from {table_name}"):
                self._start_transaction()
                self._maybe_fail(f"delete:{table_name}")
                self.children[table_name] = [
                    row
                    for row in self.children[table_name]
                    if row["import_id"] != parameters["import_id"]
                ]
                self.events.append(f"delete:{table_name}")
                return _Result()
            if sql.startswith(f"insert into {table_name}"):
                self._start_transaction()
                self._maybe_fail(f"insert:{table_name}")
                batch = [dict(row) for row in parameters]
                self.children[table_name].extend(batch)
                self.batch_lengths.append((table_name, len(batch)))
                self.events.append(f"insert:{table_name}")
                return _Result()
            if sql.startswith(f"select count(*) from {table_name}"):
                count = sum(
                    row["import_id"] == parameters["import_id"]
                    for row in self.children[table_name]
                )
                count += self.count_adjustments.get(table_name, 0)
                return _Result(scalar=count)
        if sql.startswith("update assessment_benchmark_imports set is_active = 0"):
            self._start_transaction()
            self._maybe_fail("deactivate")
            for row in self.imports:
                if row["is_active"] == 1:
                    row["is_active"] = 0
                    row["status"] = "inactive"
            self.events.append("deactivate")
            return _Result()
        if sql.startswith("update assessment_benchmark_imports set is_active = 1"):
            self._start_transaction()
            self._maybe_fail("activate")
            for row in self.imports:
                if row["import_id"] == parameters["import_id"]:
                    row["is_active"] = 1
                    row["status"] = "active"
            self.events.append("activate")
            return _Result()
        if sql.startswith("delete from assessment_benchmark_imports"):
            self._start_transaction()
            self._maybe_fail("delete:assessment_benchmark_imports")
            self.imports = [
                row
                for row in self.imports
                if not (
                    row["import_id"] == parameters["import_id"]
                    and row["is_active"] == 0
                )
            ]
            self.events.append("delete:assessment_benchmark_imports")
            return _Result()
        raise AssertionError(f"Unexpected repository SQL: {sql}")

    def commit(self) -> None:
        """Commit fake row state and clear its rollback snapshot.

        Inputs:
            None.

        Outputs:
            None. Commit count is incremented for transaction assertions.
        """

        self.commit_count += 1
        self._snapshot = None

    def rollback(self) -> None:
        """Restore row state captured before the active transaction.

        Inputs:
            None.

        Outputs:
            None. Rollback count is incremented for failure assertions.
        """

        self.rollback_count += 1
        if self._snapshot is not None:
            self.imports, self.children = deepcopy(self._snapshot)
        self._snapshot = None


def _dataset(count: int = 5) -> BenchmarkDataset:
    """Return a complete normalized dataset with configurable batch row counts.

    Inputs:
        count: Number of companies, submissions, responses, and scores.

    Outputs:
        BenchmarkDataset: Validated import payload ready for repository tests.
    """

    import_id = "new-import"
    companies = [
        BenchmarkCompany(
            import_id=import_id,
            source_company_id=f"company-{index}",
            company_name=f"Synthetic {index}",
            customer_class="class_3",
            revenue=Decimal("1000000"),
            employees=50,
            nace1="Synthetic NACE 1",
            nace2="Synthetic NACE 2",
            nace3=f"Synthetic NACE 3-{index}",
            company_size="Medium",
            legal_form="Synthetic LLC",
            geographic_presence="National",
            is_listed=False,
            is_public_contracting_client=False,
            uses_self_governance_code=False,
        )
        for index in range(count)
    ]
    submissions = [
        BenchmarkSubmission(
            import_id=import_id,
            questionnaire_id=f"questionnaire-{index}",
            source_company_id=f"company-{index}",
            submission_date=date(2026, 4, 18),
            extraction_date=date(2026, 7, 6),
            release_status="REL",
            raw_assessment_category="Visione",
        )
        for index in range(count)
    ]
    responses = [
        BenchmarkResponse(
            import_id=import_id,
            questionnaire_id=f"questionnaire-{index}",
            question_id="Q.STR.03.01",
            canonical_answer_id=f"Q.STR.03.01-L1-{index + 1:03d}",
            source_answer_id=f"R-{index}",
            source_answer_text=f"Synthetic answer {index}",
            source_answer_value="Si",
            level=1,
            selected=True,
            optional=False,
            managed=True,
            validation_status="ATT",
        )
        for index in range(count)
    ]
    scores = [
        BenchmarkScore(
            score_id=f"score-{index}",
            import_id=import_id,
            questionnaire_id=f"questionnaire-{index}",
            scope_type="overall",
            dimension=None,
            topic=None,
            question_id=None,
            calculated_score=50.0,
        )
        for index in range(count)
    ]
    return BenchmarkDataset(
        workbook_bytes=b"synthetic workbook bytes",
        companies=companies,
        submissions=submissions,
        responses=responses,
        scores=scores,
        summary=BenchmarkValidationSummary(
            import_id=import_id,
            source_filename="benchmark.xlsx",
            source_sha256="new-sha",
            scoring_version="assessment-v1",
            row_count=count,
            company_count=count,
            questionnaire_count=count,
            question_count=1,
            accepted_count=count,
            rejected_count=0,
            success=True,
            status="validated",
        ),
    )


def _active_import(source_sha256: str = "old-sha") -> dict[str, Any]:
    """Return one existing active import row for version-transition tests.

    Inputs:
        source_sha256: Source hash assigned to the active version.

    Outputs:
        dict[str, Any]: Minimal fake import row used by repository SQL.
    """

    return {
        "import_id": "old-import",
        "source_sha256": source_sha256,
        "status": "active",
        "is_active": 1,
    }


def test_repository_ensures_missing_tables_and_rejects_bad_column_contract() -> None:
    """Verify schema setup is idempotent and validates existing column names.

    Inputs:
        None. Fake HANA catalog states are complete, empty, or incompatible.

    Outputs:
        None. Missing tables are created and bad contracts raise immediately.
    """

    empty_catalog = _FakeHanaSession(tables=set())
    HanaBenchmarkRepository(empty_catalog).ensure_schema()
    assert set(EXPECTED_COLUMNS) <= empty_catalog.tables

    bad_columns = deepcopy(EXPECTED_COLUMNS)
    bad_columns["ASSESSMENT_BENCHMARK_IMPORTS"].remove("SOURCE_WORKBOOK")
    incompatible = _FakeHanaSession(columns=bad_columns)
    with pytest.raises(RuntimeError, match="incompatible contract"):
        HanaBenchmarkRepository(incompatible).ensure_schema()


def test_write_inserts_bounded_batches_and_atomically_retains_old_version() -> None:
    """Verify successful writes validate counts before one version activation.

    Inputs:
        None. Five rows per child table force three batches at size two.

    Outputs:
        None. The old version remains inactive and all inserted rows are retained.
    """

    session = _FakeHanaSession(imports=[_active_import()])
    result = write_benchmark_dataset(session, _dataset(), batch_size=2)

    assert result.import_id == "new-import"
    assert result.status == "active"
    assert result.no_op is False
    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert next(row for row in session.imports if row["import_id"] == "old-import")[
        "status"
    ] == "inactive"
    assert next(row for row in session.imports if row["import_id"] == "new-import")[
        "is_active"
    ] == 1
    assert all(len(rows) == 5 for rows in session.children.values())
    assert max(length for _table, length in session.batch_lengths) <= 2
    assert session.events.index("insert:assessment_benchmark_imports") < session.events.index(
        "deactivate"
    )
    assert session.events.index("deactivate") < session.events.index("activate")


def test_identical_active_sha_is_an_idempotent_no_op() -> None:
    """Verify re-importing the active workbook hash creates no new version rows.

    Inputs:
        None. Existing active SHA matches the incoming dataset summary.

    Outputs:
        None. The existing import ID is returned without child inserts.
    """

    session = _FakeHanaSession(imports=[_active_import("new-sha")])
    result = write_benchmark_dataset(session, _dataset(), batch_size=2)

    assert result.import_id == "old-import"
    assert result.status == "no_op"
    assert result.no_op is True
    assert session.imports == [_active_import("new-sha")]
    assert all(not rows for rows in session.children.values())
    assert session.commit_count == 1


def test_history_replacement_deletes_only_inactive_import_rows() -> None:
    """Verify explicit cleanup preserves the active import and deletion order.

    Inputs:
        None. One old active version has child rows before replacement.

    Outputs:
        None. Old children and BLOB metadata are deleted after activation while
        the new active version remains complete.
    """

    session = _FakeHanaSession(imports=[_active_import()])
    for rows in session.children.values():
        rows.append({"import_id": "old-import"})

    result = write_benchmark_dataset(
        session,
        _dataset(),
        batch_size=2,
        replace_history=True,
    )

    assert result.import_id == "new-import"
    assert [row["import_id"] for row in session.imports] == ["new-import"]
    assert all(
        {row["import_id"] for row in rows} == {"new-import"}
        for rows in session.children.values()
    )
    delete_events = [event for event in session.events if event.startswith("delete:")]
    assert delete_events == [
        "delete:assessment_benchmark_scores",
        "delete:assessment_benchmark_responses",
        "delete:assessment_benchmark_submissions",
        "delete:assessment_benchmark_companies",
        "delete:assessment_benchmark_imports",
    ]


@pytest.mark.parametrize(
    ("fail_on", "count_adjustments"),
    [
        ("insert:assessment_benchmark_responses", None),
        ("activate", None),
        (None, {"assessment_benchmark_responses": -1}),
    ],
)
def test_write_failure_rolls_back_and_leaves_old_version_active(
    fail_on: str | None,
    count_adjustments: dict[str, int] | None,
) -> None:
    """Verify insert, validation, and activation failures restore prior state.

    Inputs:
        fail_on: Optional fake SQL/event failure injection.
        count_adjustments: Optional inserted-row count corruption.

    Outputs:
        None. The transaction rolls back and the old version stays active.
    """

    session = _FakeHanaSession(
        imports=[_active_import()],
        fail_on=fail_on,
        count_adjustments=count_adjustments,
    )

    with pytest.raises(RuntimeError):
        write_benchmark_dataset(session, _dataset(), batch_size=2)

    assert session.commit_count == 0
    assert session.rollback_count == 1
    assert session.imports == [_active_import()]
    assert all(not rows for rows in session.children.values())
