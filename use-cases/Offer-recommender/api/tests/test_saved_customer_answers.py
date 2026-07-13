from __future__ import annotations

from app.services.saved_answers import (
    HanaSavedCustomerAnswerRepository,
    InMemorySavedCustomerAnswerRepository,
    SavedCustomerAnswerService,
    create_saved_answers_table_sql,
)


def test_saved_answers_table_sql_defines_account_fact_primary_key() -> None:
    """The saved-answer HANA table should be keyed by account and fact id."""
    sql = create_saved_answers_table_sql()

    assert "COA_SAVED_CUSTOMER_ANSWERS" in sql
    assert '"BILLING_ACCOUNT" NVARCHAR(100)' in sql
    assert '"FACT_ID" NVARCHAR(200)' in sql
    assert 'PRIMARY KEY ("BILLING_ACCOUNT", "FACT_ID")' in sql


def test_saved_answer_service_round_trips_json_values_and_explicit_null() -> None:
    """Saved answer values should survive JSON storage, including explicit null."""
    service = SavedCustomerAnswerService(InMemorySavedCustomerAnswerRepository())

    service.save_answer(
        "00104",
        "customer_wants_followup",
        None,
        answer_label="Not sure / Prefer not to answer",
        question_prompt="Would the customer like to answer more questions?",
        source_surface="chat",
    )
    service.save_answer("104", "payments_on_time", True, source_surface="lookup")

    answers = service.get_answer_values("104")

    assert answers == {
        "customer_wants_followup": None,
        "payments_on_time": True,
    }


def test_saved_answer_service_overwrites_and_resets_answers() -> None:
    """Latest writes should win, and account/global reset should delete overlays only."""
    repository = InMemorySavedCustomerAnswerRepository()
    service = SavedCustomerAnswerService(repository)

    service.save_answer("104", "payments_on_time", False, source_surface="chat")
    service.save_answer("104", "payments_on_time", True, source_surface="lookup")
    service.save_answer("6001", "customer_wants_followup", True, source_surface="chat")

    assert service.get_answer_values("104") == {"payments_on_time": True}
    assert service.reset_account("104") == 1
    assert service.get_answer_values("104") == {}
    assert service.get_answer_values("6001") == {"customer_wants_followup": True}
    assert service.reset_all() == 1
    assert service.get_all_answer_values() == {}


def test_saved_answer_service_skips_unknown_or_unanswerable_facts() -> None:
    """Only known customer-answerable facts should be persisted."""
    service = SavedCustomerAnswerService(InMemorySavedCustomerAnswerRepository())

    service.save_answers(
        "104",
        {
            "payments_on_time": True,
            "current_rate_plan": "E21",
            "not_a_fact": True,
        },
        source_surface="lookup",
    )

    assert service.get_answer_values("104") == {"payments_on_time": True}


class FakeCursor:
    """Small cursor double that records SQL issued by the HANA repository."""

    def __init__(self) -> None:
        """Create an empty recording cursor."""
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []
        self.rows: list[tuple[object, ...]] = []
        self.description = [
            ("BILLING_ACCOUNT",),
            ("FACT_ID",),
            ("ANSWER_JSON",),
            ("ANSWER_LABEL",),
            ("QUESTION_PROMPT",),
            ("SOURCE_SURFACE",),
            ("UPDATED_AT",),
        ]

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        """Record one SQL execution."""
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[object, ...]]:
        """Return the configured fake result rows."""
        return self.rows

    def close(self) -> None:
        """Mirror the hdbcli cursor close API."""
        return None


class FakeConnection:
    """Small connection double that exposes one reusable recording cursor."""

    def __init__(self) -> None:
        """Create a fake connection with a single cursor instance."""
        self.cursor_instance = FakeCursor()
        self.committed = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        """Return the fake recording cursor."""
        return self.cursor_instance

    def commit(self) -> None:
        """Record that a transaction was committed."""
        self.committed = True

    def close(self) -> None:
        """Record that the connection was closed."""
        self.closed = True


def test_hana_repository_uses_upsert_with_json_payload() -> None:
    """The HANA repository should write answer values as JSON through UPSERT."""
    connection = FakeConnection()
    repository = HanaSavedCustomerAnswerRepository(lambda: connection)

    repository.upsert_answer(
        "104",
        "payments_on_time",
        True,
        answer_label="Yes",
        question_prompt="Are payments on time?",
        source_surface="chat",
    )

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    params = connection.cursor_instance.executed[-1][1]

    assert any("CREATE COLUMN TABLE" in sql for sql in executed_sql)
    assert "UPSERT" in executed_sql[-1]
    assert params is not None
    assert params[0] == "104"
    assert params[1] == "payments_on_time"
    assert params[2] == "true"
    assert params[3] == "Yes"
    assert connection.committed is True
