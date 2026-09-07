"""Tests for HANA connection-loss handling at process boundaries."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import DBAPIError

from app.db import is_transient_database_error
from app.main import app
import app.routers.ai_review as ai_review_router


def _database_error(
    code: int,
    message: str,
    *,
    connection_invalidated: bool = False,
) -> DBAPIError:
    """Build a deterministic SQLAlchemy-wrapped HANA-style error.

    Inputs:
        code: Numeric driver error code placed in the original exception.
        message: Driver error text.
        connection_invalidated: Whether SQLAlchemy already classified the
            connection as invalid.

    Outputs:
        DBAPIError: Error object matching the production exception boundary.
    """

    return DBAPIError(
        "select 1 from dummy",
        {},
        Exception(code, message),
        connection_invalidated=connection_invalidated,
    )


class FailingSession:
    """Raise one configured DBAPI error and record session cleanup."""

    def __init__(self, error: DBAPIError) -> None:
        """Store the database error raised by repository SQL execution.

        Inputs:
            error: SQLAlchemy error to raise from ``execute``.

        Outputs:
            None. Cleanup counters are initialized for assertions.
        """

        self.error = error
        self.rollback_calls = 0
        self.close_calls = 0

    def execute(self, *_args: object, **_kwargs: object) -> None:
        """Raise the configured database failure for every SQL statement.

        Inputs:
            *_args: Ignored SQL statement arguments.
            **_kwargs: Ignored SQL execution keyword arguments.

        Outputs:
            None. The method always raises ``self.error``.
        """

        raise self.error

    def commit(self) -> None:
        """Accept commits if a request unexpectedly reaches that boundary."""

        return None

    def rollback(self) -> None:
        """Record transaction rollback after the failed request."""

        self.rollback_calls += 1

    def close(self) -> None:
        """Record session closure after dependency cleanup."""

        self.close_calls += 1


@pytest.mark.parametrize("code", [-10709, -10807])
def test_hana_connection_error_codes_are_transient(code: int) -> None:
    """Verify observed HANA connect and route-loss codes are retryable.

    Inputs:
        code: Observed HANA driver error code supplied by pytest.

    Outputs:
        None. The assertion confirms both outage signatures are classified as
        transient even when the dialect did not mark the connection invalid.
    """

    assert is_transient_database_error(_database_error(code, "connection lost"))


def test_invalidated_sqlalchemy_connection_is_transient() -> None:
    """Verify SQLAlchemy's dialect-level disconnect signal is honored."""

    assert is_transient_database_error(
        _database_error(
            99999,
            "dialect-specific disconnect",
            connection_invalidated=True,
        )
    )


def test_sql_programming_error_is_not_transient() -> None:
    """Verify SQL and schema errors remain visible instead of retrying forever."""

    assert not is_transient_database_error(
        _database_error(257, "sql syntax error near unexpected token")
    )


def test_api_returns_service_unavailable_for_transient_hana_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify request-scoped HANA outages become concise retryable responses.

    Inputs:
        monkeypatch: Pytest fixture replacing the cached HANA session factory.

    Outputs:
        None. Assertions confirm HTTP 503, a retry hint, and complete transaction
        cleanup instead of an uncaught ASGI exception.
    """

    session = FailingSession(
        _database_error(-10709, "Cannot resolve HANA host name")
    )

    def create_failing_session() -> FailingSession:
        """Return the shared failing session for this request."""

        return session

    def get_failing_session_factory() -> Callable[[], FailingSession]:
        """Return a session factory compatible with the API dependency."""

        return create_failing_session

    monkeypatch.setattr(
        ai_review_router,
        "get_hana_session_factory",
        get_failing_session_factory,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/api/assessment/dimensions",
            headers={"X-API-Key": "test-api-key"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The HANA database is temporarily unavailable. Retry shortly."
    }
    assert response.headers["Retry-After"] == "5"
    assert session.rollback_calls == 1
    assert session.close_calls == 1


def test_benchmark_import_preserves_transient_hana_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify benchmark validation does not disguise HANA outages as bad files.

    Inputs:
        monkeypatch: Pytest fixture replacing the cached HANA session factory.

    Outputs:
        None. Assertions confirm the shared repository boundary still returns
        HTTP 503 with its retry hint before workbook parsing starts.
    """

    session = FailingSession(_database_error(-10709, "connection lost"))

    def create_failing_session() -> FailingSession:
        """Return the shared failing session for this request."""

        return session

    def get_failing_session_factory() -> Callable[[], FailingSession]:
        """Return a session factory compatible with the API dependency."""

        return create_failing_session

    monkeypatch.setattr(
        ai_review_router,
        "get_hana_session_factory",
        get_failing_session_factory,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/import/assessment-benchmarks",
            headers={"X-API-Key": "test-api-key"},
            files={
                "workbook": (
                    "benchmark.xlsx",
                    b"not-reached",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The HANA database is temporarily unavailable. Retry shortly."
    }
    assert response.headers["Retry-After"] == "5"
    assert session.rollback_calls == 1
    assert session.close_calls == 1


def test_api_preserves_non_transient_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SQL/programming errors are not mislabeled as service outages.

    Inputs:
        monkeypatch: Pytest fixture replacing the cached HANA session factory.

    Outputs:
        None. Assertions confirm the original non-transient error propagates
        after the request transaction is rolled back and closed.
    """

    error = _database_error(257, "sql syntax error near unexpected token")
    session = FailingSession(error)

    def create_failing_session() -> FailingSession:
        """Return the shared failing session for this request."""

        return session

    def get_failing_session_factory() -> Callable[[], FailingSession]:
        """Return a session factory compatible with the API dependency."""

        return create_failing_session

    monkeypatch.setattr(
        ai_review_router,
        "get_hana_session_factory",
        get_failing_session_factory,
    )

    with TestClient(app) as client, pytest.raises(DBAPIError) as captured:
        client.get(
            "/api/assessment/dimensions",
            headers={"X-API-Key": "test-api-key"},
        )

    assert captured.value is error
    assert session.rollback_calls == 1
    assert session.close_calls == 1
