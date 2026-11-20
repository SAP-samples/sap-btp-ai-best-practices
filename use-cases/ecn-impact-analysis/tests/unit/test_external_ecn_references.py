"""Unit tests for the external-reference preflight that guards ECN deletion.

Covers ``find_external_ecn_references`` through both repository delete paths
(``delete_ecns_if_idle`` and ``clear_history_if_idle``) with the
``RecordingEngine`` fake, and the ``HistoryService`` mapping of blocked
deletions and HANA foreign-key violations to HTTP 409.

Run with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_external_ecn_references.py
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import DBAPIError

from app.ecn.errors import PersistenceStateError
from app.ecn.services.history import HistoryService
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.external_references import (
    ExternalEcnReference,
    ExternalEcnReferenceError,
)
from app.persistence.history_repository import HistoryRepository
from tests.fixtures.persistence import RecordingEngine

SELECTED = "00000000-0000-4000-8000-000000000001"
EXTERNAL = "ECN_IMPACT_LEGACY_CHILD"


def _queue_scoped_preflight(engine: RecordingEngine, foreign_keys: list[dict[str, str]]) -> None:
    """Queue the scope read, an idle job count and the given catalog foreign keys."""
    engine.queue_mappings([{"ECN_ROW_ID": SELECTED, "UPLOAD_ID": "U1", "ECN_ARTIFACT_ID": "A1"}])
    engine.queue_scalar(0)
    engine.queue_mappings(foreign_keys)


def _deletes(engine: RecordingEngine) -> list[str]:
    """Return every executed DELETE statement."""
    return [sql for sql, _ in engine.executions if sql.startswith("DELETE")]


def test_scoped_delete_is_refused_when_a_foreign_table_has_rows() -> None:
    """Rows in a table the app does not own stop the delete before any DELETE."""
    engine = RecordingEngine()
    _queue_scoped_preflight(engine, [{"TABLE_NAME": EXTERNAL, "COLUMN_NAME": "ECN_ROW_ID"}])
    engine.queue_scalar(2)  # referencing rows for the selected ECN

    with pytest.raises(ExternalEcnReferenceError) as raised:
        HistoryRepository(engine).delete_ecns_if_idle([SELECTED])

    assert raised.value.references == (ExternalEcnReference(EXTERNAL, "ECN_ROW_ID", 2),)
    assert _deletes(engine) == []
    count_sql, count_params = engine.executions[-1]
    assert f'FROM "{EXTERNAL}" WHERE "ECN_ROW_ID" IN' in count_sql
    assert count_params == {"ids": (SELECTED,)}
    assert engine.rollback_count == 1


def test_scoped_delete_proceeds_when_the_foreign_table_has_no_matching_rows() -> None:
    """A foreign table that references other ECNs only does not block this delete."""
    engine = RecordingEngine()
    _queue_scoped_preflight(engine, [{"TABLE_NAME": EXTERNAL, "COLUMN_NAME": "ECN_ROW_ID"}])
    engine.queue_scalar(0)
    engine.queue_mappings([])  # no generation batches
    counts = HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert counts is not None
    assert any('"ECN_IMPACT_ECN"' in sql for sql in _deletes(engine))
    assert not any(EXTERNAL in sql for sql in _deletes(engine))


def test_owned_tables_in_the_catalog_are_not_counted() -> None:
    """Foreign keys from app-owned tables are handled by the delete order itself."""
    engine = RecordingEngine()
    _queue_scoped_preflight(
        engine,
        [
            {"TABLE_NAME": "ECN_IMPACT_REVIEW", "COLUMN_NAME": "ECN_ROW_ID"},
            {"TABLE_NAME": "ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT", "COLUMN_NAME": "ECN_ROW_ID"},
        ],
    )
    engine.queue_mappings([])  # no generation batches
    HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert not any(sql.startswith("SELECT COUNT(1) FROM \"ECN_IMPACT_REVIEW\"") for sql, _ in engine.executions)


def test_unsafe_catalog_names_block_without_being_queried() -> None:
    """A catalog name outside the safe pattern is reported, never interpolated."""
    engine = RecordingEngine()
    _queue_scoped_preflight(engine, [{"TABLE_NAME": 'odd"name', "COLUMN_NAME": "ECN_ROW_ID"}])
    with pytest.raises(ExternalEcnReferenceError) as raised:
        HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert raised.value.references[0].row_count is None
    assert not any('odd"name' in sql for sql, _ in engine.executions)


def test_lowercase_catalog_keys_are_accepted() -> None:
    """sqlalchemy-hana may return lowercase column keys for catalog reads."""
    engine = RecordingEngine()
    _queue_scoped_preflight(engine, [{"table_name": EXTERNAL, "column_name": "ECN_ROW_ID"}])
    engine.queue_scalar(1)
    with pytest.raises(ExternalEcnReferenceError):
        HistoryRepository(engine).delete_ecns_if_idle([SELECTED])


def test_history_reset_counts_every_referencing_row() -> None:
    """The whole-history reset checks all non-null references, not a selection."""
    engine = RecordingEngine()
    engine.queue_scalar(0)
    engine.queue_mappings([{"TABLE_NAME": EXTERNAL, "COLUMN_NAME": "ECN_ROW_ID"}])
    engine.queue_scalar(3)
    with pytest.raises(ExternalEcnReferenceError):
        HistoryRepository(engine).clear_history_if_idle()
    assert _deletes(engine) == []
    assert engine.executions[-1][0].endswith('"ECN_ROW_ID" IS NOT NULL')


class _RaisingRepository:
    """Repository double whose delete methods raise a configured error."""

    def __init__(self, error: Exception) -> None:
        """Store the error to raise."""
        self.error = error

    def delete_ecns_if_idle(self, _ids):
        """Raise the configured error."""
        raise self.error

    def clear_history_if_idle(self):
        """Raise the configured error."""
        raise self.error


class _HanaError(Exception):
    """Stand-in for ``hdbcli.dbapi.Error`` carrying an ``errorcode``."""

    def __init__(self, errorcode: int) -> None:
        """Store the HANA error code."""
        super().__init__(errorcode, "simulated")
        self.errorcode = errorcode


def _blocked() -> ExternalEcnReferenceError:
    """Return a repository error for one blocking table with two rows."""
    return ExternalEcnReferenceError([ExternalEcnReference(EXTERNAL, "ECN_ROW_ID", 2)])


def test_service_maps_external_references_to_conflict_naming_the_table() -> None:
    """The 409 message tells the user which table blocks the delete."""
    service = HistoryService(_RaisingRepository(_blocked()), HistoryMutationGuard())
    with pytest.raises(PersistenceStateError) as raised:
        service.delete_ecns("DELETE", [SELECTED])
    assert raised.value.code == "ECN_DELETE_EXTERNAL_REFERENCES"
    assert raised.value.status_code == 409
    assert f"{EXTERNAL} (2 rows)" in raised.value.public_message


def test_reset_maps_external_references_to_conflict() -> None:
    """Clear history reports the same blocking tables with its own code."""
    service = HistoryService(_RaisingRepository(_blocked()), HistoryMutationGuard())
    with pytest.raises(PersistenceStateError) as raised:
        service.reset_history("RESET HISTORY")
    assert raised.value.code == "HISTORY_RESET_EXTERNAL_REFERENCES"


def test_service_maps_hana_foreign_key_violation_to_conflict() -> None:
    """A 462 error the preflight could not foresee still becomes a 409."""
    error = DBAPIError("DELETE ...", {}, _HanaError(462))
    service = HistoryService(_RaisingRepository(error), HistoryMutationGuard())
    with pytest.raises(PersistenceStateError) as raised:
        service.delete_ecns("DELETE", [SELECTED])
    assert raised.value.code == "ECN_DELETE_REFERENCED"
    assert raised.value.status_code == 409


def test_service_reraises_other_database_errors() -> None:
    """Only foreign-key violations are mapped; other failures stay 500s."""
    error = DBAPIError("DELETE ...", {}, _HanaError(133))
    service = HistoryService(_RaisingRepository(error), HistoryMutationGuard())
    with pytest.raises(DBAPIError):
        service.delete_ecns("DELETE", [SELECTED])


def test_endpoint_returns_conflict_with_the_blocking_table(monkeypatch) -> None:
    """The dialog receives a 409 whose message names the blocking table."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("API_KEY", "test-api-key")
    services = SimpleNamespace(
        history_service=HistoryService(_RaisingRepository(_blocked()), HistoryMutationGuard()),
        ecn_repository=SimpleNamespace(list_matching_ecn_ids=lambda _query: []),
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield test services."""
        yield services

    with TestClient(create_app(service_lifespan_factory=lifespan)) as client:
        response = client.post(
            "/api/ecns/delete",
            headers={"X-API-Key": "test-api-key"},
            json={"confirmation": "DELETE", "selection": {"mode": "explicit", "ecn_ids": [SELECTED]}},
        )
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "ECN_DELETE_EXTERNAL_REFERENCES"
    assert EXTERNAL in body["message"]
