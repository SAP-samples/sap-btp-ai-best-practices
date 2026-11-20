"""Unit tests for scoped ECN line-item deletion (repository, service, router).

Covers ``HistoryRepository.delete_ecns_if_idle`` with the ``RecordingEngine``
fake (no live HANA), ``HistoryService.delete_ecns`` confirmation/guard/graph
behaviour, and the ``POST /api/ecns/delete`` request contract.

Run with:

    PYTHONPATH=api python -m pytest -q tests/unit/test_ecn_deletion.py
"""

from __future__ import annotations

import re

import pytest

from app.ecn.errors import PersistenceStateError, SourceValidationError
from app.ecn.services.history import HistoryService
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.history_repository import HistoryRepository
from app.persistence.records import EcnDeletionCounts
from tests.fixtures.persistence import RecordingEngine

SELECTED = "00000000-0000-4000-8000-000000000001"
VERSION = "00000000-0000-4000-8000-000000000002"


def _scope_rows() -> list[dict[str, str]]:
    """Return the selected row plus one reassessment sharing its artifact."""
    return [
        {"ECN_ROW_ID": SELECTED, "UPLOAD_ID": "U1", "ECN_ARTIFACT_ID": "A1"},
        {"ECN_ROW_ID": VERSION, "UPLOAD_ID": "U2", "ECN_ARTIFACT_ID": "A1"},
    ]


def _targets(engine: RecordingEngine) -> list[str]:
    """Return DELETE target tables in execution order."""
    return [
        match.group(1)
        for sql, _ in engine.executions
        if (match := re.search(r'DELETE FROM "([^"]+)"', sql))
    ]


def test_active_job_on_any_version_blocks_without_deleting() -> None:
    """A running job on a reassessment version aborts the whole delete."""
    engine = RecordingEngine()
    engine.queue_mappings(_scope_rows())
    engine.queue_scalar(1)
    assert HistoryRepository(engine).delete_ecns_if_idle([SELECTED]) is None
    assert _targets(engine) == []
    active_params = engine.executions[1][1]
    assert set(active_params["ids"]) == {SELECTED, VERSION}


def test_deletes_all_versions_in_fk_order_and_never_reference_data() -> None:
    """Every version is deleted child-first; parents only when orphaned."""
    engine = RecordingEngine()
    engine.queue_mappings(_scope_rows())
    engine.queue_scalar(0)
    engine.queue_mappings([])  # external-reference preflight: no foreign tables
    engine.queue_mappings([{"BATCH_ID": "B1"}])
    engine.queue_rowcounts([1, 1, 2, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 0])
    counts = HistoryRepository(engine).delete_ecns_if_idle([SELECTED])

    assert _targets(engine) == [
        "ECN_IMPACT_REVIEW",
        "ECN_IMPACT_BATCH_JOB",
        "ECN_IMPACT_ANALYSIS",
        "ECN_IMPACT_PROVIDER_CALL",
        "ECN_IMPACT_ANALYSIS_STAGE",
        "ECN_IMPACT_LLM_ATTEMPT",
        "ECN_IMPACT_ANALYSIS_JOB",
        "ECN_IMPACT_RDF_PROJECTION",
        "ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT",
        "ECN_IMPACT_ECN",
        "ECN_IMPACT_UPLOAD_DATASET",
        "ECN_IMPACT_UPLOAD",
        "ECN_IMPACT_SOURCE_ARTIFACT",
        "ECN_IMPACT_GENERATION_BATCH",
    ]
    assert counts == EcnDeletionCounts(
        ecn_row_ids=(SELECTED, VERSION),
        ecns=2, analyses=2, attempts=3, jobs=2, reviews=1, uploads=1,
        source_artifacts=1, feasibility_assessments=1,
    )
    assert engine.begin_count == 1
    all_sql = " ".join(sql for sql, _ in engine.executions)
    for untouched in ("REFERENCE_DATASET", "BOM_ROW", "MATERIAL_MASTER", "STOCK_ROW"):
        assert f"DELETE FROM \"ECN_IMPACT_{untouched}" not in all_sql


def test_parents_are_deleted_only_when_no_remaining_ecn_references_them() -> None:
    """Uploads, artifacts and batches shared with surviving ECNs are kept by SQL guards."""
    engine = RecordingEngine()
    engine.queue_mappings(_scope_rows())
    engine.queue_scalar(0)
    engine.queue_mappings([])  # external-reference preflight: no foreign tables
    engine.queue_mappings([{"BATCH_ID": "B1"}])
    HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    by_target = {
        re.search(r'DELETE FROM "([^"]+)"', sql).group(1): (sql, params)
        for sql, params in engine.executions
        if sql.startswith("DELETE")
    }
    upload_sql, upload_params = by_target["ECN_IMPACT_UPLOAD"]
    assert 'NOT IN (SELECT "UPLOAD_ID" FROM "ECN_IMPACT_ECN")' in upload_sql
    assert set(upload_params["uploads"]) == {"U1", "U2"}
    artifact_sql, artifact_params = by_target["ECN_IMPACT_SOURCE_ARTIFACT"]
    assert 'NOT IN (SELECT "ECN_ARTIFACT_ID" FROM "ECN_IMPACT_UPLOAD")' in artifact_sql
    assert artifact_params == {"artifacts": ("A1",), "ecn_kind": "ECN_JSON"}
    assert 'NOT IN (SELECT "BATCH_ID" FROM "ECN_IMPACT_BATCH_JOB")' in by_target[
        "ECN_IMPACT_GENERATION_BATCH"
    ][0]


def test_without_generation_batches_the_batch_cleanup_is_skipped() -> None:
    """No empty IN list is ever sent to HANA."""
    engine = RecordingEngine()
    engine.queue_mappings(_scope_rows())
    engine.queue_scalar(0)
    engine.queue_mappings([])  # external-reference preflight: no foreign tables
    engine.queue_mappings([])
    HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert "ECN_IMPACT_GENERATION_BATCH" not in _targets(engine)


def test_unknown_ids_delete_nothing() -> None:
    """Repeating a delete for already-removed ECNs is a harmless no-op."""
    engine = RecordingEngine()
    engine.queue_mappings([])
    counts = HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert counts is not None and counts.ecns == 0 and counts.ecn_row_ids == ()
    assert _targets(engine) == []


class _Repository:
    """Return a configured scoped-delete result."""

    def __init__(self, result: EcnDeletionCounts | None) -> None:
        """Store the result and record received IDs."""
        self.result = result
        self.received: list[str] | None = None

    def delete_ecns_if_idle(self, ids):
        """Record IDs and return the configured result."""
        self.received = list(ids)
        return self.result


def _counts() -> EcnDeletionCounts:
    """Return counts for two deleted versions."""
    return EcnDeletionCounts((SELECTED, VERSION), 2, 1, 1, 1, 0, 1, 1)


@pytest.mark.parametrize("confirmation", ["", "delete", "DELETE ", "RESET HISTORY"])
def test_service_requires_exact_confirmation(confirmation: str) -> None:
    """Wrong phrases never reach the repository."""
    repository = _Repository(_counts())
    service = HistoryService(repository, HistoryMutationGuard())
    with pytest.raises(SourceValidationError) as raised:
        service.delete_ecns(confirmation, [SELECTED])
    assert raised.value.code == "ECN_DELETE_CONFIRMATION_INVALID"
    assert repository.received is None


def test_service_maps_active_jobs_to_conflict() -> None:
    """A blocked repository result becomes HTTP 409."""
    service = HistoryService(_Repository(None), HistoryMutationGuard())
    with pytest.raises(PersistenceStateError) as raised:
        service.delete_ecns("DELETE", [SELECTED])
    assert raised.value.code == "ECN_DELETE_ACTIVE_JOBS"
    assert raised.value.status_code == 409


def test_service_cleans_graph_for_every_deleted_version() -> None:
    """Graph cleanup receives expanded IDs, not only the selected one."""
    cleaned: list[tuple[str, ...]] = []
    service = HistoryService(
        _Repository(_counts()),
        HistoryMutationGuard(),
        delete_projection_graphs=lambda ids: cleaned.append(tuple(ids)),
    )
    counts, graph_ok = service.delete_ecns("DELETE", [SELECTED])
    assert graph_ok and counts.ecns == 2
    assert cleaned == [(SELECTED, VERSION)]


def test_service_reports_graph_failure_without_undoing_delete() -> None:
    """Graph failure is reported, relational deletion stays committed."""

    def fail(_ids) -> None:
        """Simulate an unavailable SPARQL endpoint."""
        raise RuntimeError("graph down")

    service = HistoryService(
        _Repository(_counts()), HistoryMutationGuard(), delete_projection_graphs=fail
    )
    counts, graph_ok = service.delete_ecns("DELETE", [SELECTED])
    assert counts.ecns == 2 and graph_ok is False


def _api_client(monkeypatch, service, matching=()):
    """Build a test app whose ECN repository resolves filtered selections."""
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from app.main import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("API_KEY", "test-api-key")
    services = SimpleNamespace(
        history_service=service,
        ecn_repository=SimpleNamespace(list_matching_ecn_ids=lambda _query: list(matching)),
    )

    @asynccontextmanager
    async def lifespan(_app):
        """Yield test services."""
        yield services

    return TestClient(create_app(service_lifespan_factory=lifespan))


def test_endpoint_deletes_explicit_selection_and_reports_counts(monkeypatch) -> None:
    """Explicit IDs reach the service; counts and graph status are returned."""
    repository = _Repository(_counts())
    client = _api_client(monkeypatch, HistoryService(repository, HistoryMutationGuard()))
    with client:
        response = client.post(
            "/api/ecns/delete",
            headers={"X-API-Key": "test-api-key"},
            json={"confirmation": "DELETE", "selection": {"mode": "explicit", "ecn_ids": [SELECTED]}},
        )
    assert response.status_code == 200
    assert repository.received == [SELECTED]
    assert response.json() == {
        "ecns": 2, "analyses": 1, "attempts": 1, "jobs": 1, "reviews": 0,
        "uploads": 1, "source_artifacts": 1, "feasibility_assessments": 0,
        "graph_cleanup": "complete",
    }


def test_endpoint_resolves_filtered_selection_minus_exclusions(monkeypatch) -> None:
    """'Select all matching' deletes the filter membership except excluded rows."""
    repository = _Repository(_counts())
    client = _api_client(
        monkeypatch, HistoryService(repository, HistoryMutationGuard()), matching=(SELECTED, VERSION)
    )
    with client:
        response = client.post(
            "/api/ecns/delete",
            headers={"X-API-Key": "test-api-key"},
            json={
                "confirmation": "DELETE",
                "selection": {"mode": "filtered", "excluded_ecn_ids": [VERSION]},
            },
        )
    assert response.status_code == 200
    assert repository.received == [SELECTED]


def test_endpoint_returns_conflict_and_bad_phrase_errors(monkeypatch) -> None:
    """Active jobs yield 409; a wrong phrase yields 400; nothing is deleted."""
    repository = _Repository(None)
    client = _api_client(monkeypatch, HistoryService(repository, HistoryMutationGuard()))
    body = {"selection": {"mode": "explicit", "ecn_ids": [SELECTED]}}
    with client:
        wrong = client.post("/api/ecns/delete", headers={"X-API-Key": "test-api-key"},
                            json={**body, "confirmation": "delete"})
        active = client.post("/api/ecns/delete", headers={"X-API-Key": "test-api-key"},
                             json={**body, "confirmation": "DELETE"})
    assert wrong.status_code == 400
    assert active.status_code == 409
    assert "ECN_DELETE_ACTIVE_JOBS" in active.text


def test_lowercase_hana_row_keys_are_accepted() -> None:
    """sqlalchemy-hana returns lowercase keys; scope and batch reads must normalize them."""
    engine = RecordingEngine()
    engine.queue_mappings([{k.lower(): v for k, v in row.items()} for row in _scope_rows()])
    engine.queue_scalar(0)
    engine.queue_mappings([])  # external-reference preflight: no foreign tables
    engine.queue_mappings([{"batch_id": "B1"}])
    counts = HistoryRepository(engine).delete_ecns_if_idle([SELECTED])
    assert counts is not None and counts.ecn_row_ids == (SELECTED, VERSION)
    assert "ECN_IMPACT_GENERATION_BATCH" in _targets(engine)
