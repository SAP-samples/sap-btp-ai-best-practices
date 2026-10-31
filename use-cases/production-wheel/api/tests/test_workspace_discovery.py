"""Date-range and supported-key contracts for shared dataset/run discovery."""

import pytest
from app.routers.workspace import router
from app.workspace.dependencies import get_service
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _service():
    """Return timestamped dataset/run records without database dependencies."""
    repo = MemoryRepository()
    for key, created in [
        ("early", "2026-09-01T09:00:00+00:00"),
        ("late", "2026-09-07T12:00:00+00:00"),
    ]:
        repo.insert(
            "datasets",
            key,
            {
                "dataset_id": key,
                "name": "Example",
                "status": "published",
                "metadata": {"plant": "P"},
                "created_at": created,
            },
        )
        repo.insert(
            "runs",
            key,
            {
                "run_id": key,
                "dataset_id": "source",
                "parent_run_id": "parent",
                "status": "completed",
                "created_at": created,
            },
        )
    return WorkspaceService(repo)


def test_shared_discovery_date_bounds_are_inclusive_and_timezone_aware():
    """Apply supported metadata predicates together with UTC-equivalent ranges."""
    service = _service()
    criteria = {
        "created_from": "2026-09-07T14:00:00+02:00",
        "created_to": "2026-09-07T12:00:00Z",
    }
    assert [
        row["dataset_id"]
        for row in service.list_datasets(criteria | {"plant": "P", "name": "exam"})
    ] == ["late"]
    assert [
        row["run_id"]
        for row in service.list_runs(
            criteria | {"dataset_id": "source", "parent_run_id": "parent"}
        )
    ] == ["late"]
    assert len(service.list_runs(None)) == 2
    assert service.list_runs({"dataset_id": "missing"}) == []


def test_shared_discovery_rejects_unknown_keys_and_invalid_dates():
    """Reject misspelled filters and incoherent ranges rather than hiding records."""
    service = _service()
    for criteria in (
        {"unsupported": None},
        {"created_from": "yesterday"},
        {"created_from": "2026-09-08T00:00:00Z", "created_to": "2026-09-01T00:00:00Z"},
    ):
        with pytest.raises(ValueError):
            service.list_datasets(criteria)
        with pytest.raises(ValueError):
            service.list_runs(criteria)


def test_http_discovery_exposes_dates_and_rejects_unknown_query_parameters():
    """Use the same discovery contract through documented HTTP query parameters."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = _service
    client = TestClient(app)
    response = client.get(
        "/api/datasets", params={"created_from": "2026-09-07T00:00:00Z"}
    )
    assert [row["dataset_id"] for row in response.json()["items"]] == ["late"]
    assert client.get("/api/runs", params={"created_to": "invalid"}).status_code == 422
    assert client.get("/api/datasets", params={"surprise": "x"}).status_code == 422
