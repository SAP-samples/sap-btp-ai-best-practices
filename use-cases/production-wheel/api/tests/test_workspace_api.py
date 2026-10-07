"""HTTP integration contracts use shared services without database/model costs."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.workspace import router
from app.workspace.dependencies import get_service
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def test_failed_upload_returns_readable_error_with_cors(monkeypatch):
    """Persistence failures retain CORS and do not expose database internals."""
    import io
    from zipfile import ZipFile
    from fastapi.middleware.cors import CORSMiddleware
    from production_wheel.extraction import datasets

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4173"])
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())

    def fail_save(*args):
        """Simulate a schema rejection after extraction completes."""
        raise RuntimeError("private database schema details")

    monkeypatch.setattr(service, "register_dataset", fail_save)
    monkeypatch.setattr(datasets, "extract_workbooks", lambda *args: {})
    app.dependency_overrides[get_service] = lambda: service
    source = io.BytesIO()
    with ZipFile(source, "w") as archive:
        archive.writestr("placeholder", "test")
    response = TestClient(app).post(
        "/api/datasets", files={"primary": ("source.xlsx", source.getvalue())},
        headers={"Origin": "http://localhost:4173"},
    )
    assert response.status_code == 500
    assert response.headers["access-control-allow-origin"] == "http://localhost:4173"
    assert "No new snapshot was registered" in response.json()["detail"]
    assert "private database" not in response.text
    assert service.list_datasets() == []


def test_api_query_draft_and_unknown_identifiers():
    """HTTP routes return JSON contracts and a404 instead of an internal exception."""
    app = FastAPI()
    app.include_router(router)
    s = WorkspaceService(MemoryRepository())
    app.dependency_overrides[get_service] = lambda: s
    client = TestClient(app)
    assert client.get("/api/datasets").json() == {"items": []}
    assert client.get("/api/datasets/missing").status_code == 404
    assert (
        client.post(
            "/api/query", json={"view": "SYS.USERS", "dataset_id": "x"}
        ).status_code
        == 422
    )


def test_configuration_help_and_permanent_deletion_routes():
    """The UI fetches help and deletes history without a restore route."""
    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.insert(
        "runs", "r", {"run_id": "r", "revision": 1, "status": "completed"}
    )
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as client:
        assert (
            "BASE_GROUP"
            in client.get("/api/configuration-help").json()["basis"]["text"]
        )
        assert client.delete("/api/runs/r").status_code == 200
        assert client.get("/api/runs").json()["items"] == []
        assert client.get("/api/runs?include_removed=true").json()["items"] == []
        assert client.post("/api/runs/r/restore").status_code == 404
        assert client.get("/api/runs/r").status_code == 404


def test_run_list_and_status_routes_exclude_frozen_replay_payloads():
    """Browser run reads use the same compact contract intended for chat tools."""
    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.insert(
        "runs",
        "r",
        {
            "run_id": "r",
            "revision": 1,
            "dataset_id": "dataset",
            "status": "failed",
            "stage": "publication_failed",
            "results_ready": False,
            "plant_profile": {
                "profile_id": "profile",
                "settings": {"horizon_days": 80},
                "matrix_rows": [{"volume_a": "A", "volume_b": "B", "status": "allowed"}],
            },
            "request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
            "source_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
            "validation": {"effective_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}}},
            "metadata": {"opaque": "do not return"},
        },
    )
    app.dependency_overrides[get_service] = lambda: service

    with TestClient(app) as client:
        listed = client.get("/api/runs").json()["items"]
        selected = client.get("/api/runs/r").json()

    assert listed == [{"run_id": "r", "revision": 1, "dataset_id": "dataset", "draft_id": None,
                       "title": None, "parent_run_id": None, "plant_profile_id": None,
                       "plant_profile_revision": None, "status": "failed", "stage": "publication_failed",
                       "error": None, "created_at": None, "started_at": None, "solver_finished_at": None,
                       "finished_at": None, "heartbeat_at": None, "progress": None, "results_ready": False,
                       "cancel_requested": None, "worker_id": None}]
    assert selected["configuration"]["matrix_pair_count"] == 1
    assert set(selected).isdisjoint({"plant_profile", "request", "source_request", "validation", "metadata"})


def test_failure_diagnostics_route_reads_only_bounded_checkpoint_evidence():
    """Failure diagnosis is available without returning a checkpoint or request blob."""
    import json

    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.insert(
        "runs",
        "r",
        {"run_id": "r", "revision": 1, "status": "failed", "stage": "publication_failed"},
    )
    service.repo.put_artifact(
        "r",
        "publication-checkpoint.json",
        json.dumps(
            {
                "metadata": {"complete": False, "integrity_valid": False},
                "tables": {
                    "block_failures": [
                        {
                            "plant": "P1",
                            "sefi": "S1",
                            "candidate_count": 3,
                            "error_type": "uncovered_precheck",
                            "error_message": "No eligible candidate",
                        }
                    ]
                },
            }
        ).encode(),
    )
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).get("/api/runs/r/failure-diagnostics")

    assert response.status_code == 200
    assert response.json()["failure_types"] == ["uncovered_precheck"]
    assert "request" not in response.text
    assert "publication-checkpoint" not in response.text


def test_matrix_route_pages_a_failed_run_frozen_profile_matrix():
    """Matrix review is explicit, filtered and available before result publication."""
    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.insert(
        "runs",
        "r",
        {
            "run_id": "r",
            "revision": 1,
            "status": "failed",
            "plant_profile": {
                "matrix_rows": [
                    {"volume_a": "A", "volume_b": "B", "status": "allowed"},
                    {"volume_a": "A", "volume_b": "C", "status": "blocked"},
                ]
            },
        },
    )
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).get("/api/runs/r/matrix?status=allowed&limit=1")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "r",
        "total": 1,
        "offset": 0,
        "limit": 1,
        "truncated": False,
        "rows": [{"volume_a": "A", "volume_b": "B", "status": "allowed"}],
    }


def test_results_route_uses_compact_run_and_metadata_views():
    """Completed frontier reads do not bypass the compact run-data boundary."""
    app = FastAPI()
    app.include_router(router)
    service = WorkspaceService(MemoryRepository())
    service.repo.insert(
        "runs",
        "r",
        {
            "run_id": "r",
            "revision": 1,
            "status": "completed",
            "results_ready": True,
            "plant_profile": {"matrix_rows": [{"volume_a": "A"}]},
            "request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
            "source_request": {"config": {"matrix_pairs": [{"volume_a": "A"}]}},
            "metadata": {
                "complete": True,
                "integrity_valid": True,
                "config": {"matrix_pairs": [{"volume_a": "A"}]},
                "source_provenance": {"private": "hidden"},
            },
        },
    )
    service.repo.replace_tables("r", {"solutions": [{"point_index": 1, "j_ch": 3.0}]})
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).get("/api/runs/r/results")

    assert response.status_code == 200
    payload = response.json()
    assert payload["points"] == [{"point_index": 1, "j_ch": 3.0}]
    assert payload["metadata"] == {"integrity_valid": True, "complete": True}
    assert set(payload["run"]).isdisjoint(
        {"plant_profile", "request", "source_request", "metadata"}
    )
