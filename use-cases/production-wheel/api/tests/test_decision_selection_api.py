"""HTTP and persistence contract for decision scenarios and their selection.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_decision_selection_api.py -q
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.decisions import router
from app.workspace.dependencies import get_service
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def _client(results_ready=True):
    """Build an API client over one published 9-point run on y = 1/x."""

    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert(
        "runs",
        "r",
        {
            "run_id": "r",
            "revision": 1,
            "dataset_id": "d",
            "status": "succeeded",
            "results_ready": results_ready,
            "metadata": {"frontier_points_retained": 9, "frontier_shortfall": False},
        },
    )
    repo.replace_tables(
        "r",
        {
            "solutions": [
                {
                    "point_index": x,
                    "j_ch": x,
                    "demand_weighted_mean_coverage_days": 1 / x,
                    "group_count": 10 + x,
                    "validation_status": "valid",
                }
                for x in range(1, 10)
            ],
            "groups": [
                {"point_index": 3, "frequency_per_week": 1.5},
                {"point_index": 3, "frequency_per_week": 2.0},
                {"point_index": 4, "frequency_per_week": 4.0},
            ],
            "members": [
                {"point_index": 3, "sefi": "S1", "proposed_subgroup_code": "G001"},
                {"point_index": 3, "sefi": "S2", "proposed_subgroup_code": ""},
            ],
        },
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app), service


def test_scenarios_carry_labels_and_three_metrics() -> None:
    """Every usable point is returned; key points carry a label and SEFI batches."""

    client, _ = _client()
    body = client.get("/api/runs/r/decision-scenarios").json()
    by_index = {point["point_index"]: point for point in body["points"]}
    assert body["knee_point_index"] == 3
    assert len(body["points"]) == 9
    assert by_index[3]["decision_label"] == "B"
    assert by_index[3]["sefi_batches_per_week"] == 3.5
    assert by_index[3]["group_count"] == 13
    assert by_index[9]["decision_label"] is None
    assert body["frontier_shortfall"] is False
    assert body["selection"] is None


def test_selection_is_stored_on_the_run() -> None:
    """PUT persists the choice; GET returns it and the run revision advances."""

    client, service = _client()
    revision = service.get_run("r")["revision"]
    response = client.put(
        "/api/runs/r/decision-selection",
        json={"point_index": 3},
    )
    assert response.status_code == 200
    stored = service.get_run("r")
    assert stored["revision"] == revision + 1
    assert stored["decision_selection"]["decision_label"] == "B"
    assert "runner_adjusted_sefis" not in stored["decision_selection"]
    assert client.get("/api/runs/r/decision-scenarios").json()["selection"]["point_index"] == 3


def test_points_beyond_the_key_set_can_be_selected() -> None:
    """Show-all users can still pick a fourth neighbour for extreme cases."""

    client, _ = _client()
    response = client.put("/api/runs/r/decision-selection", json={"point_index": 9})
    assert response.status_code == 200
    assert response.json()["decision_label"] is None


def test_invalid_selections_are_rejected() -> None:
    """Unknown points, the removed per-SEFI runner choice and unpublished runs return 422/409."""

    client, _ = _client()
    assert client.put("/api/runs/r/decision-selection", json={"point_index": 42}).status_code == 422
    response = client.put(
        "/api/runs/r/decision-selection",
        json={"point_index": 3, "runner_adjusted_sefis": ["S2"]},
    )
    # The cycle stock strategy now comes from the optimizer, not a SEFI tick box.
    assert response.status_code == 422
    unpublished, _ = _client(results_ready=False)
    assert unpublished.get("/api/runs/r/decision-scenarios").status_code == 409
    assert unpublished.put("/api/runs/r/decision-selection", json={"point_index": 3}).status_code == 409
