"""Behavioral contracts for HANA workspace services, queries, and run drafts."""

import pytest
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from app.workspace.models import QuerySpec


def dataset(service):
    """Stage a tiny canonical dataset for service tests."""
    return service.register_dataset(
        "fixture",
        {
            "metadata": {"plant": "P", "settings": {}},
            "issues": [],
            "tables": {
                "fini_master": [
                    {
                        "plant": "P",
                        "material": "001",
                        "sefi": "S",
                        "model_status": "modeled",
                        "forecast_litres_12m": 1000,
                        "package_volume": 1,
                        "pallet_litres_resolved": 10,
                        "eligible_lines": "L1",
                        "fixed_pv": "PV1",
                    }
                ],
                "production_versions": [
                    {
                        "plant": "P",
                        "sefi": "S",
                        "production_version": "PV1",
                        "lot_size_litres": 100,
                        "active": True,
                    }
                ],
            },
        },
        [],
    )


def test_snapshot_publication_is_idempotent_and_immutable():
    """Publishing cannot alter source data or generate duplicate dataset versions."""
    service = WorkspaceService(MemoryRepository())
    d = dataset(service)
    assert d["status"] == "review"
    assert service.publish_dataset(d["dataset_id"])["status"] == "published"
    assert service.publish_dataset(d["dataset_id"])["dataset_id"] == d["dataset_id"]
    assert len(service.list_datasets()) == 1


def test_stale_draft_cannot_overwrite_user_change():
    """Optimistic revisions prevent an agent replacing a concurrent manual edit."""
    service = WorkspaceService(MemoryRepository())
    d = dataset(service)
    service.publish_dataset(d["dataset_id"])
    draft = service.create_draft(d["dataset_id"])
    changed = service.update_draft(
        draft["draft_id"], draft["revision"], {"budget": {"frontier_points": 3}}
    )
    assert changed["budget"]["frontier_points"] == 3
    with pytest.raises(ValueError, match="revision"):
        service.update_draft(
            draft["draft_id"], draft["revision"], {"budget": {"frontier_points": 5}}
        )


def test_query_nearest_rank_and_weighted_mean_without_join_duplication():
    """Aggregate whole scope before pagination using optimizer conventions."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert("runs", "run", {"run_id": "run", "revision": 1, "results_ready": True})
    repo.replace_tables(
        "run",
        {
            "groups": [
                {
                    "point_index": 1,
                    "group_id": str(i),
                    "coverage_days": float(i),
                    "group_demand_litres": float(i),
                }
                for i in range(1, 11)
            ]
        },
    )
    result = service.query(
        QuerySpec(
            view="groups",
            run_id="run",
            point_index=1,
            metrics=[
                {"field": "coverage_days", "operation": "p90"},
                {
                    "field": "coverage_days",
                    "operation": "weighted_mean",
                    "weight_field": "group_demand_litres",
                },
            ],
            limit=1,
        )
    )
    assert result["rows"][0]["p90_coverage_days"] == 9
    assert result["rows"][0]["weighted_mean_coverage_days"] == pytest.approx(7)


def test_query_rejects_unknown_view_and_unbounded_page():
    """Query specification is a finite vocabulary, never raw SQL."""
    with pytest.raises(ValueError):
        QuerySpec(view="SYS.USERS", dataset_id="x")
    with pytest.raises(ValueError):
        QuerySpec(view="groups", run_id="x", limit=10001)


def test_run_parameter_provenance_survives_overrides_and_dataset_changes():
    """Record individual defaults/overrides and rebase only inherited calendars."""
    service = WorkspaceService(MemoryRepository())
    first = dataset(service)
    service.publish_dataset(first["dataset_id"])
    draft = service.create_draft(first["dataset_id"])
    assert draft["parameter_sources"]["budget.frontier_points"]["origin"] == "default"
    changed = service.update_draft(
        draft["draft_id"],
        draft["revision"],
        {
            "request": {"config": {"canonical_factor": 0.8}},
            "budget": {"frontier_points": 3},
        },
    )
    assert (
        changed["parameter_sources"]["request.config.canonical_factor"]["origin"]
        == "explicit"
    )
    assert (
        changed["parameter_sources"]["budget.frontier_points"]["origin"] == "explicit"
    )
    second = service.register_dataset(
        "Different calendar",
        {
            "metadata": {
                "plant": "P",
                "settings": {"demand_days": 240, "canonical_factor": 0.7},
                "parameter_sources": {
                    "demand_days": "user",
                    "canonical_factor": "user",
                },
            },
            "issues": [],
            "tables": service.input_tables(first["dataset_id"]),
        },
        [],
    )
    service.publish_dataset(second["dataset_id"])
    rebased = service.update_draft(
        changed["draft_id"], changed["revision"], {"dataset_id": second["dataset_id"]}
    )
    assert rebased["request"]["config"]["demand_days"] == 240
    assert rebased["request"]["config"]["canonical_factor"] == 0.8
    assert (
        rebased["parameter_sources"]["request.config.demand_days"][
            "dataset_parameter_origin"
        ]
        == "user"
    )
    assert (
        service.inspect_dataset(second["dataset_id"])["metadata"]["settings"][
            "canonical_factor"
        ]
        == 0.7
    )


def test_query_response_preserves_identifiers_without_inventing_units():
    """A detail page may contain identifiers and evidence without physical units."""
    service = WorkspaceService(MemoryRepository())
    source = dataset(service)
    result = service.query(
        QuerySpec(view="fini_master", dataset_id=source["dataset_id"], limit=1)
    )
    assert result["rows"][0]["material"] == "001"
    assert result["units"]["material"] is None
    assert result["evidence"]["source_version"] == source["dataset_id"]


def test_workspace_context_expires_with_the_service_process():
    """Agent selections remain usable in-session without surviving in HANA."""
    repository = MemoryRepository()
    current = WorkspaceService(repository)
    current.context("page-session", {"dataset_id": "dataset-1"})
    assert current.context("page-session")["dataset_id"] == "dataset-1"
    restarted = WorkspaceService(repository)
    assert restarted.context("page-session") == {"context_id": "page-session"}
    assert repository.list("contexts") == []
