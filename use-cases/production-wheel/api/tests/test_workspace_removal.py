"""Verify permanent deletion clears evidence and protects active jobs."""

import pytest
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def test_delete_run_removes_result_rows():
    """A completed run disappears permanently, including its evidence."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert("runs", "r", {"run_id": "r", "revision": 1, "status": "completed"})
    repo.replace_tables("r", {"groups": [{"group_id": "g"}]})
    service.remove_run("r")
    assert service.list_runs() == []
    assert service.list_runs(include_removed=True) == []
    assert repo.rows("r", "groups") == []
    with pytest.raises(KeyError):
        service.get_run("r")


def test_active_run_and_its_snapshot_cannot_be_removed():
    """Removing history cannot orphan a currently executing job."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert(
        "datasets",
        "d",
        {"dataset_id": "d", "revision": 1, "status": "published", "metadata": {}},
    )
    repo.insert(
        "runs",
        "r",
        {"run_id": "r", "dataset_id": "d", "revision": 1, "status": "running"},
    )
    with pytest.raises(ValueError, match="active"):
        service.remove_run("r")
    with pytest.raises(ValueError, match="active"):
        service.remove_dataset("d")


def test_deleted_snapshot_deletes_runs_and_cannot_launch_new_work():
    """Snapshot deletion removes dependent runs and prevents new drafts."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert(
        "datasets",
        "d",
        {"dataset_id": "d", "revision": 1, "status": "published", "metadata": {}},
    )
    repo.insert(
        "runs",
        "r",
        {"run_id": "r", "dataset_id": "d", "revision": 1, "status": "completed"},
    )
    service.remove_dataset("d")
    assert service.list_datasets() == []
    with pytest.raises(KeyError):
        service.get_run("r")
    with pytest.raises(KeyError):
        service.create_draft("d")
    assert service.list_datasets(include_removed=True) == []
