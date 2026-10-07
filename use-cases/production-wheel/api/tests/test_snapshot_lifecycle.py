"""Regression coverage for independent imports and permanent history deletion."""
import pytest
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def test_reimport_published_workbook_creates_new_review():
    """Identical source content must not silently select an existing publication."""
    service = WorkspaceService(MemoryRepository())
    extracted = {"metadata": {"input_hashes": {"primary": "same"}},
                 "tables": {"fini_master": [{"model_status": "modeled"}]}, "issues": []}
    reference = service.register_dataset("Reference", extracted, [("wheel.xlsx", b"source")])
    service.publish_dataset(reference["dataset_id"])
    imported = service.register_dataset("New import", extracted, [("wheel.xlsx", b"source")])
    assert imported["dataset_id"] != reference["dataset_id"]
    assert imported["status"] == "review"
    assert imported["name"] == "New import"
    assert len(service.list_datasets()) == 2


def test_delete_snapshot_cascades_and_preserves_other_owners():
    """Delete source, drafts, terminal runs and artifacts only for the chosen snapshot."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    for key in ("d", "other"):
        repo.insert("datasets", key, {"dataset_id": key, "revision": 1, "metadata": {}})
        repo.replace_tables(key, {"fini_master": [{"fini": key}]})
        repo.put_artifact(key, "source.xlsx", b"source")
    repo.insert("drafts", "draft", {"draft_id": "draft", "dataset_id": "d", "revision": 1})
    repo.insert("runs", "r", {"run_id": "r", "dataset_id": "d", "revision": 1, "status": "completed", "removed_at": "old"})
    repo.replace_tables("r", {"groups": [{"group_id": "g"}]})
    repo.put_artifact("r", "checkpoint", b"results")
    service.remove_dataset("d")
    for kind, key in (("datasets", "d"), ("drafts", "draft"), ("runs", "r")):
        with pytest.raises(KeyError):
            repo.get(kind, key)
    assert repo.rows("d", "fini_master") == []
    assert repo.rows("r", "groups") == []
    assert set(repo.artifacts) == {("other", "source.xlsx")}
    assert repo.rows("other", "fini_master") == [{"fini": "other"}]


def test_failed_cascade_rolls_back_deleted_children(monkeypatch):
    """A storage failure cannot leave a half-deleted snapshot or changed revisions."""
    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert("datasets", "d", {"dataset_id": "d", "revision": 1, "metadata": {}})
    repo.insert("runs", "r", {"run_id": "r", "dataset_id": "d", "revision": 1, "status": "completed"})
    repo.put_artifact("r", "checkpoint", b"results")
    original = repo.delete_owner

    def fail_parent(kind, key):
        """Simulate a late database failure after child deletion."""
        if kind == "datasets":
            raise RuntimeError("storage failure")
        original(kind, key)

    monkeypatch.setattr(repo, "delete_owner", fail_parent)
    with pytest.raises(RuntimeError, match="storage failure"):
        service.remove_dataset("d")
    assert repo.get("datasets", "d")["revision"] == 1
    assert repo.get("runs", "r")["revision"] == 1
    assert repo.artifact("r", "checkpoint") == b"results"


def test_legacy_migration_can_reuse_content_identity():
    """Historical run imports retain deterministic lineage without affecting UI imports."""
    service = WorkspaceService(MemoryRepository())
    extracted = {"metadata": {}, "tables": {"fini_master": [{"model_status": "modeled"}]}, "issues": []}
    first = service.register_dataset("history", extracted, [], reuse_content=True)
    second = service.register_dataset("history", extracted, [], reuse_content=True)
    uploaded = service.register_dataset("upload", extracted, [])
    assert first["dataset_id"] == second["dataset_id"]
    assert uploaded["dataset_id"] != first["dataset_id"]
    assert uploaded["metadata"]["content_fingerprint"] == first["metadata"]["content_fingerprint"]
