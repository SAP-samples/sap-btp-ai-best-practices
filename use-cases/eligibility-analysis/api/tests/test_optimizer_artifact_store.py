"""Tests for run-scoped artifact persistence used by workspace downloads.

Run from the repository root:
    cd api && PYTHONPATH=. ../.venv/bin/python -m unittest tests.test_optimizer_artifact_store
"""
import tempfile
import unittest
from pathlib import Path

from app.services.database.backend import BackendType, DatabaseBackend
from app.services.optimizer.artifact_store import OptimizerArtifactStore


class TestOptimizerArtifactStore(unittest.TestCase):
    """Text and binary artifacts round-trip and replace per (run, key)."""

    def test_text_and_binary_round_trip(self) -> None:
        """Saved artifacts are readable, replaceable and isolated per run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OptimizerArtifactStore(db_path=Path(tmpdir) / "artifacts.db",
                                           backend=DatabaseBackend(BackendType.SQLITE))
            store.upsert_text_artifact("run-1", "workspace-manifest", '{"v": 1}')
            store.upsert_text_artifact("run-1", "workspace-manifest", '{"v": 2}')
            self.assertEqual(store.get_text_artifact("run-1", "workspace-manifest"), '{"v": 2}')
            self.assertIsNone(store.get_text_artifact("run-2", "workspace-manifest"))

            store.put_binary("run-1", "report-pdf", b"%PDF-1")
            store.put_binary("run-1", "report-pdf", b"%PDF-2")
            self.assertEqual(store.get_binary("run-1", "report-pdf"), b"%PDF-2")
            self.assertIsNone(store.get_binary("run-1", "missing"))


if __name__ == "__main__":
    unittest.main()
