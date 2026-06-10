import tempfile
import unittest
from pathlib import Path

from app.services.optimizer.process_store import ProcessStore


class TestOptimizerProcessStore(unittest.TestCase):
    def test_progress_columns_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "optimizer_processes.db"
            store = ProcessStore(db_path=db_path)
            with store._get_connection() as conn:  # noqa: SLF001 - validated store schema
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(optimizer_processes)")
                columns = {row[1] for row in cur.fetchall()}
            self.assertIn("progress_json", columns)
            self.assertIn("progress_updated_at", columns)

    def test_list_processes_and_get_process_head_avoid_run_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "optimizer_processes.db"
            process_dir = Path(tmpdir) / "proc-1"
            process_dir.mkdir(parents=True, exist_ok=True)
            store = ProcessStore(db_path=db_path)
            process_id = "proc-1"
            store.create_process(process_id, str(process_dir), "input.xlsx", "2025-01-28")
            store.update_process(process_id, run_metadata_json='{"large": true}')

            listed = store.list_processes(limit=10, offset=0)[0]
            head = store.get_process_head(process_id)
            full = store.get_process(process_id)

            self.assertNotIn("run_metadata_json", listed)
            self.assertNotIn("run_metadata_json", head)
            self.assertIn("run_metadata_json", full)


if __name__ == "__main__":
    unittest.main()
