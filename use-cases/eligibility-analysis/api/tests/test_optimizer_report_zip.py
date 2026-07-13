import tempfile
import unittest
import zipfile
from pathlib import Path

from app.services.optimizer.process_manager import ProcessManager


class _StoreStub:
    def __init__(self, process_dir: Path):
        self._process_dir = process_dir

    def get_process(self, process_id: str):
        return {"id": process_id, "process_dir": str(self._process_dir)}


class TestOptimizerReportZip(unittest.TestCase):
    def test_generate_report_zip_includes_all_output_workbooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            process_dir = Path(tmpdir) / "process"
            process_dir.mkdir(parents=True, exist_ok=True)

            for name in [
                "selected.xlsx",
                "excluded.xlsx",
                "weekly_plan.xlsx",
                "weekly_exposure.xlsx",
                "facility_breakdown.xlsx",
            ]:
                (process_dir / name).write_bytes(b"xlsx")

            (process_dir / "run_summary.md").write_text("# Summary\n", encoding="utf-8")
            (process_dir / "run_summary.pdf").write_bytes(b"%PDF-1.4")
            (process_dir / "run_summary.docx").write_bytes(b"PK-docx")
            (process_dir / "extraction.xlsx").write_bytes(b"input-file")

            manager = ProcessManager(store=_StoreStub(process_dir))
            zip_path = manager.generate_report_zip("process-1")

            self.assertIsNotNone(zip_path)
            self.assertTrue(zip_path.exists())

            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())

            self.assertIn("selected.xlsx", names)
            self.assertIn("excluded.xlsx", names)
            self.assertIn("weekly_plan.xlsx", names)
            self.assertIn("weekly_exposure.xlsx", names)
            self.assertIn("facility_breakdown.xlsx", names)
            self.assertIn("run_summary.md", names)
            self.assertIn("run_summary.pdf", names)
            self.assertIn("run_summary.docx", names)
            self.assertNotIn("extraction.xlsx", names)


if __name__ == "__main__":
    unittest.main()
