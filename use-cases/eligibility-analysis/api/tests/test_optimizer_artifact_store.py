import json
import tempfile
import unittest
from pathlib import Path

from app.services.optimizer.artifact_store import OptimizerArtifactStore


class TestOptimizerArtifactStore(unittest.TestCase):
    def test_text_rows_materialization_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "optimizer.db"
            store = OptimizerArtifactStore(db_path=db_path)
            process_id = "proc-1"

            store.upsert_text_artifact(
                process_id,
                "run_summary.md",
                "# Summary\n",
                metadata={"kind": "summary"},
                artifact_kind="report/markdown",
            )
            self.assertTrue(store.has_text_artifact(process_id, "run_summary.md"))
            self.assertEqual(store.get_text_artifact(process_id, "run_summary.md"), "# Summary\n")
            self.assertEqual(store.get_text_artifact_metadata(process_id, "run_summary.md")["kind"], "summary")

            invoice_rows = [
                {
                    "Invoice Reference": "INV-001",
                    "Customer": "CUST-A",
                    "Company Code": "C001",
                    "Purchase Price": 100.0,
                    "excluded_stage": "rule",
                    "excluded_reason": "DUPLICATE",
                    "planned_week_start_iso": "2025-01-28",
                },
                {
                    "Invoice Reference": "INV-002",
                    "Customer": "CUST-B",
                    "Company Code": "C002",
                    "Purchase Price": 200.0,
                    "excluded_stage": "optimizer",
                    "excluded_reason": "CAP_BINDING",
                    "planned_week_start_iso": "2025-02-04",
                },
            ]
            store.replace_invoice_rows(process_id, "excluded", invoice_rows)
            self.assertTrue(store.has_invoice_rows(process_id, "excluded"))

            paged = store.load_invoice_rows(
                process_id,
                "excluded",
                excluded_reason="cap",
                limit=10,
                offset=0,
            )
            self.assertEqual(paged["total"], 1)
            self.assertEqual(paged["rows"][0]["Invoice Reference"], "INV-002")

            exposure_rows = [
                {
                    "week_start": "2025-01-28",
                    "entity_type": "facility",
                    "entity_id": "C001",
                    "used_new": 10.0,
                    "used_base": 90.0,
                    "used_total": 100.0,
                    "limit": 120.0,
                    "utilization_pct": 83.3,
                }
            ]
            store.replace_exposure_rows(process_id, exposure_rows)
            self.assertTrue(store.has_exposure_rows(process_id))
            exposure = store.load_exposure_rows(process_id, entity_type="facility", limit=10, offset=0)
            self.assertEqual(exposure["total"], 1)
            self.assertEqual(exposure["rows"][0]["entity_id"], "C001")

            summary_path = store.materialize_text_artifact(process_id, "run_summary.md")
            self.assertIsNotNone(summary_path)
            self.assertTrue(summary_path.exists())

            workbook_path = store.materialize_workbook(process_id, "excluded")
            self.assertIsNotNone(workbook_path)
            self.assertTrue(workbook_path.exists())

            store.delete_process_artifacts(process_id)
            self.assertFalse(store.has_text_artifact(process_id, "run_summary.md"))
            self.assertFalse(store.has_invoice_rows(process_id, "excluded"))
            self.assertFalse(store.has_exposure_rows(process_id))


if __name__ == "__main__":
    unittest.main()
