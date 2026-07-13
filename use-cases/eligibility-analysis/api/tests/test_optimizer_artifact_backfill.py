import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.services.optimizer.artifact_store import OptimizerArtifactStore
from app.services.optimizer.process_manager import ProcessManager, SUMMARY_ARTIFACT_KEYS
from app.services.optimizer.process_store import ProcessStore
from scripts.migrate_optimizer_artifacts_to_hana import migrate_process


def _metadata() -> dict:
    return {
        "cohort": "2025-01-28",
        "planning_mode": "multi_week",
        "planning_start_date": "2025-01-28",
        "horizon_weeks": 8,
        "metrics": {
            "planning_mode": "multi_week",
            "baseline_submitted_count": 2,
            "optimized_submitted_count": 1,
            "rule_excluded_count": 1,
            "not_selected_count": 0,
            "candidate_total_amount": 300.0,
            "selected_total_amount": 100.0,
            "selected_amount_ratio_pct": 33.3,
            "top3_customer_concentration_pct": 100.0,
            "optimizer_status": "OPTIMAL",
            "weekly_plan_count": 1,
            "facility_weekly_usage": {
                "2025-01-28": {
                    "C001": {
                        "used_new": 100.0,
                        "used_base": 900.0,
                        "used_total": 1000.0,
                        "limit": 1200.0,
                        "utilization_pct": 83.3,
                    }
                }
            },
            "customer_weekly_usage": {},
            "group_weekly_usage": {},
        },
        "weekly_plan": [
            {
                "Invoice Reference": "INV-001",
                "Customer": "CUST-A",
                "Company Code": "C001",
                "Purchase Price": 100.0,
                "planned_week_index": 1,
                "planned_week_start_iso": "2025-01-28",
                "expected_lifetime_weeks": 4,
            }
        ],
        "weekly_exposure": [
            {
                "week_start": "2025-01-28",
                "entity_type": "facility",
                "entity_id": "C001",
                "used_new": 100.0,
                "used_base": 900.0,
                "used_total": 1000.0,
                "limit": 1200.0,
                "utilization_pct": 83.3,
            }
        ],
        "rule_summaries": [],
        "load_report": {},
        "limits": {},
        "deferred_reasons": {},
        "lifetime_estimation": {"status": "not_requested"},
    }


class TestOptimizerArtifactBackfill(unittest.TestCase):
    def _build_manager(self, tmpdir: str) -> tuple[ProcessManager, str, Path]:
        tmp = Path(tmpdir)
        db_path = tmp / "optimizer.db"
        process_dir = tmp / "proc-1"
        process_dir.mkdir(parents=True, exist_ok=True)

        store = ProcessStore(db_path=db_path)
        artifact_store = OptimizerArtifactStore(db_path=db_path)
        process_id = "abc11111-1111-1111-1111-111111111111"
        store.create_process(process_id, str(process_dir), "input.xlsx", "2025-01-28")
        store.update_process(
            process_id,
            status="completed",
            run_metadata_json=json.dumps(_metadata()),
        )
        return ProcessManager(store=store, artifact_store=artifact_store), process_id, process_dir

    def test_manager_reads_and_materializes_from_artifact_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, process_id, process_dir = self._build_manager(tmpdir)

            manager.artifact_store.upsert_text_artifact(
                process_id,
                "run_summary.md",
                "# Summary\n",
                artifact_kind="report/markdown",
            )
            manager.artifact_store.replace_invoice_rows(
                process_id,
                "selected",
                _metadata()["weekly_plan"],
            )
            manager.artifact_store.replace_invoice_rows(
                process_id,
                "excluded",
                [
                    {
                        "Invoice Reference": "INV-EXC",
                        "Customer": "CUST-B",
                        "Company Code": "C001",
                        "Purchase Price": 200.0,
                        "excluded_stage": "rule",
                        "excluded_reason": "DUPLICATE",
                        "excluded_reason_detail": "dup",
                    }
                ],
            )
            manager.artifact_store.replace_invoice_rows(
                process_id,
                "weekly_plan",
                _metadata()["weekly_plan"],
            )
            manager.artifact_store.replace_exposure_rows(process_id, _metadata()["weekly_exposure"])

            selected_rows = manager.get_invoice_rows(process_id, bucket="selected", limit=10, offset=0)
            self.assertEqual(selected_rows["total"], 1)
            self.assertEqual(selected_rows["rows"][0]["invoice_ref"], "INV-001")

            exposure_rows = manager.get_weekly_exposure_rows(process_id, limit=10, offset=0)
            self.assertEqual(exposure_rows["total"], 1)
            self.assertEqual(exposure_rows["rows"][0]["entity_id"], "C001")

            selected_path = manager.get_file_path(process_id, "selected")
            summary_path = manager.get_file_path(process_id, "summary")
            pdf_path = manager.get_file_path(process_id, "pdf")
            zip_path = manager.generate_report_zip(process_id)

            self.assertIsNotNone(selected_path)
            self.assertTrue(selected_path.exists())
            self.assertIsNotNone(summary_path)
            self.assertTrue(summary_path.exists())
            self.assertIsNotNone(pdf_path)
            self.assertTrue(pdf_path.exists())
            self.assertIsNotNone(zip_path)
            self.assertTrue(zip_path.exists())

            manager.get_overview_summary(process_id)
            self.assertTrue(manager.artifact_store.has_text_artifact(process_id, SUMMARY_ARTIFACT_KEYS["overview"]))

            self.assertTrue(manager.delete_process(process_id))
            self.assertIsNone(manager.get_process(process_id))
            self.assertFalse(manager.artifact_store.has_invoice_rows(process_id, "selected"))
            self.assertFalse(manager.artifact_store.has_text_artifact(process_id, SUMMARY_ARTIFACT_KEYS["overview"]))
            self.assertFalse((process_dir).exists())

    def test_sync_process_artifacts_persists_summary_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, process_id, process_dir = self._build_manager(tmpdir)
            metadata = _metadata()

            pd.DataFrame(metadata["weekly_plan"]).to_excel(process_dir / "selected.xlsx", index=False)
            pd.DataFrame([
                {
                    "Invoice Reference": "INV-EXC",
                    "Customer": "CUST-B",
                    "Company Code": "C001",
                    "Purchase Price": 200.0,
                    "excluded_stage": "rule",
                    "excluded_reason": "DUPLICATE",
                    "excluded_reason_detail": "dup",
                }
            ]).to_excel(process_dir / "excluded.xlsx", index=False)
            pd.DataFrame(metadata["weekly_plan"]).to_excel(process_dir / "weekly_plan.xlsx", index=False)

            persisted = manager.sync_process_artifacts(process_id, force=True)

            self.assertEqual(persisted["summary_sections"], 4)
            for artifact_key in SUMMARY_ARTIFACT_KEYS.values():
                self.assertTrue(manager.artifact_store.has_text_artifact(process_id, artifact_key))

    def test_lazy_summary_backfill_persists_and_reuses_summary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, process_id, process_dir = self._build_manager(tmpdir)

            pd.DataFrame(
                [
                    {
                        "Invoice Reference": "INV-EXC",
                        "Customer": "CUST-B",
                        "Company Code": "C001",
                        "Purchase Price": 200.0,
                        "excluded_stage": "rule",
                        "excluded_reason": "DUPLICATE",
                        "excluded_reason_detail": "dup",
                    }
                ]
            ).to_excel(process_dir / "excluded.xlsx", index=False)

            summary = manager.get_exclusions_summary(process_id)
            self.assertEqual(summary["rows"][0]["reason"], "DUPLICATE")
            self.assertTrue(manager.artifact_store.has_text_artifact(process_id, SUMMARY_ARTIFACT_KEYS["exclusions"]))

            (process_dir / "excluded.xlsx").unlink()
            manager = ProcessManager(store=manager.store, artifact_store=manager.artifact_store)
            summary = manager.get_exclusions_summary(process_id)
            self.assertEqual(summary["rows"][0]["reason"], "DUPLICATE")

    def test_migrate_process_backfills_rows_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, process_id, process_dir = self._build_manager(tmpdir)
            metadata = _metadata()

            pd.DataFrame(metadata["weekly_plan"]).to_excel(process_dir / "selected.xlsx", index=False)
            pd.DataFrame([
                {
                    "Invoice Reference": "INV-EXC",
                    "Customer": "CUST-B",
                    "Company Code": "C001",
                    "Purchase Price": 200.0,
                    "excluded_stage": "rule",
                    "excluded_reason": "DUPLICATE",
                    "excluded_reason_detail": "dup",
                }
            ]).to_excel(process_dir / "excluded.xlsx", index=False)
            pd.DataFrame(metadata["weekly_plan"]).to_excel(process_dir / "weekly_plan.xlsx", index=False)
            pd.DataFrame(metadata["weekly_exposure"]).to_excel(process_dir / "weekly_exposure.xlsx", index=False)
            (process_dir / "limits.yaml").write_text("facility_limits_by_company_code: {}\n", encoding="utf-8")
            (process_dir / "rules.yaml").write_text("rules: []\n", encoding="utf-8")
            (process_dir / "run_summary.md").write_text("# Summary\n", encoding="utf-8")
            (process_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = migrate_process(process_id, manager, force=True)

            self.assertEqual(result["status"], "migrated")
            self.assertEqual(result["persisted_counts"]["selected_rows"], 1)
            self.assertTrue(manager.artifact_store.has_text_artifact(process_id, "run_summary.md"))
            self.assertTrue(manager.artifact_store.has_invoice_rows(process_id, "weekly_plan"))
            self.assertTrue(manager.artifact_store.has_exposure_rows(process_id))
            self.assertTrue(manager.artifact_store.has_text_artifact(process_id, SUMMARY_ARTIFACT_KEYS["overview"]))


if __name__ == "__main__":
    unittest.main()
