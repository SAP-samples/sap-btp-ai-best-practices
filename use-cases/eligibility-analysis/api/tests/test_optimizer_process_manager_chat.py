"""Tests for ProcessManager chat context and paged row access helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.services.optimizer.process_manager import ProcessManager
from app.services.optimizer.process_store import ProcessStore


def _base_metadata() -> dict:
    return {
        "cohort": "2025-01-28",
        "planning_mode": "multi_week",
        "source_profile": "offer_file",
        "horizon_weeks": 8,
        "metrics": {
            "planning_mode": "multi_week",
            "baseline_submitted_count": 10,
            "optimized_submitted_count": 7,
            "rule_excluded_count": 2,
            "not_selected_count": 1,
            "candidate_total_amount": 10000.0,
            "selected_total_amount": 7300.0,
            "selected_amount_ratio_pct": 73.0,
            "top3_customer_concentration_pct": 82.5,
            "optimizer_status": "OPTIMAL",
            "weekly_plan_count": 7,
            "deferred_reasons": {"CUSTOMER_CAP_BINDING": 1},
            "facility_weekly_usage": {
                "2025-01-28": {
                    "C001": {"used_new": 1000, "used_base": 2000, "used_total": 3000, "limit": 3100, "utilization_pct": 96.7}
                }
            },
            "customer_weekly_usage": {},
            "group_weekly_usage": {},
        },
        # These large arrays should not be surfaced by get_chat_context.
        "weekly_plan": [{"planned_week_index": i, "purchase_price": 100.0} for i in range(200)],
        "weekly_exposure": [{"entity_type": "facility", "entity_id": "C001"} for _ in range(500)],
    }


class TestProcessManagerChatHelpers(unittest.TestCase):
    def test_get_invoice_rows_filters_and_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            process_dir = tmp / "proc-1"
            process_dir.mkdir(parents=True, exist_ok=True)

            excluded_df = pd.DataFrame(
                {
                    "Invoice Reference": ["INV-001", "INV-002", "INV-003"],
                    "Customer": ["CUST-A", "CUST-A", "CUST-B"],
                    "Company Code": ["C001", "C001", "C002"],
                    "Purchase Price": [100.0, 200.0, 300.0],
                    "excluded_stage": ["optimizer", "rule", "optimizer"],
                    "excluded_reason": ["CUSTOMER_CAP_BINDING", "R2_DUPLICATE", "CUSTOMER_CAP_BINDING"],
                    "planned_week_start_iso": ["2025-01-28", "2025-02-04", "2025-01-28"],
                }
            )
            excluded_df.to_excel(process_dir / "excluded.xlsx", index=False)

            db_path = tmp / "store.db"
            store = ProcessStore(db_path=db_path)
            process_id = "abc11111-1111-1111-1111-111111111111"
            store.create_process(
                process_id=process_id,
                process_dir=str(process_dir),
                extraction_filename="input.xlsx",
                cohort="2025-01-28",
            )
            manager = ProcessManager(store=store)

            first_page = manager.get_invoice_rows(
                process_id=process_id,
                bucket="excluded",
                customer="CUST-A",
                limit=1,
                offset=0,
            )
            self.assertEqual(first_page["total"], 2)
            self.assertEqual(len(first_page["rows"]), 1)
            self.assertEqual(first_page["rows"][0]["customer"], "CUST-A")

            second_page = manager.get_invoice_rows(
                process_id=process_id,
                bucket="excluded",
                customer="CUST-A",
                limit=1,
                offset=1,
            )
            self.assertEqual(second_page["total"], 2)
            self.assertEqual(len(second_page["rows"]), 1)
            self.assertNotEqual(
                first_page["rows"][0]["invoice_ref"],
                second_page["rows"][0]["invoice_ref"],
            )

    def test_get_weekly_exposure_rows_sorted_and_paged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            process_dir = tmp / "proc-1"
            process_dir.mkdir(parents=True, exist_ok=True)

            exposure_df = pd.DataFrame(
                {
                    "week_start": ["2025-02-04", "2025-01-28", "2025-01-28"],
                    "entity_type": ["facility", "facility", "customer"],
                    "entity_id": ["C002", "C001", "CU1"],
                    "used_new": [1.0, 2.0, 3.0],
                    "used_base": [10.0, 20.0, 30.0],
                    "used_total": [11.0, 22.0, 33.0],
                    "limit": [100.0, 200.0, 300.0],
                    "utilization_pct": [11.0, 22.0, 33.0],
                }
            )
            exposure_df.to_excel(process_dir / "weekly_exposure.xlsx", index=False)

            db_path = tmp / "store.db"
            store = ProcessStore(db_path=db_path)
            process_id = "abc11111-1111-1111-1111-111111111111"
            store.create_process(
                process_id=process_id,
                process_dir=str(process_dir),
                extraction_filename="input.xlsx",
                cohort="2025-01-28",
            )
            manager = ProcessManager(store=store)

            result = manager.get_weekly_exposure_rows(
                process_id=process_id,
                entity_type="facility",
                limit=1,
                offset=0,
            )
            self.assertEqual(result["total"], 2)
            self.assertEqual(len(result["rows"]), 1)
            self.assertEqual(result["rows"][0]["entity_id"], "C001")

    def test_get_chat_context_is_compact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            process_dir = tmp / "proc-1"
            process_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "Invoice Reference": ["INV-001", "INV-002"],
                    "Customer": ["CUST-A", "CUST-B"],
                    "Purchase Price": [100.0, 200.0],
                }
            ).to_excel(process_dir / "selected.xlsx", index=False)
            pd.DataFrame(
                {
                    "Invoice Reference": ["INV-003"],
                    "Customer": ["CUST-A"],
                    "Purchase Price": [300.0],
                    "excluded_stage": ["optimizer"],
                    "excluded_reason": ["CUSTOMER_CAP_BINDING"],
                    "excluded_reason_detail": ["limit reached"],
                }
            ).to_excel(process_dir / "excluded.xlsx", index=False)
            pd.DataFrame(
                {
                    "planned_week_index": [1, 2],
                    "planned_week_start_iso": ["2025-01-28", "2025-02-04"],
                    "Purchase Price": [100.0, 200.0],
                }
            ).to_excel(process_dir / "weekly_plan.xlsx", index=False)
            pd.DataFrame(
                {
                    "week_start": ["2025-01-28"],
                    "entity_type": ["facility"],
                    "entity_id": ["C001"],
                    "used_new": [100.0],
                    "used_base": [500.0],
                    "used_total": [600.0],
                    "limit": [700.0],
                    "utilization_pct": [85.7],
                }
            ).to_excel(process_dir / "weekly_exposure.xlsx", index=False)

            db_path = tmp / "store.db"
            store = ProcessStore(db_path=db_path)
            process_id = "abc11111-1111-1111-1111-111111111111"
            store.create_process(
                process_id=process_id,
                process_dir=str(process_dir),
                extraction_filename="input.xlsx",
                cohort="2025-01-28",
            )
            store.update_process(
                process_id,
                status="completed",
                run_metadata_json=json.dumps(_base_metadata()),
            )
            manager = ProcessManager(store=store)

            context = manager.get_chat_context(process_id)
            self.assertIn("weekly_schedule_summary", context)
            self.assertIn("exclusion_summary", context)
            self.assertIn("kpis", context)
            self.assertNotIn("weekly_plan", context)
            self.assertNotIn("weekly_exposure", context)
            self.assertEqual(context["row_counts"]["selected"], 2)

    def test_resolve_process_id_ambiguous_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "store.db"
            store = ProcessStore(db_path=db_path)

            process_dir_1 = tmp / "proc-1"
            process_dir_2 = tmp / "proc-2"
            process_dir_1.mkdir(parents=True, exist_ok=True)
            process_dir_2.mkdir(parents=True, exist_ok=True)

            pid_1 = "8ac26111-1111-1111-1111-111111111111"
            pid_2 = "8ac26222-2222-2222-2222-222222222222"
            store.create_process(pid_1, str(process_dir_1), "a.xlsx", "2025-01-28")
            store.create_process(pid_2, str(process_dir_2), "b.xlsx", "2025-01-28")
            manager = ProcessManager(store=store)

            ambiguous = manager.resolve_process_id("8ac26")
            self.assertEqual(ambiguous["match_type"], "prefix_ambiguous")
            self.assertEqual(len(ambiguous["matches"]), 2)

            exact = manager.resolve_process_id(pid_1)
            self.assertEqual(exact["match_type"], "exact")
            self.assertEqual(exact["process_id"], pid_1)


if __name__ == "__main__":
    unittest.main()
