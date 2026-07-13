"""Tests for report_context.assemble_report_context."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.optimizer.report.report_context import (
    ReportContext,
    assemble_report_context,
)


def _make_metadata(planning_mode: str = "multi_week") -> dict:
    """Return a minimal but realistic metadata dict."""
    return {
        "cohort": "2025-01-28",
        "planning_mode": planning_mode,
        "planning_start_date": "2025-01-28",
        "horizon_weeks": 8,
        "load_report": {"raw_rows": 6, "loaded_rows": 5},
        "rule_summaries": [
            {"rule_name": "cohort_filter", "rule_type": "equals", "input_rows": 5, "excluded_rows": 0, "output_rows": 5},
        ],
        "limits": {"facility_limits_by_company_code": {"2410": 110000.0}},
        "metrics": {
            "planning_mode": planning_mode,
            "baseline_submitted_count": 5,
            "optimized_submitted_count": 4,
            "rule_excluded_count": 0,
            "not_selected_count": 1,
            "candidate_total_amount": 6200.0,
            "selected_total_amount": 5200.0,
            "selected_amount_ratio_pct": 83.87,
            "top3_customer_concentration_pct": 100.0,
            "optimizer_status": "OPTIMAL",
            "weekly_plan_count": 4,
            "horizon_weeks": 8,
            "facility_weekly_usage": {
                "2025-01-28": {
                    "2410": {"used_new": 5200, "used_base": 72500, "used_total": 77700, "limit": 110000, "utilization_pct": 70.6}
                },
            },
            "customer_weekly_usage": {
                "2025-01-28": {
                    "CUSTAI01": {"used_new": 5200, "used_base": 23000, "used_total": 28200, "limit": 60000, "utilization_pct": 47.0},
                    "CUSTAI02": {"used_new": 0, "used_base": 49500, "used_total": 49500, "limit": 50000, "utilization_pct": 99.0},
                },
            },
            "group_weekly_usage": {},
            "deferred_reasons": {"CUSTOMER_CAP_BINDING": 1},
            "lifecycle_profile": {"total_rows": 5, "missing_credit_end_pct": 0.0, "repurchase_pct": 0.0},
        },
        "weekly_plan": [
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 2500, "Invoice Reference": "INV-001"},
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 2500, "Invoice Reference": "INV-002"},
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 100, "Invoice Reference": "INV-003"},
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 100, "Invoice Reference": "INV-004"},
        ],
        "deferred_reasons": {"CUSTOMER_CAP_BINDING": 1},
        "lifetime_estimation": {"status": "init_failed", "requested_candidates": 5, "predicted_candidates": 0},
    }


class TestAssembleReportContext(unittest.TestCase):
    def test_basic_fields_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Create minimal Excel files
            sel_df = pd.DataFrame({
                "Invoice Reference": ["INV-001", "INV-002", "INV-003", "INV-004"],
                "Customer": ["CUSTAI01", "CUSTAI01", "CUSTAI01", "CUSTAI01"],
                "Company Code": ["2410", "2410", "2410", "2410"],
                "Purchase Price": [2500, 2500, 100, 100],
                "Currency": ["EUR", "EUR", "USD", "USD"],
                "Due Date": ["2026-02-26", "2026-02-26", "2026-11-24", "2025-11-24"],
                "planned_week_start_iso": ["2025-01-28"] * 4,
                "expected_lifetime_weeks": [4, 4, 4, 4],
            })
            sel_df.to_excel(output_dir / "selected.xlsx", index=False)

            exc_df = pd.DataFrame({
                "Invoice Reference": ["INV-005"],
                "Customer": ["CUSTAI02"],
                "Purchase Price": [1000],
                "excluded_stage": ["optimizer"],
                "excluded_reason": ["CUSTOMER_CAP_BINDING"],
                "excluded_reason_detail": ["Limit 50000 exceeded"],
            })
            exc_df.to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata()
            ctx = assemble_report_context(metadata, output_dir)

            self.assertIsInstance(ctx, ReportContext)
            self.assertEqual(ctx.cohort, "2025-01-28")
            self.assertEqual(ctx.planning_mode, "multi_week")
            self.assertEqual(ctx.candidate_count, 5)
            self.assertEqual(ctx.selected_count, 4)
            self.assertAlmostEqual(ctx.selected_amount, 5200.0)
            self.assertEqual(ctx.solver_status, "OPTIMAL")

    def test_selected_invoices_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            sel_df = pd.DataFrame({
                "Invoice Reference": [f"INV-{i:03d}" for i in range(30)],
                "Customer": ["C1"] * 30,
                "Purchase Price": [100.0] * 30,
            })
            sel_df.to_excel(output_dir / "selected.xlsx", index=False)
            pd.DataFrame().to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata()
            ctx = assemble_report_context(metadata, output_dir)

            # Capped at 25
            self.assertEqual(len(ctx.selected_invoices), 25)
            self.assertEqual(ctx.total_selected_invoices, 30)

    def test_exclusion_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame().to_excel(output_dir / "selected.xlsx", index=False)

            exc_df = pd.DataFrame({
                "Invoice Reference": ["A", "B", "C"],
                "Customer": ["C1", "C1", "C2"],
                "Purchase Price": [100, 200, 300],
                "excluded_stage": ["rule", "rule", "optimizer"],
                "excluded_reason": ["DUPLICATE", "DUPLICATE", "CAP_BINDING"],
                "excluded_reason_detail": ["dup", "dup", "limit"],
            })
            exc_df.to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata()
            ctx = assemble_report_context(metadata, output_dir)

            self.assertEqual(len(ctx.exclusion_summary), 2)
            dup = next(r for r in ctx.exclusion_summary if r["reason"] == "DUPLICATE")
            self.assertEqual(dup["count"], 2)
            self.assertAlmostEqual(dup["total_amount"], 300.0)

    def test_weekly_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame().to_excel(output_dir / "selected.xlsx", index=False)
            pd.DataFrame().to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata("multi_week")
            ctx = assemble_report_context(metadata, output_dir)

            self.assertTrue(len(ctx.weekly_schedule) > 0)
            self.assertEqual(ctx.weekly_schedule[0]["week_start"], "2025-01-28")
            self.assertEqual(ctx.weekly_schedule[0]["invoice_count"], 4)

    def test_binding_constraints_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame().to_excel(output_dir / "selected.xlsx", index=False)
            pd.DataFrame().to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata()
            ctx = assemble_report_context(metadata, output_dir)

            # CUSTAI02 is at 99% utilization -- should be flagged as binding
            self.assertTrue(len(ctx.binding_constraints) > 0)
            custai02 = next(
                (bc for bc in ctx.binding_constraints if bc["entity_id"] == "CUSTAI02"),
                None,
            )
            self.assertIsNotNone(custai02)
            self.assertGreaterEqual(custai02["peak_utilization_pct"], 95.0)

    def test_facility_utilization_excludes_unconstrained_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame().to_excel(output_dir / "selected.xlsx", index=False)
            pd.DataFrame().to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata()
            metadata["metrics"]["facility_weekly_usage"]["2025-01-28"]["Company Code 1"] = {
                "used_new": 0.0,
                "used_base": 232159.1,
                "used_total": 232159.1,
                "limit": 0.0,
                "utilization_pct": 0.0,
            }

            ctx = assemble_report_context(metadata, output_dir)

            self.assertTrue(ctx.facility_utilization)
            self.assertTrue(all(row["limit"] > 0 for row in ctx.facility_utilization))
            unconstrained = [r for r in ctx.facility_utilization if r["entity_id"] == "Company Code 1"]
            self.assertEqual(unconstrained, [])

    def test_single_week_no_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame().to_excel(output_dir / "selected.xlsx", index=False)
            pd.DataFrame().to_excel(output_dir / "excluded.xlsx", index=False)

            metadata = _make_metadata("single_week")
            ctx = assemble_report_context(metadata, output_dir)

            self.assertEqual(ctx.weekly_schedule, [])

    def test_missing_excel_files_graceful(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # No Excel files created
            metadata = _make_metadata()
            ctx = assemble_report_context(metadata, output_dir)

            self.assertEqual(ctx.selected_invoices, [])
            self.assertEqual(ctx.excluded_invoices, [])
            self.assertEqual(ctx.total_selected_invoices, 0)

    def test_preloaded_dataframes_override_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata = _make_metadata()
            selected_df = pd.DataFrame({
                "Invoice Reference": ["INV-001"],
                "Customer": ["CUST-A"],
                "Company Code": ["2410"],
                "Purchase Price": [150.0],
                "Currency": ["EUR"],
                "Due Date": ["2026-02-26"],
                "planned_week_start_iso": ["2025-01-28"],
                "expected_lifetime_weeks": [4],
            })
            excluded_df = pd.DataFrame({
                "Invoice Reference": ["INV-099"],
                "Customer": ["CUST-B"],
                "Purchase Price": [90.0],
                "excluded_stage": ["rule"],
                "excluded_reason": ["DUPLICATE"],
                "excluded_reason_detail": ["dup"],
            })

            ctx = assemble_report_context(
                metadata,
                output_dir,
                selected_df=selected_df,
                excluded_df=excluded_df,
            )

            self.assertEqual(ctx.total_selected_invoices, 1)
            self.assertEqual(ctx.total_excluded_invoices, 1)
            self.assertEqual(ctx.selected_invoices[0]["Invoice Reference"], "INV-001")


if __name__ == "__main__":
    unittest.main()
