"""Tests for report_builder.generate_full_report."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _make_metadata() -> dict:
    """Return minimal multi-week metadata for testing."""
    return {
        "cohort": "2025-01-28",
        "planning_mode": "multi_week",
        "planning_start_date": "2025-01-28",
        "horizon_weeks": 8,
        "load_report": {"raw_rows": 6, "loaded_rows": 5},
        "rule_summaries": [
            {"rule_name": "cohort_filter", "rule_type": "equals", "input_rows": 5, "excluded_rows": 0, "output_rows": 5},
        ],
        "limits": {},
        "metrics": {
            "planning_mode": "multi_week",
            "baseline_submitted_count": 5,
            "optimized_submitted_count": 4,
            "rule_excluded_count": 1,
            "not_selected_count": 0,
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
            "customer_weekly_usage": {},
            "group_weekly_usage": {},
            "deferred_reasons": {"CUSTOMER_CAP_BINDING": 1},
            "lifecycle_profile": {"total_rows": 5, "missing_credit_end_pct": 0.0, "repurchase_pct": 0.0},
        },
        "weekly_plan": [
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 2500},
            {"planned_week_index": 1, "planned_week_start_iso": "2025-01-28", "purchase_price": 2700},
        ],
        "deferred_reasons": {"CUSTOMER_CAP_BINDING": 1},
        "lifetime_estimation": {"status": "init_failed", "requested_candidates": 5, "predicted_candidates": 0},
    }


def _setup_output_dir(tmpdir: str) -> Path:
    output_dir = Path(tmpdir)
    sel_df = pd.DataFrame({
        "Invoice Reference": [f"INV-{i:03d}" for i in range(4)],
        "Customer": ["CUST1"] * 4,
        "Company Code": ["2410"] * 4,
        "Purchase Price": [2500, 2500, 100, 100],
        "Currency": ["EUR", "EUR", "USD", "USD"],
        "Due Date": ["2026-02-26"] * 4,
        "planned_week_start_iso": ["2025-01-28"] * 4,
        "expected_lifetime_weeks": [4] * 4,
    })
    sel_df.to_excel(output_dir / "selected.xlsx", index=False)

    exc_df = pd.DataFrame({
        "Invoice Reference": ["INV-EXC"],
        "Customer": ["CUST2"],
        "Purchase Price": [1000],
        "excluded_stage": ["rule"],
        "excluded_reason": ["DUPLICATE"],
        "excluded_reason_detail": ["dup check"],
    })
    exc_df.to_excel(output_dir / "excluded.xlsx", index=False)
    return output_dir


class TestGenerateFullReport(unittest.TestCase):
    def test_report_files_created_without_narrative(self) -> None:
        """generate_full_report with narrative disabled produces all three files."""
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()

            md_path, pdf_path, docx_path = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertTrue(docx_path.exists())

            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Optimizer Run Report", md_text)
            self.assertIn("Selection Metrics", md_text)
            self.assertIn("Selected Invoices", md_text)
            self.assertIn("Facility Utilization", md_text)
            self.assertIn("Rule Funnel", md_text)
            self.assertIn("Customer Concentration", md_text)
            self.assertIn("Customer Cap Binding", md_text)
            self.assertNotIn("CUSTOMER_CAP_BINDING", md_text)

    def test_section_headers_present(self) -> None:
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()

            md_path, _, _ = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )

            md_text = md_path.read_text(encoding="utf-8")

            expected_headers = [
                "# Optimizer Run Report",
                "## Executive Summary",
                "## Selection Metrics",
                "## Selected Invoices",
                "## Excluded Invoices",
                "## Weekly Schedule",
                "## Facility Utilization",
                "## Rule Funnel",
            ]
            for header in expected_headers:
                self.assertIn(header, md_text, f"Missing section: {header}")

    def test_selected_invoices_capped(self) -> None:
        from app.optimizer.report.report_builder import generate_full_report

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
            md_path, _, _ = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )

            md_text = md_path.read_text(encoding="utf-8")
            # Should mention remainder omitted
            self.assertIn("additional selected invoices omitted", md_text)

    def test_multi_week_conditional_schedule(self) -> None:
        """Weekly Schedule section only appears for multi_week mode."""
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()
            metadata["planning_mode"] = "single_week"
            metadata["metrics"]["planning_mode"] = "single_week"
            metadata["weekly_plan"] = []

            md_path, _, _ = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )
            md_text = md_path.read_text(encoding="utf-8")
            self.assertNotIn("## Weekly Schedule", md_text)

    @patch("app.optimizer.report.report_builder._generate_narrative")
    def test_constraint_analysis_appears_below_executive_summary(
        self,
        mock_generate_narrative,
    ) -> None:
        """Constraint analysis is rendered before selection metrics."""
        from app.optimizer.report.report_builder import generate_full_report

        mock_generate_narrative.return_value = (
            "- Summary bullet",
            "Narrative details.",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()

            md_path, _, _ = generate_full_report(
                metadata, output_dir, generate_narrative=True,
            )
            md_text = md_path.read_text(encoding="utf-8")

            idx_exec = md_text.index("## Executive Summary")
            idx_constraint = md_text.index("## Constraint Analysis & Recommendations")
            idx_metrics = md_text.index("## Selection Metrics")
            self.assertLess(idx_exec, idx_constraint)
            self.assertLess(idx_constraint, idx_metrics)

    def test_pdf_is_valid(self) -> None:
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()

            _, pdf_path, _ = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )

            # PDF files start with %PDF
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            self.assertEqual(header, b"%PDF-")

    def test_docx_is_valid(self) -> None:
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = _setup_output_dir(tmpdir)
            metadata = _make_metadata()

            _, _, docx_path = generate_full_report(
                metadata, output_dir, generate_narrative=False,
            )

            # DOCX files are ZIP archives starting with PK
            with open(docx_path, "rb") as f:
                header = f.read(2)
            self.assertEqual(header, b"PK")

    def test_can_render_from_preloaded_dataframes(self) -> None:
        from app.optimizer.report.report_builder import generate_full_report

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metadata = _make_metadata()
            selected_df = pd.DataFrame({
                "Invoice Reference": ["INV-001"],
                "Customer": ["CUST1"],
                "Company Code": ["2410"],
                "Purchase Price": [2500],
                "Currency": ["EUR"],
                "Due Date": ["2026-02-26"],
                "planned_week_start_iso": ["2025-01-28"],
                "expected_lifetime_weeks": [4],
            })
            excluded_df = pd.DataFrame({
                "Invoice Reference": ["INV-EXC"],
                "Customer": ["CUST2"],
                "Purchase Price": [1000],
                "excluded_stage": ["rule"],
                "excluded_reason": ["DUPLICATE"],
                "excluded_reason_detail": ["dup check"],
            })

            md_path, pdf_path, docx_path = generate_full_report(
                metadata,
                output_dir,
                generate_narrative=False,
                selected_df=selected_df,
                excluded_df=excluded_df,
            )

            self.assertTrue(md_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertTrue(docx_path.exists())


if __name__ == "__main__":
    unittest.main()
