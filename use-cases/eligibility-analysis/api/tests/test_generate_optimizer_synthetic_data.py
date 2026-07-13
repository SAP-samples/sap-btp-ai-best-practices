import tempfile
import unittest
from pathlib import Path

import json
import pandas as pd

from app.optimizer.io.load_extraction import load_extraction
from app.optimizer.model.lifecycle import derive_lifecycle
from scripts.generate_optimizer_synthetic_data import GeneratorConfig, generate_synthetic_package


class TestGenerateOptimizerSyntheticData(unittest.TestCase):
    def test_generate_package_without_candidate_reconciliation_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GeneratorConfig(
                start_date="2026-02-02",
                weeks=3,
                invoices_per_week=5,
                history_weeks=1,
                companies=2,
                customers_per_company=3,
                groups_per_company=2,
                seed=123,
                enable_reconciliation_date=False,
                output_root=Path(tmpdir),
                scenario_name="case_no_recon",
            )
            out = generate_synthetic_package(cfg)

            scenario_dir = Path(out["scenario_dir"])
            extraction_path = scenario_dir / "synthetic_extraction.xlsx"
            limits_path = scenario_dir / "limits.yaml"
            manifest_path = scenario_dir / "scenario_manifest.json"
            readme_path = scenario_dir / "README.md"

            self.assertTrue(extraction_path.exists())
            self.assertTrue(limits_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(readme_path.exists())

            loaded_df, report = load_extraction(extraction_path, sheet_name="SAPUI5 Export")
            self.assertGreater(report.loaded_rows, 0)
            self.assertIn("Offer File Date (UTC)", loaded_df.columns)

            raw_df = pd.read_excel(extraction_path, sheet_name="SAPUI5 Export", engine="openpyxl")
            candidates = raw_df[raw_df["Synthetic Row Type"] == "candidate"].copy()
            self.assertEqual(len(candidates), 15)
            self.assertEqual(candidates["Offer File Date (UTC)"].dt.date.nunique(), 3)
            self.assertTrue(candidates["Reconciliation File Date (UTC)"].isna().all())
            self.assertTrue((candidates["Due Date"] >= candidates["Offer File Date (UTC)"]).all())

            limits_payload = json.loads(limits_path.read_text(encoding="utf-8"))
            self.assertFalse(limits_payload["synthetic_generation"]["enabled"])
            for company, base in limits_payload["base_exposure"]["facility"].items():
                self.assertLess(base, limits_payload["facility_limits_by_company_code"][company])
            for customer, base in limits_payload["base_exposure"]["customer"].items():
                self.assertLess(base, limits_payload["customer_limits"][customer])

    def test_generate_package_with_candidate_reconciliation_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GeneratorConfig(
                start_date="2026-02-09",
                weeks=2,
                invoices_per_week=6,
                history_weeks=1,
                companies=1,
                customers_per_company=4,
                groups_per_company=2,
                seed=9,
                enable_reconciliation_date=True,
                output_root=Path(tmpdir),
                scenario_name="case_with_recon",
            )
            out = generate_synthetic_package(cfg)
            extraction_path = Path(out["synthetic_extraction"])

            raw_df = pd.read_excel(extraction_path, sheet_name="SAPUI5 Export", engine="openpyxl")
            candidates = raw_df[raw_df["Synthetic Row Type"] == "candidate"].copy()
            self.assertTrue(candidates["Reconciliation File Date (UTC)"].notna().all())

            candidates = derive_lifecycle(candidates, release_event="reconciliation_file_date")
            duration = pd.to_numeric(candidates["credit_duration_days"], errors="coerce")
            self.assertTrue((duration > 0).all())

    def test_stress_knobs_history_and_facility_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GeneratorConfig(
                start_date="2026-03-02",
                weeks=2,
                invoices_per_week=6,
                history_weeks=2,
                history_invoices_per_week=6,
                companies=1,
                customers_per_company=3,
                groups_per_company=1,
                seed=11,
                facility_limit_fraction_of_total=0.5,
                base_exposure_ratio=0.7,
                enable_reconciliation_date=True,
                output_root=Path(tmpdir),
                scenario_name="stress_case",
            )
            out = generate_synthetic_package(cfg)
            extraction_path = Path(out["synthetic_extraction"])
            limits_path = Path(out["limits_yaml"])

            raw_df = pd.read_excel(extraction_path, sheet_name="SAPUI5 Export", engine="openpyxl")
            history = raw_df[raw_df["Synthetic Row Type"] == "history"]
            candidates = raw_df[raw_df["Synthetic Row Type"] == "candidate"]
            self.assertEqual(len(history), 12)
            self.assertEqual(len(candidates), 12)

            limits_payload = json.loads(limits_path.read_text(encoding="utf-8"))
            total = float(candidates["Purchase Price"].sum())
            facility_limit = float(next(iter(limits_payload["facility_limits_by_company_code"].values())))
            self.assertAlmostEqual(facility_limit, round(total * 0.5, 2), places=2)


if __name__ == "__main__":
    unittest.main()
