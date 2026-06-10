import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from scripts.evaluate_rpt1_on_synthetic import (
    DEFAULT_CONTEXT_INPUT_PATH,
    evaluate_rpt1_on_synthetic,
)
from scripts.generate_optimizer_synthetic_data import GeneratorConfig, generate_synthetic_package


def _fake_estimator(
    candidates_df: pd.DataFrame,
    _lifecycle_source_df: pd.DataFrame,
    *,
    config: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = candidates_df.copy().reset_index(drop=True)
    summary = pd.to_datetime(out["Summary File Date (UTC)"], errors="coerce")
    recon = pd.to_datetime(out["Reconciliation File Date (UTC)"], errors="coerce")
    predicted_days = (recon - summary).dt.days
    out["expected_lifetime_days"] = predicted_days
    out["expected_lifetime_weeks"] = predicted_days.apply(
        lambda v: None if pd.isna(v) else max(1, int((float(v) + 6) // 7))
    )
    out["expected_lifetime_confidence"] = 1.0
    out["expected_lifetime_source"] = "fake_estimator"
    return out, {
        "status": "ok",
        "predicted_candidates": int(predicted_days.notna().sum()),
        "default_lifetime_weeks": int(config.default_lifetime_weeks),
    }


class TestEvaluateRPT1OnSynthetic(unittest.TestCase):
    def test_default_context_path_points_to_extraction_btp(self) -> None:
        self.assertTrue(str(DEFAULT_CONTEXT_INPUT_PATH).endswith("data/2026/EXTRACTION BTP.xlsx"))

    def test_fails_when_reconciliation_dates_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = generate_synthetic_package(
                GeneratorConfig(
                    start_date="2026-02-02",
                    weeks=2,
                    invoices_per_week=4,
                    history_weeks=1,
                    enable_reconciliation_date=False,
                    output_root=Path(tmpdir),
                    scenario_name="missing_recon",
                )
            )
            candidate_path = scenario["synthetic_extraction"]
            with self.assertRaises(ValueError) as err:
                evaluate_rpt1_on_synthetic(
                    candidate_input_path=candidate_path,
                    context_input_path=candidate_path,
                    output_dir=Path(tmpdir) / "eval",
                    estimator=_fake_estimator,
                )
            self.assertIn("--enable-reconciliation-date", str(err.exception))

    def test_evaluation_outputs_created_when_reconciliation_dates_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = generate_synthetic_package(
                GeneratorConfig(
                    start_date="2026-02-02",
                    weeks=2,
                    invoices_per_week=4,
                    history_weeks=1,
                    enable_reconciliation_date=True,
                    output_root=Path(tmpdir),
                    scenario_name="with_recon",
                )
            )
            candidate_path = scenario["synthetic_extraction"]
            out_dir = Path(tmpdir) / "eval"

            outputs = evaluate_rpt1_on_synthetic(
                candidate_input_path=candidate_path,
                context_input_path=candidate_path,
                output_dir=out_dir,
                estimator=_fake_estimator,
            )
            self.assertTrue(Path(outputs["metrics_output"]).exists())
            self.assertTrue(Path(outputs["predictions_output"]).exists())
            self.assertGreater(outputs["metrics"]["evaluated_rows"], 0)
            self.assertEqual(outputs["metrics"]["mae_days"], 0.0)


if __name__ == "__main__":
    unittest.main()
