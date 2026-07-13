from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app.services.optimizer.pipeline_runner import run_optimizer_pipeline


class TestPipelineRunnerProgress(unittest.TestCase):
    @patch("app.services.optimizer.pipeline_runner.render_run_summary_markdown", return_value="# Summary")
    @patch(
        "app.services.optimizer.pipeline_runner.compute_run_metrics",
        return_value={
            "baseline_submitted_count": 1,
            "optimized_submitted_count": 1,
            "rule_excluded_count": 0,
            "not_selected_count": 0,
            "candidate_total_amount": 100.0,
            "selected_total_amount": 100.0,
            "optimizer_status": "OPTIMAL",
        },
    )
    @patch(
        "app.services.optimizer.pipeline_runner.explain_non_selection",
        return_value=SimpleNamespace(explained_not_selected_df=pd.DataFrame()),
    )
    @patch(
        "app.services.optimizer.pipeline_runner.optimize_single_week",
        return_value=SimpleNamespace(
            status="OPTIMAL",
            selected_df=pd.DataFrame({"Invoice Reference": ["INV-1"], "Purchase Price": [100.0]}),
            not_selected_df=pd.DataFrame(),
        ),
    )
    @patch("app.services.optimizer.pipeline_runner.limits_to_money_dict", return_value={})
    @patch("app.services.optimizer.pipeline_runner.resolve_limits", return_value=SimpleNamespace())
    @patch("app.services.optimizer.pipeline_runner.load_limits_config", return_value={})
    @patch("app.services.optimizer.pipeline_runner.profile_lifecycle", return_value={})
    @patch("app.services.optimizer.pipeline_runner.derive_lifecycle", side_effect=lambda df, release_event: df)
    @patch(
        "app.services.optimizer.pipeline_runner._offer_prefilter",
        return_value=(pd.DataFrame({"Invoice Reference": ["INV-1"], "Purchase Price": [100.0]}), pd.DataFrame(), []),
    )
    @patch(
        "app.services.optimizer.pipeline_runner.load_offer_file",
        return_value=(pd.DataFrame({"Invoice Reference": ["INV-1"], "Purchase Price": [100.0]}), SimpleNamespace(to_dict=lambda: {})),
    )
    def test_progress_emits_ten_steps_for_single_week(self, *_mocks) -> None:
        events = []
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "input_path": str(Path(tmpdir) / "input.xlsx"),
                "sheet_name": "Sheet1",
                "cohort": "2026-02-18",
                "planning_mode": "single_week",
                "source_profile": "offer_file",
                "planning_start_date": "2026-02-18",
                "horizon_weeks": 8,
                "attempt_cap": 1,
                "release_event": "reconciliation_file_date",
                "lifecycle_input_path": str(Path(tmpdir) / "lifecycle.xlsx"),
                "limits_config_path": str(Path(tmpdir) / "limits.yaml"),
                "rules_config_path": str(Path(tmpdir) / "rules.yaml"),
                "output_dir": str(Path(tmpdir) / "out"),
            }
            metadata = run_optimizer_pipeline(config, progress_callback=lambda payload: events.append(payload))

        self.assertIn("metrics", metadata)
        self.assertTrue(events)
        step_indices = [int(event.get("step_index", 0)) for event in events]
        self.assertIn(1, step_indices)
        self.assertIn(7, step_indices)
        self.assertIn(10, step_indices)
        self.assertEqual(events[-1].get("step_index"), 10)
        self.assertEqual(events[-1].get("phase_status"), "completed")


if __name__ == "__main__":
    unittest.main()
