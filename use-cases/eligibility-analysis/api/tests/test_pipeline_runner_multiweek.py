import unittest

import pandas as pd

from app.services.optimizer.pipeline_runner import (
    _apply_multi_week_candidate_window,
    _disable_single_week_cohort_rule_for_multi_week,
    _offer_prefilter,
)


class TestPipelineRunnerMultiWeekRules(unittest.TestCase):
    def test_disable_single_week_cohort_rule(self) -> None:
        cfg = {
            "rules": [
                {
                    "name": "cohort_target_offer_file_date",
                    "type": "equals",
                    "column": "Offer File Date (UTC)",
                    "value_from_context": "cohort_ts",
                },
                {"name": "another_rule", "type": "regex_not_contains", "column": "Status", "pattern": "x"},
            ]
        }
        out = _disable_single_week_cohort_rule_for_multi_week(cfg)
        self.assertFalse(out["rules"][0]["enabled"])
        self.assertNotIn("enabled", out["rules"][1])

    def test_apply_multi_week_candidate_window(self) -> None:
        df = pd.DataFrame(
            {
                "Invoice Reference": ["A", "B", "C", "D"],
                "Offer File Date (UTC)": [
                    pd.Timestamp("2025-01-21"),
                    pd.Timestamp("2025-01-28"),
                    pd.Timestamp("2025-02-04"),
                    pd.Timestamp("2025-02-18"),
                ],
            }
        )
        kept, excluded, summary = _apply_multi_week_candidate_window(
            df,
            planning_start_date="2025-01-28",
            horizon_weeks=2,
        )
        self.assertSetEqual(set(kept["Invoice Reference"]), {"B", "C"})
        self.assertSetEqual(set(excluded["Invoice Reference"]), {"A", "D"})
        self.assertTrue((excluded["excluded_reason"] == "planning_window_offer_file_date").all())
        self.assertEqual(summary["input_rows"], 4)
        self.assertEqual(summary["excluded_rows"], 2)
        self.assertEqual(summary["output_rows"], 2)

    def test_offer_prefilter_excludes_negative_amounts_before_dedupe(self) -> None:
        df = pd.DataFrame(
            {
                "invoice_reference": ["INV-1", "INV-1", "INV-2", "INV-3"],
                "candidate_amount": [-10.0, 50.0, 20.0, -5.0],
            }
        )
        kept, excluded, summaries = _offer_prefilter(df)

        self.assertSetEqual(set(kept["invoice_reference"]), {"INV-1", "INV-2"})
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(excluded), 2)

        reasons = set(excluded["excluded_reason"].tolist())
        self.assertSetEqual(reasons, {"exclude_negative_purchase_price"})

        self.assertEqual(summaries[0]["rule_name"], "exclude_negative_purchase_price")
        self.assertEqual(summaries[0]["excluded_rows"], 2)
        self.assertEqual(summaries[1]["rule_name"], "deduplicate_invoice_reference_offer")
        self.assertEqual(summaries[1]["excluded_rows"], 0)


if __name__ == "__main__":
    unittest.main()
