import unittest

import pandas as pd

from app.optimizer.rules.rule_engine import apply_rules


class TestRuleEngineNumericMin(unittest.TestCase):
    def test_numeric_min_excludes_negative_purchase_price(self) -> None:
        df = pd.DataFrame({"Purchase Price": [150.0, 0.0, -25.5, None]})
        rules_config = {
            "rules": [
                {
                    "name": "exclude_negative_purchase_price",
                    "type": "numeric_min",
                    "column": "Purchase Price",
                    "min_value": 0,
                    "include_equal": True,
                }
            ]
        }

        result = apply_rules(df, rules_config=rules_config, context={})

        self.assertEqual(len(result.eligible_candidates_df), 2)
        self.assertEqual(len(result.excluded_df), 2)
        self.assertListEqual(
            result.eligible_candidates_df["Purchase Price"].tolist(),
            [150.0, 0.0],
        )
        self.assertTrue((result.excluded_df["excluded_reason"] == "exclude_negative_purchase_price").all())

    def test_numeric_min_supports_strict_threshold_from_context(self) -> None:
        df = pd.DataFrame({"Purchase Price": [0.0, 0.01, -1.0]})
        rules_config = {
            "rules": [
                {
                    "name": "strict_positive_purchase_price",
                    "type": "numeric_min",
                    "column": "Purchase Price",
                    "min_value_from_context": "amount_floor",
                    "include_equal": False,
                }
            ]
        }

        result = apply_rules(df, rules_config=rules_config, context={"amount_floor": 0})

        self.assertEqual(len(result.eligible_candidates_df), 1)
        self.assertAlmostEqual(float(result.eligible_candidates_df.iloc[0]["Purchase Price"]), 0.01, places=6)
        self.assertEqual(len(result.excluded_df), 2)


if __name__ == "__main__":
    unittest.main()
