import unittest

import pandas as pd

from app.optimizer.rules.rule_engine import apply_rules


class TestRuleEngineNumericMin(unittest.TestCase):
    def test_numeric_min_missing_min_value_defaults_to_zero(self) -> None:
        df = pd.DataFrame(
            {
                "Purchase Price": [100.0, -5.0, 0.0],
                "Invoice Reference": ["A", "B", "C"],
            }
        )
        config = {
            "rules": [
                {
                    "name": "exclude_negative_purchase_price",
                    "type": "numeric_min",
                    "column": "Purchase Price",
                    # intentionally missing min_value
                    "enabled": True,
                }
            ]
        }
        result = apply_rules(df, config, context={})
        kept_refs = set(result.eligible_candidates_df["Invoice Reference"].tolist())
        self.assertEqual(kept_refs, {"A", "C"})
        self.assertEqual(len(result.excluded_df), 1)
        self.assertEqual(result.excluded_df.iloc[0]["Invoice Reference"], "B")


if __name__ == "__main__":
    unittest.main()
