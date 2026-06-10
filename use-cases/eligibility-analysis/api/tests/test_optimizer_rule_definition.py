import unittest

from app.models.optimizer import RulesConfig


class TestOptimizerRuleDefinition(unittest.TestCase):
    def test_rules_config_preserves_numeric_min_fields(self) -> None:
        payload = {
            "rules": [
                {
                    "name": "exclude_negative_purchase_price",
                    "type": "numeric_min",
                    "column": "Purchase Price",
                    "min_value": 0,
                    "include_equal": True,
                    "enabled": True,
                }
            ]
        }
        model = RulesConfig.model_validate(payload)
        dumped = model.model_dump(exclude_none=True)
        self.assertEqual(dumped["rules"][0]["min_value"], 0)
        self.assertTrue(dumped["rules"][0]["include_equal"])


if __name__ == "__main__":
    unittest.main()
