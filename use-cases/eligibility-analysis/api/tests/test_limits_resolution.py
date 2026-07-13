import importlib.util
import unittest

import pandas as pd

from app.optimizer.model.limits import resolve_limits
from app.optimizer.opt.optimizer_multi_week import (
    MultiWeekOptimizerSettings,
    optimize_multi_week,
)

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None


class TestLimitResolutionFractions(unittest.TestCase):
    def _candidates(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Invoice Reference": ["INV-1", "INV-2"],
                "Company Code": ["C1", "C1"],
                "Customer": ["U1", "U2"],
                "Purchase Price": [100.0, 200.0],
                "Due Date": [pd.Timestamp("2026-02-20"), pd.Timestamp("2026-02-20")],
            }
        )

    def test_synthetic_mode_uses_beta_gamma_for_default_limits(self) -> None:
        cfg = {
            "facility_limits_by_company_code": {},
            "customer_limits": {},
            "group_limits": {},
            "customer_to_group": {"U1": "G1", "U2": "G1"},
            "defaults": {
                "customer_limit_fraction_of_facility": 0.15,
                "group_limit_fraction_of_facility": 0.30,
            },
            "synthetic_generation": {
                "enabled": True,
                "alpha": 1.0,
                "beta": 0.40,
                "gamma": 0.60,
            },
        }

        limits = resolve_limits(self._candidates(), cfg)
        self.assertEqual(limits.facility_limits_by_company_code["C1"], 30000)
        self.assertEqual(limits.customer_limits["U1"], 12000)
        self.assertEqual(limits.customer_limits["U2"], 12000)
        self.assertEqual(limits.group_limits["G1"], 18000)

    def test_manual_mode_keeps_defaults_for_default_limits(self) -> None:
        cfg = {
            "facility_limits_by_company_code": {"C1": 1000.0},
            "customer_limits": {},
            "group_limits": {},
            "customer_to_group": {"U1": "G1", "U2": "G1"},
            "defaults": {
                "customer_limit_fraction_of_facility": 0.20,
                "group_limit_fraction_of_facility": 0.30,
            },
            "synthetic_generation": {
                "enabled": False,
                "alpha": 0.85,
                "beta": 0.40,
                "gamma": 0.60,
            },
        }

        limits = resolve_limits(self._candidates(), cfg)
        self.assertEqual(limits.facility_limits_by_company_code["C1"], 100000)
        self.assertEqual(limits.customer_limits["U1"], 20000)
        self.assertEqual(limits.customer_limits["U2"], 20000)
        self.assertEqual(limits.group_limits["G1"], 30000)

    def test_base_exposure_is_resolved_in_cents(self) -> None:
        cfg = {
            "facility_limits_by_company_code": {"C1": 1000.0},
            "customer_limits": {"U1": 400.0},
            "group_limits": {"G1": 800.0},
            "customer_to_group": {"U1": "G1"},
            "base_exposure": {
                "facility": {"C1": 250.0},
                "customer": {"U1": 120.5},
                "group": {"G1": 300.0},
            },
            "defaults": {
                "customer_limit_fraction_of_facility": 0.20,
                "group_limit_fraction_of_facility": 0.30,
            },
            "synthetic_generation": {
                "enabled": False,
                "alpha": 0.85,
                "beta": 0.40,
                "gamma": 0.60,
            },
        }

        limits = resolve_limits(self._candidates(), cfg)
        self.assertEqual(limits.base_exposure_facility["C1"], 25000)
        self.assertEqual(limits.base_exposure_customer["U1"], 12050)
        self.assertEqual(limits.base_exposure_group["G1"], 30000)


@unittest.skipUnless(HAS_ORTOOLS, "ortools is required for multi-week optimizer tests")
class TestMultiWeekBaseExposureValidation(unittest.TestCase):
    def test_base_exposure_violation_returns_explanatory_error(self) -> None:
        candidates = pd.DataFrame(
            {
                "Invoice Reference": ["INV-1"],
                "Company Code": ["C1"],
                "Customer": ["U1"],
                "Purchase Price": [100.0],
                "Due Date": [pd.Timestamp("2026-02-20")],
                "expected_lifetime_weeks": [1],
            }
        )
        limits_cfg = {
            "facility_limits_by_company_code": {"C1": 100.0},
            "customer_limits": {"U1": 100.0},
            "group_limits": {},
            "customer_to_group": {},
            "defaults": {
                "customer_limit_fraction_of_facility": 0.15,
                "group_limit_fraction_of_facility": 0.30,
            },
            "synthetic_generation": {
                "enabled": False,
                "alpha": 0.85,
                "beta": 0.15,
                "gamma": 0.30,
            },
        }
        limits = resolve_limits(candidates, limits_cfg)
        weeks = [pd.Timestamp("2026-01-13")]
        base = {
            pd.Timestamp("2026-01-13").to_period("W-MON").start_time: {
                "facility": {"C1": 200.0},
                "customer": {"U1": 200.0},
            }
        }

        with self.assertRaises(RuntimeError) as err:
            optimize_multi_week(
                candidates,
                limits=limits,
                week_starts=weeks,
                base_weekly_exposure=base,
                settings=MultiWeekOptimizerSettings(horizon_weeks=1, attempt_cap=1),
            )

        message = str(err.exception)
        self.assertIn("INFEASIBLE", message)
        self.assertIn("base exposure exceeds limits", message)


if __name__ == "__main__":
    unittest.main()
