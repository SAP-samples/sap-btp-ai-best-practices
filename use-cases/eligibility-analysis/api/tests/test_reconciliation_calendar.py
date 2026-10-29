"""Weekly reconciliation calendar arithmetic and leakage-free customer statistics."""
import unittest

import pandas as pd

from app.optimizer.model.reconciliation_calendar import (
    as_of_customer_priors,
    customer_margin_offsets,
    first_reconciliation_on_or_after,
    planned_funding_dates,
    reconciliation_offset,
    release_timestamp,
    to_naive,
)


class CalendarTests(unittest.TestCase):
    def test_tuesday_files_and_offsets(self):
        due = to_naive(["2025-12-02", "2025-12-03", "2025-12-08"])  # Tue, Wed, Mon
        self.assertEqual(first_reconciliation_on_or_after(due).dt.strftime("%Y-%m-%d").tolist(),
                         ["2025-12-02", "2025-12-09", "2025-12-09"])
        released = to_naive(["2025-12-09 13:00", "2025-12-09 09:00", "2025-12-23 10:00"])
        self.assertEqual(reconciliation_offset(due, released).tolist(), [1, 0, 2])
        self.assertEqual(release_timestamp(due, pd.Series([1, 0, 2]), 0.5).dt.strftime("%m-%d %H").tolist(),
                         ["12-09 12", "12-09 12", "12-23 12"])

    def test_planned_funding_is_the_wednesday_of_the_first_allowed_week(self):
        """Offer mid-week -> next Monday's week; Monday midnight offer -> same week; none -> planning start."""
        funding = planned_funding_dates(pd.Series(["2025-11-26 10:00", "2025-12-01 00:00", None, "2025-11-01"]),
                                        "2025-11-24")
        self.assertEqual(funding.dt.strftime("%Y-%m-%d").tolist(),
                         ["2025-12-03", "2025-12-03", "2025-11-26", "2025-11-26"])

    def test_priors_use_only_outcomes_known_before_funding(self):
        customers = pd.Series(["A", "A", "A", "B"])
        funded = to_naive(["2025-01-01", "2025-02-01", "2025-03-01", "2025-03-01"])
        known = to_naive(["2025-01-20", "2025-02-25", "2025-04-01", "2025-04-01"])
        median, count = as_of_customer_priors(customers, funded, known, pd.Series([1.0, 3.0, 5.0, 2.0]))
        self.assertEqual(count.tolist(), [0, 1, 2, 0])
        self.assertTrue(pd.isna(median.iloc[0]))
        self.assertEqual(median.iloc[1:3].tolist(), [1.0, 2.0])

    def test_margin_requires_enough_history_and_can_be_disabled(self):
        customers = pd.Series(["A"] * 8 + ["B"] * 2)
        k = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 5, 5])
        self.assertEqual(customer_margin_offsets(customers, k, 0.75, 5), {"A": 2})
        self.assertEqual(customer_margin_offsets(customers, k, None, 5), {})


if __name__ == "__main__":
    unittest.main()
