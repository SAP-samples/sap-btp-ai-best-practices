"""RPT-1 lifetime estimator: reconciliation-offset target, customer context and margin."""
import unittest
from unittest.mock import patch

import pandas as pd

from app.optimizer.model.lifetime_estimation import (
    LifetimeEstimationConfig,
    _prepare_history_features,
    estimate_candidate_lifetime_with_rpt1,
    select_context,
)


def _history(customer_rows):
    """Observed lifecycles: due Tue 2025-03-04, funded Wed 2025-02-05, released on file k at 12:00.

    Args:
        customer_rows: {customer: [k, k, ...]} one entry per history invoice.
    """
    rows = []
    for customer, offsets in customer_rows.items():
        for position, k in enumerate(offsets):
            release = pd.Timestamp("2025-03-04 12:00") + pd.Timedelta(weeks=k)
            rows.append({"Invoice Reference": f"H-{customer}-{position}", "Company Code": "C1",
                         "Customer": customer, "Due Date": pd.Timestamp("2025-03-04"),
                         "credit_start": pd.Timestamp("2025-02-05 15:00") + pd.Timedelta(days=position),
                         "credit_release": release})
    return pd.DataFrame(rows)


def _candidates(customers):
    """Candidates due Tue 2026-03-03 with a planned funding Wednesday of 2026-02-04."""
    return pd.DataFrame({"Invoice Reference": [f"INV-{i}" for i in range(len(customers))],
                         "Company Code": ["C1"] * len(customers), "Customer": customers,
                         "Due Date": [pd.Timestamp("2026-03-03")] * len(customers),
                         "Planned Funding Date": [pd.Timestamp("2026-02-04")] * len(customers)})


class _FakeClient:
    """Stands in for RPT1Client; predicts a fixed k and records every context it receives."""
    predicted_k = 1.0
    contexts = []

    @classmethod
    def from_env(cls, **kwargs):
        return cls()

    def fit(self, **kwargs):
        _FakeClient.contexts.append(kwargs["context_df"])
        return self

    def predict(self, query_df):
        result = type("Result", (), {})()
        result.predictions_df = pd.DataFrame({"ROW_ID": query_df["ROW_ID"],
                                              "TARGET_RECON_K": [self.predicted_k] * len(query_df)})
        result.metadata = {"rpt1_usage": {"input_cells": 10, "prediction_count": len(query_df)}}
        return result


def _run(candidates, history, **config):
    _FakeClient.contexts = []
    with patch("app.optimizer.model.lifetime_estimation._load_rpt1_client_class", return_value=_FakeClient):
        return estimate_candidate_lifetime_with_rpt1(candidates, history, config=LifetimeEstimationConfig(**config))


class HistoryFeatureTests(unittest.TestCase):
    def test_offset_target_and_as_of_priors(self):
        """k is read from the Tuesday calendar; priors never use outcomes known after funding."""
        history = _prepare_history_features(_history({"U1": [1, 2, 1]}))
        self.assertEqual(sorted(history["TARGET_RECON_K"].tolist()), [1, 1, 2])
        # All three were funded before any release, so no earlier outcome was known.
        self.assertEqual(history["CUST_HISTORY_N"].tolist(), [0, 0, 0])
        self.assertTrue(history["CUST_MEDIAN_K"].isna().all())
        self.assertTrue(history["CUSTOMER_ID"].str.startswith("CUST_").all())


class ContextSelectionTests(unittest.TestCase):
    def test_customer_only_when_history_is_sufficient(self):
        history = _prepare_history_features(_history({"U1": [1] * 30, "U2": [2] * 30}))
        context = select_context(history, "U1", "C1", min_rows=10, max_rows=800)
        self.assertEqual(len(context), 30)
        self.assertTrue((context["CUSTOMER_ID"] == "CUST_U1").all())

    def test_sparse_customer_is_topped_up_and_max_is_respected(self):
        history = _prepare_history_features(_history({"U1": [1] * 3, "U2": [2] * 30}))
        self.assertEqual(len(select_context(history, "U1", "C1", min_rows=10, max_rows=800)), 10)
        self.assertEqual(len(select_context(history, "U2", "C1", min_rows=10, max_rows=12)), 12)


class EstimatorTests(unittest.TestCase):
    def test_disabled_estimator_is_safe_noop(self):
        output, report = estimate_candidate_lifetime_with_rpt1(
            _candidates(["U1"]), pd.DataFrame(), config=LifetimeEstimationConfig(enabled=False))
        self.assertEqual(report["status"], "disabled")
        self.assertEqual(len(output), 1)

    def test_margin_raises_k_to_the_customer_quantile(self):
        """Model k=1; customer p75 k=2 -> release Tue 2026-03-17 12:00, 42 days, 6 weeks."""
        output, report = _run(_candidates(["U1"]), _history({"U1": [1] * 10 + [2] * 10}))
        row = output.iloc[0]
        self.assertEqual(report["status"], "completed")
        self.assertEqual((row.expected_recon_k_model, row.expected_recon_k), (1, 2))
        self.assertEqual(row.expected_release_date, "2026-03-17")
        self.assertEqual((row.expected_lifetime_days, row.expected_lifetime_weeks), (42, 6))
        self.assertEqual(row.expected_lifetime_source, "RPT-1")
        self.assertEqual(report["margin_applied_candidates"], 1)

    def test_without_margin_the_model_k_is_used(self):
        output, _ = _run(_candidates(["U1"]), _history({"U1": [1] * 10 + [2] * 10}), release_margin_quantile=None)
        self.assertEqual((output.iloc[0].expected_recon_k, output.iloc[0].expected_lifetime_days), (1, 35))

    def test_one_customer_context_per_batch_and_progress_counters(self):
        customers = [f"U{i % 5}" for i in range(30)]
        history = _history({f"U{i}": [1] * 20 for i in range(5)})
        events = []
        with patch("app.optimizer.model.lifetime_estimation._load_rpt1_client_class", return_value=_FakeClient):
            _FakeClient.contexts = []
            output, report = estimate_candidate_lifetime_with_rpt1(
                _candidates(customers), history,
                config=LifetimeEstimationConfig(query_batch_size=10, context_min_rows=5, max_parallel_calls=2),
                progress_callback=events.append)
        self.assertEqual((report["batches_total"], report["api_calls"], report["predicted_candidates"]), (5, 5, 30))
        self.assertEqual(report["fallback_candidates"], 0)
        self.assertTrue(all(context["CUSTOMER_ID"].nunique() == 1 for context in _FakeClient.contexts))
        self.assertTrue((output["expected_lifetime_source"] == "RPT-1").all())
        self.assertTrue(any(event["batches_completed"] == 5 for event in events))

    def test_missing_funding_date_falls_back_to_four_weeks(self):
        candidates = _candidates(["U1"]).drop(columns="Planned Funding Date")
        output, report = _run(candidates, _history({"U1": [1] * 10}))
        self.assertEqual(output.iloc[0].expected_lifetime_source, "fallback_default_weeks")
        self.assertEqual(output.iloc[0].expected_lifetime_days, 28)
        self.assertEqual(report["fallback_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
