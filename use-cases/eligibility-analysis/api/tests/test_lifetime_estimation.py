import unittest
from unittest.mock import patch

import pandas as pd

from app.optimizer.model.lifetime_estimation import (
    LifetimeEstimationConfig,
    build_lifetime_payload,
    estimate_candidate_lifetime_with_rpt1,
)


class TestLifetimePayloadBuilder(unittest.TestCase):
    def test_context_size_respects_bounds(self) -> None:
        row_count = 1200
        history_df = pd.DataFrame(
            {
                "INVOICE_REF": [f"H-{i}" for i in range(row_count)],
                "COMPANY_CODE": ["C1" if i % 2 == 0 else "C2" for i in range(row_count)],
                "CUSTOMER_ID": ["U1" if i % 5 == 0 else f"U{i % 17}" for i in range(row_count)],
                "PROGRAM_ID": ["P1" if i % 7 == 0 else "P2" for i in range(row_count)],
                "FUNDING_CURRENCY": ["EUR"] * row_count,
                "ORIGINAL_CURRENCY": ["EUR"] * row_count,
                "PURCHASE_PRICE": [1000.0 + (i % 13) * 10.0 for i in range(row_count)],
                "INVOICE_AMOUNT": [1200.0 + (i % 11) * 10.0 for i in range(row_count)],
                "ISSUANCE_DATE": pd.to_datetime(["2025-01-01"] * row_count),
                "DUE_DATE": pd.to_datetime(["2025-03-01"] * row_count),
                "TENOR_DAYS": [59] * row_count,
                "TARGET_LIFETIME_DAYS": [30 + (i % 20) for i in range(row_count)],
                "CREDIT_START_DATE": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            }
        )
        candidate_row = pd.Series(
            {
                "INVOICE_REF": "NEW-1",
                "COMPANY_CODE": "C1",
                "CUSTOMER_ID": "U1",
                "PROGRAM_ID": "P1",
                "FUNDING_CURRENCY": "EUR",
                "ORIGINAL_CURRENCY": "EUR",
                "PURCHASE_PRICE": 1050.0,
                "INVOICE_AMOUNT": 1200.0,
                "ISSUANCE_DATE": pd.Timestamp("2026-02-01"),
                "DUE_DATE": pd.Timestamp("2026-03-15"),
                "TENOR_DAYS": 42,
            }
        )

        context_df, query_df = build_lifetime_payload(
            history_df,
            candidate_row,
            context_min_rows=500,
            context_max_rows=800,
        )

        self.assertGreaterEqual(len(context_df), 500)
        self.assertLessEqual(len(context_df), 800)
        self.assertEqual(len(query_df), 1)
        self.assertIn("TARGET_LIFETIME_DAYS", context_df.columns)
        self.assertNotIn("TARGET_LIFETIME_DAYS", query_df.columns)


class TestLifetimeEstimatorFallback(unittest.TestCase):
    def test_disabled_estimator_is_safe_noop(self) -> None:
        candidates_df = pd.DataFrame(
            {
                "Invoice Reference": ["INV-1"],
                "Company Code": ["C1"],
                "Customer": ["U1"],
                "Purchase Price": [1000.0],
                "Due Date": [pd.Timestamp("2026-03-15")],
            }
        )
        lifecycle_df = pd.DataFrame()
        config = LifetimeEstimationConfig(enabled=False)

        output_df, report = estimate_candidate_lifetime_with_rpt1(
            candidates_df,
            lifecycle_df,
            config=config,
        )

        self.assertEqual(report["status"], "disabled")
        self.assertEqual(report["predicted_candidates"], 0)
        self.assertEqual(len(output_df), len(candidates_df))

    def test_parallel_estimation_reports_progress_counters(self) -> None:
        candidates_df = pd.DataFrame(
            {
                "Invoice Reference": [f"INV-{i}" for i in range(30)],
                "Company Code": ["C1"] * 30,
                "Customer": [f"U{i % 5}" for i in range(30)],
                "PROGRAMA": ["P1"] * 30,
                "Funding Currency": ["EUR"] * 30,
                "Currency": ["EUR"] * 30,
                "Purchase Price": [1000.0 + i for i in range(30)],
                "Amount": [1200.0 + i for i in range(30)],
                "Issuance date": [pd.Timestamp("2026-01-01")] * 30,
                "Due Date": [pd.Timestamp("2026-03-01")] * 30,
            }
        )
        lifecycle_df = pd.DataFrame(
            {
                "Invoice Reference": [f"H-{i}" for i in range(900)],
                "Company Code": ["C1"] * 900,
                "Customer": [f"U{i % 7}" for i in range(900)],
                "PROGRAMA": ["P1"] * 900,
                "Funding Currency": ["EUR"] * 900,
                "Currency": ["EUR"] * 900,
                "Purchase Price": [950.0 + (i % 30) for i in range(900)],
                "Amount": [1150.0 + (i % 30) for i in range(900)],
                "Issuance date": [pd.Timestamp("2025-01-01")] * 900,
                "Due Date": [pd.Timestamp("2025-03-01")] * 900,
                "credit_start": pd.date_range("2024-01-01", periods=900, freq="D"),
                "credit_duration_days": [30 + (i % 10) for i in range(900)],
            }
        )

        class _FakePredictionResult:
            def __init__(self, query_df: pd.DataFrame):
                self.predictions_df = pd.DataFrame(
                    {
                        "INVOICE_REF": query_df["INVOICE_REF"].tolist(),
                        "TARGET_LIFETIME_DAYS": [42] * len(query_df),
                        "TARGET_LIFETIME_DAYS__confidence": [0.8] * len(query_df),
                    }
                )

        class _FakeClient:
            @classmethod
            def from_env(cls, **kwargs):
                return cls()

            def fit(self, **kwargs):
                return self

            def predict(self, query_df):
                return _FakePredictionResult(query_df)

        progress_events = []
        config = LifetimeEstimationConfig(
            enabled=True,
            query_batch_size=10,
            context_min_rows=100,
            context_max_rows=150,
            max_parallel_calls=2,
        )

        with patch(
            "app.optimizer.model.lifetime_estimation._load_rpt1_client_class",
            return_value=_FakeClient,
        ):
            output_df, report = estimate_candidate_lifetime_with_rpt1(
                candidates_df,
                lifecycle_df,
                config=config,
                progress_callback=lambda payload: progress_events.append(payload),
            )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["predicted_candidates"], 30)
        self.assertEqual(report["fallback_candidates"], 0)
        self.assertEqual(report["max_parallel_calls"], 2)
        self.assertEqual(report["batches_total"], 5)
        self.assertEqual(report["batches_completed"], 5)
        self.assertEqual(report["api_calls"], 5)
        self.assertTrue((output_df["expected_lifetime_source"] == "RPT-1").all())
        self.assertTrue(len(progress_events) >= 2)
        self.assertTrue(
            any(event.get("batches_completed") == report["batches_completed"] for event in progress_events)
        )


if __name__ == "__main__":
    unittest.main()
