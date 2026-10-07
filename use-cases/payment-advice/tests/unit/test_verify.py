"""
Unit tests for deterministic verification (offline).

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice.verify import verify_canonical  # noqa: E402


class Verify(unittest.TestCase):
    def _header(self, **overrides):
        base = {"payee_name": "Example Lighting", "payment_reference": "R1", "payment_amount": 100.0}
        base.update(overrides)
        return base

    def test_clean_payload_high_confidence(self) -> None:
        lines = [{"invoice_reference": "A", "net_amount": 60.0}, {"invoice_reference": "B", "net_amount": 40.0}]
        result = verify_canonical(self._header(), lines)
        self.assertFalse(result.needs_review)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.issues, [])

    def test_missing_required_field_needs_review(self) -> None:
        header = self._header()
        del header["payment_reference"]
        result = verify_canonical(header, [{"net_amount": 100.0}])
        self.assertTrue(result.needs_review)
        self.assertTrue(any("payment_reference" in i for i in result.issues))

    def test_reconciliation_mismatch_warns(self) -> None:
        lines = [{"invoice_reference": "A", "net_amount": 60.0}]  # total 60 != 100
        result = verify_canonical(self._header(), lines)
        self.assertTrue(any("reconcile" in w for w in result.warnings))
        self.assertLess(result.confidence, 1.0)

    def test_reconciliation_within_tolerance_ok(self) -> None:
        lines = [{"net_amount": 99.999}]
        result = verify_canonical(self._header(payment_amount=100.0), lines)
        self.assertEqual([w for w in result.warnings if "reconcile" in w], [])

    def test_suspected_duplicates_flagged(self) -> None:
        lines = [
            {"invoice_reference": "X", "net_amount": -5.0, "gross_amount": -5.0},
            {"invoice_reference": "X", "net_amount": -5.0, "gross_amount": -5.0},
        ]
        result = verify_canonical(self._header(payment_amount=-10.0), lines)
        self.assertEqual(result.suspected_duplicate_groups, [[0, 1]])
        self.assertTrue(any("duplicate" in w for w in result.warnings))

    def test_empty_line_items_warns(self) -> None:
        result = verify_canonical(self._header(), [])
        self.assertTrue(any("no line items" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
