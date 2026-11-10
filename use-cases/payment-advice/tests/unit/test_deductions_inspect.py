# tests/unit/test_deductions_inspect.py
import unittest
from app.payment_advice.deductions import inspect_deductions

PAYLOAD = {"line_items": [
    {"invoice_reference": "A", "gross_amount": 100.0, "net_amount": 90.0},
    {"invoice_reference": "B", "gross_amount": -50.0, "net_amount": -50.0},
    {"invoice_reference": "C", "gross_amount": 10.0, "net_amount": -1.0},
]}

class InspectTest(unittest.TestCase):
    def test_surfaces_negatives_by_sign_only(self):
        out = inspect_deductions(PAYLOAD, positive_sample=1)
        refs = {l["invoice_reference"] for l in out["negative_lines"]}
        self.assertEqual(refs, {"B", "C"})
        self.assertEqual(out["counts"], {"negative": 2, "positive": 1, "total": 3})
        self.assertEqual(len(out["positive_sample"]), 1)

    def test_non_numeric_and_absent_amounts_treated_as_positive(self):
        # Line D: amounts present but non-numeric (None / string) — should NOT trigger
        # the isinstance guard, so must be classified as positive.
        # Line E: both amount keys entirely absent — same expectation.
        payload = {"line_items": [
            {"invoice_reference": "D", "gross_amount": None, "net_amount": "N/A"},
            {"invoice_reference": "E"},
        ]}
        out = inspect_deductions(payload)
        self.assertEqual(out["negative_lines"], [],
                         "Non-numeric / absent amounts must not produce negative lines")
        refs = {l["invoice_reference"] for l in out["positive_sample"]}
        self.assertEqual(refs, {"D", "E"})
        self.assertEqual(out["counts"], {"negative": 0, "positive": 2, "total": 2})
