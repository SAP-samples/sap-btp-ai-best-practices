# tests/unit/test_deductions_docinfo.py
"""Tests for the three deterministic full-document functions added in UC-02.

Covers:
  - document_statistics  (aggregate counts, totals, currency, distinct refs)
  - list_invoice_references  (ordered flat list with pagination)
  - fetch_invoices  (str-or-list lookup with not_found reporting)
"""
import unittest
from app.payment_advice.deductions import document_statistics, list_invoice_references, fetch_invoices

PAYLOAD = {"header": {"payment_currency": "CAD"}, "line_items": [
    {"invoice_reference": "A", "gross_amount": 10.0, "net_amount": 9.0, "customer_account_reference": "SDRBG"},
    {"invoice_reference": "B", "gross_amount": -5.0, "net_amount": -5.0},
    {"invoice_reference": "C", "gross_amount": 3.0, "net_amount": 3.0, "deduction_reason": "X [1]", "customer_account_reference": "SDRBG"},
]}

class DocInfoTest(unittest.TestCase):
    def test_statistics(self):
        s = document_statistics(PAYLOAD)
        self.assertEqual(s["total_lines"], 3)
        self.assertEqual(s["anomalous_count"], 2)      # B (neg) + C (has reason)
        self.assertEqual(s["plain_count"], 1)
        self.assertEqual(s["negative_count"], 1)       # B
        self.assertEqual(s["with_deduction_reason_count"], 1)  # C
        self.assertEqual(s["currency"], "CAD")
        self.assertEqual(s["distinct_customer_account_references"], ["SDRBG"])
        self.assertAlmostEqual(s["gross_total"], 8.0)
        self.assertAlmostEqual(s["net_total"], 7.0)    # 9.0 + (-5.0) + 3.0

    def test_statistics_no_header(self):
        """document_statistics must not crash and must return currency=None when header is absent."""
        payload_no_header = {"line_items": [
            {"invoice_reference": "X", "gross_amount": 1.0, "net_amount": 1.0},
        ]}
        s = document_statistics(payload_no_header)
        self.assertIsNone(s["currency"])
        self.assertEqual(s["total_lines"], 1)

        payload_none_header = {"header": None, "line_items": []}
        s2 = document_statistics(payload_none_header)
        self.assertIsNone(s2["currency"])

    def test_list_references_pagination(self):
        out = list_invoice_references(PAYLOAD, offset=0, limit=2)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["references"], ["A", "B"])
        self.assertTrue(out["has_more"])
        out2 = list_invoice_references(PAYLOAD, offset=2, limit=2)
        self.assertEqual(out2["references"], ["C"])
        self.assertFalse(out2["has_more"])

    def test_fetch_invoices_str_and_list(self):
        one = fetch_invoices(PAYLOAD, "A")
        self.assertEqual(len(one["invoices"]), 1)
        self.assertEqual(one["invoices"][0]["line_index"], 0)
        multi = fetch_invoices(PAYLOAD, ["B", "ZZ"])
        self.assertEqual([i["invoice_reference"] for i in multi["invoices"]], ["B"])
        self.assertEqual(multi["not_found"], ["ZZ"])
