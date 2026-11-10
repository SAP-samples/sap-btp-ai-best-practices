# tests/unit/test_deductions_select.py
"""Tests for the deterministic split and assembly helpers: needs_analysis,
select_analysis_lines, assemble_interpretations.

These helpers are the UC-02 "selective-analysis" path that routes only
anomalous lines to the agent while resolving plain invoice lines locally.
"""
import unittest
from app.payment_advice.deductions import needs_analysis, select_analysis_lines, assemble_interpretations


class SelectTest(unittest.TestCase):
    def test_needs_analysis_rules(self):
        self.assertTrue(needs_analysis({"gross_amount": -1.0, "net_amount": -1.0}))
        self.assertTrue(needs_analysis({"gross_amount": 5.0, "net_amount": -1.0}))
        self.assertTrue(needs_analysis({"gross_amount": 5.0, "net_amount": 5.0, "deduction_reason": "PRICE DIFF [0100]"}))
        self.assertFalse(needs_analysis({"gross_amount": 5.0, "net_amount": 4.0}))          # positive+allowance = plain
        self.assertFalse(needs_analysis({"gross_amount": 5.0, "net_amount": 4.0, "deduction_reason": "   "}))
        self.assertFalse(needs_analysis({"gross_amount": 5.0}))

    def test_select_tags_line_index_and_filters(self):
        payload = {"line_items": [
            {"invoice_reference": "A", "gross_amount": 10.0, "net_amount": 9.0},   # plain
            {"invoice_reference": "P", "gross_amount": -3.0, "net_amount": -3.0},  # analysis
        ]}
        sel = select_analysis_lines(payload)
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["line_index"], 1)
        self.assertEqual(sel[0]["invoice_reference"], "P")

    def test_assemble_plain_matched_and_unmapped(self):
        payload = {"line_items": [
            {"invoice_reference": "A", "gross_amount": 10.0, "net_amount": 9.0},   # index 0 plain
            {"invoice_reference": "P", "gross_amount": -3.0, "net_amount": -3.0},  # index 1 analysis, matched
            {"invoice_reference": "Q", "gross_amount": -4.0, "net_amount": -4.0},  # index 2 analysis, unmapped
        ]}
        agent_entries = [{"line_index": 1, "document_nature": "chargeback", "reason_code": "R1", "rationale": "neg gross"}]
        out = assemble_interpretations(payload, agent_entries)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], {"document_nature": "invoice", "reason_code": None, "residual_items": [], "flags": []})
        self.assertEqual(out[1]["document_nature"], "chargeback")
        self.assertEqual(out[1]["reason_code"], "R1")
        self.assertNotIn("line_index", out[1])
        self.assertEqual(out[2]["reason_code"], "323")
        self.assertIn("needs-customer-confirmation", out[2]["flags"])
        self.assertIn("rationale", out[2])

    def test_plain_entries_are_independent(self):
        """Mutating one plain entry's lists must not affect sibling entries or the module constant."""
        from app.payment_advice import deductions as _mod
        payload = {"line_items": [
            {"invoice_reference": "A", "gross_amount": 10.0, "net_amount": 9.0},   # plain
            {"invoice_reference": "B", "gross_amount": 20.0, "net_amount": 18.0},  # plain
        ]}
        out = assemble_interpretations(payload, [])
        # Mutate the first entry's flags list
        out[0]["flags"].append("probe")
        # The second plain entry must be unaffected
        self.assertEqual(out[1]["flags"], [])
        # If the module-level constant still exists it must also be unaffected
        if hasattr(_mod, "_PLAIN_INTERP"):
            self.assertEqual(_mod._PLAIN_INTERP["flags"], [])
