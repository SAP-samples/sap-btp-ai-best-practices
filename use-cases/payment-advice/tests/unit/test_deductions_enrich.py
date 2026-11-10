# tests/unit/test_deductions_enrich.py
import unittest
from app.payment_advice.deductions import enrich_payload

PAYLOAD = {"header": {}, "line_items": [
    {"invoice_reference": "A", "gross_amount": 100.0},
    {"invoice_reference": "P1", "gross_amount": -10.0},
]}

class EnrichTest(unittest.TestCase):
    def test_applies_interpretations_and_defaults(self):
        # interpretations is an index-aligned list: element 0 covers line 0 ("A"),
        # element 1 is an empty dict so line 1 ("P1") receives safe defaults.
        interp = [
            {"document_nature": "invoice", "reason_code": None},
            {},
        ]
        out = enrich_payload(PAYLOAD, interp, client_key="fabrikam", playbook_revision=2)
        a = out["line_items"][0]; p = out["line_items"][1]
        self.assertEqual(a["document_nature"], "invoice")
        self.assertEqual(a["reason_code"], None)
        self.assertEqual(p["document_nature"], "unknown")        # no interpretation given
        self.assertEqual(p["reason_code"], "323")
        self.assertIn("reason-code-defaulted", p["flags"])
        self.assertEqual(out["header"]["interpretation"]["client_key"], "fabrikam")
        self.assertEqual(out["header"]["interpretation"]["playbook_revision"], 2)
        self.assertEqual(out["header"]["interpretation"]["unresolved_count"], 1)
        self.assertNotIn("document_nature", PAYLOAD["line_items"][0])  # input unmutated
        self.assertNotIn("rationale", a)  # plain line with no rationale in interp must not gain the key

    def test_rationale_passthrough(self):
        from app.payment_advice.deductions import enrich_payload
        payload = {"header": {}, "line_items": [{"invoice_reference": "P", "gross_amount": -3.0}]}
        interp = [{"document_nature": "chargeback", "reason_code": "R1", "rationale": "explained"}]
        out = enrich_payload(payload, interp, client_key="c", playbook_revision=1)
        self.assertEqual(out["line_items"][0]["rationale"], "explained")
