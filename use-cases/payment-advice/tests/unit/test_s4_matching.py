"""Offline checks for the S/4 reads of the check: fields by need, other spellings, linked items."""
import unittest
from decimal import Decimal

from app.s4.matching import collect_candidates
from tests.unit.test_s4_lookup import FakeClient, row


def advice(*lines, **header):
    """Advice result with the given lines."""
    return {"header": header, "line_items": [{"document_nature": "invoice", "net_amount": 10, **line} for line in lines]}


def fields(client):
    """The S/4 field each query of the client searched."""
    return [next(f for f in FakeClient.FIELDS if f + " eq" in call) for call in client.calls]


class TierTests(unittest.TestCase):
    """A reference asks further fields only while it has no payable item in the advice's scope."""
    def test_open_item_in_scope_stops_at_the_reference_field(self):
        client = FakeClient([row(ref="1800000102", doc="1400000001")])
        found = collect_candidates(client, advice({"invoice_reference": "1800000102"}), {})
        self.assertEqual((len(found["1800000102"]), fields(client)), (1, ["DocumentReferenceID", "InvoiceReference"]))

    def test_cleared_reference_hit_does_not_hide_the_open_document_number(self):
        client = FakeClient([row(ref="1800000102", doc="1400000001", cleared=True), row(ref="", doc="1800000102")])
        found = collect_candidates(client, advice({"invoice_reference": "1800000102"}), {})
        self.assertEqual([(i.matched_by, i.is_cleared) for i in found["1800000102"]],
                         [("DocumentReferenceID", True), ("AccountingDocument", False)])

    def test_hit_outside_the_rule_company_code_or_customer_keeps_searching(self):
        rows = [row(ref="1800000102", doc="1400000001", CompanyCode="CA02"), row(ref="", doc="1800000102")]
        client = FakeClient(rows)
        collect_candidates(client, advice({"invoice_reference": "1800000102"}, company_code="ca01"), {})
        self.assertIn("AccountingDocument", fields(client))
        client = FakeClient(rows)
        collect_candidates(client, advice({"invoice_reference": "1800000102", "customer_account": "10099999"}), {})
        self.assertIn("AccountingDocument", fields(client))
        client = FakeClient(rows)  # the demo alias maps the rule code to the code S/4 uses
        collect_candidates(client, advice({"invoice_reference": "1800000102"}, company_code="XX01"), {"XX01": "CA02"})
        self.assertNotIn("AccountingDocument", fields(client))

    def test_excluded_rows_and_empty_references_are_not_looked_up(self):
        client = FakeClient([])
        found = collect_candidates(client, advice({"invoice_reference": "A1", "excluded": True}, {"invoice_reference": ""}), {})
        self.assertEqual((found, client.calls), ({}, []))


class VariantTests(unittest.TestCase):
    """Unknown invoice references are tried in other spellings; deductions are not."""
    def test_unknown_invoice_reference_is_found_without_its_separator(self):
        client = FakeClient([row(ref="INV0012")])
        found = collect_candidates(client, advice({"invoice_reference": "inv-0012"}), {})
        hit, = found["INV-0012"]
        self.assertEqual((hit.variant, hit.reference, hit.document_reference), ("INV0012", "INV-0012", "INV0012"))

    def test_unknown_deduction_reference_is_not_tried_in_other_spellings(self):
        client = FakeClient([row(ref="DED0012")])
        found = collect_candidates(client, advice({"invoice_reference": "DED-0012", "document_nature": "chargeback",
                                                   "net_amount": -5}), {})
        self.assertEqual((found["DED-0012"], len(client.calls)), ([], 1))


class LinkedTests(unittest.TestCase):
    """Open balances come from the items that refer to the found document."""
    def test_partial_payment_reduces_the_open_amount_of_the_candidate(self):
        partial = row(ref="PAY-1", doc="1500000001", amount="-400", InvoiceReference="1400000001",
                      InvoiceReferenceFiscalYear="2025", FollowOnDocumentType="Z")
        found = collect_candidates(FakeClient([row(), partial]), advice({"invoice_reference": "8700276154"}), {})
        self.assertEqual(found["8700276154"][0].open_amount, Decimal("10196.28"))


if __name__ == "__main__":
    unittest.main()
