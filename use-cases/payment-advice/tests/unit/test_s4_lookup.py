"""Offline checks for invoice-reference lookup against S/4 customer items."""
import unittest
from decimal import Decimal

from app.s4.lookup import CHUNK, find_receivables


def row(ref="8700276154", doc="1400000001", customer="0010053628", amount="10596.28", cleared=False, **extra):
    """Build one A_OperationalAcctgDocItemCube row."""
    return {"CompanyCode": "CA01", "FiscalYear": "2025", "AccountingDocument": doc, "AccountingDocumentItem": "001",
            "Customer": customer, "DocumentReferenceID": ref, "BillingDocument": "", "AmountInTransactionCurrency": amount,
            "TransactionCurrency": "CAD", "IsCleared": cleared, "ClearingAccountingDocument": "1500000009" if cleared else "",
            **extra}


class FakeClient:
    """Answers each cube query with the rows whose field value appears in the filter."""
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def get_json(self, service, path, params):
        self.calls.append(params["$filter"])
        field = next(f for f in ("DocumentReferenceID", "AccountingDocument", "BillingDocument") if f + " eq" in params["$filter"])
        return {"d": {"results": [r for r in self.rows if r[field] and f"{field} eq '{r[field]}'" in params["$filter"]]}}


class LookupTests(unittest.TestCase):
    """References resolve in field order; only unmatched ones fall through."""
    def test_reference_resolves_by_xblnr_with_company_code_and_customer(self):
        found = find_receivables(FakeClient([row()]), [" 8700276154 ", "924610168612UL"])
        item = found["8700276154"][0]
        self.assertEqual((item.company_code, item.customer, item.matched_by), ("CA01", "0010053628", "DocumentReferenceID"))
        self.assertEqual(item.amount, Decimal("10596.28"))
        self.assertEqual(found["924610168612UL"], [])

    def test_unpadded_s4_customer_is_padded_like_rule_accounts(self):
        found = find_receivables(FakeClient([row(customer="10053628")]), ["8700276154"])  # some systems return no zeros
        self.assertEqual(found["8700276154"][0].customer, "0010053628")

    def test_unmatched_numeric_reference_falls_back_to_padded_accounting_document(self):
        client = FakeClient([row(ref="", doc="0012345678")])
        trace = []
        found = find_receivables(client, ["12345678"], trace=trace)
        self.assertEqual(found["12345678"][0].matched_by, "AccountingDocument")
        self.assertIn("AccountingDocument eq '0012345678'", client.calls[1])
        self.assertEqual([t["field"] for t in trace], ["DocumentReferenceID", "AccountingDocument"])

    def test_non_numeric_references_never_query_document_number_fields(self):
        client = FakeClient([])
        find_receivables(client, ["REPLEN COMP JULY", "372290-202618"])
        self.assertEqual(len(client.calls), 1)

    def test_ledger_duplicates_collapse_and_cleared_items_are_flagged(self):
        found = find_receivables(FakeClient([row(cleared=True), row(cleared=True)]), ["8700276154"])
        self.assertEqual(len(found["8700276154"]), 1)
        self.assertTrue(found["8700276154"][0].is_cleared)

    def test_many_references_are_chunked(self):
        client = FakeClient([])
        find_receivables(client, [f"REF{i}" for i in range(CHUNK + 1)], company_code="CA01")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("CompanyCode eq 'CA01'", client.calls[0])


if __name__ == "__main__":
    unittest.main()
