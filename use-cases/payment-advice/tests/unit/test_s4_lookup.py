"""Offline checks for invoice-reference lookup against S/4 customer items."""
import unittest
from decimal import Decimal

from app.s4.lookup import find_linked_items, find_receivables
from app.s4.odata import MAX_TERMS


def row(ref="8700276154", doc="1400000001", customer="0010053628", amount="10596.28", cleared=False, **extra):
    """Build one A_OperationalAcctgDocItemCube row."""
    return {"CompanyCode": "CA01", "FiscalYear": "2025", "AccountingDocument": doc, "AccountingDocumentItem": "001",
            "Customer": customer, "DocumentReferenceID": ref, "BillingDocument": "", "AmountInTransactionCurrency": amount,
            "TransactionCurrency": "CAD", "IsCleared": cleared, "ClearingAccountingDocument": "1500000009" if cleared else "",
            "InvoiceReference": "", **extra}


class FakeClient:
    """Answers each cube query with the rows whose field value appears in the filter."""
    FIELDS = ("InvoiceReference", "DocumentReferenceID", "AccountingDocument", "BillingDocument")

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def get_json(self, service, path, params):
        self.calls.append(params["$filter"])
        field = next(f for f in self.FIELDS if f + " eq" in params["$filter"])
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

    def test_reference_unknown_as_xblnr_and_belnr_resolves_by_billing_document(self):
        client = FakeClient([row(ref="", doc="1400000077", BillingDocument="0090001234")])
        found = find_receivables(client, ["90001234"])
        item, = found["90001234"]
        self.assertEqual((item.matched_by, item.accounting_document), ("BillingDocument", "1400000077"))
        self.assertEqual(len(client.calls), 3)

    def test_non_numeric_references_never_query_document_number_fields(self):
        client = FakeClient([])
        find_receivables(client, ["REPLEN COMP JULY", "372290-202618"])
        self.assertEqual(len(client.calls), 1)

    def test_ledger_duplicates_collapse_and_cleared_items_are_flagged(self):
        found = find_receivables(FakeClient([row(cleared=True), row(cleared=True)]), ["8700276154"])
        self.assertEqual(len(found["8700276154"]), 1)
        self.assertTrue(found["8700276154"][0].is_cleared)

    def test_reversal_document_is_skipped_and_reversed_original_is_flagged(self):
        original = row(cleared=True, IsReversed=True, ReverseDocument="1800000028")
        reversal = row(doc="1800000028", amount="-10596.28", cleared=True, IsReversal=True)
        item, = find_receivables(FakeClient([original, reversal]), ["8700276154"])["8700276154"]
        self.assertEqual((item.accounting_document, item.is_reversed, item.reverse_document),
                         ("1400000001", True, "1800000028"))

    def test_reposted_invoice_wins_over_its_reversed_original(self):
        original = row(cleared=True, IsReversed=True, ReverseDocument="1800000028")
        reposted = row(doc="1800000030")
        item, = find_receivables(FakeClient([original, reposted]), ["8700276154"])["8700276154"]
        self.assertEqual((item.accounting_document, item.is_reversed), ("1800000030", False))

    def test_many_references_are_chunked(self):
        client = FakeClient([])
        find_receivables(client, [f"REF{i}" for i in range(MAX_TERMS + 1)], company_code="CA01")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("CompanyCode eq 'CA01'", client.calls[0])


class DefectTests(unittest.TestCase):
    """Cases the first version lost or misread."""
    def test_two_references_to_the_same_item_are_both_found(self):
        # One advice line quotes the reference, another the document number of the same invoice.
        found = find_receivables(FakeClient([row(ref="INV-7", doc="0012345678")]), ["INV-7", "12345678"])
        self.assertEqual([i.matched_by for i in found["INV-7"]], ["DocumentReferenceID"])
        self.assertEqual([i.matched_by for i in found["12345678"]], ["AccountingDocument"])

    def test_references_sharing_one_padded_document_number_are_both_found(self):
        found = find_receivables(FakeClient([row(ref="", doc="0000000123")]), ["123", "0000000123"])
        self.assertEqual((len(found["123"]), len(found["0000000123"])), (1, 1))

    def test_non_ascii_digits_are_not_document_numbers(self):
        client = FakeClient([])
        find_receivables(client, ["\u0661\u0662\u0663\u0664"])  # Arabic-Indic digits: str.isdigit() is True
        self.assertEqual(len(client.calls), 1)

    def test_every_candidate_is_returned_with_what_resolution_needs(self):
        rows = [row(doc="1400000001", CompanyCode="CA01", DebitCreditCode="S", BranchAccount="10059152", SpecialGLCode=""),
                row(doc="1400000001", CompanyCode="CA02", DebitCreditCode="H", amount="-5", SpecialGLCode="A",
                    AccountingDocumentType="DG")]
        first, second = find_receivables(FakeClient(rows), ["8700276154"])["8700276154"]
        self.assertEqual((first.company_code, first.debit_credit, first.branch_account), ("CA01", "S", "0010059152"))
        self.assertEqual((second.company_code, second.debit_credit, second.special_gl, second.document_type),
                         ("CA02", "H", "A", "DG"))
        self.assertEqual(first.to_dict()["open_amount"], None)


class NumberFormatTests(unittest.TestCase):
    """S/4 returns numbers without leading zeros and "0" for none; items carry the stored width."""
    def test_nine_digit_document_number_is_found_although_s4_returns_it_unpadded(self):
        client = FakeClient([row(ref="", doc="100000014", AccountingDocumentItem="1")])
        client.rows[0]["AccountingDocument"] = "100000014"
        # the fake compares filter text, so serve the row for the padded search value as S/4 does
        client.get_json = lambda service, path, params: client.calls.append(params["$filter"]) or {"d": {"results": (
            client.rows if "AccountingDocument eq '0100000014'" in params["$filter"] else [])}}
        item, = find_receivables(client, ["100000014"])["100000014"]
        self.assertEqual((item.matched_by, item.accounting_document, item.accounting_document_item),
                         ("AccountingDocument", "0100000014", "001"))

    def test_eight_digit_billing_document_is_found(self):
        billed = row(ref="", doc="1400000077", BillingDocument="90001234")
        client = FakeClient([billed])
        client.get_json = lambda service, path, params: client.calls.append(params["$filter"]) or {"d": {"results": (
            [billed] if "BillingDocument eq '0090001234'" in params["$filter"] else [])}}
        item, = find_receivables(client, ["90001234"])["90001234"]
        self.assertEqual((item.matched_by, item.billing_document), ("BillingDocument", "0090001234"))

    def test_zero_means_no_value_and_other_numbers_get_their_stored_width(self):
        plain = row(AccountingDocumentItem="2", InvoiceReference="", InvoiceReferenceFiscalYear="0", InvoiceItemReference="0",
                    ClearingAccountingDocument="", ReverseDocument="")
        linked = row(doc="100000050", AccountingDocumentItem="3", InvoiceReference="100000014", InvoiceReferenceFiscalYear="2025",
                     InvoiceItemReference="1", cleared=True, ReverseDocument="100000099")
        first, second = find_receivables(FakeClient([plain, linked]), ["8700276154"])["8700276154"]
        self.assertEqual((first.accounting_document_item, first.invoice_reference, first.invoice_reference_year,
                          first.invoice_item_reference), ("002", "", "", ""))
        self.assertEqual((second.accounting_document, second.invoice_reference, second.invoice_reference_year,
                          second.invoice_item_reference, second.clearing_document, second.reverse_document),
                         ("0100000050", "0100000014", "2025", "001", "1500000009", "0100000099"))


class TierTests(unittest.TestCase):
    """The caller decides whether a reference still needs the next field."""
    def test_reference_goes_on_while_the_caller_needs_more(self):
        rows = [row(ref="1800000102", doc="1400000001", cleared=True), row(ref="", doc="1800000102")]
        client = FakeClient(rows)
        found = find_receivables(client, ["1800000102"], needs_more=lambda ref, items: all(i.is_cleared for i in items))
        self.assertEqual([(i.matched_by, i.is_cleared) for i in found["1800000102"]],
                         [("DocumentReferenceID", True), ("AccountingDocument", False)])
        self.assertEqual(len(client.calls), 2)  # the open document number hit satisfied the caller before pass 3

    def test_default_stops_at_the_first_field_with_a_hit(self):
        rows = [row(ref="1800000102", doc="1400000001", cleared=True), row(ref="", doc="1800000102")]
        client = FakeClient(rows)
        found = find_receivables(client, ["1800000102"])
        self.assertEqual((len(found["1800000102"]), len(client.calls)), (1, 1))


class LinkedItemTests(unittest.TestCase):
    """Partial payments and residual items are read by the document they refer to."""
    def test_items_referring_to_a_found_document_are_read_once(self):
        invoice = row()
        partial = row(ref="PAY-1", doc="1500000001", amount="-400", InvoiceReference="1400000001",
                      InvoiceReferenceFiscalYear="2025", FollowOnDocumentType="Z")
        reversal = row(ref="PAY-2", doc="1500000002", InvoiceReference="1400000001", IsReversal=True)
        client = FakeClient([invoice, partial, partial, reversal])
        found = find_receivables(client, ["8700276154"])
        trace = []
        linked = find_linked_items(client, found["8700276154"], trace)
        self.assertEqual([(i.accounting_document, i.follow_on_type, i.invoice_reference, i.invoice_reference_year)
                          for i in linked], [("1500000001", "Z", "1400000001", "2025")])
        self.assertEqual(trace[0]["field"], "InvoiceReference")
        self.assertEqual(find_linked_items(client, []), [])


if __name__ == "__main__":
    unittest.main()
