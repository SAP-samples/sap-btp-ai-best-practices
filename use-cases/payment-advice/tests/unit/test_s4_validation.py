"""Offline checks for S/4 advice validation and payment advice payload mapping.

Golden case: remittance 2000976003 (10,303.26 CAD) = one supplier invoice
8700276154 (10,596.28 gross, 211.93 discount, region A) and one customer
chargeback 924610168612UL (-81.09, reason code 316, region O).
"""
import copy
import unittest
from decimal import Decimal

from app.s4.lookup import ReceivableItem
from app.s4.payment_advice import build_payload, key_path, odata_date
from app.s4.validation import check_advice

GOLDEN_ADVICE = {
    "header": {"payment_reference": "2000976003", "payment_amount": 10303.26, "payment_date": "2025-09-18"},
    "line_items": [
        {"row_id": "a:1", "invoice_reference": "924610168612UL", "gross_amount": -81.09, "discount_amount": 0,
         "net_amount": -81.09, "document_nature": "chargeback", "reason_code": "316"},
        {"row_id": "a:2", "invoice_reference": "8700276154", "gross_amount": 10596.28, "discount_amount": 211.93,
         "net_amount": 10384.35, "document_nature": "invoice", "reason_code": None},
    ],
}


def receivable(ref="8700276154", customer="0010053628", amount="10596.28", cleared=False, doc="1400000001"):
    """Build a lookup hit as S/4 would return it for a seeded invoice."""
    return ReceivableItem(ref, "DocumentReferenceID", "CA01", "2025", doc, "001", customer, ref, Decimal(amount), "CAD",
                          cleared, "1500000009" if cleared else "")


def found(**overrides):
    """Lookup output for the golden advice, optionally overriding the invoice hit."""
    return {"8700276154": [receivable(**overrides)], "924610168612UL": []}


def codes(check, severity="blocking"):
    """Return the issue codes of one severity."""
    return {issue["code"] for issue in check["issues"] if issue["severity"] == severity}


class GoldenAdviceTests(unittest.TestCase):
    """The clean Contoso advice validates and maps field by field."""
    def test_clean_advice_is_ready_and_derives_header_from_s4(self):
        check = check_advice(GOLDEN_ADVICE, found())
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(check["derived"], {"company_code": "CA01", "customer": "0010053628",
                                            "customers": ["0010053628"], "currency": "CAD",
                                            "payer_account": None, "rule_company_code": None})
        self.assertEqual([line["kind"] for line in check["lines"]], ["deduction", "matched"])

    def test_payload_maps_invoice_and_deduction_items(self):
        payload = build_payload(GOLDEN_ADVICE, check_advice(GOLDEN_ADVICE, found()))
        items = payload.pop("to_PaymentAdviceItem")
        self.assertEqual(payload, {
            "CompanyCode": "CA01", "PaymentAdviceAccountType": "D", "PaymentAdviceAccount": "0010053628",
            "PaymentAdviceType": "04", "PaymentCurrency": "CAD", "PaidAmountInPaytCurrency": "10303.26",
            "PaymentAdviceHeaderText": "2000976003", "PaymentDate": "/Date(1758153600000)/"})
        self.assertEqual(items[0], {
            "Currency": "CAD", "GrossAmountInPaymentCurrency": "-81.09", "NetPaymentAmountInPaytCurrency": "-81.09",
            "AssignmentReference": "924610168612UL", "DocumentItemText": "924610168612UL chargeback",
            "PaymentDifferenceReason": "316"})
        self.assertEqual(items[1], {
            "Currency": "CAD", "GrossAmountInPaymentCurrency": "10596.28", "NetPaymentAmountInPaytCurrency": "10384.35",
            "DocumentReferenceID": "8700276154", "AccountingDocument": "1400000001", "FiscalYear": "2025",
            "CashDiscountAmountInPaytCrcy": "211.93"})

    def test_demo_reason_alias_changes_only_the_payload(self):
        check = check_advice(GOLDEN_ADVICE, found())
        items = build_payload(GOLDEN_ADVICE, check, reason_aliases={"316": "060"})["to_PaymentAdviceItem"]
        self.assertEqual(items[0]["PaymentDifferenceReason"], "060")
        self.assertEqual(check["lines"][0]["reason_code"], "316")

    def test_key_path_and_date_literal(self):
        key = {"CompanyCode": "CA01", "PaymentAdviceAccountType": "D", "PaymentAdviceAccount": "0010053628",
               "PaymentAdvice": "0425092812000001"}
        self.assertEqual(key_path(key), "/A_PaymentAdvice(CompanyCode='CA01',PaymentAdviceAccountType='D',"
                                        "PaymentAdviceAccount='0010053628',PaymentAdvice='0425092812000001')")
        self.assertIsNone(odata_date("not a date"))


class BlockingTests(unittest.TestCase):
    """Anything that would make S/4 clear the wrong items blocks posting."""
    def test_cleared_invoice_blocks_and_payload_is_refused(self):
        check = check_advice(GOLDEN_ADVICE, found(cleared=True))
        self.assertIn("invoice_cleared", codes(check))
        with self.assertRaises(ValueError):
            build_payload(GOLDEN_ADVICE, check)

    def test_invoice_missing_in_s4_blocks_and_no_company_code_is_derived(self):
        check = check_advice(GOLDEN_ADVICE, {"8700276154": [], "924610168612UL": []})
        self.assertEqual(codes(check), {"invoice_not_found", "no_invoice_resolved"})

    def test_totals_mismatch_blocks(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["header"]["payment_amount"] = 10303.00
        self.assertIn("totals_mismatch", codes(check_advice(advice, found())))

    def test_deduction_without_reason_code_blocks(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"][0]["reason_code"] = None
        self.assertEqual(codes(check_advice(advice, found())), {"missing_reason_code"})

    def test_ambiguous_reference_blocks(self):
        hits = {"8700276154": [receivable(), receivable(doc="1400000002")], "924610168612UL": []}
        self.assertIn("ambiguous_reference", codes(check_advice(GOLDEN_ADVICE, hits)))

    def test_same_s4_document_applied_twice_blocks(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"].append(dict(advice["line_items"][1], row_id="a:3"))
        advice["header"]["payment_amount"] = 20687.61
        self.assertIn("duplicate_application", codes(check_advice(advice, found())))


class DifferenceTests(unittest.TestCase):
    """Short payments need a reason code; lead payers use alternative accounts."""
    def test_short_payment_with_reason_code_warns_and_sends_reason(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"][1]["reason_code"] = "319"
        check = check_advice(advice, found(amount="10600.00"))
        self.assertTrue(check["ready"])
        self.assertIn("amount_difference", codes(check, "warning"))
        self.assertEqual(build_payload(advice, check)["to_PaymentAdviceItem"][1]["PaymentDifferenceReason"], "319")

    def test_short_payment_without_reason_code_blocks(self):
        self.assertIn("difference_without_reason", codes(check_advice(GOLDEN_ADVICE, found(amount="10600.00"))))

    def test_invoice_of_other_customer_becomes_alternative_account(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"].append({"invoice_reference": "8700276155", "gross_amount": 100, "discount_amount": 0,
                                     "net_amount": 100, "document_nature": "invoice"})
        advice["header"]["payment_amount"] = 10403.26
        hits = {**found(), "8700276155": [receivable("8700276155", customer="0010059152", amount="100", doc="1400000003")]}
        check = check_advice(advice, hits)
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("multiple_customers", codes(check, "warning"))
        item = build_payload(advice, check)["to_PaymentAdviceItem"][2]
        self.assertEqual((item["PaymentAdviceAltvAccount"], item["PaymentAdviceAltvAccountType"]), ("0010059152", "D"))

    def test_suspected_duplicates_only_warn(self):
        check = check_advice(GOLDEN_ADVICE, found(), {"suspected_duplicate_groups": [[0, 1]]})
        self.assertTrue(check["ready"])
        self.assertIn("suspected_duplicates", codes(check, "warning"))



def regional_advice(invoice_account="10059152", deduction_account="10058745", company_code="CA01"):
    """Golden advice after the customer's rules ran: Rg accounts per line, payer and company code on the header."""
    advice = copy.deepcopy(GOLDEN_ADVICE)
    advice["header"].update(payer_account="10053628", company_code=company_code)
    advice["line_items"][0]["customer_account"] = deduction_account
    advice["line_items"][1]["customer_account"] = invoice_account
    return advice


def demo_hits(customer="0010059152"):
    """The seeded invoice lives in demo company code Z291 under the regional customer."""
    item = receivable(customer=customer)
    return {"8700276154": [ReceivableItem(**{**item.__dict__, "company_code": "Z291"})], "924610168612UL": []}


def lookup(existing=("0010059152", "0010058745", "0010053628")):
    """Customer existence lookup returning a fixed set and recording the company code asked for."""
    asked = []

    def run(accounts, company_code):
        asked.append(company_code)
        return set(existing) & set(accounts)
    run.asked = asked
    return run


class RuleAccountTests(unittest.TestCase):
    """Rule-derived accounts and company code drive the advice and are cross-checked against S/4."""
    def test_payer_header_with_regional_alternative_accounts(self):
        customers = lookup()
        check = check_advice(regional_advice(), demo_hits(), customer_lookup=customers, aliases={"CA01": "Z291"})
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual((check["derived"]["company_code"], check["derived"]["customer"]), ("Z291", "0010053628"))
        self.assertEqual(customers.asked, ["Z291"])
        payload = build_payload(regional_advice(), check)
        self.assertEqual(payload["PaymentAdviceAccount"], "0010053628")
        deduction, invoice = payload["to_PaymentAdviceItem"]
        self.assertEqual((deduction["PaymentAdviceAltvAccount"], deduction["PaymentAdviceAltvAccountType"]), ("0010058745", "D"))
        self.assertEqual(invoice["PaymentAdviceAltvAccount"], "0010059152")
        # S/4 checks the alternative customer in the item's own company code.
        self.assertEqual((deduction["AlternativeCompanyCode"], invoice["AlternativeCompanyCode"]), ("Z291", "Z291"))

    def test_rule_account_disagreeing_with_s4_invoice_blocks(self):
        check = check_advice(regional_advice(invoice_account="10058745"), demo_hits(), customer_lookup=lookup(),
                             aliases={"CA01": "Z291"})
        self.assertIn("customer_mismatch", codes(check))

    def test_rule_account_missing_in_company_code_blocks(self):
        check = check_advice(regional_advice(), demo_hits(), customer_lookup=lookup(("0010059152", "0010053628")),
                             aliases={"CA01": "Z291"})
        self.assertEqual(codes(check), {"unknown_customer"})

    def test_company_code_without_alias_disagreeing_with_s4_blocks(self):
        check = check_advice(regional_advice(), demo_hits(), customer_lookup=lookup())
        self.assertIn("company_code_mismatch", codes(check))

    def test_rules_assign_an_advice_with_no_matched_invoice(self):
        advice = regional_advice()
        advice["line_items"] = advice["line_items"][:1]
        advice["header"].update(payment_amount=-81.09, payment_currency="CAD")
        check = check_advice(advice, {"924610168612UL": []}, customer_lookup=lookup(), aliases={"CA01": "Z291"})
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("derived_from_rules", codes(check, "warning"))
        self.assertEqual((check["derived"]["company_code"], check["derived"]["customer"]), ("Z291", "0010053628"))


if __name__ == "__main__":
    unittest.main()
