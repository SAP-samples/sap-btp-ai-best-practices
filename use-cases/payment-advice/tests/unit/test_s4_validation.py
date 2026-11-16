"""Offline checks for S/4 advice validation and payment advice payload mapping.

Golden case: remittance 2000976003 (10,303.26 CAD) = one supplier invoice
8700276154 (10,596.28 gross, 211.93 discount, region A) and one customer
chargeback 924610168612UL (-81.09, reason code 316, region O).
"""
import copy
import unittest
from decimal import Decimal

from app.s4.lookup import ReceivableItem
from app.s4.payment_advice import build_payload, compare_read_back, key_path, odata_date
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

    def test_read_back_comparison_accepts_s4_formats_and_names_differences(self):
        payload = build_payload(GOLDEN_ADVICE, check_advice(GOLDEN_ADVICE, found()))
        stored = {**{k: v for k, v in payload.items() if k != "to_PaymentAdviceItem"},
                  "PaymentAdviceAccount": "10053628",  # S/4 answers without leading zeros
                  "items": [dict(item) for item in payload["to_PaymentAdviceItem"]]}
        self.assertEqual(compare_read_back(payload, stored), {"items_sent": 2, "items_stored": 2, "differences": []})
        stored["items"][1]["NetPaymentAmountInPaytCurrency"] = "10000.00"
        stored["items"].append({})
        differences = compare_read_back(payload, stored)["differences"]
        self.assertEqual(differences, ["2 items sent, S/4 holds 3",
                                       "Item 2 NetPaymentAmountInPaytCurrency: sent '10384.35', S/4 has '10000.00'"])

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

    def test_reversed_invoice_blocks_with_its_own_issue_only(self):
        reversed_item = ReceivableItem("8700276154", "DocumentReferenceID", "CA01", "2025", "1400000001", "001",
                                       "0010053628", "8700276154", Decimal("10596.28"), "CAD", True, "1800000028",
                                       is_reversed=True, reverse_document="1800000028")
        check = check_advice(GOLDEN_ADVICE, {"8700276154": [reversed_item], "924610168612UL": []})
        self.assertEqual(codes(check), {"invoice_reversed"})
        self.assertEqual(check["lines"][1]["status"], "reversed")
        self.assertEqual(check["derived"]["currency"], "CAD")

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


class ExcludedLineTests(unittest.TestCase):
    """Rows the reviewer excluded are neither checked nor posted; the others keep their table numbers."""
    def test_excluded_row_is_skipped_and_numbers_are_kept(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        extra = {**advice["line_items"][0], "row_id": "a:0", "excluded": True}
        advice["line_items"].insert(0, extra)
        check = check_advice(advice, found())
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual([line["index"] for line in check["lines"]], [1, 2])
        self.assertEqual(len(build_payload(advice, check)["to_PaymentAdviceItem"]), 2)


class ResidualTests(unittest.TestCase):
    """A reviewer can post a reversed or unknown invoice line as a residual open item with a reason code."""
    def advice(self, reason="060"):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["header"].update(company_code="CA01", payer_account="10053628")
        advice["line_items"][1].update(s4_treatment="residual", reason_code=reason)
        return advice

    def reversed_hit(self):
        return {"8700276154": [ReceivableItem("8700276154", "DocumentReferenceID", "CA01", "2025", "1400000001", "001",
                                              "0010053628", "8700276154", Decimal("10596.28"), "CAD", True, "1800000028",
                                              is_reversed=True, reverse_document="1800000028")], "924610168612UL": []}

    def test_reversed_invoice_posted_as_residual_is_ready_without_document(self):
        check = check_advice(self.advice(), self.reversed_hit())
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual((check["lines"][1]["kind"], check["derived"]["currency"]), ("residual", "CAD"))
        item = build_payload(self.advice(), check)["to_PaymentAdviceItem"][1]
        self.assertEqual(item, {"Currency": "CAD", "GrossAmountInPaymentCurrency": "10384.35",
                                "NetPaymentAmountInPaytCurrency": "10384.35", "AssignmentReference": "8700276154",
                                "DocumentItemText": "8700276154 residual", "PaymentDifferenceReason": "060"})

    def test_residual_without_reason_code_blocks(self):
        self.assertIn("missing_reason_code", codes(check_advice(self.advice(reason=None), self.reversed_hit())))

    def test_residual_on_an_open_invoice_warns(self):
        self.assertIn("residual_with_open_invoice", codes(check_advice(self.advice(), found()), "warning"))


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


class HeaderDerivationTests(unittest.TestCase):
    """Header-level blocks: one company code, one currency, a payment amount and a currency are required."""
    def two_invoices(self, **second):
        """Golden advice plus a second invoice whose S/4 item differs from the first as given in ``second``."""
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"].append({"row_id": "a:3", "invoice_reference": "8700276155", "gross_amount": 100,
                                     "discount_amount": 0, "net_amount": 100, "document_nature": "invoice"})
        advice["header"]["payment_amount"] = 10403.26
        other = receivable("8700276155", amount="100", doc="1400000003")
        return advice, {**found(), "8700276155": [ReceivableItem(**{**other.__dict__, **second})]}

    def test_invoices_in_two_company_codes_block(self):
        check = check_advice(*self.two_invoices(company_code="CA02"))
        self.assertIn("multiple_company_codes", codes(check))
        self.assertIsNone(check["derived"]["company_code"])

    def test_invoices_in_two_currencies_block(self):
        check = check_advice(*self.two_invoices(currency="USD"))
        self.assertIn("multiple_currencies", codes(check))
        self.assertIsNone(check["derived"]["currency"])

    def test_advice_currency_differing_from_s4_blocks(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["header"]["payment_currency"] = "usd"
        self.assertIn("currency_mismatch", codes(check_advice(advice, found())))

    def test_no_match_and_no_advice_currency_blocks(self):
        advice = regional_advice()
        advice["line_items"] = advice["line_items"][:1]
        advice["header"]["payment_amount"] = -81.09
        check = check_advice(advice, {"924610168612UL": []}, customer_lookup=lookup(), aliases={"CA01": "Z291"})
        self.assertIn("missing_currency", codes(check))

    def test_missing_payment_amount_blocks(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["header"]["payment_amount"] = None
        self.assertIn("missing_payment_amount", codes(check_advice(advice, found())))

    def test_long_deduction_reference_warns_and_is_cut_in_the_assignment(self):
        advice = copy.deepcopy(GOLDEN_ADVICE)
        advice["line_items"][0]["invoice_reference"] = "DEDUCTION-REFERENCE-0001"
        check = check_advice(advice, {**found(), "DEDUCTION-REFERENCE-0001": []})
        self.assertIn("reference_truncated", codes(check, "warning"))
        item = build_payload(advice, check)["to_PaymentAdviceItem"][0]
        self.assertEqual(item["AssignmentReference"], "DEDUCTION-REFERENC")  # 18 characters
        self.assertEqual(item["DocumentItemText"], "DEDUCTION-REFERENCE-0001 chargeback")


def advice_of(*lines, amount=None, **header):
    """Advice with the given lines; the payment amount defaults to the sum of their nets."""
    rows = [{"row_id": f"r:{n}", "discount_amount": 0, "document_nature": "invoice", **line} for n, line in enumerate(lines, 1)]
    total = round(sum(row["net_amount"] for row in rows), 2) if amount is None else amount
    return {"header": {"payment_reference": "PAY-0001", "payment_amount": total, "payment_date": "2026-03-01", **header},
            "line_items": rows}


def invoice_line(reference="8700838001", gross=1000.0, net=None, **extra):
    """One invoice line paying its gross in full unless a net is given."""
    return {"invoice_reference": reference, "gross_amount": gross, "net_amount": gross if net is None else net, **extra}


def candidate(reference="8700838001", doc="1800000101", amount="1000.00", cleared=False, **extra):
    """One S/4 customer item in company code 1000 for customer 0010000001, open unless said otherwise."""
    values = {"company_code": "1000", "fiscal_year": "2026", "accounting_document_item": "001", "customer": "0010000001",
              "document_reference": reference, "currency": "EUR", "clearing_document": "1500000009" if cleared else "",
              "matched_by": "DocumentReferenceID", **extra}
    return ReceivableItem(reference=reference, accounting_document=doc, amount=Decimal(amount), is_cleared=cleared, **values)


class DuplicateTests(unittest.TestCase):
    """Several S/4 candidates for one reference: resolved by rules, by the reviewer, or left to the reviewer."""
    pair = {"8700838001": [candidate(), candidate(doc="1800000102", fiscal_year="2025")]}

    def test_unresolvable_candidates_block_and_are_listed_for_the_reviewer(self):
        check = check_advice(advice_of(invoice_line()), self.pair)
        self.assertEqual(codes(check), {"ambiguous_reference", "no_invoice_resolved"})
        self.assertEqual([c["pick"] for c in check["lines"][0]["candidates"]],
                         ["1000/1800000101/2026/001", "1000/1800000102/2025/001"])
        self.assertEqual(check["lines"][0]["candidates"][0]["state"], "open")

    def test_reviewer_pick_resolves_and_is_counted(self):
        check = check_advice(advice_of(invoice_line(s4_document="1000/1800000102/2025")), self.pair)
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(check["lines"][0]["s4_item"]["accounting_document"], "1800000102")
        self.assertEqual(check["summary"]["resolution"], {"unique": 0, "auto": 0, "reviewer": 1})

    def test_pick_that_s4_no_longer_offers_blocks(self):
        check = check_advice(advice_of(invoice_line(s4_document="1000/1800000777/2026")), self.pair)
        self.assertIn("selection_stale", codes(check))

    def test_residual_treatment_ignores_a_pick(self):
        line = invoice_line(s4_document="1000/1800000777/2026", s4_treatment="residual", reason_code="060")
        check = check_advice(advice_of(line, company_code="1000", payer_account="10000001", payment_currency="EUR"), self.pair)
        self.assertEqual((check["lines"][0]["kind"], "selection_stale" in codes(check)), ("residual", False))

    def test_open_document_is_chosen_over_a_cleared_one_without_a_warning_when_the_rules_name_the_customer(self):
        hits = {"8700838001": [candidate(doc="1800000001", cleared=True, fiscal_year="2025"), candidate()]}
        check = check_advice(advice_of(invoice_line(customer_account="10000001")), hits)
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(codes(check, "warning"), set())
        self.assertEqual(check["lines"][0]["resolution"], {"method": "auto", "warning": None,
                                                           "steps": [{"rule": "open", "kept": 1, "dropped": 1}]})

    def test_amount_decides_with_a_warning(self):
        hits = {"8700838001": [candidate(doc="1800000101", amount="250.00"), candidate(doc="1800000102")]}
        check = check_advice(advice_of(invoice_line(customer_account="10000001")), hits)
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("reference_disambiguated", codes(check, "warning"))
        self.assertEqual(check["lines"][0]["s4_item"]["accounting_document"], "1800000102")

    def test_company_code_of_the_other_lines_resolves_a_document_number_that_exists_twice(self):
        twice = [candidate("1800000500", doc="1800000500", matched_by="AccountingDocument", document_reference=""),
                 candidate("1800000500", doc="1800000500", matched_by="AccountingDocument", document_reference="",
                           company_code="2000", customer="0020000002")]
        advice = advice_of(invoice_line(), invoice_line("1800000500"))
        check = check_advice(advice, {"8700838001": [candidate()], "1800000500": twice})
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(check["derived"]["company_code"], "1000")
        self.assertEqual(check["lines"][1]["resolution"]["steps"][0]["rule"], "advice_company_code")


class CrossCompanyCodeTests(unittest.TestCase):
    """One advice may pay items of several company codes; its header is the company code that receives the payment."""
    def two_codes(self, customer="0010000001", **header):
        """Two invoice lines: the first in company code 1000, the second in 2000 for ``customer``."""
        advice = advice_of(invoice_line(), invoice_line("8700838002", gross=250.0), **header)
        elsewhere = candidate("8700838002", doc="1800000202", amount="250.00", company_code="2000", customer=customer)
        return advice, {"8700838001": [candidate()], "8700838002": [elsewhere]}

    def test_rule_company_code_heads_an_advice_with_items_elsewhere(self):
        check = check_advice(*self.two_codes(company_code="1000"))
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual((check["derived"]["company_code"], check["summary"]["company_codes"]), ("1000", {"1000": 1, "2000": 1}))
        message = next(i["message"] for i in check["issues"] if i["code"] == "cross_company_items")
        self.assertIn("2000 (lines 2)", message)

    def test_item_elsewhere_is_sent_with_its_own_company_code(self):
        advice, hits = self.two_codes(company_code="1000")
        payload = build_payload(advice, check_advice(advice, hits))
        home, elsewhere = payload["to_PaymentAdviceItem"]
        self.assertEqual(payload["CompanyCode"], "1000")
        self.assertNotIn("AlternativeCompanyCode", home)
        self.assertEqual({name: elsewhere[name] for name in ("AccountingDocument", "AlternativeCompanyCode",
                                                             "PaymentAdviceAltvAccount", "PaymentAdviceAltvAccountType")},
                         {"AccountingDocument": "1800000202", "AlternativeCompanyCode": "2000",
                          "PaymentAdviceAltvAccount": "0010000001", "PaymentAdviceAltvAccountType": "D"})

    def test_other_customer_in_the_advice_company_code_keeps_the_advice_company_code(self):
        advice = advice_of(invoice_line(), invoice_line("8700838002", gross=250.0), company_code="1000")
        other = candidate("8700838002", doc="1800000202", amount="250.00", customer="0020000002")
        payload = build_payload(advice, check_advice(advice, {"8700838001": [candidate()], "8700838002": [other]}))
        self.assertEqual((payload["to_PaymentAdviceItem"][1]["AlternativeCompanyCode"],
                          payload["to_PaymentAdviceItem"][1]["PaymentAdviceAltvAccount"]), ("1000", "0020000002"))

    def test_without_a_rule_company_code_the_receiving_one_is_not_guessed(self):
        check = check_advice(*self.two_codes())
        message = next(i["message"] for i in check["issues"] if i["code"] == "multiple_company_codes")
        self.assertIn("1000 (lines 1), 2000 (lines 2)", message)
        self.assertIsNone(check["derived"]["company_code"])

    def test_rule_company_code_that_no_item_confirms_blocks(self):
        check = check_advice(*self.two_codes(company_code="3000"))
        self.assertEqual((codes(check), check["derived"]["company_code"]), ({"company_code_mismatch"}, None))

    def test_lead_payer_comes_from_the_advice_company_code(self):
        check = check_advice(*self.two_codes(customer="0020000002", company_code="2000"))
        self.assertEqual((check["derived"]["customer"], check["derived"]["customers"]),
                         ("0020000002", ["0020000002", "0010000001"]))

    def test_customers_are_checked_in_the_company_code_they_are_used_in(self):
        advice, hits = self.two_codes(customer="0020000002", company_code="1000", payer_account="10000001")
        asked = {}
        check = check_advice(advice, hits, customer_lookup=lambda accounts, code: asked.setdefault(code, accounts) and set(accounts))
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(asked, {"1000": ["0010000001"], "2000": ["0020000002"]})

    def test_rule_account_of_a_line_elsewhere_must_exist_there(self):
        advice, hits = self.two_codes(company_code="1000")
        advice["line_items"][1]["customer_account"] = "10000001"
        check = check_advice(advice, hits, customer_lookup=lambda accounts, code: set(accounts) if code == "1000" else set())
        message = next(i["message"] for i in check["issues"] if i["code"] == "unknown_customer")
        self.assertIn("company code 2000", message)

    def test_reason_code_of_a_line_elsewhere_must_exist_in_both_company_codes(self):
        advice, hits = self.two_codes(company_code="1000")
        advice["line_items"][1].update(gross_amount=250.0, deduction_amount=10.0, net_amount=240.0, reason_code="060")
        advice["header"]["payment_amount"] = 1240.0
        both = check_advice(advice, hits, reason_codes=lambda code: {"060"})
        self.assertTrue(both["ready"], both["issues"])
        only_home = check_advice(advice, hits, reason_codes=lambda code: {"060"} if code == "1000" else {"010"})
        message = next(i["message"] for i in only_home["issues"] if i["code"] == "unknown_reason_code")
        self.assertIn("company code 2000", message)
        unverified = check_advice(advice, hits, reason_codes=lambda code: {"060"} if code == "1000" else None)
        self.assertTrue(unverified["ready"], unverified["issues"])
        self.assertIn("company code 2000", next(i["message"] for i in unverified["issues"]
                                                if i["code"] == "reason_codes_not_verified"))

    def chosen_among_several(self, customer):
        """Second line: no document in 1000; the rules pick the open one in 2000 over a cleared one in 3000."""
        advice = advice_of(invoice_line(), invoice_line("8700838002", gross=250.0), company_code="1000")
        open_one = candidate("8700838002", doc="1800000202", amount="250.00", company_code="2000", customer=customer)
        cleared = candidate("8700838002", doc="1800000303", amount="250.00", company_code="3000", customer="0030000003",
                            cleared=True)
        return advice, {"8700838001": [candidate()], "8700838002": [open_one, cleared]}

    def test_document_picked_among_several_for_an_unknown_customer_elsewhere_needs_the_reviewer(self):
        advice, hits = self.chosen_among_several("0020000002")
        check = check_advice(advice, hits)
        self.assertEqual(codes(check), {"cross_company_unconfirmed"})
        self.assertEqual([c["pick"] for c in check["lines"][1]["candidates"]], ["2000/1800000202/2026/001"])
        advice["line_items"][1]["s4_document"] = "2000/1800000202/2026/001"
        confirmed = check_advice(advice, hits)
        self.assertTrue(confirmed["ready"], confirmed["issues"])
        self.assertNotIn("candidates", confirmed["lines"][1])

    def test_customer_known_in_the_advice_company_code_needs_no_confirmation(self):
        check = check_advice(*self.chosen_among_several("0010000001"))
        self.assertTrue(check["ready"], check["issues"])

    def test_same_document_number_applied_in_two_company_codes_is_not_a_duplicate(self):
        advice = advice_of(invoice_line(), invoice_line("8700838002"), company_code="1000")
        twin = candidate("8700838002", company_code="2000")
        check = check_advice(advice, {"8700838001": [candidate()], "8700838002": [twin]})
        self.assertNotIn("duplicate_application", codes(check))


class OpenBalanceTests(unittest.TestCase):
    """The advice is compared with what is still open in S/4, not with the original document amount."""
    def test_partially_paid_invoice_matches_its_open_balance(self):
        partly_paid = candidate(open_amount=Decimal("600.00"), partial_payments=Decimal("-400.00"))
        check = check_advice(advice_of(invoice_line(gross=600.0)), {"8700838001": [partly_paid]})
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual((check["lines"][0]["open_amount"], codes(check, "warning")), ("600.00", set()))

    def test_paying_the_original_amount_of_a_partially_paid_invoice_needs_a_reason(self):
        partly_paid = candidate(open_amount=Decimal("600.00"), partial_payments=Decimal("-400.00"))
        check = check_advice(advice_of(invoice_line()), {"8700838001": [partly_paid]})
        message = next(i["message"] for i in check["issues"] if i["code"] == "difference_without_reason")
        self.assertIn("S/4 open amount 600.00 (document amount 1000.00, partial payments -400.00)", message)

    def test_residual_item_of_a_cleared_invoice_is_applied_by_item(self):
        residual = candidate(doc="1400000003", amount="150.00", accounting_document_item="002",
                             residual_of="1800000101/2026", open_amount=Decimal("150.00"))
        advice = advice_of(invoice_line(gross=150.0))
        check = check_advice(advice, {"8700838001": [residual]})
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("residual_item_applied", codes(check, "warning"))
        item, = build_payload(advice, check)["to_PaymentAdviceItem"]
        self.assertEqual((item["AccountingDocument"], item["AccountingDocumentItem"]), ("1400000003", "002"))


class MultiItemTests(unittest.TestCase):
    """Documents with several customer items are applied item by item."""
    instalments = {"8700838001": [candidate(amount="300.00"), candidate(amount="700.00", accounting_document_item="002")]}

    def test_line_paying_one_instalment_targets_that_item(self):
        advice = advice_of(invoice_line(gross=700.0))
        check = check_advice(advice, self.instalments)
        item, = build_payload(advice, check)["to_PaymentAdviceItem"]
        self.assertEqual((item["AccountingDocumentItem"], item["GrossAmountInPaymentCurrency"]), ("002", "700.00"))

    def test_line_paying_the_document_total_sends_one_item_per_instalment(self):
        advice = advice_of(invoice_line())
        check = check_advice(advice, self.instalments)
        self.assertTrue(check["ready"], check["issues"])
        items = build_payload(advice, check)["to_PaymentAdviceItem"]
        self.assertEqual([(i["AccountingDocumentItem"], i["GrossAmountInPaymentCurrency"], i["NetPaymentAmountInPaytCurrency"])
                          for i in items], [("001", "300.00", "300.00"), ("002", "700.00", "700.00")])

    def test_two_lines_paying_different_items_of_one_document_are_not_a_duplicate(self):
        advice = advice_of(invoice_line(gross=300.0), invoice_line(gross=700.0))
        self.assertTrue(check_advice(advice, self.instalments)["ready"])

    def test_amount_fitting_no_item_is_left_to_the_reviewer(self):
        check = check_advice(advice_of(invoice_line(gross=500.0)), self.instalments)
        self.assertIn("multi_item_document", codes(check))
        self.assertEqual(len(check["lines"][0]["candidates"]), 2)


class SignAndKindTests(unittest.TestCase):
    """A line must fit the side of the S/4 item it applies."""
    def test_positive_line_on_a_credit_item_blocks(self):
        credit = {"8700838001": [candidate(amount="-1000.00")]}
        check = check_advice(advice_of(invoice_line()), credit)
        self.assertIn("sign_mismatch", codes(check))
        self.assertEqual(check["lines"][0]["kind"], "unresolved")

    def test_deduction_quoting_an_invoice_is_a_deduction_not_a_second_application(self):
        advice = advice_of(invoice_line(), invoice_line(gross=-50.0, document_nature="chargeback", reason_code="060"))
        check = check_advice(advice, {"8700838001": [candidate()]})
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual([line["kind"] for line in check["lines"]], ["matched", "deduction"])
        self.assertEqual(check["lines"][1]["related_document"], "1000/1800000101/2026")

    def test_special_gl_item_blocks(self):
        check = check_advice(advice_of(invoice_line()), {"8700838001": [candidate(special_gl="A")]})
        self.assertIn("special_gl_item", codes(check))

    def test_unknown_credit_memo_becomes_a_deduction_with_a_warning(self):
        advice = advice_of(invoice_line(), invoice_line("CM-77", gross=-20.0, document_nature="credit_memo", reason_code="060"))
        check = check_advice(advice, {"8700838001": [candidate()], "CM-77": []})
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("credit_not_found", codes(check, "warning"))


class ReferenceTests(unittest.TestCase):
    """Other spellings and the customer's own document reference."""
    def test_reference_found_under_another_spelling_warns_when_the_amount_agrees(self):
        hit = candidate("INV-0012", document_reference="INV0012", variant="INV0012")
        check = check_advice(advice_of(invoice_line("INV-0012")), {"INV-0012": [hit]})
        self.assertTrue(check["ready"], check["issues"])
        self.assertIn("reference_normalized", codes(check, "warning"))

    def test_other_spelling_with_a_different_amount_is_only_suggested(self):
        hit = candidate("INV-0012", document_reference="INV0012", variant="INV0012", amount="5.00")
        check = check_advice(advice_of(invoice_line("INV-0012")), {"INV-0012": [hit]})
        self.assertIn("invoice_not_found", codes(check))
        self.assertEqual(check["lines"][0]["candidates"][0]["pick"], "1000/1800000101/2026/001")

    def test_customer_document_reference_is_used_when_the_invoice_reference_is_empty(self):
        line = {"invoice_reference": None, "customer_document_reference": "8700838001", "gross_amount": 1000.0, "net_amount": 1000.0}
        check = check_advice(advice_of(line), {"8700838001": [candidate()]})
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(check["lines"][0]["reference_source"], "customer_document_reference")

    def test_document_without_s4_reference_sends_no_reference(self):
        advice = advice_of(invoice_line("1800000101"))
        hit = candidate("1800000101", matched_by="AccountingDocument", document_reference="")
        item, = build_payload(advice, check_advice(advice, {"1800000101": [hit]}))["to_PaymentAdviceItem"]
        self.assertNotIn("DocumentReferenceID", item)
        self.assertEqual(item["AccountingDocument"], "1800000101")


class AmountTests(unittest.TestCase):
    """Bad amounts block; decimals follow the currency."""
    def test_invalid_line_amount_and_payment_amount_block_without_raising(self):
        check = check_advice(advice_of(invoice_line(gross="abc"), amount="n/a"), {"8700838001": [candidate()]})
        messages = [i["message"] for i in check["issues"] if i["code"] == "invalid_amount"]
        self.assertEqual(len(messages), 2)
        self.assertIn("gross amount 'abc'", messages[0])

    def test_amounts_follow_the_decimals_of_the_currency(self):
        yen = {"8700838001": [candidate(amount="1000", currency="JPY")]}
        advice = advice_of(invoice_line(), payment_currency="JPY")
        payload = build_payload(advice, check_advice(advice, yen))
        self.assertEqual((payload["PaidAmountInPaytCurrency"], payload["to_PaymentAdviceItem"][0]["GrossAmountInPaymentCurrency"]),
                         ("1000", "1000"))
        dinar = {"8700838001": [candidate(amount="1.234", currency="KWD")]}
        advice = advice_of(invoice_line(gross=1.234), amount=1.234, payment_currency="KWD")
        check = check_advice(advice, dinar)
        self.assertTrue(check["ready"], check["issues"])
        self.assertEqual(build_payload(advice, check)["to_PaymentAdviceItem"][0]["NetPaymentAmountInPaytCurrency"], "1.234")
        self.assertIn("invalid_amount", codes(check_advice(advice_of(invoice_line(gross=1.234)), {"8700838001": [candidate()]})))


class MasterDataTests(unittest.TestCase):
    """Reason codes and customer blocks are checked against S/4."""
    def deduction_advice(self, reason="316"):
        return advice_of(invoice_line(), invoice_line("DED-1", gross=-10.0, document_nature="chargeback", reason_code=reason))

    hits = {"8700838001": [candidate()], "DED-1": []}

    def test_reason_code_unknown_in_the_company_code_blocks(self):
        asked = []
        check = check_advice(self.deduction_advice(), self.hits, reason_codes=lambda code: asked.append(code) or {"060"})
        self.assertEqual((codes(check), asked), ({"unknown_reason_code"}, ["1000"]))

    def test_the_aliased_reason_code_is_the_one_checked(self):
        check = check_advice(self.deduction_advice(), self.hits, reason_codes=lambda code: {"060"}, reason_aliases={"316": "060"})
        self.assertTrue(check["ready"], check["issues"])

    def test_unavailable_reason_code_list_warns_instead_of_passing_silently(self):
        check = check_advice(self.deduction_advice(), self.hits, reason_codes=lambda code: None)
        self.assertTrue(check["ready"])
        self.assertIn("reason_codes_not_verified", codes(check, "warning"))

    def test_difference_reason_of_a_matched_line_is_checked_too(self):
        advice = advice_of(invoice_line(net=900.0, reason_code="319"))
        self.assertIn("unknown_reason_code", codes(check_advice(advice, self.hits, reason_codes=lambda code: {"060"})))
        wrong_length = advice_of(invoice_line(net=900.0, reason_code="31"))
        self.assertIn("missing_reason_code", codes(check_advice(wrong_length, self.hits)))

    def test_deduction_taken_on_an_invoice_line_needs_a_reason_code_even_when_the_line_adds_up(self):
        line = invoice_line(net=950.0, deduction_amount=50.0)
        self.assertIn("difference_without_reason", codes(check_advice(advice_of(line), self.hits)))
        advice = advice_of(dict(line, reason_code="060"))
        check = check_advice(advice, self.hits, reason_codes=lambda code: {"060"})
        self.assertTrue(check["ready"], check["issues"])
        item, = build_payload(advice, check)["to_PaymentAdviceItem"]
        self.assertEqual((item["DeductionAmountInPaytCurrency"], item["PaymentDifferenceReason"]), ("50.00", "060"))

    def test_posting_block_blocks_and_deletion_flag_warns(self):
        status = lambda accounts, code: {"0010000001": {"posting_blocked": True, "deletion_flag": True}}  # noqa: E731
        check = check_advice(advice_of(invoice_line()), self.hits, customer_lookup=status)
        self.assertIn("customer_blocked", codes(check))
        self.assertIn("customer_marked_for_deletion", codes(check, "warning"))

    def test_invoice_posted_on_the_head_office_fits_the_branch_account_of_the_rules(self):
        head_office = {"8700838001": [candidate(customer="0010000009", branch_account="0010000001")]}
        check = check_advice(advice_of(invoice_line(customer_account="10000001")), head_office)
        self.assertNotIn("customer_mismatch", codes(check))


class HeaderTests(unittest.TestCase):
    """Payer, payment reference and payment date."""
    hits = {"8700838001": [candidate()], "8700838002": [candidate("8700838002", doc="1800000102", customer="0010000002")]}

    def test_two_customers_with_the_same_amount_and_no_rule_payer_block(self):
        advice = advice_of(invoice_line(), invoice_line("8700838002"))
        self.assertIn("payer_ambiguous", codes(check_advice(advice, self.hits)))
        advice["header"]["payer_account"] = "10000002"
        self.assertTrue(check_advice(advice, self.hits)["ready"])

    def test_missing_payment_reference_blocks_and_a_long_one_warns(self):
        self.assertIn("missing_payment_reference", codes(check_advice(advice_of(invoice_line(), payment_reference=" "), self.hits)))
        long_reference = check_advice(advice_of(invoice_line(), payment_reference="R" * 26), self.hits)
        self.assertTrue(long_reference["ready"])
        self.assertIn("payment_reference_truncated", codes(long_reference, "warning"))

    def test_missing_or_unreadable_payment_date_warns(self):
        check = check_advice(advice_of(invoice_line(), payment_date="18.09.2025"), self.hits)
        self.assertTrue(check["ready"])
        self.assertIn("missing_payment_date", codes(check, "warning"))

    def test_summary_counts_how_lines_were_matched(self):
        advice = advice_of(invoice_line(), invoice_line("DED-1", gross=-10.0, document_nature="chargeback", reason_code="060"),
                           invoice_line("UNKNOWN-9", gross=5.0))
        summary = check_advice(advice, {**self.hits, "DED-1": [], "UNKNOWN-9": []})["summary"]
        self.assertEqual((summary["lines"], summary["matched"], summary["deductions"], summary["unresolved"]), (3, 1, 1, 1))
        self.assertEqual((summary["matched_by"], summary["resolution"]["unique"]), ({"DocumentReferenceID": 1}, 1))


class ReadBackTests(unittest.TestCase):
    """What S/4 stored is compared by the kind of each field."""
    def test_text_fields_are_compared_as_text_and_items_by_their_s4_number(self):
        payload = {"CompanyCode": "1000", "to_PaymentAdviceItem": [
            {"DocumentReferenceID": "0090", "GrossAmountInPaymentCurrency": "10.00", "AccountingDocument": "0000000042"},
            {"AssignmentReference": "1E3", "GrossAmountInPaymentCurrency": "-1.00"}]}
        stored = {"CompanyCode": "1000", "items": [
            {"PaymentAdviceItem": "00002", "AssignmentReference": "1000", "GrossAmountInPaymentCurrency": "-1.000"},
            {"PaymentAdviceItem": "00001", "DocumentReferenceID": "90", "GrossAmountInPaymentCurrency": "10", "AccountingDocument": "42"}]}
        self.assertEqual(compare_read_back(payload, stored)["differences"],
                         ["Item 1 DocumentReferenceID: sent '0090', S/4 has '90'", "Item 2 AssignmentReference: sent '1E3', S/4 has '1000'"])


if __name__ == "__main__":
    unittest.main()
