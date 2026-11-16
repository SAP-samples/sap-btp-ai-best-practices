"""Offline checks for choosing one S/4 document among several candidates."""
import unittest
from decimal import Decimal

from app.s4.lookup import ReceivableItem
from app.s4.resolution import LineContext, apply_linked, payable_in_scope, resolve

REF = "8700838001"


def item(doc="1800000101", amount="1000.00", cleared=False, company_code="1000", customer="0010000001", year="2026",
         by="DocumentReferenceID", number="001", **extra):
    """Build one S/4 candidate; open and live unless said otherwise."""
    return ReceivableItem(REF, by, company_code, year, doc, number, customer, REF, Decimal(amount), "EUR", cleared,
                          "1500000009" if cleared else "", **extra)


def context(gross="1000.00", **overrides):
    """Line context for a positive invoice line of 1,000.00."""
    amount = None if gross is None else Decimal(gross)
    values = {"gross": amount, "sign": 0 if amount is None else (amount > 0) - (amount < 0)}
    return LineContext(**{**values, **overrides})


def documents(resolution):
    """Document numbers of the chosen items."""
    return [chosen.accounting_document for chosen in resolution.items]


def rules(resolution):
    """Names of the rules that removed candidates."""
    return [step["rule"] for step in resolution.steps]


class UniqueTests(unittest.TestCase):
    """One candidate needs no rule; none gives nothing."""
    def test_single_candidate_is_resolved_without_warning(self):
        result = resolve([item()], context())
        self.assertEqual((result.status, result.method, result.warning, result.steps), ("resolved", "unique", None, []))

    def test_no_candidate(self):
        self.assertEqual(resolve([], context()).status, "none")

    def test_single_cleared_candidate_is_still_the_answer(self):
        result = resolve([item(cleared=True)], context())
        self.assertEqual((result.status, result.items[0].is_cleared), ("resolved", True))


class ScopeTests(unittest.TestCase):
    """Company code, customer and sign decide before open or cleared does."""
    def test_rule_company_code_separates_the_same_document_number(self):
        same_number = [item(company_code="1000"), item(company_code="2000", customer="0020000002")]
        result = resolve(same_number, context(company_code="2000"))
        self.assertEqual((result.items[0].company_code, result.method, result.warning, rules(result)),
                         ("2000", "auto", None, ["company_code"]))

    def test_own_cleared_invoice_beats_another_customers_open_document(self):
        ours = item(doc="1800000101", cleared=True)
        foreign = item(doc="1800000999", customer="0020000002")
        result = resolve([ours, foreign], context(account="0010000001"))
        self.assertEqual((documents(result), result.items[0].is_cleared), (["1800000101"], True))

    def test_branch_account_counts_as_the_rule_customer(self):
        head_office = item(doc="1800000101", customer="0010000009", branch_account="0010000001")
        other = item(doc="1800000999", customer="0020000002")
        self.assertEqual(documents(resolve([head_office, other], context(account="0010000001"))), ["1800000101"])

    def test_sign_separates_an_invoice_from_the_credit_memo_sharing_its_reference(self):
        invoice, credit = item(doc="1800000101"), item(doc="1700000050", amount="-120.00")
        self.assertEqual(documents(resolve([invoice, credit], context("-120.00", account="0010000001"))), ["1700000050"])
        self.assertEqual(documents(resolve([invoice, credit], context("1000.00", account="0010000001"))), ["1800000101"])

    def test_sign_is_decided_before_state_so_a_cleared_invoice_is_not_replaced_by_an_open_credit(self):
        invoice, credit = item(doc="1800000101", cleared=True), item(doc="1700000050", amount="-120.00")
        result = resolve([invoice, credit], context(account="0010000001"))
        self.assertEqual((documents(result), result.items[0].is_cleared), (["1800000101"], True))

    def test_company_code_agreed_by_the_other_lines_is_used_only_without_a_rule(self):
        pair = [item(company_code="1000"), item(company_code="2000", customer="0020000002")]
        self.assertEqual(resolve(pair, context(), consensus="2000").items[0].company_code, "2000")
        self.assertEqual(resolve(pair, context(company_code="1000"), consensus="2000").items[0].company_code, "1000")

    def test_rule_customer_is_applied_before_the_company_code(self):
        foreign = item(doc="1800000999", company_code="1000", customer="0020000002")
        ours = item(doc="1800000101", company_code="2000")
        result = resolve([foreign, ours], context(company_code="1000", account="0010000001"))
        self.assertEqual((result.items[0].company_code, rules(result), result.warning), ("2000", ["customer"], None))


class CrossCompanyCodeTests(unittest.TestCase):
    """The company code prefers a document; it does not rule out one the same customer could pay elsewhere."""
    def test_same_customer_and_amount_in_two_company_codes_goes_to_the_reviewer(self):
        pair = [item(company_code="1000"), item(company_code="2000")]
        result = resolve(pair, context(company_code="1000"))
        self.assertEqual((result.status, len(result.candidates)), ("ambiguous", 2))

    def test_amount_decides_between_the_company_codes_of_one_customer(self):
        pair = [item(company_code="1000"), item(company_code="2000", amount="500.00")]
        inside = resolve(pair, context("1000.00", company_code="1000"))
        outside = resolve(pair, context("500.00", company_code="1000"))
        self.assertEqual((inside.items[0].company_code, outside.items[0].company_code), ("1000", "2000"))
        self.assertIn("only candidate whose amount", inside.warning)
        self.assertIn("in company code 1000 as well", outside.warning)

    def test_open_document_elsewhere_replaces_the_cleared_one_only_with_the_amount(self):
        pair = [item(company_code="1000", cleared=True), item(company_code="2000", amount="750.00")]
        confirmed = resolve(pair, context("750.00", company_code="1000"))
        self.assertEqual((confirmed.items[0].company_code, confirmed.method), ("2000", "auto"))
        self.assertIn("in company code 1000 as well", confirmed.warning)
        self.assertEqual(resolve(pair, context("999.00", company_code="1000")).status, "ambiguous")

    def test_agreed_company_code_follows_the_same_rule(self):
        pair = [item(company_code="1000"), item(company_code="2000")]
        self.assertEqual(resolve(pair, context(), consensus="1000").status, "ambiguous")

    def test_document_of_the_other_side_elsewhere_is_no_rival(self):
        pair = [item(company_code="1000"), item(company_code="2000", doc="1700000050", amount="-120.00")]
        result = resolve(pair, context(company_code="1000"))
        self.assertEqual((result.items[0].company_code, rules(result)), ("1000", ["company_code"]))

    def test_cleared_or_special_document_elsewhere_is_no_rival(self):
        for other in (item(company_code="2000", cleared=True), item(company_code="2000", special_gl="A")):
            result = resolve([item(company_code="1000"), other], context(company_code="1000"))
            self.assertEqual((result.items[0].company_code, result.warning), ("1000", None))


class StateTests(unittest.TestCase):
    """Inside the scope: live before reversed, open before cleared, normal before special G/L."""
    def test_open_document_wins_over_cleared_ones_of_the_same_customer(self):
        result = resolve([item(doc="1800000001", cleared=True, year="2025"), item(doc="1800000101")],
                         context(account="0010000001"))
        self.assertEqual((documents(result), result.warning, rules(result)), (["1800000101"], None, ["open"]))

    def test_special_gl_item_loses_against_a_normal_item(self):
        result = resolve([item(doc="1600000001", special_gl="A"), item(doc="1800000101")], context(account="0010000001"))
        self.assertEqual((documents(result), rules(result)), (["1800000101"], ["normal_item"]))

    def test_agreed_company_code_does_not_confirm_the_customer(self):
        shared = [item(doc="1800000001", cleared=True, customer="0020000002"), item(doc="1800000101")]
        self.assertIn("no customer rule confirms", resolve(shared, context(), consensus="1000").warning)
        self.assertIsNone(resolve(shared, context(account="0010000001"), consensus="1000").warning)

    def test_rule_customer_does_not_confirm_the_company_code(self):
        two_codes = [item(doc="1800000001", cleared=True, company_code="2000"), item(doc="1800000101")]
        self.assertIn("no customer rule confirms", resolve(two_codes, context(account="0010000001")).warning)
        self.assertIsNone(resolve(two_codes, context(account="0010000001", company_code="1000")).warning)

    def test_without_any_rule_a_foreign_candidate_raises_a_warning(self):
        result = resolve([item(doc="1800000001", cleared=True, customer="0020000002"), item(doc="1800000101")], context())
        self.assertEqual(documents(result), ["1800000101"])
        self.assertIn("no customer rule confirms", result.warning)


class TieBreakTests(unittest.TestCase):
    """The amount may decide, with a warning; nothing else may."""
    def test_amount_decides_between_two_open_documents_and_warns(self):
        result = resolve([item(doc="1800000101", amount="250.00"), item(doc="1800000102")], context(account="0010000001"))
        self.assertEqual((documents(result), rules(result)), (["1800000102"], ["open_amount"]))
        self.assertIn("amount equals the advice amount", result.warning)

    def test_equal_candidates_stay_ambiguous_and_are_offered_to_the_reviewer(self):
        result = resolve([item(doc="1800000101"), item(doc="1800000102", year="2025")], context(account="0010000001"))
        self.assertEqual((result.status, result.items, len(result.candidates)), ("ambiguous", [], 2))

    def test_missing_gross_never_decides(self):
        result = resolve([item(doc="1800000101", amount="0.00"), item(doc="1800000102")], context(None))
        self.assertEqual(result.status, "ambiguous")

    def test_reference_field_wins_over_document_number_when_both_can_be_paid(self):
        by_reference, by_number = item(doc="1800000101"), item(doc="8700838001", by="AccountingDocument", amount="5.00")
        result = resolve([by_reference, by_number], context(account="0010000001"))
        self.assertEqual((documents(result), result.warning, rules(result)), (["1800000101"], None, ["field_priority"]))

    def test_lower_field_replaces_a_cleared_reference_hit_only_when_the_amount_agrees(self):
        cleared = item(doc="1800000101", cleared=True)
        by_number = item(doc="8700838001", by="AccountingDocument")
        agreed = resolve([cleared, by_number], context(account="0010000001"))
        self.assertEqual(documents(agreed), ["8700838001"])
        self.assertIn("found by document number", agreed.warning)
        disputed = resolve([cleared, by_number], context("400.00", account="0010000001"))
        self.assertEqual((disputed.status, len(disputed.candidates)), ("ambiguous", 2))

    def test_variant_spelling_needs_the_amount(self):
        found = item(variant="8700838001X")
        self.assertIn("read as", resolve([found], context()).warning)
        suggested = resolve([found], context("999.00"))
        self.assertEqual((suggested.status, len(suggested.candidates)), ("suggested", 1))


class PickTests(unittest.TestCase):
    """The reviewer's pick wins while it is still a candidate."""
    def test_pick_wins_over_every_rule(self):
        pair = [item(doc="1800000101"), item(doc="1800000102", year="2025", cleared=True)]
        result = resolve(pair, context(pick="1000/1800000102/2025"))
        self.assertEqual((documents(result), result.method, rules(result)), (["1800000102"], "reviewer", ["reviewer_pick"]))

    def test_pick_that_is_no_longer_a_candidate_is_stale(self):
        result = resolve([item(doc="1800000101")], context(pick="1000/1800000777/2026"))
        self.assertEqual((result.status, len(result.candidates)), ("stale_pick", 1))
        self.assertEqual(resolve([item()], context(pick="not a pick")).status, "stale_pick")

    def test_pick_names_one_item_of_a_document(self):
        instalments = [item(number="001", amount="500.00"), item(number="002", amount="500.00")]
        result = resolve(instalments, context("500.00", pick="1000/1800000101/2026/2"))
        self.assertEqual(([i.accounting_document_item for i in result.items], result.item_targeted), (["002"], True))


class MultiItemTests(unittest.TestCase):
    """A document with several customer items (instalments) is paid by item."""
    def instalments(self):
        return [item(number="001", amount="300.00"), item(number="002", amount="700.00")]

    def test_item_whose_open_amount_equals_the_gross_is_targeted(self):
        result = resolve(self.instalments(), context("700.00"))
        self.assertEqual(([i.accounting_document_item for i in result.items], result.item_targeted, result.method),
                         (["002"], True, "unique"))

    def test_document_total_pays_every_item_when_the_line_is_plain(self):
        result = resolve(self.instalments(), context("1000.00"))
        self.assertEqual(([i.accounting_document_item for i in result.items], result.item_targeted), (["001", "002"], True))

    def test_anything_else_is_left_to_the_reviewer(self):
        self.assertEqual(resolve(self.instalments(), context("1000.00", plain=False)).status, "multi_item")
        equal = [item(number="001", amount="500.00"), item(number="002", amount="500.00")]
        result = resolve(equal, context("500.00"))
        self.assertEqual((result.status, len(result.candidates)), ("multi_item", 2))

    def test_single_open_item_of_a_partly_cleared_document(self):
        result = resolve([item(number="001", amount="300.00", cleared=True), item(number="002", amount="700.00")], context("700.00"))
        self.assertEqual(([i.accounting_document_item for i in result.items], result.item_targeted), (["002"], True))


class LinkedItemTests(unittest.TestCase):
    """Partial payments and residual items decide what is really open."""
    def linked(self, doc, amount, kind, cleared=False, **extra):
        return item(doc=doc, amount=amount, cleared=cleared, by="InvoiceReference", invoice_reference="1800000101",
                    invoice_reference_year="2026", follow_on_type=kind, **extra)

    def test_open_partial_payments_reduce_the_open_amount(self):
        links = [self.linked("1400000001", "-400.00", "Z"), self.linked("1400000002", "-100.00", "Z", cleared=True)]
        enriched, = apply_linked({REF: [item()]}, links)[REF]
        self.assertEqual((enriched.open_amount, enriched.partial_payments, enriched.amount),
                         (Decimal("600.00"), Decimal("-400.00"), Decimal("1000.00")))

    def test_open_residual_item_replaces_the_cleared_invoice(self):
        residual = self.linked("1400000003", "150.00", "V", number="002")
        enriched, = apply_linked({REF: [item(cleared=True)]}, [residual])[REF]
        self.assertEqual((enriched.accounting_document, enriched.residual_of, enriched.open_amount, enriched.matched_by,
                          enriched.reference), ("1400000003", "1800000101/2026", Decimal("150.00"), "DocumentReferenceID", REF))
        result = resolve([enriched], context("150.00"))
        self.assertEqual((result.status, result.item_targeted), ("resolved", True))

    def test_residual_item_reached_through_its_invoice_and_directly_is_one_candidate(self):
        # The residual item inherits the billing document number, so the lookup can return it on its own as well.
        residual = self.linked("1400000003", "150.00", "V", number="002")
        direct = item(doc="1400000003", amount="150.00", number="002", by="BillingDocument", invoice_reference="1800000101",
                      invoice_reference_year="2026", follow_on_type="V")
        enriched, = apply_linked({REF: [item(cleared=True), direct]}, [residual])[REF]
        self.assertEqual((enriched.matched_by, enriched.residual_of), ("DocumentReferenceID", "1800000101/2026"))
        result = resolve([enriched], context("150.00"))
        self.assertEqual((result.status, result.item_targeted, result.warning), ("resolved", True, None))

    def test_residual_item_found_only_directly_is_marked_and_named_by_item(self):
        direct = item(doc="1400000003", amount="150.00", number="002", by="BillingDocument", invoice_reference="1800000101",
                      invoice_reference_year="2026", follow_on_type="V")
        enriched, = apply_linked({REF: [direct]}, [])[REF]
        self.assertEqual(enriched.residual_of, "1800000101/2026")
        self.assertTrue(resolve([enriched], context("150.00")).item_targeted)

    def test_residual_item_chosen_by_the_reviewer_is_named_by_item_too(self):
        residual = item(doc="1400000003", amount="150.00", number="002", invoice_reference="1800000101",
                        invoice_reference_year="2026", follow_on_type="V", residual_of="1800000101/2026")
        other = item(doc="1800000999", amount="150.00")
        result = resolve([residual, other], context("150.00", pick="1000/1400000003/2026/002"))
        self.assertEqual((documents(result), result.method, result.item_targeted), (["1400000003"], "reviewer", True))

    def test_credit_memo_with_invoice_reference_is_not_netted(self):
        credit = self.linked("1700000050", "-120.00", "G")
        enriched, = apply_linked({REF: [item()]}, [credit])[REF]
        self.assertEqual(enriched.open_amount, Decimal("1000.00"))

    def test_links_of_another_company_code_or_year_are_ignored(self):
        links = [self.linked("1400000001", "-400.00", "Z", company_code="2000"),
                 item(doc="1400000002", amount="-400.00", by="InvoiceReference", invoice_reference="1800000101",
                      invoice_reference_year="2025", follow_on_type="Z")]
        enriched, = apply_linked({REF: [item()]}, links)[REF]
        self.assertEqual(enriched.open_amount, Decimal("1000.00"))

    def test_link_naming_another_item_of_the_document_is_ignored(self):
        other_item = self.linked("1400000001", "-400.00", "Z", invoice_item_reference="002")
        same_item = self.linked("1400000002", "-100.00", "Z", invoice_item_reference="001")
        enriched, = apply_linked({REF: [item()]}, [other_item, same_item])[REF]
        self.assertEqual(enriched.open_amount, Decimal("900.00"))

    def test_cleared_invoice_without_residual_is_open_for_nothing(self):
        enriched, = apply_linked({REF: [item(cleared=True)]}, [])[REF]
        self.assertEqual((enriched.open_amount, enriched.is_cleared), (Decimal("0"), True))


class ScopePredicateTests(unittest.TestCase):
    """The lookup stops asking further fields once a payable item in scope exists."""
    def test_payable_item_must_fit_company_code_and_customer(self):
        self.assertTrue(payable_in_scope([item()], None, set()))
        self.assertTrue(payable_in_scope([item()], "1000", {"0010000001"}))
        self.assertFalse(payable_in_scope([item()], "2000", set()))
        self.assertFalse(payable_in_scope([item()], "1000", {"0020000002"}))
        self.assertFalse(payable_in_scope([item(cleared=True)], None, set()))
        self.assertFalse(payable_in_scope([item(special_gl="A")], None, set()))
        self.assertFalse(payable_in_scope([], None, set()))


if __name__ == "__main__":
    unittest.main()
