"""Offline checks for the deterministic customer rule engine (rules evaluated on raw columns)."""
import json
import unittest
from pathlib import Path

from app.payment_advice.rule_engine import apply_documented_mappings, apply_rules, matches, normalize_label

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "playbooks" / "contoso_golden.json").read_text())["anchors"]
# Raw view as Document AI returns it for the region-coded remittance: field "region" printed as "Rg".
REGION_SCHEMA = [{"name": "your_invoice_no", "label": "Your Invoice No", "type": "string", "scope": "line"},
                 {"name": "region", "label": "Rg", "type": "string", "scope": "line"},
                 {"name": "document_no", "label": "Document No.", "type": "string", "scope": "header"}]


def advice(rows):
    """Build (payload, raw_extraction) from (invoice reference, region, gross) rows."""
    lines = [{"invoice_reference": ref, "gross_amount": gross, "net_amount": gross} for ref, _, gross in rows]
    raw = [{"your_invoice_no": ref, **({"region": rg} if rg else {})} for ref, rg, _ in rows]
    return {"header": {"payment_reference": "2000976003"}, "line_items": lines}, \
        {"schema_fields": REGION_SCHEMA, "header": {"document_no": "2000976003"}, "line_items": raw}


class OperatorTests(unittest.TestCase):
    """Comparisons are trimmed and case-insensitive; bad rules raise."""
    def test_operators(self):
        self.assertTrue(matches("equals", " a ", "A"))
        self.assertTrue(matches("starts_with", "WEEK  END 12", "week end"))
        self.assertTrue(matches("ends_with", "294610493842ul", "UL"))
        self.assertTrue(matches("contains", "323763-202620", "-202"))
        self.assertTrue(matches("regex", "9304911684SCRSC", r"^\d{10}SC"))
        self.assertFalse(matches("equals", None, "A"))
        with self.assertRaises(ValueError):
            matches("between", "1", "2")

    def test_label_normalization(self):
        self.assertEqual(normalize_label("Column Rg"), "rg")
        self.assertEqual(normalize_label("item.invoice_number"), "invoice_number")
        self.assertEqual(normalize_label("Your Invoice No"), "your_invoice_no")


class RegionAndReasonRulesTests(unittest.TestCase):
    """Golden region-coded customer: Rg selects the account, invoice-number rules the reason code."""
    def test_region_column_selects_account_and_payer_is_the_default(self):
        payload, raw = advice([("8700276154", "A", 10596.28), ("924610168612UL", "O", -81.09),
                               ("8700427315", "W", 18689.92), ("REPLEN COMP JULY", "X", -381.30)])
        outcome = apply_rules(payload, raw, GOLDEN)
        self.assertEqual([line["customer_account"] for line in outcome["lines"]],
                         ["10059152", "10058745", "10053628", "10053628"])
        self.assertEqual(outcome["lines"][0]["customer_account_rule"], "Rg equals 'A' (priority 3)")
        self.assertEqual(outcome["lines"][3]["flags"], ["customer-account-defaulted"])
        self.assertEqual(outcome["header"]["payer_account"], "10053628")
        self.assertEqual(outcome["warnings"], [])

    def test_documented_reason_rules_and_lines_left_for_the_agent(self):
        refs = {"924610168612UL": "316", "294610493842UL": "316", "323763-202620": "319", "REPLEN COMP JULY": "316",
                "504610611121": "316", "WEEK END 12": "319", "XYZ-FS1": "323",
                "4610868892": None, "0046115053": None, "8700276154": None}
        payload, raw = advice([(ref, "A", -1.0) for ref in refs])
        outcome = apply_rules(payload, raw, GOLDEN)
        got = {ref: (line["deduction"] or {}).get("reason_code") for ref, line in zip(refs, outcome["lines"])}
        self.assertEqual(got, refs)
        self.assertIn("ends_with 'UL'", outcome["lines"][0]["deduction"]["rule"])

    def test_missing_region_column_is_reported_not_ignored(self):
        payload, raw = advice([("8700276154", None, 10.0)])
        raw["schema_fields"] = [f for f in REGION_SCHEMA if f["name"] != "region"]
        outcome = apply_rules(payload, raw, GOLDEN)
        self.assertEqual(outcome["lines"][0]["customer_account"], "10053628")
        self.assertEqual(outcome["lines"][0]["flags"], ["customer-account-defaulted"])
        self.assertIn("rule field 'Rg' was not found in the extracted columns", outcome["warnings"])


class OtherCustomerPatternsTests(unittest.TestCase):
    """The same engine expresses other customers' documented logic."""
    def test_suffix_rules_before_description_rules(self):
        anchors = {"default_customer_account": "35931406", "deduction_rules": [
            {"priority": 1, "field": "invoice_number", "operator": "ends_with", "value": "SCRSC", "reason_code": "307"},
            {"priority": 2, "field": "invoice_number", "operator": "ends_with", "value": "PCR", "reason_code": "323"},
            {"priority": 10, "field": "description", "operator": "starts_with", "value": "SPA", "reason_code": "320"},
            {"priority": 11, "field": "description", "operator": "starts_with", "value": "Co-op", "reason_code": "319"}]}
        schema = [{"name": "invoice_number", "label": "Invoice Number", "scope": "line"},
                  {"name": "invoice_description", "label": "Invoice Description", "scope": "line"}]
        raw_rows = [{"invoice_number": "9304283483SCRSC", "invoice_description": "Shortage Claim"},
                    {"invoice_number": "5401219529", "invoice_description": "SPA-104498845-5401219529"},
                    {"invoice_number": "3222045885-SE", "invoice_description": "Co-op-116803235"}]
        payload = {"header": {}, "line_items": [{"invoice_reference": r["invoice_number"]} for r in raw_rows]}
        outcome = apply_rules(payload, {"schema_fields": schema, "header": {}, "line_items": raw_rows}, anchors)
        self.assertEqual([line["deduction"]["reason_code"] for line in outcome["lines"]], ["307", "320", "319"])
        self.assertEqual({line["customer_account"] for line in outcome["lines"]}, {"35931406"})
        self.assertEqual([line["flags"] for line in outcome["lines"]], [[], [], []])

    def test_company_code_from_header_drives_the_account(self):
        anchors = {"default_customer_account": None,
                   "company_code_rules": [{"priority": 1, "field": "Sales Org. Id", "operator": "equals",
                                           "value": "DE01", "company_code": "DE01"}],
                   "customer_account_rules": [
                       {"priority": 1, "field": "company_code", "operator": "equals", "value": "US01", "customer_account": "72712269"},
                       {"priority": 2, "field": "company_code", "operator": "equals", "value": "DE01", "customer_account": "72443356"}]}
        raw = {"schema_fields": [{"name": "sales_org_id", "label": "Sales Org. Id", "scope": "header"}],
               "header": {"sales_org_id": "DE01"}, "line_items": [{}, {}]}
        outcome = apply_rules({"header": {}, "line_items": [{}, {}]}, raw, anchors)
        self.assertEqual(outcome["header"]["company_code"], "DE01")
        self.assertEqual(outcome["header"]["payer_account"], "72443356")
        self.assertEqual([line["customer_account"] for line in outcome["lines"]], ["72443356", "72443356"])

    def test_constant_company_code_and_no_rules(self):
        self.assertEqual(apply_rules({"header": {}, "line_items": []}, None, {"default_company_code": "CA01"})["header"],
                         {"payer_account": None, "company_code": "CA01", "company_code_rule": "default company code"})
        outcome = apply_rules({"header": {}, "line_items": [{"invoice_reference": "1"}]}, None, None)
        self.assertEqual(outcome["lines"], [{"customer_account": None, "customer_account_rule": None,
                                             "flags": [], "deduction": None}])
        self.assertEqual(outcome["warnings"], [])

    def test_company_code_from_constant_header_mapping(self):
        """Older playbooks state the company code only as the ILN recipient / payee constant."""
        def header(mappings, **extra):
            return apply_rules({"header": {}, "line_items": []}, None, {"field_mappings": mappings, **extra})["header"]
        stored = [{"target_field": "iln_recipient", "source_field": 'Fixed value "CA01"', "transformation": "constant"}]
        self.assertEqual((header(stored)["company_code"], header(stored)["company_code_rule"]),
                         ("CA01", "constant header mapping 'iln_recipient'"))
        self.assertEqual(header([{"target_field": "payee", "source_field": "constant", "transformation": "de01"}])["company_code"], "DE01")
        self.assertEqual(header(stored, default_company_code="NL01")["company_code"], "NL01")
        self.assertIsNone(header([{"target_field": "payee", "source_field": "constant", "transformation": "Example Lighting Canada"}])["company_code"])

    def test_documented_mappings_override_the_mapper(self):
        """Stored item-record mappings pin the columns; conflicts with the LLM mapper are repaired."""
        anchors = {"field_mappings": [
            {"target_field": "invoice_number", "source_field": "Your Invoice No", "transformation": "column mapping"},
            {"target_field": "payment_amount", "source_field": "Net", "transformation": "column mapping"},
            {"target_field": "invoice_date", "source_field": "Inv. Date", "transformation": "column mapping"},
            {"target_field": "iln_recipient", "source_field": 'Fixed value "CA01"', "transformation": "constant"},
            {"target_field": "market_name", "source_field": "Rg", "transformation": "populate from account"}]}
        schema = [{"name": "document_no", "label": "Document No", "scope": "line"},
                  {"name": "your_invoice_no", "label": "Your Invoice No", "scope": "line"},
                  {"name": "net", "label": "Net", "scope": "line"},
                  {"name": "grand_total_net", "label": "Grand Total - Net", "scope": "header"}]
        raw = {"schema_fields": schema, "header": {"grand_total_net": "18,316.12"},
               "line_items": [{"document_no": "5100818221", "your_invoice_no": "8700427315", "net": "18,316.12"}],
               "mapping": {"line_items": {"document_no": "invoice_reference", "your_invoice_no": "customer_document_reference",
                                          "net": "net_amount"}}}
        payload = {"header": {}, "line_items": [{"invoice_reference": "5100818221",
                                                 "customer_document_reference": "8700427315", "net_amount": 18316.12}]}
        fixed, documented, warnings = apply_documented_mappings(payload, raw, anchors)
        self.assertEqual(fixed["line_items"][0], {"invoice_reference": "8700427315",
                                                  "customer_document_reference": "5100818221", "net_amount": 18316.12})
        self.assertEqual(documented, {"invoice_reference": "Your Invoice No", "net_amount": "Net"})
        self.assertEqual(warnings, ["documented column 'Inv. Date' for invoice_date was not found among the "
                                    "extracted line columns; the mapper's choice was kept"])
        self.assertEqual(payload["line_items"][0]["invoice_reference"], "5100818221")  # input not mutated
        self.assertEqual(apply_documented_mappings(payload, None, anchors), (payload, {}, []))

    def test_invalid_rule_is_reported(self):
        anchors = {"deduction_rules": [{"priority": 1, "field": "invoice_number", "operator": "between", "value": "1"}]}
        outcome = apply_rules({"header": {}, "line_items": [{"invoice_reference": "1"}]}, None, anchors)
        self.assertIsNone(outcome["lines"][0]["deduction"])
        self.assertTrue(any("unknown operator" in w for w in outcome["warnings"]))


if __name__ == "__main__":
    unittest.main()
