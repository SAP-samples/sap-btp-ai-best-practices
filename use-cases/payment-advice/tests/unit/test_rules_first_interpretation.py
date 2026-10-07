"""Offline checks that documented rules run before the deduction agent and only leftovers reach it."""
import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.payment_advice.interpretation_service import interpret_events

ANCHORS = json.loads((Path(__file__).parent / "fixtures" / "playbooks" / "contoso_golden.json").read_text())["anchors"]
SCHEMA = [{"name": "document_no", "label": "Document No", "scope": "line"},
          {"name": "your_invoice_no", "label": "Your Invoice No", "scope": "line"},
          {"name": "region", "label": "Rg", "scope": "line"},
          {"name": "invoice_date", "label": "Invoice Date", "scope": "line"},
          {"name": "gross", "label": "Gross", "scope": "line"},
          {"name": "discount", "label": "Discount", "scope": "line"},
          {"name": "net", "label": "Net", "scope": "line"}]
ROWS = [("8700276154", "A", 10596.28),       # plain supplier invoice: no agent
        ("924610168612UL", "O", -81.09),      # deduction with a specific rule: settled by the rule
        ("4610868892", "A", -255.00),         # deduction, only the default applies: agent
        ("294610775961UL", "A", 82.52)]       # positive but matches a deduction rule: agent with hint


class FakeRuntime:
    """Records the prompt and answers one interpretation per routed line."""
    def __init__(self):
        self.prompts = []

    async def ainvoke(self, text, context_id, response_model):
        self.prompts.append(text)
        lines = json.loads(text[text.index("["):])
        entries = [{"line_index": line["line_index"], "document_nature": "allowance_reversal" if line["gross_amount"] > 0
                    else "chargeback", "reason_code": "316" if line["gross_amount"] > 0 else "323", "residual_items": [],
                    "flags": [], "rationale": "agent"} for line in lines]
        return SimpleNamespace(output_parsed={"interpretations": entries}, messages=[])


def run(runtime, anchors=ANCHORS, swapped=False):
    """Interpret the sample advice and return (events by type, enriched result).

    ``swapped`` simulates the LLM mapper confusing "Document No" and "Your Invoice No".
    """
    raw_rows = [{"document_no": str(600000000 + i), "your_invoice_no": r, "region": g, "invoice_date": "2025-09-01",
                 "gross": gross, "discount": 0.0, "net": gross} for i, (r, g, gross) in enumerate(ROWS)]
    numbers, invoices = ("your_invoice_no", "document_no") if swapped else ("document_no", "your_invoice_no")
    mapping = {numbers: "customer_document_reference", invoices: "invoice_reference",
               "gross": "gross_amount", "net": "net_amount"}
    payload = {"header": {"payment_reference": "2000366233", "payment_amount": 10262.71},
               "line_items": [{field: row[column] for column, field in mapping.items()} for row in raw_rows]}
    raw = {"schema_fields": SCHEMA, "header": {}, "line_items": raw_rows, "mapping": {"line_items": mapping}}

    async def collect():
        return [json.loads(line) async for line in interpret_events(
            "contoso", payload, runtime=runtime, engine=None, playbook_revision=3, raw_extraction=raw, anchors=anchors)]
    events = {event["type"]: event for event in asyncio.run(collect())}
    return events, events["result"]["enriched"]


class RulesFirstTests(unittest.TestCase):
    """Rules settle what the documents state; the agent judges only the rest."""
    def test_only_unsettled_lines_reach_the_agent_with_hints(self):
        runtime = FakeRuntime()
        events, enriched = run(runtime)
        self.assertEqual(events["analysis"]["analyzed_indexes"], [2, 3])
        self.assertEqual(events["analysis"]["rule_resolved_count"], 1)
        self.assertEqual(events["analysis"]["passthrough_count"], 1)
        prompt_lines = json.loads(runtime.prompts[0][runtime.prompts[0].index("["):])
        self.assertIn("matches documented deduction rule", prompt_lines[1]["rule_hint"])
        self.assertEqual({k: prompt_lines[1]["source_columns"][k] for k in ("your_invoice_no", "region")},
                         {"your_invoice_no": "294610775961UL", "region": "A"})

        lines = enriched["line_items"]
        self.assertEqual((lines[0]["document_nature"], lines[0]["reason_code"]), ("invoice", None))
        self.assertEqual((lines[1]["reason_code"], lines[1]["document_nature"]), ("316", "chargeback"))
        self.assertIn("Documented customer rule invoice_number", lines[1]["rationale"])
        self.assertEqual((lines[3]["document_nature"], lines[3]["reason_code"]), ("allowance_reversal", "316"))

    def test_rule_derived_accounts_and_summary(self):
        _, enriched = run(FakeRuntime())
        self.assertEqual([line["customer_account"] for line in enriched["line_items"]],
                         ["10059152", "10058745", "10059152", "10059152"])
        header = enriched["header"]
        self.assertEqual(header["payer_account"], "10053628")
        self.assertEqual(header["interpretation"]["rule_resolved_count"], 1)
        self.assertEqual(header["interpretation"]["rule_warnings"], [])
        self.assertEqual(header["interpretation"]["playbook_revision"], 3)

    def test_documented_mapping_repairs_a_swapped_invoice_number(self):
        """The mapper put Document No into invoice_reference; the documented "Your Invoice No" wins."""
        runtime = FakeRuntime()
        events, enriched = run(runtime, swapped=True)
        lines = enriched["line_items"]
        self.assertEqual([line["invoice_reference"] for line in lines], [ref for ref, _, _ in ROWS])
        self.assertEqual(lines[0]["customer_document_reference"], "600000000")
        # The same routing as with a correct mapping: the UL deduction is settled by its rule.
        self.assertEqual(events["analysis"]["analyzed_indexes"], [2, 3])
        self.assertEqual(events["analysis"]["rule_resolved_count"], 1)
        summary = enriched["header"]["interpretation"]
        self.assertEqual(summary["documented_mappings"]["invoice_reference"], "Your Invoice No")
        self.assertEqual(summary["rule_warnings"], [])

    def test_without_rules_behaviour_is_unchanged(self):
        runtime = FakeRuntime()
        events, enriched = run(runtime, anchors={})
        self.assertEqual(events["analysis"]["analyzed_indexes"], [1, 2])
        self.assertEqual(enriched["line_items"][3]["document_nature"], "invoice")
        self.assertIsNone(enriched["line_items"][0]["customer_account"])


if __name__ == "__main__":
    unittest.main()
