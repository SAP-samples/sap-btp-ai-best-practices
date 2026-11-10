"""
Unit tests for build_uc02_tools() — the LangChain tool wrappers for UC-02.

Verifies:
  - Exactly 8 tools are returned with the expected names (the full-document tool
    set introduced in UC-02 iteration 2).
  - Full-document tools read the payload bound to the current ContextVar, not
    from an explicit payload argument.
  - document_statistics reports the correct line count.
  - list_invoice_references reports the correct total.
  - fetch_invoices retrieves a line by invoice_reference.

Example run:
    PYTHONPATH=api .venv/bin/python -m unittest tests.unit.test_uc02_tools -v
"""

import unittest

from app.deduction_agent.tools.uc02_tools import build_uc02_tools
from app.deduction_agent.document_context import (
    set_current_document,
    reset_current_document,
)
from tests.unit.fakes import FakeEngine


class ToolsTest(unittest.TestCase):
    def test_tool_set_names(self):
        """Agents can inspect evidence but cannot invoke combinatorial matching."""
        tools = {t.name for t in build_uc02_tools(FakeEngine(rows=[]))}
        self.assertEqual(
            tools,
            {
                "find_customers",
                "get_customer",
                "get_deduction_rules",
                "save_deduction_rules",
                "delete_deduction_rules",
                "document_statistics",
                "list_invoice_references",
                "fetch_invoices",
            },
        )

    def test_lookup_reads_bound_document(self):
        """Full-document tools read the ContextVar-bound payload, not a passed arg."""
        payload = {
            "line_items": [
                {"invoice_reference": "A", "gross_amount": 50.0},
                {"invoice_reference": "B", "gross_amount": 50.0},
                {"invoice_reference": "P", "gross_amount": -100.0},
            ]
        }
        tools = {t.name: t for t in build_uc02_tools(FakeEngine(rows=[]))}
        token = set_current_document(payload)
        try:
            # document_statistics: 3 total lines
            stats = tools["document_statistics"].invoke({})
            self.assertEqual(stats["total_lines"], 3)

            # list_invoice_references: 3 lines have invoice_reference
            refs = tools["list_invoice_references"].invoke({})
            self.assertEqual(refs["total"], 3)

            # fetch_invoices: fetch line P by string reference
            fetched = tools["fetch_invoices"].invoke({"references": "P"})
            self.assertEqual(fetched["invoices"][0]["invoice_reference"], "P")
        finally:
            reset_current_document(token)


if __name__ == "__main__":
    unittest.main()
