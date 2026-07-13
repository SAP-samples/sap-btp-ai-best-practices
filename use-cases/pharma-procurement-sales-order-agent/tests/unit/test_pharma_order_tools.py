from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.tools.pharma_order.sap_mock_tools import (
    get_invoice_pdf,
    get_order_status,
    get_pricing_for_customer_material,
    list_blocked_orders,
    lookup_customer_by_dea,
    lookup_material_by_ndc,
    set_or_clear_order_block,
)


class PharmaOrderToolTests(unittest.TestCase):
    def test_pricing_tool_returns_synthetic_context(self):
        result = get_pricing_for_customer_material("Northstar", "Glycemor 10 mg", 1)
        self.assertEqual(result["data_status"], "synthetic_demo_data")
        self.assertGreaterEqual(len(result["data"]["results"]), 1)

    def test_order_related_tools_return_records(self):
        self.assertIn("data", get_order_status(customer_name="Northstar"))
        self.assertIn("data", list_blocked_orders(customer_name="MetroMed Wholesale"))
        self.assertIn("preview_only_no_update", str(set_or_clear_order_block(sales_order="5000001234")))

    def test_compliance_material_and_invoice_tools_return_records(self):
        self.assertIn("data", lookup_customer_by_dea(customer_name="Northstar"))
        self.assertIn("data", lookup_material_by_ndc(material_name="Glycemor"))
        self.assertIn("data", get_invoice_pdf(customer_name="Northstar"))


if __name__ == "__main__":
    unittest.main()
