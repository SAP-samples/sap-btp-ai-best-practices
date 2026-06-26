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
    get_material_availability,
    get_order_status,
    get_pricing_for_customer_material,
    lookup_customer_by_dea,
    lookup_customer_recent_orders,
    lookup_material_by_ndc,
)


def has_matches(tool_payload: dict, dataset_name: str | None = None) -> bool:
    data = tool_payload.get("data", {})
    groups = data["results"] if "results" in data else [data]
    for group in groups:
        if dataset_name and group.get("dataset") != dataset_name:
            continue
        if int(group.get("match_count", 0)) > 0:
            return True
    return False


class PharmaOrderComplexScenarioTests(unittest.TestCase):
    def test_material_name_resolution_then_price_and_availability_chain(self):
        material = lookup_material_by_ndc(ndc="90000-0100-30")
        self.assertTrue(has_matches(material, "materials"))

        pricing = get_pricing_for_customer_material("Northstar", "Glycemor 10 mg", 1)
        self.assertTrue(has_matches(pricing, "pricing"))

        availability = get_material_availability("Glycemor 10 mg", requested_quantity=1)
        self.assertTrue(has_matches(availability, "stock"))

    def test_customer_resolution_then_recent_order_chain(self):
        customer = lookup_customer_by_dea(customer_name="Northstar")
        self.assertTrue(has_matches(customer, "customer_compliance"))

        orders = lookup_customer_recent_orders(customer_name="Northstar")
        self.assertTrue(has_matches(orders, "sales_orders"))

    def test_order_status_then_invoice_chain(self):
        order = get_order_status(sales_order="50214568")
        self.assertTrue(has_matches(order, "sales_orders"))

        invoice = get_invoice_pdf(sales_order="50214568")
        self.assertTrue(has_matches(invoice, "invoices"))


if __name__ == "__main__":
    unittest.main()

