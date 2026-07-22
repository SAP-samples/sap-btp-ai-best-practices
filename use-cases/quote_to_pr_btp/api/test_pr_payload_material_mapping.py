import unittest
from unittest.mock import patch

from app.services.s4_pr_client import (
    S4PRError,
    build_pr_payload,
    create_purchase_requisition_for_poc,
)


class PRPayloadMaterialMappingTests(unittest.TestCase):
    def test_line_material_override_does_not_replace_other_fallback_lines(self) -> None:
        normalized = {
            "header": {"vendor_name": "CONTOSO MACHINE SERVICES", "currency": "USD"},
            "line_items": [
                {"description": "Plant-ready item", "quantity": 1, "unit_price": 10},
                {"description": "Unresolved item", "quantity": 2, "unit_price": 20},
            ],
            "pr_mapping": {},
        }
        prepared = build_pr_payload(
            normalized,
            {
                "supplier": "1000002",
                "material": "SP002",
                "plant": "1710",
                "purchasing_group": "001",
                "material_group": "YBPM01",
                "base_unit": "PC",
                "delivery_date": "2026-08-01",
                "line_items": [
                    {
                        "index": 0,
                        "material": "Q2PR0001",
                        "material_group": "YBPM01",
                        "base_unit": "PC",
                    }
                ],
            },
        )

        items = prepared["payload"]["to_PurchaseReqnItem"]["results"]
        self.assertTrue(prepared["ready_for_create"])
        self.assertEqual("Q2PR0001", items[0]["Material"])
        self.assertEqual("SP002", items[1]["Material"])
        self.assertEqual({"1000002"}, {item["Supplier"] for item in items})
        self.assertEqual({"1000002"}, {item["FixedSupplier"] for item in items})

    @patch("app.services.s4_pr_client.create_purchase_requisition")
    def test_poc_creation_retries_without_unavailable_fixed_supplier(self, create_mock) -> None:
        create_mock.side_effect = [
            S4PRError(
                "S/4 rejected the purchase requisition payload.",
                status_code=400,
                details="Supplier 1000001 not yet created by purchasing organization 1000",
            ),
            {"status": "created", "purchase_requisition": "10000999"},
        ]
        payload = {
            "to_PurchaseReqnItem": {
                "results": [
                    {"Material": "DEMO-MAT-001", "Supplier": "1000001", "FixedSupplier": "1000001"},
                ]
            }
        }

        result = create_purchase_requisition_for_poc(payload)

        self.assertEqual("10000999", result["purchase_requisition"])
        self.assertEqual("supplier_not_extended_for_purchasing_org", result["poc_adjustments"][0]["code"])
        retry_payload = create_mock.call_args_list[1].args[0]
        self.assertNotIn("Supplier", retry_payload["to_PurchaseReqnItem"]["results"][0])
        self.assertNotIn("FixedSupplier", retry_payload["to_PurchaseReqnItem"]["results"][0])
        self.assertEqual("1000001", payload["to_PurchaseReqnItem"]["results"][0]["Supplier"])

    def test_line_prices_preserve_quote_net_amount(self) -> None:
        normalized = {
            "header": {"vendor_name": "NORTHWIND INDUSTRIAL SUPPLY", "currency": "USD", "subtotal_amount": 417.82, "tax_amount": 25.07, "total_amount": 442.89},
            "line_items": [
                {"description": "Push plate", "quantity": 1, "unit_of_measure": "EA", "unit_price": 19.47},
                {"description": "Pull plate", "quantity": 1, "unit_of_measure": "EA", "unit_price": 55.15},
                {"description": "Door closer", "quantity": 1, "unit_of_measure": "EA", "unit_price": 343.20},
            ],
            "pr_mapping": {},
        }

        prepared = build_pr_payload(normalized, {"material": "SP002", "plant": "1710", "purchasing_group": "001"})
        items = prepared["payload"]["to_PurchaseReqnItem"]["results"]
        net_amount = sum(float(item["RequestedQuantity"]) * float(item["PurchaseRequisitionPrice"]) for item in items)

        self.assertAlmostEqual(417.82, net_amount, places=2)


if __name__ == "__main__":
    unittest.main()
