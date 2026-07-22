import unittest

from app.services.s4_master_data import rank_materials, suggest_master_data


class MaterialPlantGuardrailTest(unittest.TestCase):
    def test_exact_match_is_not_auto_applied_without_required_plant(self):
        source = {"description": "Push Plate Aluminum", "manufacturer_part_number": "70B.28"}
        products = [
            {
                "Product": "Q2PR0001",
                "ProductOldID": "70B.28",
                "ProductGroup": "YBPM01",
                "BaseUnit": "PC",
                "descriptions": ["Push Plate Aluminum"],
                "plants": ["1010"],
            }
        ]

        result = rank_materials(source, products, required_plant="1710")

        self.assertEqual(result["status"], "review")
        self.assertFalse(result["auto_apply"])
        self.assertFalse(result["candidates"][0]["plant_ready"])

    def test_exact_match_is_auto_applied_in_required_plant(self):
        source = {"description": "Push Plate Aluminum", "manufacturer_part_number": "70B.28"}
        products = [
            {
                "Product": "Q2PR0001",
                "ProductOldID": "70B.28",
                "ProductGroup": "YBPM01",
                "BaseUnit": "PC",
                "descriptions": ["Push Plate Aluminum"],
                "plants": ["1710"],
            }
        ]

        result = rank_materials(source, products, required_plant="1710")

        self.assertEqual(result["status"], "matched")
        self.assertTrue(result["auto_apply"])
        self.assertTrue(result["candidates"][0]["plant_ready"])

    def test_customer_suggestions_do_not_modify_the_pr_automatically(self):
        normalized = {
            "header": {"vendor_name": "NORTHWIND INDUSTRIAL SUPPLY"},
            "line_items": [
                {
                    "description": "Push Plate Aluminum",
                    "manufacturer_part_number": "70B.28",
                }
            ],
            "pr_mapping": {"plant": "1710"},
        }
        partners = [
            {
                "BusinessPartner": "1000001",
                "Supplier": "1000001",
                "BusinessPartnerFullName": "NORTHWIND INDUSTRIAL SUPPLY",
            }
        ]
        products = [
            {
                "Product": "Q2PR0001",
                "ProductOldID": "70B.28",
                "ProductGroup": "YBPM01",
                "BaseUnit": "PC",
                "descriptions": ["Push Plate Aluminum"],
                "plants": ["1710"],
            }
        ]

        result = suggest_master_data(
            normalized,
            business_partners=partners,
            products=products,
        )

        self.assertEqual(result["application_mode"], "suggestions_only")
        self.assertFalse(result["business_partner"]["auto_apply"])
        self.assertFalse(result["materials"][0]["auto_apply"])
        self.assertEqual(result["recommended_overrides"], {"line_items": []})


if __name__ == "__main__":
    unittest.main()
