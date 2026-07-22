import json
import tempfile
import unittest
from pathlib import Path

from app.services.purchasing_intelligence import (
    create_back_office_referral,
    create_material_proposal,
    list_material_proposals,
    purchasing_intelligence,
)


NORMALIZED = {
    "header": {"vendor_name": "CONTOSO MACHINE SERVICES"},
    "line_items": [
        {
            "description": "Small water pump shaft repair",
            "manufacturer_part_number": "PUMP-01",
            "unit_of_measure": "AU",
        }
    ],
}


class PurchasingIntelligenceTests(unittest.TestCase):
    def test_three_demo_decisions_are_explicit(self) -> None:
        expected = {
            "sample_facilities_quote.pdf": "Ready after validation",
            "sample_service_quote.pdf": "Review 1 material match",
            "sample_new_supplier_quote.pdf": "Review supplier and 9 materials",
        }
        for document, title in expected.items():
            result = purchasing_intelligence(NORMALIZED, document)
            self.assertEqual(title, result["decision"]["title"])
            self.assertEqual("demo", result["source"]["type"])

    def test_material_proposal_is_reused_without_s4_creation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = create_material_proposal(NORMALIZED, "sample_service_quote.pdf", 0, root)
            second = create_material_proposal(NORMALIZED, "sample_service_quote.pdf", 0, root)
            self.assertEqual(first["proposal_id"], second["proposal_id"])
            self.assertFalse(first["s4_material_created"])
            self.assertTrue(second["reused"])
            files = list((root / "material_proposals").glob("*.json"))
            self.assertEqual(1, len(files))
            self.assertEqual(first["proposal_id"], json.loads(files[0].read_text())["proposal_id"])

    def test_changed_item_creates_a_distinct_proposal(self) -> None:
        changed = json.loads(json.dumps(NORMALIZED))
        changed["line_items"][0]["description"] += " revised"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = create_material_proposal(NORMALIZED, "sample_service_quote.pdf", 0, root)
            second = create_material_proposal(changed, "sample_service_quote.pdf", 0, root)
            self.assertNotEqual(first["proposal_id"], second["proposal_id"])

    def test_saved_proposals_are_filtered_by_document(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = create_material_proposal(NORMALIZED, "sample_service_quote.pdf", 0, root)
            create_material_proposal(NORMALIZED, "another-document.PDF", 0, root)
            proposals = list_material_proposals(root, "sample_service_quote.pdf")
            self.assertEqual([expected["proposal_id"]], [item["proposal_id"] for item in proposals])
            self.assertFalse(proposals[0]["s4_material_created"])

    def test_back_office_referral_is_reused(self) -> None:
        normalized = {"header": {"vendor_name": "ALPINE PIPE SUPPLY"}, "line_items": []}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = create_back_office_referral(normalized, "sample_new_supplier_quote.pdf", root)
            second = create_back_office_referral(normalized, "sample_new_supplier_quote.pdf", root)
            self.assertEqual(first["referral_id"], second["referral_id"])
            self.assertTrue(second["reused"])
            self.assertEqual(1, len(list((root / "back_office_referrals").glob("*.json"))))


if __name__ == "__main__":
    unittest.main()
