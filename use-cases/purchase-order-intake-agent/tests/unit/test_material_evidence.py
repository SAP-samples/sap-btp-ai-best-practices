"""Unit tests for agent-chosen materials: part-number scoring, price check and evidence-based confidence.

Run from the api/ folder:  ../.venv/bin/python -m unittest discover -s ../tests -t ..
"""

import unittest
from datetime import date, datetime

from app.orders.matching import score_material
from app.orders.service import apply_match_confidence, material_confidence, price_check
from app.orders.validation import MasterView, ValidationContext, validate

DISC_P600 = {"MATNR": "000000000007552686", "MAKTX": "BRAND PELKIT DISC 325U 150MM P600 100PC", "MFRPN": "43038",
             "BISMT": None, "VOLUM": None}
DISC_P400 = {**DISC_P600, "MATNR": "000000000007492456", "MAKTX": "BRAND PELKIT DISC 325U 150MM P400 100PC",
             "MFRPN": "98136"}
PRINTED = "43038 BRAND 325U PKIT DISC 150MM P600 100PC 43038"
PRICE = {"KBETR": 49.0, "KONWA": "EUR", "KPEIN": 1, "KMEIN": "PC"}


def evidence_meta(confidence=0.96, **evidence):
    """MATERIAL field metadata as stored after an agent submission."""
    base = {"matched": ["grit: P600"], "conflicts": [], "alternatives": []}
    base.update(evidence)
    return {"EXTRACT_CONFIDENCE": confidence, "CONFIDENCE": confidence, "CHANGED_BY": "AGENT", "EVIDENCE": base}


class PartNumberScoring(unittest.TestCase):
    """The manufacturer part number in the description identifies the variant."""

    def test_part_number_in_description(self):
        score, matched_on, _ = score_material(DISC_P600, codes=["BRAND 5211985"], description=PRINTED)
        self.assertEqual(score, 0.95)
        self.assertIn("manufacturer_part", matched_on)

    def test_other_variant_stays_description_only(self):
        score, matched_on, _ = score_material(DISC_P400, codes=["BRAND 5211985"], description=PRINTED)
        self.assertLess(score, 0.8)
        self.assertEqual(matched_on, ["description"])


class PriceCheck(unittest.TestCase):
    """Printed unit price against the customer price (A305)."""

    def test_match_within_tolerance(self):
        text, differs = price_check({"UNIT_PRICE": 49.4, "UOM": "PC"}, PRICE)
        self.assertIn("matches", text)
        self.assertFalse(differs)

    def test_large_deviation_differs(self):
        text, differs = price_check({"UNIT_PRICE": 1.0, "UOM": "PC"}, PRICE)
        self.assertIn("differs", text)
        self.assertTrue(differs)

    def test_price_from_line_amount(self):
        _, differs = price_check({"LINE_AMOUNT": 98.0, "QUANTITY": 2, "UOM": "PC"}, PRICE)
        self.assertFalse(differs)

    def test_nothing_to_compare(self):
        self.assertEqual(price_check({"UNIT_PRICE": 49.0}, None), (None, False))
        self.assertEqual(price_check({"UNIT_PRICE": 49.0, "UOM": "L"}, PRICE), (None, False))


class EvidenceConfidence(unittest.TestCase):
    """Agent evidence plus system checks become the material confidence."""

    def test_clean_evidence_keeps_agent_confidence(self):
        meta = evidence_meta()
        material_confidence(meta, 0.95, "Printed price 49.00 matches the customer price 49.00 EUR.", False)
        self.assertEqual(meta["CONFIDENCE"], 0.96)
        self.assertFalse(meta["AMBIGUOUS"])
        self.assertIsNone(meta["CONFIDENCE_REASON"])
        self.assertIn("matches", meta["EVIDENCE"]["price_check"])

    def test_conflict_caps_confidence(self):
        meta = evidence_meta(conflicts=["pack: line 4 L, material 1 L"])
        material_confidence(meta, 0.9, None, False)
        self.assertLessEqual(meta["CONFIDENCE"], 0.6)
        self.assertIn("pack", meta["CONFIDENCE_REASON"])

    def test_price_difference_caps_confidence(self):
        meta = evidence_meta()
        material_confidence(meta, 0.9, "Printed price 1.00 differs from the customer price 49.00 EUR (-98%).", True)
        self.assertLessEqual(meta["CONFIDENCE"], 0.75)

    def test_weak_text_caps_confidence(self):
        meta = evidence_meta()
        material_confidence(meta, 0.1, None, False)
        self.assertLessEqual(meta["CONFIDENCE"], 0.7)

    def test_alternatives_make_it_ambiguous(self):
        meta = evidence_meta(alternatives=["7492456"])
        material_confidence(meta, 0.9, None, False)
        self.assertTrue(meta["AMBIGUOUS"])


class EmptyMaterial(unittest.TestCase):
    """An unresolved material shows 0 % and only the 'missing' error."""

    def test_empty_material_scores_zero_without_duplicate_issue(self):
        items = [{"ITEM_NO": 1, "LINE_TYPE": "MATERIAL", "MATERIAL": None, "QUANTITY": 1.0, "UOM": "PC"}]
        meta = {(1, "MATERIAL"): {"RAW_VALUE": "BRAND 5211985", "EXTRACT_CONFIDENCE": 0.99, "CONFIDENCE": 0.99,
                                  "CHANGED_BY": "AGENT"}}
        view = MasterView()
        apply_match_confidence({}, items, meta, view, None)
        self.assertEqual(meta[(1, "MATERIAL")]["CONFIDENCE"], 0.0)
        ctx = ValidationContext(received_at=datetime(2026, 3, 3), sales_org_hint=None, confidence_threshold=0.8,
                                order_type="TA", distr_channel="10", po_type="WWW", today=date(2026, 3, 5))
        result = validate({"PO_NUMBER": "1"}, items, meta, view, ctx, "STANDARD")
        codes = [i.code for i in result.issues if i.item_no == 1 and i.field == "MATERIAL"]
        self.assertNotIn("LOW_CONFIDENCE", codes)
        self.assertTrue(codes)


if __name__ == "__main__":
    unittest.main()
