"""Unit tests for the pure matching helpers (no database)."""

import unittest

from app.orders.matching import (
    alpha_in,
    alpha_out,
    pack_litres,
    parse_code,
    rank_customers,
    rank_materials,
    strip_accents,
    text_similarity,
)


def customer(kunnr, name1, *, name2=None, street=None, city=None, country="SG", vat=None, phone=None, emails=()):
    """Build a KNA1-like test row."""
    return {"KUNNR": kunnr, "NAME1": name1, "NAME2": name2, "STRAS": street, "PSTLZ": None, "ORT01": city,
            "LAND1": country, "STCEG": vat, "TELF1": phone, "EMAILS": list(emails)}


def material(matnr, bismt, maktx, volume):
    """Build a MARA+MAKT-like test row."""
    return {"MATNR": alpha_in(matnr, 18), "BISMT": bismt, "MAKTX": maktx, "VOLUM": volume, "VOLEH": "L"}


CUSTOMERS = [
    customer("0000100001", "HARBOR POINT MARINE SUPPLIES PTE LTD", street="12 QUAY ROAD", city="SINGAPORE",
             vat="201700001A", emails=["orders@harborpoint.example"]),
    customer("0000100002", "HARBOR POINT TRADING PTE LTD", street="99 DOCK LANE", city="SINGAPORE", vat="199900002B"),
    customer("0000100003", "CONG TY TNHH SON BIEN XANH", street="42 DUONG 30/4", city="DA NANG", country="VN"),
    customer("0000100004", "KAROSSERIE NORD GMBH", street="HAFENSTRASSE 5", city="HAMBURG", country="DE",
             phone="4012345678"),
]

MATERIALS = [
    material("7000001", "XG1421/QT", "EPOXY PRIMER WHITE BASE QT", 0.946),
    material("7000002", "XG1421/GL", "EPOXY PRIMER WHITE BASE GL", 3.785),
    material("7000003", "BSE106/20L/XA", "TOPCOAT BASE ULTRA DEEP 20L", 17.6),
    material("7000004", "BSE106/20L", "TOPCOAT BASE ULTRA DEEP 20L EXPORT", 17.6),
    material("7000005", "THN061/1LT", "EPOXY THINNER 1L", 1.0),
    material("7000006", "THN061/4LT", "EPOXY THINNER 4L", 4.0),
    material("7000007", "EGB105/A20L/XA", "PRIMER BASE 20L", 18.4),
    material("455991", None, "MIXING TONER 800 EU 1L", 1.0),
]


class FormatTests(unittest.TestCase):
    """ALPHA conversion, accent folding, pack parsing."""

    def test_alpha_conversion(self):
        self.assertEqual(alpha_in("759842", 10), "0000759842")
        self.assertEqual(alpha_in(" ab-12 ", 18), "AB-12")
        self.assertEqual(alpha_out("0000759842"), "759842")
        self.assertEqual(alpha_out("0ABC"), "0ABC")
        self.assertIsNone(alpha_in("  ", 10))

    def test_vietnamese_accents_are_folded(self):
        self.assertEqual(strip_accents("Đường Lê Lợi"), "duong le loi")

    def test_pack_sizes(self):
        self.assertEqual(pack_litres("20L"), 20.0)
        self.assertEqual(pack_litres("A4LT"), 4.0)
        self.assertEqual(pack_litres("1GL"), 3.785)
        self.assertEqual(pack_litres("QT"), 0.946)
        self.assertEqual(pack_litres("946ML"), 0.946)
        self.assertEqual(pack_litres("4 Ltr"), 4.0)
        self.assertEqual(pack_litres("2,5 l"), 2.5)
        self.assertIsNone(pack_litres("4AP"))

    def test_parse_code(self):
        self.assertEqual(parse_code("BSE106/20L/XA"), ("BSE106", 20.0, {"XA"}))
        self.assertEqual(parse_code("XG1421"), ("XG1421", None, set()))

    def test_legal_forms_do_not_count(self):
        self.assertGreater(text_similarity("Karosserie Nord", "KAROSSERIE NORD GMBH"), 0.95)


class CustomerRankingTests(unittest.TestCase):
    """Customer resolution priorities."""

    def test_tax_id_beats_similar_name(self):
        top = rank_customers(CUSTOMERS, name="Harbor Point", tax_id="201700001 A")[0]
        self.assertEqual(top["KUNNR"], "0000100001")
        self.assertEqual(top["SCORE"], 1.0)
        self.assertFalse(top["AMBIGUOUS"])

    def test_sender_domain_identifies_customer(self):
        top = rank_customers(CUSTOMERS, name="Harbor Point Marine", sender_email="Buyer <buyer@harborpoint.example>")[0]
        self.assertEqual(top["KUNNR"], "0000100001")
        self.assertIn("email_domain", top["MATCHED_ON"])

    def test_name_only_between_similar_customers_stays_below_review_threshold(self):
        """A vague name never reaches the 0.80 auto-post confidence, so the order goes to review."""
        ranked = rank_customers(CUSTOMERS, name="Harbor Point Pte Ltd")
        self.assertLess(ranked[0]["SCORE"], 0.8)
        self.assertEqual(len([c for c in ranked if c["KUNNR"] in {"0000100001", "0000100002"}]), 2)

    def test_phone_in_buyer_block_matches(self):
        top = rank_customers(CUSTOMERS, name="KAROSSERIE NORD GMBH", phone="40 1234 5678")[0]
        self.assertEqual(top["KUNNR"], "0000100004")
        self.assertIn("phone", top["MATCHED_ON"])

    def test_vietnamese_name_with_accents(self):
        top = rank_customers(CUSTOMERS, name="CÔNG TY TNHH SƠN BIỂN XANH", address="Số 42 đường 30/4, Đà Nẵng")[0]
        self.assertEqual(top["KUNNR"], "0000100003")
        self.assertGreaterEqual(top["SCORE"], 0.9)


class MaterialRankingTests(unittest.TestCase):
    """Material resolution: codes, packs, suffixes, customer codes."""

    def test_missing_prefix_letter_and_pack_select_gallon(self):
        ranked = rank_materials(MATERIALS, codes=["G1421/1GL"])
        self.assertEqual(ranked[0]["MATNR"], alpha_in("7000002", 18))
        self.assertGreaterEqual(ranked[0]["SCORE"], 0.9)
        self.assertGreater(ranked[0]["SCORE"] - ranked[1]["SCORE"], 0.2)

    def test_xa_suffix_narrows_to_local_variant(self):
        ranked = rank_materials(MATERIALS, codes=["BSE106/20L/XA"])
        self.assertEqual(ranked[0]["MATNR"], alpha_in("7000003", 18))
        self.assertEqual(ranked[0]["SCORE"], 1.0)
        self.assertFalse(ranked[0]["AMBIGUOUS"])

    def test_missing_suffix_still_matches_unique_material(self):
        top = rank_materials(MATERIALS, codes=["EGB105/A20L"])[0]
        self.assertEqual(top["MATNR"], alpha_in("7000007", 18))
        self.assertGreaterEqual(top["SCORE"], 0.8)

    def test_pack_conflict_between_code_and_line_is_reported(self):
        ranked = rank_materials(MATERIALS, codes=["THN061/1LT"], pack_text="4 Ltr")
        self.assertTrue(any("line states 4.0 L" in note for note in ranked[0]["NOTES"]))
        self.assertLess(ranked[0]["SCORE"], 0.85)

    def test_brand_prefix_and_material_number(self):
        top = rank_materials(MATERIALS, codes=["Brandname 455991"])[0]
        self.assertEqual(top["MATNR"], alpha_in("455991", 18))
        self.assertEqual(top["MATCHED_ON"], ["matnr"])

    def test_customer_material_code(self):
        top = rank_materials(MATERIALS, codes=["1-ABC-0192"], customer_codes={"1ABC0192": alpha_in("7000006", 18)})[0]
        self.assertEqual(top["MATNR"], alpha_in("7000006", 18))
        self.assertEqual(top["MATCHED_ON"], ["customer_code"])

    def test_description_only_match_is_capped(self):
        top = rank_materials(MATERIALS, description="Epoxy thinner 4 litre", pack_text="4L")[0]
        self.assertEqual(top["MATNR"], alpha_in("7000006", 18))
        self.assertLessEqual(top["SCORE"], 0.9)


if __name__ == "__main__":
    unittest.main()


class LearnedMappingTests(unittest.TestCase):
    """Reviewer corrections fed back as precedent (correction learning)."""

    def test_learned_material_outranks_description_match(self):
        learned = {"YEWF236": alpha_in("7000006", 18)}
        top = rank_materials(MATERIALS, codes=["YEW F236"], description="Epoxy primer white", learned=learned)[0]
        self.assertEqual(top["MATNR"], alpha_in("7000006", 18))
        self.assertEqual((top["SCORE"], top["MATCHED_ON"]), (0.98, ["learned"]))

    def test_exact_code_still_beats_learned(self):
        learned = {"XG1421GL": alpha_in("7000001", 18)}
        top = rank_materials(MATERIALS, codes=["XG1421/GL"], learned=learned)[0]
        self.assertEqual((top["MATNR"], top["SCORE"]), (alpha_in("7000002", 18), 1.0))

    def test_learned_customer_name(self):
        learned = {"harbor point": "0000100002"}
        top = rank_customers(CUSTOMERS, name="HARBOR POINT PTE. LTD.", learned=learned)[0]
        self.assertEqual(top["KUNNR"], "0000100002")
        self.assertIn("learned", top["MATCHED_ON"])
