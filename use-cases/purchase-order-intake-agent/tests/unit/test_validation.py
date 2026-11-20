"""Unit tests for the deterministic order validation and system fills."""

import unittest
from datetime import date, datetime

from app.orders.validation import MasterView, ValidationContext, validate

SOLD_TO, SHIP_A, SHIP_B, MAT = "0000100001", "0000100002", "0000100003", "000000000007000001"


def ctx(**overrides):
    """Default validation context."""
    values = dict(received_at=datetime(2026, 3, 3, 9, 0), sales_org_hint=None, confidence_threshold=0.8,
                  order_type="TA", distr_channel="10", po_type="WWW", today=date(2026, 3, 5))
    values.update(overrides)
    return ValidationContext(**values)


def view(ship_tos=(SHIP_A, SHIP_B), default=None, **overrides):
    """Master data: one sold-to in XX10, ship-to partners, one salable pail material."""
    partners = [{"PARVW": "WE", "VKORG": "XX10", "KUNN2": k, "DEFPA": "X" if k == default else None} for k in ship_tos]
    values = dict(
        customer={"KUNNR": SOLD_TO, "KTOKD": "0001", "AUFSD": None, "LOEVM": None},
        sales_areas=[{"KUNNR": SOLD_TO, "VKORG": "XX10", "VTWEG": "10", "SPART": "Z9", "WAERS": "SGD",
                      "ZTERM": "N030", "INCO1": "DAP", "INCO2": "PORT", "VWERK": "XX01", "AUFSD": None}],
        partners=partners,
        materials={MAT: {"MATNR": MAT, "MEINS": "PC", "LVORM": None, "MSTAE": None, "UNITS": [{"MEINH": "PC"}],
                         "SALES": [{"VKORG": "XX10", "VTWEG": "10", "VMSTA": None, "DWERK": "XX02", "VRKME": "PC"}]}},
    )
    values.update(overrides)
    return MasterView(**values)


def order(**header):
    """A complete header and one material line."""
    base = {"PO_NUMBER": "PO-1", "PO_DATE": date(2026, 3, 1), "SOLD_TO": SOLD_TO, "SHIP_TO": SHIP_A}
    base.update(header)
    return base, [{"ITEM_NO": 10, "LINE_TYPE": "MATERIAL", "MATERIAL": MAT, "QUANTITY": 4.0, "UOM": "PC"}]


def codes(result, severity):
    """(field, code) pairs of one severity."""
    return {(i.field, i.code) for i in result.issues if i.severity == severity}


class ValidationTests(unittest.TestCase):
    """Rules from the requirements and the ECC data model."""

    def test_complete_order_passes_with_system_fills(self):
        header, items = order()
        result = validate(header, items, {}, view(), ctx(), "STANDARD")
        self.assertTrue(result.auto_post_ok, result.issues)
        self.assertEqual((header["SALES_ORG"], header["CURRENCY"], header["DIVISION"], header["ORDER_TYPE"]),
                         ("XX10", "SGD", "Z9", "TA"))
        self.assertEqual(items[0]["PLANT"], "XX02")
        self.assertIn(("CURRENCY", "SYSTEM_FILL"), codes(result, "INFO"))

    def test_po_date_falls_back_to_received_date(self):
        header, items = order(PO_DATE=None)
        validate(header, items, {}, view(), ctx(), "STANDARD")
        self.assertEqual(header["PO_DATE"], date(2026, 3, 3))

    def test_single_ship_to_is_used(self):
        header, items = order(SHIP_TO=None)
        result = validate(header, items, {}, view(ship_tos=(SHIP_A,)), ctx(), "STANDARD")
        self.assertEqual(header["SHIP_TO"], SHIP_A)
        self.assertTrue(result.auto_post_ok)

    def test_default_ship_to_only_without_document_address(self):
        header, items = order(SHIP_TO=None)
        validate(header, items, {}, view(default=SHIP_B), ctx(), "STANDARD")
        self.assertEqual(header["SHIP_TO"], SHIP_B)
        header, items = order(SHIP_TO=None, SHIP_TO_ADDRESS="Unknown site 5")
        result = validate(header, items, {}, view(default=SHIP_B), ctx(), "STANDARD")
        self.assertIn(("SHIP_TO", "UNDETERMINED"), codes(result, "ERROR"))

    def test_ship_to_must_be_partner_and_customer_unblocked(self):
        header, items = order(SHIP_TO="0000999999")
        blocked = view(customer={"KUNNR": SOLD_TO, "KTOKD": "0001", "AUFSD": "01", "LOEVM": None})
        result = validate(header, items, {}, blocked, ctx(), "STANDARD")
        self.assertTrue({("SHIP_TO", "NOT_A_PARTNER"), ("SOLD_TO", "BLOCKED")} <= codes(result, "ERROR"))

    def test_material_and_quantity_rules(self):
        header, items = order()
        items[0].update(QUANTITY=2.5, UOM="L")
        not_salable = view()
        not_salable.materials[MAT]["SALES"][0]["VMSTA"] = "02"
        result = validate(header, items, {}, not_salable, ctx(), "STANDARD")
        self.assertTrue({("MATERIAL", "NOT_SALABLE"), ("UOM", "INVALID")} <= codes(result, "ERROR"))
        header, items = order()
        items[0]["QUANTITY"] = 2.5
        self.assertIn(("QUANTITY", "FRACTIONAL"), codes(validate(header, items, {}, view(), ctx(), "STANDARD"), "ERROR"))

    def test_low_confidence_blocks_only_posting_fields(self):
        header, items = order()
        meta = {(0, "PO_NUMBER"): {"CONFIDENCE": 0.6}, (0, "HEADER_NOTE"): {"CONFIDENCE": 0.5},
                (10, "MATERIAL"): {"CONFIDENCE": 0.95, "AMBIGUOUS": True},
                (0, "BUYER_TAX_ID"): {"CONFIDENCE": 0.3, "INFORMATIONAL": True},
                (0, "SHIP_TO"): {"CONFIDENCE": 0.5, "CHANGED_BY": "USER"}}
        result = validate(header, items, meta, view(), ctx(), "STANDARD")
        self.assertIn(("PO_NUMBER", "LOW_CONFIDENCE"), codes(result, "REVIEW"))
        self.assertIn(("MATERIAL", "AMBIGUOUS"), codes(result, "REVIEW"))
        self.assertIn(("HEADER_NOTE", "LOW_CONFIDENCE"), codes(result, "WARNING"))
        self.assertNotIn(("BUYER_TAX_ID", "LOW_CONFIDENCE"), codes(result, "REVIEW") | codes(result, "WARNING"))
        self.assertNotIn(("SHIP_TO", "LOW_CONFIDENCE"), codes(result, "REVIEW"))
        self.assertFalse(result.auto_post_ok)

    def test_review_reasons(self):
        header, items = order(NET_AMOUNT=100.0, CURRENCY="EUR")
        items[0]["LINE_AMOUNT"] = 80.0
        items.append({"ITEM_NO": 20, "LINE_TYPE": "CHARGE", "LINE_AMOUNT": 5.0})
        result = validate(header, items, {}, view(duplicates=["sales order 9100000001"]), ctx(), "CONTRACT",
                          review_flags=[{"item_no": 10, "field": "quantity", "message": "note says 500 pails"}])
        expected = {("NET_AMOUNT", "TOTAL_MISMATCH"), ("CURRENCY", "MISMATCH"), ("_DUPLICATE", "DUPLICATE_PO"),
                    ("_CLASSIFICATION", "CONTRACT_ORDER"), ("_AGENT_FLAG_1", "AGENT_FLAG"), ("LINE_TYPE", "CHARGE_LINE")}
        self.assertTrue(expected <= codes(result, "REVIEW"), codes(result, "REVIEW"))
        self.assertFalse(result.auto_post_ok)

    def test_not_an_order(self):
        header, items = {}, []
        result = validate(header, items, {}, MasterView(), ctx(), "NOT_AN_ORDER")
        self.assertFalse(result.auto_post_ok)
        self.assertEqual(codes(result, "INFO"), {("_CLASSIFICATION", "NOT_AN_ORDER")})

    def test_missing_essentials(self):
        header, items = {"PO_NUMBER": None, "SOLD_TO": None}, []
        result = validate(header, items, {}, MasterView(), ctx(), "STANDARD")
        self.assertTrue({("PO_NUMBER", "MISSING"), ("SOLD_TO", "MISSING"), ("_NO_MATERIAL_LINES", "NO_LINES")}
                        <= codes(result, "ERROR"))


if __name__ == "__main__":
    unittest.main()
