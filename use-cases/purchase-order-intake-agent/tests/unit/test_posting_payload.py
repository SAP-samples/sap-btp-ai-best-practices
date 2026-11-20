"""Writeback payload built from a validated order (requirements: 'Writeback API - required fields')."""

import unittest
from datetime import date

from app.orders.posting import build_posting_payload

HEADER = {
    "ORDER_ID": "order-1", "PO_NUMBER": "PO-4711", "PO_DATE": date(2026, 3, 1), "ORDER_TYPE": "TA",
    "SOLD_TO": "0000100001", "SHIP_TO": "0000100002", "SALES_ORG": "XX10", "DISTR_CHANNEL": "10",
    "DIVISION": "Z9", "PO_TYPE": "WWW", "REQ_DELIVERY_DATE": date(2026, 3, 8), "CURRENCY": "SGD",
    "PAYMENT_TERMS": "N030", "INCOTERMS": "DAP", "INCOTERMS_PLACE": "PORT", "NET_AMOUNT": 120.0,
}
ITEMS = [
    {"ITEM_NO": 10, "LINE_TYPE": "MATERIAL", "MATERIAL": "000000000007000001", "QUANTITY": 4.0, "UOM": "PC",
     "PLANT": "XX01", "CUSTOMER_MATERIAL": "C-1", "COLOUR_CODE": "RAL 9010", "DESCRIPTION": "Primer 4L",
     "LINE_AMOUNT": 100.0},
    {"ITEM_NO": 20, "LINE_TYPE": "CHARGE", "DESCRIPTION": "Freight", "LINE_AMOUNT": 20.0},
]


class PostingPayloadTests(unittest.TestCase):
    """Field names, constants and number formats of the payload."""

    def test_header_fields(self):
        payload = build_posting_payload(HEADER, ITEMS)
        self.assertEqual(
            {k: payload[k] for k in ("orderReference", "orderDate", "orderType", "soldToId", "shipToId", "salesOrgId",
                                     "distributionChannel", "division", "customerPurchaseOrderType")},
            {"orderReference": "PO-4711", "orderDate": "20260301", "orderType": "TA", "soldToId": "100001",
             "shipToId": "100002", "salesOrgId": "XX10", "distributionChannel": "10", "division": "Z9",
             "customerPurchaseOrderType": "WWW"})

    def test_only_material_lines_are_transferred(self):
        items = build_posting_payload(HEADER, ITEMS)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0], {
            "lineItemNumber": 10, "sapMaterialNumber": "7000001", "quantity": 4.0, "unitOfMeasure": "PC",
            "plant": "XX01", "customerMaterialNumber": "C-1", "tintedMaterialColourCode": "RAL 9010",
            "description": "Primer 4L", "netPrice": 25.0})


if __name__ == "__main__":
    unittest.main()
