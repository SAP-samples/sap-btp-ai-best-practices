"""Offline checks for amount parsing and currency decimals."""
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from app.s4.amounts import currency_decimals, dec, fmt, parse_amount, sign_of, tolerance


class ParseTests(unittest.TestCase):
    """Amounts become exact Decimals, or None; nothing is rounded silently and nothing raises."""
    def test_numbers_strings_and_empty_values(self):
        self.assertEqual(parse_amount(10303.26, "CAD"), Decimal("10303.26"))
        self.assertEqual(parse_amount(" -81.09 "), Decimal("-81.09"))
        self.assertEqual(parse_amount(None), Decimal("0.00"))
        self.assertEqual(parse_amount(""), Decimal("0.00"))
        self.assertEqual(parse_amount(10303.260000000002), Decimal("10303.26"))  # float noise is not an extra decimal

    def test_invalid_values_are_none(self):
        for value in ("abc", "12,50", float("nan"), float("inf"), True, "12.345"):
            self.assertIsNone(parse_amount(value, "CAD"), value)
        with self.assertRaisesRegex(ValueError, "not a valid amount in CAD"):
            dec("abc", "CAD")

    def test_currency_decides_the_decimals(self):
        self.assertEqual((currency_decimals("jpy"), currency_decimals("KWD"), currency_decimals("CAD"), currency_decimals(None)),
                         (0, 3, 2, 2))
        self.assertEqual(parse_amount("1000", "JPY"), Decimal("1000"))
        self.assertIsNone(parse_amount("1000.5", "JPY"))
        self.assertEqual(parse_amount("1.234", "KWD"), Decimal("1.234"))
        self.assertEqual((fmt(Decimal("1.5"), "KWD"), fmt(Decimal("7"), "JPY"), fmt(Decimal("7"), "CAD")), ("1.500", "7", "7.00"))
        self.assertEqual((tolerance("JPY"), tolerance("KWD"), tolerance()), (Decimal("1"), Decimal("0.001"), Decimal("0.01")))

    def test_decimals_can_be_overridden_per_currency(self):
        with patch.dict(os.environ, {"S4_CURRENCY_DECIMALS": "HUF=0, broken, cad=3"}):
            self.assertEqual((currency_decimals("HUF"), currency_decimals("CAD"), currency_decimals("EUR")), (0, 3, 2))

    def test_sign(self):
        self.assertEqual((sign_of(Decimal("5")), sign_of(Decimal("-5")), sign_of(Decimal("0"))), (1, -1, 0))


if __name__ == "__main__":
    unittest.main()
