"""Unit tests for deterministic quantity conversion (requirement R4)."""

import unittest

from app.orders.quantity import convert_quantity, resolve_unit

ALIASES = {"lít": "L", "lit": "L", "ltrs": "L", "gallon-": "GAL", "gallon": "GAL", "pc": "PC", "pail": "PC", "t": "PC"}


def material(volume=17.6, bismt="XYZ106/20L/XA", units=None):
    """Pail material sold per piece with the given fill volume."""
    return {"MATNR": "000000000007000001", "MEINS": "PC", "VOLUM": volume, "VOLEH": "L", "BISMT": bismt,
            "UNITS": units or [{"MEINH": "PC", "UMREZ": 1, "UMREN": 1}],
            "SALES": [{"VKORG": "XX10", "VTWEG": "10", "VRKME": "PC"}]}


class QuantityTests(unittest.TestCase):
    """Conversions seen on real orders."""

    def test_unit_aliases_with_accents(self):
        self.assertEqual(resolve_unit("Lít", ALIASES), "L")
        self.assertEqual(resolve_unit("Gallon-", ALIASES), "GAL")
        self.assertIsNone(resolve_unit("bags", ALIASES))

    def test_litres_to_pails_use_fill_volume(self):
        result = convert_quantity(material(17.6), 4400, "Lít", None, ALIASES)
        self.assertEqual((result["sales_qty"], result["sales_uom"], result["method"], result["exact"]),
                         (250.0, "PC", "volume", True))

    def test_fill_volume_with_decimals(self):
        result = convert_quantity(material(17.94), 9867, "Lít", None, ALIASES)
        self.assertEqual((result["sales_qty"], result["exact"]), (550.0, True))

    def test_inexact_volume_offers_nominal_pack(self):
        result = convert_quantity(material(17.6), 1000, "ltrs", None, ALIASES)
        self.assertFalse(result["exact"])
        self.assertEqual(result["alternatives"], [{"sales_qty": 50.0, "basis": "nominal pack 20.0 L"}])

    def test_gallons(self):
        result = convert_quantity(material(3.785, "XG1421/GL"), 6, "Gallon-", None, ALIASES)
        self.assertEqual((result["sales_qty"], result["exact"]), (6.0, True))

    def test_pieces_stay_pieces(self):
        result = convert_quantity(material(), 12, "pail", None, ALIASES)
        self.assertEqual((result["sales_qty"], result["method"], result["exact"]), (12.0, "same_unit", True))

    def test_alternative_unit_from_marm(self):
        units = [{"MEINH": "PC", "UMREZ": 1, "UMREN": 1}, {"MEINH": "CS", "UMREZ": 4, "UMREN": 1}]
        result = convert_quantity(material(units=units), 3, "CS", None, ALIASES)
        self.assertEqual((result["sales_qty"], result["method"]), (12.0, "marm"))

    def test_errors(self):
        self.assertEqual(convert_quantity(material(), 5, "bags", None, ALIASES)["error"], "UNKNOWN_UNIT")
        self.assertEqual(convert_quantity(material(), 0, "pc", None, ALIASES)["error"], "INVALID_QUANTITY")
        no_volume = material(volume=None, bismt=None)
        self.assertEqual(convert_quantity(no_volume, 5, "lít", None, ALIASES)["error"], "UOM_NOT_CONVERTIBLE")


if __name__ == "__main__":
    unittest.main()
