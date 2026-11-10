"""
Unit tests for the canonical schema module (offline; no SAP calls).

Covers field-definition integrity (counts, unique labels, valid formatting types),
CANONICAL_SCHEMA consistency, and payload validation.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from app.payment_advice import canonical as C  # noqa: E402

# Formatting types SAP accepts (from /schemas/capabilities).
_VALID_FORMATTING = {"string", "number", "date", "discount", "currency", "country/region"}


class FieldDefinitions(unittest.TestCase):
    def test_counts(self) -> None:
        header, line = C.build_field_definitions()
        self.assertEqual(len(header), 6)
        self.assertEqual(len(line), 10)

    def test_labels_unique_across_header_and_line(self) -> None:
        header, line = C.build_field_definitions()
        labels = [f.label for f in header + line]
        self.assertTrue(all(labels), "every field must carry a label")
        self.assertEqual(len(labels), len(set(labels)), "labels must be unique across both arrays")

    def test_formatting_types_valid(self) -> None:
        header, line = C.build_field_definitions()
        for field in header + line:
            self.assertIn(field.formattingType, _VALID_FORMATTING)

    def test_currency_field_is_string_not_numeric(self) -> None:
        # payment_currency holds an ISO code; numeric normalization would corrupt it.
        meta = {name: ftype for name, ftype, _ in C.HEADER_FIELD_META}
        self.assertEqual(meta["payment_currency"], "string")
        self.assertEqual(meta["payment_amount"], "number")


class CanonicalSchemaConsistency(unittest.TestCase):
    def test_schema_names_match_meta(self) -> None:
        header_names = [f["name"] for f in C.CANONICAL_SCHEMA["headerFields"]]
        line_names = [f["name"] for f in C.CANONICAL_SCHEMA["lineItemFields"]]
        self.assertEqual(header_names, list(C.HEADER_FIELD_NAMES))
        self.assertEqual(line_names, list(C.LINE_FIELD_NAMES))


class Validation(unittest.TestCase):
    def _valid_header(self) -> dict:
        return {"payee_name": "Example Lighting", "payment_reference": "REF1", "payment_amount": 100.0}

    def test_valid_payload(self) -> None:
        issues = C.validate_canonical(self._valid_header(), [{"invoice_reference": "1", "net_amount": 5}])
        self.assertEqual(issues, [])

    def test_missing_required_header(self) -> None:
        header = self._valid_header()
        del header["payment_reference"]
        issues = C.validate_canonical(header, [])
        self.assertTrue(any("payment_reference" in i for i in issues))

    def test_blank_required_header(self) -> None:
        header = self._valid_header()
        header["payee_name"] = "   "
        issues = C.validate_canonical(header, [])
        self.assertTrue(any("payee_name" in i for i in issues))

    def test_unknown_line_field_flagged(self) -> None:
        issues = C.validate_canonical(self._valid_header(), [{"bogus_field": 1}])
        self.assertTrue(any("bogus_field" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
