"""Unit tests for the UC-02 INTERPRETATION_SCHEMA structure.

Verifies that the entry sub-schema includes the `line_index` and `rationale`
properties introduced in the selective-analysis redesign, that `line_index` is
marked as required, and that the document_nature vocabulary still contains the
full controlled vocabulary (including `allowance_reversal`).

Run:
    PYTHONPATH=api .venv/bin/python -m unittest tests/unit/test_interpretation_schema.py -v
"""
import unittest

from app.deduction_agent.interpretation_schema import INTERPRETATION_SCHEMA


class SchemaTest(unittest.TestCase):
    """Tests for the INTERPRETATION_SCHEMA entry sub-schema."""

    def _entry(self) -> dict:
        """Return the items sub-schema (one interpretation entry)."""
        return INTERPRETATION_SCHEMA["properties"]["interpretations"]["items"]

    def test_entry_requires_line_index_and_has_rationale(self):
        """Entry sub-schema must have line_index and rationale properties,
        line_index must be in required, and document_nature description must
        mention allowance_reversal (confirming vocabulary is intact)."""
        entry = self._entry()

        # Both new properties must exist
        self.assertIn("line_index", entry["properties"],
                      "line_index property is missing from entry sub-schema")
        self.assertIn("rationale", entry["properties"],
                      "rationale property is missing from entry sub-schema")

        # line_index must be required
        self.assertIn("line_index", entry.get("required", []),
                      "line_index must appear in the required list of the entry sub-schema")

        # document_nature controlled vocabulary must still mention allowance_reversal
        self.assertIn(
            "allowance_reversal",
            entry["properties"]["document_nature"]["description"],
            "document_nature description must still mention allowance_reversal",
        )

    def test_line_index_type_is_integer(self):
        """line_index property must declare type integer."""
        entry = self._entry()
        self.assertEqual(
            entry["properties"]["line_index"]["type"],
            "integer",
            "line_index type must be 'integer'",
        )

    def test_rationale_type_is_string(self):
        """rationale property must declare type string."""
        entry = self._entry()
        self.assertEqual(
            entry["properties"]["rationale"]["type"],
            "string",
            "rationale type must be 'string'",
        )

    def test_top_level_required_interpretations(self):
        """Top-level schema must still require the interpretations array."""
        self.assertIn("interpretations", INTERPRETATION_SCHEMA.get("required", []),
                      "Top-level schema must still list 'interpretations' as required")


if __name__ == "__main__":
    unittest.main()
