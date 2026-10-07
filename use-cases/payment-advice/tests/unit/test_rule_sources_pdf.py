"""
Unit tests for PDF support in the rule-source reader (offline).

Verifies the new ``.pdf`` branch end to end against a generated PDF:
  - a text PDF is accepted by the allowlist and its text is extracted with a
    ``[Page N]`` locator;
  - a PDF with no extractable text (the scanned/image failure mode) raises a
    clear error instead of returning empty content;
  - an unsupported suffix is still rejected.

Uses ``fpdf2`` (already a project dependency) to build the fixtures.

Run from the repo root:
    PYTHONPATH=api python -m unittest discover -s tests/unit -q
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2] / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

from fpdf import FPDF  # noqa: E402

from app.deduction_agent import rule_sources  # noqa: E402
from app.deduction_agent.rule_sources import (  # noqa: E402
    RuleSourceError,
    bind_rule_sources,
    list_rule_sources,
    read_rule_source,
    reset_rule_sources,
)

_RULE_TEXT = "Contoso deduction rule: invoice_reference ends with UL maps to reason code 316"


def _write_text_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, text)
    pdf.output(str(path))


def _write_blank_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()  # a page with no text content
    pdf.output(str(path))


class RuleSourcesPdf(unittest.TestCase):
    def setUp(self) -> None:
        # The rule-source allowlist is a ContextVar; guard against leakage from
        # other tests in the full-suite run so absolute-None assertions hold.
        rule_sources._BOUND_SOURCES.set(None)

    def test_text_pdf_is_accepted_and_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "contoso_rules.pdf"
            _write_text_pdf(pdf_path, _RULE_TEXT)

            token = bind_rule_sources([pdf_path])
            try:
                sources = list_rule_sources()
                self.assertEqual(sources[0]["source_name"], "contoso_rules.pdf")
                self.assertEqual(sources[0]["format"], "pdf")

                chunk = read_rule_source("contoso_rules.pdf")
                self.assertIn("[Page 1]", chunk["content"])
                # Extraction can drop spaces depending on font metrics; assert on
                # distinctive tokens that must survive.
                self.assertIn("Contoso", chunk["content"])
                self.assertIn("316", chunk["content"])
            finally:
                reset_rule_sources(token)

    def test_scanned_pdf_without_text_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "scanned.pdf"
            _write_blank_pdf(pdf_path)

            token = bind_rule_sources([pdf_path])
            try:
                with self.assertRaises(RuleSourceError):
                    read_rule_source("scanned.pdf")
            finally:
                reset_rule_sources(token)

    def test_unsupported_suffix_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            txt_path = Path(tmp) / "rules.txt"
            txt_path.write_text("not a supported rule source", encoding="utf-8")
            with self.assertRaises(RuleSourceError):
                bind_rule_sources([txt_path])
            # Binding failed, so nothing is bound for this execution.
            self.assertIsNone(rule_sources._BOUND_SOURCES.get())


if __name__ == "__main__":
    unittest.main()
