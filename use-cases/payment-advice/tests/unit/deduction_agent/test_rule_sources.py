"""Tests for safe, agent-readable Word and Excel rule sources.

Run with::

    PYTHONPATH=api .venv/bin/python -m unittest \
        tests.unit.deduction_agent.test_rule_sources -v
"""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from openpyxl import Workbook

from app.deduction_agent.rule_sources import (
    RuleSourceError,
    bind_rule_sources,
    build_rule_source_tools,
    read_rule_source,
    reset_rule_sources,
)


_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

_DOCX_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Deduction rules</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Invoice suffix</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Reason code</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>SC</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>307</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""


def _write_docx(path: Path) -> None:
    """Write the smallest DOCX fixture needed to exercise body and table parsing."""

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        archive.writestr("word/document.xml", _DOCX_DOCUMENT)


class RuleSourceTests(unittest.TestCase):
    """Verify extraction, pagination, authorization, and tool exposure."""

    def test_reads_bound_docx_and_xlsx_with_traceable_locations(self) -> None:
        """Removing either parser or its location labels must break this test."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docx_path = root / "rules.docx"
            xlsx_path = root / "mapping.xlsx"
            _write_docx(docx_path)

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Rules"
            sheet.append(["Invoice suffix", "Reason code"])
            sheet.append(["SC", 307])
            workbook.save(xlsx_path)

            token = bind_rule_sources([docx_path, xlsx_path])
            try:
                tools = {tool.name: tool for tool in build_rule_source_tools()}
                self.assertEqual(set(tools), {"list_rule_sources", "read_rule_source"})

                listed = tools["list_rule_sources"].invoke({})
                self.assertEqual(
                    [(item["source_name"], item["format"]) for item in listed],
                    [("mapping.xlsx", "xlsx"), ("rules.docx", "docx")],
                )

                first = read_rule_source("rules.docx", max_characters=35)
                second = read_rule_source(
                    "rules.docx",
                    cursor=first["next_cursor"],
                    max_characters=10_000,
                )
                docx_text = first["content"] + second["content"]
                self.assertIn("[Paragraph 1] Deduction rules", docx_text)
                self.assertIn("[Table 1 row 2] SC | 307", docx_text)
                self.assertTrue(first["has_more"])
                self.assertFalse(second["has_more"])

                xlsx_text = read_rule_source("mapping.xlsx")["content"]
                self.assertIn("[Sheet: Rules]", xlsx_text)
                self.assertIn("A1 = Invoice suffix", xlsx_text)
                self.assertIn("B2 = 307", xlsx_text)
            finally:
                reset_rule_sources(token)

    def test_rejects_sources_that_were_not_bound_by_the_caller(self) -> None:
        """Replacing the allowlist lookup with path resolution must break this test."""

        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / "allowed.docx"
            _write_docx(allowed)
            token = bind_rule_sources([allowed])
            try:
                with self.assertRaisesRegex(RuleSourceError, "not bound"):
                    read_rule_source("../secret.docx")
            finally:
                reset_rule_sources(token)

    def test_rejects_non_pdf_word_or_excel_authoring_sources(self) -> None:
        """Broadening the authoring reader beyond its approved formats must be explicit."""

        with tempfile.TemporaryDirectory() as tmp:
            # PDF is now an approved authoring source, so a plain-text file is the
            # unsupported example here.
            unsupported = Path(tmp) / "incoming.txt"
            unsupported.write_text("plain text is not an authoring source", encoding="utf-8")
            with self.assertRaisesRegex(RuleSourceError, "PDF, DOCX and XLSX"):
                bind_rule_sources([unsupported])


if __name__ == "__main__":
    unittest.main()
