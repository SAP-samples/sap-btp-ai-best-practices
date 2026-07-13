"""Tests for the markdown-to-PDF converter."""

import tempfile
import unittest
from pathlib import Path

from app.optimizer.report.markdown_to_pdf import markdown_to_pdf


_SAMPLE_MARKDOWN = """\
# Test Report

**Generated:** 2025-01-28

---

## Selection Metrics

| Metric | Value |
|--------|-------|
| Candidates | 100 |
| Selected | 80 |
| Ratio | 80.0% |

## Details

- **Item A**: Some description here
- Item B: Another description

### Sub-section

1. First step
2. Second step

Regular paragraph with **bold** words in the middle.

## Long Table Test

| Invoice Reference | Customer | Company Code | Purchase Price | Currency | Due Date | Planned Week | Lifetime |
|-------------------|----------|--------------|----------------|----------|----------|--------------|----------|
| VERY-LONG-INVOICE-REFERENCE-001 | CUSTOMER-WITH-A-VERY-LONG-NAME | 2410 | 1,234,567.89 | EUR | 2026-02-26 | 2025-01-28 | 52 |
| INV-002 | CUST02 | 2410 | 100.00 | USD | 2025-11-24 | 2025-01-28 | 4 |
"""


class TestMarkdownToPdf(unittest.TestCase):
    def test_creates_valid_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test.pdf"
            result = markdown_to_pdf(_SAMPLE_MARKDOWN, output)

            self.assertEqual(result, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)

            with open(output, "rb") as f:
                header = f.read(5)
            self.assertEqual(header, b"%PDF-")

    def test_empty_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "empty.pdf"
            result = markdown_to_pdf("", output)
            self.assertTrue(output.exists())

    def test_table_with_many_columns_no_crash(self) -> None:
        """Tables with many columns should not crash or raise."""
        wide_table = (
            "| " + " | ".join(f"Col{i}" for i in range(10)) + " |\n"
            "| " + " | ".join("---" for _ in range(10)) + " |\n"
            "| " + " | ".join(f"Val{i}" for i in range(10)) + " |\n"
        )
        md = f"# Wide Table\n\n{wide_table}"
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "wide.pdf"
            result = markdown_to_pdf(md, output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
