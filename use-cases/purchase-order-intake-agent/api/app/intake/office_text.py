"""Convert Word (.docx) and Excel (.xlsx) purchase orders to plain text for the agent.

* ``docx_to_text`` uses only the standard library (a .docx is a zip with WordprocessingML);
  paragraphs and tables are emitted in document order, table cells separated by `` | ``.
* ``xlsx_to_text`` uses openpyxl in read-only mode with row/column caps so a huge sheet
  cannot flood the prompt.

The model receives this text in the first message; the original file stays available to the
reviewer in the UI.
"""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MAX_XLSX_ROWS = 500
MAX_XLSX_COLS = 40


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Concatenate the runs of one Word paragraph (tabs and breaks preserved)."""
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag in (f"{W}br", f"{W}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def docx_to_text(data: bytes) -> str:
    """Extract the text of a .docx document.

    Args:
        data: Raw .docx bytes.

    Returns:
        Paragraphs separated by newlines; each table row as ``cell | cell | cell``.

    Raises:
        ValueError: If the bytes are not a valid .docx file.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError(f"not a readable .docx file: {exc}") from exc

    body = root.find(f"{W}body")
    lines: list[str] = []
    for block in list(body) if body is not None else []:
        if block.tag == f"{W}p":
            text = _paragraph_text(block)
            if text:
                lines.append(text)
        elif block.tag == f"{W}tbl":
            for row in block.iter(f"{W}tr"):
                cells = [" ".join(filter(None, (_paragraph_text(p) for p in cell.iter(f"{W}p"))))
                         for cell in row.iter(f"{W}tc")]
                if any(cells):
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def xlsx_to_text(data: bytes) -> str:
    """Extract the cell values of every sheet of an .xlsx workbook.

    Args:
        data: Raw .xlsx bytes.

    Returns:
        ``## Sheet: <name>`` headers followed by one ``value | value`` line per non-empty row
        (computed values, not formulas), capped at MAX_XLSX_ROWS x MAX_XLSX_COLS per sheet.

    Raises:
        ValueError: If the bytes are not a readable workbook.
    """
    from openpyxl import load_workbook  # imported lazily: only needed for Excel uploads

    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - openpyxl raises several unrelated types
        raise ValueError(f"not a readable .xlsx file: {exc}") from exc

    lines: list[str] = []
    try:
        for sheet in workbook.worksheets:
            lines.append(f"## Sheet: {sheet.title}")
            for index, row in enumerate(sheet.iter_rows(max_col=MAX_XLSX_COLS, values_only=True)):
                if index >= MAX_XLSX_ROWS:
                    lines.append(f"... truncated after {MAX_XLSX_ROWS} rows")
                    break
                values = ["" if value is None else str(value).strip() for value in row]
                while values and not values[-1]:
                    values.pop()
                if any(values):
                    lines.append(" | ".join(values))
    finally:
        workbook.close()
    return "\n".join(lines)
