"""Generic markdown-to-PDF converter using fpdf2.

Parses a subset of markdown (headers, tables, bold, bullets, numbered lists,
horizontal rules) and renders to a styled PDF with OpenSans fonts and the
project's teal colour scheme.

Tables use fpdf2's ``.table()`` context-manager API which automatically wraps
text inside cells and prevents the column-overflow problems of manual
``cell()`` calls.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Teal colour used for H1/H2 headers
_TEAL = (46, 134, 171)
_DARK_GRAY = (70, 70, 70)
_BODY_COLOR = (50, 50, 50)
_HEADER_BG = (240, 240, 240)
_INLINE_EMPHASIS_PATTERN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
_IMAGE_PATTERN = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_FULL_CELL_EMPHASIS_PATTERNS = (
    (re.compile(r"^\*\*\*([^*]+)\*\*\*$"), True, True),
    (re.compile(r"^\*\*([^*]+)\*\*$"), True, False),
    (re.compile(r"^\*([^*]+)\*$"), False, True),
)


def _set_inline_font(pdf, bold: bool, italic: bool, font_size: int) -> None:
    if italic:
        pdf.set_font("Helvetica", "BI" if bold else "I", font_size)
    elif bold:
        pdf.set_font("OpenSans", "B", font_size)
    else:
        pdf.set_font("OpenSans", "", font_size)


def _render_line_with_inline_styles(
    pdf, line: str, font_size: int = 10, line_height: int = 6
) -> None:
    """Render a line containing inline ``**bold**`` and ``*italic*`` segments."""
    _set_inline_font(pdf, bold=False, italic=False, font_size=font_size)
    pdf.set_text_color(*_BODY_COLOR)

    parts = _INLINE_EMPHASIS_PATTERN.split(line)
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) >= 4:
            _set_inline_font(pdf, bold=True, italic=False, font_size=font_size)
            pdf.write(line_height, part[2:-2])
            _set_inline_font(pdf, bold=False, italic=False, font_size=font_size)
        elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
            _set_inline_font(pdf, bold=False, italic=True, font_size=font_size)
            pdf.write(line_height, part[1:-1])
            _set_inline_font(pdf, bold=False, italic=False, font_size=font_size)
        elif part:
            pdf.write(line_height, part)

    pdf.ln(line_height)


def _render_markdown_table(pdf, table_lines: List[str]) -> None:
    """Render a markdown table using fpdf2's ``table()`` context-manager.

    This approach handles automatic column-width calculation, text wrapping
    inside cells, and multi-line content without overflow.  Cells containing
    ``**bold**`` markers are rendered with bold emphasis.
    """
    from fpdf import FontFace

    if len(table_lines) < 2:
        return

    header = [cell.strip() for cell in table_lines[0].split("|") if cell.strip()]
    if not header:
        return

    data_rows: List[List[str]] = []
    for line in table_lines[2:]:
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                while len(cells) < len(header):
                    cells.append("")
                data_rows.append(cells[:len(header)])

    headings_style = FontFace(emphasis="BOLD", fill_color=_HEADER_BG)
    bold_style = FontFace(emphasis="BOLD")
    italic_style = FontFace(family="Helvetica", emphasis="I")
    bold_italic_style = FontFace(family="Helvetica", emphasis="BI")

    pdf.set_font("OpenSans", "", 9)
    pdf.set_text_color(*_BODY_COLOR)

    with pdf.table(
        headings_style=headings_style,
        cell_fill_color=(255, 255, 255),
        cell_fill_mode="ROWS",
        line_height=pdf.font_size * 1.8,
        text_align="LEFT",
        first_row_as_headings=True,
        borders_layout="SINGLE_TOP_LINE",
    ) as table:
        # Header row
        row = table.row()
        for h in header:
            row.cell(h)
        # Data rows -- apply bold to cells wrapped in **
        for data_row in data_rows:
            row = table.row()
            for cell_text in data_row:
                stripped = cell_text.strip()
                style = None
                clean_text = cell_text
                for pattern, is_bold, is_italic in _FULL_CELL_EMPHASIS_PATTERNS:
                    match = pattern.match(stripped)
                    if not match:
                        continue
                    clean_text = match.group(1)
                    if is_bold and is_italic:
                        style = bold_italic_style
                    elif is_bold:
                        style = bold_style
                    else:
                        style = italic_style
                    break
                row.cell(clean_text, style=style)

    pdf.ln(3)


def markdown_to_pdf(markdown_content: str, output_path: Path) -> Path:
    """Convert markdown text to a styled PDF.

    Args:
        markdown_content: Markdown string to render.
        output_path: Destination file path (will be created/overwritten).

    Returns:
        The *output_path* for convenience.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = Path(__file__).parent / "fonts" / "Open_Sans" / "static"
    pdf.add_font("OpenSans", fname=str(font_dir / "OpenSans-Regular.ttf"))
    pdf.add_font("OpenSans", style="B", fname=str(font_dir / "OpenSans-Bold.ttf"))

    pdf.add_page()

    lines = markdown_content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        pdf.set_x(pdf.l_margin)

        # Table block
        if line.strip().startswith("|"):
            table_lines: List[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            _render_markdown_table(pdf, table_lines)
            continue

        # H1
        if line.startswith("# "):
            pdf.set_font("OpenSans", "B", 18)
            pdf.set_text_color(*_TEAL)
            pdf.multi_cell(0, 10, line[2:])
            pdf.ln(5)
        # H2
        elif line.startswith("## "):
            pdf.set_font("OpenSans", "B", 14)
            pdf.set_text_color(*_TEAL)
            pdf.multi_cell(0, 8, line[3:])
            pdf.ln(3)
        # H3
        elif line.startswith("### "):
            pdf.set_font("OpenSans", "B", 12)
            pdf.set_text_color(*_DARK_GRAY)
            pdf.multi_cell(0, 7, line[4:])
            pdf.ln(2)
        # Bullet points
        elif line.startswith("- ") or line.startswith("* "):
            content = line[2:]
            if _INLINE_EMPHASIS_PATTERN.search(content):
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(*_BODY_COLOR)
                pdf.write(6, "  - ")
                _render_line_with_inline_styles(pdf, content, 10, 6)
            else:
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(*_BODY_COLOR)
                pdf.multi_cell(0, 6, "  - " + content)
        # Numbered lists
        elif re.match(r"^\d+\.\s", line):
            if _INLINE_EMPHASIS_PATTERN.search(line):
                _render_line_with_inline_styles(pdf, line, 10, 6)
            else:
                pdf.set_font("OpenSans", "", 10)
                pdf.set_text_color(*_BODY_COLOR)
                pdf.multi_cell(0, 6, line)
        # Horizontal rule
        elif line.strip() == "---":
            pdf.ln(5)
        # Image embedding
        elif _IMAGE_PATTERN.match(line.strip()):
            image_match = _IMAGE_PATTERN.match(line.strip())
            img_path = Path(image_match.group(2))
            if img_path.exists():
                available_width = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(str(img_path), w=min(available_width, 170))
                pdf.ln(5)
        # Lines with inline emphasis
        elif _INLINE_EMPHASIS_PATTERN.search(line):
            _render_line_with_inline_styles(pdf, line, 10, 6)
        # Regular text
        elif line.strip():
            pdf.set_font("OpenSans", "", 10)
            pdf.set_text_color(*_BODY_COLOR)
            pdf.multi_cell(0, 6, line)

        i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    logger.info("PDF report written to %s", output_path)
    return output_path
