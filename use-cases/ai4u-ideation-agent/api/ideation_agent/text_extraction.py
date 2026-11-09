"""Text extraction from uploaded files.

Extracts plain text from common file formats using lightweight libraries
already in the project's dependencies (pypdf, python-docx).
The extracted text is used by the handover pre-fill pipeline and the
handover agent's search_context tool.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum characters to return — matches the truncation limit in
# handover_prefill._run_extraction so the LLM never receives more than
# it can process.
_MAX_CHARS = 24_000


def extract_text_from_file(filename: str, data: bytes) -> Optional[str]:
    """Extract plain text from a file's raw bytes.

    Args:
        filename: Original filename (used to determine format by extension).
        data: Raw file bytes.

    Returns:
        Extracted text (truncated to ~24 000 chars), or None if the format
        is unsupported or extraction fails.
    """
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()

    try:
        text: Optional[str] = None

        if ext in _TEXT_EXTENSIONS:
            text = _extract_plain_text(data)
        elif ext == "pdf":
            text = _extract_pdf(data)
        elif ext in ("docx",):
            text = _extract_docx(data)
        elif ext in ("doc",):
            # Legacy .doc format — python-docx can't read it.
            logger.info("Legacy .doc format not supported for text extraction: %s", filename)
            return None
        elif ext in ("xlsx", "xls"):
            text = _extract_xlsx(data)
        else:
            logger.info("No text extraction available for .%s files: %s", ext, filename)
            return None

        if text and len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "\n\n[... truncated]"

        return text if text and text.strip() else None

    except Exception as exc:
        logger.warning("Text extraction failed for %s: %s", filename, exc)
        return None


# ── Supported text extensions ──────────────────────────────────────────────

_TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "text",
    "csv", "tsv",
    "json", "xml", "yaml", "yml",
    "py", "js", "ts", "html", "htm", "css", "sql",
    "rtf",  # Best-effort: read raw text, may include RTF tags
    "log", "ini", "cfg", "conf", "env",
    "srt", "vtt",  # Subtitle formats
}


# ── Extraction helpers ─────────────────────────────────────────────────────


def _extract_plain_text(data: bytes) -> Optional[str]:
    """Decode UTF-8 text (with fallback to latin-1)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except Exception:
            return None


def _extract_pdf(data: bytes) -> Optional[str]:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — cannot extract text from PDF")
        return None

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages) if pages else None


def _extract_docx(data: bytes) -> Optional[str]:
    """Extract text from a .docx file using python-docx."""
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed — cannot extract text from DOCX")
        return None

    doc = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))

    return "\n\n".join(paragraphs) if paragraphs else None


def _extract_xlsx(data: bytes) -> Optional[str]:
    """Best-effort text extraction from .xlsx using openpyxl (if available)."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        logger.info("openpyxl not installed — cannot extract text from XLSX")
        return None

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = []

    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            header = f"[Sheet: {ws.title}]"
            sheets.append(header + "\n" + "\n".join(rows))

    wb.close()
    return "\n\n".join(sheets) if sheets else None
