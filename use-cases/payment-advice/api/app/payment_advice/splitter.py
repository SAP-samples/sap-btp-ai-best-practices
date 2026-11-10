"""
Format-aware document splitter for the Payment Advice Extractor (UC-01).

SAP Document AI enforces per-job limits (100 pages / 2000 line items / 49 columns).
When an input advice exceeds them it must be split into standalone parts that each
fit, extracted separately, and recombined downstream (see ``aggregation`` /
``extract``). This module owns the *size check* and the *splitting*.

Handled formats:

- **PDF** (``.pdf``): counted by page; split into <=100-page parts (``pypdf``).
- **Tabular** (``.xlsx``, ``.csv``, ``.tsv``): counted by data rows and columns;
  split into parts of <=2000 data rows, each keeping the header row.
- **Image** (``.jpg/.jpeg/.png/.tif/.tiff``): treated as a single page; never split.
- **Opaque** (everything else, e.g. ``.txt``, ``.docx``, ``.eml``): line items cannot
  be measured reliably before extraction, so the file is sent whole. If such a file
  is genuinely oversized, Document AI will report it downstream.

A file with more than 49 columns cannot be fixed by row-chunking, so it is refused
with ``ColumnLimitExceeded`` (this is not expected to occur in practice).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook
from pypdf import PdfReader, PdfWriter

from .config import MAX_COLUMNS, MAX_LINE_ITEMS, MAX_PAGES

# Format buckets keyed by lowercase file extension (including the dot).
_PDF_EXT = {".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_TABULAR_EXT = {".xlsx", ".csv", ".tsv"}

FORMAT_PDF = "pdf"
FORMAT_IMAGE = "image"
FORMAT_TABULAR = "tabular"
FORMAT_OPAQUE = "opaque"


class ColumnLimitExceeded(ValueError):
    """Raised when a tabular document has more columns than Document AI allows."""


@dataclass(frozen=True)
class SizeReport:
    """
    Result of probing a document against the Document AI limits.

    Attributes:
        path: The probed file.
        fmt: One of FORMAT_PDF / FORMAT_IMAGE / FORMAT_TABULAR / FORMAT_OPAQUE.
        pages: Page count for PDF/image, else ``None``.
        rows: Data-row count for tabular, else ``None``.
        columns: Column count for tabular, else ``None``.
        oversized: Whether the document exceeds a splittable limit.
        reason: Human-readable explanation of the size decision.
    """

    path: Path
    fmt: str
    pages: int | None
    rows: int | None
    columns: int | None
    oversized: bool
    reason: str


def detect_format(path: str | Path) -> str:
    """Classify a file into a format bucket by its extension."""
    ext = Path(path).suffix.lower()
    if ext in _PDF_EXT:
        return FORMAT_PDF
    if ext in _IMAGE_EXT:
        return FORMAT_IMAGE
    if ext in _TABULAR_EXT:
        return FORMAT_TABULAR
    return FORMAT_OPAQUE


def _read_table(path: Path) -> tuple[list, list[list]]:
    """
    Read a tabular file into (header_row, data_rows).

    The first non-empty row is treated as the header; subsequent non-empty rows are
    data. This heuristic only affects genuinely oversized files (small files are
    never split), so a messy preamble on a small sheet is harmless.

    Returns:
        A tuple of (header cells, list of data rows). Empty header if the file has
        no non-empty rows.
    """
    ext = path.suffix.lower()
    if ext == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        raw_rows = [list(row) for row in sheet.iter_rows(values_only=True)]
        workbook.close()
    else:
        delimiter = "\t" if ext == ".tsv" else ","
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            raw_rows = [list(row) for row in csv.reader(handle, delimiter=delimiter)]

    def is_empty(row: list) -> bool:
        return all(cell is None or str(cell).strip() == "" for cell in row)

    non_empty = [row for row in raw_rows if not is_empty(row)]
    if not non_empty:
        return [], []
    return non_empty[0], non_empty[1:]


def _write_table_chunk(
    path: Path, header: list, data_rows: list[list], out_dir: Path, index: int
) -> Path:
    """Write one tabular chunk (header + rows) in the source format; return its path."""
    ext = path.suffix.lower()
    out_path = out_dir / f"{path.stem}.part{index:02d}{ext}"
    if ext == ".xlsx":
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(header)
        for row in data_rows:
            sheet.append(row)
        workbook.save(out_path)
    else:
        delimiter = "\t" if ext == ".tsv" else ","
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(data_rows)
    return out_path


def probe(
    path: str | Path,
    *,
    max_pages: int = MAX_PAGES,
    max_line_items: int = MAX_LINE_ITEMS,
    max_columns: int = MAX_COLUMNS,
) -> SizeReport:
    """
    Measure a document against the Document AI limits.

    Args:
        path: The input file.
        max_pages / max_line_items / max_columns: Document AI limits.

    Returns:
        A ``SizeReport`` describing format, measured size, and whether it is oversized.
    """
    path = Path(path)
    fmt = detect_format(path)

    if fmt == FORMAT_PDF:
        pages = len(PdfReader(str(path)).pages)
        oversized = pages > max_pages
        reason = f"PDF with {pages} pages (limit {max_pages})"
        return SizeReport(path, fmt, pages, None, None, oversized, reason)

    if fmt == FORMAT_IMAGE:
        # ponytail: single-frame image = 1 page; multi-page TIFF is not handled
        # (add frame counting if a client ever sends >100-page TIFFs).
        return SizeReport(path, fmt, 1, None, None, False, "image treated as one page")

    if fmt == FORMAT_TABULAR:
        header, data_rows = _read_table(path)
        rows = len(data_rows)
        columns = len(header)
        oversized = rows > max_line_items
        reason = f"tabular with {rows} rows x {columns} cols (limits {max_line_items}/{max_columns})"
        return SizeReport(path, fmt, None, rows, columns, oversized, reason)

    return SizeReport(
        path,
        FORMAT_OPAQUE,
        None,
        None,
        None,
        False,
        "opaque format sent whole (line items not measurable before extraction)",
    )


def split_document(
    path: str | Path,
    out_dir: str | Path,
    *,
    max_pages: int = MAX_PAGES,
    max_line_items: int = MAX_LINE_ITEMS,
    max_columns: int = MAX_COLUMNS,
) -> list[Path]:
    """
    Split a document into Document-AI-sized parts, or return it unchanged.

    Args:
        path: The input file.
        out_dir: Directory to write chunk files into (created if missing).
        max_pages / max_line_items / max_columns: Document AI limits.

    Returns:
        A list of file paths to send to Document AI. Contains the original path
        (a single element) when the document fits or cannot be split by us.

    Raises:
        ColumnLimitExceeded: When a tabular document exceeds ``max_columns``.
    """
    path = Path(path)
    out_dir = Path(out_dir)
    report = probe(
        path,
        max_pages=max_pages,
        max_line_items=max_line_items,
        max_columns=max_columns,
    )

    if report.columns is not None and report.columns > max_columns:
        raise ColumnLimitExceeded(
            f"{path.name} has {report.columns} columns (limit {max_columns}); "
            "row-chunking cannot resolve a column-count overflow"
        )

    if not report.oversized:
        return [path]

    out_dir.mkdir(parents=True, exist_ok=True)

    if report.fmt == FORMAT_PDF:
        return _split_pdf(path, out_dir, max_pages)

    if report.fmt == FORMAT_TABULAR:
        header, data_rows = _read_table(path)
        parts: list[Path] = []
        for index, start in enumerate(range(0, len(data_rows), max_line_items), start=1):
            chunk_rows = data_rows[start : start + max_line_items]
            parts.append(_write_table_chunk(path, header, chunk_rows, out_dir, index))
        return parts

    # Oversized image/opaque cannot be split here; send whole (should not occur).
    return [path]


def _split_pdf(path: Path, out_dir: Path, max_pages: int) -> list[Path]:
    """Split a PDF into <=max_pages-page parts; return the chunk paths in order."""
    reader = PdfReader(str(path))
    total = len(reader.pages)
    parts: list[Path] = []
    for index, start in enumerate(range(0, total, max_pages), start=1):
        writer = PdfWriter()
        for page in reader.pages[start : start + max_pages]:
            writer.add_page(page)
        out_path = out_dir / f"{path.stem}.part{index:02d}.pdf"
        with out_path.open("wb") as handle:
            writer.write(handle)
        parts.append(out_path)
    return parts
