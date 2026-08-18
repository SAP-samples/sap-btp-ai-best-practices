"""
pdf_layout_service.py
---------------------
DOC AI NEW — PDF Layout Service.

Extracts word-level bounding boxes from PDF files using pdfplumber.
Provides real spatial coordinates for annotation generation.

Each word is returned as:
{
    "text":   str,
    "page":   int,   # 1-based
    "x0":     float, # left edge
    "x1":     float, # right edge
    "top":    float, # top edge (distance from top of page)
    "bottom": float, # bottom edge
}
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pdfplumber  # type: ignore

    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False
    logger.warning(
        "pdfplumber is not installed. "
        "PDF layout extraction will be unavailable. "
        "Run: pip install pdfplumber"
    )


class PdfLayoutService:
    """
    Extracts word-level bounding boxes from a PDF using pdfplumber.

    Works only for searchable (text-layer) PDFs.
    For scanned PDFs, returns an empty list — coordinates will fall back to zeros.
    """

    def extract_words(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        Extract all words with bounding box coordinates from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of word dicts with keys: text, page, x0, x1, top, bottom.
            Returns empty list if pdfplumber is unavailable or extraction fails.
        """
        if not _PDFPLUMBER_AVAILABLE:
            logger.warning(
                "pdfplumber not available — skipping layout extraction for '%s'.",
                pdf_path.name,
            )
            return []

        words: list[dict[str, Any]] = []
        total_pages = 0

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_words = page.extract_words(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=False,
                        use_text_flow=False,
                    )
                    for w in page_words:
                        text = str(w.get("text", "")).strip()
                        if not text:
                            continue
                        words.append(
                            {
                                "text": text,
                                "page": page_num,
                                "x0": round(float(w.get("x0", 0.0)), 2),
                                "x1": round(float(w.get("x1", 0.0)), 2),
                                "top": round(float(w.get("top", 0.0)), 2),
                                "bottom": round(float(w.get("bottom", 0.0)), 2),
                            }
                        )
        except Exception as exc:
            logger.error(
                "PDF layout extraction failed for '%s': %s", pdf_path.name, exc
            )
            return []

        logger.info(
            "PDF layout extracted: %d words across %d page(s) from '%s'.",
            len(words),
            total_pages,
            pdf_path.name,
        )
        return words