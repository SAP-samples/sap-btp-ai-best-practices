"""
pdf_detection_service.py
------------------------
DOC AI NEW — PDF Type Detection Service.

Determines whether a PDF is:
  - Searchable: contains extractable text
  - Scanned: contains only images (requires OCR)

Strategy:
  - Attempt text extraction using pypdf
  - If extracted text is empty or near-empty → scanned PDF
  - Otherwise → searchable PDF
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum number of meaningful characters to consider a PDF searchable
MIN_TEXT_LENGTH: int = 50


class PdfDetectionService:
    """
    Detects whether a PDF is searchable or scanned.

    Usage:
        service = PdfDetectionService()
        is_scanned = service.is_scanned_pdf(pdf_path)
    """

    def is_scanned_pdf(self, pdf_path: Path) -> bool:
        """
        Determine if a PDF is scanned (image-only) or searchable (has text).

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            True if the PDF is scanned (no extractable text), False if searchable.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text = self._extract_text(pdf_path)
        cleaned = text.strip()

        is_scanned = len(cleaned) < MIN_TEXT_LENGTH

        logger.info(
            "PDF type detection: '%s' → %s (extracted %d chars)",
            pdf_path.name,
            "SCANNED" if is_scanned else "SEARCHABLE",
            len(cleaned),
        )

        return is_scanned

    def _extract_text(self, pdf_path: Path) -> str:
        """
        Attempt to extract text from a PDF using pypdf.

        Falls back to empty string if pypdf is not available or extraction fails.
        """
        try:
            import pypdf  # type: ignore

            text_parts: list[str] = []
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                    except Exception as exc:
                        logger.debug("Could not extract text from page: %s", exc)

            return "\n".join(text_parts)

        except ImportError:
            logger.warning(
                "pypdf not installed. Falling back to LLM-only extraction. "
                "Install with: pip install pypdf"
            )
            # Without pypdf, assume searchable (LLM handles both cases)
            return "fallback_text_extraction_not_available"

        except Exception as exc:
            logger.warning("Text extraction failed: %s. Treating as scanned.", exc)
            return ""

    def get_pdf_info(self, pdf_path: Path) -> dict:
        """
        Return detailed PDF type information.

        Returns:
            Dict with keys: is_scanned, text_length, pdf_type
        """
        text = self._extract_text(pdf_path)
        cleaned = text.strip()
        is_scanned = len(cleaned) < MIN_TEXT_LENGTH

        return {
            "is_scanned": is_scanned,
            "text_length": len(cleaned),
            "pdf_type": "scanned" if is_scanned else "searchable",
            "filename": pdf_path.name,
        }


def is_scanned_pdf(pdf_path: Path) -> bool:
    """Convenience function to detect if a PDF is scanned."""
    return PdfDetectionService().is_scanned_pdf(pdf_path)