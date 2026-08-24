"""
annotation_generation_service.py
----------------------------------
DOC AI NEW — Annotation Generation Service.

Generates annotations from LLM extraction results using REAL bounding boxes
extracted from the PDF layout via pdfplumber + rapidfuzz matching.

Strategy:
1. Extract all words + coordinates from the PDF (PdfLayoutService).
2. For each extracted field value, search the PDF words using fuzzy matching.
3. Build the bounding box from the matched word sequence.
4. Fall back to zeros only when the value cannot be located.

Success criterion: no annotation sent to SAP with x=0, y=0, w=0, h=0
unless the value genuinely cannot be found in the PDF.
"""

import logging
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz  # type: ignore

from modules.docai_new.pdf_layout_service import PdfLayoutService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fields to annotate
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "customer_name",
    "customer_address",
    "customer_tax_id",
    "invoice_number",
    "invoice_date",
    "due_date",
    "subtotal",
    "tax_amount",
    "total_amount",
]

LINE_ITEM_SUB_FIELDS = ["description", "quantity", "unit_price", "line_total"]

# Minimum rapidfuzz similarity score (0–100) to accept a match
_MATCH_THRESHOLD = 90.0

# Zero-coordinate sentinel
_ZERO_COORDS: dict[str, Any] = {
    "page": 1,
    "x": 0.0,
    "y": 0.0,
    "width": 0.0,
    "height": 0.0,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AnnotationGenerationService:
    """
    Generates annotations from LLM extraction results.

    When a pdf_path is supplied, real bounding boxes are computed by
    matching each extracted value against the PDF word layout.
    Without pdf_path the service falls back to zero coordinates (legacy).

    Each annotation contains:
    - fieldName
    - value
    - page
    - boundingBox (x, y, width, height)
    """

    def __init__(self) -> None:
        self._layout = PdfLayoutService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_annotations(
        self,
        extraction_result: dict[str, Any],
        pdf_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate annotations from extraction result.

        Args:
            extraction_result: Result from FreePromptExtractionService.
            pdf_path: Optional path to the source PDF.
                      When provided, real bounding boxes are computed.

        Returns:
            List of annotation dicts.
        """
        pdf_words: list[dict[str, Any]] = []
        if pdf_path is not None:
            pdf_words = self._layout.extract_words(pdf_path)
            logger.info(
                "PDF layout loaded: %d words available for coordinate matching.",
                len(pdf_words),
            )

        annotations: list[dict[str, Any]] = []

        # ── Core fields ──────────────────────────────────────────────────
        for field_name in REQUIRED_FIELDS:
            value = extraction_result.get(field_name)
            if value is None:
                continue

            coords = self._find_field_coordinates(str(value), pdf_words)
            annotation = {
                "fieldName": field_name,
                "value": str(value),
                "page": coords["page"],
                "boundingBox": {
                    "x": coords["x"],
                    "y": coords["y"],
                    "width": coords["width"],
                    "height": coords["height"],
                },
            }
            annotations.append(annotation)
            self._log_annotation(field_name, str(value), coords)

        # ── Line items ───────────────────────────────────────────────────
        line_items = extraction_result.get("line_items") or []
        for idx, item in enumerate(line_items):
            for sub_field in LINE_ITEM_SUB_FIELDS:
                val = item.get(sub_field)
                if val is None:
                    continue
                coords = self._find_field_coordinates(str(val), pdf_words)
                field_label = f"line_item_{idx + 1}_{sub_field}"
                annotations.append(
                    {
                        "fieldName": field_label,
                        "value": str(val),
                        "page": coords["page"],
                        "boundingBox": {
                            "x": coords["x"],
                            "y": coords["y"],
                            "width": coords["width"],
                            "height": coords["height"],
                        },
                    }
                )
                self._log_annotation(field_label, str(val), coords)

        # ── Summary ──────────────────────────────────────────────────────
        located = sum(
            1
            for a in annotations
            if a["boundingBox"]["width"] > 0 or a["boundingBox"]["height"] > 0
        )
        logger.info(
            "Annotations generated: %d total | %d with real coordinates | %d with zeros.",
            len(annotations),
            located,
            len(annotations) - located,
        )
        return annotations

    def generate_sap_annotations(
        self,
        extraction_result: dict[str, Any],
        pdf_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate annotations in SAP Document AI format.

        Returns list of annotation objects compatible with SAP DocAI template API.
        """
        raw_annotations = self.generate_annotations(extraction_result, pdf_path=pdf_path)
        sap_annotations = []

        for ann in raw_annotations:
            bb = ann["boundingBox"]
            sap_ann = {
                "label": ann["fieldName"],
                "value": ann["value"],
                "page": ann["page"],
                "coordinates": [
                    bb["x"],
                    bb["y"],
                    bb["x"] + bb["width"],
                    bb["y"] + bb["height"],
                ],
            }
            sap_annotations.append(sap_ann)

        return sap_annotations

    # ------------------------------------------------------------------
    # Coordinate matching
    # ------------------------------------------------------------------

    def _find_field_coordinates(
        self,
        value: str,
        pdf_words: list[dict[str, Any]],
        threshold: float = _MATCH_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Find the bounding box of *value* inside the PDF word list.

        Algorithm:
        1. Normalize both the target value and each candidate window.
        2. Slide a window of varying size over words sorted by position.
        3. Score each window with rapidfuzz.fuzz.ratio.
        4. Return the bounding box of the best-scoring window if score >= threshold.
        5. Fall back to zeros if no match found.

        Args:
            value:     The extracted field value to locate.
            pdf_words: Word list from PdfLayoutService.extract_words().
            threshold: Minimum similarity score (0–100) to accept a match.

        Returns:
            Dict with keys: page, x, y, width, height.
        """
        if not pdf_words or not value or not value.strip():
            return dict(_ZERO_COORDS)

        normalized_value = self._normalize_text(value)
        if not normalized_value:
            return dict(_ZERO_COORDS)

        n_tokens = len(normalized_value.split())

        best_score = 0.0
        best_page = 1
        best_words: list[dict[str, Any]] = []

        # Group words by page
        pages: dict[int, list[dict[str, Any]]] = {}
        for word in pdf_words:
            pages.setdefault(word["page"], []).append(word)

        for page_num, page_words in pages.items():
            # Sort by approximate line (bucket top to nearest 5px) then x0
            sorted_words = sorted(
                page_words,
                key=lambda w: (round(w["top"] / 5) * 5, w["x0"]),
            )
            total = len(sorted_words)

            # Try window sizes around the expected token count
            min_win = max(1, n_tokens - 1)
            max_win = min(n_tokens + 3, total)

            for win_size in range(min_win, max_win + 1):
                for i in range(total - win_size + 1):
                    window = sorted_words[i : i + win_size]
                    window_text = self._normalize_text(
                        " ".join(w["text"] for w in window)
                    )
                    score = fuzz.ratio(normalized_value, window_text)

                    if score > best_score:
                        best_score = score
                        best_page = page_num
                        best_words = window

        if best_score >= threshold and best_words:
            x0 = min(w["x0"] for w in best_words)
            x1 = max(w["x1"] for w in best_words)
            top = min(w["top"] for w in best_words)
            bottom = max(w["bottom"] for w in best_words)
            return {
                "page": best_page,
                "x": round(x0, 2),
                "y": round(top, 2),
                "width": round(x1 - x0, 2),
                "height": round(bottom - top, 2),
            }

        logger.debug(
            "No match found for value '%s' (best score=%.1f < threshold=%.1f).",
            value[:50],
            best_score,
            threshold,
        )
        return dict(_ZERO_COORDS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for fuzzy comparison.

        Steps:
        1. Unicode NFKD decomposition → strip combining marks (accents, umlauts).
        2. Lowercase.
        3. Collapse whitespace.
        """
        # NFKD decomposition
        nfkd = unicodedata.normalize("NFKD", text)
        # Strip combining characters (accents, umlauts, etc.)
        stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Lowercase + collapse whitespace
        return " ".join(stripped.lower().split())

    @staticmethod
    def _log_annotation(
        field_name: str, value: str, coords: dict[str, Any]
    ) -> None:
        """Log annotation details for validation."""
        has_coords = coords["width"] > 0 or coords["height"] > 0
        status = "LOCATED" if has_coords else "ZERO   "
        logger.info(
            "[%s] %-30s | value=%-35s | page=%d | x=%-7.1f y=%-7.1f w=%-7.1f h=%.1f",
            status,
            field_name,
            value[:35],
            coords["page"],
            coords["x"],
            coords["y"],
            coords["width"],
            coords["height"],
        )