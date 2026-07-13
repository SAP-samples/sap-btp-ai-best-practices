"""
Utility functions for extracting text and rasterizing pages from PDF documents.

The functions in this module are designed to work in Cloud Foundry, so we rely
only on the pure-Python PyMuPDF package (``fitz``) to avoid system-level
dependencies such as Poppler or Ghostscript.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageImage:
    """
    Container describing an image representation of a single PDF page.

    The bytes attribute stores a PNG image that can be base64-encoded and sent
    to vision-capable LLMs. Width and height are included so that prompts can
    reference the rendered resolution if needed.
    """

    index: int
    bytes: bytes
    width: int
    height: int


def open_document(pdf_path: str | Path) -> fitz.Document:
    """
    Open a PDF document and return the PyMuPDF document object.

    Callers are responsible for closing the document (using a context manager
    is recommended). Raising the error lets the caller decide how to handle
    invalid or corrupted PDFs.
    """

    path = Path(pdf_path)
    return fitz.open(path.as_posix())


def extract_full_text(pdf_path: str | Path) -> str:
    """
    Extract the concatenated text of all pages in a PDF document.

    Returns a single string containing the text from every page. The caller can
    use the length of the returned string to decide whether the document should
    be processed in text mode (>100 characters) or image mode.
    """

    with open_document(pdf_path) as document:
        page_texts = []
        for page in document:  # type: ignore[assignment]
            page_texts.append(page.get_text("text"))
        return "\n".join(page_texts).strip()


def iter_page_text(pdf_path: str | Path) -> Iterator[Tuple[int, str]]:
    """
    Yield ``(page_index, text)`` tuples for each page in the PDF.

    This helper is useful if the caller needs to inspect text content on a
    page-by-page basis, for example to provide additional context to an LLM.
    """

    with open_document(pdf_path) as document:
        for index, page in enumerate(document):  # type: ignore[assignment]
            yield index, page.get_text("text").strip()


def extract_first_page_image(
    pdf_path: str | Path,
    *,
    dpi: int | None = None,
) -> PageImage:
    """
    Extract a rendered PNG image of the first page (index 0) of the PDF.

    This is useful for sending the first page to vision-capable LLMs alongside
    full document text, allowing the model to extract vendor logos and other
    visual information from letterheads while also processing the complete text.

    The rendering resolution can be controlled via the ``PAGE_IMAGE_DPI``
    environment variable or by passing the dpi keyword argument. A default of
    150 DPI strikes a balance between readability and payload size.
    """

    resolved_dpi = dpi or int(os.environ.get("PAGE_IMAGE_DPI", "150"))
    scale = resolved_dpi / 72.0

    with open_document(pdf_path) as document:
        if len(document) == 0:
            raise ValueError(f"PDF document {pdf_path} has no pages")

        page = document[0]
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        buffer = io.BytesIO()
        buffer.write(pixmap.tobytes("png"))
        return PageImage(
            index=0,
            bytes=buffer.getvalue(),
            width=pixmap.width,
            height=pixmap.height,
        )


def iter_page_images(
    pdf_path: str | Path,
    *,
    dpi: int | None = None,
) -> Iterator[PageImage]:
    """
    Yield rendered PNG images for each page in the PDF.

    The rendering resolution can be controlled via the ``PAGE_IMAGE_DPI``
    environment variable or by passing the dpi keyword argument. A default of
    150 DPI strikes a balance between readability and payload size for LLM
    prompts.
    """

    resolved_dpi = dpi or int(os.environ.get("PAGE_IMAGE_DPI", "150"))
    scale = resolved_dpi / 72.0

    with open_document(pdf_path) as document:
        for index, page in enumerate(document):  # type: ignore[assignment]
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            buffer = io.BytesIO()
            buffer.write(pixmap.tobytes("png"))
            yield PageImage(
                index=index,
                bytes=buffer.getvalue(),
                width=pixmap.width,
                height=pixmap.height,
            )

