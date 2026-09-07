"""Shared helpers for Document Manager upload and corpus ingestion."""

from pathlib import Path
from typing import Any


TEXT_RAG_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xlsm", ".eml"}
"""File extensions supported by the text extraction and RAG indexer."""


def validate_document_file_name(file_name: str | None) -> str:
    """Validate an uploaded corpus document name and return its basename.

    Inputs:
        file_name: Browser-provided filename.

    Outputs:
        str: Normalized basename accepted by the document corpus.

    Raises:
        ValueError: Raised when the filename is empty or has an unsupported
        extension.
    """
    if not file_name:
        raise ValueError("All uploaded files must include a filename.")
    visible_file_name = Path(file_name).name
    if not visible_file_name:
        raise ValueError("All uploaded files must include a non-empty filename.")
    suffix = Path(visible_file_name).suffix.lower()
    if suffix not in TEXT_RAG_ALLOWED_EXTENSIONS:
        raise ValueError(
            "Document Manager supports PDF, DOCX, XLSX, XLSM, and EML files."
        )
    return visible_file_name


def document_upload_payload(
    file_name: str,
    content_type: str | None,
    content: bytes,
) -> dict[str, Any]:
    """Build the repository payload for one uploaded document.

    Inputs:
        file_name: Validated user-visible filename.
        content_type: Browser-provided content type.
        content: Uploaded document bytes.

    Outputs:
        dict[str, Any]: Repository payload with normalized content type.
    """
    return {
        "file_name": file_name,
        "content_type": content_type or "application/octet-stream",
        "content": content,
    }
