"""Decide how each file is presented to the order agent.

``MODEL_INPUT`` values:

* ``NATIVE`` - sent to the multimodal model as a file/image block (PDF, PNG, JPEG, GIF, WEBP),
  so layout, logos and scanned content are visible to the model;
* ``TEXT``   - converted to text and placed in the first message (DOCX, XLSX, TXT/CSV,
  forwarded e-mails);
* ``NONE``   - stored and shown to the reviewer, but not given to the model (unsupported type,
  logo/icon images, duplicates, conversion errors); ``MODEL_INPUT_REASON`` says why.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import PurePath

from .office_text import docx_to_text, xlsx_to_text

NATIVE_MIME = {"application/pdf", "image/png", "image/jpeg", "image/gif", "image/webp"}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Roles of stored files (POAGENT_ATTACHMENT.ROLE).
ROLE_SOURCE_EMAIL = "SOURCE_EMAIL"      # the uploaded .eml file itself
ROLE_DOCUMENT = "DOCUMENT"              # attachment or loose uploaded file
ROLE_INLINE_IMAGE = "INLINE_IMAGE"      # image embedded in an e-mail body (Content-ID)
ROLE_FORWARDED_EMAIL = "FORWARDED_EMAIL"  # message/rfc822 part (forwarded e-mail)


@dataclass
class StoredFile:
    """One file to be stored in POAGENT_ATTACHMENT.

    Attributes:
        file_name: Display name.
        mime_type: Content type.
        content: Raw bytes.
        role: One of the ROLE_* constants.
        model_input: NATIVE, TEXT or NONE.
        model_input_reason: Why the file is (not) given to the model, or its pixel size.
        extracted_text: Text given to the model for TEXT files.
        children: Files nested in a forwarded e-mail.
    """

    file_name: str
    mime_type: str
    content: bytes
    role: str
    model_input: str
    model_input_reason: str | None = None
    extracted_text: str | None = None
    children: list["StoredFile"] = field(default_factory=list)


def guess_mime(file_name: str, declared: str | None = None) -> str:
    """Best-effort content type from the declared type and the file extension.

    Args:
        file_name: File name (extension is used).
        declared: Content type sent by the client / MIME header.

    Returns:
        Content type; generic ``application/octet-stream`` types are refined by extension.
    """
    if declared and declared not in {"application/octet-stream", "binary/octet-stream"}:
        return declared.lower()
    return mimetypes.guess_type(file_name)[0] or "application/octet-stream"


def classify_document(file_name: str, mime_type: str, content: bytes) -> StoredFile:
    """Classify an attachment or uploaded file and extract text where needed.

    Args:
        file_name: File name.
        mime_type: Content type (see ``guess_mime``).
        content: Raw bytes.

    Returns:
        StoredFile with role DOCUMENT and the MODEL_INPUT decision.
    """
    suffix = PurePath(file_name).suffix.lower()
    stored = StoredFile(file_name, mime_type, content, ROLE_DOCUMENT, "NONE")
    try:
        if mime_type in NATIVE_MIME:
            stored.model_input = "NATIVE"
        elif mime_type == DOCX_MIME or suffix == ".docx":
            stored.model_input, stored.extracted_text = "TEXT", docx_to_text(content)
        elif mime_type == XLSX_MIME or suffix == ".xlsx":
            stored.model_input, stored.extracted_text = "TEXT", xlsx_to_text(content)
        elif mime_type.startswith("text/") or suffix in {".txt", ".csv"}:
            stored.model_input, stored.extracted_text = "TEXT", content.decode("utf-8", errors="replace")
        elif suffix == ".msg":
            stored.model_input_reason = "Outlook .msg files are not supported"
        else:
            stored.model_input_reason = f"unsupported file type {mime_type}"
    except ValueError as exc:
        stored.model_input, stored.model_input_reason = "NONE", str(exc)
    return stored
