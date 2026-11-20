"""Bounded, signature-checked transient drawing intake."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from fastapi import UploadFile

from app.drawing_feasibility.errors import DrawingUploadTooLargeError, DrawingValidationError
from app.drawing_feasibility.models import DrawingInput


_ALLOWED_MEDIA_TYPES = {
    "application/pdf": {".pdf"},
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
}
_SIGNATURES = {
    "application/pdf": lambda content: content.startswith(b"%PDF-"),
    "image/png": lambda content: content.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda content: content.startswith(b"\xff\xd8\xff"),
}
_FILENAME_MAX = 500
_MEDIA_TYPE_MAX = 100


@dataclass(frozen=True)
class ValidatedDrawingMetadata:
    """Safe, presentation-ready upload metadata after allowlist validation."""

    file_name: str
    media_type: str


def validate_drawing_metadata(
    file_name: str | None, media_type: str | None
) -> ValidatedDrawingMetadata:
    """Accept only one supported drawing filename/media-type combination."""
    name = PurePath(file_name or "").name.strip()
    content_type = (media_type or "").strip().lower()
    if (
        not name
        or len(name) > _FILENAME_MAX
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise DrawingValidationError(
            "DRAWING_FILE_NAME_INVALID", "Drawing file name is invalid"
        )
    if (
        not content_type
        or len(content_type) > _MEDIA_TYPE_MAX
        or content_type not in _ALLOWED_MEDIA_TYPES
    ):
        raise DrawingValidationError(
            "DRAWING_FILE_TYPE_UNSUPPORTED",
            "Only PDF, PNG, and JPEG drawings are supported",
        )
    if PurePath(name).suffix.lower() not in _ALLOWED_MEDIA_TYPES[content_type]:
        raise DrawingValidationError(
            "DRAWING_FILE_TYPE_UNSUPPORTED",
            "Drawing filename and media type do not match a supported format",
        )
    return ValidatedDrawingMetadata(file_name=name, media_type=content_type)


async def read_drawing_upload(upload: UploadFile, maximum_bytes: int) -> DrawingInput:
    """Read one upload in bounded chunks and verify its declared binary format.

    Drawing bytes exist only in this returned request-local object. The caller
    must not persist, log, or echo them.
    """
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    metadata = validate_drawing_metadata(upload.filename, upload.content_type)
    chunks: list[bytes] = []
    size = 0
    try:
        while True:
            chunk = await upload.read(min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise DrawingUploadTooLargeError(maximum_bytes)
            chunks.append(chunk)
    finally:
        await upload.close()
    content = b"".join(chunks)
    if not content or not _SIGNATURES[metadata.media_type](content):
        raise DrawingValidationError(
            "DRAWING_CONTENT_INVALID", "Drawing content is empty or invalid"
        )
    return DrawingInput(
        file_name=metadata.file_name, media_type=metadata.media_type, content=content
    )
