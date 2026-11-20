"""Upload metadata validation and bounded byte reading for the ECN intake layer.

Provides ``validate_upload_metadata`` to check filename and media type against
kind-specific allowlists, and ``read_upload_bytes`` to stream an UploadFile in
chunks while enforcing a hard byte limit without trusting Content-Length.

Sensitive source content is never embedded in error messages or details.

Usage::

    from app.ecn.intake.files import validate_upload_metadata, read_upload_bytes
    from app.ecn.models import ArtifactKind

    metadata = validate_upload_metadata("ECN.JSON", "application/json", ArtifactKind.ECN_JSON)
    content = await read_upload_bytes(upload, maximum_bytes=25 * 1024 * 1024)
"""

import re
from dataclasses import dataclass

from app.ecn.errors import SourceValidationError, UploadTooLargeError
from app.ecn.models import ArtifactKind, JsonValue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f\x80-\x9f]")
"""Matches ASCII C0 control chars (0x00-0x1F), DEL (0x7F), and C1 (0x80-0x9F)."""

_FILE_NAME_MAX = 500
"""Maximum allowed length for a sanitised upload filename (characters)."""

_MEDIA_TYPE_MAX = 200
"""Maximum allowed length for a normalised media type string (characters)."""

# Accepted normalised media types per artifact kind.
_ALLOWED_MEDIA: dict[ArtifactKind, frozenset[str]] = {
    ArtifactKind.ECN_JSON: frozenset({
        "application/json",
        "text/json",
        "application/octet-stream",
        "",
    }),
    ArtifactKind.STOCK_WORKBOOK: frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/octet-stream"}),
    ArtifactKind.BOM_WORKBOOK: frozenset({
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "",
    }),
    ArtifactKind.CONDITIONAL_RULE_WORKBOOK: frozenset({
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "",
    }),
    ArtifactKind.MATERIAL_MASTER_WORKBOOK: frozenset({
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "",
    }),
    ArtifactKind.PURCHASE_INFO_WORKBOOK: frozenset({
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "",
    }),
    ArtifactKind.PURCHASE_INFO_PRICE_CSV: frozenset({
        "text/csv",
        "application/csv",
        "application/octet-stream",
        "",
    }),
}

# Required file extension per artifact kind (case-insensitive match applied by caller).
_REQUIRED_EXTENSION: dict[ArtifactKind, str] = {
    ArtifactKind.ECN_JSON: ".json",
    ArtifactKind.STOCK_WORKBOOK: ".xlsx",
    ArtifactKind.BOM_WORKBOOK: ".xlsx",
    ArtifactKind.CONDITIONAL_RULE_WORKBOOK: ".xlsx",
    ArtifactKind.MATERIAL_MASTER_WORKBOOK: ".xlsx",
    ArtifactKind.PURCHASE_INFO_WORKBOOK: ".xlsx",
    ArtifactKind.PURCHASE_INFO_PRICE_CSV: ".csv",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedUploadMetadata:
    """Hold safe display metadata produced by kind-specific allowlist checks.

    ``file_name`` is the sanitised basename (control characters removed,
    whitespace trimmed, no path separators).  ``media_type`` is the
    normalised (stripped, lowercased) media type string.

    Args:
        file_name: Sanitised upload filename (basename only).
        media_type: Normalised media type string; empty string when absent.
    """

    file_name: str
    media_type: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_basename(raw: str | None) -> str:
    """Extract the basename, strip control characters, and trim whitespace.

    Takes the final segment after any ``/`` or ``\\`` path separator, removes
    all ASCII/C1 control characters, trims external whitespace, and rejects
    names that are empty or exceed ``_FILE_NAME_MAX`` characters.

    Args:
        raw: The raw filename as supplied by the caller; ``None`` is rejected.

    Returns:
        The sanitised, non-empty basename string.

    Raises:
        SourceValidationError: If ``raw`` is ``None``, empty, or too long after
            sanitisation.  Error details do not contain the raw filename.
    """
    if raw is None:
        raise SourceValidationError(
            "UPLOAD_FILE_NAME_MISSING",
            "A file name is required",
            status_code=400,
        )
    # Strip to the final path segment only (no filesystem path creation).
    segment = raw
    for sep in ("/", "\\"):
        if sep in segment:
            segment = segment.rsplit(sep, 1)[-1]
    # Remove ASCII C0, DEL, and C1 control characters, then trim whitespace.
    cleaned = _CONTROL_CHAR.sub("", segment).strip()
    if not cleaned:
        raise SourceValidationError(
            "UPLOAD_FILE_NAME_EMPTY",
            "File name must not be empty after sanitisation",
            status_code=400,
        )
    if len(cleaned) > _FILE_NAME_MAX:
        raise SourceValidationError(
            "UPLOAD_FILE_NAME_TOO_LONG",
            f"File name must not exceed {_FILE_NAME_MAX} characters",
            status_code=400,
        )
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_upload_metadata(
    file_name: str | None,
    media_type: str | None,
    artifact_kind: ArtifactKind,
) -> ValidatedUploadMetadata:
    """Validate upload filename and media type against kind-specific allowlists.

    Performs basename extraction, control-character removal, and whitespace
    trimming on the filename.  Normalises the media type to stripped lowercase
    and checks both extension and media type against ``_ALLOWED_MEDIA`` and
    ``_REQUIRED_EXTENSION``.

    Error details never contain the raw filename or media type to avoid
    leaking unexpected user-controlled content.

    Args:
        file_name: Raw filename as received from the upload field; may be
            ``None`` when the multipart field provides no name.
        media_type: Raw ``Content-Type`` header value; may be ``None`` or empty.
        artifact_kind: The expected kind of the artifact being uploaded.

    Returns:
        A ``ValidatedUploadMetadata`` with sanitised ``file_name`` and
        normalised ``media_type``.

    Raises:
        SourceValidationError: HTTP 400 when filename or media type violates
            the allowlist rules.
    """
    safe_name = _safe_basename(file_name)

    # Normalise media type: strip outer whitespace and lowercase.
    raw_media = media_type or ""
    normalized_media = raw_media.strip().lower()
    if len(normalized_media) > _MEDIA_TYPE_MAX:
        raise SourceValidationError(
            "UPLOAD_MEDIA_TYPE_TOO_LONG",
            f"Media type must not exceed {_MEDIA_TYPE_MAX} characters",
            status_code=400,
        )

    # Extension check (case-insensitive).
    required_ext = _REQUIRED_EXTENSION[artifact_kind]
    if not safe_name.lower().endswith(required_ext):
        raise SourceValidationError(
            "UPLOAD_EXTENSION_INVALID",
            f"Files of kind {artifact_kind} must use the {required_ext} extension",
            status_code=400,
            details={"artifact_kind": str(artifact_kind), "required_extension": required_ext},
        )

    # Media type allowlist check.
    allowed = _ALLOWED_MEDIA[artifact_kind]
    if normalized_media not in allowed:
        raise SourceValidationError(
            "UPLOAD_MEDIA_TYPE_INVALID",
            f"Media type is not accepted for uploads of kind {artifact_kind}",
            status_code=400,
            details={"artifact_kind": str(artifact_kind)},
        )

    return ValidatedUploadMetadata(file_name=safe_name, media_type=normalized_media)


async def read_upload_bytes(
    upload: object,
    maximum_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> bytes:
    """Read an upload in bounded chunks, rejecting the first byte beyond the limit.

    Calls ``await upload.read(chunk_size)`` repeatedly.  The limit is enforced
    on the *actual* streamed byte count; Content-Length is never consulted.
    Neither closes nor rewinds the upload object — the caller owns lifecycle.

    Args:
        upload: Any async-readable object with an ``async def read(size) -> bytes``
            method; typically a FastAPI ``UploadFile``.
        maximum_bytes: Maximum number of bytes to accept (must be positive).
        chunk_size: Number of bytes to request per read call (must be positive).
            Defaults to 1 MiB.

    Returns:
        The complete upload content as a single ``bytes`` object.

    Raises:
        ValueError: If ``maximum_bytes`` or ``chunk_size`` is not positive.
        UploadTooLargeError: HTTP 413 when the stream exceeds ``maximum_bytes``.
            The error details include ``maximum_bytes`` but never the content.
    """
    if maximum_bytes <= 0 or chunk_size <= 0:
        raise ValueError("upload and chunk limits must be positive")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk: bytes = await upload.read(chunk_size)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > maximum_bytes:
            raise UploadTooLargeError(
                "UPLOAD_TOO_LARGE",
                f"Uploaded file exceeds the {maximum_bytes}-byte limit",
                status_code=413,
                details={"maximum_bytes": maximum_bytes},
            )
        chunks.append(chunk)
