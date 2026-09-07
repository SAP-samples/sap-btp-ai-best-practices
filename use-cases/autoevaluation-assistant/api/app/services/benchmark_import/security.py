"""Bounded ZIP safety validation for untrusted XLSX benchmark sources."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

from .constants import (
    MAX_COMPRESSION_RATIO,
    MAX_SOURCE_BYTES,
    MAX_UNCOMPRESSED_BYTES,
    MAX_ZIP_MEMBERS,
)


class WorkbookSafetyError(ValueError):
    """Describe one blocking source-size or XLSX container safety failure.

    Inputs:
        code: Stable validation code exposed to import callers.
        message: Human-readable safety failure description.

    Outputs:
        Exception carrying structured code and message attributes.
    """

    def __init__(self, code: str, message: str) -> None:
        """Initialize a structured workbook safety failure.

        Inputs:
            code: Stable machine-readable validation category.
            message: Human-readable reason the source is unsafe.

        Outputs:
            None. ``ValueError`` is initialized with the same message.
        """

        self.code = code
        super().__init__(message)


def validate_xlsx_container(content: bytes) -> None:
    """Reject oversized, malformed, encrypted, or ZIP-bomb-like XLSX sources.

    Inputs:
        content: Raw workbook bytes supplied by an upload or CLI path.

    Outputs:
        None. Safe containers return without reading member payloads.

    Raises:
        WorkbookSafetyError: If source or archive limits are violated.
    """

    if len(content) > MAX_SOURCE_BYTES:
        raise WorkbookSafetyError(
            "source_too_large",
            f"Workbook exceeds the {MAX_SOURCE_BYTES}-byte source limit",
        )
    if not content or not zipfile.is_zipfile(io.BytesIO(content)):
        raise WorkbookSafetyError("malformed_xlsx", "Workbook is not a valid XLSX ZIP")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ZIP_MEMBERS:
                raise WorkbookSafetyError(
                    "unsafe_zip",
                    f"Workbook ZIP contains {len(members)} members; limit is {MAX_ZIP_MEMBERS}",
                )
            total_uncompressed = 0
            for member in members:
                total_uncompressed += member.file_size
                path = PurePosixPath(member.filename)
                if member.flag_bits & 0x1:
                    raise WorkbookSafetyError(
                        "unsafe_zip",
                        f"Workbook ZIP member is encrypted: {member.filename}",
                    )
                if path.is_absolute() or ".." in path.parts:
                    raise WorkbookSafetyError(
                        "unsafe_zip",
                        f"Workbook ZIP member has an unsafe path: {member.filename}",
                    )
                ratio = member.file_size / max(member.compress_size, 1)
                if ratio > MAX_COMPRESSION_RATIO:
                    raise WorkbookSafetyError(
                        "unsafe_zip",
                        f"Workbook ZIP member compression ratio is unsafe: {member.filename}",
                    )
            if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
                raise WorkbookSafetyError(
                    "unsafe_zip",
                    "Workbook ZIP exceeds the uncompressed-size limit",
                )
    except zipfile.BadZipFile as exc:
        raise WorkbookSafetyError(
            "malformed_xlsx",
            "Workbook is not a readable XLSX ZIP",
        ) from exc
