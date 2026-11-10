"""Image file processor for multimodal ideation agent.

Validates, resizes, and converts uploaded images into LangChain-compatible
content blocks for native multimodal input to the LLM.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB (supports PDF uploads)
MAX_FILES_PER_MESSAGE = 5
MAX_IMAGE_DIMENSION = 2048  # px longest side
MAX_IMAGE_PIXELS = 100_000_000  # Pillow decompression bomb guard

ALLOWED_MIME_TYPES = {
    # Images
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    # PDF
    "application/pdf",
    # Office XML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Legacy Office
    "application/msword",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
    # OpenDocument
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
    # RTF
    "application/rtf",
    # Data
    "text/csv",
    "text/tab-separated-values",
    # Email / Web Archive
    "message/rfc822",
    # Text / Code
    "text/plain",
    "text/markdown",
    "application/json",
    "text/xml",
    "text/yaml",
    "text/x-python",
    "text/javascript",
    "text/html",
    "text/css",
    "application/sql",
}

# Magic byte signatures for image format validation
_MAGIC_BYTES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP starts with RIFF....WEBP
}

# SVG detection patterns (reject for security — XSS vector)
_SVG_SIGNATURES = [b"<?xml", b"<svg", b"<!DOCTYPE svg"]


# --- Data classes ------------------------------------------------------------

@dataclass
class ProcessedImage:
    """Result of successful image processing."""
    content_block: dict
    """LangChain-compatible content block: {"type": "image_url", "image_url": {...}}"""
    original_size_bytes: int
    processed_size_bytes: int
    width: int
    height: int
    original_format: str


@dataclass
class ProcessingError:
    """Result of failed image processing."""
    error: str
    code: str  # "invalid_type" | "corrupt" | "too_large" | "security"


# --- Public API --------------------------------------------------------------

def process_image(
    file_data: bytes,
    filename: str,
    content_type: Optional[str] = None,
) -> ProcessedImage | ProcessingError:
    """Process an uploaded image file into a LangChain content block.

    Pipeline:
    1. Validate file size
    2. Validate magic bytes (reject MIME spoofing, SVG)
    3. Open with Pillow (validates image integrity)
    4. Strip EXIF metadata
    5. Resize if too large
    6. Convert to PNG and base64 encode
    7. Return content block

    Args:
        file_data: Raw file bytes.
        filename: Original filename (for logging/error messages).
        content_type: Declared content type from upload (optional, not trusted).

    Returns:
        ProcessedImage on success, ProcessingError on failure.
    """
    # 1. File size check
    if len(file_data) > MAX_FILE_SIZE_BYTES:
        return ProcessingError(
            error=f"File too large: {len(file_data) / 1024 / 1024:.1f} MB "
                  f"(max {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB)",
            code="too_large",
        )

    if len(file_data) == 0:
        return ProcessingError(error="Empty file", code="invalid_type")

    # 2. Magic bytes validation
    detected_mime = _detect_mime_type(file_data)
    if detected_mime is None:
        return ProcessingError(
            error=f"Unsupported file type for '{filename}'. "
                  f"Accepted: PNG, JPEG, GIF, WEBP.",
            code="invalid_type",
        )

    # Check for SVG (security — embedded scripts)
    if _is_svg(file_data):
        return ProcessingError(
            error="SVG files are not supported (security restriction).",
            code="security",
        )

    # Cross-check declared content_type if provided
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        # Declared type doesn't match what we accept, but magic bytes passed —
        # trust the magic bytes over the declared type
        logger.warning(
            "MIME mismatch: declared=%s detected=%s file=%s — using detected",
            content_type, detected_mime, filename,
        )

    # 3. Open with Pillow (validates image integrity)
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(file_data))
        img.load()  # Force full decode — catches truncated/corrupt images
    except Exception as exc:
        logger.warning("Failed to open image '%s': %s", filename, exc)
        return ProcessingError(
            error=f"Cannot open image '{filename}': file appears corrupt or invalid.",
            code="corrupt",
        )

    original_format = img.format or "UNKNOWN"

    # 4. Strip EXIF metadata (privacy — GPS, device info)
    img = ImageOps.exif_transpose(img)  # Apply EXIF rotation first
    # Remove all metadata by copying pixel data to a new image
    img_clean = Image.new(img.mode, img.size)
    if hasattr(img, "get_flattened_data"):
        img_clean.putdata(img.get_flattened_data())
    else:
        img_clean.putdata(list(img.getdata()))
    img = img_clean

    # Convert to RGB if needed (RGBA → RGB for smaller PNG, P/L → RGB for compatibility)
    if img.mode in ("RGBA", "LA"):
        # Composite onto white background to preserve transparency info visually
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img, mask=img.split()[1])
        img = background
    elif img.mode not in ("RGB",):
        img = img.convert("RGB")

    # 5. Resize if longest side exceeds limit
    width, height = img.size
    if max(width, height) > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        width, height = img.size

    # 6. Encode to PNG → base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()
    b64_data = base64.b64encode(png_bytes).decode("ascii")

    # 7. Build content block
    content_block = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64_data}",
            "detail": "high",
        },
    }

    return ProcessedImage(
        content_block=content_block,
        original_size_bytes=len(file_data),
        processed_size_bytes=len(png_bytes),
        width=width,
        height=height,
        original_format=original_format,
    )


def validate_file_count(count: int) -> Optional[str]:
    """Validate the number of files in a message. Returns error string or None."""
    if count > MAX_FILES_PER_MESSAGE:
        return (
            f"Too many files: {count} (max {MAX_FILES_PER_MESSAGE} per message)"
        )
    return None


# --- Private helpers ---------------------------------------------------------

def _detect_mime_type(data: bytes) -> Optional[str]:
    """Detect image MIME type from magic bytes. Returns None if not recognized."""
    for magic, mime in _MAGIC_BYTES.items():
        if magic == b"RIFF":
            # WebP: starts with RIFF + 4 bytes size + WEBP
            if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                return mime
        elif data.startswith(magic):
            return mime
    return None


def _is_svg(data: bytes) -> bool:
    """Check if file content looks like SVG (even if disguised)."""
    # Check first 1KB for SVG signatures (may have BOM or whitespace prefix)
    header = data[:1024].lstrip()
    header_lower = header.lower()
    for sig in _SVG_SIGNATURES:
        if header_lower.startswith(sig):
            return True
    return False
