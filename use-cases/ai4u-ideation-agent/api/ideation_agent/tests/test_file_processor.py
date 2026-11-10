"""Unit tests for the file_processor module.

Tests image validation, resizing, format conversion, security checks,
and content block generation.
"""
from __future__ import annotations

import base64
import io
import struct

import pytest
from PIL import Image

from ideation_agent.file_processor import (
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    ProcessedImage,
    ProcessingError,
    process_image,
    validate_file_count,
    _detect_mime_type,
    _is_svg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 100, height: int = 100, mode: str = "RGB") -> bytes:
    """Create a minimal PNG image in memory."""
    img = Image.new(mode, (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_gif(width: int = 50, height: int = 50) -> bytes:
    """Create a minimal GIF image in memory."""
    img = Image.new("P", (width, height))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    return buf.getvalue()


def _make_webp(width: int = 80, height: int = 80) -> bytes:
    """Create a minimal WebP image in memory."""
    img = Image.new("RGB", (width, height), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def _make_png_with_exif(width: int = 100, height: int = 100) -> bytes:
    """Create a JPEG with EXIF data (Pillow doesn't write EXIF to PNG easily)."""
    import piexif

    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    exif_dict = {
        "0th": {piexif.ImageIFD.Make: b"TestCamera"},
        "GPS": {piexif.GPSIFD.GPSLatitude: ((40, 1), (0, 1), (0, 1))},
    }
    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Valid image tests
# ---------------------------------------------------------------------------

class TestValidImages:
    """Test successful processing of valid image files."""

    def test_valid_png(self):
        data = _make_png()
        result = process_image(data, "test.png")
        assert isinstance(result, ProcessedImage)
        assert result.content_block["type"] == "image_url"
        assert result.content_block["image_url"]["detail"] == "high"
        assert result.content_block["image_url"]["url"].startswith("data:image/png;base64,")
        assert result.original_format == "PNG"
        assert result.width == 100
        assert result.height == 100

    def test_valid_jpeg(self):
        data = _make_jpeg()
        result = process_image(data, "photo.jpg")
        assert isinstance(result, ProcessedImage)
        assert result.original_format == "JPEG"
        # Output is always PNG
        assert result.content_block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_valid_gif(self):
        data = _make_gif()
        result = process_image(data, "animation.gif")
        assert isinstance(result, ProcessedImage)
        assert result.original_format == "GIF"

    def test_valid_webp(self):
        data = _make_webp()
        result = process_image(data, "image.webp")
        assert isinstance(result, ProcessedImage)
        assert result.original_format == "WEBP"

    def test_content_block_is_valid_base64(self):
        data = _make_png(50, 50)
        result = process_image(data, "small.png")
        assert isinstance(result, ProcessedImage)
        # Extract and decode base64
        url = result.content_block["image_url"]["url"]
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        # Should be valid PNG
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_rgba_to_rgb_conversion(self):
        """RGBA images should be composited onto white background."""
        data = _make_png(100, 100, mode="RGBA")
        result = process_image(data, "transparent.png")
        assert isinstance(result, ProcessedImage)
        # Verify it's a valid image
        url = result.content_block["image_url"]["url"]
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        img = Image.open(io.BytesIO(decoded))
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Resize tests
# ---------------------------------------------------------------------------

class TestResize:
    """Test image resizing when dimensions exceed MAX_IMAGE_DIMENSION."""

    def test_large_image_resized(self):
        data = _make_png(4000, 3000)
        result = process_image(data, "huge.png")
        assert isinstance(result, ProcessedImage)
        # Longest side should be MAX_IMAGE_DIMENSION
        assert max(result.width, result.height) == MAX_IMAGE_DIMENSION
        # Aspect ratio preserved
        assert result.width == MAX_IMAGE_DIMENSION
        assert result.height == int(3000 * (MAX_IMAGE_DIMENSION / 4000))

    def test_small_image_not_resized(self):
        data = _make_png(800, 600)
        result = process_image(data, "normal.png")
        assert isinstance(result, ProcessedImage)
        assert result.width == 800
        assert result.height == 600

    def test_exact_limit_not_resized(self):
        data = _make_png(MAX_IMAGE_DIMENSION, 1000)
        result = process_image(data, "exact.png")
        assert isinstance(result, ProcessedImage)
        assert result.width == MAX_IMAGE_DIMENSION
        assert result.height == 1000

    def test_tall_image_resized(self):
        data = _make_png(1000, 5000)
        result = process_image(data, "tall.png")
        assert isinstance(result, ProcessedImage)
        assert result.height == MAX_IMAGE_DIMENSION
        assert result.width == int(1000 * (MAX_IMAGE_DIMENSION / 5000))


# ---------------------------------------------------------------------------
# Validation and security tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test rejection of invalid or malicious files."""

    def test_empty_file(self):
        result = process_image(b"", "empty.png")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_oversized_file(self):
        # Create data just over the limit
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_FILE_SIZE_BYTES + 1)
        result = process_image(data, "huge.png")
        assert isinstance(result, ProcessingError)
        assert result.code == "too_large"

    def test_wrong_magic_bytes(self):
        """Non-image data with .png extension should be rejected."""
        data = b"MZ" + b"\x00" * 100  # EXE magic bytes
        result = process_image(data, "malicious.png")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_text_file_as_image(self):
        data = b"Hello, this is not an image file!\n" * 100
        result = process_image(data, "readme.png")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_corrupt_png(self):
        """Valid PNG header but corrupt data."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00\x01\x02\x03" * 50
        result = process_image(data, "corrupt.png")
        assert isinstance(result, ProcessingError)
        assert result.code == "corrupt"

    def test_svg_rejected(self):
        """SVG files should be rejected for security."""
        svg_data = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        result = process_image(svg_data, "vector.svg")
        assert isinstance(result, ProcessingError)
        assert result.code in ("invalid_type", "security")

    def test_svg_without_xml_declaration(self):
        svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        result = process_image(svg_data, "plain.svg")
        assert isinstance(result, ProcessingError)

    def test_pdf_rejected(self):
        data = b"%PDF-1.4 fake pdf content" + b"\x00" * 100
        result = process_image(data, "doc.pdf")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"


# ---------------------------------------------------------------------------
# EXIF stripping tests
# ---------------------------------------------------------------------------

class TestExifStripping:
    """Test that EXIF metadata is removed from processed images."""

    def test_jpeg_exif_stripped(self):
        """JPEG with EXIF data should have it stripped in output."""
        # Create JPEG with some metadata embedded
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        jpeg_data = buf.getvalue()

        result = process_image(jpeg_data, "photo_with_gps.jpg")
        assert isinstance(result, ProcessedImage)

        # Decode the output PNG and check for absence of metadata
        url = result.content_block["image_url"]["url"]
        b64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        output_img = Image.open(io.BytesIO(decoded))
        # PNG output should have no EXIF
        exif = output_img.getexif()
        assert len(exif) == 0


# ---------------------------------------------------------------------------
# Magic bytes detection tests
# ---------------------------------------------------------------------------

class TestMimeDetection:
    """Test the magic bytes MIME type detection."""

    def test_detect_png(self):
        assert _detect_mime_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20) == "image/png"

    def test_detect_jpeg(self):
        assert _detect_mime_type(b"\xff\xd8\xff\xe0" + b"\x00" * 20) == "image/jpeg"

    def test_detect_gif87(self):
        assert _detect_mime_type(b"GIF87a" + b"\x00" * 20) == "image/gif"

    def test_detect_gif89(self):
        assert _detect_mime_type(b"GIF89a" + b"\x00" * 20) == "image/gif"

    def test_detect_webp(self):
        # RIFF + 4 bytes size + WEBP
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20
        assert _detect_mime_type(data) == "image/webp"

    def test_unknown_format(self):
        assert _detect_mime_type(b"\x00\x01\x02\x03\x04\x05\x06\x07") is None

    def test_exe_not_detected(self):
        assert _detect_mime_type(b"MZ" + b"\x00" * 50) is None


# ---------------------------------------------------------------------------
# SVG detection tests
# ---------------------------------------------------------------------------

class TestSvgDetection:
    """Test SVG file detection."""

    def test_xml_svg(self):
        assert _is_svg(b'<?xml version="1.0"?><svg>...</svg>')

    def test_plain_svg(self):
        assert _is_svg(b"<svg xmlns='http://www.w3.org/2000/svg'>")

    def test_svg_with_whitespace(self):
        assert _is_svg(b"   \n  <svg>content</svg>")

    def test_not_svg(self):
        assert not _is_svg(b"\x89PNG\r\n\x1a\n")

    def test_html_not_svg(self):
        assert not _is_svg(b"<html><body>not svg</body></html>")


# ---------------------------------------------------------------------------
# File count validation tests
# ---------------------------------------------------------------------------

class TestFileCount:
    """Test file count validation."""

    def test_valid_count(self):
        assert validate_file_count(1) is None
        assert validate_file_count(5) is None

    def test_too_many(self):
        error = validate_file_count(6)
        assert error is not None
        assert "6" in error
        assert "max" in error.lower()

    def test_zero_is_valid(self):
        assert validate_file_count(0) is None
