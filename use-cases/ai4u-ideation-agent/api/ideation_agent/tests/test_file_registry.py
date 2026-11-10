"""Unit tests for the file processing registry.

Tests the registry routing, image processor, PDF processor, Office/text/email
processors, and error handling.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

from ideation_agent.file_registry import (
    ProcessedFile,
    ProcessingError,
    process_file,
    process_image_file,
    process_pdf_file,
    PROCESSORS,
    SUPPORTED_MIME_TYPES,
    CONTENT_MAX_DIMENSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_rgba_png(width: int = 50, height: int = 50) -> bytes:
    img = Image.new("RGBA", (width, height), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Registry routing tests
# ---------------------------------------------------------------------------

class TestRegistryRouting:
    """Test that process_file routes by MIME type correctly."""

    def test_routes_png(self):
        result = process_file(_make_png(), "test.png", "image/png", "id-1")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "image/png"

    def test_routes_jpeg(self):
        result = process_file(_make_jpeg(), "test.jpg", "image/jpeg", "id-2")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "image/jpeg"

    def test_routes_by_magic_bytes_over_declared_type(self):
        """If declared MIME is wrong but magic bytes are valid, still works."""
        png_data = _make_png()
        result = process_file(png_data, "photo.jpg", "image/jpeg", "id-3")
        assert isinstance(result, ProcessedFile)
        # Magic bytes detect PNG
        assert result.original_type == "image/png"

    def test_unsupported_type_error(self):
        result = process_file(b"\x00\x01\x02\x03", "data.bin", "application/octet-stream", "id-4")
        assert isinstance(result, ProcessingError)
        assert result.code == "unsupported_type"

    def test_supported_mime_types(self):
        """All image MIME types should be registered."""
        assert "image/png" in SUPPORTED_MIME_TYPES
        assert "image/jpeg" in SUPPORTED_MIME_TYPES
        assert "image/gif" in SUPPORTED_MIME_TYPES
        assert "image/webp" in SUPPORTED_MIME_TYPES


# ---------------------------------------------------------------------------
# Image processor tests
# ---------------------------------------------------------------------------

class TestImageProcessor:
    """Test process_image_file output."""

    def test_returns_processed_file(self):
        result = process_image_file(_make_png(200, 150), "test.png", "file-id")
        assert isinstance(result, ProcessedFile)
        assert result.file_id == "file-id"
        assert result.filename == "test.png"
        assert result.num_outputs == 1
        assert result.dimensions == {"width": 200, "height": 150}

    def test_output_is_png_bytes(self):
        result = process_image_file(_make_png(), "test.png", "id")
        assert isinstance(result, ProcessedFile)
        assert len(result.outputs) == 1
        assert result.outputs[0][:8] == b"\x89PNG\r\n\x1a\n"

    def test_content_output_resized_to_max(self):
        """Large image content output should be <= 2048px on longest side."""
        result = process_image_file(_make_png(4000, 3000), "huge.png", "id")
        assert isinstance(result, ProcessedFile)
        output_img = Image.open(io.BytesIO(result.outputs[0]))
        assert max(output_img.size) <= CONTENT_MAX_DIMENSION

    def test_small_image_not_upscaled(self):
        """Small images should not be enlarged."""
        result = process_image_file(_make_png(50, 30), "small.png", "id")
        assert isinstance(result, ProcessedFile)
        assert result.dimensions == {"width": 50, "height": 30}

    def test_rgba_converted_to_rgb(self):
        """RGBA images should be composited to RGB."""
        result = process_image_file(_make_rgba_png(), "alpha.png", "id")
        assert isinstance(result, ProcessedFile)
        # Output should be RGB (no alpha channel)
        output_img = Image.open(io.BytesIO(result.outputs[0]))
        assert output_img.mode == "RGB"

    def test_corrupt_image_error(self):
        corrupt = b"\x89PNG\r\n\x1a\n" + b"\xff" * 200
        result = process_image_file(corrupt, "broken.png", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "corrupt"

    def test_svg_rejected(self):
        svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>'
        result = process_image_file(svg, "vector.svg", "id")
        assert isinstance(result, ProcessingError)
        # SVG has no valid image magic bytes, so it hits invalid_type first
        assert result.code in ("security", "invalid_type")

    def test_non_image_magic_bytes(self):
        """File with non-image magic bytes should be rejected."""
        exe_data = b"MZ" + b"\x00" * 500
        result = process_image_file(exe_data, "file.png", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"


# ---------------------------------------------------------------------------
# PDF processor tests
# ---------------------------------------------------------------------------

# Minimal valid PDF (1 page, valid structure)
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n183\n"
    b"%%EOF\n"
)


class TestPdfProcessor:
    """Test process_pdf_file functionality."""

    def test_valid_pdf_returns_processed_file(self):
        result = process_pdf_file(_MINIMAL_PDF, "report.pdf", "pdf-id-1")
        assert isinstance(result, ProcessedFile)
        assert result.file_id == "pdf-id-1"
        assert result.filename == "report.pdf"
        assert result.original_type == "application/pdf"
        assert result.num_outputs == 1
        assert result.dimensions is None
        assert result.metadata == {"format": "pdf"}

    def test_pdf_output_is_raw_pdf_bytes(self):
        """PDF output should be the raw PDF bytes (not rendered PNG)."""
        result = process_pdf_file(_MINIMAL_PDF, "doc.pdf", "id")
        assert isinstance(result, ProcessedFile)
        assert len(result.outputs) == 1
        assert result.outputs[0] == _MINIMAL_PDF

    def test_pdf_magic_bytes_detection(self):
        """_detect_mime_type should recognize %PDF."""
        from ideation_agent.file_registry import _detect_mime_type
        assert _detect_mime_type(_MINIMAL_PDF) == "application/pdf"

    def test_process_file_routes_pdf(self):
        """process_file routes application/pdf correctly."""
        result = process_file(_MINIMAL_PDF, "paper.pdf", "application/pdf", "id-pdf")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/pdf"

    def test_process_file_routes_pdf_by_magic_bytes(self):
        """process_file detects PDF via magic bytes even if declared type is wrong."""
        result = process_file(_MINIMAL_PDF, "file.bin", "application/octet-stream", "id-pdf2")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/pdf"

    def test_invalid_header_rejected(self):
        """Non-PDF data should be rejected."""
        result = process_pdf_file(b"NOT A PDF FILE DATA", "fake.pdf", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_encrypted_pdf_rejected(self):
        """Password-protected PDFs should be rejected with clear message."""
        encrypted_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>\n"
            b"/Encrypt<</Filter/Standard/V 2/Length 128>>\n"
            b"endobj\n"
            b"%%EOF\n"
        )
        result = process_pdf_file(encrypted_pdf, "secret.pdf", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "corrupt"
        assert "encrypted" in result.error.lower() or "password" in result.error.lower()

    def test_empty_pdf_header_only(self):
        """A file with just %PDF and nothing else should still process (LLM handles it)."""
        # This is a very short/malformed PDF, but has valid header
        short_pdf = b"%PDF-1.0\n%%EOF\n"
        result = process_pdf_file(short_pdf, "tiny.pdf", "id")
        # Should succeed — we let the LLM handle potentially malformed PDFs
        assert isinstance(result, ProcessedFile)

    def test_pdf_in_supported_mime_types(self):
        """application/pdf should be in SUPPORTED_MIME_TYPES."""
        assert "application/pdf" in SUPPORTED_MIME_TYPES

    def test_pdf_in_processors(self):
        """application/pdf should be registered in PROCESSORS."""
        assert "application/pdf" in PROCESSORS


# ---------------------------------------------------------------------------
# Office document processor tests
# ---------------------------------------------------------------------------

import zipfile
import io


def _make_docx_bytes():
    """Create a minimal DOCX file (ZIP with [Content_Types].xml)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>')
        zf.writestr("word/document.xml", "<document>Hello World</document>")
    return buf.getvalue()


def _make_pptx_bytes():
    """Create a minimal PPTX file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
            '</Types>')
        zf.writestr("ppt/presentation.xml", "<presentation>Slide 1</presentation>")
    return buf.getvalue()


def _make_xlsx_bytes():
    """Create a minimal XLSX file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '</Types>')
        zf.writestr("xl/workbook.xml", "<workbook>Sheet1</workbook>")
    return buf.getvalue()


def _make_odt_bytes():
    """Create a minimal ODT file (ZIP with mimetype)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", "<office:document>Test</office:document>")
    return buf.getvalue()


class TestOfficeProcessor:
    """Test process_office_file and ZIP-based MIME detection."""

    def test_docx_detection_and_processing(self):
        """DOCX files should be detected and processed correctly."""
        docx_data = _make_docx_bytes()
        result = process_file(docx_data, "report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "id-docx")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert result.outputs[0] == docx_data
        assert result.num_outputs == 1

    def test_pptx_detection_and_processing(self):
        """PPTX files should be detected and processed correctly."""
        pptx_data = _make_pptx_bytes()
        result = process_file(pptx_data, "slides.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "id-pptx")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        assert result.outputs[0] == pptx_data

    def test_xlsx_detection_and_processing(self):
        """XLSX files should be detected and processed correctly."""
        xlsx_data = _make_xlsx_bytes()
        result = process_file(xlsx_data, "data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "id-xlsx")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert result.outputs[0] == xlsx_data

    def test_odt_detection_and_processing(self):
        """ODT files should be detected via mimetype file in ZIP."""
        odt_data = _make_odt_bytes()
        result = process_file(odt_data, "document.odt", "application/vnd.oasis.opendocument.text", "id-odt")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/vnd.oasis.opendocument.text"

    def test_zip_magic_bytes_detection(self):
        """ZIP-based files should be detected by PK magic bytes."""
        from ideation_agent.file_registry import _detect_mime_type
        docx_data = _make_docx_bytes()
        assert _detect_mime_type(docx_data) == "_zip_based"

    def test_docx_detected_by_magic_bytes_not_declared_mime(self):
        """DOCX should be correctly identified even with wrong declared MIME."""
        docx_data = _make_docx_bytes()
        result = process_file(docx_data, "report.docx", "application/octet-stream", "id")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def test_corrupt_zip_rejected(self):
        """Invalid ZIP data with PK header should be handled gracefully."""
        corrupt = b"PK\x03\x04" + b"\x00" * 100  # Invalid ZIP
        result = process_file(corrupt, "bad.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "id")
        # Should fail gracefully (either unsupported or invalid type error)
        assert isinstance(result, ProcessingError)

    def test_office_types_in_supported_mime_types(self):
        """All Office MIME types should be in SUPPORTED_MIME_TYPES."""
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in SUPPORTED_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.presentationml.presentation" in SUPPORTED_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in SUPPORTED_MIME_TYPES
        assert "application/msword" in SUPPORTED_MIME_TYPES
        assert "application/vnd.ms-powerpoint" in SUPPORTED_MIME_TYPES
        assert "application/vnd.ms-excel" in SUPPORTED_MIME_TYPES


# ---------------------------------------------------------------------------
# RTF processor tests
# ---------------------------------------------------------------------------

class TestRtfProcessor:
    """Test process_rtf_file functionality."""

    def test_valid_rtf(self):
        """Valid RTF should be processed correctly."""
        rtf_data = b"{\\rtf1\\ansi Hello World}"
        result = process_file(rtf_data, "note.rtf", "application/rtf", "id-rtf")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/rtf"
        assert result.outputs[0] == rtf_data

    def test_rtf_detected_by_magic_bytes(self):
        """RTF should be detected via magic bytes."""
        from ideation_agent.file_registry import _detect_mime_type
        rtf_data = b"{\\rtf1\\ansi Test content}"
        assert _detect_mime_type(rtf_data) == "application/rtf"

    def test_invalid_rtf_rejected(self):
        """Non-RTF data should be rejected by the RTF processor."""
        from ideation_agent.file_registry import process_rtf_file
        result = process_rtf_file(b"NOT RTF CONTENT", "fake.rtf", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"


# ---------------------------------------------------------------------------
# Text/Code file processor tests
# ---------------------------------------------------------------------------

class TestTextProcessor:
    """Test process_text_file functionality."""

    def test_plain_text(self):
        """Plain text files should be processed correctly."""
        text_data = b"Hello, this is a plain text file.\nLine 2."
        result = process_file(text_data, "readme.txt", "text/plain", "id-txt")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "text/plain"
        assert result.outputs[0] == text_data
        assert result.metadata == {"format": "txt"}

    def test_markdown(self):
        """Markdown files should be processed correctly."""
        md_data = b"# Title\n\nSome **bold** text."
        result = process_file(md_data, "docs.md", "text/markdown", "id-md")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "text/markdown"

    def test_json(self):
        """JSON files should be processed correctly."""
        json_data = b'{"key": "value", "number": 42}'
        result = process_file(json_data, "config.json", "application/json", "id-json")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/json"

    def test_python_code(self):
        """Python source files should be processed correctly."""
        py_data = b'def hello():\n    print("Hello, world!")\n'
        result = process_file(py_data, "script.py", "text/x-python", "id-py")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "text/x-python"

    def test_csv(self):
        """CSV files should be processed correctly."""
        csv_data = b"name,age,city\nAlice,30,Berlin\nBob,25,Munich\n"
        result = process_file(csv_data, "data.csv", "text/csv", "id-csv")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "text/csv"

    def test_empty_text_rejected(self):
        """Empty text files should be rejected."""
        from ideation_agent.file_registry import process_text_file
        result = process_text_file(b"", "empty.txt", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_binary_content_rejected(self):
        """Binary content masquerading as text should be rejected."""
        from ideation_agent.file_registry import process_text_file
        binary_data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08"
        result = process_text_file(binary_data, "data.txt", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_extension_based_detection(self):
        """Text files should be detected by extension when no magic bytes match."""
        sql_data = b"SELECT * FROM users WHERE active = 1;"
        result = process_file(sql_data, "query.sql", "application/sql", "id-sql")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "application/sql"

    def test_text_types_in_supported_mime_types(self):
        """All text/code MIME types should be registered."""
        assert "text/plain" in SUPPORTED_MIME_TYPES
        assert "text/markdown" in SUPPORTED_MIME_TYPES
        assert "application/json" in SUPPORTED_MIME_TYPES
        assert "text/csv" in SUPPORTED_MIME_TYPES
        assert "text/tab-separated-values" in SUPPORTED_MIME_TYPES


# ---------------------------------------------------------------------------
# Email file processor tests
# ---------------------------------------------------------------------------

class TestEmailProcessor:
    """Test process_email_file functionality."""

    def test_valid_eml(self):
        """Valid .eml file should be processed correctly."""
        eml_data = (
            b"From: sender@example.com\r\n"
            b"To: recipient@example.com\r\n"
            b"Subject: Test Email\r\n"
            b"Date: Thu, 7 Aug 2026 10:00:00 +0000\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"This is the email body.\r\n"
        )
        result = process_file(eml_data, "message.eml", "message/rfc822", "id-eml")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "message/rfc822"
        assert result.outputs[0] == eml_data

    def test_mhtml_web_archive(self):
        """MHTML web archive should be processed correctly."""
        mhtml_data = (
            b"From: <Saved by Browser>\r\n"
            b"Subject: Saved Page\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: multipart/related;\r\n"
            b"\tboundary=\"----=_NextPart_000\"\r\n"
            b"\r\n"
            b"------=_NextPart_000\r\n"
            b"Content-Type: text/html\r\n"
            b"\r\n"
            b"<html><body>Saved page</body></html>\r\n"
        )
        result = process_file(mhtml_data, "page.mhtml", "message/rfc822", "id-mht")
        assert isinstance(result, ProcessedFile)
        assert result.original_type == "message/rfc822"

    def test_invalid_email_rejected(self):
        """Non-email data should be rejected."""
        from ideation_agent.file_registry import process_email_file
        result = process_email_file(b"This is just random text, no headers.", "fake.eml", "id")
        assert isinstance(result, ProcessingError)
        assert result.code == "invalid_type"

    def test_empty_email_rejected(self):
        """Empty files should be rejected."""
        from ideation_agent.file_registry import process_email_file
        result = process_email_file(b"", "empty.eml", "id")
        assert isinstance(result, ProcessingError)

    def test_email_type_in_supported(self):
        """message/rfc822 should be registered."""
        assert "message/rfc822" in SUPPORTED_MIME_TYPES


# ---------------------------------------------------------------------------
# DOCUMENT_MIME_TYPES set tests
# ---------------------------------------------------------------------------

class TestDocumentMimeTypes:
    """Test the DOCUMENT_MIME_TYPES constant used for LLM resolution."""

    def test_pdf_in_document_types(self):
        from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
        assert "application/pdf" in DOCUMENT_MIME_TYPES

    def test_office_in_document_types(self):
        from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in DOCUMENT_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.presentationml.presentation" in DOCUMENT_MIME_TYPES
        assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in DOCUMENT_MIME_TYPES

    def test_text_in_document_types(self):
        from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
        assert "text/plain" in DOCUMENT_MIME_TYPES
        assert "text/csv" in DOCUMENT_MIME_TYPES
        assert "application/json" in DOCUMENT_MIME_TYPES

    def test_email_in_document_types(self):
        from ideation_agent.file_registry import DOCUMENT_MIME_TYPES
        assert "message/rfc822" in DOCUMENT_MIME_TYPES

