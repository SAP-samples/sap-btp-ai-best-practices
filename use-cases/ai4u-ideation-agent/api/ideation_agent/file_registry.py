"""File processing registry — routes uploads by MIME type.

Extensible architecture: adding new format support = writing a
processor function + registering it. No structural changes needed.

Each processor returns a standardized ProcessedFile with:
- outputs: list of processed bytes (PNG for images, raw bytes for documents)

Supported format categories:
- Images: PNG, JPEG, GIF, WebP
- PDF: Native vision processing (text + images + diagrams)
- Office XML: DOCX, PPTX, XLSX (text extraction via LLM)
- Legacy Office: DOC, PPT, XLS (text extraction via LLM)
- OpenDocument: ODT, ODP, ODS (text extraction via LLM)
- RTF: Rich Text Format (text extraction via LLM)
- Data: CSV, TSV (spreadsheet-specific processing by LLM)
- Email: EML, MHT/MHTML (text extraction via LLM)
- Text/Code: TXT, MD, JSON, XML, YAML, PY, JS, HTML, SQL, etc.
"""
from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

CONTENT_MAX_DIMENSION = 2048  # px longest side for LLM input
MAX_IMAGE_PIXELS = 178_000_000  # Pillow decompression bomb guard (supports up to ~13000x13000)

# SVG detection patterns (reject for security — XSS vector)
_SVG_SIGNATURES = [b"<?xml", b"<svg", b"<!DOCTYPE svg"]

# Magic byte signatures for file format validation
_MAGIC_BYTES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP starts with RIFF....WEBP
    b"%PDF": "application/pdf",
    b"{\\rtf": "application/rtf",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "_ole_compound",  # Legacy Office (DOC/XLS/PPT)
    b"PK\x03\x04": "_zip_based",  # DOCX/PPTX/XLSX/ODT/ODP/ODS (needs further inspection)
}

# Extension → MIME type mapping for text-based files (no reliable magic bytes)
_TEXT_EXTENSION_MAP: Dict[str, str] = {
    ".txt": "text/plain",
    ".text": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".json": "application/json",
    ".xml": "text/xml",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/javascript",
    ".html": "text/html",
    ".htm": "text/html",
    ".css": "text/css",
    ".sql": "application/sql",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".eml": "message/rfc822",
    ".mht": "message/rfc822",
    ".mhtml": "message/rfc822",
    ".rtf": "application/rtf",
}

# MIME types that are "document" types (sent to LLM via Responses API input_file)
DOCUMENT_MIME_TYPES = {
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


# --- Data classes ------------------------------------------------------------

@dataclass
class ProcessedFile:
    """Standardized output from any file processor."""
    file_id: str
    filename: str
    original_type: str  # e.g. "image/png", "application/pdf"
    outputs: List[bytes]  # List of processed outputs (PNG for images, raw PDF for documents)
    num_outputs: int = field(init=False)
    dimensions: Optional[Dict[str, int]] = None  # width/height for images
    metadata: Optional[Dict[str, Any]] = None  # Extra info (e.g. page_count for PDFs)

    def __post_init__(self):
        self.num_outputs = len(self.outputs)


@dataclass
class ProcessingError:
    """Result of failed file processing."""
    error: str
    code: str  # "unsupported_type" | "invalid_type" | "corrupt" | "too_large" | "security"


# --- Type alias for processor functions --------------------------------------

ProcessorFn = Callable[[bytes, str], ProcessedFile | ProcessingError]


# --- Image processor ---------------------------------------------------------

def process_image_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process an image file into a full-res output for LLM analysis.

    Pipeline:
    1. Validate magic bytes (reject SVG for security)
    2. Open with Pillow (validates image integrity)
    3. Strip EXIF metadata (privacy — GPS, device info)
    4. Convert to RGB
    5. Generate content output (2048px max)
    6. Return ProcessedFile with PNG bytes

    Args:
        file_data: Raw image bytes.
        filename: Original filename for error messages.
        file_id: UUID assigned by the upload endpoint.
    """
    # 1. Validate magic bytes
    detected_mime = _detect_mime_type(file_data)
    if detected_mime is None:
        return ProcessingError(
            error=f"Unsupported file type for '{filename}'. Accepted: PNG, JPEG, GIF, WEBP.",
            code="invalid_type",
        )

    if _is_svg(file_data):
        return ProcessingError(
            error="SVG files are not supported (security restriction).",
            code="security",
        )

    # 2. Open with Pillow
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(file_data))
        img.load()
    except Exception as exc:
        logger.warning("Failed to open image '%s': %s", filename, exc)
        return ProcessingError(
            error=f"Cannot open image '{filename}': file appears corrupt or invalid.",
            code="corrupt",
        )

    original_format = img.format or "UNKNOWN"

    # 3. Apply EXIF rotation (before any processing)
    img = ImageOps.exif_transpose(img)

    # 4. Convert to RGB (alpha compositing needs full-res for correctness)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img, mask=img.split()[1])
        img = background
    elif img.mode not in ("RGB",):
        img = img.convert("RGB")

    # 5. Resize to content max FIRST (avoids expensive ops on huge images)
    width, height = img.size
    if max(width, height) > CONTENT_MAX_DIMENSION:
        ratio = CONTENT_MAX_DIMENSION / max(width, height)
        img = img.resize(
            (int(width * ratio), int(height * ratio)), Image.LANCZOS
        )

    # 6. Save as PNG (metadata is stripped implicitly — fresh encode)
    content_width, content_height = img.size
    content_buffer = io.BytesIO()
    img.save(content_buffer, format="PNG", optimize=True)
    content_bytes = content_buffer.getvalue()

    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type=detected_mime or f"image/{original_format.lower()}",
        outputs=[content_bytes],
        dimensions={"width": content_width, "height": content_height},
    )


# --- PDF processor -----------------------------------------------------------

# PDF validation — minimal structural check (no heavy dependencies)
_PDF_HEADER = b"%PDF"
_PDF_EOF_MARKER = b"%%EOF"


def process_pdf_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process a PDF file for native LLM input.

    Unlike images, PDFs are NOT rendered to PNGs. The raw PDF bytes are stored
    and sent directly to the LLM via the Responses API `input_file` block,
    which provides full native understanding of text, images, diagrams, and
    scanned content.

    Pipeline:
    1. Validate magic bytes (%PDF header)
    2. Basic structural validation (check for %%EOF marker)
    3. Return ProcessedFile with raw PDF bytes as the single output

    Args:
        file_data: Raw PDF bytes.
        filename: Original filename for error messages.
        file_id: UUID assigned by the upload endpoint.
    """
    # 1. Validate magic bytes
    if not file_data.startswith(_PDF_HEADER):
        return ProcessingError(
            error=f"File '{filename}' is not a valid PDF (invalid header).",
            code="invalid_type",
        )

    # 2. Basic structural validation
    # Check for %%EOF marker in the last 1024 bytes (allows for trailing whitespace)
    tail = file_data[-1024:]
    if _PDF_EOF_MARKER not in tail:
        logger.warning("PDF '%s' missing %%%%EOF marker — may be truncated", filename)
        # We still accept it — many PDFs work fine without a strict %%EOF check,
        # and the LLM will handle what it can. Only reject truly empty/corrupt PDFs.

    # 3. Check for encrypted/password-protected PDFs
    # Quick heuristic: look for /Encrypt dictionary in the first 4KB
    header_region = file_data[:4096]
    if b"/Encrypt" in header_region:
        return ProcessingError(
            error=f"Cannot process '{filename}': PDF is encrypted/password-protected. "
                  f"Please upload an unlocked version.",
            code="corrupt",
        )

    # 4. Return raw PDF bytes as the output (LLM receives natively)
    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type="application/pdf",
        outputs=[file_data],
        dimensions=None,
        metadata={"format": "pdf"},
    )


# --- Office document processor ------------------------------------------------

def process_office_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process an Office/OpenDocument file for native LLM input.

    Handles ZIP-based formats (DOCX, PPTX, XLSX, ODT, ODP, ODS) and
    legacy OLE formats (DOC, PPT, XLS). Raw bytes are stored and sent
    directly to the LLM via the Responses API `input_file` block.

    Pipeline:
    1. Validate file structure (ZIP or OLE magic bytes)
    2. Detect specific format from ZIP contents or extension
    3. Return ProcessedFile with raw bytes

    Args:
        file_data: Raw document bytes.
        filename: Original filename for error messages.
        file_id: UUID assigned by the upload endpoint.
    """
    detected_mime = _detect_document_type(file_data, filename)
    if detected_mime is None:
        return ProcessingError(
            error=f"Cannot identify document type for '{filename}'. File may be corrupt.",
            code="invalid_type",
        )

    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type=detected_mime,
        outputs=[file_data],
        dimensions=None,
        metadata={"format": _extension_from_filename(filename)},
    )


# --- RTF processor -----------------------------------------------------------

def process_rtf_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process an RTF file for native LLM input.

    Pipeline:
    1. Validate RTF header ({\\rtf)
    2. Return ProcessedFile with raw bytes
    """
    if not file_data.startswith(b"{\\rtf"):
        return ProcessingError(
            error=f"File '{filename}' is not a valid RTF document.",
            code="invalid_type",
        )

    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type="application/rtf",
        outputs=[file_data],
        dimensions=None,
        metadata={"format": "rtf"},
    )


# --- Text/Code file processor ------------------------------------------------

def process_text_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process a text/code file for native LLM input.

    Handles plain text, markdown, JSON, XML, YAML, Python, JavaScript,
    HTML, CSS, SQL, CSV, TSV, and other text-based formats.

    Pipeline:
    1. Validate encoding (must be decodable as UTF-8 or Latin-1)
    2. Reject binary content
    3. Return ProcessedFile with raw bytes

    Args:
        file_data: Raw file bytes.
        filename: Original filename for error messages.
        file_id: UUID assigned by the upload endpoint.
    """
    if len(file_data) == 0:
        return ProcessingError(
            error=f"File '{filename}' is empty.",
            code="invalid_type",
        )

    # Validate this is actually text (not binary masquerading as text)
    try:
        file_data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            file_data.decode("latin-1")
        except UnicodeDecodeError:
            return ProcessingError(
                error=f"File '{filename}' does not appear to be a text file (binary content detected).",
                code="invalid_type",
            )

    # Check for null bytes (strong indicator of binary content)
    if b"\x00" in file_data[:8192]:
        return ProcessingError(
            error=f"File '{filename}' contains binary content. Please upload a text-based file.",
            code="invalid_type",
        )

    # Determine MIME type from extension
    ext = _extension_from_filename(filename)
    mime_type = _TEXT_EXTENSION_MAP.get(f".{ext}", "text/plain")

    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type=mime_type,
        outputs=[file_data],
        dimensions=None,
        metadata={"format": ext or "txt"},
    )


# --- Email file processor ----------------------------------------------------

def process_email_file(file_data: bytes, filename: str, file_id: str = "") -> ProcessedFile | ProcessingError:
    """Process an email (.eml) or web archive (.mht/.mhtml) file for LLM input.

    Pipeline:
    1. Basic RFC822 header validation
    2. Return ProcessedFile with raw bytes
    """
    if len(file_data) == 0:
        return ProcessingError(
            error=f"File '{filename}' is empty.",
            code="invalid_type",
        )

    # Basic validation: look for RFC822-like headers in the first 2KB
    header = file_data[:2048]
    try:
        header_text = header.decode("utf-8", errors="replace")
    except Exception:
        header_text = ""

    # Check for common email/MIME headers
    has_headers = any(
        marker in header_text
        for marker in ("From:", "Subject:", "Date:", "MIME-Version:", "Content-Type:")
    )

    if not has_headers:
        return ProcessingError(
            error=f"File '{filename}' does not appear to be a valid email or web archive file.",
            code="invalid_type",
        )

    return ProcessedFile(
        file_id=file_id,
        filename=filename,
        original_type="message/rfc822",
        outputs=[file_data],
        dimensions=None,
        metadata={"format": _extension_from_filename(filename) or "eml"},
    )


# --- Processor registry ------------------------------------------------------

PROCESSORS: Dict[str, ProcessorFn] = {
    # Images
    "image/png": process_image_file,
    "image/jpeg": process_image_file,
    "image/gif": process_image_file,
    "image/webp": process_image_file,
    # PDF
    "application/pdf": process_pdf_file,
    # Office XML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": process_office_file,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": process_office_file,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": process_office_file,
    # Legacy Office
    "application/msword": process_office_file,
    "application/vnd.ms-powerpoint": process_office_file,
    "application/vnd.ms-excel": process_office_file,
    # OpenDocument
    "application/vnd.oasis.opendocument.text": process_office_file,
    "application/vnd.oasis.opendocument.presentation": process_office_file,
    "application/vnd.oasis.opendocument.spreadsheet": process_office_file,
    # RTF
    "application/rtf": process_rtf_file,
    # Data
    "text/csv": process_text_file,
    "text/tab-separated-values": process_text_file,
    # Email / Web Archive
    "message/rfc822": process_email_file,
    # Text / Code
    "text/plain": process_text_file,
    "text/markdown": process_text_file,
    "application/json": process_text_file,
    "text/xml": process_text_file,
    "text/yaml": process_text_file,
    "text/x-python": process_text_file,
    "text/javascript": process_text_file,
    "text/html": process_text_file,
    "text/css": process_text_file,
    "application/sql": process_text_file,
}

SUPPORTED_MIME_TYPES = set(PROCESSORS.keys())


def process_file(
    file_data: bytes,
    filename: str,
    content_type: str,
    file_id: str = "",
) -> ProcessedFile | ProcessingError:
    """Route file processing by MIME type.

    Detection strategy:
    1. Magic bytes for binary formats (images, PDF, RTF, Office)
    2. ZIP inspection for Office XML / OpenDocument
    3. Extension-based for text/code formats
    4. Declared MIME as fallback

    Args:
        file_data: Raw file bytes.
        filename: Original filename.
        content_type: Declared MIME type (used for routing; magic bytes validated inside processor).
        file_id: UUID assigned by the upload endpoint.
    """
    # Normalize content type
    mime = (content_type or "").lower().split(";")[0].strip()

    # Try magic-byte detection first (most reliable)
    detected = _detect_mime_type(file_data)

    if detected == "_zip_based":
        # ZIP-based format — inspect contents to determine specific type
        detected = _detect_zip_format(file_data, filename)
    elif detected == "_ole_compound":
        # Legacy Office — use extension to determine specific type
        detected = _detect_ole_format(filename)

    if detected and detected in PROCESSORS:
        mime = detected
    elif mime not in PROCESSORS:
        # Try extension-based detection for text/code files
        ext_mime = _detect_from_extension(filename)
        if ext_mime and ext_mime in PROCESSORS:
            mime = ext_mime
        else:
            return ProcessingError(
                error=f"Unsupported file type: {content_type or 'unknown'}. "
                      f"Supported formats: images, PDF, Office documents, text/code files.",
                code="unsupported_type",
            )

    processor = PROCESSORS[mime]
    return processor(file_data, filename, file_id)


# --- Private helpers ---------------------------------------------------------

def _detect_mime_type(data: bytes) -> Optional[str]:
    """Detect file MIME type from magic bytes.

    Returns actual MIME type for direct-match formats, or sentinel strings
    '_zip_based' / '_ole_compound' for formats that need further inspection.
    """
    for magic, mime in _MAGIC_BYTES.items():
        if magic == b"RIFF":
            if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                return mime
        elif data.startswith(magic):
            return mime
    return None


def _detect_zip_format(data: bytes, filename: str) -> Optional[str]:
    """Inspect a ZIP file to determine if it's DOCX, PPTX, XLSX, or OpenDocument.

    Office XML formats contain [Content_Types].xml with distinguishing strings.
    OpenDocument formats contain a 'mimetype' file at the archive root.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            names = zf.namelist()

            # OpenDocument: 'mimetype' file at root
            if "mimetype" in names:
                try:
                    mimetype_content = zf.read("mimetype").decode("utf-8", errors="replace").strip()
                    if "opendocument.text" in mimetype_content:
                        return "application/vnd.oasis.opendocument.text"
                    elif "opendocument.presentation" in mimetype_content:
                        return "application/vnd.oasis.opendocument.presentation"
                    elif "opendocument.spreadsheet" in mimetype_content:
                        return "application/vnd.oasis.opendocument.spreadsheet"
                except Exception:
                    pass

            # Office XML: [Content_Types].xml
            if "[Content_Types].xml" in names:
                try:
                    ct_content = zf.read("[Content_Types].xml").decode("utf-8", errors="replace")
                    if "wordprocessingml" in ct_content:
                        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif "presentationml" in ct_content:
                        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    elif "spreadsheetml" in ct_content:
                        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                except Exception:
                    pass

    except (zipfile.BadZipFile, Exception) as e:
        logger.debug("ZIP inspection failed for '%s': %s", filename, e)
        return None

    # Fallback: use extension if ZIP structure is valid but type unclear
    return _detect_office_from_extension(filename)


def _detect_ole_format(filename: str) -> Optional[str]:
    """Determine legacy Office type from file extension (OLE compound files).

    Legacy .doc/.xls/.ppt all share the same magic bytes, so we use the extension.
    """
    ext = _extension_from_filename(filename)
    ole_ext_map = {
        "doc": "application/msword",
        "xls": "application/vnd.ms-excel",
        "ppt": "application/vnd.ms-powerpoint",
    }
    return ole_ext_map.get(ext)


def _detect_office_from_extension(filename: str) -> Optional[str]:
    """Fallback: determine Office type from extension when ZIP inspection is ambiguous."""
    ext = _extension_from_filename(filename)
    ext_map = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "odt": "application/vnd.oasis.opendocument.text",
        "odp": "application/vnd.oasis.opendocument.presentation",
        "ods": "application/vnd.oasis.opendocument.spreadsheet",
    }
    return ext_map.get(ext)


def _detect_from_extension(filename: str) -> Optional[str]:
    """Detect MIME type from file extension for text-based formats."""
    ext = _extension_from_filename(filename)
    if ext:
        return _TEXT_EXTENSION_MAP.get(f".{ext}")
    return None


def _detect_document_type(data: bytes, filename: str) -> Optional[str]:
    """Detect specific document type for office files (used by process_office_file)."""
    # ZIP-based?
    if data.startswith(b"PK\x03\x04"):
        return _detect_zip_format(data, filename)
    # OLE Compound?
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return _detect_ole_format(filename)
    return None


def _extension_from_filename(filename: str) -> str:
    """Extract lowercase extension from filename (without dot)."""
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return ""


def _is_svg(data: bytes) -> bool:
    """Check if file content looks like SVG."""
    header = data[:1024].lstrip()
    header_lower = header.lower()
    for sig in _SVG_SIGNATURES:
        if header_lower.startswith(sig):
            return True
    return False
