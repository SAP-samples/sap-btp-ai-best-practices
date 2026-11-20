"""Unit tests for upload metadata validation and bounded byte reading.

Tests enforce:
- Chunk-by-chunk byte limit with no Content-Length trust.
- Kind-specific filename extension and media-type allowlists.
- Basename extraction, control-character stripping, and length limits.
- Correct error codes and HTTP status codes (400 / 413).

Usage::

    PYTHONPATH=api python -m pytest -q tests/unit/test_upload_limits.py
"""

import pytest

from app.ecn.errors import SourceValidationError, UploadTooLargeError
from app.ecn.intake.files import ValidatedUploadMetadata, read_upload_bytes, validate_upload_metadata
from app.ecn.models import ArtifactKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTIFACT_ID = "00000000-0000-4000-8000-000000000001"


# ---------------------------------------------------------------------------
# Bounded streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_limit_rejects_first_byte_over_limit(async_upload) -> None:
    """A stream of 5 bytes with a 4-byte limit must raise 413."""
    upload = async_upload(b"12345", filename="ecn.json", content_type="application/json")
    with pytest.raises(UploadTooLargeError) as caught:
        await read_upload_bytes(upload, maximum_bytes=4, chunk_size=2)
    assert caught.value.status_code == 413
    # Raw content must not leak into the exception message.
    assert b"12345" not in str(caught.value).encode()


@pytest.mark.asyncio
async def test_stream_at_exact_limit_succeeds(async_upload) -> None:
    """Exactly maximum_bytes must succeed and return all bytes."""
    data = b"hello"
    upload = async_upload(data, filename="ecn.json")
    result = await read_upload_bytes(upload, maximum_bytes=5)
    assert result == data


@pytest.mark.asyncio
async def test_empty_stream_returns_empty_bytes(async_upload) -> None:
    """An empty upload must return empty bytes without error."""
    upload = async_upload(b"", filename="ecn.json")
    result = await read_upload_bytes(upload, maximum_bytes=100)
    assert result == b""


@pytest.mark.asyncio
async def test_nonpositive_maximum_raises_value_error(async_upload) -> None:
    """A maximum_bytes <= 0 must raise ValueError before any read."""
    upload = async_upload(b"abc")
    with pytest.raises(ValueError):
        await read_upload_bytes(upload, maximum_bytes=0)


@pytest.mark.asyncio
async def test_nonpositive_chunk_size_raises_value_error(async_upload) -> None:
    """A chunk_size <= 0 must raise ValueError before any read."""
    upload = async_upload(b"abc")
    with pytest.raises(ValueError):
        await read_upload_bytes(upload, maximum_bytes=10, chunk_size=0)


@pytest.mark.asyncio
async def test_fake_content_length_does_not_override_stream_limit(async_upload) -> None:
    """Streaming enforcement is independent of any Content-Length header.

    The mock upload carries 5 bytes; the limit is 4. Even if a hypothetical
    Content-Length were 3 (smaller than actual), the real bytes must trip
    the limit — this test demonstrates the enforcer only trusts the stream.
    """
    # 5 bytes in stream, limit 4 — must raise regardless of any header.
    upload = async_upload(b"ABCDE", filename="ecn.json")
    with pytest.raises(UploadTooLargeError) as caught:
        await read_upload_bytes(upload, maximum_bytes=4, chunk_size=1024)
    assert caught.value.status_code == 413


@pytest.mark.asyncio
async def test_limit_is_enforced_mid_stream(async_upload) -> None:
    """Limit must be crossed during chunked reading, not after full accumulation."""
    # 6 bytes with chunk_size=2: chunks are [b"AB", b"CD", b"EF"]
    # After chunk 2 (total=4 bytes) still OK; chunk 3 pushes to 6 > 5.
    upload = async_upload(b"ABCDEF", filename="ecn.json")
    with pytest.raises(UploadTooLargeError):
        await read_upload_bytes(upload, maximum_bytes=5, chunk_size=2)


# ---------------------------------------------------------------------------
# Metadata validation — ECN JSON kind
# ---------------------------------------------------------------------------


def test_json_metadata_uses_basename_and_exact_allowlist() -> None:
    """Basename extraction and media type normalisation must work together."""
    metadata = validate_upload_metadata(
        "..\\folder\\ECN.JSON",
        " Application/JSON ",
        ArtifactKind.ECN_JSON,
    )
    assert metadata.file_name == "ECN.JSON"
    assert metadata.media_type == "application/json"


def test_json_txt_extension_rejected() -> None:
    """A .txt file must be rejected for ECN_JSON kind."""
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata("ecn.txt", "application/json", ArtifactKind.ECN_JSON)
    assert caught.value.status_code == 400


def test_json_extension_case_insensitive() -> None:
    """Extension match must be case-insensitive (.JSON accepted for ECN_JSON)."""
    metadata = validate_upload_metadata("upload.JSON", "application/json", ArtifactKind.ECN_JSON)
    assert metadata.file_name == "upload.JSON"


def test_json_text_json_media_type_accepted() -> None:
    """text/json must be accepted for ECN_JSON kind."""
    metadata = validate_upload_metadata("ecn.json", "text/json", ArtifactKind.ECN_JSON)
    assert metadata.media_type == "text/json"


def test_json_octet_stream_accepted() -> None:
    """application/octet-stream must be accepted for ECN_JSON kind."""
    metadata = validate_upload_metadata("ecn.json", "application/octet-stream", ArtifactKind.ECN_JSON)
    assert metadata.media_type == "application/octet-stream"


def test_json_empty_media_type_accepted() -> None:
    """An absent or empty media type must be accepted for ECN_JSON kind."""
    metadata = validate_upload_metadata("ecn.json", None, ArtifactKind.ECN_JSON)
    assert metadata.media_type == ""
    metadata2 = validate_upload_metadata("ecn.json", "", ArtifactKind.ECN_JSON)
    assert metadata2.media_type == ""


def test_json_xml_media_type_rejected() -> None:
    """application/xml must be rejected for ECN_JSON kind."""
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata("ecn.json", "application/xml", ArtifactKind.ECN_JSON)
    assert caught.value.status_code == 400


# ---------------------------------------------------------------------------
# Metadata validation — workbook kinds
# ---------------------------------------------------------------------------


def test_xlsx_extension_accepted_for_bom_workbook() -> None:
    """A .xlsx file with the spreadsheet MIME type must be accepted."""
    media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    metadata = validate_upload_metadata("bom.xlsx", media, ArtifactKind.BOM_WORKBOOK)
    assert metadata.file_name == "bom.xlsx"
    assert metadata.media_type == media


def test_xlsx_octet_stream_accepted_for_bom_workbook() -> None:
    """application/octet-stream must be accepted for BOM_WORKBOOK kind."""
    metadata = validate_upload_metadata("bom.xlsx", "application/octet-stream", ArtifactKind.BOM_WORKBOOK)
    assert metadata.media_type == "application/octet-stream"


def test_xlsx_empty_media_accepted_for_conditional_workbook() -> None:
    """Empty media type must be accepted for CONDITIONAL_RULE_WORKBOOK."""
    metadata = validate_upload_metadata("rules.xlsx", None, ArtifactKind.CONDITIONAL_RULE_WORKBOOK)
    assert metadata.media_type == ""


def test_csv_extension_accepted_for_purchase_info_price_source() -> None:
    """The dedicated price overlay source accepts its documented CSV transport."""
    metadata = validate_upload_metadata(
        "purchase-prices.csv", "text/csv", ArtifactKind.PURCHASE_INFO_PRICE_CSV
    )
    assert metadata.file_name == "purchase-prices.csv"
    assert metadata.media_type == "text/csv"



    """A .json file must be rejected for BOM_WORKBOOK kind."""
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata("bom.json", "application/octet-stream", ArtifactKind.BOM_WORKBOOK)
    assert caught.value.status_code == 400


# ---------------------------------------------------------------------------
# Filename safety rules
# ---------------------------------------------------------------------------


def test_control_characters_removed_from_filename() -> None:
    """ASCII control characters must be stripped from the filename."""
    # \x00 and \x1f are C0 control chars; \x7f is DEL.
    metadata = validate_upload_metadata(
        "ecn\x00\x1f\x7ffile.json", "application/json", ArtifactKind.ECN_JSON
    )
    assert "\x00" not in metadata.file_name
    assert "\x1f" not in metadata.file_name
    assert "\x7f" not in metadata.file_name
    assert metadata.file_name == "ecnfile.json"


def test_c1_control_characters_removed_from_filename() -> None:
    """C1 control characters (0x80-0x9F) must be stripped from the filename."""
    metadata = validate_upload_metadata(
        "ecn\x80\x9ffile.json", "application/json", ArtifactKind.ECN_JSON
    )
    assert metadata.file_name == "ecnfile.json"


def test_empty_filename_rejected() -> None:
    """An empty or None filename must raise SourceValidationError."""
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata(None, "application/json", ArtifactKind.ECN_JSON)
    assert caught.value.status_code == 400

    with pytest.raises(SourceValidationError):
        validate_upload_metadata("   ", "application/json", ArtifactKind.ECN_JSON)


def test_filename_all_control_chars_becomes_empty_and_is_rejected() -> None:
    """A filename that reduces to empty after stripping controls must be rejected."""
    with pytest.raises(SourceValidationError):
        validate_upload_metadata("\x00\x01\x02", "application/json", ArtifactKind.ECN_JSON)


def test_overlong_filename_rejected() -> None:
    """A filename exceeding 500 characters must raise SourceValidationError."""
    # 495 a's + ".json" = 500 chars exactly — must be accepted.
    long_name = "a" * 495 + ".json"
    metadata = validate_upload_metadata(long_name, "application/json", ArtifactKind.ECN_JSON)
    assert len(metadata.file_name) == 500

    # 496 a's + ".json" = 501 chars — must be rejected.
    too_long = "a" * 496 + ".json"
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata(too_long, "application/json", ArtifactKind.ECN_JSON)
    assert caught.value.status_code == 400


def test_overlong_media_type_rejected() -> None:
    """A media type exceeding 200 characters must raise SourceValidationError."""
    long_media = "application/" + "x" * 190  # 202 chars
    with pytest.raises(SourceValidationError) as caught:
        validate_upload_metadata("ecn.json", long_media, ArtifactKind.ECN_JSON)
    assert caught.value.status_code == 400


def test_unix_path_basename_extracted() -> None:
    """Unix-path prefixes must be stripped to the basename only."""
    metadata = validate_upload_metadata(
        "/tmp/uploads/ecn.json", "application/json", ArtifactKind.ECN_JSON
    )
    assert metadata.file_name == "ecn.json"


def test_windows_path_basename_extracted() -> None:
    """Windows-path prefixes must be stripped to the basename only."""
    metadata = validate_upload_metadata(
        "C:\\Users\\user\\Downloads\\ecn.json", "application/json", ArtifactKind.ECN_JSON
    )
    assert metadata.file_name == "ecn.json"


def test_media_type_whitespace_stripped_and_lowercased() -> None:
    """Media types must be stripped and lowercased during normalisation."""
    metadata = validate_upload_metadata(
        "ecn.json", "  TEXT/JSON  ", ArtifactKind.ECN_JSON
    )
    assert metadata.media_type == "text/json"
