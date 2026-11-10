"""Integration tests for the file upload endpoint.

Tests the POST /api/ideation-agent/upload endpoint including auth,
file validation, and response format with Object Store.
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.auth import UserIdentity, get_current_user


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TEST_USER = UserIdentity(
    user_id="test-xsuaa-user",
    email="test.user@sap.com",
    given_name="Test",
    family_name="User",
    user_uuid="abc-123",
    auth_provider="xsuaa",
)


def _make_png(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal PNG image in memory."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_object_store():
    """Mock the Object Store so we don't need real credentials."""
    mock_store = AsyncMock()
    mock_store.upload = AsyncMock(return_value=None)
    mock_store.download = AsyncMock(return_value=b"\x89PNG fake data")
    mock_store.exists = AsyncMock(return_value=True)

    with patch("app.routers.ideation_upload.get_object_store", return_value=mock_store):
        yield mock_store


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def auth_override():
    """Override get_current_user for all tests."""
    async def _override():
        return _TEST_USER

    app.dependency_overrides[get_current_user] = _override
    yield
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

class TestUploadSuccess:
    """Test successful file uploads."""

    def test_upload_valid_png(self, client, mock_object_store):
        png_data = _make_png()
        response = client.post(
            "/api/ideation-agent/upload?context_id=test-ctx-123",
            files={"file": ("test.png", io.BytesIO(png_data), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test.png"
        assert data["original_type"] == "image/png"
        assert data["num_outputs"] == 1
        # Should NOT include content_block (no base64 to client)
        assert "content_block" not in data
        # Thumbnail URL removed — UI uses icons now
        assert "thumbnail_url" not in data

    def test_upload_valid_jpeg(self, client, mock_object_store):
        jpeg_data = _make_jpeg()
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-456",
            files={"file": ("photo.jpg", io.BytesIO(jpeg_data), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "photo.jpg"
        assert data["original_type"] == "image/jpeg"
        assert data["num_outputs"] == 1

    def test_upload_returns_dimensions(self, client, mock_object_store):
        png_data = _make_png(200, 150)
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-dim",
            files={"file": ("sized.png", io.BytesIO(png_data), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["dimensions"]["width"] == 200
        assert data["dimensions"]["height"] == 150

    def test_upload_stores_in_object_store(self, client, mock_object_store):
        """Verify Object Store receives original + output + meta.json."""
        png_data = _make_png()
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-store",
            files={"file": ("test.png", io.BytesIO(png_data), "image/png")},
        )
        assert response.status_code == 200
        file_id = response.json()["file_id"]

        # Object Store should receive 3 uploads: original, output 0, meta.json
        assert mock_object_store.upload.call_count == 3

        # Verify keys contain context_id and file_id
        call_keys = [call.args[0] for call in mock_object_store.upload.call_args_list]
        assert any(f"ctx-store/{file_id}/original" in k for k in call_keys)
        assert any(f"ctx-store/{file_id}/0.png" in k for k in call_keys)
        assert any(f"ctx-store/{file_id}/meta.json" in k for k in call_keys)

    def test_context_id_required(self, client, mock_object_store):
        """Upload without context_id should fail with 422."""
        png_data = _make_png()
        response = client.post(
            "/api/ideation-agent/upload",
            files={"file": ("test.png", io.BytesIO(png_data), "image/png")},
        )
        assert response.status_code == 422  # Missing required query param


# ---------------------------------------------------------------------------
# Auth test
# ---------------------------------------------------------------------------

class TestUploadAuth:
    """Test that unauthenticated requests are rejected."""

    def test_no_auth_returns_401(self, client):
        """Without auth override, endpoint should require a token."""
        app.dependency_overrides.pop(get_current_user, None)
        png_data = _make_png()
        response = client.post(
            "/api/ideation-agent/upload?context_id=test-ctx",
            files={"file": ("test.png", io.BytesIO(png_data), "image/png")},
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Validation error cases
# ---------------------------------------------------------------------------

class TestUploadValidation:
    """Test file validation error handling."""

    def test_invalid_file_type(self, client, mock_object_store):
        """Non-image file should be rejected with 400."""
        txt_data = b"This is not an image file\n" * 10
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("readme.txt", io.BytesIO(txt_data), "text/plain")},
        )
        assert response.status_code == 400

    def test_exe_renamed_to_png(self, client, mock_object_store):
        """EXE file renamed to .png should be rejected (magic bytes check)."""
        exe_data = b"MZ" + b"\x00" * 500
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("malware.png", io.BytesIO(exe_data), "image/png")},
        )
        assert response.status_code == 400

    def test_oversized_file(self, client, mock_object_store):
        """File exceeding 10 MB should be rejected with 413."""
        big_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (10 * 1024 * 1024 + 1)
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("big.png", io.BytesIO(big_data), "image/png")},
        )
        assert response.status_code == 413

    def test_corrupt_image(self, client, mock_object_store):
        """Corrupt PNG data should be rejected with 400."""
        corrupt_data = b"\x89PNG\r\n\x1a\n" + b"\xff" * 200
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("broken.png", io.BytesIO(corrupt_data), "image/png")},
        )
        assert response.status_code == 400
        assert "corrupt" in response.json()["detail"].lower()

    def test_svg_rejected(self, client, mock_object_store):
        """SVG files should be rejected for security."""
        svg_data = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("vector.svg", io.BytesIO(svg_data), "image/svg+xml")},
        )
        assert response.status_code == 400

    def test_empty_file(self, client, mock_object_store):
        """Empty file should be rejected with 400."""
        response = client.post(
            "/api/ideation-agent/upload?context_id=ctx-val",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# File retrieval endpoints
# ---------------------------------------------------------------------------

class TestFileRetrieval:
    """Test file serving endpoints (full, output)."""

    def test_get_full_image(self, client, mock_object_store):
        """GET /files/{file_id}/full should return full-res PNG."""
        with patch("app.routers.ideation_upload._verify_file_access", new_callable=AsyncMock):
            mock_object_store.download = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nfake full image")
            response = client.get(
                "/api/ideation-agent/files/test-file-id/full?context_id=ctx-123"
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_get_output_by_index(self, client, mock_object_store):
        """GET /files/{file_id}/output/0 should return specific output."""
        with patch("app.routers.ideation_upload._verify_file_access", new_callable=AsyncMock):
            mock_object_store.download = AsyncMock(return_value=b"\x89PNG output page 0")
            response = client.get(
                "/api/ideation-agent/files/test-file-id/output/0?context_id=ctx-123"
            )
            assert response.status_code == 200

    def test_file_not_found(self, client, mock_object_store):
        """Missing file should return 404."""
        with patch("app.routers.ideation_upload._verify_file_access", new_callable=AsyncMock):
            mock_object_store.download = AsyncMock(return_value=None)
            response = client.get(
                "/api/ideation-agent/files/nonexistent/full?context_id=ctx-123"
            )
            assert response.status_code == 404
