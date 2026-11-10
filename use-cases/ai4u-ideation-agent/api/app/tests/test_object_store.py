"""Unit tests for the Object Store service.

Tests credential resolution, key prefixing, and method behavior
with mocked aioboto3 clients.
"""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.object_store import (
    ObjectStore,
    _get_credentials,
    KEY_PREFIX,
    get_object_store,
)


# ---------------------------------------------------------------------------
# Credential resolution tests
# ---------------------------------------------------------------------------

class TestCredentialResolution:
    """Test _get_credentials() resolves from VCAP_SERVICES or env vars."""

    def test_env_vars_fallback(self, monkeypatch):
        """When no VCAP_SERVICES, should fall back to explicit env vars."""
        monkeypatch.delenv("VCAP_SERVICES", raising=False)
        monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("OBJECT_STORE_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("OBJECT_STORE_BUCKET", "test-bucket")
        monkeypatch.setenv("OBJECT_STORE_HOST", "s3.eu-central-1.amazonaws.com")
        monkeypatch.setenv("OBJECT_STORE_REGION", "eu-central-1")

        creds = _get_credentials()
        assert creds["access_key_id"] == "test-key"
        assert creds["secret_access_key"] == "test-secret"
        assert creds["bucket"] == "test-bucket"
        assert creds["host"] == "s3.eu-central-1.amazonaws.com"
        assert creds["region"] == "eu-central-1"

    def test_vcap_services_priority(self, monkeypatch):
        """VCAP_SERVICES should take priority over env vars."""
        vcap = {
            "objectstore": [{
                "credentials": {
                    "access_key_id": "vcap-key",
                    "secret_access_key": "vcap-secret",
                    "bucket": "vcap-bucket",
                    "host": "vcap-host.com",
                    "region": "us-east-1",
                }
            }]
        }
        monkeypatch.setenv("VCAP_SERVICES", json.dumps(vcap))
        monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY_ID", "env-key")

        creds = _get_credentials()
        assert creds["access_key_id"] == "vcap-key"
        assert creds["bucket"] == "vcap-bucket"
        assert creds["region"] == "us-east-1"

    def test_empty_vcap_falls_through(self, monkeypatch):
        """Empty VCAP_SERVICES should fall through to env vars."""
        monkeypatch.setenv("VCAP_SERVICES", "{}")
        monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY_ID", "fallback-key")
        monkeypatch.setenv("OBJECT_STORE_SECRET_ACCESS_KEY", "fallback-secret")
        monkeypatch.setenv("OBJECT_STORE_BUCKET", "fallback-bucket")

        creds = _get_credentials()
        assert creds["access_key_id"] == "fallback-key"

    def test_default_region(self, monkeypatch):
        """Default region should be eu-central-1."""
        monkeypatch.delenv("VCAP_SERVICES", raising=False)
        monkeypatch.delenv("OBJECT_STORE_REGION", raising=False)
        monkeypatch.setenv("OBJECT_STORE_ACCESS_KEY_ID", "k")
        monkeypatch.setenv("OBJECT_STORE_SECRET_ACCESS_KEY", "s")
        monkeypatch.setenv("OBJECT_STORE_BUCKET", "b")

        creds = _get_credentials()
        assert creds["region"] == "eu-central-1"


# ---------------------------------------------------------------------------
# ObjectStore method tests
# ---------------------------------------------------------------------------

class TestObjectStoreMethods:
    """Test ObjectStore upload/download/delete/exists with mocked S3 client."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client context manager."""
        mock_client = AsyncMock()
        mock_client.put_object = AsyncMock()
        mock_client.get_object = AsyncMock()
        mock_client.delete_object = AsyncMock()
        mock_client.head_object = AsyncMock()
        return mock_client

    @pytest.fixture
    def store(self, mock_s3_client):
        """Create an ObjectStore instance with mocked session."""
        creds = {
            "access_key_id": "test-key",
            "secret_access_key": "test-secret",
            "bucket": "test-bucket",
            "host": "s3.eu-central-1.amazonaws.com",
            "region": "eu-central-1",
        }
        obj_store = ObjectStore(creds)

        # Patch the session's client to return our mock
        async def _mock_client(**kwargs):
            return mock_s3_client

        # We need to mock the async context manager
        class MockContextManager:
            async def __aenter__(self):
                return mock_s3_client

            async def __aexit__(self, *args):
                pass

        obj_store._session = MagicMock()
        obj_store._session.client = MagicMock(return_value=MockContextManager())
        return obj_store

    @pytest.mark.asyncio
    async def test_upload_prefixes_key(self, store, mock_s3_client):
        """Upload should prefix the key with KEY_PREFIX."""
        await store.upload("ctx/file-id/0.png", b"data", "image/png")
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args.kwargs
        assert call_kwargs["Key"] == f"{KEY_PREFIX}/ctx/file-id/0.png"
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["ContentType"] == "image/png"
        assert call_kwargs["Body"] == b"data"

    @pytest.mark.asyncio
    async def test_download_returns_bytes(self, store, mock_s3_client):
        """Download should return the object body bytes."""
        mock_body = AsyncMock()
        mock_body.read = AsyncMock(return_value=b"file content")
        mock_s3_client.get_object = AsyncMock(return_value={"Body": mock_body})

        result = await store.download("ctx/file-id/thumb.png")
        assert result == b"file content"

    @pytest.mark.asyncio
    async def test_download_returns_none_for_missing(self, store, mock_s3_client):
        """Download should return None for non-existent keys."""
        mock_s3_client.get_object = AsyncMock(
            side_effect=Exception("NoSuchKey: The specified key does not exist")
        )
        result = await store.download("missing/key.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_prefixes_key(self, store, mock_s3_client):
        """Delete should prefix the key."""
        await store.delete("ctx/file-id/original.png")
        mock_s3_client.delete_object.assert_called_once()
        call_kwargs = mock_s3_client.delete_object.call_args.kwargs
        assert call_kwargs["Key"] == f"{KEY_PREFIX}/ctx/file-id/original.png"

    @pytest.mark.asyncio
    async def test_exists_returns_true(self, store, mock_s3_client):
        """Exists should return True when head_object succeeds."""
        result = await store.exists("ctx/file-id/0.png")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_error(self, store, mock_s3_client):
        """Exists should return False when head_object raises."""
        mock_s3_client.head_object = AsyncMock(side_effect=Exception("404"))
        result = await store.exists("missing.png")
        assert result is False


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

class TestSingleton:
    """Test the get_object_store() singleton."""

    @pytest.mark.asyncio
    async def test_raises_without_credentials(self, monkeypatch):
        """Should raise RuntimeError if credentials not configured."""
        monkeypatch.delenv("VCAP_SERVICES", raising=False)
        monkeypatch.delenv("OBJECT_STORE_ACCESS_KEY_ID", raising=False)
        monkeypatch.setenv("OBJECT_STORE_BUCKET", "")

        # Reset singleton
        import app.services.object_store as module
        module._store = None

        with pytest.raises(RuntimeError, match="credentials not configured"):
            await get_object_store()
