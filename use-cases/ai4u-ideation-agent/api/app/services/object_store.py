"""Object Store service — async S3 client for BTP Object Store.

Provides upload, download, delete, and exists operations using aioboto3.
Singleton pattern with asyncio.Lock (same as hana_store.py).

Credentials resolved from:
1. VCAP_SERVICES["objectstore"][0]["credentials"] (Cloud Foundry)
2. Environment variables (local dev fallback)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import aioboto3

logger = logging.getLogger(__name__)

_store: Optional["ObjectStore"] = None
_store_lock = asyncio.Lock()


def _get_credentials() -> Dict[str, Any]:
    """Resolve Object Store credentials from VCAP_SERVICES or env vars.

    BTP Object Store credentials have NO namespace field — just
    access_key_id, secret_access_key, bucket, host, region.
    """
    # Try VCAP_SERVICES first (Cloud Foundry production)
    vcap = os.getenv("VCAP_SERVICES", "")
    if vcap:
        try:
            services = json.loads(vcap)
            creds = services.get("objectstore", [{}])[0].get("credentials", {})
            if creds.get("access_key_id"):
                return {
                    "access_key_id": creds["access_key_id"],
                    "secret_access_key": creds["secret_access_key"],
                    "bucket": creds["bucket"],
                    "host": creds.get("host", ""),
                    "region": creds.get("region", "eu-central-1"),
                }
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Fallback to explicit env vars (local development)
    return {
        "access_key_id": os.getenv("OBJECT_STORE_ACCESS_KEY_ID", ""),
        "secret_access_key": os.getenv("OBJECT_STORE_SECRET_ACCESS_KEY", ""),
        "bucket": os.getenv("OBJECT_STORE_BUCKET", ""),
        "host": os.getenv("OBJECT_STORE_HOST", ""),
        "region": os.getenv("OBJECT_STORE_REGION", "eu-central-1"),
    }


# Key prefix for all files in this application
KEY_PREFIX = "ai4u/files"


class ObjectStore:
    """Async S3 client wrapper for BTP Object Store."""

    def __init__(self, credentials: Dict[str, Any]):
        self._creds = credentials
        self._session = aioboto3.Session()
        self._bucket = credentials["bucket"]
        self._endpoint_url = f"https://{credentials['host']}" if credentials.get("host") else None

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "service_name": "s3",
            "aws_access_key_id": self._creds["access_key_id"],
            "aws_secret_access_key": self._creds["secret_access_key"],
            "region_name": self._creds["region"],
        }
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        return kwargs

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        """Upload bytes to Object Store.

        Args:
            key: Object key (relative, will be prefixed with KEY_PREFIX).
            data: File content as bytes.
            content_type: MIME type for the object.
        """
        full_key = f"{KEY_PREFIX}/{key}"
        async with self._session.client(**self._client_kwargs()) as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=full_key,
                Body=data,
                ContentType=content_type,
            )
        logger.debug("Uploaded %s (%d bytes)", full_key, len(data))

    async def download(self, key: str) -> Optional[bytes]:
        """Download bytes from Object Store. Returns None if not found.

        Args:
            key: Object key (relative, will be prefixed with KEY_PREFIX).
        """
        full_key = f"{KEY_PREFIX}/{key}"
        try:
            async with self._session.client(**self._client_kwargs()) as client:
                response = await client.get_object(Bucket=self._bucket, Key=full_key)
                data = await response["Body"].read()
                return data
        except Exception as exc:
            if "NoSuchKey" in str(exc) or "404" in str(exc):
                return None
            raise

    async def delete(self, key: str) -> None:
        """Delete an object from Object Store.

        Args:
            key: Object key (relative, will be prefixed with KEY_PREFIX).
        """
        full_key = f"{KEY_PREFIX}/{key}"
        async with self._session.client(**self._client_kwargs()) as client:
            await client.delete_object(Bucket=self._bucket, Key=full_key)
        logger.debug("Deleted %s", full_key)

    async def exists(self, key: str) -> bool:
        """Check if an object exists in Object Store.

        Args:
            key: Object key (relative, will be prefixed with KEY_PREFIX).
        """
        full_key = f"{KEY_PREFIX}/{key}"
        try:
            async with self._session.client(**self._client_kwargs()) as client:
                await client.head_object(Bucket=self._bucket, Key=full_key)
                return True
        except Exception:
            return False


async def get_object_store() -> ObjectStore:
    """Get or create the singleton ObjectStore instance."""
    global _store
    if _store is not None:
        return _store

    async with _store_lock:
        if _store is not None:
            return _store

        creds = _get_credentials()
        if not creds.get("access_key_id") or not creds.get("bucket"):
            raise RuntimeError(
                "Object Store credentials not configured. Set OBJECT_STORE_ACCESS_KEY_ID, "
                "OBJECT_STORE_SECRET_ACCESS_KEY, OBJECT_STORE_BUCKET, etc. or bind an "
                "objectstore service in Cloud Foundry."
            )

        _store = ObjectStore(creds)
        logger.info(
            "Object Store initialized: bucket=%s region=%s",
            creds["bucket"],
            creds["region"],
        )
        return _store
