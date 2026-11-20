"""Shared pytest fixtures for the unit test suite.

Provides async_upload, a factory fixture that creates in-memory mock UploadFile
objects suitable for testing read_upload_bytes without real file I/O.

Usage::

    async def test_something(async_upload) -> None:
        upload = async_upload(b"hello", filename="ecn.json", content_type="application/json")
        data = await read_upload_bytes(upload, maximum_bytes=1024)
"""

import io
from typing import Callable

import pytest


class _AsyncBytesUpload:
    """Minimal async-readable mock for FastAPI UploadFile.

    Only implements the ``read(size)`` coroutine; all other UploadFile
    attributes are intentionally absent so that tests cannot rely on them.

    Args:
        data: The bytes the mock will yield on successive ``read`` calls.
        filename: Exposed as ``filename`` attribute (mirrors UploadFile).
        content_type: Exposed as ``content_type`` attribute.
    """

    def __init__(self, data: bytes, *, filename: str = "test", content_type: str = "") -> None:
        self._buf = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        """Async read from the in-memory buffer — mirrors UploadFile.read."""
        return self._buf.read(size)


@pytest.fixture
def async_upload() -> Callable[..., _AsyncBytesUpload]:
    """Return a factory that creates in-memory async-readable UploadFile mocks.

    Each call to the returned factory produces a fresh ``_AsyncBytesUpload``
    whose internal buffer is fully rewound.  Tests pass only the fields they
    care about; unspecified fields use safe defaults.

    Returns:
        A callable ``(data, *, filename, content_type) -> _AsyncBytesUpload``.
    """
    def _factory(
        data: bytes,
        *,
        filename: str = "test",
        content_type: str = "",
    ) -> _AsyncBytesUpload:
        """Build one mock upload backed by ``data``.

        Args:
            data: Raw bytes the mock will stream.
            filename: Filename attribute on the mock.
            content_type: Content-type attribute on the mock.

        Returns:
            A fresh ``_AsyncBytesUpload`` instance.
        """
        return _AsyncBytesUpload(data, filename=filename, content_type=content_type)

    return _factory
