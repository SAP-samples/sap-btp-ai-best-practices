"""Unit tests for the in-memory file output cache."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.services.file_cache import (
    cache_file_outputs,
    get_file_outputs,
    _cache,
    CACHE_TTL_SECONDS,
    CACHE_MAX_ENTRIES,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with a clean cache."""
    _cache.clear()
    yield
    _cache.clear()


class TestCacheBasics:
    """Test basic cache operations."""

    def test_cache_and_retrieve(self):
        """Stored outputs should be retrievable."""
        outputs = [b"png_bytes_page_0"]
        cache_file_outputs("ctx-1", "file-1", outputs)
        result = get_file_outputs("ctx-1", "file-1")
        assert result == outputs

    def test_miss_returns_none(self):
        """Non-existent key should return None."""
        result = get_file_outputs("ctx-1", "nonexistent")
        assert result is None

    def test_consume_once(self):
        """Second retrieval should return None (consume-once semantics)."""
        cache_file_outputs("ctx-1", "file-1", [b"data"])
        assert get_file_outputs("ctx-1", "file-1") == [b"data"]
        assert get_file_outputs("ctx-1", "file-1") is None

    def test_multiple_outputs(self):
        """Multiple output pages should all be stored."""
        outputs = [b"page0", b"page1", b"page2"]
        cache_file_outputs("ctx-1", "file-1", outputs)
        result = get_file_outputs("ctx-1", "file-1")
        assert result == outputs
        assert len(result) == 3

    def test_independent_entries(self):
        """Different file_ids should be independent."""
        cache_file_outputs("ctx-1", "file-a", [b"a_data"])
        cache_file_outputs("ctx-1", "file-b", [b"b_data"])
        assert get_file_outputs("ctx-1", "file-a") == [b"a_data"]
        assert get_file_outputs("ctx-1", "file-b") == [b"b_data"]

    def test_different_contexts_independent(self):
        """Same file_id in different contexts should be independent."""
        cache_file_outputs("ctx-1", "file-1", [b"ctx1_data"])
        cache_file_outputs("ctx-2", "file-1", [b"ctx2_data"])
        assert get_file_outputs("ctx-1", "file-1") == [b"ctx1_data"]
        assert get_file_outputs("ctx-2", "file-1") == [b"ctx2_data"]


class TestCacheTTL:
    """Test TTL expiration."""

    def test_expired_entry_returns_none(self):
        """Entries past TTL should return None."""
        cache_file_outputs("ctx-1", "file-1", [b"data"])
        # Simulate time passing
        entry = _cache[("ctx-1", "file-1")]
        entry.created_at = time.time() - CACHE_TTL_SECONDS - 1
        assert get_file_outputs("ctx-1", "file-1") is None

    def test_fresh_entry_returns_data(self):
        """Entries within TTL should return data."""
        cache_file_outputs("ctx-1", "file-1", [b"data"])
        # Immediately retrieve (well within TTL)
        assert get_file_outputs("ctx-1", "file-1") == [b"data"]


class TestCacheEviction:
    """Test eviction behavior."""

    def test_max_entries_enforced(self):
        """Cache should not grow beyond CACHE_MAX_ENTRIES."""
        for i in range(CACHE_MAX_ENTRIES + 10):
            cache_file_outputs("ctx", f"file-{i}", [b"data"])
        assert len(_cache) <= CACHE_MAX_ENTRIES

    def test_expired_entries_evicted_on_write(self):
        """Writing new entries should evict expired ones."""
        # Add an entry and make it expired
        cache_file_outputs("ctx", "old-file", [b"old"])
        _cache[("ctx", "old-file")].created_at = time.time() - CACHE_TTL_SECONDS - 1

        # New write triggers eviction
        cache_file_outputs("ctx", "new-file", [b"new"])

        assert ("ctx", "old-file") not in _cache
        assert get_file_outputs("ctx", "new-file") == [b"new"]
