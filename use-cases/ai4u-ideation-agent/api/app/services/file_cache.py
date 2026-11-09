"""In-memory TTL cache for processed file bytes.

Eliminates the S3 round-trip when the A2A message/send arrives
shortly after the upload endpoint processes the file. The upload
stores processed output bytes here; the A2A handler consumes them
~1s later instead of re-downloading from Object Store.

Single-process only (safe for uvicorn without --workers).
Fallback: cache miss → S3 download (always correct for history
replay, multi-instance, or expired entries).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_TTL_SECONDS = 60  # Generous; actual gap is <1s
CACHE_MAX_ENTRIES = 50  # Safety cap (~50 * 2MB = 100MB worst case)


@dataclass
class _CachedFileOutputs:
    """Cached processed outputs for a single file upload."""

    outputs: List[bytes]
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > CACHE_TTL_SECONDS


# Cache keyed by (context_id, file_id)
_cache: Dict[tuple, _CachedFileOutputs] = {}


def cache_file_outputs(context_id: str, file_id: str, outputs: List[bytes]) -> None:
    """Store processed file outputs in the TTL cache.

    Called by the upload endpoint after process_file() succeeds.
    Evicts expired entries opportunistically.
    """
    _evict_expired()

    # Safety: don't exceed max entries (drop oldest if full)
    if len(_cache) >= CACHE_MAX_ENTRIES:
        _evict_oldest()

    key = (context_id, file_id)
    _cache[key] = _CachedFileOutputs(outputs=outputs)
    logger.debug("Cached file outputs: context=%s file=%s outputs=%d", context_id, file_id, len(outputs))


def get_file_outputs(context_id: str, file_id: str) -> Optional[List[bytes]]:
    """Retrieve cached file outputs. Returns None on miss.

    Removes the entry on hit (consume-once: saves memory).
    """
    key = (context_id, file_id)
    entry = _cache.pop(key, None)

    if entry is None:
        return None

    if entry.is_expired:
        logger.debug("Cache expired for context=%s file=%s", context_id, file_id)
        return None

    logger.info("File cache hit: skipped S3 download for %s/%s (%d outputs)", context_id, file_id, len(entry.outputs))
    return entry.outputs


def _evict_expired() -> None:
    """Remove all expired entries."""
    now = time.time()
    expired_keys = [k for k, v in _cache.items() if now - v.created_at > CACHE_TTL_SECONDS]
    for k in expired_keys:
        del _cache[k]
    if expired_keys:
        logger.debug("Evicted %d expired cache entries", len(expired_keys))


def _evict_oldest() -> None:
    """Remove the oldest entry to make room."""
    if not _cache:
        return
    oldest_key = min(_cache, key=lambda k: _cache[k].created_at)
    del _cache[oldest_key]
