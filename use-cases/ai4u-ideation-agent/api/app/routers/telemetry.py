"""Telemetry receiver — accepts version-check pings from customer instances.

This endpoint is UNAUTHENTICATED (customer instances don't have SAP auth tokens).
It doubles as a version-check endpoint: returns LOCAL_VERSION as plain text.
Only registered on the SAP-hosted instance (INSTALLATION_ID=sap-hosted-primary).
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from ..services.version_check import LOCAL_VERSION

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory rate limiting ──────────────────────────────────────────────────
_RATE_WINDOW = 300  # 5 minutes
_RATE_LIMIT = 10  # max pings per installation per window
_MAX_TRACKED_IDS = 10_000  # memory cap
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_last_prune: float = 0.0

_VALID_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")


def _prune_rate_buckets() -> None:
    """Remove stale entries to prevent unbounded memory growth."""
    global _last_prune
    now = time.time()
    if now - _last_prune < _RATE_WINDOW:
        return
    _last_prune = now
    stale_keys = [
        k for k, v in _rate_buckets.items() if not v or (now - v[-1]) > _RATE_WINDOW
    ]
    for k in stale_keys:
        del _rate_buckets[k]


def _is_rate_limited(installation_id: str) -> bool:
    """Simple sliding-window rate limiter per installation_id."""
    _prune_rate_buckets()

    # Memory cap: reject new IDs beyond limit
    if installation_id not in _rate_buckets and len(_rate_buckets) >= _MAX_TRACKED_IDS:
        return True

    now = time.time()
    bucket = _rate_buckets[installation_id]
    # Prune old entries for this ID
    _rate_buckets[installation_id] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(_rate_buckets[installation_id]) >= _RATE_LIMIT:
        return True
    _rate_buckets[installation_id].append(now)
    return False


@router.get("/version")
async def telemetry_version(
    id: Optional[str] = Query(None, max_length=64),
    v: Optional[str] = Query(None, max_length=32),
    c: int = Query(0, ge=0, le=999_999_999),
    u: int = Query(0, ge=0, le=999_999_999),
    s: int = Query(0, ge=0, le=999_999_999),
) -> PlainTextResponse:
    """Receive telemetry ping and return current version.

    This endpoint is UNAUTHENTICATED — customer instances don't have SAP auth tokens.
    It doubles as the version-check endpoint (returns the VERSION string as plain text).
    """
    # Always return version string (even if telemetry recording fails)
    if not id:
        return PlainTextResponse(LOCAL_VERSION)

    # Validate installation_id format
    if not _VALID_ID_RE.match(id):
        return PlainTextResponse(LOCAL_VERSION)

    # Rate limit per installation
    if _is_rate_limited(id):
        return PlainTextResponse(LOCAL_VERSION)

    # Record telemetry (best-effort, never blocks the version response)
    try:
        from ..services.hana_telemetry import get_telemetry_store

        store = await get_telemetry_store()
        await store.upsert_installation(id, v or "unknown", c, u, s)
        await store.record_daily_snapshot(id, v or "unknown", c, u, s)
    except Exception as exc:
        logger.warning("Telemetry record failed for %s: %s", id, exc)

    return PlainTextResponse(LOCAL_VERSION)
