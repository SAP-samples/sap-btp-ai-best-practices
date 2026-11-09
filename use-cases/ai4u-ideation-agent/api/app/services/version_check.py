"""Version check service — compares local version to upstream and sends anonymous telemetry."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# --- Configuration ---
_DEFAULT_UPDATE_URL = (
    "https://ai4u-ideation-agent-api.cfapps.eu10-005.hana.ondemand.com/api/telemetry/version"
)

UPDATE_CHECK_URL: str = os.getenv("UPDATE_CHECK_URL", _DEFAULT_UPDATE_URL).strip()
UPDATE_CHECK_ENABLED: bool = os.getenv("UPDATE_CHECK_ENABLED", "true").strip().lower() in (
    "true",
    "1",
    "yes",
)
INSTALLATION_ID: str = os.getenv("INSTALLATION_ID", "").strip()
CACHE_TTL_SECONDS: int = 3600  # 1 hour


# --- Local version (read once at module load) ---


def _read_local_version() -> str:
    """Read VERSION file from project root.

    In CF deployment, deploy.sh copies VERSION into api/, so it sits next to
    requirements.txt. In local dev, it's at the repo root (one level above api/).
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "VERSION",  # api/VERSION
        Path(__file__).resolve().parent.parent.parent.parent / "VERSION",  # repo root
    ]
    for path in candidates:
        if path.is_file():
            content = path.read_text().strip()
            if content:
                return content
    return "0.0.0"


LOCAL_VERSION: str = _read_local_version()


# --- In-memory cache ---


class _UpstreamCache:
    """Simple TTL cache for the upstream version string."""

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.fetched_at: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.version is not None and (time.time() - self.fetched_at) < CACHE_TTL_SECONDS

    def set(self, version: str) -> None:
        self.version = version
        self.fetched_at = time.time()

    def clear(self) -> None:
        self.version = None
        self.fetched_at = 0.0


_cache = _UpstreamCache()


# --- Semver comparison ---


def _parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Parse 'X.Y.Z' into (major, minor, patch). Ignores pre-release suffixes."""
    parts = version_str.strip().split(".")
    return (
        int(parts[0]) if len(parts) > 0 else 0,
        int(parts[1]) if len(parts) > 1 else 0,
        int(parts[2].split("-")[0]) if len(parts) > 2 else 0,
    )


def _is_newer(upstream: str, local: str) -> bool:
    """Return True if upstream is strictly newer than local."""
    return _parse_semver(upstream) > _parse_semver(local)


# --- Public API ---


# --- Installation ID resolution ---

_resolved_id: Optional[str] = None
_id_lock = asyncio.Lock()


async def _get_installation_id() -> str:
    """Resolve installation ID: env var → HANA TELEMETRY_CONFIG → generate UUID4."""
    global _resolved_id
    if _resolved_id:
        return _resolved_id
    async with _id_lock:
        if _resolved_id:
            return _resolved_id
        # Priority 1: env var (SAP instance uses "sap-hosted-primary")
        if INSTALLATION_ID:
            _resolved_id = INSTALLATION_ID
            return _resolved_id
        # Priority 2: HANA persistence
        try:
            from ideation_agent.hana_store import _execute, _fetch_one

            # Ensure table exists
            try:
                await _execute(
                    """CREATE COLUMN TABLE TELEMETRY_CONFIG (
                        KEY VARCHAR(64) PRIMARY KEY,
                        VAL VARCHAR(256)
                    )"""
                )
            except Exception:
                pass  # Table already exists

            row = await _fetch_one(
                "SELECT VAL FROM TELEMETRY_CONFIG WHERE KEY = ?",
                ("installation_id",),
            )
            if row and row.get("val"):
                _resolved_id = row["val"]
                return _resolved_id
            # Priority 3: Generate and persist
            new_id = str(uuid.uuid4())
            await _execute(
                "INSERT INTO TELEMETRY_CONFIG (KEY, VAL) VALUES (?, ?)",
                ("installation_id", new_id),
            )
            _resolved_id = new_id
            return _resolved_id
        except Exception as exc:
            logger.debug("HANA unavailable for installation ID, using ephemeral: %s", exc)
            # Priority 4: Ephemeral in-memory UUID
            _resolved_id = str(uuid.uuid4())
            return _resolved_id


async def _gather_telemetry_stats() -> dict:
    """Gather anonymous aggregate counts. Each table queried independently."""
    stats = {"c": 0, "u": 0, "s": 0}
    try:
        from ideation_agent.hana_store import _fetch_one

        # Conversations count
        try:
            row = await _fetch_one("SELECT COUNT(*) AS cnt FROM CONVERSATIONS")
            stats["c"] = int(row["cnt"]) if row else 0
        except Exception:
            pass

        # Unique users (from CONVERSATIONS — always exists, unlike IDEATION_USERS which is CDC-fallback only)
        try:
            row = await _fetch_one("SELECT COUNT(DISTINCT USER_ID) AS cnt FROM CONVERSATIONS")
            stats["u"] = int(row["cnt"]) if row else 0
        except Exception:
            pass

        # Submissions count
        try:
            row = await _fetch_one("SELECT COUNT(*) AS cnt FROM IDEATION_SUBMISSIONS")
            stats["s"] = int(row["cnt"]) if row else 0
        except Exception:
            pass
    except Exception:
        pass
    return stats


# --- Public API ---


async def check_for_update() -> dict:
    """Check if a newer version is available upstream.

    Returns:
        {
            "updateAvailable": bool | None,  (None = check failed or disabled)
            "currentVersion": str,
            "latestVersion": str | None,
            "checkEnabled": bool,
        }
    """
    if not UPDATE_CHECK_ENABLED:
        return {
            "updateAvailable": None,
            "currentVersion": LOCAL_VERSION,
            "latestVersion": None,
            "checkEnabled": False,
        }

    # Return cached result if still valid
    if _cache.is_valid:
        return {
            "updateAvailable": _is_newer(_cache.version, LOCAL_VERSION),
            "currentVersion": LOCAL_VERSION,
            "latestVersion": _cache.version,
            "checkEnabled": True,
        }

    # SAP-hosted instance: skip HTTP call to self, just self-report to HANA
    if INSTALLATION_ID == "sap-hosted-primary":
        try:
            stats = await _gather_telemetry_stats()
            from .hana_telemetry import get_telemetry_store

            store = await get_telemetry_store()
            await store.upsert_installation(
                INSTALLATION_ID, LOCAL_VERSION, stats["c"], stats["u"], stats["s"]
            )
            await store.record_daily_snapshot(
                INSTALLATION_ID, LOCAL_VERSION, stats["c"], stats["u"], stats["s"]
            )
        except Exception as exc:
            logger.debug("Self-report telemetry failed: %s", exc)
        _cache.set(LOCAL_VERSION)
        return {
            "updateAvailable": False,
            "currentVersion": LOCAL_VERSION,
            "latestVersion": LOCAL_VERSION,
            "checkEnabled": True,
        }

    # Fetch upstream VERSION file (with telemetry params appended)
    try:
        install_id = await _get_installation_id()
        stats = await _gather_telemetry_stats()
        params = {
            "id": install_id,
            "v": LOCAL_VERSION,
            "c": stats["c"],
            "u": stats["u"],
            "s": stats["s"],
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(UPDATE_CHECK_URL, params=params)
            resp.raise_for_status()
            upstream_version = resp.text.strip()

        # Validate it looks like a version string (raises ValueError if not)
        _parse_semver(upstream_version)
        _cache.set(upstream_version)

        return {
            "updateAvailable": _is_newer(upstream_version, LOCAL_VERSION),
            "currentVersion": LOCAL_VERSION,
            "latestVersion": upstream_version,
            "checkEnabled": True,
        }
    except Exception as exc:
        logger.warning("Version check failed: %s", exc)
        return {
            "updateAvailable": None,
            "currentVersion": LOCAL_VERSION,
            "latestVersion": None,
            "checkEnabled": True,
        }
