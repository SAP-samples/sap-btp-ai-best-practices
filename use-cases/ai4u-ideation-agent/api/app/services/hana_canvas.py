"""HANA Cloud persistence layer for Canvas tile positions.

Stores per-user, per-workspace tile layouts in SAP HANA Cloud.
Two-table schema: CANVASES (metadata) + CANVAS_TILES (tile positions).
Follows the same hdbcli + asyncio.to_thread pattern as hana_events.py.

Collaborative mode: Company and Event canvases are shared (single board per group).
Personal canvases remain per-user.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

MAX_TILES_PER_CANVAS = 200

# Per-canvas lock to serialize concurrent saves (keyed by canvas_id)
_save_locks: Dict[str, asyncio.Lock] = {}

# Per-canvas notification: SSE loops await this instead of polling.
_canvas_changed: Dict[str, asyncio.Event] = {}


def _get_save_lock(canvas_id: str) -> asyncio.Lock:
    """Get or create a per-canvas asyncio.Lock to serialize saves."""
    if canvas_id not in _save_locks:
        _save_locks[canvas_id] = asyncio.Lock()
    return _save_locks[canvas_id]


def notify_canvas_changed(canvas_id: str) -> None:
    """Signal all SSE waiters that this canvas has new data."""
    ev = _canvas_changed.get(canvas_id)
    if ev:
        ev.set()


async def wait_for_canvas_change(canvas_id: str, timeout: float = 2.0) -> bool:
    """Wait up to `timeout` seconds for a canvas write signal."""
    if canvas_id not in _canvas_changed:
        _canvas_changed[canvas_id] = asyncio.Event()
    ev = _canvas_changed[canvas_id]
    ev.clear()
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


def _extract_email_domain(email: str) -> str:
    """Extract domain from email address."""
    if not email or "@" not in email:
        return ""
    return email.split("@", 1)[1].lower()


def _is_public_email_domain(domain: str) -> bool:
    """Check if domain is a public/free email provider."""
    public_domains = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "live.com", "icloud.com", "mail.com", "protonmail.com",
        "aol.com", "zoho.com", "yandex.com", "gmx.com",
    }
    return domain.lower() in public_domains


class HanaCanvasStore:
    """HANA-backed store for canvas tile persistence."""

    async def init_schema(self) -> None:
        """Create tables and indexes (idempotent)."""
        from ideation_agent.hana_store import _execute

        # --- CANVASES table ---
        try:
            await _execute("""CREATE COLUMN TABLE CANVASES (
                CANVAS_ID        NVARCHAR(36)  PRIMARY KEY,
                USER_ID          NVARCHAR(256) NOT NULL,
                WORKSPACE_ID     NVARCHAR(36)  NOT NULL,
                NAME             NVARCHAR(256) NOT NULL DEFAULT 'Value vs Feasibility',
                CREATED_AT       TIMESTAMP     NOT NULL,
                UPDATED_AT       TIMESTAMP     NOT NULL
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # --- CANVAS_TILES table ---
        try:
            await _execute("""CREATE COLUMN TABLE CANVAS_TILES (
                TILE_ID          NVARCHAR(64)  PRIMARY KEY,
                CANVAS_ID        NVARCHAR(36)  NOT NULL,
                TILE_TYPE        NVARCHAR(20)  NOT NULL,
                CONTEXT_ID       NVARCHAR(64)  DEFAULT NULL,
                TITLE            NVARCHAR(512) DEFAULT '',
                BODY             NCLOB,
                POS_X            INTEGER       NOT NULL,
                POS_Y            INTEGER       NOT NULL,
                COLOR            NVARCHAR(20)  DEFAULT '',
                Z_INDEX          INTEGER       DEFAULT 0,
                CREATED_AT       TIMESTAMP     NOT NULL,
                UPDATED_AT       TIMESTAMP     NOT NULL
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # --- Schema migration: add ownership columns to CANVASES ---
        for alter_sql in [
            "ALTER TABLE CANVASES ADD (OWNER_TYPE NVARCHAR(20) DEFAULT 'user')",
            "ALTER TABLE CANVASES ADD (OWNER_ID NVARCHAR(256) DEFAULT '')",
        ]:
            try:
                await _execute(alter_sql)
            except Exception:
                pass  # Column already exists

        # --- Schema migration: add lock columns to CANVAS_TILES ---
        for alter_sql in [
            "ALTER TABLE CANVAS_TILES ADD (LOCKED_BY NVARCHAR(256) DEFAULT NULL)",
            "ALTER TABLE CANVAS_TILES ADD (LOCKED_AT TIMESTAMP DEFAULT NULL)",
        ]:
            try:
                await _execute(alter_sql)
            except Exception:
                pass  # Column already exists

        # Indexes
        for sql in [
            "CREATE INDEX IX_CANVASES_USER_WS ON CANVASES(USER_ID, WORKSPACE_ID)",
            "CREATE INDEX IX_CANVAS_TILES_CANVAS ON CANVAS_TILES(CANVAS_ID)",
            "CREATE INDEX IX_CANVAS_TILES_CTX ON CANVAS_TILES(CANVAS_ID, CONTEXT_ID)",
            "CREATE INDEX IX_CANVASES_OWNER ON CANVASES(OWNER_TYPE, OWNER_ID)",
        ]:
            try:
                await _execute(sql)
            except Exception:
                pass

    # ── Canvas resolution ────────────────────────────────────────

    async def get_or_create_canvas(self, user_id: str, workspace_id: str) -> Dict[str, Any]:
        """Get the default canvas for (user, workspace), creating one if it doesn't exist."""
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT CANVAS_ID, NAME, CREATED_AT, UPDATED_AT FROM CANVASES WHERE USER_ID = ? AND WORKSPACE_ID = ?",
            (user_id, workspace_id),
        )
        if row:
            return row

        # Create default canvas
        canvas_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await _execute(
            """INSERT INTO CANVASES (CANVAS_ID, USER_ID, WORKSPACE_ID, NAME, OWNER_TYPE, OWNER_ID, CREATED_AT, UPDATED_AT)
               VALUES (?, ?, ?, 'Value vs Feasibility', 'user', ?, ?, ?)""",
            (canvas_id, user_id, workspace_id, user_id, now, now),
        )
        return {"canvas_id": canvas_id, "name": "Value vs Feasibility", "created_at": now, "updated_at": now}

    async def get_or_create_shared_canvas(self, owner_type: str, owner_id: str) -> Dict[str, Any]:
        """Get or create the single shared canvas for a group (company domain or event_id).

        Uses a catch-duplicate-then-retry pattern to prevent race conditions
        when two users access a shared canvas for the first time simultaneously.
        """
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT CANVAS_ID, NAME, CREATED_AT, UPDATED_AT FROM CANVASES WHERE OWNER_TYPE = ? AND OWNER_ID = ?",
            (owner_type, owner_id),
        )
        if row:
            return row

        # Create shared canvas — handle race condition gracefully
        canvas_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        try:
            await _execute(
                """INSERT INTO CANVASES (CANVAS_ID, USER_ID, WORKSPACE_ID, NAME, OWNER_TYPE, OWNER_ID, CREATED_AT, UPDATED_AT)
                   VALUES (?, ?, ?, 'Value vs Feasibility', ?, ?, ?, ?)""",
                (canvas_id, "__shared__", "__shared__", owner_type, owner_id, now, now),
            )
        except Exception as exc:
            # Another user may have created it concurrently — retry SELECT
            err_str = str(exc).lower()
            if "unique" in err_str or "duplicate" in err_str or "301" in err_str:
                row = await _fetch_one(
                    "SELECT CANVAS_ID, NAME, CREATED_AT, UPDATED_AT FROM CANVASES WHERE OWNER_TYPE = ? AND OWNER_ID = ?",
                    (owner_type, owner_id),
                )
                if row:
                    return row
            raise

        return {"canvas_id": canvas_id, "name": "Value vs Feasibility", "created_at": now, "updated_at": now}

    # ── Tile loading ─────────────────────────────────────────────

    async def get_workspace_tiles(self, user_id: str, workspace_id: str) -> List[Dict[str, Any]]:
        """Load all tiles for a user's personal canvas in a given workspace."""
        canvas = await self.get_or_create_canvas(user_id, workspace_id)
        return await self._load_canvas_tiles(canvas["canvas_id"])

    async def get_shared_tiles(self, canvas_id: str) -> List[Dict[str, Any]]:
        """Load all tiles for a shared canvas."""
        return await self._load_canvas_tiles(canvas_id)

    async def _load_canvas_tiles(self, canvas_id: str) -> List[Dict[str, Any]]:
        """Load active tiles for a canvas with STILL_VALID check."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT
                 t.TILE_ID, t.TILE_TYPE, t.CONTEXT_ID, t.TITLE, t.BODY,
                 t.POS_X, t.POS_Y, t.COLOR, t.Z_INDEX, t.LOCKED_BY,
                 CASE WHEN t.CONTEXT_ID IS NULL THEN 1
                      WHEN EXISTS (SELECT 1 FROM IDEATION_SUBMISSIONS s WHERE s.CONTEXT_ID = t.CONTEXT_ID) THEN 1
                      ELSE 0
                 END AS STILL_VALID
               FROM CANVAS_TILES t
               WHERE t.CANVAS_ID = ?
               ORDER BY t.Z_INDEX ASC""",
            (canvas_id,),
        )
        return rows

    # ── Full-board save ──────────────────────────────────────────

    async def bulk_save(self, canvas_id: str, tiles: List[Dict[str, Any]], skip_prune: bool = False) -> None:
        """Save all tiles for a canvas (upsert + optional prune).

        Uses UPSERT (INSERT ... WITH PRIMARY KEY) for each tile. On personal
        canvases, prunes tiles no longer in the incoming set. On shared canvases
        (skip_prune=True), only upserts — removals are handled via explicit DELETE.
        """
        from ideation_agent.hana_store import _execute, _fetch_all

        if len(tiles) > MAX_TILES_PER_CANVAS:
            raise ValueError(f"Maximum {MAX_TILES_PER_CANVAS} tiles per canvas exceeded")

        now = datetime.now(timezone.utc)

        lock = _get_save_lock(canvas_id)
        async with lock:
            # Upsert each tile
            incoming_ids = set()
            for tile in tiles:
                tile_id = tile["tile_id"]
                incoming_ids.add(tile_id)
                await _execute(
                    """UPSERT CANVAS_TILES
                       (TILE_ID, CANVAS_ID, TILE_TYPE, CONTEXT_ID, TITLE, BODY, POS_X, POS_Y, COLOR, Z_INDEX, CREATED_AT, UPDATED_AT)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       WITH PRIMARY KEY""",
                    (
                        tile_id,
                        canvas_id,
                        tile["tile_type"],
                        tile.get("context_id") or None,
                        tile.get("title", ""),
                        tile.get("body", ""),
                        int(tile["pos_x"]),
                        int(tile["pos_y"]),
                        tile.get("color", ""),
                        int(tile.get("z_index", 0)),
                        now,
                        now,
                    ),
                )

            # Prune tiles no longer on board (personal canvases only)
            if not skip_prune:
                existing_rows = await _fetch_all(
                    "SELECT TILE_ID FROM CANVAS_TILES WHERE CANVAS_ID = ?",
                    (canvas_id,),
                )
                for row in existing_rows:
                    if row["tile_id"] not in incoming_ids:
                        await _execute(
                            "DELETE FROM CANVAS_TILES WHERE TILE_ID = ? AND CANVAS_ID = ?",
                            (row["tile_id"], canvas_id),
                        )

            # Update canvas timestamp
            await _execute(
                "UPDATE CANVASES SET UPDATED_AT = ? WHERE CANVAS_ID = ?",
                (now, canvas_id),
            )

        # Notify SSE waiters outside the lock
        notify_canvas_changed(canvas_id)

    async def delete_tile(self, canvas_id: str, tile_id: str) -> bool:
        """Delete a single tile from a canvas. Returns True if found and deleted."""
        from ideation_agent.hana_store import _execute, _fetch_one

        existing = await _fetch_one(
            "SELECT TILE_ID FROM CANVAS_TILES WHERE TILE_ID = ? AND CANVAS_ID = ?",
            (tile_id, canvas_id),
        )
        if not existing:
            return False

        now = datetime.now(timezone.utc)
        await _execute(
            "DELETE FROM CANVAS_TILES WHERE TILE_ID = ? AND CANVAS_ID = ?",
            (tile_id, canvas_id),
        )
        # Bump canvas timestamp so SSE picks up the deletion
        await _execute(
            "UPDATE CANVASES SET UPDATED_AT = ? WHERE CANVAS_ID = ?",
            (now, canvas_id),
        )
        notify_canvas_changed(canvas_id)
        return True

    async def get_canvas_updated_at(self, canvas_id: str) -> Optional[datetime]:
        """Get the UPDATED_AT timestamp for a canvas (used by SSE poll loop)."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT UPDATED_AT FROM CANVASES WHERE CANVAS_ID = ?",
            (canvas_id,),
        )
        return row["updated_at"] if row else None

    # ── Tile locking ────────────────────────────────────────────

    async def acquire_lock(self, canvas_id: str, tile_id: str, user_id: str) -> bool:
        """Try to lock a tile for editing. Returns True if lock acquired."""
        from ideation_agent.hana_store import _execute, _fetch_one
        from datetime import timedelta

        LOCK_TIMEOUT = timedelta(seconds=300)
        now = datetime.now(timezone.utc)
        cutoff = now - LOCK_TIMEOUT

        row = await _fetch_one(
            "SELECT LOCKED_BY, LOCKED_AT FROM CANVAS_TILES WHERE TILE_ID = ? AND CANVAS_ID = ?",
            (tile_id, canvas_id),
        )
        if not row:
            return False

        locked_by = row.get("locked_by")
        locked_at = row.get("locked_at")

        if locked_by and locked_by != user_id and locked_at and locked_at > cutoff:
            return False  # genuinely locked by someone else

        await _execute(
            "UPDATE CANVAS_TILES SET LOCKED_BY = ?, LOCKED_AT = ? WHERE TILE_ID = ? AND CANVAS_ID = ?",
            (user_id, now, tile_id, canvas_id),
        )
        notify_canvas_changed(canvas_id)
        return True

    async def release_lock(self, canvas_id: str, tile_id: str, user_id: str) -> None:
        """Release lock (only if owned by this user)."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "UPDATE CANVAS_TILES SET LOCKED_BY = NULL, LOCKED_AT = NULL WHERE TILE_ID = ? AND CANVAS_ID = ? AND LOCKED_BY = ?",
            (tile_id, canvas_id, user_id),
        )
        notify_canvas_changed(canvas_id)


# ── Presence tracking (in-memory, per-process) ───────────────────

_canvas_presence: Dict[str, Dict[str, dict]] = {}


def join_canvas(canvas_id: str, user_id: str, display_name: str) -> None:
    """Register a user as present on a canvas."""
    if canvas_id not in _canvas_presence:
        _canvas_presence[canvas_id] = {}
    _canvas_presence[canvas_id][user_id] = {
        "user_id": user_id,
        "display_name": display_name,
        "initials": "".join(w[0].upper() for w in display_name.split()[:2]) or "?",
    }
    notify_canvas_changed(canvas_id)


def leave_canvas(canvas_id: str, user_id: str) -> None:
    """Remove a user from canvas presence."""
    if canvas_id in _canvas_presence:
        _canvas_presence[canvas_id].pop(user_id, None)
        if not _canvas_presence[canvas_id]:
            del _canvas_presence[canvas_id]
        notify_canvas_changed(canvas_id)


def get_canvas_viewers(canvas_id: str) -> list:
    """Get list of users currently viewing a canvas."""
    return list((_canvas_presence.get(canvas_id) or {}).values())


# ── Access control helper ────────────────────────────────────────

async def resolve_canvas_with_access_check(
    workspace_id: str, user_id: str, email: str
) -> Tuple[Dict[str, Any], bool]:
    """Resolve workspace to a canvas + verify access.

    Returns: (canvas_dict, is_shared)
    Raises: HTTPException(403) if access denied.
    """
    store = await get_canvas_store()

    if workspace_id == "__personal__":
        canvas = await store.get_or_create_canvas(user_id, workspace_id)
        return canvas, False

    elif workspace_id == "__company__":
        domain = _extract_email_domain(email)
        if not domain or _is_public_email_domain(domain):
            raise HTTPException(status_code=403, detail="Company canvas requires a corporate email")
        canvas = await store.get_or_create_shared_canvas("company", domain)
        return canvas, True

    else:
        # Event workspace — check membership
        from .hana_events import get_event_store
        event_store = await get_event_store()
        role = await event_store.get_member_role(workspace_id, user_id)
        if not role:
            raise HTTPException(status_code=403, detail="Not a member of this event")
        canvas = await store.get_or_create_shared_canvas("event", workspace_id)
        return canvas, True


# Module-level singleton
_store: Optional[HanaCanvasStore] = None
_store_lock = asyncio.Lock()


async def get_canvas_store() -> HanaCanvasStore:
    """Get or create the singleton canvas store (initializes schema on first call)."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaCanvasStore()
        try:
            await store.init_schema()
            logger.info("HANA canvas store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA canvas schema: %s", exc)
            raise
        _store = store
        return _store
