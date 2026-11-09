"""HANA Cloud persistence layer for Conversation Shares.

Stores share links and membership for conversation sharing. Follows the same
hdbcli + asyncio.to_thread pattern as hana_events.py.
"""

from __future__ import annotations

import asyncio
import logging
import random
import string
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SHARE_CODE_LENGTH = 6
_SHARE_CODE_CHARS = string.ascii_uppercase + string.digits


def _generate_share_code() -> str:
    """Generate a random 6-char uppercase alphanumeric code."""
    return "".join(random.choices(_SHARE_CODE_CHARS, k=_SHARE_CODE_LENGTH))


class HanaShareStore:
    """HANA-backed store for conversation share links and members."""

    async def init_schema(self) -> None:
        """Create tables and indexes (idempotent)."""
        from ideation_agent.hana_store import _execute

        # --- CONVERSATION_SHARES table ---
        try:
            await _execute("""CREATE COLUMN TABLE CONVERSATION_SHARES (
                SHARE_ID        NVARCHAR(36)   PRIMARY KEY,
                CONVERSATION_ID NVARCHAR(36)   NOT NULL,
                SHARE_CODE      NVARCHAR(8)    NOT NULL UNIQUE,
                PERMISSION      NVARCHAR(10)   NOT NULL DEFAULT 'read',
                CREATED_BY_UID  NVARCHAR(256)  NOT NULL,
                CREATED_AT      TIMESTAMP      NOT NULL,
                REVOKED_AT      TIMESTAMP      DEFAULT NULL
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # --- CONVERSATION_SHARE_MEMBERS table ---
        try:
            await _execute("""CREATE COLUMN TABLE CONVERSATION_SHARE_MEMBERS (
                CONVERSATION_ID NVARCHAR(36)   NOT NULL,
                UID             NVARCHAR(256)  NOT NULL,
                EMAIL           NVARCHAR(320)  DEFAULT '',
                DISPLAY_NAME    NVARCHAR(256)  DEFAULT '',
                PERMISSION      NVARCHAR(10)   NOT NULL DEFAULT 'read',
                JOINED_VIA      NVARCHAR(36)   NOT NULL,
                JOINED_AT       TIMESTAMP      NOT NULL,
                PRIMARY KEY (CONVERSATION_ID, UID)
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # Indexes
        for sql in [
            "CREATE INDEX IX_SHARES_CONV ON CONVERSATION_SHARES(CONVERSATION_ID)",
            "CREATE INDEX IX_SHARES_CODE ON CONVERSATION_SHARES(SHARE_CODE)",
            "CREATE INDEX IX_SHARE_MEMBERS_UID ON CONVERSATION_SHARE_MEMBERS(UID)",
        ]:
            try:
                await _execute(sql)
            except Exception:
                pass

        # Migration: add CREATED_BY_NAME column if missing
        try:
            await _execute(
                "ALTER TABLE CONVERSATION_SHARES ADD (CREATED_BY_NAME NVARCHAR(256) DEFAULT '')"
            )
        except Exception:
            pass  # Column already exists

    # ── Share CRUD ──────────────────────────────────────────────────────────

    async def create_share(
        self,
        conversation_id: str,
        uid: str,
        permission: str = "read",
        creator_name: str = "",
    ) -> Dict[str, Any]:
        """Create a new share link for a conversation.

        Retries code generation on collision (max 5 attempts).
        Returns share metadata including the generated code.
        """
        from ideation_agent.hana_store import _execute

        share_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        for _ in range(5):
            share_code = _generate_share_code()
            try:
                await _execute(
                    """INSERT INTO CONVERSATION_SHARES
                    (SHARE_ID, CONVERSATION_ID, SHARE_CODE, PERMISSION, CREATED_BY_UID, CREATED_BY_NAME, CREATED_AT)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (share_id, conversation_id, share_code, permission, uid, creator_name, now),
                )
                break
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    continue
                raise
        else:
            raise RuntimeError("Failed to generate unique share code after 5 attempts")

        return {
            "share_id": share_id,
            "conversation_id": conversation_id,
            "share_code": share_code,
            "permission": permission,
            "created_at": now.isoformat(),
        }

    async def get_share_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Lookup a share by its code. Returns None if not found or revoked."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            """SELECT SHARE_ID, CONVERSATION_ID, SHARE_CODE, PERMISSION,
                      CREATED_BY_UID, CREATED_AT, REVOKED_AT
            FROM CONVERSATION_SHARES WHERE SHARE_CODE = ? AND REVOKED_AT IS NULL""",
            (code.upper(),),
        )
        if row and isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()
        return row

    async def get_shares_for_conversation(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get all active (non-revoked) share links for a conversation."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT SHARE_ID, SHARE_CODE, PERMISSION, CREATED_AT
            FROM CONVERSATION_SHARES
            WHERE CONVERSATION_ID = ? AND REVOKED_AT IS NULL
            ORDER BY CREATED_AT DESC""",
            (conversation_id,),
        )
        for row in rows:
            if isinstance(row.get("created_at"), datetime):
                row["created_at"] = row["created_at"].isoformat()
        return rows

    async def revoke_share(self, share_id: str, uid: str) -> bool:
        """Revoke a share link. Only the creator can revoke. Returns True on success."""
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT SHARE_ID FROM CONVERSATION_SHARES WHERE SHARE_ID = ? AND CREATED_BY_UID = ?",
            (share_id, uid),
        )
        if not row:
            return False

        now = datetime.now(timezone.utc)
        await _execute(
            "UPDATE CONVERSATION_SHARES SET REVOKED_AT = ? WHERE SHARE_ID = ?",
            (now, share_id),
        )
        return True

    # ── Membership ──────────────────────────────────────────────────────────

    async def join_share(
        self,
        conversation_id: str,
        uid: str,
        email: str = "",
        display_name: str = "",
        permission: str = "read",
        share_id: str = "",
    ) -> bool:
        """Join a shared conversation. Returns True on success, False if already member.

        If already a member with lower permission, upgrades to higher permission.
        """
        from ideation_agent.hana_store import _execute, _fetch_one

        now = datetime.now(timezone.utc)

        existing = await _fetch_one(
            "SELECT PERMISSION FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ? AND UID = ?",
            (conversation_id, uid),
        )

        if existing:
            # Upgrade permission if new access is higher
            if existing["permission"] == "read" and permission == "write":
                await _execute(
                    """UPDATE CONVERSATION_SHARE_MEMBERS
                    SET PERMISSION = 'write', JOINED_VIA = ?, JOINED_AT = ?
                    WHERE CONVERSATION_ID = ? AND UID = ?""",
                    (share_id, now, conversation_id, uid),
                )
                return True
            return False  # Already a member with same or higher permission

        try:
            await _execute(
                """INSERT INTO CONVERSATION_SHARE_MEMBERS
                (CONVERSATION_ID, UID, EMAIL, DISPLAY_NAME, PERMISSION, JOINED_VIA, JOINED_AT)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (conversation_id, uid, email, display_name, permission, share_id, now),
            )
            return True
        except Exception as exc:
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower() or "301" in str(exc).lower():
                return False
            raise

    async def get_member_permission(
        self, conversation_id: str, uid: str
    ) -> Optional[str]:
        """Get user's permission for a conversation (or None if not a member)."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT PERMISSION FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ? AND UID = ?",
            (conversation_id, uid),
        )
        return row["permission"] if row else None

    async def get_members(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all members of a shared conversation."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT UID, EMAIL, DISPLAY_NAME, PERMISSION, JOINED_AT
            FROM CONVERSATION_SHARE_MEMBERS
            WHERE CONVERSATION_ID = ?
            ORDER BY JOINED_AT ASC""",
            (conversation_id,),
        )
        for row in rows:
            if isinstance(row.get("joined_at"), datetime):
                row["joined_at"] = row["joined_at"].isoformat()
        return rows

    async def remove_member(self, conversation_id: str, uid: str) -> bool:
        """Remove a member from a shared conversation. Returns True if deleted."""
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT UID FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ? AND UID = ?",
            (conversation_id, uid),
        )
        if not row:
            return False
        await _execute(
            "DELETE FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ? AND UID = ?",
            (conversation_id, uid),
        )
        return True

    async def update_members_permission(
        self, conversation_id: str, permission: str
    ) -> None:
        """Update all members of a conversation to a new permission level."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "UPDATE CONVERSATION_SHARE_MEMBERS SET PERMISSION = ? WHERE CONVERSATION_ID = ?",
            (permission, conversation_id),
        )

    async def update_member_permission(
        self, conversation_id: str, uid: str, permission: str
    ) -> bool:
        """Update a single member's permission. Returns True if the member exists."""
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT UID FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ? AND UID = ?",
            (conversation_id, uid),
        )
        if not row:
            return False
        await _execute(
            "UPDATE CONVERSATION_SHARE_MEMBERS SET PERMISSION = ? WHERE CONVERSATION_ID = ? AND UID = ?",
            (permission, conversation_id, uid),
        )
        return True

    async def get_shared_conversation_ids(self, owner_uid: str) -> Dict[str, int]:
        """Get conversation IDs owned by the user that have shared members, with count."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT M.CONVERSATION_ID, COUNT(*) AS CNT
            FROM CONVERSATION_SHARE_MEMBERS M
            INNER JOIN CONVERSATIONS C ON C.ID = M.CONVERSATION_ID
            WHERE C.USER_ID = ?
            GROUP BY M.CONVERSATION_ID""",
            (owner_uid,),
        )
        return {row["conversation_id"]: row["cnt"] for row in rows}

    async def update_share_permission(
        self, conversation_id: str, permission: str
    ) -> None:
        """Update all active share links for a conversation to a new permission level."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "UPDATE CONVERSATION_SHARES SET PERMISSION = ? WHERE CONVERSATION_ID = ? AND REVOKED_AT IS NULL",
            (permission, conversation_id),
        )

    async def get_shared_conversations(self, uid: str, event_id: str | None = None) -> List[Dict[str, Any]]:
        """Get all conversations shared with a user (for sidebar listing).

        Args:
            event_id: If provided, only return shared conversations belonging to that event.
                      If None, returns all shared conversations (no event filter).
        """
        from ideation_agent.hana_store import _fetch_all

        if event_id:
            rows = await _fetch_all(
                """SELECT C.ID, C.TITLE, C.STATUS, C.MESSAGE_COUNT, C.EVENT_ID,
                          C.CREATED_AT, C.UPDATED_AT,
                          C.USER_ID AS OWNER_UID,
                          M.PERMISSION, M.JOINED_AT,
                          S.CREATED_BY_NAME AS OWNER_NAME
                FROM CONVERSATIONS C
                INNER JOIN CONVERSATION_SHARE_MEMBERS M ON C.ID = M.CONVERSATION_ID
                LEFT JOIN CONVERSATION_SHARES S ON S.CONVERSATION_ID = C.ID
                    AND S.CREATED_BY_NAME != '' AND S.REVOKED_AT IS NULL
                WHERE M.UID = ? AND C.EVENT_ID = ?
                ORDER BY C.UPDATED_AT DESC""",
                (uid, event_id),
            )
        else:
            rows = await _fetch_all(
                """SELECT C.ID, C.TITLE, C.STATUS, C.MESSAGE_COUNT, C.EVENT_ID,
                          C.CREATED_AT, C.UPDATED_AT,
                          C.USER_ID AS OWNER_UID,
                          M.PERMISSION, M.JOINED_AT,
                          S.CREATED_BY_NAME AS OWNER_NAME
                FROM CONVERSATIONS C
                INNER JOIN CONVERSATION_SHARE_MEMBERS M ON C.ID = M.CONVERSATION_ID
                LEFT JOIN CONVERSATION_SHARES S ON S.CONVERSATION_ID = C.ID
                    AND S.CREATED_BY_NAME != '' AND S.REVOKED_AT IS NULL
                WHERE M.UID = ?
                ORDER BY C.UPDATED_AT DESC""",
                (uid,),
            )
        # Deduplicate rows (LEFT JOIN on shares may produce duplicates)
        seen = set()
        unique_rows = []
        for row in rows:
            if row["id"] not in seen:
                seen.add(row["id"])
                unique_rows.append(row)
        for row in unique_rows:
            for key in ("created_at", "updated_at", "joined_at"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
        return unique_rows

    async def delete_shares_for_conversation(self, conversation_id: str) -> None:
        """Delete all shares and members for a conversation (cascade on delete)."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "DELETE FROM CONVERSATION_SHARE_MEMBERS WHERE CONVERSATION_ID = ?",
            (conversation_id,),
        )
        await _execute(
            "DELETE FROM CONVERSATION_SHARES WHERE CONVERSATION_ID = ?",
            (conversation_id,),
        )


# Module-level singleton
_store: Optional[HanaShareStore] = None
_store_lock = asyncio.Lock()


async def get_share_store() -> HanaShareStore:
    """Get or create the singleton share store (initializes schema on first call)."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaShareStore()
        try:
            await store.init_schema()
            logger.info("HANA share store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA shares schema: %s", exc)
            raise
        _store = store
        return _store
