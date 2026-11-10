"""HANA Cloud persistence layer for Events (scoped leaderboards).

Stores event metadata and membership. Follows the same hdbcli + asyncio.to_thread
pattern as hana_submissions.py and hana_store.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import string
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JOIN_CODE_LENGTH = 6
_JOIN_CODE_CHARS = string.ascii_uppercase + string.digits


def _generate_join_code() -> str:
    """Generate a random 6-char uppercase alphanumeric code."""
    return "".join(random.choices(_JOIN_CODE_CHARS, k=_JOIN_CODE_LENGTH))


class HanaEventStore:
    """HANA-backed store for Events and EventMembers."""

    async def init_schema(self) -> None:
        """Create tables and indexes (idempotent)."""
        from ideation_agent.hana_store import _execute

        # --- EVENTS table ---
        try:
            await _execute("""CREATE COLUMN TABLE EVENTS (
                EVENT_ID         NVARCHAR(36)  PRIMARY KEY,
                JOIN_CODE        NVARCHAR(8)   NOT NULL UNIQUE,
                NAME             NVARCHAR(256) NOT NULL,
                CREATED_BY_UID   NVARCHAR(256) NOT NULL,
                CREATED_BY_EMAIL NVARCHAR(320) NOT NULL,
                STATUS           NVARCHAR(20)  DEFAULT 'active',
                CREATED_AT       TIMESTAMP     NOT NULL
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # --- EVENT_MEMBERS table ---
        try:
            await _execute("""CREATE COLUMN TABLE EVENT_MEMBERS (
                EVENT_ID    NVARCHAR(36)  NOT NULL,
                UID         NVARCHAR(256) NOT NULL,
                EMAIL       NVARCHAR(320) NOT NULL,
                GIVEN_NAME  NVARCHAR(256) DEFAULT '',
                FAMILY_NAME NVARCHAR(256) DEFAULT '',
                ROLE        NVARCHAR(20)  DEFAULT 'member',
                JOINED_AT   TIMESTAMP     NOT NULL,
                PRIMARY KEY (EVENT_ID, UID)
            )""")
        except Exception as exc:
            err_str = str(exc).lower()
            if "already exists" not in err_str and "duplicate" not in err_str and "288" not in err_str:
                raise

        # Indexes
        for sql in [
            "CREATE INDEX IX_EVENT_CODE ON EVENTS(JOIN_CODE)",
            "CREATE INDEX IX_EVENT_MEMBERS_UID ON EVENT_MEMBERS(UID)",
        ]:
            try:
                await _execute(sql)
            except Exception:
                pass

    # ── Event CRUD ──────────────────────────────────────────────────────────

    async def create_event(
        self,
        name: str,
        uid: str,
        email: str,
        given_name: str = "",
        family_name: str = "",
    ) -> Dict[str, Any]:
        """Create a new event and auto-join the creator as admin.

        Retries code generation on collision (max 5 attempts).
        """
        from ideation_agent.hana_store import _execute

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Generate unique join code (retry on collision)
        for _ in range(5):
            join_code = _generate_join_code()
            try:
                await _execute(
                    """INSERT INTO EVENTS (EVENT_ID, JOIN_CODE, NAME, CREATED_BY_UID, CREATED_BY_EMAIL, STATUS, CREATED_AT)
                    VALUES (?, ?, ?, ?, ?, 'active', ?)""",
                    (event_id, join_code, name, uid, email, now),
                )
                break
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    continue
                raise
        else:
            raise RuntimeError("Failed to generate unique join code after 5 attempts")

        # Auto-join creator as admin
        await _execute(
            """INSERT INTO EVENT_MEMBERS (EVENT_ID, UID, EMAIL, GIVEN_NAME, FAMILY_NAME, ROLE, JOINED_AT)
            VALUES (?, ?, ?, ?, ?, 'admin', ?)""",
            (event_id, uid, email, given_name, family_name, now),
        )

        return {
            "event_id": event_id,
            "join_code": join_code,
            "name": name,
            "status": "active",
            "created_at": now.isoformat(),
        }

    async def get_event_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Lookup event by join code (any status)."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            """SELECT EVENT_ID, JOIN_CODE, NAME, CREATED_BY_UID, CREATED_BY_EMAIL, STATUS, CREATED_AT
            FROM EVENTS WHERE JOIN_CODE = ?""",
            (code.upper(),),
        )
        if row and isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()
        return row

    async def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Lookup event by primary key."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            """SELECT EVENT_ID, JOIN_CODE, NAME, CREATED_BY_UID, CREATED_BY_EMAIL, STATUS, CREATED_AT
            FROM EVENTS WHERE EVENT_ID = ?""",
            (event_id,),
        )
        if row and isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()
        return row

    # ── Membership ──────────────────────────────────────────────────────────

    async def join_event(
        self,
        event_id: str,
        uid: str,
        email: str,
        given_name: str = "",
        family_name: str = "",
    ) -> bool:
        """Join an event as member. Returns True on success, False if already member."""
        from ideation_agent.hana_store import _execute

        now = datetime.now(timezone.utc)
        try:
            await _execute(
                """INSERT INTO EVENT_MEMBERS (EVENT_ID, UID, EMAIL, GIVEN_NAME, FAMILY_NAME, ROLE, JOINED_AT)
                VALUES (?, ?, ?, ?, ?, 'member', ?)""",
                (event_id, uid, email, given_name, family_name, now),
            )
            return True
        except Exception as exc:
            # Already a member (PK violation)
            if "unique" in str(exc).lower() or "duplicate" in str(exc).lower() or "301" in str(exc).lower():
                return False
            raise

    async def leave_event(self, event_id: str, uid: str) -> bool:
        """Remove a member from an event. Returns True if row deleted."""
        from ideation_agent.hana_store import _execute, _fetch_one

        row = await _fetch_one(
            "SELECT UID FROM EVENT_MEMBERS WHERE EVENT_ID = ? AND UID = ?",
            (event_id, uid),
        )
        if not row:
            return False
        await _execute(
            "DELETE FROM EVENT_MEMBERS WHERE EVENT_ID = ? AND UID = ?",
            (event_id, uid),
        )
        return True

    async def is_member(self, event_id: str, uid: str) -> bool:
        """Check if user is a member of the event."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT UID FROM EVENT_MEMBERS WHERE EVENT_ID = ? AND UID = ?",
            (event_id, uid),
        )
        return row is not None

    async def get_member_role(self, event_id: str, uid: str) -> Optional[str]:
        """Get user's role in an event (or None if not member)."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT ROLE FROM EVENT_MEMBERS WHERE EVENT_ID = ? AND UID = ?",
            (event_id, uid),
        )
        return row["role"] if row else None

    async def count_event_members(self, event_id: str) -> int:
        """Count members in an event."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT COUNT(*) AS CNT FROM EVENT_MEMBERS WHERE EVENT_ID = ?",
            (event_id,),
        )
        return int(row["cnt"]) if row else 0

    async def get_event_members(self, event_id: str) -> List[Dict[str, Any]]:
        """Get all members with roles."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT UID, EMAIL, GIVEN_NAME, FAMILY_NAME, ROLE, JOINED_AT
            FROM EVENT_MEMBERS WHERE EVENT_ID = ? ORDER BY JOINED_AT ASC""",
            (event_id,),
        )
        for row in rows:
            if isinstance(row.get("joined_at"), datetime):
                row["joined_at"] = row["joined_at"].isoformat()
        return rows

    # ── Event status ────────────────────────────────────────────────────────

    async def set_event_status(self, event_id: str, uid: str, status: str) -> bool:
        """Toggle event status (admin only). Returns True on success."""
        from ideation_agent.hana_store import _execute

        role = await self.get_member_role(event_id, uid)
        if role != "admin":
            return False

        await _execute(
            "UPDATE EVENTS SET STATUS = ? WHERE EVENT_ID = ?",
            (status, event_id),
        )
        return True

    # ── Role management ─────────────────────────────────────────────────────

    async def set_member_role(
        self, event_id: str, admin_uid: str, target_uid: str, role: str
    ) -> bool:
        """Promote/demote a member (admin only). Returns True on success."""
        from ideation_agent.hana_store import _execute

        # Verify caller is admin
        caller_role = await self.get_member_role(event_id, admin_uid)
        if caller_role != "admin":
            return False

        # Target must be a member
        target_role = await self.get_member_role(event_id, target_uid)
        if target_role is None:
            return False

        await _execute(
            "UPDATE EVENT_MEMBERS SET ROLE = ? WHERE EVENT_ID = ? AND UID = ?",
            (role, event_id, target_uid),
        )
        return True

    # ── User queries ────────────────────────────────────────────────────────

    async def get_user_events(self, uid: str) -> List[Dict[str, Any]]:
        """Get all events a user is member of (active + ended)."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            """SELECT E.EVENT_ID, E.JOIN_CODE, E.NAME, E.STATUS, E.CREATED_AT,
                      E.CREATED_BY_UID, E.CREATED_BY_EMAIL,
                      M.ROLE, M.JOINED_AT
            FROM EVENTS E
            INNER JOIN EVENT_MEMBERS M ON E.EVENT_ID = M.EVENT_ID
            WHERE M.UID = ?
            ORDER BY E.CREATED_AT DESC""",
            (uid,),
        )
        for row in rows:
            for key in ("created_at", "joined_at"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
        return rows

    # ── Event leaderboard ───────────────────────────────────────────────────

    async def get_event_leaderboard(
        self, event_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get ranked submissions for an event.

        Deduplicates by CONTEXT_ID: only the highest-scoring submission per
        conversation is kept, attributed to the conversation owner.
        Non-member editors can submit (their entry is attributed to the owner),
        but only entries where the final attributed UID is an event member survive.
        """
        from ideation_agent.hana_store import _fetch_all

        # Fetch more to account for duplicates that collapse
        fetch_limit = limit * 3

        # No JOIN on EVENT_MEMBERS here — include ALL submissions for this event,
        # even from non-member editors. Membership is checked after owner attribution.
        rows = await _fetch_all(
            f"""SELECT S.SUBMISSION_ID, S.UID, S.EMAIL, S.INDUSTRY, S.TITLE,
                       S.PROBLEM, S.SOLUTION, S.APP_TYPE, S.EVALUATOR_FEEDBACK,
                       S.EVALUATOR_DETAILED_ANALYSIS, S.EVALUATOR_SUGGESTIONS_JSON,
                       S.APPROPRIATENESS_LEVEL, S.SCORE, S.RATING, S.SUBMITTED_AT,
                       S.CONTEXT_ID
                FROM IDEATION_SUBMISSIONS S
                WHERE S.EVENT_ID = ? AND S.IS_ON_LEADERBOARD = TRUE
                ORDER BY S.SCORE DESC, S.SUBMITTED_AT DESC
                LIMIT {int(fetch_limit)}""",
            (event_id,),
        )
        for row in rows:
            if isinstance(row.get("submitted_at"), datetime):
                row["submitted_at"] = row["submitted_at"].isoformat()
            if row.get("evaluator_suggestions_json"):
                try:
                    row["evaluator_suggestions"] = json.loads(row["evaluator_suggestions_json"])
                except (json.JSONDecodeError, TypeError):
                    row["evaluator_suggestions"] = []
            else:
                row["evaluator_suggestions"] = []

        # Deduplicate by CONTEXT_ID — keep highest score per conversation
        best_by_context: dict = {}
        for row in rows:
            ctx = row.get("context_id", "")
            if not ctx:
                best_by_context[row.get("submission_id", "")] = row
                continue
            existing = best_by_context.get(ctx)
            if existing is None or int(row.get("score", 0)) > int(existing.get("score", 0)):
                best_by_context[ctx] = row

        # Batch-fetch conversation owners to attribute entries
        context_ids = [
            ctx for ctx in best_by_context.keys()
            if ctx and ctx != best_by_context[ctx].get("submission_id", "")
        ]
        owner_map: dict = {}  # context_id -> owner UID
        if context_ids:
            placeholders = ", ".join(["?"] * len(context_ids))
            owner_rows = await _fetch_all(
                f"SELECT ID, USER_ID FROM CONVERSATIONS WHERE ID IN ({placeholders})",
                tuple(context_ids),
            )
            owner_map = {r["id"]: r["user_id"] for r in owner_rows}

        # Determine which UIDs will be the final attributed owners
        # (owner_map values for context-based entries, original uid for non-context entries)
        all_attributed_uids = set()
        for row in best_by_context.values():
            ctx = row.get("context_id", "")
            owner_uid = owner_map.get(ctx)
            all_attributed_uids.add(owner_uid if owner_uid else row.get("uid", ""))

        # Batch-check event membership for all attributed UIDs
        member_uids: set = set()
        uids_to_check = [uid for uid in all_attributed_uids if uid]
        if uids_to_check:
            placeholders = ", ".join(["?"] * len(uids_to_check))
            member_rows = await _fetch_all(
                f"""SELECT UID, EMAIL, GIVEN_NAME, FAMILY_NAME
                    FROM EVENT_MEMBERS WHERE EVENT_ID = ? AND UID IN ({placeholders})""",
                (event_id, *uids_to_check),
            )
            member_uids = {r["uid"] for r in member_rows}
        else:
            member_rows = []

        # Build UID → display info from member_rows
        uid_info: dict = {}
        for mr in member_rows:
            uid_info[mr["uid"]] = {
                "email": mr.get("email", ""),
                "given_name": mr.get("given_name", ""),
                "family_name": mr.get("family_name", ""),
            }

        # Override attribution to conversation owner and filter by membership
        deduped = []
        for row in best_by_context.values():
            ctx = row.get("context_id", "")
            owner_uid = owner_map.get(ctx)
            attributed_uid = owner_uid if owner_uid else row.get("uid", "")

            # Only include entries where the attributed user is an event member
            if attributed_uid not in member_uids:
                continue

            # Override display info to the owner
            if owner_uid and owner_uid in uid_info:
                info = uid_info[owner_uid]
                row["uid"] = owner_uid
                row["email"] = info["email"]
                row["given_name"] = info["given_name"]
                row["family_name"] = info["family_name"]
            elif attributed_uid in uid_info:
                info = uid_info[attributed_uid]
                row["given_name"] = info.get("given_name", "")
                row["family_name"] = info.get("family_name", "")

            deduped.append(row)

        # Re-sort and limit
        deduped.sort(key=lambda r: (-int(r.get("score", 0)), r.get("submitted_at", "")))
        return deduped[:limit]


# Module-level singleton
_store: Optional[HanaEventStore] = None
_store_lock = asyncio.Lock()


async def get_event_store() -> HanaEventStore:
    """Get or create the singleton event store (initializes schema on first call)."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaEventStore()
        try:
            await store.init_schema()
            logger.info("HANA event store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA events schema: %s", exc)
            raise
        _store = store
        return _store
