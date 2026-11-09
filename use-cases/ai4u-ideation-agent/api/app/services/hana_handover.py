"""HANA persistence layer for the Handover Workspace.

Follows the same async singleton pattern as other HANA services in this
project. Imports shared helpers from ``ideation_agent.hana_store``.

Tables:
    HANDOVER            — Main handover records (one per workspace)
    HANDOVER_FILES      — File attachments with zone categorization
    HANDOVER_CONTEXT    — Linked context sources (ideation conversations,
                          transcripts, documents)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HanaHandoverStore:
    """Async HANA store for handover workspace records."""

    async def init_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        from ideation_agent.hana_store import _execute

        table_ddls = [
            """
            CREATE COLUMN TABLE HANDOVER (
                ID                      NVARCHAR(36) PRIMARY KEY,
                USER_ID                 NVARCHAR(256) NOT NULL,
                STATUS                  NVARCHAR(20) DEFAULT 'draft',
                CUSTOMER                NVARCHAR(500),
                USECASE_TITLE           NVARCHAR(500),
                PRIORITY                NVARCHAR(20),
                PAYLOAD_JSON            NCLOB,
                LINKED_CONVERSATION_ID  NVARCHAR(36),
                CREATED_AT              TIMESTAMP,
                UPDATED_AT              TIMESTAMP,
                SUBMITTED_AT            TIMESTAMP
            )
            """,
            """
            CREATE COLUMN TABLE HANDOVER_FILES (
                FILE_ID         NVARCHAR(36) PRIMARY KEY,
                HANDOVER_ID     NVARCHAR(36) NOT NULL,
                ZONE            NVARCHAR(32) NOT NULL,
                FILENAME        NVARCHAR(500),
                CONTENT_TYPE    NVARCHAR(200),
                SIZE            BIGINT,
                DATA            BLOB,
                CREATED_AT      TIMESTAMP
            )
            """,
            """
            CREATE COLUMN TABLE HANDOVER_CONTEXT (
                ID              NVARCHAR(36) PRIMARY KEY,
                HANDOVER_ID     NVARCHAR(36) NOT NULL,
                SOURCE_TYPE     NVARCHAR(20) NOT NULL,
                SOURCE_REF      NCLOB,
                EXTRACTED_FIELDS NCLOB,
                CREATED_AT      TIMESTAMP
            )
            """,
        ]

        for ddl in table_ddls:
            try:
                await _execute(ddl)
            except Exception as exc:
                err_str = str(exc).lower()
                if (
                    "already exists" in err_str
                    or "duplicate" in err_str
                    or "288" in err_str
                ):
                    continue
                raise

        # Indexes
        index_ddls = [
            "CREATE INDEX IDX_HANDOVER_USER ON HANDOVER (USER_ID)",
            "CREATE INDEX IDX_HANDOVER_STATUS ON HANDOVER (STATUS)",
            "CREATE INDEX IDX_HANDOVER_LINKED ON HANDOVER (LINKED_CONVERSATION_ID)",
            "CREATE INDEX IDX_HANDOVER_FILES_HID ON HANDOVER_FILES (HANDOVER_ID)",
            "CREATE INDEX IDX_HANDOVER_CONTEXT_HID ON HANDOVER_CONTEXT (HANDOVER_ID)",
        ]
        for ddl in index_ddls:
            try:
                await _execute(ddl)
            except Exception:
                pass  # Index may already exist

        # Schema migrations (idempotent column additions / type changes)
        migrations = [
            "ALTER TABLE HANDOVER ADD (AGENT_MESSAGES_JSON NCLOB DEFAULT NULL)",
            "ALTER TABLE HANDOVER_CONTEXT ALTER (SOURCE_REF NCLOB)",
            "ALTER TABLE HANDOVER_CONTEXT ADD (CONTENT_TEXT NCLOB DEFAULT NULL)",
        ]
        for ddl in migrations:
            try:
                await _execute(ddl)
            except Exception:
                pass  # Column already exists

    # ─── Handover CRUD ──────────────────────────────────────────────────

    async def create(self, user_id: str, payload: dict, linked_conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new handover workspace. Returns the created record dict."""
        from ideation_agent.hana_store import _execute

        handover_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        customer = payload.get("customer")
        usecase_title = payload.get("usecase_title")
        priority = payload.get("priority")
        payload_json = json.dumps(payload)

        await _execute(
            "INSERT INTO HANDOVER "
            "(ID, USER_ID, STATUS, CUSTOMER, USECASE_TITLE, PRIORITY, "
            "PAYLOAD_JSON, LINKED_CONVERSATION_ID, CREATED_AT, UPDATED_AT, SUBMITTED_AT) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                handover_id, user_id, "draft",
                customer, usecase_title, priority,
                payload_json, linked_conversation_id,
                now, now, None,
            ),
        )

        return {
            "id": handover_id,
            "user_id": user_id,
            "status": "draft",
            "customer": customer,
            "usecase_title": usecase_title,
            "priority": priority,
            "linked_conversation_id": linked_conversation_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "submitted_at": None,
        }

    async def get(self, handover_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single handover by ID. Returns None if not found."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT ID, USER_ID, STATUS, CUSTOMER, USECASE_TITLE, PRIORITY, "
            "PAYLOAD_JSON, LINKED_CONVERSATION_ID, CREATED_AT, UPDATED_AT, SUBMITTED_AT "
            "FROM HANDOVER WHERE ID = ?",
            (handover_id,),
        )
        if row is None:
            return None
        return self._row_to_dict(row)

    async def list_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """List all handover workspaces for a user, newest first."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            "SELECT ID, USER_ID, STATUS, CUSTOMER, USECASE_TITLE, PRIORITY, "
            "PAYLOAD_JSON, LINKED_CONVERSATION_ID, CREATED_AT, UPDATED_AT, SUBMITTED_AT "
            "FROM HANDOVER WHERE USER_ID = ? ORDER BY UPDATED_AT DESC",
            (user_id,),
        )
        return [self._row_to_dict(row) for row in rows]

    async def update_payload(self, handover_id: str, payload: dict) -> bool:
        """Merge-update the payload JSON and promoted columns. Returns True if row existed."""
        from ideation_agent.hana_store import _execute, _fetch_one

        # Fetch current payload to merge
        existing = await _fetch_one(
            "SELECT PAYLOAD_JSON FROM HANDOVER WHERE ID = ?",
            (handover_id,),
        )
        if existing is None:
            return False

        current_payload = self._parse_payload(existing.get("payload_json"))
        current_payload.update(payload)

        now = datetime.now(timezone.utc)
        customer = current_payload.get("customer")
        usecase_title = current_payload.get("usecase_title")
        priority = current_payload.get("priority")
        payload_json = json.dumps(current_payload)

        await _execute(
            "UPDATE HANDOVER SET "
            "CUSTOMER = ?, USECASE_TITLE = ?, PRIORITY = ?, PAYLOAD_JSON = ?, "
            "STATUS = CASE WHEN STATUS = 'draft' THEN 'in_progress' ELSE STATUS END, "
            "UPDATED_AT = ? "
            "WHERE ID = ?",
            (customer, usecase_title, priority, payload_json, now, handover_id),
        )
        return True

    async def mark_complete(self, handover_id: str) -> bool:
        """Mark a handover as validated/complete. Returns True if row existed."""
        from ideation_agent.hana_store import _execute

        now = datetime.now(timezone.utc)
        await _execute(
            "UPDATE HANDOVER SET STATUS = 'validated', SUBMITTED_AT = ?, UPDATED_AT = ? "
            "WHERE ID = ?",
            (now, now, handover_id),
        )
        return True

    async def delete(self, handover_id: str) -> bool:
        """Delete a handover and its associated files and context sources.

        Uses a transaction so partial failures roll back cleanly.
        """
        from ideation_agent.hana_store import _execute_transaction

        await _execute_transaction([
            ("DELETE FROM HANDOVER_FILES WHERE HANDOVER_ID = ?", (handover_id,)),
            ("DELETE FROM HANDOVER_CONTEXT WHERE HANDOVER_ID = ?", (handover_id,)),
            ("DELETE FROM HANDOVER WHERE ID = ?", (handover_id,)),
        ])
        return True

    async def get_status_for_conversation(self, conversation_id: str) -> Optional[str]:
        """Get handover status for a linked ideation conversation.

        Used by canvas tiles to show the 'validated' badge.
        """
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT TOP 1 STATUS FROM HANDOVER WHERE LINKED_CONVERSATION_ID = ? "
            "ORDER BY UPDATED_AT DESC",
            (conversation_id,),
        )
        if row is None:
            return None
        return row.get("status")

    # ─── Files ──────────────────────────────────────────────────────────

    async def add_file(
        self,
        handover_id: str,
        zone: str,
        filename: str,
        content_type: str,
        size: int,
        data: bytes,
    ) -> Dict[str, Any]:
        """Store an uploaded file as a BLOB linked to the handover."""
        from ideation_agent.hana_store import _execute

        file_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        await _execute(
            "INSERT INTO HANDOVER_FILES "
            "(FILE_ID, HANDOVER_ID, ZONE, FILENAME, CONTENT_TYPE, SIZE, DATA, CREATED_AT) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, handover_id, zone, filename, content_type, size, data, now),
        )

        return {
            "file_id": file_id,
            "filename": filename,
            "size": size,
            "zone": zone,
            "content_type": content_type,
            "created_at": now.isoformat(),
        }

    async def list_files(self, handover_id: str) -> List[Dict[str, Any]]:
        """List file metadata (no BLOB data) for a handover, grouped by zone."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            "SELECT FILE_ID, FILENAME, CONTENT_TYPE, SIZE, ZONE, CREATED_AT "
            "FROM HANDOVER_FILES WHERE HANDOVER_ID = ? ORDER BY ZONE, CREATED_AT",
            (handover_id,),
        )
        return [
            {
                "file_id": row["file_id"],
                "filename": row["filename"],
                "content_type": row["content_type"],
                "size": row["size"],
                "zone": row["zone"],
                "created_at": self._iso(row.get("created_at")),
            }
            for row in rows
        ]

    async def delete_file(self, file_id: str, handover_id: str) -> bool:
        """Delete a single file by ID (scoped to handover for security)."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "DELETE FROM HANDOVER_FILES WHERE FILE_ID = ? AND HANDOVER_ID = ?",
            (file_id, handover_id),
        )
        return True

    async def get_file_content(
        self, file_id: str, handover_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single file's BLOB data and metadata (scoped to handover)."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT FILE_ID, FILENAME, CONTENT_TYPE, SIZE, DATA "
            "FROM HANDOVER_FILES WHERE FILE_ID = ? AND HANDOVER_ID = ?",
            (file_id, handover_id),
        )
        if row is None:
            return None

        # HANA LOB objects may need .read() to materialize; _fetch_one reads
        # inside the cursor scope so the data should already be bytes, but
        # handle the LOB proxy case defensively.
        data = row.get("data")
        read_fn = getattr(data, "read", None)
        if callable(read_fn):
            data = read_fn()

        return {
            "file_id": row["file_id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "size": row["size"],
            "data": data,
        }

    # ─── Context Sources ────────────────────────────────────────────────

    async def add_context(
        self,
        handover_id: str,
        source_type: str,
        source_ref: Optional[str] = None,
        extracted_fields: Optional[dict] = None,
        content_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Link a context source (ideation conversation, document, etc.)."""
        from ideation_agent.hana_store import _execute

        context_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        extracted_json = json.dumps(extracted_fields) if extracted_fields is not None else None

        await _execute(
            "INSERT INTO HANDOVER_CONTEXT "
            "(ID, HANDOVER_ID, SOURCE_TYPE, SOURCE_REF, EXTRACTED_FIELDS, CONTENT_TEXT, CREATED_AT) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (context_id, handover_id, source_type, source_ref, extracted_json, content_text, now),
        )

        return {
            "id": context_id,
            "source_type": source_type,
            "source_ref": source_ref,
            "extracted_fields": extracted_fields,
            "content_text": content_text,
            "created_at": now.isoformat(),
        }

    async def list_context(self, handover_id: str) -> List[Dict[str, Any]]:
        """List context sources for a handover."""
        from ideation_agent.hana_store import _fetch_all

        rows = await _fetch_all(
            "SELECT ID, SOURCE_TYPE, SOURCE_REF, EXTRACTED_FIELDS, CONTENT_TEXT, CREATED_AT "
            "FROM HANDOVER_CONTEXT WHERE HANDOVER_ID = ? ORDER BY CREATED_AT",
            (handover_id,),
        )
        return [
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_ref": self._read_lob_text(row.get("source_ref")),
                "extracted_fields": self._parse_nullable_json(row.get("extracted_fields")),
                "content_text": self._read_lob_text(row.get("content_text")),
                "created_at": self._iso(row.get("created_at")),
            }
            for row in rows
        ]

    async def delete_context(self, context_id: str, handover_id: str) -> bool:
        """Remove a context source link (scoped to handover for security)."""
        from ideation_agent.hana_store import _execute

        await _execute(
            "DELETE FROM HANDOVER_CONTEXT WHERE ID = ? AND HANDOVER_ID = ?",
            (context_id, handover_id),
        )
        return True

    async def update_context_extracted(
        self, context_id: str, extracted_fields: dict, handover_id: str
    ) -> None:
        """Update the cached extracted_fields for a context source."""
        from ideation_agent.hana_store import _execute

        extracted_json = json.dumps(extracted_fields)
        await _execute(
            "UPDATE HANDOVER_CONTEXT SET EXTRACTED_FIELDS = ? "
            "WHERE ID = ? AND HANDOVER_ID = ?",
            (extracted_json, context_id, handover_id),
        )

    # ─── Agent Messages ────────────────────────────────────────────────

    async def get_agent_messages(self, handover_id: str) -> List[Dict[str, Any]]:
        """Get stored agent chat messages for a handover."""
        from ideation_agent.hana_store import _fetch_one

        row = await _fetch_one(
            "SELECT AGENT_MESSAGES_JSON FROM HANDOVER WHERE ID = ?",
            (handover_id,),
        )
        if row is None:
            return []
        raw = row.get("agent_messages_json")
        if raw is None:
            return []
        # Handle HANA LOB objects (can only be read once)
        read = getattr(raw, "read", None)
        if callable(read):
            raw = read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                result = json.loads(raw)
                return result if isinstance(result, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    async def save_agent_messages(
        self, handover_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Persist agent chat messages (full replace, capped at 100)."""
        from ideation_agent.hana_store import _execute

        _MAX_AGENT_MESSAGES = 100
        if len(messages) > _MAX_AGENT_MESSAGES:
            messages = messages[-_MAX_AGENT_MESSAGES:]

        messages_json = json.dumps(messages)
        await _execute(
            "UPDATE HANDOVER SET AGENT_MESSAGES_JSON = ?, UPDATED_AT = ? WHERE ID = ?",
            (messages_json, datetime.now(timezone.utc), handover_id),
        )

    # ─── Helpers ────────────────────────────────────────────────────────

    def _row_to_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a raw HANA row to a response dict."""
        payload = self._parse_payload(row.get("payload_json"))
        return {
            "id": row["id"],
            "user_id": row.get("user_id"),
            "status": row.get("status", "draft"),
            "customer": row.get("customer"),
            "usecase_title": row.get("usecase_title"),
            "priority": row.get("priority"),
            "payload": payload,
            "linked_conversation_id": row.get("linked_conversation_id"),
            "created_at": self._iso(row.get("created_at")),
            "updated_at": self._iso(row.get("updated_at")),
            "submitted_at": self._iso(row.get("submitted_at")),
        }

    def _parse_payload(self, raw: Any) -> dict:
        """Parse a JSON payload from HANA (handles LOB objects, bytes, strings)."""
        if raw is None:
            return {}
        # hdbcli may return a LOB object with .read()
        read = getattr(raw, "read", None)
        if callable(read):
            raw = read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def _parse_nullable_json(self, raw: Any) -> Optional[dict]:
        """Parse a JSON column that may be NULL, preserving the NULL distinction.

        Returns None for DB NULL (not yet populated), dict for valid JSON.
        Unlike ``_parse_payload``, does NOT coerce NULL to ``{}``.
        """
        if raw is None:
            return None
        # hdbcli may return a LOB object with .read()
        read = getattr(raw, "read", None)
        if callable(read):
            raw = read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def _iso(self, value: Any) -> Optional[str]:
        """Serialize a datetime to ISO 8601 string."""
        if value is None:
            return None
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            return isoformat()
        return str(value) if value else None

    def _read_lob_text(self, raw: Any) -> Optional[str]:
        """Read an NCLOB column value to a plain Python string."""
        if raw is None:
            return None
        read = getattr(raw, "read", None)
        if callable(read):
            raw = read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return str(raw) if raw is not None else None


# ─── Module-level singleton ─────────────────────────────────────────────────

_store: Optional[HanaHandoverStore] = None
_store_lock = asyncio.Lock()


async def get_handover_store() -> HanaHandoverStore:
    """Get or create the singleton HanaHandoverStore instance."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaHandoverStore()
        await store.init_schema()
        _store = store
        return _store
