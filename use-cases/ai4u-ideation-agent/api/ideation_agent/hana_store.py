"""HANA Cloud persistence layer for ideation agent conversations.

Stores conversation metadata and messages in SAP HANA Cloud.
All hdbcli calls are wrapped with asyncio.to_thread() for async compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONN = None
_CONN_LOCK = asyncio.Lock()


def _get_hana_credentials() -> Dict[str, Any]:
    """Read HANA credentials from env vars (preferred) or VCAP_SERVICES (CF fallback).

    Explicit HANA_HOST env var takes priority over VCAP_SERVICES so that
    production deployments can use a technical user with DDL privileges
    (HDI shared containers block direct CREATE TABLE).
    """
    if os.getenv("HANA_HOST"):
        return {
            "host": os.getenv("HANA_HOST", ""),
            "port": int(os.getenv("HANA_PORT", "443")),
            "user": os.getenv("HANA_USER", ""),
            "password": os.getenv("HANA_PASSWORD", ""),
            "schema": os.getenv("HANA_SCHEMA", ""),
            "encrypt": os.getenv("HANA_ENCRYPT", "true").lower() in ("true", "1", "yes"),
        }

    vcap = os.getenv("VCAP_SERVICES", "")
    if vcap:
        try:
            services = json.loads(vcap)
            hana_creds = services.get("hana", [{}])[0].get("credentials", {})
            if hana_creds:
                return {
                    "host": hana_creds.get("host", ""),
                    "port": int(hana_creds.get("port", 443)),
                    "user": hana_creds.get("hdi_user", hana_creds.get("user", "")),
                    "password": hana_creds.get("hdi_password", hana_creds.get("password", "")),
                    "schema": hana_creds.get("schema", ""),
                    "encrypt": True,
                }
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return {
        "host": "",
        "port": 443,
        "user": "",
        "password": "",
        "schema": "",
        "encrypt": True,
    }


def _connect_sync():
    """Create a synchronous HANA connection."""
    from hdbcli import dbapi

    creds = _get_hana_credentials()
    if not creds["host"] or not creds["user"]:
        raise RuntimeError(
            "HANA credentials not configured. Set HANA_HOST, HANA_USER, HANA_PASSWORD "
            "or bind a HANA service in Cloud Foundry."
        )
    conn = dbapi.connect(
        address=creds["host"],
        port=creds["port"],
        user=creds["user"],
        password=creds["password"],
        encrypt=creds["encrypt"],
        sslValidateCertificate=False,
        autocommit=True,
    )
    if creds.get("schema"):
        cursor = conn.cursor()
        cursor.execute(f'SET SCHEMA "{creds["schema"]}"')
        cursor.close()
    return conn


async def _get_connection():
    """Get or create the HANA connection (singleton with reconnect)."""
    global _CONN
    async with _CONN_LOCK:
        if _CONN is not None:
            try:
                await asyncio.to_thread(_CONN.isconnected)
                return _CONN
            except Exception:
                _CONN = None

        _CONN = await asyncio.to_thread(_connect_sync)
        return _CONN


async def _execute(sql: str, params: tuple = ()) -> None:
    """Execute a SQL statement (no result)."""
    conn = await _get_connection()

    def _run():
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            cursor.close()

    await asyncio.to_thread(_run)


async def _execute_transaction(statements: list) -> None:
    """Execute multiple SQL statements in a single transaction.

    Args:
        statements: List of (sql, params) tuples.

    Rolls back on any failure so partial mutations never commit.
    """
    conn = await _get_connection()

    def _run():
        conn.setautocommit(False)
        try:
            for sql, params in statements:
                cursor = conn.cursor()
                try:
                    cursor.execute(sql, params)
                finally:
                    cursor.close()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.setautocommit(True)

    await asyncio.to_thread(_run)


async def _fetch_all(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SQL query and return all rows as dicts."""
    conn = await _get_connection()

    def _run():
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            columns = [desc[0].lower() for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

    return await asyncio.to_thread(_run)


async def _fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute a SQL query and return one row as dict (or None)."""
    conn = await _get_connection()

    def _run():
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            columns = [desc[0].lower() for desc in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None
        finally:
            cursor.close()

    return await asyncio.to_thread(_run)


class HanaConversationStore:
    """HANA-backed conversation store for ideation agent."""

    async def init_schema(self) -> None:
        """Create tables if they don't exist (idempotent).

        Uses try/catch per DDL statement instead of system view queries,
        which may not be accessible in HDI container contexts.
        """
        table_ddls = [
            """CREATE TABLE CONVERSATIONS (
                ID VARCHAR(36) PRIMARY KEY,
                USER_ID VARCHAR(256) NOT NULL,
                COMPANY_NAME NVARCHAR(256),
                TITLE NVARCHAR(512),
                STATUS VARCHAR(20) DEFAULT 'active',
                MESSAGE_COUNT INTEGER DEFAULT 0,
                AGENT_STATE_JSON NCLOB,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE CONVERSATION_MESSAGES (
                ID VARCHAR(36) PRIMARY KEY,
                CONVERSATION_ID VARCHAR(36) NOT NULL,
                SEQUENCE_NUM INTEGER NOT NULL,
                ROLE VARCHAR(20) NOT NULL,
                CONTENT NCLOB,
                METADATA_JSON NCLOB,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE CONVERSATION_RESEARCH (
                ID VARCHAR(36) PRIMARY KEY,
                CONVERSATION_ID VARCHAR(36) NOT NULL,
                RESEARCH_TYPE VARCHAR(40) NOT NULL,
                STATUS VARCHAR(20) NOT NULL,
                PAYLOAD_JSON NCLOB,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
        ]
        for ddl in table_ddls:
            try:
                await _execute(ddl)
            except Exception as exc:
                err_str = str(exc).lower()
                if "already exists" in err_str or "duplicate" in err_str or "288" in err_str:
                    continue
                raise

        # Migration: add EVENT_ID column to CONVERSATIONS (for event-scoped conversations)
        try:
            await _execute(
                "ALTER TABLE CONVERSATIONS ADD (EVENT_ID NVARCHAR(36) DEFAULT NULL)"
            )
        except Exception:
            pass  # Column already exists

        for sql in [
            "CREATE INDEX IDX_CONV_USER ON CONVERSATIONS(USER_ID)",
            "CREATE INDEX IDX_CONV_UPDATED ON CONVERSATIONS(USER_ID, UPDATED_AT DESC)",
            "CREATE INDEX IDX_MSG_CONV ON CONVERSATION_MESSAGES(CONVERSATION_ID, SEQUENCE_NUM)",
            "CREATE INDEX IDX_RESEARCH_CONV ON CONVERSATION_RESEARCH(CONVERSATION_ID, RESEARCH_TYPE)",
            "CREATE INDEX IX_CONV_EVENT ON CONVERSATIONS(EVENT_ID, USER_ID)",
        ]:
            try:
                await _execute(sql)
            except Exception:
                pass

    async def create_conversation(
        self,
        conv_id: str,
        user_id: str,
        company_name: str = "",
        title: str = "",
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation record."""
        now = datetime.now(timezone.utc)
        await _execute(
            """
            INSERT INTO CONVERSATIONS (ID, USER_ID, COMPANY_NAME, TITLE, STATUS, MESSAGE_COUNT, EVENT_ID, CREATED_AT, UPDATED_AT)
            VALUES (?, ?, ?, ?, 'active', 0, ?, ?, ?)
            """,
            (conv_id, user_id, company_name or "", title or "", event_id or None, now, now),
        )
        return {
            "id": conv_id,
            "user_id": user_id,
            "company_name": company_name,
            "title": title,
            "status": "active",
            "message_count": 0,
            "event_id": event_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

    async def list_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0, event_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List conversations for a user, ordered by most recent first.

        Args:
            event_id: If provided, filter to conversations in that event.
                      If None (default), filter to personal conversations (no event).
        """
        if event_id:
            rows = await _fetch_all(
                f"""
                SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS, MESSAGE_COUNT, EVENT_ID, CREATED_AT, UPDATED_AT
                FROM CONVERSATIONS
                WHERE USER_ID = ? AND EVENT_ID = ?
                ORDER BY UPDATED_AT DESC
                LIMIT {int(limit)} OFFSET {int(offset)}
                """,
                (user_id, event_id),
            )
        else:
            rows = await _fetch_all(
                f"""
                SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS, MESSAGE_COUNT, EVENT_ID, CREATED_AT, UPDATED_AT
                FROM CONVERSATIONS
                WHERE USER_ID = ? AND (EVENT_ID IS NULL OR EVENT_ID = '')
                ORDER BY UPDATED_AT DESC
                LIMIT {int(limit)} OFFSET {int(offset)}
                """,
                (user_id,),
            )
        for row in rows:
            for key in ("created_at", "updated_at"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
        return rows

    async def get_conversation(
        self, conv_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single conversation (checks user ownership)."""
        row = await _fetch_one(
            """
            SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS, MESSAGE_COUNT,
                   AGENT_STATE_JSON, EVENT_ID, CREATED_AT, UPDATED_AT
            FROM CONVERSATIONS
            WHERE ID = ? AND USER_ID = ?
            """,
            (conv_id, user_id),
        )
        if row:
            for key in ("created_at", "updated_at"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
            if row.get("agent_state_json"):
                try:
                    row["agent_state"] = json.loads(row["agent_state_json"])
                except (json.JSONDecodeError, TypeError):
                    row["agent_state"] = None
            else:
                row["agent_state"] = None
        return row

    async def get_conversation_by_id(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID without ownership check.

        Used after shared access has been verified externally (e.g. via share membership).
        """
        row = await _fetch_one(
            """
            SELECT ID, USER_ID, COMPANY_NAME, TITLE, STATUS, MESSAGE_COUNT,
                   AGENT_STATE_JSON, EVENT_ID, CREATED_AT, UPDATED_AT
            FROM CONVERSATIONS
            WHERE ID = ?
            """,
            (conv_id,),
        )
        if row:
            for key in ("created_at", "updated_at"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
            if row.get("agent_state_json"):
                try:
                    row["agent_state"] = json.loads(row["agent_state_json"])
                except (json.JSONDecodeError, TypeError):
                    row["agent_state"] = None
            else:
                row["agent_state"] = None
        return row

    async def update_conversation(self, conv_id: str, **fields) -> None:
        """Update conversation fields (title, status, message_count, agent_state_json, etc.)."""
        allowed = {"COMPANY_NAME", "TITLE", "STATUS", "MESSAGE_COUNT", "AGENT_STATE_JSON", "UPDATED_AT"}
        updates = {k.upper(): v for k, v in fields.items() if k.upper() in allowed}
        if not updates:
            return
        if "UPDATED_AT" not in updates:
            updates["UPDATED_AT"] = datetime.now(timezone.utc)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = tuple(updates.values()) + (conv_id,)
        await _execute(
            f"UPDATE CONVERSATIONS SET {set_clause} WHERE ID = ?",
            values,
        )

    async def delete_conversation(self, conv_id: str, user_id: str) -> bool:
        """Delete a conversation, its messages, and its research (checks user ownership)."""
        row = await _fetch_one(
            "SELECT ID FROM CONVERSATIONS WHERE ID = ? AND USER_ID = ?",
            (conv_id, user_id),
        )
        if not row:
            return False
        await _execute(
            "DELETE FROM CONVERSATION_RESEARCH WHERE CONVERSATION_ID = ?",
            (conv_id,),
        )
        await _execute(
            "DELETE FROM CONVERSATION_MESSAGES WHERE CONVERSATION_ID = ?",
            (conv_id,),
        )
        await _execute("DELETE FROM CONVERSATIONS WHERE ID = ?", (conv_id,))
        return True

    async def append_message(
        self,
        conv_id: str,
        msg_id: str,
        sequence_num: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a message to a conversation."""
        metadata_json = json.dumps(metadata) if metadata else None
        now = datetime.now(timezone.utc)
        await _execute(
            """
            INSERT INTO CONVERSATION_MESSAGES (ID, CONVERSATION_ID, SEQUENCE_NUM, ROLE, CONTENT, METADATA_JSON, CREATED_AT)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, conv_id, sequence_num, role, content, metadata_json, now),
        )

    async def get_messages(
        self, conv_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation, ordered by sequence."""
        sql = """
            SELECT ID, CONVERSATION_ID, SEQUENCE_NUM, ROLE, CONTENT, METADATA_JSON, CREATED_AT
            FROM CONVERSATION_MESSAGES
            WHERE CONVERSATION_ID = ?
            ORDER BY SEQUENCE_NUM ASC
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = await _fetch_all(sql, (conv_id,))
        for row in rows:
            if isinstance(row.get("created_at"), datetime):
                row["created_at"] = row["created_at"].isoformat()
            if row.get("metadata_json"):
                try:
                    row["metadata"] = json.loads(row["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    row["metadata"] = None
            else:
                row["metadata"] = None
        return rows

    async def get_message_count(self, conv_id: str) -> int:
        """Get the number of messages in a conversation."""
        row = await _fetch_one(
            "SELECT COUNT(*) AS CNT FROM CONVERSATION_MESSAGES WHERE CONVERSATION_ID = ?",
            (conv_id,),
        )
        return int(row["cnt"]) if row else 0

    async def delete_messages_from(self, conv_id: str, sequence_num: int) -> None:
        """Delete every message with SEQUENCE_NUM >= sequence_num (server-side rewind)."""
        await _execute(
            "DELETE FROM CONVERSATION_MESSAGES WHERE CONVERSATION_ID = ? AND SEQUENCE_NUM >= ?",
            (conv_id, sequence_num),
        )

    async def rewind_to(
        self,
        conv_id: str,
        sequence_num: int,
        agent_state_json: Optional[str],
        message_count: int,
    ) -> None:
        """Truncate to before `sequence_num` and restore agent state atomically.

        Deletes every message row with SEQUENCE_NUM >= sequence_num and resets
        AGENT_STATE_JSON + MESSAGE_COUNT in a single transaction, so a rewind can
        never leave the message rows and the agent-state snapshot out of sync.
        `agent_state_json` may be None (restores a fresh/empty state).
        """
        now = datetime.now(timezone.utc)
        await _execute_transaction([
            (
                "DELETE FROM CONVERSATION_MESSAGES WHERE CONVERSATION_ID = ? AND SEQUENCE_NUM >= ?",
                (conv_id, sequence_num),
            ),
            (
                "UPDATE CONVERSATIONS SET AGENT_STATE_JSON = ?, MESSAGE_COUNT = ?, UPDATED_AT = ? WHERE ID = ?",
                (agent_state_json, message_count, now, conv_id),
            ),
        ])

    # ── Research persistence ─────────────────────────────────────────────

    async def save_research(
        self,
        conv_id: str,
        research_type: str,
        status: str,
        payload: Dict[str, Any],
    ) -> None:
        """Upsert research result for a conversation (DELETE+INSERT)."""
        await _execute(
            "DELETE FROM CONVERSATION_RESEARCH WHERE CONVERSATION_ID = ? AND RESEARCH_TYPE = ?",
            (conv_id, research_type),
        )
        now = datetime.now(timezone.utc)
        await _execute(
            """
            INSERT INTO CONVERSATION_RESEARCH (ID, CONVERSATION_ID, RESEARCH_TYPE, STATUS, PAYLOAD_JSON, CREATED_AT, UPDATED_AT)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), conv_id, research_type, status, json.dumps(payload), now, now),
        )

    async def get_research(
        self, conv_id: str, research_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single research result by type. Returns parsed payload or None."""
        row = await _fetch_one(
            "SELECT STATUS, PAYLOAD_JSON, UPDATED_AT FROM CONVERSATION_RESEARCH "
            "WHERE CONVERSATION_ID = ? AND RESEARCH_TYPE = ?",
            (conv_id, research_type),
        )
        if not row:
            return None
        payload = None
        if row.get("payload_json"):
            try:
                payload = json.loads(row["payload_json"])
            except (json.JSONDecodeError, TypeError):
                payload = None
        return {"status": row["status"], "payload": payload, "updated_at": row.get("updated_at")}

    async def get_all_research(self, conv_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all research results for a conversation, keyed by research_type."""
        rows = await _fetch_all(
            "SELECT RESEARCH_TYPE, STATUS, PAYLOAD_JSON, UPDATED_AT FROM CONVERSATION_RESEARCH "
            "WHERE CONVERSATION_ID = ?",
            (conv_id,),
        )
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            payload = None
            if row.get("payload_json"):
                try:
                    payload = json.loads(row["payload_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            result[row["research_type"]] = {
                "status": row["status"],
                "payload": payload,
                "updated_at": row.get("updated_at"),
            }
        return result


# Module-level singleton
_store: Optional[HanaConversationStore] = None
_store_lock = asyncio.Lock()


async def get_store() -> HanaConversationStore:
    """Get or create the singleton store instance (initializes schema on first call)."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is not None:
            return _store
        store = HanaConversationStore()
        try:
            await store.init_schema()
            logger.info("HANA conversation store initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize HANA schema: %s", exc)
            raise
        _store = store
        return _store
