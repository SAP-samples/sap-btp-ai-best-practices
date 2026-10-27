"""Persist full conversation state in the runtime user's SAP HANA schema.

One row per conversation context stores the complete LangChain message list
(user turns, assistant turns, tool calls, and tool results), an optional pending
human-approval action, and the columns that make concurrent access safe across
several application instances:

- ``LEASE_OWNER`` / ``LEASE_EXPIRES_AT``: a time-bounded row lease. Only the
  lease holder may run a turn for the context, so two Cloud Foundry instances
  never execute the same conversation at the same time.
- ``VERSION``: an optimistic-concurrency counter. A save succeeds only when the
  version read at the start of the turn is still current and the caller still
  holds the lease, so a turn whose lease expired can never overwrite newer data.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

from .config import MemorySettings

ConnectionFactory = Callable[[], Any]

# Version marker written into every MESSAGES payload. Version 1 rows (a bare
# JSON list of {"role", "content"} pairs) are still readable.
PAYLOAD_FORMAT = 2

# HANA SQL error code for a unique-constraint violation.
_HANA_UNIQUE_VIOLATION = 301

# Columns every memory table must have, mapped to (type, nullable).
_CORE_COLUMNS = {
    "CONTEXT_ID": ("NVARCHAR", "FALSE"),
    "MESSAGES": ("NCLOB", "FALSE"),
    "UPDATED_AT": ("TIMESTAMP", "FALSE"),
}

# Columns added by payload format 2, with the DDL used to add them to a
# version 1 table. Existing rows receive VERSION 0 and no lease or action.
_V2_COLUMNS = {
    "PENDING_ACTION": (("NCLOB", "TRUE"), '"PENDING_ACTION" NCLOB'),
    "VERSION": (("BIGINT", "FALSE"), '"VERSION" BIGINT DEFAULT 0 NOT NULL'),
    "LEASE_OWNER": (("NVARCHAR", "TRUE"), '"LEASE_OWNER" NVARCHAR(64)'),
    "LEASE_EXPIRES_AT": (("TIMESTAMP", "TRUE"), '"LEASE_EXPIRES_AT" TIMESTAMP'),
}


class ContextBusyError(RuntimeError):
    """Raised when another request holds the lease for a conversation context."""


class ConcurrentUpdateError(RuntimeError):
    """Raised when a save would overwrite a newer version of a conversation."""


@dataclass
class StoredConversation:
    """Hold one context's persisted state as read at the start of a turn.

    Attributes:
        messages: Full LangChain message history, oldest first.
        version: Optimistic-concurrency version to pass back to ``save``.
        pending_action: Tool calls awaiting human approval, or ``None``.
    """

    messages: list[BaseMessage] = field(default_factory=list)
    version: int = 0
    pending_action: dict[str, Any] | None = None


class ConversationStore(Protocol):
    """Define the persistence and concurrency contract required by AgentRuntime."""

    def ensure(self) -> None:
        """Create, migrate, or validate backing storage before the first write."""

    def acquire(self, context_id: str, owner: str, lease_seconds: int) -> bool:
        """Try once to take the context lease; return whether ``owner`` holds it."""

    def load(self, context_id: str) -> StoredConversation:
        """Return the stored state for a context identifier."""

    def save(
        self,
        context_id: str,
        owner: str,
        expected_version: int,
        messages: list[BaseMessage],
        pending_action: dict[str, Any] | None,
    ) -> None:
        """Write a new state and release the lease, or raise ConcurrentUpdateError."""

    def release(self, context_id: str, owner: str) -> None:
        """Release the lease if ``owner`` still holds it."""

    def clear(self, context_id: str) -> None:
        """Delete stored state for a context identifier."""

    def close(self) -> None:
        """Release persistence resources."""


class NullConversationStore:
    """Provide a no-op store when HANA memory is disabled.

    Every turn starts from an empty history and nothing is shared, so the lease
    is always granted and no cross-instance coordination is needed.
    """

    def ensure(self) -> None:
        """Perform no setup."""

    def acquire(self, context_id: str, owner: str, lease_seconds: int) -> bool:
        """Always grant the lease because there is no shared state."""

        return True

    def load(self, context_id: str) -> StoredConversation:
        """Return no prior messages."""

        return StoredConversation()

    def save(
        self,
        context_id: str,
        owner: str,
        expected_version: int,
        messages: list[BaseMessage],
        pending_action: dict[str, Any] | None,
    ) -> None:
        """Discard messages intentionally."""

    def release(self, context_id: str, owner: str) -> None:
        """Release nothing."""

    def clear(self, context_id: str) -> None:
        """Clear nothing intentionally."""

    def close(self) -> None:
        """Release no resources."""


class HanaConversationStore:
    """Store one leased, versioned JSON conversation row per context in SAP HANA."""

    def __init__(
        self,
        settings: MemorySettings,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Initialize the HANA store.

        Args:
            settings: Validated table name and memory limits.
            connection_factory: Optional test seam returning hdbcli-like connections.
        """

        self.table_name = settings.table_name
        self._connection_factory = connection_factory or _hana_connection
        self._ready = False
        self._lock = threading.Lock()

    @property
    def quoted_table(self) -> str:
        """Return the already validated table identifier with HANA quoting."""

        return f'"{self.table_name}"'

    def ensure(self) -> None:
        """Create the memory table, or migrate and validate an existing one.

        A missing table is created with the full format 2 contract. An existing
        format 1 table receives the missing columns through ``ALTER TABLE``.
        Several instances may start at once, so a failed ``ALTER`` is accepted
        when a re-read shows another instance already added the column.
        """

        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            connection = self._connection_factory()
            cursor = connection.cursor()
            try:
                cursor.execute("SELECT CURRENT_SCHEMA FROM DUMMY")
                schema = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?",
                    (schema, self.table_name),
                )
                if cursor.fetchone()[0]:
                    self._migrate(cursor, schema)
                else:
                    cursor.execute(
                        f"""
                        CREATE COLUMN TABLE {self.quoted_table} (
                          "CONTEXT_ID" NVARCHAR(256) NOT NULL PRIMARY KEY,
                          "MESSAGES" NCLOB NOT NULL,
                          "PENDING_ACTION" NCLOB,
                          "VERSION" BIGINT DEFAULT 0 NOT NULL,
                          "LEASE_OWNER" NVARCHAR(64),
                          "LEASE_EXPIRES_AT" TIMESTAMP,
                          "UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
                        )
                        """
                    )
                connection.commit()
                self._ready = True
            finally:
                cursor.close()
                connection.close()

    def acquire(self, context_id: str, owner: str, lease_seconds: int) -> bool:
        """Try once to take the lease for a context.

        The conditional ``UPDATE`` only matches a free or expired lease. The
        follow-up read confirms which owner HANA kept, so the result is correct
        even if two instances update the row at the same moment. A missing row
        is inserted already leased; a duplicate-key error means another
        instance created it first and therefore holds the lease.

        Args:
            context_id: Conversation key.
            owner: Unique token identifying this turn.
            lease_seconds: Lease lifetime; must exceed the longest expected turn.

        Returns:
            ``True`` when ``owner`` now holds the lease, otherwise ``False``.
        """

        self.ensure()
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                UPDATE {self.quoted_table}
                SET "LEASE_OWNER" = ?,
                    "LEASE_EXPIRES_AT" = ADD_SECONDS(CURRENT_UTCTIMESTAMP, ?)
                WHERE "CONTEXT_ID" = ?
                  AND ("LEASE_OWNER" IS NULL OR "LEASE_EXPIRES_AT" < CURRENT_UTCTIMESTAMP)
                """,
                (owner, lease_seconds, context_id),
            )
            updated = cursor.rowcount
            connection.commit()
            if updated == 1:
                cursor.execute(
                    f'SELECT "LEASE_OWNER" FROM {self.quoted_table} WHERE "CONTEXT_ID" = ?',
                    (context_id,),
                )
                row = cursor.fetchone()
                return bool(row) and row[0] == owner

            cursor.execute(
                f'SELECT COUNT(*) FROM {self.quoted_table} WHERE "CONTEXT_ID" = ?',
                (context_id,),
            )
            if cursor.fetchone()[0]:
                return False
            try:
                cursor.execute(
                    f"""
                    INSERT INTO {self.quoted_table}
                      ("CONTEXT_ID", "MESSAGES", "VERSION", "LEASE_OWNER", "LEASE_EXPIRES_AT")
                    VALUES (?, ?, 0, ?, ADD_SECONDS(CURRENT_UTCTIMESTAMP, ?))
                    """,
                    (context_id, _encode_messages([]), owner, lease_seconds),
                )
                connection.commit()
            except Exception as exc:
                if getattr(exc, "errorcode", None) == _HANA_UNIQUE_VIOLATION:
                    return False
                raise
            return True
        finally:
            cursor.close()
            connection.close()

    def load(self, context_id: str) -> StoredConversation:
        """Load and decode the stored state for one context."""

        self.ensure()
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                SELECT "MESSAGES", "PENDING_ACTION", "VERSION"
                FROM {self.quoted_table} WHERE "CONTEXT_ID" = ?
                """,
                (context_id,),
            )
            row = cursor.fetchone()
            if not row:
                return StoredConversation()
            pending = json.loads(row[1]) if row[1] else None
            return StoredConversation(
                messages=_decode_messages(row[0], context_id),
                version=int(row[2]),
                pending_action=pending,
            )
        finally:
            cursor.close()
            connection.close()

    def save(
        self,
        context_id: str,
        owner: str,
        expected_version: int,
        messages: list[BaseMessage],
        pending_action: dict[str, Any] | None,
    ) -> None:
        """Write the new state, bump the version, and release the lease atomically.

        Args:
            context_id: Conversation key.
            owner: Lease token used by ``acquire``.
            expected_version: Version returned by ``load`` for this turn.
            messages: Complete history to persist (attachment bytes removed).
            pending_action: Tool calls awaiting approval, or ``None``.

        Raises:
            ConcurrentUpdateError: When the version moved on or the lease was
                lost; nothing is written in that case.
        """

        self.ensure()
        pending = json.dumps(pending_action, ensure_ascii=False) if pending_action else None
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                UPDATE {self.quoted_table}
                SET "MESSAGES" = ?,
                    "PENDING_ACTION" = ?,
                    "VERSION" = "VERSION" + 1,
                    "LEASE_OWNER" = NULL,
                    "LEASE_EXPIRES_AT" = NULL,
                    "UPDATED_AT" = CURRENT_UTCTIMESTAMP
                WHERE "CONTEXT_ID" = ? AND "VERSION" = ? AND "LEASE_OWNER" = ?
                """,
                (_encode_messages(messages), pending, context_id, expected_version, owner),
            )
            updated = cursor.rowcount
            connection.commit()
        finally:
            cursor.close()
            connection.close()
        if updated != 1:
            raise ConcurrentUpdateError(
                f"Conversation {context_id!r} changed or its lease expired during the "
                "turn; the result was not saved"
            )

    def release(self, context_id: str, owner: str) -> None:
        """Release the lease without writing state, if ``owner`` still holds it."""

        self.ensure()
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                UPDATE {self.quoted_table}
                SET "LEASE_OWNER" = NULL, "LEASE_EXPIRES_AT" = NULL
                WHERE "CONTEXT_ID" = ? AND "LEASE_OWNER" = ?
                """,
                (context_id, owner),
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def clear(self, context_id: str) -> None:
        """Delete a context row using a bound value parameter."""

        self.ensure()
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f'DELETE FROM {self.quoted_table} WHERE "CONTEXT_ID" = ?',
                (context_id,),
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def close(self) -> None:
        """Release no persistent connection because operations are short-lived."""

    def _columns(self, cursor: Any, schema: str) -> dict[str, tuple[Any, ...]]:
        """Return existing column name -> (type, nullable, length) from the catalog."""

        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            """,
            (schema, self.table_name),
        )
        return {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}

    def _migrate(self, cursor: Any, schema: str) -> None:
        """Add missing format 2 columns, then reject any incompatible column.

        Raises:
            RuntimeError: When a required column is missing or has the wrong type.
        """

        actual = self._columns(cursor, schema)
        missing = [name for name in _V2_COLUMNS if name not in actual]
        if missing and all(name in actual for name in _CORE_COLUMNS):
            for name in missing:
                try:
                    cursor.execute(
                        f"ALTER TABLE {self.quoted_table} ADD ({_V2_COLUMNS[name][1]})"
                    )
                except Exception:
                    # Another instance may have added the column concurrently;
                    # the validation below decides whether this is a real failure.
                    pass
            actual = self._columns(cursor, schema)

        expected = {**_CORE_COLUMNS, **{name: spec[0] for name, spec in _V2_COLUMNS.items()}}
        problems: list[str] = []
        for column, contract in expected.items():
            value = actual.get(column)
            if value is None or value[:2] != contract:
                problems.append(f"{column}: expected {contract}, got {value}")
        context = actual.get("CONTEXT_ID")
        if context and context[2] != 256:
            problems.append(f"CONTEXT_ID: expected length 256, got {context[2]}")
        if problems:
            raise RuntimeError(
                f"HANA table {schema}.{self.table_name} has an incompatible contract: "
                + "; ".join(problems)
            )


def create_conversation_store(settings: MemorySettings) -> ConversationStore:
    """Return HANA persistence when enabled, otherwise the no-op implementation."""

    if settings.enabled:
        return HanaConversationStore(settings)
    return NullConversationStore()


def _encode_messages(messages: list[BaseMessage]) -> str:
    """Serialize messages into the versioned JSON payload stored in MESSAGES."""

    return json.dumps(
        {"format": PAYLOAD_FORMAT, "messages": messages_to_dict(messages)},
        ensure_ascii=False,
    )


def _decode_messages(raw: str, context_id: str) -> list[BaseMessage]:
    """Deserialize a format 2 payload, or upgrade a format 1 role/content list.

    Raises:
        RuntimeError: When the stored JSON matches neither format.
    """

    payload = json.loads(raw)
    if isinstance(payload, dict) and payload.get("format") == PAYLOAD_FORMAT:
        return messages_from_dict(payload["messages"])
    if isinstance(payload, list):
        converted: list[BaseMessage] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                converted.append(HumanMessage(content=str(item["content"])))
            elif item.get("role") == "assistant":
                converted.append(AIMessage(content=str(item["content"])))
        return converted
    raise RuntimeError(f"Invalid conversation JSON for context {context_id!r}")


def _hana_connection() -> Any:
    """Create one hdbcli connection from standard environment variables."""

    from hdbcli import dbapi

    required = ["HANA_ADDRESS", "HANA_PORT", "HANA_USER", "HANA_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing HANA environment variable(s): {', '.join(missing)}")
    return dbapi.connect(
        address=os.environ["HANA_ADDRESS"],
        port=int(os.environ["HANA_PORT"]),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        encrypt=_as_bool(os.getenv("HANA_ENCRYPT", "true")),
        sslValidateCertificate=_as_bool(
            os.getenv("HANA_SSL_VALIDATE_CERTIFICATE", "true")
        ),
    )


def _as_bool(value: str) -> bool:
    """Parse a conventional environment boolean or reject ambiguous input."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")
