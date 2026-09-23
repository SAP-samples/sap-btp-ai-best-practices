"""Persist concise conversation history in the runtime user's SAP HANA schema."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from typing import Any, Protocol

from .config import MemorySettings

Conversation = list[dict[str, str]]
ConnectionFactory = Callable[[], Any]


class ConversationStore(Protocol):
    """Define the small persistence contract required by AgentRuntime."""

    max_messages: int

    def ensure(self) -> None:
        """Create or validate backing storage before the first write."""

    def load(self, context_id: str) -> Conversation:
        """Return stored turns for a context identifier."""

    def save(self, context_id: str, messages: Conversation) -> None:
        """Replace stored turns for a context identifier."""

    def clear(self, context_id: str) -> None:
        """Delete stored turns for a context identifier."""

    def close(self) -> None:
        """Release persistence resources."""


class NullConversationStore:
    """Provide a no-op store when HANA memory is disabled."""

    def __init__(self, max_messages: int = 40) -> None:
        """Initialize the no-op store with the same retention contract."""

        self.max_messages = max_messages

    def ensure(self) -> None:
        """Perform no setup."""

    def load(self, context_id: str) -> Conversation:
        """Return no prior messages."""

        return []

    def save(self, context_id: str, messages: Conversation) -> None:
        """Discard messages intentionally."""

    def clear(self, context_id: str) -> None:
        """Clear nothing intentionally."""

    def close(self) -> None:
        """Release no resources."""


class HanaConversationStore:
    """Store one bounded JSON conversation row per context in SAP HANA."""

    def __init__(
        self,
        settings: MemorySettings,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Initialize the HANA store.

        Args:
            settings: Validated table name and retention limit.
            connection_factory: Optional test seam returning hdbcli-like connections.
        """

        self.table_name = settings.table_name
        self.max_messages = settings.max_messages
        self._connection_factory = connection_factory or _hana_connection
        self._ready = False
        self._lock = threading.Lock()

    @property
    def quoted_table(self) -> str:
        """Return the already validated table identifier with HANA quoting."""

        return f'"{self.table_name}"'

    def ensure(self) -> None:
        """Create the memory table or validate its existing column contract."""

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
                exists = bool(cursor.fetchone()[0])
                if exists:
                    self._validate_contract(cursor, schema)
                else:
                    cursor.execute(
                        f"""
                        CREATE COLUMN TABLE {self.quoted_table} (
                          "CONTEXT_ID" NVARCHAR(256) NOT NULL PRIMARY KEY,
                          "MESSAGES" NCLOB NOT NULL,
                          "UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
                        )
                        """
                    )
                connection.commit()
                self._ready = True
            finally:
                cursor.close()
                connection.close()

    def load(self, context_id: str) -> Conversation:
        """Load and validate the JSON conversation for one context."""

        self.ensure()
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f'SELECT "MESSAGES" FROM {self.quoted_table} WHERE "CONTEXT_ID" = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            if not row:
                return []
            payload = json.loads(row[0])
            if not isinstance(payload, list):
                raise RuntimeError(f"Invalid conversation JSON for context {context_id!r}")
            return [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in payload
                if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
            ][-self.max_messages :]
        finally:
            cursor.close()
            connection.close()

    def save(self, context_id: str, messages: Conversation) -> None:
        """Upsert the latest bounded conversation using parameterized HANA SQL."""

        self.ensure()
        payload = json.dumps(messages[-self.max_messages :], ensure_ascii=False)
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"""
                MERGE INTO {self.quoted_table} AS target
                USING (SELECT ? AS "CONTEXT_ID", ? AS "MESSAGES" FROM DUMMY) AS source
                ON target."CONTEXT_ID" = source."CONTEXT_ID"
                WHEN MATCHED THEN UPDATE SET
                  target."MESSAGES" = source."MESSAGES",
                  target."UPDATED_AT" = CURRENT_UTCTIMESTAMP
                WHEN NOT MATCHED THEN INSERT ("CONTEXT_ID", "MESSAGES", "UPDATED_AT")
                  VALUES (source."CONTEXT_ID", source."MESSAGES", CURRENT_UTCTIMESTAMP)
                """,
                (context_id, payload),
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

    def _validate_contract(self, cursor: Any, schema: str) -> None:
        """Reject an existing table whose required columns are incompatible."""

        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE, LENGTH
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            """,
            (schema, self.table_name),
        )
        actual = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}
        problems: list[str] = []
        expected = {
            "CONTEXT_ID": ("NVARCHAR", "FALSE"),
            "MESSAGES": ("NCLOB", "FALSE"),
            "UPDATED_AT": ("TIMESTAMP", "FALSE"),
        }
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
    return NullConversationStore(settings.max_messages)


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
