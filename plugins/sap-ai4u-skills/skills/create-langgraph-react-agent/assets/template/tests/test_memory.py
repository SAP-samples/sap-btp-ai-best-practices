"""Verify HANA table setup, migration, leasing, versioned saves, and payloads."""

import json
from collections.abc import Iterable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from template_agent.config import MemorySettings
from template_agent.memory import (
    ConcurrentUpdateError,
    HanaConversationStore,
    _encode_messages,
)

V1_COLUMNS = [
    ("CONTEXT_ID", "NVARCHAR", "FALSE", 256),
    ("MESSAGES", "NCLOB", "FALSE", None),
    ("UPDATED_AT", "TIMESTAMP", "FALSE", None),
]
V2_COLUMNS = [
    *V1_COLUMNS,
    ("PENDING_ACTION", "NCLOB", "TRUE", None),
    ("VERSION", "BIGINT", "FALSE", None),
    ("LEASE_OWNER", "NVARCHAR", "TRUE", 64),
    ("LEASE_EXPIRES_AT", "TIMESTAMP", "TRUE", None),
]


class HanaError(Exception):
    """Mimic an hdbcli error carrying a HANA error code."""

    def __init__(self, errorcode: int) -> None:
        """Store the numeric HANA error code."""

        super().__init__(f"HANA error {errorcode}")
        self.errorcode = errorcode


class FakeCursor:
    """Return scripted rows and row counts, and record executed SQL."""

    def __init__(
        self,
        fetchone: Iterable[Any] = (),
        fetchall: Iterable[list[Any]] = (),
        rowcounts: Iterable[int] = (),
        failures: dict[str, Exception] | None = None,
    ) -> None:
        """Initialize response queues and statement failures keyed by SQL fragment."""

        self.fetchone_values = list(fetchone)
        self.fetchall_values = list(fetchall)
        self.rowcounts = list(rowcounts)
        self.failures = dict(failures or {})
        self.rowcount = -1
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        """Record one statement, raise a scripted failure, or set its row count."""

        self.executed.append((sql, params))
        for fragment, error in list(self.failures.items()):
            if fragment in sql:
                del self.failures[fragment]
                raise error
        if sql.strip().split()[0] in {"UPDATE", "INSERT", "DELETE"}:
            self.rowcount = self.rowcounts.pop(0) if self.rowcounts else 1

    def fetchone(self) -> Any:
        """Return the next scripted row."""

        return self.fetchone_values.pop(0)

    def fetchall(self) -> list[Any]:
        """Return the next scripted row list."""

        return self.fetchall_values.pop(0)

    def close(self) -> None:
        """Close nothing."""


class FakeConnection:
    """Wrap one fake cursor with commit/close counters."""

    def __init__(self, cursor: FakeCursor) -> None:
        """Store the cursor."""

        self.value = cursor
        self.commits = 0

    def cursor(self) -> FakeCursor:
        """Return the stored cursor."""

        return self.value

    def commit(self) -> None:
        """Record a commit."""

        self.commits += 1

    def close(self) -> None:
        """Close nothing."""


def _ready_store(cursor: FakeCursor) -> HanaConversationStore:
    """Return a store that skips catalog setup and always uses ``cursor``."""

    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    store._ready = True
    return store


def test_creates_full_table_when_absent() -> None:
    """A fresh schema receives the complete format 2 table in one statement."""

    cursor = FakeCursor(fetchone=[("SCHEMA",), (0,)])
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    store.ensure()

    create_sql = next(sql for sql, _ in cursor.executed if "CREATE COLUMN TABLE" in sql)
    for column in ("PENDING_ACTION", "VERSION", "LEASE_OWNER", "LEASE_EXPIRES_AT"):
        assert column in create_sql


def test_migrates_version_1_table_with_alter_statements() -> None:
    """An existing role/content table gains the four format 2 columns."""

    cursor = FakeCursor(fetchone=[("SCHEMA",), (1,)], fetchall=[V1_COLUMNS, V2_COLUMNS])
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    store.ensure()

    alters = [sql for sql, _ in cursor.executed if sql.startswith("ALTER TABLE")]
    assert len(alters) == 4
    assert any('"VERSION" BIGINT DEFAULT 0 NOT NULL' in sql for sql in alters)


def test_migration_accepts_column_added_by_another_instance() -> None:
    """A failed ALTER is fine when the re-read catalog shows the column exists."""

    cursor = FakeCursor(
        fetchone=[("SCHEMA",), (1,)],
        fetchall=[V1_COLUMNS, V2_COLUMNS],
        failures={'ADD ("VERSION"': HanaError(324)},
    )
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    store.ensure()
    assert store._ready


def test_rejects_incompatible_existing_table() -> None:
    """Fail startup when an existing column has the wrong type."""

    stale = [
        ("CONTEXT_ID", "NVARCHAR", "FALSE", 100),
        ("MESSAGES", "NVARCHAR", "TRUE", 100),
    ]
    cursor = FakeCursor(fetchone=[("SCHEMA",), (1,)], fetchall=[stale])
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    with pytest.raises(RuntimeError, match="incompatible contract"):
        store.ensure()


def test_acquire_takes_free_lease_and_confirms_owner() -> None:
    """A conditional UPDATE plus read-back grants the lease to this owner."""

    cursor = FakeCursor(fetchone=[("owner-a",)], rowcounts=[1])
    assert _ready_store(cursor).acquire("ctx", "owner-a", 300) is True
    update_sql, params = cursor.executed[0]
    assert "LEASE_EXPIRES_AT\" < CURRENT_UTCTIMESTAMP" in update_sql
    assert params == ("owner-a", 300, "ctx")


def test_acquire_reports_busy_lease() -> None:
    """An existing row with a live lease is not taken."""

    cursor = FakeCursor(fetchone=[(1,)], rowcounts=[0])
    assert _ready_store(cursor).acquire("ctx", "owner-b", 300) is False


def test_acquire_rejects_when_concurrent_update_kept_other_owner() -> None:
    """The read-back decides the winner if two updates both reported a row."""

    cursor = FakeCursor(fetchone=[("owner-a",)], rowcounts=[1])
    assert _ready_store(cursor).acquire("ctx", "owner-b", 300) is False


def test_acquire_inserts_missing_row_already_leased() -> None:
    """The first turn of a new context creates its row holding the lease."""

    cursor = FakeCursor(fetchone=[(0,)], rowcounts=[0, 1])
    assert _ready_store(cursor).acquire("ctx", "owner-a", 300) is True
    insert_sql, params = cursor.executed[-1]
    assert insert_sql.strip().startswith("INSERT INTO")
    assert params[0] == "ctx" and params[2] == "owner-a"


def test_acquire_loses_insert_race_on_duplicate_key() -> None:
    """A unique violation means another instance created and leased the row."""

    cursor = FakeCursor(
        fetchone=[(0,)],
        rowcounts=[0],
        failures={"INSERT INTO": HanaError(301)},
    )
    assert _ready_store(cursor).acquire("ctx", "owner-b", 300) is False


def test_acquire_propagates_unexpected_insert_error() -> None:
    """Only the duplicate-key error is treated as a lost race."""

    cursor = FakeCursor(
        fetchone=[(0,)],
        rowcounts=[0],
        failures={"INSERT INTO": HanaError(258)},
    )
    with pytest.raises(HanaError):
        _ready_store(cursor).acquire("ctx", "owner-b", 300)


def test_save_is_fenced_by_version_and_lease_owner() -> None:
    """A save binds the expected version and owner and releases the lease."""

    cursor = FakeCursor(rowcounts=[1])
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    _ready_store(cursor).save("ctx", "owner-a", 3, messages, {"tool_calls": []})

    sql, params = cursor.executed[0]
    assert '"VERSION" = "VERSION" + 1' in sql
    assert '"LEASE_OWNER" = NULL' in sql
    assert params[2:] == ("ctx", 3, "owner-a")
    assert json.loads(params[1]) == {"tool_calls": []}


def test_save_rejects_stale_turn_without_writing() -> None:
    """Zero updated rows means a newer turn won; raise instead of overwriting."""

    cursor = FakeCursor(rowcounts=[0])
    with pytest.raises(ConcurrentUpdateError):
        _ready_store(cursor).save("ctx", "owner-a", 3, [], None)


def test_load_round_trips_tool_calls_and_pending_action() -> None:
    """Format 2 payloads restore tool calls, tool results, and pending actions."""

    messages = [
        HumanMessage(content="look it up"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content="result", tool_call_id="c1"),
        AIMessage(content="done"),
    ]
    pending = {"tool_calls": [{"name": "delete", "args": {}, "id": "c2"}]}
    cursor = FakeCursor(fetchone=[(_encode_messages(messages), json.dumps(pending), 7)])

    stored = _ready_store(cursor).load("ctx")

    assert stored.version == 7
    assert stored.pending_action == pending
    assert stored.messages[1].tool_calls[0]["name"] == "search"
    assert isinstance(stored.messages[2], ToolMessage)
    assert stored.messages[2].tool_call_id == "c1"


def test_load_upgrades_version_1_payload() -> None:
    """Rows written by the role/content format stay readable."""

    legacy = json.dumps(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    )
    cursor = FakeCursor(fetchone=[(legacy, None, 0)])

    stored = _ready_store(cursor).load("ctx")

    assert [type(m) for m in stored.messages] == [HumanMessage, AIMessage]
    assert stored.pending_action is None


def test_release_and_clear_bind_values() -> None:
    """Release only matches this owner; clear deletes by bound key."""

    cursor = FakeCursor(rowcounts=[1, 1])
    store = _ready_store(cursor)
    store.release("ctx", "owner-a")
    store.clear("ctx")
    assert cursor.executed[0][1] == ("ctx", "owner-a")
    assert cursor.executed[1][1] == ("ctx",)
