"""Verify HANA table setup and parameterized conversation DML."""

from collections.abc import Iterable
from typing import Any

import pytest

from template_agent.config import MemorySettings
from template_agent.memory import HanaConversationStore


class FakeCursor:
    """Return scripted database rows and record executed SQL."""

    def __init__(self, fetchone: Iterable[Any] = (), fetchall: Iterable[Any] = ()) -> None:
        """Initialize response queues."""

        self.fetchone_values = list(fetchone)
        self.fetchall_value = list(fetchall)
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        """Record one statement and its bound parameters."""

        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        """Return the next scripted row."""

        return self.fetchone_values.pop(0)

    def fetchall(self) -> list[Any]:
        """Return the scripted row list."""

        return self.fetchall_value

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


def test_creates_table_then_uses_parameterized_merge_select_delete() -> None:
    """Exercise the complete repository SQL surface without a live database."""

    setup = FakeCursor(fetchone=[("SCHEMA",), (0,)])
    save = FakeCursor()
    load = FakeCursor(fetchone=[('[{"role":"user","content":"hello"}]',)])
    clear = FakeCursor()
    connections = [
        FakeConnection(setup),
        FakeConnection(save),
        FakeConnection(load),
        FakeConnection(clear),
    ]
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: connections.pop(0),
    )

    store.ensure()
    store.save("ctx", [{"role": "user", "content": "hello"}])
    assert store.load("ctx") == [{"role": "user", "content": "hello"}]
    store.clear("ctx")

    assert any("CREATE COLUMN TABLE" in sql for sql, _ in setup.executed)
    merge_sql, merge_params = save.executed[0]
    assert "MERGE INTO" in merge_sql
    assert merge_params[0] == "ctx"
    assert load.executed[0][1] == ("ctx",)
    assert clear.executed[0][1] == ("ctx",)


def test_rejects_incompatible_existing_table() -> None:
    """Fail startup when the existing table contract is stale."""

    cursor = FakeCursor(
        fetchone=[("SCHEMA",), (1,)],
        fetchall=[
            ("CONTEXT_ID", "NVARCHAR", "FALSE", 100),
            ("MESSAGES", "NVARCHAR", "TRUE", 100),
        ],
    )
    store = HanaConversationStore(
        MemorySettings(enabled=True),
        connection_factory=lambda: FakeConnection(cursor),
    )
    with pytest.raises(RuntimeError, match="incompatible contract"):
        store.ensure()
