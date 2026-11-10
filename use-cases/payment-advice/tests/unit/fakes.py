"""
Minimal SQLAlchemy Engine / Connection / Result fake for offline unit tests.

This module provides FakeEngine, a lightweight double that emulates just enough
of the SQLAlchemy surface used by payment_advice data-access modules so that
tests can run without a live SAP HANA connection.

The fake is intentionally thin — it does not validate SQL, enforce transactions,
or simulate rollback semantics. Its only job is to:

  - Return caller-seeded rows from SELECT-like calls.
  - Record every execute() call so tests can assert on SQL and writes.
  - Expose the same context-manager protocol as Engine.connect() / Engine.begin().

Usage
-----
    from tests.unit.fakes import FakeEngine

    engine = FakeEngine(rows=[{"REVISION": 2}])
    result = save_playbook(engine, "acme", "text", {}, user_confirmed=True)
    assert engine.executed_writes   # INSERT or UPDATE was issued

The FakeEngine is consumed by:
  - tests/unit/test_deduction_rules.py  (Task 2)
  - tests/unit/test_playbook_tools.py   (Task 3, future)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


# ---------------------------------------------------------------------------
# Result surface
# ---------------------------------------------------------------------------

class _FakeMappings:
    """Emulates the object returned by Result.mappings()."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def first(self) -> dict | None:
        """Return the first seeded row dict, or None if the result is empty."""
        return self._rows[0] if self._rows else None

    def all(self) -> list[dict]:
        """Return all seeded row dicts."""
        return list(self._rows)


class FakeResult:
    """
    Minimal stand-in for a SQLAlchemy CursorResult.

    Supports the subset of the Result API used by payment_advice modules:
      - .mappings().first() / .mappings().all()
      - .scalar()        — first column value of the first row, or None
      - .rowcount        — len(seeded rows), used by DELETE result checks
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    @property
    def rowcount(self) -> int:
        """Number of seeded rows (proxy for affected-row count in write operations)."""
        return len(self._rows)

    def scalar(self) -> Any:
        """Return the first column value of the first row, or None if empty."""
        if not self._rows:
            return None
        return next(iter(self._rows[0].values()))

    def mappings(self) -> _FakeMappings:
        """Return a mappings view over the seeded rows."""
        return _FakeMappings(self._rows)


# ---------------------------------------------------------------------------
# Connection surface
# ---------------------------------------------------------------------------

class _FakeConnection:
    """
    Stand-in for a SQLAlchemy Connection.

    Every execute() call:
      - Records the SQL string in engine.last_sql and params in engine.last_params.
      - If the SQL contains INSERT, UPDATE, or DELETE (case-insensitive), appends
        the SQL string to engine.executed_writes.
      - Always returns a FakeResult seeded with the engine's rows.
    """

    def __init__(self, engine: "FakeEngine") -> None:
        self._engine = engine

    def execute(self, statement: Any, params: dict | None = None) -> FakeResult:
        """
        Record the call and return a result backed by the engine's seeded rows.

        Args:
            statement: Any object whose str() yields the SQL text (e.g. sqlalchemy.text()).
            params:    Optional parameter dict passed alongside the statement.

        Returns:
            FakeResult seeded with FakeEngine._rows.
        """
        sql = str(statement)
        self._engine.last_sql = sql
        self._engine.last_params = params

        # Classify as a write if any DML keyword is present.
        sql_upper = sql.upper()
        if any(kw in sql_upper for kw in ("INSERT", "UPDATE", "DELETE")):
            self._engine.executed_writes.append(sql)

        return FakeResult(self._engine._rows)


# ---------------------------------------------------------------------------
# Engine surface
# ---------------------------------------------------------------------------

class FakeEngine:
    """
    Minimal SQLAlchemy Engine fake for offline unit tests.

    Emulates Engine.connect() and Engine.begin() as context managers that each
    yield a _FakeConnection. All execute() calls are recorded on the engine so
    tests can assert on SQL text and whether writes occurred.

    Args:
        rows: Seeded rows that the fake SELECT results will return.  Each element
              must be a plain dict keyed by column name.  Defaults to [] (no row).

    Attributes:
        last_sql (str | None):       SQL text of the most recent execute() call.
        last_params (dict | None):   Params dict of the most recent execute() call.
        executed_writes (list[str]): SQL texts of every INSERT/UPDATE/DELETE executed.
    """

    def __init__(self, rows: list[dict] | None = None) -> None:
        # Seeded rows returned by every FakeResult produced from this engine.
        self._rows: list[dict] = rows if rows is not None else []

        # Inspection attributes — cleared on construction, updated by _FakeConnection.
        self.last_sql: str | None = None
        self.last_params: dict | None = None
        self.executed_writes: list[str] = []

    @contextmanager
    def connect(self):
        """
        Context manager emulating Engine.connect().

        Yields a _FakeConnection.  Use for read-only operations (SELECT).
        """
        yield _FakeConnection(self)

    @contextmanager
    def begin(self):
        """
        Context manager emulating Engine.begin().

        Yields a _FakeConnection.  Use for transactional writes (INSERT/UPDATE/DELETE).
        Auto-commits on clean exit (no rollback simulation).
        """
        yield _FakeConnection(self)
