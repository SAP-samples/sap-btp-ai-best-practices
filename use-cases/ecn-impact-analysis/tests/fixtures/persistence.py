"""Fake persistence helpers for unit and integration tests.

This module provides recording fakes for the persistence layer so that unit
tests can verify SQL construction and engine creation without a live HANA
instance.

Classes
-------
``FakeEngineFactory``
    Records ``sqlalchemy.create_engine`` calls.  Used by hana engine tests.
``RecordingConnection``
    Queues scalar results and records ``execute()`` calls; supports
    ``current_schema`` and ``table_exists`` assertions.
``FakeInspector``
    Returns Inspector-shaped dicts for a single pre-seeded table; used to
    drive ``validate_table_contract`` tests.
``FakeEngine`` / ``_FakeEngineConnection`` / ``_FakeSchemaInspector``
    Full engine fake for ``ensure_schema`` behavioral tests.
``RecordingEngine``
    Minimal engine fake for repository unit tests (Tasks 5–8+).  Supports
    ``queue_scalar``, ``queue_mappings``, ``queue_mapping``,
    ``queue_rowcount``, and ``queue_exception`` for all read/write query
    patterns used by ``ArtifactRepository``, ``UploadRepository``,
    ``EcnRepository``, and ``AnalysisRepository``.

    Task 7 (EcnRepository) uses the same ``RecordingEngine`` as earlier tasks:

    * ``queue_scalar(2)`` then ``queue_mappings([...])`` for ``list_ecns``
      (count query + list query on one ``connect()`` context).
    * ``queue_mapping({"EVIDENCE_JSON": ...})`` for ``get_evidence``.
    * ``queue_mappings([{"JOB_ID": ...}, ...])`` for ``list_queued_job_ids``.
    * ``queue_mapping(None)`` simulates a missing row (``one_or_none()`` → ``None``).

Do not read from ``data/`` in this module; all test data is synthetic.

Example usage::

    from tests.fixtures.persistence import FakeEngineFactory, RecordingConnection

    def test_engine_creates_correct_url(monkeypatch):
        factory = FakeEngineFactory()
        monkeypatch.setattr("app.persistence.hana.create_engine", factory)
        create_hana_engine({...})
        assert factory.calls[0].url.drivername == "hana"
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError


# ===========================================================================
# Task-3 helpers: engine-factory recording
# ===========================================================================


@dataclass
class RecordedEngineCall:
    """Hold the arguments of one ``sqlalchemy.create_engine`` call.

    Attributes:
        url: The SQLAlchemy ``URL`` object passed to ``create_engine``.
        kwargs: Additional keyword arguments (e.g. ``pool_pre_ping``).
    """

    url: URL
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeEngineFactory:
    """Record calls to ``sqlalchemy.create_engine`` for unit-test assertions.

    Patch ``app.persistence.hana.create_engine`` with an instance of this
    class.  After calling the function under test, inspect ``calls`` to
    verify the URL structure and pool settings that were used.

    The factory does NOT attempt a real database connection; it returns a
    plain object sentinel so that callers requiring only construction-time
    assertions do not need a live HANA instance.

    Example::

        factory = FakeEngineFactory()
        monkeypatch.setattr("app.persistence.hana.create_engine", factory)
        create_hana_engine({"HANA_ADDRESS": "h", "HANA_PORT": "443",
                            "HANA_USER": "u", "HANA_PASSWORD": "p"})
        assert factory.calls[0].url.drivername == "hana"
        assert factory.calls[0].kwargs["pool_pre_ping"] is True
    """

    def __init__(self) -> None:
        """Initialise with an empty call log."""
        self.calls: list[RecordedEngineCall] = []

    def __call__(self, url: URL, **kwargs: Any) -> Any:
        """Record one ``create_engine`` call and return a sentinel object.

        Args:
            url: The SQLAlchemy URL passed by the factory function.
            **kwargs: Keyword arguments (e.g. ``pool_pre_ping=True``).

        Returns:
            A plain ``object()`` sentinel; not a real SQLAlchemy ``Engine``.
        """
        self.calls.append(RecordedEngineCall(url=url, kwargs=dict(kwargs)))
        return object()


# ===========================================================================
# Task-4 helpers: schema recording and fakes
# ===========================================================================


# ---------------------------------------------------------------------------
# Minimal result wrappers used by RecordingConnection
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal result-set wrapper supporting ``scalar_one()`` and ``first()``.

    Args:
        value: The single value to return from both accessor methods.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one(self) -> Any:
        """Return the single pre-queued value.

        Returns:
            The value supplied at construction time.
        """
        return self._value

    def first(self) -> Any:
        """Return the single pre-queued value (mimics a row or None).

        Returns:
            The value supplied at construction time.
        """
        return self._value


# ---------------------------------------------------------------------------
# RecordingConnection
# ---------------------------------------------------------------------------


class RecordingConnection:
    """Record ``execute()`` calls and serve pre-queued scalar results.

    Used to test ``current_schema`` and ``table_exists`` without a live
    HANA instance.  Call ``queue_scalar`` before each ``execute`` that needs
    to return a value; inspect ``statements`` after to verify SQL text and
    parameters.

    Attributes:
        statements: List of ``(sql_text, params_dict)`` tuples recorded in
            order of execution.

    Example::

        conn = RecordingConnection()
        conn.queue_scalar("APP_RUNTIME")
        result = conn.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY"))
        assert result.scalar_one() == "APP_RUNTIME"
        assert conn.statements[0] == ("SELECT CURRENT_SCHEMA FROM DUMMY", {})
    """

    def __init__(self) -> None:
        """Initialise with empty queue and statement log."""
        self._queue: list[Any] = []
        self.statements: list[tuple[str, dict[str, Any]]] = []

    def queue_scalar(self, value: Any) -> None:
        """Enqueue one value to be returned by the next ``execute`` call.

        Args:
            value: The value that the next result's ``scalar_one()`` or
                ``first()`` should return.
        """
        self._queue.append(value)

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        """Record the statement and return the next queued result.

        Args:
            stmt: A SQLAlchemy ``text()`` clause or similar; ``str()`` is
                called to extract the SQL text.
            params: Optional parameter dict; stored alongside the SQL text.

        Returns:
            A ``_FakeResult`` wrapping the next queued value, or ``None``
            if the queue is empty.
        """
        self.statements.append((str(stmt), params or {}))
        value = self._queue.pop(0) if self._queue else None
        return _FakeResult(value)

    def commit(self) -> None:
        """No-op; provided so the connection is usable in tests that call commit."""


# ---------------------------------------------------------------------------
# Fake type objects for Inspector-shaped column dicts
# ---------------------------------------------------------------------------


def make_fake_type(type_name: str, **attrs: Any) -> Any:
    """Create a fake SQLAlchemy-like type object for inspector-shaped dicts.

    The returned object's ``__class__.__name__`` matches ``type_name`` so that
    ``_reflected_column`` in ``schema.py`` produces the correct comparison tuple
    without requiring a live HANA dialect.

    Args:
        type_name: The HANA type name, e.g. ``"NVARCHAR"`` or ``"BIGINT"``.
            This becomes the dynamic class name so ``obj.__class__.__name__``
            returns it exactly.
        **attrs: Optional attributes to attach (e.g. ``length=36``,
            ``precision=31``, ``scale=14``).

    Returns:
        An instance of a dynamically-created class named ``type_name`` with
        the supplied attributes set as class-level attributes.

    Example::

        t = make_fake_type("NVARCHAR", length=36)
        assert t.__class__.__name__ == "NVARCHAR"
        assert t.length == 36
        assert getattr(t, "precision", None) is None
    """
    cls = type(type_name, (), dict(attrs))
    return cls()


def col_to_reflected(col: Any) -> dict[str, Any]:
    """Convert a ``ColumnContract`` to an Inspector-shaped reflected column dict.

    Produces the dict format that the SQLAlchemy HANA inspector returns for
    ``get_columns``, using ``make_fake_type`` so that ``schema._reflected_column``
    yields a comparison tuple that matches ``_expected_column_tuple``.

    Args:
        col: A ``ColumnContract`` instance with ``name``, ``hana_type``,
            ``nullable``, ``length``, ``precision``, and ``scale`` attributes.

    Returns:
        A dict with keys ``name``, ``type``, and ``nullable``.
    """
    type_attrs: dict[str, Any] = {}
    if col.length is not None:
        type_attrs["length"] = col.length
    if col.precision is not None:
        type_attrs["precision"] = col.precision
    if col.scale is not None:
        type_attrs["scale"] = col.scale
    return {
        "name": col.name,
        "type": make_fake_type(col.hana_type, **type_attrs),
        "nullable": col.nullable,
    }


# ---------------------------------------------------------------------------
# FakeInspector — single-table, fully explicit
# ---------------------------------------------------------------------------


class FakeInspector:
    """Fake SQLAlchemy Inspector for ``validate_table_contract`` unit tests.

    Holds pre-seeded data for one table (by name) and exposes the four
    Inspector methods called by ``validate_table_contract``.  Tests mutate
    ``columns``, ``pks``, ``uqs``, or ``fks`` directly to inject mismatches.

    Attributes:
        connection: The ``RecordingConnection`` companion for this inspector.
        columns: Dict mapping table name → list of reflected column dicts.
    """

    def __init__(
        self,
        connection: RecordingConnection,
        columns: dict[str, list[dict[str, Any]]],
        pks: dict[str, list[str]],
        uqs: dict[str, list[list[str]]],
        fks: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Seed the inspector with pre-built data for all four Inspector methods.

        Args:
            connection: The companion recording connection.
            columns: Table-name → list of reflected column dicts.
            pks: Table-name → ordered list of PK column names.
            uqs: Table-name → list of UQ column-name lists.
            fks: Table-name → list of FK dicts (``constrained_columns``,
                ``referred_schema``, ``referred_table``, ``referred_columns``).
        """
        self.connection = connection
        self.columns: dict[str, list[dict[str, Any]]] = columns
        self._pks = pks
        self._uqs = uqs
        self._fks = fks

    def get_columns(self, table_name: str, schema: str | None = None) -> list[dict[str, Any]]:
        """Return the seeded column list for the given table.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            A copy of the seeded column list for the table.
        """
        return list(self.columns.get(table_name, []))

    def get_pk_constraint(self, table_name: str, schema: str | None = None) -> dict[str, Any]:
        """Return the seeded PK constraint dict for the given table.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            A dict with ``constrained_columns`` (list of str) and ``name``
            (``None``).
        """
        return {
            "constrained_columns": list(self._pks.get(table_name, [])),
            "name": None,
        }

    def get_unique_constraints(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the seeded UQ list for the given table.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            A list of dicts, each with ``column_names`` (list of str) and a
            generated ``name`` string.
        """
        return [
            {"column_names": list(uq), "name": f"gen_uq_{i}"}
            for i, uq in enumerate(self._uqs.get(table_name, []))
        ]

    def get_foreign_keys(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the seeded FK list for the given table.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            A copy of the seeded FK list for the table.
        """
        return list(self._fks.get(table_name, []))


def build_inspector_for(
    contract: Any,
    table_names: Any | None = None,
    schema: str = "APP_RUNTIME",
) -> FakeInspector:
    """Build a ``FakeInspector`` fully seeded from a single ``TableContract``.

    Converts every column to a reflected dict, and every FK to a resolved
    FK dict whose ``referred_table`` comes from ``table_names``.

    Args:
        contract: A ``TableContract`` instance.
        table_names: Optional ``PersistenceTableNames``; defaults to the
            production names.
        schema: The schema string; used only for FK ``referred_schema`` docs.

    Returns:
        A ``FakeInspector`` seeded with correct data for ``contract``.
    """
    from app.persistence.table_contracts import PersistenceTableNames

    names = table_names or PersistenceTableNames()
    cols = [col_to_reflected(c) for c in contract.columns]
    pks = {contract.name: list(contract.primary_key)}
    uqs = {contract.name: [list(uq) for uq in contract.unique_constraints]}
    fks = {
        contract.name: [
            {
                "constrained_columns": list(fk.columns),
                "referred_schema": None,
                "referred_table": getattr(names, fk.referenced_table_key),
                "referred_columns": list(fk.referenced_columns),
            }
            for fk in contract.foreign_keys
        ]
    }
    conn = RecordingConnection()
    return FakeInspector(conn, {contract.name: cols}, pks, uqs, fks)


# ===========================================================================
# FakeEngine for ensure_schema behavioral tests
# ===========================================================================


class _FakeEngineConnection:
    """Connection returned by ``FakeEngine.connect()`` context manager.

    Handles SELECT CURRENT_SCHEMA, SYS.TABLES existence checks, and DDL
    execution without touching a live HANA instance.

    Args:
        engine: The owning ``FakeEngine`` whose state is read and updated.
    """

    def __init__(self, engine: "FakeEngine") -> None:
        self._engine = engine

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        """Dispatch the statement to the appropriate fake handler.

        * ``SELECT CURRENT_SCHEMA`` → returns ``engine.schema``.
        * ``SYS.TABLES`` → checks ``existing_tables | _created_tables``.
        * DDL → records in ``ddl_statements``; raises for ``_ddl_failures``.

        Args:
            stmt: A SQLAlchemy ``text()`` clause.
            params: Optional parameter dict.

        Returns:
            A ``_FakeResult`` appropriate for the statement type.

        Raises:
            SQLAlchemyError: When the table name extracted from a DDL
                statement appears in ``engine._ddl_failures``.
        """
        sql = str(stmt)
        if "CURRENT_SCHEMA" in sql:
            return _FakeResult(self._engine.schema)

        if "SYS.TABLES" in sql:
            table_name = (params or {}).get("table_name", "")
            all_known = self._engine.existing_tables | self._engine._created_tables
            return _FakeResult(1 if table_name in all_known else None)

        # DDL — extract the quoted table name from CREATE COLUMN TABLE "..." (...)
        match = re.search(r'"(ECN_IMPACT[^"]+)"', sql)
        tname = match.group(1) if match else None
        if tname and tname in self._engine._ddl_failures:
            raise SQLAlchemyError(f"Simulated DDL failure for {tname!r}")
        self._engine.ddl_statements.append(sql)
        if tname:
            self._engine._created_tables.add(tname)
        return _FakeResult(None)

    def commit(self) -> None:
        """No-op; HANA DDL auto-commits; the fake mirrors that model."""


class _FakeSchemaInspector:
    """Inspector returned by monkeypatched ``inspect()`` during FakeEngine tests.

    Returns correct contract data for all tables, with optional column drops
    for tables registered as incompatible via ``FakeEngine.mark_incompatible``.

    Args:
        engine: The owning ``FakeEngine`` providing incompatibility state.
        table_names: ``PersistenceTableNames`` used to build expected contracts
            and resolve FK targets.
    """

    def __init__(self, engine: "FakeEngine", table_names: Any) -> None:
        from app.persistence.table_contracts import build_table_contracts

        self._engine = engine
        self._contracts = {c.name: c for c in build_table_contracts(table_names)}
        self._table_names = table_names

    def get_columns(self, table_name: str, schema: str | None = None) -> list[dict[str, Any]]:
        """Return reflected column dicts, optionally with one column dropped.

        Increments ``engine.inspection_count[table_name]`` so tests can assert
        that re-inspection occurs after table creation.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            List of reflected column dicts for the table.
        """
        self._engine.inspection_count[table_name] = (
            self._engine.inspection_count.get(table_name, 0) + 1
        )
        contract = self._contracts.get(table_name)
        if contract is None:
            return []
        cols = [col_to_reflected(c) for c in contract.columns]
        # Drop column if table is marked incompatible
        if table_name in self._engine._incompatible_tables:
            drop_col = self._engine._incompatible_tables[table_name]
            cols = [c for c in cols if c["name"] != drop_col]
        return cols

    def get_pk_constraint(self, table_name: str, schema: str | None = None) -> dict[str, Any]:
        """Return the contract's PK columns.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            Dict with ``constrained_columns`` and ``name=None``.
        """
        contract = self._contracts.get(table_name)
        if contract is None:
            return {"constrained_columns": [], "name": None}
        return {"constrained_columns": list(contract.primary_key), "name": None}

    def get_unique_constraints(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the contract's UQ constraints.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            List of dicts with ``column_names`` and generated ``name``.
        """
        contract = self._contracts.get(table_name)
        if contract is None:
            return []
        return [
            {"column_names": list(uq), "name": f"gen_uq_{i}"}
            for i, uq in enumerate(contract.unique_constraints)
        ]

    def get_foreign_keys(
        self, table_name: str, schema: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the contract's FK list with physical table names resolved.

        Args:
            table_name: Physical HANA table name.
            schema: Ignored; present for interface compatibility.

        Returns:
            List of FK dicts with ``constrained_columns``, ``referred_schema``
            (``None``), ``referred_table``, and ``referred_columns``.
        """
        contract = self._contracts.get(table_name)
        if contract is None:
            return []
        return [
            {
                "constrained_columns": list(fk.columns),
                "referred_schema": None,
                "referred_table": getattr(self._table_names, fk.referenced_table_key),
                "referred_columns": list(fk.referenced_columns),
            }
            for fk in contract.foreign_keys
        ]


# ===========================================================================
# Task-5 helpers: transaction-recording engine for artifact repository tests
# ===========================================================================


_UNSET: object = object()


class _MappingResult:
    """Minimal SQLAlchemy result fake supporting mappings/one_or_none/all/scalar_one.

    The recording engine queues row batches, scalar values, or rowcounts; this
    wrapper exposes the same API surface that repository methods call so that
    unit tests can run without a live HANA connection.

    ``rowcount`` mirrors the SQLAlchemy ``CursorResult.rowcount`` attribute used
    by conditional UPDATE statements (e.g. claim and provider-outcome writes) to
    detect whether a row was actually matched.

    Args:
        rows: Pre-queued list of row dicts to return.
        scalar: Pre-queued scalar value; ``_UNSET`` when this result holds rows.
        rowcount: Pre-queued rowcount integer; defaults to ``0`` when not set.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        scalar: Any = _UNSET,
        rowcount: int = 0,
    ) -> None:
        self._rows = rows
        self._scalar = scalar
        self.rowcount = rowcount

    def mappings(self) -> "_MappingResult":
        """Return self; mirrors SQLAlchemy's chaining pattern."""
        return self

    def one_or_none(self) -> dict[str, Any] | None:
        """Return the first queued row or None when the queue was empty.

        Returns:
            The first row dict, or ``None``.
        """
        return self._rows[0] if self._rows else None

    def all(self) -> list[dict[str, Any]]:
        """Return all queued rows as a list.

        Returns:
            A copy of the queued row list.
        """
        return list(self._rows)

    def scalar_one(self) -> Any:
        """Return the pre-queued scalar value.

        Used by ``COALESCE(MAX(...))`` queries in ``publish_upload``.

        Returns:
            The scalar value supplied at construction time.
        """
        return self._scalar


class _RecordingConnection:
    """Connection-like object returned by RecordingEngine context managers.

    Records every ``execute`` call and delegates result retrieval to the
    owning engine's unified queue so that consecutive queries drain the queue
    in order.  Supports both row-mapping results and scalar results.

    Args:
        engine: The owning ``RecordingEngine`` that holds the unified queue.
    """

    def __init__(self, engine: "RecordingEngine") -> None:
        self._engine = engine

    def execute(self, stmt: Any, params: Any = None) -> _MappingResult:
        """Record the statement and return the next queued result.

        The result may be a row-mapping batch (for SELECT queries that use
        ``mappings().one_or_none()`` or ``mappings().all()``) or a scalar
        value (for ``scalar_one()`` calls such as ``COALESCE(MAX(...))+1``).

        Args:
            stmt: A SQLAlchemy ``text()`` clause; ``str()`` extracts the SQL.
            params: Optional bound-parameter dict or list of dicts for
                multi-row ``executemany`` style inserts.

        Returns:
            A ``_MappingResult`` wrapping the next dequeued result.
        """
        self._engine.executions.append((str(stmt), params if params is not None else {}))
        return self._engine._pop_result()

    def commit(self) -> None:
        """No-op; mirrors the SQLAlchemy Connection commit API."""


class RecordingEngine:
    """Minimal engine fake for repository unit tests.

    Counts ``connect()`` and ``begin()`` entries, records every SQL
    statement and its bound parameters, and serves pre-queued results
    so that each repository method's SQL and transaction boundaries can be
    asserted without a live HANA instance.

    The unified queue supports four entry kinds:

    * ``("rows", list[dict])`` — returned via ``mappings().one_or_none()``
      or ``mappings().all()``.
    * ``("scalar", value)`` — returned via ``scalar_one()``.
    * ``("rowcount", int)`` — returned via ``.rowcount``; used to simulate
      the affected-row count of conditional UPDATE statements.
    * ``("exception", exc)`` — raised when the next ``execute()`` is called,
      simulating database-level errors such as ``IntegrityError``.

    Attributes:
        begin_count: Number of times ``begin()`` was entered.
        connect_count: Number of times ``connect()`` was entered.
        executions: Ordered list of ``(sql_text, params)`` tuples.

    Example::

        engine = RecordingEngine()
        engine.queue_mappings([{"ARTIFACT_ID": "..."}])
        repo = ArtifactRepository(engine)
        artifact = repo.get_artifact("00000000-0000-4000-8000-000000000001")
        statement, params = engine.executions[0]
        assert "ARTIFACT_ID" in statement
    """

    def __init__(self) -> None:
        """Initialise with zero counters and empty unified queue."""
        self.begin_count: int = 0
        self.connect_count: int = 0
        self.rollback_count: int = 0
        self.executions: list[tuple[str, Any]] = []
        self._queue: list[tuple[str, Any]] = []

    def queue_mapping(self, row: dict[str, Any] | None) -> None:
        """Enqueue one row dict (or None) to be returned by the next execute call.

        Passing ``None`` causes ``one_or_none()`` to return ``None``, which
        simulates "no row found" for preflight idempotency checks.

        Args:
            row: A column-name → value mapping, or ``None`` for an empty result.
        """
        self._queue.append(("rows", [row]))

    def queue_mappings(self, rows: list[dict[str, Any]]) -> None:
        """Enqueue a list of row dicts to be returned by the next execute call.

        Args:
            rows: A list of column-name → value mappings to return together.
        """
        self._queue.append(("rows", list(rows)))

    def queue_scalar(self, value: Any) -> None:
        """Enqueue a scalar value to be returned by the next ``scalar_one()`` call.

        Used to simulate ``COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1`` results.

        Args:
            value: The scalar value (e.g. ``1`` for the first publication order).
        """
        self._queue.append(("scalar", value))

    def queue_exception(self, exc: Exception) -> None:
        """Enqueue an exception to be raised by the next ``execute()`` call.

        Used to simulate database-level errors such as ``IntegrityError`` on a
        duplicate ``REQUEST_ID`` (idempotency race condition tests).

        Args:
            exc: The exception instance to raise.
        """
        self._queue.append(("exception", exc))

    def queue_rowcount(self, n: int) -> None:
        """Enqueue an integer rowcount to be returned by the next ``execute()`` call.

        Used to simulate the rowcount returned by an UPDATE statement so that
        conditional claim and provider-outcome writes can be tested without a
        live HANA instance.  A rowcount of ``1`` simulates a successful conditional
        update; ``0`` simulates a race loss or a state mismatch.

        Args:
            n: The integer rowcount the next result's ``.rowcount`` should return.
        """
        self._queue.append(("rowcount", n))

    def queue_rowcounts(self, counts: list[int]) -> None:
        """Enqueue multiple integer rowcounts in order for multi-statement transactions.

        Convenience wrapper that calls ``queue_rowcount`` for each element in
        ``counts``.  Used to set up the result sequence for methods that execute
        several conditional UPDATE or DELETE statements in a single transaction.

        Args:
            counts: A list of integer rowcounts to enqueue in order; each element
                is the simulated rowcount of the corresponding statement.
        """
        for n in counts:
            self._queue.append(("rowcount", n))

    def _pop_result(self) -> _MappingResult:
        """Dequeue and return the next result, or raise the queued exception.

        Returns:
            A ``_MappingResult`` wrapping the next dequeued rows, scalar value,
            or rowcount depending on the queued kind.

        Raises:
            Exception: When the next queued entry is an exception.
        """
        if not self._queue:
            return _MappingResult([])
        kind, value = self._queue.pop(0)
        if kind == "exception":
            raise value
        if kind == "scalar":
            return _MappingResult([], scalar=value)
        if kind == "rowcount":
            return _MappingResult([], rowcount=value)
        return _MappingResult(value)

    @contextmanager
    def connect(self):  # type: ignore[override]
        """Yield a recording connection and increment the connect counter.

        Yields:
            A ``_RecordingConnection`` bound to this engine.
        """
        self.connect_count += 1
        yield _RecordingConnection(self)

    @contextmanager
    def begin(self):  # type: ignore[override]
        """Yield a recording connection within a transaction and count it.

        Increments ``rollback_count`` when the context exits with an unhandled
        exception, mirroring the rollback that a real SQLAlchemy engine would
        issue.  This allows tests to assert that the repository correctly rolls
        back on mismatch or constraint failure.

        Yields:
            A ``_RecordingConnection`` bound to this engine.

        Raises:
            Exception: Re-raises any exception that propagates out of the block.
        """
        self.begin_count += 1
        try:
            yield _RecordingConnection(self)
        except Exception:
            self.rollback_count += 1
            raise


class FakeEngine:
    """Fake SQLAlchemy engine for ``ensure_schema`` behavioral unit tests.

    Controls which tables appear to exist, which are incompatible, and which
    DDL statements should raise.  Monkeypatch ``app.persistence.schema.inspect``
    to return ``_FakeSchemaInspector(engine, table_names)`` alongside.

    Attributes:
        schema: The schema string returned by ``SELECT CURRENT_SCHEMA``.
        existing_tables: Physical table names that the fake treats as
            already present before any DDL runs.
        ddl_statements: Ordered list of DDL strings executed via the engine.
        inspection_count: Dict mapping table name → number of ``get_columns``
            calls; used to assert re-inspection after creation.

    Example::

        engine = FakeEngine()
        engine.existing_tables = {PersistenceTableNames().source_artifact}
        engine.mark_incompatible(PersistenceTableNames().source_artifact, "FILE_NAME")
        # ensure_schema raises StartupConfigurationError; no DDL runs.
    """

    def __init__(self) -> None:
        """Initialise with empty state and default schema."""
        self.schema: str = "APP_RUNTIME"
        self.existing_tables: set[str] = set()
        self.ddl_statements: list[str] = []
        self._incompatible_tables: dict[str, str] = {}
        self._ddl_failures: set[str] = set()
        self._created_tables: set[str] = set()
        self.inspection_count: dict[str, int] = {}

    def mark_incompatible(self, table_name: str, column_to_drop: str) -> None:
        """Mark a table so that validation returns a column mismatch.

        When ``validate_table_contract`` is called for ``table_name``,
        ``_FakeSchemaInspector.get_columns`` omits ``column_to_drop``,
        causing the column tuple comparison to fail.

        Args:
            table_name: Physical HANA table name to corrupt.
            column_to_drop: Name of the column to omit from the reflected data.
        """
        self._incompatible_tables[table_name] = column_to_drop

    def mark_ddl_failure(self, table_name: str) -> None:
        """Mark a table so that its CREATE COLUMN TABLE raises SQLAlchemyError.

        Args:
            table_name: Physical HANA table name whose DDL should fail.
        """
        self._ddl_failures.add(table_name)

    @contextmanager
    def connect(self):  # type: ignore[override]
        """Yield a ``_FakeEngineConnection`` for use as a context manager.

        Yields:
            A ``_FakeEngineConnection`` bound to this engine.
        """
        yield _FakeEngineConnection(self)

    def dispose(self) -> None:
        """No-op; provided for interface compatibility with real Engine."""
