"""Detect rows in tables this app does not own that still reference ECN rows.

Why this exists
---------------
The ECN deletion paths (``HistoryRepository.delete_ecns_if_idle`` and
``HistoryRepository.clear_history_if_idle``) delete every table the app owns
in a foreign-key-safe order.  A HANA schema can, however, contain extra tables
created outside this codebase (for example leftovers of a removed feature)
that declare a ``RESTRICT`` foreign key on ``ECN_IMPACT_ECN``.  Deleting an ECN
that such a table still references fails with HANA error 462 halfway through
the transaction.

``find_external_ecn_references`` runs inside the deletion transaction, before
any DELETE, and reports those tables so the caller can refuse the deletion with
a clear message instead of an internal server error.  It never deletes or
changes data in tables the app does not own.

SQL safety
----------
Catalog names read from ``REFERENTIAL_CONSTRAINTS`` are only interpolated into
SQL after they match ``^[A-Z0-9_]{1,127}$``.  A name outside that pattern is
reported as blocking without counting its rows.  ECN IDs are bound parameters.

Usage::

    with engine.begin() as connection:
        blocking = find_external_ecn_references(connection, tables, ecn_ids)
        if blocking:
            raise ExternalEcnReferenceError(blocking)
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import bindparam, text

from app.persistence.codec import normalize_row_keys
from app.persistence.table_contracts import PersistenceTableNames


# Plain upper-case HANA identifiers only; anything else is never interpolated.
_SAFE_CATALOG_NAME = re.compile(r"^[A-Z0-9_]{1,127}$")

# Every foreign key in the connected schema that points at the ECN table.
_REFERENCING_FOREIGN_KEYS_SQL = (
    'SELECT DISTINCT "TABLE_NAME", "COLUMN_NAME" FROM "REFERENTIAL_CONSTRAINTS" '
    'WHERE "SCHEMA_NAME" = CURRENT_SCHEMA AND "REFERENCED_TABLE_NAME" = :ecn_table '
    'ORDER BY "TABLE_NAME", "COLUMN_NAME"'
)


@dataclass(frozen=True)
class ExternalEcnReference:
    """One table outside the app's contracts that still references ECN rows.

    Args:
        table_name: Physical table name as stored in the HANA catalog.
        column_name: Foreign-key column pointing at ``ECN_ROW_ID``.
        row_count: Number of referencing rows in scope, or ``None`` when the
            catalog name was not safe to query and the rows were not counted.
    """

    table_name: str
    column_name: str
    row_count: int | None


class ExternalEcnReferenceError(Exception):
    """Raised when tables the app does not own block an ECN deletion.

    The deletion transaction is left without any DELETE executed, so raising
    this error rolls back nothing but the read-only checks.

    Args:
        references: Every blocking table with its referencing row count.
    """

    def __init__(self, references: Sequence[ExternalEcnReference]) -> None:
        """Store the blocking references and build a short diagnostic message."""
        self.references = tuple(references)
        super().__init__(describe_external_references(self.references))


def describe_external_references(references: Sequence[ExternalEcnReference]) -> str:
    """Return a human-readable list such as ``"T1 (2 rows), T2 (rows not counted)"``.

    Args:
        references: Blocking references to describe.

    Returns:
        Comma-separated table names with their row counts.
    """
    parts = []
    for reference in references:
        if reference.row_count is None:
            parts.append(f"{reference.table_name} (rows not counted)")
        else:
            noun = "row" if reference.row_count == 1 else "rows"
            parts.append(f"{reference.table_name} ({reference.row_count} {noun})")
    return ", ".join(parts)


def _owned_table_names(tables: PersistenceTableNames) -> frozenset[str]:
    """Return every physical table name the app's contracts own.

    Args:
        tables: The active table-name set (production or test-suffixed).

    Returns:
        All field values of ``PersistenceTableNames``.
    """
    return frozenset(
        str(getattr(tables, field.name)) for field in dataclasses.fields(tables)
    )


def find_external_ecn_references(
    connection: object,
    tables: PersistenceTableNames,
    ecn_row_ids: Sequence[str] | None,
) -> list[ExternalEcnReference]:
    """List tables the app does not own whose rows still reference the given ECNs.

    Steps:

    1. Read every foreign key in ``CURRENT_SCHEMA`` that references
       ``tables.ecn``.
    2. Drop the tables listed in ``PersistenceTableNames``; the deletion code
       already handles those.
    3. For each remaining table, count the rows whose foreign-key column is in
       ``ecn_row_ids`` (or all non-null rows when ``ecn_row_ids`` is ``None``,
       which is the whole-history reset).

    Args:
        connection: Open SQLAlchemy connection inside the deletion transaction.
        tables: Physical table names; their values define what the app owns.
        ecn_row_ids: ECN row IDs about to be deleted, or ``None`` for all rows.

    Returns:
        References with at least one row (or an uncounted unsafe name), sorted
        by table name.  An empty list means the deletion can proceed.
    """
    owned = _owned_table_names(tables)
    foreign_keys = [
        normalize_row_keys(row)
        for row in connection.execute(  # type: ignore[attr-defined]
            text(_REFERENCING_FOREIGN_KEYS_SQL), {"ecn_table": tables.ecn}
        ).mappings().all()
    ]

    blocking: list[ExternalEcnReference] = []
    for foreign_key in foreign_keys:
        table_name = str(foreign_key["TABLE_NAME"])
        column_name = str(foreign_key["COLUMN_NAME"])
        if table_name in owned:
            continue
        if not (_SAFE_CATALOG_NAME.match(table_name) and _SAFE_CATALOG_NAME.match(column_name)):
            # Unsafe to interpolate: report it as blocking without querying it.
            blocking.append(ExternalEcnReference(table_name, column_name, None))
            continue
        row_count = _count_referencing_rows(connection, table_name, column_name, ecn_row_ids)
        if row_count:
            blocking.append(ExternalEcnReference(table_name, column_name, row_count))
    return sorted(blocking, key=lambda reference: reference.table_name)


def _count_referencing_rows(
    connection: object,
    table_name: str,
    column_name: str,
    ecn_row_ids: Sequence[str] | None,
) -> int:
    """Count rows of one external table that reference the ECNs in scope.

    Args:
        connection: Open SQLAlchemy connection.
        table_name: Catalog table name already validated as safe.
        column_name: Catalog column name already validated as safe.
        ecn_row_ids: ECN IDs in scope, or ``None`` to count every non-null row.

    Returns:
        Number of referencing rows.
    """
    base = f'SELECT COUNT(1) FROM "{table_name}" WHERE "{column_name}" '
    if ecn_row_ids is None:
        statement = text(base + "IS NOT NULL")
        params: dict[str, object] = {}
    else:
        statement = text(base + "IN :ids").bindparams(bindparam("ids", expanding=True))
        params = {"ids": tuple(ecn_row_ids)}
    return int(connection.execute(statement, params).scalar_one())  # type: ignore[attr-defined]
