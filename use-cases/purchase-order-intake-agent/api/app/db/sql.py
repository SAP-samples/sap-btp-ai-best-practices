"""Small helpers that turn HANA query results into plain Python dicts.

* Column names are upper-cased (the HANA SQLAlchemy dialect lower-cases them) so rows keep
  SAP-style keys (``KUNNR``, ``ORDER_ID``).
* ``Decimal`` values become ``float``; LOB handles (BLOB/NCLOB) are read into bytes/str.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .engine import get_engine


def _value(value: Any) -> Any:
    """Convert one database value to a plain Python value."""
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "read"):  # hdbcli LOB handle
        return value.read()
    return value


def fetch_all(sql: str, conn: Connection | None = None, **params: Any) -> list[dict[str, Any]]:
    """Run a SELECT and return all rows as dicts.

    Args:
        sql: SQL text with named parameters.
        conn: Existing connection (inside a transaction); a pooled one is used when omitted.
        **params: Bind parameters.

    Returns:
        Rows as dicts with upper-case keys.
    """
    if conn is None:
        with get_engine().connect() as own:
            return fetch_all(sql, own, **params)
    result = conn.execute(text(sql), params)
    return [{key.upper(): _value(value) for key, value in row._mapping.items()} for row in result]


def fetch_one(sql: str, conn: Connection | None = None, **params: Any) -> dict[str, Any] | None:
    """Run a SELECT and return the first row, or None."""
    rows = fetch_all(sql, conn, **params)
    return rows[0] if rows else None


def insert_row(conn: Connection, table: str, row: dict[str, Any]) -> None:
    """Insert one row; keys are column names (quoted, so they must match the DDL exactly).

    Args:
        conn: Connection inside a transaction.
        table: Target table.
        row: Column -> value mapping.
    """
    columns = list(row)
    conn.execute(
        text(f'INSERT INTO "{table}" ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in columns)}) '
             f'VALUES ({", ".join(":" + c for c in columns)})'),
        row,
    )


def insert_many(conn: Connection, table: str, rows: list[dict[str, Any]]) -> None:
    """Insert many rows in one round trip (executemany); rows are padded to the same columns.

    Args:
        conn: Connection inside a transaction.
        table: Target table.
        rows: Column -> value mappings (missing columns are inserted as NULL).
    """
    if not rows:
        return
    columns = list(dict.fromkeys(column for row in rows for column in row))
    conn.execute(
        text(f'INSERT INTO "{table}" ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in columns)}) '
             f'VALUES ({", ".join(":" + c for c in columns)})'),
        [{column: row.get(column) for column in columns} for row in rows],
    )
