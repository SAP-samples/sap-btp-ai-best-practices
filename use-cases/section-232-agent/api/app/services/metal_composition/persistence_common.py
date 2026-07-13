"""Shared HANA persistence helpers for metal composition stores."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.utils.hana import HANAConnection


def quote_identifier(identifier: str) -> str:
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def qualified_table(table: str, schema: Optional[str]) -> str:
    if schema:
        return f"{quote_identifier(schema)}.{quote_identifier(table)}"
    return quote_identifier(table)


def ensure_index(
    connection: HANAConnection,
    table: str,
    *,
    schema: Optional[str],
    columns: Sequence[str],
) -> None:
    if not columns:
        return

    ensure = getattr(connection, "_ensure_index", None)
    if callable(ensure):
        ensure(table, schema=schema, columns=tuple(columns))
        return

    index_name = f"IDX_{table}_{'_'.join(columns)}"
    quoted_columns = ", ".join(quote_identifier(column) for column in columns)
    try:
        connection.execute(
            f"CREATE INDEX {quote_identifier(index_name)} "
            f"ON {qualified_table(table, schema)} ({quoted_columns})"
        )
    except Exception:
        pass


def fetch_rows(
    connection: HANAConnection,
    sql: str,
    params: Sequence[object] | None = None,
) -> List[Dict[str, object]]:
    with connection.cursor() as cursor:
        cursor.execute(sql, list(params or []))
        rows = cursor.fetchall()
        columns = [str(description[0]).lower() for description in (cursor.description or [])]
    return [
        {columns[index]: value for index, value in enumerate(row)}
        for row in rows
    ]


def column_exists(
    connection: HANAConnection,
    table: str,
    column: str,
    *,
    schema: Optional[str],
) -> bool:
    connection_column_exists = getattr(connection, "column_exists", None)
    if callable(connection_column_exists):
        try:
            return bool(connection_column_exists(table, column, schema=schema))
        except TypeError:
            return bool(connection_column_exists(table, column))
        except Exception:
            return False

    table_columns = getattr(connection, "table_columns", None)
    if callable(table_columns):
        try:
            columns = table_columns(table, schema=schema)
        except TypeError:
            columns = table_columns(table)
        except Exception:
            columns = []
        return column in {str(value) for value in (columns or [])}

    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {qualified_table(table, schema)} WHERE 1 = 0")
            columns = [str(description[0]) for description in (cursor.description or [])]
    except Exception:
        return False
    return column in columns


def ensure_column(
    connection: HANAConnection,
    table: str,
    column: str,
    definition: str,
    *,
    schema: Optional[str],
) -> None:
    if column_exists(connection, table, column, schema=schema):
        return

    table_name = qualified_table(table, schema)
    statements = [
        f"ALTER TABLE {table_name} ADD COLUMN {quote_identifier(column)} {definition}",
        f"ALTER TABLE {table_name} ADD ({quote_identifier(column)} {definition})",
    ]
    for statement in statements:
        try:
            connection.execute(statement)
            return
        except Exception:
            continue


def execute_compatible_update(
    connection: HANAConnection,
    sql: str,
    *,
    fallback_sql: str | None = None,
) -> None:
    try:
        connection.execute(sql)
        return
    except Exception:
        if not fallback_sql:
            raise
    connection.execute(fallback_sql)
