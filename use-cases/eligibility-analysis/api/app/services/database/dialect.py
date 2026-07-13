"""
SQL Dialect Helpers

Pure functions that return backend-specific SQL fragments.
"""

from .backend import get_backend


def sql_group_concat_distinct(column: str) -> str:
    if get_backend().is_hana:
        return f"STRING_AGG(DISTINCT {column}, ',')"
    return f"GROUP_CONCAT(DISTINCT {column})"


def sql_date(column: str) -> str:
    if get_backend().is_hana:
        return f"TO_DATE({column})"
    return f"DATE({column})"


def sql_reset_identity(table: str, column: str) -> str:
    if get_backend().is_hana:
        return (
            f"ALTER TABLE {table} ALTER "
            f"({column} BIGINT GENERATED ALWAYS AS IDENTITY (RESTART WITH 1))"
        )
    return f"DELETE FROM sqlite_sequence WHERE name='{table}'"


def sql_auto_id_column(column: str) -> str:
    if get_backend().is_hana:
        return f"{column} BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
    return f"{column} INTEGER PRIMARY KEY AUTOINCREMENT"


def sql_add_column(table: str, column: str, col_type: str) -> str:
    if get_backend().is_hana:
        return f"ALTER TABLE {table} ADD ({column} {col_type})"
    return f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
