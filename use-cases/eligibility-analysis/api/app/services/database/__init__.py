"""
Database Abstraction Layer

Provides a dual-backend (SQLite / SAP HANA Cloud) database interface.
Backend is selected automatically based on environment variables.
"""

from .backend import BackendType, DatabaseBackend, get_backend
from .dialect import (
    sql_add_column,
    sql_auto_id_column,
    sql_date,
    sql_group_concat_distinct,
    sql_reset_identity,
)
from .parsing import parse_date, parse_datetime, parse_decimal
from .types import map_type

__all__ = [
    "BackendType",
    "DatabaseBackend",
    "get_backend",
    "map_type",
    "parse_date",
    "parse_datetime",
    "parse_decimal",
    "sql_add_column",
    "sql_auto_id_column",
    "sql_date",
    "sql_group_concat_distinct",
    "sql_reset_identity",
]
