"""
DDL Type Mapping

Maps abstract type names to backend-specific SQL column types.
Used by _ensure_additive_columns() for ALTER TABLE ADD COLUMN statements.
"""

from .backend import get_backend

_TYPE_MAP = {
    "TEXT": ("TEXT", "NVARCHAR(5000)"),
    "NCLOB": ("TEXT", "NCLOB"),
    "INTEGER": ("INTEGER", "INTEGER"),
    "REAL": ("REAL", "DOUBLE"),
    "DOUBLE": ("REAL", "DOUBLE"),
    "TIMESTAMP": ("TIMESTAMP", "TIMESTAMP"),
    "BOOLEAN": ("BOOLEAN", "BOOLEAN"),
    "BIGINT": ("INTEGER", "BIGINT"),
}


def map_type(abstract_ddl: str) -> str:
    """
    Map an abstract column type to the backend-specific SQL type.

    Handles compound DDL strings like "TEXT DEFAULT 'foo'" by mapping
    only the base type and preserving the rest.
    """
    parts = abstract_ddl.strip().split(None, 1)
    base = parts[0].upper()
    suffix = f" {parts[1]}" if len(parts) > 1 else ""

    if base in _TYPE_MAP:
        idx = 1 if get_backend().is_hana else 0
        return _TYPE_MAP[base][idx] + suffix
    return abstract_ddl
