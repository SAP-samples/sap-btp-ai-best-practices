"""Initialize workspace tables in HANA; SQLite is an explicitly injected test double."""

from contextlib import contextmanager
import json
from threading import RLock

_SCHEMA_LOCK = RLock()
TABLES = {
    "RECEIVABLES_ANALYSIS_EXPORTS": {
        "scope_token": "NVARCHAR(64) PRIMARY KEY", "analysis_id": "NVARCHAR(36) NOT NULL",
        "eligibility": "BLOB NOT NULL", "insights_excel": "BLOB NOT NULL", "insights_pdf": "BLOB NOT NULL",
    },
    "RECEIVABLES_ANALYSES": {
        "analysis_id": "NVARCHAR(36) PRIMARY KEY", "request_key": "NVARCHAR(128) NOT NULL UNIQUE",
        "input_hash": "NVARCHAR(64) NOT NULL", "source_hash": "NVARCHAR(64) NOT NULL",
        "created_at": "NVARCHAR(40) NOT NULL", "metadata": "NCLOB NOT NULL", "content": "BLOB NOT NULL",
    },
    "RECEIVABLES_SOURCE_ROWS": {
        "analysis_id": "NVARCHAR(36) NOT NULL", "row_id": "NVARCHAR(64) NOT NULL",
        "row_number": "INTEGER NOT NULL", "payload": "NCLOB NOT NULL",
    },
    "RECEIVABLES_RUNS": {
        "run_id": "NVARCHAR(36) PRIMARY KEY", "analysis_id": "NVARCHAR(36) NOT NULL",
        "revision": "INTEGER NOT NULL", "status": "NVARCHAR(50) NOT NULL", "payload": "NCLOB NOT NULL",
    },
}


def decode_json(value):
    """Decode an NCLOB/string result into the saved JSON snapshot."""
    if hasattr(value, "read"):
        value = value.read()
    return json.loads(value)


@contextmanager
def transaction(backend, db_path=None):
    """Commit all writes together or roll back; disable HANA's default autocommit."""
    with backend.get_connection(db_path) as connection:
        if backend.is_hana:
            connection.setautocommit(False)
        try:
            yield backend.cursor(connection)
            backend.commit(connection)
        except Exception:
            connection.rollback()
            raise


def ensure_schema(backend, db_path=None, tables=None):
    """Create missing tables and validate required column contracts before first use."""
    if not backend.is_hana and db_path is None:
        raise ValueError("Workspace requires HANA; an explicit SQLite path is allowed only for tests")
    with _SCHEMA_LOCK, backend.get_connection(db_path) as connection:
        cursor = backend.cursor(connection)
        for table, columns in (tables or TABLES).items():
            if not backend.table_exists(connection, table):
                definitions = [f"{name} {kind}" for name, kind in columns.items()]
                if table == "RECEIVABLES_SOURCE_ROWS":
                    definitions.append("PRIMARY KEY (analysis_id, row_id)")
                cursor.execute(f"CREATE TABLE {table} ({', '.join(definitions)})")
                backend.commit(connection)
            found = {name.lower() for name in backend.get_table_columns(connection, table)}
            if missing := set(columns) - found:
                raise RuntimeError(f"{table} requires a schema migration; missing columns: {sorted(missing)}")
            if backend.is_hana:
                cursor.execute("SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE FROM SYS.TABLE_COLUMNS "
                               "WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?", (table,))
                contract = {row[0].lower(): (row[1], row[2]) for row in cursor.fetchall()}
                for name, definition in columns.items():
                    expected = definition.split('(')[0].split()[0]
                    actual, nullable = contract[name]
                    required = "NOT NULL" in definition or "PRIMARY KEY" in definition
                    if actual != expected or (required and nullable != "FALSE"):
                        raise RuntimeError(f"Incompatible HANA column {table}.{name}")
