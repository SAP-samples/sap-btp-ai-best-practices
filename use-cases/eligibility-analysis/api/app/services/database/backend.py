"""
Database Backend Abstraction

Provides a unified interface for SQLite and SAP HANA Cloud backends.
Backend selection is automatic based on environment variables:
  - If `HANA_ADDRESS` is set and non-empty -> HANA
  - Otherwise -> SQLite

HANA settings: HANA_ADDRESS, HANA_PORT, HANA_USER, HANA_PASSWORD, HANA_ENCRYPT and the
optional HANA_SCHEMA. HANA_SCHEMA is applied as the `currentSchema` connection property,
so unqualified table names resolve to that schema instead of the login user's own one.

HANA connections are pooled (thread-safe queue, size 5).
SQLite connections are opened per call (cheap, file-local).
"""

import logging
import os
import queue
import sqlite3
import threading
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Generator, List, Optional, Set

logger = logging.getLogger(__name__)


class BackendType(Enum):
    SQLITE = "sqlite"
    HANA = "hana"


def detect_backend() -> BackendType:
    """
    Pick the database backend from the environment.

    Returns:
        BackendType.HANA when HANA_ADDRESS is set and non-empty, else BackendType.SQLITE.
    """
    addr = os.getenv("HANA_ADDRESS", "").strip()
    if addr:
        return BackendType.HANA
    return BackendType.SQLITE


class DictRow:
    """
    Wraps a HANA result tuple to support row["column_name"] access,
    matching sqlite3.Row behavior.

    Supports: row["col"], row[0], dict(row), len(row), iteration, keys().
    """

    __slots__ = ("_data", "_columns", "_index")

    def __init__(self, data: tuple, columns: List[str]):
        self._data = data
        self._columns = [c.lower() for c in columns]
        self._index = {col: i for i, col in enumerate(self._columns)}

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self._index.get(key.lower())
            if idx is None:
                raise KeyError(key)
            return self._data[idx]
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return list(self._columns)

    def values(self):
        return list(self._data)

    def items(self):
        return list(zip(self._columns, self._data))


class DictCursor:
    """
    Wraps an hdbcli cursor to return DictRow objects and provide
    a lastrowid property via CURRENT_IDENTITY_VALUE().
    """

    def __init__(self, cursor, connection):
        self._cursor = cursor
        self._conn = connection
        self._columns: List[str] = []
        self._lastrowid: Optional[int] = None

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> Optional[int]:
        return self._lastrowid

    def execute(self, sql: str, params=None):
        if params is not None:
            self._cursor.execute(sql, params)
        else:
            self._cursor.execute(sql)
        if self._cursor.description:
            self._columns = [desc[0] for desc in self._cursor.description]
        stripped = sql.strip().upper()
        if stripped.startswith("INSERT"):
            try:
                id_cursor = self._conn.cursor()
                id_cursor.execute("SELECT CURRENT_IDENTITY_VALUE() FROM DUMMY")
                row = id_cursor.fetchone()
                self._lastrowid = int(row[0]) if row and row[0] is not None else None
                id_cursor.close()
            except Exception:
                self._lastrowid = None

    def executemany(self, sql: str, seq_of_params):
        """Send one parameter batch to HANA instead of one network round trip per row."""
        self._cursor.executemany(sql, seq_of_params)
        if self._cursor.description:
            self._columns = [desc[0] for desc in self._cursor.description]

    def fetchone(self) -> Optional[DictRow]:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return DictRow(row, self._columns)

    def fetchall(self) -> List[DictRow]:
        rows = self._cursor.fetchall()
        return [DictRow(r, self._columns) for r in rows]

    def close(self):
        self._cursor.close()


class DatabaseBackend:
    """
    Unified database backend supporting both SQLite and SAP HANA.

    Usage:
        db = get_backend()
        with db.get_connection(db_path) as conn:
            cur = db.cursor(conn)
            cur.execute("SELECT ...")
            rows = cur.fetchall()
            db.commit(conn)
    """

    _POOL_SIZE = 5

    def __init__(self, backend_type: BackendType):
        self._type = backend_type
        self._pool: Optional[queue.Queue] = None
        self._pool_lock = threading.Lock()
        if self._type == BackendType.HANA:
            self._init_hana_pool()

    @property
    def is_hana(self) -> bool:
        return self._type == BackendType.HANA

    @property
    def backend_type(self) -> BackendType:
        return self._type

    def _init_hana_pool(self) -> None:
        self._pool = queue.Queue(maxsize=self._POOL_SIZE)
        for _ in range(self._POOL_SIZE):
            conn = self._create_hana_connection()
            self._pool.put(conn)
        logger.info("HANA connection pool initialized (size=%d)", self._POOL_SIZE)

    def _create_hana_connection(self):
        """
        Open one hdbcli connection from the HANA_* environment variables.

        HANA_SCHEMA (optional) is passed as `currentSchema`, so the app can log in as one
        user (HANA_USER) while reading and writing tables in another schema. An empty value
        keeps HANA's default: the login user's own schema.

        Returns:
            An open hdbcli connection whose CURRENT_SCHEMA is HANA_SCHEMA when set.
        """
        from hdbcli import dbapi

        address = os.getenv("HANA_ADDRESS", "")
        port = int(os.getenv("HANA_PORT", "443"))
        user = os.getenv("HANA_USER", "")
        password = os.getenv("HANA_PASSWORD", "")
        encrypt = os.getenv("HANA_ENCRYPT", "true").lower() in ("true", "1", "yes")
        schema = os.getenv("HANA_SCHEMA", "").strip()

        conn = dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password,
            encrypt=encrypt,
            # Optional schema override; empty keeps the login user's default schema.
            currentSchema=schema,
        )
        logger.debug("Created new HANA connection to %s:%d (schema=%s)", address, port, schema or "<user default>")
        return conn

    def _validate_hana_connection(self, conn) -> bool:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUMMY")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    @contextmanager
    def get_connection(self, db_path: Optional[Path] = None) -> Generator:
        if self._type == BackendType.SQLITE:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        else:
            conn = self._pool.get(timeout=30)
            try:
                if not self._validate_hana_connection(conn):
                    logger.warning("Stale HANA connection; creating replacement")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = self._create_hana_connection()
                yield conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    conn = self._create_hana_connection()
                except Exception:
                    conn = None
                raise
            finally:
                if conn is not None:
                    self._pool.put(conn)

    def cursor(self, conn):
        if self._type == BackendType.SQLITE:
            return conn.cursor()
        return DictCursor(conn.cursor(), conn)

    def commit(self, conn) -> None:
        conn.commit()

    def table_exists(self, conn, table_name: str) -> bool:
        if self._type == BackendType.SQLITE:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cursor.fetchone() is not None
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM TABLES WHERE TABLE_NAME = ? AND SCHEMA_NAME = CURRENT_SCHEMA",
                (table_name.upper(),),
            )
            result = cursor.fetchone() is not None
            cursor.close()
            return result

    def index_exists(self, conn, index_name: str) -> bool:
        if self._type == BackendType.SQLITE:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                (index_name,),
            )
            return cursor.fetchone() is not None
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM INDEXES WHERE INDEX_NAME = ? AND SCHEMA_NAME = CURRENT_SCHEMA",
                (index_name.upper(),),
            )
            result = cursor.fetchone() is not None
            cursor.close()
            return result

    def get_table_columns(self, conn, table_name: str) -> Set[str]:
        if self._type == BackendType.SQLITE:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            return {row[1] for row in cursor.fetchall()}
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COLUMN_NAME FROM TABLE_COLUMNS "
                "WHERE TABLE_NAME = ? AND SCHEMA_NAME = CURRENT_SCHEMA",
                (table_name.upper(),),
            )
            cols = {row[0].lower() for row in cursor.fetchall()}
            cursor.close()
            return cols


_backend_instance: Optional[DatabaseBackend] = None
_backend_lock = threading.Lock()


def get_backend() -> DatabaseBackend:
    global _backend_instance
    if _backend_instance is None:
        with _backend_lock:
            if _backend_instance is None:
                bt = detect_backend()
                _backend_instance = DatabaseBackend(bt)
                logger.info("Database backend initialized: %s", bt.value)
    return _backend_instance
