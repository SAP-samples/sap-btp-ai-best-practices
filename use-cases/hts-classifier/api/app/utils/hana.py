"""SAP HANA utilities used by the API."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def _quote_identifier(identifier: str) -> str:
    return f'"{str(identifier).replace(chr(34), chr(34) * 2)}"'


def _qualified_table(table: str, schema: str | None = None) -> str:
    if schema:
        return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
    return _quote_identifier(table)


def _normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    return value


def _hana_type_for_series(column: str, series: pd.Series) -> str:
    if column in {"source_row_id", "prepared__row_id"}:
        return "BIGINT"
    if pd.api.types.is_bool_dtype(series):
        return "SMALLINT"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"

    max_len = int(
        series.fillna("")
        .astype(str)
        .map(len)
        .max()
        or 0
    )
    if column == "normalized_product_code":
        return "NVARCHAR(255)"
    width = min(max(128, max_len + 32), 5000)
    return f"NVARCHAR({width})"


class HANAConnection:
    """Handle SAP HANA connections and DataFrame operations.

    Each thread that uses this instance gets its own underlying hdbcli
    connection, stored in a ``threading.local()`` object.  This makes it safe
    for multiple threads (e.g. the classification-job worker, batch
    ``ThreadPoolExecutor`` threads, and FastAPI request-handler threads) to
    share a single ``HANAConnection`` instance without risking interleaved
    cursor operations on the same hdbcli connection.
    """

    def __init__(self) -> None:
        load_dotenv()

        self.address = os.getenv("HANA_ADDRESS")
        self.port = os.getenv("HANA_PORT", "443")
        self.user = os.getenv("HANA_USER")
        self.password = os.getenv("HANA_PASSWORD")
        self.encrypt = os.getenv("HANA_ENCRYPT", "True").strip().lower() == "true"

        self._local = threading.local()
        self._dbapi = None

        if not all([self.address, self.port, self.user, self.password]):
            logger.warning(
                "HANA credentials missing in .env file (HANA_ADDRESS, HANA_PORT, HANA_USER, HANA_PASSWORD)"
            )

    @property
    def connection(self) -> Any:
        """Return the current thread's hdbcli connection, or ``None``."""
        return getattr(self._local, "connection", None)

    @connection.setter
    def connection(self, value: Any) -> None:
        self._local.connection = value

    def _require_dbapi(self):
        if self._dbapi is None:
            try:
                from hdbcli import dbapi  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "hdbcli is not installed. Add it to the API runtime to enable HANA-backed metal composition."
                ) from exc
            self._dbapi = dbapi
        return self._dbapi

    def connect(self):
        if self.connection is not None:
            return self.connection

        if not all([self.address, self.port, self.user, self.password]):
            raise ValueError("HANA credentials missing")

        dbapi = self._require_dbapi()
        logger.info("Connecting to HANA at %s:%s", self.address, self.port)
        connection = dbapi.connect(
            address=self.address,
            port=int(self.port),
            user=self.user,
            password=self.password,
            encrypt=self.encrypt,
        )
        set_autocommit = getattr(connection, "setautocommit", None)
        if callable(set_autocommit):
            # Job submission and ownership updates rely on multi-statement
            # transactions. hdbcli defaults to autocommit, which can expose a
            # queued job before its item rows and ownership rows are complete.
            set_autocommit(False)
        self.connection = connection
        return self.connection

    def disconnect(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        connection = self.connect()
        cursor = connection.cursor()
        try:
            yield cursor
            if self._transaction_depth == 0:
                connection.commit()
        except Exception:
            if self._transaction_depth > 0:
                self._rollback_only = True
            else:
                connection.rollback()
            raise
        finally:
            cursor.close()

    @property
    def _transaction_depth(self) -> int:
        return int(getattr(self._local, "transaction_depth", 0) or 0)

    @_transaction_depth.setter
    def _transaction_depth(self, value: int) -> None:
        self._local.transaction_depth = int(value)

    @property
    def _rollback_only(self) -> bool:
        return bool(getattr(self._local, "rollback_only", False))

    @_rollback_only.setter
    def _rollback_only(self, value: bool) -> None:
        self._local.rollback_only = bool(value)

    @contextmanager
    def transaction(self) -> Iterator["HANAConnection"]:
        connection = self.connect()
        is_outermost = self._transaction_depth == 0
        if is_outermost:
            self._rollback_only = False
        self._transaction_depth += 1
        try:
            yield self
        except Exception:
            if self._transaction_depth > 0:
                self._rollback_only = True
            if is_outermost:
                connection.rollback()
                self._rollback_only = False
            raise
        else:
            if is_outermost:
                if self._rollback_only:
                    connection.rollback()
                else:
                    connection.commit()
                self._rollback_only = False
        finally:
            self._transaction_depth = max(0, self._transaction_depth - 1)

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self.cursor() as cursor:
            cursor.execute(sql, params or [])

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.cursor() as cursor:
            cursor.executemany(sql, list(rows))

    def fetch_dataframe(
        self,
        table: str,
        *,
        schema: Optional[str] = None,
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        sql = f"SELECT * FROM {_qualified_table(table, schema)}"
        if where_clause:
            sql = f"{sql} WHERE {where_clause}"
        with self.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def table_exists(self, table: str, *, schema: Optional[str] = None) -> bool:
        try:
            with self.cursor() as cursor:
                cursor.execute(f"SELECT 1 FROM {_qualified_table(table, schema)} WHERE 1 = 0")
            return True
        except Exception:
            return False

    def table_columns(self, table: str, *, schema: Optional[str] = None) -> List[str]:
        with self.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {_qualified_table(table, schema)} WHERE 1 = 0")
            return [str(description[0]) for description in (cursor.description or [])]

    def column_exists(self, table: str, column: str, *, schema: Optional[str] = None) -> bool:
        try:
            return column in self.table_columns(table, schema=schema)
        except Exception:
            return False

    def test_connection(self) -> Dict[str, Any]:
        try:
            with self.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUMMY")
                test_value = cursor.fetchone()[0]

            if test_value == 1:
                return {
                    "success": True,
                    "message": "Successfully connected to SAP HANA and executed 'SELECT 1 FROM DUMMY'",
                }
            return {
                "success": False,
                "message": "Test query did not return expected value",
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            logger.error("HANA connection test failed: %s", exc)
            return {"success": False, "message": str(exc)}
        finally:
            self.disconnect()

    def _create_column_table(
        self,
        table: str,
        *,
        schema: Optional[str],
        frame: pd.DataFrame,
        primary_key: Optional[str],
    ) -> None:
        columns_sql: List[str] = []
        for column in frame.columns:
            column_sql = f"{_quote_identifier(column)} {_hana_type_for_series(column, frame[column])}"
            if primary_key and column == primary_key:
                column_sql = f"{column_sql} NOT NULL"
            columns_sql.append(column_sql)
        if primary_key:
            columns_sql.append(f"PRIMARY KEY ({_quote_identifier(primary_key)})")

        ddl = (
            f"CREATE COLUMN TABLE {_qualified_table(table, schema)} "
            f"({', '.join(columns_sql)})"
        )
        self.execute(ddl)

    def _drop_table_if_exists(self, table: str, *, schema: Optional[str]) -> None:
        if not self.table_exists(table, schema=schema):
            return
        try:
            self.execute(f"DROP TABLE {_qualified_table(table, schema)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to drop HANA table %s: %s", _qualified_table(table, schema), exc)

    def _insert_dataframe(
        self,
        table: str,
        *,
        schema: Optional[str],
        frame: pd.DataFrame,
        batch_size: int = 500,
    ) -> None:
        if frame.empty:
            return

        columns_sql = ", ".join(_quote_identifier(column) for column in frame.columns)
        placeholders = ", ".join("?" for _ in frame.columns)
        sql = (
            f"INSERT INTO {_qualified_table(table, schema)} "
            f"({columns_sql}) VALUES ({placeholders})"
        )
        rows = [
            tuple(_normalize_scalar(value) for value in row)
            for row in frame.itertuples(index=False, name=None)
        ]
        for idx in range(0, len(rows), batch_size):
            self.executemany(sql, rows[idx : idx + batch_size])

    def _ensure_index(
        self,
        table: str,
        *,
        schema: Optional[str],
        columns: Sequence[str],
    ) -> None:
        if not columns:
            return
        index_name = f"IDX_{table}_{'_'.join(columns)}"
        quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
        try:
            self.execute(
                f"CREATE INDEX {_quote_identifier(index_name)} "
                f"ON {_qualified_table(table, schema)} ({quoted_columns})"
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Skipping HANA index creation for %s: %s", index_name, exc)

    def refresh_serving_table(
        self,
        *,
        frame: pd.DataFrame,
        table: str,
        schema: Optional[str],
        primary_key: str,
        index_columns: Sequence[str],
    ) -> Dict[str, Any]:
        if frame.empty:
            raise ValueError("Cannot refresh HANA serving table with an empty frame")

        stage_table = f"{table}__STAGING"
        self._drop_table_if_exists(stage_table, schema=schema)
        self._create_column_table(stage_table, schema=schema, frame=frame, primary_key=primary_key)
        self._insert_dataframe(stage_table, schema=schema, frame=frame)

        stage_count = int(self.fetch_dataframe(stage_table, schema=schema).shape[0])
        if stage_count != len(frame):
            raise RuntimeError(
                f"Staging load validation failed for HANA table {stage_table}: "
                f"expected {len(frame)} rows, got {stage_count}"
            )

        if not self.table_exists(table, schema=schema):
            self._create_column_table(table, schema=schema, frame=frame, primary_key=primary_key)

        try:
            self.execute(f"DELETE FROM {_qualified_table(table, schema)}")
            self.execute(
                f"INSERT INTO {_qualified_table(table, schema)} "
                f"SELECT * FROM {_qualified_table(stage_table, schema)}"
            )
        except Exception:
            self._drop_table_if_exists(table, schema=schema)
            self._create_column_table(table, schema=schema, frame=frame, primary_key=primary_key)
            self._insert_dataframe(table, schema=schema, frame=frame)

        self._ensure_index(table, schema=schema, columns=index_columns)
        self._drop_table_if_exists(stage_table, schema=schema)
        return {
            "table": table,
            "schema": schema,
            "row_count": int(len(frame)),
        }
