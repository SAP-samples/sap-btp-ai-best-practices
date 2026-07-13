"""
Optimizer artifact persistence and materialization.

Stores optimizer run artifacts in the configured database backend so completed
runs remain available even after the local process directory disappears.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Generator, Iterable, List, Optional

import pandas as pd

from ..database import get_backend
from .process_store import _get_default_db_path

logger = logging.getLogger(__name__)

TEMP_ROOT = Path(gettempdir()) / "optimizer-artifacts"

_ARTIFACTS_SQLITE_DDL = """
CREATE TABLE optimizer_process_artifacts (
    process_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    storage_mode TEXT NOT NULL,
    text_content TEXT,
    binary_content BLOB,
    metadata_json TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (process_id, artifact_key)
)
"""

_ARTIFACTS_HANA_DDL = """
CREATE TABLE optimizer_process_artifacts (
    process_id NVARCHAR(5000) NOT NULL,
    artifact_key NVARCHAR(5000) NOT NULL,
    artifact_kind NVARCHAR(5000) NOT NULL,
    storage_mode NVARCHAR(5000) NOT NULL,
    text_content NCLOB,
    binary_content BLOB,
    metadata_json NCLOB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (process_id, artifact_key)
)
"""

_INVOICE_ROWS_SQLITE_DDL = """
CREATE TABLE optimizer_process_invoice_rows (
    process_id TEXT NOT NULL,
    bucket TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    invoice_ref TEXT,
    company_code TEXT,
    customer TEXT,
    purchase_price REAL,
    due_date TEXT,
    status TEXT,
    excluded_stage TEXT,
    excluded_reason TEXT,
    excluded_reason_detail TEXT,
    planned_week_index INTEGER,
    planned_week_start TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (process_id, bucket, row_index)
)
"""

_INVOICE_ROWS_HANA_DDL = """
CREATE TABLE optimizer_process_invoice_rows (
    process_id NVARCHAR(5000) NOT NULL,
    bucket NVARCHAR(5000) NOT NULL,
    row_index INTEGER NOT NULL,
    invoice_ref NVARCHAR(5000),
    company_code NVARCHAR(5000),
    customer NVARCHAR(5000),
    purchase_price DOUBLE,
    due_date NVARCHAR(5000),
    status NVARCHAR(5000),
    excluded_stage NVARCHAR(5000),
    excluded_reason NVARCHAR(5000),
    excluded_reason_detail NCLOB,
    planned_week_index INTEGER,
    planned_week_start NVARCHAR(5000),
    payload_json NCLOB NOT NULL,
    PRIMARY KEY (process_id, bucket, row_index)
)
"""

_EXPOSURE_ROWS_SQLITE_DDL = """
CREATE TABLE optimizer_process_exposure_rows (
    process_id TEXT NOT NULL,
    row_index INTEGER NOT NULL,
    week_start TEXT,
    entity_type TEXT,
    entity_id TEXT,
    used_new REAL,
    used_base REAL,
    used_total REAL,
    limit_value REAL,
    utilization_pct REAL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (process_id, row_index)
)
"""

_EXPOSURE_ROWS_HANA_DDL = """
CREATE TABLE optimizer_process_exposure_rows (
    process_id NVARCHAR(5000) NOT NULL,
    row_index INTEGER NOT NULL,
    week_start NVARCHAR(5000),
    entity_type NVARCHAR(5000),
    entity_id NVARCHAR(5000),
    used_new DOUBLE,
    used_base DOUBLE,
    used_total DOUBLE,
    limit_value DOUBLE,
    utilization_pct DOUBLE,
    payload_json NCLOB NOT NULL,
    PRIMARY KEY (process_id, row_index)
)
"""


def _coerce_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return str(value)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return None


class OptimizerArtifactStore:
    """Stores optimizer artifacts in SQLite or HANA using the shared backend."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _get_default_db_path()
        self._db = get_backend()
        self._ensure_tables()

    @contextmanager
    def _get_connection(self) -> Generator:
        with self._db.get_connection(self.db_path) as conn:
            yield conn

    def _ensure_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if not self._db.table_exists(conn, "optimizer_process_artifacts"):
                cursor.execute(_ARTIFACTS_HANA_DDL if self._db.is_hana else _ARTIFACTS_SQLITE_DDL)
            if not self._db.table_exists(conn, "optimizer_process_invoice_rows"):
                cursor.execute(_INVOICE_ROWS_HANA_DDL if self._db.is_hana else _INVOICE_ROWS_SQLITE_DDL)
            if not self._db.table_exists(conn, "optimizer_process_exposure_rows"):
                cursor.execute(_EXPOSURE_ROWS_HANA_DDL if self._db.is_hana else _EXPOSURE_ROWS_SQLITE_DDL)
            self._db.commit(conn)

    def has_text_artifact(self, process_id: str, artifact_key: str) -> bool:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT 1 FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            return cursor.fetchone() is not None

    def upsert_text_artifact(
        self,
        process_id: str,
        artifact_key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        artifact_kind: str = "text",
        storage_mode: str = "text",
    ) -> None:
        metadata_json = json.dumps(metadata or {}, default=str)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            cursor.execute(
                """
                INSERT INTO optimizer_process_artifacts (
                    process_id,
                    artifact_key,
                    artifact_kind,
                    storage_mode,
                    text_content,
                    binary_content,
                    metadata_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    process_id,
                    artifact_key,
                    artifact_kind,
                    storage_mode,
                    content,
                    None,
                    metadata_json,
                ),
            )
            self._db.commit(conn)

    def get_text_artifact(self, process_id: str, artifact_key: str) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT text_content FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return str(row[0]) if row[0] is not None else None

    def get_text_artifact_metadata(self, process_id: str, artifact_key: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT metadata_json FROM optimizer_process_artifacts WHERE process_id = ? AND artifact_key = ?",
                (process_id, artifact_key),
            )
            row = cursor.fetchone()
            if row is None or not row[0]:
                return {}
            try:
                return json.loads(str(row[0]))
            except Exception:
                return {}

    def has_invoice_rows(self, process_id: str, bucket: str) -> bool:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT 1 FROM optimizer_process_invoice_rows WHERE process_id = ? AND bucket = ? LIMIT 1",
                (process_id, bucket),
            )
            return cursor.fetchone() is not None

    def has_exposure_rows(self, process_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT 1 FROM optimizer_process_exposure_rows WHERE process_id = ? LIMIT 1",
                (process_id,),
            )
            return cursor.fetchone() is not None

    def replace_invoice_rows(self, process_id: str, bucket: str, rows: Iterable[Dict[str, Any]]) -> int:
        normalized_rows = list(rows)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_process_invoice_rows WHERE process_id = ? AND bucket = ?",
                (process_id, bucket),
            )
            for idx, row in enumerate(normalized_rows):
                payload_json = json.dumps(row, default=str)
                cursor.execute(
                    """
                    INSERT INTO optimizer_process_invoice_rows (
                        process_id,
                        bucket,
                        row_index,
                        invoice_ref,
                        company_code,
                        customer,
                        purchase_price,
                        due_date,
                        status,
                        excluded_stage,
                        excluded_reason,
                        excluded_reason_detail,
                        planned_week_index,
                        planned_week_start,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        process_id,
                        bucket,
                        idx,
                        _coerce_string(row.get("Invoice Reference", row.get("invoice_reference", row.get("invoice_ref")))),
                        _coerce_string(row.get("Company Code", row.get("company_code", row.get("seller_id_external")))),
                        _coerce_string(row.get("Customer", row.get("customer", row.get("debtor_id")))),
                        _coerce_float(row.get("Purchase Price", row.get("purchase_price", row.get("candidate_amount")))),
                        _coerce_string(row.get("Due Date", row.get("due_date"))),
                        _coerce_string(row.get("Status", row.get("status"))),
                        _coerce_string(row.get("excluded_stage")),
                        _coerce_string(row.get("excluded_reason")),
                        _coerce_string(row.get("excluded_reason_detail")),
                        _coerce_int(row.get("planned_week_index")),
                        _coerce_string(row.get("planned_week_start_iso", row.get("planned_week_start", row.get("week_start")))),
                        payload_json,
                    ),
                )
            self._db.commit(conn)
        return len(normalized_rows)

    def replace_exposure_rows(self, process_id: str, rows: Iterable[Dict[str, Any]]) -> int:
        normalized_rows = list(rows)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_process_exposure_rows WHERE process_id = ?",
                (process_id,),
            )
            for idx, row in enumerate(normalized_rows):
                payload_json = json.dumps(row, default=str)
                cursor.execute(
                    """
                    INSERT INTO optimizer_process_exposure_rows (
                        process_id,
                        row_index,
                        week_start,
                        entity_type,
                        entity_id,
                        used_new,
                        used_base,
                        used_total,
                        limit_value,
                        utilization_pct,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        process_id,
                        idx,
                        _coerce_string(row.get("week_start")),
                        _coerce_string(row.get("entity_type")),
                        _coerce_string(row.get("entity_id")),
                        _coerce_float(row.get("used_new")),
                        _coerce_float(row.get("used_base")),
                        _coerce_float(row.get("used_total", row.get("used"))),
                        _coerce_float(row.get("limit")),
                        _coerce_float(row.get("utilization_pct")),
                        payload_json,
                    ),
                )
            self._db.commit(conn)
        return len(normalized_rows)

    def _load_payloads(self, sql: str, params: tuple[Any, ...]) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        payloads: List[Dict[str, Any]] = []
        for row in rows:
            try:
                payloads.append(json.loads(str(row[0])))
            except Exception:
                logger.warning("Failed decoding optimizer artifact payload")
        return payloads

    def get_all_invoice_rows(self, process_id: str, bucket: str) -> List[Dict[str, Any]]:
        return self._load_payloads(
            """
            SELECT payload_json
            FROM optimizer_process_invoice_rows
            WHERE process_id = ? AND bucket = ?
            ORDER BY row_index ASC
            """,
            (process_id, bucket),
        )

    def get_all_exposure_rows(self, process_id: str) -> List[Dict[str, Any]]:
        return self._load_payloads(
            """
            SELECT payload_json
            FROM optimizer_process_exposure_rows
            WHERE process_id = ?
            ORDER BY row_index ASC
            """,
            (process_id,),
        )

    def load_invoice_rows(
        self,
        process_id: str,
        bucket: str,
        *,
        invoice_ref: Optional[str] = None,
        customer: Optional[str] = None,
        company_code: Optional[str] = None,
        excluded_stage: Optional[str] = None,
        excluded_reason: Optional[str] = None,
        week_start: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        conditions = ["process_id = ?", "bucket = ?"]
        params: List[Any] = [process_id, bucket]

        def _like_clause(column: str, value: Optional[str]) -> None:
            if not value:
                return
            conditions.append(f"LOWER(COALESCE({column}, '')) LIKE ?")
            params.append(f"%{str(value).lower()}%")

        _like_clause("invoice_ref", invoice_ref)
        _like_clause("customer", customer)
        _like_clause("company_code", company_code)
        _like_clause("excluded_stage", excluded_stage)
        _like_clause("excluded_reason", excluded_reason)
        _like_clause("planned_week_start", week_start)

        where_sql = " AND ".join(conditions)
        order_sql = {
            "selected": "COALESCE(due_date, ''), COALESCE(invoice_ref, '')",
            "excluded": "COALESCE(excluded_stage, ''), COALESCE(excluded_reason, ''), COALESCE(invoice_ref, '')",
            "weekly_plan": "COALESCE(planned_week_start, ''), COALESCE(planned_week_index, 0), COALESCE(invoice_ref, '')",
        }.get(bucket, "row_index")

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"SELECT COUNT(*) FROM optimizer_process_invoice_rows WHERE {where_sql}",
                tuple(params),
            )
            count_row = cursor.fetchone()
            total = int(count_row[0]) if count_row else 0

            cursor.execute(
                f"""
                SELECT payload_json
                FROM optimizer_process_invoice_rows
                WHERE {where_sql}
                ORDER BY {order_sql}, row_index ASC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [limit, offset]),
            )
            page_rows = cursor.fetchall()

        payloads: List[Dict[str, Any]] = []
        for row in page_rows:
            try:
                payloads.append(json.loads(str(row[0])))
            except Exception:
                logger.warning("Failed decoding invoice row payload for process %s", process_id)
        return {"rows": payloads, "total": total, "limit": int(limit), "offset": int(offset)}

    def load_exposure_rows(
        self,
        process_id: str,
        *,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        week_start: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        conditions = ["process_id = ?"]
        params: List[Any] = [process_id]

        def _like_clause(column: str, value: Optional[str]) -> None:
            if not value:
                return
            conditions.append(f"LOWER(COALESCE({column}, '')) LIKE ?")
            params.append(f"%{str(value).lower()}%")

        _like_clause("entity_type", entity_type)
        _like_clause("entity_id", entity_id)
        _like_clause("week_start", week_start)
        where_sql = " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"SELECT COUNT(*) FROM optimizer_process_exposure_rows WHERE {where_sql}",
                tuple(params),
            )
            count_row = cursor.fetchone()
            total = int(count_row[0]) if count_row else 0

            cursor.execute(
                f"""
                SELECT payload_json
                FROM optimizer_process_exposure_rows
                WHERE {where_sql}
                ORDER BY COALESCE(week_start, ''), COALESCE(entity_type, ''), COALESCE(entity_id, ''), row_index ASC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [limit, offset]),
            )
            page_rows = cursor.fetchall()

        payloads: List[Dict[str, Any]] = []
        for row in page_rows:
            try:
                payloads.append(json.loads(str(row[0])))
            except Exception:
                logger.warning("Failed decoding exposure row payload for process %s", process_id)
        return {"rows": payloads, "total": total, "limit": int(limit), "offset": int(offset)}

    def delete_process_artifacts(self, process_id: str) -> None:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_process_invoice_rows WHERE process_id = ?",
                (process_id,),
            )
            cursor.execute(
                "DELETE FROM optimizer_process_exposure_rows WHERE process_id = ?",
                (process_id,),
            )
            cursor.execute(
                "DELETE FROM optimizer_process_artifacts WHERE process_id = ?",
                (process_id,),
            )
            self._db.commit(conn)

    def materialize_text_artifact(
        self,
        process_id: str,
        artifact_key: str,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        content = self.get_text_artifact(process_id, artifact_key)
        if content is None:
            return None
        target_dir = TEMP_ROOT / process_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (filename or artifact_key)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    def materialize_workbook(self, process_id: str, artifact_type: str) -> Optional[Path]:
        target_dir = TEMP_ROOT / process_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if artifact_type == "weekly_exposure":
            rows = self.get_all_exposure_rows(process_id)
            if not rows:
                return None
            path = target_dir / "weekly_exposure.xlsx"
            pd.DataFrame(rows).to_excel(path, index=False)
            return path

        bucket_map = {
            "selected": ("selected", "selected.xlsx", "selected"),
            "excluded": ("excluded", "excluded.xlsx", "excluded"),
            "weekly_plan": ("weekly_plan", "weekly_plan.xlsx", "weekly_plan"),
        }
        bucket_info = bucket_map.get(artifact_type)
        if bucket_info is None:
            return None

        bucket, filename, sheet_name = bucket_info
        rows = self.get_all_invoice_rows(process_id, bucket)
        if not rows:
            return None
        path = target_dir / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    def clear_temp_artifacts(self, process_id: str) -> None:
        target_dir = TEMP_ROOT / process_id
        if not target_dir.exists():
            return
        for child in target_dir.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
        try:
            target_dir.rmdir()
        except OSError:
            pass
