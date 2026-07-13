"""
Optimizer Process Store

Database-backed persistence for tracking optimization processes.
Supports both SQLite (local) and SAP HANA Cloud (production) backends.
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from ..database import get_backend, map_type, sql_add_column

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / "data"

_PROCESS_LIST_COLUMNS = (
    "id",
    "status",
    "created_at",
    "started_at",
    "completed_at",
    "extraction_filename",
    "cohort",
    "planning_mode",
    "source_profile",
    "process_dir",
    "candidate_count",
    "selected_count",
    "excluded_count",
    "candidate_amount",
    "selected_amount",
    "optimizer_status",
    "error_message",
)

_PROCESS_HEAD_COLUMNS = (
    "id",
    "status",
    "completed_at",
    "progress_updated_at",
    "process_dir",
)


def _get_default_db_path() -> Path:
    db_dir = DEFAULT_DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "optimizer_processes.db"


_SQLITE_DDL = """
CREATE TABLE optimizer_processes (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'created',
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    extraction_filename TEXT,
    cohort TEXT,
    cohort_match_granularity TEXT,
    sheet_name TEXT DEFAULT 'SAPUI5 Export',
    release_event TEXT DEFAULT 'reconciliation_file_date',
    planning_mode TEXT DEFAULT 'single_week',
    planning_start_date TEXT,
    horizon_weeks INTEGER DEFAULT 8,
    attempt_cap INTEGER DEFAULT 1,
    source_profile TEXT DEFAULT 'extraction_file',
    release_event_mode TEXT DEFAULT 'reconciliation_file_date',
    lifecycle_input_path TEXT,
    solver_max_time_seconds INTEGER DEFAULT 60,
    solver_random_seed INTEGER DEFAULT 0,
    solver_num_search_workers INTEGER DEFAULT 1,
    process_dir TEXT NOT NULL,
    run_metadata_json TEXT,
    error_message TEXT,
    candidate_count INTEGER,
    selected_count INTEGER,
    excluded_count INTEGER,
    candidate_amount REAL,
    selected_amount REAL,
    optimizer_status TEXT,
    progress_json TEXT,
    progress_updated_at TIMESTAMP,
    weekly_plan_output TEXT,
    weekly_exposure_output TEXT
)
"""

_HANA_DDL = """
CREATE TABLE optimizer_processes (
    id NVARCHAR(5000) PRIMARY KEY,
    status NVARCHAR(5000) NOT NULL DEFAULT 'created',
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    extraction_filename NVARCHAR(5000),
    cohort NVARCHAR(5000),
    cohort_match_granularity NVARCHAR(5000),
    sheet_name NVARCHAR(5000) DEFAULT 'SAPUI5 Export',
    release_event NVARCHAR(5000) DEFAULT 'reconciliation_file_date',
    planning_mode NVARCHAR(5000) DEFAULT 'single_week',
    planning_start_date NVARCHAR(5000),
    horizon_weeks INTEGER DEFAULT 8,
    attempt_cap INTEGER DEFAULT 1,
    source_profile NVARCHAR(5000) DEFAULT 'extraction_file',
    release_event_mode NVARCHAR(5000) DEFAULT 'reconciliation_file_date',
    lifecycle_input_path NVARCHAR(5000),
    solver_max_time_seconds INTEGER DEFAULT 60,
    solver_random_seed INTEGER DEFAULT 0,
    solver_num_search_workers INTEGER DEFAULT 1,
    process_dir NVARCHAR(5000) NOT NULL,
    run_metadata_json NCLOB,
    error_message NCLOB,
    candidate_count INTEGER,
    selected_count INTEGER,
    excluded_count INTEGER,
    candidate_amount DOUBLE,
    selected_amount DOUBLE,
    optimizer_status NVARCHAR(5000),
    progress_json NCLOB,
    progress_updated_at TIMESTAMP,
    weekly_plan_output NCLOB,
    weekly_exposure_output NCLOB
)
"""


class ProcessStore:
    """Database CRUD for optimizer processes."""

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
            if not self._db.table_exists(conn, "optimizer_processes"):
                cursor = self._db.cursor(conn)
                cursor.execute(_HANA_DDL if self._db.is_hana else _SQLITE_DDL)
                self._db.commit(conn)
            self._ensure_additive_columns(conn)
            self._db.commit(conn)
            if self._db.is_hana:
                logger.info("Optimizer process store ensured (HANA)")
            else:
                logger.info("Optimizer process store ensured at %s", self.db_path)

    def _ensure_additive_columns(self, conn) -> None:
        """Add new columns on existing DBs without destructive migration."""
        existing = self._db.get_table_columns(conn, "optimizer_processes")

        required_columns = {
            "planning_mode": "TEXT DEFAULT 'single_week'",
            "planning_start_date": "TEXT",
            "horizon_weeks": "INTEGER DEFAULT 8",
            "attempt_cap": "INTEGER DEFAULT 1",
            "source_profile": "TEXT DEFAULT 'extraction_file'",
            "release_event_mode": "TEXT DEFAULT 'reconciliation_file_date'",
            "lifecycle_input_path": "TEXT",
            "progress_json": "NCLOB",
            "progress_updated_at": "TIMESTAMP",
            "weekly_plan_output": "NCLOB",
            "weekly_exposure_output": "NCLOB",
        }

        cursor = self._db.cursor(conn)
        for col, ddl in required_columns.items():
            if col in existing:
                continue
            stmt = sql_add_column("optimizer_processes", col, map_type(ddl))
            cursor.execute(stmt)

    def create_process(
        self,
        process_id: str,
        process_dir: str,
        extraction_filename: str,
        cohort: Optional[str] = None,
        sheet_name: str = "SAPUI5 Export",
        source_profile: str = "extraction_file",
    ) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                """
                INSERT INTO optimizer_processes
                (
                    id,
                    status,
                    created_at,
                    extraction_filename,
                    cohort,
                    sheet_name,
                    planning_mode,
                    source_profile,
                    release_event,
                    release_event_mode,
                    horizon_weeks,
                    attempt_cap,
                    solver_max_time_seconds,
                    process_dir
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    process_id,
                    "created",
                    now,
                    extraction_filename,
                    cohort,
                    sheet_name,
                    "single_week",
                    source_profile,
                    "reconciliation_file_date",
                    "reconciliation_file_date",
                    8,
                    1,
                    60,
                    process_dir,
                ),
            )
            self._db.commit(conn)
        return self.get_process(process_id)

    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "SELECT * FROM optimizer_processes WHERE id = ?",
                (process_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def get_process_head(self, process_id: str) -> Optional[Dict[str, Any]]:
        columns_sql = ", ".join(_PROCESS_HEAD_COLUMNS)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"SELECT {columns_sql} FROM optimizer_processes WHERE id = ?",
                (process_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def list_processes(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        columns_sql = ", ".join(_PROCESS_LIST_COLUMNS)
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if status:
                cursor.execute(
                    """
                    SELECT {columns_sql} FROM optimizer_processes
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """.format(columns_sql=columns_sql),
                    (status, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT {columns_sql} FROM optimizer_processes
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """.format(columns_sql=columns_sql),
                    (limit, offset),
                )
            return [dict(row) for row in cursor.fetchall()]

    def count_processes(self, status: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            if status:
                cursor.execute(
                    "SELECT COUNT(*) FROM optimizer_processes WHERE status = ?",
                    (status,),
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM optimizer_processes")
            row = cursor.fetchone()
            return int(row[0] if row else 0)

    def update_process(self, process_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not kwargs:
            return self.get_process(process_id)

        set_clauses = []
        values = []
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)
        values.append(process_id)

        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                f"UPDATE optimizer_processes SET {', '.join(set_clauses)} WHERE id = ?",
                tuple(values),
            )
            self._db.commit(conn)

        return self.get_process(process_id)

    def delete_process(self, process_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = self._db.cursor(conn)
            cursor.execute(
                "DELETE FROM optimizer_processes WHERE id = ?",
                (process_id,),
            )
            self._db.commit(conn)
            return cursor.rowcount > 0
