"""Asynchronous extraction job storage and in-process execution."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from ..models.extraction import ExtractionJobStatusResponse, ExtractionJobSubmitResponse, JobResultFile
from .extraction_service import ExtractionConfig, run_extraction_for_paths

logger = logging.getLogger(__name__)

JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"

ACTIVE_STATUSES = {JOB_STATUS_QUEUED, JOB_STATUS_RUNNING}
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

DEFAULT_JOBS_TABLE = "EXTRACTION_JOBS"
DEFAULT_FILES_TABLE = "EXTRACTION_JOB_FILES"


class QueueFullError(RuntimeError):
    """Raised when the API has reached the configured active job limit."""


class JobNotFoundError(RuntimeError):
    """Raised when a caller requests an unknown extraction job."""


class JobResultNotReadyError(RuntimeError):
    """Raised when a caller downloads a job that has not succeeded yet."""


class JobStorageError(RuntimeError):
    """Raised when HANA-backed job storage cannot be initialized or queried."""


@dataclass(frozen=True, slots=True)
class JobFilePayload:
    """Uploaded file content stored before a job starts.

    Args:
        filename: Original browser-provided file name.
        content_type: Browser-provided MIME type, defaulting to PDF.
        content: Raw uploaded PDF bytes.
    """

    filename: str
    content_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class StoredJobFile:
    """File content retrieved from job storage for worker execution.

    Args:
        file_index: Stable order of the uploaded file.
        filename: Original browser-provided file name.
        content_type: Stored MIME type.
        content: Raw PDF bytes.
    """

    file_index: int
    filename: str
    content_type: str
    content: bytes


class ExtractionJobRepository(Protocol):
    """Storage contract used by the in-process job manager."""

    def create_job(self, job_id: str, files: Sequence[JobFilePayload], config: ExtractionConfig, created_at: str) -> None:
        """Persist a new queued job and its uploaded files."""

    def count_active_jobs(self) -> int:
        """Return the number of queued or running jobs."""

    def get_files(self, job_id: str) -> List[StoredJobFile]:
        """Return uploaded files for a job in upload order."""

    def get_status(self, job_id: str) -> ExtractionJobStatusResponse:
        """Return the current status payload for a job."""

    def get_result_file(self, job_id: str) -> JobResultFile:
        """Return the completed Excel artifact for a job."""

    def mark_running(self, job_id: str, started_at: str) -> None:
        """Mark a queued job as running."""

    def update_progress(self, job_id: str, progress: int, stage: str, message: str) -> None:
        """Persist coarse job progress."""

    def mark_succeeded(
        self,
        job_id: str,
        result_payload: Dict[str, Any],
        output_filename: str,
        content_type: str,
        content: bytes,
        finished_at: str,
    ) -> None:
        """Persist the final payload and Excel artifact for a successful job."""

    def mark_failed(self, job_id: str, error_message: str, finished_at: str) -> None:
        """Persist an unrecoverable job failure."""


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format.

    Returns:
        A timestamp string with a trailing ``Z`` for UTC.
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hana_timestamp(value: str) -> datetime:
    """Convert an ISO timestamp string to a HANA-compatible UTC datetime.

    Args:
        value: ISO timestamp ending in ``Z`` or an explicit offset.

    Returns:
        A timezone-naive UTC datetime suitable for HANA TIMESTAMP columns.
    """

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _to_iso(value: Any) -> Optional[str]:
    """Convert HANA timestamp values into API timestamp strings.

    Args:
        value: Value returned by hdbcli or in-memory storage.

    Returns:
        A UTC ISO string, or ``None`` when the database value is null.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    text = str(value)
    if text.endswith("Z"):
        return text
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    return parsed.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_loads(value: Any, default: Any) -> Any:
    """Decode JSON stored in HANA NCLOB columns.

    Args:
        value: Serialized JSON string or null.
        default: Value returned when the database column is empty.

    Returns:
        The parsed JSON value or the provided default.
    """

    if value in (None, ""):
        return default
    return json.loads(str(value))


def _bytes_from_blob(value: Any) -> bytes:
    """Normalize hdbcli BLOB values into bytes.

    Args:
        value: hdbcli BLOB, bytes, memoryview, or bytearray.

    Returns:
        Raw bytes stored in the database.
    """

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "read"):
        return value.read()
    return bytes(value)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert HANA boolean-like values to Python bools.

    Args:
        value: Boolean, numeric, string, or null database value.
        default: Value returned when ``value`` is null.

    Returns:
        Normalized boolean value.
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_filename(filename: str, fallback: str = "document.pdf") -> str:
    """Normalize a browser file name before writing it to a temp directory.

    Args:
        filename: Browser-provided file name.
        fallback: Name used when the input is blank or path-like.

    Returns:
        A path-free file name safe for local materialization.
    """

    candidate = Path(filename or fallback).name
    return candidate or fallback


class InMemoryExtractionJobRepository:
    """Process-local job repository used by unit tests.

    The class mirrors the HANA repository contract without external services.
    It is intentionally not used by production code.
    """

    def __init__(self) -> None:
        """Create empty in-memory job, file, and result stores."""

        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._files: Dict[str, List[StoredJobFile]] = {}
        self._results: Dict[str, JobResultFile] = {}

    def create_job(self, job_id: str, files: Sequence[JobFilePayload], config: ExtractionConfig, created_at: str) -> None:
        """Persist a queued job and its uploaded files in memory."""

        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": JOB_STATUS_QUEUED,
                "progress": 0,
                "stage": "queued",
                "message": "Job queued.",
                "created_at": created_at,
                "updated_at": created_at,
                "started_at": None,
                "finished_at": None,
                "file_count": len(files),
                "llm_verify": config.llm_verify,
                "top_k": config.top_k,
                "runtime_seconds": None,
                "reference_data_version": None,
                "output_filename": None,
                "output_size": None,
                "headers_preview": [],
                "line_items_preview": [],
                "errors": [],
                "warnings": [],
            }
            self._files[job_id] = [
                StoredJobFile(
                    file_index=index,
                    filename=file.filename,
                    content_type=file.content_type,
                    content=file.content,
                )
                for index, file in enumerate(files)
            ]

    def count_active_jobs(self) -> int:
        """Return the count of queued and running in-memory jobs."""

        with self._lock:
            return sum(1 for job in self._jobs.values() if job["status"] in ACTIVE_STATUSES)

    def get_files(self, job_id: str) -> List[StoredJobFile]:
        """Return uploaded files for an in-memory job."""

        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(f"Job not found: {job_id}")
            return list(self._files.get(job_id, []))

    def get_status(self, job_id: str) -> ExtractionJobStatusResponse:
        """Return an API status payload for an in-memory job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            return ExtractionJobStatusResponse(**job, download_url=f"/api/extraction/jobs/{job_id}/download")

    def get_result_file(self, job_id: str) -> JobResultFile:
        """Return the completed in-memory result file."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            result = self._results.get(job_id)
            if job["status"] != JOB_STATUS_SUCCEEDED or result is None:
                raise JobResultNotReadyError("Job result is not ready.")
            return result

    def mark_running(self, job_id: str, started_at: str) -> None:
        """Mark an in-memory job as running."""

        with self._lock:
            job = self._require_job(job_id)
            job.update(
                {
                    "status": JOB_STATUS_RUNNING,
                    "progress": 10,
                    "stage": "preparing",
                    "message": "Preparing uploaded PDFs.",
                    "started_at": started_at,
                    "updated_at": started_at,
                }
            )

    def update_progress(self, job_id: str, progress: int, stage: str, message: str) -> None:
        """Persist coarse in-memory progress for a job."""

        with self._lock:
            job = self._require_job(job_id)
            job.update({"progress": progress, "stage": stage, "message": message, "updated_at": utc_now_iso()})

    def mark_succeeded(
        self,
        job_id: str,
        result_payload: Dict[str, Any],
        output_filename: str,
        content_type: str,
        content: bytes,
        finished_at: str,
    ) -> None:
        """Persist successful in-memory job output and metadata."""

        with self._lock:
            job = self._require_job(job_id)
            job.update(
                {
                    "status": JOB_STATUS_SUCCEEDED,
                    "progress": 100,
                    "stage": "completed",
                    "message": "Extraction complete.",
                    "updated_at": finished_at,
                    "finished_at": finished_at,
                    "runtime_seconds": result_payload.get("runtime_seconds"),
                    "reference_data_version": result_payload.get("reference_data_version"),
                    "output_filename": output_filename,
                    "output_size": len(content),
                    "headers_preview": result_payload.get("headers_preview") or [],
                    "line_items_preview": result_payload.get("line_items_preview") or [],
                    "errors": result_payload.get("errors") or [],
                    "warnings": result_payload.get("warnings") or [],
                }
            )
            self._results[job_id] = JobResultFile(filename=output_filename, content_type=content_type, content=content)

    def mark_failed(self, job_id: str, error_message: str, finished_at: str) -> None:
        """Persist an in-memory job failure."""

        with self._lock:
            job = self._require_job(job_id)
            job.update(
                {
                    "status": JOB_STATUS_FAILED,
                    "stage": "failed",
                    "message": "Extraction failed.",
                    "updated_at": finished_at,
                    "finished_at": finished_at,
                    "errors": [error_message],
                }
            )

    def _require_job(self, job_id: str) -> Dict[str, Any]:
        """Return a mutable in-memory job record or raise when missing."""

        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return job


class HanaExtractionJobRepository:
    """HANA-backed job repository for extraction submissions and artifacts."""

    _JOB_COLUMNS = {
        "JOB_ID": {"NVARCHAR"},
        "STATUS": {"NVARCHAR"},
        "CREATED_AT": {"TIMESTAMP"},
        "UPDATED_AT": {"TIMESTAMP"},
        "STARTED_AT": {"TIMESTAMP"},
        "FINISHED_AT": {"TIMESTAMP"},
        "CONFIG_JSON": {"NCLOB"},
        "PROGRESS": {"INTEGER"},
        "STAGE": {"NVARCHAR"},
        "MESSAGE": {"NVARCHAR"},
        "FILE_COUNT": {"INTEGER"},
        "LLM_VERIFY": {"BOOLEAN", "TINYINT"},
        "TOP_K": {"INTEGER"},
        "RUNTIME_SECONDS": {"DOUBLE"},
        "REFERENCE_DATA_VERSION": {"NVARCHAR"},
        "OUTPUT_FILENAME": {"NVARCHAR"},
        "OUTPUT_MIME_TYPE": {"NVARCHAR"},
        "OUTPUT_SIZE": {"BIGINT"},
        "RESULT_BLOB": {"BLOB"},
        "RESULT_METADATA_JSON": {"NCLOB"},
        "HEADERS_PREVIEW_JSON": {"NCLOB"},
        "LINE_ITEMS_PREVIEW_JSON": {"NCLOB"},
        "ERRORS_JSON": {"NCLOB"},
        "WARNINGS_JSON": {"NCLOB"},
    }
    _FILE_COLUMNS = {
        "JOB_ID": {"NVARCHAR"},
        "FILE_INDEX": {"INTEGER"},
        "FILENAME": {"NVARCHAR"},
        "MIME_TYPE": {"NVARCHAR"},
        "SIZE_BYTES": {"BIGINT"},
        "CONTENT_BLOB": {"BLOB"},
    }

    def __init__(
        self,
        jobs_table: str | None = None,
        files_table: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Create a repository configured from HANA environment variables.

        Args:
            jobs_table: Optional override for the job table name.
            files_table: Optional override for the uploaded-files table name.
            schema: Optional schema override. Defaults to ``HANA_SCHEMA`` or
                the runtime user's current schema.
        """

        self.jobs_table = self._validate_identifier(
            jobs_table or os.getenv("HANA_EXTRACTION_JOBS_TABLE", DEFAULT_JOBS_TABLE)
        )
        self.files_table = self._validate_identifier(
            files_table or os.getenv("HANA_EXTRACTION_JOB_FILES_TABLE", DEFAULT_FILES_TABLE)
        )
        self.schema = self._validate_identifier(schema or os.getenv("HANA_SCHEMA", "").strip() or None)
        self._ready = False
        self._ready_lock = threading.Lock()

    def create_job(self, job_id: str, files: Sequence[JobFilePayload], config: ExtractionConfig, created_at: str) -> None:
        """Insert a queued job row and uploaded PDF BLOBs into HANA."""

        def _write(connection) -> None:
            now = _hana_timestamp(created_at)
            job_sql = (
                f"INSERT INTO {self._qualified(self.jobs_table)} "
                '("JOB_ID", "STATUS", "CREATED_AT", "UPDATED_AT", "CONFIG_JSON", "PROGRESS", '
                '"STAGE", "MESSAGE", "FILE_COUNT", "LLM_VERIFY", "TOP_K", "ERRORS_JSON", "WARNINGS_JSON") '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            file_sql = (
                f"INSERT INTO {self._qualified(self.files_table)} "
                '("JOB_ID", "FILE_INDEX", "FILENAME", "MIME_TYPE", "SIZE_BYTES", "CONTENT_BLOB") '
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            cursor = connection.cursor()
            try:
                cursor.execute(
                    job_sql,
                    (
                        job_id,
                        JOB_STATUS_QUEUED,
                        now,
                        now,
                        json.dumps(asdict(config)),
                        0,
                        "queued",
                        "Job queued.",
                        len(files),
                        bool(config.llm_verify),
                        int(config.top_k),
                        "[]",
                        "[]",
                    ),
                )
                rows = [
                    (
                        job_id,
                        index,
                        file.filename,
                        file.content_type,
                        len(file.content),
                        file.content,
                    )
                    for index, file in enumerate(files)
                ]
                if rows:
                    cursor.executemany(file_sql, rows)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

        self._with_connection(_write)

    def count_active_jobs(self) -> int:
        """Count queued and running jobs in HANA."""

        def _read(connection) -> int:
            sql = (
                f"SELECT COUNT(*) FROM {self._qualified(self.jobs_table)} "
                'WHERE "STATUS" IN (?, ?)'
            )
            cursor = connection.cursor()
            try:
                cursor.execute(sql, (JOB_STATUS_QUEUED, JOB_STATUS_RUNNING))
                return int(cursor.fetchone()[0])
            finally:
                cursor.close()

        return self._with_connection(_read)

    def get_files(self, job_id: str) -> List[StoredJobFile]:
        """Fetch uploaded PDF BLOBs from HANA for a job."""

        def _read(connection) -> List[StoredJobFile]:
            self._ensure_job_exists(connection, job_id)
            sql = (
                f"SELECT \"FILE_INDEX\", \"FILENAME\", \"MIME_TYPE\", \"CONTENT_BLOB\" "
                f"FROM {self._qualified(self.files_table)} "
                'WHERE "JOB_ID" = ? ORDER BY "FILE_INDEX"'
            )
            cursor = connection.cursor()
            try:
                cursor.execute(sql, (job_id,))
                return [
                    StoredJobFile(
                        file_index=int(row[0]),
                        filename=str(row[1]),
                        content_type=str(row[2]),
                        content=_bytes_from_blob(row[3]),
                    )
                    for row in cursor.fetchall()
                ]
            finally:
                cursor.close()

        return self._with_connection(_read)

    def get_status(self, job_id: str) -> ExtractionJobStatusResponse:
        """Return a status payload built from the HANA job row."""

        def _read(connection) -> ExtractionJobStatusResponse:
            row = self._fetch_job_row(connection, job_id)
            return self._row_to_status(row)

        return self._with_connection(_read)

    def get_result_file(self, job_id: str) -> JobResultFile:
        """Return a completed Excel artifact stored in HANA."""

        def _read(connection) -> JobResultFile:
            row = self._fetch_job_row(connection, job_id)
            if row["STATUS"] != JOB_STATUS_SUCCEEDED or row["RESULT_BLOB"] is None:
                raise JobResultNotReadyError("Job result is not ready.")
            return JobResultFile(
                filename=row["OUTPUT_FILENAME"] or "commodity_codes.xlsx",
                content_type=row["OUTPUT_MIME_TYPE"] or EXCEL_MIME_TYPE,
                content=_bytes_from_blob(row["RESULT_BLOB"]),
            )

        return self._with_connection(_read)

    def mark_running(self, job_id: str, started_at: str) -> None:
        """Mark a HANA job as running with a preparing stage."""

        self._update_job(
            job_id,
            {
                "STATUS": JOB_STATUS_RUNNING,
                "PROGRESS": 10,
                "STAGE": "preparing",
                "MESSAGE": "Preparing uploaded PDFs.",
                "STARTED_AT": _hana_timestamp(started_at),
                "UPDATED_AT": _hana_timestamp(started_at),
            },
        )

    def update_progress(self, job_id: str, progress: int, stage: str, message: str) -> None:
        """Update coarse progress values for a HANA job."""

        self._update_job(
            job_id,
            {
                "PROGRESS": int(progress),
                "STAGE": stage,
                "MESSAGE": message,
                "UPDATED_AT": _hana_timestamp(utc_now_iso()),
            },
        )

    def mark_succeeded(
        self,
        job_id: str,
        result_payload: Dict[str, Any],
        output_filename: str,
        content_type: str,
        content: bytes,
        finished_at: str,
    ) -> None:
        """Store successful job metadata and the result workbook in HANA."""

        self._update_job(
            job_id,
            {
                "STATUS": JOB_STATUS_SUCCEEDED,
                "PROGRESS": 100,
                "STAGE": "completed",
                "MESSAGE": "Extraction complete.",
                "UPDATED_AT": _hana_timestamp(finished_at),
                "FINISHED_AT": _hana_timestamp(finished_at),
                "RUNTIME_SECONDS": result_payload.get("runtime_seconds"),
                "REFERENCE_DATA_VERSION": result_payload.get("reference_data_version"),
                "OUTPUT_FILENAME": output_filename,
                "OUTPUT_MIME_TYPE": content_type,
                "OUTPUT_SIZE": len(content),
                "RESULT_BLOB": content,
                "RESULT_METADATA_JSON": json.dumps(
                    {key: value for key, value in result_payload.items() if key not in {"headers_preview", "line_items_preview"}}
                ),
                "HEADERS_PREVIEW_JSON": json.dumps(result_payload.get("headers_preview") or []),
                "LINE_ITEMS_PREVIEW_JSON": json.dumps(result_payload.get("line_items_preview") or []),
                "ERRORS_JSON": json.dumps(result_payload.get("errors") or []),
                "WARNINGS_JSON": json.dumps(result_payload.get("warnings") or []),
            },
        )

    def mark_failed(self, job_id: str, error_message: str, finished_at: str) -> None:
        """Store failed job state and diagnostic message in HANA."""

        self._update_job(
            job_id,
            {
                "STATUS": JOB_STATUS_FAILED,
                "STAGE": "failed",
                "MESSAGE": "Extraction failed.",
                "UPDATED_AT": _hana_timestamp(finished_at),
                "FINISHED_AT": _hana_timestamp(finished_at),
                "ERRORS_JSON": json.dumps([error_message]),
            },
        )

    def _connect(self):
        """Open a HANA connection using the deployment environment."""

        try:
            from hdbcli import dbapi
        except ImportError as exc:  # pragma: no cover - depends on deployment dependency
            raise JobStorageError("hdbcli is required for HANA-backed extraction jobs.") from exc

        address = os.getenv("hana_address", "").strip()
        port_raw = os.getenv("hana_port", "").strip()
        user = os.getenv("hana_user", "").strip()
        password = os.getenv("hana_password", "").strip()
        if not all([address, port_raw, user, password]):
            raise JobStorageError("Missing HANA credentials for extraction job storage.")

        encrypt = "true" if os.getenv("hana_encrypt", "true").strip().lower() in {"1", "true", "yes", "on"} else "false"
        validate_cert = (
            "true"
            if os.getenv("hana_ssl_validate_certificate", "false").strip().lower() in {"1", "true", "yes", "on"}
            else "false"
        )
        return dbapi.connect(
            address=address,
            port=int(port_raw),
            user=user,
            password=password,
            encrypt=encrypt,
            sslValidateCertificate=validate_cert,
        )

    def _with_connection(self, callback: Callable[[Any], Any]) -> Any:
        """Run a repository operation with initialized HANA tables."""

        connection = self._connect()
        try:
            self._ensure_ready(connection)
            return callback(connection)
        except JobNotFoundError:
            raise
        except JobResultNotReadyError:
            raise
        except Exception as exc:
            raise JobStorageError(f"Extraction job storage operation failed: {exc}") from exc
        finally:
            connection.close()

    def _ensure_ready(self, connection) -> None:
        """Create and validate required HANA tables once per process."""

        if self._ready:
            return
        with self._ready_lock:
            if self._ready:
                return
            if self.schema is None:
                self.schema = self._current_schema(connection)
            self._ensure_table(connection, self.jobs_table, self._JOB_COLUMNS, self._create_jobs_table_sql())
            self._ensure_table(connection, self.files_table, self._FILE_COLUMNS, self._create_files_table_sql())
            connection.commit()
            logger.info("Extraction job tables ready in HANA schema %s", self.schema)
            self._ready = True

    def _ensure_table(self, connection, table: str, required_columns: Dict[str, set[str]], create_sql: str) -> None:
        """Create a HANA table or validate its existing column contract."""

        if not self._table_exists(connection, table):
            cursor = connection.cursor()
            try:
                cursor.execute(create_sql)
            finally:
                cursor.close()
            return

        columns = self._table_columns(connection, table)
        problems = []
        for column, expected_types in required_columns.items():
            data_type = columns.get(column)
            if data_type is None:
                problems.append(f"missing {column}")
            elif data_type not in expected_types:
                problems.append(f"{column} has {data_type}, expected one of {sorted(expected_types)}")
        if problems:
            raise JobStorageError(f"HANA table {self._qualified(table)} has incompatible columns: {', '.join(problems)}")

    def _table_exists(self, connection, table: str) -> bool:
        """Return whether the configured HANA table already exists."""

        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?",
                (self.schema, table),
            )
            return int(cursor.fetchone()[0]) > 0
        finally:
            cursor.close()

    def _table_columns(self, connection, table: str) -> Dict[str, str]:
        """Return a mapping of HANA column names to data type names."""

        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT COLUMN_NAME, DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?",
                (self.schema, table),
            )
            return {str(row[0]): str(row[1]).upper() for row in cursor.fetchall()}
        finally:
            cursor.close()

    def _current_schema(self, connection) -> str:
        """Read the runtime user's current HANA schema."""

        cursor = connection.cursor()
        try:
            cursor.execute("SELECT CURRENT_SCHEMA FROM DUMMY")
            return self._validate_identifier(str(cursor.fetchone()[0]))
        finally:
            cursor.close()

    def _fetch_job_row(self, connection, job_id: str) -> Dict[str, Any]:
        """Fetch one HANA job row as a dictionary."""

        sql = f"SELECT * FROM {self._qualified(self.jobs_table)} WHERE \"JOB_ID\" = ?"
        cursor = connection.cursor()
        try:
            cursor.execute(sql, (job_id,))
            row = cursor.fetchone()
            if row is None:
                raise JobNotFoundError(f"Job not found: {job_id}")
            columns = [column[0] for column in cursor.description]
            return dict(zip(columns, row))
        finally:
            cursor.close()

    def _ensure_job_exists(self, connection, job_id: str) -> None:
        """Raise when a HANA job ID is unknown."""

        self._fetch_job_row(connection, job_id)

    def _row_to_status(self, row: Dict[str, Any]) -> ExtractionJobStatusResponse:
        """Convert a HANA job row into the public polling response."""

        job_id = str(row["JOB_ID"])
        config = _json_loads(row.get("CONFIG_JSON"), {})
        return ExtractionJobStatusResponse(
            job_id=job_id,
            status=str(row["STATUS"]),
            progress=int(row["PROGRESS"] or 0),
            stage=str(row["STAGE"] or ""),
            message=str(row["MESSAGE"] or ""),
            created_at=_to_iso(row["CREATED_AT"]) or "",
            updated_at=_to_iso(row["UPDATED_AT"]) or "",
            started_at=_to_iso(row.get("STARTED_AT")),
            finished_at=_to_iso(row.get("FINISHED_AT")),
            file_count=int(row.get("FILE_COUNT") or 0),
            llm_verify=_coerce_bool(row.get("LLM_VERIFY"), bool(config.get("llm_verify", False))),
            top_k=int(row.get("TOP_K") or config.get("top_k") or 5),
            runtime_seconds=row.get("RUNTIME_SECONDS"),
            reference_data_version=row.get("REFERENCE_DATA_VERSION"),
            output_filename=row.get("OUTPUT_FILENAME"),
            output_size=row.get("OUTPUT_SIZE"),
            download_url=f"/api/extraction/jobs/{job_id}/download",
            headers_preview=_json_loads(row.get("HEADERS_PREVIEW_JSON"), []),
            line_items_preview=_json_loads(row.get("LINE_ITEMS_PREVIEW_JSON"), []),
            errors=_json_loads(row.get("ERRORS_JSON"), []),
            warnings=_json_loads(row.get("WARNINGS_JSON"), []),
        )

    def _update_job(self, job_id: str, values: Dict[str, Any]) -> None:
        """Update selected HANA job columns using parameterized values."""

        if not values:
            return

        def _write(connection) -> None:
            self._ensure_job_exists(connection, job_id)
            assignments = ", ".join(f'"{column}" = ?' for column in values)
            sql = f"UPDATE {self._qualified(self.jobs_table)} SET {assignments} WHERE \"JOB_ID\" = ?"
            cursor = connection.cursor()
            try:
                cursor.execute(sql, (*values.values(), job_id))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                cursor.close()

        self._with_connection(_write)

    def _create_jobs_table_sql(self) -> str:
        """Return the HANA DDL for the extraction jobs table."""

        return f"""
CREATE COLUMN TABLE {self._qualified(self.jobs_table)} (
  "JOB_ID" NVARCHAR(64) NOT NULL,
  "STATUS" NVARCHAR(20) NOT NULL,
  "CREATED_AT" TIMESTAMP NOT NULL,
  "UPDATED_AT" TIMESTAMP NOT NULL,
  "STARTED_AT" TIMESTAMP,
  "FINISHED_AT" TIMESTAMP,
  "CONFIG_JSON" NCLOB NOT NULL,
  "PROGRESS" INTEGER NOT NULL,
  "STAGE" NVARCHAR(80) NOT NULL,
  "MESSAGE" NVARCHAR(500) NOT NULL,
  "FILE_COUNT" INTEGER NOT NULL,
  "LLM_VERIFY" BOOLEAN NOT NULL,
  "TOP_K" INTEGER NOT NULL,
  "RUNTIME_SECONDS" DOUBLE,
  "REFERENCE_DATA_VERSION" NVARCHAR(100),
  "OUTPUT_FILENAME" NVARCHAR(255),
  "OUTPUT_MIME_TYPE" NVARCHAR(128),
  "OUTPUT_SIZE" BIGINT,
  "RESULT_BLOB" BLOB,
  "RESULT_METADATA_JSON" NCLOB,
  "HEADERS_PREVIEW_JSON" NCLOB,
  "LINE_ITEMS_PREVIEW_JSON" NCLOB,
  "ERRORS_JSON" NCLOB,
  "WARNINGS_JSON" NCLOB,
  PRIMARY KEY ("JOB_ID")
)
""".strip()

    def _create_files_table_sql(self) -> str:
        """Return the HANA DDL for the uploaded job files table."""

        return f"""
CREATE COLUMN TABLE {self._qualified(self.files_table)} (
  "JOB_ID" NVARCHAR(64) NOT NULL,
  "FILE_INDEX" INTEGER NOT NULL,
  "FILENAME" NVARCHAR(255) NOT NULL,
  "MIME_TYPE" NVARCHAR(128) NOT NULL,
  "SIZE_BYTES" BIGINT NOT NULL,
  "CONTENT_BLOB" BLOB NOT NULL,
  PRIMARY KEY ("JOB_ID", "FILE_INDEX")
)
""".strip()

    def _qualified(self, table: str) -> str:
        """Return a quoted HANA table identifier, including schema when set."""

        if self.schema:
            return f"{self._quote_identifier(self.schema)}.{self._quote_identifier(table)}"
        return self._quote_identifier(table)

    def _validate_identifier(self, value: str | None) -> str | None:
        """Validate a HANA identifier configured through the environment."""

        if value is None:
            return None
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise JobStorageError(f"Invalid HANA identifier: {value!r}")
        return value

    def _quote_identifier(self, value: str) -> str:
        """Quote a validated HANA identifier."""

        return f'"{value.replace(chr(34), chr(34) * 2)}"'


class ExtractionJobManager:
    """Submit and execute extraction jobs in a bounded in-process executor."""

    def __init__(
        self,
        repository: ExtractionJobRepository,
        run_pipeline: Callable[[Sequence[Path], ExtractionConfig], Dict[str, Any]] = run_extraction_for_paths,
        max_workers: int | None = None,
        max_queued_jobs: int | None = None,
    ) -> None:
        """Create an in-process extraction job manager.

        Args:
            repository: Persistent storage for job state and artifacts.
            run_pipeline: Function that runs the existing extraction pipeline.
            max_workers: Number of concurrent background jobs.
            max_queued_jobs: Maximum number of queued plus running jobs.
        """

        worker_count = max_workers or int(os.getenv("EXTRACTION_JOB_WORKERS", "1"))
        self.repository = repository
        self.run_pipeline = run_pipeline
        self.max_queued_jobs = max_queued_jobs or int(os.getenv("EXTRACTION_MAX_QUEUED_JOBS", "20"))
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="extraction-job")
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, files: Sequence[JobFilePayload], config: ExtractionConfig) -> ExtractionJobSubmitResponse:
        """Persist a job and schedule it for background execution.

        Args:
            files: Uploaded PDF payloads.
            config: Extraction pipeline configuration.

        Returns:
            Job submission metadata for the HTTP response.

        Raises:
            ValueError: If no files are provided or a file is not a PDF.
            QueueFullError: If active job capacity is exhausted.
        """

        if not files:
            raise ValueError("At least one PDF must be uploaded.")
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise ValueError(f"Unsupported file type for '{file.filename}'. Only PDF is allowed.")

        with self._lock:
            active_count = self.repository.count_active_jobs()
            if active_count >= self.max_queued_jobs:
                raise QueueFullError("Too many extraction jobs are already queued or running.")

            job_id = str(uuid.uuid4())
            created_at = utc_now_iso()
            self.repository.create_job(job_id, files, config, created_at)
            future = self._executor.submit(self._run_job, job_id, config)
            self._futures[job_id] = future

        return ExtractionJobSubmitResponse(
            job_id=job_id,
            status=JOB_STATUS_QUEUED,
            status_url=f"/api/extraction/jobs/{job_id}",
            download_url=f"/api/extraction/jobs/{job_id}/download",
            created_at=created_at,
        )

    def get_status(self, job_id: str) -> ExtractionJobStatusResponse:
        """Return current status for a submitted job."""

        return self.repository.get_status(job_id)

    def get_result_file(self, job_id: str) -> JobResultFile:
        """Return a completed Excel file for a submitted job."""

        return self.repository.get_result_file(job_id)

    def wait_for_job(self, job_id: str, timeout_seconds: float) -> None:
        """Wait for a background job to finish; intended for tests.

        Args:
            job_id: Job identifier returned by ``submit``.
            timeout_seconds: Maximum time to wait.

        Raises:
            TimeoutError: If the background future does not finish in time.
        """

        future = self._futures.get(job_id)
        if future is None:
            deadline = datetime.now(timezone.utc).timestamp() + timeout_seconds
            while datetime.now(timezone.utc).timestamp() < deadline:
                status = self.repository.get_status(job_id).status
                if status not in ACTIVE_STATUSES:
                    return
            raise TimeoutError()
        future.result(timeout=timeout_seconds)

    def _run_job(self, job_id: str, config: ExtractionConfig) -> None:
        """Execute one job and persist its final state."""

        started_at = utc_now_iso()
        try:
            self.repository.mark_running(job_id, started_at)
            stored_files = self.repository.get_files(job_id)
            self.repository.update_progress(job_id, 25, "pipeline", "Running extraction and classification.")
            with tempfile.TemporaryDirectory(prefix=f"extraction_job_{job_id}_") as temp_dir:
                pdf_paths = self._materialize_files(stored_files, Path(temp_dir))
                result_payload = self.run_pipeline(pdf_paths, config)
                output_path = Path(str(result_payload["output_path"]))
                if not output_path.exists():
                    raise RuntimeError(f"Pipeline did not create output workbook: {output_path}")
                self.repository.update_progress(job_id, 90, "storing_result", "Storing generated workbook.")
                content = output_path.read_bytes()
                self.repository.mark_succeeded(
                    job_id=job_id,
                    result_payload=result_payload,
                    output_filename=output_path.name,
                    content_type=EXCEL_MIME_TYPE,
                    content=content,
                    finished_at=utc_now_iso(),
                )
                self._cleanup_output_file(output_path)
        except Exception as exc:  # pragma: no cover - exact branches depend on runtime pipeline failures
            logger.exception("Extraction job %s failed", job_id)
            self.repository.mark_failed(job_id, str(exc), utc_now_iso())

    def _materialize_files(self, files: Iterable[StoredJobFile], temp_dir: Path) -> List[Path]:
        """Write stored upload BLOBs into a temporary worker directory.

        Args:
            files: Stored PDF payloads for one job.
            temp_dir: Job-specific temporary directory.

        Returns:
            Ordered list of local PDF paths for the extraction pipeline.
        """

        paths: List[Path] = []
        for file in files:
            filename = _safe_filename(file.filename)
            path = temp_dir / f"{file.file_index:03d}_{filename}"
            path.write_bytes(file.content)
            paths.append(path.resolve())
        return paths

    def _cleanup_output_file(self, output_path: Path) -> None:
        """Remove the transient workbook after it has been stored in HANA.

        Args:
            output_path: Local workbook created by the existing extraction
                pipeline.
        """

        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove transient extraction output %s", output_path, exc_info=True)
