"""Job registry for long-running optimizer solves.

The registry tracks one row per launched solve so the agent (and later a UI) can
poll status across the multi-minute run. Two implementations share one contract:

- :class:`HanaJobStore` -- the production store. It persists to the HANA table
  ``PRODUCTION_WHEEL_JOBS`` (all project tables use the ``PRODUCTION_WHEEL_*``
  prefix) and auto-creates that table on first use, so the code drops into a fresh
  customer landscape without a manual migration step.
- :class:`InMemoryJobStore` -- a dict-backed store for tests and for local
  development without HANA credentials.

Timestamps are stored as ISO-8601 UTC strings for deterministic round-tripping.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

JOBS_TABLE = "PRODUCTION_WHEEL_JOBS"


class JobStatus(StrEnum):
    """Lifecycle state of one optimizer solve job."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One tracked optimizer solve.

    Attributes:
        job_id: Opaque unique identifier.
        status: Current :class:`JobStatus`.
        mode: Short solve descriptor (for example ``"greenfield_pareto"``).
        run_dir: Extraction run directory the solve reads.
        output_dir: Directory the solve writes its bundle into.
        request_json: The serialized SolveRequest driving the solve.
        pid: Process id of the detached launcher shell.
        submitted_at: ISO-8601 UTC time the job was launched.
        started_at: ISO-8601 UTC time work began (same as submitted for now).
        finished_at: ISO-8601 UTC time a terminal state was observed.
        return_code: CLI exit code once known (0 valid, 2 invalid/partial).
        log_path: File capturing combined stdout/stderr of the solve.
        error: Human-readable failure detail when status is FAILED.
    """

    job_id: str
    status: JobStatus
    mode: str
    run_dir: str
    output_dir: str
    request_json: str
    pid: int | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping (status rendered as its string value)."""

        data = asdict(self)
        data["status"] = self.status.value
        return data


@runtime_checkable
class JobStore(Protocol):
    """Minimal persistence contract used by the runner."""

    def ensure_table(self) -> None:
        """Create the backing table if it does not already exist."""

    def insert(self, record: JobRecord) -> None:
        """Persist a new job record."""

    def update(self, record: JobRecord) -> None:
        """Overwrite an existing job record by ``job_id``."""

    def get(self, job_id: str) -> JobRecord | None:
        """Return one record or ``None`` when unknown."""

    def list(self) -> list[JobRecord]:
        """Return all records, newest submission first."""


class InMemoryJobStore:
    """Process-local job store for tests and HANA-less local development."""

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._records: dict[str, JobRecord] = {}

    def ensure_table(self) -> None:
        """No-op: the in-memory store needs no schema."""

    def insert(self, record: JobRecord) -> None:
        """Store a new record, rejecting duplicate identifiers."""

        if record.job_id in self._records:
            raise ValueError(f"duplicate job_id: {record.job_id}")
        self._records[record.job_id] = record

    def update(self, record: JobRecord) -> None:
        """Replace an existing record."""

        if record.job_id not in self._records:
            raise KeyError(record.job_id)
        self._records[record.job_id] = record

    def get(self, job_id: str) -> JobRecord | None:
        """Return the record for ``job_id`` or ``None``."""

        return self._records.get(job_id)

    def list(self) -> list[JobRecord]:
        """Return all records ordered by submission time, newest first."""

        return sorted(
            self._records.values(),
            key=lambda record: record.submitted_at or "",
            reverse=True,
        )


# Ordered columns kept in one place so DDL, INSERT, and SELECT stay in sync.
_COLUMNS: tuple[str, ...] = (
    "JOB_ID",
    "STATUS",
    "MODE",
    "RUN_DIR",
    "OUTPUT_DIR",
    "REQUEST_JSON",
    "PID",
    "SUBMITTED_AT",
    "STARTED_AT",
    "FINISHED_AT",
    "RETURN_CODE",
    "LOG_PATH",
    "ERROR",
)

_CREATE_TABLE_SQL = f"""
CREATE TABLE {JOBS_TABLE} (
    JOB_ID NVARCHAR(64) PRIMARY KEY,
    STATUS NVARCHAR(16) NOT NULL,
    MODE NVARCHAR(64),
    RUN_DIR NVARCHAR(1024),
    OUTPUT_DIR NVARCHAR(1024),
    REQUEST_JSON NCLOB,
    PID INTEGER,
    SUBMITTED_AT NVARCHAR(32),
    STARTED_AT NVARCHAR(32),
    FINISHED_AT NVARCHAR(32),
    RETURN_CODE INTEGER,
    LOG_PATH NVARCHAR(1024),
    ERROR NCLOB
)
"""


def _record_to_row(record: JobRecord) -> dict[str, object]:
    """Map a record to bound-parameter values keyed by column name."""

    return {
        "JOB_ID": record.job_id,
        "STATUS": record.status.value,
        "MODE": record.mode,
        "RUN_DIR": record.run_dir,
        "OUTPUT_DIR": record.output_dir,
        "REQUEST_JSON": record.request_json,
        "PID": record.pid,
        "SUBMITTED_AT": record.submitted_at,
        "STARTED_AT": record.started_at,
        "FINISHED_AT": record.finished_at,
        "RETURN_CODE": record.return_code,
        "LOG_PATH": record.log_path,
        "ERROR": record.error,
    }


def _row_to_record(row: dict[str, object]) -> JobRecord:
    """Map a fetched row (column name -> value) back to a :class:`JobRecord`."""

    return JobRecord(
        job_id=str(row["JOB_ID"]),
        status=JobStatus(str(row["STATUS"])),
        mode=str(row["MODE"]) if row["MODE"] is not None else "",
        run_dir=str(row["RUN_DIR"]) if row["RUN_DIR"] is not None else "",
        output_dir=str(row["OUTPUT_DIR"]) if row["OUTPUT_DIR"] is not None else "",
        request_json=str(row["REQUEST_JSON"]) if row["REQUEST_JSON"] is not None else "",
        pid=int(row["PID"]) if row["PID"] is not None else None,
        submitted_at=_optional_str(row.get("SUBMITTED_AT")),
        started_at=_optional_str(row.get("STARTED_AT")),
        finished_at=_optional_str(row.get("FINISHED_AT")),
        return_code=int(row["RETURN_CODE"]) if row.get("RETURN_CODE") is not None else None,
        log_path=_optional_str(row.get("LOG_PATH")),
        error=_optional_str(row.get("ERROR")),
    )


def _optional_str(value: object) -> str | None:
    """Return ``None`` unchanged, otherwise the string form of ``value``."""

    return None if value is None else str(value)


class HanaJobStore:
    """HANA-backed job registry that auto-creates ``PRODUCTION_WHEEL_JOBS``.

    A fresh :class:`HANAConnection` is opened per operation (job writes are
    infrequent relative to the multi-minute solves they track), and the table is
    ensured once per store instance.
    """

    def __init__(self, connection_factory=None) -> None:
        """Store an optional factory returning a ``HANAConnection`` (for tests)."""

        if connection_factory is None:
            from app.utils.hana import HANAConnection

            connection_factory = HANAConnection
        self._connection_factory = connection_factory
        self._ensured = False

    def _connect(self):
        """Open and return a connected HANA helper."""

        helper = self._connection_factory()
        helper.connect()
        return helper

    def ensure_table(self) -> None:
        """Create the jobs table when the schema does not already contain it."""

        if self._ensured:
            return
        from sqlalchemy import text

        helper = self._connect()
        try:
            exists = helper.connection.execute(
                text(
                    "SELECT COUNT(*) FROM SYS.TABLES "
                    "WHERE TABLE_NAME = :name AND SCHEMA_NAME = CURRENT_SCHEMA"
                ),
                {"name": JOBS_TABLE},
            ).fetchone()[0]
            if not exists:
                logger.info("Creating HANA table %s", JOBS_TABLE)
                helper.connection.execute(text(_CREATE_TABLE_SQL))
                helper.connection.commit()
            self._ensured = True
        finally:
            helper.disconnect()

    def insert(self, record: JobRecord) -> None:
        """Insert one job row after ensuring the table exists."""

        self.ensure_table()
        from sqlalchemy import text

        columns = ", ".join(_COLUMNS)
        placeholders = ", ".join(f":{column}" for column in _COLUMNS)
        helper = self._connect()
        try:
            helper.connection.execute(
                text(f"INSERT INTO {JOBS_TABLE} ({columns}) VALUES ({placeholders})"),
                _record_to_row(record),
            )
            helper.connection.commit()
        finally:
            helper.disconnect()

    def update(self, record: JobRecord) -> None:
        """Overwrite mutable columns of an existing job row."""

        self.ensure_table()
        from sqlalchemy import text

        mutable = [column for column in _COLUMNS if column != "JOB_ID"]
        assignments = ", ".join(f"{column} = :{column}" for column in mutable)
        helper = self._connect()
        try:
            helper.connection.execute(
                text(f"UPDATE {JOBS_TABLE} SET {assignments} WHERE JOB_ID = :JOB_ID"),
                _record_to_row(record),
            )
            helper.connection.commit()
        finally:
            helper.disconnect()

    def get(self, job_id: str) -> JobRecord | None:
        """Return one job row mapped to a record, or ``None`` when absent."""

        self.ensure_table()
        from sqlalchemy import text

        columns = ", ".join(_COLUMNS)
        helper = self._connect()
        try:
            result = helper.connection.execute(
                text(f"SELECT {columns} FROM {JOBS_TABLE} WHERE JOB_ID = :job_id"),
                {"job_id": job_id},
            )
            row = result.fetchone()
            if row is None:
                return None
            return _row_to_record(dict(zip(_COLUMNS, row)))
        finally:
            helper.disconnect()

    def list(self) -> list[JobRecord]:
        """Return every job row, newest submission first."""

        self.ensure_table()
        from sqlalchemy import text

        columns = ", ".join(_COLUMNS)
        helper = self._connect()
        try:
            result = helper.connection.execute(
                text(f"SELECT {columns} FROM {JOBS_TABLE} ORDER BY SUBMITTED_AT DESC")
            )
            return [_row_to_record(dict(zip(_COLUMNS, row))) for row in result.fetchall()]
        finally:
            helper.disconnect()


def store_from_env() -> JobStore:
    """Return the default production store (HANA-backed)."""

    return HanaJobStore()
