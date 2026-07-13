"""Durable classification job persistence for async execution."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .persistence_common import (
    ensure_index,
    fetch_rows,
    qualified_table as _qualified_table,
)
from .timing import utc_now_iso

logger = logging.getLogger(__name__)

ClassificationJobType = Literal["single", "batch", "chat"]
ClassificationJobStatus = Literal["queued", "running", "completed", "failed", "partial_failed"]
ClassificationJobItemStatus = Literal["queued", "running", "completed", "failed"]
DocumentMode = Literal["text_only", "with_documents"]
SUPERSEDED_ERROR_MESSAGE = "Superseded by newer classification request"
PUBLIC_QUEUED_STATUS = "queued"
MM_QUEUED_DB_STATUS = "queued_mm"
ACTIVE_DB_STATUSES = (MM_QUEUED_DB_STATUS, PUBLIC_QUEUED_STATUS, "running")
ACTIVE_DB_STATUS_SQL = "'queued_mm', 'queued', 'running'"
QUEUED_DB_STATUS_SQL = "'queued_mm', 'queued'"


def _is_queued_db_status(status: object) -> bool:
    """Return whether a persisted status represents pending queue work."""

    return str(status) in {MM_QUEUED_DB_STATUS, PUBLIC_QUEUED_STATUS}


def _is_active_db_status(status: object) -> bool:
    """Return whether a persisted status represents non-terminal queue work."""

    return str(status) in set(ACTIVE_DB_STATUSES)


def _public_status(status: object) -> str:
    """Map internal durable queue statuses to public API statuses."""

    status_text = str(status)
    if _is_queued_db_status(status_text):
        return PUBLIC_QUEUED_STATUS
    return status_text

@dataclass(frozen=True)
class ClassificationJobItemSeed:
    item_id: str
    dataset_scope: str
    position: int
    document_mode: DocumentMode = "text_only"
    include_token_usage: bool = False


@dataclass(frozen=True)
class PersistedClassificationJob:
    job_id: str
    job_type: ClassificationJobType
    status: ClassificationJobStatus
    submitted_at: str
    total_count: int
    completed_count: int
    failed_count: int
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None


@dataclass(frozen=True)
class PersistedClassificationJobItem:
    job_item_id: str
    job_id: str
    item_id: str
    dataset_scope: str
    position: int
    document_mode: DocumentMode
    include_token_usage: bool
    status: ClassificationJobItemStatus
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    last_classified_at: Optional[str] = None


def _final_job_status(*, total_count: int, completed_count: int, failed_count: int) -> ClassificationJobStatus:
    if total_count <= 0:
        return "completed"
    if failed_count <= 0 and completed_count == total_count:
        return "completed"
    if completed_count <= 0 and failed_count == total_count:
        return "failed"
    return "partial_failed"


class InMemoryClassificationJobStore:
    """Test-friendly in-memory job store mirroring the HANA-backed semantics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: Dict[str, Dict[str, object]] = {}
        self._job_items: Dict[str, List[Dict[str, object]]] = {}
        self._ownership: Dict[Tuple[str, str], Dict[str, str]] = {}

    def submit_job(
        self,
        *,
        job_type: ClassificationJobType,
        items: Sequence[ClassificationJobItemSeed],
    ) -> PersistedClassificationJob:
        with self._lock:
            submitted_at = utc_now_iso()
            job_id = str(uuid.uuid4())
            total_count = len(items)
            status = MM_QUEUED_DB_STATUS if total_count > 0 else "completed"
            finished_at = None if total_count > 0 else submitted_at
            self._jobs[job_id] = {
                "job_id": job_id,
                "job_type": job_type,
                "status": status,
                "submitted_at": submitted_at,
                "started_at": None,
                "finished_at": finished_at,
                "total_count": total_count,
                "completed_count": 0,
                "failed_count": 0,
                "error_message": None,
                "claimed_by": None,
                "claimed_at": None,
            }
            self._job_items[job_id] = [
                {
                    "job_item_id": str(uuid.uuid4()),
                    "job_id": job_id,
                    "item_id": item.item_id,
                    "dataset_scope": item.dataset_scope,
                    "position": int(item.position),
                    "document_mode": item.document_mode,
                    "include_token_usage": bool(item.include_token_usage),
                    "status": MM_QUEUED_DB_STATUS,
                    "started_at": None,
                    "finished_at": None,
                    "error_message": None,
                    "last_classified_at": None,
                }
                for item in items
            ]
            self._assign_ownership_locked(job_id, items)
            return self._row_to_job(self._jobs[job_id])

    def get_job(self, job_id: str) -> Optional[PersistedClassificationJob]:
        with self._lock:
            row = self._jobs.get(job_id)
            if row is None:
                return None
            return self._row_to_job(row)

    def get_job_items(self, job_id: str) -> List[PersistedClassificationJobItem]:
        with self._lock:
            rows = sorted(self._job_items.get(job_id, []), key=lambda row: int(row["position"]))
            return [self._row_to_job_item(row) for row in rows]

    def get_active_item_ids(self, keys: Iterable[Tuple[str, str]]) -> List[str]:
        requested = {(item_id, dataset_scope) for item_id, dataset_scope in keys}
        active_item_ids: List[str] = []
        with self._lock:
            for job_id, job_row in self._jobs.items():
                if not _is_active_db_status(job_row["status"]):
                    continue
                for item_row in self._job_items.get(job_id, []):
                    key = (str(item_row["item_id"]), str(item_row["dataset_scope"]))
                    if not _is_active_db_status(item_row["status"]):
                        continue
                    if key in requested and str(item_row["item_id"]) not in active_item_ids:
                        active_item_ids.append(str(item_row["item_id"]))
        return active_item_ids

    def supersede_active_items(
        self,
        keys: Iterable[Tuple[str, str]],
        *,
        error_message: str = SUPERSEDED_ERROR_MESSAGE,
    ) -> int:
        requested = {(item_id, dataset_scope) for item_id, dataset_scope in keys}
        if not requested:
            return 0
        with self._lock:
            now = utc_now_iso()
            touched_jobs: Set[str] = set()
            superseded = 0
            for job_id, item_rows in self._job_items.items():
                for item_row in item_rows:
                    key = (str(item_row["item_id"]), str(item_row["dataset_scope"]))
                    if key not in requested or not _is_active_db_status(item_row["status"]):
                        continue
                    item_row["status"] = "failed"
                    item_row["started_at"] = item_row.get("started_at") or now
                    item_row["finished_at"] = now
                    item_row["error_message"] = error_message
                    superseded += 1
                    touched_jobs.add(job_id)
            self._clear_ownership_locked(requested)
            for job_id in touched_jobs:
                self._refresh_job_counts_locked(job_id)
            return superseded

    def is_item_owned_by_job(self, item_id: str, dataset_scope: str, job_id: str) -> bool:
        with self._lock:
            row = self._ownership.get((item_id, dataset_scope))
            return row is not None and row["job_id"] == job_id

    def claim_next_queued_job(self, *, worker_id: str) -> Optional[PersistedClassificationJob]:
        with self._lock:
            queued_rows = [
                row for row in self._jobs.values() if row["status"] == MM_QUEUED_DB_STATUS
            ]
            if not queued_rows:
                return None
            queued_rows.sort(key=lambda row: (str(row["submitted_at"]), str(row["job_id"])))
            row = queued_rows[0]
            claimed_at = utc_now_iso()
            row["status"] = "running"
            row["claimed_by"] = worker_id
            row["claimed_at"] = claimed_at
            row["started_at"] = row.get("started_at") or claimed_at
            return self._row_to_job(row)

    def mark_job_items_running(self, job_id: str) -> None:
        with self._lock:
            now = utc_now_iso()
            for item_row in self._job_items.get(job_id, []):
                if not _is_queued_db_status(item_row["status"]):
                    continue
                item_row["status"] = "running"
                item_row["started_at"] = item_row.get("started_at") or now
            self._refresh_job_counts_locked(job_id)

    def record_job_item_result(
        self,
        job_id: str,
        *,
        position: int,
        status: ClassificationJobItemStatus,
        error_message: Optional[str] = None,
        last_classified_at: Optional[str] = None,
    ) -> None:
        with self._lock:
            for item_row in self._job_items.get(job_id, []):
                if int(item_row["position"]) != int(position):
                    continue
                now = utc_now_iso()
                item_row["status"] = status
                item_row["started_at"] = item_row.get("started_at") or now
                item_row["finished_at"] = now
                item_row["error_message"] = error_message
                item_row["last_classified_at"] = last_classified_at
                break
            self._refresh_job_counts_locked(job_id)

    def fail_job(self, job_id: str, *, error_message: str) -> None:
        with self._lock:
            now = utc_now_iso()
            for item_row in self._job_items.get(job_id, []):
                if item_row["status"] in {"completed", "failed"}:
                    continue
                item_row["status"] = "failed"
                item_row["started_at"] = item_row.get("started_at") or now
                item_row["finished_at"] = now
                item_row["error_message"] = error_message
            self._refresh_job_counts_locked(job_id, error_message=error_message)

    def cancel_all_active_jobs(self) -> int:
        """Fail all queued/running jobs and their items. Returns the count of cancelled jobs."""
        with self._lock:
            now = utc_now_iso()
            cancelled = 0
            cleared_keys: Set[Tuple[str, str]] = set()
            for job_id, job_row in self._jobs.items():
                if not _is_active_db_status(job_row["status"]):
                    continue
                cancelled += 1
                for item_row in self._job_items.get(job_id, []):
                    if item_row["status"] in {"completed", "failed"}:
                        continue
                    cleared_keys.add((str(item_row["item_id"]), str(item_row["dataset_scope"])))
                    item_row["status"] = "failed"
                    item_row["started_at"] = item_row.get("started_at") or now
                    item_row["finished_at"] = now
                    item_row["error_message"] = "Cancelled by classification reset"
                self._refresh_job_counts_locked(job_id, error_message="Cancelled by classification reset")
            self._clear_ownership_locked(cleared_keys)
            return cancelled

    def _refresh_job_counts_locked(self, job_id: str, *, error_message: Optional[str] = None) -> None:
        row = self._jobs.get(job_id)
        if row is None:
            return
        item_rows = self._job_items.get(job_id, [])
        total_count = len(item_rows)
        completed_count = sum(1 for item_row in item_rows if item_row["status"] == "completed")
        failed_count = sum(1 for item_row in item_rows if item_row["status"] == "failed")
        queued_count = sum(1 for item_row in item_rows if _is_queued_db_status(item_row["status"]))
        running_count = sum(1 for item_row in item_rows if item_row["status"] == "running")
        terminal_count = sum(1 for item_row in item_rows if item_row["status"] in {"completed", "failed"})
        row["total_count"] = total_count
        row["completed_count"] = completed_count
        row["failed_count"] = failed_count
        if error_message is not None:
            row["error_message"] = error_message
        if total_count <= 0:
            row["status"] = "completed"
            row["finished_at"] = row.get("finished_at") or utc_now_iso()
            return
        if terminal_count < total_count:
            row["status"] = "running" if running_count > 0 else MM_QUEUED_DB_STATUS
            row["finished_at"] = None
            if queued_count > 0 and running_count <= 0 and error_message is None:
                row["error_message"] = None
            return
        row["status"] = _final_job_status(
            total_count=total_count,
            completed_count=completed_count,
            failed_count=failed_count,
        )
        row["finished_at"] = utc_now_iso()
        if row["status"] == "completed":
            row["error_message"] = None

    def _assign_ownership_locked(self, job_id: str, items: Sequence[ClassificationJobItemSeed]) -> None:
        assigned_at = utc_now_iso()
        for item in items:
            self._ownership[(item.item_id, item.dataset_scope)] = {
                "job_id": job_id,
                "assigned_at": assigned_at,
            }

    def _clear_ownership_locked(self, keys: Iterable[Tuple[str, str]]) -> None:
        for key in keys:
            self._ownership.pop(key, None)

    @staticmethod
    def _row_to_job(row: Dict[str, object]) -> PersistedClassificationJob:
        return PersistedClassificationJob(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),  # type: ignore[arg-type]
            status=_public_status(row["status"]),  # type: ignore[arg-type]
            submitted_at=str(row["submitted_at"]),
            started_at=(None if row.get("started_at") is None else str(row["started_at"])),
            finished_at=(None if row.get("finished_at") is None else str(row["finished_at"])),
            total_count=int(row.get("total_count") or 0),
            completed_count=int(row.get("completed_count") or 0),
            failed_count=int(row.get("failed_count") or 0),
            error_message=(None if row.get("error_message") is None else str(row["error_message"])),
            claimed_by=(None if row.get("claimed_by") is None else str(row["claimed_by"])),
            claimed_at=(None if row.get("claimed_at") is None else str(row["claimed_at"])),
        )

    @staticmethod
    def _row_to_job_item(row: Dict[str, object]) -> PersistedClassificationJobItem:
        return PersistedClassificationJobItem(
            job_item_id=str(row["job_item_id"]),
            job_id=str(row["job_id"]),
            item_id=str(row["item_id"]),
            dataset_scope=str(row["dataset_scope"]),
            position=int(row["position"]),
            document_mode=str(row.get("document_mode") or "text_only"),  # type: ignore[arg-type]
            include_token_usage=row.get("include_token_usage") in {True, 1, "1", "true", "True"},
            status=_public_status(row["status"]),  # type: ignore[arg-type]
            started_at=(None if row.get("started_at") is None else str(row["started_at"])),
            finished_at=(None if row.get("finished_at") is None else str(row["finished_at"])),
            error_message=(None if row.get("error_message") is None else str(row["error_message"])),
            last_classified_at=(
                None if row.get("last_classified_at") is None else str(row["last_classified_at"])
            ),
        )


class ClassificationJobStore:
    """Persist async classification jobs in SAP HANA."""

    def __init__(
        self,
        settings: MetalCompositionSettings,
        *,
        connection: Optional[HANAConnection] = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or HANAConnection()
        self.schema = settings.ui_state_hana_schema or settings.hana_schema or None
        self.jobs_table = settings.ui_state_classification_jobs_table
        self.job_items_table = settings.ui_state_classification_job_items_table
        self.ownership_table = settings.ui_state_classification_ownership_table
        self._initialize()

    def _initialize(self) -> None:
        if not self.connection.table_exists(self.jobs_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.jobs_table, self.schema)} (
                    "JOB_ID" NVARCHAR(36) PRIMARY KEY,
                    "JOB_TYPE" NVARCHAR(32) NOT NULL,
                    "STATUS" NVARCHAR(32) NOT NULL,
                    "SUBMITTED_AT" NVARCHAR(64) NOT NULL,
                    "STARTED_AT" NVARCHAR(64),
                    "FINISHED_AT" NVARCHAR(64),
                    "TOTAL_COUNT" INTEGER NOT NULL,
                    "COMPLETED_COUNT" INTEGER NOT NULL,
                    "FAILED_COUNT" INTEGER NOT NULL,
                    "ERROR_MESSAGE" NCLOB,
                    "CLAIMED_BY" NVARCHAR(255),
                    "CLAIMED_AT" NVARCHAR(64)
                )
                """
            )
            ensure_index(
                self.connection,
                self.jobs_table,
                schema=self.schema,
                columns=("STATUS", "SUBMITTED_AT"),
            )
        if not self.connection.table_exists(self.job_items_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.job_items_table, self.schema)} (
                    "JOB_ITEM_ID" NVARCHAR(36) PRIMARY KEY,
                    "JOB_ID" NVARCHAR(36) NOT NULL,
                    "ITEM_ID" NVARCHAR(255) NOT NULL,
                    "DATASET_SCOPE" NVARCHAR(255) NOT NULL,
                    "POSITION" INTEGER NOT NULL,
                    "DOCUMENT_MODE" NVARCHAR(32) DEFAULT 'text_only' NOT NULL,
                    "INCLUDE_TOKEN_USAGE" INTEGER DEFAULT 0 NOT NULL,
                    "STATUS" NVARCHAR(32) NOT NULL,
                    "STARTED_AT" NVARCHAR(64),
                    "FINISHED_AT" NVARCHAR(64),
                    "ERROR_MESSAGE" NCLOB,
                    "LAST_CLASSIFIED_AT" NVARCHAR(64)
                )
                """
            )
            ensure_index(
                self.connection,
                self.job_items_table,
                schema=self.schema,
                columns=("JOB_ID", "POSITION"),
            )
            ensure_index(
                self.connection,
                self.job_items_table,
                schema=self.schema,
                columns=("ITEM_ID", "DATASET_SCOPE"),
            )
        else:
            if not self.connection.column_exists(self.job_items_table, "DOCUMENT_MODE", schema=self.schema):
                self.connection.execute(
                    f"""
                    ALTER TABLE {_qualified_table(self.job_items_table, self.schema)}
                    ADD ("DOCUMENT_MODE" NVARCHAR(32) DEFAULT 'text_only' NOT NULL)
                    """
                )
            if not self.connection.column_exists(self.job_items_table, "INCLUDE_TOKEN_USAGE", schema=self.schema):
                self.connection.execute(
                    f"""
                    ALTER TABLE {_qualified_table(self.job_items_table, self.schema)}
                    ADD ("INCLUDE_TOKEN_USAGE" INTEGER DEFAULT 0 NOT NULL)
                    """
                )
        if not self.connection.table_exists(self.ownership_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.ownership_table, self.schema)} (
                    "ITEM_ID" NVARCHAR(255) NOT NULL,
                    "DATASET_SCOPE" NVARCHAR(255) NOT NULL,
                    "JOB_ID" NVARCHAR(36) NOT NULL,
                    "ASSIGNED_AT" NVARCHAR(64) NOT NULL,
                    PRIMARY KEY ("ITEM_ID", "DATASET_SCOPE")
                )
                """
            )

    def _fetch_rows(self, sql: str, params: Sequence[object] | None = None) -> List[Dict[str, object]]:
        return fetch_rows(self.connection, sql, params)

    def submit_job(
        self,
        *,
        job_type: ClassificationJobType,
        items: Sequence[ClassificationJobItemSeed],
    ) -> PersistedClassificationJob:
        submitted_at = utc_now_iso()
        job_id = str(uuid.uuid4())
        total_count = len(items)
        status = MM_QUEUED_DB_STATUS if total_count > 0 else "completed"
        finished_at = None if total_count > 0 else submitted_at
        logger.info(
            "submit_job: inserting job %s (type=%s, items=%d, status=%s)",
            job_id, job_type, total_count, status,
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {_qualified_table(self.jobs_table, self.schema)} (
                    "JOB_ID", "JOB_TYPE", "STATUS", "SUBMITTED_AT", "STARTED_AT", "FINISHED_AT",
                    "TOTAL_COUNT", "COMPLETED_COUNT", "FAILED_COUNT", "ERROR_MESSAGE", "CLAIMED_BY", "CLAIMED_AT"
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    job_id,
                    job_type,
                    status,
                    submitted_at,
                    None,
                    finished_at,
                    total_count,
                    0,
                    0,
                    None,
                    None,
                    None,
                ],
            )
            for item in items:
                cursor.execute(
                    f"""
                    INSERT INTO {_qualified_table(self.job_items_table, self.schema)} (
                        "JOB_ITEM_ID", "JOB_ID", "ITEM_ID", "DATASET_SCOPE", "POSITION", "DOCUMENT_MODE",
                        "INCLUDE_TOKEN_USAGE",
                        "STATUS", "STARTED_AT", "FINISHED_AT", "ERROR_MESSAGE", "LAST_CLASSIFIED_AT"
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(uuid.uuid4()),
                        job_id,
                        item.item_id,
                        item.dataset_scope,
                        int(item.position),
                        item.document_mode,
                        1 if item.include_token_usage else 0,
                        MM_QUEUED_DB_STATUS,
                        None,
                        None,
                        None,
                        None,
                    ],
                )
                cursor.execute(
                    f'DELETE FROM {_qualified_table(self.ownership_table, self.schema)} '
                    f'WHERE "ITEM_ID" = ? AND "DATASET_SCOPE" = ?',
                    [item.item_id, item.dataset_scope],
                )
                cursor.execute(
                    f"""
                    INSERT INTO {_qualified_table(self.ownership_table, self.schema)} (
                        "ITEM_ID", "DATASET_SCOPE", "JOB_ID", "ASSIGNED_AT"
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [item.item_id, item.dataset_scope, job_id, submitted_at],
                )
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Unable to load classification job {job_id}")
        logger.info("submit_job: committed job %s to HANA (status=%s)", job_id, job.status)
        return job

    def get_job(self, job_id: str) -> Optional[PersistedClassificationJob]:
        rows = self._fetch_rows(
            f"""
            SELECT "JOB_ID", "JOB_TYPE", "STATUS", "SUBMITTED_AT", "STARTED_AT", "FINISHED_AT",
                   "TOTAL_COUNT", "COMPLETED_COUNT", "FAILED_COUNT", "ERROR_MESSAGE",
                   "CLAIMED_BY", "CLAIMED_AT"
            FROM {_qualified_table(self.jobs_table, self.schema)}
            WHERE "JOB_ID" = ?
            """,
            [job_id],
        )
        if not rows:
            return None
        return self._row_to_job(rows[0])

    def get_job_items(self, job_id: str) -> List[PersistedClassificationJobItem]:
        rows = self._fetch_rows(
            f"""
            SELECT "JOB_ITEM_ID", "JOB_ID", "ITEM_ID", "DATASET_SCOPE", "POSITION",
                   "DOCUMENT_MODE", "INCLUDE_TOKEN_USAGE", "STATUS", "STARTED_AT", "FINISHED_AT",
                   "ERROR_MESSAGE", "LAST_CLASSIFIED_AT"
            FROM {_qualified_table(self.job_items_table, self.schema)}
            WHERE "JOB_ID" = ?
            ORDER BY "POSITION" ASC, "JOB_ITEM_ID" ASC
            """,
            [job_id],
        )
        return [self._row_to_job_item(row) for row in rows]

    def get_active_item_ids(self, keys: Iterable[Tuple[str, str]]) -> List[str]:
        key_list = list(keys)
        if not key_list:
            return []
        clauses = " OR ".join('("I"."ITEM_ID" = ? AND "I"."DATASET_SCOPE" = ?)' for _ in key_list)
        params: List[str] = []
        for item_id, dataset_scope in key_list:
            params.extend((item_id, dataset_scope))
        rows = self._fetch_rows(
            f"""
            SELECT DISTINCT "I"."ITEM_ID"
            FROM {_qualified_table(self.job_items_table, self.schema)} AS "I"
            INNER JOIN {_qualified_table(self.jobs_table, self.schema)} AS "J"
                ON "J"."JOB_ID" = "I"."JOB_ID"
            WHERE "J"."STATUS" IN ({ACTIVE_DB_STATUS_SQL})
              AND "I"."STATUS" IN ({ACTIVE_DB_STATUS_SQL})
              AND ({clauses})
            ORDER BY "I"."ITEM_ID" ASC
            """,
            params,
        )
        return [str(row["item_id"]) for row in rows]

    def supersede_active_items(
        self,
        keys: Iterable[Tuple[str, str]],
        *,
        error_message: str = SUPERSEDED_ERROR_MESSAGE,
    ) -> int:
        """Fail queued/running job items for the provided item keys.

        Inputs:
            keys: Item id and dataset-scope pairs that should be retired before
                a newer classification job is submitted.
            error_message: Failure reason to persist on the superseded rows.

        Expected output:
            The number of active job-item rows selected for superseding.
        """
        key_list = list(dict.fromkeys(keys))
        if not key_list:
            return 0
        now = utc_now_iso()
        clauses = " OR ".join('("ITEM_ID" = ? AND "DATASET_SCOPE" = ?)' for _ in key_list)
        params: List[object] = [now, now, error_message]
        for item_id, dataset_scope in key_list:
            params.extend((item_id, dataset_scope))
        touched_rows = self._fetch_rows(
            f"""
            SELECT "JOB_ID", COUNT(*) AS "ITEM_COUNT"
            FROM {_qualified_table(self.job_items_table, self.schema)}
            WHERE "STATUS" IN ({ACTIVE_DB_STATUS_SQL}) AND ({clauses})
            GROUP BY "JOB_ID"
            """,
            params[3:],
        )
        if not touched_rows:
            return 0
        superseded_count = sum(int(row["item_count"] or 0) for row in touched_rows)
        self.connection.execute(
            f"""
            UPDATE {_qualified_table(self.job_items_table, self.schema)}
            SET "STATUS" = 'failed',
                "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END,
                "FINISHED_AT" = CASE WHEN "FINISHED_AT" IS NULL THEN ? ELSE "FINISHED_AT" END,
                "ERROR_MESSAGE" = ?
            WHERE "STATUS" IN ({ACTIVE_DB_STATUS_SQL}) AND ({clauses})
            """,
            params,
        )
        delete_sql = (
            f'DELETE FROM {_qualified_table(self.ownership_table, self.schema)} '
            f'WHERE {clauses}'
        )
        delete_params: List[object] = []
        for item_id, dataset_scope in key_list:
            delete_params.extend((item_id, dataset_scope))
        self.connection.execute(delete_sql, delete_params)
        for row in touched_rows:
            self._refresh_job_counts(str(row["job_id"]))
        return superseded_count

    def is_item_owned_by_job(self, item_id: str, dataset_scope: str, job_id: str) -> bool:
        rows = self._fetch_rows(
            f"""
            SELECT "JOB_ID"
            FROM {_qualified_table(self.ownership_table, self.schema)}
            WHERE "ITEM_ID" = ? AND "DATASET_SCOPE" = ?
            """,
            [item_id, dataset_scope],
        )
        if not rows:
            return False
        return str(rows[0]["job_id"]) == job_id

    def claim_next_queued_job(self, *, worker_id: str) -> Optional[PersistedClassificationJob]:
        rows = self._fetch_rows(
            f"""
            SELECT "JOB_ID"
            FROM {_qualified_table(self.jobs_table, self.schema)}
            WHERE "STATUS" = {repr(MM_QUEUED_DB_STATUS)}
            ORDER BY "SUBMITTED_AT" ASC, "JOB_ID" ASC
            LIMIT 1
            """
        )
        if not rows:
            return None
        job_id = str(rows[0]["job_id"])
        logger.info("claim_next_queued_job: found queued job %s, attempting claim", job_id)
        claimed_at = utc_now_iso()
        with self.connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {_qualified_table(self.jobs_table, self.schema)}
                SET "STATUS" = 'running',
                    "CLAIMED_BY" = ?,
                    "CLAIMED_AT" = ?,
                    "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END
                WHERE "JOB_ID" = ? AND "STATUS" = {repr(MM_QUEUED_DB_STATUS)}
                """,
                [worker_id, claimed_at, claimed_at, job_id],
            )
            if int(cursor.rowcount or 0) <= 0:
                logger.warning("claim_next_queued_job: UPDATE rowcount=0, another worker may have claimed job %s", job_id)
                return None
        logger.info("claim_next_queued_job: successfully claimed job %s", job_id)
        return self.get_job(job_id)

    def mark_job_items_running(self, job_id: str) -> None:
        now = utc_now_iso()
        self.connection.execute(
            f"""
            UPDATE {_qualified_table(self.job_items_table, self.schema)}
            SET "STATUS" = 'running',
                "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END
            WHERE "JOB_ID" = ? AND "STATUS" = {repr(MM_QUEUED_DB_STATUS)}
            """,
            [now, job_id],
        )
        self._refresh_job_counts(job_id)

    def record_job_item_result(
        self,
        job_id: str,
        *,
        position: int,
        status: ClassificationJobItemStatus,
        error_message: Optional[str] = None,
        last_classified_at: Optional[str] = None,
    ) -> None:
        now = utc_now_iso()
        self.connection.execute(
            f"""
            UPDATE {_qualified_table(self.job_items_table, self.schema)}
            SET "STATUS" = ?,
                "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END,
                "FINISHED_AT" = ?,
                "ERROR_MESSAGE" = ?,
                "LAST_CLASSIFIED_AT" = ?
            WHERE "JOB_ID" = ? AND "POSITION" = ?
            """,
            [status, now, now, error_message, last_classified_at, job_id, int(position)],
        )
        self._refresh_job_counts(job_id)

    def fail_job(self, job_id: str, *, error_message: str) -> None:
        now = utc_now_iso()
        self.connection.execute(
            f"""
            UPDATE {_qualified_table(self.job_items_table, self.schema)}
            SET "STATUS" = 'failed',
                "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END,
                "FINISHED_AT" = CASE WHEN "FINISHED_AT" IS NULL THEN ? ELSE "FINISHED_AT" END,
                "ERROR_MESSAGE" = CASE WHEN "ERROR_MESSAGE" IS NULL THEN ? ELSE "ERROR_MESSAGE" END
            WHERE "JOB_ID" = ? AND "STATUS" NOT IN ('completed', 'failed')
            """,
            [now, now, error_message, job_id],
        )
        self._refresh_job_counts(job_id, error_message=error_message)

    def cancel_all_active_jobs(self) -> int:
        """Fail all queued/running jobs and their items. Returns the count of cancelled jobs."""
        now = utc_now_iso()
        error_message = "Cancelled by classification reset"
        active_rows = self._fetch_rows(
            f"""
            SELECT "JOB_ID"
            FROM {_qualified_table(self.jobs_table, self.schema)}
            WHERE "STATUS" IN ({ACTIVE_DB_STATUS_SQL})
            """
        )
        if not active_rows:
            return 0
        active_item_rows = self._fetch_rows(
            f"""
            SELECT DISTINCT "ITEM_ID", "DATASET_SCOPE"
            FROM {_qualified_table(self.job_items_table, self.schema)}
            WHERE "STATUS" IN ({ACTIVE_DB_STATUS_SQL})
            """
        )
        for row in active_rows:
            job_id = str(row["job_id"])
            self.connection.execute(
                f"""
                UPDATE {_qualified_table(self.job_items_table, self.schema)}
                SET "STATUS" = 'failed',
                    "STARTED_AT" = CASE WHEN "STARTED_AT" IS NULL THEN ? ELSE "STARTED_AT" END,
                    "FINISHED_AT" = CASE WHEN "FINISHED_AT" IS NULL THEN ? ELSE "FINISHED_AT" END,
                    "ERROR_MESSAGE" = CASE WHEN "ERROR_MESSAGE" IS NULL THEN ? ELSE "ERROR_MESSAGE" END
                WHERE "JOB_ID" = ? AND "STATUS" NOT IN ('completed', 'failed')
                """,
                [now, now, error_message, job_id],
            )
            self._refresh_job_counts(job_id, error_message=error_message)
        if active_item_rows:
            clauses = " OR ".join('("ITEM_ID" = ? AND "DATASET_SCOPE" = ?)' for _ in active_item_rows)
            params: List[object] = []
            for row in active_item_rows:
                params.extend((str(row["item_id"]), str(row["dataset_scope"])))
            self.connection.execute(
                f'DELETE FROM {_qualified_table(self.ownership_table, self.schema)} WHERE {clauses}',
                params,
            )
        return len(active_rows)

    def _refresh_job_counts(self, job_id: str, *, error_message: Optional[str] = None) -> None:
        rows = self._fetch_rows(
            f"""
            SELECT COUNT(*) AS "total_count",
                   SUM(CASE WHEN "STATUS" = 'completed' THEN 1 ELSE 0 END) AS "completed_count",
                   SUM(CASE WHEN "STATUS" = 'failed' THEN 1 ELSE 0 END) AS "failed_count",
                   SUM(CASE WHEN "STATUS" IN ({QUEUED_DB_STATUS_SQL}) THEN 1 ELSE 0 END) AS "queued_count",
                   SUM(CASE WHEN "STATUS" = 'running' THEN 1 ELSE 0 END) AS "running_count",
                   SUM(CASE WHEN "STATUS" IN ('completed', 'failed') THEN 1 ELSE 0 END) AS "terminal_count"
            FROM {_qualified_table(self.job_items_table, self.schema)}
            WHERE "JOB_ID" = ?
            """,
            [job_id],
        )
        row = rows[0] if rows else {
            "total_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "queued_count": 0,
            "running_count": 0,
            "terminal_count": 0,
        }
        total_count = int(row["total_count"] or 0)
        completed_count = int(row["completed_count"] or 0)
        failed_count = int(row["failed_count"] or 0)
        queued_count = int(row["queued_count"] or 0)
        running_count = int(row["running_count"] or 0)
        terminal_count = int(row["terminal_count"] or 0)
        finished_at = None
        status: str = MM_QUEUED_DB_STATUS
        final_error = error_message
        if total_count <= 0:
            status = "completed"
            finished_at = utc_now_iso()
            final_error = None
        elif terminal_count >= total_count:
            status = _final_job_status(
                total_count=total_count,
                completed_count=completed_count,
                failed_count=failed_count,
            )
            finished_at = utc_now_iso()
            if status == "completed":
                final_error = None
        elif running_count > 0:
            status = "running"
        self.connection.execute(
            f"""
            UPDATE {_qualified_table(self.jobs_table, self.schema)}
            SET "STATUS" = ?,
                "TOTAL_COUNT" = ?,
                "COMPLETED_COUNT" = ?,
                "FAILED_COUNT" = ?,
                "FINISHED_AT" = CASE WHEN ? IS NULL THEN "FINISHED_AT" ELSE ? END,
                "ERROR_MESSAGE" = ?
            WHERE "JOB_ID" = ?
            """,
            [status, total_count, completed_count, failed_count, finished_at, finished_at, final_error, job_id],
        )

    @staticmethod
    def _row_to_job(row: Dict[str, object]) -> PersistedClassificationJob:
        return PersistedClassificationJob(
            job_id=str(row["job_id"]),
            job_type=str(row["job_type"]),  # type: ignore[arg-type]
            status=_public_status(row["status"]),  # type: ignore[arg-type]
            submitted_at=str(row["submitted_at"]),
            started_at=(None if row["started_at"] is None else str(row["started_at"])),
            finished_at=(None if row["finished_at"] is None else str(row["finished_at"])),
            total_count=int(row["total_count"] or 0),
            completed_count=int(row["completed_count"] or 0),
            failed_count=int(row["failed_count"] or 0),
            error_message=(None if row["error_message"] is None else str(row["error_message"])),
            claimed_by=(None if row["claimed_by"] is None else str(row["claimed_by"])),
            claimed_at=(None if row["claimed_at"] is None else str(row["claimed_at"])),
        )

    @staticmethod
    def _row_to_job_item(row: Dict[str, object]) -> PersistedClassificationJobItem:
        return PersistedClassificationJobItem(
            job_item_id=str(row["job_item_id"]),
            job_id=str(row["job_id"]),
            item_id=str(row["item_id"]),
            dataset_scope=str(row["dataset_scope"]),
            position=int(row["position"]),
            document_mode=str(row.get("document_mode") or "text_only"),  # type: ignore[arg-type]
            include_token_usage=row.get("include_token_usage") in {True, 1, "1", "true", "True"},
            status=_public_status(row["status"]),  # type: ignore[arg-type]
            started_at=(None if row["started_at"] is None else str(row["started_at"])),
            finished_at=(None if row["finished_at"] is None else str(row["finished_at"])),
            error_message=(None if row["error_message"] is None else str(row["error_message"])),
            last_classified_at=(
                None if row["last_classified_at"] is None else str(row["last_classified_at"])
            ),
        )
