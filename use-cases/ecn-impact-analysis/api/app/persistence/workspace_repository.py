"""Explicit batch scheduling and append-only engineering review persistence."""

import json
from datetime import datetime, timezone
from functools import wraps

from sqlalchemy import bindparam, text

from app.ecn.errors import PersistenceNotFoundError, PersistenceStateError
from app.ecn.models import OUTPUT_SCHEMA_VERSION, PROMPT_VERSION
from app.ecn.services.lifecycle import HistoryMutationGuard
from app.persistence.codec import normalize_row_keys, read_lob_text
from app.persistence.hana import from_hana_timestamp, to_hana_timestamp
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier


def serialized_action(method):
    """Serialize short state changes with the shared history/publication mutation guard."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        """Run one action atomically with respect to other local lifecycle mutations."""
        with self.guard.hold():
            return method(self, *args, **kwargs)

    return guarded


class WorkspaceRepository:
    """Keep user actions durable while preserving source and attempt records."""

    def __init__(self, engine, table_names=None, guard=None):
        """Bind the HANA engine and optional isolated test namespace."""
        self.engine = engine
        self.guard = guard or HistoryMutationGuard()
        self.tables = table_names or PersistenceTableNames()

    def jobs(self, ecn_ids):
        """Read the exact selected jobs, without scheduling or retrieving source BLOBs."""
        statement = text(
            f'SELECT "JOB_ID", "ECN_ROW_ID", "STATE", "MODEL_NAME", "PROMPT_VERSION" '
            f'FROM {quote_identifier(self.tables.analysis_job)} WHERE "ECN_ROW_ID" IN :ids'
        ).bindparams(bindparam("ids", expanding=True))
        with self.engine.connect() as connection:
            found = [
                normalize_row_keys(r)
                for r in connection.execute(statement, {"ids": ecn_ids})
                .mappings()
                .all()
            ]
        if len(found) != len(set(ecn_ids)):
            raise PersistenceNotFoundError(
                "ECN_NOT_FOUND",
                "One or more selected ECNs no longer exist.",
                status_code=404,
            )
        lookup = {r["ECN_ROW_ID"]: r for r in found}
        return [lookup[key] for key in ecn_ids]

    def batch(self, batch_id):
        """Return a persisted selection and its current job states for polling/replay."""
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        f'SELECT * FROM {quote_identifier(self.tables.generation_batch)} WHERE "BATCH_ID" = :id'
                    ),
                    {"id": batch_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            row = normalize_row_keys(row)
            selection = json.loads(read_lob_text(row["SELECTION_JSON"]))
        jobs = self.jobs(selection["ecn_ids"])
        return {
            "batch_id": batch_id,
            "cancelled": bool(row["CANCELLED"]),
            "created_at": from_hana_timestamp(row["CREATED_AT"])
            if isinstance(row["CREATED_AT"], datetime)
            else row["CREATED_AT"],
            "selection": selection,
            "jobs": [{k.lower(): v for k, v in job.items()} for job in jobs],
            "finished": all(
                job["STATE"] not in {"QUEUED", "RUNNING", "BLOCKED"} for job in jobs
            ),
        }

    @serialized_action
    def create_batch(self, batch_id, ecn_ids):
        """Queue selected jobs on the current prompt while retaining prior results."""
        existing = self.batch(batch_id)
        if existing:
            if existing["selection"]["ecn_ids"] != ecn_ids:
                raise PersistenceStateError(
                    "BATCH_REQUEST_CONFLICT",
                    "This request ID belongs to a different selection.",
                    status_code=409,
                )
            return existing, ()
        jobs = self.jobs(ecn_ids)
        scheduled = []
        with self.engine.begin() as connection:
            for job in jobs:
                if job["STATE"] in {"READY", "FAILED", "BLOCKED", "COMPLETED"}:
                    count = connection.execute(
                        text(
                            f"UPDATE {quote_identifier(self.tables.analysis_job)} "
                            'SET "STATE" = \'QUEUED\', "ERROR_CODE" = NULL, "ERROR_MESSAGE" = NULL, "STARTED_AT" = NULL, "FINISHED_AT" = NULL, '
                            '"PROMPT_VERSION" = :prompt_version, "OUTPUT_SCHEMA_VERSION" = :schema_version '
                            'WHERE "JOB_ID" = :id AND "STATE" = :previous'
                        ),
                        {"id": job["JOB_ID"], "previous": job["STATE"],
                         "prompt_version": PROMPT_VERSION, "schema_version": OUTPUT_SCHEMA_VERSION},
                    ).rowcount
                    if count == 1:
                        scheduled.append(job["JOB_ID"])
            selection = {"ecn_ids": ecn_ids, "scheduled_job_ids": scheduled}
            connection.execute(
                text(
                    f"INSERT INTO {quote_identifier(self.tables.generation_batch)} "
                    '("BATCH_ID", "SELECTION_JSON", "CREATED_AT", "CANCELLED") VALUES (:id, :selection, :created, 0)'
                ),
                {
                    "id": batch_id,
                    "selection": json.dumps(selection),
                    "created": to_hana_timestamp(datetime.now(timezone.utc)),
                },
            )
            for job_id in scheduled:
                connection.execute(
                    text(
                        f'DELETE FROM {quote_identifier(self.tables.batch_job)} WHERE "JOB_ID" = :id'
                    ),
                    {"id": job_id},
                )
                connection.execute(
                    text(
                        f'INSERT INTO {quote_identifier(self.tables.batch_job)} ("JOB_ID", "BATCH_ID") VALUES (:job, :batch)'
                    ),
                    {"job": job_id, "batch": batch_id},
                )
        return self.batch(batch_id), tuple(scheduled)

    @serialized_action
    def cancel_batch(self, batch_id):
        """Stop this batch's unstarted jobs; never cancel running provider calls or other batches."""
        batch = self.batch(batch_id)
        if batch is None:
            raise PersistenceNotFoundError(
                "BATCH_NOT_FOUND", "Generation batch was not found.", status_code=404
            )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f'UPDATE {quote_identifier(self.tables.generation_batch)} SET "CANCELLED" = 1 WHERE "BATCH_ID" = :id'
                ),
                {"id": batch_id},
            )
            for job_id in batch["selection"]["scheduled_job_ids"]:
                connection.execute(
                    text(
                        f"UPDATE {quote_identifier(self.tables.analysis_job)} "
                        'SET "STATE" = \'READY\', "ERROR_CODE" = NULL, "ERROR_MESSAGE" = NULL '
                        f'WHERE "JOB_ID" = :id AND "STATE" IN (\'QUEUED\', \'BLOCKED\') AND "JOB_ID" IN (SELECT "JOB_ID" FROM {quote_identifier(self.tables.batch_job)} WHERE "BATCH_ID" = :batch)'
                    ),
                    {"id": job_id, "batch": batch_id},
                )
        return self.batch(batch_id)

    def reviews(self, ecn_row_id):
        """Read all attributed review statements against an exact immutable assessment."""
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        f"SELECT * FROM {quote_identifier(self.tables.review)} "
                        'WHERE "ECN_ROW_ID" = :id ORDER BY "CREATED_AT", "REVIEW_ID"'
                    ),
                    {"id": ecn_row_id},
                )
                .mappings()
                .all()
            )
            # Acceptances may carry no comment; a NULL NCLOB reads back as "".
            return [
                {
                    k.lower(): (
                        (read_lob_text(v) if v is not None else "")
                        if k == "COMMENTS"
                        else from_hana_timestamp(v)
                        if k == "CREATED_AT" and isinstance(v, datetime)
                        else v
                    )
                    for k, v in normalize_row_keys(r).items()
                    if k != "ACTOR_ID"
                }
                for r in rows
            ]

    @serialized_action
    def save_review(self, values):
        """Append an idempotent human statement; never overwrite source facts or previous reviews."""
        columns = tuple(values)
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    text(
                        f'SELECT * FROM {quote_identifier(self.tables.review)} WHERE "REVIEW_ID" = :id'
                    ),
                    {"id": values["REVIEW_ID"]},
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                stored = normalize_row_keys(existing)
                if any(
                    (
                        (read_lob_text(stored[k]) if stored[k] is not None else "")
                        if k == "COMMENTS"
                        else stored[k]
                    )
                    != value
                    for k, value in values.items()
                    if k != "CREATED_AT"
                ):
                    raise PersistenceStateError(
                        "REVIEW_REQUEST_CONFLICT",
                        "This request ID belongs to a different review submission.",
                        status_code=409,
                    )
                return
            connection.execute(
                text(
                    f"INSERT INTO {quote_identifier(self.tables.review)} "
                    "("
                    + ", ".join(f'"{c}"' for c in columns)
                    + ") VALUES ("
                    + ", ".join(f":{c}" for c in columns)
                    + ")"
                ),
                values,
            )

    def versions(self, ecn_row_id):
        """List assessments of the same original source position, preserving previous review links."""
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        f'SELECT E."ECN_ROW_ID", U."CREATED_AT", E."EVIDENCE_HASH" '
                        f'FROM {quote_identifier(self.tables.ecn)} E JOIN {quote_identifier(self.tables.upload)} U ON E."UPLOAD_ID" = U."UPLOAD_ID" '
                        f'JOIN {quote_identifier(self.tables.ecn)} S ON E."SOURCE_ORDINAL" = S."SOURCE_ORDINAL" '
                        f'JOIN {quote_identifier(self.tables.upload)} V ON S."UPLOAD_ID" = V."UPLOAD_ID" AND U."ECN_ARTIFACT_ID" = V."ECN_ARTIFACT_ID" '
                        'WHERE S."ECN_ROW_ID" = :id ORDER BY U."CREATED_AT" DESC, E."ECN_ROW_ID"'
                    ),
                    {"id": ecn_row_id},
                )
                .mappings()
                .all()
            )
            return [
                {
                    k.lower(): from_hana_timestamp(v)
                    if k == "CREATED_AT" and isinstance(v, datetime)
                    else v
                    for k, v in normalize_row_keys(row).items()
                }
                for row in rows
            ]
