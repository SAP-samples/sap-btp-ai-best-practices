"""Safe persistence boundary for automated PS4 drawing assessments."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.persistence.codec import read_lob_text
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

ManufacturingFeasibilityState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED"]


@dataclass(frozen=True)
class ManufacturingFeasibilityRecord:
    """Stored report metadata and JSON; raw drawing bytes are never persisted."""

    assessment_id: str
    ecn_row_id: str
    material_number: str
    drawing_file_name: str
    drawing_sha256: str
    state: ManufacturingFeasibilityState
    report: dict[str, object] | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime


class ManufacturingFeasibilityRepository:
    """Persist and retrieve one automatic drawing assessment per ECN/material/drawing."""

    def __init__(self, engine: Engine, tables: PersistenceTableNames | None = None) -> None:
        self._engine = engine
        self._tables = tables or PersistenceTableNames()

    def create_queued(
        self,
        record: ManufacturingFeasibilityRecord,
    ) -> bool:
        """Insert a queued assessment and return False when it was already created."""
        table = quote_identifier(self._tables.manufacturing_feasibility_assessment)
        with self._engine.begin() as connection:
            existing = connection.execute(
                text(
                    f"SELECT 1 FROM {table} WHERE \"ECN_ROW_ID\" = :ecn_row_id "
                    'AND "MATERIAL_NUMBER" = :material_number '
                    'AND "DRAWING_SHA256" = :drawing_sha256'
                ),
                {
                    "ecn_row_id": record.ecn_row_id,
                    "material_number": record.material_number,
                    "drawing_sha256": record.drawing_sha256,
                },
            ).first()
            if existing is not None:
                return False
            connection.execute(
                text(
                    f"INSERT INTO {table} ("
                    '"ASSESSMENT_ID", "ECN_ROW_ID", "MATERIAL_NUMBER", '
                    '"DRAWING_FILE_NAME", "DRAWING_SHA256", "STATE", "REPORT_JSON", '
                    '"ERROR_CODE", "CREATED_AT", "UPDATED_AT") VALUES ('
                    ':assessment_id, :ecn_row_id, :material_number, :drawing_file_name, '
                    ':drawing_sha256, :state, :report_json, :error_code, :created_at, :updated_at)'
                ),
                {
                    "assessment_id": record.assessment_id,
                    "ecn_row_id": record.ecn_row_id,
                    "material_number": record.material_number,
                    "drawing_file_name": record.drawing_file_name,
                    "drawing_sha256": record.drawing_sha256,
                    "state": record.state,
                    "report_json": None,
                    "error_code": None,
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                },
            )
            return True

    def mark_running(self, assessment_id: str, updated_at: datetime) -> bool:
        """Transition a queued assessment to running without reopening finished work."""
        table = quote_identifier(self._tables.manufacturing_feasibility_assessment)
        with self._engine.begin() as connection:
            result = connection.execute(
                text(
                    f'UPDATE {table} SET "STATE" = :state, "UPDATED_AT" = :updated_at '
                    'WHERE "ASSESSMENT_ID" = :assessment_id AND "STATE" = :queued'
                ),
                {
                    "state": "RUNNING",
                    "updated_at": updated_at,
                    "assessment_id": assessment_id,
                    "queued": "QUEUED",
                },
            )
            return result.rowcount == 1

    def complete(
        self, assessment_id: str, report: dict[str, object], updated_at: datetime
    ) -> None:
        """Store the validated report JSON after the transient drawing call completes."""
        self._update_terminal(assessment_id, "COMPLETED", report, None, updated_at)

    def fail(self, assessment_id: str, error_code: str, updated_at: datetime) -> None:
        """Store only a controlled failure code; never persist exception or drawing content."""
        self._update_terminal(assessment_id, "FAILED", None, error_code, updated_at)

    def _update_terminal(
        self,
        assessment_id: str,
        state: Literal["COMPLETED", "FAILED"],
        report: dict[str, object] | None,
        error_code: str | None,
        updated_at: datetime,
    ) -> None:
        table = quote_identifier(self._tables.manufacturing_feasibility_assessment)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    f'UPDATE {table} SET "STATE" = :state, "REPORT_JSON" = :report_json, '
                    '"ERROR_CODE" = :error_code, "UPDATED_AT" = :updated_at '
                    'WHERE "ASSESSMENT_ID" = :assessment_id'
                ),
                {
                    "state": state,
                    "report_json": json.dumps(report, ensure_ascii=False, separators=(",", ":"))
                    if report is not None
                    else None,
                    "error_code": error_code,
                    "updated_at": updated_at,
                    "assessment_id": assessment_id,
                },
            )

    def get_completed(
        self, ecn_row_id: str, material_number: str, *, connection: Connection | None = None
    ) -> dict[str, object] | None:
        """Read the latest completed report, optionally on a caller-owned connection."""
        table = quote_identifier(self._tables.manufacturing_feasibility_assessment)
        with (self._engine.connect() if connection is None else nullcontext(connection)) as connection:
            value = connection.execute(
                text(
                    f'SELECT "REPORT_JSON" FROM {table} WHERE "ECN_ROW_ID" = :ecn_row_id '
                    'AND "MATERIAL_NUMBER" = :material_number AND "STATE" = :state '
                    'ORDER BY "UPDATED_AT" DESC'
                ),
                {
                    "ecn_row_id": ecn_row_id,
                    "material_number": material_number,
                    "state": "COMPLETED",
                },
            ).scalar()
            if value is None:
                return None
            return json.loads(read_lob_text(value))
