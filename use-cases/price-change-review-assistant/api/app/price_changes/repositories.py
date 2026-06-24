from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .models import AgentProposal, EmailAttachment, EmailAttachmentSummary, GmailEmail
from .schema import ensure_email_attachments_table, ensure_processing_progress_tables


REQUIRED_DRAFT_FIELDS = {
    "supplier_id": "supplier_id",
    "material_number": "material_code",
    "original_price": "original_price",
    "requested_new_price": "requested_new_price",
    "currency": "currency",
    "uom": "uom",
    "effective_from": "effective_from",
}


def normalize_decimal_bind_value(value: Any) -> Decimal | None:
    """Normalize optional draft price values before binding to HANA DECIMAL columns.

    Args:
        value: Price value from proposal assembly, possibly using a decimal comma.

    Returns:
        Decimal value suitable for HANA numeric binding, or None when missing or
        not parseable.
    """
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    text_value = text_value.replace("€", "").replace("$", "").replace(" ", "").replace("'", "")
    if "," in text_value and "." in text_value:
        if text_value.rfind(",") > text_value.rfind("."):
            text_value = text_value.replace(".", "").replace(",", ".")
        else:
            text_value = text_value.replace(",", "")
    elif "," in text_value:
        text_value = text_value.replace(",", ".")
    try:
        return Decimal(text_value)
    except InvalidOperation:
        return None


def draft_validation_errors(draft: dict[str, Any]) -> list[str]:
    """Return required-field errors for a draft row.

    Args:
        draft: Draft row using persisted database column names.

    Returns:
        Human-readable validation errors for missing required draft fields.
    """
    return [
        f"{field_name} is required"
        for field_name, draft_key in REQUIRED_DRAFT_FIELDS.items()
        for value in [draft.get(draft_key)]
        if value is None or value == ""
    ]


def parse_json_string_list(value: Any) -> list[str]:
    """Parse a JSON string list stored in HANA draft metadata.

    Args:
        value: Raw JSON text or list-like value.

    Returns:
        List of string values, or an empty list when missing or invalid.
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def merge_review_validation_errors(existing_errors: list[str], required_errors: list[str]) -> list[str]:
    """Merge semantic review errors with freshly computed required-field errors.

    Args:
        existing_errors: Errors previously stored on the draft.
        required_errors: Current required-field validation errors.

    Returns:
        Deduplicated errors that preserve non-required review reasons while
        dropping stale required-field errors after users fill missing fields.
    """
    required_error_names = {f"{field_name} is required" for field_name in REQUIRED_DRAFT_FIELDS}
    merged: list[str] = []
    for error in existing_errors:
        if error in required_error_names:
            continue
        if error not in merged:
            merged.append(error)
    for error in required_errors:
        if error not in merged:
            merged.append(error)
    return merged


def parse_json_object(value: Any) -> dict[str, Any]:
    """Parse a JSON object stored in HANA NCLOB columns.

    Args:
        value: Raw JSON value from HANA.

    Returns:
        Parsed object, or an empty object when missing or invalid.
    """
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def s4_write_audit_payload(raw_agent_output_json: Any) -> dict[str, Any]:
    """Return the mutable audit payload stored with a draft.

    Args:
        raw_agent_output_json: Existing raw agent output JSON.

    Returns:
        Parsed audit payload with S/4 write attempt containers.
    """
    payload = parse_json_object(raw_agent_output_json)
    attempts = payload.get("s4_write_attempts")
    if not isinstance(attempts, list):
        payload["s4_write_attempts"] = []
    return payload


def normalize_draft_review_state(draft: dict[str, Any]) -> dict[str, Any]:
    """Normalize draft status and validation errors for review lists.

    Args:
        draft: Draft row returned from HANA.

    Returns:
        Draft row with current required-field errors merged with semantic review errors.
    """
    normalized = dict(draft)
    errors = merge_review_validation_errors(
        parse_json_string_list(normalized.get("validation_errors_json")),
        draft_validation_errors(normalized),
    )
    normalized["validation_errors_json"] = json.dumps(errors, ensure_ascii=True)
    if normalized.get("status") in {None, "ready_for_review", "needs_human_review"}:
        normalized["status"] = "needs_human_review" if errors else "ready_for_review"
    audit_payload = parse_json_object(normalized.get("raw_agent_output_json"))
    s4_write_result = audit_payload.get("s4_write_result")
    if isinstance(s4_write_result, dict):
        normalized["s4_write_result"] = s4_write_result
    return normalized


class PriceChangeRepository:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def email_exists(self, gmail_message_id: str) -> bool:
        result = self.connection.execute(
            text("SELECT gmail_message_id FROM gmail_emails WHERE gmail_message_id = :gmail_message_id"),
            {"gmail_message_id": gmail_message_id},
        )
        return result.mappings().first() is not None

    def get_last_successful_fetch_at(self, mailbox_id: str) -> datetime | None:
        result = self.connection.execute(
            text(
                "SELECT last_successful_fetch_at FROM gmail_fetch_state "
                "WHERE mailbox_id = :mailbox_id AND last_status = 'success'"
            ),
            {"mailbox_id": mailbox_id},
        ).mappings().first()
        return None if result is None else result["last_successful_fetch_at"]

    def record_fetch_started(self, mailbox_id: str, started_at: datetime) -> None:
        self.connection.execute(
            text(
                "UPSERT gmail_fetch_state (mailbox_id, last_started_at, last_status) "
                "VALUES (:mailbox_id, :last_started_at, 'running') WITH PRIMARY KEY"
            ),
            {"mailbox_id": mailbox_id, "last_started_at": started_at},
        )

    def record_fetch_finished(
        self,
        mailbox_id: str,
        started_at: datetime,
        finished_at: datetime,
        status: str,
        summary_json: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        successful_at = finished_at if status == "success" else None
        self.connection.execute(
            text(
                "UPSERT gmail_fetch_state (mailbox_id, last_successful_fetch_at, last_started_at, "
                "last_finished_at, last_status, summary_json, error_message) VALUES "
                "(:mailbox_id, :last_successful_fetch_at, :last_started_at, :last_finished_at, "
                ":last_status, :summary_json, :error_message) WITH PRIMARY KEY"
            ),
            {
                "mailbox_id": mailbox_id,
                "last_successful_fetch_at": successful_at,
                "last_started_at": started_at,
                "last_finished_at": finished_at,
                "last_status": status,
                "summary_json": json.dumps(summary_json, ensure_ascii=True),
                "error_message": error_message,
            },
        )

    def create_processing_run(self, processing_run_id: str, source_type: str) -> None:
        """Create or reset one persistent processing progress run.

        Args:
            processing_run_id: Browser-generated run id used by polling.
            source_type: Source flow such as gmail_fetch or manual.

        Returns:
            None. The run is persisted in HANA.
        """
        ensure_processing_progress_tables(self.connection)
        now = datetime.now(timezone.utc)
        self.connection.execute(
            text(
                "UPSERT price_change_processing_runs (processing_run_id, source_type, status, started_at, "
                "finished_at, last_heartbeat_at, current_stage, current_message, error_message) VALUES "
                "(:processing_run_id, :source_type, :status, :started_at, NULL, :last_heartbeat_at, "
                "NULL, NULL, NULL) WITH PRIMARY KEY"
            ),
            {
                "processing_run_id": processing_run_id,
                "source_type": source_type,
                "status": "running",
                "started_at": now,
                "last_heartbeat_at": now,
            },
        )

    def append_processing_event(
        self,
        processing_run_id: str,
        stage: str,
        message: str,
        level: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one sanitized progress event and update the run heartbeat.

        Args:
            processing_run_id: Run id that owns the event.
            stage: Stable sanitized stage key.
            message: User-facing progress message.
            level: Event severity.
            metadata: Optional sanitized metadata.

        Returns:
            None. The event and current run stage are persisted in HANA.
        """
        ensure_processing_progress_tables(self.connection)
        sequence_result = self.connection.execute(
            text(
                "SELECT MAX(sequence_number) AS sequence_number FROM price_change_processing_events "
                "WHERE processing_run_id = :processing_run_id"
            ),
            {"processing_run_id": processing_run_id},
        ).mappings().first()
        current_sequence = 0
        if sequence_result and sequence_result.get("sequence_number") is not None:
            current_sequence = int(sequence_result["sequence_number"])
        sequence_number = current_sequence + 1
        now = datetime.now(timezone.utc)
        self.connection.execute(
            text(
                "INSERT INTO price_change_processing_events (event_id, processing_run_id, sequence_number, "
                "event_time, level, stage, message, metadata_json) VALUES (:event_id, :processing_run_id, "
                ":sequence_number, :event_time, :level, :stage, :message, :metadata_json)"
            ),
            {
                "event_id": f"evt-{uuid.uuid4().hex}",
                "processing_run_id": processing_run_id,
                "sequence_number": sequence_number,
                "event_time": now,
                "level": level,
                "stage": stage,
                "message": message,
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=True),
            },
        )
        self.connection.execute(
            text(
                "UPDATE price_change_processing_runs SET last_heartbeat_at = :last_heartbeat_at, "
                "current_stage = :current_stage, current_message = :current_message "
                "WHERE processing_run_id = :processing_run_id"
            ),
            {
                "processing_run_id": processing_run_id,
                "last_heartbeat_at": now,
                "current_stage": stage,
                "current_message": message,
            },
        )

    def complete_processing_run(self, processing_run_id: str) -> None:
        """Mark a processing run successful.

        Args:
            processing_run_id: Run id to complete.

        Returns:
            None. The run status and finish timestamp are persisted.
        """
        self._finish_processing_run(processing_run_id, "success")

    def fail_processing_run(self, processing_run_id: str, error_message: str) -> None:
        """Mark a processing run failed with a sanitized error message.

        Args:
            processing_run_id: Run id to fail.
            error_message: User-safe error summary.

        Returns:
            None. The run status, finish timestamp, and error are persisted.
        """
        self._finish_processing_run(processing_run_id, "failed", error_message)

    def _finish_processing_run(
        self,
        processing_run_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Persist terminal status fields for a processing run.

        Args:
            processing_run_id: Run id to update.
            status: Terminal run status.
            error_message: Optional user-safe error summary.

        Returns:
            None.
        """
        ensure_processing_progress_tables(self.connection)
        now = datetime.now(timezone.utc)
        if status != "failed":
            self.connection.execute(
                text(
                    "UPDATE price_change_processing_runs SET status = :status, finished_at = :finished_at, "
                    "last_heartbeat_at = :last_heartbeat_at, error_message = :error_message "
                    "WHERE processing_run_id = :processing_run_id"
                ),
                {
                    "processing_run_id": processing_run_id,
                    "status": status,
                    "finished_at": now,
                    "last_heartbeat_at": now,
                    "error_message": error_message,
                },
            )
            return
        self.connection.execute(
            text(
                "UPDATE price_change_processing_runs SET status = :status, finished_at = :finished_at, "
                "last_heartbeat_at = :last_heartbeat_at, current_stage = :current_stage, "
                "current_message = :current_message, error_message = :error_message "
                "WHERE processing_run_id = :processing_run_id"
            ),
            {
                "processing_run_id": processing_run_id,
                "status": status,
                "finished_at": now,
                "last_heartbeat_at": now,
                "current_stage": "analysis_failed",
                "current_message": "Analysis failed",
                "error_message": error_message,
            },
        )

    def get_processing_run_snapshot(self, processing_run_id: str) -> dict[str, Any]:
        """Return one processing run and its ordered event list.

        Args:
            processing_run_id: Run id requested by the UI.

        Returns:
            Dictionary with `run` and `events` keys.

        Raises:
            KeyError: If the run id does not exist.
        """
        ensure_processing_progress_tables(self.connection)
        run = self.connection.execute(
            text("SELECT * FROM price_change_processing_runs WHERE processing_run_id = :processing_run_id"),
            {"processing_run_id": processing_run_id},
        ).mappings().first()
        if run is None:
            raise KeyError(processing_run_id)
        events = self.connection.execute(
            text(
                "SELECT * FROM price_change_processing_events WHERE processing_run_id = :processing_run_id "
                "ORDER BY sequence_number"
            ),
            {"processing_run_id": processing_run_id},
        ).mappings().all()
        return {
            "run": dict(run),
            "events": [self._normalize_processing_event(dict(event)) for event in events],
        }

    def _normalize_processing_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Parse one progress event row for API responses.

        Args:
            event: Raw HANA event row.

        Returns:
            Event row with metadata parsed from JSON.
        """
        normalized = dict(event)
        normalized["metadata"] = parse_json_object(normalized.get("metadata_json"))
        return normalized

    def insert_gmail_email(self, email: GmailEmail) -> None:
        self.connection.execute(
            text(
                "INSERT INTO gmail_emails (gmail_message_id, thread_id, sender_name, sender_email, "
                "subject, email_date, body, fetched_at, processed_at, processing_status) VALUES "
                "(:gmail_message_id, :thread_id, :sender_name, :sender_email, :subject, :email_date, "
                ":body, :fetched_at, :processed_at, :processing_status)"
            ),
            email.model_dump(),
        )

    def insert_email_attachments(
        self,
        gmail_message_id: str,
        attachments: list[EmailAttachment],
    ) -> list[EmailAttachment]:
        """Persist supported email attachments in HANA.

        Args:
            gmail_message_id: Gmail or synthetic manual message id that owns the attachments.
            attachments: Validated attachment models with base64 content.

        Returns:
            Persisted attachments with generated ids and owner message id.
        """
        ensure_email_attachments_table(self.connection)
        statement = text(
            "INSERT INTO email_attachments (attachment_id, gmail_message_id, source, provider_attachment_id, "
            "filename, mime_type, file_extension, size_bytes, content_base64, text_preview, created_at) VALUES "
            "(:attachment_id, :gmail_message_id, :source, :provider_attachment_id, :filename, :mime_type, "
            ":file_extension, :size_bytes, :content_base64, :text_preview, :created_at)"
        )
        stored: list[EmailAttachment] = []
        for attachment in attachments:
            persisted = attachment.model_copy(
                update={
                    "attachment_id": attachment.attachment_id or f"att-{uuid.uuid4().hex}",
                    "gmail_message_id": gmail_message_id,
                }
            )
            params = persisted.model_dump()
            params["created_at"] = datetime.now(timezone.utc)
            self.connection.execute(statement, params)
            stored.append(persisted)
        return stored

    def list_email_attachment_summaries(self, gmail_message_id: str) -> list[dict[str, Any]]:
        """List attachment metadata for one email without raw content.

        Args:
            gmail_message_id: Gmail or synthetic manual message id.

        Returns:
            JSON-serializable attachment metadata rows.
        """
        ensure_email_attachments_table(self.connection)
        result = self.connection.execute(
            text(
                "SELECT attachment_id, filename, mime_type, file_extension, size_bytes "
                "FROM email_attachments WHERE gmail_message_id = :gmail_message_id "
                "ORDER BY created_at, filename"
            ),
            {"gmail_message_id": gmail_message_id},
        ).mappings().all()
        return [
            EmailAttachmentSummary.model_validate(dict(row)).model_dump()
            for row in result
        ]

    def get_email_attachment(self, attachment_id: str) -> dict[str, Any]:
        """Return one persisted attachment row including raw base64 content.

        Args:
            attachment_id: Attachment id.

        Returns:
            Attachment row.

        Raises:
            KeyError: If the attachment id does not exist.
        """
        ensure_email_attachments_table(self.connection)
        result = self.connection.execute(
            text("SELECT * FROM email_attachments WHERE attachment_id = :attachment_id"),
            {"attachment_id": attachment_id},
        ).mappings().first()
        if result is None:
            raise KeyError(attachment_id)
        return dict(result)

    def update_email_status(
        self,
        gmail_message_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        self.connection.execute(
            text(
                "UPDATE gmail_emails SET processing_status = :status, processed_at = :processed_at, "
                "error_message = :error_message WHERE gmail_message_id = :gmail_message_id"
            ),
            {
                "gmail_message_id": gmail_message_id,
                "status": status,
                "processed_at": datetime.now(timezone.utc),
                "error_message": error_message,
            },
        )

    def insert_extraction(
        self,
        gmail_message_id: str,
        model_name: str,
        raw_model_output: str,
        normalized_json: dict[str, Any],
        status: str,
        validation_errors: list[str],
    ) -> str:
        extraction_id = f"ext-{uuid.uuid4().hex}"
        self.connection.execute(
            text(
                "INSERT INTO email_extractions (extraction_id, gmail_message_id, model_name, raw_model_output, "
                "normalized_json, is_price_request, reason, confidence, status, validation_errors, created_at) "
                "VALUES (:extraction_id, :gmail_message_id, :model_name, :raw_model_output, :normalized_json, "
                ":is_price_request, :reason, :confidence, :status, :validation_errors, :created_at)"
            ),
            {
                "extraction_id": extraction_id,
                "gmail_message_id": gmail_message_id,
                "model_name": model_name,
                "raw_model_output": raw_model_output,
                "normalized_json": json.dumps(normalized_json, ensure_ascii=True),
                "is_price_request": bool(normalized_json.get("is_price_request", False)),
                "reason": normalized_json.get("reason"),
                "confidence": normalized_json.get("confidence"),
                "status": status,
                "validation_errors": json.dumps(validation_errors, ensure_ascii=True),
                "created_at": datetime.now(timezone.utc),
            },
        )
        return extraction_id

    def find_supplier_by_email(self, email: str) -> dict[str, Any]:
        result = self.connection.execute(
            text(
                "SELECT supplier_id, company, email FROM suppliers "
                "WHERE LOWER(email) = LOWER(:email)"
            ),
            {"email": email},
        ).mappings().all()
        if len(result) == 1:
            return {"status": "found", "supplier": dict(result[0])}
        if len(result) > 1:
            return {"status": "ambiguous", "candidates": [dict(row) for row in result]}
        return {"status": "not_found", "candidates": []}

    def find_supplier_by_id(self, supplier_id: str) -> dict[str, Any]:
        result = self.connection.execute(
            text(
                "SELECT supplier_id, company, email FROM suppliers "
                "WHERE LOWER(supplier_id) = LOWER(:supplier_id) OR LOWER(supplier) = LOWER(:supplier_id)"
            ),
            {"supplier_id": supplier_id},
        ).mappings().all()
        if len(result) == 1:
            return {"status": "found", "supplier": dict(result[0])}
        if len(result) > 1:
            return {"status": "ambiguous", "candidates": [dict(row) for row in result]}
        return {"status": "not_found", "candidates": []}

    def find_supplier_by_name(self, name_or_company: str) -> dict[str, Any]:
        result = self.connection.execute(
            text(
                "SELECT supplier_id, company, email FROM suppliers "
                "WHERE LOWER(company) LIKE LOWER(:query) OR LOWER(contact_name) LIKE LOWER(:query)"
            ),
            {"query": f"%{name_or_company}%"},
        ).mappings().all()
        return {
            "status": "found" if len(result) == 1 else "ambiguous" if result else "not_found",
            "candidates": [dict(row) for row in result],
        }

    def find_material_by_number(self, material_number: str) -> dict[str, Any]:
        result = self.connection.execute(
            text(
                "SELECT material_code, material_description, supplier_id, current_price, currency, uom "
                "FROM materials WHERE material_code = :material_code"
            ),
            {"material_code": material_number},
        ).mappings().all()
        return {
            "status": "found" if len(result) == 1 else "ambiguous" if result else "not_found",
            "candidates": [dict(row) for row in result],
        }

    def search_materials_by_description(
        self,
        query: str,
        supplier_id: str | None = None,
    ) -> dict[str, Any]:
        sql = (
            "SELECT material_code, material_description, supplier_id, current_price, currency, uom "
            "FROM materials WHERE LOWER(material_description) LIKE LOWER(:query)"
        )
        params: dict[str, Any] = {"query": f"%{query}%"}
        if supplier_id:
            sql += " AND supplier_id = :supplier_id"
            params["supplier_id"] = supplier_id
        result = self.connection.execute(text(sql), params).mappings().all()
        return {
            "status": "found" if len(result) == 1 else "ambiguous" if result else "not_found",
            "candidates": [dict(row) for row in result],
        }

    def get_current_supplier_material_price(self, supplier_id: str, material_number: str) -> dict[str, Any]:
        result = self.connection.execute(
            text(
                "SELECT supplier_id, material_code, current_price, currency, uom FROM supplier_material_prices "
                "WHERE supplier_id = :supplier_id AND material_code = :material_code"
            ),
            {"supplier_id": supplier_id, "material_code": material_number},
        ).mappings().first()
        return {"status": "found", "price": dict(result)} if result else {"status": "not_found", "price": None}

    def insert_price_change_draft(
        self,
        proposal: AgentProposal,
        extraction_id: str,
        item_index: int,
        raw_agent_output: dict[str, Any],
    ) -> str:
        draft_id = f"draft-{uuid.uuid4().hex}"
        self.connection.execute(
            text(
                "INSERT INTO price_change_drafts (draft_id, gmail_message_id, extraction_id, item_index, "
                "supplier_id, supplier_name, supplier_email, material_code, material_description, original_price, "
                "requested_new_price, currency, uom, price_change_mode, price_change_value, effective_from, "
                "effective_to, confidence, status, explanation, candidate_materials_json, candidate_suppliers_json, "
                "validation_errors_json, raw_agent_output_json, created_at, updated_at) VALUES "
                "(:draft_id, :gmail_message_id, :extraction_id, :item_index, :supplier_id, :supplier_name, "
                ":supplier_email, :material_code, :material_description, :original_price, :requested_new_price, "
                ":currency, :uom, :price_change_mode, :price_change_value, :effective_from, :effective_to, "
                ":confidence, :status, :explanation, :candidate_materials_json, :candidate_suppliers_json, "
                ":validation_errors_json, :raw_agent_output_json, :created_at, :updated_at)"
            ),
            {
                "draft_id": draft_id,
                "gmail_message_id": proposal.gmail_message_id,
                "extraction_id": extraction_id,
                "item_index": item_index,
                "supplier_id": proposal.supplier_id,
                "supplier_name": proposal.supplier_name,
                "supplier_email": proposal.supplier_email,
                "material_code": proposal.material_number,
                "material_description": proposal.material_description,
                "original_price": normalize_decimal_bind_value(proposal.original_price),
                "requested_new_price": normalize_decimal_bind_value(proposal.requested_new_price),
                "currency": proposal.currency,
                "uom": proposal.uom,
                "price_change_mode": proposal.price_change_mode,
                "price_change_value": proposal.price_change_value,
                "effective_from": proposal.effective_from,
                "effective_to": proposal.effective_to,
                "confidence": proposal.confidence,
                "status": proposal.status,
                "explanation": proposal.explanation,
                "candidate_materials_json": json.dumps(
                    [item.model_dump() for item in proposal.candidate_materials],
                    ensure_ascii=True,
                ),
                "candidate_suppliers_json": json.dumps(
                    [item.model_dump() for item in proposal.candidate_suppliers],
                    ensure_ascii=True,
                ),
                "validation_errors_json": json.dumps(proposal.validation_errors, ensure_ascii=True),
                "raw_agent_output_json": json.dumps(raw_agent_output, ensure_ascii=True),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        return draft_id

    def list_price_change_drafts(self) -> list[dict[str, Any]]:
        result = self.connection.execute(
            text(
                "SELECT d.*, e.subject, e.body, e.sender_name, e.sender_email, e.email_date "
                "FROM price_change_drafts d LEFT JOIN gmail_emails e ON d.gmail_message_id = e.gmail_message_id "
                "WHERE d.status IN ('ready_for_review', 'needs_human_review', 'failed') "
                "ORDER BY d.created_at DESC"
            )
        ).mappings().all()
        return self._with_attachment_metadata(
            [normalize_draft_review_state(dict(row)) for row in result]
        )

    def list_price_change_history(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List one newest-first page of terminal price-change draft history.

        Args:
            limit: Maximum rows to return.
            offset: Number of terminal history rows to skip.

        Returns:
            Approved/rejected draft rows enriched with source email and attachment metadata.
        """
        result = self.connection.execute(
            text(
                "SELECT d.*, e.subject, e.body, e.sender_name, e.sender_email, e.email_date "
                "FROM price_change_drafts d LEFT JOIN gmail_emails e ON d.gmail_message_id = e.gmail_message_id "
                "WHERE d.status IN ('approved', 'rejected') "
                "ORDER BY d.updated_at DESC "
                "LIMIT :limit OFFSET :offset"
            ),
            {"limit": limit, "offset": offset},
        ).mappings().all()
        return self._with_attachment_metadata(
            [normalize_draft_review_state(dict(row)) for row in result]
        )

    def price_change_history_summary(self) -> dict[str, int]:
        """Count terminal price-change history rows by review status.

        Returns:
            Dictionary containing total, approved, and rejected counts.
        """
        result = self.connection.execute(
            text(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END), 0) AS approved, "
                "COALESCE(SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END), 0) AS rejected "
                "FROM price_change_drafts "
                "WHERE status IN ('approved', 'rejected')"
            )
        ).mappings().first()
        row = dict(result or {})
        return {
            "total": int(row.get("total") or 0),
            "approved": int(row.get("approved") or 0),
            "rejected": int(row.get("rejected") or 0),
        }

    def reset_price_change_history(self) -> dict[str, int]:
        """Delete terminal history rows and private audit records.

        Emails, attachments, and extractions are deleted only when no active draft
        still references the same Gmail message or extraction id.

        Returns:
            Dictionary containing the number of approved/rejected draft rows deleted.
        """
        ensure_email_attachments_table(self.connection)
        count_result = self.connection.execute(
            text(
                "SELECT COUNT(*) AS deleted_history "
                "FROM price_change_drafts "
                "WHERE status IN ('approved', 'rejected')"
            )
        ).mappings().first()
        deleted_history = int(dict(count_result or {}).get("deleted_history") or 0)
        terminal_statuses = "'approved', 'rejected'"

        self.connection.execute(
            text(
                "DELETE FROM email_attachments "
                "WHERE gmail_message_id IN ("
                "SELECT historical_emails.gmail_message_id FROM ("
                "SELECT DISTINCT gmail_message_id FROM price_change_drafts "
                f"WHERE status IN ({terminal_statuses}) AND gmail_message_id IS NOT NULL"
                ") historical_emails "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM price_change_drafts active_drafts "
                "WHERE active_drafts.gmail_message_id = historical_emails.gmail_message_id "
                f"AND active_drafts.status NOT IN ({terminal_statuses})"
                ")"
                ")"
            )
        )
        self.connection.execute(
            text(
                "DELETE FROM email_extractions "
                "WHERE extraction_id IN ("
                "SELECT historical_extractions.extraction_id FROM ("
                "SELECT DISTINCT extraction_id FROM price_change_drafts "
                f"WHERE status IN ({terminal_statuses}) AND extraction_id IS NOT NULL"
                ") historical_extractions "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM price_change_drafts active_drafts "
                "WHERE active_drafts.extraction_id = historical_extractions.extraction_id "
                f"AND active_drafts.status NOT IN ({terminal_statuses})"
                ")"
                ")"
            )
        )
        self.connection.execute(
            text(
                "DELETE FROM gmail_emails "
                "WHERE gmail_message_id IN ("
                "SELECT historical_emails.gmail_message_id FROM ("
                "SELECT DISTINCT gmail_message_id FROM price_change_drafts "
                f"WHERE status IN ({terminal_statuses}) AND gmail_message_id IS NOT NULL"
                ") historical_emails "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM price_change_drafts active_drafts "
                "WHERE active_drafts.gmail_message_id = historical_emails.gmail_message_id "
                f"AND active_drafts.status NOT IN ({terminal_statuses})"
                ")"
                ")"
            )
        )
        self.connection.execute(
            text("DELETE FROM price_change_drafts WHERE status IN ('approved', 'rejected')")
        )
        return {"deleted_history": deleted_history}

    def _with_attachment_metadata(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach per-email attachment metadata JSON to draft/history rows.

        Args:
            rows: Draft or history rows after status normalization.

        Returns:
            Rows with an attachments_json field.
        """
        cache: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            gmail_message_id = row.get("gmail_message_id")
            if not gmail_message_id:
                row["attachments_json"] = "[]"
                continue
            message_id = str(gmail_message_id)
            if message_id not in cache:
                cache[message_id] = self.list_email_attachment_summaries(message_id)
            row["attachments_json"] = json.dumps(cache[message_id], ensure_ascii=True)
        return rows

    def patch_price_change_draft(self, draft_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "supplier_id",
            "supplier_name",
            "supplier_email",
            "material_code",
            "material_description",
            "original_price",
            "requested_new_price",
            "currency",
            "uom",
            "effective_from",
            "effective_to",
            "explanation",
        }
        patch_values = {key: value for key, value in values.items() if key in allowed}
        if not patch_values:
            return self.get_price_change_draft(draft_id)
        current = self.get_price_change_draft(draft_id)
        merged = {**current, **patch_values}
        normalized = normalize_draft_review_state(merged)
        patch_values["status"] = normalized["status"]
        patch_values["validation_errors_json"] = normalized["validation_errors_json"]
        assignments = ", ".join(f"{key} = :{key}" for key in patch_values)
        params = {**patch_values, "draft_id": draft_id, "updated_at": datetime.now(timezone.utc)}
        self.connection.execute(
            text(f"UPDATE price_change_drafts SET {assignments}, updated_at = :updated_at WHERE draft_id = :draft_id"),
            params,
        )
        return self.get_price_change_draft(draft_id)

    def record_s4_write_attempt(self, draft_id: str, s4_write_result: dict[str, Any]) -> dict[str, Any]:
        """Persist one S/4 write attempt on a draft without changing review status.

        Args:
            draft_id: Draft id.
            s4_write_result: Sanitized S/4 write result.

        Returns:
            Updated draft row.
        """
        draft = self.get_price_change_draft(draft_id)
        payload = s4_write_audit_payload(draft.get("raw_agent_output_json"))
        attempts = payload["s4_write_attempts"]
        attempts.append(
            {
                "attempted_at": datetime.now(timezone.utc).isoformat(),
                "result": s4_write_result,
            }
        )
        payload["s4_write_result"] = s4_write_result
        self.connection.execute(
            text(
                "UPDATE price_change_drafts SET raw_agent_output_json = :raw_agent_output_json, "
                "updated_at = :updated_at WHERE draft_id = :draft_id"
            ),
            {
                "draft_id": draft_id,
                "raw_agent_output_json": json.dumps(payload, ensure_ascii=True),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        return self.get_price_change_draft(draft_id)

    def approve_price_change_draft(
        self,
        draft_id: str,
        s4_write_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Mark a draft approved after a verified user-triggered S/4 write.

        Args:
            draft_id: Draft id.
            s4_write_result: Successful S/4 write result to persist for audit.

        Returns:
            Updated approved draft row.
        """
        draft = self.get_price_change_draft(draft_id)
        validation_errors = draft_validation_errors(draft)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        if s4_write_result is not None:
            self.record_s4_write_attempt(draft_id, s4_write_result)
        self._set_draft_status(draft_id, "approved", [])
        return self.get_price_change_draft(draft_id)

    def reject_price_change_draft(self, draft_id: str) -> dict[str, Any]:
        draft = self.get_price_change_draft(draft_id)
        self._set_draft_status(draft_id, "rejected", draft_validation_errors(draft))
        return self.get_price_change_draft(draft_id)

    def _set_draft_status(self, draft_id: str, status: str, validation_errors: list[str]) -> None:
        self.connection.execute(
            text(
                "UPDATE price_change_drafts SET status = :status, validation_errors_json = :validation_errors_json, "
                "updated_at = :updated_at WHERE draft_id = :draft_id"
            ),
            {
                "draft_id": draft_id,
                "status": status,
                "validation_errors_json": json.dumps(validation_errors, ensure_ascii=True),
                "updated_at": datetime.now(timezone.utc),
            },
        )

    def get_price_change_draft(self, draft_id: str) -> dict[str, Any]:
        result = self.connection.execute(
            text("SELECT * FROM price_change_drafts WHERE draft_id = :draft_id"),
            {"draft_id": draft_id},
        ).mappings().first()
        if result is None:
            raise KeyError(draft_id)
        return normalize_draft_review_state(dict(result))
