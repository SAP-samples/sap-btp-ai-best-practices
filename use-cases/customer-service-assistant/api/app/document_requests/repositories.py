from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from .models import DocumentAttachment, DraftEmailProposal, GmailEmail, OrderStatus
from .scenario_emails import normalize_line_identifier, product_matches

# Draft statuses still waiting for a reviewer decision.
ACTIVE_DRAFT_STATUSES = ["ready_for_review", "needs_human_review", "failed"]


def json_dumps(value: Any) -> str:
    """Serialize JSON with ASCII-safe output for HANA NCLOB columns.

    Args:
        value: JSON-compatible value.

    Returns:
        JSON string that can be stored in HANA text columns.
    """
    return json.dumps(value, ensure_ascii=True)


def attachment_from_row(row: dict[str, Any]) -> DocumentAttachment:
    """Build an attachment model from a HANA shipment document row.

    Args:
        row: Shipment document row containing metadata and binary content.

    Returns:
        DocumentAttachment with normalized field names.
    """
    return DocumentAttachment(
        document_id=str(row["document_id"]),
        file_name=str(row["file_name"]),
        document_type=row["document_type"],
        mime_type=str(row.get("mime_type") or "application/octet-stream"),
        size_bytes=int(row.get("file_size_bytes") or 0),
        content=bytes(row.get("file_content") or b""),
    )


class DocumentRequestRepository:
    """Persist and query shipment document-request data in HANA."""

    def __init__(self, connection: Connection) -> None:
        """Initialize the repository.

        Args:
            connection: Transactional SQLAlchemy connection.
        """
        self.connection = connection

    def get_last_successful_fetch_at(self, mailbox_id: str) -> datetime | None:
        """Return the Gmail watermark: when the last successful run started.

        Args:
            mailbox_id: Run key (Gmail mailbox id).

        Returns:
            Watermark timestamp, or None when no run has succeeded yet.
        """
        result = self.connection.execute(
            text("SELECT last_successful_fetch_at FROM document_request_fetch_state WHERE mailbox_id = :mailbox_id"),
            {"mailbox_id": mailbox_id},
        ).mappings().first()
        return None if result is None else result["last_successful_fetch_at"]

    def try_start_fetch_run(self, run_key: str, started_at: datetime, stale_before: datetime) -> bool:
        """Atomically mark a run as running unless another live run holds it.

        Works across threads and Cloud Foundry instances because the decision is
        a single conditional UPDATE on one HANA row. A run still flagged as
        running but started before ``stale_before`` is treated as crashed and
        can be taken over.

        Args:
            run_key: Run identifier (mailbox id, or "scenarios").
            started_at: Start timestamp of the new run.
            stale_before: Runs started before this instant are considered dead.

        Returns:
            True when this caller now owns the run, False when one is active.
        """
        try:
            self.connection.execute(
                text(
                    "INSERT INTO document_request_fetch_state (mailbox_id, last_status) "
                    "SELECT :mailbox_id, 'idle' FROM DUMMY WHERE NOT EXISTS "
                    "(SELECT 1 FROM document_request_fetch_state WHERE mailbox_id = :mailbox_id)"
                ),
                {"mailbox_id": run_key},
            )
        except IntegrityError:
            # A concurrent caller created the row first; the UPDATE below decides.
            pass
        result = self.connection.execute(
            text(
                "UPDATE document_request_fetch_state SET last_status = 'running', last_started_at = :started_at, "
                "last_finished_at = NULL, summary_json = NULL, error_message = NULL "
                "WHERE mailbox_id = :mailbox_id AND (last_status IS NULL OR last_status <> 'running' "
                "OR last_started_at IS NULL OR last_started_at < :stale_before)"
            ),
            {"mailbox_id": run_key, "started_at": started_at, "stale_before": stale_before},
        )
        return result.rowcount == 1

    def update_fetch_progress(self, run_key: str, summary_json: dict[str, Any]) -> None:
        """Store intermediate run counters so the UI can show progress.

        Args:
            run_key: Run identifier.
            summary_json: JSON-compatible FetchSummary.
        """
        self.connection.execute(
            text("UPDATE document_request_fetch_state SET summary_json = :summary_json WHERE mailbox_id = :mailbox_id"),
            {"mailbox_id": run_key, "summary_json": json_dumps(summary_json)},
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
        """Record the end of a run.

        On success the watermark becomes ``started_at`` (not ``finished_at``) so
        emails that arrived while the run was in progress are picked up next
        time. On failure the previous watermark is kept.

        Args:
            mailbox_id: Run identifier.
            started_at: Run start timestamp.
            finished_at: Run finish timestamp.
            status: Final status, "success" or "failed".
            summary_json: JSON-compatible FetchSummary.
            error_message: Optional failure message.
        """
        self.connection.execute(
            text(
                "UPDATE document_request_fetch_state SET "
                "last_successful_fetch_at = COALESCE(:watermark, last_successful_fetch_at), "
                "last_finished_at = :last_finished_at, last_status = :last_status, "
                "summary_json = :summary_json, error_message = :error_message WHERE mailbox_id = :mailbox_id"
            ),
            {
                "mailbox_id": mailbox_id,
                "watermark": started_at if status == "success" else None,
                "last_finished_at": finished_at,
                "last_status": status,
                "summary_json": json_dumps(summary_json),
                "error_message": error_message,
            },
        )

    def get_fetch_state(self, run_key: str) -> dict[str, Any] | None:
        """Return the stored state row of a run.

        Args:
            run_key: Run identifier.

        Returns:
            Row with last_status, timestamps, summary_json and error_message, or None.
        """
        result = self.connection.execute(
            text(
                "SELECT last_status, last_started_at, last_finished_at, summary_json, error_message "
                "FROM document_request_fetch_state WHERE mailbox_id = :mailbox_id"
            ),
            {"mailbox_id": run_key},
        ).mappings().first()
        return dict(result) if result else None

    def email_exists(self, gmail_message_id: str) -> bool:
        """Return True when a Gmail message was already stored.

        Args:
            gmail_message_id: Gmail API message id.

        Returns:
            True when a row already exists for the message.
        """
        result = self.connection.execute(
            text("SELECT gmail_message_id FROM document_request_emails WHERE gmail_message_id = :gmail_message_id"),
            {"gmail_message_id": gmail_message_id},
        ).mappings().first()
        return result is not None

    def insert_gmail_email(self, email: GmailEmail) -> None:
        """Persist a normalized Gmail email before extraction.

        Args:
            email: Normalized Gmail email model.
        """
        self.connection.execute(
            text(
                "INSERT INTO document_request_emails (gmail_message_id, thread_id, rfc_message_id, sender_name, "
                "sender_email, subject, email_date, body, fetched_at, processed_at, processing_status) VALUES "
                "(:gmail_message_id, :thread_id, :rfc_message_id, :sender_name, :sender_email, :subject, "
                ":email_date, :body, :fetched_at, :processed_at, :processing_status)"
            ),
            email.model_dump(),
        )

    def update_email_status(self, gmail_message_id: str, status: str, error_message: str | None = None) -> None:
        """Update processing status and optional error text for a stored email.

        Args:
            gmail_message_id: Gmail API message id.
            status: New processing status.
            error_message: Optional error details.
        """
        self.connection.execute(
            text(
                "UPDATE document_request_emails SET processing_status = :status, processed_at = :processed_at, "
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
        """Persist one extraction attempt and return its id.

        Args:
            gmail_message_id: Source Gmail message id.
            model_name: Extractor model name.
            raw_model_output: Raw text returned by the model.
            normalized_json: Normalized extraction JSON.
            status: Extraction persistence status.
            validation_errors: Normalization or validation errors.

        Returns:
            Generated extraction id.
        """
        extraction_id = f"ext-{uuid.uuid4().hex}"
        self.connection.execute(
            text(
                "INSERT INTO document_request_extractions (extraction_id, gmail_message_id, model_name, "
                "raw_model_output, normalized_json, is_document_request, client_country, sales_order_id, "
                "reason, confidence, status, validation_errors, created_at) VALUES (:extraction_id, "
                ":gmail_message_id, :model_name, :raw_model_output, :normalized_json, :is_document_request, "
                ":client_country, :sales_order_id, :reason, :confidence, :status, :validation_errors, :created_at)"
            ),
            {
                "extraction_id": extraction_id,
                "gmail_message_id": gmail_message_id,
                "model_name": model_name,
                "raw_model_output": raw_model_output,
                "normalized_json": json_dumps(normalized_json),
                "is_document_request": bool(normalized_json.get("is_document_request", False)),
                "client_country": normalized_json.get("client_country"),
                "sales_order_id": normalized_json.get("sales_order_id"),
                "reason": normalized_json.get("reason"),
                "confidence": normalized_json.get("confidence"),
                "status": status,
                "validation_errors": json_dumps(validation_errors),
                "created_at": datetime.now(timezone.utc),
            },
        )
        return extraction_id

    def list_documents_for_country(self, client_country: str) -> list[dict[str, Any]]:
        """Return all active documents for a country, including binary content.

        Args:
            client_country: Country bucket such as RU, GB, or ES.

        Returns:
            Shipment document rows ordered for deterministic matching.
        """
        result = self.connection.execute(
            text(
                "SELECT document_id, client_country, sales_order_id, outbound_delivery_id, document_type, "
                "file_name, mime_type, file_size_bytes, raw_text, file_content FROM shipment_documents "
                "WHERE client_country = :client_country AND is_active = TRUE "
                "ORDER BY document_type, file_name"
            ),
            {"client_country": client_country},
        ).mappings().all()
        return [dict(row) for row in result]

    def list_documents_for_sales_order(
        self, sales_order_id: str, document_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return all active documents for a sales order, optionally filtered by type.

        Args:
            sales_order_id: Sales order identifier (e.g. "6090004185").
            document_types: Optional list of document types to filter by.

        Returns:
            Shipment document rows ordered for deterministic matching.
        """
        if document_types:
            result = self.connection.execute(
                text(
                    "SELECT document_id, client_country, sales_order_id, outbound_delivery_id, document_type, "
                    "file_name, mime_type, file_size_bytes, raw_text, file_content FROM shipment_documents "
                    "WHERE sales_order_id = :sales_order_id AND document_type IN :document_types "
                    "AND is_active = TRUE ORDER BY document_type, file_name"
                ).bindparams(bindparam("document_types", expanding=True)),
                {"sales_order_id": sales_order_id, "document_types": document_types},
            ).mappings().all()
        else:
            result = self.connection.execute(
                text(
                    "SELECT document_id, client_country, sales_order_id, outbound_delivery_id, document_type, "
                    "file_name, mime_type, file_size_bytes, raw_text, file_content FROM shipment_documents "
                    "WHERE sales_order_id = :sales_order_id AND is_active = TRUE "
                    "ORDER BY document_type, file_name"
                ),
                {"sales_order_id": sales_order_id},
            ).mappings().all()
        return [dict(row) for row in result]

    def get_order_status(self, sales_order_id: str) -> OrderStatus | None:
        """Return HANA-derived order status for a sales order.

        Args:
            sales_order_id: Sales order number.

        Returns:
            OrderStatus when present, otherwise None.
        """
        result = self.connection.execute(
            text(
                "SELECT sales_order_id, client_country, picking_status, packing_status, pgi_status, shipped, "
                "shipped_date, back_order, expected_completion FROM order_status "
                "WHERE sales_order_id = :sales_order_id"
            ),
            {"sales_order_id": sales_order_id},
        ).mappings().first()
        if result is None:
            return None
        return OrderStatus.model_validate(dict(result))

    def get_order_context(
        self,
        sales_order_id: str,
        line_item_number: str | None = None,
        product_id: str | None = None,
        batch_number: str | None = None,
    ) -> dict[str, Any]:
        """Return HANA order context rows filtered by line, product, or batch.

        Args:
            sales_order_id: Sales order number.
            line_item_number: Optional line item to match after SAP-line normalization.
            product_id: Optional product/material id.
            batch_number: Optional batch/lot id.

        Returns:
            Dict with status_lines and item_details lists.
        """
        status_lines = self._list_status_lines_for_sales_order(sales_order_id)
        item_details = self._list_item_details_for_sales_order(sales_order_id)
        normalized_line = normalize_line_identifier(line_item_number)
        if normalized_line:
            status_lines = [row for row in status_lines if row.get("line_number") == normalized_line]
            item_details = [row for row in item_details if row.get("item_number") == normalized_line]
        if product_id:
            status_lines = [row for row in status_lines if product_matches(row.get("product_id"), product_id)]
            item_details = [row for row in item_details if product_matches(row.get("product_id"), product_id)]
        if batch_number:
            status_lines = [row for row in status_lines if str(row.get("batch_number") or "") == str(batch_number)]
            item_details = [row for row in item_details if str(row.get("lot_number") or "") == str(batch_number)]
        return {"status_lines": status_lines, "item_details": item_details}

    def _list_status_lines_for_sales_order(self, sales_order_id: str) -> list[dict[str, Any]]:
        """Return all HANA status lines for a sales order.

        Args:
            sales_order_id: Sales order number.

        Returns:
            Status line rows ordered by outbound delivery and line.
        """
        result = self.connection.execute(
            text(
                "SELECT status_line_id, scenario_id, scenario_email_id, order_date, order_time, client_country, "
                "route_id, sales_order_id, outbound_delivery_id, document_type_code, line_number, product_id, "
                "batch_number, quantity_ordered, quantity_packed, dangerous_goods, pick_date, pack_box, "
                "status_pick, status_gi FROM sales_order_status_lines "
                "WHERE sales_order_id = :sales_order_id ORDER BY outbound_delivery_id, line_number, product_id"
            ),
            {"sales_order_id": sales_order_id},
        ).mappings().all()
        return [dict(row) for row in result]

    def _list_item_details_for_sales_order(self, sales_order_id: str) -> list[dict[str, Any]]:
        """Return all HANA item detail rows for a sales order.

        Args:
            sales_order_id: Sales order number.

        Returns:
            Item detail rows ordered by item and product.
        """
        result = self.connection.execute(
            text(
                "SELECT item_detail_id, scenario_id, scenario_email_id, sales_order_id, customer_ref, order_type, "
                "item_number, sold_to_name, quantity, uom, product_id, lot_number, description, line_weight_kg, "
                "invoice_id, invoice_date, foreign_amount, commodity_code, commodity_description, "
                "country_of_origin, expiry_date, manufacturing_date FROM sales_order_item_details "
                "WHERE sales_order_id = :sales_order_id ORDER BY item_number, product_id, lot_number"
            ),
            {"sales_order_id": sales_order_id},
        ).mappings().all()
        return [dict(row) for row in result]

    def get_document_content(self, document_id: str) -> dict[str, Any] | None:
        """Fetch a single document's binary content and metadata for preview.

        Args:
            document_id: Shipment document primary key.

        Returns:
            Dict with file_content, mime_type, file_name or None if not found.
        """
        result = self.connection.execute(
            text(
                "SELECT file_content, mime_type, file_name FROM shipment_documents "
                "WHERE document_id = :document_id AND is_active = TRUE"
            ),
            {"document_id": document_id},
        ).mappings().first()
        return dict(result) if result else None

    def insert_document_email_draft(
        self,
        proposal: DraftEmailProposal,
        gmail_draft_id: str | None,
        gmail_draft_message_id: str | None,
        raw_agent_output: dict[str, Any],
    ) -> str:
        """Persist a draft email review row and return its draft id.

        Args:
            proposal: Draft email proposal built by the agent.
            gmail_draft_id: Gmail draft id when one was created.
            gmail_draft_message_id: Gmail draft message id when available.
            raw_agent_output: JSON-compatible agent output for audit.

        Returns:
            Generated review draft id.
        """
        draft_id = f"draft-{uuid.uuid4().hex}"
        selected_documents = [item.model_dump_for_ui() for item in proposal.selected_documents]
        self.connection.execute(
            text(
                "INSERT INTO document_email_drafts (draft_id, gmail_message_id, extraction_id, gmail_draft_id, "
                "gmail_draft_message_id, client_country, to_email, subject, body, status, selected_documents_json, "
                "confidence, explanation, validation_errors_json, raw_agent_output_json, created_at, updated_at) "
                "VALUES (:draft_id, :gmail_message_id, :extraction_id, :gmail_draft_id, :gmail_draft_message_id, "
                ":client_country, :to_email, :subject, :body, :status, :selected_documents_json, :confidence, "
                ":explanation, :validation_errors_json, :raw_agent_output_json, :created_at, :updated_at)"
            ),
            {
                "draft_id": draft_id,
                "gmail_message_id": proposal.gmail_message_id,
                "extraction_id": proposal.extraction_id,
                "gmail_draft_id": gmail_draft_id,
                "gmail_draft_message_id": gmail_draft_message_id,
                "client_country": proposal.client_country,
                "to_email": proposal.to_email,
                "subject": proposal.subject,
                "body": proposal.body,
                "status": proposal.status,
                "selected_documents_json": json_dumps(selected_documents),
                "confidence": proposal.confidence,
                "explanation": proposal.explanation,
                "validation_errors_json": json_dumps(proposal.validation_errors),
                "raw_agent_output_json": json_dumps(raw_agent_output),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        return draft_id

    def list_active_drafts(self, page: int = 1, page_size: int = 50) -> list[dict[str, Any]]:
        """Return active draft rows joined with their original email.

        Args:
            page: One-based page number.
            page_size: Number of rows to return.

        Returns:
            Draft rows in ready, review-needed, or failed statuses.
        """
        return self._list_drafts_by_statuses(ACTIVE_DRAFT_STATUSES, page, page_size)

    def list_history(self, page: int = 1, page_size: int = 50) -> list[dict[str, Any]]:
        """Return approved and rejected draft rows joined with their original email.

        Args:
            page: One-based page number.
            page_size: Number of rows to return.

        Returns:
            Draft rows that have been approved or rejected.
        """
        return self._list_drafts_by_statuses(["approved", "rejected"], page, page_size)

    def count_active_drafts(self) -> int:
        """Return the number of active review draft rows.

        Returns:
            Count of ready, review-needed, and failed draft rows.
        """
        return self._count_drafts_by_statuses(ACTIVE_DRAFT_STATUSES)

    def count_history(self) -> int:
        """Return the number of historical draft rows.

        Returns:
            Count of approved and rejected draft rows.
        """
        return self._count_drafts_by_statuses(["approved", "rejected"])

    def summarize_active_drafts(self) -> dict[str, int]:
        """Return status counts for all active review draft rows.

        Returns:
            Summary with total, ready_for_review, needs_human_review, and failed counts.
        """
        return self._summarize_drafts_by_statuses(ACTIVE_DRAFT_STATUSES)

    def summarize_history(self) -> dict[str, int]:
        """Return status counts for all historical draft rows.

        Returns:
            Summary with total, approved, and rejected counts.
        """
        return self._summarize_drafts_by_statuses(["approved", "rejected"])

    def get_draft(self, draft_id: str) -> dict[str, Any]:
        """Return one draft row or raise KeyError.

        Args:
            draft_id: Review draft id.

        Returns:
            Draft row dictionary.
        """
        result = self.connection.execute(
            text(
                "SELECT d.*, e.thread_id AS source_thread_id, e.rfc_message_id AS source_rfc_message_id, "
                "e.sender_name, e.sender_email, e.subject AS original_subject, e.body AS original_body, "
                "e.email_date FROM document_email_drafts d "
                "LEFT JOIN document_request_emails e ON d.gmail_message_id = e.gmail_message_id "
                "WHERE d.draft_id = :draft_id"
            ),
            {"draft_id": draft_id},
        ).mappings().first()
        if result is None:
            raise KeyError(draft_id)
        return dict(result)

    def update_draft_body(self, draft_id: str, body: str) -> dict[str, Any]:
        """Persist the final reviewer-edited draft body.

        Args:
            draft_id: Review draft id.
            body: Plain-text body currently present in the editable review field.

        Returns:
            Updated draft row.
        """
        self.connection.execute(
            text(
                "UPDATE document_email_drafts SET body = :body, updated_at = :updated_at "
                "WHERE draft_id = :draft_id"
            ),
            {"draft_id": draft_id, "body": body, "updated_at": datetime.now(timezone.utc)},
        )
        return self.get_draft(draft_id)

    def get_selected_draft_attachments(self, draft: dict[str, Any]) -> list[DocumentAttachment]:
        """Load full attachment content for the documents selected on a draft.

        Args:
            draft: Draft row containing selected_documents_json metadata.

        Returns:
            DocumentAttachment models including binary content for Gmail MIME rebuilds.
        """
        try:
            selected = json.loads(str(draft.get("selected_documents_json") or "[]"))
        except json.JSONDecodeError:
            selected = []
        attachments: list[DocumentAttachment] = []
        for item in selected:
            if not isinstance(item, dict) or not item.get("document_id"):
                continue
            document = self.get_document_content(str(item["document_id"]))
            if document:
                attachments.append(attachment_from_row(document))
        return attachments

    def lock_active_draft(self, draft_id: str) -> bool:
        """Lock a draft row for an approve/reject decision if it is still open.

        The conditional UPDATE takes a HANA row lock held until this
        transaction ends, so a concurrent second decision on the same draft
        waits. The status is re-read afterwards (READ COMMITTED gives each
        statement a fresh snapshot), so the waiting request sees the first
        decision and is refused instead of sending the email twice.

        Args:
            draft_id: Review draft id.

        Returns:
            True when the draft is still open and now locked by this transaction.
        """
        result = self.connection.execute(
            text(
                "UPDATE document_email_drafts SET updated_at = :updated_at "
                "WHERE draft_id = :draft_id AND status IN :statuses"
            ).bindparams(bindparam("statuses", expanding=True)),
            {"draft_id": draft_id, "updated_at": datetime.now(timezone.utc), "statuses": ACTIVE_DRAFT_STATUSES},
        )
        if result.rowcount != 1:
            return False
        status = self.connection.execute(
            text("SELECT status FROM document_email_drafts WHERE draft_id = :draft_id"), {"draft_id": draft_id}
        ).scalar()
        return status in ACTIVE_DRAFT_STATUSES

    def mark_draft_sent(self, draft_id: str, sent_message_id: str | None) -> dict[str, Any]:
        """Mark a draft approved after Gmail sends it.

        Args:
            draft_id: Review draft id.
            sent_message_id: Gmail sent message id when available.

        Returns:
            Updated draft row.
        """
        self.connection.execute(
            text(
                "UPDATE document_email_drafts SET status = 'approved', sent_message_id = :sent_message_id, "
                "updated_at = :updated_at WHERE draft_id = :draft_id"
            ),
            {"draft_id": draft_id, "sent_message_id": sent_message_id, "updated_at": datetime.now(timezone.utc)},
        )
        return self.get_draft(draft_id)

    def mark_draft_rejected(self, draft_id: str) -> dict[str, Any]:
        """Mark a draft rejected.

        Args:
            draft_id: Review draft id.

        Returns:
            Updated draft row.
        """
        self.connection.execute(
            text(
                "UPDATE document_email_drafts SET status = 'rejected', updated_at = :updated_at "
                "WHERE draft_id = :draft_id"
            ),
            {"draft_id": draft_id, "updated_at": datetime.now(timezone.utc)},
        )
        return self.get_draft(draft_id)

    def _list_drafts_by_statuses(self, statuses: list[str], page: int, page_size: int) -> list[dict[str, Any]]:
        """Return draft rows matching a set of statuses.

        Args:
            statuses: Draft statuses to include.
            page: One-based page number.
            page_size: Number of rows to return.

        Returns:
            Joined draft and source-email rows.
        """
        offset = (page - 1) * page_size
        result = self.connection.execute(
            text(
                "SELECT d.*, e.sender_name, e.sender_email, e.subject AS original_subject, e.body AS original_body, "
                "e.email_date FROM document_email_drafts d "
                "LEFT JOIN document_request_emails e ON d.gmail_message_id = e.gmail_message_id "
                "WHERE d.status IN :statuses "
                "ORDER BY COALESCE(d.updated_at, d.created_at) DESC, d.draft_id DESC "
                "LIMIT :limit OFFSET :offset"
            ).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": statuses, "limit": page_size, "offset": offset},
        ).mappings().all()
        return [dict(row) for row in result]

    def _count_drafts_by_statuses(self, statuses: list[str]) -> int:
        """Return a count of draft rows matching a set of statuses.

        Args:
            statuses: Draft statuses to include.

        Returns:
            Number of matching draft rows.
        """
        result = self.connection.execute(
            text("SELECT COUNT(*) AS row_count FROM document_email_drafts WHERE status IN :statuses").bindparams(
                bindparam("statuses", expanding=True)
            ),
            {"statuses": statuses},
        ).mappings().first()
        return int(result["row_count"] if result else 0)

    def _summarize_drafts_by_statuses(self, statuses: list[str]) -> dict[str, int]:
        """Return total and per-status counts for draft rows.

        Args:
            statuses: Draft statuses to include.

        Returns:
            Dict containing total and one count per requested status.
        """
        result = self.connection.execute(
            text(
                "SELECT status, COUNT(*) AS row_count FROM document_email_drafts "
                "WHERE status IN :statuses GROUP BY status"
            ).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": statuses},
        ).mappings().all()
        summary = {status: 0 for status in statuses}
        for row in result:
            summary[str(row["status"])] = int(row["row_count"])
        summary["total"] = sum(summary.values())
        return summary

    def clear_history(self) -> dict[str, int]:
        """Delete historical drafts and linked email/extraction traces.

        Historical rows are approved or rejected document request drafts. Shipment
        document tables are intentionally left untouched.

        Returns:
            Counts of deleted drafts, extractions, and source emails.
        """
        statuses = ["approved", "rejected"]
        rows = self.connection.execute(
            text(
                "SELECT DISTINCT gmail_message_id FROM document_email_drafts "
                "WHERE status IN :statuses AND gmail_message_id IS NOT NULL"
            ).bindparams(bindparam("statuses", expanding=True)),
            {"statuses": statuses},
        ).mappings().all()
        gmail_message_ids = [str(row["gmail_message_id"]) for row in rows]
        draft_count = self._count_drafts_by_statuses(statuses)
        extraction_count = (
            self._count_rows_by_gmail_ids("document_request_extractions", gmail_message_ids) if gmail_message_ids else 0
        )
        email_count = self._count_rows_by_gmail_ids("document_request_emails", gmail_message_ids) if gmail_message_ids else 0

        self.connection.execute(
            text("DELETE FROM document_email_drafts WHERE status IN :statuses").bindparams(
                bindparam("statuses", expanding=True)
            ),
            {"statuses": statuses},
        )
        if not gmail_message_ids:
            return {"drafts": draft_count, "extractions": 0, "emails": 0}
        self.connection.execute(
            text("DELETE FROM document_request_extractions WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        )
        self.connection.execute(
            text("DELETE FROM document_request_emails WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        )
        return {"drafts": draft_count, "extractions": extraction_count, "emails": email_count}

    def _count_rows_by_gmail_ids(self, table_name: str, gmail_message_ids: list[str]) -> int:
        """Return the number of rows in a table for a list of Gmail ids.

        Args:
            table_name: Trusted repository-owned table name.
            gmail_message_ids: Gmail message ids to count.

        Returns:
            Count of matching rows.
        """
        result = self.connection.execute(
            text(f"SELECT COUNT(*) AS row_count FROM {table_name} WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        ).mappings().first()
        return int(result["row_count"] if result else 0)

    def delete_emails_by_ids(self, gmail_message_ids: list[str]) -> None:
        """Delete mock email rows and their related extractions and drafts.

        Args:
            gmail_message_ids: Gmail message IDs to remove.
        """
        self.connection.execute(
            text("DELETE FROM document_email_drafts WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        )
        self.connection.execute(
            text("DELETE FROM document_request_extractions WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        )
        self.connection.execute(
            text("DELETE FROM document_request_emails WHERE gmail_message_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": gmail_message_ids},
        )
