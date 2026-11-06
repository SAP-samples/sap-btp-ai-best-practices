from __future__ import annotations

import re
from typing import Any

from .models import DocumentAttachment, OrderStatus
from .repositories import attachment_from_row
from .scenario_emails import normalize_line_identifier


DOCUMENT_TYPE_ALIASES: dict[str, list[str]] = {
    "invoice": ["invoice", "billing_document"],
    "billing_document": ["billing_document", "invoice"],
}


def expand_document_types(types: list[str]) -> list[str]:
    """Expand document type list with known aliases (invoice ↔ billing_document)."""
    expanded: list[str] = []
    for t in types:
        for alias in DOCUMENT_TYPE_ALIASES.get(t, [t]):
            if alias not in expanded:
                expanded.append(alias)
    return expanded


def normalize_identifier(value: str) -> str:
    """Normalize an identifier for simple case-insensitive text matching.

    Args:
        value: Raw identifier or nearby text from an email or document.

    Returns:
        Lowercase alphanumeric-only string.
    """
    return re.sub(r"[^a-z0-9]", "", value.lower())


def row_matches_identifier(row: dict[str, Any], identifier: str) -> bool:
    """Return True when a document row contains the requested identifier.

    Args:
        row: HANA shipment document row.
        identifier: Requested identifier extracted from the email.

    Returns:
        True when file name or raw text contains the normalized identifier.
    """
    haystack = " ".join(
        [
            str(row.get("file_name") or ""),
            str(row.get("raw_text") or ""),
        ]
    )
    return normalize_identifier(identifier) in normalize_identifier(haystack)


def choose_documents_for_request(
    repository: Any,
    client_country: str | None,
    requested_documents: list[dict[str, Any]],
    send_all: bool,
    sales_order_id: str | None = None,
    line_item_number: str | None = None,
    order_status: OrderStatus | None = None,
) -> tuple[list[DocumentAttachment], list[str]]:
    """Select HANA documents for a request using sales order logic.

    When a sales_order_id is provided, documents are fetched by sales order.
    The scenario logic determines which documents to attach:
      - If PGI is not completed: no documents (status-only response)
      - If a specific line item is requested: only packing list
      - Otherwise (full order, PGI complete): all available documents

    Falls back to country-based lookup when no sales_order_id is given.

    Args:
        repository: Object exposing document query methods.
        client_country: Country bucket extracted from the email (fallback).
        requested_documents: Requested document type and identifiers.
        send_all: Whether to attach all documents for broad requests.
        sales_order_id: Sales order number for order-based lookup.
        line_item_number: Specific line item when only partial docs needed.
        order_status: Order status to decide whether docs can be sent.

    Returns:
        Tuple of selected attachments and validation errors.
    """
    if sales_order_id:
        return _choose_documents_by_sales_order(
            repository=repository,
            sales_order_id=sales_order_id,
            line_item_number=line_item_number,
            order_status=order_status,
            requested_documents=requested_documents,
        )

    if not client_country:
        return [], ["sales_order_id or client_country is required"]

    rows = repository.list_documents_for_country(client_country)
    if not rows:
        return [], [f"No documents are available for {client_country}"]

    if not requested_documents:
        return [attachment_from_row(row) for row in rows], []

    selected: list[DocumentAttachment] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for request in requested_documents:
        document_type = request.get("document_type")
        identifiers = [str(item) for item in request.get("identifiers") or [] if str(item).strip()]
        type_rows = [row for row in rows if not document_type or row.get("document_type") == document_type]
        if not type_rows:
            errors.append(f"No {document_type or 'requested'} documents are available for {client_country}")
            continue
        if not identifiers:
            for row in type_rows:
                if row["document_id"] not in seen_ids:
                    selected.append(attachment_from_row(row))
                    seen_ids.add(str(row["document_id"]))
            continue
        for identifier in identifiers:
            matches = [row for row in type_rows if row_matches_identifier(row, identifier)]
            if not matches:
                errors.append(f"No {document_type or 'requested'} document matched identifier {identifier} for {client_country}")
                continue
            for row in matches:
                if row["document_id"] not in seen_ids:
                    selected.append(attachment_from_row(row))
                    seen_ids.add(str(row["document_id"]))

    return selected, errors


def _choose_documents_by_sales_order(
    repository: Any,
    sales_order_id: str,
    line_item_number: str | None,
    order_status: OrderStatus | None,
    requested_documents: list[dict[str, Any]],
) -> tuple[list[DocumentAttachment], list[str]]:
    """Select documents by sales order ID with scenario logic.

    Scenario rules:
      - PGI not completed → no documents attached (status-only)
      - Specific line item requested → packing list only
      - Full order + PGI complete → all applicable docs

    Args:
        repository: Object exposing list_documents_for_sales_order.
        sales_order_id: Sales order number.
        line_item_number: Specific line item number (Scenario 2 behavior).
        order_status: Status data for the order.
        requested_documents: Explicit document requests from extraction.

    Returns:
        Tuple of selected attachments and validation errors.
    """
    errors: list[str] = []

    if order_status and order_status.pgi_status != "completed":
        return [], []

    if line_item_number:
        line_item_number = normalize_line_identifier(line_item_number)
        document_types = ["packing_list"]
    elif requested_documents:
        raw_types = [
            r["document_type"] for r in requested_documents
            if r.get("document_type")
        ]
        document_types = expand_document_types(raw_types) if raw_types else None
    else:
        document_types = None

    rows = repository.list_documents_for_sales_order(sales_order_id, document_types)
    if not rows:
        if order_status and order_status.pgi_status == "completed":
            errors.append(f"No documents found for sales order {sales_order_id}")
        return [], errors

    selected = [attachment_from_row(row) for row in rows]
    return selected, errors


def build_reply_subject(original_subject: str | None) -> str:
    """Build the outgoing email subject.

    Args:
        original_subject: Subject from the source Gmail message.

    Returns:
        Reply subject prefixed with Re: when needed.
    """
    base = original_subject or "Requested shipment documents"
    return base if base.lower().startswith("re:") else f"Re: {base}"


def build_reply_body(
    sales_order_id: str | None,
    attachments: list[DocumentAttachment],
    order_status: OrderStatus | None = None,
    client_country: str | None = None,
) -> str:
    """Build a response body with order status and attachment listing.

    Args:
        sales_order_id: Sales order number for the response.
        attachments: Selected attachment models.
        order_status: Order status to include in the response body.
        client_country: Fallback context when no sales order.

    Returns:
        Plain text draft email body.
    """
    parts: list[str] = ["Hello,\n"]

    order_ref = f"order {sales_order_id}" if sales_order_id else (client_country or "your shipment")

    if order_status:
        parts.append(f"Here is the current status of {order_ref}:\n")
        parts.append(f"  Picking: {_format_status(order_status.picking_status)}")
        parts.append(f"  Packing: {_format_status(order_status.packing_status)}")
        parts.append(f"  Goods Issue (PGI): {_format_status(order_status.pgi_status)}")
        if order_status.shipped and order_status.shipped_date:
            parts.append(f"  Shipped on: {order_status.shipped_date}")
        if order_status.back_order:
            parts.append("  Back order: Yes — some items are still pending")
        if order_status.expected_completion:
            parts.append(f"  Expected completion: {order_status.expected_completion}")
        parts.append("")

    if attachments:
        names = "\n".join(f"  - {item.file_name}" for item in attachments)
        parts.append(f"Please find attached the requested documents for {order_ref}:\n{names}\n")
    elif order_status and order_status.pgi_status != "completed":
        parts.append(
            f"Documents for {order_ref} are not yet available as the goods issue "
            "has not been completed. We will follow up once shipment is finalized.\n"
        )

    parts.append("Regards,\nCustomer Service Assistant")
    return "\n".join(parts)


def _format_status(status: str) -> str:
    """Convert a status slug to human-readable text."""
    return status.replace("_", " ").capitalize()
