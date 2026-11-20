"""Read access to stored orders, messages and attachments (app tables ``POAGENT_*``).

Rows are returned with upper-case column keys and internal SAP number formats; the API layer
converts them for display. All functions are blocking; async callers use ``asyncio.to_thread``.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text

from ..db.engine import get_engine
from ..db.schema import clip_row
from ..db.sql import fetch_all, fetch_one, insert_many
from .matching import compact_id, name_key

ISSUE_BLOCKING = ("ERROR", "REVIEW")


def list_orders(statuses: list[str] | None = None, query: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """List orders for the overview table, newest first.

    Args:
        statuses: Optional STATUS filter.
        query: Optional case-insensitive search in PO number, customer, sender and subject.
        limit: Maximum rows.

    Returns:
        Header summary rows with SENDER, SUBJECT, RECEIVED_AT, SOURCE_TYPE, ITEM_COUNT,
        ISSUE_COUNT (ERROR/REVIEW issues) and MIN_CONFIDENCE.
    """
    params: dict[str, Any] = {"limit": int(limit)}
    where = []
    if statuses:
        names = [f"s{i}" for i in range(len(statuses))]
        where.append(f'h."STATUS" IN ({", ".join(":" + n for n in names)})')
        params.update(dict(zip(names, statuses)))
    if query:
        params["q"] = f"%{query.lower()}%"
        where.append('(LOWER(COALESCE(h."PO_NUMBER", \'\')) LIKE :q OR LOWER(COALESCE(h."SOLD_TO_NAME", \'\')) LIKE :q '
                     'OR LOWER(COALESCE(m."SENDER", \'\')) LIKE :q OR LOWER(COALESCE(m."SUBJECT", \'\')) LIKE :q)')
    sql = f"""
        SELECT h."ORDER_ID", h."STATUS", h."CLASSIFICATION", h."STATUS_MESSAGE", h."PO_NUMBER", h."PO_DATE",
               h."SOLD_TO", h."SOLD_TO_NAME", h."SHIP_TO", h."SHIP_TO_NAME", h."SALES_ORG", h."CURRENCY",
               h."NET_AMOUNT", h."SALES_DOC", h."CREATED_AT", h."UPDATED_AT",
               m."SENDER", m."SUBJECT", m."RECEIVED_AT", m."SOURCE_TYPE",
               (SELECT COUNT(*) FROM "POAGENT_ORDER_ITEM" i WHERE i."ORDER_ID" = h."ORDER_ID") AS "ITEM_COUNT",
               (SELECT COUNT(*) FROM "POAGENT_FIELD_STATE" f WHERE f."ORDER_ID" = h."ORDER_ID"
                   AND f."ISSUE_SEVERITY" IN ('ERROR', 'REVIEW')) AS "ISSUE_COUNT",
               (SELECT MIN(f."CONFIDENCE") FROM "POAGENT_FIELD_STATE" f WHERE f."ORDER_ID" = h."ORDER_ID") AS "MIN_CONFIDENCE"
          FROM "POAGENT_ORDER_HEADER" h
          JOIN "POAGENT_INBOUND_MESSAGE" m ON m."MESSAGE_ID" = h."MESSAGE_ID"
         {"WHERE " + " AND ".join(where) if where else ""}
         ORDER BY m."RECEIVED_AT" DESC, h."CREATED_AT" DESC
         LIMIT :limit
    """
    return fetch_all(sql, **params)


def get_header(order_id: str, conn=None) -> dict[str, Any] | None:
    """Return the full POAGENT_ORDER_HEADER row of an order, or None."""
    return fetch_one('SELECT * FROM "POAGENT_ORDER_HEADER" WHERE "ORDER_ID" = :o', conn, o=order_id)


def get_items(order_id: str, conn=None) -> list[dict[str, Any]]:
    """Return the order lines sorted by item number."""
    return fetch_all('SELECT * FROM "POAGENT_ORDER_ITEM" WHERE "ORDER_ID" = :o ORDER BY "ITEM_NO"', conn, o=order_id)


def get_field_states(order_id: str, conn=None) -> list[dict[str, Any]]:
    """Return all per-field metadata rows of an order."""
    return fetch_all('SELECT * FROM "POAGENT_FIELD_STATE" WHERE "ORDER_ID" = :o ORDER BY "ITEM_NO", "FIELD_NAME"',
                     conn, o=order_id)


def get_message(message_id: str, conn=None) -> dict[str, Any] | None:
    """Return the inbound message row."""
    return fetch_one('SELECT * FROM "POAGENT_INBOUND_MESSAGE" WHERE "MESSAGE_ID" = :m', conn, m=message_id)


def get_attachments(message_id: str, with_content: bool = False, conn=None) -> list[dict[str, Any]]:
    """Return the files of a message in storage order.

    Args:
        message_id: Message id.
        with_content: Also load the BLOB content (needed by the agent, not by the UI list).
        conn: Optional connection.

    Returns:
        Attachment rows.
    """
    columns = '*' if with_content else ('"ATTACHMENT_ID", "MESSAGE_ID", "PARENT_ID", "SEQ", "FILE_NAME", "MIME_TYPE", '
                                       '"SIZE_BYTES", "ROLE", "MODEL_INPUT", "MODEL_INPUT_REASON", "EXTRACTED_TEXT"')
    return fetch_all(f'SELECT {columns} FROM "POAGENT_ATTACHMENT" WHERE "MESSAGE_ID" = :m ORDER BY "SEQ"',
                     conn, m=message_id)


def get_attachment_content(order_id: str, attachment_id: str) -> dict[str, Any] | None:
    """Return file name, MIME type and bytes of one attachment that belongs to the order.

    Args:
        order_id: Order id (guards against reading files of other orders).
        attachment_id: Attachment id.

    Returns:
        Row with FILE_NAME, MIME_TYPE, CONTENT, or None.
    """
    return fetch_one(
        'SELECT a."FILE_NAME", a."MIME_TYPE", a."CONTENT" FROM "POAGENT_ATTACHMENT" a '
        'JOIN "POAGENT_ORDER_HEADER" h ON h."MESSAGE_ID" = a."MESSAGE_ID" '
        'WHERE h."ORDER_ID" = :o AND a."ATTACHMENT_ID" = :a',
        o=order_id, a=attachment_id,
    )


def get_latest_run(order_id: str) -> dict[str, Any] | None:
    """Return the most recent agent run summary (without the trace)."""
    return fetch_one(
        'SELECT "RUN_ID", "STARTED_AT", "FINISHED_AT", "OUTCOME", "MODEL", "LLM_CALLS", "INPUT_TOKENS", '
        '"CACHED_INPUT_TOKENS", "OUTPUT_TOKENS", "ERROR_TEXT" FROM "POAGENT_AGENT_RUN" WHERE "ORDER_ID" = :o '
        'ORDER BY "STARTED_AT" DESC LIMIT 1',
        o=order_id,
    )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def update_header(conn, order_id: str, values: dict[str, Any]) -> None:
    """Update header columns of one order (UPDATED_AT is set automatically).

    Args:
        conn: Connection inside a transaction.
        order_id: Order id.
        values: Column -> value (unknown columns are ignored, strings clipped).
    """
    row = clip_row("POAGENT_ORDER_HEADER", values)
    for column in ("ORDER_ID", "CREATED_AT", "UPDATED_AT"):
        row.pop(column, None)
    if not row:
        return
    assignments = ", ".join(f'"{column}" = :{column}' for column in row)
    conn.execute(
        text(f'UPDATE "POAGENT_ORDER_HEADER" SET {assignments}, "UPDATED_AT" = CURRENT_UTCTIMESTAMP '
             f'WHERE "ORDER_ID" = :order_id'),
        {**row, "order_id": order_id},
    )


def replace_items_and_fields(conn, order_id: str, items: list[dict[str, Any]], field_states: list[dict[str, Any]]) -> None:
    """Replace all lines and field-state rows of an order.

    Args:
        conn: Connection inside a transaction.
        order_id: Order id.
        items: POAGENT_ORDER_ITEM rows (ORDER_ID is set here).
        field_states: POAGENT_FIELD_STATE rows (ORDER_ID is set here).
    """
    conn.execute(text('DELETE FROM "POAGENT_ORDER_ITEM" WHERE "ORDER_ID" = :o'), {"o": order_id})
    conn.execute(text('DELETE FROM "POAGENT_FIELD_STATE" WHERE "ORDER_ID" = :o'), {"o": order_id})
    # One executemany per table: row-by-row inserts cost one HANA Cloud round trip each.
    insert_many(conn, "POAGENT_ORDER_ITEM", [clip_row("POAGENT_ORDER_ITEM", {**i, "ORDER_ID": order_id}) for i in items])
    insert_many(conn, "POAGENT_FIELD_STATE",
                [clip_row("POAGENT_FIELD_STATE", {**s, "ORDER_ID": order_id}) for s in field_states])


def set_status(order_id: str, status: str, message: str | None = None, conn=None, **extra: Any) -> None:
    """Set the processing status (and optional status message / extra header columns).

    Args:
        order_id: Order id.
        status: New STATUS.
        message: STATUS_MESSAGE shown in the UI.
        conn: Optional connection inside a transaction.
        **extra: Further header columns to update.
    """
    values = {"STATUS": status, "STATUS_MESSAGE": message, **extra}
    if conn is not None:
        update_header(conn, order_id, values)
        return
    with get_engine().begin() as own:
        update_header(own, order_id, values)


def delete_order(conn, order: dict[str, Any]) -> None:
    """Delete an order and everything created for it (PoC re-testing; see ``service.delete_order``).

    Removes field states, lines, agent runs, reviewer corrections made on this order, the mock ECC
    sales document posted for it (VBAK/VBKD/VBAP/VBPA) and the header. The inbound message and
    its attachments are removed too when no other order was created from the same message.

    Args:
        conn: Open connection inside a transaction.
        order: Header row of the order (ORDER_ID, MESSAGE_ID, SALES_DOC).
    """
    params = {"o": order["ORDER_ID"]}
    for table in ("POAGENT_FIELD_STATE", "POAGENT_ORDER_ITEM", "POAGENT_AGENT_RUN", "POAGENT_CORRECTION_LOG",
                  "POAGENT_ORDER_HEADER"):
        conn.execute(text(f'DELETE FROM "{table}" WHERE "ORDER_ID" = :o'), params)
    if order.get("SALES_DOC"):
        for table in ("VBAP_POAGENT", "VBPA_POAGENT", "VBKD_POAGENT", "VBAK_POAGENT"):
            conn.execute(text(f'DELETE FROM "{table}" WHERE "VBELN" = :v'), {"v": order["SALES_DOC"]})
    others = conn.execute(text('SELECT COUNT(*) FROM "POAGENT_ORDER_HEADER" WHERE "MESSAGE_ID" = :m'),
                          {"m": order["MESSAGE_ID"]}).scalar()
    if not others:
        for table in ("POAGENT_ATTACHMENT", "POAGENT_INBOUND_MESSAGE"):
            conn.execute(text(f'DELETE FROM "{table}" WHERE "MESSAGE_ID" = :m'), {"m": order["MESSAGE_ID"]})


def find_duplicates(sold_to: str, po_number: str, order_id: str) -> list[str]:
    """Find posted sales documents and other open orders with the same sold-to and PO number.

    Args:
        sold_to: Sold-to (internal format).
        po_number: Customer PO number.
        order_id: Current order (excluded).

    Returns:
        Human-readable references, e.g. ["sales order 9100000004", "order in review 3f2a..."].
    """
    refs = [f"sales order {row['VBELN']}" for row in fetch_all(
        'SELECT k."VBELN" FROM "VBKD_POAGENT" k JOIN "VBAK_POAGENT" a ON a."VBELN" = k."VBELN" '
        'WHERE a."KUNNR" = :s AND k."POSNR" = \'000000\' AND k."BSTKD" = :p', s=sold_to, p=po_number)]
    refs += [f"order {row['ORDER_ID'][:8]} ({row['STATUS']})" for row in fetch_all(
        'SELECT "ORDER_ID", "STATUS" FROM "POAGENT_ORDER_HEADER" WHERE "SOLD_TO" = :s AND "PO_NUMBER" = :p '
        'AND "ORDER_ID" <> :o AND "STATUS" IN (\'QUEUED\', \'PROCESSING\', \'NEEDS_REVIEW\') ',
        s=sold_to, p=po_number, o=order_id)]
    return refs


def insert_corrections(conn, order_id: str, corrections: list[dict[str, Any]]) -> None:
    """Append reviewer corrections to POAGENT_CORRECTION_LOG (long-term agent memory).

    Args:
        conn: Connection inside a transaction.
        order_id: Order id.
        corrections: Rows with ITEM_NO, FIELD_NAME, SOLD_TO, RAW_VALUE, AGENT_VALUE, USER_VALUE.
    """
    insert_many(conn, "POAGENT_CORRECTION_LOG", [
        clip_row("POAGENT_CORRECTION_LOG", {**c, "CORRECTION_ID": str(uuid.uuid4()), "ORDER_ID": order_id})
        for c in corrections])


def learned_materials(sold_to: str) -> dict[str, str]:
    """Reviewer-confirmed material mappings of a sold-to: {normalised printed code: MATNR} (latest wins)."""
    rows = fetch_all(
        'SELECT "RAW_VALUE", "USER_VALUE" FROM "POAGENT_CORRECTION_LOG" WHERE "SOLD_TO" = :s AND "FIELD_NAME" = \'MATERIAL\' '
        'AND "RAW_VALUE" IS NOT NULL AND "USER_VALUE" IS NOT NULL ORDER BY "CREATED_AT"', s=sold_to)
    return {compact_id(r["RAW_VALUE"]): r["USER_VALUE"] for r in rows if compact_id(r["RAW_VALUE"])}


def learned_customers() -> dict[str, str]:
    """Reviewer-confirmed sold-to mappings: {name_key(printed buyer name): KUNNR} (latest wins)."""
    rows = fetch_all(
        'SELECT "RAW_VALUE", "USER_VALUE" FROM "POAGENT_CORRECTION_LOG" WHERE "FIELD_NAME" = \'SOLD_TO\' '
        'AND "RAW_VALUE" IS NOT NULL AND "USER_VALUE" IS NOT NULL ORDER BY "CREATED_AT"')
    return {name_key(r["RAW_VALUE"]): r["USER_VALUE"] for r in rows if name_key(r["RAW_VALUE"])}


def recent_corrections(sold_to: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the latest reviewer corrections for a sold-to (long-term agent memory)."""
    return fetch_all(
        'SELECT "FIELD_NAME", "RAW_VALUE", "AGENT_VALUE", "USER_VALUE", "CREATED_AT" FROM "POAGENT_CORRECTION_LOG" '
        'WHERE "SOLD_TO" = :s ORDER BY "CREATED_AT" DESC LIMIT :n', s=sold_to, n=int(limit))
