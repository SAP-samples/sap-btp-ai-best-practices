"""Build the agent's first message (text + native attachments) for one order.

Layout of the message text (kept stable so provider prompt caching applies across tool rounds):

    ## Inbound message          id, received date, sales-org hint, sender, recipients, subject
    ## E-mail body              full plain-text body (HTML already converted), never truncated
    ## Attachment manifest      [A1] file (type, size) -> attached natively | text below | not attached: reason
    ## Forwarded e-mails / converted documents   full text of TEXT files, labelled with their [A#]
    ## Reviewer note            only when a reviewer requested reprocessing

PDFs and images marked NATIVE are written to a temporary directory and passed as template
``Attachment`` objects, capped by ``MAX_NATIVE_FILES`` / ``MAX_NATIVE_MB``: order documents first,
then inline images by size (largest first). Anything beyond the caps stays listed as not attached.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent_core import Attachment
from ..orders import repository
from ..orders.settings import get_settings


@dataclass
class AgentInput:
    """Prepared input of one agent run; call ``cleanup`` when the run is over."""

    text: str
    attachments: list[Attachment]
    tempdir: tempfile.TemporaryDirectory | None
    native_files: list[str]

    def cleanup(self) -> None:
        """Delete the temporary attachment files."""
        if self.tempdir is not None:
            self.tempdir.cleanup()


def _size(num_bytes: int | None) -> str:
    """Human-readable file size."""
    value = float(num_bytes or 0)
    for unit in ("B", "KB", "MB"):
        if value < 1024 or unit == "MB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} MB"


def _safe_name(label: str, file_name: str) -> str:
    """File name for the temp copy: label prefix + sanitised original name."""
    stem = re.sub(r"[^\w.\- ]", "_", file_name).strip() or "file"
    return f"{label}_{stem}"[:150]


def select_native(attachments: list[dict[str, Any]], max_files: int, max_mb: int) -> set[str]:
    """Choose which NATIVE files are sent to the model within the caps.

    Args:
        attachments: Attachment rows (with SIZE_BYTES, ROLE, MODEL_INPUT, SEQ).
        max_files: Maximum number of files.
        max_mb: Maximum total size.

    Returns:
        ATTACHMENT_IDs to send natively.
    """
    natives = [a for a in attachments if a["MODEL_INPUT"] == "NATIVE"]
    documents = [a for a in natives if a["ROLE"] != "INLINE_IMAGE"]
    inline = sorted((a for a in natives if a["ROLE"] == "INLINE_IMAGE"), key=lambda a: -(a["SIZE_BYTES"] or 0))
    chosen, total = set(), 0
    for row in [*documents, *inline]:
        size = row["SIZE_BYTES"] or 0
        if len(chosen) < max_files and total + size <= max_mb * 1024 * 1024:
            chosen.add(row["ATTACHMENT_ID"])
            total += size
    return chosen


def build_agent_input(order_id: str) -> AgentInput:
    """Load an order's message and files from HANA and build the agent input (blocking).

    Args:
        order_id: Order id.

    Returns:
        AgentInput with message text, native attachments and the temp directory holding them.

    Raises:
        LookupError: If the order does not exist.
    """
    settings = get_settings()
    header = repository.get_header(order_id)
    if header is None:
        raise LookupError(f"order {order_id} not found")
    message = repository.get_message(header["MESSAGE_ID"]) or {}
    files = repository.get_attachments(header["MESSAGE_ID"], with_content=True)
    labels = {row["ATTACHMENT_ID"]: f"A{index}" for index, row in enumerate(files, start=1)}
    native_ids = select_native(files, settings.max_native_files, settings.max_native_mb)

    lines = [
        "Process this inbound purchase-order message following your workflow.",
        "",
        "## Inbound message",
        f"- Order id: {order_id}",
        f"- Received: {message.get('RECEIVED_AT')}",
        f"- Sales organisation of the receiving inbox: {message.get('SALES_ORG_HINT') or 'not specified'}",
        f"- Source: {message.get('SOURCE_TYPE')}",
        f"- From: {message.get('SENDER') or '-'}",
        f"- To/Cc: {message.get('RECIPIENTS') or '-'}",
        f"- Subject: {message.get('SUBJECT') or '-'}",
        "",
        "## E-mail body",
        message.get("BODY_TEXT") or "(empty)",
        "",
        "## Attachment manifest",
    ]
    parents = {row["ATTACHMENT_ID"]: row.get("PARENT_ID") for row in files}
    text_sections: list[str] = []
    tempdir = tempfile.TemporaryDirectory(prefix="po-agent-") if native_ids else None
    attachments: list[Attachment] = []
    native_names: list[str] = []
    for row in files:
        label = labels[row["ATTACHMENT_ID"]]
        depth = 0
        parent = parents.get(row["ATTACHMENT_ID"])
        while parent:
            depth += 1
            parent = parents.get(parent)
        inside = f", inside {labels[row['PARENT_ID']]}" if row.get("PARENT_ID") in labels else ""
        kind = {"SOURCE_EMAIL": "original e-mail", "DOCUMENT": "document", "INLINE_IMAGE": "image in e-mail body",
                "FORWARDED_EMAIL": "forwarded e-mail"}.get(row["ROLE"], row["ROLE"].lower())
        if row["ATTACHMENT_ID"] in native_ids:
            disposition = "attached natively"
            path = Path(tempdir.name) / _safe_name(label, row["FILE_NAME"])
            path.write_bytes(row["CONTENT"] or b"")
            attachments.append(Attachment(path=path, mime_type=row["MIME_TYPE"]))
            native_names.append(path.name)
        elif row["MODEL_INPUT"] == "NATIVE":
            disposition = "not attached: native file cap reached"
        elif row["MODEL_INPUT"] == "TEXT":
            disposition = "converted text below"
            text_sections += ["", f"### [{label}] {row['FILE_NAME']}", row.get("EXTRACTED_TEXT") or ""]
        else:
            disposition = f"not attached: {row.get('MODEL_INPUT_REASON') or 'not supported'}"
        detail = f"; {row['MODEL_INPUT_REASON']}" if row["MODEL_INPUT"] == "NATIVE" and row.get("MODEL_INPUT_REASON") else ""
        lines.append(f"{'  ' * depth}- [{label}] {row['FILE_NAME']} ({kind}{inside}, {row.get('MIME_TYPE')}, "
                     f"{_size(row.get('SIZE_BYTES'))}{detail}) -> {disposition}")
    if not files:
        lines.append("- (no files)")
    if text_sections:
        lines += ["", "## Forwarded e-mails and converted documents", *text_sections]
    if header.get("REVIEWER_NOTE"):
        lines += ["", "## Reviewer note (reprocessing requested)", header["REVIEWER_NOTE"]]
    if native_names:
        lines += ["", f"Native attachments follow in this order: {', '.join(native_names)}"]
    return AgentInput(text="\n".join(lines), attachments=attachments, tempdir=tempdir, native_files=native_names)
