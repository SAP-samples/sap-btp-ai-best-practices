"""Turn an upload request into stored inbound messages and queued orders.

Grouping rules (one order is created per message):

* every ``.eml`` file is its own message (``SOURCE_TYPE = EML``); the original file is stored
  as ``SOURCE_EMAIL`` for download, its parts as documents / inline images / forwarded e-mails;
* all other files uploaded together, plus the optional pasted e-mail text, form one message
  (``SOURCE_TYPE = UPLOAD``);
* pasted text alone is a message of its own (``SOURCE_TYPE = TEXT``);
* pasted text together with exactly one ``.eml`` and no other files is appended to that
  e-mail's body.

Production note: the e-mail trigger (Power Automate / MS Graph webhook / CPI Mail adapter)
posts the raw ``.eml`` to the same upload endpoint, so this module is the inbox integration seam.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePath

from ..db.engine import get_engine
from ..db.sql import insert_row
from ..orders.settings import get_settings
from .documents import ROLE_SOURCE_EMAIL, StoredFile, classify_document, guess_mime
from .email_parsing import filter_inline_images, parse_eml

ACCEPTED_SUFFIXES = {".eml", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".docx", ".xlsx", ".txt", ".csv"}


class IntakeError(Exception):
    """Upload rejected; ``status`` is the HTTP status code to return."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class UploadedFile:
    """One file received by the upload endpoint."""

    file_name: str
    content_type: str | None
    content: bytes


@dataclass
class IntakeMessage:
    """One inbound message ready to be stored (becomes one order)."""

    source_type: str
    body_text: str
    sales_org_hint: str | None
    sender: str | None = None
    recipients: str | None = None
    subject: str | None = None
    sent_at: datetime | None = None
    files: list[StoredFile] = field(default_factory=list)


def _now() -> datetime:
    """Current time as naive UTC (HANA TIMESTAMP has no time zone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def build_messages(files: list[UploadedFile], email_text: str | None, sales_org: str | None) -> list[IntakeMessage]:
    """Validate an upload and group it into messages (no database access).

    Args:
        files: Uploaded files.
        email_text: Optional pasted e-mail content.
        sales_org: Optional sales organisation chosen at upload (simulates the inbox mapping).

    Returns:
        Messages to store, in upload order.

    Raises:
        IntakeError: Empty upload (400), unsupported file type (400) or upload too large (413).
    """
    settings = get_settings()
    text = (email_text or "").strip()
    sales_org = (sales_org or "").strip().upper() or None
    if not files and not text:
        raise IntakeError(400, "Upload at least one file or paste the e-mail content.")
    total = sum(len(f.content) for f in files) + len(text.encode())
    if total > settings.max_upload_mb * 1024 * 1024:
        raise IntakeError(413, f"Upload exceeds {settings.max_upload_mb} MB.")
    for uploaded in files:
        suffix = PurePath(uploaded.file_name).suffix.lower()
        if suffix == ".msg":
            raise IntakeError(400, f"{uploaded.file_name}: Outlook .msg is not supported; save the e-mail as .eml.")
        if suffix not in ACCEPTED_SUFFIXES:
            raise IntakeError(400, f"{uploaded.file_name}: unsupported file type. Accepted: {', '.join(sorted(ACCEPTED_SUFFIXES))}")
        if not uploaded.content:
            raise IntakeError(400, f"{uploaded.file_name}: file is empty.")

    messages: list[IntakeMessage] = []
    loose: list[StoredFile] = []
    for uploaded in files:
        if PurePath(uploaded.file_name).suffix.lower() == ".eml":
            parsed = parse_eml(uploaded.content)
            filter_inline_images(parsed.files, settings.inline_image_min_px)
            original = StoredFile(uploaded.file_name, "message/rfc822", uploaded.content, ROLE_SOURCE_EMAIL,
                                  "NONE", "original e-mail; its body and parts are given separately")
            messages.append(IntakeMessage(
                source_type="EML", body_text=parsed.body_text, sales_org_hint=sales_org, sender=parsed.sender,
                recipients=parsed.recipients, subject=parsed.subject, sent_at=parsed.sent_at,
                files=[original, *parsed.files],
            ))
        else:
            mime = guess_mime(uploaded.file_name, uploaded.content_type)
            loose.append(classify_document(uploaded.file_name, mime, uploaded.content))

    if loose:
        subject = loose[0].file_name if len(loose) == 1 else f"{loose[0].file_name} (+{len(loose) - 1} files)"
        messages.append(IntakeMessage("UPLOAD", text, sales_org, subject=subject, files=loose))
    elif text and len(messages) == 1:
        messages[0].body_text += "\n\n--- Additional text provided at upload ---\n" + text
    elif text:
        messages.append(IntakeMessage("TEXT", text, sales_org, subject=text.splitlines()[0][:200]))
    return messages


def _store_files(conn, message_id: str, files: list[StoredFile], parent_id: str | None, seq: list[int]) -> None:
    """Insert files depth-first, linking forwarded-e-mail children through PARENT_ID."""
    for stored in files:
        attachment_id = str(uuid.uuid4())
        seq[0] += 1
        insert_row(conn, "POAGENT_ATTACHMENT", {
            "ATTACHMENT_ID": attachment_id, "MESSAGE_ID": message_id, "PARENT_ID": parent_id, "SEQ": seq[0],
            "FILE_NAME": stored.file_name[:500], "MIME_TYPE": stored.mime_type[:200], "SIZE_BYTES": len(stored.content),
            "ROLE": stored.role, "MODEL_INPUT": stored.model_input,
            "MODEL_INPUT_REASON": (stored.model_input_reason or None) and stored.model_input_reason[:200],
            "CONTENT": stored.content, "EXTRACTED_TEXT": stored.extracted_text,
        })
        _store_files(conn, message_id, stored.children, attachment_id, seq)


def store_messages(messages: list[IntakeMessage]) -> list[dict[str, str]]:
    """Persist messages, their files and one QUEUED order per message (single transaction).

    Args:
        messages: Output of ``build_messages``.

    Returns:
        ``[{"order_id", "message_id", "source"}]`` in input order.
    """
    created = []
    now = _now()
    with get_engine().begin() as conn:
        for message in messages:
            message_id, order_id = str(uuid.uuid4()), str(uuid.uuid4())
            insert_row(conn, "POAGENT_INBOUND_MESSAGE", {
                "MESSAGE_ID": message_id, "SOURCE_TYPE": message.source_type,
                "SENDER": (message.sender or None) and message.sender[:320],
                "RECIPIENTS": (message.recipients or None) and message.recipients[:2000],
                "SUBJECT": (message.subject or None) and message.subject[:1000],
                "SENT_AT": message.sent_at,
                # The e-mail's own date is the "document received date" of the requirements.
                "RECEIVED_AT": message.sent_at or now,
                "BODY_TEXT": message.body_text, "SALES_ORG_HINT": message.sales_org_hint,
            })
            _store_files(conn, message_id, message.files, None, [0])
            insert_row(conn, "POAGENT_ORDER_HEADER", {
                "ORDER_ID": order_id, "MESSAGE_ID": message_id, "STATUS": "QUEUED", "ATTEMPTS": 0,
                "CREATED_AT": now, "UPDATED_AT": now,
            })
            created.append({"order_id": order_id, "message_id": message_id, "source": message.subject or message.source_type})
    return created


def intake_upload(files: list[UploadedFile], email_text: str | None, sales_org: str | None) -> list[dict[str, str]]:
    """Validate, group and store an upload (blocking; call via ``asyncio.to_thread``).

    Args:
        files: Uploaded files.
        email_text: Optional pasted e-mail content.
        sales_org: Optional sales organisation hint.

    Returns:
        Created orders (see ``store_messages``).

    Raises:
        IntakeError: When the upload is rejected.
    """
    return store_messages(build_messages(files, email_text, sales_org))
