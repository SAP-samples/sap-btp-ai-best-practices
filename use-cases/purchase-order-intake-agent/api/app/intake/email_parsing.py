"""Parse .eml files into header fields, body text and the files the order agent may need.

Handled MIME structures (as seen on real customer order e-mails):

* ``multipart/mixed`` with a text/HTML body and attached PDFs, Office files or images;
* ``multipart/related`` bodies with inline images referenced by ``Content-ID`` - kept as
  ``INLINE_IMAGE`` and given to the model, because a customer may paste a screenshot of the
  order into the body; only logos/icons and duplicates are filtered out (``filter_inline_images``);
* forwarded e-mails (``message/rfc822`` parts), parsed recursively; their text becomes a
  ``FORWARDED_EMAIL`` file and their own attachments are nested as children;
* HTML-only bodies, converted to plain text (HTML is never rendered anywhere).

Example::

    parsed = parse_eml(Path("order.eml").read_bytes())
    filter_inline_images(parsed.files, min_px=150)
"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from html.parser import HTMLParser
from typing import Iterator

from .documents import (
    ROLE_DOCUMENT,
    ROLE_FORWARDED_EMAIL,
    ROLE_INLINE_IMAGE,
    StoredFile,
    classify_document,
    guess_mime,
)
from .image_info import image_size


@dataclass
class ParsedEmail:
    """Result of parsing one e-mail.

    Attributes:
        sender: From header.
        recipients: To + Cc headers.
        subject: Subject header.
        sent_at: Date header as naive UTC datetime, if parseable.
        body_text: Plain-text body (HTML converted).
        files: Attachments, inline images and forwarded e-mails (with nested children).
    """

    sender: str | None
    recipients: str | None
    subject: str | None
    sent_at: datetime | None
    body_text: str
    files: list[StoredFile] = field(default_factory=list)


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text converter keeping line structure of paragraphs, rows and cells."""

    BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: D102 - HTMLParser hook
        if tag in ("style", "script", "head"):
            self._skip += 1
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        elif tag == "td":
            self.parts.append(" | ")

    def handle_endtag(self, tag):  # noqa: D102 - HTMLParser hook
        if tag in ("style", "script", "head") and self._skip:
            self._skip -= 1

    def handle_data(self, data):  # noqa: D102 - HTMLParser hook
        if not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    """Convert HTML to readable plain text (no rendering, scripts/styles dropped).

    Args:
        markup: HTML source.

    Returns:
        Text with paragraph/row breaks and collapsed blank lines.
    """
    extractor = _TextExtractor()
    extractor.feed(markup)
    lines = [" ".join(line.split()) for line in html.unescape("".join(extractor.parts)).splitlines()]
    text, blank = [], 0
    for line in lines:
        blank = blank + 1 if not line else 0
        if blank <= 1:
            text.append(line)
    return "\n".join(text).strip()


def _body_text(message: EmailMessage) -> str:
    """Return the plain-text body, falling back to converted HTML."""
    body = message.get_body(preferencelist=("plain", "html"))
    if body is None:
        return ""
    content = body.get_content()
    return html_to_text(content) if body.get_content_type() == "text/html" else content.strip()


def _sent_at(message: EmailMessage) -> datetime | None:
    """Parse the Date header to naive UTC, or None."""
    try:
        value = message["Date"].datetime if message["Date"] else None
    except (AttributeError, TypeError, ValueError):
        return None
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _header(message: EmailMessage, *names: str) -> str | None:
    """Join the given headers (e.g. To and Cc) into one string."""
    values = [str(message[name]) for name in names if message[name]]
    return ", ".join(values) or None


def _forwarded(inner: EmailMessage) -> StoredFile:
    """Build the FORWARDED_EMAIL file of a message/rfc822 part, with nested files."""
    parsed = _parse_message(inner)
    header_lines = [f"From: {parsed.sender or ''}", f"To: {parsed.recipients or ''}",
                    f"Date: {inner['Date'] or ''}", f"Subject: {parsed.subject or ''}"]
    name = (parsed.subject or "forwarded e-mail").strip()[:120] + ".eml"
    return StoredFile(
        file_name=name, mime_type="message/rfc822", content=inner.as_bytes(), role=ROLE_FORWARDED_EMAIL,
        model_input="TEXT", extracted_text="\n".join(header_lines) + "\n\n" + parsed.body_text,
        children=parsed.files,
    )


def _collect(part: EmailMessage, files: list[StoredFile], in_related: bool = False) -> None:
    """Recursively collect non-body parts of a MIME tree.

    Args:
        part: Current MIME part.
        files: Output list.
        in_related: True inside ``multipart/related`` (parts referenced from the HTML body).
    """
    ctype = part.get_content_type()
    if ctype == "message/rfc822":
        payload = part.get_payload()
        inner = payload[0] if isinstance(payload, list) else payload
        files.append(_forwarded(inner))
        return
    if part.is_multipart():
        for sub in part.iter_parts():
            _collect(sub, files, in_related or ctype == "multipart/related")
        return
    disposition = part.get_content_disposition()
    file_name = part.get_filename()
    if ctype in ("text/plain", "text/html") and disposition != "attachment" and not file_name:
        return  # body alternative, already read by get_body
    content = part.get_payload(decode=True) or b""
    # Inline = referenced from the body: Content-ID inside multipart/related (clients differ on the
    # disposition they set), or an explicit "inline" disposition.
    if ctype.startswith("image/") and ((in_related and part["Content-ID"]) or disposition == "inline"):
        cid = (part["Content-ID"] or "").strip("<> ")
        name = file_name or f"inline-{cid or len(files) + 1}.{ctype.split('/')[1]}"
        files.append(StoredFile(name, ctype, content, ROLE_INLINE_IMAGE, "NATIVE"))
        return
    name = file_name or f"attachment-{len(files) + 1}"
    files.append(classify_document(name, guess_mime(name, ctype), content))


def _parse_message(message: EmailMessage) -> ParsedEmail:
    """Parse one (possibly nested) e-mail message."""
    files: list[StoredFile] = []
    if message.is_multipart():
        for part in message.iter_parts():
            _collect(part, files)
    return ParsedEmail(
        sender=_header(message, "From"),
        recipients=_header(message, "To", "Cc"),
        subject=str(message["Subject"]) if message["Subject"] else None,
        sent_at=_sent_at(message),
        body_text=_body_text(message),
        files=files,
    )


def parse_eml(data: bytes) -> ParsedEmail:
    """Parse raw .eml bytes.

    Args:
        data: RFC 5322 message bytes.

    Returns:
        ParsedEmail with headers, body text and files.
    """
    message = BytesParser(policy=policy.default).parsebytes(data)
    return _parse_message(message)


def iter_files(files: list[StoredFile]) -> Iterator[StoredFile]:
    """Yield files depth-first, including children of forwarded e-mails."""
    for stored in files:
        yield stored
        yield from iter_files(stored.children)


def filter_inline_images(files: list[StoredFile], min_px: int) -> None:
    """Mark logo/icon-sized and duplicate inline images as not sent to the model.

    Inline images are kept by default (they may contain the order). An image is dropped
    (``MODEL_INPUT = NONE`` with a reason) only when its shorter side is below ``min_px`` or
    an identical image was already kept in the same message. Images of unknown format keep
    ``NATIVE``. Kept images record their pixel size in ``model_input_reason``.

    Args:
        files: Files of one message (mutated in place, recursively).
        min_px: Minimum shorter side in pixels for a content image.
    """
    kept: dict[str, str] = {}
    for stored in iter_files(files):
        if stored.role != ROLE_INLINE_IMAGE:
            continue
        digest = hashlib.sha256(stored.content).hexdigest()
        size = image_size(stored.content)
        if digest in kept:
            stored.model_input, stored.model_input_reason = "NONE", f"duplicate of {kept[digest]}"
        elif size and min(size) < min_px:
            stored.model_input, stored.model_input_reason = "NONE", f"likely logo/icon ({size[0]}x{size[1]} px)"
        else:
            kept[digest] = stored.file_name
            stored.model_input_reason = f"{size[0]}x{size[1]} px" if size else None


__all__ = ["ParsedEmail", "parse_eml", "filter_inline_images", "iter_files", "html_to_text", "ROLE_DOCUMENT"]
