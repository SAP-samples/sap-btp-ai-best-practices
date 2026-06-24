from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime, parseaddr
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup
import requests

from .attachments import (
    AttachmentTooLargeError,
    UnsupportedAttachmentTypeError,
    build_email_attachment,
    decode_base64url_bytes,
    ensure_supported_attachment,
    sanitize_filename,
)
from .btp_destination import (
    BtpDestinationClient,
    DestinationAuthenticationError,
    RuntimeDestination,
)
from .models import EmailAttachment, GmailEmail
from .settings import PriceChangeSettings

REQUEST_TIMEOUT_SECONDS = 30


def build_gmail_query_start(
    now: datetime,
    last_successful_fetch_at: datetime | None,
    fetch_max_days: int,
) -> datetime:
    """Calculate the earliest Gmail date to query for new messages.

    Args:
        now: Current UTC-aware timestamp for the fetch run.
        last_successful_fetch_at: Last completed fetch timestamp from HANA.
        fetch_max_days: Maximum lookback window in days.

    Returns:
        Query start timestamp bounded by fetch_max_days.
    """
    floor = now - timedelta(days=fetch_max_days)
    if last_successful_fetch_at is None:
        return floor
    if last_successful_fetch_at.tzinfo is None:
        last_successful_fetch_at = last_successful_fetch_at.replace(tzinfo=timezone.utc)
    return max(last_successful_fetch_at, floor)


def build_gmail_query(query_start: datetime) -> str:
    """Build a Gmail search query from the selected query start date.

    Args:
        query_start: Earliest timestamp to include in Gmail search.

    Returns:
        Gmail query string using the after: date filter.
    """
    return f"after:{query_start.astimezone(timezone.utc):%Y/%m/%d}"


def build_gmail_service(settings: PriceChangeSettings) -> "GmailDestinationApiClient":
    """Build a Gmail API client backed by BTP Destination authentication.

    Args:
        settings: Runtime settings containing destination lookup configuration.

    Returns:
        GmailDestinationApiClient ready to call Gmail REST endpoints.
    """
    return GmailDestinationApiClient(BtpDestinationClient.from_settings(settings))


def decode_base64url(data: str) -> str:
    """Decode Gmail's base64url message body encoding.

    Args:
        data: Base64url text without guaranteed padding.

    Returns:
        Decoded UTF-8 text with invalid bytes replaced.
    """
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")


def get_header(payload: dict[str, Any], name: str) -> str:
    """Read a message header from a Gmail payload.

    Args:
        payload: Gmail message payload object.
        name: Case-insensitive header name.

    Returns:
        Header value, or an empty string when not present.
    """
    for header in payload.get("headers", []):
        if str(header.get("name", "")).lower() == name.lower():
            return str(header.get("value", ""))
    return ""


def extract_body(payload: dict[str, Any]) -> str:
    """Extract readable body text from a Gmail message payload.

    Args:
        payload: Gmail message payload object.

    Returns:
        Plain text body when available, otherwise HTML converted to text.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []
    stack = [payload]
    while stack:
        part = stack.pop()
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if data and mime_type == "text/plain":
            plain_parts.append(decode_base64url(str(data)))
        elif data and mime_type == "text/html":
            html_parts.append(decode_base64url(str(data)))
        stack.extend(part.get("parts", []))
    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return BeautifulSoup("\n".join(html_parts), "html.parser").get_text("\n").strip()
    return ""


def iter_attachment_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Gmail MIME parts that carry attachment filenames and data.

    Args:
        payload: Gmail message payload object.

    Returns:
        Attachment MIME parts from any nesting level.
    """
    attachment_parts: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        part = stack.pop()
        body = part.get("body", {})
        if part.get("filename") and (body.get("attachmentId") or body.get("data")):
            attachment_parts.append(part)
        stack.extend(part.get("parts", []))
    return attachment_parts


def parse_email_date(raw_date: str) -> datetime | None:
    """Parse an email Date header into a timezone-aware datetime.

    Args:
        raw_date: Raw RFC 2822-style Date header.

    Returns:
        Parsed datetime, or None when the header is empty.
    """
    if not raw_date:
        return None
    parsed = parsedate_to_datetime(raw_date)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def email_from_message(message: dict[str, Any]) -> GmailEmail:
    """Convert a Gmail API message object into the app's GmailEmail model.

    Args:
        message: Gmail API message returned with format=full.

    Returns:
        GmailEmail normalized for HANA persistence and extraction.
    """
    payload = message.get("payload", {})
    raw_from = get_header(payload, "From")
    sender_name, sender_email = parseaddr(raw_from)
    return GmailEmail(
        gmail_message_id=str(message["id"]),
        thread_id=str(message.get("threadId", "")),
        sender_name=sender_name or None,
        sender_email=sender_email or None,
        subject=get_header(payload, "Subject") or None,
        email_date=parse_email_date(get_header(payload, "Date")),
        body=extract_body(payload),
    )


@dataclass
class GmailAttachmentDownloadResult:
    """Result of downloading supported attachments for one Gmail message.

    Attributes:
        attachments: Validated supported attachments with base64 content.
        skipped: Unsupported or over-limit attachments skipped intentionally.
        failed: Supported attachments that could not be downloaded or decoded.
    """

    attachments: list[EmailAttachment]
    skipped: int = 0
    failed: int = 0


class GmailApiError(RuntimeError):
    """Raised when a Gmail API request fails."""


class GmailDestinationApiClient:
    """Small Gmail REST client that obtains bearer tokens from BTP Destination."""

    def __init__(
        self,
        destination_client: BtpDestinationClient,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the Gmail client.

        Args:
            destination_client: Client used to resolve Gmail base URL and auth header.
            session: Optional requests-compatible session for HTTP calls.
        """
        self.destination_client = destination_client
        self.session = session or requests.Session()

    def list_messages(
        self,
        user_id: str,
        label_ids: list[str],
        query: str,
        max_results: int,
        page_token: str | None,
    ) -> dict[str, Any]:
        """List Gmail message references for a mailbox.

        Args:
            user_id: Gmail mailbox id, usually "me".
            label_ids: Gmail labels to constrain the search.
            query: Gmail search query.
            max_results: Maximum messages per page.
            page_token: Optional pagination token.

        Returns:
            Gmail API list response.
        """
        params: dict[str, Any] = {
            "labelIds": label_ids,
            "q": query,
            "maxResults": max_results,
        }
        if page_token:
            params["pageToken"] = page_token
        return self._request_json("GET", f"users/{quote(user_id, safe='')}/messages", params=params)

    def get_message(self, user_id: str, message_id: str, format_: str = "full") -> dict[str, Any]:
        """Fetch one Gmail message.

        Args:
            user_id: Gmail mailbox id, usually "me".
            message_id: Gmail message id.
            format_: Gmail response format.

        Returns:
            Gmail API message response.
        """
        return self._request_json(
            "GET",
            f"users/{quote(user_id, safe='')}/messages/{quote(message_id, safe='')}",
            params={"format": format_},
        )

    def get_attachment(self, user_id: str, message_id: str, attachment_id: str) -> dict[str, Any]:
        """Fetch one Gmail message attachment payload.

        Args:
            user_id: Gmail mailbox id, usually "me".
            message_id: Gmail message id.
            attachment_id: Gmail attachment id from a MIME part body.

        Returns:
            Gmail attachment response containing base64url data.
        """
        return self._request_json(
            "GET",
            (
                f"users/{quote(user_id, safe='')}/messages/{quote(message_id, safe='')}"
                f"/attachments/{quote(attachment_id, safe='')}"
            ),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a Gmail request and retry once after a stale bearer token.

        Args:
            method: HTTP method.
            path: Path under the destination Gmail base URL.
            params: Optional query parameters.

        Returns:
            Parsed JSON response object.
        """
        for attempt in range(2):
            runtime_destination = self.destination_client.resolve_runtime_destination()
            response = self._send_request(method, runtime_destination, path, params)
            if response.status_code == 401 and attempt == 0:
                self.destination_client.clear_cache()
                continue
            self._raise_for_gmail_error(response, path)
            payload = response.json()
            if not isinstance(payload, dict):
                raise GmailApiError("Gmail API response did not contain a JSON object.")
            return payload
        raise GmailApiError("Gmail API request failed after refreshing destination auth.")

    def _send_request(
        self,
        method: str,
        runtime_destination: RuntimeDestination,
        path: str,
        params: dict[str, Any] | None,
    ) -> requests.Response:
        """Send one HTTP request to Gmail.

        Args:
            method: HTTP method.
            runtime_destination: Resolved Gmail destination URL and auth headers.
            path: Path under the destination Gmail base URL.
            params: Optional query parameters.

        Returns:
            Raw requests response.
        """
        url = f"{runtime_destination.url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Accept": "application/json", **runtime_destination.headers}
        return self.session.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def _raise_for_gmail_error(self, response: requests.Response, path: str) -> None:
        """Raise an application error for failed Gmail responses.

        Args:
            response: Raw requests response.
            path: Gmail path being called.
        """
        if response.status_code < 400:
            return
        if "invalid_grant" in response.text:
            raise DestinationAuthenticationError(
                "The Google refresh token configured on the Gmail destination was rejected. "
                "Please re-authorize Gmail with google_tmp/google_fetch.py and replace the destination property."
            )
        raise GmailApiError(f"Gmail API request to {path} failed with HTTP {response.status_code}.")


class GmailFetchService:
    """Fetch Gmail inbox message references and full messages for processing."""

    def __init__(self, service: GmailDestinationApiClient, mailbox_id: str = "me") -> None:
        """Initialize the fetch service.

        Args:
            service: Gmail API client.
            mailbox_id: Gmail mailbox id, usually "me".
        """
        self.service = service
        self.mailbox_id = mailbox_id

    def list_inbox_message_refs(self, query: str) -> list[dict[str, str]]:
        """List all inbox message references matching a Gmail query.

        Args:
            query: Gmail search query.

        Returns:
            List of Gmail message reference objects.
        """
        message_refs: list[dict[str, str]] = []
        page_token: str | None = None
        while True:
            result = self.service.list_messages(
                user_id=self.mailbox_id,
                label_ids=["INBOX"],
                query=query,
                max_results=500,
                page_token=page_token,
            )
            message_refs.extend(result.get("messages", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                return message_refs

    def get_full_messages(self, message_refs: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Fetch full Gmail messages for the provided message references.

        Args:
            message_refs: Gmail message references with id fields.

        Returns:
            Full Gmail message objects.
        """
        return [
            self.service.get_message(
                user_id=self.mailbox_id,
                message_id=item["id"],
                format_="full",
            )
            for item in message_refs
        ]

    def get_attachment_data(self, message_id: str, part: dict[str, Any]) -> bytes:
        """Read attachment bytes from inline data or the Gmail attachment endpoint.

        Args:
            message_id: Gmail message id.
            part: Gmail MIME attachment part.

        Returns:
            Raw attachment bytes.
        """
        body = part.get("body", {})
        inline_data = body.get("data")
        if inline_data:
            return decode_base64url_bytes(str(inline_data))
        attachment_id = str(body["attachmentId"])
        result = self.service.get_attachment(self.mailbox_id, message_id, attachment_id)
        return decode_base64url_bytes(str(result["data"]))

    def download_supported_attachments(self, message: dict[str, Any]) -> GmailAttachmentDownloadResult:
        """Download supported PDF, CSV, and XLSX attachments for one Gmail message.

        Args:
            message: Full Gmail message object.

        Returns:
            Download result with attachment models and skip/failure counters.
        """
        message_id = str(message["id"])
        attachments: list[EmailAttachment] = []
        skipped = 0
        failed = 0
        for part in iter_attachment_parts(message.get("payload", {})):
            filename = sanitize_filename(str(part.get("filename", "")))
            mime_type = str(part.get("mimeType", ""))
            try:
                ensure_supported_attachment(filename, mime_type)
            except UnsupportedAttachmentTypeError:
                skipped += 1
                continue
            if len(attachments) >= 5:
                skipped += 1
                continue
            body = part.get("body", {})
            attachment_id = str(body.get("attachmentId") or "")
            try:
                data = self.get_attachment_data(message_id, part)
                attachments.append(
                    build_email_attachment(
                        filename=filename,
                        mime_type=mime_type,
                        content=data,
                        source="gmail",
                        provider_attachment_id=attachment_id or None,
                        gmail_message_id=message_id,
                    )
                )
            except (AttachmentTooLargeError, UnsupportedAttachmentTypeError):
                skipped += 1
            except Exception:
                failed += 1
        return GmailAttachmentDownloadResult(attachments=attachments, skipped=skipped, failed=failed)
