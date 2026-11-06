from __future__ import annotations

import base64
import mimetypes
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup
import requests

from .btp_destination import BtpDestinationClient, DestinationAuthenticationError, RuntimeDestination
from .models import DocumentAttachment, GmailEmail
from .settings import DocumentRequestSettings

REQUEST_TIMEOUT_SECONDS = 30
GMAIL_DRAFT_AUTH_SCOPES = (
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://mail.google.com/",
)


def build_gmail_query_start(now: datetime, last_successful_fetch_at: datetime | None, fetch_max_days: int) -> datetime:
    """Calculate the earliest Gmail date to query.

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


def build_gmail_query(query_start: datetime | str) -> str:
    """Build a Gmail search query from a start date.

    Args:
        query_start: Datetime or already formatted Gmail after: date.

    Returns:
        Gmail query that excludes sent mail.
    """
    if isinstance(query_start, datetime):
        value = f"{query_start.astimezone(timezone.utc):%Y/%m/%d}"
    else:
        value = query_start
    return f"after:{value} -in:sent"


def build_gmail_service(settings: DocumentRequestSettings) -> "GmailDestinationApiClient":
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


def parse_email_date(raw_date: str) -> datetime | None:
    """Parse an RFC 2822 email date header.

    Args:
        raw_date: Raw Date header value.

    Returns:
        Parsed datetime, or None when the header is empty.
    """
    if not raw_date:
        return None
    parsed = parsedate_to_datetime(raw_date)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


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
        rfc_message_id=get_header(payload, "Message-ID") or None,
        sender_name=sender_name or None,
        sender_email=sender_email or None,
        subject=get_header(payload, "Subject") or None,
        email_date=parse_email_date(get_header(payload, "Date")),
        body=extract_body(payload),
    )


def build_mime_reply_with_attachments(
    source_email: GmailEmail,
    to_email: str,
    from_email: str | None,
    subject: str,
    body: str,
    attachments: list[DocumentAttachment],
) -> str:
    """Build a Gmail raw MIME message containing selected attachments.

    Args:
        source_email: Original Gmail message context for reply threading.
        to_email: Recipient email address.
        from_email: Optional sender email address.
        subject: Draft reply subject.
        body: Plain text draft body.
        attachments: HANA-selected binary attachments.

    Returns:
        Base64url-encoded raw MIME message for Gmail.
    """
    mime_message = EmailMessage()
    mime_message["To"] = to_email
    if from_email:
        mime_message["From"] = from_email
    mime_message["Subject"] = subject
    if source_email.rfc_message_id:
        mime_message["In-Reply-To"] = source_email.rfc_message_id
        mime_message["References"] = source_email.rfc_message_id
    mime_message.set_content(body)
    for attachment in attachments:
        content_type = attachment.mime_type or mimetypes.guess_type(attachment.file_name)[0] or "application/octet-stream"
        maintype, subtype = content_type.split("/", 1)
        mime_message.add_attachment(
            attachment.content,
            maintype=maintype,
            subtype=subtype,
            filename=attachment.file_name,
        )
    return base64.urlsafe_b64encode(mime_message.as_bytes()).decode("ascii")


class GmailApiError(RuntimeError):
    """Raised when a Gmail API request fails."""


def extract_gmail_error_detail(response: requests.Response) -> str:
    """Extract a concise Gmail API error detail from a failed response.

    Args:
        response: Gmail HTTP response.

    Returns:
        Human-readable error details without authorization headers or tokens.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500].strip()
    if not isinstance(payload, dict):
        return str(payload)[:500]
    error = payload.get("error")
    if not isinstance(error, dict):
        return str(error or payload)[:500]

    details: list[str] = []
    message = error.get("message")
    status = error.get("status")
    if message:
        details.append(str(message))
    if status and str(status) not in details:
        details.append(str(status))

    reasons: list[str] = []
    error_items = error.get("errors")
    if isinstance(error_items, list):
        for item in error_items:
            if isinstance(item, dict):
                reason = item.get("reason") or item.get("message")
                if reason and str(reason) not in reasons:
                    reasons.append(str(reason))
    detail_items = error.get("details")
    if isinstance(detail_items, list):
        for item in detail_items:
            if isinstance(item, dict):
                reason = item.get("reason")
                if reason and str(reason) not in reasons:
                    reasons.append(str(reason))
    if reasons:
        details.append("Reason: " + ", ".join(reasons))
    return "; ".join(details)


def is_insufficient_scope_error(response: requests.Response) -> bool:
    """Detect Gmail errors caused by missing OAuth authorization scopes.

    Args:
        response: Gmail HTTP response.

    Returns:
        True when the response body indicates the access token lacks required scopes.
    """
    body = response.text.lower()
    return any(
        marker in body
        for marker in (
            "insufficient authentication scopes",
            "insufficientpermissions",
            "access_token_scope_insufficient",
        )
    )


class GmailDestinationApiClient:
    """Small Gmail REST client that obtains bearer tokens from BTP Destination."""

    def __init__(self, destination_client: BtpDestinationClient, session: requests.Session | None = None) -> None:
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
        """List Gmail message references.

        Args:
            user_id: Gmail mailbox id, usually "me".
            label_ids: Gmail labels to constrain the search.
            query: Gmail search query.
            max_results: Maximum messages per page.
            page_token: Optional pagination token.

        Returns:
            Gmail API list response.
        """
        params: dict[str, Any] = {"labelIds": label_ids, "q": query, "maxResults": max_results}
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

    def create_draft(self, user_id: str, raw_message: str, thread_id: str | None = None) -> dict[str, Any]:
        """Create a Gmail draft from a raw MIME message.

        Args:
            user_id: Gmail mailbox id, usually "me".
            raw_message: Base64url-encoded raw MIME message.
            thread_id: Optional Gmail thread id for reply threading.

        Returns:
            Gmail draft response.
        """
        message: dict[str, Any] = {"raw": raw_message}
        if thread_id:
            message["threadId"] = thread_id
        return self._request_json("POST", f"users/{quote(user_id, safe='')}/drafts", body={"message": message})

    def update_draft(self, user_id: str, draft_id: str, raw_message: str, thread_id: str | None = None) -> dict[str, Any]:
        """Replace an existing Gmail draft with a refreshed MIME message.

        Args:
            user_id: Gmail mailbox id, usually "me".
            draft_id: Gmail draft id to update.
            raw_message: Base64url-encoded raw MIME message.
            thread_id: Optional Gmail thread id for reply threading.

        Returns:
            Gmail draft response after the update.
        """
        message: dict[str, Any] = {"raw": raw_message}
        if thread_id:
            message["threadId"] = thread_id
        return self._request_json(
            "PUT",
            f"users/{quote(user_id, safe='')}/drafts/{quote(draft_id, safe='')}",
            body={"id": draft_id, "message": message},
        )

    def send_draft(self, user_id: str, draft_id: str) -> dict[str, Any]:
        """Send an existing Gmail draft.

        Args:
            user_id: Gmail mailbox id, usually "me".
            draft_id: Gmail draft id to send.

        Returns:
            Gmail sent message response.
        """
        return self._request_json("POST", f"users/{quote(user_id, safe='')}/drafts/send", body={"id": draft_id})

    def delete_draft(self, user_id: str, draft_id: str) -> None:
        """Delete a Gmail draft.

        Args:
            user_id: Gmail mailbox id, usually "me".
            draft_id: Gmail draft id to delete.
        """
        self._request_json("DELETE", f"users/{quote(user_id, safe='')}/drafts/{quote(draft_id, safe='')}")

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a Gmail request and retry once after a stale bearer token.

        Args:
            method: HTTP method.
            path: Path under the destination Gmail base URL.
            params: Optional query parameters.
            body: Optional JSON body.

        Returns:
            Parsed JSON response object, or an empty dict for empty responses.
        """
        for attempt in range(2):
            runtime_destination = self.destination_client.resolve_runtime_destination()
            response = self._send_request(method, runtime_destination, path, params, body)
            if response.status_code == 401 and attempt == 0:
                self.destination_client.clear_cache()
                continue
            self._raise_for_gmail_error(response, path)
            if response.status_code == 204 or not response.text:
                return {}
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
        body: dict[str, Any] | None,
    ) -> requests.Response:
        """Send one HTTP request to Gmail.

        Args:
            method: HTTP method.
            runtime_destination: Resolved destination with base URL and auth headers.
            path: Gmail REST path.
            params: Optional query parameters.
            body: Optional JSON body.

        Returns:
            Raw requests response.
        """
        url = f"{runtime_destination.url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Accept": "application/json", **runtime_destination.headers}
        return self.session.request(method, url, headers=headers, params=params, json=body, timeout=REQUEST_TIMEOUT_SECONDS)

    def _raise_for_gmail_error(self, response: requests.Response, path: str) -> None:
        """Raise an application error for failed Gmail responses.

        Args:
            response: Gmail HTTP response.
            path: Gmail REST path used in the request.
        """
        if response.status_code < 400:
            return
        if "invalid_grant" in response.text:
            raise DestinationAuthenticationError(
                "The Google refresh token configured on the Gmail destination was rejected. "
                "Re-authorize Gmail and update the destination refresh-token property."
            )
        detail = extract_gmail_error_detail(response)
        message = f"Gmail API request to {path} failed with HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        if response.status_code == 403 and is_insufficient_scope_error(response):
            scopes = ", ".join(GMAIL_DRAFT_AUTH_SCOPES)
            message = (
                f"{message}. The configured Google OAuth refresh token does not include Gmail draft permissions. "
                f"Re-authorize the Gmail destination with one of these scopes: {scopes}. "
                "Then replace the destination refresh-token property or set GMAIL_DESTINATION_NAME to a "
                "compose-capable destination."
            )
        raise GmailApiError(message + ".")


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
            Gmail message reference objects.
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
        """Fetch full Gmail messages for the provided references.

        Args:
            message_refs: Gmail message references with id fields.

        Returns:
            Full Gmail message payloads.
        """
        return [
            self.service.get_message(user_id=self.mailbox_id, message_id=item["id"], format_="full")
            for item in message_refs
        ]
