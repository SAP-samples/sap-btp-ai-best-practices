from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any


def decode_json_b64(value: str | None) -> dict[str, Any] | None:
    """Decode a base64-encoded JSON object from an environment variable.

    Args:
        value: Base64-encoded JSON string, or None when unset.

    Returns:
        Decoded dictionary when a value is provided, otherwise None.
    """
    if not value:
        return None
    decoded = base64.b64decode(value.encode("ascii")).decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("Decoded JSON value must be an object")
    return payload


@dataclass(frozen=True)
class DocumentRequestSettings:
    """Runtime configuration for the Customer Service Assistant API.

    Attributes:
        hana_address: SAP HANA host name.
        hana_port: SAP HANA SQL port.
        hana_user: SAP HANA database user.
        hana_password: SAP HANA database password.
        hana_encrypt: Whether HANA TLS encryption is enabled.
        fetch_max_days: Maximum Gmail lookback window.
        fetch_max_workers: Emails processed in parallel during one background run.
        extractor_model: Fast model used for email classification and extraction.
        agent_model: Full model reserved for agent orchestration prompts.
        gmail_mailbox_id: Gmail mailbox id, usually "me".
        gmail_destination_name: BTP destination name that can read Gmail and create/send drafts.
        gmail_destination_level: Destination lookup level, "subaccount" or "instance".
        gmail_refresh_token_property: Destination Additional Property holding the Google refresh token.
        destination_service_name: Bound CF Destination service instance name.
        from_email: Optional From header for draft messages.
    """

    hana_address: str | None
    hana_port: str | None
    hana_user: str | None
    hana_password: str | None
    hana_encrypt: bool
    fetch_max_days: int
    fetch_max_workers: int
    extractor_model: str
    agent_model: str
    gmail_mailbox_id: str
    gmail_destination_name: str
    gmail_destination_level: str
    gmail_refresh_token_property: str
    destination_service_name: str | None
    from_email: str | None

    @classmethod
    def from_env(cls) -> "DocumentRequestSettings":
        """Build settings from process environment variables.

        Returns:
            DocumentRequestSettings populated with configured values and defaults.
        """
        return cls(
            hana_address=os.getenv("HANA_ADDRESS"),
            hana_port=os.getenv("HANA_PORT"),
            hana_user=os.getenv("HANA_USER"),
            hana_password=os.getenv("HANA_PASSWORD"),
            hana_encrypt=os.getenv("HANA_ENCRYPT", "true").lower() == "true",
            fetch_max_days=int(os.getenv("FETCH_MAX_DAYS", "7")),
            fetch_max_workers=max(1, int(os.getenv("FETCH_MAX_WORKERS", "4"))),
            extractor_model=os.getenv("DOCUMENT_REQUEST_EXTRACTOR_MODEL", "gpt-6-luna"),
            agent_model=os.getenv("DOCUMENT_REQUEST_AGENT_MODEL", "gpt-6-luna"),
            gmail_mailbox_id=os.getenv("GMAIL_MAILBOX_ID", "me"),
            gmail_destination_name=os.getenv("GMAIL_DESTINATION_NAME", "GMAIL_API_READONLY"),
            gmail_destination_level=os.getenv("GMAIL_DESTINATION_LEVEL", "subaccount"),
            gmail_refresh_token_property=os.getenv("GMAIL_REFRESH_TOKEN_PROPERTY", "GMAIL_REFRESH_TOKEN"),
            destination_service_name=os.getenv("DESTINATION_SERVICE_NAME", "shipment-document-request-destination"),
            from_email=os.getenv("DOCUMENT_REQUEST_FROM_EMAIL"),
        )
