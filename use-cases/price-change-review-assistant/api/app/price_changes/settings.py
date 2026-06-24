from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any


def normalize_reasoning_effort(value: str | None) -> str:
    """Normalize an optional Responses API reasoning-effort setting.

    Args:
        value: Raw environment variable value.

    Returns:
        Lowercase reasoning-effort value, defaulting to low when unset.
    """
    normalized = (value or "low").strip().lower()
    return normalized or "low"


def parse_bool_env(value: str | None, default: bool = False) -> bool:
    """Parse an environment boolean value.

    Args:
        value: Raw environment variable value.
        default: Value returned when the variable is missing or blank.

    Returns:
        Parsed boolean value.
    """
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_positive_int_env(value: str | None, default: int) -> int:
    """Parse a positive integer environment value.

    Args:
        value: Raw environment variable value.
        default: Value returned when the variable is missing, blank, or invalid.

    Returns:
        Positive integer value or the provided default.
    """
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def decode_json_b64(value: str | None) -> dict[str, Any] | None:
    """Decode a base64-encoded JSON object from an environment variable.

    Args:
        value: Base64-encoded JSON string, or None when unset.

    Returns:
        Decoded JSON object, or None when no value was provided.
    """
    if not value:
        return None
    decoded = base64.b64decode(value.encode("ascii")).decode("utf-8")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("Decoded JSON value must be an object")
    return payload


@dataclass(frozen=True)
class PriceChangeSettings:
    """Runtime configuration for the price-change API.

    Attributes:
        hana_address: SAP HANA host name.
        hana_port: SAP HANA SQL port.
        hana_user: SAP HANA database user.
        hana_password: SAP HANA database password.
        hana_encrypt: Whether HANA TLS encryption is enabled.
        google_credentials_json_b64: Deprecated base64 Google OAuth client JSON.
        google_token_json_b64: Deprecated base64 Google OAuth token JSON.
        fetch_max_days: Maximum Gmail lookback window.
        extractor_model: SAP GenAI Hub model used for email extraction.
        attachment_extractor_model: SAP GenAI Hub model used when attachments require Responses inputs.
        agent_model: SAP GenAI Hub model used for proposal resolution.
        agent_reasoning_effort: OpenAI Responses API reasoning effort for the agent.
        completion_batch_size: Maximum canonical items to send through each completion batch.
        gmail_mailbox_id: Gmail mailbox id, usually "me".
        gmail_destination_name: BTP destination name for Gmail API access.
        gmail_destination_level: Destination lookup level, "subaccount" or "instance".
        gmail_refresh_token_property: Destination Additional Property holding the Google refresh token.
        destination_service_name: Bound CF Destination service instance name.
        s4_destination_name: BTP destination name for S/4 OData access.
        destination_service_uri: Destination service-key API URI for S/4 destination lookup.
        destination_token_base_url: OAuth issuer for Destination service-key tokens.
        destination_client_id: Destination service-key client id.
        destination_client_secret: Destination service-key client secret.
        connectivity_proxy_host: BTP Connectivity proxy host.
        connectivity_proxy_port: BTP Connectivity proxy port.
        connectivity_token_base_url: OAuth issuer for Connectivity service-key tokens.
        connectivity_client_id: Connectivity service-key client id.
        connectivity_client_secret: Connectivity service-key client secret.
        s4_connectivity_mode: S/4 connection mode, `auto`, `direct`, or `btp`.
        btp_dest_debug: Whether to emit extra non-secret BTP connectivity diagnostics.
    """

    hana_address: str | None
    hana_port: str | None
    hana_user: str | None
    hana_password: str | None
    hana_encrypt: bool
    google_credentials_json_b64: str | None
    google_token_json_b64: str | None
    fetch_max_days: int
    extractor_model: str
    agent_model: str
    gmail_mailbox_id: str
    attachment_extractor_model: str = "gpt-5.4"
    agent_reasoning_effort: str = "low"
    completion_batch_size: int = 25
    gmail_destination_name: str = "GMAIL_API_READONLY"
    gmail_destination_level: str = "subaccount"
    gmail_refresh_token_property: str = "GMAIL_REFRESH_TOKEN"
    destination_service_name: str | None = "email-price-classifier-destination"
    s4_destination_name: str | None = None
    destination_service_uri: str | None = None
    destination_token_base_url: str | None = None
    destination_client_id: str | None = None
    destination_client_secret: str | None = None
    connectivity_proxy_host: str | None = None
    connectivity_proxy_port: str | None = None
    connectivity_token_base_url: str | None = None
    connectivity_client_id: str | None = None
    connectivity_client_secret: str | None = None
    s4_connectivity_mode: str = "auto"
    btp_dest_debug: bool = False

    @classmethod
    def from_env(cls) -> "PriceChangeSettings":
        """Build price-change settings from process environment variables.

        Returns:
            PriceChangeSettings populated with configured values and safe defaults.
        """
        return cls(
            hana_address=os.getenv("HANA_ADDRESS"),
            hana_port=os.getenv("HANA_PORT"),
            hana_user=os.getenv("HANA_USER"),
            hana_password=os.getenv("HANA_PASSWORD"),
            hana_encrypt=os.getenv("HANA_ENCRYPT", "true").lower() == "true",
            google_credentials_json_b64=os.getenv("GOOGLE_CREDENTIALS_JSON_B64"),
            google_token_json_b64=os.getenv("GOOGLE_TOKEN_JSON_B64"),
            fetch_max_days=int(os.getenv("FETCH_MAX_DAYS", "7")),
            extractor_model=os.getenv("PRICE_CHANGE_EXTRACTOR_MODEL", "gpt-4o-mini"),
            agent_model=os.getenv("PRICE_CHANGE_AGENT_MODEL", "gpt-5.4"),
            gmail_mailbox_id=os.getenv("GMAIL_MAILBOX_ID", "me"),
            attachment_extractor_model=os.getenv(
                "PRICE_CHANGE_ATTACHMENT_EXTRACTOR_MODEL",
                "gpt-5.4",
            ),
            agent_reasoning_effort=normalize_reasoning_effort(
                os.getenv("PRICE_CHANGE_AGENT_REASONING_EFFORT")
            ),
            completion_batch_size=parse_positive_int_env(
                os.getenv("PRICE_CHANGE_COMPLETION_BATCH_SIZE"),
                default=25,
            ),
            gmail_destination_name=os.getenv("GMAIL_DESTINATION_NAME", "GMAIL_API_READONLY"),
            gmail_destination_level=os.getenv("GMAIL_DESTINATION_LEVEL", "subaccount"),
            gmail_refresh_token_property=os.getenv(
                "GMAIL_REFRESH_TOKEN_PROPERTY",
                "GMAIL_REFRESH_TOKEN",
            ),
            destination_service_name=os.getenv(
                "DESTINATION_SERVICE_NAME",
                "email-price-classifier-destination",
            ),
            s4_destination_name=os.getenv("S4_DESTINATION_NAME"),
            destination_service_uri=os.getenv("DESTINATION_SERVICE_URI"),
            destination_token_base_url=os.getenv("DESTINATION_TOKEN_BASE_URL"),
            destination_client_id=os.getenv("DESTINATION_CLIENT_ID"),
            destination_client_secret=os.getenv("DESTINATION_CLIENT_SECRET"),
            connectivity_proxy_host=os.getenv("CONNECTIVITY_PROXY_HOST"),
            connectivity_proxy_port=os.getenv("CONNECTIVITY_PROXY_PORT"),
            connectivity_token_base_url=os.getenv("CONNECTIVITY_TOKEN_BASE_URL"),
            connectivity_client_id=os.getenv("CONNECTIVITY_CLIENT_ID"),
            connectivity_client_secret=os.getenv("CONNECTIVITY_CLIENT_SECRET"),
            s4_connectivity_mode=os.getenv("S4_CONNECTIVITY_MODE", "auto"),
            btp_dest_debug=parse_bool_env(os.getenv("BTP_DEST_DEBUG"), default=False),
        )

    def google_credentials_json(self) -> dict[str, Any] | None:
        """Return deprecated Google OAuth client JSON from base64 env config.

        Returns:
            Decoded Google OAuth client JSON, or None when unset.
        """
        return decode_json_b64(self.google_credentials_json_b64)

    def google_token_json(self) -> dict[str, Any] | None:
        """Return deprecated Google OAuth token JSON from base64 env config.

        Returns:
            Decoded Google OAuth token JSON, or None when unset.
        """
        return decode_json_b64(self.google_token_json_b64)
