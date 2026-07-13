from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


DEFAULT_TOKEN_TTL_SECONDS = 300
TOKEN_EXPIRY_SAFETY_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 30
S4_BTP_REQUIRED_ENV_VARS = (
    "S4_DESTINATION_NAME",
    "DESTINATION_SERVICE_URI",
    "DESTINATION_TOKEN_BASE_URL",
    "DESTINATION_CLIENT_ID",
    "DESTINATION_CLIENT_SECRET",
    "CONNECTIVITY_PROXY_HOST",
    "CONNECTIVITY_PROXY_PORT",
    "CONNECTIVITY_TOKEN_BASE_URL",
    "CONNECTIVITY_CLIENT_ID",
    "CONNECTIVITY_CLIENT_SECRET",
)


class S4BtpConnectivityError(RuntimeError):
    """Raised when S/4 Destination or Connectivity runtime setup fails."""


@dataclass(frozen=True)
class S4RuntimeContext:
    """Runtime request context for one S/4 OData call.

    Args:
        base_url: Base S/4 URL to use for service paths.
        client: Optional SAP client number for request query parameters.
        headers: Extra request headers, including backend and proxy auth.
        proxies: Optional requests proxy mapping.
        verify: TLS verification setting for requests.
        expires_at: UTC expiry for token-derived context, or None for direct access.
    """

    base_url: str
    client: str | None
    headers: dict[str, str]
    proxies: dict[str, str] | None
    verify: bool
    expires_at: datetime | None = None


@dataclass(frozen=True)
class S4BtpConnectivityConfig:
    """Configuration needed to reach S/4 through BTP Destination and Connectivity.

    Args:
        destination_name: BTP destination name, for example S4_SIA_550_HTTP.
        destination_service_uri: Destination service API base URI.
        destination_token_base_url: OAuth issuer base URL for Destination token.
        destination_client_id: Destination service-key client id.
        destination_client_secret: Destination service-key client secret.
        connectivity_proxy_host: Internal Connectivity proxy host.
        connectivity_proxy_port: Internal Connectivity proxy port.
        connectivity_token_base_url: OAuth issuer base URL for Connectivity token.
        connectivity_client_id: Connectivity service-key client id.
        connectivity_client_secret: Connectivity service-key client secret.
        fallback_sap_client: SAP client from direct S4_CLIENT, used when the destination has none.
        verify: TLS verification setting for the backend S/4 request.
        debug: Whether callers should emit extra non-secret diagnostics.
    """

    destination_name: str
    destination_service_uri: str
    destination_token_base_url: str
    destination_client_id: str
    destination_client_secret: str
    connectivity_proxy_host: str
    connectivity_proxy_port: str
    connectivity_token_base_url: str
    connectivity_client_id: str
    connectivity_client_secret: str
    fallback_sap_client: str | None
    verify: bool
    debug: bool = False


def parse_bool_env(value: str | None, default: bool) -> bool:
    """Parse a boolean environment-style value.

    Args:
        value: Raw environment variable value.
        default: Value returned when the variable is missing or empty.

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


def mask_secret(value: str | None) -> str:
    """Mask a potentially sensitive value for logs and errors.

    Args:
        value: Secret value.

    Returns:
        Redacted string.
    """
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return f"{value[:4]}***{value[-2:]}"


def is_sensitive_key(key: str) -> bool:
    """Return whether a JSON key name usually contains sensitive data.

    Args:
        key: JSON object key.

    Returns:
        True when values under this key should be redacted.
    """
    normalized = key.lower()
    return any(
        marker in normalized
        for marker in (
            "authorization",
            "client",
            "password",
            "secret",
            "token",
            "user",
        )
    )


def sanitize_obj(value: Any) -> Any:
    """Redact sensitive values in a JSON-compatible object.

    Args:
        value: Object to sanitize.

    Returns:
        Sanitized object with the same shape.
    """
    if isinstance(value, dict):
        return {
            key: "***" if is_sensitive_key(str(key)) else sanitize_obj(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_obj(item) for item in value]
    return value


def sanitize_text(value: str) -> str:
    """Redact credentials and tokens from diagnostic text.

    Args:
        value: Raw text that may contain JSON or HTTP auth strings.

    Returns:
        Redacted text snippet.
    """
    try:
        parsed = json.loads(value)
    except Exception:
        sanitized = value
    else:
        sanitized = json.dumps(sanitize_obj(parsed), ensure_ascii=True)

    patterns = [
        (r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/=]+)", r"\1***"),
        (r"(?i)(basic\s+)([A-Za-z0-9\-._~+/=]+)", r"\1***"),
        (r'("access_token"\s*:\s*")[^"]+"', r'\1***"'),
        (r'("client_secret"\s*:\s*")[^"]+"', r'\1***"'),
        (r'("password"\s*:\s*")[^"]+"', r'\1***"'),
    ]
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized[:500]


def load_s4_btp_connectivity_config_from_env(
    env: Mapping[str, str] | None = None,
) -> S4BtpConnectivityConfig | None:
    """Load optional S/4 BTP connectivity config from environment variables.

    Args:
        env: Optional environment mapping. Defaults to os.environ.

    Returns:
        Complete config when all required values are present, otherwise None.
    """
    source = env or os.environ
    values = {
        name: str(source.get(name, "")).strip()
        for name in S4_BTP_REQUIRED_ENV_VARS
    }
    if not all(values.values()):
        return None
    fallback_sap_client = str(source.get("S4_CLIENT", "")).strip() or None
    return S4BtpConnectivityConfig(
        destination_name=values["S4_DESTINATION_NAME"],
        destination_service_uri=values["DESTINATION_SERVICE_URI"],
        destination_token_base_url=values["DESTINATION_TOKEN_BASE_URL"],
        destination_client_id=values["DESTINATION_CLIENT_ID"],
        destination_client_secret=values["DESTINATION_CLIENT_SECRET"],
        connectivity_proxy_host=values["CONNECTIVITY_PROXY_HOST"],
        connectivity_proxy_port=values["CONNECTIVITY_PROXY_PORT"],
        connectivity_token_base_url=values["CONNECTIVITY_TOKEN_BASE_URL"],
        connectivity_client_id=values["CONNECTIVITY_CLIENT_ID"],
        connectivity_client_secret=values["CONNECTIVITY_CLIENT_SECRET"],
        fallback_sap_client=fallback_sap_client,
        verify=parse_bool_env(source.get("S4_VERIFY"), default=True),
        debug=parse_bool_env(source.get("BTP_DEST_DEBUG"), default=False),
    )


def extract_error_message(response: requests.Response) -> str:
    """Extract a sanitized response error message.

    Args:
        response: Failed HTTP response.

    Returns:
        Redacted error text.
    """
    try:
        payload = response.json()
    except ValueError:
        return sanitize_text(response.text)
    return sanitize_text(json.dumps(payload))


def destination_configuration(destination_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the destinationConfiguration object from a Destination response.

    Args:
        destination_payload: Destination service response.

    Returns:
        Destination configuration dictionary.
    """
    config = destination_payload.get("destinationConfiguration", destination_payload)
    return config if isinstance(config, dict) else {}


def extract_destination_auth_header(destination_payload: dict[str, Any]) -> dict[str, str]:
    """Extract backend auth headers returned by Destination service.

    Args:
        destination_payload: Destination service response.

    Returns:
        Authentication headers for the backend S/4 call.
    """
    auth_tokens = destination_payload.get("authTokens") or []
    if not isinstance(auth_tokens, list) or not auth_tokens:
        return {}
    first_token = auth_tokens[0]
    if not isinstance(first_token, dict):
        return {}
    http_header = first_token.get("http_header") or first_token.get("httpHeader")
    if isinstance(http_header, dict):
        key = http_header.get("key")
        value = http_header.get("value")
        if key and value:
            return {str(key): str(value)}
        if http_header:
            return {str(key): str(value) for key, value in http_header.items()}
    token_value = first_token.get("value")
    token_type = first_token.get("type")
    if token_value and token_type:
        return {"Authorization": f"{token_type} {token_value}"}
    if token_value:
        return {"Authorization": str(token_value)}
    return {}


def extract_sap_client(destination_payload: dict[str, Any]) -> str | None:
    """Extract optional sap-client from destination properties.

    Args:
        destination_payload: Destination service response.

    Returns:
        SAP client value or None.
    """
    config = destination_configuration(destination_payload)
    for key, value in config.items():
        if key.lower() in {"sap-client", "sap_client", "sap.client"} and value:
            return str(value)
    properties = (
        config.get("properties")
        or config.get("Properties")
        or config.get("additionalProperties")
        or config.get("AdditionalProperties")
    )
    if isinstance(properties, dict):
        for key, value in properties.items():
            if str(key).lower() in {"sap-client", "sap_client", "sap.client"} and value:
                return str(value)
    if isinstance(properties, list):
        for item in properties:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("name")
            value = item.get("value")
            if key and str(key).lower() in {"sap-client", "sap_client", "sap.client"} and value:
                return str(value)
    return None


def auth_token_expiry(
    token_payload: dict[str, Any],
    now: Callable[[], datetime],
) -> datetime:
    """Calculate a safe token expiry timestamp.

    Args:
        token_payload: OAuth or Destination auth token payload.
        now: Current UTC time provider.

    Returns:
        Expiry timestamp with a safety window removed.
    """
    raw_expires_in = token_payload.get("expires_in") or token_payload.get("expiresIn")
    try:
        expires_in = int(raw_expires_in) if raw_expires_in is not None else DEFAULT_TOKEN_TTL_SECONDS
    except (TypeError, ValueError):
        expires_in = DEFAULT_TOKEN_TTL_SECONDS
    cache_seconds = max(1, expires_in - TOKEN_EXPIRY_SAFETY_SECONDS)
    return now() + timedelta(seconds=cache_seconds)


class S4BtpConnectivityClient:
    """Resolve S/4 runtime access through BTP Destination and Connectivity."""

    def __init__(
        self,
        config: S4BtpConnectivityConfig,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Create the runtime resolver.

        Args:
            config: S/4 BTP connectivity configuration.
            session: Optional requests session for tests.
            now: Optional UTC clock provider for tests.
        """
        self.config = config
        self.session = session or requests.Session()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._cached_context: S4RuntimeContext | None = None

    def resolve_runtime_context(self) -> S4RuntimeContext:
        """Resolve and cache the S/4 BTP runtime request context.

        Returns:
            S/4 runtime request context.
        """
        cached = self._cached_context
        if cached and cached.expires_at and cached.expires_at > self.now():
            return cached

        destination_token, destination_expires_at = self._fetch_oauth_token(
            self.config.destination_token_base_url,
            self.config.destination_client_id,
            self.config.destination_client_secret,
            "Destination",
        )
        destination_payload = self._fetch_destination(destination_token)
        connectivity_token, connectivity_expires_at = self._fetch_oauth_token(
            self.config.connectivity_token_base_url,
            self.config.connectivity_client_id,
            self.config.connectivity_client_secret,
            "Connectivity",
        )
        context = self._build_runtime_context(
            destination_payload,
            connectivity_token,
            min(destination_expires_at, connectivity_expires_at),
        )
        self._cached_context = context
        return context

    def clear_cache(self) -> None:
        """Clear the cached runtime context."""
        self._cached_context = None

    def _fetch_oauth_token(
        self,
        token_base_url: str,
        client_id: str,
        client_secret: str,
        service_name: str,
    ) -> tuple[str, datetime]:
        """Fetch a client-credentials OAuth token.

        Args:
            token_base_url: OAuth issuer base URL.
            client_id: OAuth client id.
            client_secret: OAuth client secret.
            service_name: Service name for error messages.

        Returns:
            Access token and safe expiry timestamp.
        """
        url = f"{token_base_url.rstrip('/')}/oauth/token"
        response = self.session.request(
            "POST",
            url,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise S4BtpConnectivityError(
                f"{service_name} token request failed with HTTP {response.status_code}: "
                f"{extract_error_message(response)}"
            )
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise S4BtpConnectivityError(f"{service_name} token response did not include access_token.")
        return str(payload["access_token"]), auth_token_expiry(payload, self.now)

    def _fetch_destination(self, destination_token: str) -> dict[str, Any]:
        """Fetch the configured S/4 destination.

        Args:
            destination_token: Destination service bearer token.

        Returns:
            Destination service response payload.
        """
        url = (
            f"{self.config.destination_service_uri.rstrip('/')}/"
            f"destination-configuration/v1/destinations/{self.config.destination_name}"
        )
        response = self.session.request(
            "GET",
            url,
            headers={
                "Authorization": f"Bearer {destination_token}",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise S4BtpConnectivityError(
                f"S/4 destination fetch failed with HTTP {response.status_code}: "
                f"{extract_error_message(response)}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise S4BtpConnectivityError("S/4 destination response was not a JSON object.")
        return payload

    def _build_runtime_context(
        self,
        destination_payload: dict[str, Any],
        connectivity_token: str,
        service_tokens_expire_at: datetime,
    ) -> S4RuntimeContext:
        """Build request headers, proxies, and base URL for S/4 calls.

        Args:
            destination_payload: Destination service response.
            connectivity_token: Connectivity proxy bearer token.
            service_tokens_expire_at: Earliest service-token expiry.

        Returns:
            S/4 runtime request context.
        """
        config = destination_configuration(destination_payload)
        base_url = str(config.get("URL") or config.get("url") or "").strip().rstrip("/")
        if not base_url:
            raise S4BtpConnectivityError("S/4 destination URL is missing.")

        proxy_type = str(config.get("ProxyType") or config.get("proxyType") or "")
        if proxy_type != "OnPremise":
            raise S4BtpConnectivityError("S/4 destination must be configured with ProxyType=OnPremise.")

        auth_type = str(config.get("Authentication") or config.get("authentication") or "")
        headers = {"Accept": "application/json"}
        headers.update(extract_destination_auth_header(destination_payload))
        if auth_type == "BasicAuthentication" and "Authorization" not in headers:
            raise S4BtpConnectivityError("S/4 destination did not provide a backend auth header.")
        headers["Proxy-Authorization"] = f"Bearer {connectivity_token}"

        proxy_url = f"http://{self.config.connectivity_proxy_host}:{self.config.connectivity_proxy_port}"
        auth_tokens = destination_payload.get("authTokens") or []
        destination_auth_expires_at = service_tokens_expire_at
        if isinstance(auth_tokens, list) and auth_tokens and isinstance(auth_tokens[0], dict):
            destination_auth_expires_at = auth_token_expiry(auth_tokens[0], self.now)

        return S4RuntimeContext(
            base_url=base_url,
            client=extract_sap_client(destination_payload) or self.config.fallback_sap_client,
            headers=headers,
            proxies={"http": proxy_url, "https": proxy_url},
            verify=self.config.verify,
            expires_at=min(service_tokens_expire_at, destination_auth_expires_at),
        )
