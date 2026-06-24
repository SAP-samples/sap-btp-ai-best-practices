from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

from .settings import PriceChangeSettings

DEFAULT_RUNTIME_TOKEN_TTL_SECONDS = 300
TOKEN_EXPIRY_SAFETY_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 30


class DestinationServiceError(RuntimeError):
    """Raised when the SAP BTP Destination service cannot resolve Gmail auth."""


class DestinationAuthenticationError(DestinationServiceError):
    """Raised when Google rejects the refresh token configured in the destination."""


@dataclass(frozen=True)
class RuntimeDestination:
    """Resolved runtime destination data for calling Gmail.

    Attributes:
        url: Base URL returned by the destination configuration.
        headers: Prepared target-system authentication headers.
        expires_at: UTC time when the prepared auth header should be refreshed.
    """

    url: str
    headers: dict[str, str]
    expires_at: datetime | None


def load_vcap_services(raw_vcap: str | None = None) -> dict[str, Any]:
    """Load and parse the Cloud Foundry VCAP_SERVICES payload.

    Args:
        raw_vcap: Optional raw JSON payload. When omitted, VCAP_SERVICES is read
            from the process environment.

    Returns:
        Parsed VCAP_SERVICES object.

    Raises:
        DestinationServiceError: If VCAP_SERVICES is missing or invalid JSON.
    """
    raw = raw_vcap if raw_vcap is not None else os.getenv("VCAP_SERVICES")
    if not raw:
        raise DestinationServiceError(
            "VCAP_SERVICES is required to resolve GMAIL_API_READONLY. "
            "Bind the API app to the BTP Destination service or run locally with "
            "`cds bind --exec -- uvicorn app.main:app --host 0.0.0.0`."
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DestinationServiceError(f"VCAP_SERVICES is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DestinationServiceError("VCAP_SERVICES must decode to a JSON object.")
    return parsed


def find_destination_binding(
    vcap_services: dict[str, Any],
    destination_service_name: str | None,
) -> dict[str, Any]:
    """Find the bound SAP BTP Destination service instance.

    Args:
        vcap_services: Parsed VCAP_SERVICES object.
        destination_service_name: Optional CF service instance name to select.

    Returns:
        The matching VCAP service binding object.

    Raises:
        DestinationServiceError: If no matching destination binding is present.
    """
    candidates: list[dict[str, Any]] = []
    destination_bindings: list[dict[str, Any]] = []
    for service_label, instances in vcap_services.items():
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            label = str(instance.get("label", ""))
            tags = instance.get("tags", [])
            name = str(instance.get("name", ""))
            has_destination_tag = isinstance(tags, list) and "destination" in tags
            is_destination = service_label == "destination" or label == "destination" or has_destination_tag
            if not is_destination:
                continue
            destination_bindings.append(instance)
            if destination_service_name and name != destination_service_name:
                continue
            candidates.append(instance)

    if not candidates:
        if destination_service_name and len(destination_bindings) == 1:
            return destination_bindings[0]
        if destination_service_name:
            raise DestinationServiceError(
                "No bound Destination service instance named "
                f"{destination_service_name!r} was found in VCAP_SERVICES."
            )
        raise DestinationServiceError("No bound Destination service instance was found in VCAP_SERVICES.")
    if len(candidates) > 1 and not destination_service_name:
        names = ", ".join(str(candidate.get("name", "<unnamed>")) for candidate in candidates)
        raise DestinationServiceError(
            "Multiple Destination service bindings were found. "
            f"Set DESTINATION_SERVICE_NAME to one of: {names}."
        )
    return candidates[0]


def extract_destination_property(destination_config: dict[str, Any], property_name: str) -> str:
    """Extract a destination Additional Property by name.

    Args:
        destination_config: Destination configuration returned by the v1 API.
        property_name: Additional Property key to read.

    Returns:
        The configured property value.

    Raises:
        DestinationServiceError: If the property is missing or empty.
    """
    candidates: list[Any] = [
        destination_config.get(property_name),
        destination_config.get("properties", {}).get(property_name)
        if isinstance(destination_config.get("properties"), dict)
        else None,
        destination_config.get("Properties", {}).get(property_name)
        if isinstance(destination_config.get("Properties"), dict)
        else None,
    ]
    additional_properties = destination_config.get("additionalProperties")
    if isinstance(additional_properties, list):
        for item in additional_properties:
            if isinstance(item, dict) and item.get("key") == property_name:
                candidates.append(item.get("value"))

    for candidate in candidates:
        if candidate:
            return str(candidate)

    destination_name = destination_config.get("Name") or destination_config.get("name") or "<unknown>"
    raise DestinationServiceError(
        f"Destination {destination_name!r} is missing Additional Property {property_name!r}."
    )


def destination_level_to_v1_collection(destination_level: str) -> str:
    """Map a destination level to the v1 Destination service collection name.

    Args:
        destination_level: Configured level such as "subaccount" or "instance".

    Returns:
        Destination service v1 collection segment.

    Raises:
        DestinationServiceError: If the level is unsupported.
    """
    normalized = destination_level.strip().lower().replace("-", "_")
    if normalized == "subaccount":
        return "subaccountDestinations"
    if normalized in {"instance", "service_instance"}:
        return "instanceDestinations"
    raise DestinationServiceError(
        "GMAIL_DESTINATION_LEVEL must be either 'subaccount' or 'instance'."
    )


def destination_level_to_v2_suffix(destination_level: str) -> str:
    """Map a destination level to the v2 Destination service lookup suffix.

    Args:
        destination_level: Configured level such as "subaccount" or "instance".

    Returns:
        Destination service v2 level suffix.

    Raises:
        DestinationServiceError: If the level is unsupported.
    """
    normalized = destination_level.strip().lower().replace("-", "_")
    if normalized == "subaccount":
        return "subaccount"
    if normalized in {"instance", "service_instance"}:
        return "instance"
    raise DestinationServiceError(
        "GMAIL_DESTINATION_LEVEL must be either 'subaccount' or 'instance'."
    )


def normalize_auth_header(auth_token: dict[str, Any]) -> dict[str, str]:
    """Normalize a Destination service auth token into HTTP headers.

    Args:
        auth_token: First auth token from the v2 destination consumption response.

    Returns:
        Prepared HTTP headers for the target Gmail request.

    Raises:
        DestinationServiceError: If the response does not contain a usable header.
    """
    http_header = auth_token.get("http_header") or auth_token.get("httpHeader")
    if isinstance(http_header, dict):
        if "key" in http_header and "value" in http_header:
            return {str(http_header["key"]): str(http_header["value"])}
        if http_header:
            return {str(key): str(value) for key, value in http_header.items()}
    raise DestinationServiceError("Destination runtime response did not include a usable auth HTTP header.")


def raise_for_auth_token_error(auth_token: dict[str, Any], destination_name: str, property_name: str) -> None:
    """Raise a clear error when Destination service returns token retrieval failure.

    Args:
        auth_token: First auth token from the v2 destination consumption response.
        destination_name: Destination name used in the runtime lookup.
        property_name: Additional Property that stores the Google refresh token.

    Raises:
        DestinationAuthenticationError: If the token contains a Google refresh-token error.
    """
    error = auth_token.get("error")
    if not error:
        return
    raise DestinationAuthenticationError(
        "The Google refresh token configured on the Gmail destination was rejected. "
        "Please re-authorize Gmail with google_tmp/google_fetch.py and replace the "
        f"{property_name} Additional Property on {destination_name}. "
        f"Destination service error: {error}"
    )


def extract_error_message(response: requests.Response) -> str:
    """Extract a short, non-secret error message from an HTTP response.

    Args:
        response: HTTP response returned by requests.

    Returns:
        Best-effort error text suitable for exception messages.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict):
        detail = payload.get("error_description") or payload.get("message") or payload.get("error")
        if detail:
            return str(detail)
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("status") or error)
    return str(payload)[:500]


def response_has_invalid_grant(response: requests.Response) -> bool:
    """Detect Google invalid_grant errors returned through Destination service.

    Args:
        response: HTTP response returned by requests.

    Returns:
        True when the response indicates the Google refresh token is invalid.
    """
    try:
        payload = response.json()
    except ValueError:
        return "invalid_grant" in response.text
    return "invalid_grant" in json.dumps(payload)


class BtpDestinationClient:
    """Resolve Gmail runtime auth through the SAP BTP Destination service."""

    def __init__(
        self,
        destination_name: str,
        destination_level: str,
        refresh_token_property: str,
        destination_service_name: str | None,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the client with destination lookup settings.

        Args:
            destination_name: BTP destination name to resolve.
            destination_level: Destination lookup level.
            refresh_token_property: Additional Property that stores the Google refresh token.
            destination_service_name: Bound Destination service instance name.
            session: Optional requests-compatible session for HTTP calls.
            now: Optional UTC clock provider for tests.
        """
        self.destination_name = destination_name
        self.destination_level = destination_level
        self.refresh_token_property = refresh_token_property
        self.destination_service_name = destination_service_name
        self.session = session or requests.Session()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._cached_runtime_destination: RuntimeDestination | None = None

    @classmethod
    def from_settings(cls, settings: PriceChangeSettings) -> "BtpDestinationClient":
        """Create a BTP destination client from application settings.

        Args:
            settings: Price-change runtime settings.

        Returns:
            Configured BtpDestinationClient.
        """
        return cls(
            destination_name=settings.gmail_destination_name,
            destination_level=settings.gmail_destination_level,
            refresh_token_property=settings.gmail_refresh_token_property,
            destination_service_name=settings.destination_service_name,
        )

    def clear_cache(self) -> None:
        """Clear the cached Gmail runtime auth header."""
        self._cached_runtime_destination = None

    def resolve_runtime_destination(self) -> RuntimeDestination:
        """Resolve and cache Gmail runtime destination data.

        Returns:
            RuntimeDestination with Gmail base URL and prepared auth headers.

        Raises:
            DestinationServiceError: If destination resolution fails.
            DestinationAuthenticationError: If Google rejects the refresh token.
        """
        cached = self._cached_runtime_destination
        if cached and cached.expires_at and cached.expires_at > self.now():
            return cached

        vcap = load_vcap_services()
        binding = find_destination_binding(vcap, self.destination_service_name)
        service_token = self._fetch_destination_service_token(binding)
        destination_config = self._fetch_destination_config(binding, service_token)
        refresh_token = extract_destination_property(destination_config, self.refresh_token_property)
        runtime_destination = self._consume_runtime_destination(binding, service_token, refresh_token)
        self._cached_runtime_destination = runtime_destination
        return runtime_destination

    def _fetch_destination_service_token(self, binding: dict[str, Any]) -> str:
        """Fetch a client-credentials token for the Destination service.

        Args:
            binding: Destination service binding from VCAP_SERVICES.

        Returns:
            Bearer token for the Destination service REST API.
        """
        credentials = self._binding_credentials(binding)
        token_url = f"{credentials['url'].rstrip('/')}/oauth/token"
        response = self.session.request(
            "POST",
            token_url,
            data={"grant_type": "client_credentials"},
            auth=(credentials["clientid"], credentials["clientsecret"]),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._raise_for_destination_error(response, "fetching Destination service token")
        payload = response.json()
        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        if not access_token:
            raise DestinationServiceError("Destination service token response did not include access_token.")
        return str(access_token)

    def _fetch_destination_config(self, binding: dict[str, Any], service_token: str) -> dict[str, Any]:
        """Read destination configuration to obtain the stored refresh token.

        Args:
            binding: Destination service binding from VCAP_SERVICES.
            service_token: Bearer token for the Destination service API.

        Returns:
            Destination configuration object.
        """
        credentials = self._binding_credentials(binding)
        collection = destination_level_to_v1_collection(self.destination_level)
        url = (
            f"{credentials['uri'].rstrip('/')}/destination-configuration/v1/"
            f"{collection}/{self.destination_name}"
        )
        response = self.session.request(
            "GET",
            url,
            headers={
                "Authorization": f"Bearer {service_token}",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._raise_for_destination_error(response, "reading Gmail destination configuration")
        payload = response.json()
        destination_config = payload.get("destinationConfiguration", payload)
        if not isinstance(destination_config, dict):
            raise DestinationServiceError("Destination configuration response did not contain an object.")
        return destination_config

    def _consume_runtime_destination(
        self,
        binding: dict[str, Any],
        service_token: str,
        refresh_token: str,
    ) -> RuntimeDestination:
        """Consume the destination with X-refresh-token to obtain Gmail auth.

        Args:
            binding: Destination service binding from VCAP_SERVICES.
            service_token: Bearer token for the Destination service API.
            refresh_token: Google refresh token from the destination Additional Property.

        Returns:
            RuntimeDestination for Gmail API calls.
        """
        credentials = self._binding_credentials(binding)
        level_suffix = destination_level_to_v2_suffix(self.destination_level)
        url = (
            f"{credentials['uri'].rstrip('/')}/destination-configuration/v2/"
            f"destinations/{self.destination_name}@{level_suffix}"
        )
        response = self.session.request(
            "GET",
            url,
            headers={
                "Authorization": f"Bearer {service_token}",
                "Accept": "application/json",
                "X-refresh-token": refresh_token,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._raise_for_destination_error(response, "resolving Gmail runtime destination")
        payload = response.json()
        if not isinstance(payload, dict):
            raise DestinationServiceError("Destination runtime response did not contain an object.")
        destination_config = payload.get("destinationConfiguration")
        auth_tokens = payload.get("authTokens")
        if not isinstance(destination_config, dict):
            raise DestinationServiceError("Destination runtime response is missing destinationConfiguration.")
        if not isinstance(auth_tokens, list) or not auth_tokens or not isinstance(auth_tokens[0], dict):
            raise DestinationServiceError("Destination runtime response is missing authTokens.")
        raise_for_auth_token_error(auth_tokens[0], self.destination_name, self.refresh_token_property)
        base_url = destination_config.get("URL") or destination_config.get("url")
        if not base_url:
            raise DestinationServiceError("Destination runtime response did not include a target URL.")
        return RuntimeDestination(
            url=str(base_url),
            headers=normalize_auth_header(auth_tokens[0]),
            expires_at=self._calculate_runtime_expiry(auth_tokens[0]),
        )

    def _calculate_runtime_expiry(self, auth_token: dict[str, Any]) -> datetime:
        """Calculate when cached Gmail runtime auth should be refreshed.

        Args:
            auth_token: First auth token from the v2 destination consumption response.

        Returns:
            UTC datetime when the cached runtime token expires.
        """
        raw_expires_in = auth_token.get("expires_in") or auth_token.get("expiresIn")
        try:
            expires_in = int(raw_expires_in) if raw_expires_in is not None else DEFAULT_RUNTIME_TOKEN_TTL_SECONDS
        except (TypeError, ValueError):
            expires_in = DEFAULT_RUNTIME_TOKEN_TTL_SECONDS
        cache_seconds = max(1, expires_in - TOKEN_EXPIRY_SAFETY_SECONDS)
        return self.now() + timedelta(seconds=cache_seconds)

    def _binding_credentials(self, binding: dict[str, Any]) -> dict[str, str]:
        """Validate and return Destination service binding credentials.

        Args:
            binding: Destination service binding from VCAP_SERVICES.

        Returns:
            Binding credentials with required string fields.
        """
        credentials = binding.get("credentials")
        if not isinstance(credentials, dict):
            raise DestinationServiceError("Destination service binding has no credentials object.")
        required = ["clientid", "clientsecret", "url", "uri"]
        missing = [name for name in required if not credentials.get(name)]
        if missing:
            raise DestinationServiceError(
                "Destination service binding is missing credential fields: " + ", ".join(missing)
            )
        return {name: str(credentials[name]) for name in required}

    def _raise_for_destination_error(self, response: requests.Response, action: str) -> None:
        """Raise an application-specific error for failed Destination responses.

        Args:
            response: HTTP response returned by requests.
            action: Short description of the Destination service operation.
        """
        if response.status_code < 400:
            return
        if response_has_invalid_grant(response):
            raise DestinationAuthenticationError(
                "The Google refresh token configured on the Gmail destination was rejected. "
                "Please re-authorize Gmail with google_tmp/google_fetch.py and replace the "
                f"{self.refresh_token_property} Additional Property on {self.destination_name}."
            )
        raise DestinationServiceError(
            f"Destination service failed while {action}: HTTP {response.status_code}: "
            f"{extract_error_message(response)}"
        )
