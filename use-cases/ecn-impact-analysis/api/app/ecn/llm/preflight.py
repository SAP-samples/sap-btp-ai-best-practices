"""Safe, explicit SAP AI Core OAuth and model-deployment preflight checks."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol

import requests
from pydantic import BaseModel, ConfigDict

from app.ecn.settings import DEFAULT_LLM_MODEL


class ProviderAvailability(StrEnum):
    """Describe one credential or model capability without technical payloads."""

    NOT_CHECKED = "NOT_CHECKED"
    AVAILABLE = "AVAILABLE"
    CONFIGURATION_MISSING = "CONFIGURATION_MISSING"
    AUTH_FAILED = "AUTH_FAILED"
    FORBIDDEN = "FORBIDDEN"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class CapabilityStatus(BaseModel):
    """Expose the safe status of one configured AI Core capability.

    Args:
        capability: Stable role such as ``text_generation``.
        model: Exact configured model name; never a URL or credential.
        status: Current safe availability classification.
        http_status: Provider HTTP status when one was received.
        provider_code: Optional bounded machine code from the provider.
        request_id: Optional bounded request identifier for support correlation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str
    model: str
    status: ProviderAvailability
    http_status: int | None = None
    provider_code: str | None = None
    request_id: str | None = None


class OAuthStatus(BaseModel):
    """Expose safe OAuth preflight state without endpoint or token details."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProviderAvailability
    http_status: int | None = None
    provider_code: str | None = None
    request_id: str | None = None


class ProviderStatusSnapshot(BaseModel):
    """Return one cached or freshly checked provider status snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checked_at: datetime | None
    oauth: OAuthStatus
    models: tuple[CapabilityStatus, ...]


@dataclass(frozen=True)
class AiCorePreflightConfig:
    """Hold AI Core configuration while suppressing sensitive values from repr.

    Args:
        auth_url: OAuth token endpoint.
        client_id: OAuth client identifier.
        client_secret: OAuth client secret.
        base_url: AI Core API base URL.
        resource_group: AI Core resource group header value.
        llm_model: Exact text-generation model name.
    """

    auth_url: str = field(repr=False)
    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    base_url: str = field(repr=False)
    resource_group: str = field(repr=False)
    llm_model: str = DEFAULT_LLM_MODEL

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "AiCorePreflightConfig":
        """Load only required AI Core fields from an environment mapping."""
        source = os.environ if environ is None else environ
        return cls(
            auth_url=source.get("AICORE_AUTH_URL", "").strip(),
            client_id=source.get("AICORE_CLIENT_ID", "").strip(),
            client_secret=source.get("AICORE_CLIENT_SECRET", "").strip(),
            base_url=source.get("AICORE_BASE_URL", "").strip().rstrip("/"),
            resource_group=source.get("AICORE_RESOURCE_GROUP", "").strip(),
            llm_model=source.get("LLM_MODEL", DEFAULT_LLM_MODEL).strip()
            or DEFAULT_LLM_MODEL,
        )

    @property
    def complete(self) -> bool:
        """Return whether every credential and routing field is nonblank."""
        return all(
            (
                self.auth_url,
                self.client_id,
                self.client_secret,
                self.base_url,
                self.resource_group,
            )
        )


class HttpSession(Protocol):
    """Define the synchronous HTTP surface used by preflight."""

    def post(self, url: str, **kwargs: object) -> object:
        """Issue one OAuth POST request."""

    def get(self, url: str, **kwargs: object) -> object:
        """Issue one AI Core discovery GET request."""


def _bounded(value: object, maximum: int = 200) -> str | None:
    """Return a trimmed bounded provider metadata value."""
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered[:maximum] if rendered else None


def _response_payload(response: object) -> Mapping[str, object]:
    """Return a JSON mapping and ignore malformed or non-object bodies."""
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - provider bodies are never surfaced.
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _safe_response_metadata(response: object) -> tuple[int | None, str | None, str | None]:
    """Extract only HTTP status, machine code, and request ID."""
    status_raw = getattr(response, "status_code", None)
    try:
        status = int(status_raw) if status_raw is not None else None
    except (TypeError, ValueError):
        status = None
    payload = _response_payload(response)
    error = payload.get("error")
    provider_code = (
        _bounded(error.get("code"), 120)
        if isinstance(error, Mapping)
        else _bounded(payload.get("code"), 120)
    )
    raw_headers = getattr(response, "headers", {})
    headers = raw_headers if isinstance(raw_headers, Mapping) else {}
    request_id = next(
        (
            _bounded(value)
            for key, value in headers.items()
            if str(key).casefold() in {"x-request-id", "x-correlation-id"}
        ),
        None,
    )
    return status, provider_code, request_id


def _availability_for_status(status: int | None) -> ProviderAvailability:
    """Map a provider HTTP status to a bounded availability state."""
    if status == 401:
        return ProviderAvailability.AUTH_FAILED
    if status == 403:
        return ProviderAvailability.FORBIDDEN
    if status == 429:
        return ProviderAvailability.RATE_LIMITED
    if status is not None and 200 <= status <= 299:
        return ProviderAvailability.AVAILABLE
    return ProviderAvailability.PROVIDER_ERROR


def _resources(payload: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    """Return mapping resources from a standard AI Core list response."""
    values = payload.get("resources", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(item for item in values if isinstance(item, Mapping))


def _contains_exact_value(value: object, expected: str) -> bool:
    """Find an exact case-insensitive scalar value inside a JSON-like object."""
    if isinstance(value, Mapping):
        return any(_contains_exact_value(item, expected) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_exact_value(item, expected) for item in value)
    return isinstance(value, str) and value.strip().casefold() == expected.casefold()


def _oauth_token_url(auth_url: str) -> str:
    """Return a service-key OAuth URL with the token path appended once."""
    normalized = auth_url.rstrip("/")
    return (
        normalized
        if normalized.endswith("/oauth/token")
        else f"{normalized}/oauth/token"
    )


def _lm_url(base_url: str, resource: str) -> str:
    """Return an AI Core LM collection URL without duplicating the v2 segment."""
    normalized = base_url.rstrip("/")
    api_root = normalized if normalized.endswith("/v2") else f"{normalized}/v2"
    return f"{api_root}/lm/{resource}"


class ProviderStatusService:
    """Run explicit provider preflight and retain only its latest safe snapshot."""

    def __init__(
        self,
        config: AiCorePreflightConfig,
        *,
        session: HttpSession | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        """Store credentials privately and prepare an initial NOT_CHECKED state."""
        self._config = config
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds
        self._status = self._empty_snapshot()

    def _model_specs(self) -> tuple[tuple[str, str], ...]:
        """Return the fixed capability-to-configured-model mapping.

        Only the shared text-generation model is checked, because it is the
        only AI Core model deployment the application calls.
        """
        return (("text_generation", self._config.llm_model),)

    def _empty_snapshot(self) -> ProviderStatusSnapshot:
        """Build the safe state used before any explicit network check."""
        return ProviderStatusSnapshot(
            checked_at=None,
            oauth=OAuthStatus(status=ProviderAvailability.NOT_CHECKED),
            models=tuple(
                CapabilityStatus(
                    capability=capability,
                    model=model,
                    status=ProviderAvailability.NOT_CHECKED,
                )
                for capability, model in self._model_specs()
            ),
        )

    def get_status(self) -> ProviderStatusSnapshot:
        """Return the latest safe snapshot without issuing a provider request."""
        return self._status

    def generation_blocker(self) -> str | None:
        """Return a safe text-generation blocker code from the cached snapshot.

        ``NOT_CHECKED`` intentionally permits scheduling so callers that do not
        opt into startup preflight retain historical behavior. Once a check has
        produced a failure, jobs can be held without repeating provider calls.
        """
        status_to_code = {
            ProviderAvailability.CONFIGURATION_MISSING: "PROVIDER_AUTH_FAILED",
            ProviderAvailability.AUTH_FAILED: "PROVIDER_AUTH_FAILED",
            ProviderAvailability.FORBIDDEN: "PROVIDER_FORBIDDEN",
            ProviderAvailability.NOT_AVAILABLE: "MODEL_NOT_AVAILABLE",
            ProviderAvailability.RATE_LIMITED: "PROVIDER_RATE_LIMITED",
            ProviderAvailability.TIMEOUT: "PROVIDER_TIMEOUT",
            ProviderAvailability.PROVIDER_ERROR: "PROVIDER_SERVER_ERROR",
        }
        if self._status.oauth.status not in {
            ProviderAvailability.NOT_CHECKED,
            ProviderAvailability.AVAILABLE,
        }:
            return status_to_code[self._status.oauth.status]
        text_generation = next(
            (
                model
                for model in self._status.models
                if model.capability == "text_generation"
            ),
            None,
        )
        if text_generation is None or text_generation.status in {
            ProviderAvailability.NOT_CHECKED,
            ProviderAvailability.AVAILABLE,
        }:
            return None
        return status_to_code[text_generation.status]

    def check(self) -> ProviderStatusSnapshot:
        """Check OAuth, configurations, and running deployments synchronously."""
        checked_at = datetime.now(timezone.utc)
        if not self._config.complete:
            self._status = ProviderStatusSnapshot(
                checked_at=checked_at,
                oauth=OAuthStatus(status=ProviderAvailability.CONFIGURATION_MISSING),
                models=tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=ProviderAvailability.NOT_CHECKED,
                    )
                    for capability, model in self._model_specs()
                ),
            )
            return self._status
        try:
            oauth_response = self._session.post(
                _oauth_token_url(self._config.auth_url),
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._config.client_id,
                    "client_secret": self._config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self._timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - classify without exception text.
            status = (
                ProviderAvailability.TIMEOUT
                if "timeout" in type(error).__name__.casefold()
                else ProviderAvailability.PROVIDER_ERROR
            )
            self._status = ProviderStatusSnapshot(
                checked_at=checked_at,
                oauth=OAuthStatus(status=status),
                models=tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=ProviderAvailability.NOT_CHECKED,
                    )
                    for capability, model in self._model_specs()
                ),
            )
            return self._status
        http_status, provider_code, request_id = _safe_response_metadata(oauth_response)
        oauth_availability = _availability_for_status(http_status)
        if oauth_availability is not ProviderAvailability.AVAILABLE:
            self._status = ProviderStatusSnapshot(
                checked_at=checked_at,
                oauth=OAuthStatus(
                    status=oauth_availability,
                    http_status=http_status,
                    provider_code=provider_code,
                    request_id=request_id,
                ),
                models=tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=ProviderAvailability.NOT_CHECKED,
                    )
                    for capability, model in self._model_specs()
                ),
            )
            return self._status
        token = _bounded(_response_payload(oauth_response).get("access_token"), 20_000)
        if token is None:
            oauth = OAuthStatus(status=ProviderAvailability.PROVIDER_ERROR, http_status=http_status)
            self._status = ProviderStatusSnapshot(
                checked_at=checked_at,
                oauth=oauth,
                models=tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=ProviderAvailability.NOT_CHECKED,
                    )
                    for capability, model in self._model_specs()
                ),
            )
            return self._status
        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": self._config.resource_group,
        }
        try:
            configurations_response = self._session.get(
                _lm_url(self._config.base_url, "configurations"),
                headers=headers,
                timeout=self._timeout_seconds,
            )
            deployments_response = self._session.get(
                _lm_url(self._config.base_url, "deployments"),
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - never surface endpoint or token.
            availability = (
                ProviderAvailability.TIMEOUT
                if "timeout" in type(error).__name__.casefold()
                else ProviderAvailability.PROVIDER_ERROR
            )
            model_statuses = tuple(
                CapabilityStatus(
                    capability=capability,
                    model=model,
                    status=availability,
                )
                for capability, model in self._model_specs()
            )
        else:
            config_status, config_code, config_request = _safe_response_metadata(
                configurations_response
            )
            deployment_status, deployment_code, deployment_request = _safe_response_metadata(
                deployments_response
            )
            discovery_status = _availability_for_status(config_status)
            if discovery_status is ProviderAvailability.AVAILABLE:
                discovery_status = _availability_for_status(deployment_status)
            if discovery_status is not ProviderAvailability.AVAILABLE:
                model_statuses = tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=discovery_status,
                        http_status=(
                            config_status
                            if _availability_for_status(config_status)
                            is not ProviderAvailability.AVAILABLE
                            else deployment_status
                        ),
                        provider_code=config_code or deployment_code,
                        request_id=config_request or deployment_request,
                    )
                    for capability, model in self._model_specs()
                )
            else:
                configurations = _resources(_response_payload(configurations_response))
                deployments = _resources(_response_payload(deployments_response))
                config_ids_by_model: dict[str, set[str]] = {}
                for _, model in self._model_specs():
                    config_ids_by_model[model] = {
                        str(configuration.get("id"))
                        for configuration in configurations
                        if configuration.get("id") is not None
                        and _contains_exact_value(configuration, model)
                    }
                running_config_ids = {
                    str(deployment.get("configurationId"))
                    for deployment in deployments
                    if str(deployment.get("status", "")).upper() == "RUNNING"
                    and deployment.get("configurationId") is not None
                }
                model_statuses = tuple(
                    CapabilityStatus(
                        capability=capability,
                        model=model,
                        status=(
                            ProviderAvailability.AVAILABLE
                            if config_ids_by_model[model] & running_config_ids
                            else ProviderAvailability.NOT_AVAILABLE
                        ),
                    )
                    for capability, model in self._model_specs()
                )
        self._status = ProviderStatusSnapshot(
            checked_at=checked_at,
            oauth=OAuthStatus(
                status=ProviderAvailability.AVAILABLE,
                http_status=http_status,
                request_id=request_id,
            ),
            models=model_statuses,
        )
        return self._status
