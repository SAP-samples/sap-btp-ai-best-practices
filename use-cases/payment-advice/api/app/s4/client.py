"""Small S/4HANA HTTP client with direct and BTP connectivity modes.

Adapted from the generic part of an earlier S/4 lookup module.

Connectivity mode is chosen by `S4_CONNECTIVITY_MODE`:

- `direct`: `S4_BASE_URL` + `S4_USERNAME` + `S4_PASSWORD` (local development
  over the SAP VPN; the S/4 host is not resolvable without it).
- `btp`: Destination + Connectivity service variables (see
  `app.s4.connectivity`); only works inside Cloud Foundry because the
  Connectivity proxy host is internal.
- `auto` (default): btp inside Cloud Foundry when the BTP variables are
  complete, otherwise direct when complete, otherwise btp when complete.

OData writes (POST/PATCH) first fetch an `X-CSRF-Token`; the shared
`requests.Session` keeps the session cookies the token is bound to.

Example:
    client = S4Client(load_s4_config())
    rows = odata_rows(client.get_json("API_BUSINESS_PARTNER", "/A_BusinessPartner", {"$top": "1"}))
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import urllib3
from dotenv import load_dotenv

from .connectivity import (
    S4BtpConnectivityClient,
    S4RuntimeContext,
    load_s4_btp_connectivity_config_from_env,
)

ODATA_ROOT = "/sap/opu/odata/sap/"
TIMEOUT_SECONDS = 120
API_DIR = Path(__file__).resolve().parents[2]


class S4ConfigError(RuntimeError):
    """Raised when S/4 connection settings are incomplete or invalid."""


class S4HTTPError(RuntimeError):
    """Raised for a failed S/4 HTTP call; keeps the body for error details.

    Args:
        method: HTTP method.
        url: Requested URL (no credentials).
        status_code: HTTP status, or 0 for connection-level failures.
        message: HTTP reason or connection error text.
        body: Optional response body.
    """

    def __init__(self, method: str, url: str, status_code: int, message: str, body: str | None = None) -> None:
        super().__init__(f"{method} {url} failed with HTTP {status_code}: {message}")
        self.method, self.url, self.status_code, self.message, self.body = method, url, status_code, message, body

    @property
    def detail(self) -> str:
        """Return the most specific S/4 error text available (OData message or HTTP reason)."""
        return odata_error_message(self.body) or self.message


@dataclass(frozen=True)
class S4Config:
    """Runtime S/4 settings.

    Args:
        base_url: S/4 base URL for direct mode (empty in btp mode).
        client: SAP client number sent as `sap-client`, if any.
        verify: TLS certificate verification.
        username: Basic-auth user for direct mode.
        password: Basic-auth password for direct mode.
        runtime_context_provider: Token-backed context provider for btp mode.
        mode: Resolved connectivity mode, `direct` or `btp` (for diagnostics).
    """

    base_url: str
    client: str | None
    verify: bool
    username: str | None = None
    password: str | None = None
    runtime_context_provider: Callable[[], S4RuntimeContext] | None = None
    mode: str = "direct"


def is_cloud_foundry_runtime() -> bool:
    """Return True when running inside a Cloud Foundry container."""
    return bool(os.getenv("VCAP_APPLICATION") or os.getenv("CF_INSTANCE_GUID"))


def normalize_connectivity_mode(value: str | None) -> str:
    """Normalize `S4_CONNECTIVITY_MODE` to `auto`, `direct` or `btp`.

    Raises:
        S4ConfigError: For unsupported values.
    """
    normalized = (value or "auto").strip().lower().replace("_", "-")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized == "direct":
        return "direct"
    if normalized in {"btp", "destination", "btp-destination"}:
        return "btp"
    raise S4ConfigError("S4_CONNECTIVITY_MODE must be one of: auto, direct, btp.")


def parse_bool(value: str | None, default: bool = True) -> bool:
    """Parse an environment boolean; missing or empty returns `default`.

    Raises:
        S4ConfigError: For values that are not recognisable booleans.
    """
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise S4ConfigError(f"Invalid boolean value: {value!r}")


def direct_config_from_env() -> S4Config:
    """Build direct (basic-auth) settings from `S4_BASE_URL`, `S4_USERNAME`, `S4_PASSWORD`.

    Raises:
        S4ConfigError: When a required variable is missing.
    """
    required = {name: os.getenv(name, "").strip() for name in ("S4_BASE_URL", "S4_USERNAME", "S4_PASSWORD")}
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise S4ConfigError(f"Missing required S/4 environment variables: {', '.join(missing)}")
    return S4Config(
        base_url=required["S4_BASE_URL"].rstrip("/"),
        client=os.getenv("S4_CLIENT", "").strip() or None,
        verify=parse_bool(os.getenv("S4_VERIFY"), default=True),
        username=required["S4_USERNAME"],
        password=required["S4_PASSWORD"],
        mode="direct",
    )


def btp_config_from_env() -> S4Config:
    """Build Destination/Connectivity-backed settings from the BTP environment variables.

    Raises:
        S4ConfigError: When the BTP variables are incomplete.
    """
    btp = load_s4_btp_connectivity_config_from_env()
    if btp is None:
        raise S4ConfigError("S4_CONNECTIVITY_MODE=btp requires complete Destination and Connectivity variables.")
    return S4Config(
        base_url="",
        client=btp.fallback_sap_client,
        verify=btp.verify,
        runtime_context_provider=S4BtpConnectivityClient(btp).resolve_runtime_context,
        mode="btp",
    )


def load_s4_config(env_file: str | Path | None = None) -> S4Config:
    """Load S/4 settings, honouring `S4_CONNECTIVITY_MODE` (see module docstring).

    Args:
        env_file: Optional explicit `.env`; defaults to `api/.env`. Existing
            process variables always win.

    Returns:
        Resolved S4Config.
    """
    load_dotenv(env_file or API_DIR / ".env", override=False)
    mode = normalize_connectivity_mode(os.getenv("S4_CONNECTIVITY_MODE"))
    if mode == "direct":
        return direct_config_from_env()
    if mode == "btp":
        return btp_config_from_env()
    has_btp = load_s4_btp_connectivity_config_from_env() is not None
    if has_btp and is_cloud_foundry_runtime():
        return btp_config_from_env()
    if all(os.getenv(name) for name in ("S4_BASE_URL", "S4_USERNAME", "S4_PASSWORD")):
        return direct_config_from_env()
    return btp_config_from_env() if has_btp else direct_config_from_env()


def odata_str(value: str) -> str:
    """Quote a string as an OData V2 literal, escaping single quotes."""
    return "'" + str(value).replace("'", "''") + "'"


def odata_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the `d.results` rows of an OData V2 collection response."""
    return list((payload.get("d") or {}).get("results") or [])


def odata_entity(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the `d` object of an OData V2 single-entity response ({} when absent)."""
    data = payload.get("d")
    return data if isinstance(data, dict) else {}


def odata_error_message(body: str | None) -> str | None:
    """Extract `error.message.value` (plus inner detail messages) from an OData V2 error body."""
    try:
        error = json.loads(body or "").get("error") or {}
    except (ValueError, AttributeError):
        return None
    message = (error.get("message") or {}).get("value")
    details = [d.get("message") for d in (error.get("innererror") or {}).get("errordetails") or [] if d.get("message")]
    extra = [d for d in details if d != message]
    return "; ".join([message, *extra]) if message else ("; ".join(extra) or None)


class S4Client:
    """HTTP client for S/4 OData services and plain ICF paths (e.g. SOAP).

    Args:
        config: Resolved S/4 settings.
        session: Optional requests session (tests inject a fake).
    """

    def __init__(self, config: S4Config, session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        if config.username and config.password:
            self.session.auth = (config.username, config.password)
        self.session.headers.update({"Accept": "application/json"})
        if not config.verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def context(self) -> S4RuntimeContext:
        """Return the current request context (cached BTP tokens or static direct settings)."""
        if self.config.runtime_context_provider is not None:
            return self.config.runtime_context_provider()
        return S4RuntimeContext(self.config.base_url, self.config.client, {}, None, self.config.verify)

    def _send(self, method: str, path: str, *, params: dict[str, Any] | None = None,
              headers: dict[str, str] | None = None, timeout: int = TIMEOUT_SECONDS, **kwargs: Any) -> requests.Response:
        """Send one request to `<base_url><path>` and raise S4HTTPError on failure.

        Args:
            method: HTTP method.
            path: Absolute path on the S/4 host, starting with `/`.
            params: Query parameters; `sap-client` is added when configured.
            headers: Extra headers, merged over the context (auth/proxy) headers.
            timeout: Seconds before the request is abandoned.
            kwargs: Passed to `requests` (`json`, `data`).

        Returns:
            The successful response.
        """
        context = self.context()
        url = f"{context.base_url}{path}"
        query = dict(params or {})
        if context.client:
            query.setdefault("sap-client", context.client)
        try:
            response = self.session.request(method, url, params=query, headers={**context.headers, **(headers or {})},
                                            timeout=timeout, proxies=context.proxies, verify=context.verify, **kwargs)
        except requests.RequestException as exc:
            raise S4HTTPError(method, url, 0, str(exc)) from exc
        if response.status_code >= 400:
            raise S4HTTPError(method, url, response.status_code, response.reason or "", response.text)
        return response

    def get_json(self, service: str, path: str = "", params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET an OData V2 JSON response from `/sap/opu/odata/sap/<service><path>`."""
        response = self._send("GET", f"{ODATA_ROOT}{service}{path}", params=params)
        return json.loads(response.text) if response.text else {}

    def get_text(self, service: str, path: str = "", params: dict[str, Any] | None = None) -> str:
        """GET a text body (e.g. `$metadata` XML) from an OData service."""
        return self._send("GET", f"{ODATA_ROOT}{service}{path}", params=params, headers={"Accept": "*/*"}).text

    def fetch_csrf_token(self, service: str) -> str:
        """Fetch the CSRF token that OData writes on `service` must echo back.

        Raises:
            S4HTTPError: When S/4 does not return a token.
        """
        response = self._send("GET", f"{ODATA_ROOT}{service}/", headers={"X-CSRF-Token": "Fetch"}, timeout=60)
        token = response.headers.get("X-CSRF-Token") or response.headers.get("x-csrf-token")
        if not token:
            raise S4HTTPError("GET", f"{ODATA_ROOT}{service}/", 0, "Missing CSRF token in response headers")
        return token

    def _write(self, method: str, service: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a CSRF-protected JSON write and return the parsed response body ({} when empty)."""
        headers = {"X-CSRF-Token": self.fetch_csrf_token(service), "Content-Type": "application/json"}
        response = self._send(method, f"{ODATA_ROOT}{service}{path}", headers=headers, json=payload)
        return json.loads(response.text) if response.text else {}

    def post_json(self, service: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST (create / deep insert) a JSON entity to an OData service."""
        return self._write("POST", service, path, payload)

    def patch_json(self, service: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """PATCH (partial update) an OData entity."""
        return self._write("PATCH", service, path, payload)

    def request_path(self, method: str, path: str, body: str | bytes | None = None,
                     headers: dict[str, str] | None = None, params: dict[str, Any] | None = None) -> requests.Response:
        """Send a raw request to any ICF path (used for the Journal Entry SOAP service)."""
        return self._send(method, path, params=params, headers=headers, data=body)
