from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
import urllib3

from .s4_btp_connectivity import (
    S4BtpConnectivityClient,
    S4RuntimeContext,
    load_s4_btp_connectivity_config_from_env,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is present in api/requirements.txt.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        """Fallback no-op dotenv loader when python-dotenv is absent."""
        return False


BUSINESS_PARTNER_SERVICE_NAME = "API_BUSINESS_PARTNER"
INFORECORD_SERVICE_NAME = "API_INFORECORD_PROCESS_SRV"
PRODUCT_SERVICE_NAME = "API_PRODUCT_SRV"
DEFAULT_EXTERNAL_MATERIAL_PREFIX = "A"


class S4ConfigError(RuntimeError):
    """Raised when S/4 runtime configuration is incomplete or invalid."""


class S4HTTPError(RuntimeError):
    """Raised for failed S/4 HTTP calls.

    Args:
        method: HTTP method.
        url: Requested URL.
        status_code: HTTP status, or 0 for connection-level failures.
        message: HTTP reason or connection error text.
        body: Optional response body.
    """

    def __init__(self, method: str, url: str, status_code: int, message: str, body: str | None = None) -> None:
        """Create an S/4 HTTP error.

        Returns:
            None.
        """
        super().__init__(f"{method} {url} failed with HTTP {status_code}: {message}")
        self.method = method
        self.url = url
        self.status_code = status_code
        self.message = message
        self.body = body


@dataclass(frozen=True)
class S4Config:
    """Runtime configuration for S/4 OData reads.

    Args:
        base_url: Base S/4 HTTPS URL for direct mode.
        client: Optional SAP client number for direct mode or destination fallback.
        verify: Whether TLS certificate verification is enabled.
        username: Optional S/4 basic-auth username for direct mode.
        password: Optional S/4 basic-auth password for direct mode.
        runtime_context_provider: Optional dynamic context provider for BTP destination mode.
    """

    base_url: str
    client: str | None
    verify: bool
    username: str | None = None
    password: str | None = None
    runtime_context_provider: Callable[[], S4RuntimeContext] | None = None


def is_cloud_foundry_runtime() -> bool:
    """Return whether the process is running inside Cloud Foundry.

    Returns:
        True when Cloud Foundry runtime markers are present.
    """
    return bool(os.getenv("VCAP_APPLICATION") or os.getenv("CF_INSTANCE_GUID"))


def normalize_connectivity_mode(value: str | None) -> str:
    """Normalize the S/4 connectivity mode.

    Args:
        value: Raw `S4_CONNECTIVITY_MODE` value.

    Returns:
        One of `auto`, `direct`, or `btp`.

    Raises:
        S4ConfigError: If the mode is unsupported.
    """
    normalized = (value or "auto").strip().lower().replace("_", "-")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized == "direct":
        return "direct"
    if normalized in {"btp", "destination", "btp-destination"}:
        return "btp"
    raise S4ConfigError(
        "S4_CONNECTIVITY_MODE must be one of: auto, direct, btp."
    )


def parse_bool(value: str | None, default: bool = True) -> bool:
    """Parse an environment boolean.

    Args:
        value: Raw environment value.
        default: Value to return when `value` is missing.

    Returns:
        Parsed boolean.
    """
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise S4ConfigError(f"Invalid boolean value: {value!r}")


def direct_s4_env_values() -> dict[str, str | None]:
    """Read direct S/4 connection values from the environment.

    Returns:
        Mapping of direct S/4 environment variable names to their values.
    """
    return {
        "S4_BASE_URL": os.getenv("S4_BASE_URL"),
        "S4_USERNAME": os.getenv("S4_USERNAME"),
        "S4_PASSWORD": os.getenv("S4_PASSWORD"),
    }


def direct_s4_config_is_complete() -> bool:
    """Return whether direct S/4 connection settings are complete.

    Returns:
        True when local direct S/4 URL and credentials are present.
    """
    return all(value for value in direct_s4_env_values().values())


def direct_s4_config_from_env() -> S4Config:
    """Build direct S/4 configuration from environment variables.

    Returns:
        Direct S4Config.

    Raises:
        S4ConfigError: If required direct settings are missing.
    """
    required_env = direct_s4_env_values()
    missing = [name for name, value in required_env.items() if not value]
    if missing:
        raise S4ConfigError(f"Missing required S/4 environment variables: {', '.join(sorted(missing))}")

    base_url = required_env["S4_BASE_URL"]
    username = required_env["S4_USERNAME"]
    password = required_env["S4_PASSWORD"]
    assert base_url is not None
    assert username is not None
    assert password is not None

    client = os.getenv("S4_CLIENT")
    return S4Config(
        base_url=base_url.rstrip("/"),
        client=client.strip() if client and client.strip() else None,
        verify=parse_bool(os.getenv("S4_VERIFY"), default=True),
        username=username,
        password=password,
    )


def btp_s4_config_from_env() -> S4Config:
    """Build BTP Destination-backed S/4 configuration from environment variables.

    Returns:
        BTP Destination-backed S4Config.

    Raises:
        S4ConfigError: If required BTP settings are missing.
    """
    btp_config = load_s4_btp_connectivity_config_from_env()
    if btp_config is None:
        raise S4ConfigError(
            "S4_CONNECTIVITY_MODE=btp requires complete S/4 Destination and Connectivity environment variables."
        )
    btp_client = S4BtpConnectivityClient(btp_config)
    return S4Config(
        base_url="",
        client=btp_config.fallback_sap_client,
        verify=btp_config.verify,
        username=None,
        password=None,
        runtime_context_provider=btp_client.resolve_runtime_context,
    )


def load_s4_config(env_file: str | Path | None = None) -> S4Config:
    """Load S/4 connection settings for the API package.

    Args:
        env_file: Optional explicit `.env` path.

    Returns:
        S/4 configuration from environment variables.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        api_dir = Path(__file__).resolve().parents[2]
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(api_dir / ".env", override=False)
        load_dotenv(repo_root / ".env", override=False)

    connectivity_mode = normalize_connectivity_mode(os.getenv("S4_CONNECTIVITY_MODE"))
    btp_config = load_s4_btp_connectivity_config_from_env()
    direct_config_available = direct_s4_config_is_complete()

    if connectivity_mode == "direct":
        return direct_s4_config_from_env()
    if connectivity_mode == "btp":
        return btp_s4_config_from_env()
    if btp_config is not None and is_cloud_foundry_runtime():
        return btp_s4_config_from_env()
    if direct_config_available:
        return direct_s4_config_from_env()
    if btp_config is not None:
        return btp_s4_config_from_env()
    return direct_s4_config_from_env()


def odata_str(value: str) -> str:
    """Quote a string for OData V2 filter/key syntax.

    Args:
        value: Raw string value.

    Returns:
        Single-quoted OData string literal with escaped quotes.
    """
    return "'" + value.replace("'", "''") + "'"


def normalize_material_code(value: str) -> str:
    """Normalize a material number to the PoC's externally assigned S/4 code.

    Args:
        value: Material number with or without a leading letter.

    Returns:
        Uppercase S/4 material code.
    """
    normalized = value.strip().upper()
    if normalized and normalized[0].isdigit():
        return f"{DEFAULT_EXTERNAL_MATERIAL_PREFIX}{normalized}"
    return normalized


class S4Client:
    """Small S/4 OData JSON client used by API lookup tools."""

    def __init__(self, config: S4Config) -> None:
        """Create an authenticated S/4 OData client.

        Args:
            config: S/4 connection settings.

        Returns:
            None.
        """
        self.config = config
        self.session = requests.Session()
        if config.username and config.password:
            self.session.auth = (config.username, config.password)
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self.session.verify = config.verify
        if not config.verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _runtime_context(self) -> S4RuntimeContext:
        """Return the current S/4 request context.

        Returns:
            Direct or BTP Destination-backed request context.
        """
        if self.config.runtime_context_provider is not None:
            return self.config.runtime_context_provider()
        return S4RuntimeContext(
            base_url=self.config.base_url,
            client=self.config.client,
            headers={},
            proxies=None,
            verify=self.config.verify,
        )

    def _headers(
        self,
        context: S4RuntimeContext,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Merge dynamic S/4 headers with caller-provided headers.

        Args:
            context: Current request context.
            headers: Optional caller headers.

        Returns:
            Header override dictionary or None.
        """
        merged: dict[str, str] = {}
        merged.update(context.headers)
        if headers:
            merged.update(headers)
        return merged or None

    def service_root(self, service_name: str, context: S4RuntimeContext | None = None) -> str:
        """Build a service root URL.

        Args:
            service_name: Technical OData service name.
            context: Optional already-resolved runtime context.

        Returns:
            Absolute service root URL.
        """
        active_context = context or self._runtime_context()
        return f"{active_context.base_url}/sap/opu/odata/sap/{service_name}"

    def service_url(
        self,
        service_name: str,
        path: str = "",
        context: S4RuntimeContext | None = None,
    ) -> str:
        """Build a service entity URL.

        Args:
            service_name: Technical OData service name.
            path: Entity path within the service.
            context: Optional already-resolved runtime context.

        Returns:
            Absolute request URL.
        """
        return f"{self.service_root(service_name, context=context)}{path}"

    def _params(
        self,
        extra: dict[str, Any] | None = None,
        client: str | None = None,
    ) -> dict[str, Any]:
        """Merge request parameters with the configured SAP client.

        Args:
            extra: Optional caller-provided parameters.
            client: Optional runtime SAP client.

        Returns:
            Request query parameters.
        """
        params: dict[str, Any] = {}
        if extra:
            params.update(extra)
        if client:
            params.setdefault("sap-client", client)
        return params

    def get_json(
        self,
        service_name: str,
        path: str = "",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """GET a JSON response from an S/4 OData service.

        Args:
            service_name: Technical OData service name.
            path: Entity path within the service.
            params: Optional query parameters.
            headers: Optional request headers.

        Returns:
            Parsed JSON body and response headers.
        """
        context = self._runtime_context()
        url = self.service_url(service_name, path, context=context)
        try:
            response = self.session.get(
                url,
                params=self._params(params, client=context.client),
                headers=self._headers(context, headers),
                timeout=120,
                proxies=context.proxies,
                verify=context.verify,
            )
            response.raise_for_status()
            return (json.loads(response.text) if response.text else {}), dict(response.headers)
        except requests.HTTPError as exc:
            r = exc.response
            raise S4HTTPError("GET", url, r.status_code, r.reason, r.text) from exc
        except requests.ConnectionError as exc:
            raise S4HTTPError("GET", url, 0, str(exc)) from exc

    def _request(
        self,
        method: str,
        service_name: str,
        path: str = "",
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], str]:
        """Issue a JSON S/4 OData request.

        Args:
            method: HTTP method.
            service_name: Technical OData service name.
            path: Entity path within the service.
            params: Optional query parameters.
            payload: Optional JSON payload.
            headers: Optional headers.

        Returns:
            Status code, response headers, and response body text.
        """
        context = self._runtime_context()
        url = self.service_url(service_name, path, context=context)
        try:
            response = self.session.request(
                method,
                url,
                params=self._params(params, client=context.client),
                json=payload,
                headers=self._headers(context, headers),
                timeout=120,
                proxies=context.proxies,
                verify=context.verify,
            )
            response.raise_for_status()
            return response.status_code, dict(response.headers), response.text
        except requests.HTTPError as exc:
            r = exc.response
            raise S4HTTPError(method, url, r.status_code, r.reason, r.text) from exc
        except requests.ConnectionError as exc:
            raise S4HTTPError(method, url, 0, str(exc)) from exc

    def _request_raw(
        self,
        method: str,
        service_name: str,
        path: str = "",
        params: dict[str, Any] | None = None,
        body: str | bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], str]:
        """Issue a raw S/4 OData request.

        Args:
            method: HTTP method.
            service_name: Technical OData service name.
            path: Entity path within the service.
            params: Optional query parameters.
            body: Raw request body.
            headers: Optional headers.

        Returns:
            Status code, response headers, and response body text.
        """
        context = self._runtime_context()
        url = self.service_url(service_name, path, context=context)
        try:
            response = self.session.request(
                method,
                url,
                params=self._params(params, client=context.client),
                data=body,
                headers=self._headers(context, headers),
                timeout=120,
                proxies=context.proxies,
                verify=context.verify,
            )
            response.raise_for_status()
            return response.status_code, dict(response.headers), response.text
        except requests.HTTPError as exc:
            r = exc.response
            raise S4HTTPError(method, url, r.status_code, r.reason, r.text) from exc
        except requests.ConnectionError as exc:
            raise S4HTTPError(method, url, 0, str(exc)) from exc

    def get_text(
        self,
        service_name: str,
        path: str = "",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """GET a text response from an S/4 OData service.

        Args:
            service_name: Technical OData service name.
            path: Entity path within the service.
            params: Optional query parameters.
            headers: Optional headers.

        Returns:
            Response body and headers.
        """
        _status_code, response_headers, body = self._request_raw(
            "GET",
            service_name,
            path=path,
            params=params,
            headers=headers,
        )
        return body, response_headers

    def fetch_csrf_token(self, service_name: str) -> str:
        """Fetch a CSRF token for an S/4 OData write service.

        Args:
            service_name: Technical OData service name.

        Returns:
            CSRF token.
        """
        context = self._runtime_context()
        url = self.service_root(service_name, context=context)
        try:
            response = self.session.get(
                url,
                params=self._params(client=context.client),
                headers=self._headers(context, {"X-CSRF-Token": "Fetch"}),
                timeout=60,
                proxies=context.proxies,
                verify=context.verify,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            r = exc.response
            raise S4HTTPError("GET", url, r.status_code, r.reason, r.text) from exc
        except requests.ConnectionError as exc:
            raise S4HTTPError("GET", url, 0, str(exc)) from exc
        token = response.headers.get("X-CSRF-Token")
        if not token:
            raise S4HTTPError("GET", url, 0, "Missing CSRF token in response headers", response.text)
        return token

    def post_json(
        self,
        service_name: str,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], int, dict[str, str]]:
        """POST a JSON payload to an S/4 OData service.

        Args:
            service_name: Technical OData service name.
            path: Entity path within the service.
            payload: JSON payload.
            headers: Optional headers.

        Returns:
            Parsed response body, HTTP status code, and headers.
        """
        csrf_token = self.fetch_csrf_token(service_name)
        request_headers: dict[str, str] = {"X-CSRF-Token": csrf_token}
        if headers:
            request_headers.update(headers)
        status_code, response_headers, body = self._request(
            "POST",
            service_name,
            path=path,
            payload=payload,
            headers=request_headers,
        )
        return (json.loads(body) if body else {}), status_code, response_headers

    def patch_json(
        self,
        service_name: str,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], int, dict[str, str]]:
        """PATCH a JSON payload to an S/4 OData service.

        Args:
            service_name: Technical OData service name.
            path: Entity path within the service.
            payload: JSON payload.
            headers: Optional headers.

        Returns:
            Parsed response body, HTTP status code, and headers.
        """
        csrf_token = self.fetch_csrf_token(service_name)
        request_headers: dict[str, str] = {"X-CSRF-Token": csrf_token}
        if headers:
            request_headers.update(headers)
        status_code, response_headers, body = self._request(
            "PATCH",
            service_name,
            path=path,
            payload=payload,
            headers=request_headers,
        )
        return (json.loads(body) if body else {}), status_code, response_headers

    def post_batch(self, service_name: str, body: str, content_type: str) -> tuple[str, int, dict[str, str]]:
        """POST a multipart OData `$batch` request.

        Args:
            service_name: Technical OData service name.
            body: Multipart batch body.
            content_type: Multipart content type with boundary.

        Returns:
            Response body, HTTP status code, and headers.
        """
        csrf_token = self.fetch_csrf_token(service_name)
        headers = {
            "X-CSRF-Token": csrf_token,
            "Accept": "application/json",
            "Content-Type": content_type,
        }
        status_code, response_headers, response_body = self._request_raw(
            "POST",
            service_name,
            path="/$batch",
            body=body,
            headers=headers,
        )
        return response_body, status_code, response_headers


def clean_optional(value: Any) -> str | None:
    """Normalize optional S/4 string values.

    Args:
        value: Raw value from S/4.

    Returns:
        Stripped string, or `None` when the value is blank.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def odata_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract OData V2 collection rows from a JSON payload.

    Args:
        payload: Parsed OData JSON response.

    Returns:
        List of entity rows.
    """
    return list(payload.get("d", {}).get("results", []))


def odata_entity(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract an OData V2 entity row from a JSON payload.

    Args:
        payload: Parsed OData JSON response.

    Returns:
        Entity row, or an empty dict when the payload is empty.
    """
    data = payload.get("d", {})
    return data if isinstance(data, dict) else {}


def failed_api_result(exc: S4HTTPError) -> dict[str, Any]:
    """Convert an S/4 HTTP failure into an agent-tool-safe response.

    Args:
        exc: S/4 HTTP exception.

    Returns:
        JSON-serializable failure response.
    """
    return {
        "status": "failed_api",
        "message": exc.message,
        "http_status": exc.status_code,
        "api_error_body": exc.body,
    }


def result_from_candidates(candidates: list[dict[str, Any]], found_key: str | None = None) -> dict[str, Any]:
    """Build the lookup result shape expected by existing agent code.

    Args:
        candidates: Candidate rows found in S/4.
        found_key: Optional singular key to include when exactly one candidate exists.

    Returns:
        `found`, `ambiguous`, or `not_found` lookup response.
    """
    if len(candidates) == 1:
        result: dict[str, Any] = {"status": "found", "candidates": candidates}
        if found_key:
            result[found_key] = candidates[0]
        return result
    if len(candidates) > 1:
        return {"status": "ambiguous", "candidates": candidates}
    return {"status": "not_found", "candidates": []}


def money_is_positive(value: Any) -> bool:
    """Return whether an S/4 amount value is positive.

    Args:
        value: Raw decimal value from S/4.

    Returns:
        True when the value parses and is greater than zero.
    """
    try:
        return Decimal(str(value)) > Decimal("0")
    except InvalidOperation:
        return False


def condition_detail_to_price(detail: dict[str, Any]) -> dict[str, str | None]:
    """Normalize an S/4 purchasing condition record into a price object.

    Args:
        detail: `A_PurInfoRecdPrcgCndn` row.

    Returns:
        Price fields used by the agent tool response.
    """
    return {
        "amount": clean_optional(detail.get("ConditionRateValue")),
        "currency": clean_optional(detail.get("ConditionRateValueUnit")),
        "quantity": clean_optional(detail.get("ConditionQuantity")),
        "uom": clean_optional(detail.get("ConditionQuantityUnit")),
    }


def net_price_to_price(info_row: dict[str, Any]) -> dict[str, str | None]:
    """Normalize an S/4 info-record net price into a price object.

    Args:
        info_row: `A_PurgInfoRecdOrgPlantData` row.

    Returns:
        Price fields used by the agent tool response.
    """
    return {
        "amount": clean_optional(info_row.get("NetPriceAmount")),
        "currency": clean_optional(info_row.get("Currency")),
        "quantity": clean_optional(info_row.get("MaterialPriceUnitQty")) or "1",
        "uom": clean_optional(info_row.get("PurgDocOrderQuantityUnit"))
        or clean_optional(info_row.get("PurchaseOrderPriceUnit")),
    }


@dataclass(frozen=True)
class S4LookupConfig:
    """Configuration for S/4-backed agent lookup tools.

    Args:
        purchasing_organization: Purchasing organization for supplier price context.
        info_record_category: Purchasing info-record category.
        plant: Optional plant; blank means general supplier/material pricing.
        condition_type: Purchasing base price condition type.
        language: Product description language.
        top: Maximum search candidates to return.
    """

    purchasing_organization: str = "1010"
    info_record_category: str = "0"
    plant: str = ""
    condition_type: str = "PPR0"
    language: str = "EN"
    top: int = 20


class S4LookupRepository:
    """S/4-backed lookup repository for LangGraph price-change tools.

    This repository intentionally mirrors the old HANA lookup method names and
    return shapes so the agent can keep its existing tool contract while S/4
    becomes the source of truth for supplier, material, and current price data.
    """

    def __init__(self, client: S4Client, config: S4LookupConfig | None = None) -> None:
        """Create an S/4 lookup repository.

        Args:
            client: Authenticated S/4 OData client.
            config: Optional lookup defaults.

        Returns:
            None.
        """
        self.client = client
        self.config = config or S4LookupConfig()

    @classmethod
    def from_env(cls) -> "S4LookupRepository":
        """Build a repository from S/4 environment variables.

        Returns:
            S/4 lookup repository using `.env`/environment credentials.
        """
        return cls(S4Client(load_s4_config()))

    def find_supplier_by_email(self, email: str) -> dict[str, Any]:
        """Find supplier candidates by email-address substring in S/4.

        Args:
            email: Full or partial supplier contact email/domain.

        Returns:
            Existing agent lookup response with matching supplier candidates and
            a `supplier` row when exactly one S/4 supplier owns a matching email.
        """
        normalized = clean_optional(email)
        if not normalized:
            return {"status": "not_found", "candidates": []}
        try:
            payload, _ = self.client.get_json(
                BUSINESS_PARTNER_SERVICE_NAME,
                path="/A_AddressEmailAddress",
                params={
                    "$format": "json",
                    "$filter": f"substringof({odata_str(normalized)},EmailAddress)",
                    "$top": str(self.config.top),
                },
            )
            candidates: list[dict[str, Any]] = []
            for email_row in odata_rows(payload):
                candidates.extend(self._supplier_candidates_for_email_row(email_row))
            return result_from_candidates(self._dedupe_suppliers(candidates)[: self.config.top], found_key="supplier")
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def find_supplier_by_id(self, supplier_id: str) -> dict[str, Any]:
        """Find supplier candidates by local id or S/4 supplier-number substring.

        Args:
            supplier_id: Local supplier account fragment or S/4 supplier-number
                fragment from the source document.

        Returns:
            Existing agent lookup response with candidates from both supplier
            account-number contains search and S/4 supplier-number contains
            search, plus a `supplier` row when exactly one candidate remains.
        """
        normalized = clean_optional(supplier_id)
        if not normalized:
            return {"status": "not_found", "candidates": []}
        try:
            candidates = [
                *self._supplier_candidates_for_supplier_account_number(normalized),
                *self._supplier_candidates_for_supplier_number_contains(normalized),
            ]
            return result_from_candidates(self._dedupe_suppliers(candidates)[: self.config.top], found_key="supplier")
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def find_supplier_by_name(self, name_or_company: str) -> dict[str, Any]:
        """Find suppliers by company name in S/4 Business Partner data.

        Args:
            name_or_company: Supplier company/name fragment.

        Returns:
            Existing agent lookup response with candidate supplier rows.
        """
        query = clean_optional(name_or_company)
        if not query:
            return {"status": "not_found", "candidates": []}
        try:
            payload, _ = self.client.get_json(
                BUSINESS_PARTNER_SERVICE_NAME,
                path="/A_BusinessPartner",
                params={
                    "$format": "json",
                    "$select": "BusinessPartner,Supplier,OrganizationBPName1,BusinessPartnerName",
                    "$filter": f"substringof({odata_str(query)},OrganizationBPName1)",
                    "$top": str(self.config.top),
                },
            )
            candidates = [
                self._supplier_candidate_from_business_partner(row)
                for row in odata_rows(payload)
                if clean_optional(row.get("Supplier"))
            ]
            return result_from_candidates(self._dedupe_suppliers(candidates)[: self.config.top], found_key="supplier")
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def find_material_by_number(self, material_number: str) -> dict[str, Any]:
        """Find material candidates by product-number substring in S/4.

        Args:
            material_number: Full or partial product/material number from the
                source document.

        Returns:
            Existing agent material lookup response with matching material
            candidates from `/A_Product` expanded with descriptions.
        """
        normalized = clean_optional(material_number)
        if not normalized:
            return {"status": "not_found", "candidates": []}
        try:
            payload, _ = self.client.get_json(
                PRODUCT_SERVICE_NAME,
                path="/A_Product",
                params={
                    "$format": "json",
                    "$filter": f"substringof({odata_str(normalized)},Product)",
                    "$expand": "to_Description",
                    "$top": str(self.config.top),
                },
            )
            candidates = [self._material_candidate_from_product(row) for row in odata_rows(payload)]
            return result_from_candidates(candidates)
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def find_supplier_material_price_candidates(
        self,
        supplier_candidates: list[dict[str, Any]],
        material_candidates: list[dict[str, Any]],
        purchasing_organizations: list[str] | None = None,
        info_record_categories: list[str] | None = None,
        plants: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find exact price-context rows for selected supplier/material candidates.

        Args:
            supplier_candidates: Supplier candidate rows. `supplier` is used
                first; `supplier_id` is only used when it does not look like a
                local id such as `SUP001`.
            material_candidates: Material candidate rows. `material_code` is
                used first, then `material_number` when no code is present.
            purchasing_organizations: Optional purchasing organizations to
                include as exact filters. Defaults to the configured one.
            info_record_categories: Optional info-record categories to include
                as exact filters. Defaults to the configured one.
            plants: Optional plants to include as exact filters. Defaults to
                the configured plant value, including blank plant.

        Returns:
            Lookup response with normalized price-context candidates. The result
            is `found` for one row, `ambiguous` for multiple rows, and
            `not_found` when S/4 returns no matching info-record rows.
        """
        suppliers = self._candidate_supplier_numbers(supplier_candidates)
        materials = self._candidate_material_codes(material_candidates)
        purchasing_orgs = self._bounded_clean_values(
            purchasing_organizations if purchasing_organizations is not None else [self.config.purchasing_organization]
        )
        categories = self._bounded_clean_values(
            info_record_categories if info_record_categories is not None else [self.config.info_record_category]
        )
        plant_values = self._bounded_clean_values(plants if plants is not None else [self.config.plant], keep_blank=True)
        if not suppliers or not materials:
            return {"status": "not_found", "candidates": []}
        try:
            info_rows: list[dict[str, Any]] = []
            pair_clauses = self._supplier_material_pair_clauses(suppliers, materials)
            if not pair_clauses:
                return {"status": "not_found", "candidates": []}
            for pair_chunk in self._chunk_values(pair_clauses, self._supplier_material_pair_chunk_size()):
                payload, _ = self.client.get_json(
                    INFORECORD_SERVICE_NAME,
                    path="/A_PurgInfoRecdOrgPlantData",
                    params={
                        "$format": "json",
                        "$select": (
                            "PurchasingInfoRecord,PurchasingInfoRecordCategory,PurchasingOrganization,"
                            "Plant,Supplier,Material,NetPriceAmount,Currency,PurgDocOrderQuantityUnit,"
                            "MaterialPriceUnitQty,PurchaseOrderPriceUnit"
                        ),
                        "$filter": self._supplier_material_price_candidate_filter(
                            pair_chunk,
                            purchasing_orgs,
                            categories,
                            plant_values,
                        ),
                        "$top": str(self.config.top),
                    },
                )
                info_rows.extend(odata_rows(payload))
            candidates = self._dedupe_price_context_candidates(
                [
                    self._price_context_candidate_from_info_record(
                        row,
                        supplier_candidates=supplier_candidates,
                        material_candidates=material_candidates,
                    )
                    for row in info_rows
                ]
            )[: self.config.top]
            return result_from_candidates(candidates)
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def search_materials_by_description(self, query: str, supplier_id: str | None = None) -> dict[str, Any]:
        """Search S/4 product descriptions, optionally scoped to a supplier.

        Args:
            query: Product description fragment.
            supplier_id: Optional local supplier id or S/4 supplier number.

        Returns:
            Existing agent material lookup response with matching candidates.
        """
        normalized = clean_optional(query)
        if not normalized:
            return {"status": "not_found", "candidates": []}
        try:
            supplier = self._supplier_number_for_optional_scope(supplier_id)
            if isinstance(supplier, dict):
                return supplier
            payload, _ = self.client.get_json(
                PRODUCT_SERVICE_NAME,
                path="/A_ProductDescription",
                params={
                    "$format": "json",
                    "$filter": (
                        f"substringof({odata_str(normalized)},ProductDescription) "
                        f"and Language eq {odata_str(self.config.language)}"
                    ),
                    "$top": str(self.config.top),
                },
            )
            candidates = [self._material_candidate_from_description(row) for row in odata_rows(payload)]
            if supplier:
                candidates = [
                    candidate
                    for candidate in candidates
                    if self._supplier_has_material(supplier, str(candidate["material_code"]))
                ]
            return result_from_candidates(candidates)
        except S4HTTPError as exc:
            return failed_api_result(exc)

    def get_current_supplier_material_price(self, supplier_id: str, material_number: str) -> dict[str, Any]:
        """Read current supplier-material price from S/4 info-record APIs.

        Args:
            supplier_id: Local `SUP###` supplier id or S/4 supplier number.
            material_number: Product/material number.

        Returns:
            HANA-compatible price response where `price.current_price` contains
            the current amount when S/4 has a readable PPR0 or NetPriceAmount.
        """
        try:
            supplier_lookup = self._find_exact_supplier_for_price_context(supplier_id)
        except S4HTTPError as exc:
            return {**failed_api_result(exc), "price": None}
        supplier = supplier_lookup.get("supplier")
        if supplier_lookup.get("status") != "found" or not supplier:
            return {
                "status": "supplier_missing",
                "price": None,
                "supplier_lookup": supplier_lookup,
            }
        material_code = normalize_material_code(material_number)
        context = {
            "supplier_id": str(supplier.get("supplier_id") or supplier_id),
            "supplier": str(supplier.get("supplier")),
            "purchasing_organization": str(
                supplier.get("purchasing_organization") or self.config.purchasing_organization
            ),
            "plant": self.config.plant,
            "info_record_category": self.config.info_record_category,
        }
        try:
            result = self._read_current_supplier_material_price(material_code, context)
        except S4HTTPError as exc:
            return {**failed_api_result(exc), "price": None}
        if result["status"] != "found":
            return {
                "status": result["status"],
                "price": None,
                "message": result.get("message"),
                "s4_result": result,
            }
        price = result.get("price") or {}
        return {
            "status": "found",
            "source": result.get("source"),
            "price": {
                "supplier_id": context["supplier_id"],
                "supplier": context["supplier"],
                "material_code": material_code,
                "current_price": clean_optional(price.get("amount")),
                "currency": clean_optional(price.get("currency")),
                "uom": clean_optional(price.get("uom")),
                "quantity": clean_optional(price.get("quantity")),
            },
            "s4_result": result,
        }

    def _find_exact_supplier_for_price_context(self, supplier_id: str) -> dict[str, Any]:
        """Resolve one supplier exactly for current-price lookup context.

        Args:
            supplier_id: Local supplier account number or S/4 supplier number
                that should already be selected or known complete.

        Returns:
            Existing supplier lookup shape using exact S/4 filters. Local
            `SUP...` ids resolve only through exact SupplierAccountNumber;
            other values prefer exact S/4 Supplier and then fall back to exact
            SupplierAccountNumber for numeric local account numbers.
        """
        normalized = clean_optional(supplier_id)
        if not normalized:
            return {"status": "not_found", "candidates": []}

        if normalized.upper().startswith("SUP"):
            candidates = self._supplier_candidates_for_exact_supplier_account_number(normalized)
        else:
            candidates = self._supplier_candidates_for_supplier_number(normalized)
            if not candidates:
                candidates = self._supplier_candidates_for_exact_supplier_account_number(normalized)
        return result_from_candidates(self._dedupe_suppliers(candidates)[: self.config.top], found_key="supplier")

    def _read_current_supplier_material_price(
        self,
        material_code: str,
        context: dict[str, str],
    ) -> dict[str, Any]:
        """Read the current price for one resolved supplier/material context.

        Args:
            material_code: S/4 product/material code.
            context: Supplier and purchasing organization values.

        Returns:
            Current-price result preferring PPR0 over NetPriceAmount.
        """
        info_rows = self._find_current_price_info_record_rows(material_code, context)
        if not info_rows:
            return {
                "status": "info_record_missing",
                "message": "No supplier/material purchasing info record exists in S/4.",
                "supplier_id": context["supplier_id"],
                "material_code": material_code,
            }
        if len(info_rows) > 1:
            return {
                "status": "info_record_ambiguous",
                "message": "Multiple supplier/material purchasing info records exist in S/4.",
                "matches": info_rows,
            }

        validity_rows = self._find_condition_validity_rows(material_code, context)
        if len(validity_rows) == 1:
            condition_record = clean_optional(validity_rows[0].get("ConditionRecord"))
            if condition_record:
                detail = self._get_condition_record(condition_record)
                return {
                    "status": "found",
                    "source": "ppr0",
                    "supplier_id": context["supplier_id"],
                    "material_code": material_code,
                    "info_record": info_rows[0],
                    "condition_validity": validity_rows[0],
                    "condition_record": detail,
                    "price": condition_detail_to_price(detail),
                }
        if len(validity_rows) > 1:
            return {
                "status": "price_ambiguous",
                "message": "Multiple positive PPR0 condition rows exist for this supplier/material.",
                "matches": validity_rows,
            }
        if money_is_positive(info_rows[0].get("NetPriceAmount")):
            return {
                "status": "found",
                "source": "net_price_amount",
                "supplier_id": context["supplier_id"],
                "material_code": material_code,
                "info_record": info_rows[0],
                "condition_validity": None,
                "condition_record": None,
                "price": net_price_to_price(info_rows[0]),
            }
        return {
            "status": "price_missing",
            "message": "Info record exists, but neither PPR0 nor positive NetPriceAmount is defined.",
            "supplier_id": context["supplier_id"],
            "material_code": material_code,
            "info_record": info_rows[0],
            "ppr0_rows": validity_rows,
        }

    def _find_current_price_info_record_rows(
        self,
        material_code: str,
        context: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Find purchasing info-record org/plant rows for a supplier/material pair.

        Args:
            material_code: S/4 product/material code.
            context: Supplier and purchasing organization values.

        Returns:
            Matching info-record rows.
        """
        payload, _ = self.client.get_json(
            INFORECORD_SERVICE_NAME,
            path="/A_PurgInfoRecdOrgPlantData",
            params={
                "$format": "json",
                "$select": (
                    "PurchasingInfoRecord,PurchasingInfoRecordCategory,PurchasingOrganization,"
                    "Plant,Supplier,Material,NetPriceAmount,Currency,PurgDocOrderQuantityUnit,"
                    "MaterialPriceUnitQty,PurchaseOrderPriceUnit"
                ),
                "$filter": (
                    f"Supplier eq {odata_str(context['supplier'])} "
                    f"and Material eq {odata_str(material_code)} "
                    f"and PurchasingOrganization eq {odata_str(context['purchasing_organization'])} "
                    f"and PurchasingInfoRecordCategory eq {odata_str(context['info_record_category'])} "
                    f"and Plant eq {odata_str(context['plant'])}"
                ),
            },
        )
        return odata_rows(payload)

    def _find_condition_validity_rows(
        self,
        material_code: str,
        context: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Find active PPR0 validity rows for a supplier/material pair.

        Args:
            material_code: S/4 product/material code.
            context: Supplier and purchasing organization values.

        Returns:
            PPR0 validity rows.
        """
        payload, _ = self.client.get_json(
            INFORECORD_SERVICE_NAME,
            path="/A_PurInfoRecdPrcgCndnValidity",
            params={
                "$format": "json",
                "$select": (
                    "ConditionRecord,ConditionValidityEndDate,ConditionValidityStartDate,ConditionType,"
                    "PurchasingOrganization,PurchasingInfoRecordCategory,Supplier,Material,Plant"
                ),
                "$filter": (
                    f"ConditionType eq {odata_str(self.config.condition_type)} "
                    f"and Supplier eq {odata_str(context['supplier'])} "
                    f"and Material eq {odata_str(material_code)} "
                    f"and PurchasingOrganization eq {odata_str(context['purchasing_organization'])} "
                    f"and PurchasingInfoRecordCategory eq {odata_str(context['info_record_category'])} "
                    f"and Plant eq {odata_str(context['plant'])}"
                ),
            },
        )
        return odata_rows(payload)

    def _get_condition_record(self, condition_record: str) -> dict[str, Any]:
        """Read one purchasing info-record pricing condition detail.

        Args:
            condition_record: S/4 condition record id.

        Returns:
            Condition detail row.
        """
        payload, _ = self.client.get_json(
            INFORECORD_SERVICE_NAME,
            path=f"/A_PurInfoRecdPrcgCndn({odata_str(condition_record)})",
            params={"$format": "json"},
        )
        return odata_entity(payload)

    def _candidate_supplier_numbers(self, supplier_candidates: list[dict[str, Any]]) -> list[str]:
        """Extract bounded S/4 supplier numbers from supplier candidates.

        Args:
            supplier_candidates: Supplier candidate rows from S/4 lookup tools.

        Returns:
            Deduplicated supplier numbers, preferring `supplier` and only using
            `supplier_id` when it is not a local `SUP###` identifier.
        """
        values: list[str] = []
        for candidate in supplier_candidates[: self.config.top]:
            supplier = clean_optional(candidate.get("supplier"))
            supplier_id = clean_optional(candidate.get("supplier_id"))
            if supplier:
                values.append(supplier)
            elif supplier_id and not supplier_id.upper().startswith("SUP"):
                values.append(supplier_id)
        return self._dedupe_values(values)[: self.config.top]

    def _candidate_material_codes(self, material_candidates: list[dict[str, Any]]) -> list[str]:
        """Extract bounded S/4 material codes from material candidates.

        Args:
            material_candidates: Material candidate rows from S/4 lookup tools.

        Returns:
            Deduplicated material codes, preferring `material_code` and falling
            back to `material_number` when no code is available.
        """
        values: list[str] = []
        for candidate in material_candidates[: self.config.top]:
            material_code = clean_optional(candidate.get("material_code"))
            material_number = clean_optional(candidate.get("material_number"))
            if material_code:
                values.append(material_code)
            elif material_number:
                values.append(material_number)
        return self._dedupe_values(values)[: self.config.top]

    def _bounded_clean_values(self, values: list[str] | None, keep_blank: bool = False) -> list[str]:
        """Clean, deduplicate, and cap optional purchasing-context values.

        Args:
            values: Raw dimension values.
            keep_blank: Whether a blank value is a valid exact filter value.

        Returns:
            Deduplicated values capped to the configured top limit.
        """
        cleaned: list[str] = []
        for value in (values or [])[: self.config.top]:
            normalized = clean_optional(value)
            if normalized is not None:
                cleaned.append(normalized)
            elif keep_blank and value is not None:
                cleaned.append("")
        return self._dedupe_values(cleaned)[: self.config.top]

    def _dedupe_values(self, values: list[str]) -> list[str]:
        """Deduplicate string values while preserving input order.

        Args:
            values: Candidate string values.

        Returns:
            Ordered unique values.
        """
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _supplier_material_price_candidate_filter(
        self,
        pair_clauses: list[str],
        purchasing_organizations: list[str],
        info_record_categories: list[str],
        plants: list[str],
    ) -> str:
        """Build an exact-match OData filter for supplier/material price candidates.

        Args:
            pair_clauses: Exact supplier/material pair clauses for this query
                chunk.
            purchasing_organizations: Bounded purchasing organizations.
            info_record_categories: Bounded info-record categories.
            plants: Bounded plant values, including blank when configured.

        Returns:
            OData filter that evaluates one chunk of supplier/material pairs
            and optional purchasing-context dimensions.
        """
        filters = [f"({' or '.join(pair_clauses)})"]
        if purchasing_organizations:
            filters.append(self._or_filter("PurchasingOrganization", purchasing_organizations))
        if info_record_categories:
            filters.append(self._or_filter("PurchasingInfoRecordCategory", info_record_categories))
        if plants:
            filters.append(self._or_filter("Plant", plants))
        return " and ".join(filters)

    def _supplier_material_pair_clauses(self, suppliers: list[str], materials: list[str]) -> list[str]:
        """Build exact clauses for the full bounded supplier/material product.

        Args:
            suppliers: Bounded S/4 supplier numbers.
            materials: Bounded S/4 material codes.

        Returns:
            Exact OData pair clauses covering every supplier/material pair after
            each input dimension has already been capped to the configured top.
        """
        return [
            f"(Supplier eq {odata_str(supplier)} and Material eq {odata_str(material)})"
            for supplier in suppliers[: self.config.top]
            for material in materials[: self.config.top]
        ]

    def _supplier_material_pair_chunk_size(self) -> int:
        """Return the maximum number of supplier/material pairs per GET.

        Returns:
            Positive chunk size for price-context OData filters. The configured
            top value is reused as the per-request pair limit, while the caller
            still evaluates every dimension-bounded pair across chunks.
        """
        return max(1, self.config.top)

    def _chunk_values(self, values: list[str], chunk_size: int) -> list[list[str]]:
        """Split values into fixed-size chunks.

        Args:
            values: Ordered values to split.
            chunk_size: Maximum number of values per chunk.

        Returns:
            Ordered list of chunks preserving the original values.
        """
        bounded_chunk_size = max(1, chunk_size)
        return [values[index : index + bounded_chunk_size] for index in range(0, len(values), bounded_chunk_size)]

    def _or_filter(self, field_name: str, values: list[str]) -> str:
        """Build an OData OR clause for exact values of one field.

        Args:
            field_name: S/4 OData field name.
            values: Exact values for the field.

        Returns:
            Parenthesized OR clause.
        """
        return "(" + " or ".join(f"{field_name} eq {odata_str(value)}" for value in values[: self.config.top]) + ")"

    def _price_context_candidate_from_info_record(
        self,
        info_row: dict[str, Any],
        *,
        supplier_candidates: list[dict[str, Any]],
        material_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Normalize one S/4 info-record org/plant row as a price candidate.

        Args:
            info_row: `A_PurgInfoRecdOrgPlantData` row.
            supplier_candidates: Source supplier candidates used to build the
                query, for preserving local metadata when available.
            material_candidates: Source material candidates used to build the
                query, for preserving material descriptions when available.

        Returns:
            Agent-compatible price-context candidate row using NetPriceAmount
            fields and source metadata from the original candidates.
        """
        supplier = clean_optional(info_row.get("Supplier"))
        material_code = clean_optional(info_row.get("Material"))
        supplier_source = self._source_supplier_candidate(supplier, supplier_candidates)
        material_source = self._source_material_candidate(material_code, material_candidates)
        price = net_price_to_price(info_row)
        return {
            "supplier": supplier,
            "supplier_id": clean_optional(supplier_source.get("supplier_id")) if supplier_source else None,
            "company": clean_optional(supplier_source.get("company")) if supplier_source else None,
            "material_code": material_code,
            "material_description": (
                clean_optional(material_source.get("material_description")) if material_source else None
            ),
            "purchasing_info_record": clean_optional(info_row.get("PurchasingInfoRecord")),
            "purchasing_organization": clean_optional(info_row.get("PurchasingOrganization")),
            "info_record_category": clean_optional(info_row.get("PurchasingInfoRecordCategory")),
            "plant": clean_optional(info_row.get("Plant")) or "",
            "current_price": clean_optional(price.get("amount")),
            "currency": clean_optional(price.get("currency")),
            "quantity": clean_optional(price.get("quantity")),
            "uom": clean_optional(price.get("uom")),
            "source": "net_price_amount",
        }

    def _source_supplier_candidate(
        self,
        supplier: str | None,
        supplier_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find the original supplier candidate for a returned S/4 supplier.

        Args:
            supplier: Returned S/4 supplier number.
            supplier_candidates: Candidate rows used to build the query.

        Returns:
            Matching source candidate, or an empty dict.
        """
        if not supplier:
            return {}
        for candidate in supplier_candidates:
            candidate_supplier = clean_optional(candidate.get("supplier"))
            candidate_supplier_id = clean_optional(candidate.get("supplier_id"))
            if candidate_supplier == supplier or candidate_supplier_id == supplier:
                return candidate
        return {}

    def _source_material_candidate(
        self,
        material_code: str | None,
        material_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find the original material candidate for a returned S/4 material.

        Args:
            material_code: Returned S/4 material code.
            material_candidates: Candidate rows used to build the query.

        Returns:
            Matching source candidate, or an empty dict.
        """
        if not material_code:
            return {}
        for candidate in material_candidates:
            candidate_code = clean_optional(candidate.get("material_code"))
            candidate_number = clean_optional(candidate.get("material_number"))
            if candidate_code == material_code or candidate_number == material_code:
                return candidate
        return {}

    def _dedupe_price_context_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate price-context rows while preserving order.

        Args:
            candidates: Normalized price-context candidate rows.

        Returns:
            Deduplicated candidates.
        """
        seen: set[tuple[str | None, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for candidate in candidates:
            key = (
                candidate.get("supplier"),
                candidate.get("material_code"),
                candidate.get("purchasing_info_record"),
                candidate.get("purchasing_organization"),
                candidate.get("info_record_category"),
                candidate.get("plant"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _supplier_candidates_for_email_row(self, email_row: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve one email-address row to supplier candidates.

        Args:
            email_row: `A_AddressEmailAddress` row.

        Returns:
            Supplier candidates connected to the email address.
        """
        address_id = clean_optional(email_row.get("AddressID"))
        if not address_id:
            return []
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_BusinessPartnerAddress",
            params={
                "$format": "json",
                "$select": "BusinessPartner,AddressID",
                "$filter": f"AddressID eq {odata_str(address_id)}",
                "$top": str(self.config.top),
            },
        )
        candidates: list[dict[str, Any]] = []
        for address_row in odata_rows(payload):
            business_partner = clean_optional(address_row.get("BusinessPartner"))
            if not business_partner:
                continue
            bp_payload, _ = self.client.get_json(
                BUSINESS_PARTNER_SERVICE_NAME,
                path=f"/A_BusinessPartner({odata_str(business_partner)})",
                params={
                    "$format": "json",
                    "$select": "BusinessPartner,Supplier,OrganizationBPName1,BusinessPartnerName",
                },
            )
            bp = odata_entity(bp_payload)
            if clean_optional(bp.get("Supplier")):
                candidates.append(
                    self._supplier_candidate_from_business_partner(
                        bp,
                        email=clean_optional(email_row.get("EmailAddress")),
                    )
                )
        return candidates

    def _supplier_candidates_for_exact_supplier_account_number(
        self,
        supplier_account_number: str,
    ) -> list[dict[str, Any]]:
        """Resolve one exact supplier account number to S/4 candidates.

        Args:
            supplier_account_number: Complete supplier account number already
                selected for an exact price-context lookup.

        Returns:
            Matching supplier candidates scoped to the configured purchasing
            organization. Returned supplier numbers are hydrated with exact
            Business Partner reads.
        """
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_SupplierPurchasingOrg",
            params={
                "$format": "json",
                "$filter": (
                    f"SupplierAccountNumber eq {odata_str(supplier_account_number)} "
                    f"and PurchasingOrganization eq {odata_str(self.config.purchasing_organization)}"
                ),
                "$top": str(self.config.top),
            },
        )
        candidates: list[dict[str, Any]] = []
        for purchasing_row in odata_rows(payload):
            supplier = clean_optional(purchasing_row.get("Supplier"))
            if supplier:
                candidates.extend(
                    self._supplier_candidates_for_supplier_number(
                        supplier,
                        purchasing_org_row=purchasing_row,
                    )
                )
        return candidates

    def _supplier_candidates_for_supplier_account_number(self, supplier_account_number: str) -> list[dict[str, Any]]:
        """Resolve supplier account-number fragments to S/4 supplier candidates.

        Args:
            supplier_account_number: Full or partial account number from the
                source document.

        Returns:
            Matching supplier candidates scoped to the configured purchasing
            organization. Returned supplier numbers are hydrated with exact
            Business Partner reads.
        """
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_SupplierPurchasingOrg",
            params={
                "$format": "json",
                "$filter": (
                    f"substringof({odata_str(supplier_account_number)},SupplierAccountNumber) "
                    f"and PurchasingOrganization eq {odata_str(self.config.purchasing_organization)}"
                ),
                "$top": str(self.config.top),
            },
        )
        candidates: list[dict[str, Any]] = []
        for purchasing_row in odata_rows(payload):
            supplier = clean_optional(purchasing_row.get("Supplier"))
            if supplier:
                candidates.extend(
                    self._supplier_candidates_for_supplier_number(
                        supplier,
                        purchasing_org_row=purchasing_row,
                    )
                )
        return candidates

    def _supplier_candidates_for_supplier_number_contains(self, supplier: str) -> list[dict[str, Any]]:
        """Search S/4 suppliers by supplier-number substring.

        Args:
            supplier: Full or partial S/4 supplier number from the source
                document.

        Returns:
            Matching supplier candidates from `/A_BusinessPartner`. Purchasing
            organization data is hydrated per candidate through the existing
            candidate builder.
        """
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_BusinessPartner",
            params={
                "$format": "json",
                "$select": "BusinessPartner,Supplier,OrganizationBPName1,BusinessPartnerName",
                "$filter": f"substringof({odata_str(supplier)},Supplier)",
                "$top": str(self.config.top),
            },
        )
        return [
            self._supplier_candidate_from_business_partner(row)
            for row in odata_rows(payload)
            if clean_optional(row.get("Supplier"))
        ]

    def _supplier_candidates_for_supplier_number(
        self,
        supplier: str,
        purchasing_org_row: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve an S/4 supplier number to Business Partner candidates.

        Args:
            supplier: S/4 supplier account number.
            purchasing_org_row: Optional already-read purchasing org row.

        Returns:
            Matching supplier candidates.
        """
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_BusinessPartner",
            params={
                "$format": "json",
                "$select": "BusinessPartner,Supplier,OrganizationBPName1,BusinessPartnerName",
                "$filter": f"Supplier eq {odata_str(supplier)}",
                "$top": str(self.config.top),
            },
        )
        return [
            self._supplier_candidate_from_business_partner(row, purchasing_org_row=purchasing_org_row)
            for row in odata_rows(payload)
            if clean_optional(row.get("Supplier"))
        ]

    def _supplier_candidate_from_business_partner(
        self,
        business_partner: dict[str, Any],
        *,
        email: str | None = None,
        purchasing_org_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a supplier candidate row from Business Partner data.

        Args:
            business_partner: `A_BusinessPartner` row.
            email: Optional email that led to the supplier.
            purchasing_org_row: Optional `A_SupplierPurchasingOrg` row.

        Returns:
            Agent-compatible supplier row.
        """
        supplier = clean_optional(business_partner.get("Supplier")) or ""
        purchasing = purchasing_org_row or self._first_supplier_purchasing_org(supplier)
        supplier_id = clean_optional(purchasing.get("SupplierAccountNumber")) if purchasing else None
        return {
            "supplier_id": supplier_id or supplier,
            "supplier": supplier,
            "business_partner": clean_optional(business_partner.get("BusinessPartner")),
            "company": (
                clean_optional(business_partner.get("OrganizationBPName1"))
                or clean_optional(business_partner.get("BusinessPartnerName"))
            ),
            "email": email,
            "purchasing_organization": (
                clean_optional(purchasing.get("PurchasingOrganization")) if purchasing else self.config.purchasing_organization
            ),
            "purchase_order_currency": clean_optional(purchasing.get("PurchaseOrderCurrency")) if purchasing else None,
            "contact_name": clean_optional(purchasing.get("SupplierRespSalesPersonName")) if purchasing else None,
        }

    def _first_supplier_purchasing_org(self, supplier: str) -> dict[str, Any]:
        """Read the purchasing organization row for one S/4 supplier.

        Args:
            supplier: S/4 supplier account number.

        Returns:
            First matching purchasing org row, or an empty dict.
        """
        if not supplier:
            return {}
        payload, _ = self.client.get_json(
            BUSINESS_PARTNER_SERVICE_NAME,
            path="/A_SupplierPurchasingOrg",
            params={
                "$format": "json",
                "$filter": (
                    f"Supplier eq {odata_str(supplier)} "
                    f"and PurchasingOrganization eq {odata_str(self.config.purchasing_organization)}"
                ),
                "$top": "1",
            },
        )
        rows = odata_rows(payload)
        return rows[0] if rows else {}

    def _material_candidate_from_product(self, product: dict[str, Any]) -> dict[str, Any]:
        """Convert an S/4 product row to the agent material row shape.

        Args:
            product: `A_Product` row expanded with `to_Description`.

        Returns:
            Agent-compatible material row.
        """
        descriptions = product.get("to_Description", {}).get("results", [])
        description = ""
        for row in descriptions:
            if clean_optional(row.get("Language")) == self.config.language:
                description = clean_optional(row.get("ProductDescription")) or ""
                break
        if not description and descriptions:
            description = clean_optional(descriptions[0].get("ProductDescription")) or ""
        return {
            "material_code": clean_optional(product.get("Product")),
            "material_description": description,
            "supplier_id": None,
            "current_price": None,
            "currency": None,
            "uom": clean_optional(product.get("BaseUnit")),
        }

    def _material_candidate_from_description(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert an S/4 product-description row to the agent material row shape.

        Args:
            row: `A_ProductDescription` row.

        Returns:
            Agent-compatible material row.
        """
        return {
            "material_code": clean_optional(row.get("Product")),
            "material_description": clean_optional(row.get("ProductDescription")),
            "supplier_id": None,
            "current_price": None,
            "currency": None,
            "uom": None,
        }

    def _supplier_number_for_optional_scope(self, supplier_id: str | None) -> str | dict[str, Any] | None:
        """Resolve an optional supplier scope to an S/4 supplier number.

        Args:
            supplier_id: Optional local supplier id or S/4 supplier number.

        Returns:
            S/4 supplier number, `None` when no scope is requested, or an error
            lookup response when the supplier cannot be resolved.
        """
        if not clean_optional(supplier_id):
            return None
        supplier_lookup = self.find_supplier_by_id(str(supplier_id))
        supplier = supplier_lookup.get("supplier")
        if supplier_lookup.get("status") != "found" or not supplier:
            return {"status": "not_found", "candidates": []}
        return str(supplier.get("supplier"))

    def _supplier_has_material(self, supplier: str, material_code: str) -> bool:
        """Return whether S/4 has a supplier/material purchasing info record.

        Args:
            supplier: S/4 supplier account number.
            material_code: S/4 product/material code.

        Returns:
            True when at least one matching purchasing info-record org/plant row exists.
        """
        payload, _ = self.client.get_json(
            INFORECORD_SERVICE_NAME,
            path="/A_PurgInfoRecdOrgPlantData",
            params={
                "$format": "json",
                "$select": "PurchasingInfoRecord,Material,Supplier",
                "$filter": (
                    f"Supplier eq {odata_str(supplier)} "
                    f"and Material eq {odata_str(material_code)} "
                    f"and PurchasingOrganization eq {odata_str(self.config.purchasing_organization)} "
                    f"and PurchasingInfoRecordCategory eq {odata_str(self.config.info_record_category)} "
                    f"and Plant eq {odata_str(self.config.plant)}"
                ),
                "$top": "1",
            },
        )
        return bool(odata_rows(payload))

    def _dedupe_suppliers(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate supplier candidates while preserving order.

        Args:
            candidates: Supplier candidates.

        Returns:
            Deduplicated supplier candidates.
        """
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for candidate in candidates:
            key = str(candidate.get("supplier") or candidate.get("supplier_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped
