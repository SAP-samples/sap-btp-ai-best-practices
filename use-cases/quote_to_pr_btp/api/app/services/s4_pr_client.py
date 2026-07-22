"""S/4HANA Purchase Requisition client and payload builder.

The client prefers BTP Destination + Connectivity Service when those
credentials are present. It falls back to direct Basic Auth for local tests.
PR creation is explicit: the UI must first build/review the payload and then
send confirm_create=true.
"""

from __future__ import annotations

import datetime as dt
import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

PR_SERVICE_PATH = "/sap/opu/odata/sap/API_PURCHASEREQ_PROCESS_SRV"
PR_HEADER_ENTITY_SET = "A_PurchaseRequisitionHeader"


class S4PRError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@dataclass(frozen=True)
class S4PRConfig:
    base_url: str
    client: str | None
    verify_tls: bool
    timeout_seconds: int
    username: str | None = None
    password: str | None = None
    destination_name: str | None = None
    destination_service_uri: str | None = None
    destination_token_base_url: str | None = None
    destination_client_id: str | None = None
    destination_client_secret: str | None = None
    connectivity_token_base_url: str | None = None
    connectivity_client_id: str | None = None
    connectivity_client_secret: str | None = None
    connectivity_proxy_host: str | None = None
    connectivity_proxy_port: str | None = None

    @classmethod
    def from_env(cls) -> "S4PRConfig":
        verify = _parse_bool(os.getenv("S4_VERIFY"), default=True)
        timeout_raw = os.getenv("S4_TIMEOUT") or os.getenv("S4_TIMEOUT_SECONDS") or "60"
        try:
            timeout = max(5, int(timeout_raw))
        except ValueError:
            timeout = 60

        destination_name = _env("S4_DESTINATION_NAME")
        connection_mode = (_env("S4_CONNECTION_MODE") or "auto").lower()
        if connection_mode not in {"auto", "direct", "destination"}:
            raise S4PRError("S4_CONNECTION_MODE must be auto, direct, or destination.")
        running_in_cf = bool(os.getenv("VCAP_APPLICATION") or os.getenv("VCAP_SERVICES")) or os.getenv("APP_ENV") == "production"
        destination_configured = bool(
            destination_name and _env("DESTINATION_SERVICE_URI") and _env("DESTINATION_TOKEN_BASE_URL")
        )
        destination_mode = connection_mode == "destination" or (
            connection_mode == "auto" and running_in_cf and destination_configured
        )
        if destination_mode and not destination_configured:
            raise S4PRError(
                "S/4 Destination mode is not configured. Check S4_DESTINATION_NAME and Destination service credentials."
            )
        if destination_mode:
            return cls(
                base_url="",
                client=_env("S4_CLIENT"),
                verify_tls=verify,
                timeout_seconds=timeout,
                destination_name=destination_name,
                destination_service_uri=_env("DESTINATION_SERVICE_URI"),
                destination_token_base_url=_env("DESTINATION_TOKEN_BASE_URL"),
                destination_client_id=_env("DESTINATION_CLIENT_ID"),
                destination_client_secret=os.getenv("DESTINATION_CLIENT_SECRET"),
                connectivity_token_base_url=_env("CONNECTIVITY_TOKEN_BASE_URL"),
                connectivity_client_id=_env("CONNECTIVITY_CLIENT_ID"),
                connectivity_client_secret=os.getenv("CONNECTIVITY_CLIENT_SECRET"),
                connectivity_proxy_host=_env("CONNECTIVITY_PROXY_HOST"),
                connectivity_proxy_port=_env("CONNECTIVITY_PROXY_PORT") or "20003",
            )

        base_url = (_env("S4_BASE_URL") or "").rstrip("/")
        username = _env("S4_USERNAME")
        password = os.getenv("S4_PASSWORD") or None
        missing = [name for name, value in [("S4_BASE_URL", base_url), ("S4_USERNAME", username), ("S4_PASSWORD", password)] if not value]
        if missing:
            raise S4PRError(f"S/4 connection is not configured. Missing: {', '.join(missing)}")
        return cls(
            base_url=base_url,
            client=_env("S4_CLIENT"),
            verify_tls=verify,
            timeout_seconds=timeout,
            username=username,
            password=password,
        )

    @property
    def service_url(self) -> str:
        return self.base_url.rstrip("/") + PR_SERVICE_PATH

    @property
    def uses_destination(self) -> bool:
        return bool(self.destination_name)


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _sanitize_error_text(value: str) -> str:
    text = value or ""
    patterns = [
        (r'("access_token"\s*:\s*")[^"]+', r'\1***'),
        (r'("client_secret"\s*:\s*")[^"]+', r'\1***'),
        (r'("password"\s*:\s*")[^"]+', r'\1***'),
        (r'(?i)(authorization:\s*)([^\s]+)', r'\1***'),
        (r'(?i)(bearer\s+)([A-Za-z0-9\-\._~+/=]+)', r'\1***'),
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text[:1200]


def _json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _token_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/oauth/token"


def _get_oauth_token(base_url: str | None, client_id: str | None, client_secret: str | None, label: str) -> str:
    missing = [name for name, value in [(f"{label}_TOKEN_BASE_URL", base_url), (f"{label}_CLIENT_ID", client_id), (f"{label}_CLIENT_SECRET", client_secret)] if not value]
    if missing:
        raise S4PRError(f"BTP {label.lower()} credentials are incomplete. Missing: {', '.join(missing)}")
    try:
        response = requests.post(
            _token_url(str(base_url)),
            auth=HTTPBasicAuth(str(client_id), str(client_secret)),
            data={"grant_type": "client_credentials"},
            timeout=(10, 60),
        )
    except requests.RequestException as exc:
        raise S4PRError(f"Could not get BTP {label.lower()} token.", details=_sanitize_error_text(str(exc))) from exc
    if response.status_code >= 400:
        raise S4PRError(f"BTP {label.lower()} token request failed.", status_code=response.status_code, details=_sanitize_error_text(response.text))
    token = response.json().get("access_token")
    if not token:
        raise S4PRError(f"BTP {label.lower()} token response did not contain access_token.")
    return str(token)


def _extract_auth_header(destination_json: dict[str, Any]) -> dict[str, str]:
    tokens = destination_json.get("authTokens") or destination_json.get("authToken") or []
    if isinstance(tokens, dict):
        tokens = [tokens]
    for token in tokens:
        if not isinstance(token, dict):
            continue
        header_name = token.get("http_header", {}).get("key") if isinstance(token.get("http_header"), dict) else None
        header_value = token.get("http_header", {}).get("value") if isinstance(token.get("http_header"), dict) else None
        if header_name and header_value:
            return {str(header_name): str(header_value)}
        value = token.get("value")
        token_type = token.get("type") or token.get("tokenType") or "Bearer"
        if value:
            return {"Authorization": f"{token_type} {value}"}
    return {}


def _get_destination(config: S4PRConfig, token: str) -> dict[str, Any]:
    url = f"{config.destination_service_uri.rstrip('/')}/destination-configuration/v1/destinations/{config.destination_name}"
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=(10, 60))
    except requests.RequestException as exc:
        raise S4PRError("Could not read BTP Destination.", details=_sanitize_error_text(str(exc))) from exc
    if response.status_code >= 400:
        raise S4PRError("BTP Destination request failed.", status_code=response.status_code, details=_sanitize_error_text(response.text))
    return response.json()


def _destination_client(destination_json: dict[str, Any]) -> str | None:
    cfg = destination_json.get("destinationConfiguration") or {}
    for key, value in cfg.items():
        if str(key).lower() in {"sap-client", "sap_client", "sap.client"} and value:
            return str(value)
    return None


def _destination_request_context(config: S4PRConfig) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
    destination_token = _get_oauth_token(config.destination_token_base_url, config.destination_client_id, config.destination_client_secret, "DESTINATION")
    destination_json = _get_destination(config, destination_token)
    dest_cfg = destination_json.get("destinationConfiguration") or {}
    base_url = str(dest_cfg.get("URL") or dest_cfg.get("url") or "").rstrip("/")
    if not base_url:
        raise S4PRError("BTP Destination does not contain backend URL.")
    headers = {"Accept": "application/json"}
    headers.update(_extract_auth_header(destination_json))
    proxies = None
    proxy_type = str(dest_cfg.get("ProxyType") or dest_cfg.get("proxyType") or "")
    if proxy_type == "OnPremise":
        connectivity_token = _get_oauth_token(config.connectivity_token_base_url, config.connectivity_client_id, config.connectivity_client_secret, "CONNECTIVITY")
        if not config.connectivity_proxy_host or not config.connectivity_proxy_port:
            raise S4PRError("Connectivity proxy host/port is not configured for OnPremise destination.")
        proxy_url = f"http://{config.connectivity_proxy_host}:{config.connectivity_proxy_port}"
        proxies = {"http": proxy_url, "https": proxy_url}
        headers["Proxy-Authorization"] = f"Bearer {connectivity_token}"
    if "Authorization" not in headers:
        raise S4PRError("BTP Destination did not provide backend Authorization header.")
    client = config.client or _destination_client(destination_json)
    context = {
        "headers": headers,
        "proxies": proxies,
        "verify": config.verify_tls,
        "timeout": (10, config.timeout_seconds),
        "proxy_type": proxy_type or "Internet",
    }
    return base_url + PR_SERVICE_PATH, context, client, destination_json


def _direct_request_context(config: S4PRConfig) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
    token = requests.auth._basic_auth_str(config.username or "", config.password or "")
    return config.service_url, {
        "headers": {"Accept": "application/json", "Authorization": token},
        "proxies": None,
        "verify": config.verify_tls,
        "timeout": (10, config.timeout_seconds),
        "proxy_type": "Direct",
    }, config.client, {}


def _request_context(config: S4PRConfig) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
    if config.uses_destination:
        return _destination_request_context(config)
    return _direct_request_context(config)


def _params(client: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    params = {"$format": "json"}
    if client:
        params["sap-client"] = client
    if extra:
        params.update(extra)
    return params


def preflight_purchase_requisition_api() -> dict[str, Any]:
    """Check metadata and CSRF without creating anything."""

    config = S4PRConfig.from_env()
    service_url, request_config, client, _destination = _request_context(config)
    session = requests.Session()
    result: dict[str, Any] = {
        "configured": True,
        "connection_mode": "BTP Destination" if config.uses_destination else "Direct Basic Auth",
        "destination_name": config.destination_name,
        "service": "API_PURCHASEREQ_PROCESS_SRV",
        "entity_set": PR_HEADER_ENTITY_SET,
        "sap_client_configured": bool(client),
        "verify_tls": config.verify_tls,
        "proxy_type": request_config.get("proxy_type"),
    }

    try:
        metadata_url = f"{service_url}/$metadata"
        start = time.perf_counter()
        metadata_response = session.get(
            metadata_url,
            params={"sap-client": client} if client else None,
            headers={**request_config["headers"], "Accept": "application/xml"},
            proxies=request_config["proxies"],
            verify=request_config["verify"],
            timeout=request_config["timeout"],
        )
        result["metadata_status"] = metadata_response.status_code
        result["metadata_latency_ms"] = int((time.perf_counter() - start) * 1000)
        result["metadata_available"] = metadata_response.status_code < 400 and "A_PurchaseRequisitionHeader" in metadata_response.text
        if metadata_response.status_code >= 400:
            result["metadata_error"] = _sanitize_error_text(metadata_response.text)

        headers = {**request_config["headers"], "X-CSRF-Token": "Fetch", "Accept": "application/json"}
        start = time.perf_counter()
        csrf_response = session.get(
            f"{service_url}/{PR_HEADER_ENTITY_SET}",
            params=_params(client, {"$top": "0"}),
            headers=headers,
            proxies=request_config["proxies"],
            verify=request_config["verify"],
            timeout=request_config["timeout"],
        )
        result["csrf_status"] = csrf_response.status_code
        result["csrf_latency_ms"] = int((time.perf_counter() - start) * 1000)
        result["csrf_available"] = bool(csrf_response.headers.get("X-CSRF-Token")) and csrf_response.status_code < 400
        result["cookies_received"] = bool(csrf_response.cookies)
        if csrf_response.status_code >= 400:
            result["csrf_error"] = _sanitize_error_text(csrf_response.text)
    except requests.RequestException as exc:
        logger.exception("S/4 PR preflight failed")
        raise S4PRError("S/4 PR preflight failed.", details=_sanitize_error_text(str(exc))) from exc
    return result


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _to_decimal_string(value: Any, default: str | None = None) -> str | None:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float, Decimal)):
        return format(Decimal(str(value)).normalize(), "f")
    cleaned = str(value).strip().replace(",", "")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if cleaned in {"", ".", "-"}:
        return default
    try:
        return format(Decimal(cleaned).normalize(), "f")
    except InvalidOperation:
        return default


def _decimal_or_none(value: Any) -> Decimal | None:
    text = _to_decimal_string(value)
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _unit_price_from_line_total(line_total: Any, quantity: Any) -> str | None:
    total = _decimal_or_none(line_total)
    qty = _decimal_or_none(quantity)
    if total is None or qty in (None, Decimal("0")):
        return None
    try:
        return str((total / qty).normalize())
    except InvalidOperation:
        return None


def _to_text(value: Any, *, max_len: int | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if max_len and len(text) > max_len:
        return text[:max_len]
    return text


def _safe_sap_text(value: Any, *, max_len: int = 40, fallback: str = "Quote-based PR test item") -> str:
    text = _to_text(value, max_len=max_len) or fallback
    # Keep S/4 short text conservative. Some punctuation from quotes can fail EDM facet validation.
    text = re.sub(r"[^A-Za-z0-9 -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return (text or fallback)[:max_len]


def _parse_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _sap_date(value: Any, *, fallback_days: int = 14) -> str:
    parsed = _parse_date(value) or (dt.date.today() + dt.timedelta(days=fallback_days))
    timestamp = dt.datetime(parsed.year, parsed.month, parsed.day, tzinfo=dt.timezone.utc).timestamp()
    return f"/Date({int(timestamp * 1000)})/"


def _env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return fallback
    return value.strip()


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _line_items(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    items = normalized.get("line_items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _normalize_sap_uom(value: Any, *, each_unit: str = "PC") -> tuple[str, bool]:
    """Return an S/4 unit code and whether a business alias was replaced."""

    raw = _to_text(value, max_len=12).upper().replace(".", "").strip()
    if not raw:
        return each_unit, True
    each_aliases = {"EA", "EAC", "EACH", "PC", "PCS", "PCE", "PIECE", "PIECES", "UNIT", "UNITS"}
    if raw in each_aliases:
        normalized = each_unit.upper()
        return normalized, normalized != raw
    return raw[:3], raw[:3] != raw


def build_pr_payload(normalized: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an S/4 PR OData payload plus readiness diagnostics."""

    overrides = overrides or {}
    header = normalized.get("header") if isinstance(normalized.get("header"), dict) else {}
    pr_mapping = normalized.get("pr_mapping") if isinstance(normalized.get("pr_mapping"), dict) else {}
    source_items = _line_items(normalized)
    if not source_items:
        source_items = [{"description": _first_non_empty(header.get("notes"), header.get("quote_number"), "Quote line item"), "quantity": 1}]

    document_type = _to_text(_first_non_empty(overrides.get("document_type"), pr_mapping.get("purchase_requisition_type")), max_len=4) or "NB"
    currency = _to_text(_first_non_empty(overrides.get("currency"), header.get("currency"), _env_default("S4_PR_DEFAULT_CURRENCY", "USD")), max_len=5) or "USD"
    supplier = _to_text(_first_non_empty(overrides.get("supplier"), _env_default("S4_PR_DEFAULT_SUPPLIER", "")), max_len=10)
    plant = _to_text(_first_non_empty(overrides.get("plant"), pr_mapping.get("plant"), _env_default("S4_PR_DEFAULT_PLANT", "1710")), max_len=4)
    purchasing_org = _to_text(_first_non_empty(overrides.get("purchasing_org"), pr_mapping.get("purchasing_org"), _env_default("S4_PR_DEFAULT_PURCHASING_ORG", "")), max_len=4)
    purchasing_group = _to_text(_first_non_empty(overrides.get("purchasing_group"), pr_mapping.get("purchasing_group"), _env_default("S4_PR_DEFAULT_PURCHASING_GROUP", "001")), max_len=3)
    company_code = _to_text(_first_non_empty(overrides.get("company_code"), _env_default("S4_PR_DEFAULT_COMPANY_CODE", "1710")), max_len=4)
    material_group = _to_text(_first_non_empty(overrides.get("material_group"), _env_default("S4_PR_DEFAULT_MATERIAL_GROUP", "YBPM01")), max_len=9)
    default_material = _to_text(_first_non_empty(overrides.get("material"), pr_mapping.get("sap_material_id"), _env_default("S4_PR_DEFAULT_MATERIAL", "SP002")), max_len=40)
    fallback_uom = _to_text(_first_non_empty(overrides.get("base_unit"), _env_default("S4_PR_DEFAULT_BASE_UNIT", "PC")), max_len=3) or "PC"
    fallback_unit_price = _to_decimal_string(_first_non_empty(overrides.get("unit_price"), _env_default("S4_PR_DEFAULT_UNIT_PRICE", "130.50")), default="130.50")
    account_assignment = _to_text(_first_non_empty(overrides.get("account_assignment_category"), pr_mapping.get("account_assignment_category"), _env_default("S4_PR_DEFAULT_ACCOUNT_ASSIGNMENT_CATEGORY")), max_len=1)
    default_delivery_date = _first_non_empty(overrides.get("delivery_date"), pr_mapping.get("need_by_date"))
    requisitioner = _to_text(_first_non_empty(overrides.get("requisitioner_name"), pr_mapping.get("requested_by"), header.get("requester_name"), header.get("buyer_contact")), max_len=12)
    source_determination = _boolish(overrides.get("source_determination"), default=False)
    validation_only = _boolish(overrides.get("validation_only"), default=False)
    raw_line_overrides = overrides.get("line_items")
    line_overrides = [item for item in raw_line_overrides if isinstance(item, dict)] if isinstance(raw_line_overrides, list) else []

    def line_override_for(index: int) -> dict[str, Any]:
        for position, item_override in enumerate(line_overrides):
            configured_index = item_override.get("index")
            if configured_index is None and position == index:
                return item_override
            try:
                if configured_index is not None and int(configured_index) == index:
                    return item_override
            except (TypeError, ValueError):
                continue
        return {}

    missing_header = [name for name, value in {"Plant": plant, "Purchasing Group": purchasing_group}.items() if not value]
    items: list[dict[str, Any]] = []
    missing_by_item: list[dict[str, Any]] = []
    used_fallback_uom = False
    uom_normalizations: list[dict[str, Any]] = []
    used_fallback_unit_price = False
    used_fallback_delivery_date = False
    for index, source in enumerate(source_items, start=1):
        item_override = line_override_for(index - 1)
        material = _to_text(
            _first_non_empty(
                item_override.get("material"),
                source.get("sap_material_id"),
                source.get("material"),
                default_material,
            ),
            max_len=40,
        )
        item_material_group = _to_text(
            _first_non_empty(item_override.get("material_group"), material_group),
            max_len=9,
        )
        description = _safe_sap_text(_first_non_empty(source.get("description"), source.get("service_description"), header.get("notes"), "Quote-based PR test item"), max_len=40)
        quantity = _to_decimal_string(_first_non_empty(overrides.get("default_quantity"), source.get("quantity")), default="1")
        source_uom = source.get("unit_of_measure")
        requested_uom = _first_non_empty(item_override.get("base_unit"), overrides.get("base_unit"), source_uom, fallback_uom)
        uom, uom_was_normalized = _normalize_sap_uom(requested_uom, each_unit=fallback_uom)
        used_fallback_uom = used_fallback_uom or (
            not item_override.get("base_unit") and not overrides.get("base_unit") and (not source_uom or uom_was_normalized)
        )
        if uom_was_normalized:
            uom_normalizations.append({"item": index * 10, "source": requested_uom, "sap_value": uom})
        derived_unit_price = _unit_price_from_line_total(source.get("line_total"), quantity)
        single_line_total = header.get("total_amount") if len(source_items) == 1 else None
        source_unit_price = _first_non_empty(source.get("unit_price"), derived_unit_price, single_line_total)
        unit_price = _to_decimal_string(_first_non_empty(source_unit_price, fallback_unit_price), default=fallback_unit_price)
        used_fallback_unit_price = used_fallback_unit_price or not bool(source_unit_price)
        delivery_date = _first_non_empty(source.get("expected_delivery_date"), default_delivery_date)
        used_fallback_delivery_date = used_fallback_delivery_date or not bool(delivery_date)
        supplier_material = _to_text(_first_non_empty(source.get("vendor_material_number"), source.get("manufacturer_part_number")), max_len=35)

        item: dict[str, Any] = {
            "PurchaseRequisitionItem": str(index * 10),
            "PurchaseRequisitionType": document_type,
            "PurchaseRequisitionItemText": description,
            "RequestedQuantity": quantity,
            "BaseUnit": uom,
            "PurchaseRequisitionPrice": unit_price,
            "PurReqnPriceQuantity": "1",
            "DeliveryDate": _sap_date(delivery_date),
            "PurReqnItemCurrency": currency,
            "MaterialGroup": item_material_group,
            "PurchasingGroup": purchasing_group,
            "Plant": plant,
            "CompanyCode": company_code,
        }
        optional_fields = {
            "Material": material,
            "PurchasingOrganization": purchasing_org,
            "Supplier": supplier,
            "FixedSupplier": supplier,
            "RequisitionerName": requisitioner,
            "SupplierMaterialNumber": supplier_material,
        }
        item.update({key: value for key, value in optional_fields.items() if value})
        if not material and account_assignment:
            item["AccountAssignmentCategory"] = account_assignment
        items.append(item)

        missing_item_fields: list[str] = []
        if not description and not material:
            missing_item_fields.append("Material or item text")
        if not quantity:
            missing_item_fields.append("Requested quantity")
        if not uom:
            missing_item_fields.append("Base unit")
        if not material and not account_assignment:
            missing_item_fields.append("Material or account assignment category")
        if missing_item_fields:
            missing_by_item.append({"item": index * 10, "missing_fields": missing_item_fields})

    quote_label = " ".join(str(part) for part in [header.get("quote_number"), header.get("vendor_name")] if part)
    description = _to_text(_first_non_empty(overrides.get("description"), quote_label, "Quote-based PR"), max_len=40) or "Quote-based PR"
    payload: dict[str, Any] = {
        "PurchaseRequisitionType": document_type,
        "PurReqnDescription": description,
        "SourceDetermination": source_determination,
        "to_PurchaseReqnItem": {"results": items},
    }
    if validation_only:
        payload["PurReqnDoOnlyValidation"] = True

    missing_fields = missing_header + [f"Item {entry['item']}: {', '.join(entry['missing_fields'])}" for entry in missing_by_item]
    readiness_score = max(0, 100 - len(missing_fields) * 12)
    return {
        "ready_for_create": not missing_fields,
        "readiness_score": readiness_score,
        "missing_fields": missing_fields,
        "payload": payload,
        "source_summary": {
            "vendor_name": header.get("vendor_name"),
            "quote_number": header.get("quote_number"),
            "currency": currency,
            "line_item_count": len(items),
            "uom_normalizations": uom_normalizations,
            "defaulted_fields": {
                "material": bool(
                    default_material
                    and not line_overrides
                    and not _first_non_empty(overrides.get("material"), pr_mapping.get("sap_material_id"))
                ),
                "plant": bool(plant and not _first_non_empty(overrides.get("plant"), pr_mapping.get("plant"))),
                "purchasing_group": bool(purchasing_group and not _first_non_empty(overrides.get("purchasing_group"), pr_mapping.get("purchasing_group"))),
                "company_code": bool(company_code and not overrides.get("company_code")),
                "material_group": bool(material_group and not line_overrides and not overrides.get("material_group")),
                "base_unit": used_fallback_uom,
                "unit_price": used_fallback_unit_price,
                "delivery_date": used_fallback_delivery_date,
            },
        },
        "notes": [
            "Payload is prepared for SAP API_PURCHASEREQ_PROCESS_SRV / A_PurchaseRequisitionHeader.",
            "Prototype constants are used where quote extraction does not provide SAP master data.",
        ],
    }


def _fetch_csrf(session: requests.Session, service_url: str, request_config: dict[str, Any], client: str | None) -> tuple[str, requests.cookies.RequestsCookieJar]:
    headers = {**request_config["headers"], "X-CSRF-Token": "Fetch", "Accept": "application/json"}
    response = session.get(
        f"{service_url}/{PR_HEADER_ENTITY_SET}",
        params=_params(client, {"$top": "0"}),
        headers=headers,
        proxies=request_config["proxies"],
        verify=request_config["verify"],
        timeout=request_config["timeout"],
    )
    if response.status_code >= 400:
        raise S4PRError("Could not fetch S/4 CSRF token.", status_code=response.status_code, details=_sanitize_error_text(response.text))
    token = response.headers.get("X-CSRF-Token")
    if not token:
        raise S4PRError("S/4 did not return a CSRF token.", status_code=response.status_code)
    return token, response.cookies


def create_purchase_requisition(payload: dict[str, Any]) -> dict[str, Any]:
    config = S4PRConfig.from_env()
    service_url, request_config, client, _destination = _request_context(config)
    session = requests.Session()
    token, cookies = _fetch_csrf(session, service_url, request_config, client)
    headers = {
        **request_config["headers"],
        "X-CSRF-Token": token,
        "X-Requested-With": "XMLHttpRequest",
        "Prefer": "return=representation",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        start = time.perf_counter()
        response = session.post(
            f"{service_url}/{PR_HEADER_ENTITY_SET}",
            params={"sap-client": client} if client else None,
            headers=headers,
            cookies=cookies,
            proxies=request_config["proxies"],
            verify=request_config["verify"],
            timeout=request_config["timeout"],
            json=payload,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
    except requests.RequestException as exc:
        logger.exception("S/4 PR creation request failed")
        raise S4PRError("S/4 PR creation request failed.", details=_sanitize_error_text(str(exc))) from exc

    body = _json_or_text(response)
    if response.status_code >= 400:
        raise S4PRError("S/4 rejected the purchase requisition payload.", status_code=response.status_code, details=_sanitize_error_text(response.text))
    created_pr = None
    if isinstance(body, dict):
        data = body.get("d") if isinstance(body.get("d"), dict) else body
        created_pr = data.get("PurchaseRequisition") if isinstance(data, dict) else None
    if not created_pr:
        entity_location = response.headers.get("OData-EntityId") or response.headers.get("Location") or ""
        match = re.search(r"A_PurchaseRequisitionHeader\('([^']+)'\)", entity_location)
        if match:
            created_pr = match.group(1)
    if not created_pr:
        raise S4PRError(
            "S/4 accepted the request but did not return a Purchase Requisition number.",
            status_code=response.status_code,
            details=_sanitize_error_text(str(body)),
        )
    return {
        "status": "created",
        "http_status": response.status_code,
        "latency_ms": elapsed_ms,
        "connection_mode": "BTP Destination" if config.uses_destination else "Direct Basic Auth",
        "purchase_requisition": created_pr,
        "response": body,
    }


def create_purchase_requisition_for_poc(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a PR while keeping nonessential demo master-data gaps nonblocking."""

    try:
        return create_purchase_requisition(payload)
    except S4PRError as exc:
        error_text = f"{exc} {exc.details or ''}"
        if not re.search(
            r"Supplier\s+\S+\s+(?:is\s+)?(?:not yet created by|not maintained for)\s+purchasing organization",
            error_text,
            re.IGNORECASE,
        ):
            raise

        retry_payload = copy.deepcopy(payload)
        items = retry_payload.get("to_PurchaseReqnItem", {}).get("results", [])
        removed_suppliers = sorted({str(item.get("FixedSupplier") or item.get("Supplier") or "") for item in items if item.get("FixedSupplier") or item.get("Supplier")})
        for item in items:
            item.pop("Supplier", None)
            item.pop("FixedSupplier", None)

        created = create_purchase_requisition(retry_payload)
        created["poc_adjustments"] = [
            {
                "code": "supplier_not_extended_for_purchasing_org",
                "message": "PR created without a fixed supplier; the matched supplier remains a recommendation.",
                "suppliers": removed_suppliers,
            }
        ]
        return created
