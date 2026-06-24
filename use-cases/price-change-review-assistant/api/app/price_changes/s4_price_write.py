from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from .s4_lookup import (
    INFORECORD_SERVICE_NAME,
    S4HTTPError,
    S4LookupRepository,
    clean_optional,
    normalize_material_code,
    odata_entity,
    odata_rows,
    odata_str,
)


PURCHASING_CONDITION_RECORD_SERVICE_NAME = "API_PURGPRCGCONDITIONRECORD_SRV"
DEFAULT_CONDITION_TYPE = "PPR0"
DEFAULT_PURCHASING_ORGANIZATION = "1010"
DEFAULT_INFO_RECORD_CATEGORY = "0"
DEFAULT_OPEN_ENDED_VALID_TO = "9999-12-31"
DEFAULT_CONDITION_QUANTITY = "1"
GENERAL_CONDITION_TABLE = "018"
PLANT_CONDITION_TABLE = "017"
SUCCESS_STATUSES = {"created_condition", "delimited_and_created"}


class PriceConditionError(ValueError):
    """Raised when a reviewed draft cannot be converted into an S/4 price write."""


@dataclass(frozen=True)
class SupplierPriceContext:
    """S/4 context needed to price one supplier/material combination.

    Args:
        supplier_id: Local supplier identifier.
        supplier: S/4 supplier account number.
        purchasing_organization: Purchasing organization used for condition lookup.
        plant: Optional plant. Blank means general purchasing price.
        info_record_category: Purchasing info-record category.
    """

    supplier_id: str
    supplier: str
    purchasing_organization: str
    plant: str = ""
    info_record_category: str = DEFAULT_INFO_RECORD_CATEGORY


@dataclass(frozen=True)
class PriceConditionConfig:
    """Configuration for PPR0 purchase price condition writes.

    Args:
        condition_type: Purchasing condition type to maintain.
        open_ended_valid_to: Date used when a draft has no explicit end date.
        condition_quantity: Pricing quantity for the condition value.
    """

    condition_type: str = DEFAULT_CONDITION_TYPE
    open_ended_valid_to: str = DEFAULT_OPEN_ENDED_VALID_TO
    condition_quantity: str = DEFAULT_CONDITION_QUANTITY


@dataclass(frozen=True)
class PriceChangeRequest:
    """Reviewed supplier-material price change request.

    Args:
        supplier_id: Local supplier identifier.
        material_code: S/4 material/product code.
        new_price: New condition rate value.
        currency: ISO currency code.
        uom: Unit of measure for the condition quantity.
        effective_from: Inclusive validity start date in ISO format.
        effective_to: Inclusive validity end date in ISO format.
    """

    supplier_id: str
    material_code: str
    new_price: str
    currency: str
    uom: str
    effective_from: str
    effective_to: str


@dataclass(frozen=True)
class BatchPayload:
    """Multipart OData batch request body and content type.

    Args:
        body: Complete multipart `$batch` request body.
        content_type: HTTP `Content-Type` header including the boundary.
        batch_boundary: Batch boundary string.
        changeset_boundary: Changeset boundary string.
    """

    body: str
    content_type: str
    batch_boundary: str
    changeset_boundary: str


def date_to_odata_datetime(date_str: str) -> str:
    """Convert an ISO date to the OData datetime literal body value.

    Args:
        date_str: ISO date.

    Returns:
        OData datetime string at midnight.
    """
    return f"{date_str}T00:00:00"


def choose_condition_table(plant: str | None) -> str:
    """Choose the purchasing condition table based on plant scope.

    Args:
        plant: Optional S/4 plant.

    Returns:
        `018` for general supplier/material prices, `017` for plant-specific prices.
    """
    return PLANT_CONDITION_TABLE if clean_optional(plant) else GENERAL_CONDITION_TABLE


def parse_s4_date(value: Any) -> date | None:
    """Parse an S/4 JSON date field into a Python date.

    Args:
        value: OData date value such as `/Date(253402214400000)/` or ISO datetime text.

    Returns:
        Parsed date or `None` when the value is blank.
    """
    text = clean_optional(value)
    if text is None:
        return None
    if text.startswith("/Date(") and text.endswith(")/"):
        millis = int(text[len("/Date(") : -len(")/")])
        if millis >= 253402214400000:
            return date(9999, 12, 31)
        return datetime.fromtimestamp(millis / 1000, tz=UTC).date()
    return date.fromisoformat(text[:10])


def normalize_price(value: Any) -> str:
    """Normalize a reviewed amount to a two-decimal S/4 condition value.

    Args:
        value: Raw amount from HANA or the API.

    Returns:
        Amount formatted with two decimal places.
    """
    try:
        return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as exc:
        raise PriceConditionError(f"Invalid requested price: {value!r}") from exc


def require_text(draft: dict[str, Any], field_name: str) -> str:
    """Read a required draft field as stripped text.

    Args:
        draft: Draft row.
        field_name: Required field name.

    Returns:
        Non-empty field value.
    """
    value = clean_optional(draft.get(field_name))
    if value is None:
        raise PriceConditionError(f"{field_name} is required")
    return value


def build_price_request_from_draft(
    draft: dict[str, Any],
    config: PriceConditionConfig | None = None,
) -> PriceChangeRequest:
    """Convert a reviewed HANA draft into a PPR0 write request.

    Args:
        draft: Reviewed draft row.
        config: Optional condition defaults.

    Returns:
        PriceChangeRequest for S/4.
    """
    active_config = config or PriceConditionConfig()
    return PriceChangeRequest(
        supplier_id=require_text(draft, "supplier_id"),
        material_code=normalize_material_code(require_text(draft, "material_code")),
        new_price=normalize_price(require_text(draft, "requested_new_price")),
        currency=require_text(draft, "currency").upper(),
        uom=require_text(draft, "uom").upper(),
        effective_from=require_text(draft, "effective_from"),
        effective_to=clean_optional(draft.get("effective_to")) or active_config.open_ended_valid_to,
    )


def build_supplier_context(
    supplier: dict[str, Any],
    supplier_id: str,
    config: PriceConditionConfig | None = None,
) -> SupplierPriceContext:
    """Convert an S/4 supplier lookup row into pricing context.

    Args:
        supplier: S/4 supplier candidate from S4LookupRepository.
        supplier_id: Local supplier id from the reviewed draft.
        config: Optional condition defaults.

    Returns:
        SupplierPriceContext for PPR0 write calls.
    """
    _config = config or PriceConditionConfig()
    supplier_number = clean_optional(supplier.get("supplier"))
    if not supplier_number:
        raise PriceConditionError("S/4 supplier number is required")
    return SupplierPriceContext(
        supplier_id=clean_optional(supplier.get("supplier_id")) or supplier_id,
        supplier=supplier_number,
        purchasing_organization=clean_optional(supplier.get("purchasing_organization"))
        or DEFAULT_PURCHASING_ORGANIZATION,
        plant="",
        info_record_category=DEFAULT_INFO_RECORD_CATEGORY,
    )


def build_info_record_filter(request: PriceChangeRequest, context: SupplierPriceContext) -> str:
    """Build an info-record org/plant lookup filter.

    Args:
        request: Reviewed price-change request.
        context: Supplier pricing context.

    Returns:
        OData `$filter` expression.
    """
    return " and ".join(
        [
            f"Supplier eq {odata_str(context.supplier)}",
            f"Material eq {odata_str(request.material_code)}",
            f"PurchasingOrganization eq {odata_str(context.purchasing_organization)}",
            f"PurchasingInfoRecordCategory eq {odata_str(context.info_record_category)}",
            f"Plant eq {odata_str(context.plant)}",
        ]
    )


def build_condition_validity_filter(
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> str:
    """Build a condition-validity lookup filter for PPR0 rows.

    Args:
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        OData `$filter` expression.
    """
    filters = [
        f"ConditionType eq {odata_str(config.condition_type.strip().upper())}",
        f"Supplier eq {odata_str(context.supplier)}",
        f"Material eq {odata_str(request.material_code)}",
        f"PurchasingOrganization eq {odata_str(context.purchasing_organization)}",
        f"PurchasingInfoRecordCategory eq {odata_str(context.info_record_category)}",
    ]
    if context.plant:
        filters.append(f"Plant eq {odata_str(context.plant)}")
    return " and ".join(filters)


def build_condition_create_payload(
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> dict[str, Any]:
    """Build an `A_PurgPrcgConditionRecord` create payload.

    Args:
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        JSON payload for `API_PURGPRCGCONDITIONRECORD_SRV`.
    """
    validity: dict[str, Any] = {
        "ConditionValidityEndDate": date_to_odata_datetime(request.effective_to),
        "ConditionValidityStartDate": date_to_odata_datetime(request.effective_from),
        "ConditionType": config.condition_type.strip().upper(),
        "Supplier": context.supplier,
        "Material": request.material_code,
        "PurchasingOrganization": context.purchasing_organization,
        "PurchasingInfoRecordCategory": context.info_record_category,
    }
    if context.plant:
        validity["Plant"] = context.plant
    return {
        "ConditionTable": choose_condition_table(context.plant),
        "ConditionType": config.condition_type.strip().upper(),
        "ConditionRateValue": request.new_price,
        "ConditionRateValueUnit": request.currency,
        "ConditionQuantity": config.condition_quantity,
        "ConditionQuantityUnit": request.uom,
        "to_PurgPrcgCndnRecdValidity": [validity],
    }


def find_info_record_rows(client: Any, request: PriceChangeRequest, context: SupplierPriceContext) -> list[dict[str, Any]]:
    """Read matching purchasing info-record org/plant rows.

    Args:
        client: Authenticated S/4 client.
        request: Reviewed price-change request.
        context: Supplier pricing context.

    Returns:
        Matching info-record rows.
    """
    payload, _headers = client.get_json(
        INFORECORD_SERVICE_NAME,
        path="/A_PurgInfoRecdOrgPlantData",
        params={
            "$format": "json",
            "$select": (
                "PurchasingInfoRecord,PurchasingInfoRecordCategory,PurchasingOrganization,"
                "Plant,Supplier,Material,NetPriceAmount,Currency,PurgDocOrderQuantityUnit"
            ),
            "$filter": build_info_record_filter(request, context),
        },
    )
    return odata_rows(payload)


def find_condition_validity_rows(
    client: Any,
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> list[dict[str, Any]]:
    """Read matching PPR0 condition validity rows.

    Args:
        client: Authenticated S/4 client.
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        Matching condition-validity rows.
    """
    payload, _headers = client.get_json(
        INFORECORD_SERVICE_NAME,
        path="/A_PurInfoRecdPrcgCndnValidity",
        params={
            "$format": "json",
            "$select": (
                "ConditionRecord,ConditionType,ConditionValidityStartDate,ConditionValidityEndDate,"
                "PurchasingOrganization,PurchasingInfoRecordCategory,Supplier,Material,Plant"
            ),
            "$filter": build_condition_validity_filter(request, context, config),
        },
    )
    return odata_rows(payload)


def get_condition_record(client: Any, condition_record: str) -> dict[str, Any]:
    """Read one purchasing pricing condition detail row.

    Args:
        client: Authenticated S/4 client.
        condition_record: S/4 condition record id.

    Returns:
        Condition detail row.
    """
    payload, _headers = client.get_json(
        INFORECORD_SERVICE_NAME,
        path=f"/A_PurInfoRecdPrcgCndn({odata_str(condition_record)})",
        params={"$format": "json"},
    )
    return odata_entity(payload)


def is_open_ended_validity(row: dict[str, Any]) -> bool:
    """Check whether a condition validity row ends on `9999-12-31`.

    Args:
        row: Condition validity row.

    Returns:
        True when the validity is open-ended.
    """
    return parse_s4_date(row.get("ConditionValidityEndDate")) == date(9999, 12, 31)


def validity_overlaps_request(row: dict[str, Any], request: PriceChangeRequest) -> bool:
    """Determine whether an existing validity row overlaps the requested window.

    Args:
        row: Existing condition validity row.
        request: Requested price-change window.

    Returns:
        True when the ranges overlap.
    """
    existing_start = parse_s4_date(row.get("ConditionValidityStartDate"))
    existing_end = parse_s4_date(row.get("ConditionValidityEndDate"))
    if existing_start is None or existing_end is None:
        return False
    requested_start = date.fromisoformat(request.effective_from)
    requested_end = date.fromisoformat(request.effective_to)
    return existing_start <= requested_end and existing_end >= requested_start


def overlapping_validities(rows: list[dict[str, Any]], request: PriceChangeRequest) -> list[dict[str, Any]]:
    """Filter condition validity rows to those overlapping a request.

    Args:
        rows: Existing condition validity rows.
        request: Requested price-change window.

    Returns:
        Overlapping rows.
    """
    return [row for row in rows if validity_overlaps_request(row, request)]


def delimit_end_date_for_request(request: PriceChangeRequest) -> str:
    """Compute the end date for an old open-ended row.

    Args:
        request: Requested price-change window.

    Returns:
        ISO date for the day before the new validity starts.
    """
    return (date.fromisoformat(request.effective_from) - timedelta(days=1)).isoformat()


def build_validity_entity_path(condition_record: str, validity_end_date: str) -> str:
    """Build the condition-validity key path used for PATCH.

    Args:
        condition_record: S/4 condition record id.
        validity_end_date: Existing validity end date in ISO format.

    Returns:
        OData entity path.
    """
    return (
        "/A_PurgPrcgCndnRecdValidity("
        f"ConditionRecord={odata_str(condition_record)},"
        f"ConditionValidityEndDate=datetime{odata_str(date_to_odata_datetime(validity_end_date))})"
    )


def build_delimit_and_create_batch(
    existing_validity: dict[str, Any],
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> BatchPayload:
    """Build one OData `$batch` changeset to delimit and create PPR0 rows.

    Args:
        existing_validity: Existing open-ended condition validity row.
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        Multipart batch payload.
    """
    batch_boundary = f"batch_{uuid.uuid4().hex}"
    changeset_boundary = f"changeset_{uuid.uuid4().hex}"
    existing_end = parse_s4_date(existing_validity.get("ConditionValidityEndDate"))
    condition_record = clean_optional(existing_validity.get("ConditionRecord"))
    if existing_end is None or condition_record is None:
        raise PriceConditionError("existing validity row must include ConditionRecord and ConditionValidityEndDate")
    delimit_to = delimit_end_date_for_request(request)
    patch_path = build_validity_entity_path(condition_record, existing_end.isoformat()).lstrip("/")
    patch_payload = {"ConditionValidityEndDate": date_to_odata_datetime(delimit_to)}
    create_payload = build_condition_create_payload(request, context, config)
    body = "\r\n".join(
        [
            f"--{batch_boundary}",
            f"Content-Type: multipart/mixed; boundary={changeset_boundary}",
            "",
            f"--{changeset_boundary}",
            "Content-Type: application/http",
            "Content-Transfer-Encoding: binary",
            "",
            f"PATCH {patch_path} HTTP/1.1",
            "Content-Type: application/json",
            "Accept: application/json",
            "",
            json.dumps(patch_payload, ensure_ascii=True),
            f"--{changeset_boundary}",
            "Content-Type: application/http",
            "Content-Transfer-Encoding: binary",
            "",
            "POST A_PurgPrcgConditionRecord HTTP/1.1",
            "Content-Type: application/json",
            "Accept: application/json",
            "",
            json.dumps(create_payload, ensure_ascii=True),
            f"--{changeset_boundary}--",
            f"--{batch_boundary}--",
            "",
        ]
    )
    return BatchPayload(
        body=body,
        content_type=f"multipart/mixed; boundary={batch_boundary}",
        batch_boundary=batch_boundary,
        changeset_boundary=changeset_boundary,
    )


def parse_s4_error_body(exc: S4HTTPError) -> dict[str, Any] | str | None:
    """Parse an S/4 error body for reports.

    Args:
        exc: S/4 HTTP exception.

    Returns:
        Parsed JSON, raw text, or `None`.
    """
    if not exc.body:
        return None
    try:
        return json.loads(exc.body)
    except json.JSONDecodeError:
        return exc.body


def service_unavailable_result(exc: S4HTTPError) -> dict[str, Any]:
    """Build a clear result for inactive condition-record service errors.

    Args:
        exc: S/4 HTTP exception.

    Returns:
        JSON-serializable service unavailable result.
    """
    return {
        "status": "service_unavailable",
        "service": PURCHASING_CONDITION_RECORD_SERVICE_NAME,
        "condition_type": DEFAULT_CONDITION_TYPE,
        "message": (
            "S/4 price update could not be applied: API_PURGPRCGCONDITIONRECORD_SRV is unavailable. "
            "Ask the SAP team to activate SAP_COM_0294."
        ),
        "http_status": exc.status_code,
        "api_error_body": parse_s4_error_body(exc),
    }


def api_failure_result(status: str, exc: S4HTTPError, **extra: Any) -> dict[str, Any]:
    """Build a standard API failure report.

    Args:
        status: Report status.
        exc: S/4 HTTP exception.
        extra: Extra report fields.

    Returns:
        JSON-serializable failure result.
    """
    return {
        "status": status,
        "message": str(exc),
        "http_status": exc.status_code,
        "api_error_body": parse_s4_error_body(exc),
        **extra,
    }


def status_for_condition_api_error(exc: S4HTTPError) -> dict[str, Any]:
    """Convert a condition API exception into a status report.

    Args:
        exc: S/4 HTTP exception.

    Returns:
        Service-unavailable or failed-API status.
    """
    if exc.status_code in {403, 404}:
        return service_unavailable_result(exc)
    return api_failure_result("failed_api", exc)


def extract_condition_record_id(response: dict[str, Any]) -> str | None:
    """Extract a condition record id from an S/4 create response.

    Args:
        response: Parsed S/4 response.

    Returns:
        Condition record id or `None`.
    """
    data = response.get("d", response)
    record_id = data.get("ConditionRecord") if isinstance(data, dict) else None
    return str(record_id) if record_id else None


def expected_readback_values(
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> dict[str, str]:
    """Build expected values for readback verification.

    Args:
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        Expected S/4 field values.
    """
    return {
        "ConditionType": config.condition_type.strip().upper(),
        "Supplier": context.supplier,
        "Material": request.material_code,
        "PurchasingOrganization": context.purchasing_organization,
        "PurchasingInfoRecordCategory": context.info_record_category,
        "ConditionValidityStartDate": request.effective_from,
        "ConditionValidityEndDate": request.effective_to,
        "ConditionRateValue": request.new_price,
        "ConditionRateValueUnit": request.currency,
    }


def compare_readback(
    validity: dict[str, Any],
    condition_record: dict[str, Any],
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
) -> dict[str, dict[str, str | None]]:
    """Compare S/4 readback values against the requested price change.

    Args:
        validity: Condition validity row.
        condition_record: Condition detail row.
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.

    Returns:
        Mismatched fields keyed by field name.
    """
    actual: dict[str, str | None] = {
        "ConditionType": clean_optional(validity.get("ConditionType")),
        "Supplier": clean_optional(validity.get("Supplier")),
        "Material": clean_optional(validity.get("Material")),
        "PurchasingOrganization": clean_optional(validity.get("PurchasingOrganization")),
        "PurchasingInfoRecordCategory": clean_optional(validity.get("PurchasingInfoRecordCategory")),
        "ConditionValidityStartDate": (
            parsed.isoformat()
            if (parsed := parse_s4_date(validity.get("ConditionValidityStartDate"))) is not None
            else None
        ),
        "ConditionValidityEndDate": (
            parsed.isoformat()
            if (parsed := parse_s4_date(validity.get("ConditionValidityEndDate"))) is not None
            else None
        ),
        "ConditionRateValue": clean_optional(condition_record.get("ConditionRateValue")),
        "ConditionRateValueUnit": clean_optional(condition_record.get("ConditionRateValueUnit")),
    }
    mismatches: dict[str, dict[str, str | None]] = {}
    for field_name, expected in expected_readback_values(request, context, config).items():
        if actual.get(field_name) != expected:
            mismatches[field_name] = {"expected": expected, "actual": actual.get(field_name)}
    return mismatches


def verify_price_change_readback(
    client: Any,
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
    *,
    condition_record_id: str | None = None,
) -> dict[str, Any]:
    """Read back S/4 after a condition write and verify requested values.

    Args:
        client: Authenticated S/4 client.
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.
        condition_record_id: Optional record id from create response.

    Returns:
        Verification result.
    """
    seed_condition = get_condition_record(client, condition_record_id) if condition_record_id else {}
    validity_rows = find_condition_validity_rows(client, request, context, config)
    candidates = [
        row
        for row in validity_rows
        if parse_s4_date(row.get("ConditionValidityStartDate")) == date.fromisoformat(request.effective_from)
        and parse_s4_date(row.get("ConditionValidityEndDate")) == date.fromisoformat(request.effective_to)
    ]
    if condition_record_id:
        candidates = [row for row in candidates if clean_optional(row.get("ConditionRecord")) == condition_record_id]
    if not candidates:
        return {
            "status": "readback_mismatch",
            "mismatches": {"condition_validity": {"expected": "matching row", "actual": None}},
        }
    if len(candidates) > 1:
        return {
            "status": "readback_mismatch",
            "mismatches": {"condition_validity": {"expected": "single row", "actual": str(len(candidates))}},
        }
    condition_record = seed_condition
    if not condition_record:
        record_id = clean_optional(candidates[0].get("ConditionRecord"))
        condition_record = get_condition_record(client, record_id) if record_id else {}
    mismatches = compare_readback(candidates[0], condition_record, request, context, config)
    if mismatches:
        return {
            "status": "readback_mismatch",
            "condition_validity": candidates[0],
            "condition_record": condition_record,
            "mismatches": mismatches,
        }
    return {
        "status": "verified",
        "condition_validity": candidates[0],
        "condition_record": condition_record,
    }


def set_supplier_material_price(
    client: Any,
    request: PriceChangeRequest,
    context: SupplierPriceContext,
    config: PriceConditionConfig,
    *,
    post: bool = True,
) -> dict[str, Any]:
    """Create a PPR0 condition or delimit an old row and create a new one.

    Args:
        client: Authenticated S/4 client.
        request: Reviewed price-change request.
        context: Supplier pricing context.
        config: Condition configuration.
        post: When false, return dry-run statuses without writing.

    Returns:
        JSON-serializable action report.
    """
    create_payload = build_condition_create_payload(request, context, config)
    try:
        info_rows = find_info_record_rows(client, request, context)
    except S4HTTPError as exc:
        return api_failure_result("failed_api", exc)
    if not info_rows:
        return {
            "status": "info_record_missing",
            "message": "No matching supplier/material purchasing info record found.",
            "info_record_filter": build_info_record_filter(request, context),
            "payload": create_payload,
        }
    if len(info_rows) > 1:
        return {
            "status": "info_record_ambiguous",
            "message": "Multiple matching supplier/material purchasing info records found.",
            "matches": info_rows,
            "payload": create_payload,
        }
    try:
        condition_rows = find_condition_validity_rows(client, request, context, config)
    except S4HTTPError as exc:
        return status_for_condition_api_error(exc)
    overlaps = overlapping_validities(condition_rows, request)
    if len(overlaps) > 1:
        return {
            "status": "overlap_ambiguous",
            "message": "Multiple overlapping PPR0 validity rows found; refusing to write automatically.",
            "matches": overlaps,
            "payload": create_payload,
        }
    if not overlaps:
        if not post:
            return {
                "status": "would_create_condition",
                "message": "Dry run: no overlapping PPR0 row found; a new condition would be created.",
                "info_record": info_rows[0],
                "payload": create_payload,
            }
        try:
            response_body, status_code, _headers = client.post_json(
                PURCHASING_CONDITION_RECORD_SERVICE_NAME,
                "/A_PurgPrcgConditionRecord",
                create_payload,
            )
        except S4HTTPError as exc:
            return status_for_condition_api_error(exc)
        condition_record_id = extract_condition_record_id(response_body)
        try:
            readback = verify_price_change_readback(
                client,
                request,
                context,
                config,
                condition_record_id=condition_record_id,
            )
        except S4HTTPError as exc:
            return api_failure_result("readback_mismatch", exc, api_response=response_body)
        if readback["status"] != "verified":
            return {
                **readback,
                "api_response": response_body,
                "http_status": status_code,
                "payload": create_payload,
            }
        return {
            "status": "created_condition",
            "message": f"PPR0 condition created with HTTP {status_code}.",
            "condition_record": condition_record_id,
            "payload": create_payload,
            "readback": readback,
        }
    overlap = overlaps[0]
    if not is_open_ended_validity(overlap):
        return {
            "status": "overlap_ambiguous",
            "message": "An overlapping PPR0 row exists, but it is not open-ended.",
            "matches": overlaps,
            "payload": create_payload,
        }
    delimit_to = delimit_end_date_for_request(request)
    existing_start = parse_s4_date(overlap.get("ConditionValidityStartDate"))
    if existing_start is not None and date.fromisoformat(delimit_to) < existing_start:
        return {
            "status": "overlap_ambiguous",
            "message": "The requested start date would delimit the existing row before its start date.",
            "matches": overlaps,
            "payload": create_payload,
        }
    batch = build_delimit_and_create_batch(overlap, request, context, config)
    if not post:
        return {
            "status": "would_delimit_and_create",
            "message": "Dry run: one open-ended overlapping PPR0 row would be delimited and a new row created.",
            "info_record": info_rows[0],
            "existing_condition": overlap,
            "delimit_to": delimit_to,
            "payload": create_payload,
            "batch_body": batch.body,
        }
    try:
        batch_response, status_code, _headers = client.post_batch(
            PURCHASING_CONDITION_RECORD_SERVICE_NAME,
            batch.body,
            batch.content_type,
        )
    except S4HTTPError as exc:
        return status_for_condition_api_error(exc)
    try:
        readback = verify_price_change_readback(client, request, context, config)
    except S4HTTPError as exc:
        return api_failure_result("readback_mismatch", exc, api_response=batch_response)
    if readback["status"] != "verified":
        return {
            **readback,
            "api_response": batch_response,
            "http_status": status_code,
            "payload": create_payload,
        }
    return {
        "status": "delimited_and_created",
        "message": f"Existing PPR0 row delimited and new row created with HTTP {status_code}.",
        "existing_condition": overlap,
        "delimit_to": delimit_to,
        "payload": create_payload,
        "readback": readback,
    }


def is_successful_write_result(result: dict[str, Any]) -> bool:
    """Return whether an S/4 write result is an approval-success status.

    Args:
        result: S/4 write result.

    Returns:
        True when the price condition was written and verified.
    """
    return result.get("status") in SUCCESS_STATUSES


class S4PriceApprovalService:
    """Application service for user-triggered S/4 price approval writes."""

    def __init__(
        self,
        lookup_repository: S4LookupRepository,
        config: PriceConditionConfig | None = None,
    ) -> None:
        """Create a price approval writer.

        Args:
            lookup_repository: S/4 lookup repository used for supplier resolution and client access.
            config: Optional PPR0 write configuration.

        Returns:
            None.
        """
        self.lookup_repository = lookup_repository
        self.config = config or PriceConditionConfig()

    def approve_draft(self, draft: dict[str, Any]) -> dict[str, Any]:
        """Attempt to write one reviewed draft to S/4 PPR0 condition records.

        Args:
            draft: Reviewed price-change draft from HANA.

        Returns:
            S/4 write result.
        """
        try:
            request = build_price_request_from_draft(draft, self.config)
        except PriceConditionError as exc:
            return {"status": "failed_api", "message": str(exc)}
        supplier_lookup = self.lookup_repository.find_supplier_by_id(request.supplier_id)
        supplier = supplier_lookup.get("supplier")
        if supplier_lookup.get("status") != "found" or not isinstance(supplier, dict):
            return {
                "status": "supplier_missing",
                "message": "S/4 supplier could not be resolved for the reviewed supplier id.",
                "supplier_lookup": supplier_lookup,
            }
        try:
            context = build_supplier_context(supplier, request.supplier_id, self.config)
        except PriceConditionError as exc:
            return {"status": "failed_api", "message": str(exc)}
        return set_supplier_material_price(
            self.lookup_repository.client,
            request,
            context,
            self.config,
            post=True,
        )
