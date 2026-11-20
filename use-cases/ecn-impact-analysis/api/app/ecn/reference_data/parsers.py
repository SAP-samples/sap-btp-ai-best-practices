"""Strict workbook parsers for Settings-managed reusable source data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from csv import reader as csv_reader
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from io import BytesIO, StringIO
from itertools import chain
import re
from uuid import NAMESPACE_URL, uuid5

from openpyxl import load_workbook

from app.ecn.errors import SourceValidationError
from app.ecn.models import JsonValue, canonical_uuid
from app.ecn.reference_data.models import (
    NormalizedReferenceRow,
    ParsedReferenceWorkbook,
    ReferenceDatasetType,
)


_MATERIAL_HEADERS = (
    "Material",
    "Material Description",
    "Plnt",
    "ProcType",
    "Cross plant status",
    "Plant specific status",
    "Matl Group",
    "Commodity Code number",
)

_PURCHASE_HEADERS = (
    "Info Rec.",
    "Vendor",
    "Plnt",
    "Cat",
    "PGr",
    "Tx",
    "FFD",
    "Material",
)

_PURCHASE_PRICE_HEADERS = (
    "Info Rec.",
    "Vendor",
    "Name 1",
    "Material",
    "Plnt",
    "PGr",
    "Net Price",
    "Crcy",
    "Per",
    "OPU",
)

_GROUPED_DECIMAL = re.compile(
    r"^(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$"
)


def _text(value: object) -> str | None:
    """Normalize an optional worksheet scalar to stable trimmed text."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        rendered = str(int(value))
    else:
        rendered = str(value)
    stripped = rendered.strip()
    return stripped or None


def _row_id(artifact_id: str, sheet_name: str, row_number: int, slot: int = 0) -> str:
    """Derive a stable UUID from immutable artifact and source coordinates."""
    return str(
        uuid5(
            NAMESPACE_URL,
            f"ecn-reference:{artifact_id}:{sheet_name}:{row_number}:{slot}",
        )
    )


def _headers(row: Sequence[object]) -> tuple[str, ...]:
    """Normalize one worksheet header row for exact validation."""
    return tuple(_text(value) or "" for value in row)


def _header_positions(
    row: Sequence[object], required: Sequence[str]
) -> dict[str, int]:
    """Return exact required header positions or raise a safe validation error."""
    normalized = _headers(row)
    missing = [header for header in required if normalized.count(header) != 1]
    if missing:
        raise SourceValidationError(
            "REFERENCE_HEADERS_INVALID",
            "Required reference-data headers are missing or duplicated.",
            status_code=400,
            details={"headers": missing},
        )
    return {header: normalized.index(header) for header in required}


def _value(row: Sequence[object], index: int) -> str | None:
    """Read one optional cell from a possibly short worksheet row."""
    return _text(row[index]) if index < len(row) else None


def _tabular_rows(
    *,
    artifact_id: str,
    sheet_name: str,
    rows: Iterable[Sequence[object]],
    required_headers: Sequence[str],
    field_map: dict[str, str],
) -> tuple[NormalizedReferenceRow, ...]:
    """Parse one exact header table into normalized named business fields."""
    iterator = iter(rows)
    try:
        header_row = next(iterator)
    except StopIteration as exc:
        raise SourceValidationError(
            "REFERENCE_WORKBOOK_EMPTY",
            "The reference workbook is empty.",
            status_code=400,
        ) from exc
    positions = _header_positions(header_row, required_headers)
    parsed: list[NormalizedReferenceRow] = []
    for row_number, row in enumerate(iterator, start=2):
        values: dict[str, JsonValue] = {
            target: _value(row, positions[source])
            for source, target in field_map.items()
        }
        if not any(value is not None for value in values.values()):
            continue
        parsed.append(
            NormalizedReferenceRow(
                row_id=_row_id(artifact_id, sheet_name, row_number),
                artifact_id=artifact_id,
                sheet_name=sheet_name,
                source_row_number=row_number,
                values=values,
            )
        )
    return tuple(parsed)


def _parse_material_master(
    artifact_id: str, sheet_name: str, rows: Iterable[Sequence[object]]
) -> ParsedReferenceWorkbook:
    """Parse either delivered material layout; disambiguate MS only by exact layout."""
    rows = iter(rows)
    header = next(rows, ())
    fields = {
        "Material": "material_number", "Material Description": "description",
        "Plnt": "plant_code", "ProcType": "procurement_type",
        "Cross plant status": "cross_plant_status", "Plant specific status": "plant_status",
        "Matl Group": "material_group", "Commodity Code number": "commodity_code",
    }
    required = _MATERIAL_HEADERS
    if _headers(header) == ("Material", "Material Description", "Plnt", "ProcType", "MS", "MS", "Matl Group", "Material Group Desc."):
        header = (*header[:4], "Cross plant status", "Plant specific status", *header[6:])
        fields.pop("Commodity Code number")
        fields["Material Group Desc."] = "material_group_description"
        required = tuple(fields)
    rows = chain((header,), rows)
    parsed = _tabular_rows(
        artifact_id=artifact_id,
        sheet_name=sheet_name,
        rows=rows,
        required_headers=required,
        field_map=fields,
    )
    return ParsedReferenceWorkbook(ReferenceDatasetType.MATERIAL_MASTER, parsed, {})


def _parse_purchase_info(
    artifact_id: str, sheet_name: str, rows: Iterable[Sequence[object]]
) -> ParsedReferenceWorkbook:
    """Parse either delivered purchase layout without inventing missing legacy fields."""
    rows = iter(rows)
    header = next(rows, ())
    fields = {
        "Info Rec.": "info_record", "Vendor": "vendor", "Plnt": "plant_code",
        "Cat": "category", "PGr": "purchasing_group", "Tx": "tax_code",
        "FFD": "fixed_firm_date", "Material": "material_number",
    }
    required = _PURCHASE_HEADERS
    if _headers(header) == ("Info Rec.", "Vendor", "Name 1", "Material", "Plnt", "PGr"):
        fields = {key: value for key, value in fields.items() if key not in {"Cat", "Tx", "FFD"}}
        fields["Name 1"] = "supplier_name"
        required = tuple(fields)
    rows = chain((header,), rows)
    parsed = _tabular_rows(
        artifact_id=artifact_id,
        sheet_name=sheet_name,
        rows=rows,
        required_headers=required,
        field_map=fields,
    )
    return ParsedReferenceWorkbook(
        ReferenceDatasetType.PURCHASE_INFO_RECORDS, parsed, {}
    )


def _grouped_decimal_text(
    value: object,
    *,
    field: str,
    allow_zero: bool,
    sheet_name: str,
    row_number: int,
) -> str:
    """Normalize a positive source decimal with comma grouping, never guessing locale."""
    raw = _text(value)
    if raw is None or _GROUPED_DECIMAL.fullmatch(raw) is None:
        raise SourceValidationError(
            "PURCHASE_INFO_PRICE_NUMBER_INVALID",
            "Purchase-info price values must use digits, comma grouping, and an optional decimal point.",
            status_code=400,
            details={"field": field, "sheet": sheet_name, "row": row_number},
        )
    try:
        number = Decimal(raw.replace(",", ""))
    except InvalidOperation as exc:
        raise SourceValidationError(
            "PURCHASE_INFO_PRICE_NUMBER_INVALID",
            "Purchase-info price values must be finite decimal numbers.",
            status_code=400,
            details={"field": field, "sheet": sheet_name, "row": row_number},
        ) from exc
    if not number.is_finite() or number < 0 or (not allow_zero and number == 0):
        raise SourceValidationError(
            "PURCHASE_INFO_PRICE_NUMBER_INVALID",
            "Net price must be non-negative and price quantity must be positive.",
            status_code=400,
            details={"field": field, "sheet": sheet_name, "row": row_number},
        )
    return format(number, "f")


def parse_purchase_info_price_csv(
    content: bytes, artifact_id: str
) -> ParsedReferenceWorkbook:
    """Parse the approved purchase-info price CSV into source-backed overlay rows.

    The source uses comma digit grouping and an optional decimal point. Its six
    legacy purchase-info columns are retained exactly as normalized fields so a
    later reconciliation can create a non-destructive Purchase Info successor.
    """
    canonical_uuid(artifact_id)
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SourceValidationError(
            "PURCHASE_INFO_PRICE_CSV_INVALID",
            "Purchase-info price CSV must be UTF-8 encoded.",
            status_code=400,
        ) from exc
    rows = tuple(tuple(row) for row in csv_reader(StringIO(decoded)))
    parsed = _tabular_rows(
        artifact_id=artifact_id,
        sheet_name="CSV",
        rows=rows,
        required_headers=_PURCHASE_PRICE_HEADERS,
        field_map={
            "Info Rec.": "info_record",
            "Vendor": "vendor",
            "Name 1": "supplier_name",
            "Material": "material_number",
            "Plnt": "plant_code",
            "PGr": "purchasing_group",
            "Net Price": "net_price",
            "Crcy": "currency",
            "Per": "price_per",
            "OPU": "order_price_unit",
        },
    )
    normalized_rows: list[NormalizedReferenceRow] = []
    zero_price_count = 0
    for row in parsed:
        values = dict(row.values)
        required = (
            "info_record",
            "vendor",
            "supplier_name",
            "material_number",
            "plant_code",
            "purchasing_group",
            "net_price",
            "currency",
            "price_per",
            "order_price_unit",
        )
        if any(values.get(field) is None for field in required):
            raise SourceValidationError(
                "PURCHASE_INFO_PRICE_KEY_MISSING",
                "Purchase-info price rows require every documented source field.",
                status_code=400,
                details={"sheet": row.sheet_name, "row": row.source_row_number},
            )
        values["net_price"] = _grouped_decimal_text(
            values["net_price"],
            field="Net Price",
            allow_zero=True,
            sheet_name=row.sheet_name,
            row_number=row.source_row_number,
        )
        values["price_per"] = _grouped_decimal_text(
            values["price_per"],
            field="Per",
            allow_zero=False,
            sheet_name=row.sheet_name,
            row_number=row.source_row_number,
        )
        if Decimal(values["net_price"]) == 0:
            zero_price_count += 1
        normalized_rows.append(replace(row, values=values))
    warnings = {"PURCHASE_INFO_PRICE_ZERO": zero_price_count} if zero_price_count else {}
    return ParsedReferenceWorkbook(
        ReferenceDatasetType.PURCHASE_INFO_PRICES,
        tuple(normalized_rows),
        warnings,
    )


def _parse_stock(artifact_id: str, sheet_name: str, rows: Iterable[Sequence[object]]) -> ParsedReferenceWorkbook:
    """Read reported quantities with exact storage coordinates and explicit unit/date gaps."""
    parsed = _tabular_rows(
        artifact_id=artifact_id, sheet_name=sheet_name, rows=rows,
        required_headers=("Material", "Plant", "Storage Location", "Unrestricted"),
        field_map={"Material": "material_number", "Plant": "plant_code",
                   "Storage Location": "storage_location", "Unrestricted": "unrestricted_quantity"},
    )
    for row in parsed:
        value = row.values["unrestricted_quantity"]
        try:
            if value is None or not Decimal(value).is_finite():
                raise InvalidOperation
        except (InvalidOperation, ValueError):
            raise SourceValidationError("STOCK_QUANTITY_INVALID", "Stock quantity must be a finite number.", status_code=400,
                                        details={"sheet": sheet_name, "row": row.source_row_number}) from None
        if not all(row.values[key] for key in ("material_number", "plant_code", "storage_location")):
            raise SourceValidationError("STOCK_KEY_MISSING", "Stock requires material, plant and storage location.", status_code=400)
    return ParsedReferenceWorkbook(ReferenceDatasetType.STOCK, parsed, {"STOCK_UNIT_UNAVAILABLE": 1, "STOCK_DATE_UNAVAILABLE": 1})


def parse_reference_workbook(
    dataset_type: ReferenceDatasetType,
    content: bytes,
    artifact_id: str,
) -> ParsedReferenceWorkbook:
    """Parse one supported non-BOM/non-conditional-rule reference workbook.

    Supported types are STOCK, MATERIAL_MASTER, and PURCHASE_INFO_RECORDS.
    BOM and CONDITIONAL_RULES use their own established parsers and must not
    be passed here.

    Args:
        dataset_type: Exact dataset contract to apply.
        content: Original uploaded XLSX bytes.
        artifact_id: Immutable artifact UUID used for stable row identifiers.

    Returns:
        Normalized source rows plus grouped validation warnings.

    Raises:
        SourceValidationError: If the XLSX, worksheet, or headers are invalid.
        ValueError: If this generic parser receives BOM or conditional rules.
    """
    canonical_uuid(artifact_id)
    if dataset_type is ReferenceDatasetType.PURCHASE_INFO_PRICES:
        return parse_purchase_info_price_csv(content, artifact_id)
    if dataset_type in {
        ReferenceDatasetType.BOM,
        ReferenceDatasetType.CONDITIONAL_RULES,
    }:
        raise ValueError("BOM and conditional rules use their established parsers")
    workbook = None
    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=False)
        if not workbook.sheetnames:
            raise SourceValidationError(
                "REFERENCE_WORKBOOK_EMPTY",
                "The reference workbook contains no worksheets.",
                status_code=400,
            )
        sheet = workbook.active
        rows = tuple(tuple(row) for row in sheet.iter_rows(values_only=True))
        if dataset_type is ReferenceDatasetType.STOCK:
            return _parse_stock(artifact_id, sheet.title, rows)
        if dataset_type is ReferenceDatasetType.MATERIAL_MASTER:
            return _parse_material_master(artifact_id, sheet.title, rows)
        return _parse_purchase_info(artifact_id, sheet.title, rows)
    except SourceValidationError:
        raise
    except Exception as exc:
        raise SourceValidationError(
            "REFERENCE_WORKBOOK_INVALID",
            "The reference workbook is unreadable.",
            status_code=400,
        ) from exc
    finally:
        if workbook is not None:
            workbook.close()
