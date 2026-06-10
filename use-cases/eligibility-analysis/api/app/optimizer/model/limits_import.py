"""Import manual limits payloads from Excel/YAML/JSON files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml


GBP_PER_EUR = 0.87
SUPPORTED_CURRENCIES = {"EUR", "GBP"}

EXCEL_REQUIRED_COLUMNS = (
    "ID Seller",
    "Group Debtor",
    "ID Debtor",
    "Seller Limit",
    "Currency Seller LM",
    "Group Limit",
    "Currency Group LM",
    "Debtor Limit",
    "Currency Debtor LM",
    "Facility Limit",
    "Currency Facility",
)

DEFAULT_LIMITS = {
    "customer_limit_fraction_of_facility": 0.15,
    "group_limit_fraction_of_facility": 0.30,
}


class LimitsImportError(ValueError):
    """Raised when manual limits import file cannot be parsed."""


def _round_money(amount: float) -> float:
    return round(float(amount), 2)


def _normalize_id(value: object, *, allow_empty: bool = False) -> str | None:
    if value is None:
        return None if allow_empty else ""
    if pd.isna(value):
        return None if allow_empty else ""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None if allow_empty else ""
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            return text
        return text

    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text:
        return None if allow_empty else ""
    return text


def _normalize_currency(value: object, *, field_name: str, row_number: int) -> str:
    if value is None or pd.isna(value):
        raise LimitsImportError(f"Missing currency for {field_name} at row {row_number}.")
    currency = str(value).strip().upper()
    if currency not in SUPPORTED_CURRENCIES:
        raise LimitsImportError(
            f"Unsupported currency '{currency}' for {field_name} at row {row_number}. "
            f"Supported: EUR, GBP."
        )
    return currency


def _read_amount(value: object, *, field_name: str, row_number: int) -> float:
    if value is None or pd.isna(value):
        raise LimitsImportError(f"Missing amount for {field_name} at row {row_number}.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LimitsImportError(
            f"Invalid numeric amount '{value}' for {field_name} at row {row_number}."
        ) from exc


def _to_eur(
    amount: float,
    currency: str,
    *,
    summary: Dict[str, Any],
) -> float:
    if currency == "EUR":
        return _round_money(amount)
    if currency == "GBP":
        summary["gbp_conversions"] += 1
        return _round_money(amount / GBP_PER_EUR)
    raise LimitsImportError(f"Unsupported currency '{currency}'.")


def _ensure_consistent_amount(
    mapping: Dict[str, float],
    key: str,
    amount: float,
    *,
    entity_name: str,
) -> None:
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = amount
        return
    if _round_money(existing) != _round_money(amount):
        raise LimitsImportError(
            f"Conflicting {entity_name} limit for '{key}': {existing} vs {amount}."
        )


def _parse_excel_limits(
    file_content: bytes,
    *,
    existing_limits: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    workbook = pd.ExcelFile(BytesIO(file_content), engine="openpyxl")
    if not workbook.sheet_names:
        raise LimitsImportError("Limits workbook has no sheets.")
    sheet_name = workbook.sheet_names[0]

    raw_df = pd.read_excel(workbook, sheet_name=sheet_name, engine="openpyxl")
    if raw_df.empty:
        raise LimitsImportError("Limits workbook is empty.")

    normalized_columns = {col: str(col).strip() for col in raw_df.columns}
    df = raw_df.rename(columns=normalized_columns)

    missing = [col for col in EXCEL_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise LimitsImportError("Missing required columns: " + ", ".join(missing))

    summary: Dict[str, Any] = {
        "source_type": "excel",
        "sheet_name": sheet_name,
        "rows_total": int(len(df)),
        "rows_processed": 0,
        "warnings_count": 0,
        "warnings": [],
        "gbp_conversions": 0,
    }

    facility_limits: Dict[str, float] = {}
    customer_limits: Dict[str, float] = {}
    group_limits: Dict[str, float] = {}
    customer_to_group: Dict[str, str] = {}

    for idx, row in df.iterrows():
        excel_row = int(idx) + 2
        seller_id = _normalize_id(row.get("ID Seller"), allow_empty=False)
        debtor_id = _normalize_id(row.get("ID Debtor"), allow_empty=False)
        group_id = _normalize_id(row.get("Group Debtor"), allow_empty=True)

        if not seller_id:
            summary["warnings"].append(f"Row {excel_row}: missing ID Seller, skipped.")
            continue
        if not debtor_id:
            summary["warnings"].append(f"Row {excel_row}: missing ID Debtor, skipped.")
            continue

        facility_amount_raw = row.get("Facility Limit")
        facility_currency_raw = row.get("Currency Facility")
        seller_amount_raw = row.get("Seller Limit")
        seller_currency_raw = row.get("Currency Seller LM")
        use_seller_fallback = facility_amount_raw is None or pd.isna(facility_amount_raw)
        if use_seller_fallback and (seller_amount_raw is not None and not pd.isna(seller_amount_raw)):
            facility_amount_raw = seller_amount_raw
            facility_currency_raw = seller_currency_raw

        if facility_amount_raw is not None and not pd.isna(facility_amount_raw):
            facility_amount = _read_amount(
                facility_amount_raw,
                field_name="Facility/Seller Limit",
                row_number=excel_row,
            )
            facility_currency = _normalize_currency(
                facility_currency_raw,
                field_name="Facility/Seller Limit",
                row_number=excel_row,
            )
            facility_eur = _to_eur(facility_amount, facility_currency, summary=summary)
            _ensure_consistent_amount(
                facility_limits,
                seller_id,
                facility_eur,
                entity_name="facility",
            )
        else:
            summary["warnings"].append(
                f"Row {excel_row}: no Facility Limit or Seller Limit for seller '{seller_id}'."
            )

        debtor_amount = _read_amount(
            row.get("Debtor Limit"),
            field_name="Debtor Limit",
            row_number=excel_row,
        )
        debtor_currency = _normalize_currency(
            row.get("Currency Debtor LM"),
            field_name="Debtor Limit",
            row_number=excel_row,
        )
        debtor_limit_eur = _to_eur(debtor_amount, debtor_currency, summary=summary)
        _ensure_consistent_amount(
            customer_limits,
            debtor_id,
            debtor_limit_eur,
            entity_name="customer",
        )

        if group_id:
            existing_group = customer_to_group.get(debtor_id)
            if existing_group is not None and existing_group != group_id:
                raise LimitsImportError(
                    f"Conflicting group mapping for customer '{debtor_id}': "
                    f"'{existing_group}' vs '{group_id}'."
                )
            customer_to_group[debtor_id] = group_id

            group_amount_raw = row.get("Group Limit")
            group_currency_raw = row.get("Currency Group LM")
            if group_amount_raw is not None and not pd.isna(group_amount_raw):
                group_amount = _read_amount(
                    group_amount_raw,
                    field_name="Group Limit",
                    row_number=excel_row,
                )
                group_currency = _normalize_currency(
                    group_currency_raw,
                    field_name="Group Limit",
                    row_number=excel_row,
                )
                group_limit_eur = _to_eur(group_amount, group_currency, summary=summary)
                _ensure_consistent_amount(
                    group_limits,
                    group_id,
                    group_limit_eur,
                    entity_name="group",
                )
            else:
                summary["warnings"].append(
                    f"Row {excel_row}: group '{group_id}' has no Group Limit value."
                )

        summary["rows_processed"] += 1

    summary["warnings_count"] = len(summary["warnings"])

    defaults = dict(DEFAULT_LIMITS)
    if isinstance(existing_limits, dict):
        current_defaults = existing_limits.get("defaults", {})
        if isinstance(current_defaults, dict):
            defaults.update(current_defaults)

    payload: Dict[str, Any] = {
        "facility_limits_by_company_code": facility_limits,
        "customer_limits": customer_limits,
        "group_limits": group_limits,
        "customer_to_group": customer_to_group,
        "base_exposure": {
            "facility": {},
            "customer": {},
            "group": {},
        },
        "defaults": defaults,
        "synthetic_generation": {"enabled": False},
    }
    return payload, summary


def _parse_structured_text_limits(
    file_content: bytes,
    *,
    filename: str,
    existing_limits: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ext = Path(filename or "").suffix.lower()
    text = file_content.decode("utf-8")
    if ext == ".json":
        payload = yaml.safe_load(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise LimitsImportError("Structured limits file must contain a top-level object.")

    defaults = dict(DEFAULT_LIMITS)
    if isinstance(existing_limits, dict):
        current_defaults = existing_limits.get("defaults", {})
        if isinstance(current_defaults, dict):
            defaults.update(current_defaults)
    payload_defaults = payload.get("defaults")
    if isinstance(payload_defaults, dict):
        defaults.update(payload_defaults)

    payload.setdefault("facility_limits_by_company_code", {})
    payload.setdefault("customer_limits", {})
    payload.setdefault("group_limits", {})
    payload.setdefault("customer_to_group", {})
    payload.setdefault("base_exposure", {"facility": {}, "customer": {}, "group": {}})
    payload.setdefault("synthetic_generation", {"enabled": False})
    payload["synthetic_generation"]["enabled"] = False
    payload["defaults"] = defaults

    summary = {
        "source_type": "structured_text",
        "rows_total": None,
        "rows_processed": None,
        "warnings_count": 0,
        "warnings": [],
        "gbp_conversions": 0,
    }
    return payload, summary


def import_limits_payload(
    file_content: bytes,
    *,
    filename: str,
    existing_limits: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Import limits payload from Excel, YAML, YML, or JSON."""
    ext = Path(filename or "").suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return _parse_excel_limits(file_content, existing_limits=existing_limits)
    if ext in {".yaml", ".yml", ".json"}:
        return _parse_structured_text_limits(
            file_content,
            filename=filename,
            existing_limits=existing_limits,
        )
    raise LimitsImportError(
        f"Unsupported limits file extension '{ext or '<none>'}'. "
        "Use .xlsx, .xls, .yaml, .yml, or .json."
    )
