"""Shared column contract and identifier normalization for credit-settings Excel imports.

Used by api/app/services/workspace/settings_import.py.
"""

from __future__ import annotations

import pandas as pd


# Columns required in the "limits" worksheet of an uploaded credit-settings workbook.
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


def _normalize_id(value: object, *, allow_empty: bool = False) -> str | None:
    """Normalize an Excel identifier cell to text (integral floats lose their ".0").

    Args:
        value: Raw cell value.
        allow_empty: Return None instead of "" for blank cells.

    Returns:
        The identifier as text, "" or None for blank cells.
    """
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
