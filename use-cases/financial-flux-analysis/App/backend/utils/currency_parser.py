"""
Utility module for parsing currency amount strings from SAP HANA.

Handles European-format amounts with currency codes such as:
    "- 561.434,27 USD"  ->  -561434.27
    "1.234,99 USD"      ->  1234.99
    "0,00 EUR"          ->  0.0
"""

import logging
import re
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# Regex to strip trailing 3-letter currency codes (e.g. USD, EUR, GBP)
_CURRENCY_CODE_RE = re.compile(r"\s*[A-Z]{3}\s*$")


def parse_currency_amount(value: Union[str, float, int, None]) -> float:
    """
    Parse a currency amount string into a Python float.

    Supports:
    - Null / NaN values  -> 0.0
    - Already-numeric values (int / float) -> float(value)
    - European format strings with optional currency code:
        "- 561.434,27 USD" -> -561434.27
        "1.234,99 USD"     ->  1234.99
        "-100,50"          -> -100.50
        "0,00"             ->  0.0

    Never raises an exception; logs a warning and returns 0.0 on parse failure.

    Args:
        value: Raw value from the database column.

    Returns:
        Parsed float amount.
    """
    # Handle None
    if value is None:
        return 0.0

    # Handle pandas NaN / NaT
    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    # Already numeric — return directly
    if isinstance(value, (int, float)):
        return float(value)

    # Convert to string and strip surrounding whitespace
    raw = str(value).strip()

    if not raw:
        return 0.0

    # Remove trailing currency code (e.g. "USD", "EUR")
    raw = _CURRENCY_CODE_RE.sub("", raw).strip()

    # Detect and remove leading negative sign (handles "- 561.434,27" with space)
    negative = raw.startswith("-")
    if negative:
        raw = raw[1:].strip()

    # European number format:
    #   Thousand separator = "."  (e.g. 1.234.567)
    #   Decimal separator  = ","  (e.g. ,27)
    # Strategy: remove all dots, then replace comma with dot
    raw = raw.replace(".", "").replace(",", ".")

    try:
        result = float(raw)
        return -result if negative else result
    except (ValueError, TypeError):
        logger.warning("parse_currency_amount: could not parse value %r — returning 0.0", value)
        return 0.0


def apply_currency_parser(series: pd.Series) -> pd.Series:
    """
    Apply parse_currency_amount to an entire pandas Series.

    Args:
        series: Series containing raw currency strings.

    Returns:
        Series of float values.
    """
    return series.apply(parse_currency_amount)