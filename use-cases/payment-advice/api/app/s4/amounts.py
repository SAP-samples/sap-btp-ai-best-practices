"""Amounts for the S/4 check: safe parsing and the number of decimals each currency has.

Extracted and corrected amounts arrive as floats, strings or nothing at all.
This module turns them into exact `Decimal` values without ever raising in the
middle of a check and without rounding silently:

- `parse_amount` returns None for anything that is not a finite number, and
  for a value with more decimals than its currency allows (a rounded amount
  would no longer be what the customer paid). Validation reports None as a
  blocking `invalid_amount`.
- `currency_decimals` knows the currencies without decimals (for example JPY)
  and with three decimals (for example KWD); every other currency has two.
  The list follows ISO 4217. A system whose currency settings differ can
  override single currencies with `S4_CURRENCY_DECIMALS="HUF=0,TWD=0"`.
- `tolerance` is one minor unit of the currency (0.01 for two decimals).

Example:
    parse_amount(10303.26, "CAD")   -> Decimal("10303.26")
    parse_amount("1000", "JPY")     -> Decimal("1000")
    parse_amount("12.345", "CAD")   -> None
    fmt(Decimal("1.5"), "KWD")      -> "1.500"
"""
from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Any

# ISO 4217 currencies whose minor unit is not 1/100.
ZERO_DECIMAL = frozenset({"BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW", "PYG", "RWF", "UGX", "UYI",
                          "VND", "VUV", "XAF", "XOF", "XPF"})
THREE_DECIMAL = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})
DEFAULT_DECIMALS = 2
# Float arithmetic in a browser can leave 10303.260000000002; anything larger than this is a real extra decimal.
NOISE = Decimal("0.000001")


def _overrides() -> dict[str, int]:
    """Parse `S4_CURRENCY_DECIMALS` ("HUF=0,TWD=0") into {currency: decimals}; malformed entries are ignored."""
    result: dict[str, int] = {}
    for entry in os.getenv("S4_CURRENCY_DECIMALS", "").split(","):
        code, _, digits = entry.partition("=")
        if code.strip() and digits.strip().isdigit():
            result[code.strip().upper()] = int(digits.strip())
    return result


def currency_decimals(currency: str | None) -> int:
    """Return how many decimals amounts in ``currency`` have (2 when the currency is unknown or missing)."""
    code = (currency or "").strip().upper()
    overrides = _overrides()
    if code in overrides:
        return overrides[code]
    if code in ZERO_DECIMAL:
        return 0
    return 3 if code in THREE_DECIMAL else DEFAULT_DECIMALS


def unit(currency: str | None = None) -> Decimal:
    """Return the minor unit of ``currency`` as a Decimal (0.01, 1 or 0.001)."""
    return Decimal(1).scaleb(-currency_decimals(currency))


def tolerance(currency: str | None = None) -> Decimal:
    """Return the largest difference two amounts in ``currency`` may have and still count as equal."""
    return unit(currency)


def parse_amount(value: Any, currency: str | None = None) -> Decimal | None:
    """Convert an extracted amount to an exact Decimal in the precision of ``currency``.

    Args:
        value: Float, int, string or None; None and "" mean zero.
        currency: Currency of the amount; decides the number of decimals.

    Returns:
        The amount, or None when the value is not a finite number or has more
        decimals than the currency allows.
    """
    if value in (None, ""):
        return Decimal(0).quantize(unit(currency))
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    rounded = number.quantize(unit(currency))
    return rounded if abs(number - rounded) <= NOISE else None


def dec(value: Any, currency: str | None = None) -> Decimal:
    """Convert an amount that was already validated; unlike `parse_amount` it raises on a bad value.

    Raises:
        ValueError: The value is not a valid amount in ``currency``.
    """
    amount = parse_amount(value, currency)
    if amount is None:
        raise ValueError(f"{value!r} is not a valid amount" + (f" in {currency}" if currency else ""))
    return amount


def fmt(amount: Decimal, currency: str | None = None) -> str:
    """Format an amount with exactly the decimals of ``currency`` (the form sent to S/4)."""
    return str(amount.quantize(unit(currency)))


def sign_of(amount: Decimal) -> int:
    """Return 1 for a positive amount, -1 for a negative one and 0 for zero."""
    return (amount > 0) - (amount < 0)
