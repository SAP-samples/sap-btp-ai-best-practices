"""
Shared value-parsing utilities for database rows.

SQLite returns TEXT columns as Python strings, while HANA may return
native ``datetime``, ``date``, or ``Decimal`` objects.  These helpers
normalise both representations into the expected Python type.
"""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Optional


def parse_decimal(value) -> Optional[Decimal]:
    """Parse a value to Decimal, returning None on failure."""
    if not value:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, ArithmeticError):
        return None


def parse_date(value) -> Optional[date]:
    """Parse a value to ``date``, handling strings and native datetime/date."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except Exception:
        return None


def parse_datetime(value) -> Optional[datetime]:
    """Parse a value to ``datetime``, handling strings and native datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None
