"""Pure cell normalization helpers for benchmark workbook rows."""

from __future__ import annotations

import math
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .validation import RowValueError, WarningAccumulator


def normalized_text(value: Any) -> str:
    """Normalize Unicode, accents, punctuation, case, and whitespace for QA.

    Inputs:
        value: Source or canonical text value.

    Outputs:
        str: Accent-free alphanumeric tokens separated by single spaces.
    """

    source = "" if value is None else str(value)
    decomposed = unicodedata.normalize("NFKD", source).casefold()
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in without_accents
        ).split()
    )


def identifier(value: Any) -> str:
    """Normalize Excel numeric or text identifiers without a decimal suffix.

    Inputs:
        value: Cell value representing a source identity.

    Outputs:
        str: Trimmed identifier, with integral floats rendered as integers.
    """

    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def text_or_none(value: Any) -> str | None:
    """Return trimmed source text or ``None`` for blank values.

    Inputs:
        value: Arbitrary worksheet cell value.

    Outputs:
        str | None: Trimmed text when non-blank, otherwise ``None``.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def is_placeholder(
    value: Any,
    *,
    numeric_zero_is_placeholder: bool = True,
) -> bool:
    """Return whether a profile cell is blank or a numeric zero placeholder.

    Inputs:
        value: Workbook profile cell value.
        numeric_zero_is_placeholder: Whether numeric/text zero represents
            missing profile data. Boolean profile fields disable this because
            zero is a valid explicit false value.

    Outputs:
        bool: ``True`` for blank, zero, or zero-like text placeholders.
    """

    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Decimal)) and value == 0:
        return numeric_zero_is_placeholder
    normalized = str(value).strip().casefold()
    if normalized in {"", "none", "null"}:
        return True
    return numeric_zero_is_placeholder and normalized in {"0", "0.0"}


def profile_value(
    value: Any,
    column: str,
    company_id: str,
    warnings: WarningAccumulator,
    *,
    numeric_zero_is_placeholder: bool = True,
) -> Any | None:
    """Normalize blank/zero profile placeholders to null with one warning.

    Inputs:
        value: Raw profile cell value.
        column: Source profile column name.
        company_id: Source company identity used for warning deduplication.
        warnings: Aggregate warning collector.
        numeric_zero_is_placeholder: Whether numeric/text zero represents
            missing data rather than an explicit boolean false.

    Outputs:
        Any | None: Original value when meaningful, otherwise ``None``.
    """

    if not is_placeholder(
        value,
        numeric_zero_is_placeholder=numeric_zero_is_placeholder,
    ):
        return value
    warnings.add(
        "profile_placeholder_normalized",
        "Blank or zero company profile placeholders were normalized to null",
        f"company {company_id}: {column}",
        token=f"{company_id}:{column}",
    )
    return None


def decimal_or_none(value: Any, field: str) -> Decimal | None:
    """Parse an optional numeric profile value as ``Decimal``.

    Inputs:
        value: Normalized source numeric value or ``None``.
        field: Human-readable source field used in validation messages.

    Outputs:
        Decimal | None: Exact decimal value, or ``None`` when absent.

    Raises:
        RowValueError: If a non-blank value is not numeric.
    """

    if value is None:
        return None
    try:
        parsed = Decimal(
            str(value).strip().replace("€", "").replace(" ", "").replace(",", ".")
        )
    except (InvalidOperation, ValueError) as exc:
        raise RowValueError(
            "invalid_profile_value",
            f"{field} must be numeric when provided",
        ) from exc
    if not parsed.is_finite():
        raise RowValueError(
            "invalid_profile_value",
            f"{field} must be finite when provided",
        )
    return parsed


def integer_or_none(value: Any, field: str) -> int | None:
    """Parse an optional integral profile value.

    Inputs:
        value: Normalized source numeric value or ``None``.
        field: Human-readable source field used in validation messages.

    Outputs:
        int | None: Integral value, or ``None`` when absent.

    Raises:
        RowValueError: If a non-blank value is not an integer.
    """

    decimal_value = decimal_or_none(value, field)
    if decimal_value is None:
        return None
    if decimal_value != decimal_value.to_integral_value():
        raise RowValueError(
            "invalid_profile_value",
            f"{field} must be an integer when provided",
        )
    try:
        return int(decimal_value)
    except (OverflowError, ValueError) as exc:
        raise RowValueError(
            "invalid_profile_value",
            f"{field} must be a finite integer when provided",
        ) from exc


def date_or_none(value: Any, field: str) -> date | None:
    """Parse an optional ISO date or Excel datetime cell.

    Inputs:
        value: Source date cell value.
        field: Source column name used in validation messages.

    Outputs:
        date | None: Calendar date, or ``None`` for blank input.

    Raises:
        RowValueError: If the value cannot be interpreted as a date.
    """

    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise RowValueError(
            "invalid_date",
            f"{field} must be an ISO date",
        ) from exc


def customer_class(value: Any) -> str:
    """Normalize source classes 1-5 to application ``class_N`` identifiers.

    Inputs:
        value: Numeric or textual workbook class value.

    Outputs:
        str: One of ``class_1`` through ``class_5``.

    Raises:
        RowValueError: If the class is missing or outside the supported range.
    """

    text = identifier(value).casefold()
    if text.startswith("class_"):
        text = text.removeprefix("class_")
    try:
        decimal_value = Decimal(text)
        if decimal_value != decimal_value.to_integral_value():
            raise ValueError("class must be an integer")
        number = int(decimal_value)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise RowValueError(
            "invalid_class",
            f"Unsupported company class: {value!r}",
        ) from exc
    if number not in range(1, 6):
        raise RowValueError(
            "invalid_class",
            f"Unsupported company class: {value!r}",
        )
    return f"class_{number}"


def boolean(value: Any, field: str, *, allow_none: bool = False) -> bool | None:
    """Normalize Italian/English workbook boolean values.

    Inputs:
        value: Boolean, numeric 0/1, or common yes/no text.
        field: Source column name used in validation messages.
        allow_none: Whether a blank value is accepted as ``None``.

    Outputs:
        bool | None: Normalized boolean, or optional ``None``.

    Raises:
        RowValueError: If the value is blank when required or unrecognized.
    """

    if value is None or str(value).strip() == "":
        if allow_none:
            return None
        raise RowValueError("invalid_boolean", f"{field} must contain yes or no")
    if isinstance(value, bool):
        return value
    normalized = normalized_text(value)
    if normalized in {"si", "yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise RowValueError(
        "invalid_boolean",
        f"{field} contains an unsupported boolean value: {value!r}",
    )


def level(value: Any) -> int:
    """Parse and validate one maturity level from 1 through 5.

    Inputs:
        value: Numeric or text worksheet level value.

    Outputs:
        int: Validated maturity level.

    Raises:
        RowValueError: If the source value is not an integer from 1 to 5.
    """

    try:
        decimal_value = Decimal(identifier(value))
        if decimal_value != decimal_value.to_integral_value():
            raise ValueError("level must be an integer")
        parsed = int(decimal_value)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise RowValueError("invalid_level", f"Invalid maturity level: {value!r}") from exc
    if parsed not in range(1, 6):
        raise RowValueError("invalid_level", f"Invalid maturity level: {value!r}")
    return parsed


def float_or_none(value: Any, field: str) -> float | None:
    """Parse an optional finite numeric audit score.

    Inputs:
        value: Source score cell value.
        field: Source field name used in validation messages.

    Outputs:
        float | None: Finite numeric score, or ``None`` when blank.

    Raises:
        RowValueError: If the supplied value is not finite and numeric.
    """

    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RowValueError(
            "invalid_score",
            f"{field} must be numeric when provided",
        ) from exc
    if not math.isfinite(parsed):
        raise RowValueError("invalid_score", f"{field} must be finite")
    return parsed
