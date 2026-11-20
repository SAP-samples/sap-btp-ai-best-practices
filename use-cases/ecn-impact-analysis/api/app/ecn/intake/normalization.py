"""Source-object normalisation, BOM cell value normalisation, effectivity parsing,
and the four-way raw EBOM classification heuristic.

Provides:

- ``normalize_source_object`` / ``_walk_affected`` / ``_walk_resulting``
  for ECN JSON source-object normalisation (original Task 10 code).
- ``source_text``, ``ordered_unique``, ``json_safe_cell``, ``normalize_quantity``,
  ``fit_bom_identifier`` for lossless BOM cell value handling.
- ``EffectivityFormat``, ``EffectivityField``, ``normalize_effectivity_value``
  for locale-explicit effectivity date parsing.
- ``classify_raw_row`` for the versioned four-way raw-EBOM classification heuristic.

WTPart identity rules (source-object section):
- ``OR:wt.part.WTPart:<opaque>`` and ``VR:wt.part.WTPart:<opaque>`` both
  identify the same technical class; the opaque suffixes identify different
  Windchill object layers and are never compared.
- If both ID and VersionID are present and their technical classes disagree,
  the object is NOT seeded as a WTPart and a TECHNICAL_IDENTITY_CONTRADICTION
  qualifier is recorded.
- CAD/EPM, WTDocument, drawing, and specification objects always have
  ``is_technical_wtpart=False``; number shape never changes this.

Usage::

    from app.ecn.intake.normalization import (
        normalize_source_object, classify_raw_row, normalize_effectivity_value,
        EffectivityFormat, EffectivityField,
    )
"""

import math
import re
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from app.ecn.errors import SourceValidationError
from app.ecn.models import (
    BomRowClassification,
    JsonValue,
    NormalizedEffectivity,
    NormalizedSourceObject,
    SourceObjectRole,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TECHNICAL_ID = re.compile(r"^(?P<form>OR|VR):(?P<class>[^:]+):(?P<opaque>.+)$")
"""Matches Windchill technical identity strings of the form OR/VR:class:opaque."""

_WTPART_CLASS = "wt.part.WTPart"
"""The exact Windchill class name that marks a part master record."""

# Maximum lengths for identifier fields on source objects.
# These align with the corresponding NVARCHAR(n) BOM columns used for joins.
_NUMBER_MAX = 200
"""Maximum trimmed length for a source object number (join key)."""

# ---------------------------------------------------------------------------
# Enumerations for effectivity parsing
# ---------------------------------------------------------------------------


class EffectivityFormat(StrEnum):
    """Identify the wire format used to represent a date in a BOM source cell.

    ``RAW_DD_MM_YYYY`` applies to the raw EBOM sheet where date cells are
    stored as text strings in day-first European locale (``%d.%m.%Y``).
    ``EXCEL_TYPED_OR_ISO`` applies to the enriched EBOM and MBOM sheets where
    openpyxl returns ``date``/``datetime`` objects or an ISO-format string.
    """

    RAW_DD_MM_YYYY = "RAW_DD_MM_YYYY"
    EXCEL_TYPED_OR_ISO = "EXCEL_TYPED_OR_ISO"


class EffectivityField(StrEnum):
    """Identify whether an effectivity cell is a validity-start or validity-end date.

    Only ``VALID_FROM`` dates with a successfully parsed year of 1900 are
    marked as migration placeholders.  ``VALID_TO`` year-1900 dates are not.
    """

    VALID_FROM = "VALID_FROM"
    VALID_TO = "VALID_TO"


# ---------------------------------------------------------------------------
# Raw classification regex (versioned heuristic)
# ---------------------------------------------------------------------------

DOCUMENT_REFERENCE = re.compile(r"^.+?\s+ZV[DSG]\s+\S+\s+\S+$")
"""Matches component strings that embed a document-reference key.

Pattern: one or more non-whitespace segments, then a ZVD/ZVS/ZVG token,
then two further non-whitespace tokens, all separated by single whitespace.
Used by the four-way raw-EBOM classification heuristic (rule 2 of 4).
"""

# ---------------------------------------------------------------------------
# Quantity normalization dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantityNormalization:
    """Carry an exact HANA-safe quantity or a non-rounding rejection result.

    ``value`` holds the parsed ``Decimal`` when the input is exactly
    representable as ``DECIMAL(31,14)``; it is ``None`` on blank input or
    on any non-rounding rejection.  ``qualifiers`` is empty on success and
    contains ``"QUANTITY_NOT_EXACTLY_REPRESENTABLE"`` on rejection.

    Args:
        value: The parsed ``Decimal``, or ``None`` when absent or rejected.
        qualifiers: Empty tuple on success; ``("QUANTITY_NOT_EXACTLY_REPRESENTABLE",)``
            on rejection.
    """

    value: Decimal | None
    qualifiers: tuple[str, ...]


# ---------------------------------------------------------------------------
# Technical identity helpers
# ---------------------------------------------------------------------------


def _technical_class(value: object) -> str | None:
    """Return the Windchill technical class from an OR/VR identity string.

    Accepts only the exact forms ``OR:class:opaque`` and ``VR:class:opaque``.
    Any other shape, including non-strings, returns ``None``.

    Args:
        value: A potential Windchill identity string.

    Returns:
        The class segment (e.g. ``"wt.part.WTPart"``), or ``None`` if the
        value does not match the expected pattern.
    """
    if not isinstance(value, str):
        return None
    match = TECHNICAL_ID.fullmatch(value.strip())
    return None if match is None else match.group("class")


def _is_consistent_wtpart(
    technical_id: object,
    version_id: object,
) -> tuple[bool, tuple[str, ...]]:
    """Classify WTPart identity while allowing distinct OR/VR opaque suffixes.

    A positive classification requires all present class segments to agree
    and to equal ``wt.part.WTPart``.  If both IDs are present and disagree,
    returns ``(False, ("TECHNICAL_IDENTITY_CONTRADICTION",))``.

    Args:
        technical_id: The raw ``ID`` field value from the source object.
        version_id: The raw ``VersionID`` field value from the source object.

    Returns:
        A ``(is_wtpart, qualifiers)`` tuple.  ``is_wtpart`` is ``True`` only
        when both present IDs agree on the ``wt.part.WTPart`` class.
    """
    classes = [
        cls
        for cls in (_technical_class(technical_id), _technical_class(version_id))
        if cls is not None
    ]
    if not classes:
        return False, ()
    if len(set(classes)) > 1:
        # Two different classes → contradiction; object must not seed BOM matching.
        return False, ("TECHNICAL_IDENTITY_CONTRADICTION",)
    return classes[0] == _WTPART_CLASS, ()


# ---------------------------------------------------------------------------
# Collection walkers
# ---------------------------------------------------------------------------


def _walk_affected(
    ecn: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Yield affected objects with exact object and link paths in original order.

    Traverses ``$.CNAffectLinks[*].AffectedObjects[*]``.  Non-dict link or
    object members are silently skipped; the caller is responsible for
    recording path-specific qualifiers.

    Args:
        ecn: A single decoded ECN dict from the source JSON array.

    Yields:
        ``(source_path, link_path, link_dict, object_dict)`` tuples in
        document order.
    """
    links = ecn.get("CNAffectLinks", [])
    if not isinstance(links, list):
        return
    for link_index, link in enumerate(links):
        if not isinstance(link, dict):
            continue
        objects = link.get("AffectedObjects", [])
        if not isinstance(objects, list):
            continue
        for object_index, source_object in enumerate(objects):
            if isinstance(source_object, dict):
                link_path = f"$.CNAffectLinks[{link_index}]"
                yield (
                    f"{link_path}.AffectedObjects[{object_index}]",
                    link_path,
                    link,
                    source_object,
                )


def _walk_resulting(
    ecn: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Yield resulting objects with exact three-level paths in original order.

    Traverses ``$.ImplementationPlan[*].ResultingLinks[*].ResultingObjects[*]``.
    Non-dict plan-item, link, or object members are silently skipped.

    Args:
        ecn: A single decoded ECN dict from the source JSON array.

    Yields:
        ``(source_path, link_path, link_dict, object_dict)`` tuples in
        document order.
    """
    plan = ecn.get("ImplementationPlan", [])
    if not isinstance(plan, list):
        return
    for plan_index, plan_item in enumerate(plan):
        if not isinstance(plan_item, dict):
            continue
        result_links = plan_item.get("ResultingLinks", [])
        if not isinstance(result_links, list):
            continue
        for link_index, link in enumerate(result_links):
            if not isinstance(link, dict):
                continue
            objects = link.get("ResultingObjects", [])
            if not isinstance(objects, list):
                continue
            for object_index, source_object in enumerate(objects):
                if isinstance(source_object, dict):
                    link_path = (
                        f"$.ImplementationPlan[{plan_index}]"
                        f".ResultingLinks[{link_index}]"
                    )
                    yield (
                        f"{link_path}.ResultingObjects[{object_index}]",
                        link_path,
                        link,
                        source_object,
                    )


# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------


def _normalize_state(
    raw_state: JsonValue,
) -> tuple[str | None, tuple[str, ...]]:
    """Normalise the State field from a string or a {Value, Display} dict.

    Returns ``(None, ("SOURCE_FIELD_SHAPE_UNSUPPORTED",))`` for any other shape
    so that the source value remains in ``raw_object`` but no prose is produced.

    Args:
        raw_state: The raw value of the ``State`` key from a source object.

    Returns:
        ``(state_string_or_none, qualifiers_tuple)`` where qualifiers is empty
        on success and contains ``SOURCE_FIELD_SHAPE_UNSUPPORTED`` on failure.
    """
    if raw_state is None:
        return None, ()
    if isinstance(raw_state, str):
        stripped = raw_state.strip()
        return stripped if stripped else None, ()
    if isinstance(raw_state, dict):
        value_field = raw_state.get("Value")
        if isinstance(value_field, str):
            stripped = value_field.strip()
            return stripped if stripped else None, ()
    # Unsupported shape (list, number, bool, or dict without a string Value).
    return None, ("SOURCE_FIELD_SHAPE_UNSUPPORTED",)


# ---------------------------------------------------------------------------
# Relationship-value extraction
# ---------------------------------------------------------------------------


def _extract_relationship_values(
    link: dict[str, Any],
) -> tuple[dict[str, JsonValue], tuple[str, ...]]:
    """Extract scalar relationship properties from an owning link object.

    Includes all scalar (non-dict, non-list) keys from the link.  Error-shaped
    strings (beginning ``wt.util.WTException:``) are retained as values but
    trigger a ``SOURCE_RELATION_ERROR_VALUE`` qualifier so downstream code can
    treat them with appropriate scepticism.

    Args:
        link: The owning link dict at the affected or resulting traversal level.

    Returns:
        ``(relationship_values, qualifiers)`` where ``relationship_values``
        contains only scalar entries and ``qualifiers`` is a tuple of
        ``SOURCE_RELATION_ERROR_VALUE`` entries.
    """
    values: dict[str, JsonValue] = {}
    qualifiers: list[str] = []
    for key, val in link.items():
        # Only scalar (non-nested) values are preserved.
        if isinstance(val, (dict, list)):
            continue
        values[key] = val
        if isinstance(val, str) and val.startswith("wt.util.WTException:"):
            qualifiers.append("SOURCE_RELATION_ERROR_VALUE")
    return values, tuple(qualifiers)


# ---------------------------------------------------------------------------
# Identifier helper for source-object fields
# ---------------------------------------------------------------------------


def _safe_identifier(value: object) -> str | None:
    """Trim external whitespace from an identifier field; return None if absent or blank.

    Does NOT case-fold, strip leading zeros, or coerce numeric-looking values.
    Non-string values that are not None are not coerced; they return None
    (the caller preserves the raw value in ``raw_object``).

    Args:
        value: Raw field value from the source object.

    Returns:
        The whitespace-trimmed string, or ``None`` when absent, blank, or
        non-string.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


# ---------------------------------------------------------------------------
# BOM cell value normalisation — public interface
# ---------------------------------------------------------------------------


def source_text(value: object) -> str | None:
    """Convert a BOM cell value to a trimmed string without case folding.

    For strings: trims external whitespace only.  Case, leading zeros, and
    internal whitespace are preserved unchanged.  Returns ``None`` for ``None``
    or a blank/whitespace-only string.

    For non-string non-``None`` values (e.g. integers or floats that openpyxl
    returns for numeric cells): converts via ``str()``.  Leading zeros already
    lost in numeric cells cannot be recovered this way.  Date/datetime objects
    are NOT converted here; use ``json_safe_cell`` or ``normalize_effectivity_value``
    for those.

    Args:
        value: Raw cell value from a BOM worksheet.

    Returns:
        The trimmed string, or ``None`` when absent or blank.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    # Non-string (int, float, etc.): convert to string.
    return str(value)


def ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return non-blank strings once each, preserving first-seen order.

    Strings that are empty or contain only whitespace are silently dropped.
    Duplicates beyond the first occurrence are also dropped.  The original
    order of first-occurrence values is preserved.

    Args:
        values: An iterable of strings to deduplicate.

    Returns:
        A tuple of unique non-blank strings in first-seen order.
    """
    return tuple(dict.fromkeys(v for v in values if v.strip()))


def json_safe_cell(value: object) -> tuple[JsonValue, tuple[str, ...]]:
    """Convert one workbook cell value to a JSON-safe form without dropping it.

    Conversion rules (applied in order):
    - ``None``, ``bool``, ``int``, finite ``float``, ``str``: returned unchanged.
    - ``date`` / ``datetime``: converted to their ISO-format string.
    - ``Decimal``: converted to its exact string representation.
    - Any other type (including non-finite floats): ``str(value)`` is returned
      together with the qualifier ``"SOURCE_CELL_TYPE_STRINGIFIED"``.

    The qualifier tuple is empty when no conversion was needed and non-empty
    only for the stringification fallback.

    Args:
        value: A raw cell value as returned by openpyxl or a test fixture.

    Returns:
        ``(json_value, qualifiers)`` where ``qualifiers`` is empty or contains
        ``"SOURCE_CELL_TYPE_STRINGIFIED"``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value, ()  # type: ignore[return-value]
    if isinstance(value, float) and math.isfinite(value):
        return value, ()
    # datetime must be checked before date because datetime is a subclass of date.
    if isinstance(value, datetime):
        return value.isoformat(), ()
    if isinstance(value, date):
        return value.isoformat(), ()
    if isinstance(value, Decimal):
        return str(value), ()
    # Fallback: opaque or unsupported type — stringify and flag.
    return str(value), ("SOURCE_CELL_TYPE_STRINGIFIED",)


def normalize_quantity(value: object) -> QuantityNormalization:
    """Accept only quantities exactly representable as HANA ``DECIMAL(31,14)``.

    Applies ``Decimal(str(value))`` to parse the raw input, then verifies
    that the integer part has at most 17 significant digits and the fractional
    part has at most 14 digits (after ignoring insignificant trailing zeros).
    Never rounds.  A nonblank failure returns ``value=None`` and the qualifier
    ``"QUANTITY_NOT_EXACTLY_REPRESENTABLE"``; the raw cell content is preserved
    in ``BomRow.row_json`` by the caller.

    Blank input (``None`` or a whitespace-only string) returns ``(None, ())``
    without a qualifier, indicating the field is simply absent.

    Args:
        value: Raw cell value from a BOM quantity column.

    Returns:
        A ``QuantityNormalization`` with either the accepted ``Decimal`` value
        or ``None`` and the rejection qualifier.
    """
    _REJECT = QuantityNormalization(None, ("QUANTITY_NOT_EXACTLY_REPRESENTABLE",))
    # Blank / absent input.
    if value is None:
        return QuantityNormalization(None, ())
    if isinstance(value, str) and not value.strip():
        return QuantityNormalization(None, ())
    # Parse.
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return _REJECT
    if not decimal_value.is_finite():
        return _REJECT
    # Normalise to remove insignificant trailing zeros, then check bounds.
    normalized = decimal_value.normalize()
    sign, digits, exponent = normalized.as_tuple()
    # Number of integer digits = len(digits) + exponent when exponent >= 0,
    # otherwise len(digits) minus any fractional digits.
    integer_digits = max(len(digits) + exponent, 0)
    fractional_digits = max(-exponent, 0)
    if integer_digits > 17 or fractional_digits > 14:
        return _REJECT
    return QuantityNormalization(decimal_value, ())


def fit_bom_identifier(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
    sheet_name: str,
    source_row_number: int,
) -> str | None:
    """Trim and bound-check a BOM identifier that serves as a join key.

    Join/source identifiers must never be truncated silently.  If the trimmed
    value exceeds ``maximum_length``, a ``SourceValidationError`` is raised
    with ``SOURCE_VALUE_TOO_LONG`` and the coordinates (field, sheet, row,
    maximum_length) — but NOT the value itself, to avoid embedding source
    content in error records.

    Args:
        value: Raw cell value.  Non-string non-``None`` values are coerced via
            ``source_text``; ``None`` and blank values return ``None``.
        field_name: Python field name of the target ``BomRow`` column (for
            diagnostics).
        maximum_length: Maximum allowed trimmed length in characters.
        sheet_name: Worksheet name used in the error ``details``.
        source_row_number: 1-based source row number used in the error ``details``.

    Returns:
        The trimmed identifier string, or ``None`` when absent/blank.

    Raises:
        SourceValidationError: When the trimmed value exceeds ``maximum_length``.
            The error code is ``"SOURCE_VALUE_TOO_LONG"`` and ``details`` carries
            ``field_name``, ``sheet_name``, ``source_row_number``, and
            ``maximum_length`` — never the value.
    """
    text = source_text(value)
    if text is None:
        return None
    if len(text) > maximum_length:
        raise SourceValidationError(
            "SOURCE_VALUE_TOO_LONG",
            "A source identifier exceeds its supported column length",
            status_code=400,
            details={
                "field_name": field_name,
                "sheet_name": sheet_name,
                "source_row_number": source_row_number,
                "maximum_length": maximum_length,
            },
        )
    return text


# ---------------------------------------------------------------------------
# Effectivity normalization — public interface
# ---------------------------------------------------------------------------


def normalize_effectivity_value(
    value: object,
    source_format: EffectivityFormat,
    field: EffectivityField,
) -> NormalizedEffectivity:
    """Parse one effectivity date cell without locale guessing or current-state inference.

    Parsing rules by format:

    - ``RAW_DD_MM_YYYY``: the raw text is parsed as ``%d.%m.%Y`` (European
      day-first locale).  ``"02.01.1900"`` is 2 January 1900, not 1 February.
    - ``EXCEL_TYPED_OR_ISO``: accepts a Python ``date`` or ``datetime`` object
      returned by openpyxl, or an ISO-format ``YYYY-MM-DD`` string.

    Migration placeholder logic:

    - ``is_migration_placeholder`` is ``True`` only when ``field`` is
      ``VALID_FROM`` and the successfully parsed year equals 1900.
    - ``VALID_TO`` year-1900 dates are never flagged as placeholders.

    Blank input (``None``, empty string, whitespace) produces all-null output
    with no warning — a blank cell means the field is absent, not that
    effectivity is infinite or that a negative fact has been observed.

    This function models intervals only.  It never computes an active/current
    boolean because no analysis-as-of date is available at parse time.

    Args:
        value: Raw cell value.  May be ``None``, a string, a ``date``, or a
            ``datetime``.
        source_format: Wire format of the cell in this workbook role.
        field: Whether this is a VALID_FROM or VALID_TO cell.

    Returns:
        A ``NormalizedEffectivity`` with ``raw_text``, ``parsed_date``,
        ``is_migration_placeholder``, and ``parse_warning``.
    """
    # Determine raw_text: date/datetime objects produce their ISO string;
    # everything else goes through source_text (strips whitespace, returns None
    # for blank).
    if isinstance(value, (date, datetime)):
        raw_text: str | None = value.isoformat()
    else:
        raw_text = source_text(value)

    if raw_text is None:
        return NormalizedEffectivity(None, None, False, None)

    # Parse the date, applying the appropriate locale.
    try:
        if source_format is EffectivityFormat.RAW_DD_MM_YYYY:
            parsed: date = datetime.strptime(raw_text, "%d.%m.%Y").date()
        elif isinstance(value, datetime):
            # datetime is a subclass of date; extract the date component.
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        else:
            # String in ISO format for the enriched/MBOM sheets.
            parsed = date.fromisoformat(raw_text)
    except (TypeError, ValueError):
        return NormalizedEffectivity(raw_text, None, False, "EFFECTIVITY_PARSE_FAILED")

    # Migration placeholder: VALID_FROM with year 1900 only.
    migration = (field is EffectivityField.VALID_FROM) and (parsed.year == 1900)
    return NormalizedEffectivity(raw_text, parsed, migration, None)


# ---------------------------------------------------------------------------
# Four-way raw EBOM classification heuristic
# ---------------------------------------------------------------------------


def classify_raw_row(
    item_value: object,
    type_value: object,
    component_number: object,
) -> BomRowClassification:
    """Classify one raw EBOM row using the documented versioned source heuristic.

    Precedence (applied in strict order):

    1. A whitespace-trimmed raw ``Item`` value that begins with an uppercase
       ``"D"`` → ``CONFIRMED_DOCUMENT``.
    2. Otherwise, a trimmed ``Component number`` matching
       ``^.+?\\s+ZV[DSG]\\s+\\S+\\s+\\S+$`` → ``DOCUMENT_REFERENCE_CANDIDATE``.
    3. Otherwise, an exact trimmed ``Typ == "ND"`` → ``MATERIAL_LIKE_OBSERVED``.
    4. Otherwise → ``UNRESOLVED``.

    Positions ``150`` and ``160`` are observed position values in the ``Item``
    column and receive no special classification; they fall through to rules 3/4
    like any other non-``D``-prefixed item.

    A blank ``Typ`` cell (empty string or ``None``) yields ``UNRESOLVED`` when
    neither of the first two rules fires.  A missing ``Typ`` header would be
    caught at workbook-contract validation time (Task 12).

    Only ``MATERIAL_LIKE_OBSERVED`` rows are seeded as candidate materials in
    downstream analysis; they each receive the qualifier
    ``"TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED"`` at seed time.

    Args:
        item_value: Raw cell value from the ``Item`` column (position/document key).
        type_value: Raw cell value from the ``Typ`` column.
        component_number: Raw cell value from the ``Component number`` column.

    Returns:
        One of ``CONFIRMED_DOCUMENT``, ``DOCUMENT_REFERENCE_CANDIDATE``,
        ``MATERIAL_LIKE_OBSERVED``, or ``UNRESOLVED``.
    """
    item = source_text(item_value)
    row_type = source_text(type_value)
    component = source_text(component_number)

    # Rule 1: D-prefix item → confirmed document node.
    if item is not None and item.startswith("D"):
        return BomRowClassification.CONFIRMED_DOCUMENT

    # Rule 2: Document-reference key embedded in the component string.
    if component is not None and DOCUMENT_REFERENCE.fullmatch(component):
        return BomRowClassification.DOCUMENT_REFERENCE_CANDIDATE

    # Rule 3: Exact "ND" type → material-like observation.
    if row_type == "ND":
        return BomRowClassification.MATERIAL_LIKE_OBSERVED

    # Rule 4: No conclusive evidence.
    return BomRowClassification.UNRESOLVED


# ---------------------------------------------------------------------------
# Source-object normalisation — public interface
# ---------------------------------------------------------------------------


def normalize_source_object(
    source_object: dict[str, JsonValue],
    owning_link: dict[str, JsonValue],
    role: SourceObjectRole,
    source_path: str,
    link_path: str,
    source_ordinal: int,
) -> NormalizedSourceObject:
    """Normalise one source object from an ECN affect or result link.

    Assigns a new canonical UUID, extracts and trims identifiers from the
    source object, classifies WTPart technical identity, normalises State,
    and extracts relationship values from the owning link.

    Args:
        source_object: The decoded source object dict from the ECN JSON.
        owning_link: The containing affect/result link dict used to extract
            relationship properties.
        role: Whether this object is ``AFFECTED`` or ``RESULTING``.
        source_path: Exact JSON-path string for this object, e.g.
            ``"$.CNAffectLinks[0].AffectedObjects[0]"``.
        link_path: Exact JSON-path string for the owning link, e.g.
            ``"$.CNAffectLinks[0]"``.
        source_ordinal: Zero-based index of this object within its role list.

    Returns:
        A frozen ``NormalizedSourceObject`` with all normalised fields,
        identity classification, and quality qualifiers.
    """
    qualifiers: list[str] = []

    # --- Technical identity classification ---
    raw_id = source_object.get("ID")
    raw_version_id = source_object.get("VersionID")
    is_wtpart, identity_qualifiers = _is_consistent_wtpart(raw_id, raw_version_id)
    qualifiers.extend(identity_qualifiers)

    # --- State normalisation ---
    state_val, state_qualifiers = _normalize_state(source_object.get("State"))
    qualifiers.extend(state_qualifiers)

    # --- Relationship values from owning link ---
    relationship_values, rel_qualifiers = _extract_relationship_values(
        owning_link  # type: ignore[arg-type]
    )
    qualifiers.extend(rel_qualifiers)

    # --- Identifier and display fields (trim only, no case folding) ---
    technical_id = _safe_identifier(raw_id)
    version_id = _safe_identifier(raw_version_id)

    # Number is the primary BOM join key — never truncate; reject over-limit values.
    # Note: importing fit_identifier from json_parser would create a circular import,
    # so the same length-guard logic is applied inline here.
    raw_number = source_object.get("Number")
    number: str | None
    if raw_number is None:
        number = None
    elif not isinstance(raw_number, str):
        number = None  # non-string: raw value preserved in raw_object
    else:
        stripped_number = raw_number.strip()
        if not stripped_number:
            number = None
        elif len(stripped_number) > _NUMBER_MAX:
            raise SourceValidationError(
                "SOURCE_VALUE_TOO_LONG",
                "Number exceeds its supported length",
                status_code=400,
                details={
                    "field": "Number",
                    "source_path": source_path,
                    "maximum_length": _NUMBER_MAX,
                },
            )
        else:
            number = stripped_number

    name = _safe_identifier(source_object.get("Name"))
    description = _safe_identifier(source_object.get("Description"))
    object_type = _safe_identifier(source_object.get("ObjectType"))
    revision = _safe_identifier(source_object.get("Revision"))
    version = _safe_identifier(source_object.get("Version"))
    created_on = _safe_identifier(source_object.get("CreatedOn"))
    last_modified = _safe_identifier(source_object.get("LastModified"))
    source_subtype = _safe_identifier(source_object.get("Subtype"))

    return NormalizedSourceObject(
        source_object_id=str(uuid.uuid4()),
        source_path=source_path,
        link_path=link_path,
        role=role,
        source_ordinal=source_ordinal,
        technical_id=technical_id,
        version_id=version_id,
        number=number,
        name=name,
        description=description,
        object_type=object_type,
        revision=revision,
        version=version,
        state=state_val,
        created_on=created_on,
        last_modified=last_modified,
        source_subtype=source_subtype,
        relationship_values=relationship_values,
        raw_object=dict(source_object),  # type: ignore[arg-type]
        is_technical_wtpart=is_wtpart,
        quality_qualifiers=tuple(qualifiers),
    )
