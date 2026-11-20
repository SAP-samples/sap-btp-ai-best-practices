"""UTF-8 ECN JSON decoding, field-fit helpers, and top-level ECN normalisation.

Provides:
- ``fit_identifier``: trim an identifier field; raise SourceValidationError if
  non-string or over the configured length.
- ``fit_display_text``: soft-truncate a display field; add a qualifier when
  truncated.
- ``parse_ecn_json``: decode one UTF-8 JSON array, normalise every ECN entry,
  and return a ``ParsedEcnJson`` with all normalised records.

Field-fit policy:
- NEVER truncate identifier/join-key fields; reject over-limit values with
  ``SOURCE_VALUE_TOO_LONG`` (safe field coordinates, no raw value in details).
- Truncate ONLY display ``title`` (500) and ``description_summary`` (2000) with
  ``DISPLAY_VALUE_TRUNCATED``; full originals remain in ``source_json``.
- Error-shaped strings beginning ``wt.util.WTException:`` are excluded from
  ``description_text`` with ``SOURCE_ERROR_VALUE_EXCLUDED_FROM_PROSE``.
- Non-string / error-shaped field values stay in ``source_json`` and add a
  normalisation qualifier; they are never stringified as prose.

Usage::

    from app.ecn.intake.json_parser import parse_ecn_json

    parsed = parse_ecn_json(file_bytes, artifact_id)
    for ecn in parsed.ecns:
        print(ecn.ecn_number, ecn.title)
"""

import json
import uuid
from typing import Any

from app.ecn.errors import SourceValidationError
from app.ecn.intake.normalization import (
    _walk_affected,
    _walk_resulting,
    normalize_source_object,
)
from app.ecn.models import (
    NormalizedEcn,
    NormalizedSourceObject,
    ParsedEcnJson,
    SourceObjectRole,
    canonical_uuid,
)

# ---------------------------------------------------------------------------
# Field-length limits (must match HANA column widths in table_contracts.py)
# ---------------------------------------------------------------------------

_ECN_NUMBER_MAX = 80
"""Maximum trimmed length for an ECN number; matches ECN_NUMBER NVARCHAR(80)."""

_TITLE_MAX = 500
"""Maximum stored length for an ECN display title; matched by TITLE NVARCHAR(500)."""

_DESCRIPTION_SUMMARY_MAX = 2000
"""Maximum stored length for description_summary; matched by DESCRIPTION_SUMMARY NVARCHAR(2000)."""

_ERROR_PREFIX = "wt.util.WTException:"
"""Prefix that marks a Windchill server-side error string."""

# ---------------------------------------------------------------------------
# Public field-fit helpers
# ---------------------------------------------------------------------------


def fit_identifier(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
    source_path: str,
) -> str | None:
    """Trim one source identifier without coercing it or silently truncating it.

    Returns ``None`` for absent or blank values.  Raises ``SourceValidationError``
    when ``value`` is non-string (and non-None), or when the trimmed string
    exceeds ``maximum_length``.  Raw values are never embedded in error details.

    Args:
        value: The raw field value from the source object.
        field_name: Human-readable name for the field (used in error messages).
        maximum_length: The maximum allowed length after trimming.
        source_path: The JSON path to this field (used in error details for
            safe diagnostic coordinates).

    Returns:
        The whitespace-trimmed string, or ``None`` when absent or blank.

    Raises:
        SourceValidationError: Code ``SOURCE_FIELD_TYPE_INVALID`` (HTTP 400) if
            ``value`` is non-string and non-None; code ``SOURCE_VALUE_TOO_LONG``
            (HTTP 400) if the trimmed string exceeds ``maximum_length``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise SourceValidationError(
            "SOURCE_FIELD_TYPE_INVALID",
            f"{field_name} must be text",
            status_code=400,
            details={"field": field_name, "source_path": source_path},
        )
    normalized = value.strip()
    if len(normalized) > maximum_length:
        raise SourceValidationError(
            "SOURCE_VALUE_TOO_LONG",
            f"{field_name} exceeds its supported length",
            status_code=400,
            details={
                "field": field_name,
                "source_path": source_path,
                "maximum_length": maximum_length,
            },
        )
    return normalized


def fit_display_text(
    value: object,
    *,
    maximum_length: int,
    qualifier: str,
) -> tuple[str | None, tuple[str, ...]]:
    """Soft-truncate a display text field, adding a qualifier when truncated.

    Non-string or blank values return ``(None, ())``.  When truncation occurs
    the full original is expected to remain in ``source_json``; this function
    only produces the stored (possibly shorter) representation.

    Args:
        value: The raw field value.
        maximum_length: The maximum character length to retain.
        qualifier: The qualifier code to add when truncation occurs (typically
            ``"DISPLAY_VALUE_TRUNCATED"``).

    Returns:
        ``(truncated_value_or_none, qualifiers_tuple)``.
    """
    if not isinstance(value, str):
        return None, ()
    stripped = value.strip()
    if not stripped:
        return None, ()
    if len(stripped) > maximum_length:
        return stripped[:maximum_length], (qualifier,)
    return stripped, ()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reject_nonstandard_constant(value: str) -> None:
    """Reject NaN and Infinity tokens that Python's JSON decoder otherwise accepts.

    Despite being deprecated since Python 3.1, ``parse_constant`` is still
    silently accepted by ``json.loads`` in Python 3.12 and the callback IS
    invoked for NaN/Infinity tokens (confirmed: ``json.loads('[NaN]')`` returns
    ``[float('nan')]`` without this guard in Python 3.12.9).  Removing this
    callback would silently accept non-standard IEEE constants.

    Args:
        value: The non-standard constant string encountered by the decoder.

    Raises:
        ValueError: Always; the message includes the offending token.
    """
    raise ValueError(f"non-standard JSON constant: {value}")


def _canonical_source_json(root: list[Any]) -> str:
    """Produce a compact, deterministic JSON string for the full source array.

    Uses compact separators, ``ensure_ascii=False`` to preserve Unicode, and
    sorted object keys so semantically equivalent JSON has stable text.

    Args:
        root: The fully decoded JSON root list.

    Returns:
        A compact JSON string.
    """
    return json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_description_text(
    ecn: dict[str, Any],
) -> tuple[str | None, list[str]]:
    """Build description_text from the three prose fields in fixed order.

    Includes only nonblank string values that do not begin with the Windchill
    error prefix.  Error-shaped strings add ``SOURCE_ERROR_VALUE_EXCLUDED_FROM_PROSE``.
    Non-string values silently skip (the caller handles non-string qualifiers
    separately).

    Args:
        ecn: The raw decoded ECN dict.

    Returns:
        ``(description_text_or_none, qualifiers)`` where qualifiers contains
        ``SOURCE_ERROR_VALUE_EXCLUDED_FROM_PROSE`` for each error-shaped value
        that was excluded.
    """
    _PROSE_KEYS = ("DescriptionSummary", "Description", "SOLUTION_DESCRIPTION")
    qualifiers: list[str] = []
    blocks: list[str] = []
    for key in _PROSE_KEYS:
        val = ecn.get(key)
        if not isinstance(val, str):
            continue
        if val.startswith(_ERROR_PREFIX):
            qualifiers.append("SOURCE_ERROR_VALUE_EXCLUDED_FROM_PROSE")
            continue
        stripped = val.strip()
        if stripped:
            blocks.append(f"{key}:\n{stripped}")
    text = "\n\n".join(blocks) if blocks else None
    return text, qualifiers


def _normalize_ecn(
    raw: Any,
    artifact_id: str,
    source_ordinal: int,
) -> NormalizedEcn:
    """Normalise one element from the ECN JSON root array.

    Validates that the element is a dict, maps top-level fields, builds
    description_text, and walks both affected and resulting collections.

    Args:
        raw: One element from the decoded JSON root list.
        artifact_id: Canonical UUID of the containing source artifact.
        source_ordinal: Zero-based index of this element in the source array.

    Returns:
        A frozen ``NormalizedEcn`` with all normalised fields.

    Raises:
        SourceValidationError: HTTP 400 if ``raw`` is not a dict (non-object
            array member), or if the ECN number string exceeds 80 characters.
    """
    if not isinstance(raw, dict):
        raise SourceValidationError(
            "ECN_JSON_MEMBER_NOT_OBJECT",
            f"Every ECN array member must be a JSON object (found at index {source_ordinal})",
            status_code=400,
            details={"source_ordinal": source_ordinal},
        )

    qualifiers: list[str] = []
    source_path = f"$[{source_ordinal}]"

    # --- ECN Number (identifier — hard reject if over 80 chars) ---
    raw_number = raw.get("Number")
    ecn_number: str | None
    if raw_number is None or (isinstance(raw_number, str) and not raw_number.strip()):
        ecn_number = None
    elif not isinstance(raw_number, str):
        # Non-string number: preserve in source_json, add qualifier, set to None.
        ecn_number = None
        qualifiers.append("SOURCE_FIELD_TYPE_INVALID")
    else:
        # Validates length; raises SourceValidationError on excess.
        ecn_number = fit_identifier(
            raw_number,
            field_name="Number",
            maximum_length=_ECN_NUMBER_MAX,
            source_path=source_path,
        )

    # --- Title (soft truncate to 500) ---
    raw_title = raw.get("Name")
    title, title_qualifiers = fit_display_text(
        raw_title, maximum_length=_TITLE_MAX, qualifier="DISPLAY_VALUE_TRUNCATED"
    )
    qualifiers.extend(title_qualifiers)

    # --- Description summary (soft truncate to 2000) ---
    raw_summary = raw.get("DescriptionSummary")
    description_summary, summary_qualifiers = fit_display_text(
        raw_summary,
        maximum_length=_DESCRIPTION_SUMMARY_MAX,
        qualifier="DISPLAY_VALUE_TRUNCATED",
    )
    qualifiers.extend(summary_qualifiers)

    # --- Description text (fixed prose order, error-shaped strings excluded) ---
    description_text, prose_qualifiers = _build_description_text(raw)
    qualifiers.extend(prose_qualifiers)

    # --- Affected objects ($.CNAffectLinks[*].AffectedObjects[*]) ---
    affected_objects: list[NormalizedSourceObject] = []
    raw_links = raw.get("CNAffectLinks")
    if raw_links is not None and not isinstance(raw_links, list):
        qualifiers.append("SOURCE_AFFECT_LINKS_COLLECTION_INVALID")
    else:
        for ordinal, (sp, lp, link, obj) in enumerate(
            _walk_affected(raw)  # type: ignore[arg-type]
        ):
            affected_objects.append(
                normalize_source_object(
                    obj, link, SourceObjectRole.AFFECTED, sp, lp, ordinal
                )
            )

    # --- Resulting objects ($.ImplementationPlan[*].ResultingLinks[*].ResultingObjects[*]) ---
    resulting_objects: list[NormalizedSourceObject] = []
    raw_plan = raw.get("ImplementationPlan")
    if raw_plan is not None and not isinstance(raw_plan, list):
        qualifiers.append("SOURCE_IMPLEMENTATION_PLAN_COLLECTION_INVALID")
    else:
        for ordinal, (sp, lp, link, obj) in enumerate(
            _walk_resulting(raw)  # type: ignore[arg-type]
        ):
            resulting_objects.append(
                normalize_source_object(
                    obj, link, SourceObjectRole.RESULTING, sp, lp, ordinal
                )
            )

    return NormalizedEcn(
        ecn_row_id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        source_ordinal=source_ordinal,
        ecn_number=ecn_number,
        title=title,
        description_summary=description_summary,
        description_text=description_text,
        affected_objects=tuple(affected_objects),
        resulting_objects=tuple(resulting_objects),
        source_json=raw,  # type: ignore[arg-type]
        normalization_qualifiers=tuple(qualifiers),
    )


# ---------------------------------------------------------------------------
# Public parse entry point
# ---------------------------------------------------------------------------


def parse_ecn_json(
    content: bytes,
    artifact_id: str,
    *,
    source_positions: tuple[int, ...] | None = None,
) -> ParsedEcnJson:
    """Decode one UTF-8 ECN array, normalise all entries, and return a ParsedEcnJson.

    Validates that ``artifact_id`` is a canonical UUID, decodes the bytes as
    strict UTF-8, parses the JSON while rejecting non-standard constants
    (NaN, Infinity), requires a top-level array, and normalises each member.

    Args:
        content: Raw upload bytes expected to contain a UTF-8 JSON array.
        source_positions: Optional one-based selection; original ordinals are retained.
        artifact_id: Canonical UUID string of the source artifact; used as the
            ``artifact_id`` on every produced ``NormalizedEcn``.

    Returns:
        A ``ParsedEcnJson`` with the tuple of normalised ECNs and the compact
        canonical JSON string of the source array.

    Raises:
        SourceValidationError: HTTP 400 for invalid UTF-8 (code
            ``ECN_JSON_ENCODING_INVALID``), JSON syntax errors (code
            ``ECN_JSON_SYNTAX_INVALID``), non-array root (code
            ``ECN_JSON_ROOT_INVALID``), non-object array members (code
            ``ECN_JSON_MEMBER_NOT_OBJECT``), or over-limit ECN number
            (code ``SOURCE_VALUE_TOO_LONG``).
    """
    canonical_uuid(artifact_id)

    # --- UTF-8 decode ---
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceValidationError(
            "ECN_JSON_ENCODING_INVALID",
            "ECN JSON must use UTF-8 encoding",
            status_code=400,
            details={"byte_offset": exc.start},
        ) from exc

    # --- JSON parse (no NaN/Infinity) ---
    try:
        root = json.loads(decoded, parse_constant=_reject_nonstandard_constant)
    except json.JSONDecodeError as exc:
        raise SourceValidationError(
            "ECN_JSON_SYNTAX_INVALID",
            "ECN JSON syntax is invalid",
            status_code=400,
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc
    except ValueError as exc:
        raise SourceValidationError(
            "ECN_JSON_SYNTAX_INVALID",
            "ECN JSON syntax is invalid",
            status_code=400,
            details={"line": 1, "column": 1},
        ) from exc

    del (
        decoded
    )  # Release the large decoded buffer before constructing normalized records.

    # --- Root must be an array ---
    if not isinstance(root, list):
        raise SourceValidationError(
            "ECN_JSON_ROOT_INVALID",
            "ECN JSON must contain a top-level array",
            status_code=400,
        )

    # A selection is one-based for operators; enumerate the original array so
    # source ordinals and nested paths never shift when importing a later batch.
    selected = None if source_positions is None else set(source_positions)
    if selected is not None and (
        not selected or min(selected) < 1 or max(selected) > len(root)
    ):
        raise SourceValidationError(
            "ECN_SELECTION_INVALID",
            "Select positions within the source artifact.",
            status_code=400,
        )
    ecns = tuple(
        _normalize_ecn(item, artifact_id, ordinal)
        for ordinal, item in enumerate(root)
        if selected is None or ordinal + 1 in selected
    )

    return ParsedEcnJson(
        ecns=ecns,
        normalized_content=_canonical_source_json(root)
        if selected is None
        else "",
    )
