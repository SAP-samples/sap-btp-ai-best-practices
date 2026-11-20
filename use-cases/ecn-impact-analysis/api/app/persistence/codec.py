"""Canonical JSON serialization, HANA LOB boundary helpers, and domain row mappers.

This module owns all conversion between Python domain objects and the flat
column-keyed dicts that the repository layer passes to or receives from
SQLAlchemy ``text()`` statements.

Design decisions
----------------
* Canonical JSON is produced with ``ensure_ascii=False``, ``sort_keys=True``,
  and ``separators=(",", ":")``.  The hash recorded in ``EVIDENCE_HASH`` is
  SHA-256 over the exact canonical UTF-8 bytes so that two encoders with the
  same input always produce the same digest.
* LOB readers handle the three shapes the hdbcli driver may return: plain
  ``str`` (already decoded NCLOB), ``bytes`` (BLOB), or an object with a
  ``read()`` method (streaming LOB).  Any other shape raises a safe
  ``PersistenceStateError``.
* ``usage_to_columns`` uses ``getattr`` for every token field so that the
  current four-field ``TokenUsage`` object is supported without modification,
  and a future richer object will be picked up automatically.
* Row mappers are explicit field-by-field; they never use pickle, ``vars()``,
  or ``asdict()``.  A malformed stored payload always raises
  ``PersistenceStateError("PERSISTED_JSON_INVALID", ...)``.
* ``artifact_from_row`` accepts an optional ``prefix`` so that it can decode
  columns from a joined query (e.g. ``"ecn.ARTIFACT_ID"``).
* Five pure API response mappers (Task 7) convert persistence records to API
  response types.  They perform no SQL and are the only conversion boundary
  consumed by HTTP route handlers.  Routes must not reconstruct response
  structures ad hoc.

Do not import SQLAlchemy here; this module has no connection lifecycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel

from app.ecn.analysis_models import AnalysisCompleteness, AnalysisResponse
from app.ecn.api_models import (
    AnalysisFilterState,
    ArtifactView,
    BusinessEvidenceDetailResponse,
    BusinessEvidenceItemView,
    BusinessEvidencePageResponse,
    BusinessEvidenceSourceView,
    BusinessEvidenceTechnicalView,
    EcnDetailResponse,
    EcnListItem,
    EvidenceSectionSummaries,
    EvidenceSummary,
    ImpactCounts,
)
from app.ecn.evidence.business import (
    BusinessEvidencePage,
    BusinessEvidenceRecord,
    business_evidence_record_from_item,
)
from app.ecn.errors import PersistenceStateError
from app.ecn.models import (
    AnalysisSectionKey,
    ArtifactKind,
    BomKind,
    BomRow,
    BomRowClassification,
    ChangeCandidate,
    ChangeCandidateKind,
    EvidenceBundle,
    EvidenceBundleSections,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceSource,
    EvidenceTier,
    JsonValue,
    ParseState,
    SourceArtifact,
    canonical_uuid,
)
from app.persistence.hana import from_hana_timestamp, to_hana_timestamp
from app.persistence.records import (
    AnalysisCompletion,
    ArtifactMetadataRecord,
    AttemptUsageColumns,
    EcnDetailRecord,
    EcnPublication,
    StoredEcnRecord,
    UploadPublication,
    UploadRecord,
)

# ---------------------------------------------------------------------------
# Canonical JSON serialization
# ---------------------------------------------------------------------------


def _json_compatible(value: object) -> JsonValue:
    """Convert approved immutable records and scalar types to lossless JSON values.

    Handles: ``None``, ``str``, ``int``, ``float``, ``bool``, ``StrEnum``,
    ``date``/``datetime``, ``Decimal``, Pydantic ``BaseModel`` (via
    ``model_dump(mode="json")``), dataclasses (via field iteration), tuples
    and lists (element-wise), and ``dict[str, …]`` (value-wise).  Any other
    type raises ``PersistenceStateError``.

    Args:
        value: Any approved domain value to convert.

    Returns:
        A JSON-compatible Python value (``str``, ``int``, ``float``, ``bool``,
        ``None``, ``list``, or ``dict[str, …]``).

    Raises:
        PersistenceStateError: When ``value`` is of an unsupported type.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value  # type: ignore[return-value]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, BaseModel):
        # Pydantic models: delegate to model_dump which handles all field
        # types (enums → str, datetimes → isoformat, etc.) then recurse so
        # that the result is processed through our own converter for
        # consistent representation.
        return _json_compatible(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_compatible(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_compatible(v) for k, v in value.items()}
    raise PersistenceStateError(
        "PERSISTENCE_VALUE_UNSUPPORTED",
        "A value cannot be serialized for persistence",
        status_code=500,
    )


def canonical_json(value: object) -> str:
    """Encode an approved value as deterministic Unicode JSON text.

    Uses ``ensure_ascii=False`` to preserve non-ASCII characters as-is,
    ``sort_keys=True`` for reproducible key order, and compact separators
    so that the output is byte-for-byte stable across Python versions.

    Args:
        value: Any domain value accepted by ``_json_compatible``.

    Returns:
        A UTF-8-safe JSON string with sorted keys and compact separators.
    """
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# HANA LOB boundary helpers
# ---------------------------------------------------------------------------


def read_lob_text(value: object) -> str:
    """Read a NCLOB or NVARCHAR column value and return a Python string.

    Accepts: a ``str`` (already decoded), a bytes-like object (decoded as
    UTF-8), or any object with a ``read()`` method that returns ``str`` or a
    bytes-like value. All other types raise ``PersistenceStateError``.

    Args:
        value: The column value as returned by the HANA driver.

    Returns:
        The decoded string content.

    Raises:
        PersistenceStateError: When ``value`` is of an unsupported type.
    """
    if isinstance(value, str):
        return value
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("utf-8")
        read_method = getattr(value, "read", None)
        if read_method is not None:
            raw = read_method()
            if isinstance(raw, str):
                return raw
            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PersistenceStateError(
            "LOB_READ_ERROR",
            "A NCLOB column value is not valid UTF-8",
            status_code=500,
        ) from exc
    raise PersistenceStateError(
        "LOB_READ_ERROR",
        "A NCLOB column value has an unsupported type",
        status_code=500,
    )


def read_lob_bytes(value: object) -> bytes:
    """Read a BLOB column value and return raw bytes.

    Accepts: a bytes-like object (copied to immutable ``bytes``), a ``str``
    (encoded as UTF-8), or any object with a ``read()`` method that returns a
    bytes-like value or ``str``. All other types raise ``PersistenceStateError``.

    Args:
        value: The column value as returned by the HANA driver.

    Returns:
        The raw bytes content.

    Raises:
        PersistenceStateError: When ``value`` is of an unsupported type.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    read_method = getattr(value, "read", None)
    if read_method is not None:
        raw = read_method()
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
        if isinstance(raw, str):
            return raw.encode("utf-8")
    raise PersistenceStateError(
        "LOB_READ_ERROR",
        "A BLOB column value has an unsupported type",
        status_code=500,
    )


# ---------------------------------------------------------------------------
# ISO-8601 boundary parsing
# ---------------------------------------------------------------------------


def _parse_iso_datetime(value: object) -> datetime | None:
    """Parse a stored ISO-8601 string into a datetime, tolerating absence.

    The source ResolutionDate is stored as the raw ISO-8601 string exactly as it
    arrived from the source system (e.g. ``"2017-01-25T11:56:05+01:00"``).  This
    boundary helper turns it into a datetime for the API layer.  Absent, blank, or
    unparseable values return ``None`` rather than raising, so a single malformed
    stored value can never fail an entire list or detail read; the raw value always
    remains available in ``source_json`` for audit.

    Args:
        value: The stored value (typically ``str`` from ``JSON_VALUE``) or ``None``.

    Returns:
        A parsed ``datetime`` (timezone-aware when the string carried an offset),
        or ``None`` when the value is absent, blank, or not ISO-8601.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dialect-agnostic row-key normalization
# ---------------------------------------------------------------------------


def normalize_row_keys(row: Mapping[str, object]) -> dict[str, object]:
    """Return a plain dict with every top-level HANA result-row key uppercased.

    sqlalchemy-hana may return column keys in a different case from the
    uppercase names used in SQL projections (e.g. ``"upload_id"`` instead of
    ``"UPLOAD_ID"``).  Normalizing to UPPERCASE at the read boundary makes all
    ``row["COLUMN_NAME"]`` accesses dialect-agnostic.

    This function is idempotent for already-uppercase mappings (including the
    ``RecordingEngine`` dicts used in unit tests), so the existing test suite
    requires no changes.

    IMPORTANT: only the top-level HANA result-row *keys* are uppercased.  The
    *values* — including NCLOB text that encodes JSON application payloads —
    are passed through completely unchanged.  Never call this on decoded JSON
    dicts or nested application structures.

    Args:
        row: A SQLAlchemy result-row mapping, a ``Mapping``, or a plain dict.
            SQLAlchemy's ``_mapping`` attribute is resolved automatically when
            the caller passes a ``Row`` rather than a ``RowMapping``.

    Returns:
        A new ``dict[str, object]`` with every key converted to an uppercase
        string; values are shallow-copied (references, not deep copies).
    """
    source: Mapping[str, object] = getattr(row, "_mapping", row)
    return {str(key).upper(): value for key, value in dict(source).items()}


# ---------------------------------------------------------------------------
# Token-usage column projection
# ---------------------------------------------------------------------------


def usage_to_columns(
    usage: object | None, *, usage_available: bool
) -> AttemptUsageColumns:
    """Map current or future normalized token usage without treating absence as zero.

    Uses ``getattr`` for every field so that the current four-field
    ``TokenUsage`` object works without modification, and a future object with
    additional attributes (``cache_write_input_tokens``, ``input_total_tokens``)
    is automatically detected.  For the current object, ``cached_input_tokens``
    maps to ``cache_read_input_tokens``; absent future fields map to ``None``.

    When ``usage_available=False``, all integer columns are ``None`` regardless
    of any zero defaults on the ``usage`` object.

    Args:
        usage: A token-usage object (typically ``TokenUsage``) or ``None``.
        usage_available: ``True`` when ``usage`` carries real provider data.

    Returns:
        An ``AttemptUsageColumns`` record ready for insertion.

    Raises:
        ValueError: When ``usage_available=True`` but ``usage`` is ``None``,
            or when any token count is a negative integer.
    """
    if not usage_available:
        return AttemptUsageColumns(None, None, None, None, None, None, False)
    if usage is None:
        raise ValueError("available usage requires a usage object")

    # cache_read_input_tokens: try the future attribute name first, then fall
    # back to the current four-field attribute name.
    cached = getattr(usage, "cache_read_input_tokens", None)
    if cached is None:
        cached = getattr(usage, "cached_input_tokens", None)

    values: dict[str, int | None] = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cache_read_input_tokens": cached,
        # Future attributes absent on the current class → NULL, not zero.
        "cache_write_input_tokens": getattr(usage, "cache_write_input_tokens", None),
        "input_total_tokens": getattr(usage, "input_total_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    if any(
        v is not None and (not isinstance(v, int) or v < 0)
        for v in values.values()
    ):
        raise ValueError("token usage values must be nonnegative integers or null")
    return AttemptUsageColumns(**values, usage_available=True)


# ---------------------------------------------------------------------------
# SourceArtifact row mapper
# ---------------------------------------------------------------------------


def artifact_to_row(artifact: SourceArtifact) -> dict[str, object]:
    """Convert a ``SourceArtifact`` to a flat dict keyed by HANA column names.

    ``CREATED_AT`` is converted to a naive UTC datetime via
    ``to_hana_timestamp`` so that the value is safe for a HANA TIMESTAMP column.
    ``CONTENT_BYTES`` is stored as raw bytes; ``NORMALIZED_CONTENT`` is stored
    as a string or ``None``.

    Args:
        artifact: A fully validated ``SourceArtifact`` domain record.

    Returns:
        A dict with 11 entries keyed by the exact column names in
        ``ECN_IMPACT_SOURCE_ARTIFACT``.
    """
    return {
        "ARTIFACT_ID": artifact.artifact_id,
        "ARTIFACT_KIND": artifact.artifact_kind.value,
        "FILE_NAME": artifact.file_name,
        "MEDIA_TYPE": artifact.media_type,
        "BYTE_SIZE": artifact.byte_size,
        "SHA256": artifact.sha256,
        "CONTENT_BYTES": artifact.content_bytes,
        "NORMALIZED_CONTENT": artifact.normalized_content,
        "PARSE_STATE": artifact.parse_state.value,
        "PARSE_ERROR_CODE": artifact.parse_error_code,
        "CREATED_AT": to_hana_timestamp(artifact.created_at),
    }


def artifact_from_row(
    row: Mapping[str, object], prefix: str = ""
) -> SourceArtifact:
    """Reconstruct a ``SourceArtifact`` from a HANA column row mapping.

    The optional ``prefix`` prepends to every column name so that joined
    queries can supply per-table aliases (e.g. ``prefix="ecn_artifact."``).
    ``CREATED_AT`` is restored with ``tzinfo=utc`` via ``from_hana_timestamp``.
    ``CONTENT_BYTES`` and ``NORMALIZED_CONTENT`` are passed through the LOB
    boundary helpers so that hdbcli-style LOB objects are transparently decoded.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences between driver and application code are invisible to the rest
    of this function.  When a ``prefix`` is supplied the lookup key is
    uppercased as ``(prefix + name).upper()`` so that mixed-case aliased
    columns (e.g. ``"ecn_artifact.ARTIFACT_ID"``) are resolved correctly after
    normalization.

    Args:
        row: A mapping of column name → column value as returned by the driver.
        prefix: Optional string prepended to every column name; the entire
            ``prefix + name`` string is uppercased for the lookup.

    Returns:
        A fully validated ``SourceArtifact`` domain record.
    """
    row = normalize_row_keys(row)

    def col(name: str) -> object:
        return row[(prefix + name).upper()]

    normalized_raw = col("NORMALIZED_CONTENT")
    normalized: str | None = None
    if normalized_raw is not None:
        normalized = read_lob_text(normalized_raw)

    return SourceArtifact(
        artifact_id=str(col("ARTIFACT_ID")),
        artifact_kind=ArtifactKind(str(col("ARTIFACT_KIND"))),
        file_name=str(col("FILE_NAME")),
        media_type=str(col("MEDIA_TYPE")),
        byte_size=int(col("BYTE_SIZE")),  # type: ignore[arg-type]
        sha256=str(col("SHA256")),
        content_bytes=read_lob_bytes(col("CONTENT_BYTES")),
        normalized_content=normalized,
        parse_state=ParseState(str(col("PARSE_STATE"))),
        parse_error_code=(str(col("PARSE_ERROR_CODE")) if col("PARSE_ERROR_CODE") is not None else None),
        created_at=from_hana_timestamp(col("CREATED_AT")),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# BomRow row mapper
# ---------------------------------------------------------------------------


def bom_row_to_row(row: BomRow) -> dict[str, object]:
    """Convert a ``BomRow`` to a flat dict keyed by HANA column names.

    ``ROW_JSON`` encodes the full ``row_json`` tuple as a canonical JSON array.
    ``QUALITY_JSON`` encodes the quality metadata as a canonical JSON object
    with exactly two keys: ``details`` and ``qualifiers``.  Boolean columns
    (``IS_DOCUMENTATION``, ``IS_MIGRATION_DATE``) are stored as ``0``/``1``
    TINYINT values.  ``QUANTITY`` is stored as a ``Decimal`` (HANA accepts
    Python ``Decimal`` for DECIMAL columns).  Date columns are stored as
    Python ``date`` objects.

    Args:
        row: A fully validated ``BomRow`` domain record.

    Returns:
        A dict with 35 entries keyed by the exact column names in
        ``ECN_IMPACT_BOM_ROW``.
    """
    row_json_str = canonical_json(list(row.row_json))
    quality_json_str = canonical_json(
        {
            "details": row.quality_details,
            "qualifiers": list(row.quality_qualifiers),
        }
    )
    return {
        "BOM_ROW_ID": row.bom_row_id,
        "ARTIFACT_ID": row.artifact_id,
        "BOM_KIND": row.bom_kind.value,
        "SHEET_NAME": row.sheet_name,
        "SOURCE_ROW_NUMBER": row.source_row_number,
        "PARENT_BOM_ROW_ID": row.parent_bom_row_id,
        "DOTTED_LEVEL": row.dotted_level,
        "HIERARCHY_PATH": row.hierarchy_path,
        "MAIN_VALUE": row.main_value,
        "SUB_VALUE": row.sub_value,
        "VC_VALUE": row.vc_value,
        "LEVEL_4_VALUE": row.level_4_value,
        "LEVEL_5_VALUE": row.level_5_value,
        "LEVEL_6_VALUE": row.level_6_value,
        "ITEM_VALUE": row.item_value,
        "POSITION_VALUE": row.position_value,
        "MATERIAL_NUMBER": row.material_number,
        "PARENT_MATERIAL_NUMBER": row.parent_material_number,
        "COMPONENT_NUMBER": row.component_number,
        "CHANGE_NUMBER": row.change_number,
        "CHANGE_NUMBER_TO": row.change_number_to,
        "DESCRIPTION_TEXT": row.description_text,
        "QUANTITY": row.quantity,
        "UNIT": row.unit,
        "PLANT": row.plant,
        "ALTERNATIVE": row.alternative,
        "VALID_FROM_RAW": row.valid_from_raw,
        "VALID_FROM_DATE": row.valid_from_date,
        "VALID_TO_RAW": row.valid_to_raw,
        "VALID_TO_DATE": row.valid_to_date,
        "ROW_CLASSIFICATION": row.row_classification.value,
        "IS_DOCUMENTATION": 1 if row.is_documentation else 0,
        "IS_MIGRATION_DATE": 1 if row.is_migration_date else 0,
        "ROW_JSON": row_json_str,
        "QUALITY_JSON": quality_json_str,
    }


def bom_row_from_row(row: Mapping[str, object]) -> BomRow:
    """Reconstruct a ``BomRow`` from a HANA column row mapping.

    ``ROW_JSON`` is parsed as a JSON array and converted to a tuple of dicts.
    ``QUALITY_JSON`` must be a JSON object with exactly ``details`` and
    ``qualifiers`` keys; any other structure raises ``PersistenceStateError``.
    TINYINT booleans are converted via ``bool()``.  ``QUANTITY`` is
    reconstructed as a ``Decimal`` from the string representation.
    ``HIERARCHY_PATH`` and ``DESCRIPTION_TEXT`` are NCLOB columns that may
    be ``None``; non-null values are passed through ``read_lob_text``.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: A mapping of column name → column value as returned by the driver.

    Returns:
        A fully validated ``BomRow`` domain record.

    Raises:
        PersistenceStateError: When ``ROW_JSON`` or ``QUALITY_JSON`` is
            malformed or has an unexpected structure.
    """
    row = normalize_row_keys(row)
    row_json_text = read_lob_text(row["ROW_JSON"])

    try:
        row_json_raw = json.loads(row_json_text)
    except json.JSONDecodeError as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "ROW_JSON could not be parsed as JSON",
            status_code=500,
        ) from exc

    if not isinstance(row_json_raw, list):
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "ROW_JSON must be a JSON array",
            status_code=500,
        )
    row_json: tuple[dict[str, JsonValue], ...] = tuple(row_json_raw)

    quality_text = read_lob_text(row["QUALITY_JSON"])

    try:
        quality_raw = json.loads(quality_text)
    except json.JSONDecodeError as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "QUALITY_JSON could not be parsed as JSON",
            status_code=500,
        ) from exc

    if not isinstance(quality_raw, dict) or set(quality_raw.keys()) != {"details", "qualifiers"}:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "QUALITY_JSON must be an object with exactly 'details' and 'qualifiers'",
            status_code=500,
        )
    quality_details: dict[str, JsonValue] = quality_raw["details"]
    quality_qualifiers: tuple[str, ...] = tuple(quality_raw["qualifiers"])

    quantity_raw = row["QUANTITY"]
    quantity = Decimal(str(quantity_raw)) if quantity_raw is not None else None

    hierarchy_raw = row["HIERARCHY_PATH"]
    hierarchy_path = read_lob_text(hierarchy_raw) if hierarchy_raw is not None else None

    desc_raw = row["DESCRIPTION_TEXT"]
    description_text = read_lob_text(desc_raw) if desc_raw is not None else None

    valid_from_date_raw = row["VALID_FROM_DATE"]
    valid_from_date: date | None = None
    if valid_from_date_raw is not None:
        if isinstance(valid_from_date_raw, datetime):
            valid_from_date = valid_from_date_raw.date()
        elif isinstance(valid_from_date_raw, date):
            valid_from_date = valid_from_date_raw
        else:
            valid_from_date = date.fromisoformat(str(valid_from_date_raw))

    valid_to_date_raw = row["VALID_TO_DATE"]
    valid_to_date: date | None = None
    if valid_to_date_raw is not None:
        if isinstance(valid_to_date_raw, datetime):
            valid_to_date = valid_to_date_raw.date()
        elif isinstance(valid_to_date_raw, date):
            valid_to_date = valid_to_date_raw
        else:
            valid_to_date = date.fromisoformat(str(valid_to_date_raw))

    return BomRow(
        bom_row_id=str(row["BOM_ROW_ID"]),
        artifact_id=str(row["ARTIFACT_ID"]),
        sheet_name=str(row["SHEET_NAME"]),
        bom_kind=BomKind(str(row["BOM_KIND"])),
        source_row_number=int(row["SOURCE_ROW_NUMBER"]),  # type: ignore[arg-type]
        parent_bom_row_id=(
            str(row["PARENT_BOM_ROW_ID"]) if row["PARENT_BOM_ROW_ID"] is not None else None
        ),
        dotted_level=(
            str(row["DOTTED_LEVEL"]) if row["DOTTED_LEVEL"] is not None else None
        ),
        hierarchy_path=hierarchy_path,
        main_value=(str(row["MAIN_VALUE"]) if row["MAIN_VALUE"] is not None else None),
        sub_value=(str(row["SUB_VALUE"]) if row["SUB_VALUE"] is not None else None),
        vc_value=(str(row["VC_VALUE"]) if row["VC_VALUE"] is not None else None),
        level_4_value=(
            str(row["LEVEL_4_VALUE"]) if row["LEVEL_4_VALUE"] is not None else None
        ),
        level_5_value=(
            str(row["LEVEL_5_VALUE"]) if row["LEVEL_5_VALUE"] is not None else None
        ),
        level_6_value=(
            str(row["LEVEL_6_VALUE"]) if row["LEVEL_6_VALUE"] is not None else None
        ),
        item_value=(str(row["ITEM_VALUE"]) if row["ITEM_VALUE"] is not None else None),
        position_value=(
            str(row["POSITION_VALUE"]) if row["POSITION_VALUE"] is not None else None
        ),
        material_number=(
            str(row["MATERIAL_NUMBER"]) if row["MATERIAL_NUMBER"] is not None else None
        ),
        parent_material_number=(
            str(row["PARENT_MATERIAL_NUMBER"])
            if row["PARENT_MATERIAL_NUMBER"] is not None
            else None
        ),
        component_number=(
            str(row["COMPONENT_NUMBER"]) if row["COMPONENT_NUMBER"] is not None else None
        ),
        change_number=(
            str(row["CHANGE_NUMBER"]) if row["CHANGE_NUMBER"] is not None else None
        ),
        change_number_to=(
            str(row["CHANGE_NUMBER_TO"]) if row["CHANGE_NUMBER_TO"] is not None else None
        ),
        description_text=description_text,
        unit=(str(row["UNIT"]) if row["UNIT"] is not None else None),
        plant=(str(row["PLANT"]) if row["PLANT"] is not None else None),
        alternative=(str(row["ALTERNATIVE"]) if row["ALTERNATIVE"] is not None else None),
        valid_from_raw=(
            str(row["VALID_FROM_RAW"]) if row["VALID_FROM_RAW"] is not None else None
        ),
        valid_to_raw=(
            str(row["VALID_TO_RAW"]) if row["VALID_TO_RAW"] is not None else None
        ),
        quantity=quantity,
        valid_from_date=valid_from_date,
        valid_to_date=valid_to_date,
        row_classification=BomRowClassification(str(row["ROW_CLASSIFICATION"])),
        is_documentation=bool(row["IS_DOCUMENTATION"]),
        is_migration_date=bool(row["IS_MIGRATION_DATE"]),
        row_json=row_json,
        quality_qualifiers=quality_qualifiers,
        quality_details=quality_details,
    )


# ---------------------------------------------------------------------------
# UploadRecord row mapper
# ---------------------------------------------------------------------------


def upload_record_from_row(row: Mapping[str, object]) -> UploadRecord:
    """Reconstruct an ``UploadRecord`` from a HANA upload table row.

    The ``CREATED_BY``, ``PARSER_VERSION``, and logic-version columns present
    in the upload table are intentionally omitted; ``UploadRecord`` is the
    minimal read projection used by the HTTP layer.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: A mapping of column name → value for the upload table.

    Returns:
        A frozen ``UploadRecord`` instance.
    """
    row = normalize_row_keys(row)
    return UploadRecord(
        upload_id=str(row["UPLOAD_ID"]),
        request_id=str(row["REQUEST_ID"]),
        ecn_artifact_id=str(row["ECN_ARTIFACT_ID"]),
        bom_artifact_id=str(row["BOM_ARTIFACT_ID"]),
        rule_artifact_id=(
            str(row["RULE_ARTIFACT_ID"]) if row["RULE_ARTIFACT_ID"] is not None else None
        ),
        bom_reused=bool(row["BOM_REUSED"]),
        imported_ecn_count=int(row["IMPORTED_ECN_COUNT"]),  # type: ignore[arg-type]
        publication_order=int(row["PUBLICATION_ORDER"]),  # type: ignore[arg-type]
        created_at=from_hana_timestamp(row["CREATED_AT"]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# StoredEcnRecord row mapper
# ---------------------------------------------------------------------------


def _decode_change_candidate(data: dict[str, object]) -> ChangeCandidate:
    """Reconstruct one ``ChangeCandidate`` from its canonical JSON representation.

    Args:
        data: A dict with exactly the fields of ``ChangeCandidate``.

    Returns:
        A frozen ``ChangeCandidate`` instance.

    Raises:
        PersistenceStateError: When the dict is missing required keys or
            contains unrecognised enum values.
    """
    try:
        return ChangeCandidate(
            candidate_id=str(data["candidate_id"]),
            kind=ChangeCandidateKind(str(data["kind"])),
            affected_object_ids=tuple(str(x) for x in data["affected_object_ids"]),  # type: ignore[union-attr]
            resulting_object_ids=tuple(str(x) for x in data["resulting_object_ids"]),  # type: ignore[union-attr]
            source_paths=tuple(str(x) for x in data["source_paths"]),  # type: ignore[union-attr]
            qualifiers=tuple(str(x) for x in data["qualifiers"]),  # type: ignore[union-attr]
            explicit_relation_text=(
                str(data["explicit_relation_text"])
                if data.get("explicit_relation_text") is not None
                else None
            ),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "A stored ChangeCandidate record is malformed",
            status_code=500,
        ) from exc


def stored_ecn_from_row(row: Mapping[str, object]) -> StoredEcnRecord:
    """Reconstruct a ``StoredEcnRecord`` from a HANA ECN table row.

    ``SOURCE_JSON`` and ``CHANGE_CANDIDATES_JSON`` are NCLOB columns that
    are decoded via ``read_lob_text`` then parsed with ``json.loads``.
    ``DESCRIPTION_TEXT`` is nullable NCLOB.  ``CREATED_AT`` is naive UTC and
    is restored via ``from_hana_timestamp``.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences are invisible to the rest of this function.

    Args:
        row: A mapping of column name → value for the ECN table row.

    Returns:
        A frozen ``StoredEcnRecord`` instance.

    Raises:
        PersistenceStateError: When any JSON column is malformed.
    """
    row = normalize_row_keys(row)
    try:
        source_json: dict[str, JsonValue] = json.loads(
            read_lob_text(row["SOURCE_JSON"])
        )
    except Exception as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "SOURCE_JSON column is malformed",
            status_code=500,
        ) from exc

    try:
        candidates_raw = json.loads(read_lob_text(row["CHANGE_CANDIDATES_JSON"]))
        change_candidates = tuple(
            _decode_change_candidate(item) for item in candidates_raw
        )
    except PersistenceStateError:
        raise
    except Exception as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "CHANGE_CANDIDATES_JSON column is malformed",
            status_code=500,
        ) from exc

    desc_raw = row["DESCRIPTION_TEXT"]
    description_text = read_lob_text(desc_raw) if desc_raw is not None else None

    # ResolutionDate is a top-level source field kept as its raw ISO-8601 string;
    # non-string source values are treated as absent.
    resolution_raw = source_json.get("ResolutionDate")
    resolution_date = resolution_raw if isinstance(resolution_raw, str) else None

    return StoredEcnRecord(
        ecn_row_id=str(row["ECN_ROW_ID"]),
        upload_id=str(row["UPLOAD_ID"]),
        source_ordinal=int(row["SOURCE_ORDINAL"]),  # type: ignore[arg-type]
        ecn_number=(str(row["ECN_NUMBER"]) if row["ECN_NUMBER"] is not None else None),
        title=(str(row["TITLE"]) if row["TITLE"] is not None else None),
        description_summary=(
            str(row["DESCRIPTION_SUMMARY"])
            if row["DESCRIPTION_SUMMARY"] is not None
            else None
        ),
        description_text=description_text,
        counts=ImpactCounts(
            affected_item_count=int(row["AFFECTED_ITEM_COUNT"]),  # type: ignore[arg-type]
            raw_ecn_match_count=int(row["RAW_ECN_MATCH_COUNT"]),  # type: ignore[arg-type]
            ebom_match_count=int(row["EBOM_MATCH_COUNT"]),  # type: ignore[arg-type]
            ebom_context_count=int(row["EBOM_CONTEXT_COUNT"]),  # type: ignore[arg-type]
            mbom_match_count=int(row["MBOM_MATCH_COUNT"]),  # type: ignore[arg-type]
            warning_count=int(row["WARNING_COUNT"]),  # type: ignore[arg-type]
        ),
        source_json=source_json,
        change_candidates=change_candidates,
        resolution_date=resolution_date,
        evidence_hash=str(row["EVIDENCE_HASH"]),
        parser_version=str(row["PARSER_VERSION"]),
        deterministic_logic_version=str(row["DETERMINISTIC_LOGIC_VERSION"]),
        rule_evaluator_version=str(row["RULE_EVALUATOR_VERSION"]),
        created_at=from_hana_timestamp(row["CREATED_AT"]),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# EvidenceBundle JSON decoder
# ---------------------------------------------------------------------------


def _decode_evidence_source(data: dict[str, object]) -> EvidenceSource:
    """Reconstruct one ``EvidenceSource`` from its canonical JSON representation.

    Args:
        data: A dict with the fields of ``EvidenceSource``.

    Returns:
        A frozen ``EvidenceSource`` instance.
    """
    return EvidenceSource(
        artifact_id=str(data["artifact_id"]),
        artifact_kind=ArtifactKind(str(data["artifact_kind"])),
        json_path=(str(data["json_path"]) if data.get("json_path") is not None else None),
        sheet_name=(
            str(data["sheet_name"]) if data.get("sheet_name") is not None else None
        ),
        bom_row_id=(
            str(data["bom_row_id"]) if data.get("bom_row_id") is not None else None
        ),
        source_row_number=(
            int(data["source_row_number"])  # type: ignore[arg-type]
            if data.get("source_row_number") is not None
            else None
        ),
    )


def _decode_evidence_item(data: dict[str, object]) -> EvidenceItem:
    """Reconstruct one ``EvidenceItem`` from its canonical JSON representation.

    Args:
        data: A dict with the fields of ``EvidenceItem``.

    Returns:
        A frozen ``EvidenceItem`` instance.
    """
    tier_raw = data.get("tier")
    tier = EvidenceTier(str(tier_raw)) if tier_raw is not None else None
    return EvidenceItem(
        evidence_id=str(data["evidence_id"]),
        section_key=AnalysisSectionKey(str(data["section_key"])),
        kind=EvidenceKind(str(data["kind"])),
        tier=tier,
        summary=str(data["summary"]),
        sources=tuple(
            _decode_evidence_source(s) for s in data["sources"]  # type: ignore[union-attr]
        ),
        values=dict(data["values"]),  # type: ignore[arg-type]
        related_ids=tuple(str(x) for x in data["related_ids"]),  # type: ignore[union-attr]
        quality_qualifiers=tuple(str(x) for x in data["quality_qualifiers"]),  # type: ignore[union-attr]
    )


def _decode_evidence_bundle_sections(data: dict[str, object]) -> EvidenceBundleSections:
    """Reconstruct ``EvidenceBundleSections`` from its canonical JSON representation.

    Args:
        data: A dict keyed by the six ``AnalysisSectionKey`` string values.

    Returns:
        A frozen ``EvidenceBundleSections`` instance.
    """

    def _items(key: str) -> tuple[EvidenceItem, ...]:
        return tuple(_decode_evidence_item(item) for item in data.get(key, []))  # type: ignore[union-attr]

    return EvidenceBundleSections(
        executive_summary=_items("executive_summary"),
        change_intent=_items("change_intent"),
        affected_materials=_items("affected_materials"),
        ebom_mbom_impact=_items("ebom_mbom_impact"),
        configuration_scope=_items("configuration_scope"),
        uncertainties_and_missing_evidence=_items("uncertainties_and_missing_evidence"),
    )


def _decode_evidence_context(data: dict[str, object]) -> EvidenceContext:
    """Reconstruct an ``EvidenceContext`` from its canonical JSON representation.

    ``analysis_as_of_date`` is parsed from an ISO date string or left as
    ``None`` when the JSON value is null.

    Args:
        data: A dict with the fields of ``EvidenceContext``.

    Returns:
        A frozen ``EvidenceContext`` instance.
    """
    aod_raw = data.get("analysis_as_of_date")
    analysis_as_of_date = (
        date.fromisoformat(str(aod_raw)) if aod_raw is not None else None
    )
    plant_assignment_raw = data.get("plant_assignment", {})
    if not isinstance(plant_assignment_raw, Mapping):
        raise TypeError("plant_assignment must be an object")
    return EvidenceContext(
        upload_id=str(data["upload_id"]),
        ecn_artifact_id=str(data["ecn_artifact_id"]),
        bom_artifact_id=str(data["bom_artifact_id"]),
        rule_artifact_id=(
            str(data["rule_artifact_id"])
            if data.get("rule_artifact_id") is not None
            else None
        ),
        parser_version=str(data["parser_version"]),
        deterministic_logic_version=str(data["deterministic_logic_version"]),
        rule_evaluator_version=str(data["rule_evaluator_version"]),
        analysis_as_of_date=analysis_as_of_date,
        plant_assignment=dict(plant_assignment_raw),
        dataset_context=tuple(data.get("dataset_context", ())),
        data_group=str(data.get("data_group", "customer")),
        rule_source_artifact_ids=dict(data.get("rule_source_artifact_ids", {})),
    )


def evidence_bundle_from_json(value: str) -> EvidenceBundle:
    """Reconstruct an ``EvidenceBundle`` from its canonical JSON string.

    The string must have been produced by ``canonical_json(bundle)`` or by
    an equivalent serializer that follows the same field order and type
    conventions.  Any structural mismatch, missing key, or unrecognised enum
    value raises ``PersistenceStateError("PERSISTED_JSON_INVALID")``.

    Args:
        value: A canonical JSON string encoding an ``EvidenceBundle``.

    Returns:
        A fully validated ``EvidenceBundle`` domain record.

    Raises:
        PersistenceStateError: When the JSON is malformed or any nested
            record cannot be reconstructed.
    """
    try:
        data = json.loads(value)
        return EvidenceBundle(
            ecn_row_id=str(data["ecn_row_id"]),
            context=_decode_evidence_context(data["context"]),
            sections=_decode_evidence_bundle_sections(data["sections"]),
            evidence_hash=str(data["evidence_hash"]),
        )
    except PersistenceStateError:
        raise
    except Exception as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "The stored evidence bundle could not be parsed",
            status_code=500,
        ) from exc


# ---------------------------------------------------------------------------
# AnalysisResponse JSON decoder
# ---------------------------------------------------------------------------


def analysis_response_from_json(value: str) -> AnalysisResponse:
    """Reconstruct an ``AnalysisResponse`` from its canonical JSON string.

    Delegates to Pydantic's ``model_validate_json`` which handles StrEnum
    coercion from JSON string values even in strict mode.

    Args:
        value: A JSON string encoding a full six-section analysis response.

    Returns:
        A validated ``AnalysisResponse`` Pydantic model.

    Raises:
        PersistenceStateError: When the JSON is malformed or structurally
            invalid according to the Pydantic schema.
    """
    try:
        return AnalysisResponse.model_validate_json(value, context={"historical": True})
    except Exception as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "The stored analysis response could not be parsed",
            status_code=500,
        ) from exc


# ---------------------------------------------------------------------------
# Upload-publication row encoders (Task 6)
# ---------------------------------------------------------------------------


def upload_publication_to_row(
    publication: UploadPublication, *, publication_order: int
) -> dict[str, object]:
    """Convert an ``UploadPublication`` to a flat dict keyed by upload-table column names.

    All 13 columns of ``ECN_IMPACT_UPLOAD`` are included.  ``BOM_REUSED`` is
    stored as ``1``/``0`` (TINYINT).  ``RULE_ARTIFACT_ID`` is ``None`` when no
    rule artifact was supplied.  ``CREATED_AT`` is converted to naive UTC via
    ``to_hana_timestamp``.

    Args:
        publication: A validated ``UploadPublication`` command record.
        publication_order: The monotonically increasing sequence integer
            allocated by ``COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1``.

    Returns:
        A dict with 13 entries keyed by the exact column names in
        ``ECN_IMPACT_UPLOAD``.
    """
    return {
        "UPLOAD_ID": publication.upload_id,
        "REQUEST_ID": publication.request_id,
        "PUBLICATION_ORDER": publication_order,
        "ECN_ARTIFACT_ID": publication.ecn_artifact.artifact_id,
        "BOM_ARTIFACT_ID": publication.bom_artifact.artifact_id,
        "RULE_ARTIFACT_ID": (
            publication.rule_artifact.artifact_id
            if publication.rule_artifact is not None
            else None
        ),
        "BOM_REUSED": 1 if publication.bom_reused else 0,
        "IMPORTED_ECN_COUNT": len(publication.ecns),
        "CREATED_BY": publication.actor_id,
        "CREATED_AT": to_hana_timestamp(publication.created_at),
        "PARSER_VERSION": publication.parser_version,
        "DETERMINISTIC_LOGIC_VERSION": publication.deterministic_logic_version,
        "RULE_EVALUATOR_VERSION": publication.rule_evaluator_version,
    }


def ecn_publication_to_row(
    publication: UploadPublication, item: EcnPublication
) -> dict[str, object]:
    """Convert one ``EcnPublication`` to a flat dict keyed by ECN-table column names.

    All 21 columns of ``ECN_IMPACT_ECN`` are included.  ``SOURCE_JSON`` and
    ``CHANGE_CANDIDATES_JSON`` are canonical JSON strings.  ``EVIDENCE_JSON``
    is the full ``EvidenceBundle`` serialised as canonical JSON.
    ``CREATED_AT`` is converted to naive UTC via ``to_hana_timestamp``.

    Args:
        publication: The owning ``UploadPublication`` (provides upload ID and
            version strings shared across all ECN rows in this batch).
        item: One ``EcnPublication`` carrying the normalized ECN, counts,
            change candidates, and evidence bundle.

    Returns:
        A dict with 21 entries keyed by the exact column names in
        ``ECN_IMPACT_ECN``.
    """
    return {
        "ECN_ROW_ID": item.ecn.ecn_row_id,
        "UPLOAD_ID": publication.upload_id,
        "SOURCE_ORDINAL": item.ecn.source_ordinal,
        "ECN_NUMBER": item.ecn.ecn_number,
        "TITLE": item.ecn.title,
        "DESCRIPTION_SUMMARY": item.ecn.description_summary,
        "DESCRIPTION_TEXT": item.ecn.description_text,
        "AFFECTED_ITEM_COUNT": item.counts.affected_item_count,
        "RAW_ECN_MATCH_COUNT": item.counts.raw_ecn_match_count,
        "EBOM_MATCH_COUNT": item.counts.ebom_match_count,
        "EBOM_CONTEXT_COUNT": item.counts.ebom_context_count,
        "MBOM_MATCH_COUNT": item.counts.mbom_match_count,
        "WARNING_COUNT": item.counts.warning_count,
        "SOURCE_JSON": canonical_json(item.ecn.source_json),
        "CHANGE_CANDIDATES_JSON": canonical_json(list(item.change_candidates)),
        "EVIDENCE_JSON": canonical_json(item.evidence),
        "EVIDENCE_HASH": item.evidence.evidence_hash,
        "PARSER_VERSION": publication.parser_version,
        "DETERMINISTIC_LOGIC_VERSION": publication.deterministic_logic_version,
        "RULE_EVALUATOR_VERSION": publication.rule_evaluator_version,
        "CREATED_AT": to_hana_timestamp(publication.created_at),
    }


def analysis_completion_to_row(completion: AnalysisCompletion) -> dict[str, object]:
    """Convert an ``AnalysisCompletion`` to a flat dict keyed by HANA analysis-table column names.

    All 18 columns of ``ECN_IMPACT_ANALYSIS`` are included.  ``CONTENT_JSON`` is
    the canonical JSON encoding of the full ``AnalysisResponse``.  ``CREATED_AT``
    is converted to naive UTC via ``to_hana_timestamp``.  All keys are UPPERCASE
    so that SQLAlchemy ``text()`` bind parameters match the dict directly.

    Args:
        completion: A fully validated ``AnalysisCompletion`` carrying the
            accepted response, completeness classification, evidence hash,
            and all pinned-version fields.

    Returns:
        A dict with 18 entries keyed by the exact column names in
        ``ECN_IMPACT_ANALYSIS``, ready for a parameterized INSERT.
    """
    return {
        "ANALYSIS_ID": completion.analysis_id,
        "ECN_ROW_ID": completion.ecn_row_id,
        "JOB_ID": completion.job_id,
        "SUCCESSFUL_ATTEMPT_ID": completion.successful_attempt_id,
        "COMPLETENESS_STATE": completion.completeness.value,
        "CONTENT_JSON": canonical_json(completion.response),
        "EVIDENCE_HASH": completion.evidence_hash,
        "PROMPT_VERSION": completion.prompt_version,
        "MODEL_NAME": completion.model_name,
        "MODEL_DEPLOYMENT": completion.model_deployment,
        "DETERMINISTIC_LOGIC_VERSION": completion.deterministic_logic_version,
        "RULE_EVALUATOR_VERSION": completion.rule_evaluator_version,
        "OUTPUT_SCHEMA_VERSION": completion.output_schema_version,
        "CREATED_AT": to_hana_timestamp(completion.created_at),
        "DISCOVERY_STATUS": completion.discovery_status,
        "DISCOVERY_ERROR_CODE": completion.discovery_error_code,
        "DISCOVERY_HASH": completion.discovery_hash,
        "DISCOVERED_EVIDENCE_JSON": completion.discovered_evidence_json,
    }


def job_publication_to_row(
    publication: UploadPublication, item: EcnPublication
) -> dict[str, object]:
    """Convert one ``EcnPublication`` to a flat dict keyed by analysis-job column names.

    All 13 columns of ``ECN_IMPACT_ANALYSIS_JOB`` are included.  The initial
    ``STATE`` is always ``"QUEUED"`` and ``ATTEMPT_COUNT`` is ``0``.
    ``STARTED_AT``, ``FINISHED_AT``, ``ERROR_CODE``, and ``ERROR_MESSAGE`` are
    all ``None`` at creation time.  ``CREATED_AT`` is converted to naive UTC.

    Args:
        publication: The owning ``UploadPublication`` (provides the creation
            timestamp).
        item: One ``EcnPublication`` carrying the pinned model and prompt
            configuration for this job.

    Returns:
        A dict with 13 entries keyed by the exact column names in
        ``ECN_IMPACT_ANALYSIS_JOB``.
    """
    return {
        "JOB_ID": item.job_id,
        "ECN_ROW_ID": item.ecn.ecn_row_id,
        "STATE": "READY",
        "ATTEMPT_COUNT": 0,
        "CREATED_AT": to_hana_timestamp(publication.created_at),
        "STARTED_AT": None,
        "FINISHED_AT": None,
        "PROMPT_VERSION": item.prompt_version,
        "OUTPUT_SCHEMA_VERSION": item.output_schema_version,
        "MODEL_NAME": item.model_name,
        "MODEL_CONFIG_JSON": item.model_config_json,
        "ERROR_CODE": None,
        "ERROR_MESSAGE": None,
    }


# ---------------------------------------------------------------------------
# ECN list row mapper (Task 7)
# ---------------------------------------------------------------------------


def ecn_list_item_from_row(row: Mapping[str, object]) -> EcnListItem:
    """Map one ECN list-query result row to an ``EcnListItem`` API response.

    Reads columns produced by the list SQL:
    ``ECN_ROW_ID``, ``ECN_NUMBER``, ``TITLE``, ``DESCRIPTION_SUMMARY``,
    ``SOURCE_ORDINAL``, the six count columns, ``UPLOADED_AT`` (aliased from
    ``U."CREATED_AT"``), ``SOURCE_FILE_NAME`` (aliased from ``SA."FILE_NAME"``),
    ``DERIVED_ANALYSIS_STATE`` (CASE alias), ``COMPLETENESS_STATE``
    (nullable, from the LEFT JOIN analysis table), and ``AFFECTED_PLANTS_JSON``
    (nullable JSON array of plant labels; ``None`` until first assessed), and
    ``REVIEW_OUTCOME`` (newest engineering review decision; ``None`` if none).

    ``display_ecn_number`` falls back to ``"Missing ECN ID"`` when
    ``ECN_NUMBER`` is ``NULL``.

    The row is normalized to uppercase keys at entry so that dialect casing
    differences (including aliased columns) are invisible to the rest of this
    function.

    Args:
        row: A column-keyed mapping from the list SQL result.

    Returns:
        A fully validated ``EcnListItem`` Pydantic model.
    """
    row = normalize_row_keys(row)
    ecn_number_raw = row["ECN_NUMBER"]
    ecn_number: str | None = str(ecn_number_raw) if ecn_number_raw is not None else None
    display_ecn_number = ecn_number if ecn_number else "Missing ECN ID"

    completeness_raw = row["COMPLETENESS_STATE"]
    completeness = (
        AnalysisCompleteness(str(completeness_raw)) if completeness_raw is not None else None
    )
    analysis_content = row.get("ANALYSIS_CONTENT_JSON")
    overall_impact = (
        analysis_response_from_json(read_lob_text(analysis_content)).overall_impact
        if analysis_content is not None
        else None
    )
    plants_raw = row.get("AFFECTED_PLANTS_JSON")
    affected_plants = json.loads(str(plants_raw)) if plants_raw is not None else None

    return EcnListItem(
        ecn_row_id=str(row["ECN_ROW_ID"]),
        ecn_number=ecn_number,
        display_ecn_number=display_ecn_number,
        title=(str(row["TITLE"]) if row["TITLE"] is not None else None),
        description_summary=(
            str(row["DESCRIPTION_SUMMARY"]) if row["DESCRIPTION_SUMMARY"] is not None else None
        ),
        source_ordinal=int(row["SOURCE_ORDINAL"]),  # type: ignore[arg-type]
        resolution_date=_parse_iso_datetime(row.get("RESOLUTION_DATE")),
        uploaded_at=from_hana_timestamp(row["UPLOADED_AT"]),  # type: ignore[arg-type]
        source_file_name=str(row["SOURCE_FILE_NAME"]),
        counts=ImpactCounts(
            affected_item_count=int(row["AFFECTED_ITEM_COUNT"]),  # type: ignore[arg-type]
            raw_ecn_match_count=int(row["RAW_ECN_MATCH_COUNT"]),  # type: ignore[arg-type]
            ebom_match_count=int(row["EBOM_MATCH_COUNT"]),  # type: ignore[arg-type]
            ebom_context_count=int(row["EBOM_CONTEXT_COUNT"]),  # type: ignore[arg-type]
            mbom_match_count=int(row["MBOM_MATCH_COUNT"]),  # type: ignore[arg-type]
            warning_count=int(row["WARNING_COUNT"]),  # type: ignore[arg-type]
        ),
        analysis_state=AnalysisFilterState(str(row["DERIVED_ANALYSIS_STATE"])),
        has_partial_analysis=bool(int(row.get("HAS_PARTIAL_ANALYSIS") or 0)),
        completeness=completeness,
        overall_impact=overall_impact,
        affected_plants=affected_plants,
        review_outcome=(str(row["REVIEW_OUTCOME"]) if row.get("REVIEW_OUTCOME") is not None else None),
    )


# ---------------------------------------------------------------------------
# Pure API response mappers (Task 7) — no SQL, no connection lifecycle
# ---------------------------------------------------------------------------


def artifact_view_from_record(
    artifact: SourceArtifact | ArtifactMetadataRecord,
) -> ArtifactView:
    """Map immutable artifact metadata without exposing bytes or normalized content.

    The returned ``ArtifactView`` carries only safe display fields.  Raw bytes
    (``content_bytes``) and parsed text (``normalized_content``) are
    intentionally excluded from the API surface.

    Args:
        artifact: Validated full artifact or metadata-only read record.

    Returns:
        An ``ArtifactView`` Pydantic model with safe metadata only.
    """
    return ArtifactView(
        artifact_id=artifact.artifact_id,
        kind=artifact.artifact_kind,
        file_name=artifact.file_name,
        sha256=artifact.sha256,
    )


def evidence_section_summaries_from_bundle(
    bundle: EvidenceBundle,
) -> EvidenceSectionSummaries:
    """Map all six deterministic evidence groups into compact API summaries.

    Iterates every ``AnalysisSectionKey`` in declaration order and converts
    each ``EvidenceItem`` to an ``EvidenceSummary`` (compact view without
    source pointers or structured values).  Source tuples become compact dicts;
    the stored domain records are not mutated.

    Args:
        bundle: The full ``EvidenceBundle`` for one ECN occurrence.

    Returns:
        An ``EvidenceSectionSummaries`` model with all six section lists.
    """
    return EvidenceSectionSummaries.model_validate(
        {
            section_key.value: [
                EvidenceSummary(
                    evidence_id=item.evidence_id,
                    kind=item.kind,
                    tier=item.tier,
                    summary=item.summary,
                    quality_qualifiers=list(item.quality_qualifiers),
                )
                for item in getattr(bundle.sections, section_key.value)
            ]
            for section_key in AnalysisSectionKey
        }
    )


def ecn_detail_response_from_record(record: EcnDetailRecord) -> EcnDetailResponse:
    """Flatten one persistence detail record into the exact public response contract.

    This is the single conversion boundary between ``EcnDetailRecord`` and
    ``EcnDetailResponse``.  HTTP route handlers must call this function; they
    must not reconstruct the response payload manually so that provenance,
    ordinal, and deterministic evidence groups are never accidentally omitted.

    Args:
        record: A fully populated ``EcnDetailRecord`` from the repository.

    Returns:
        An ``EcnDetailResponse`` Pydantic model ready for HTTP serialisation.
    """
    return EcnDetailResponse(
        ecn_row_id=record.ecn.ecn_row_id,
        ecn_number=record.ecn.ecn_number,
        display_ecn_number=record.ecn.ecn_number or "Missing ECN ID",
        title=record.ecn.title,
        description_summary=record.ecn.description_summary,
        description_text=record.ecn.description_text,
        source_ordinal=record.ecn.source_ordinal,
        source_status=record.ecn.source_status,
        requestor_name=record.ecn.requestor_name,
        affected_products=list(record.ecn.affected_products),
        resolution_date=_parse_iso_datetime(record.ecn.resolution_date),
        uploaded_at=record.upload.created_at,
        source_file_name=record.ecn_artifact.file_name,
        ecn_artifact=artifact_view_from_record(record.ecn_artifact),
        bom_artifact=artifact_view_from_record(record.bom_artifact),
        rule_artifact=(
            artifact_view_from_record(record.rule_artifact)
            if record.rule_artifact is not None
            else None
        ),
        counts=record.ecn.counts,
        job=record.job,
        analysis=record.analysis,
    )


def evidence_detail_response_from_item(
    ecn_row_id: str,
    item: EvidenceItem,
) -> BusinessEvidenceDetailResponse:
    """Map an evidence item only after repository same-ECN membership verification.

    ``ecn_row_id`` is validated as a canonical UUID here so that the
    response always carries a well-formed identity even if the caller already
    validated upstream.  The repository is responsible for confirming that
    ``item`` was retrieved from the bundle of ``ecn_row_id`` before calling
    this mapper.

    Args:
        ecn_row_id: Canonical UUID string of the owning ECN occurrence.
        item: The evidence item returned by ``EcnRepository.get_evidence``.

    Returns:
        An ``EvidenceDetailResponse`` ready for HTTP serialisation.

    Raises:
        ValueError: When ``ecn_row_id`` is not a canonical UUID string.
    """
    canonical_uuid(ecn_row_id)
    return BusinessEvidenceDetailResponse(
        ecn_row_id=ecn_row_id,
        item=business_evidence_item_view(
            business_evidence_record_from_item(item), include_technical=True
        ),
    )


def business_evidence_item_view(
    record: BusinessEvidenceRecord,
    *,
    include_technical: bool,
) -> BusinessEvidenceItemView:
    """Map one domain business record to its public, optionally technical DTO.

    Args:
        record: Grouped deterministic business evidence record.
        include_technical: Whether to include internal IDs, enums, and raw values.

    Returns:
        Public business evidence item.
    """
    technical = record.technical_details
    return BusinessEvidenceItemView(
        label=record.label,
        summary=record.summary,
        material_number=record.material_number,
        description=record.description,
        revision=record.revision,
        direction=record.direction,
        parent=record.parent,
        bom_path=list(record.bom_path),
        quantity=record.quantity,
        effectivity=record.effectivity,
        plant=record.plant,
        matched_row_count=record.matched_row_count,
        sources=[
            BusinessEvidenceSourceView(
                artifact_kind=source.artifact_kind,
                json_path=source.json_path,
                sheet_name=source.sheet_name,
                source_row_number=source.source_row_number,
            )
            for source in record.sources
        ],
        technical_details=(
            BusinessEvidenceTechnicalView(
                internal_evidence_ids=list(technical.internal_evidence_ids),
                raw_kinds=list(technical.raw_kinds),
                raw_tiers=list(technical.raw_tiers),
                raw_values=list(technical.raw_values),
                quality_qualifiers=list(technical.quality_qualifiers),
                related_ids=list(technical.related_ids),
            )
            if include_technical
            else None
        ),
    )


def business_evidence_page_response(
    page: BusinessEvidencePage,
    *,
    include_technical: bool,
) -> BusinessEvidencePageResponse:
    """Map one domain evidence page to the stable public response contract.

    Args:
        page: Validated business evidence page.
        include_technical: Whether each row includes raw audit details.

    Returns:
        Public paginated response.
    """
    return BusinessEvidencePageResponse(
        section=page.section,
        items=[
            business_evidence_item_view(item, include_technical=include_technical)
            for item in page.items
        ],
        page=page.page,
        page_size=page.page_size,
        total_items=page.total_items,
        total_pages=page.total_pages,
    )
