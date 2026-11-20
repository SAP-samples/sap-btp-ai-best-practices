"""Bounded in-memory OOXML workbook parser for BOM evidence layers.

Provides two public functions:

- ``detect_sheet_roles`` – resolve exactly one worksheet per logical BOM role
  from normalized header signatures; raise ``SourceValidationError`` on
  missing or duplicate roles.
- ``parse_bom_workbook`` – validate bytes, open the workbook, detect roles,
  iterate cells with hard row/cell limits enforced during streaming, build
  ``BomRow`` objects for every nonblank data row, reconstruct enriched and raw
  hierarchies, and return a ``ParsedBomWorkbook``.

Safe error codes (all HTTP 400 ``SourceValidationError``):
- ``BOM_WORKBOOK_CORRUPT``           – not a ZIP / not openable OOXML.
- ``BOM_SHEET_ROLE_MISSING``         – fewer than one candidate for a role.
- ``BOM_SHEET_ROLE_DUPLICATE``       – more than one candidate for a role.
- ``BOM_REQUIRED_HEADER_DUPLICATE``  – a required/mapped header appears twice.
- ``BOM_ROW_LIMIT_EXCEEDED``         – total visited data rows crossed the limit.
- ``BOM_CELL_LIMIT_EXCEEDED``        – total visited cells crossed the limit.
- ``SOURCE_VALUE_TOO_LONG``          – a bounded identifier exceeds its HANA column width.

None of the error details contain raw cell content or source row payloads.

Usage::

    from app.ecn.intake.workbook_parser import parse_bom_workbook

    result = parse_bom_workbook(
        content=file_bytes,
        artifact_id="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        maximum_rows=100_000,
        maximum_cells=3_000_000,
    )
    for row in result.rows:
        print(row.bom_kind, row.component_number)
"""

from __future__ import annotations

import uuid
import zipfile
from io import BytesIO
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import openpyxl
from openpyxl import Workbook
from openpyxl.utils.exceptions import InvalidFileException

from app.ecn.errors import SourceValidationError
from app.ecn.models import (
    BomKind,
    BomRow,
    BomRowClassification,
    JsonValue,
    ParsedBomWorkbook,
    canonical_uuid,
)
from app.ecn.intake.normalization import (
    EffectivityField,
    EffectivityFormat,
    classify_raw_row,
    fit_bom_identifier,
    json_safe_cell,
    normalize_effectivity_value,
    normalize_quantity,
    ordered_unique,
    source_text,
)
from app.ecn.intake.workbook_contracts import (
    WORKBOOK_SHEET_CONTRACTS,
    WorkbookSheetContract,
    normalize_header,
)
from app.ecn.intake.hierarchy import (
    reconstruct_enriched_hierarchy,
    reconstruct_raw_hierarchy,
)

# ---------------------------------------------------------------------------
# Processing order for roles
# ---------------------------------------------------------------------------

_ROLE_ORDER: tuple[BomKind, ...] = (
    BomKind.ENRICHED_EBOM,
    BomKind.RAW_EBOM_130,
    BomKind.MBOM_130,
)
"""Fixed processing order for BOM roles; determines output row sequence."""

# ---------------------------------------------------------------------------
# Column-width limits for bounded NVARCHAR fields on BomRow.
# Each value MUST match the corresponding NVARCHAR(n) length in
# api/app/persistence/table_contracts.py (BOM_ROW_TABLE column definitions).
# NCLOB columns (description_text, hierarchy_path) are NOT listed here because
# they have no character-length bound at the parsing stage.
# ---------------------------------------------------------------------------

_FIELD_MAX_LENGTHS: dict[str, int] = {
    "dotted_level":           100,  # DOTTED_LEVEL NVARCHAR(100)
    "main_value":             200,  # MAIN_VALUE NVARCHAR(200)
    "sub_value":              200,  # SUB_VALUE NVARCHAR(200)
    "vc_value":               200,  # VC_VALUE NVARCHAR(200)
    "level_4_value":          200,  # LEVEL_4_VALUE NVARCHAR(200)
    "level_5_value":          200,  # LEVEL_5_VALUE NVARCHAR(200)
    "level_6_value":          200,  # LEVEL_6_VALUE NVARCHAR(200)
    "item_value":             200,  # ITEM_VALUE NVARCHAR(200)
    "position_value":         200,  # POSITION_VALUE NVARCHAR(200)
    "material_number":        200,  # MATERIAL_NUMBER NVARCHAR(200)
    "parent_material_number": 200,  # PARENT_MATERIAL_NUMBER NVARCHAR(200)
    "component_number":       200,  # COMPONENT_NUMBER NVARCHAR(200)
    "change_number":          200,  # CHANGE_NUMBER NVARCHAR(200)
    "change_number_to":       200,  # CHANGE_NUMBER_TO NVARCHAR(200)
    "unit":                    40,  # UNIT NVARCHAR(40)
    "alternative":            200,  # ALTERNATIVE NVARCHAR(200)
    "valid_from_raw":         100,  # VALID_FROM_RAW NVARCHAR(100)
    "valid_to_raw":           100,  # VALID_TO_RAW NVARCHAR(100)
}


def _fit_id(
    field_name: str,
    value: object,
    sheet_name: str,
    source_row_number: int,
) -> str | None:
    """Trim and bound-check a BomRow identifier using the HANA catalog width.

    Delegates to ``fit_bom_identifier`` with the maximum length looked up from
    ``_FIELD_MAX_LENGTHS``.  Raises ``SourceValidationError(SOURCE_VALUE_TOO_LONG)``
    with safe sheet/row/field coordinates when the trimmed value is too long.

    Args:
        field_name: BomRow Python field name (must be a key in _FIELD_MAX_LENGTHS).
        value: Raw cell value from the worksheet.
        sheet_name: Worksheet title for error diagnostics.
        source_row_number: 1-based source row index for error diagnostics.

    Returns:
        The trimmed identifier string, or ``None`` when absent/blank.

    Raises:
        SourceValidationError: When the trimmed length exceeds the catalog width.
    """
    return fit_bom_identifier(
        value,
        field_name=field_name,
        maximum_length=_FIELD_MAX_LENGTHS[field_name],
        sheet_name=sheet_name,
        source_row_number=source_row_number,
    )


def _guard_eff_raw(
    raw_text: str | None,
    field_name: str,
    sheet_name: str,
    source_row_number: int,
) -> str | None:
    """Guard a normalized effectivity raw-text string against NVARCHAR(100) width.

    The ``raw_text`` from ``normalize_effectivity_value`` is either a date ISO
    string (always short) or ``source_text(cell_value)`` — which could be an
    arbitrary-length string if the cell contained non-date text.  This helper
    rejects values that would exceed VALID_FROM_RAW / VALID_TO_RAW NVARCHAR(100)
    before the row is returned.

    Delegates to ``fit_bom_identifier`` so that the raised error carries
    safe field/sheet/row coordinates and never the raw value.

    Args:
        raw_text: Already-normalized raw text from NormalizedEffectivity.
        field_name: One of ``"valid_from_raw"`` or ``"valid_to_raw"``.
        sheet_name: Worksheet title for error diagnostics.
        source_row_number: 1-based source row index for error diagnostics.

    Returns:
        The raw text as-is when within the column width, or ``None`` when absent.

    Raises:
        SourceValidationError: When ``raw_text`` exceeds 100 characters.
    """
    return fit_bom_identifier(
        raw_text,
        field_name=field_name,
        maximum_length=_FIELD_MAX_LENGTHS[field_name],
        sheet_name=sheet_name,
        source_row_number=source_row_number,
    )


# ---------------------------------------------------------------------------
# Internal error helpers
# ---------------------------------------------------------------------------


def _workbook_error(code: str, message: str, **details: JsonValue) -> SourceValidationError:
    """Return a safe ``SourceValidationError`` with the given code.

    Args:
        code: Stable error code string.
        message: Human-readable public message safe for API responses.
        **details: Optional key/value diagnostic fields (no source content).

    Returns:
        A ``SourceValidationError`` with ``status_code=400``.
    """
    return SourceValidationError(
        code,
        message,
        status_code=400,
        details=dict(details) if details else None,
    )


# ---------------------------------------------------------------------------
# Public: detect_sheet_roles
# ---------------------------------------------------------------------------


def detect_sheet_roles(workbook: Workbook) -> dict[BomKind, str]:
    """Resolve exactly one worksheet per logical BOM role from normalized header signatures.

    Each worksheet's first row is read and normalized.  A worksheet is
    associated with a role when every string in the contract's
    ``required_headers`` is present in the normalized header set.  Extra
    worksheets that match no role are silently ignored.

    Args:
        workbook: An open openpyxl ``Workbook`` (caller owns lifecycle).

    Returns:
        A ``dict`` mapping each ``BomKind`` to its matching worksheet title.

    Raises:
        SourceValidationError: ``BOM_SHEET_ROLE_MISSING`` when fewer than one
            candidate is found for any role; ``BOM_SHEET_ROLE_DUPLICATE`` when
            more than one candidate is found.
    """
    candidates: dict[BomKind, list[str]] = {
        contract.bom_kind: [] for contract in WORKBOOK_SHEET_CONTRACTS
    }

    for worksheet in workbook.worksheets:
        first_row = next(
            worksheet.iter_rows(min_row=1, max_row=1, values_only=True), ()
        )
        normalized_set = {normalize_header(cell) for cell in first_row}
        for contract in WORKBOOK_SHEET_CONTRACTS:
            if set(contract.required_headers).issubset(normalized_set):
                candidates[contract.bom_kind].append(worksheet.title)

    _raise_for_missing_or_duplicate_roles(candidates)
    return {kind: names[0] for kind, names in candidates.items()}


def _raise_for_missing_or_duplicate_roles(
    candidates: dict[BomKind, list[str]],
) -> None:
    """Raise the appropriate role error when candidate counts are not exactly one.

    Args:
        candidates: Map from BomKind to list of matching sheet titles.

    Raises:
        SourceValidationError: ``BOM_SHEET_ROLE_MISSING`` or
            ``BOM_SHEET_ROLE_DUPLICATE`` as appropriate.
    """
    for kind, names in candidates.items():
        if len(names) == 0:
            raise _workbook_error(
                "BOM_SHEET_ROLE_MISSING",
                f"No worksheet found matching the {kind} header signature",
                missing_role=str(kind),
            )
        if len(names) > 1:
            raise _workbook_error(
                "BOM_SHEET_ROLE_DUPLICATE",
                f"More than one worksheet matches the {kind} header signature",
                duplicate_role=str(kind),
                sheet_count=len(names),
            )


# ---------------------------------------------------------------------------
# Public: parse_bom_workbook
# ---------------------------------------------------------------------------


def parse_bom_workbook(
    content: bytes,
    artifact_id: str,
    maximum_rows: int,
    maximum_cells: int,
) -> ParsedBomWorkbook:
    """Parse three required BOM evidence layers from bounded in-memory OOXML bytes.

    Processing order:
    1. Validate the canonical artifact UUID and positive limits.
    2. Confirm the bytes form an OOXML ZIP.
    3. Open with ``openpyxl.load_workbook(read_only=True, data_only=False)``.
    4. Detect exactly one worksheet per role via normalized header signatures.
    5. For each role in fixed order (enriched, raw, MBOM):
       a. Read the header row and validate for duplicate required headers.
       b. Iterate data rows, enforcing row/cell limits during streaming.
       c. Build one ``BomRow`` per nonblank row.
    6. Reconstruct enriched and raw hierarchies on their complete row sequences.
    7. Return a ``ParsedBomWorkbook`` with rows in role/sheet order.

    ``row_count`` equals the number of retained nonblank rows.  ``cell_count``
    equals the number of header and data cells visited across all role sheets
    (using the header row width per role).

    Args:
        content: Raw workbook bytes exactly as received from the upload.
        artifact_id: Canonical UUID string of the parent artifact.
        maximum_rows: Hard upper bound on total visited data rows (must be > 0).
        maximum_cells: Hard upper bound on total visited cells (must be > 0).

    Returns:
        A frozen ``ParsedBomWorkbook`` with all rows, role names, and counters.

    Raises:
        ValueError: If ``artifact_id`` is not a canonical UUID string, or if
            either limit is not a positive integer.
        SourceValidationError: HTTP 400 for any workbook-level contract violation
            (corrupt file, missing/duplicate role, duplicate required header,
            row/cell limit exceeded).
    """
    canonical_uuid(artifact_id)
    if maximum_rows <= 0 or maximum_cells <= 0:
        raise ValueError("workbook row and cell limits must be positive")

    stream = BytesIO(content)
    if not zipfile.is_zipfile(stream):
        raise _workbook_error("BOM_WORKBOOK_CORRUPT", "BOM workbook is not valid OOXML")
    stream.seek(0)

    try:
        workbook = openpyxl.load_workbook(stream, read_only=True, data_only=False)
    except (
        BadZipFile,
        EOFError,
        InvalidFileException,
        KeyError,
        OSError,
        ParseError,
        ValueError,
    ) as exc:
        raise _workbook_error(
            "BOM_WORKBOOK_CORRUPT", "BOM workbook cannot be opened"
        ) from exc

    try:
        try:
            roles = detect_sheet_roles(workbook)
            return _parse_detected_roles(
                workbook, roles, artifact_id, maximum_rows, maximum_cells
            )
        except SourceValidationError:
            raise
        except (BadZipFile, EOFError, OSError, ParseError) as exc:
            raise _workbook_error(
                "BOM_WORKBOOK_CORRUPT", "BOM workbook cannot be read"
            ) from exc
    finally:
        workbook.close()


# ---------------------------------------------------------------------------
# Internal: role parsing
# ---------------------------------------------------------------------------


def _parse_detected_roles(
    workbook: Workbook,
    roles: dict[BomKind, str],
    artifact_id: str,
    maximum_rows: int,
    maximum_cells: int,
) -> ParsedBomWorkbook:
    """Iterate each detected role sheet in fixed order and build BomRow objects.

    Args:
        workbook: Open openpyxl workbook (caller owns lifecycle).
        roles: Map from BomKind to sheet title, as returned by detect_sheet_roles.
        artifact_id: Canonical UUID of the parent artifact.
        maximum_rows: Row limit enforced during streaming.
        maximum_cells: Cell limit enforced during streaming.

    Returns:
        A frozen ``ParsedBomWorkbook``.
    """
    contracts: dict[BomKind, WorkbookSheetContract] = {
        c.bom_kind: c for c in WORKBOOK_SHEET_CONTRACTS
    }

    all_rows: list[BomRow] = []
    total_cells = 0
    total_rows_visited = 0

    for kind in _ROLE_ORDER:
        sheet_name = roles[kind]
        contract = contracts[kind]
        worksheet = workbook[sheet_name]

        # ------------------------------------------------------------------
        # Read header row and determine column width.
        # ------------------------------------------------------------------
        header_row = next(
            worksheet.iter_rows(min_row=1, max_row=1, values_only=True), ()
        )
        headers: list[object] = list(header_row)
        width = len(headers)

        # Count header cells towards the cell limit.
        total_cells += width
        if total_cells > maximum_cells:
            raise _workbook_error(
                "BOM_CELL_LIMIT_EXCEEDED",
                "Workbook exceeds the configured maximum cell count",
                maximum_cells=maximum_cells,
                observed_cells=total_cells,
            )

        # Validate no duplicate required headers.
        _check_header_duplicates(contract, headers, sheet_name)

        # Build field-key → 0-based column index map.
        field_col = _build_field_col(contract, headers)

        # ------------------------------------------------------------------
        # Iterate data rows.
        # ------------------------------------------------------------------
        role_rows_pre_hierarchy: list[BomRow] = []

        for source_row_number, row_values in enumerate(
            worksheet.iter_rows(min_row=2, values_only=True), start=2
        ):
            # Truncate/pad to header width so every row has exactly `width` slots.
            values: list[object] = list(row_values[:width])
            values.extend([None] * max(0, width - len(values)))

            # Count cells for this data row.
            total_cells += width
            if total_cells > maximum_cells:
                raise _workbook_error(
                    "BOM_CELL_LIMIT_EXCEEDED",
                    "Workbook exceeds the configured maximum cell count",
                    maximum_cells=maximum_cells,
                    observed_cells=total_cells,
                )

            # Count this row towards the row limit (before blank check).
            total_rows_visited += 1
            if total_rows_visited > maximum_rows:
                raise _workbook_error(
                    "BOM_ROW_LIMIT_EXCEEDED",
                    "Workbook exceeds the configured maximum data row count",
                    maximum_rows=maximum_rows,
                    observed_rows=total_rows_visited,
                )

            # Skip all-blank rows (do not create a BomRow for them).
            if all(v is None for v in values):
                continue

            # Build row_json entries from all source columns.
            row_json_entries, cell_qualifiers = _build_row_json(headers, values)

            # Delegate to role-specific row builder.
            bom_row = _build_role_row(
                kind=kind,
                contract=contract,
                field_col=field_col,
                headers=headers,
                values=values,
                row_json_entries=row_json_entries,
                cell_qualifiers=cell_qualifiers,
                artifact_id=artifact_id,
                sheet_name=sheet_name,
                source_row_number=source_row_number,
            )
            role_rows_pre_hierarchy.append(bom_row)

        # ------------------------------------------------------------------
        # Reconstruct hierarchy for the complete role row sequence.
        # ------------------------------------------------------------------
        if kind is BomKind.ENRICHED_EBOM:
            role_rows: list[BomRow] = list(
                reconstruct_enriched_hierarchy(role_rows_pre_hierarchy)
            )
        elif kind is BomKind.RAW_EBOM_130:
            role_rows = list(reconstruct_raw_hierarchy(role_rows_pre_hierarchy))
        else:
            role_rows = role_rows_pre_hierarchy

        all_rows.extend(role_rows)

    return ParsedBomWorkbook(
        rows=tuple(all_rows),
        sheet_roles=roles,
        row_count=len(all_rows),
        cell_count=total_cells,
        quality_qualifiers=(),
    )


# ---------------------------------------------------------------------------
# Internal: header helpers
# ---------------------------------------------------------------------------


def _check_header_duplicates(
    contract: WorkbookSheetContract,
    headers: list[object],
    sheet_name: str,
) -> None:
    """Raise BOM_REQUIRED_HEADER_DUPLICATE when any required header appears more than once.

    Only required/mapped headers (those in ``contract.required_headers``) are
    checked.  Non-required headers (e.g. the two MS columns) may appear any
    number of times without error.

    Args:
        contract: The matched role contract.
        headers: Raw header values from the worksheet's first row.
        sheet_name: Sheet title for error diagnostics.

    Raises:
        SourceValidationError: When a required header appears more than once.
    """
    required_set = set(contract.required_headers)
    seen: dict[str, int] = {}
    for raw in headers:
        normalized = normalize_header(raw)
        if normalized in required_set:
            seen[normalized] = seen.get(normalized, 0) + 1
            if seen[normalized] > 1:
                raise _workbook_error(
                    "BOM_REQUIRED_HEADER_DUPLICATE",
                    "A required header appears more than once in a matched worksheet",
                    role=str(contract.bom_kind),
                    safe_sheet_name=sheet_name,
                    duplicate_header=normalized,
                )


def _build_field_col(
    contract: WorkbookSheetContract,
    headers: list[object],
) -> dict[str, int]:
    """Build a mapping from field key to 0-based column index.

    Uses the first occurrence of each normalized header string.  Duplicate
    unrelated headers (like two MS columns) are not in field_headers and
    therefore do not create conflicts here.

    Args:
        contract: The matched role contract with field_headers mapping.
        headers: Raw header values from the worksheet's first row.

    Returns:
        A dict of field_key → 0-based column index for all resolvable fields.
    """
    normalized_headers = [normalize_header(h) for h in headers]
    result: dict[str, int] = {}
    for field_key, source_header in contract.field_headers.items():
        try:
            result[field_key] = normalized_headers.index(source_header)
        except ValueError:
            # Header not found — the column is absent; field will be None.
            pass
    return result


# ---------------------------------------------------------------------------
# Internal: ROW_JSON builder
# ---------------------------------------------------------------------------


def _build_row_json(
    headers: list[object],
    values: list[object],
) -> tuple[tuple[dict[str, JsonValue], ...], tuple[str, ...]]:
    """Preserve every source column position, including duplicate and blank headers.

    Each entry in the returned tuple captures: ``column_index`` (1-based),
    ``raw_header`` (the literal cell text, or ``""`` for None), ``normalized_header``
    (whitespace-normalized), and ``value`` (JSON-safe cell value).

    Stringification qualifiers from ``json_safe_cell`` are accumulated and
    deduplicated across all columns.

    Args:
        headers: Raw header values (length determines schema width).
        values: Raw cell values for this row, pre-padded to len(headers).

    Returns:
        A ``(entries_tuple, qualifiers_tuple)`` pair where ``qualifiers_tuple``
        contains unique ``SOURCE_CELL_TYPE_STRINGIFIED`` codes.
    """
    entries: list[dict[str, JsonValue]] = []
    qualifiers: list[str] = []
    width = max(len(headers), len(values))
    for offset in range(width):
        raw_header_value = headers[offset] if offset < len(headers) else None
        raw_header = "" if raw_header_value is None else str(raw_header_value)
        value = values[offset] if offset < len(values) else None
        json_value, cell_qualifiers = json_safe_cell(value)
        entries.append({
            "column_index": offset + 1,
            "raw_header": raw_header,
            "normalized_header": normalize_header(raw_header_value),
            "value": json_value,
        })
        qualifiers.extend(cell_qualifiers)
    return tuple(entries), ordered_unique(qualifiers)


# ---------------------------------------------------------------------------
# Internal: field-value extractor
# ---------------------------------------------------------------------------


def _field_value(
    field_key: str,
    field_col: dict[str, int],
    values: list[object],
) -> object:
    """Return the raw cell value for a named BomRow field, or None if absent.

    Args:
        field_key: Logical field key (e.g. ``"dotted_level"``).
        field_col: Map from field key to 0-based column index.
        values: Row cell values pre-padded to header width.

    Returns:
        The raw cell value, or ``None`` when the column was not found or is empty.
    """
    idx = field_col.get(field_key)
    if idx is None or idx >= len(values):
        return None
    return values[idx]


# ---------------------------------------------------------------------------
# Internal: role-specific BomRow builders
# ---------------------------------------------------------------------------


def _build_role_row(
    kind: BomKind,
    contract: WorkbookSheetContract,
    field_col: dict[str, int],
    headers: list[object],
    values: list[object],
    row_json_entries: tuple[dict[str, JsonValue], ...],
    cell_qualifiers: tuple[str, ...],
    artifact_id: str,
    sheet_name: str,
    source_row_number: int,
) -> BomRow:
    """Dispatch to the appropriate role-specific builder.

    Args:
        kind: The BOM role kind for this row.
        contract: The matched role contract (unused directly; passed for context).
        field_col: Map from field key to 0-based column index.
        headers: Raw header values.
        values: Raw cell values for this row.
        row_json_entries: Pre-built ROW_JSON tuple.
        cell_qualifiers: Qualifiers from json_safe_cell across all columns.
        artifact_id: Canonical UUID of the parent artifact.
        sheet_name: Worksheet title this row was read from.
        source_row_number: 1-based row index in the source worksheet.

    Returns:
        A frozen ``BomRow`` instance.
    """
    fv = lambda key: _field_value(key, field_col, values)  # noqa: E731

    if kind is BomKind.ENRICHED_EBOM:
        return _build_enriched_row(
            fv, row_json_entries, cell_qualifiers,
            artifact_id, sheet_name, source_row_number,
        )
    if kind is BomKind.RAW_EBOM_130:
        return _build_raw_row(
            fv, row_json_entries, cell_qualifiers,
            artifact_id, sheet_name, source_row_number,
        )
    return _build_mbom_row(
        fv, row_json_entries, cell_qualifiers,
        artifact_id, sheet_name, source_row_number,
    )


def _build_enriched_row(
    fv,
    row_json_entries: tuple[dict[str, JsonValue], ...],
    cell_qualifiers: tuple[str, ...],
    artifact_id: str,
    sheet_name: str,
    source_row_number: int,
) -> BomRow:
    """Build one ENRICHED_EBOM BomRow from extracted field values.

    Mappings:
    - ``dotted_level`` ← Level (integer stored as string via source_text).
    - ``position_value`` ← Post number.
    - ``component_number = material_number`` ← Component.
    - ``item_value`` ← None (no Item column in this layout).
    - ``parent_material_number`` ← Parent.
    - ``main/sub/vc/level_4/5/6`` ← named slot columns.
    - ``hierarchy_path`` / ``parent_bom_row_id`` are set by reconstruct_enriched_hierarchy.
    - ``row_classification`` ← NOT_APPLICABLE.

    Args:
        fv: Callable ``(field_key) -> raw_cell_value``.
        row_json_entries: Pre-built ROW_JSON tuple.
        cell_qualifiers: Qualifiers from json_safe_cell.
        artifact_id: Canonical artifact UUID.
        sheet_name: Source worksheet title.
        source_row_number: 1-based source row index.

    Returns:
        A frozen ``BomRow``.
    """
    qualifiers: list[str] = list(cell_qualifiers)

    # All bounded identifier fields go through _fit_id so that values exceeding
    # their HANA NVARCHAR column width raise SOURCE_VALUE_TOO_LONG with safe
    # coordinates and no raw content in the error details.
    fi = lambda name: _fit_id(name, fv(name), sheet_name, source_row_number)  # noqa: E731

    component_number = fi("component_number")

    # Effectivity — enriched uses EXCEL_TYPED_OR_ISO format.
    eff_from = normalize_effectivity_value(
        fv("valid_from"), EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    eff_to = normalize_effectivity_value(
        fv("valid_to"), EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_TO
    )
    if eff_from.parse_warning:
        qualifiers.append(eff_from.parse_warning)
    if eff_to.parse_warning:
        qualifiers.append(eff_to.parse_warning)

    # Guard effectivity raw-text strings against VALID_FROM_RAW / VALID_TO_RAW
    # NVARCHAR(100).  Date isoformat strings are always short; the guard protects
    # against arbitrarily-long non-date cell text passed through source_text().
    valid_from_raw = _guard_eff_raw(eff_from.raw_text, "valid_from_raw", sheet_name, source_row_number)
    valid_to_raw = _guard_eff_raw(eff_to.raw_text, "valid_to_raw", sheet_name, source_row_number)

    # Quantity.
    qty = normalize_quantity(fv("quantity"))
    qualifiers.extend(qty.qualifiers)

    return BomRow(
        bom_row_id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        sheet_name=sheet_name,
        bom_kind=BomKind.ENRICHED_EBOM,
        source_row_number=source_row_number,
        parent_bom_row_id=None,
        dotted_level=fi("dotted_level"),
        hierarchy_path=None,
        main_value=fi("main_value"),
        sub_value=fi("sub_value"),
        vc_value=fi("vc_value"),
        level_4_value=fi("level_4_value"),
        level_5_value=fi("level_5_value"),
        level_6_value=fi("level_6_value"),
        item_value=None,
        position_value=fi("position_value"),
        material_number=component_number,
        parent_material_number=fi("parent_material_number"),
        component_number=component_number,
        change_number=fi("change_number"),
        change_number_to=fi("change_number_to"),
        description_text=source_text(fv("description_text")),  # NCLOB — no length bound
        unit=fi("unit"),
        plant=None,
        alternative=fi("alternative"),
        valid_from_raw=valid_from_raw,
        valid_to_raw=valid_to_raw,
        quantity=qty.value,
        valid_from_date=eff_from.parsed_date,
        valid_to_date=eff_to.parsed_date,
        row_classification=BomRowClassification.NOT_APPLICABLE,
        is_documentation=False,
        is_migration_date=eff_from.is_migration_placeholder,
        row_json=row_json_entries,
        quality_qualifiers=ordered_unique(qualifiers),
        quality_details={},
    )


def _build_raw_row(
    fv,
    row_json_entries: tuple[dict[str, JsonValue], ...],
    cell_qualifiers: tuple[str, ...],
    artifact_id: str,
    sheet_name: str,
    source_row_number: int,
) -> BomRow:
    """Build one RAW_EBOM_130 BomRow from extracted field values.

    Mappings:
    - ``dotted_level`` ← Level no. (dotted string like ``.1``, ``..2``).
    - ``item_value = position_value`` ← Item.
    - ``component_number = material_number`` ← Component number.
    - ``type_value`` (Typ) is used for classification only, not stored as a field.
    - ``row_classification`` ← from classify_raw_row(item, typ, component).
    - ``is_documentation`` ← True when classification is CONFIRMED_DOCUMENT.
    - ``TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED`` ← added for MATERIAL_LIKE_OBSERVED rows.
    - ``hierarchy_path`` / ``parent_bom_row_id`` set by reconstruct_raw_hierarchy.
    - Effectivity uses RAW_DD_MM_YYYY format (European day-first text strings).

    Args:
        fv: Callable ``(field_key) -> raw_cell_value``.
        row_json_entries: Pre-built ROW_JSON tuple.
        cell_qualifiers: Qualifiers from json_safe_cell.
        artifact_id: Canonical artifact UUID.
        sheet_name: Source worksheet title.
        source_row_number: 1-based source row index.

    Returns:
        A frozen ``BomRow``.
    """
    qualifiers: list[str] = list(cell_qualifiers)

    fi = lambda name: _fit_id(name, fv(name), sheet_name, source_row_number)  # noqa: E731

    # classification needs the raw values before fit; classify_raw_row internally
    # calls source_text, so passing the same raw values is correct.
    item_value = fi("item_value")
    component_number = fi("component_number")
    type_value = fv("type_value")

    classification = classify_raw_row(item_value, type_value, component_number)
    is_doc = classification is BomRowClassification.CONFIRMED_DOCUMENT

    if classification is BomRowClassification.MATERIAL_LIKE_OBSERVED:
        qualifiers.append("TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED")

    # Effectivity — raw sheet uses European text format.
    eff_from = normalize_effectivity_value(
        fv("valid_from"), EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_FROM
    )
    eff_to = normalize_effectivity_value(
        fv("valid_to"), EffectivityFormat.RAW_DD_MM_YYYY, EffectivityField.VALID_TO
    )
    if eff_from.parse_warning:
        qualifiers.append(eff_from.parse_warning)
    if eff_to.parse_warning:
        qualifiers.append(eff_to.parse_warning)

    valid_from_raw = _guard_eff_raw(eff_from.raw_text, "valid_from_raw", sheet_name, source_row_number)
    valid_to_raw = _guard_eff_raw(eff_to.raw_text, "valid_to_raw", sheet_name, source_row_number)

    # Quantity.
    qty = normalize_quantity(fv("quantity"))
    qualifiers.extend(qty.qualifiers)

    return BomRow(
        bom_row_id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        sheet_name=sheet_name,
        bom_kind=BomKind.RAW_EBOM_130,
        source_row_number=source_row_number,
        parent_bom_row_id=None,
        dotted_level=fi("dotted_level"),
        hierarchy_path=None,
        main_value=None,
        sub_value=None,
        vc_value=None,
        level_4_value=None,
        level_5_value=None,
        level_6_value=None,
        item_value=item_value,
        position_value=item_value,
        material_number=component_number,
        parent_material_number=None,
        component_number=component_number,
        change_number=fi("change_number"),
        change_number_to=fi("change_number_to"),
        description_text=source_text(fv("description_text")),  # NCLOB — no length bound
        unit=fi("unit"),
        plant=None,
        alternative=fi("alternative"),
        valid_from_raw=valid_from_raw,
        valid_to_raw=valid_to_raw,
        quantity=qty.value,
        valid_from_date=eff_from.parsed_date,
        valid_to_date=eff_to.parsed_date,
        row_classification=classification,
        is_documentation=is_doc,
        is_migration_date=eff_from.is_migration_placeholder,
        row_json=row_json_entries,
        quality_qualifiers=ordered_unique(qualifiers),
        quality_details={},
    )


def _build_mbom_row(
    fv,
    row_json_entries: tuple[dict[str, JsonValue], ...],
    cell_qualifiers: tuple[str, ...],
    artifact_id: str,
    sheet_name: str,
    source_row_number: int,
) -> BomRow:
    """Build one MBOM_130 BomRow from extracted field values.

    Mappings:
    - ``item_value = parent_material_number`` ← Item (parent context, not position).
    - ``component_number = material_number`` ← Component.
    - ``position_value`` ← None (MBOM has no position column).
    - ``hierarchy_path`` ← ``Item/Component`` when both exist.
    - ``row_classification`` ← NOT_APPLICABLE.
    - Sequence and Employment probability remain only in ROW_JSON.
    - Effectivity uses EXCEL_TYPED_OR_ISO format.
    - Blank plant/alternative/effectivity values remain unavailable, not negative facts.

    Args:
        fv: Callable ``(field_key) -> raw_cell_value``.
        row_json_entries: Pre-built ROW_JSON tuple.
        cell_qualifiers: Qualifiers from json_safe_cell.
        artifact_id: Canonical artifact UUID.
        sheet_name: Source worksheet title.
        source_row_number: 1-based source row index.

    Returns:
        A frozen ``BomRow``.
    """
    qualifiers: list[str] = list(cell_qualifiers)

    fi = lambda name: _fit_id(name, fv(name), sheet_name, source_row_number)  # noqa: E731

    # MBOM Item column maps to both item_value and parent_material_number.
    item_value = _fit_id("item_value", fv("parent_material_number"), sheet_name, source_row_number)
    component_number = fi("component_number")

    # Hierarchy path: Item/Component when both present.
    path_parts = [p for p in [item_value, component_number] if p]
    hierarchy_path: str | None = "/".join(path_parts) if path_parts else None

    # Effectivity — MBOM uses EXCEL_TYPED_OR_ISO format.
    eff_from = normalize_effectivity_value(
        fv("valid_from"), EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_FROM
    )
    eff_to = normalize_effectivity_value(
        fv("valid_to"), EffectivityFormat.EXCEL_TYPED_OR_ISO, EffectivityField.VALID_TO
    )
    if eff_from.parse_warning:
        qualifiers.append(eff_from.parse_warning)
    if eff_to.parse_warning:
        qualifiers.append(eff_to.parse_warning)

    valid_from_raw = _guard_eff_raw(eff_from.raw_text, "valid_from_raw", sheet_name, source_row_number)
    valid_to_raw = _guard_eff_raw(eff_to.raw_text, "valid_to_raw", sheet_name, source_row_number)

    # Quantity.
    qty = normalize_quantity(fv("quantity"))
    qualifiers.extend(qty.qualifiers)

    return BomRow(
        bom_row_id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        sheet_name=sheet_name,
        bom_kind=BomKind.MBOM_130,
        source_row_number=source_row_number,
        parent_bom_row_id=None,
        dotted_level=None,
        hierarchy_path=hierarchy_path,
        main_value=None,
        sub_value=None,
        vc_value=None,
        level_4_value=None,
        level_5_value=None,
        level_6_value=None,
        item_value=item_value,
        position_value=None,
        material_number=component_number,
        parent_material_number=item_value,
        component_number=component_number,
        change_number=fi("change_number"),
        change_number_to=fi("change_number_to"),
        description_text=source_text(fv("description_text")),  # NCLOB — no length bound
        unit=fi("unit"),
        plant=None,
        alternative=fi("alternative"),
        valid_from_raw=valid_from_raw,
        valid_to_raw=valid_to_raw,
        quantity=qty.value,
        valid_from_date=eff_from.parsed_date,
        valid_to_date=eff_to.parsed_date,
        row_classification=BomRowClassification.NOT_APPLICABLE,
        is_documentation=False,
        is_migration_date=eff_from.is_migration_placeholder,
        row_json=row_json_entries,
        quality_qualifiers=ordered_unique(qualifiers),
        quality_details={},
    )
