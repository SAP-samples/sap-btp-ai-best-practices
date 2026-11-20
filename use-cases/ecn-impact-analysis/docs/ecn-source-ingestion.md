# ECN and BOM Source Ingestion

## Phase Scope and Byte-only Runtime Boundary

Phase 01 (Data Foundation) defines the intake, normalization, and persistence
contracts for source artifacts. All parsing operates exclusively on supplied bytes;
there are no network calls or filesystem reads beyond the upload stream. Phase 02's
ingestion service composes those contracts with deterministic matching, rules,
evidence, and atomic publication. LLM generation and HTTP routes remain outside
this service boundary.

## Deterministic preparation and publication

1. Check `REQUEST_ID`; a committed replay returns without taking either gate.
2. Under the capacity-one preparation gate, parse the required ECN JSON and resolve an uploaded BOM workbook versus the latest valid BOM snapshot.
3. Parse a replacement workbook or stream normalized rows from HANA in 5,000-row batches; build exact indexes, conservative candidates, relevant-rule non-evaluations, evidence, counts, and queued job aggregates. `analysis_as_of_date` is fixed to `None`, rule facts are empty, and the activation policy is absent in MVP v1.
4. Release the preparation gate, then recheck `REQUEST_ID` under the shared history mutation guard.
5. Publish the complete aggregate in one HANA transaction and release the mutation guard.

A replacement becomes current only after this transaction commits. Existing ECNs remain pinned to their original BOM artifact. Reuse references the existing artifact and rows and does not copy them into the new upload. The same request ID replays the committed upload; a deliberate new ID creates new immutable ECN occurrences even when file hashes or business ECN numbers repeat. Repeated and missing business ECN numbers therefore remain distinct through their parser-assigned row IDs and source ordinals.

The first successful upload must include a BOM workbook. A later upload may omit it and reuse the latest BOM artifact from a successfully published upload. Immediately before publication, the service revalidates a reused artifact while holding the mutation guard. This check uses an ID-only `SELECT 1`; it never loads the workbook BLOB or normalized content into the guarded critical section. If a concurrent idle reset removed it during preparation, `BOM_SNAPSHOT_INVALIDATED` returns HTTP 409 before publication; the operator must retry with a BOM workbook. Missing rule configuration creates a Configuration Scope evidence gap. Invalid JSON, an invalid workbook, a missing first BOM, an invalidated reused snapshot, or another deterministic preparation failure publishes nothing.

Only a created repository result contains `queued_job_ids`. That tuple is formed directly from the committed publication's ECN jobs in source-ordinal order and is the authoritative post-commit HTTP/runtime scheduling handoff. The service does not re-query jobs by upload ID. Replays and request-ID race losses return an empty tuple and schedule nothing.

### Service records and boundaries

| Record | Input/output contract |
|---|---|
| `UploadFileBytes` | Input file name, validated media type, and bounded bytes |
| `UploadBytesRequest` | Request UUID, required ECN file, optional replacement BOM file, and optional hashed actor ID |
| `UploadPublication` | Fully prepared source artifacts, optional new BOM rows, all ECN evidence/counts, and queued-job records for one transaction |
| `IngestionResult` | Authoritative committed/replayed `UploadRecord`, `created` flag, and created-only job IDs |

`IngestionService.prepare_upload` performs only reads and pure preparation; it does not write to HANA. `IngestionService.ingest_upload` owns replay checks and calls `UploadRepository.publish_upload` once under the short mutation guard. `IngestionPreparationGate` protects only parsing, stored-row loading, indexing, and deterministic derivation. It never blocks list/detail reads or provider work. `HistoryMutationGuard` protects only final lifecycle mutations and is shared with reset and later retry/claim services.

The upload boundary enforces `max_json_upload_bytes` and `max_workbook_upload_bytes`. Workbook parsing inherits `max_workbook_rows` and `max_workbook_cells`; their defaults are 100,000 rows and 3,000,000 visited cells. The normalized ECN artifact stores the exact canonical content returned by `parse_ecn_json`. A BOM artifact stores only a compact manifest; normalized tabular rows live in `ECN_IMPACT_BOM_ROW` in HANA.

Focused verification:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_ingestion_service.py \
  tests/unit/test_artifact_repository.py \
  tests/unit/test_upload_repository.py
PYTHONPATH=api python -m pytest -q tests/unit
python -m compileall -q api/app/ecn/services
```

## Accepted Extensions, Media Types, and Size Limits

| Artifact kind               | Extension | Accepted media types                                                              |
|-----------------------------|-----------|-----------------------------------------------------------------------------------|
| `ECN_JSON`                  | `.json`   | `application/json`, `text/json`, `application/octet-stream`, empty               |
| `BOM_WORKBOOK`              | `.xlsx`   | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`, `application/octet-stream`, empty |
| `CONDITIONAL_RULE_WORKBOOK` | `.xlsx`   | same as BOM_WORKBOOK                                                              |

Size limits are enforced at the upload boundary (HTTP 413 via `UploadTooLargeError`).
Workbook row and cell limits are enforced during OOXML streaming (HTTP 400 via
`SourceValidationError`).

## ECN JSON Root and Field Mapping

The ECN JSON file must be a UTF-8 encoded top-level array.  Each element is one
ECN object.  Fields are extracted via the path below; missing or non-string fields
are skipped without raising.

| Source JSON key               | `NormalizedEcn` field        | Notes                                        |
|-------------------------------|------------------------------|----------------------------------------------|
| `Number`                      | `ecn_number`                 | NVARCHAR(80); rejection if longer            |
| `Title`                       | `title`                      | Soft-truncated to 500 chars                  |
| `DescriptionSummary`          | `description_summary`        | Soft-truncated to 2000 chars                 |
| `DescriptionText`             | `description_text`           | Stored as NCLOB; WTException strings excluded |
| `CNAffectLinks[*].AffectedObjects[*]` | `affected_objects`  | See Source Object section                    |
| `ImplementationPlan[*].ResultingLinks[*].ResultingObjects[*]` | `resulting_objects` | See Source Object section |

## Affected and Resulting Source Paths

Affected objects are traversed at `$.CNAffectLinks[i].AffectedObjects[j]`.
Resulting objects are traversed at `$.ImplementationPlan[p].ResultingLinks[l].ResultingObjects[j]`.
Both paths are stored as exact JSON-pointer strings in `NormalizedSourceObject.source_path`.

## Technical WTPart and Document Identity Rules

- `OR:wt.part.WTPart:<opaque>` and `VR:wt.part.WTPart:<opaque>` are both WTPart identifiers.
- If `ID` and `VersionID` are both present and their technical classes disagree, the object is
  NOT classified as a WTPart and `TECHNICAL_IDENTITY_CONTRADICTION` is added to qualifiers.
- CAD/EPM, WTDocument, drawing, and specification objects always have `is_technical_wtpart=False`.
- Number shape never changes the WTPart classification.

## Workbook Role Signatures

Roles are detected purely from normalized (whitespace-collapsed, no case-fold) header signatures.
Sheet names and sheet order are irrelevant.  Extra unrecognized sheets are allowed.

### ENRICHED_EBOM (18 required headers)

Level, Post number, Component, Item designation, Component quantity, Quantity unit,
N° of modification, Start of validity, Up to modif. Number, Expiration date,
Parent, Main, Sub, VC, Level 4, Level 5, Level 6, Alternative

### RAW_EBOM_130 (12 required headers)

Level no., Item, Component number, Object description, Quantity, Un,
Valid from, Valid to, Change No., Chg No. To, Typ, AltItm

Note: the raw sheet contains two `MS` columns that are not required headers.
They are retained positionally in ROW_JSON.

### MBOM_130 (11 required headers)

Item, Component, Component name, Quantity, N° of modification, Start of validity,
at the number of modif., Expiration date, alternative postal group, Sequence,
Employment probability

## Enriched EBOM Field Mapping and Named Hierarchy

| Source header         | `BomRow` field(s)                           |
|-----------------------|----------------------------------------------|
| Level                 | `dotted_level` (stored as string, e.g. "1") |
| Post number           | `position_value`                             |
| Component             | `component_number`, `material_number`         |
| Item designation      | `description_text`                           |
| Component quantity    | `quantity`                                   |
| Quantity unit         | `unit`                                       |
| N° of modification    | `change_number`                              |
| Start of validity     | `valid_from_date`, `valid_from_raw`           |
| Up to modif. Number   | `change_number_to`                           |
| Expiration date       | `valid_to_date`, `valid_to_raw`               |
| Parent                | `parent_material_number`                     |
| Main                  | `main_value`                                 |
| Sub                   | `sub_value`                                  |
| VC                    | `vc_value`                                   |
| Level 4               | `level_4_value`                              |
| Level 5               | `level_5_value`                              |
| Level 6               | `level_6_value`                              |
| Alternative           | `alternative`                                |
| (none)                | `item_value` = None                          |

`row_classification` is always `NOT_APPLICABLE`.  Enriched effectivity uses
`EXCEL_TYPED_OR_ISO` format (openpyxl returns date/datetime objects or ISO strings).

Named path slots are: Main (slot 1), Sub (slot 2), VC (slot 3), Level 4 (slot 4),
Level 5 (slot 5), Level 6 (slot 6).  For a row at level L, slot L should equal
`component_number`, all earlier slots should be nonblank, and all later slots should
be blank.  Violations add `ENRICHED_PATH_CONTRADICTION` to qualifiers.  **Raw dotted
depth is never mapped to Main/Sub/VC semantics.**

## Raw 130-percent EBOM Field Mapping and Dotted Hierarchy

| Source header     | `BomRow` field(s)                           |
|-------------------|----------------------------------------------|
| Level no.         | `dotted_level` (dotted string like ".1")    |
| Item              | `item_value`, `position_value`               |
| Component number  | `component_number`, `material_number`         |
| Object description| `description_text`                           |
| Quantity          | `quantity`                                   |
| Un                | `unit`                                       |
| Valid from        | `valid_from_date`, `valid_from_raw`           |
| Valid to          | `valid_to_date`, `valid_to_raw`               |
| Change No.        | `change_number`                              |
| Chg No. To        | `change_number_to`                           |
| Typ               | used for classification only (not a field)  |
| AltItm            | `alternative`                                |

Raw effectivity uses `RAW_DD_MM_YYYY` format (`%d.%m.%Y` European locale).
`02.01.1900` parses as 2 January 1900, not 1 February.

Dotted hierarchy depth is the number of leading dots in `Level no.`
(e.g., `..2` is depth 2).  The numeric label mismatch adds
`DOTTED_LEVEL_LABEL_MISMATCH` but does not change depth.

Positions `150` and `160` are observed `Item` column values (not special codes);
they fall through the classification heuristic like any other non-D item.

**The four-way classification heuristic (applied in strict precedence order):**

1. Trimmed `Item` starts with uppercase "D" → `CONFIRMED_DOCUMENT`
2. `Component number` matches `^.+?\s+ZV[DSG]\s+\S+\s+\S+$` → `DOCUMENT_REFERENCE_CANDIDATE`
3. Trimmed `Typ == "ND"` → `MATERIAL_LIKE_OBSERVED`
4. Otherwise → `UNRESOLVED`

## Documentation Classification and Nearest Material Owner

A D-prefix on the raw `Item` column value is the sole indicator of a confirmed
documentation row.  The D-prefix rule applies only in the raw EBOM `Item` column
and not in any other field or role.

After parent links are assigned, each `CONFIRMED_DOCUMENT` row receives
`document_owner_bom_row_id` and `document_owner_material_number` in its
`quality_details` by traversing the parent chain to the nearest
`MATERIAL_LIKE_OBSERVED` ancestor.  If no such ancestor exists,
`DOCUMENT_OWNER_UNRESOLVED` is added to qualifiers.

Key invariants:
- Documentation rows are **never** filtered from the output.
- `parent_material_number` is the direct structural parent's component, even when
  that parent is also documentation.
- Document owner resolution skips over intermediate documentation rows; it does
  **not** select merely the nearest non-D row.
- Every `MATERIAL_LIKE_OBSERVED` row carries `TYPE_ND_OBSERVED_NOT_BUSINESS_CONFIRMED`.

## MBOM Field Mapping

| Source header                | `BomRow` field(s)                               |
|------------------------------|-------------------------------------------------|
| Item                         | `item_value`, `parent_material_number`           |
| Component                    | `component_number`, `material_number`            |
| Component name               | `description_text`                              |
| Quantity                     | `quantity`                                      |
| N° of modification           | `change_number`                                 |
| Start of validity            | `valid_from_date`, `valid_from_raw`              |
| at the number of modif.      | `change_number_to`                              |
| Expiration date              | `valid_to_date`, `valid_to_raw`                  |
| alternative postal group     | `alternative`                                   |
| Sequence                     | ROW_JSON only (not a `BomRow` field)            |
| Employment probability       | ROW_JSON only (not a `BomRow` field)            |

`position_value` is always `None` for MBOM rows.
`hierarchy_path` is `Item/Component` when both are nonblank, otherwise the
available part(s) or None.  `row_classification` is always `NOT_APPLICABLE`.
MBOM effectivity uses `EXCEL_TYPED_OR_ISO` format.

Blank plant/alternative/effectivity values remain **unavailable** — they are not
negative facts and no default is substituted.

## Positional ROW_JSON and QUALITY_JSON

Every source row produces a `ROW_JSON` entry for every column in that sheet's
header row (including blank headers and blank cells):

```json
[
  {
    "column_index": 1,
    "raw_header": "Item",
    "normalized_header": "Item",
    "value": "00150"
  }
]
```

Duplicate headers (e.g., two `MS` columns at positions 11 and 12) remain as
**distinct positional entries** — neither overwrites the other.

`QUALITY_JSON` is the canonical `{"details": {...}, "qualifiers": [...]}` object
stored per row, capturing all parse-time qualifiers and structured diagnostics.

## Quantity and HANA Column-fit Policy

Quantities are validated against `DECIMAL(31,14)`: integer part ≤ 17 digits,
fractional part ≤ 14 digits, no rounding.  Out-of-range values set `quantity=None`
and add `QUANTITY_NOT_EXACTLY_REPRESENTABLE` to qualifiers.  The raw cell value is
preserved in ROW_JSON.

## Effectivity and Year-1900 Migration Policy

Year-1900 `valid_from` dates are preserved as migration placeholders.  Specifically:
- `is_migration_date=True` and `valid_from_date=date(1900, ...)` are recorded.
- **No substitute date is injected.**
- `VALID_TO` year-1900 dates are not flagged as placeholders.
- No active/current boolean is computed — no analysis-as-of date is available
  at parse time.  Effectivity evaluation against a business date occurs only
  in the analysis phase.

## Safe Error Codes and HTTP Status Boundary

| Code                          | HTTP | Trigger                                             |
|-------------------------------|------|-----------------------------------------------------|
| `BOM_WORKBOOK_CORRUPT`        | 400  | Not a ZIP or cannot be opened as OOXML              |
| `BOM_SHEET_ROLE_MISSING`      | 400  | Fewer than one sheet matches a required role        |
| `BOM_SHEET_ROLE_DUPLICATE`    | 400  | More than one sheet matches a required role         |
| `BOM_REQUIRED_HEADER_DUPLICATE` | 400 | A required/mapped header appears twice in a sheet |
| `BOM_ROW_LIMIT_EXCEEDED`      | 400  | Total visited data rows exceeds configured limit    |
| `BOM_CELL_LIMIT_EXCEEDED`     | 400  | Total visited cells exceeds configured limit        |
| `ECN_JSON_SYNTAX_INVALID`     | 400  | JSON file is malformed or not a top-level array     |
| `SOURCE_VALUE_TOO_LONG`       | 400  | An identifier field exceeds its HANA column length  |
| `UPLOAD_TOO_LARGE`            | 413  | Byte stream exceeds the upload size limit           |
| (framework validation)        | 422  | Request-field schema violations (later layers)      |

None of the 400 error details include raw cell values, ECN content, or row payloads.

## Related Files

| Path                                          | Role                                       |
|-----------------------------------------------|--------------------------------------------|
| `api/app/ecn/models.py`                       | Immutable domain records (`BomRow`, etc.)  |
| `api/app/ecn/intake/workbook_contracts.py`    | Role signatures and field mappings         |
| `api/app/ecn/intake/workbook_parser.py`       | OOXML parsing, bounded iteration, row build |
| `api/app/ecn/intake/hierarchy.py`             | Dotted/named hierarchy reconstruction      |
| `api/app/ecn/intake/normalization.py`         | Cell normalization, effectivity, classify  |
| `api/app/ecn/intake/json_parser.py`           | ECN JSON decoding and normalization        |
| `api/app/ecn/services/lifecycle.py`           | Shared preparation and mutation guards     |
| `api/app/ecn/services/ingestion.py`           | Deterministic preparation and publication  |
| `api/app/persistence/upload_repository.py`    | Atomic aggregate publication               |
| `api/app/ecn/errors.py`                       | Typed error hierarchy                      |
| `tests/unit/test_bom_parser.py`               | Parser unit tests (synthetic fixtures)     |
| `tests/unit/test_bom_hierarchy.py`            | Hierarchy unit tests (synthetic fixtures)  |
| `tests/unit/test_source_normalization.py`     | Normalization and contract unit tests      |
| `tests/unit/test_ingestion_service.py`        | Preparation, reuse, replay, and race tests |
| `tests/fixtures/builders.py`                  | In-memory workbook and row builders        |
| `tests/integration/test_development_fixtures.py` | Optional gated fixture tests            |

## Synthetic and Optional Development-fixture Verification

All committed unit tests use synthetic in-memory XLSX fixtures produced by
`tests/fixtures/builders.bom_workbook_bytes(...)`.  They never read from `data/`.

The development-fixture test in `tests/integration/test_development_fixtures.py`
is marked `development_fixture` and is **skipped by default**.  To run it:

```bash
RUN_DEVELOPMENT_FIXTURE_TESTS=1 PYTHONPATH=api python -m pytest \
  tests/integration/test_development_fixtures.py -m development_fixture
```

It requires a file matching `data/EBOM MBOM*.xlsx` (gitignored).  No sheet names,
row totals, or business identifiers are asserted — only that all three roles are
detected and at least one row was parsed.  Fixture files may be replaced without
updating the committed test.
