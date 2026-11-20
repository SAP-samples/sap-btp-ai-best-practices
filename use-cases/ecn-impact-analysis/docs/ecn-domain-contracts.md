# ECN Domain Contracts

This document defines the stable data boundaries, type contracts, invariants,
and API envelopes for the ECN impact analysis pipeline.  Later tasks, phases,
and the UI layer consume these definitions verbatim; changing a field name,
type, or bound here is a breaking change.

---

## Scope and Non-goals

**In scope:**

- Immutable identity and occurrence records (source artifacts, BOM rows, ECN rows).
- Deterministic evidence items and section groupings derived without LLM involvement.
- Strict six-section LLM analysis output with evidence-scope validation.
- Every API envelope that the HTTP layer produces or consumes.
- Runtime settings and the reproducibility model-configuration string.

**Out of scope:**

- Prompt templates, LLM provider configuration beyond `EcnSettings`, and
  streaming concerns.
- BOM parsing, effectivity evaluation, and rule evaluation internals.
- HANA table DDL and column-level constraints (see Task 3).
- Authentication and authorization (handled at the FastAPI layer).

---

## Immutable Identities and Occurrences

Every record in the pipeline is identified by a **canonical UUID string**
(`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).  The `canonical_uuid()` helper in
`app.ecn.models` rejects braced, bare-hex, and URN forms so that every stored
identifier is round-trip safe.

A **source artifact** (`SourceArtifact`) is the original uploaded file (ECN
JSON, BOM workbook, or conditional-rule workbook).  Artifacts are immutable
once stored.  Their `sha256` field is a 64-character lowercase hex digest
that the HANA layer stores as `NVARCHAR(64)`.

A **normalized ECN** (`NormalizedEcn`) is one ECN entry parsed from an ECN
JSON artifact.  Its `source_ordinal` is the zero-based index within the
source file.  Multiple ECNs from the same artifact share the same
`artifact_id`.

---

## Source, BOM, ECN, and Evidence Records

| Record | Key field | Notes |
|---|---|---|
| `SourceArtifact` | `artifact_id` | Immutable; `ArtifactKind` discriminates file type |
| `BomRow` | `bom_row_id` | `source_row_number` is 1-based and positive |
| `NormalizedEcn` | `ecn_row_id` | `ecn_number` is nullable |
| `EvidenceItem` | `evidence_id` | Belongs to exactly one `AnalysisSectionKey` section |
| `EvidenceBundle` | `ecn_row_id` | Aggregates all items for one ECN by section |

`EvidenceSource` points to the artifact location (JSON path or BOM row) that
supports one evidence item.  The `EvidenceContext` record captures the exact
version tags in effect when the bundle was derived:

- `PARSER_VERSION = "ecn-source-v1"` — ECN JSON normalization
- `DETERMINISTIC_LOGIC_VERSION = "ecn-evidence-v1"` — deterministic evidence engine
- `RULE_EVALUATOR_VERSION = "ecn-rules-v1"` — conditional-rule evaluator
- `PROMPT_VERSION = "ecn-impact-v1"` — LLM prompt template
- `OUTPUT_SCHEMA_VERSION = "ecn-impact-sections-v1"` — LLM structured output schema

---

## Six Analysis Sections and Status Invariants

The LLM analysis output is structured into exactly **six fixed sections**,
identified by `AnalysisSectionKey`:

| Key | Field name |
|---|---|
| `EXECUTIVE_SUMMARY` | `executive_summary` |
| `CHANGE_INTENT` | `change_intent` |
| `AFFECTED_MATERIALS` | `affected_materials` |
| `EBOM_MBOM_IMPACT` | `ebom_mbom_impact` |
| `CONFIGURATION_SCOPE` | `configuration_scope` |
| `UNCERTAINTIES_AND_MISSING_EVIDENCE` | `uncertainties_and_missing_evidence` |

Each section is an `AnalysisSection` model (`strict=True`, `extra="forbid"`)
with these status invariants enforced by a Pydantic `model_validator` (never
by bare `assert` so that optimized Python cannot disable them):

| `SectionStatus` | `evidence_ids` | `missing_evidence` |
|---|---|---|
| `SUFFICIENT` | one or more IDs | empty |
| `NOT_APPLICABLE` | one or more IDs (explaining non-applicability) | empty |
| `INSUFFICIENT_EVIDENCE` | zero or more IDs | one or more concrete entries |

All list entries are trimmed to nonblank and must be unique.

`validate_analysis_response(raw_json, allowed_evidence_ids_by_section)` adds a
second validation pass: every evidence ID cited in a section must belong to
**that section's** allowed set for the current ECN bundle.  An ID that is
valid in one section is rejected if cited from another section.  The function
uses `ValidationError.from_exception_data` so the error location path always
names the offending section.

`derive_completeness(response)` derives the overall `AnalysisCompleteness`:

- Returns `COMPLETED_WITH_GAPS` if **any** section has `INSUFFICIENT_EVIDENCE`.
- Returns `COMPLETED` otherwise (`NOT_APPLICABLE` alone does not create a gap).

**Important:** `COMPLETED_WITH_GAPS` is an **API/result projection** computed
at read time from the stored analysis completeness.  The mutable HANA job table
uses `JobState`, which contains only `QUEUED`, `RUNNING`, `COMPLETED`, and
`FAILED`.  `COMPLETED_WITH_GAPS` is **never** stored in the job row; it is
derived via `derive_completeness()` and exposed through `AnalysisFilterState`
in API responses.

---

## API States, Counts, and Pagination

### AnalysisFilterState

The `AnalysisFilterState` enum adds `COMPLETED_WITH_GAPS` to the four
`JobState` values:

```
QUEUED | RUNNING | COMPLETED | COMPLETED_WITH_GAPS | FAILED
```

`COMPLETED_WITH_GAPS` is absent from `JobState` and is never written to the
job table.

### Count equations

`UploadResponse` enforces:

```
queued_count + running_count + completed_count
+ completed_with_gaps_count + failed_count == imported_ecn_count
```

`AnalysisProgressResponse` enforces two equations:

```
unfinished_count == queued_count + running_count
total_count == unfinished_count + completed_count
              + completed_with_gaps_count + failed_count
```

Both equations are enforced by Pydantic `model_validator(mode="after")`.

### Pagination

`EcnListQuery` enforces `1 <= page_size <= 100`.  `EcnListResponse` returns
`total_items` and `total_pages` as nonnegative integers.

### Sort allowlist

The `sort` field on `EcnListQuery` is a closed `Literal` accepting exactly:
`uploaded_at`, `ecn_number`, `description`, `affected_item_count`,
`ebom_match_count`, `mbom_match_count`, `warning_count`, `analysis_state`.

---

## Runtime Settings and Reproducibility Versions

`EcnSettings` is a frozen (`frozen=True`) Pydantic model loaded once at
startup by `load_ecn_settings()`.  Environment variable names map to fields
via `ENVIRONMENT_FIELDS`:

| Environment variable | Field | Default |
|---|---|---|
| `LLM_MAX_CONCURRENCY` | `llm_max_concurrency` | `4` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `llm_request_timeout_seconds` | `180` |
| `MAX_JSON_UPLOAD_BYTES` | `max_json_upload_bytes` | 128 MiB |
| `MAX_WORKBOOK_UPLOAD_BYTES` | `max_workbook_upload_bytes` | 100 MiB |
| `MAX_WORKBOOK_ROWS` | `max_workbook_rows` | `100 000` |
| `MAX_WORKBOOK_CELLS` | `max_workbook_cells` | `3 000 000` |
| `MAX_PROMPT_EVIDENCE_ITEMS` | `max_prompt_evidence_items` | `500` |
| `MAX_PROMPT_CHARACTERS` | `max_prompt_characters` | `250 000` |
| `LLM_MODEL` | `llm_model_name` | `"gpt-5.4"` |

Empty string values are ignored (treated as absent), so deployment tooling may
set variables to `""` to restore the default.

Plants are not configured through environment variables. The brand suffix to
plant mapping and the rule states used for plant assignment are maintained in
Settings and stored in HANA (see [Rule-based plant assignment](rule-based-plant-assignment.md)).

`canonical_model_config(settings)` emits a sorted, compact JSON string
containing exactly four keys that are pinned in every stored analysis record
for reproducibility:

```json
{
  "max_prompt_characters": 250000,
  "max_prompt_evidence_items": 500,
  "request_timeout_seconds": 180,
  "strict_structured_output": true
}
```

`strict_structured_output` is always `true`; it is not configurable.

---

## Safe Errors

`ErrorResponse` exposes only a stable code slug (max 80 chars, aligned to the
HANA `NVARCHAR(80)` error-log column) and a safe message (max 2000 chars,
aligned to `NVARCHAR(2000)`).  Sensitive source content must never appear in
either field.

`HistoryResetRequest` requires the exact phrase `"RESET HISTORY"` to prevent
accidental deletion.  `HistoryResetResponse` reports dependency-ordered
deletion counts (attempts → analyses → jobs → ECNs → uploads → BOM rows →
source artifacts).

---

## Related Files

| File | Purpose |
|---|---|
| `api/app/ecn/models.py` | Enums, dataclasses, version constants (Task 1) |
| `api/app/ecn/errors.py` | Typed error hierarchy (Task 1) |
| `api/app/ecn/analysis_models.py` | Six-section Pydantic models (Task 2) |
| `api/app/ecn/api_models.py` | API envelope models (Task 2) |
| `api/app/ecn/settings.py` | `EcnSettings`, `load_ecn_settings`, `canonical_model_config` (Task 2) |
| `tests/fixtures/builders.py` | Deterministic test-data builders |
| `tests/unit/test_analysis_contract.py` | Section invariant and completeness tests |
| `tests/unit/test_ecn_settings.py` | Settings defaults, overrides, and canonical config |
| `tests/unit/test_ecn_api_models.py` | API model bounds, equations, and enum tests |

---

## Focused Verification

Run the three Task 2 test files in isolation to verify the contracts defined
here:

```bash
PYTHONPATH=api python -m pytest -q tests/unit/test_analysis_contract.py tests/unit/test_ecn_settings.py tests/unit/test_ecn_api_models.py
```

To run the complete unit suite (including Task 1 tests):

```bash
PYTHONPATH=api python -m pytest -q tests/unit
```

To verify all Python files under `api/app` compile without errors:

```bash
python -m compileall -q api/app
```
