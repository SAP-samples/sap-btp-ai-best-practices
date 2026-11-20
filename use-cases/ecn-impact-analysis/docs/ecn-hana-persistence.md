# ECN HANA Persistence

## Phase Scope

This document covers the schema initialization and structural contract
validation implemented in **Phase 01, Task 4** of the ECN Impact MVP.
The scope is limited to creating and validating the seven physical HANA
column tables at startup.  FastAPI lifespan wiring — calling `ensure_schema`
during application startup — is deferred to a later API phase.

**Files owned by this phase:**

| File | Role |
|------|------|
| `api/app/persistence/schema.py` | `current_schema`, `table_exists`, `validate_table_contract`, `ensure_schema` |
| `api/app/persistence/table_contracts.py` | Column, FK, UQ, and table contracts; `build_table_contracts`; `render_create_table` |
| `api/app/persistence/hana.py` | `create_hana_engine`, timestamp helpers |
| `tests/unit/test_hana_schema.py` | Unit tests with fake inspector and fake engine |
| `tests/integration/test_hana_persistence.py` | Opt-in live HANA test with UUID namespace |
| `tests/fixtures/persistence.py` | `RecordingConnection`, `FakeInspector`, `FakeEngine`, helpers |

---

## Environment Inputs and CURRENT_SCHEMA

`create_hana_engine` reads five environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `HANA_ADDRESS` | Yes | HANA host |
| `HANA_PORT` | Yes | Integer, 1–65535 |
| `HANA_USER` | Yes | Database username |
| `HANA_PASSWORD` | Yes | Database password |
| `HANA_ENCRYPT` | No | `true/1/yes` or `false/0/no`; defaults to `true` |

`current_schema(connection)` executes `SELECT CURRENT_SCHEMA FROM DUMMY` and
returns the authenticated user's schema name.  The result is stripped of
whitespace; a blank or `None` result raises `StartupConfigurationError` with
code `HANA_SCHEMA_UNAVAILABLE`.  All subsequent `SYS.TABLES` queries and DDL
statements operate within this schema.

---

## Seven Tables and 128-column Contract

`build_table_contracts(table_names)` returns a tuple of seven `TableContract`
instances in dependency-safe creation order:

| Order | Key | Physical name (default) | Columns |
|-------|-----|------------------------|---------|
| 1 | `source_artifact` | `ECN_IMPACT_SOURCE_ARTIFACT` | 11 |
| 2 | `upload` | `ECN_IMPACT_UPLOAD` | 13 |
| 3 | `bom_row` | `ECN_IMPACT_BOM_ROW` | 35 |
| 4 | `ecn` | `ECN_IMPACT_ECN` | 21 |
| 5 | `analysis_job` | `ECN_IMPACT_ANALYSIS_JOB` | 13 |
| 6 | `llm_attempt` | `ECN_IMPACT_LLM_ATTEMPT` | 21 |
| 7 | `analysis` | `ECN_IMPACT_ANALYSIS` | 14 |

Total across all tables: **128 columns**.

Each `ForeignKeyContract` uses a `referenced_table_key` (e.g., `"source_artifact"`)
resolved through `PersistenceTableNames`.  When a suffixed namespace is used
(e.g., from `PersistenceTableNames().with_suffix("_TEST")`), all FK targets
resolve to the suffixed names so that the isolated namespace is self-contained.

---

## Creation and Dependency Order

`ensure_schema(engine, table_names)` creates missing tables in the order
defined by `build_table_contracts`:

```
source_artifact → upload → bom_row → ecn → analysis_job → llm_attempt → analysis
```

Each table is created with:
- A `CREATE COLUMN TABLE` statement (no `IF NOT EXISTS`).
- An unnamed `PRIMARY KEY` clause.
- Unnamed `UNIQUE` clauses.
- Unnamed `FOREIGN KEY … REFERENCES` clauses with targets resolved through
  `table_names`.

After each table is created, `validate_table_contract` is called immediately
on a fresh connection to confirm the table matches the contract before the
next table is created.

---

## Existing-table Compatibility Validation

`validate_table_contract(connection, schema, contract, table_names)` uses the
SQLAlchemy HANA inspector to compare a live table against its contract.

**Comparisons performed:**

1. **Columns** — exact ordered tuple of `(name_upper, type_class_upper, nullable, length, precision, scale)`.  Extra or missing columns both fail.
2. **Column type** — `reflected_type.__class__.__name__.upper()` against `contract.hana_type.upper()`.
3. **NVARCHAR / LOB length** — `getattr(type, "length", None)` compared exactly.
4. **DECIMAL precision / scale** — `getattr(type, "precision/scale", None)` compared exactly.
5. **Nullability** — `bool(column["nullable"])`.
6. **Primary key** — ordered column name tuple.
7. **Unique constraints** — sorted set of column-name tuples; generated constraint names are ignored.
8. **Foreign keys** — for each FK in declaration order: local columns, referenced schema (`None` treated as current schema), physical referenced table (resolved via `table_names`), referenced columns.

Any mismatch raises `StartupConfigurationError` with code
`HANA_SCHEMA_INCOMPATIBLE` and a message containing `"incompatible"`.

---

## HANA DDL Auto-commit and Retry Behavior

HANA DDL statements (`CREATE TABLE`, `DROP TABLE`) may auto-commit
regardless of any surrounding transaction.  `ensure_schema` is designed
with this in mind:

- **Validation runs first**: all existing tables are validated in one pass
  before any DDL is attempted.  If any existing table is incompatible, the
  error propagates and no DDL runs.
- **One connection per table**: each `CREATE COLUMN TABLE` statement runs in
  its own `engine.connect()` context followed by an explicit `commit()`.
- **No transactional rollback**: the creation loop does NOT use a single
  `engine.begin()` or claim rollback semantics.
- **Safe on retry**: if creation succeeds for table N but fails for table N+1
  (raising `StartupConfigurationError` with code
  `HANA_SCHEMA_INITIALIZATION_FAILED`), table N is left on disk.  On the
  next startup attempt, table N is found, validated as compatible, and
  skipped.  The remaining tables are then created.  This makes `ensure_schema`
  idempotent: re-running it on a fully-created schema completes without
  executing any DDL.

---

## Identifier and Parameter Safety

- All physical table names must match `^ECN_IMPACT_[A-Z0-9_]{1,110}$`.
  `quote_identifier(name)` validates and double-quotes each name before
  use in DDL.
- `table_exists` uses `SCHEMA_NAME = :schema AND TABLE_NAME = :table_name`
  bound parameters so neither value is ever string-interpolated into SQL.
- Passwords and driver exception text are never embedded in
  `StartupConfigurationError` messages.

---

## Live-test Isolation and Cleanup

The live integration test in `tests/integration/test_hana_persistence.py`
is decorated with `@pytest.mark.hana_live` and uses
`@pytest.mark.skipif(os.getenv("RUN_HANA_INTEGRATION_TESTS") != "1", ...)`.

It creates a UUID-suffixed namespace (`_T<10-hex-chars>`) so that test tables
never clash with production tables.  The `finally` block drops only the seven
suffixed tables in reverse dependency order (FK-safe).  It never touches
unsuffixed application tables and never calls history reset.

**Default-skipped run (CI, unit tests):**

```bash
PYTHONPATH=api python -m pytest -q tests/integration/test_hana_persistence.py
# Expected: 1 skipped
```

**Explicit live run:**

```bash
RUN_HANA_INTEGRATION_TESTS=1 \
    HANA_ADDRESS=<host> HANA_PORT=<port> \
    HANA_USER=<user> HANA_PASSWORD=<pass> HANA_ENCRYPT=true \
    PYTHONPATH=api python -m pytest -q tests/integration/test_hana_persistence.py
```

---

---

## Canonical JSON and LOB Boundary

`canonical_json(value)` in `api/app/persistence/codec.py` produces a
UTF-8-safe JSON string that is byte-for-byte reproducible across calls:

| Setting | Value |
|---------|-------|
| `ensure_ascii` | `False` — non-ASCII characters are stored as-is |
| `sort_keys` | `True` — all dict keys at every nesting level are sorted |
| `separators` | `(",", ":")` — no whitespace in the output |

Any SHA-256 digest stored alongside a payload is computed over the exact
canonical UTF-8 bytes so that two encoders given the same input always
produce the same digest.

`read_lob_text(value)` and `read_lob_bytes(value)` handle the three shapes
the hdbcli driver may return:

| Shape | Handling |
|-------|----------|
| `str` | Returned as-is (`read_lob_text`) or encoded UTF-8 (`read_lob_bytes`) |
| `bytes` | Decoded UTF-8 (`read_lob_text`) or returned as-is (`read_lob_bytes`) |
| Object with `.read()` | `read()` is called; result is then handled as `str` or `bytes` |
| Any other type | `PersistenceStateError("LOB_READ_ERROR", …, status_code=500)` |

`QUALITY_JSON` stored for BOM rows is always an object with exactly two keys:
`details` (a dict) and `qualifiers` (an array of strings).  `ROW_JSON` is a
JSON array of cell-entry objects.  Any deviation from these shapes at read time
raises `PersistenceStateError("PERSISTED_JSON_INVALID", …, status_code=500)`.

---

## Artifact Repository

`ArtifactRepository` in `api/app/persistence/artifact_repository.py` exposes
five public methods:

| Method | Connection type | Purpose |
|--------|-----------------|---------|
| `get_artifact(artifact_id)` | `engine.connect()` | Single SELECT by PK |
| `get_current_rule_artifact()` | `engine.connect()` | Latest CONDITIONAL_RULE_WORKBOOK |
| `publish_rule_artifact(artifact)` | `engine.begin()` | Append-only INSERT |
| `iter_bom_rows(artifact_id, batch_size)` | `engine.connect()` per page | Paginated SELECT |

**Short-lived connections**: every method opens and closes its own connection.
No `Connection` is held as instance state.

**Append-only rule publication**: `publish_rule_artifact` accepts only a
`CONDITIONAL_RULE_WORKBOOK` artifact with `parse_state=VALID`.  Validation
happens before the transaction opens so that no transaction is started for
invalid inputs.  A duplicate-key `IntegrityError` is converted to
`PersistenceStateError("ARTIFACT_ALREADY_EXISTS")` to avoid leaking driver
error text.

**Rule ordering**: `get_current_rule_artifact` orders by
`"CREATED_AT" DESC, "ARTIFACT_ID" DESC` for a deterministic tie-breaker.

**Connection-closed pagination**: `iter_bom_rows` fetches one page of
`batch_size` rows, closes the connection, decodes the batch into `BomRow`
objects, and only then yields.  This ensures the generator never holds a HANA
connection while the caller processes each page.  Pages are ordered by
`BOM_KIND, SHEET_NAME, SOURCE_ROW_NUMBER, BOM_ROW_ID` for stability.

**Identifier safety**: the only string interpolated into SQL is the table
identifier, guarded by `quote_identifier` which enforces
`^ECN_IMPACT_[A-Z0-9_]{1,110}$`.  All data values are bound parameters.

---

## Token-column Compatibility

`usage_to_columns(usage, *, usage_available)` in `codec.py` maps a
`TokenUsage`-shaped object to `AttemptUsageColumns` without modifying the
current four-field `TokenUsage` class.

**Current four-field mapping**:

| `TokenUsage` field | `AttemptUsageColumns` field | Notes |
|--------------------|----------------------------|-------|
| `input_tokens` | `input_tokens` | Direct |
| `cached_input_tokens` | `cache_read_input_tokens` | Field rename |
| `output_tokens` | `output_tokens` | Direct |
| `total_tokens` | `total_tokens` | Direct |
| *(absent)* | `cache_write_input_tokens` | `NULL` — not yet surfaced |
| *(absent)* | `input_total_tokens` | `NULL` — not yet surfaced |

**`usage_available` flag**: when `False`, every integer column is stored as
`NULL` regardless of the zero defaults on the `TokenUsage` object.  This
prevents `TokenUsage()` (all zeros) from being stored as if real provider data
was available.

**Forward compatibility**: `usage_to_columns` uses `getattr(usage, name, None)`
for every field.  When a future richer `TokenUsage` adds
`cache_write_input_tokens` or `input_total_tokens`, those values will be
automatically picked up without any change to this function.  Task 8 adds the
richer attributes and the associated compatibility tests; this phase stores
`NULL` for those two columns.

---

---

## Upload Publication Transaction

`UploadRepository.publish_upload(publication)` in
`api/app/persistence/upload_repository.py` is the only method that writes
upload-related rows.  It runs a single `engine.begin()` transaction that
executes the following six insert groups in dependency order:

| Step | Table | Condition |
|------|-------|-----------|
| 1 | `ECN_IMPACT_SOURCE_ARTIFACT` (ECN JSON) | Always |
| 2 | `ECN_IMPACT_SOURCE_ARTIFACT` (BOM workbook) | `bom_reused=False` only |
| 3 | `ECN_IMPACT_BOM_ROW` (batch) | `bom_reused=False` and rows present |
| 4 | `ECN_IMPACT_UPLOAD` | Always |
| 5 | `ECN_IMPACT_ECN` (batch) | Always (one row per ECN) |
| 6 | `ECN_IMPACT_ANALYSIS_JOB` (batch) | Always (one job per ECN) |

`PUBLICATION_ORDER` is allocated with
`COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1` inside the transaction before
any inserts.  This guarantees a gap-free, monotonically increasing sequence
within the process, but relies on the caller serializing concurrent
publications at the service layer.  Cross-instance sequence allocation is out
of scope for this MVP.

**Idempotency — preflight fast path**: before opening a transaction,
`publish_upload` calls `find_by_request_id`.  If the row exists, it returns
`UploadPublishResult(existing, created=False)` without starting a transaction
(`begin_count` stays zero for the replay).

**Idempotency — race path**: if two calls with the same `request_id` enter the
transaction concurrently, the one that commits second encounters an
`IntegrityError` on the `UNIQUE(REQUEST_ID)` constraint.  `publish_upload`
catches `IntegrityError`, rolls back, re-queries by `request_id`, and returns
the winner's record with `created=False`.

**Distinguishing races**: only a `REQUEST_ID` uniqueness collision is treated
as an idempotent replay.  Any other uniqueness violation (e.g. a programming
error that produces a duplicate `PUBLICATION_ORDER`) raises
`PersistenceStateError("UPLOAD_PUBLICATION_CONFLICT", status_code=409)`.

**Preflight validation** runs before any database access.  Invalid inputs raise
`ValueError` without opening a connection.  A new `request_id` always produces
new ECN occurrence rows even when the source file hashes match a prior upload.

---

## BOM Snapshot Reuse

When the caller determines that the current BOM workbook has not changed since
the last upload (e.g. by comparing SHA-256 digests), it sets `bom_reused=True`
in `UploadPublication`.

Under reuse semantics:

* `new_bom_rows` must be empty (validated before the transaction opens).
* Steps 2 and 3 of the publication transaction are skipped entirely.
* The upload row references the existing BOM artifact's `ARTIFACT_ID` without
  any copy of the artifact bytes or BOM rows.
* The BOM snapshot is read back via `get_latest_bom_snapshot()`, which joins
  `ECN_IMPACT_UPLOAD` to `ECN_IMPACT_SOURCE_ARTIFACT` on
  `BOM_ARTIFACT_ID = ARTIFACT_ID` and selects the row with the greatest
  `PUBLICATION_ORDER`.

`get_latest_bom_snapshot()` orders by `U."PUBLICATION_ORDER" DESC,
U."UPLOAD_ID" DESC` to provide a deterministic tie-breaker.  It binds
`artifact_kind = "BOM_WORKBOOK"` as a parameter so that independently
published conditional-rule artifacts are never returned.

---

---

## Historical ECN Reads

`EcnRepository` in `api/app/persistence/ecn_repository.py` provides read-only access to
the historical occurrence table.  All queries use bound parameters; the only string
interpolated into SQL is a table identifier validated by `quote_identifier`.

### Filter Behaviour

| Filter field | SQL condition | Notes |
|---|---|---|
| `query` | `UPPER(E."ECN_NUMBER") LIKE :query_pattern` and two more columns | Case-insensitive contains; value always bound |
| `ecn_number` | `E."ECN_NUMBER" = :ecn_number` | Exact equality; no case-fold; Pydantic strips whitespace |
| `analysis_state=COMPLETED_WITH_GAPS` | `J."STATE" = :completed_job_state AND A."COMPLETENESS_STATE" = :gap_state` | Both conditions required |
| `analysis_state=COMPLETED` | `J."STATE" = :completed_job_state AND (A."COMPLETENESS_STATE" IS NULL OR A."COMPLETENESS_STATE" = :completed_analysis_state)` | Excludes gap rows |
| `analysis_state` (other) | `J."STATE" = :analysis_state_value` | Direct job state match |
| `has_warnings=True` | `E."WARNING_COUNT" > 0` | |
| `has_warnings=False` | `E."WARNING_COUNT" = 0` | |

Both the count query and the list query receive **identical WHERE clauses and bound
parameters** so that `total_items` exactly matches the filtered result set.

### Derived Gap State

The `DERIVED_ANALYSIS_STATE` CASE expression is aliased in the list SELECT:

```sql
CASE
    WHEN J."STATE" = 'COMPLETED' AND A."COMPLETENESS_STATE" = 'COMPLETED_WITH_GAPS'
    THEN 'COMPLETED_WITH_GAPS'
    WHEN J."STATE" = 'COMPLETED' THEN 'COMPLETED'
    ELSE J."STATE"
END AS "DERIVED_ANALYSIS_STATE"
```

`AnalysisFilterState.COMPLETED_WITH_GAPS` is an API-only projection; the HANA job
table stores only `JobState.COMPLETED`.  A completed job with no analysis row yet
shows as `COMPLETED`.

### Stable Tie-breakers and Sort Allowlist

All eight sort keys map to fixed SQL expressions via `SORT_EXPRESSIONS`.  The raw
query sort value is used only as a dict key; it is never interpolated into SQL.

Default tie-breaker appended after every primary sort:
`U."PUBLICATION_ORDER" DESC, E."SOURCE_ORDINAL" ASC, E."ECN_ROW_ID" ASC`

Nullable sort keys (`ecn_number`, `description`, `analysis_state`) append
`NULLS LAST` so that rows without a value sort deterministically.

### Six Count Definitions

| `ImpactCounts` field | Source column |
|---|---|
| `affected_item_count` | `E."AFFECTED_ITEM_COUNT"` |
| `raw_ecn_match_count` | `E."RAW_ECN_MATCH_COUNT"` |
| `ebom_match_count` | `E."EBOM_MATCH_COUNT"` |
| `ebom_context_count` | `E."EBOM_CONTEXT_COUNT"` |
| `mbom_match_count` | `E."MBOM_MATCH_COUNT"` |
| `warning_count` | `E."WARNING_COUNT"` |

### Detail Immutability

`get_ecn_detail(ecn_row_id)` executes a single bounded JOIN across all seven
structural tables, returning a frozen `EcnDetailRecord`.  It performs no LLM
inference, no state mutation, and no job scheduling.  Artifact columns are
decoded with per-table prefixes (`ecn_artifact.`, `bom_artifact.`,
`rule_artifact.`) so that `artifact_from_row(row, prefix=...)` can reuse the
existing codec without new mapper functions.

### Pure Public-Response Mapper Boundary

Five pure mapper functions in `codec.py` serve as the **only** conversion
boundary between persistence records and HTTP response types.  Route handlers
must call these functions; they must not reconstruct response payloads ad hoc.

| Function | Input | Output |
|---|---|---|
| `ecn_list_item_from_row(row)` | list query result row | `EcnListItem` |
| `artifact_view_from_record(artifact)` | `SourceArtifact` | `ArtifactView` |
| `evidence_section_summaries_from_bundle(bundle)` | `EvidenceBundle` | `EvidenceSectionSummaries` |
| `ecn_detail_response_from_record(record)` | `EcnDetailRecord` | `EcnDetailResponse` |
| `evidence_detail_response_from_item(ecn_row_id, item)` | `str` + `EvidenceItem` | `EvidenceDetailResponse` |

All five mappers are pure functions with no SQL and no connection lifecycle.

### Python-side Same-ECN Evidence Isolation

`get_evidence(ecn_row_id, evidence_id)` is a security boundary:

1. Selects **only** `EVIDENCE_JSON WHERE E."ECN_ROW_ID" = :ecn_row_id` — one
   column, one row.
2. Decodes the bundle in Python via `evidence_bundle_from_json`.
3. Searches the decoded bundle for `evidence_id` in Python.
4. An `evidence_id` that belongs to a different ECN's bundle is never found
   because the wrong EVIDENCE_JSON was never fetched.
5. The `evidence_id` lookup value is **never sent to HANA** — it is matched
   entirely in application code.

The `evidence_id` parameter is also pre-validated: empty strings and values
longer than 200 characters are rejected before the DB connection is opened.

### Why UI Search Never Relaxes Source Join Identity

The free-text `query` filter searches `ECN_NUMBER`, `TITLE`, and
`DESCRIPTION_SUMMARY` columns.  It deliberately never relaxes the `UPLOAD_ID`
or `ECN_ROW_ID` joins so that a text search cannot surface occurrences from a
different upload context.  Each ECN occurrence is treated as an immutable
snapshot; search results are filtered rows of the same historical table, not
cross-upload aggregates.

---

## Related Files

```
api/
  app/
    ecn/
      errors.py              — StartupConfigurationError, PersistenceStateError definitions
    persistence/
      hana.py                — create_hana_engine, to_hana_timestamp, from_hana_timestamp
      schema.py              — current_schema, table_exists, validate_table_contract, ensure_schema
      table_contracts.py     — PersistenceTableNames, build_table_contracts, render_create_table
      records.py             — frozen repository command/result records (Task 5)
      codec.py               — canonical_json, LOB helpers, row mappers, usage_to_columns (Task 5)
      artifact_repository.py — ArtifactRepository: artifact/BOM reads (Task 5)
      upload_repository.py   — UploadRepository: atomic publication, snapshot lookups (Task 6)
tests/
  fixtures/
    persistence.py           — RecordingConnection, FakeInspector, FakeEngine, RecordingEngine
    builders.py              — domain builders + source_artifact_row, bom_row_mapping, upload_row,
                               upload_publication (Tasks 5–6)
  integration/
    test_hana_persistence.py — live HANA test (gated, UUID-suffixed namespace)
  unit/
    test_hana_schema.py      — schema unit tests (fake inspector, fake engine)
    test_hana_table_contracts.py — contract and DDL rendering tests
    test_artifact_repository.py  — codec round-trips, recording-engine SQL assertions (Task 5)
    test_upload_repository.py    — upload lookup, BOM snapshot, atomic publication (Task 6)
pytest.ini                   — marker registration (hana_live, llm_live, development_fixture)
```

---

## Verification

Run unit tests only (no HANA credentials needed):

```bash
PYTHONPATH=api python -m pytest -q tests/unit/test_hana_schema.py
PYTHONPATH=api python -m pytest -q tests/unit/test_hana_table_contracts.py
PYTHONPATH=api python -m pytest -q tests/unit
python -m compileall -q api/app
```

Run the full test suite including default-skipped integration test:

```bash
PYTHONPATH=api python -m pytest -q tests/unit tests/integration
# Expected: all unit tests pass, 1 integration test skipped
```

---

## Analysis Claim and Attempt Transaction

**File:** `api/app/persistence/analysis_repository.py`

`AnalysisRepository` implements the two write paths needed by the LLM analysis
worker: claiming a queued job and recording the provider call outcome.

### AttemptSeed → Pinned-Model AttemptStart boundary

The worker supplies an `AttemptSeed` that contains identity, correlation, and
provider fields **but no `model_name`**.  This is deliberate: the repository
reads `MODEL_NAME` from the persisted job row and constructs `AttemptStart`
with `model_name=config.model_name`.  No caller-supplied value can override
the model that was pinned at ingestion time.

### Claim transaction flow

`claim_job_and_start_attempt(job_id, AttemptSeed)` runs one `engine.begin()`
transaction:

1. **SELECT** the job row (joined to ECN for `UPLOAD_ID` and `EVIDENCE_JSON`)
   filtering only by `JOB_ID` — not by state.
2. **Python state check**: missing row → `NOT_FOUND`; `STATE != 'QUEUED'` →
   `NOT_QUEUED`.  These are separate so the caller can distinguish the two.
3. **Pinned config parse**: `MODEL_CONFIG_JSON` is validated with `_PinnedPayload`
   (strict Pydantic, `extra="forbid"`).  If invalid → conditional UPDATE:

   ```sql
   UPDATE ECN_IMPACT_ANALYSIS_JOB
   SET "STATE" = 'FAILED',
       "FINISHED_AT" = :finished_at,
       "ERROR_CODE" = 'PINNED_CONFIG_INVALID',
       "ERROR_MESSAGE" = 'The pinned generation configuration is invalid'
   WHERE "JOB_ID" = :job_id AND "STATE" = 'QUEUED'
   ```

   If that UPDATE wins (rowcount=1) → `PINNED_CONFIG_INVALID`.  If it loses
   the race (rowcount=0) → `NOT_QUEUED`.  The error message never contains the
   raw malformed JSON.

4. **Atomic claim UPDATE** (one-winner rule):

   ```sql
   UPDATE ECN_IMPACT_ANALYSIS_JOB
   SET "STATE" = 'RUNNING',
       "ATTEMPT_COUNT" = "ATTEMPT_COUNT" + 1,
       "STARTED_AT" = :started_at,
       "FINISHED_AT" = NULL,
       "ERROR_CODE" = NULL,
       "ERROR_MESSAGE" = NULL
   WHERE "JOB_ID" = :job_id AND "STATE" = 'QUEUED'
   ```

   Only `rowcount == 1` permits proceeding.  The race loser receives
   `NOT_QUEUED` with no further side-effects — no attempt row is inserted.

5. **Attempt INSERT**: a row with `PROVIDER_OUTCOME='STARTED'`,
   `VALIDATION_OUTCOME='NOT_RUN'`, all six token columns NULL, and
   `USAGE_AVAILABLE=0` is inserted into `ECN_IMPACT_LLM_ATTEMPT`.
   `attempt_number` is `ATTEMPT_COUNT + 1` (read from the pre-UPDATE value).

6. Returns `ClaimJobResult(CLAIMED, ClaimedJob)` with the decoded evidence
   bundle, pinned config, and the exact `AttemptStart` that was persisted.

### All four `ClaimJobStatus` outcomes

| Status | Meaning | `claimed_job` |
|--------|---------|---------------|
| `CLAIMED` | Job is now RUNNING; attempt inserted | Non-null |
| `NOT_FOUND` | No row for the given `job_id` | `None` |
| `NOT_QUEUED` | Row exists but `STATE != 'QUEUED'`; or race loss | `None` |
| `PINNED_CONFIG_INVALID` | Config JSON is malformed/version-invalid; job marked FAILED | `None` |

### `NOT_CALLED` prompt-projection failure semantics

`NOT_CALLED` is the terminal `ProviderCallOutcome` when a deterministic step
(e.g. prompt projection, context assembly) fails **before any LLM request is
made**.  Its constraints are enforced before any SQL:

* `error_code` must be non-null (records why the call was skipped).
* `usage` must be `None` and `usage_available` must be `False`.
* `latency_ms` must be `None` (no call = no latency).

Violating any constraint raises `ValueError` without touching the database.

### Provider-versus-validation separation

`record_provider_outcome` writes `PROVIDER_OUTCOME`, token usage columns, and
`FINISHED_AT`.  It deliberately **does not write `VALIDATION_OUTCOME`**.
Schema validation is a separate step with its own write (Task 9); keeping the
two writes separate prevents a failed validation from hiding a successful
provider call, and vice versa.

### Current token compatibility

`usage_to_columns` uses `getattr` with fallback for every token field so that
the current four-field `TokenUsage` (which has `cached_input_tokens` but not
`cache_write_input_tokens` or `input_total_tokens`) works without modification.
Future richer `TokenUsage` objects will be picked up automatically.  The current
mapping:

| `TokenUsage` field | HANA column |
|--------------------|-------------|
| `input_tokens` | `INPUT_TOKENS` |
| `cached_input_tokens` | `CACHE_READ_INPUT_TOKENS` |
| *(absent)* | `CACHE_WRITE_INPUT_TOKENS` → NULL |
| *(absent)* | `INPUT_TOTAL_TOKENS` → NULL |
| `output_tokens` | `OUTPUT_TOKENS` |
| `total_tokens` | `TOTAL_TOKENS` |

### No SDK or model calls inside transactions

The `engine.begin()` context in both `claim_job_and_start_attempt` and
`record_provider_outcome` holds open a HANA connection.  No model SDK calls,
HTTP requests, or I/O of any kind are made inside these transaction blocks.
All provider calls happen entirely outside the transaction window.

---

## Terminal Analysis Transactions

### Completion preconditions

`complete_job(job_id, AnalysisCompletion)` executes exactly one `engine.begin()`
transaction containing three writes.  Before the writes, a precondition SELECT
loads a row joining `ECN_IMPACT_ANALYSIS_JOB`, `ECN_IMPACT_ECN`, and
`ECN_IMPACT_LLM_ATTEMPT` (LEFT JOIN by attempt ID) with a correlated EXISTS
sub-query to check for an existing analysis.  The Python-side check
`_require_completion_preconditions` validates all of the following before any
write is attempted:

| Precondition | Reason |
|---|---|
| `job_id == completion.job_id` | Prevents a completion from silently borrowing another job |
| `JOB_STATE == 'RUNNING'` | Only RUNNING jobs can be completed |
| `JOB_ECN_ROW_ID == completion.ecn_row_id` | ECN must match across all layers |
| `ATTEMPT_JOB_ID == job_id` | Attempt must belong to this exact job (cross-job write rejected) |
| `PROVIDER_OUTCOME == 'SUCCESS'` | Only a successful provider call justifies a valid analysis |
| `VALIDATION_OUTCOME == 'NOT_RUN'` | Attempt must not already be validated |
| `EVIDENCE_HASH matches` | Immutable evidence integrity |
| `PROMPT_VERSION matches` | Pinned prompt version consistency |
| `OUTPUT_SCHEMA_VERSION matches` | Pinned schema version consistency |
| `MODEL_NAME matches` | Pinned model consistency |
| `DETERMINISTIC_LOGIC_VERSION matches` | Stored ECN version consistency |
| `RULE_EVALUATOR_VERSION matches` | Stored ECN version consistency |
| `ATTEMPT_MODEL_DEPLOYMENT matches` | Deployment recorded on the attempt must match completion |
| `ANALYSIS_EXISTS == 0` | No existing analysis for this job (UQ constraint guard) |
| `completeness == derive_completeness(response)` | Caller cannot inject an inconsistent completeness |

Any mismatch raises `PersistenceStateError("ANALYSIS_STATE_CONFLICT", status_code=409)`.
All three writes are rolled back atomically.

### Three-write transaction

1. **INSERT** into `ECN_IMPACT_ANALYSIS` with all 14 columns including canonical-JSON
   `CONTENT_JSON` and the derived `COMPLETENESS_STATE`.
2. **UPDATE** `ECN_IMPACT_LLM_ATTEMPT` where `ATTEMPT_ID = :attempt_id AND JOB_ID = :job_id
   AND PROVIDER_OUTCOME='SUCCESS' AND VALIDATION_OUTCOME='NOT_RUN'` — sets
   `VALIDATION_OUTCOME='VALID'`.
3. **UPDATE** `ECN_IMPACT_ANALYSIS_JOB` where `JOB_ID = :job_id AND STATE='RUNNING'` —
   sets `STATE='COMPLETED'` and clears error columns.

Both conditional UPDATEs must return `rowcount==1`.  If either returns 0, all three
writes are rolled back and `ANALYSIS_STATE_CONFLICT` is raised.

A unique-constraint `IntegrityError` on the analysis INSERT is also caught and
converted to `ANALYSIS_STATE_CONFLICT` so callers always see the same typed error.

### No method updates an existing analysis

`complete_job` is the only path that writes to `ECN_IMPACT_ANALYSIS`.  Once
inserted, an analysis row is immutable.  No UPDATE, DELETE, or REPLACE is
ever performed on this table by the repository layer.

### `fail_job` safe error write

`fail_job(job_id, attempt_id, error, finished_at)` executes one transaction with
two conditional writes:

1. UPDATE `ECN_IMPACT_LLM_ATTEMPT` where `ATTEMPT_ID = :ATTEMPT_ID AND JOB_ID = :JOB_ID
   AND PROVIDER_OUTCOME != 'STARTED'` — sets `VALIDATION_OUTCOME` to
   `error.validation_outcome`.
2. UPDATE `ECN_IMPACT_ANALYSIS_JOB` where `JOB_ID = :JOB_ID AND STATE='RUNNING'` —
   sets `STATE='FAILED'` with the sanitized error code and message.

Error classification:

| Scenario | `ValidationOutcome` |
|---|---|
| Provider returned SUCCESS; response failed schema/evidence validation | `INVALID` |
| Provider returned NOT_CALLED, ERROR, or TIMEOUT | `NOT_RUN` |

Error text sanitization: all C0 control characters (except `\t`, `\n`, `\r`),
DEL, and the C1 block are stripped.  Codes are capped at 80 characters;
messages at 2000 characters.  Source or provider exception text is never accepted
as a public message.

---

## Retry and Restart Recovery

### Discriminated retry result

`queue_failed_job(ecn_row_id)` returns a `RetryQueueResult` with a
`RetryQueueStatus` enum that is never overloaded:

| Status | When | `job_id` |
|---|---|---|
| `ECN_NOT_FOUND` | No ECN row exists for `ecn_row_id` | `None` |
| `NOT_RETRYABLE` | ECN exists but has no job, job is not FAILED, or analysis exists | existing job ID or `None` |
| `QUEUED` | FAILED→QUEUED UPDATE succeeded | the re-queued job ID |

The SELECT starts at `ECN_IMPACT_ECN` and LEFT JOINs the canonical job and any
existing analysis.  This means a missing ECN returns `row is None`, while a known
ECN with a wrong-state job returns a non-null row that fails the `STATE == 'FAILED'`
and `ANALYSIS_ID IS NULL` checks.  All known-ECN conflicts map to `NOT_RETRYABLE`.

The FAILED→QUEUED UPDATE preserves all pinned fields (PROMPT_VERSION,
OUTPUT_SCHEMA_VERSION, MODEL_NAME, MODEL_CONFIG_JSON, ATTEMPT_COUNT) and clears
only the state, timestamps, and error columns.

### Mutually exclusive progress counts

`get_progress(upload_id=None)` returns an `AnalysisProgressResponse` with five
mutually exclusive state counts derived from a single aggregate SELECT:

| Field | Condition |
|---|---|
| `queued_count` | `STATE = 'QUEUED'` |
| `running_count` | `STATE = 'RUNNING'` |
| `completed_count` | `STATE = 'COMPLETED' AND COMPLETENESS_STATE = 'COMPLETED'` |
| `completed_with_gaps_count` | `STATE = 'COMPLETED' AND COMPLETENESS_STATE = 'COMPLETED_WITH_GAPS'` |
| `failed_count` | `STATE = 'FAILED'` |
| `unfinished_count` | Derived = `queued_count + running_count` |
| `total_count` | Derived = sum of all five state counts |

The `AnalysisProgressResponse` Pydantic model validates that `unfinished_count`
and `total_count` are consistent with the five state counts.  The model is
`frozen=True`.

When `upload_id` is supplied, the ECN table is INNER-JOINed and a WHERE clause
filters by `UPLOAD_ID`, restricting all five counts to jobs originating from
that upload batch.

### Restart recovery preserves finished provider telemetry

`fail_interrupted_jobs(finished_at)` executes one transaction with two writes:

1. **Attempt UPDATE** — targets only rows where
   `PROVIDER_OUTCOME='STARTED' AND FINISHED_AT IS NULL`:
   - Sets `PROVIDER_OUTCOME='INTERRUPTED'`, `VALIDATION_OUTCOME='NOT_RUN'`,
     `FINISHED_AT=:finished_at`.
   - **All six token/usage columns are intentionally excluded from the SET
     clause.** Attempts that already received a terminal provider outcome from
     `record_provider_outcome` have `FINISHED_AT IS NOT NULL` and are excluded
     by the WHERE predicate; their usage columns are preserved untouched.

2. **Job UPDATE** — targets all jobs where `STATE IN ('QUEUED', 'RUNNING')`:
   - Sets `STATE='FAILED'`, `FINISHED_AT=:finished_at`,
     `ERROR_CODE='PROCESS_RESTARTED'`.
   - Completed jobs and their analyses are untouched.

Returns the rowcount from the job UPDATE (number of jobs transitioned to FAILED).

---

## History Reset

### Guarded dependency-ordered deletion

`HistoryRepository.clear_history_if_idle()` in
`api/app/persistence/history_repository.py` performs a two-phase operation inside
a single `engine.begin()` transaction:

**Phase 1 — active-job check**: counts QUEUED and RUNNING jobs.  If the count is
non-zero, returns `None` immediately without issuing any DELETE.

**Phase 2 — ordered deletion** (only when count is zero):

| Step | Table | Condition |
|---|---|---|
| 1 | `ECN_IMPACT_ANALYSIS` | All rows |
| 2 | `ECN_IMPACT_LLM_ATTEMPT` | All rows |
| 3 | `ECN_IMPACT_ANALYSIS_JOB` | All rows |
| 4 | `ECN_IMPACT_ECN` | All rows |
| 5 | `ECN_IMPACT_UPLOAD` | All rows |
| 6 | `ECN_IMPACT_BOM_ROW` | All rows |
| 7 | `ECN_IMPACT_SOURCE_ARTIFACT` | Only `ARTIFACT_KIND IN ('ECN_JSON', 'BOM_WORKBOOK')` |

This FK-safe reverse-dependency order ensures no foreign-key violations occur.

### Conditional-rule preservation

`CONDITIONAL_RULE_WORKBOOK` source artifacts are always preserved across history
resets.  The final DELETE filters by `ARTIFACT_KIND IN (:ecn_kind, :bom_kind)`
with bound params `ecn_kind='ECN_JSON'` and `bom_kind='BOM_WORKBOOK'`.  No
`CONDITIONAL_RULE_WORKBOOK` value is ever passed as a parameter.

### No DROP or TRUNCATE

The repository never issues `DROP TABLE` or `TRUNCATE TABLE`.  All seven table
definitions are preserved across history resets so that the application can
resume normal operation without re-running `ensure_schema`.

### Returns HistoryDeletionCounts or None

`clear_history_if_idle` returns a `HistoryDeletionCounts` frozen dataclass with
per-table row counts when deletion succeeded, or `None` when active work blocked
deletion:

```python
@dataclass(frozen=True)
class HistoryDeletionCounts:
    analyses: int
    attempts: int
    jobs: int
    ecns: int
    uploads: int
    bom_rows: int
    source_artifacts: int
```

### Process-guard caller requirement

The database-level active-job check inside the transaction provides only a
database-level race guard.  The calling code must serialize
`clear_history_if_idle` against upload publication via a process-wide mutex or
equivalent mechanism.  Without this serialization a concurrent publication may
create a new upload between the active-job check and the first DELETE, leaving
the newly published data orphaned.
