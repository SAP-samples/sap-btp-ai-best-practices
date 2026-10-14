# HANA DDL and DML Patterns

Use this reference when the generated project writes SAP HANA SQL directly, uses `hdbcli`, or uses `sqlalchemy-hana`.

## Table of Contents

- Dialect boundary
- Schema and identifier handling
- Generated app table names and table contracts
- Idempotent object creation
- Initialization timing for generated apps
- Spreadsheet-backed submission forms
- Inserts, updates, upserts, and merges
- Data type choices
- Transactions and DDL
- Cloud Foundry checks
- Failure signatures

## Dialect Boundary

The SAP HANA database SQL reference is the source of truth for HANA Cloud database apps. Do not copy syntax from:

- PostgreSQL, SQLite, MySQL, DuckDB, or generic SQL examples.
- SAP HANA Cloud data lake relational engine pages, unless the target is actually data lake.
- SAP CAP/CDS examples, unless the generated app is using CAP/CDS and deploys those artifacts.

Concrete example: HANA Cloud data lake relational engine documents `CREATE TABLE [ IF NOT EXISTS ]`, but SAP HANA database `CREATE TABLE` syntax does not include that clause. A HANA Cloud database app that executes `CREATE TABLE IF NOT EXISTS` fails with a syntax error near `IF`.

## Schema and Identifier Handling

Rules:

- Unquoted identifiers are normalized by HANA. Prefer uppercase ASCII object names such as `APP_SUBMISSIONS`.
- Quoted identifiers are case-sensitive and must be quoted every time. Avoid mixed-case quoted object names unless required.
- Reserved words must not be used as identifiers unless quoted. Prefer renaming over quoting.
- Value parameters can be bound. Schema, table, and column names cannot.
- User-provided strings must never become identifiers without allowlist validation.

Identifier helper:

```python
import re

_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,126}$")

def quote_ident(identifier: str) -> str:
    normalized = identifier.strip().upper()
    if not _IDENT_RE.fullmatch(normalized):
        raise ValueError(f"Unsafe HANA identifier: {identifier!r}")
    return f'"{normalized}"'

def qname(schema: str | None, table: str) -> str:
    if schema:
        return f"{quote_ident(schema)}.{quote_ident(table)}"
    return quote_ident(table)
```

Use the runtime user's default schema by default:

```python
with engine.begin() as conn:
    current = conn.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY")).scalar()
    logger.info("Using HANA current schema %s", current)
```

Do not read `HANA_SCHEMA` and automatically apply it. If the user explicitly requests a named schema and privileges/provisioning are verified, set it consistently for every pooled connection or fully qualify object names:

```python
from sqlalchemy import event, text

schema = os.environ["HANA_SCHEMA"]

@event.listens_for(engine, "connect")
def set_hana_schema(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"SET SCHEMA {quote_ident(schema)}")
    finally:
        cursor.close()
```

hdbcli supports the `currentSchema` connection property. If using SQLAlchemy `connect_args`, verify with `SELECT CURRENT_SCHEMA FROM DUMMY` after connection creation.

### Default User Schema Policy

Generated apps should use the runtime user's default schema unless the user explicitly asks for a separate schema. The runtime user may have `CREATE TABLE` in its own default schema even when it lacks `CREATE TABLE` in the injected `HANA_SCHEMA`.

Default behavior:

- Omit `HANA_SCHEMA` / `currentSchema` / `SET SCHEMA` in generated app code.
- Verify `SELECT CURRENT_SCHEMA FROM DUMMY`; this will often equal `CURRENT_USER`.
- Document that application tables are stored in the runtime user's default schema.
- If the app performs runtime DDL, create a uniquely named probe table, insert one row, drop the probe table, and verify it is absent from `SYS.TABLES` before signing off.
- Do not switch to a configured schema just because `HANA_SCHEMA` appears in the deployment environment.

Minimal probe shape:

```sql
SELECT CURRENT_SCHEMA FROM DUMMY;

CREATE COLUMN TABLE "APP_DDL_PROBE_<RANDOM>" (
  ID INTEGER NOT NULL,
  NOTE NVARCHAR(50)
);

INSERT INTO "APP_DDL_PROBE_<RANDOM>" (ID, NOTE)
VALUES (1, 'probe');

DROP TABLE "APP_DDL_PROBE_<RANDOM>";
```

If the probe succeeds, runtime DDL is possible in the default schema. If the user explicitly requires a named schema, use a proper migration/provisioning step or verify DDL privileges in that schema before delivery.

## Generated App Table Names and Table Contracts

The runtime user's default schema can be shared by many generated apps and retries. A generic table name such as `APP_SUBMISSIONS` may already exist from an older app with a different schema. Existence-only idempotency is therefore unsafe.

Default behavior for generated apps:

- Use a stable project-specific table name, for example `APP_PROJECT_SUBMISSIONS`, derived from the app slug or generated project ID.
- Keep the name deterministic across restarts of the same app, but unique across different generated apps.
- If a shared table is explicitly required, treat its schema as an external contract and validate it before using it.
- If an existing table has missing or incompatible columns, fail startup/readiness with a clear message or run an intentional migration. Do not allow the first user request to discover the mismatch.

Minimal contract check:

```python
from sqlalchemy import text

EXPECTED_SUBMISSION_COLUMNS = {
    "ID": ("BIGINT", "FALSE"),
    "COLUMN_A": ("NVARCHAR", "FALSE"),
    "COLUMN_B": ("NVARCHAR", "FALSE"),
    "DESCRIPTION_TEXT": ("NCLOB", "TRUE"),
    "SUBMITTED_AT": ("TIMESTAMP", "FALSE"),
}

def load_table_columns(conn, schema: str, table: str) -> dict[str, tuple[str, str]]:
    rows = conn.execute(
        text("""
            SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = :schema
              AND TABLE_NAME = :table
        """),
        {"schema": schema.upper(), "table": table.upper()},
    ).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}

def assert_required_columns(conn, schema: str, table: str) -> None:
    actual = load_table_columns(conn, schema, table)
    problems = []
    for name, expected in EXPECTED_SUBMISSION_COLUMNS.items():
        if actual.get(name) != expected:
            problems.append(f"{name}: expected {expected}, got {actual.get(name)}")
    if problems:
        raise RuntimeError(
            f"HANA table {schema}.{table} exists with incompatible contract: "
            + "; ".join(problems)
        )
```

For identity columns, primary keys, unique constraints, indexes, and defaults, add catalog checks or a small transactional probe when the app depends on those behaviors.

## Idempotent Object Creation

### Preferred: Catalog Check Plus Create

Use catalog checks for generated application startup or migration code:

```python
from sqlalchemy import text

def table_exists(conn, schema: str, table: str) -> bool:
    return bool(conn.execute(
        text("""
            SELECT COUNT(*)
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = :schema
              AND TABLE_NAME = :table
        """),
        {"schema": schema.upper(), "table": table.upper()},
    ).scalar())

def ensure_submissions_table(conn, schema: str | None, table_name: str) -> None:
    table = qname(schema, table_name)
    lookup_schema = schema or None

    if not lookup_schema:
        lookup_schema = conn.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY")).scalar()

    if table_exists(conn, lookup_schema, table_name):
        assert_required_columns(conn, lookup_schema, table_name)
        return

    conn.execute(text(f"""
        CREATE COLUMN TABLE {table} (
          ID BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
          COLUMN_A NVARCHAR(200) NOT NULL,
          COLUMN_B NVARCHAR(500) NOT NULL,
          DESCRIPTION_TEXT NCLOB,
          SUBMITTED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
        )
    """))
```

Notes:

- `SYS.TABLES` is filtered by privileges. If the user cannot see a table they should own, the privilege model is already a deployment problem.
- If object names are quoted mixed-case, catalog checks must use the exact object name, not uppercase.
- Race conditions are possible if multiple instances create the same object at the same time. For generated apps, prefer a deployment/startup migration before scaling above one instance. If concurrent startup is expected, catch the "already exists" exception around `CREATE TABLE`.

### SQLScript Exception Handler Alternative

Use this mostly in migration scripts, not inside hot request handlers:

```sql
DO
BEGIN
  DECLARE EXIT HANDLER FOR SQLEXCEPTION
  BEGIN
    IF ::SQL_ERROR_CODE <> 288 THEN
      RESIGNAL;
    END IF;
  END;

  CREATE COLUMN TABLE APP_SUBMISSIONS (
    ID BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    COLUMN_A NVARCHAR(500) NOT NULL,
    SUBMITTED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
  );
END;
```

The SAP docs show error-code-based exception handling in anonymous blocks. Verify the exact code for the object operation if the handler is meant to ignore only one expected condition. Catalog checks are easier to reason about in generated app code.

### Columns and Indexes

Do not assume `ADD COLUMN IF NOT EXISTS` or `CREATE INDEX IF NOT EXISTS`. Check the relevant system view first:

```sql
SELECT COUNT(*)
FROM SYS.TABLE_COLUMNS
WHERE SCHEMA_NAME = ?
  AND TABLE_NAME = ?
  AND COLUMN_NAME = ?;
```

```sql
SELECT COUNT(*)
FROM SYS.INDEXES
WHERE SCHEMA_NAME = ?
  AND TABLE_NAME = ?
  AND INDEX_NAME = ?;
```

Then run HANA `ALTER TABLE ... ADD (...)` or `CREATE INDEX ...` only when needed.

## Initialization Timing for Generated Apps

Required table creation and table-contract validation must be guaranteed before the first user write. Do not put the only call to `ensure_table()` inside `/api/readiness`.

Why: Cloud Foundry can route traffic once the process is up, and generated UIs may call lookup endpoints and then submit without ever calling `/api/readiness`. If submit is the first HANA-touching request, readiness-only DDL leaves the app inserting into a missing table.

Preferred order:

1. Define a single `ensure_hana_ready()` function that opens a HANA transaction, reads `CURRENT_SCHEMA`, checks `SYS.TABLES`, validates `SYS.TABLE_COLUMNS`, and creates/migrates only when needed.
2. Call it from FastAPI lifespan/startup or an explicit deployment migration before accepting user writes.
3. Let `/api/readiness` call the same function or a cheaper verification path, but never make readiness the only initializer.
4. If the app intentionally starts when HANA is temporarily unavailable, call the same cached guard before `INSERT` and return 503 if initialization still fails.

FastAPI shape:

```python
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI

_hana_ready = False
_hana_ready_lock = Lock()

def ensure_hana_ready() -> None:
    global _hana_ready
    if _hana_ready:
        return
    with _hana_ready_lock:
        if _hana_ready:
            return
        with get_engine().begin() as conn:
            ensure_submissions_table(
                conn, None, "APP_PROJECT_SUBMISSIONS"
            )
        _hana_ready = True

@asynccontextmanager
async def lifespan(app):
    ensure_hana_ready()
    yield

app = FastAPI(lifespan=lifespan)

def insert_submission(payload: dict) -> dict:
    ensure_hana_ready()
    with get_engine().begin() as conn:
        conn.execute(INSERT_SUBMISSION, payload)
        row = conn.execute(text("""
            SELECT CURRENT_IDENTITY_VALUE() AS ID,
                   CURRENT_UTCTIMESTAMP AS SUBMITTED_AT
            FROM DUMMY
        """)).fetchone()
```

Notes:

- The guard is process-local. With multiple CF instances, each instance may run the catalog check once; the DDL remains idempotent.
- If concurrent instance startup is possible, catch the expected "table already exists" race around `CREATE COLUMN TABLE`, then re-run contract validation.
- After `_hana_ready` is true, submit handlers should be DML-only.
- Health endpoints should remain lightweight. Readiness endpoints should report HANA initialization failures, but they are not a substitute for startup/deployment initialization.

## Spreadsheet-Backed Submission Forms

Use this when a prompt asks for dropdowns populated from an uploaded spreadsheet and a submit action persisted to SAP HANA.

Keep the two data sources separate:

- Spreadsheet lookup data remains in `api/data`, is read with `openpyxl` at API startup or on the first lookup request, and is cached in memory.
- HANA stores only durable user-created submissions, not copied spreadsheet lookup rows.
- The generated API must include `openpyxl` in `api/requirements.txt` when runtime code reads `.xlsx` files.
- Actual workbook headers should be mapped to role names in code constants; examples in this skill use generic roles such as `COLUMN_A_SOURCE` and `COLUMN_B_SOURCE`.

Recommended API shape:

- `GET /api/form/column-a-options` returns distinct normalized non-empty values from the primary workbook column.
- `GET /api/form/column-b-options?column_a=...` returns distinct normalized non-empty dependent values filtered by the selected primary value.
- `POST /api/form/submit` validates the submitted primary/dependent values against the cached lookup map, then inserts only the submitted record into HANA.

Recommended HANA contract:

```python
SUBMISSIONS_TABLE = "APP_PROJECT_SUBMISSIONS"
EXPECTED_SUBMISSION_COLUMNS = {
    "ID": ("BIGINT", "FALSE"),
    "COLUMN_A": ("NVARCHAR", "FALSE"),
    "COLUMN_B": ("NVARCHAR", "FALSE"),
    "DESCRIPTION_TEXT": ("NCLOB", "TRUE"),
    "SUBMITTED_AT": ("TIMESTAMP", "FALSE"),
}
```

Recommended table:

```sql
CREATE COLUMN TABLE "APP_PROJECT_SUBMISSIONS" (
  ID BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  COLUMN_A NVARCHAR(200) NOT NULL,
  COLUMN_B NVARCHAR(500) NOT NULL,
  DESCRIPTION_TEXT NCLOB,
  SUBMITTED_AT TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
);
```

Recommended insert:

```python
INSERT_SUBMISSION = text("""
    INSERT INTO "APP_PROJECT_SUBMISSIONS"
      (COLUMN_A, COLUMN_B, DESCRIPTION_TEXT, SUBMITTED_AT)
    VALUES
      (:column_a, :column_b, :description_text, CURRENT_UTCTIMESTAMP)
""")
```

Delivery checks:

- `ensure_hana_ready()` runs from startup/lifespan, deployment migration, or a cached guard before submit.
- `ensure_hana_ready()` checks `CURRENT_SCHEMA`, `SYS.TABLES`, and `SYS.TABLE_COLUMNS`; it does not accept existence-only success.
- Submit handlers call `ensure_hana_ready()` before the insert when startup initialization is not guaranteed.
- Lookup endpoints work without HANA, but submit must fail with a 503 and full server-side logs if HANA is unavailable.
- No submission path writes local JSON, CSV, Excel, SQLite, or temporary files as persistence.

## Inserts, Updates, Upserts, and Merges

Always include target columns:

```python
INSERT_SUBMISSION = text("""
    INSERT INTO "APP_PROJECT_SUBMISSIONS"
      (COLUMN_A, COLUMN_B, DESCRIPTION_TEXT, SUBMITTED_AT)
    VALUES
      (:column_a, :column_b, :description_text, CURRENT_UTCTIMESTAMP)
""")
```

Do not interpolate values:

```python
# WRONG
conn.execute(text(f"INSERT INTO USERS (EMAIL) VALUES ('{email}')"))

# CORRECT
conn.execute(text("INSERT INTO USERS (EMAIL) VALUES (:email)"), {"email": email})
```

Use `UPSERT` for primary-key based upserts:

```sql
UPSERT APP_SUBMISSIONS
  (ID, COLUMN_A, DESCRIPTION_TEXT, SUBMITTED_AT)
VALUES
  (?, ?, ?, CURRENT_UTCTIMESTAMP)
WITH PRIMARY KEY;
```

Use `MERGE INTO` when the match condition is more complex or source data is a query:

```sql
MERGE INTO CUSTOMER_ASSIGNMENTS AS target
USING (
  SELECT ? AS CUSTOMER_ID, ? AS OWNER_ID FROM DUMMY
) AS source
ON target.CUSTOMER_ID = source.CUSTOMER_ID
WHEN MATCHED THEN
  UPDATE SET OWNER_ID = source.OWNER_ID
WHEN NOT MATCHED THEN
  INSERT (CUSTOMER_ID, OWNER_ID)
  VALUES (source.CUSTOMER_ID, source.OWNER_ID);
```

Do not use PostgreSQL `ON CONFLICT`, MySQL `ON DUPLICATE KEY`, or SQLite `INSERT OR REPLACE`.

### Generated Identity Values

Do not use `IDENTITY_CURRENT()` or `IDENTITY_CURRENT('TABLE')`. On SAP HANA database, that generic function name fails with `invalid name of function or procedure: IDENTITY_CURRENT`.

Use `CURRENT_IDENTITY_VALUE()` immediately after the insert on the same connection/session:

```python
with engine.begin() as conn:
    conn.execute(
        text("""
            INSERT INTO "APP_PROJECT_SUBMISSIONS"
              (COLUMN_A, COLUMN_B, DESCRIPTION_TEXT)
            VALUES
              (:column_a, :column_b, :description_text)
        """),
        params,
    )
    row = conn.execute(text("""
        SELECT CURRENT_IDENTITY_VALUE() AS ID,
               CURRENT_UTCTIMESTAMP AS SUBMITTED_AT
        FROM DUMMY
    """)).fetchone()
```

Rules:

- The lookup must run on the same connection/session as the insert. With SQLAlchemy, keep it inside the same `engine.begin()` block.
- `CURRENT_IDENTITY_VALUE()` is session-scoped, not table-name-scoped. If triggers or other logic can insert into another identity table in the same session before the lookup, do not rely on it.
- If ambiguity matters, generate the key in the application (`NVARCHAR(36)` UUID/ULID or `VARBINARY(16)`) and insert it explicitly, or use a project-reviewed HANA sequence pattern.
- If the API returns `SUBMITTED_AT`, either set it in the application and insert that exact value, or select the inserted row by the generated/application key after insert.

## Data Type Choices

Use deliberate HANA types:

| Need | HANA type |
| --- | --- |
| Surrogate integer key | `BIGINT GENERATED BY DEFAULT AS IDENTITY` |
| UI/user text up to known limit | `NVARCHAR(n)` |
| Long text | `NCLOB` |
| Money/exact decimal | `DECIMAL(p,s)` |
| Approximate measurement | `DOUBLE` or `REAL` |
| Timestamp audit | `TIMESTAMP` with `CURRENT_UTCTIMESTAMP` or app UTC value |
| JSON payload | `NCLOB` with application JSON validation unless a project-specific HANA JSON feature is verified |
| UUID | `NVARCHAR(36)` or `VARBINARY(16)`, chosen consistently |

Avoid unsupported or foreign-dialect types such as `TEXT`, `VARCHAR(MAX)`, `JSONB`, `SERIAL`, `BIGSERIAL`, and `DATETIME2`.

## Transactions and DDL

HANA has separate behavior for client autocommit and DDL autocommit:

- hdbcli autocommit is enabled by default.
- HANA `SET TRANSACTION AUTOCOMMIT DDL` controls DDL-specific auto-commit behavior, and the documented default is `ON` for sessions.
- DDL failure can leave the app in a partially initialized state if schema setup is mixed with user request DML.

Best practice for generated apps:

- Keep schema creation/migration in startup/lifespan or deployment migration when feasible.
- If startup migration is not guaranteed before user traffic, submit handlers must call a cached one-time initialization guard before first DML.
- Use `engine.begin()` for DML units of work.
- Do not rely on rollback to undo DDL unless you explicitly configured and verified DDL autocommit behavior.
- If startup creates tables, fail readiness clearly when migration fails. Do not let the first user submission discover syntax errors or missing tables.

## Cloud Foundry Checks

For HANA-backed Cloud Foundry apps:

- Manifest must provide `HANA_ADDRESS`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD`, and `HANA_ENCRYPT`. `HANA_SCHEMA` is optional and should not be used by default in generated app code.
- HANA Cloud commonly uses port `443` and encryption.
- `VCAP_SERVICES` may be empty when credentials are injected as user-provided env vars. Do not require service binding if the host flow supplies env vars.
- Never log passwords, full SQLAlchemy URLs, or raw env dumps.
- Readiness should verify local required files and, if the feature depends on HANA being ready, a cheap HANA check such as `SELECT 1 FROM DUMMY`, a schema check, and the required table contract when the table already exists.
- Do not assume CF, the UI, or the app host will call `/api/readiness` before the first user submit. Startup/lifespan, deployment migration, or a cached submit guard must guarantee initialization.
- If `HANA_SCHEMA` is present but the project did not explicitly require it, ignore it and use the runtime user's `CURRENT_SCHEMA`. If the project explicitly requires `HANA_SCHEMA`, document one of: pre-provisioned table, granted DDL privilege, or a failed privilege check.

## Failure Signatures

Use these to diagnose deployed failures quickly:

| Log/Error | Likely cause |
| --- | --- |
| `sql syntax error: incorrect syntax near "IF"` | Generic `IF NOT EXISTS` / `IF EXISTS` DDL was sent to HANA database. |
| `invalid table name` | Table was not created, wrong schema/current schema, quoted-name case mismatch, or insufficient object visibility. |
| `invalid table name` on the first `POST /submit` for a project-specific table | Required table creation was placed only in readiness, and readiness was not called before user traffic. Move initialization to startup/deploy migration or add a cached ensure guard before insert. |
| `invalid column name: COLUMN_A` or another expected app column | The table name exists but points to a stale/shared table with a different contract. Use app-specific table names and validate `SYS.TABLE_COLUMNS` before skipping DDL. |
| `invalid name of function or procedure: IDENTITY_CURRENT` | Generic identity lookup was copied from another dialect. Use SAP HANA `CURRENT_IDENTITY_VALUE()` on the same connection/session, or generate keys in the app. |
| `insufficient privilege` | User lacks schema/table/create/alter/drop privilege. Do not mask as a generic network error. |
| `insufficient privilege` on `CREATE COLUMN TABLE` after `SET SCHEMA` | Runtime user can connect but cannot create objects in the configured schema. Generated apps should not set schema by default; use the runtime user's current schema unless a named schema is explicitly required and provisioned. |
| `feature not supported` | SQL copied from another dialect or unsupported HANA edition/context. |
| `cannot use duplicate table name` or table already exists code | Missing idempotency or concurrent create. |
| App returns HTTP 200 with `{ok:false}` | API may be hiding database failure from HTTP status. Check server logs and consider returning a 5xx for failed persistence. |

## Source Anchors

- SAP HANA database `CREATE TABLE` statement: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/create-table-statement-data-definition
- SAP HANA data lake relational engine `CREATE TABLE` statement with `IF NOT EXISTS` syntax: https://help.sap.com/docs/hana-cloud-data-lake/administration-guide-for-data-lake-relational-engine/6c3afae6093f4327a6aa7fb91c1caafe.html
- SAP HANA SQLScript anonymous block and exception handler examples: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-sqlscript-reference/anonymous-block
- SAP HANA `SYS.TABLES` and `SYS.TABLE_COLUMNS` system views: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/tables-system-view
- SAP HANA hdbcli connection properties, including `currentSchema`: https://help.sap.com/docs/SAP_HANA_CLIENT/f1b440ded6144a54ada97ff95dac7adf/ee592e89dcce4480a99571a4ae7a702f.html
- SAP HANA `INSERT` statement guidance to use column lists: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/insert-statement-data-manipulation
- SAP HANA `CURRENT_IDENTITY_VALUE()` function: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/current-identity-value-function-miscellaneous
- SAP HANA `UPSERT` and `MERGE INTO`: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/upsert-statement-data-manipulation and https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/merge-into-statement-data-manipulation
- SAP HANA `SET TRANSACTION AUTOCOMMIT DDL`: https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-sql-reference-guide/set-transaction-autocommit-ddl-statement-transaction-management
