# HANA connection and schema override

## What it is

The API stores its workspace (uploads, source rows, runs, exports) in SAP HANA Cloud
tables named `RECEIVABLES_*`. On startup `ensure_schema()` creates any missing table.

HANA resolves unqualified table names against `CURRENT_SCHEMA`, which by default is the
**login user's own schema**. The optional `HANA_SCHEMA` setting decouples the two: the app
logs in as `HANA_USER` but reads and writes tables in `HANA_SCHEMA`. This lets you switch
the database user (for example after a password expiry) without moving the data.

## Configuration

| Variable        | Required | Default | Meaning                                                    |
|-----------------|----------|---------|------------------------------------------------------------|
| `HANA_ADDRESS`  | yes      | -       | HANA Cloud host. If empty, the backend falls back to SQLite (tests only). |
| `HANA_PORT`     | no       | `443`   | SQL port.                                                  |
| `HANA_USER`     | yes      | -       | Login user.                                                |
| `HANA_PASSWORD` | yes      | -       | Login password.                                            |
| `HANA_ENCRYPT`  | no       | `true`  | TLS (`true`/`1`/`yes`).                                    |
| `HANA_SCHEMA`   | no       | empty   | Schema holding the app tables. Empty = `HANA_USER`'s own schema. |

- Local: set the values in `api/.env` (template: `api/.env.example`).
- Cloud Foundry: `deploy.sh` reads the same names from `api/.env` and passes them to
  `cf push` as the vars `hana_address`, `hana_port`, `hana_user`, `hana_password`,
  `hana_encrypt` and `hana_schema`, which `manifest.yaml` maps back to the `HANA_*` env vars.

## Related files

- `api/app/services/database/backend.py`: `detect_backend()` and
  `DatabaseBackend._create_hana_connection()`, the only HANA connection site. All pooled
  connections are created there.
- `api/app/services/workspace/schema.py`: `ensure_schema()` creates and validates tables.
  Its metadata queries filter on `SCHEMA_NAME = CURRENT_SCHEMA`, so they follow `HANA_SCHEMA`.
- `manifest.yaml`, `deploy.sh`, `api/.env.example`.

Key snippet (`backend.py`):

```python
schema = os.getenv("HANA_SCHEMA", "").strip()
conn = dbapi.connect(
    address=address, port=port, user=user, password=password, encrypt=encrypt,
    # Optional schema override; empty keeps the login user's default schema.
    currentSchema=schema,
)
```

## Required privileges

`HANA_USER` needs on `HANA_SCHEMA`: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and, because
tables are auto-created, `CREATE ANY` (plus `ALTER`/`DROP` for migrations).

## Troubleshooting: connected but empty workspace

If the app starts but shows no saved analyses after a credential change, it is most likely
working in the wrong schema (and has silently created empty tables there). Check:

```sql
SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY;
SELECT SCHEMA_NAME, COUNT(*) FROM TABLES WHERE TABLE_NAME LIKE 'RECEIVABLES_%' GROUP BY SCHEMA_NAME;
```

Set `HANA_SCHEMA` to the schema that holds the data. Login error code 10 means the
credentials are wrong, expired or locked, which is not a schema problem.

## How to test

```bash
cd api
python - <<'PY'
from dotenv import load_dotenv
load_dotenv(".env")
from app.services.database.backend import get_backend

backend = get_backend()
with backend.get_connection() as conn:
    cur = conn.cursor()
    cur.execute("SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY")
    print(cur.fetchone())
    print("RECEIVABLES_ANALYSES visible:", backend.table_exists(conn, "RECEIVABLES_ANALYSES"))
PY
```

The second value must equal `HANA_SCHEMA` (or the login user when it is empty).
