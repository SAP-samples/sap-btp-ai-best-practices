"""Transactional HANA repository; MemoryRepository is a deterministic test double.

Analytical rows occupy registered relational tables. JSON is confined to entity
configuration, nested evidence, and artifacts, never used as the query data model.
"""

from __future__ import annotations
import copy
import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import INPUT_VIEWS, RESULT_VIEWS

CATALOG = json.loads(Path(__file__).with_name("columns.json").read_text())
KINDS = {
    "ai_settings": "AI_SETTINGS",
    "profiles": "PLANT_PROFILES",
    "constraint_code_jobs": "CONSTRAINT_CODE_JOBS",
    "datasets": "DATASETS",
    "drafts": "RUN_DRAFTS",
    "runs": "RUNS",
    "contexts": "CONTEXTS",
}
PREFIX = "PRODUCTION_WHEEL_"
STORAGE_VIEWS = INPUT_VIEWS | RESULT_VIEWS | {"plant_profile_matrix"}


def identifier(name: str) -> str:
    """Quote an application-owned SQL identifier after strict validation."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("invalid column identifier")
    return '"' + name.upper() + '"'


def connect_hana():
    """Open the runtime user's default schema, using millisecond connection timeout."""
    from dotenv import load_dotenv
    from hdbcli import dbapi

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    return dbapi.connect(
        address=os.environ["HANA_ADDRESS"],
        port=int(os.getenv("HANA_PORT", "443")),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        encrypt=os.getenv("HANA_ENCRYPT", "true").lower() == "true",
        connectTimeout=10000,
    )


def _json(value: Any) -> str:
    """Serialize configuration and nested evidence deterministically, rejecting NaN."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False
    )


def _read_filters(view, filters=None):
    """Validate the finite repository predicate vocabulary and typed point value.

    Only a positive integer point_index on catalogued point-scoped result views
    is accepted. Column names and operators are never taken from caller SQL.
    """
    if view not in STORAGE_VIEWS:
        raise ValueError("unknown analytical view")
    filters = dict(filters or {})
    if set(filters) - {"point_index"}:
        raise ValueError("unsupported repository read filter")
    if "point_index" in filters:
        if view not in RESULT_VIEWS or "point_index" not in CATALOG.get(view, {}):
            raise ValueError("view does not support point_index filtering")
        value = filters["point_index"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("point_index filter requires a positive integer")
    return filters


def _matches_read_filters(row, filters):
    """Match memory rows to the same numeric point predicate used by HANA."""
    if not filters:
        return True
    try:
        return float(row.get("point_index")) == filters["point_index"]
    except (TypeError, ValueError):
        return False


class MemoryRepository:
    """In-memory repository used only by unit tests, with rollback and CAS semantics."""

    def __init__(self):
        """Initialize isolated tables, entities, and binary artifacts."""
        self.entities = {}
        self.tables = {}
        self.artifacts = {}
        self.lock = threading.RLock()

    def ensure(self):
        """No setup is required for this test-only repository."""

    @contextmanager
    def transaction(self):
        """Rollback all mutations in a failed transaction."""
        with self.lock:
            old = copy.deepcopy((self.entities, self.tables, self.artifacts))
            try:
                yield self
            except BaseException:
                self.entities, self.tables, self.artifacts = old
                raise

    def insert(self, kind, key, value):
        """Insert a versioned entity, rejecting duplicate identities."""
        with self.lock:
            if (kind, key) in self.entities:
                raise ValueError("duplicate entity")
            self.entities[kind, key] = copy.deepcopy(value)

    def get(self, kind, key):
        """Return a detached entity or raise KeyError for an unknown ID."""
        with self.lock:
            return copy.deepcopy(self.entities[kind, key])

    def lock_entity(self, kind, key):
        """Read while the caller transaction owns the memory repository lock."""
        return self.get(kind, key)

    def list(self, kind, filters=None):
        """List entities matching exact metadata fields, newest first."""
        with self.lock:
            items = [
                copy.deepcopy(v)
                for (k, _), v in self.entities.items()
                if k == kind and all(v.get(f) == x for f, x in (filters or {}).items())
            ]
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def cas(self, kind, key, revision, changes):
        """Atomically patch one entity only when its revision is unchanged."""
        with self.lock:
            value = self.get(kind, key)
            if value.get("revision", 1) != revision:
                raise ValueError("stale revision; reload current state")
            value.update(copy.deepcopy(changes))
            value["revision"] = revision + 1
            self.entities[kind, key] = value
            return copy.deepcopy(value)

    def replace_tables(self, owner_id, tables):
        """Replace a publication's registered relational row collections."""
        for view, rows in tables.items():
            if view not in STORAGE_VIEWS:
                raise ValueError(f"unknown view: {view}")
            self.tables[owner_id, view] = copy.deepcopy(rows)

    def delete_owner(self, kind, key):
        """Permanently delete one entity ID and its owned rows/artifacts; return nothing."""
        with self.transaction():
            self.get(kind, key)
            del self.entities[kind, key]
            self.tables = {k: v for k, v in self.tables.items() if k[0] != key}
            self.artifacts = {k: v for k, v in self.artifacts.items() if k[0] != key}

    def rows(self, owner_id, view, filters=None):
        """Return detached rows after optional point filtering, matching HANA reads."""
        predicates = _read_filters(view, filters)
        with self.lock:
            return copy.deepcopy(
                [
                    row
                    for row in self.tables.get((owner_id, view), [])
                    if _matches_read_filters(row, predicates)
                ]
            )

    def put_artifact(self, owner_id, name, data):
        """Persist an opaque upload, report, or restart checkpoint."""
        self.artifacts[owner_id, name] = bytes(data)

    def artifact(self, owner_id, name):
        """Read an immutable source or a saved report/checkpoint."""
        return self.artifacts[owner_id, name]


class HanaRepository:
    """HANA storage with per-thread transactions and additive catalog migrations."""

    def __init__(self, connection_factory=connect_hana):
        """Initialize connection factory, schema cache, and thread-local transactions."""
        self.factory = connection_factory
        self.local = threading.local()
        self.lock = threading.RLock()
        self.ready = False
        self.specs = {}

    @contextmanager
    def transaction(self):
        """Use one HANA transaction across nested service/repository operations."""
        if getattr(self.local, "connection", None) is not None:
            yield self
            return
        conn = self.factory()
        conn.setautocommit(False)
        self.local.connection = conn
        try:
            yield self
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            self.local.connection = None
            conn.close()

    @contextmanager
    def cursor(self):
        """Open a cursor in the current or a new transaction and always close it."""
        with self.transaction():
            cursor = self.local.connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def _ensure_table(self, name, spec, primary, extension_fields=None):
        """Ensure tables, updating spec with existing types only for uncatalogued fields.

        Extension columns may have been created by earlier numeric inference.
        Preserve supported physical types and their values; declared catalog
        fields still require the explicit schema contract.
        """
        with self.lock:
            extension_fields = extension_fields or {}
            if self.specs.get(name) == spec:
                return
            # DDL uses a separate committed connection: HANA DDL must not commit
            # an in-flight dataset/result publication transaction.
            conn = self.factory()
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME=CURRENT_SCHEMA AND TABLE_NAME=?",
                    (name,),
                )
                existing = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
                if not existing:
                    columns = ", ".join(f"{identifier(k)} {v}" for k, v in spec.items())
                    pk = ", ".join(identifier(k) for k in primary)
                    cur.execute(
                        f"CREATE COLUMN TABLE {identifier(name)} ({columns}, PRIMARY KEY ({pk}))"
                    )
                else:
                    for key, sqltype in spec.items():
                        expected = sqltype.split("(")[0].split()[0]
                        found = existing.get(key.upper())
                        if key in extension_fields and found is not None:
                            if found[0] in ("DOUBLE", "INTEGER", "NCLOB"):
                                sqltype = _extension_type(found[0], extension_fields[key])
                                spec[key] = sqltype
                                if sqltype == found[0]:
                                    continue
                                expected = "NVARCHAR"
                            if found[0] == "NVARCHAR":
                                spec[key] = f"NVARCHAR({int(found[1])})"
                                continue
                        if found is None:
                            cur.execute(
                                f"ALTER TABLE {identifier(name)} ADD ({identifier(key)} {sqltype})"
                            )
                        elif (
                            found[0] in ("DOUBLE", "INTEGER") and expected == "NVARCHAR"
                        ):
                            cur.execute(
                                f"ALTER TABLE {identifier(name)} ALTER ({identifier(key)} {sqltype})"
                            )
                        elif found[0] != expected:
                            raise RuntimeError(
                                f"incompatible HANA column {name}.{key}: {found[0]} vs {expected}"
                            )
                        elif expected == "NVARCHAR" and found[1] < int(
                            re.search(r"\d+", sqltype)[0]
                        ):
                            raise RuntimeError(f"undersized HANA column {name}.{key}")
                conn.commit()
                self.specs[name] = dict(spec)
            finally:
                cur.close()
                conn.close()

    def ensure(self):
        """Create/validate metadata and source-artifact tables before any writes."""
        if self.ready:
            return
        for suffix in KINDS.values():
            self._ensure_table(
                PREFIX + suffix,
                {
                    "id": "NVARCHAR(64)",
                    "revision": "INTEGER",
                    "status": "NVARCHAR(40)",
                    "dataset_id": "NVARCHAR(64)",
                    "name": "NVARCHAR(255)",
                    "created_at": "NVARCHAR(40)",
                    "payload": "NCLOB",
                },
                ["id"],
            )
        self._ensure_table(
            PREFIX + "ARTIFACTS",
            {
                "owner_id": "NVARCHAR(64)",
                "name": "NVARCHAR(255)",
                "sha256": "NVARCHAR(64)",
                "content": "BLOB",
            },
            ["owner_id", "name"],
        )
        from .schema_migrations import migrate_option_metrics

        migrate_option_metrics(self.factory)
        self.ready = True

    def insert(self, kind, key, value):
        """Insert an entity whose searchable identity/state fields are relational."""
        self.ensure()
        table = PREFIX + KINDS[kind]
        with self.cursor() as c:
            c.execute(
                f'INSERT INTO {identifier(table)} ("ID","REVISION","STATUS","DATASET_ID","NAME","CREATED_AT","PAYLOAD") VALUES (?,?,?,?,?,?,?)',
                (
                    key,
                    value.get("revision", 1),
                    value.get("status"),
                    value.get("dataset_id"),
                    value.get("name"),
                    value.get("created_at"),
                    _json(value),
                ),
            )

    def get(self, kind, key):
        """Fetch one metadata/configuration entity by stable ID."""
        self.ensure()
        with self.cursor() as c:
            c.execute(
                f'SELECT "PAYLOAD" FROM {identifier(PREFIX + KINDS[kind])} WHERE "ID"=?',
                (key,),
            )
            row = c.fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def lock_entity(self, kind, key):
        """Lock a metadata row in the caller transaction without changing its revision."""
        self.ensure()
        with self.cursor() as cursor:
            cursor.execute(f'SELECT "PAYLOAD" FROM {identifier(PREFIX + KINDS[kind])} WHERE "ID"=? FOR UPDATE', (key,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def list(self, kind, filters=None):
        """List metadata entities; analytical data is always queried separately."""
        self.ensure()
        with self.cursor() as c:
            c.execute(
                f'SELECT "PAYLOAD" FROM {identifier(PREFIX + KINDS[kind])} ORDER BY "CREATED_AT" DESC'
            )
            rows = c.fetchall()
        return (
            [
                v
                for row in rows
                if all(
                    (v := json.loads(row[0])).get(k) == val
                    for k, val in (filters or {}).items()
                )
            ]
            if filters
            else [json.loads(row[0]) for row in rows]
        )

    def cas(self, kind, key, revision, changes):
        """Compare-and-swap protects drafts and atomically claims queued jobs."""
        value = self.get(kind, key)
        if value.get("revision", 1) != revision:
            raise ValueError("stale revision; reload current state")
        value.update(changes)
        value["revision"] = revision + 1
        with self.cursor() as c:
            c.execute(
                f'UPDATE {identifier(PREFIX + KINDS[kind])} SET "REVISION"=?,"STATUS"=?,"DATASET_ID"=?,"NAME"=?,"PAYLOAD"=? WHERE "ID"=? AND "REVISION"=?',
                (
                    revision + 1,
                    value.get("status"),
                    value.get("dataset_id"),
                    value.get("name"),
                    _json(value),
                    key,
                    revision,
                ),
            )
            if c.rowcount != 1:
                raise ValueError("stale revision; reload current state")
        return value

    def _row_spec(self, view, rows):
        """Use the checked-in field catalog with additive application-owned evidence."""
        spec = {
            "owner_id": "NVARCHAR(64)",
            "row_id": "INTEGER",
            **CATALOG.get(view, {}),
        }
        table = PREFIX + "DATA_" + view.upper()
        extensions = {}
        for row in rows:
            for k, v in row.items():
                identifier(k)
                if k not in CATALOG.get(view, {}) and k not in ("owner_id", "row_id"):
                    extensions.setdefault(k, []).append(v)
                    # Reuse validated physical types; new evidence remains stable
                    # NCLOB regardless of which value or null appears first.
                    spec[k] = self.specs.get(table, {}).get(k, "NCLOB")
        for key, values in extensions.items():
            spec[key] = _extension_type(spec[key], values)
        # Empty new audit tables still exist, so absence is queryable after restart.
        self._ensure_table(
            table, spec, ["owner_id", "row_id"], extension_fields=extensions
        )
        return spec

    def delete_owner(self, kind, key):
        """Delete a catalog entity and its HANA rows/blobs atomically by bound owner ID."""
        self.ensure()
        with self.cursor() as c:
            self.get(kind, key)
            # Some analytical tables may never have been materialized. Discover
            # existing tables, then restrict deletion to registered app views.
            c.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME=CURRENT_SCHEMA")
            existing = {row[0] for row in c.fetchall()}
            for view in sorted(STORAGE_VIEWS):
                table = PREFIX + "DATA_" + view.upper()
                if table in existing:
                    c.execute(f'DELETE FROM {identifier(table)} WHERE "OWNER_ID"=?', (key,))
            c.execute(f'DELETE FROM {identifier(PREFIX + "ARTIFACTS")} WHERE "OWNER_ID"=?', (key,))
            c.execute(f'DELETE FROM {identifier(PREFIX + KINDS[kind])} WHERE "ID"=?', (key,))

    def replace_tables(self, owner_id, tables):
        """Write normalized column rows in batches inside atomic publication."""
        self.ensure()
        specs = {
            view: self._row_spec(view, rows)
            for view, rows in tables.items()
            if view in STORAGE_VIEWS
        }
        if set(specs) != set(tables):
            raise ValueError("unknown analytical view")
        with self.transaction():
            for view, rows in tables.items():
                spec = specs[view]
                keys = list(spec)
                table = identifier(PREFIX + "DATA_" + view.upper())
                with self.cursor() as c:
                    c.execute(f'DELETE FROM {table} WHERE "OWNER_ID"=?', (owner_id,))
                    sql = f"INSERT INTO {table} ({','.join(identifier(k) for k in keys)}) VALUES ({','.join('?' for _ in keys)})"
                    batch = []
                    for i, row in enumerate(rows):
                        values = {"owner_id": owner_id, "row_id": i, **row}
                        batch.append(
                            tuple(_sql_value(values.get(k), spec[k]) for k in keys)
                        )
                        if len(batch) >= 250:
                            c.executemany(sql, batch)
                            batch = []
                    if batch:
                        c.executemany(sql, batch)

    def rows(self, owner_id, view, filters=None):
        """Read owner/point-scoped rows using bound predicates before data transfer."""
        predicates = _read_filters(view, filters)
        table = PREFIX + "DATA_" + view.upper()
        with self.cursor() as c:
            c.execute(
                "SELECT COLUMN_NAME FROM SYS.TABLE_COLUMNS WHERE SCHEMA_NAME=CURRENT_SCHEMA AND TABLE_NAME=? ORDER BY POSITION",
                (table,),
            )
            columns = [r[0].lower() for r in c.fetchall()]
            if not columns:
                return []
            where = '"OWNER_ID"=?'
            parameters = [owner_id]
            if "point_index" in predicates:
                if "point_index" not in columns:
                    raise ValueError("stored view has no point_index column")
                where += ' AND "POINT_INDEX"=?'
                parameters.append(predicates["point_index"])
            c.execute(
                f'SELECT * FROM {identifier(table)} WHERE {where} ORDER BY "ROW_ID"',
                tuple(parameters),
            )
            rows = c.fetchall()
        return [
            {
                k: _read_value(v)
                for k, v in zip(columns, row)
                if k not in ("owner_id", "row_id")
            }
            for row in rows
        ]

    def put_artifact(self, owner_id, name, data):
        """Save source bytes, report text, or a durable publication checkpoint."""
        self.ensure()
        with self.cursor() as c:
            c.execute(
                f'UPSERT {identifier(PREFIX + "ARTIFACTS")} ("OWNER_ID","NAME","SHA256","CONTENT") VALUES (?,?,?,?) WITH PRIMARY KEY',
                (owner_id, name, hashlib.sha256(data).hexdigest(), bytes(data)),
            )

    def artifact(self, owner_id, name):
        """Return verified artifact bytes, detecting storage corruption."""
        self.ensure()
        with self.cursor() as c:
            c.execute(
                f'SELECT "CONTENT","SHA256" FROM {identifier(PREFIX + "ARTIFACTS")} WHERE "OWNER_ID"=? AND "NAME"=?',
                (owner_id, name),
            )
            row = c.fetchone()
        if row is None:
            raise KeyError(name)
        data = bytes(row[0])
        if hashlib.sha256(data).hexdigest() != row[1]:
            raise ValueError("artifact checksum mismatch")
        return data


def _extension_type(sqltype, values):
    """Preserve numeric extension types unless source text requires lossless widening.

    Excel error strings remain source evidence. Numeric-to-NVARCHAR conversion
    follows the existing HANA migration path and retains earlier snapshot values.
    """
    if sqltype in ("DOUBLE", "INTEGER"):
        try:
            for value in values:
                _sql_value(value, sqltype)
        except (ValueError, TypeError, OverflowError):
            return "NVARCHAR(5000)"
    return sqltype


def _sql_value(value, sqltype):
    """Normalize CSV-era scalars without converting material or line identifiers."""
    if value is None or value == "":
        return None
    if sqltype in ("DOUBLE", "INTEGER"):
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return int(value.lower() == "true")
        return float(value) if sqltype == "DOUBLE" else int(value)
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _read_value(value):
    """Decode nested evidence only; identifiers remain strings."""
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value
