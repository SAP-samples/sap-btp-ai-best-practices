"""
Customer registry access for the Payment Advice Extractor (UC-01 / UC-02).

Reads and writes the two configuration tables defined in ``hana_schema``:

    PAYMENT_ADVICE_EXTRACTOR_CUSTOMERS
    PAYMENT_ADVICE_EXTRACTOR_CUSTOMER_SCHEMAS

The ``--client`` CLI argument is normalized to a stable ``client_key`` (see
``normalize_client_key``) and used to look up whether the client is *critical*
(dedicated schema always) and which SAP Document AI schema is bound to it.

Seeding: ``seed_customers`` inserts the critical customers listed in a JSON seed
file (see ``load_seed_customers``) idempotently and never overwrites an edited row.
The packaged ``seed_customers.json`` holds the anonymized demo list; a landscape
with its own customers points ``PAYMENT_ADVICE_SEED_CUSTOMERS_PATH`` at a private
(git-ignored) file in the same format.

Fuzzy search (UC-02): ``find_customers`` runs entirely server-side using HANA's
built-in ``CONTAINS(..., FUZZY(0.7))`` predicate so only the top-N matches are
returned over the wire rather than downloading the full customer list.

Scale upgrade path: if the table grows large, add a HANA full-text index::

    CREATE FULLTEXT INDEX ft_customers_display_name
        ON "PAYMENT_ADVICE_EXTRACTOR_CUSTOMERS"("DISPLAY_NAME")
        FUZZY SEARCH INDEX ON;

This makes CONTAINS/FUZZY use the index rather than a full-table scan.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .hana_schema import CUSTOMERS, CUSTOMER_SCHEMAS


# Packaged seed list (anonymized demo customers). Relative paths given through
# PAYMENT_ADVICE_SEED_CUSTOMERS_PATH resolve against the api/ directory, which is
# also the working directory of the Cloud Foundry app.
_DEFAULT_SEED_PATH = Path(__file__).resolve().parent / "seed_customers.json"
_API_DIR = Path(__file__).resolve().parents[2]


def normalize_client_key(raw: str) -> str:
    """
    Normalize a raw client name/argument into a stable ``client_key``.

    Lowercases, and collapses any run of non-alphanumeric characters into a single
    underscore (trimmed at the ends). Examples:
        "Northwind"              -> "northwind"
        "Acme Hardware Supply"   -> "acme_hardware_supply"

    Args:
        raw: The user-supplied client identifier.

    Returns:
        The normalized client key.

    Raises:
        ValueError: If ``raw`` is empty or normalizes to an empty string.
    """
    if not raw or not raw.strip():
        raise ValueError("client key must be a non-empty string")
    key = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    if not key:
        raise ValueError(f"client key {raw!r} normalizes to an empty string")
    return key


def load_seed_customers(path: str | os.PathLike[str] | None = None) -> tuple[tuple[str, str], ...]:
    """
    Read the critical-customer seed list from a JSON file.

    File format: a list of ``{"client_key": "...", "display_name": "..."}`` objects.
    ``client_key`` must already be normalized (``normalize_client_key``).

    Args:
        path: Explicit file. Falls back to ``PAYMENT_ADVICE_SEED_CUSTOMERS_PATH``
            (relative paths resolve against ``api/``), then the packaged
            ``seed_customers.json``.

    Returns:
        ``(client_key, display_name)`` pairs.

    Raises:
        ValueError: If the file is missing, not valid JSON, or an entry is malformed
            or duplicated. Fails at startup rather than seeding a partial list.
    """
    env_path = os.getenv("PAYMENT_ADVICE_SEED_CUSTOMERS_PATH", "").strip()
    resolved = Path(path) if path else (_API_DIR / env_path if env_path else _DEFAULT_SEED_PATH)
    try:
        entries = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read seed customers from {resolved}: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError(f"{resolved}: expected a JSON list of customers")
    seeds: list[tuple[str, str]] = []
    for entry in entries:
        key = entry.get("client_key") if isinstance(entry, dict) else None
        name = entry.get("display_name") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not isinstance(name, str) or not name.strip():
            raise ValueError(f"{resolved}: each entry needs client_key and display_name, got {entry!r}")
        if key != normalize_client_key(key):
            raise ValueError(f"{resolved}: client_key {key!r} is not normalized "
                             f"(expected {normalize_client_key(key)!r})")
        if key in {k for k, _ in seeds}:
            raise ValueError(f"{resolved}: duplicate client_key {key!r}")
        seeds.append((key, name.strip()))
    return tuple(seeds)


def _upper(row: Any) -> dict:
    """Uppercase a HANA result row's keys (the hdbcli driver returns lowercase)."""
    return {str(key).upper(): value for key, value in row.items()}


@dataclass(frozen=True)
class Customer:
    """One row of PAYMENT_ADVICE_EXTRACTOR_CUSTOMERS."""

    client_key: str
    display_name: str
    is_critical: bool
    status: str


@dataclass(frozen=True)
class CustomerSchema:
    """One row of PAYMENT_ADVICE_EXTRACTOR_CUSTOMER_SCHEMAS."""

    client_key: str
    schema_id: str
    schema_version: str
    is_canonical: bool
    source: str


def seed_customers(
    engine: Engine,
    customers: tuple[tuple[str, str], ...] | None = None,
) -> int:
    """
    Insert any missing critical-customer rows; never overwrite existing ones.

    Args:
        engine: SAP HANA SQLAlchemy engine.
        customers: (client_key, display_name) pairs to seed as critical; defaults
            to ``load_seed_customers()``.

    Returns:
        The number of rows actually inserted.
    """
    if customers is None:
        customers = load_seed_customers()
    inserted = 0
    with engine.begin() as connection:
        for client_key, display_name in customers:
            exists = connection.execute(
                text(f'SELECT COUNT(*) FROM "{CUSTOMERS}" WHERE "CLIENT_KEY" = :ck'),
                {"ck": client_key},
            ).scalar()
            if exists:
                continue
            connection.execute(
                text(
                    f'''INSERT INTO "{CUSTOMERS}"
                        ("CLIENT_KEY", "DISPLAY_NAME", "IS_CRITICAL", "STATUS")
                        VALUES (:ck, :name, TRUE, 'active')'''
                ),
                {"ck": client_key, "name": display_name},
            )
            inserted += 1
    return inserted


def get_customer(engine: Engine, client_key: str) -> Customer | None:
    """Return the customer row for ``client_key``, or ``None`` if not registered."""
    with engine.connect() as connection:
        row = connection.execute(
            text(
                f'''SELECT "CLIENT_KEY", "DISPLAY_NAME", "IS_CRITICAL", "STATUS"
                    FROM "{CUSTOMERS}" WHERE "CLIENT_KEY" = :ck'''
            ),
            {"ck": client_key},
        ).mappings().first()
    if row is None:
        return None
    row = _upper(row)
    return Customer(
        client_key=row["CLIENT_KEY"],
        display_name=row["DISPLAY_NAME"],
        is_critical=bool(row["IS_CRITICAL"]),
        status=row["STATUS"],
    )


def list_customers(engine: Engine) -> list[Customer]:
    """Return all registered customers ordered by client key."""
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f'''SELECT "CLIENT_KEY", "DISPLAY_NAME", "IS_CRITICAL", "STATUS"
                    FROM "{CUSTOMERS}" ORDER BY "CLIENT_KEY"'''
            )
        ).mappings().all()
    customers: list[Customer] = []
    for raw in rows:
        row = _upper(raw)
        customers.append(
            Customer(
                client_key=row["CLIENT_KEY"],
                display_name=row["DISPLAY_NAME"],
                is_critical=bool(row["IS_CRITICAL"]),
                status=row["STATUS"],
            )
        )
    return customers


def set_critical(engine: Engine, client_key: str, is_critical: bool) -> Customer | None:
    """
    Set a customer's criticality flag (promote/demote).

    Returns the updated Customer, or None if the client key is not registered.
    """
    with engine.begin() as connection:
        result = connection.execute(
            text(
                f'''UPDATE "{CUSTOMERS}"
                    SET "IS_CRITICAL" = :flag, "UPDATED_AT" = CURRENT_UTCTIMESTAMP
                    WHERE "CLIENT_KEY" = :ck'''
            ),
            {"flag": is_critical, "ck": client_key},
        )
        if result.rowcount == 0:
            return None
    return get_customer(engine, client_key)


def get_bound_schema(engine: Engine, client_key: str) -> CustomerSchema | None:
    """
    Return the most recently created schema bound to ``client_key``, or ``None``.

    For the POC a client has at most one active schema; if several rows exist the
    newest by ``CREATED_AT`` wins.
    """
    with engine.connect() as connection:
        row = connection.execute(
            text(
                f'''SELECT "CLIENT_KEY", "SCHEMA_ID", "SCHEMA_VERSION",
                           "IS_CANONICAL", "SOURCE"
                    FROM "{CUSTOMER_SCHEMAS}"
                    WHERE "CLIENT_KEY" = :ck
                    ORDER BY "CREATED_AT" DESC'''
            ),
            {"ck": client_key},
        ).mappings().first()
    if row is None:
        return None
    row = _upper(row)
    return CustomerSchema(
        client_key=row["CLIENT_KEY"],
        schema_id=row["SCHEMA_ID"],
        schema_version=row["SCHEMA_VERSION"],
        is_canonical=bool(row["IS_CANONICAL"]),
        source=row["SOURCE"],
    )


def bind_schema(engine: Engine, schema: CustomerSchema) -> None:
    """
    Insert or update a client->schema binding (used by generate/promote in step 6).

    Keyed by (CLIENT_KEY, SCHEMA_ID, SCHEMA_VERSION). Re-binding the same version
    updates its ``IS_CANONICAL`` / ``SOURCE`` flags and refreshes ``CREATED_AT``,
    so it becomes the binding ``get_bound_schema`` returns (rollback to an older version).
    """
    with engine.begin() as connection:
        exists = connection.execute(
            text(
                f'''SELECT COUNT(*) FROM "{CUSTOMER_SCHEMAS}"
                    WHERE "CLIENT_KEY" = :ck AND "SCHEMA_ID" = :sid
                      AND "SCHEMA_VERSION" = :ver'''
            ),
            {"ck": schema.client_key, "sid": schema.schema_id, "ver": schema.schema_version},
        ).scalar()
        params = {
            "ck": schema.client_key,
            "sid": schema.schema_id,
            "ver": schema.schema_version,
            "canon": schema.is_canonical,
            "src": schema.source,
        }
        if exists:
            connection.execute(
                text(
                    f'''UPDATE "{CUSTOMER_SCHEMAS}"
                        SET "IS_CANONICAL" = :canon, "SOURCE" = :src, "CREATED_AT" = CURRENT_UTCTIMESTAMP
                        WHERE "CLIENT_KEY" = :ck AND "SCHEMA_ID" = :sid
                          AND "SCHEMA_VERSION" = :ver'''
                ),
                params,
            )
        else:
            connection.execute(
                text(
                    f'''INSERT INTO "{CUSTOMER_SCHEMAS}"
                        ("CLIENT_KEY", "SCHEMA_ID", "SCHEMA_VERSION", "IS_CANONICAL", "SOURCE")
                        VALUES (:ck, :sid, :ver, :canon, :src)'''
                ),
                params,
            )


def find_customers(
    engine: Engine,
    query: str,
    limit: int = 10,
) -> list[tuple[str, str, float]]:
    """Fuzzy-search customer display names in HANA (CONTAINS ... FUZZY), ranked by SCORE().

    Runs entirely server-side; never downloads the full customer list.
    Returns an empty list immediately for an empty or blank query without
    executing any SQL.

    Args:
        engine: SAP HANA SQLAlchemy engine.
        query:  The search string typed by the user.
        limit:  Maximum number of results to return (default 10).

    Returns:
        A list of (client_key, display_name, score) tuples ordered by score
        descending.  ``score`` is the HANA SCORE() float (0.0–1.0).
    """
    # Short-circuit for empty input — no SQL emitted.
    if not query or not query.strip():
        return []

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f'''SELECT "CLIENT_KEY", "DISPLAY_NAME", SCORE() AS "SCORE"
                    FROM "{CUSTOMERS}"
                    WHERE CONTAINS("DISPLAY_NAME", :q, FUZZY(0.7))
                    ORDER BY "SCORE" DESC
                    LIMIT :n'''
            ),
            {"q": query.strip(), "n": int(limit)},
        ).mappings().all()

    # Map raw HANA rows (keys may be lowercase from hdbcli) to typed tuples.
    out: list[tuple[str, str, float]] = []
    for raw in rows:
        row = _upper(raw)
        out.append((row["CLIENT_KEY"], row["DISPLAY_NAME"], float(row["SCORE"])))
    return out
