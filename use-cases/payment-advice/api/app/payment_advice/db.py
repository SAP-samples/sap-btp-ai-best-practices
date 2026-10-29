"""
SAP HANA engine helper and startup bootstrap for the Payment Advice Extractor.

Reuses the template's ``HANAConnection`` (which registers the ``sqlalchemy_hana``
dialect and builds the engine from ``HANA_*`` environment variables) and adds a
one-call ``bootstrap`` that auto-creates the configuration tables and seeds the
known critical customers. This makes the tool self-installing in a customer
landscape: run it once and the schema/seed data appear.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from app.utils.hana import HANAConnection

from .customers import seed_customers
from .hana_schema import ensure_tables


def get_engine() -> Engine:
    """
    Build and return a SAP HANA SQLAlchemy engine from ``HANA_*`` env variables.

    Returns:
        A configured SQLAlchemy ``Engine``. Callers use ``engine.begin()`` /
        ``engine.connect()`` for their own connections.
    """
    conn = HANAConnection()
    conn.connect()  # registers the sqlalchemy_hana dialect and builds conn.engine
    # ponytail: HANAConnection.connect() also opens one live connection; close it
    # so we don't leak an idle session. The engine's own pool serves real work.
    if conn.connection is not None:
        conn.connection.close()
    return conn.engine


def bootstrap(engine: Engine | None = None, *, seed: bool = True) -> str:
    """
    Ensure configuration tables exist and (optionally) seed critical customers.

    Idempotent: safe to run on every startup. Creates missing tables, validates
    existing ones, and inserts any missing seed customers without overwriting rows.

    Args:
        engine: Optional existing engine; a new one is built when omitted.
        seed: Whether to seed the known critical customers.

    Returns:
        The HANA schema name the tables live in.
    """
    engine = engine or get_engine()
    schema = ensure_tables(engine)
    if seed:
        seed_customers(engine)
    return schema
