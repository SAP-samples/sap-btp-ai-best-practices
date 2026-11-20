"""Shared SQLAlchemy engine for SAP HANA Cloud.

One engine (with a connection pool) is created lazily and reused by every request,
background worker and CLI script in the process. Connection settings come from the
environment so the same code runs locally (``api/.env``) and on Cloud Foundry
(manifest variables):

    HANA_ADDRESS, HANA_PORT, HANA_USER, HANA_PASSWORD   connection (required)
    HANA_ENCRYPT                                        "true" (default) / "false"
    HANA_SSL_VALIDATE_CERTIFICATE                       "true" (default) / "false"
    HANA_SCHEMA                                         optional dedicated schema; when set,
                                                        every pooled connection runs
                                                        ``SET SCHEMA`` so all SQL is unqualified

All database functions in the application are synchronous; async code calls them
through ``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Variables without which no connection can be built.
REQUIRED_ENV = ("HANA_ADDRESS", "HANA_PORT", "HANA_USER", "HANA_PASSWORD")


def _flag(name: str, default: str = "true") -> str:
    """Read a boolean-like environment variable and normalise it for hdbcli.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        "true" or "false".
    """
    return "true" if os.getenv(name, default).strip().lower() in {"1", "true", "yes"} else "false"


def hana_configured() -> bool:
    """Tell whether all required HANA connection variables are set.

    Returns:
        True when HANA_ADDRESS, HANA_PORT, HANA_USER and HANA_PASSWORD are non-empty.
    """
    return all(os.getenv(name) for name in REQUIRED_ENV)


def configured_schema() -> str | None:
    """Return the dedicated schema name from HANA_SCHEMA, upper-cased, or None.

    Returns:
        Schema name, or None to use the user's default schema.

    Raises:
        ValueError: If the name contains characters outside [A-Z0-9_].
    """
    schema = os.getenv("HANA_SCHEMA", "").strip().upper()
    if not schema:
        return None
    if not schema.replace("_", "").isalnum():
        raise ValueError(f"HANA_SCHEMA must contain only letters, digits and '_': {schema!r}")
    return schema


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create (once) and return the pooled SQLAlchemy engine.

    Returns:
        The process-wide Engine.

    Raises:
        RuntimeError: If a required connection variable is missing.
    """
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"HANA is not configured; missing env vars: {', '.join(missing)}")

    # URL.create escapes special characters in the password (a plain f-string would break).
    url = URL.create(
        "hana",
        username=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        host=os.environ["HANA_ADDRESS"],
        port=int(os.environ["HANA_PORT"]),
        query={
            "encrypt": _flag("HANA_ENCRYPT"),
            "sslValidateCertificate": _flag("HANA_SSL_VALIDATE_CERTIFICATE"),
        },
    )
    # pool_pre_ping replaces connections the HANA side has closed while idle.
    engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20, pool_recycle=1800)

    schema = configured_schema()
    if schema:
        @event.listens_for(engine, "connect")
        def _set_schema(dbapi_connection, _record) -> None:  # pragma: no cover - needs HANA
            """Point every new pooled connection at the dedicated schema, creating it if missing."""
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM SYS.SCHEMAS WHERE SCHEMA_NAME = ?", (schema,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f'CREATE SCHEMA "{schema}"')
                    logger.info("Created HANA schema %s", schema)
                cursor.execute(f'SET SCHEMA "{schema}"')
            finally:
                cursor.close()

    logger.info("HANA engine created for %s (schema=%s)", os.environ["HANA_ADDRESS"], schema or "<user default>")
    return engine
