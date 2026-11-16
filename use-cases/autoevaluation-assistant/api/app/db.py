"""Database helpers for HANA-backed application repositories."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

TRANSIENT_HANA_CONNECTION_ERROR_CODES = {-10709, -10807}
"""HANA DBAPI codes observed for connect and active-connection route loss."""


def is_transient_database_error(error: DBAPIError) -> bool:
    """Return whether a SQLAlchemy DBAPI error represents lost connectivity.

    Inputs:
        error: SQLAlchemy wrapper around a database-driver failure.

    Outputs:
        bool: ``True`` when SQLAlchemy invalidated the connection or the HANA
        driver reported one of the observed connection-loss error codes;
        ``False`` for SQL/schema/programming failures that must remain visible.
    """

    if error.connection_invalidated:
        return True
    original_args = getattr(error.orig, "args", ())
    error_code = original_args[0] if original_args else None
    return error_code in TRANSIENT_HANA_CONNECTION_ERROR_CODES


def build_hana_url() -> URL:
    """Build a SQLAlchemy HANA connection URL from environment variables.

    Inputs:
        The function reads ``HANA_ADDRESS``, ``HANA_PORT``, ``HANA_USER``,
        ``HANA_PASSWORD``, and optional ``HANA_ENCRYPT`` from the process
        environment. ``HANA_ENCRYPT`` defaults to ``true``.

    Outputs:
        URL: A SQLAlchemy URL for the ``sqlalchemy-hana`` HANA dialect. The URL
        object preserves raw credential values while rendering reserved
        characters safely escaped when converted to a string.

    Raises:
        ValueError: Raised when any required HANA credential is missing.
    """
    required_values = {
        "HANA_ADDRESS": os.getenv("HANA_ADDRESS"),
        "HANA_PORT": os.getenv("HANA_PORT"),
        "HANA_USER": os.getenv("HANA_USER"),
        "HANA_PASSWORD": os.getenv("HANA_PASSWORD"),
    }
    missing_values = [
        key for key, value in required_values.items() if value is None or value == ""
    ]
    if missing_values:
        raise ValueError(
            "Missing required HANA environment variables: "
            + ", ".join(sorted(missing_values))
        )

    encrypt = os.getenv("HANA_ENCRYPT", "true").lower()
    return URL.create(
        drivername="hana",
        username=required_values["HANA_USER"],
        password=required_values["HANA_PASSWORD"],
        host=required_values["HANA_ADDRESS"],
        port=int(required_values["HANA_PORT"] or "0"),
        query={"encrypt": encrypt},
    )


def build_hana_connect_args() -> dict[str, str]:
    """Build driver-level HANA connection properties from the environment.

    HANA resolves unqualified table names against ``CURRENT_SCHEMA``, which
    defaults to the login user's own schema. The optional ``HANA_SCHEMA``
    variable lets the application log in as one database user (for example a
    technical or admin user) while reading and writing the tables of another
    schema. It is forwarded to ``hdbcli`` as the ``currentSchema`` property.

    Inputs:
        The function reads optional ``HANA_SCHEMA`` from the process
        environment. Surrounding whitespace is ignored.

    Outputs:
        dict[str, str]: ``{"currentSchema": <schema>}`` when ``HANA_SCHEMA`` is
        set, or an empty dictionary so the login user's default schema is used.
    """
    schema = os.getenv("HANA_SCHEMA", "").strip()
    return {"currentSchema": schema} if schema else {}


def create_hana_engine() -> Engine:
    """Create a SQLAlchemy engine configured for HANA repository access.

    Inputs:
        The function reads HANA connection settings through ``build_hana_url``
        and the optional schema override (``HANA_SCHEMA``) through
        ``build_hana_connect_args``.

    Outputs:
        Engine: A SQLAlchemy engine with connection pre-ping enabled, SQL
        echoing disabled, and every pooled connection bound to ``HANA_SCHEMA``
        when it is set (otherwise to the login user's own schema).
    """
    return create_engine(
        build_hana_url(),
        echo=False,
        pool_pre_ping=True,
        connect_args=build_hana_connect_args(),
    )


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional SQLAlchemy session context.

    Inputs:
        session_factory: A configured SQLAlchemy ``sessionmaker`` that creates
        ``Session`` instances.

    Outputs:
        Iterator[Session]: Yields one active session, commits on normal exit,
        rolls back when an exception is raised, and always closes the session.
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
