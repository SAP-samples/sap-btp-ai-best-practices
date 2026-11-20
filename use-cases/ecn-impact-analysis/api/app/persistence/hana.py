"""Safe SQLAlchemy HANA engine factory and UTC timestamp helpers.

This module provides:

* ``create_hana_engine`` — builds a pooled HANA engine from validated
  environment credentials without ever string-interpolating the password.
* ``to_hana_timestamp`` — converts a tz-aware Python ``datetime`` to the naive
  UTC value expected by SAP HANA's TIMESTAMP column type.
* ``from_hana_timestamp`` — restores UTC ``tzinfo`` to a naive datetime read
  back from a HANA TIMESTAMP column.

This module is the sole production HANA engine boundary.
Credentials are never logged, and driver exceptions are never re-raised
verbatim (they could contain credential fragments in their message text).

Usage::

    from app.persistence.hana import create_hana_engine, to_hana_timestamp

    engine = create_hana_engine()          # reads HANA_* from os.environ
    engine = create_hana_engine(environ)   # reads from the given mapping
"""

import os
from collections.abc import Mapping
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from app.ecn.errors import StartupConfigurationError

# ---------------------------------------------------------------------------
# HANA_ENCRYPT value sets — case-insensitive, lowercase-normalised
# ---------------------------------------------------------------------------

# Values that map to encrypt=true (secure connection)
TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes"})

# Values that map to encrypt=false (unencrypted; typically only local dev)
FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no"})


def create_hana_engine(environ: Mapping[str, str] | None = None) -> Engine:
    """Create one pooled HANA engine from validated runtime credentials.

    Reads ``HANA_ADDRESS``, ``HANA_PORT``, ``HANA_USER``, ``HANA_PASSWORD``,
    and optionally ``HANA_ENCRYPT`` (default ``"true"``) from ``environ``.
    Uses SQLAlchemy's ``URL.create`` so the password is never interpolated
    into a connection string.

    Raises ``StartupConfigurationError`` for any of:

    * Any required field missing or whitespace-only.
    * ``HANA_PORT`` not parseable as an integer.
    * ``HANA_PORT`` outside the valid 1-65535 range.
    * ``HANA_ENCRYPT`` not in the allowed set
      (``true | 1 | yes | false | 0 | no``).

    Sensitive values (address, port, user, password, driver exception text)
    are never embedded in the raised error's ``public_message`` or
    ``details``.

    Args:
        environ: Optional credential mapping; defaults to ``os.environ``
            when ``None``.  Passing an explicit dict is preferred in tests.

    Returns:
        A ``sqlalchemy.engine.Engine`` configured with ``pool_pre_ping=True``.

    Raises:
        StartupConfigurationError: When credentials are incomplete, the port
            is invalid, or the encryption setting is unrecognised.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ

    # --- Required fields ---------------------------------------------------
    required = {
        name: source.get(name, "").strip()
        for name in ("HANA_ADDRESS", "HANA_PORT", "HANA_USER", "HANA_PASSWORD")
    }
    if any(not value for value in required.values()):
        raise StartupConfigurationError(
            "HANA_CONFIGURATION_INVALID",
            "HANA credentials are incomplete",
            status_code=500,
        )

    # --- Port validation ---------------------------------------------------
    try:
        port = int(required["HANA_PORT"])
    except ValueError as exc:
        raise StartupConfigurationError(
            "HANA_CONFIGURATION_INVALID",
            "HANA port is invalid",
            status_code=500,
        ) from exc
    if not 1 <= port <= 65535:
        raise StartupConfigurationError(
            "HANA_CONFIGURATION_INVALID",
            "HANA port is invalid",
            status_code=500,
        )

    # --- Encrypt setting ---------------------------------------------------
    encrypt_raw = source.get("HANA_ENCRYPT", "true").strip().lower()
    if encrypt_raw not in TRUE_VALUES | FALSE_VALUES:
        raise StartupConfigurationError(
            "HANA_CONFIGURATION_INVALID",
            "HANA encryption setting is invalid",
            status_code=500,
        )
    encrypt_value = "true" if encrypt_raw in TRUE_VALUES else "false"

    # --- Build URL without string interpolation ----------------------------
    url = URL.create(
        "hana",
        username=required["HANA_USER"],
        password=required["HANA_PASSWORD"],
        host=required["HANA_ADDRESS"],
        port=port,
        query={"encrypt": encrypt_value},
    )

    # pool_pre_ping=True causes SQLAlchemy to emit a lightweight ping before
    # handing out a pooled connection, recycling stale connections silently.
    return create_engine(url, pool_pre_ping=True)


def to_hana_timestamp(value: datetime) -> datetime:
    """Convert an aware datetime to a naive UTC HANA TIMESTAMP value.

    SAP HANA TIMESTAMP columns store naive datetime values.  This helper
    converts the input to UTC and strips the ``tzinfo`` so the value is
    safe for direct insertion into a TIMESTAMP column.

    Args:
        value: A timezone-aware ``datetime`` to convert.

    Returns:
        A naive ``datetime`` representing the same instant in UTC.

    Raises:
        ValueError: If ``value`` does not carry timezone information (is
            naive).  All datetimes stored by this application must be
            tz-aware at the domain layer.
    """
    if value.tzinfo is None:
        raise ValueError(
            "audit datetime must be timezone-aware; naive datetimes are ambiguous"
        )
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def from_hana_timestamp(value: datetime) -> datetime:
    """Restore UTC timezone information to a naive datetime read from HANA.

    Inverse of ``to_hana_timestamp``.  Attaches ``tzinfo=utc`` to a naive
    datetime that was read back from a HANA TIMESTAMP column and is known
    to represent a UTC instant.

    Args:
        value: A naive ``datetime`` value as returned by the HANA driver for
            a TIMESTAMP column.

    Returns:
        The same instant as a timezone-aware ``datetime`` with ``tzinfo`` set
        to ``datetime.timezone.utc``.
    """
    return value.replace(tzinfo=timezone.utc)
