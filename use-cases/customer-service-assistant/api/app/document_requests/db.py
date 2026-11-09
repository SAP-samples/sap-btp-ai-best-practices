from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from .settings import DocumentRequestSettings


@lru_cache(maxsize=4)
def create_hana_engine(settings: DocumentRequestSettings) -> Engine:
    """Create (once per settings value) a pooled SQLAlchemy engine for SAP HANA.

    The engine is cached because it owns the connection pool: building a new
    engine per request would open a fresh HANA connection every time.
    DocumentRequestSettings is a frozen dataclass, so it is hashable and can be
    used as the cache key.

    Args:
        settings: Runtime settings containing HANA connection values.

    Returns:
        SQLAlchemy Engine connected through the HANA dialect.
    """
    if not all([settings.hana_address, settings.hana_port, settings.hana_user, settings.hana_password]):
        raise ValueError("Missing HANA connection settings")
    user = quote_plus(str(settings.hana_user))
    password = quote_plus(str(settings.hana_password))
    address = str(settings.hana_address)
    port = str(settings.hana_port)
    encrypt = "true" if settings.hana_encrypt else "false"
    return create_engine(f"hana://{user}:{password}@{address}:{port}?encrypt={encrypt}", echo=False, future=True, pool_pre_ping=True)


@contextmanager
def hana_connection(engine: Engine) -> Iterator[Connection]:
    """Open a HANA transaction and yield its connection.

    Args:
        engine: SQLAlchemy engine created by create_hana_engine.

    Yields:
        Transactional SQLAlchemy connection.
    """
    with engine.begin() as connection:
        yield connection
