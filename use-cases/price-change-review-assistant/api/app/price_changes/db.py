from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from .settings import PriceChangeSettings


def create_hana_engine(settings: PriceChangeSettings) -> Engine:
    if not all([settings.hana_address, settings.hana_port, settings.hana_user, settings.hana_password]):
        raise ValueError("Missing HANA connection settings")
    user = quote_plus(str(settings.hana_user))
    password = quote_plus(str(settings.hana_password))
    address = str(settings.hana_address)
    port = str(settings.hana_port)
    return create_engine(f"hana://{user}:{password}@{address}:{port}", echo=False, future=True)


@contextmanager
def hana_connection(engine: Engine) -> Iterator[Connection]:
    with engine.begin() as connection:
        yield connection
