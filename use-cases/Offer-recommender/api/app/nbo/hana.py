from __future__ import annotations

import os
from contextlib import closing
from dataclasses import dataclass

import pandas as pd

from app.nbo.config import (
    TABLE_ACTIVE_OFFERING,
    TABLE_ACCOUNT_PROFILE,
    TABLE_COMMERCIAL,
    TABLE_COMMERCIAL_SEGMENT,
    TABLE_DER_PROFILE,
    TABLE_PROGRAM_CONTRACT,
    TABLE_PROGRAM_EVENT_HISTORY,
    TABLE_PROGRAM_SAMPLES,
    TABLE_RESIDENTIAL,
    TABLE_RESIDENTIAL_SEGMENT,
)


DATASET_TABLES: dict[str, str] = {
    "residential": TABLE_RESIDENTIAL,
    "res_segment": TABLE_RESIDENTIAL_SEGMENT,
    "commercial": TABLE_COMMERCIAL,
    "comm_segment": TABLE_COMMERCIAL_SEGMENT,
    "active_offering": TABLE_ACTIVE_OFFERING,
    "program_contract": TABLE_PROGRAM_CONTRACT,
    "program_samples": TABLE_PROGRAM_SAMPLES,
    "account_profile": TABLE_ACCOUNT_PROFILE,
    "der_profile": TABLE_DER_PROFILE,
    "program_event_history": TABLE_PROGRAM_EVENT_HISTORY,
}

_TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSEY_VALUES = {"0", "false", "f", "no", "n", "off"}


class HanaConfigurationError(RuntimeError):
    """Raised when required HANA configuration is missing or invalid."""


class HanaDependencyError(RuntimeError):
    """Raised when HANA runtime dependencies are unavailable."""


@dataclass(frozen=True)
class HanaSettings:
    address: str
    port: int
    user: str
    password: str
    encrypt: bool

    @classmethod
    def from_env(cls) -> "HanaSettings":
        raw_values = {
            "hana_address": os.getenv("hana_address", "").strip(),
            "hana_port": os.getenv("hana_port", "").strip(),
            "hana_user": os.getenv("hana_user", "").strip(),
            "hana_password": os.getenv("hana_password", "").strip(),
            "hana_encrypt": os.getenv("hana_encrypt", "").strip(),
        }
        missing = [name for name, value in raw_values.items() if not value]
        if missing:
            joined = ", ".join(sorted(missing))
            raise HanaConfigurationError(
                "Missing required HANA configuration. Set these environment variables: "
                f"{joined}"
            )

        try:
            port = int(raw_values["hana_port"])
        except ValueError as exc:
            raise HanaConfigurationError(
                f"Invalid hana_port value: {raw_values['hana_port']!r}"
            ) from exc

        encrypt_value = raw_values["hana_encrypt"].lower()
        if encrypt_value in _TRUTHY_VALUES:
            encrypt = True
        elif encrypt_value in _FALSEY_VALUES:
            encrypt = False
        else:
            raise HanaConfigurationError(
                "Invalid hana_encrypt value. Use one of: "
                "true, false, 1, 0, yes, no."
            )

        return cls(
            address=raw_values["hana_address"],
            port=port,
            user=raw_values["hana_user"],
            password=raw_values["hana_password"],
            encrypt=encrypt,
        )


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def connect(settings: HanaSettings):
    try:
        from hdbcli import dbapi
    except ImportError as exc:
        raise HanaDependencyError(
            "hdbcli is required for HANA-backed runtime access."
        ) from exc

    return dbapi.connect(
        address=settings.address,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        encrypt=settings.encrypt,
    )


def fetch_table(connection, table_name: str) -> pd.DataFrame:
    with closing(connection.cursor()) as cursor:
        cursor.execute(f"SELECT * FROM {quote_identifier(table_name)}")
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description or ()]
    return pd.DataFrame(rows, columns=columns)


def load_runtime_datasets(settings: HanaSettings | None = None) -> dict[str, pd.DataFrame]:
    config = settings or HanaSettings.from_env()
    with closing(connect(config)) as connection:
        return {
            dataset_name: fetch_table(connection, table_name)
            for dataset_name, table_name in DATASET_TABLES.items()
        }
