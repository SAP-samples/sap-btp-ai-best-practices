"""HANA-backed reference data loading for deployment runtime."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict

import pandas as pd


DEFAULT_TABLES = {
    "catalog": "REFERENCE_COMMODITY_CATALOG",
    "unspsc": "REFERENCE_UNSPSC_MAPPING",
    "supplier_groups": "REFERENCE_SUPPLIER_GROUPS",
}

DATA_VERSION_COLUMN = "DATA_VERSION"

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, "ReferenceDataBundle"] = {}


class ReferenceDataError(RuntimeError):
    """Raised when reference data cannot be loaded or validated."""


@dataclass(frozen=True)
class ReferenceDataBundle:
    catalog_df: pd.DataFrame
    unspsc_df: pd.DataFrame
    supplier_groups_df: pd.DataFrame
    data_version: str


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ReferenceDataError(f"Missing required environment variable: {name}")
    return value


def _bool_to_hana_flag(value: str | None, default: str = "false") -> str:
    raw = (value or default).strip().lower()
    return "true" if raw in {"1", "true", "yes", "y", "on"} else "false"


def _quoted_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _qualified_table_name(schema: str | None, table: str) -> str:
    if schema:
        return f"{_quoted_identifier(schema)}.{_quoted_identifier(table)}"
    return _quoted_identifier(table)


def _connect():
    try:
        from hdbcli import dbapi
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
        raise ReferenceDataError(
            "hdbcli is required for HANA-backed reference data. Add it to the deployment environment."
        ) from exc

    address = _require_env("hana_address")
    port = int(_require_env("hana_port"))
    user = _require_env("hana_user")
    password = _require_env("hana_password")
    encrypt = _bool_to_hana_flag(os.getenv("hana_encrypt"), default="true")
    validate_cert = _bool_to_hana_flag(os.getenv("hana_ssl_validate_certificate"), default="false")

    try:
        return dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password,
            encrypt=encrypt,
            sslValidateCertificate=validate_cert,
        )
    except Exception as exc:  # pragma: no cover - network/runtime dependency
        raise ReferenceDataError(f"Failed to connect to HANA at {address}:{port}: {exc}") from exc


def _fetch_dataframe(connection, schema: str | None, table: str) -> pd.DataFrame:
    qualified_name = _qualified_table_name(schema, table)
    sql = f"SELECT * FROM {qualified_name}"
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
    except Exception as exc:  # pragma: no cover - depends on HANA runtime
        raise ReferenceDataError(f"Failed to query {qualified_name}: {exc}") from exc
    finally:
        cursor.close()
    return pd.DataFrame(rows, columns=columns)


def _extract_version(name: str, df: pd.DataFrame) -> str:
    if DATA_VERSION_COLUMN not in df.columns:
        raise ReferenceDataError(f"{name} is missing required column {DATA_VERSION_COLUMN}.")
    versions = sorted({str(value).strip() for value in df[DATA_VERSION_COLUMN].dropna() if str(value).strip()})
    if len(versions) != 1:
        raise ReferenceDataError(f"{name} must contain exactly one {DATA_VERSION_COLUMN}; found {versions}.")
    return versions[0]


def _drop_version_column(df: pd.DataFrame) -> pd.DataFrame:
    if DATA_VERSION_COLUMN in df.columns:
        return df.drop(columns=[DATA_VERSION_COLUMN]).copy()
    return df.copy()


def _build_cache_key() -> str:
    schema = os.getenv("HANA_SCHEMA", "").strip()
    expected_version = os.getenv("HANA_REFERENCE_DATA_VERSION", "").strip()
    table_env = tuple(
        os.getenv(env_name, default).strip()
        for env_name, default in (
            ("HANA_COMMODITY_CATALOG_TABLE", DEFAULT_TABLES["catalog"]),
            ("HANA_UNSPSC_MAPPING_TABLE", DEFAULT_TABLES["unspsc"]),
            ("HANA_SUPPLIER_GROUPS_TABLE", DEFAULT_TABLES["supplier_groups"]),
        )
    )
    return "|".join((schema, expected_version, *table_env))


def load_reference_data(force_refresh: bool = False) -> ReferenceDataBundle:
    cache_key = _build_cache_key()
    with _CACHE_LOCK:
        if not force_refresh and cache_key in _CACHE:
            return _CACHE[cache_key]

    schema = os.getenv("HANA_SCHEMA", "").strip() or None
    expected_version = os.getenv("HANA_REFERENCE_DATA_VERSION", "").strip()
    table_names: Dict[str, str] = {
        "catalog": os.getenv("HANA_COMMODITY_CATALOG_TABLE", DEFAULT_TABLES["catalog"]).strip() or DEFAULT_TABLES["catalog"],
        "unspsc": os.getenv("HANA_UNSPSC_MAPPING_TABLE", DEFAULT_TABLES["unspsc"]).strip() or DEFAULT_TABLES["unspsc"],
        "supplier_groups": os.getenv("HANA_SUPPLIER_GROUPS_TABLE", DEFAULT_TABLES["supplier_groups"]).strip() or DEFAULT_TABLES["supplier_groups"],
    }

    connection = _connect()
    try:
        catalog_df = _fetch_dataframe(connection, schema, table_names["catalog"])
        unspsc_df = _fetch_dataframe(connection, schema, table_names["unspsc"])
        supplier_groups_df = _fetch_dataframe(connection, schema, table_names["supplier_groups"])
    finally:
        connection.close()

    versions = {
        "catalog": _extract_version(table_names["catalog"], catalog_df),
        "unspsc": _extract_version(table_names["unspsc"], unspsc_df),
        "supplier_groups": _extract_version(table_names["supplier_groups"], supplier_groups_df),
    }
    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) != 1:
        raise ReferenceDataError(f"Reference tables have mismatched data versions: {versions}")

    data_version = unique_versions[0]
    if expected_version and data_version != expected_version:
        raise ReferenceDataError(
            f"Reference data version mismatch: expected {expected_version}, found {data_version}"
        )

    bundle = ReferenceDataBundle(
        catalog_df=_drop_version_column(catalog_df),
        unspsc_df=_drop_version_column(unspsc_df),
        supplier_groups_df=_drop_version_column(supplier_groups_df),
        data_version=data_version,
    )

    with _CACHE_LOCK:
        _CACHE[cache_key] = bundle
    return bundle

