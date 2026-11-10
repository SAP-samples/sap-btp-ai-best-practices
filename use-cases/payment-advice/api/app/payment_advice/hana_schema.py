"""
SAP HANA table DDL and catalog-contract validation for the Payment Advice Extractor.

All configuration tables use the prefix ``PAYMENT_ADVICE_EXTRACTOR_`` so they are
easy to identify among other tables in a shared tenant. UC-01 owns two tables:

    PAYMENT_ADVICE_EXTRACTOR_CUSTOMERS         one row per client (criticality flag)
    PAYMENT_ADVICE_EXTRACTOR_CUSTOMER_SCHEMAS  Document AI schema(s) bound to a client

``DEDUCTION_RULES`` (UC-02) holds per-client deduction playbooks (skill catalog).
``EMAIL_MAP`` (UC-03) is intentionally not created here; it is added when that use
case is built.

``ensure_tables`` is idempotent and self-healing: it creates only missing tables and
validates the catalog contract of any table that already exists, so the same code can
be run unchanged in a customer landscape. This pattern is adapted from the reused
DocumentAI-Agent ``hana_schema`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Engine


PREFIX = "PAYMENT_ADVICE_EXTRACTOR_"
CUSTOMERS = PREFIX + "CUSTOMERS"
CUSTOMER_SCHEMAS = PREFIX + "CUSTOMER_SCHEMAS"
DEDUCTION_RULES = PREFIX + "DEDUCTION_RULES"

# Allowed values for CUSTOMER_SCHEMAS.SOURCE (documentation; not a DB constraint).
SCHEMA_SOURCES = ("dedicated", "generated", "canonical")


@dataclass(frozen=True)
class ColumnContract:
    """Expected HANA catalog metadata for one table column."""

    data_type: str
    nullable: bool
    length: int | None = None
    default: str | None = None
    primary_key: bool = False


TABLE_CONTRACTS: dict[str, dict[str, ColumnContract]] = {
    CUSTOMERS: {
        "CLIENT_KEY": ColumnContract("NVARCHAR", False, length=60, primary_key=True),
        "DISPLAY_NAME": ColumnContract("NVARCHAR", False, length=200),
        "IS_CRITICAL": ColumnContract("BOOLEAN", False, default="FALSE"),
        "STATUS": ColumnContract("NVARCHAR", False, length=20, default="'active'"),
        "CREATED_AT": ColumnContract("TIMESTAMP", False, default="CURRENT_UTCTIMESTAMP"),
        "UPDATED_AT": ColumnContract("TIMESTAMP", False, default="CURRENT_UTCTIMESTAMP"),
    },
    CUSTOMER_SCHEMAS: {
        "CLIENT_KEY": ColumnContract("NVARCHAR", False, length=60, primary_key=True),
        "SCHEMA_ID": ColumnContract("NVARCHAR", False, length=100, primary_key=True),
        "SCHEMA_VERSION": ColumnContract("NVARCHAR", False, length=20, primary_key=True),
        "IS_CANONICAL": ColumnContract("BOOLEAN", False, default="FALSE"),
        "SOURCE": ColumnContract("NVARCHAR", False, length=20),
        "CREATED_AT": ColumnContract("TIMESTAMP", False, default="CURRENT_UTCTIMESTAMP"),
    },
    # UC-02: per-client deduction playbooks (skill catalog for the interpretation agent).
    DEDUCTION_RULES: {
        "CLIENT_KEY": ColumnContract("NVARCHAR", False, length=60, primary_key=True),
        "PLAYBOOK_TEXT": ColumnContract("NCLOB", True),
        "ANCHORS_JSON": ColumnContract("NCLOB", True),
        "REVISION": ColumnContract("INTEGER", False, default="0"),
        "UPDATED_BY": ColumnContract("NVARCHAR", True, length=120),
        "CREATED_AT": ColumnContract("TIMESTAMP", False, default="CURRENT_UTCTIMESTAMP"),
        "UPDATED_AT": ColumnContract("TIMESTAMP", False, default="CURRENT_UTCTIMESTAMP"),
    },
}

TABLE_DDL: dict[str, str] = {
    CUSTOMERS: f'''\
CREATE COLUMN TABLE "{CUSTOMERS}" (
  "CLIENT_KEY" NVARCHAR(60) NOT NULL PRIMARY KEY,
  "DISPLAY_NAME" NVARCHAR(200) NOT NULL,
  "IS_CRITICAL" BOOLEAN DEFAULT FALSE NOT NULL,
  "STATUS" NVARCHAR(20) DEFAULT 'active' NOT NULL,
  "CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL,
  "UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
)''',
    CUSTOMER_SCHEMAS: f'''\
CREATE COLUMN TABLE "{CUSTOMER_SCHEMAS}" (
  "CLIENT_KEY" NVARCHAR(60) NOT NULL,
  "SCHEMA_ID" NVARCHAR(100) NOT NULL,
  "SCHEMA_VERSION" NVARCHAR(20) NOT NULL,
  "IS_CANONICAL" BOOLEAN DEFAULT FALSE NOT NULL,
  "SOURCE" NVARCHAR(20) NOT NULL,
  "CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL,
  PRIMARY KEY ("CLIENT_KEY", "SCHEMA_ID", "SCHEMA_VERSION")
)''',
    # UC-02: per-client deduction playbooks (skill catalog for the interpretation agent).
    DEDUCTION_RULES: f'''\
CREATE COLUMN TABLE "{DEDUCTION_RULES}" (
  "CLIENT_KEY" NVARCHAR(60) NOT NULL PRIMARY KEY,
  "PLAYBOOK_TEXT" NCLOB,
  "ANCHORS_JSON" NCLOB,
  "REVISION" INTEGER DEFAULT 0 NOT NULL,
  "UPDATED_BY" NVARCHAR(120),
  "CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL,
  "UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL
)''',
}


def _normalized_default(value: Any) -> str | None:
    """
    Normalize HANA catalog default expressions for stable comparison.

    HANA stores BOOLEAN defaults as ``0`` / ``1`` even when the DDL used
    ``FALSE`` / ``TRUE``, and wraps string defaults in quotes/parentheses. We fold
    these to a canonical form so contract validation does not reject a well-formed
    table.
    """
    if value is None:
        return None
    normalized = "".join(str(value).upper().split())
    if not normalized or normalized == "NULL":
        return None
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1]
    if normalized.endswith("()"):
        normalized = normalized[:-2]
    if normalized in {"FALSE", "0"}:
        return "FALSE"
    if normalized in {"TRUE", "1"}:
        return "TRUE"
    # String defaults like 'active' are stored as 'ACTIVE' with quotes; strip quotes.
    if normalized.startswith("'") and normalized.endswith("'"):
        normalized = normalized[1:-1]
    return normalized


def _upper_keys(row: Any) -> dict[str, Any]:
    """Return a dict with uppercase keys for one HANA result row.

    The ``hdbcli`` driver reports column names in lowercase; we address columns by
    their SQL identifiers, so normalizing each row keeps the validator readable.
    """
    return {str(key).upper(): value for key, value in row.items()}


def assert_table_contract(
    connection: Any,
    schema: str,
    table_name: str,
    expected: Mapping[str, ColumnContract],
) -> None:
    """Compare existing HANA columns, defaults, and primary key with one contract."""
    rows = connection.execute(
        text(
            """
            SELECT COLUMN_NAME, DATA_TYPE_NAME, LENGTH, IS_NULLABLE,
                   DEFAULT_VALUE, GENERATION_TYPE
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = :schema AND TABLE_NAME = :table
            """
        ),
        {"schema": schema, "table": table_name},
    ).mappings().all()
    actual = {str(_upper_keys(row)["COLUMN_NAME"]).upper(): _upper_keys(row) for row in rows}

    primary_key_rows = connection.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM SYS.CONSTRAINTS
            WHERE SCHEMA_NAME = :schema AND TABLE_NAME = :table
              AND IS_PRIMARY_KEY = 'TRUE'
            """
        ),
        {"schema": schema, "table": table_name},
    ).mappings().all()
    actual_primary_key = {str(_upper_keys(row)["COLUMN_NAME"]).upper() for row in primary_key_rows}
    expected_primary_key = {col for col, c in expected.items() if c.primary_key}

    mismatches: list[str] = []
    for column, contract in expected.items():
        metadata = actual.get(column)
        if metadata is None:
            mismatches.append(f"{column}: missing")
            continue
        actual_type = str(metadata.get("DATA_TYPE_NAME", "")).upper()
        actual_nullable = str(metadata.get("IS_NULLABLE", "")).upper() == "TRUE"
        if actual_type != contract.data_type or actual_nullable != contract.nullable:
            mismatches.append(
                f"{column}: expected {contract.data_type} nullable={contract.nullable}, "
                f"got {actual_type} nullable={actual_nullable}"
            )
        if contract.length is not None and metadata.get("LENGTH") != contract.length:
            mismatches.append(
                f"{column}: expected length {contract.length}, got {metadata.get('LENGTH')}"
            )
        if contract.default is not None and _normalized_default(
            metadata.get("DEFAULT_VALUE")
        ) != _normalized_default(contract.default):
            mismatches.append(
                f"{column}: expected default {contract.default}, got {metadata.get('DEFAULT_VALUE')}"
            )

    if actual_primary_key != expected_primary_key:
        mismatches.append(
            f"primary key: expected {sorted(expected_primary_key)}, got {sorted(actual_primary_key)}"
        )

    if mismatches:
        raise RuntimeError(
            f"HANA table {schema}.{table_name} has incompatible contract: " + "; ".join(mismatches)
        )


def ensure_tables(engine: Engine) -> str:
    """
    Create absent Payment Advice Extractor tables; validate existing ones.

    Runs inside a single transaction against ``CURRENT_SCHEMA`` (technical users
    usually have DDL rights only there).

    Returns:
        The HANA schema name the tables live in.
    """
    with engine.begin() as connection:
        schema = connection.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY")).scalar()
        if not schema:
            raise RuntimeError("SAP HANA did not return CURRENT_SCHEMA")

        for table_name, expected in TABLE_CONTRACTS.items():
            exists = connection.execute(
                text(
                    """
                    SELECT COUNT(*) FROM SYS.TABLES
                    WHERE SCHEMA_NAME = :schema AND TABLE_NAME = :table
                    """
                ),
                {"schema": schema, "table": table_name},
            ).scalar()
            if exists:
                assert_table_contract(connection, schema, table_name, expected)
            else:
                connection.execute(text(TABLE_DDL[table_name]))
    return str(schema)
