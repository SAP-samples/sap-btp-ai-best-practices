"""HANA schema initialization and structural contract validation.

This module provides:

* ``current_schema`` — executes ``SELECT CURRENT_SCHEMA FROM DUMMY`` and
  returns the authenticated user's schema name.
* ``table_exists`` — queries ``SYS.TABLES`` with bound parameters to check
  physical table presence.
* ``validate_table_contract`` — uses the SQLAlchemy inspector to compare an
  existing table's reflected columns, primary key, unique constraints, and
  foreign keys against the logical contract.
* ``ensure_schema`` — creates every missing ECN impact table in dependency
  order and validates all existing tables before any DDL is attempted.

None of these functions connects to HANA; each requires a caller-supplied
``Connection`` or ``Engine`` so that connection lifecycle is managed outside
this module.  FastAPI lifespan wiring is deferred to a later phase.

Usage::

    from app.persistence.schema import ensure_schema
    from app.persistence.hana import create_hana_engine

    engine = create_hana_engine()
    schema = ensure_schema(engine)
"""

from __future__ import annotations

from dataclasses import replace

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.ecn.errors import StartupConfigurationError
from app.persistence.table_contracts import (
    ColumnContract,
    PersistenceTableNames,
    TableContract,
    build_table_contracts,
    quote_identifier,
    render_create_table,
)


# ---------------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------------


def current_schema(connection: Connection) -> str:
    """Return the current HANA schema selected for the runtime user.

    Executes ``SELECT CURRENT_SCHEMA FROM DUMMY`` against the given
    connection and returns the schema name as a stripped string.

    Args:
        connection: An open SQLAlchemy ``Connection`` to the HANA instance.

    Returns:
        The current schema name, e.g. ``"DBUSER"``.

    Raises:
        StartupConfigurationError: With code ``HANA_SCHEMA_UNAVAILABLE`` if
            the query returns ``None`` or a whitespace-only string.
    """
    value = connection.execute(text("SELECT CURRENT_SCHEMA FROM DUMMY")).scalar_one()
    if not isinstance(value, str) or not value.strip():
        raise StartupConfigurationError(
            "HANA_SCHEMA_UNAVAILABLE",
            "Current HANA schema is unavailable",
            status_code=500,
        )
    return value.strip()


# ---------------------------------------------------------------------------
# Table existence check
# ---------------------------------------------------------------------------


def table_exists(connection: Connection, schema: str, table_name: str) -> bool:
    """Check the physical HANA table catalog with bound metadata values.

    Queries ``SYS.TABLES`` using ``SCHEMA_NAME`` and ``TABLE_NAME`` as bound
    parameters so that neither value is ever string-interpolated into SQL.
    The identifier is validated by ``quote_identifier`` before the query runs.

    Args:
        connection: An open SQLAlchemy ``Connection`` to the HANA instance.
        schema: The target schema name, e.g. ``"DBUSER"``.
        table_name: The physical table name; must match the ECN impact
            identifier allowlist (``^ECN_IMPACT_[A-Z0-9_]{1,110}$``).

    Returns:
        ``True`` when the table exists in ``SYS.TABLES``, ``False`` otherwise.

    Raises:
        ValueError: If ``table_name`` fails the identifier allowlist check.
    """
    # Validate the identifier against the allowlist before using it in SQL.
    # The returned quoted string is discarded; the check is the point.
    quote_identifier(table_name)
    return (
        connection.execute(
            text(
                "SELECT 1 FROM SYS.TABLES "
                "WHERE SCHEMA_NAME = :schema AND TABLE_NAME = :table_name"
            ),
            {"schema": schema, "table_name": table_name},
        ).first()
        is not None
    )


# ---------------------------------------------------------------------------
# Column comparison helpers
# ---------------------------------------------------------------------------


def _reflected_column(
    column: dict,
) -> tuple[str, str, bool, int | None, int | None, int | None]:
    """Project dialect metadata into the versioned column-comparison tuple.

    Reads the column's SQLAlchemy type object attributes rather than comparing
    ``repr()`` strings, so the comparison survives dialect version changes.

    Args:
        column: A reflected column dict as returned by the HANA inspector's
            ``get_columns`` method.  Must have ``name``, ``type``, and
            ``nullable`` keys.

    Returns:
        A 6-tuple: ``(name_upper, type_class_upper, nullable, length,
        precision, scale)``.
    """
    reflected_type = column["type"]
    return (
        str(column["name"]).upper(),
        reflected_type.__class__.__name__.upper(),
        bool(column["nullable"]),
        getattr(reflected_type, "length", None),
        getattr(reflected_type, "precision", None),
        getattr(reflected_type, "scale", None),
    )


def _expected_column_tuple(
    column: ColumnContract,
) -> tuple[str, str, bool, int | None, int | None, int | None]:
    """Convert a ColumnContract to the same 6-tuple used for reflected columns.

    Args:
        column: A ``ColumnContract`` from the logical table definition.

    Returns:
        A 6-tuple matching the format of ``_reflected_column`` output.
    """
    return (
        column.name.upper(),
        column.hana_type.upper(),
        column.nullable,
        column.length,
        column.precision,
        column.scale,
    )


# ---------------------------------------------------------------------------
# Schema-error factory
# ---------------------------------------------------------------------------


def _schema_error(table_name: str, category: str) -> StartupConfigurationError:
    """Build a safe StartupConfigurationError for a schema incompatibility.

    The message includes the table name and category but never driver exception
    text, reflected default values, or raw source content.

    Args:
        table_name: The physical HANA table name where the mismatch was found.
        category: A short mismatch description, e.g. ``"columns"`` or
            ``"primary_key"``.

    Returns:
        A ``StartupConfigurationError`` with code ``HANA_SCHEMA_INCOMPATIBLE``
        and status 500.
    """
    return StartupConfigurationError(
        "HANA_SCHEMA_INCOMPATIBLE",
        f"Table {table_name!r} is incompatible with the expected contract: {category}",
        status_code=500,
    )


# ---------------------------------------------------------------------------
# Structural validators
# ---------------------------------------------------------------------------


def _validate_primary_key(
    inspector: object, schema: str, contract: TableContract
) -> None:
    """Raise if the reflected PK columns differ from the contract.

    Compares the ordered list of reflected PK column names (normalised to
    uppercase) against the contract's ``primary_key`` tuple.

    Args:
        inspector: An open SQLAlchemy inspector bound to the live connection.
        schema: The HANA schema containing the table.
        contract: The expected table contract.

    Raises:
        StartupConfigurationError: When the reflected PK columns differ from
            the contract in order or content.
    """
    pk_info = inspector.get_pk_constraint(contract.name, schema=schema)
    actual_pk = tuple(str(col).upper() for col in pk_info.get("constrained_columns", []))
    expected_pk = tuple(col.upper() for col in contract.primary_key)
    if actual_pk != expected_pk:
        raise _schema_error(contract.name, "primary_key")


def _validate_unique_constraints(
    inspector: object, schema: str, contract: TableContract
) -> None:
    """Raise if the reflected unique constraints differ from the contract.

    Generated constraint names are ignored; only the ordered column tuples
    within each constraint are compared.  The set of constraint tuples is
    sorted so that declaration order in the live table is not significant.

    Args:
        inspector: An open SQLAlchemy inspector bound to the live connection.
        schema: The HANA schema containing the table.
        contract: The expected table contract.

    Raises:
        StartupConfigurationError: When the set of unique column tuples
            differs from the contract.
    """
    actual_uqs = inspector.get_unique_constraints(contract.name, schema=schema)
    # Sort both sides by column-tuple so declaration order is not significant.
    actual_tuples = sorted(
        tuple(str(col).upper() for col in uq["column_names"]) for uq in actual_uqs
    )
    expected_tuples = sorted(
        tuple(col.upper() for col in uq) for uq in contract.unique_constraints
    )
    if actual_tuples != expected_tuples:
        raise _schema_error(contract.name, "unique_constraints")


def _validate_foreign_keys(
    inspector: object,
    schema: str,
    contract: TableContract,
    table_names: PersistenceTableNames,
) -> None:
    """Raise if the reflected FK set differs from the contract.

    The comparison is **order-insensitive** because HANA's ``get_foreign_keys()``
    reflection order is not guaranteed to match the contract's declaration
    order.  Both sides are normalised to sorted tuples so that declaration
    order never causes false incompatibility reports.

    Each FK is represented as a 4-tuple:
    ``(local_cols_upper, referred_schema, referred_table, referred_cols_upper)``
    where ``referred_schema`` defaults to the current schema when ``None``
    (HANA returns ``None`` for same-schema FKs).

    Args:
        inspector: An open SQLAlchemy inspector bound to the live connection.
        schema: The HANA schema containing the table and FK targets.
        contract: The expected table contract.
        table_names: The active ``PersistenceTableNames`` used to resolve
            ``referenced_table_key`` to a physical table name.

    Raises:
        StartupConfigurationError: When the FK count or the sorted FK tuple
            set differs from the contract.
    """
    actual_fks = inspector.get_foreign_keys(contract.name, schema=schema)
    if len(actual_fks) != len(contract.foreign_keys):
        raise _schema_error(contract.name, "foreign_keys")

    # Build normalised 4-tuples for both sides so sorting is well-defined.
    # HANA may return referred_schema=None for same-schema FKs — normalise to
    # the current schema so the comparison is symmetric with the expected side.
    # The HANA dialect returns column names and table names in LOWERCASE, so
    # every string component is uppercased before comparison.
    def _actual_fk_tuple(fk: dict) -> tuple:
        return (
            tuple(str(c).upper() for c in fk["constrained_columns"]),
            fk.get("referred_schema") or schema,
            str(fk["referred_table"]).upper(),  # HANA dialect returns lowercase names
            tuple(str(c).upper() for c in fk["referred_columns"]),
        )

    def _expected_fk_tuple(fk: object) -> tuple:
        return (
            tuple(c.upper() for c in fk.columns),
            schema,  # expected FKs always target the current schema
            getattr(table_names, fk.referenced_table_key).upper(),
            tuple(c.upper() for c in fk.referenced_columns),
        )

    actual_sorted = sorted(_actual_fk_tuple(fk) for fk in actual_fks)
    expected_sorted = sorted(_expected_fk_tuple(fk) for fk in contract.foreign_keys)
    if actual_sorted != expected_sorted:
        raise _schema_error(contract.name, "foreign_keys")


# ---------------------------------------------------------------------------
# Full contract validation
# ---------------------------------------------------------------------------


def validate_table_contract(
    connection: Connection,
    schema: str,
    contract: TableContract,
    table_names: PersistenceTableNames,
) -> None:
    """Reject any existing table that differs from the complete logical contract.

    Uses the SQLAlchemy HANA inspector to reflect the live table structure and
    compares it against the contract for: ordered column names and attributes,
    primary key, unique constraints (ignoring generated names), and foreign
    keys (local columns, referenced schema, referenced physical table,
    referenced columns).

    Args:
        connection: An open SQLAlchemy ``Connection`` to the HANA instance.
        schema: The HANA schema containing the table.
        contract: The expected ``TableContract`` for this table.
        table_names: The active ``PersistenceTableNames``; used to resolve FK
            target table names so that suffixed test namespaces self-reference
            correctly.

    Raises:
        StartupConfigurationError: With code ``HANA_SCHEMA_INCOMPATIBLE`` and
            a message containing ``"incompatible"`` when any structural
            attribute differs from the contract.
    """
    inspector = inspect(connection)
    actual_columns = tuple(
        _reflected_column(c) for c in inspector.get_columns(contract.name, schema=schema)
    )
    expected_columns = tuple(_expected_column_tuple(c) for c in contract.columns)
    if actual_columns != expected_columns:
        raise _schema_error(contract.name, "columns")
    _validate_primary_key(inspector, schema, contract)
    _validate_unique_constraints(inspector, schema, contract)
    _validate_foreign_keys(inspector, schema, contract, table_names)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


def ensure_schema(
    engine: Engine,
    table_names: PersistenceTableNames = PersistenceTableNames(),
) -> str:
    """Create missing HANA tables and reject any incompatible existing contract.

    **Validation phase**: every already-existing table is validated against its
    complete contract before any DDL is attempted.  If any existing table is
    incompatible the error is propagated and no DDL runs.

    **Creation phase**: each missing table is created individually in dependency
    order (source_artifact → upload → bom_row → ecn → analysis_job →
    llm_attempt → analysis).  Each table is created in its own connection and
    immediately re-validated after creation.

    Note: HANA DDL may auto-commit, so tables created before a later DDL
    failure are intentionally left on disk.  On the next startup attempt those
    tables are found, validated as compatible, and skipped.  Do NOT wrap all
    DDL in a single ``engine.begin()`` or rely on rollback.

    Args:
        engine: A SQLAlchemy ``Engine`` connected to the target HANA instance.
        table_names: The ``PersistenceTableNames`` instance defining which
            physical tables to create or validate.  Defaults to the production
            names; pass a suffixed instance for isolated live tests.

    Returns:
        The current HANA schema name as a string.

    Raises:
        StartupConfigurationError: With code ``HANA_SCHEMA_INCOMPATIBLE``
            when an existing table's structure does not match the contract, or
            with code ``HANA_SCHEMA_INITIALIZATION_FAILED`` when a DDL
            statement could not be executed.
    """
    contracts = build_table_contracts(table_names)
    # These optional display and scope fields are approved additive alterations.
    # Validate the remaining contract before any DDL; never repair incompatible data.
    additive = {"material_master_row": "MATERIAL_GROUP_DESCRIPTION", "purchase_info_row": "SUPPLIER_NAME", "reference_dataset": "CONTEXT_JSON", "ecn": "AFFECTED_PLANTS_JSON"}
    additions = []

    # --- Validation phase: inspect every existing table BEFORE any DDL ------
    # All validation happens in a single connection; DDL follows only if every
    # existing table is compatible.
    with engine.connect() as connection:
        schema = current_schema(connection)
        existing = {
            contract.key: table_exists(connection, schema, contract.name)
            for contract in contracts
        }
        for contract in contracts:
            if existing[contract.key]:
                optional = additive.get(contract.key)
                columns = {str(c["name"]).upper() for c in inspect(connection).get_columns(contract.name, schema=schema)} if optional else set()
                if optional and optional not in columns:
                    validate_table_contract(connection, schema, replace(contract, columns=tuple(c for c in contract.columns if c.name != optional)), table_names)
                    additions.append((contract, optional))
                else:
                    validate_table_contract(connection, schema, contract, table_names)

    for contract, column in additions:
        with engine.connect() as connection:
            column_type = "NCLOB" if column == "CONTEXT_JSON" else "NVARCHAR(2000)"
            connection.execute(text(f'ALTER TABLE {quote_identifier(contract.name)} ADD ("{column}" {column_type})'))
            connection.commit()
            validate_table_contract(connection, schema, contract, table_names)

    # --- Creation phase: one connection per table, each committed individually
    # HANA DDL may auto-commit, so tables created before a later failure are
    # intentionally left for the next idempotent startup attempt.  Do NOT
    # wrap all DDL in one engine.begin() or claim rollback.
    for contract in contracts:
        if existing[contract.key]:
            continue
        try:
            with engine.connect() as connection:
                connection.execute(text(render_create_table(contract, table_names)))
                connection.commit()
        except SQLAlchemyError as exc:
            raise StartupConfigurationError(
                "HANA_SCHEMA_INITIALIZATION_FAILED",
                f"Could not initialize required table {contract.name!r}",
                status_code=500,
            ) from exc
        # Re-inspect the newly created table immediately to confirm it matches
        # the contract before proceeding to the next table.
        with engine.connect() as connection:
            validate_table_contract(connection, schema, contract, table_names)

    return schema
