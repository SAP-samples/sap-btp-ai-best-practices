"""Unit tests for app.persistence.schema — schema inspection and initialization.

Covers:
- ``current_schema``: SQL text, statement recording, blank-value rejection.
- ``validate_table_contract``: one compatible pass plus every mismatch class
  (extra column, missing column, name/order, type, length, precision, scale,
  nullability, PK, UQ, local FK columns, referenced suffixed table,
  referenced column).
- ``ensure_schema``: validate-before-create ordering, dependency order,
  no ``IF NOT EXISTS``, only-missing-tables creation, re-inspect per table,
  DDL failure wrapping, idempotent retry, no unsuffixed names in suffixed runs.

Run with::

    PYTHONPATH=api python -m pytest -q tests/unit/test_hana_schema.py
"""

from __future__ import annotations

import pytest

from app.ecn.errors import StartupConfigurationError
from app.persistence.schema import current_schema, ensure_schema, validate_table_contract
from app.persistence.table_contracts import PersistenceTableNames, build_table_contracts
from tests.fixtures.persistence import (
    FakeEngine,
    FakeInspector,
    RecordingConnection,
    _FakeSchemaInspector,
    build_inspector_for,
    col_to_reflected,
    make_fake_type,
)

# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recording_connection() -> RecordingConnection:
    """Return a fresh recording connection with an empty statement log.

    Returns:
        A ``RecordingConnection`` with no pre-queued scalars.
    """
    return RecordingConnection()


@pytest.fixture
def compatible_inspector(monkeypatch: pytest.MonkeyPatch) -> FakeInspector:
    """Return a FakeInspector seeded with correct data for source_artifact.

    Monkeypatches ``app.persistence.schema.inspect`` so that
    ``validate_table_contract`` uses this inspector for any connection.

    Returns:
        A ``FakeInspector`` with source_artifact columns, PK, no UQ, no FK.
    """
    contract = build_table_contracts()[0]  # source_artifact
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    return insp


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    """Return a FakeEngine wired with a FakeSchemaInspector using default names.

    Monkeypatches ``app.persistence.schema.inspect`` so that
    ``validate_table_contract`` inside ``ensure_schema`` uses the fake.

    Returns:
        A ``FakeEngine`` with no pre-existing tables and default schema.
    """
    engine = FakeEngine()
    monkeypatch.setattr(
        "app.persistence.schema.inspect",
        lambda _conn: _FakeSchemaInspector(engine, PersistenceTableNames()),
    )
    return engine


# ===========================================================================
# current_schema
# ===========================================================================


def test_current_schema_uses_hana_runtime_schema(
    recording_connection: RecordingConnection,
) -> None:
    """current_schema executes the exact DUMMY query and returns the schema name.

    Verifies that:
    * The SQL text is exactly ``SELECT CURRENT_SCHEMA FROM DUMMY``.
    * No parameters are passed.
    * The return value equals the queued scalar.
    """
    recording_connection.queue_scalar("APP_RUNTIME")
    assert current_schema(recording_connection) == "APP_RUNTIME"
    assert recording_connection.statements == [
        ("SELECT CURRENT_SCHEMA FROM DUMMY", {}),
    ]


def test_current_schema_strips_whitespace(recording_connection: RecordingConnection) -> None:
    """current_schema strips surrounding whitespace from the returned value.

    Returns:
        Passes when the returned schema name has no leading/trailing spaces.
    """
    recording_connection.queue_scalar("  APP_RUNTIME  ")
    assert current_schema(recording_connection) == "APP_RUNTIME"


def test_current_schema_raises_on_blank_value(
    recording_connection: RecordingConnection,
) -> None:
    """current_schema raises StartupConfigurationError for whitespace-only results.

    Verifies:
    * Code is ``HANA_SCHEMA_UNAVAILABLE``.
    * status_code is 500.
    """
    recording_connection.queue_scalar("   ")
    with pytest.raises(StartupConfigurationError) as exc_info:
        current_schema(recording_connection)
    assert exc_info.value.code == "HANA_SCHEMA_UNAVAILABLE"
    assert exc_info.value.status_code == 500


def test_current_schema_raises_on_none_value(
    recording_connection: RecordingConnection,
) -> None:
    """current_schema raises StartupConfigurationError when DUMMY returns None."""
    recording_connection.queue_scalar(None)
    with pytest.raises(StartupConfigurationError) as exc_info:
        current_schema(recording_connection)
    assert exc_info.value.code == "HANA_SCHEMA_UNAVAILABLE"


# ===========================================================================
# validate_table_contract — compatible pass
# ===========================================================================


def test_compatible_contract_passes_validation(
    compatible_inspector: FakeInspector,
) -> None:
    """A table whose reflected structure exactly matches the contract passes.

    Uses the source_artifact contract (no UQ, no FK) so the only checks are
    columns, PK, empty UQ set, and empty FK set.
    """
    contract = build_table_contracts()[0]
    # Should not raise
    validate_table_contract(
        compatible_inspector.connection,
        "APP_RUNTIME",
        contract,
        PersistenceTableNames(),
    )


# ===========================================================================
# validate_table_contract — column mismatch classes
# ===========================================================================


def test_extra_column_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """An extra column in the reflected table raises StartupConfigurationError.

    Appends one column to the seeded source_artifact column list before
    calling validate_table_contract.
    """
    contract = build_table_contracts()[0]
    compatible_inspector.columns[contract.name].append(
        {"name": "UNPLANNED", "type": make_fake_type("NVARCHAR", length=10), "nullable": True}
    )
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_missing_column_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """Fewer reflected columns than expected raises StartupConfigurationError."""
    contract = build_table_contracts()[0]
    # Remove the last column so actual has 10 instead of 11
    compatible_inspector.columns[contract.name].pop()
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_column_order_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """Columns in the wrong order raise StartupConfigurationError."""
    contract = build_table_contracts()[0]
    cols = compatible_inspector.columns[contract.name]
    # Swap first two columns to change order
    cols[0], cols[1] = cols[1], cols[0]
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_column_type_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """A type mismatch on any column raises StartupConfigurationError."""
    contract = build_table_contracts()[0]
    cols = compatible_inspector.columns[contract.name]
    # Replace BYTE_SIZE (BIGINT) with INTEGER
    for i, col in enumerate(cols):
        if col["name"] == "BYTE_SIZE":
            cols[i] = {"name": "BYTE_SIZE", "type": make_fake_type("INTEGER"), "nullable": False}
            break
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_nvarchar_length_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An NVARCHAR length mismatch raises StartupConfigurationError."""
    contract = build_table_contracts()[0]  # source_artifact, ARTIFACT_ID NVARCHAR(36)
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    cols = insp.columns[contract.name]
    # Change ARTIFACT_ID length from 36 to 100
    for i, col in enumerate(cols):
        if col["name"] == "ARTIFACT_ID":
            cols[i] = {
                "name": "ARTIFACT_ID",
                "type": make_fake_type("NVARCHAR", length=100),
                "nullable": False,
            }
            break
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_decimal_precision_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DECIMAL precision mismatch raises StartupConfigurationError.

    Uses bom_row which has QUANTITY DECIMAL(31,14).
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["bom_row"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    # Change QUANTITY precision from 31 to 10
    for i, col in enumerate(insp.columns[contract.name]):
        if col["name"] == "QUANTITY":
            insp.columns[contract.name][i] = {
                "name": "QUANTITY",
                "type": make_fake_type("DECIMAL", precision=10, scale=14),
                "nullable": True,
            }
            break
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_decimal_scale_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DECIMAL scale mismatch raises StartupConfigurationError.

    Uses bom_row which has QUANTITY DECIMAL(31,14).
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["bom_row"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    # Change QUANTITY scale from 14 to 4
    for i, col in enumerate(insp.columns[contract.name]):
        if col["name"] == "QUANTITY":
            insp.columns[contract.name][i] = {
                "name": "QUANTITY",
                "type": make_fake_type("DECIMAL", precision=31, scale=4),
                "nullable": True,
            }
            break
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_nullability_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """A nullability mismatch raises StartupConfigurationError.

    Flips ARTIFACT_ID from nullable=False to nullable=True.
    """
    contract = build_table_contracts()[0]
    for i, col in enumerate(compatible_inspector.columns[contract.name]):
        if col["name"] == "ARTIFACT_ID":
            compatible_inspector.columns[contract.name][i] = {
                "name": "ARTIFACT_ID",
                "type": make_fake_type("NVARCHAR", length=36),
                "nullable": True,  # contract says False
            }
            break
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


# ===========================================================================
# validate_table_contract — PK, UQ, FK mismatches
# ===========================================================================


def test_wrong_primary_key_is_an_incompatible_contract(
    compatible_inspector: FakeInspector,
) -> None:
    """A PK column mismatch raises StartupConfigurationError.

    Replaces the seeded PK column with a wrong name.
    """
    contract = build_table_contracts()[0]  # source_artifact PK: ARTIFACT_ID
    compatible_inspector._pks[contract.name] = ["WRONG_PK"]
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            compatible_inspector.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_unique_constraint_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A UQ column mismatch raises StartupConfigurationError.

    Uses upload which has UQs on REQUEST_ID and PUBLICATION_ORDER.
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["upload"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    # Replace PUBLICATION_ORDER UQ with a wrong column name
    insp._uqs[contract.name] = [["REQUEST_ID"], ["WRONG_COL"]]
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_fk_local_columns_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local FK column mismatch raises StartupConfigurationError.

    Uses upload which has ECN_ARTIFACT_ID → source_artifact.ARTIFACT_ID.
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["upload"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    # Change the first FK's local column to a wrong name
    insp._fks[contract.name][0]["constrained_columns"] = ["WRONG_LOCAL"]
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_wrong_fk_referenced_table_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A referenced table mismatch in a suffixed namespace raises StartupConfigurationError.

    Uses upload with a suffixed namespace.  The FK inspector returns the
    unsuffixed source_artifact name, but the contract expects the suffixed one.
    """
    suffix = "_TSCHEMA"
    names = PersistenceTableNames().with_suffix(suffix)
    contracts = {c.key: c for c in build_table_contracts(names)}
    contract = contracts["upload"]
    # Build inspector with the suffixed names (correct referenced tables)
    insp = build_inspector_for(contract, table_names=names)
    # Now replace referred_table of the first FK with the unsuffixed name
    insp._fks[contract.name][0]["referred_table"] = "ECN_IMPACT_SOURCE_ARTIFACT"
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            names,  # expects suffixed target
        )


def test_wrong_fk_referenced_column_is_an_incompatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A referenced column mismatch raises StartupConfigurationError.

    Uses upload's first FK which references source_artifact.ARTIFACT_ID.
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["upload"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    insp._fks[contract.name][0]["referred_columns"] = ["WRONG_REF_COL"]
    with pytest.raises(StartupConfigurationError, match="incompatible"):
        validate_table_contract(
            insp.connection,
            "APP_RUNTIME",
            contract,
            PersistenceTableNames(),
        )


def test_fk_order_insensitive_validation_passes_for_reversed_fks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_table_contract passes when HANA returns FKs in a different order.

    HANA's get_foreign_keys() reflection order is not guaranteed to match the
    contract's declaration order.  This is the regression test for the live-HANA
    failure where upload's three FKs (ECN_ARTIFACT_ID, BOM_ARTIFACT_ID,
    RULE_ARTIFACT_ID — all referencing source_artifact) were returned in
    reversed order, causing a positional zip comparison to false-report
    "foreign_keys incompatible" on the idempotent startup re-validation.

    Uses upload (3 FKs to source_artifact) with the FK list fully reversed.
    The contract is correct; only the reflection order differs.
    """
    contracts = {c.key: c for c in build_table_contracts()}
    contract = contracts["upload"]
    insp = build_inspector_for(contract)
    monkeypatch.setattr("app.persistence.schema.inspect", lambda _conn: insp)
    # Reverse the FK list to simulate HANA returning FKs in non-declaration order
    insp._fks[contract.name] = list(reversed(insp._fks[contract.name]))
    # Should NOT raise — comparison is now order-insensitive
    validate_table_contract(
        insp.connection,
        "APP_RUNTIME",
        contract,
        PersistenceTableNames(),
    )


# ===========================================================================
# ensure_schema — ordering and DDL properties
# ===========================================================================


def test_ensure_schema_validates_existing_before_creating_missing(
    fake_engine: FakeEngine,
) -> None:
    """ensure_schema validates all existing tables before running any DDL.

    Marks source_artifact as existing and incompatible.  Asserts the error
    is raised without any DDL statement having been executed.
    """
    fake_engine.schema = "APP_RUNTIME"
    fake_engine.existing_tables = {PersistenceTableNames().source_artifact}
    fake_engine.mark_incompatible(PersistenceTableNames().source_artifact, "FILE_NAME")
    with pytest.raises(StartupConfigurationError):
        ensure_schema(fake_engine)
    assert fake_engine.ddl_statements == []


def test_ensure_schema_creates_in_dependency_order_without_if_not_exists(
    fake_engine: FakeEngine,
) -> None:
    """ensure_schema creates tables in creation order with no IF NOT EXISTS.

    When no tables exist, every table is created in catalog dependency order.
    """
    fake_engine.schema = "APP_RUNTIME"
    assert ensure_schema(fake_engine) == "APP_RUNTIME"
    created = [stmt.split('"')[1] for stmt in fake_engine.ddl_statements]
    assert created == [contract.name for contract in build_table_contracts()]
    assert all("IF NOT EXISTS" not in stmt for stmt in fake_engine.ddl_statements)


def test_ensure_schema_creates_rdf_projection_after_ecn(
    fake_engine: FakeEngine,
) -> None:
    """A fresh schema creates ECN_IMPACT_RDF_PROJECTION after its ecn dependency.

    The RDF projection-status table carries an FK to ecn, so the bootstrap must
    emit its CREATE COLUMN TABLE after the ecn table when starting from an empty
    schema.
    """
    fake_engine.schema = "APP_RUNTIME"
    ensure_schema(fake_engine)
    created = [stmt.split('"')[1] for stmt in fake_engine.ddl_statements]
    names = PersistenceTableNames()
    assert names.rdf_projection in created
    assert created.index(names.ecn) < created.index(names.rdf_projection)


def test_ensure_schema_only_creates_missing_tables(
    fake_engine: FakeEngine,
) -> None:
    """ensure_schema skips tables that already exist.

    Pre-seeds source_artifact and upload as existing.  The fake schema
    inspector returns correct data for them. Asserts only remaining tables
    appear in DDL statements.
    """
    names = PersistenceTableNames()
    fake_engine.existing_tables = {names.source_artifact, names.upload}
    ensure_schema(fake_engine)
    created = [stmt.split('"')[1] for stmt in fake_engine.ddl_statements]
    assert names.source_artifact not in created
    assert names.upload not in created
    assert len(created) == len(build_table_contracts()) - 2


def test_ensure_schema_reinspects_each_created_table(
    fake_engine: FakeEngine,
) -> None:
    """ensure_schema calls validate_table_contract for every newly created table.

    After each CREATE TABLE succeeds, ensure_schema calls validate_table_contract
    which triggers get_columns on the new table.  Asserts inspection_count > 0
    for every table after a full run.
    """
    fake_engine.schema = "APP_RUNTIME"
    ensure_schema(fake_engine)
    for name in vars(PersistenceTableNames()).values():
        assert fake_engine.inspection_count.get(name, 0) >= 1, (
            f"Table {name!r} was never re-inspected after creation"
        )


def test_ensure_schema_ddl_failure_raises_initialization_failed(
    fake_engine: FakeEngine,
) -> None:
    """A SQLAlchemyError during DDL is re-raised as HANA_SCHEMA_INITIALIZATION_FAILED."""
    fake_engine.mark_ddl_failure(PersistenceTableNames().source_artifact)
    with pytest.raises(StartupConfigurationError) as exc_info:
        ensure_schema(fake_engine)
    assert exc_info.value.code == "HANA_SCHEMA_INITIALIZATION_FAILED"
    assert exc_info.value.status_code == 500


def test_ensure_schema_idempotent_when_all_tables_exist(
    fake_engine: FakeEngine,
) -> None:
    """ensure_schema returns the schema without any DDL when all tables are compatible.

    Pre-seeds all seven tables as existing; the fake inspector returns correct
    data.  Asserts no DDL is executed and the schema string is returned.
    """
    fake_engine.existing_tables = set(vars(PersistenceTableNames()).values())
    result = ensure_schema(fake_engine)
    assert result == "APP_RUNTIME"
    assert fake_engine.ddl_statements == []


def test_ensure_schema_prior_committed_table_accepted_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table auto-committed before a failure is accepted as compatible on retry.

    Simulates partial creation: source_artifact is created (auto-committed),
    then upload DDL fails.  On the second call source_artifact exists and is
    compatible, so only the remaining six tables need creation.
    """
    engine = FakeEngine()
    monkeypatch.setattr(
        "app.persistence.schema.inspect",
        lambda _conn: _FakeSchemaInspector(engine, PersistenceTableNames()),
    )
    names = PersistenceTableNames()
    engine.mark_ddl_failure(names.upload)

    # First attempt: source_artifact is created, then upload fails
    with pytest.raises(StartupConfigurationError) as exc_info:
        ensure_schema(engine)
    assert exc_info.value.code == "HANA_SCHEMA_INITIALIZATION_FAILED"
    assert names.source_artifact in engine._created_tables

    # Second attempt: source_artifact now exists + is compatible; no DDL failure
    engine._ddl_failures.discard(names.upload)
    result = ensure_schema(engine)
    assert result == "APP_RUNTIME"
    # source_artifact was not re-created (already in existing on second run).
    # Use startswith to match only the CREATE statement, not FK REFERENCES clauses
    # in other tables' DDL that also mention source_artifact by name.
    source_artifact_creates = [
        s for s in engine.ddl_statements
        if s.startswith(f'CREATE COLUMN TABLE "{names.source_artifact}"')
    ]
    assert len(source_artifact_creates) == 1  # created only once across both runs


def test_ensure_schema_no_unsuffixed_name_with_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DDL statements contain only suffixed table names when a suffix is provided.

    Ensures that no default production table name appears in any DDL when
    ensure_schema is called with a suffixed PersistenceTableNames.
    """
    suffix = "_TTEST"
    names = PersistenceTableNames().with_suffix(suffix)
    engine = FakeEngine()
    monkeypatch.setattr(
        "app.persistence.schema.inspect",
        lambda _conn: _FakeSchemaInspector(engine, names),
    )
    ensure_schema(engine, names)
    default_names = vars(PersistenceTableNames()).values()
    for stmt in engine.ddl_statements:
        for default_name in default_names:
            assert f'"{default_name}"' not in stmt, (
                f"Unsuffixed name {default_name!r} found in DDL: {stmt[:80]!r}"
            )
