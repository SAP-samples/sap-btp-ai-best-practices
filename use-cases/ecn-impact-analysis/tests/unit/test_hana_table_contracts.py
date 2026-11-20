"""Comprehensive unit tests for the HANA persistence catalog and engine factory.

Covers:
- The exact ordered persistence catalog (creation order, column counts, column
  attributes, PKs, unique constraints, FKs).
- Deterministic DDL rendering with ``CREATE COLUMN TABLE``, identifier quoting,
  no ``IF NOT EXISTS``, and no bindable parameters.
- Suffixed test-namespace isolation ensuring FK targets never reference the
  default physical names.
- ``create_hana_engine`` credential validation (incomplete creds, bad port,
  bad encrypt, whitespace) without leaking sensitive values.
- ``to_hana_timestamp`` and ``from_hana_timestamp`` UTC round-trip.

Run with:
    PYTHONPATH=api python -m pytest -q tests/unit/test_hana_table_contracts.py
"""

import re
from datetime import datetime, timezone

import pytest

from app.ecn.errors import StartupConfigurationError
from app.persistence.hana import create_hana_engine, from_hana_timestamp, to_hana_timestamp
from app.persistence.table_contracts import (
    ColumnContract,
    ForeignKeyContract,
    PersistenceTableNames,
    TableContract,
    build_table_contracts,
    render_create_table,
)
from tests.fixtures.persistence import FakeEngineFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contracts_by_key() -> dict[str, TableContract]:
    """Return all persistence contracts keyed by their logical key string."""
    return {c.key: c for c in build_table_contracts()}


def _get_col(contract: TableContract, name: str) -> ColumnContract:
    """Return the named column from a contract, raising AssertionError if absent.

    Args:
        contract: The table contract to search.
        name: The exact column name to find.

    Returns:
        The matching ``ColumnContract``.

    Raises:
        AssertionError: When no column with the given name exists.
    """
    for col in contract.columns:
        if col.name == name:
            return col
    raise AssertionError(f"Column {name!r} not found in table {contract.key!r}")


# ---------------------------------------------------------------------------
# Step 1 tests: catalog creation order, column counts, suffixed FK isolation,
# and identifier guard (required by the brief verbatim)
# ---------------------------------------------------------------------------


def test_catalog_has_exact_tables_column_counts_and_creation_order() -> None:
    """Assert all application tables use dependency-safe creation order."""
    contracts = build_table_contracts()
    assert [contract.key for contract in contracts] == [
        "source_artifact", "reference_dataset", "upload", "bom_row",
        "conditional_rule_row", "material_master_row", "purchase_info_row", "purchase_info_price_row", "stock_row",
        "upload_dataset", "ecn", "manufacturing_feasibility_assessment", "rdf_projection",
        "analysis_job", "llm_attempt", "provider_call", "analysis_stage", "analysis", "generation_batch", "batch_job", "review",
        "plant_mapping", "app_setting",
    ]
    assert {contract.key: len(contract.columns) for contract in contracts} == {
        "generation_batch": 4, "batch_job": 2, "review": 9,
        "plant_mapping": 4, "app_setting": 3,
        "source_artifact": 11,
        "reference_dataset": 13,
        "upload": 13,
        "bom_row": 35,
        "conditional_rule_row": 8,
        "material_master_row": 13,
        "purchase_info_row": 13,
        "purchase_info_price_row": 14,
        "stock_row": 8,
        "upload_dataset": 4,
        "ecn": 22,
        "manufacturing_feasibility_assessment": 10,
        "rdf_projection": 15,
        "analysis_job": 13,
        "llm_attempt": 21,
        "provider_call": 21,
        "analysis_stage": 9,
        "analysis": 18,
    }
    assert sum(len(contract.columns) for contract in contracts) == 283


def test_reference_dataset_type_has_exactly_six_maintained_values() -> None:
    """Six source types remain after retiring historical/rule Make/Buy ingestion."""
    from app.ecn.reference_data.models import ReferenceDatasetType
    assert {item.value for item in ReferenceDatasetType} == {
        "STOCK",
        "BOM",
        "CONDITIONAL_RULES",
        "MATERIAL_MASTER",
        "PURCHASE_INFO_RECORDS",
        "PURCHASE_INFO_PRICES",
    }


def test_contract_keys_include_only_maintained_row_tables() -> None:
    """Every row-table key in the catalog must belong to a maintained source type."""
    maintained_row_keys = {
        "bom_row",
        "conditional_rule_row",
        "material_master_row",
        "purchase_info_row",
        "purchase_info_price_row",
        "stock_row",
    }
    contract_keys = {c.key for c in build_table_contracts()}
    actual_row_keys = {k for k in contract_keys if k.endswith("_row")}
    unexpected = actual_row_keys - maintained_row_keys
    assert not unexpected, (
        f"Unexpected row table keys found in catalog (retired types?): {unexpected}"
    )


def test_suffixed_catalog_never_references_default_tables() -> None:
    """Suffixed contracts must resolve FKs to suffixed targets, not default names."""
    names = PersistenceTableNames().with_suffix("_A1B2C3D4")
    ddl = "\n".join(render_create_table(c, names) for c in build_table_contracts(names))
    for physical_name in vars(names).values():
        assert f'"{physical_name}"' in ddl
    assert 'REFERENCES "ECN_IMPACT_SOURCE_ARTIFACT"' not in ddl


def test_identifier_guard_rejects_sql_text() -> None:
    """PersistenceTableNames must reject identifiers that contain SQL-injection characters."""
    with pytest.raises(ValueError, match="table identifier"):
        PersistenceTableNames(source_artifact='ECN_IMPACT_X";DROP TABLE X')


# ---------------------------------------------------------------------------
# Per-table column, PK, UQ, FK assertions
# ---------------------------------------------------------------------------


def test_source_artifact_columns_and_constraints() -> None:
    """Assert all 11 source_artifact columns with exact types and nullability."""
    sa = _contracts_by_key()["source_artifact"]

    # Ordered column names
    assert tuple(c.name for c in sa.columns) == (
        "ARTIFACT_ID", "ARTIFACT_KIND", "FILE_NAME", "MEDIA_TYPE", "BYTE_SIZE",
        "SHA256", "CONTENT_BYTES", "NORMALIZED_CONTENT", "PARSE_STATE",
        "PARSE_ERROR_CODE", "CREATED_AT",
    )

    # Column attribute assertions
    assert _get_col(sa, "ARTIFACT_ID") == ColumnContract("ARTIFACT_ID", "NVARCHAR", False, length=36)
    assert _get_col(sa, "ARTIFACT_KIND") == ColumnContract("ARTIFACT_KIND", "NVARCHAR", False, length=40)
    assert _get_col(sa, "FILE_NAME") == ColumnContract("FILE_NAME", "NVARCHAR", False, length=500)
    assert _get_col(sa, "MEDIA_TYPE") == ColumnContract("MEDIA_TYPE", "NVARCHAR", False, length=200)
    assert _get_col(sa, "BYTE_SIZE") == ColumnContract("BYTE_SIZE", "BIGINT", False)
    assert _get_col(sa, "SHA256") == ColumnContract("SHA256", "NVARCHAR", False, length=64)
    assert _get_col(sa, "CONTENT_BYTES") == ColumnContract("CONTENT_BYTES", "BLOB", False)
    assert _get_col(sa, "NORMALIZED_CONTENT") == ColumnContract("NORMALIZED_CONTENT", "NCLOB", True)
    assert _get_col(sa, "PARSE_STATE") == ColumnContract("PARSE_STATE", "NVARCHAR", False, length=32)
    assert _get_col(sa, "PARSE_ERROR_CODE") == ColumnContract("PARSE_ERROR_CODE", "NVARCHAR", True, length=80)
    assert _get_col(sa, "CREATED_AT") == ColumnContract("CREATED_AT", "TIMESTAMP", False)

    # Structural constraints
    assert sa.primary_key == ("ARTIFACT_ID",)
    assert sa.unique_constraints == ()
    assert sa.foreign_keys == ()


def test_upload_columns_and_constraints() -> None:
    """Assert all 13 upload columns, two single-column UQs, and three source FKs."""
    u = _contracts_by_key()["upload"]

    assert tuple(c.name for c in u.columns) == (
        "UPLOAD_ID", "REQUEST_ID", "PUBLICATION_ORDER", "ECN_ARTIFACT_ID",
        "BOM_ARTIFACT_ID", "RULE_ARTIFACT_ID", "BOM_REUSED", "IMPORTED_ECN_COUNT",
        "CREATED_BY", "CREATED_AT", "PARSER_VERSION", "DETERMINISTIC_LOGIC_VERSION",
        "RULE_EVALUATOR_VERSION",
    )

    assert _get_col(u, "UPLOAD_ID") == ColumnContract("UPLOAD_ID", "NVARCHAR", False, length=36)
    assert _get_col(u, "REQUEST_ID") == ColumnContract("REQUEST_ID", "NVARCHAR", False, length=36)
    assert _get_col(u, "PUBLICATION_ORDER") == ColumnContract("PUBLICATION_ORDER", "BIGINT", False)
    assert _get_col(u, "ECN_ARTIFACT_ID") == ColumnContract("ECN_ARTIFACT_ID", "NVARCHAR", False, length=36)
    assert _get_col(u, "BOM_ARTIFACT_ID") == ColumnContract("BOM_ARTIFACT_ID", "NVARCHAR", False, length=36)
    assert _get_col(u, "RULE_ARTIFACT_ID") == ColumnContract("RULE_ARTIFACT_ID", "NVARCHAR", True, length=36)
    assert _get_col(u, "BOM_REUSED") == ColumnContract("BOM_REUSED", "TINYINT", False)
    assert _get_col(u, "IMPORTED_ECN_COUNT") == ColumnContract("IMPORTED_ECN_COUNT", "INTEGER", False)
    assert _get_col(u, "CREATED_BY") == ColumnContract("CREATED_BY", "NVARCHAR", True, length=128)
    assert _get_col(u, "CREATED_AT") == ColumnContract("CREATED_AT", "TIMESTAMP", False)
    assert _get_col(u, "PARSER_VERSION") == ColumnContract("PARSER_VERSION", "NVARCHAR", False, length=40)
    assert _get_col(u, "DETERMINISTIC_LOGIC_VERSION") == ColumnContract(
        "DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(u, "RULE_EVALUATOR_VERSION") == ColumnContract(
        "RULE_EVALUATOR_VERSION", "NVARCHAR", False, length=40
    )

    assert u.primary_key == ("UPLOAD_ID",)
    assert u.unique_constraints == (("REQUEST_ID",), ("PUBLICATION_ORDER",))
    assert u.foreign_keys == (
        ForeignKeyContract(("ECN_ARTIFACT_ID",), "source_artifact", ("ARTIFACT_ID",)),
        ForeignKeyContract(("BOM_ARTIFACT_ID",), "source_artifact", ("ARTIFACT_ID",)),
        ForeignKeyContract(("RULE_ARTIFACT_ID",), "source_artifact", ("ARTIFACT_ID",)),
    )


def test_bom_row_columns_and_constraints() -> None:
    """Assert all 35 bom_row columns including DECIMAL and DATE, plus self-FK."""
    br = _contracts_by_key()["bom_row"]

    assert tuple(c.name for c in br.columns) == (
        "BOM_ROW_ID", "ARTIFACT_ID", "BOM_KIND", "SHEET_NAME", "SOURCE_ROW_NUMBER",
        "PARENT_BOM_ROW_ID", "DOTTED_LEVEL", "HIERARCHY_PATH", "MAIN_VALUE",
        "SUB_VALUE", "VC_VALUE", "LEVEL_4_VALUE", "LEVEL_5_VALUE", "LEVEL_6_VALUE",
        "ITEM_VALUE", "POSITION_VALUE", "MATERIAL_NUMBER", "PARENT_MATERIAL_NUMBER",
        "COMPONENT_NUMBER", "CHANGE_NUMBER", "CHANGE_NUMBER_TO", "DESCRIPTION_TEXT",
        "QUANTITY", "UNIT", "PLANT", "ALTERNATIVE", "VALID_FROM_RAW",
        "VALID_FROM_DATE", "VALID_TO_RAW", "VALID_TO_DATE", "ROW_CLASSIFICATION",
        "IS_DOCUMENTATION", "IS_MIGRATION_DATE", "ROW_JSON", "QUALITY_JSON",
    )

    assert _get_col(br, "BOM_ROW_ID") == ColumnContract("BOM_ROW_ID", "NVARCHAR", False, length=36)
    assert _get_col(br, "ARTIFACT_ID") == ColumnContract("ARTIFACT_ID", "NVARCHAR", False, length=36)
    assert _get_col(br, "BOM_KIND") == ColumnContract("BOM_KIND", "NVARCHAR", False, length=32)
    assert _get_col(br, "SHEET_NAME") == ColumnContract("SHEET_NAME", "NVARCHAR", False, length=500)
    assert _get_col(br, "SOURCE_ROW_NUMBER") == ColumnContract("SOURCE_ROW_NUMBER", "INTEGER", False)
    assert _get_col(br, "PARENT_BOM_ROW_ID") == ColumnContract(
        "PARENT_BOM_ROW_ID", "NVARCHAR", True, length=36
    )
    assert _get_col(br, "DOTTED_LEVEL") == ColumnContract("DOTTED_LEVEL", "NVARCHAR", True, length=100)
    assert _get_col(br, "HIERARCHY_PATH") == ColumnContract("HIERARCHY_PATH", "NCLOB", True)
    assert _get_col(br, "MAIN_VALUE") == ColumnContract("MAIN_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "SUB_VALUE") == ColumnContract("SUB_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "VC_VALUE") == ColumnContract("VC_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "LEVEL_4_VALUE") == ColumnContract("LEVEL_4_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "LEVEL_5_VALUE") == ColumnContract("LEVEL_5_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "LEVEL_6_VALUE") == ColumnContract("LEVEL_6_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "ITEM_VALUE") == ColumnContract("ITEM_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "POSITION_VALUE") == ColumnContract("POSITION_VALUE", "NVARCHAR", True, length=200)
    assert _get_col(br, "MATERIAL_NUMBER") == ColumnContract(
        "MATERIAL_NUMBER", "NVARCHAR", True, length=200
    )
    assert _get_col(br, "PARENT_MATERIAL_NUMBER") == ColumnContract(
        "PARENT_MATERIAL_NUMBER", "NVARCHAR", True, length=200
    )
    assert _get_col(br, "COMPONENT_NUMBER") == ColumnContract(
        "COMPONENT_NUMBER", "NVARCHAR", True, length=200
    )
    assert _get_col(br, "CHANGE_NUMBER") == ColumnContract("CHANGE_NUMBER", "NVARCHAR", True, length=200)
    assert _get_col(br, "CHANGE_NUMBER_TO") == ColumnContract(
        "CHANGE_NUMBER_TO", "NVARCHAR", True, length=200
    )
    assert _get_col(br, "DESCRIPTION_TEXT") == ColumnContract("DESCRIPTION_TEXT", "NCLOB", True)
    assert _get_col(br, "QUANTITY") == ColumnContract(
        "QUANTITY", "DECIMAL", True, precision=31, scale=14
    )
    assert _get_col(br, "UNIT") == ColumnContract("UNIT", "NVARCHAR", True, length=40)
    assert _get_col(br, "PLANT") == ColumnContract("PLANT", "NVARCHAR", True, length=80)
    assert _get_col(br, "ALTERNATIVE") == ColumnContract("ALTERNATIVE", "NVARCHAR", True, length=200)
    assert _get_col(br, "VALID_FROM_RAW") == ColumnContract("VALID_FROM_RAW", "NVARCHAR", True, length=100)
    assert _get_col(br, "VALID_FROM_DATE") == ColumnContract("VALID_FROM_DATE", "DATE", True)
    assert _get_col(br, "VALID_TO_RAW") == ColumnContract("VALID_TO_RAW", "NVARCHAR", True, length=100)
    assert _get_col(br, "VALID_TO_DATE") == ColumnContract("VALID_TO_DATE", "DATE", True)
    assert _get_col(br, "ROW_CLASSIFICATION") == ColumnContract(
        "ROW_CLASSIFICATION", "NVARCHAR", False, length=40
    )
    assert _get_col(br, "IS_DOCUMENTATION") == ColumnContract("IS_DOCUMENTATION", "TINYINT", False)
    assert _get_col(br, "IS_MIGRATION_DATE") == ColumnContract("IS_MIGRATION_DATE", "TINYINT", False)
    assert _get_col(br, "ROW_JSON") == ColumnContract("ROW_JSON", "NCLOB", False)
    assert _get_col(br, "QUALITY_JSON") == ColumnContract("QUALITY_JSON", "NCLOB", False)

    assert br.primary_key == ("BOM_ROW_ID",)
    assert br.unique_constraints == (("ARTIFACT_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),)
    assert br.foreign_keys == (
        ForeignKeyContract(("ARTIFACT_ID",), "source_artifact", ("ARTIFACT_ID",)),
        ForeignKeyContract(("PARENT_BOM_ROW_ID",), "bom_row", ("BOM_ROW_ID",)),
    )


def test_ecn_columns_and_constraints() -> None:
    """Assert all 22 ecn columns and the (UPLOAD_ID, SOURCE_ORDINAL) UQ."""
    ec = _contracts_by_key()["ecn"]

    assert tuple(c.name for c in ec.columns) == (
        "ECN_ROW_ID", "UPLOAD_ID", "SOURCE_ORDINAL", "ECN_NUMBER", "TITLE",
        "DESCRIPTION_SUMMARY", "DESCRIPTION_TEXT", "AFFECTED_ITEM_COUNT",
        "RAW_ECN_MATCH_COUNT", "EBOM_MATCH_COUNT", "EBOM_CONTEXT_COUNT",
        "MBOM_MATCH_COUNT", "WARNING_COUNT", "SOURCE_JSON", "CHANGE_CANDIDATES_JSON",
        "EVIDENCE_JSON", "EVIDENCE_HASH", "PARSER_VERSION",
        "DETERMINISTIC_LOGIC_VERSION", "RULE_EVALUATOR_VERSION", "CREATED_AT",
        "AFFECTED_PLANTS_JSON",
    )
    assert _get_col(ec, "AFFECTED_PLANTS_JSON") == ColumnContract(
        "AFFECTED_PLANTS_JSON", "NVARCHAR", True, length=2000
    )

    assert _get_col(ec, "ECN_ROW_ID") == ColumnContract("ECN_ROW_ID", "NVARCHAR", False, length=36)
    assert _get_col(ec, "UPLOAD_ID") == ColumnContract("UPLOAD_ID", "NVARCHAR", False, length=36)
    assert _get_col(ec, "SOURCE_ORDINAL") == ColumnContract("SOURCE_ORDINAL", "INTEGER", False)
    assert _get_col(ec, "ECN_NUMBER") == ColumnContract("ECN_NUMBER", "NVARCHAR", True, length=80)
    assert _get_col(ec, "TITLE") == ColumnContract("TITLE", "NVARCHAR", True, length=500)
    assert _get_col(ec, "DESCRIPTION_SUMMARY") == ColumnContract(
        "DESCRIPTION_SUMMARY", "NVARCHAR", True, length=2000
    )
    assert _get_col(ec, "DESCRIPTION_TEXT") == ColumnContract("DESCRIPTION_TEXT", "NCLOB", True)
    assert _get_col(ec, "AFFECTED_ITEM_COUNT") == ColumnContract("AFFECTED_ITEM_COUNT", "INTEGER", False)
    assert _get_col(ec, "RAW_ECN_MATCH_COUNT") == ColumnContract("RAW_ECN_MATCH_COUNT", "INTEGER", False)
    assert _get_col(ec, "EBOM_MATCH_COUNT") == ColumnContract("EBOM_MATCH_COUNT", "INTEGER", False)
    assert _get_col(ec, "EBOM_CONTEXT_COUNT") == ColumnContract("EBOM_CONTEXT_COUNT", "INTEGER", False)
    assert _get_col(ec, "MBOM_MATCH_COUNT") == ColumnContract("MBOM_MATCH_COUNT", "INTEGER", False)
    assert _get_col(ec, "WARNING_COUNT") == ColumnContract("WARNING_COUNT", "INTEGER", False)
    assert _get_col(ec, "SOURCE_JSON") == ColumnContract("SOURCE_JSON", "NCLOB", False)
    assert _get_col(ec, "CHANGE_CANDIDATES_JSON") == ColumnContract(
        "CHANGE_CANDIDATES_JSON", "NCLOB", False
    )
    assert _get_col(ec, "EVIDENCE_JSON") == ColumnContract("EVIDENCE_JSON", "NCLOB", False)
    assert _get_col(ec, "EVIDENCE_HASH") == ColumnContract("EVIDENCE_HASH", "NVARCHAR", False, length=64)
    assert _get_col(ec, "PARSER_VERSION") == ColumnContract("PARSER_VERSION", "NVARCHAR", False, length=40)
    assert _get_col(ec, "DETERMINISTIC_LOGIC_VERSION") == ColumnContract(
        "DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(ec, "RULE_EVALUATOR_VERSION") == ColumnContract(
        "RULE_EVALUATOR_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(ec, "CREATED_AT") == ColumnContract("CREATED_AT", "TIMESTAMP", False)

    assert ec.primary_key == ("ECN_ROW_ID",)
    assert ec.unique_constraints == (("UPLOAD_ID", "SOURCE_ORDINAL"),)
    assert ec.foreign_keys == (
        ForeignKeyContract(("UPLOAD_ID",), "upload", ("UPLOAD_ID",)),
    )


def test_rdf_projection_columns_and_constraints() -> None:
    """Assert all 15 rdf_projection columns, PK on ECN_ROW_ID, and FK to ecn.

    The per-ECN RDF projection-status table is the relational anchor for the
    RDF projection lifecycle: nine NOT NULL expected-count columns, five
    nullable verification/error columns, and two nullable timestamps.  STATUS
    is a plain NVARCHAR(20) (no check constraint, per the bootstrap's
    introspection limits).
    """
    rp = _contracts_by_key()["rdf_projection"]

    assert tuple(c.name for c in rp.columns) == (
        "ECN_ROW_ID", "SOURCE_SHA256", "PROJECTION_VERSION", "STATUS",
        "EXPECTED_OBJECT_COUNT", "EXPECTED_ARRAY_COUNT", "EXPECTED_SCALAR_COUNT",
        "EXPECTED_NULL_COUNT", "EXPECTED_NODE_COUNT", "VERIFIED_NODE_COUNT",
        "VERIFIED_TRIPLE_COUNT", "ERROR_CODE", "ERROR_MESSAGE", "LAST_ATTEMPT_AT",
        "CURRENT_AT",
    )

    assert _get_col(rp, "ECN_ROW_ID") == ColumnContract("ECN_ROW_ID", "NVARCHAR", False, length=36)
    assert _get_col(rp, "SOURCE_SHA256") == ColumnContract("SOURCE_SHA256", "NVARCHAR", False, length=64)
    assert _get_col(rp, "PROJECTION_VERSION") == ColumnContract(
        "PROJECTION_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(rp, "STATUS") == ColumnContract("STATUS", "NVARCHAR", False, length=20)
    assert _get_col(rp, "EXPECTED_OBJECT_COUNT") == ColumnContract("EXPECTED_OBJECT_COUNT", "BIGINT", False)
    assert _get_col(rp, "EXPECTED_ARRAY_COUNT") == ColumnContract("EXPECTED_ARRAY_COUNT", "BIGINT", False)
    assert _get_col(rp, "EXPECTED_SCALAR_COUNT") == ColumnContract("EXPECTED_SCALAR_COUNT", "BIGINT", False)
    assert _get_col(rp, "EXPECTED_NULL_COUNT") == ColumnContract("EXPECTED_NULL_COUNT", "BIGINT", False)
    assert _get_col(rp, "EXPECTED_NODE_COUNT") == ColumnContract("EXPECTED_NODE_COUNT", "BIGINT", False)
    assert _get_col(rp, "VERIFIED_NODE_COUNT") == ColumnContract("VERIFIED_NODE_COUNT", "BIGINT", True)
    assert _get_col(rp, "VERIFIED_TRIPLE_COUNT") == ColumnContract("VERIFIED_TRIPLE_COUNT", "BIGINT", True)
    assert _get_col(rp, "ERROR_CODE") == ColumnContract("ERROR_CODE", "NVARCHAR", True, length=80)
    assert _get_col(rp, "ERROR_MESSAGE") == ColumnContract("ERROR_MESSAGE", "NVARCHAR", True, length=500)
    assert _get_col(rp, "LAST_ATTEMPT_AT") == ColumnContract("LAST_ATTEMPT_AT", "TIMESTAMP", True)
    assert _get_col(rp, "CURRENT_AT") == ColumnContract("CURRENT_AT", "TIMESTAMP", True)

    assert rp.primary_key == ("ECN_ROW_ID",)
    assert rp.unique_constraints == ()
    assert rp.foreign_keys == (
        ForeignKeyContract(("ECN_ROW_ID",), "ecn", ("ECN_ROW_ID",)),
    )


def test_rdf_projection_is_created_after_ecn_and_before_independent_tables() -> None:
    """rdf_projection follows its ecn FK target and precedes tables not depending on it."""
    keys = [contract.key for contract in build_table_contracts()]
    # Its only FK target (ecn) must be created first.
    assert keys.index("ecn") < keys.index("rdf_projection")
    # It must precede the analysis/workspace tables that do not depend on it.
    for independent_key in (
        "analysis_job", "llm_attempt", "provider_call", "analysis",
        "generation_batch", "batch_job", "review",
    ):
        assert keys.index("rdf_projection") < keys.index(independent_key)


def test_analysis_job_columns_and_constraints() -> None:
    """Assert all 13 analysis_job columns, single ECN_ROW_ID UQ, and ECN FK."""
    aj = _contracts_by_key()["analysis_job"]

    assert tuple(c.name for c in aj.columns) == (
        "JOB_ID", "ECN_ROW_ID", "STATE", "ATTEMPT_COUNT", "CREATED_AT",
        "STARTED_AT", "FINISHED_AT", "PROMPT_VERSION", "OUTPUT_SCHEMA_VERSION",
        "MODEL_NAME", "MODEL_CONFIG_JSON", "ERROR_CODE", "ERROR_MESSAGE",
    )

    assert _get_col(aj, "JOB_ID") == ColumnContract("JOB_ID", "NVARCHAR", False, length=36)
    assert _get_col(aj, "ECN_ROW_ID") == ColumnContract("ECN_ROW_ID", "NVARCHAR", False, length=36)
    assert _get_col(aj, "STATE") == ColumnContract("STATE", "NVARCHAR", False, length=32)
    assert _get_col(aj, "ATTEMPT_COUNT") == ColumnContract("ATTEMPT_COUNT", "INTEGER", False)
    assert _get_col(aj, "CREATED_AT") == ColumnContract("CREATED_AT", "TIMESTAMP", False)
    assert _get_col(aj, "STARTED_AT") == ColumnContract("STARTED_AT", "TIMESTAMP", True)
    assert _get_col(aj, "FINISHED_AT") == ColumnContract("FINISHED_AT", "TIMESTAMP", True)
    assert _get_col(aj, "PROMPT_VERSION") == ColumnContract("PROMPT_VERSION", "NVARCHAR", False, length=40)
    assert _get_col(aj, "OUTPUT_SCHEMA_VERSION") == ColumnContract(
        "OUTPUT_SCHEMA_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(aj, "MODEL_NAME") == ColumnContract("MODEL_NAME", "NVARCHAR", False, length=200)
    assert _get_col(aj, "MODEL_CONFIG_JSON") == ColumnContract("MODEL_CONFIG_JSON", "NCLOB", False)
    assert _get_col(aj, "ERROR_CODE") == ColumnContract("ERROR_CODE", "NVARCHAR", True, length=80)
    assert _get_col(aj, "ERROR_MESSAGE") == ColumnContract("ERROR_MESSAGE", "NVARCHAR", True, length=2000)

    assert aj.primary_key == ("JOB_ID",)
    assert aj.unique_constraints == (("ECN_ROW_ID",),)
    assert aj.foreign_keys == (
        ForeignKeyContract(("ECN_ROW_ID",), "ecn", ("ECN_ROW_ID",)),
    )


def test_llm_attempt_columns_and_constraints() -> None:
    """Assert all 21 llm_attempt columns including token BIGINT nullables."""
    la = _contracts_by_key()["llm_attempt"]

    assert tuple(c.name for c in la.columns) == (
        "ATTEMPT_ID", "JOB_ID", "ATTEMPT_NUMBER", "CORRELATION_ID", "PROVIDER",
        "MODEL_NAME", "MODEL_DEPLOYMENT", "LLM_ENDPOINT", "STARTED_AT",
        "FINISHED_AT", "PROVIDER_OUTCOME", "VALIDATION_OUTCOME", "ERROR_CODE",
        "LATENCY_MS", "INPUT_TOKENS", "CACHE_READ_INPUT_TOKENS",
        "CACHE_WRITE_INPUT_TOKENS", "INPUT_TOTAL_TOKENS", "OUTPUT_TOKENS",
        "TOTAL_TOKENS", "USAGE_AVAILABLE",
    )

    assert _get_col(la, "ATTEMPT_ID") == ColumnContract("ATTEMPT_ID", "NVARCHAR", False, length=36)
    assert _get_col(la, "JOB_ID") == ColumnContract("JOB_ID", "NVARCHAR", False, length=36)
    assert _get_col(la, "ATTEMPT_NUMBER") == ColumnContract("ATTEMPT_NUMBER", "INTEGER", False)
    assert _get_col(la, "CORRELATION_ID") == ColumnContract("CORRELATION_ID", "NVARCHAR", False, length=100)
    assert _get_col(la, "PROVIDER") == ColumnContract("PROVIDER", "NVARCHAR", False, length=100)
    assert _get_col(la, "MODEL_NAME") == ColumnContract("MODEL_NAME", "NVARCHAR", False, length=200)
    assert _get_col(la, "MODEL_DEPLOYMENT") == ColumnContract(
        "MODEL_DEPLOYMENT", "NVARCHAR", True, length=200
    )
    assert _get_col(la, "LLM_ENDPOINT") == ColumnContract("LLM_ENDPOINT", "NVARCHAR", False, length=100)
    assert _get_col(la, "STARTED_AT") == ColumnContract("STARTED_AT", "TIMESTAMP", False)
    assert _get_col(la, "FINISHED_AT") == ColumnContract("FINISHED_AT", "TIMESTAMP", True)
    assert _get_col(la, "PROVIDER_OUTCOME") == ColumnContract(
        "PROVIDER_OUTCOME", "NVARCHAR", False, length=32
    )
    assert _get_col(la, "VALIDATION_OUTCOME") == ColumnContract(
        "VALIDATION_OUTCOME", "NVARCHAR", False, length=32
    )
    assert _get_col(la, "ERROR_CODE") == ColumnContract("ERROR_CODE", "NVARCHAR", True, length=80)
    assert _get_col(la, "LATENCY_MS") == ColumnContract("LATENCY_MS", "BIGINT", True)
    assert _get_col(la, "INPUT_TOKENS") == ColumnContract("INPUT_TOKENS", "BIGINT", True)
    assert _get_col(la, "CACHE_READ_INPUT_TOKENS") == ColumnContract(
        "CACHE_READ_INPUT_TOKENS", "BIGINT", True
    )
    assert _get_col(la, "CACHE_WRITE_INPUT_TOKENS") == ColumnContract(
        "CACHE_WRITE_INPUT_TOKENS", "BIGINT", True
    )
    assert _get_col(la, "INPUT_TOTAL_TOKENS") == ColumnContract("INPUT_TOTAL_TOKENS", "BIGINT", True)
    assert _get_col(la, "OUTPUT_TOKENS") == ColumnContract("OUTPUT_TOKENS", "BIGINT", True)
    assert _get_col(la, "TOTAL_TOKENS") == ColumnContract("TOTAL_TOKENS", "BIGINT", True)
    assert _get_col(la, "USAGE_AVAILABLE") == ColumnContract("USAGE_AVAILABLE", "TINYINT", False)

    assert la.primary_key == ("ATTEMPT_ID",)
    assert la.unique_constraints == (("JOB_ID", "ATTEMPT_NUMBER"),)
    assert la.foreign_keys == (
        ForeignKeyContract(("JOB_ID",), "analysis_job", ("JOB_ID",)),
    )


def test_analysis_columns_and_constraints() -> None:
    """Assert all 18 analysis columns, two single-column UQs, and three FKs."""
    an = _contracts_by_key()["analysis"]

    assert tuple(c.name for c in an.columns) == (
        "ANALYSIS_ID", "ECN_ROW_ID", "JOB_ID", "SUCCESSFUL_ATTEMPT_ID",
        "COMPLETENESS_STATE", "CONTENT_JSON", "EVIDENCE_HASH", "PROMPT_VERSION",
        "MODEL_NAME", "MODEL_DEPLOYMENT", "DETERMINISTIC_LOGIC_VERSION",
        "RULE_EVALUATOR_VERSION", "OUTPUT_SCHEMA_VERSION", "CREATED_AT",
        "DISCOVERY_STATUS", "DISCOVERY_ERROR_CODE", "DISCOVERY_HASH",
        "DISCOVERED_EVIDENCE_JSON",
    )

    assert _get_col(an, "ANALYSIS_ID") == ColumnContract("ANALYSIS_ID", "NVARCHAR", False, length=36)
    assert _get_col(an, "ECN_ROW_ID") == ColumnContract("ECN_ROW_ID", "NVARCHAR", False, length=36)
    assert _get_col(an, "JOB_ID") == ColumnContract("JOB_ID", "NVARCHAR", False, length=36)
    assert _get_col(an, "SUCCESSFUL_ATTEMPT_ID") == ColumnContract(
        "SUCCESSFUL_ATTEMPT_ID", "NVARCHAR", False, length=36
    )
    assert _get_col(an, "COMPLETENESS_STATE") == ColumnContract(
        "COMPLETENESS_STATE", "NVARCHAR", False, length=40
    )
    assert _get_col(an, "CONTENT_JSON") == ColumnContract("CONTENT_JSON", "NCLOB", False)
    assert _get_col(an, "EVIDENCE_HASH") == ColumnContract("EVIDENCE_HASH", "NVARCHAR", False, length=64)
    assert _get_col(an, "PROMPT_VERSION") == ColumnContract("PROMPT_VERSION", "NVARCHAR", False, length=40)
    assert _get_col(an, "MODEL_NAME") == ColumnContract("MODEL_NAME", "NVARCHAR", False, length=200)
    assert _get_col(an, "MODEL_DEPLOYMENT") == ColumnContract(
        "MODEL_DEPLOYMENT", "NVARCHAR", True, length=200
    )
    assert _get_col(an, "DETERMINISTIC_LOGIC_VERSION") == ColumnContract(
        "DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(an, "RULE_EVALUATOR_VERSION") == ColumnContract(
        "RULE_EVALUATOR_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(an, "OUTPUT_SCHEMA_VERSION") == ColumnContract(
        "OUTPUT_SCHEMA_VERSION", "NVARCHAR", False, length=40
    )
    assert _get_col(an, "CREATED_AT") == ColumnContract("CREATED_AT", "TIMESTAMP", False)
    assert _get_col(an, "DISCOVERY_STATUS") == ColumnContract(
        "DISCOVERY_STATUS", "NVARCHAR", False, length=20
    )
    assert _get_col(an, "DISCOVERY_ERROR_CODE") == ColumnContract(
        "DISCOVERY_ERROR_CODE", "NVARCHAR", True, length=80
    )
    assert _get_col(an, "DISCOVERY_HASH") == ColumnContract(
        "DISCOVERY_HASH", "NVARCHAR", False, length=64
    )
    assert _get_col(an, "DISCOVERED_EVIDENCE_JSON") == ColumnContract(
        "DISCOVERED_EVIDENCE_JSON", "NCLOB", False
    )

    assert an.primary_key == ("ANALYSIS_ID",)
    assert an.unique_constraints == (("ECN_ROW_ID",), ("JOB_ID",))
    assert an.foreign_keys == (
        ForeignKeyContract(("ECN_ROW_ID",), "ecn", ("ECN_ROW_ID",)),
        ForeignKeyContract(("JOB_ID",), "analysis_job", ("JOB_ID",)),
        ForeignKeyContract(("SUCCESSFUL_ATTEMPT_ID",), "llm_attempt", ("ATTEMPT_ID",)),
    )


# ---------------------------------------------------------------------------
# DDL format assertions
# ---------------------------------------------------------------------------


def test_ddl_format_for_all_tables() -> None:
    """Assert DDL uses CREATE COLUMN TABLE, quotes identifiers, has no IF NOT EXISTS."""
    names = PersistenceTableNames()
    contracts = build_table_contracts(names)
    for contract in contracts:
        ddl = render_create_table(contract, names)
        assert ddl.startswith("CREATE COLUMN TABLE"), (
            f"Table {contract.key!r} DDL must start with CREATE COLUMN TABLE"
        )
        assert "IF NOT EXISTS" not in ddl, (
            f"Table {contract.key!r} DDL must not contain IF NOT EXISTS"
        )
        # Physical table name must be quoted
        assert f'"{contract.name}"' in ddl, (
            f"Table {contract.key!r} DDL must quote the table name"
        )
        # No bindable parameters
        assert "?" not in ddl, f"Table {contract.key!r} DDL must not contain bind placeholders"
        assert ":param" not in ddl


def test_ddl_source_artifact_is_correct() -> None:
    """Assert source_artifact DDL contains expected clauses in rendered form."""
    names = PersistenceTableNames()
    contracts_map = {c.key: c for c in build_table_contracts(names)}
    ddl = render_create_table(contracts_map["source_artifact"], names)
    assert '"ARTIFACT_ID" NVARCHAR(36) NOT NULL' in ddl
    assert '"NORMALIZED_CONTENT" NCLOB' in ddl
    assert '"PARSE_ERROR_CODE" NVARCHAR(80)' in ddl
    assert 'PRIMARY KEY ("ARTIFACT_ID")' in ddl
    # No explicit NULL keyword (HANA default is nullable; NOT NULL is explicit)
    assert "IF NOT EXISTS" not in ddl


def test_ddl_bom_row_decimal_and_date_rendered_correctly() -> None:
    """Assert DECIMAL and DATE columns render without spurious parentheses."""
    names = PersistenceTableNames()
    contracts_map = {c.key: c for c in build_table_contracts(names)}
    ddl = render_create_table(contracts_map["bom_row"], names)
    assert '"QUANTITY" DECIMAL(31,14)' in ddl
    assert '"VALID_FROM_DATE" DATE' in ddl
    assert '"VALID_TO_DATE" DATE' in ddl


def test_ddl_upload_contains_foreign_key_to_source_artifact() -> None:
    """Assert upload DDL references source_artifact by its physical name."""
    names = PersistenceTableNames()
    contracts_map = {c.key: c for c in build_table_contracts(names)}
    ddl = render_create_table(contracts_map["upload"], names)
    expected_target = f'REFERENCES "{names.source_artifact}"'
    assert expected_target in ddl


def test_ddl_bom_row_self_fk_references_bom_row() -> None:
    """Assert bom_row DDL contains a self-referential FK for PARENT_BOM_ROW_ID."""
    names = PersistenceTableNames()
    contracts_map = {c.key: c for c in build_table_contracts(names)}
    ddl = render_create_table(contracts_map["bom_row"], names)
    assert f'REFERENCES "{names.bom_row}"' in ddl


def test_default_table_names_match_expected_values() -> None:
    """Assert default PersistenceTableNames uses the exact specified defaults."""
    names = PersistenceTableNames()
    assert names.source_artifact == "ECN_IMPACT_SOURCE_ARTIFACT"
    assert names.upload == "ECN_IMPACT_UPLOAD"
    assert names.bom_row == "ECN_IMPACT_BOM_ROW"
    assert names.ecn == "ECN_IMPACT_ECN"
    assert names.analysis_job == "ECN_IMPACT_ANALYSIS_JOB"
    assert names.llm_attempt == "ECN_IMPACT_LLM_ATTEMPT"
    assert names.analysis == "ECN_IMPACT_ANALYSIS"
    assert names.reference_dataset == "ECN_IMPACT_REFERENCE_DATASET"
    assert names.upload_dataset == "ECN_IMPACT_UPLOAD_DATASET"
    assert names.provider_call == "ECN_IMPACT_PROVIDER_CALL"
    assert names.analysis_stage == "ECN_IMPACT_ANALYSIS_STAGE"
    assert names.material_master_row == "ECN_IMPACT_MATERIAL_MASTER_ROW"
    assert names.purchase_info_row == "ECN_IMPACT_PURCHASE_INFO_ROW"
    assert names.purchase_info_price_row == "ECN_IMPACT_PURCHASE_INFO_PRICE_ROW"
    assert names.manufacturing_feasibility_assessment == "ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT"
    assert names.conditional_rule_row == "ECN_IMPACT_CONDITIONAL_RULE_ROW"
    assert names.rdf_projection == "ECN_IMPACT_RDF_PROJECTION"
    assert names.plant_mapping == "ECN_IMPACT_PLANT_MAPPING"
    assert names.app_setting == "ECN_IMPACT_APP_SETTING"
    # PersistenceTableNames must contain exactly the set of maintained fields.
    # Any field not in this set indicates a retired type was not fully cleaned up.
    expected_fields = {
        "source_artifact", "upload", "bom_row", "ecn", "analysis_job",
        "llm_attempt", "analysis", "reference_dataset", "upload_dataset",
        "provider_call", "analysis_stage", "material_master_row", "purchase_info_row", "purchase_info_price_row", "manufacturing_feasibility_assessment",
        "conditional_rule_row", "batch_job", "generation_batch", "review",
        "stock_row", "rdf_projection", "plant_mapping", "app_setting",
    }
    actual_fields = set(names.__dataclass_fields__)
    assert actual_fields == expected_fields, (
        f"PersistenceTableNames has unexpected fields: {actual_fields - expected_fields}"
    )


def test_with_suffix_appends_to_all_seven_names() -> None:
    """Assert with_suffix produces valid suffixed names for all seven tables."""
    default = PersistenceTableNames()
    suffixed = default.with_suffix("_TEST")
    assert suffixed.source_artifact == "ECN_IMPACT_SOURCE_ARTIFACT_TEST"
    assert suffixed.upload == "ECN_IMPACT_UPLOAD_TEST"
    assert suffixed.bom_row == "ECN_IMPACT_BOM_ROW_TEST"
    assert suffixed.ecn == "ECN_IMPACT_ECN_TEST"
    assert suffixed.analysis_job == "ECN_IMPACT_ANALYSIS_JOB_TEST"
    assert suffixed.llm_attempt == "ECN_IMPACT_LLM_ATTEMPT_TEST"
    assert suffixed.analysis == "ECN_IMPACT_ANALYSIS_TEST"
    assert suffixed.reference_dataset == "ECN_IMPACT_REFERENCE_DATASET_TEST"
    assert suffixed.purchase_info_price_row == "ECN_IMPACT_PURCHASE_INFO_PRICE_ROW_TEST"
    assert suffixed.provider_call == "ECN_IMPACT_PROVIDER_CALL_TEST"


def test_purchase_info_price_contract_keeps_source_price_and_matching_fields() -> None:
    """Price overlays retain source identity without mutating purchase setup rows."""
    contract = next(
        item for item in build_table_contracts() if item.key == "purchase_info_price_row"
    )
    assert {column.name for column in contract.columns} >= {
        "DATASET_ID",
        "SOURCE_ROW_NUMBER",
        "INFO_RECORD",
        "MATERIAL_NUMBER",
        "PLANT_CODE",
        "NET_PRICE",
        "CURRENCY",
        "PRICE_PER",
        "ORDER_PRICE_UNIT",
    }
    assert contract.foreign_keys[0].referenced_table_key == "reference_dataset"



    """Create metadata before source rows and attempts before provider calls."""
    contracts = build_table_contracts()
    keys = [contract.key for contract in contracts]
    assert keys.index("source_artifact") < keys.index("reference_dataset")
    assert keys.index("reference_dataset") < keys.index("material_master_row")
    assert keys.index("upload") < keys.index("upload_dataset")
    assert keys.index("llm_attempt") < keys.index("provider_call")
    provider = next(item for item in contracts if item.key == "provider_call")
    assert {column.name for column in provider.columns} >= {
        "CALL_ID",
        "ATTEMPT_ID",
        "STAGE",
        "HTTP_STATUS",
        "PROVIDER_CODE",
        "PROVIDER_REQUEST_ID",
        "RETRY_AFTER_SECONDS",
        "LATENCY_MS",
        "USAGE_AVAILABLE",
    }


def test_reference_dataset_contract_retains_scope_activation_and_counts() -> None:
    """Store immutable publication metadata needed by Settings and provenance."""
    contract = next(
        item for item in build_table_contracts() if item.key == "reference_dataset"
    )
    assert {column.name for column in contract.columns} >= {
        "DATASET_ID",
        "DATASET_TYPE",
        "ARTIFACT_ID",
        "PUBLICATION_ORDER",
        "SCOPE_CODE",
        "SCOPE_NAME",
        "ROW_COUNT",
        "WARNING_COUNT",
        "WARNING_JSON",
        "CREATED_AT",
    }


def test_with_suffix_rejects_invalid_suffix() -> None:
    """Assert with_suffix raises ValueError for suffixes with lowercase or special chars."""
    default = PersistenceTableNames()
    with pytest.raises(ValueError):
        default.with_suffix("_lowercase")
    with pytest.raises(ValueError):
        default.with_suffix("")
    with pytest.raises(ValueError):
        # Exceeds 16-character limit
        default.with_suffix("_TOOLONGSUFFIX12345")


def test_build_table_contracts_uses_provided_names() -> None:
    """Assert build_table_contracts uses the given table_names for physical names."""
    names = PersistenceTableNames().with_suffix("_MYTEST")
    contracts = build_table_contracts(names)
    by_key = {c.key: c for c in contracts}
    assert by_key["source_artifact"].name == "ECN_IMPACT_SOURCE_ARTIFACT_MYTEST"
    assert by_key["upload"].name == "ECN_IMPACT_UPLOAD_MYTEST"
    assert by_key["analysis"].name == "ECN_IMPACT_ANALYSIS_MYTEST"


# ---------------------------------------------------------------------------
# Engine construction and timestamp helpers (Step 5 of the brief)
# ---------------------------------------------------------------------------


def test_engine_rejects_incomplete_credentials_without_exposing_values() -> None:
    """Incomplete credentials raise StartupConfigurationError without leaking values."""
    with pytest.raises(StartupConfigurationError, match="HANA credentials are incomplete") as caught:
        create_hana_engine({"HANA_ADDRESS": "secret-host", "HANA_PORT": "443"})
    assert "secret-host" not in str(caught.value)


def test_hana_timestamp_boundary_is_utc_and_round_trips() -> None:
    """to_hana_timestamp strips tz; from_hana_timestamp restores UTC tz."""
    aware = datetime(2026, 9, 14, 10, 30, tzinfo=timezone.utc)
    stored = to_hana_timestamp(aware)
    assert stored == datetime(2026, 9, 14, 10, 30)
    assert stored.tzinfo is None
    assert from_hana_timestamp(stored) == aware


def test_engine_creates_hana_url_with_credentials_and_pool_pre_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid credentials produce a hana:// URL with pool_pre_ping=True."""
    factory = FakeEngineFactory()
    monkeypatch.setattr("app.persistence.hana.create_engine", factory)
    create_hana_engine({
        "HANA_ADDRESS": "db.example.com",
        "HANA_PORT": "443",
        "HANA_USER": "DBUSER",
        "HANA_PASSWORD": "secret-pass",
        "HANA_ENCRYPT": "false",
    })
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call.url.drivername == "hana"
    assert call.url.password == "secret-pass"
    assert call.kwargs.get("pool_pre_ping") is True
    assert call.url.query["encrypt"] == "false"


def test_engine_encrypt_true_maps_to_true_query_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HANA_ENCRYPT=true results in encrypt=true query parameter."""
    factory = FakeEngineFactory()
    monkeypatch.setattr("app.persistence.hana.create_engine", factory)
    create_hana_engine({
        "HANA_ADDRESS": "db.example.com",
        "HANA_PORT": "443",
        "HANA_USER": "DBUSER",
        "HANA_PASSWORD": "pass",
        "HANA_ENCRYPT": "true",
    })
    assert factory.calls[0].url.query["encrypt"] == "true"


def test_engine_encrypt_aliases_1_yes_map_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HANA_ENCRYPT aliases '1' and 'yes' resolve to encrypt=true."""
    factory = FakeEngineFactory()
    monkeypatch.setattr("app.persistence.hana.create_engine", factory)
    for alias in ("1", "yes"):
        factory.calls.clear()
        create_hana_engine({
            "HANA_ADDRESS": "h",
            "HANA_PORT": "443",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
            "HANA_ENCRYPT": alias,
        })
        assert factory.calls[0].url.query["encrypt"] == "true", f"alias {alias!r} failed"


def test_engine_rejects_noninteger_port() -> None:
    """A non-integer HANA_PORT raises StartupConfigurationError."""
    with pytest.raises(StartupConfigurationError, match="HANA port is invalid"):
        create_hana_engine({
            "HANA_ADDRESS": "h",
            "HANA_PORT": "notanumber",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
        })


def test_engine_rejects_out_of_range_port() -> None:
    """Ports outside 1-65535 raise StartupConfigurationError."""
    with pytest.raises(StartupConfigurationError, match="HANA port is invalid"):
        create_hana_engine({
            "HANA_ADDRESS": "h",
            "HANA_PORT": "0",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
        })
    with pytest.raises(StartupConfigurationError, match="HANA port is invalid"):
        create_hana_engine({
            "HANA_ADDRESS": "h",
            "HANA_PORT": "65536",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
        })


def test_engine_rejects_whitespace_only_required_values() -> None:
    """Required fields that contain only whitespace are treated as missing."""
    with pytest.raises(StartupConfigurationError, match="HANA credentials are incomplete"):
        create_hana_engine({
            "HANA_ADDRESS": "   ",
            "HANA_PORT": "443",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
        })


def test_engine_rejects_invalid_encrypt_value() -> None:
    """An HANA_ENCRYPT value outside the allowed set raises StartupConfigurationError."""
    with pytest.raises(StartupConfigurationError, match="HANA encryption setting is invalid"):
        create_hana_engine({
            "HANA_ADDRESS": "h",
            "HANA_PORT": "443",
            "HANA_USER": "u",
            "HANA_PASSWORD": "p",
            "HANA_ENCRYPT": "maybe",
        })


def test_to_hana_timestamp_rejects_naive_datetime() -> None:
    """to_hana_timestamp raises ValueError for a naive (tz-unaware) datetime."""
    naive = datetime(2026, 9, 14, 10, 30)
    with pytest.raises(ValueError, match="timezone-aware"):
        to_hana_timestamp(naive)
