"""Immutable HANA table catalog for the ECN impact pipeline.

This module defines the complete HANA structural contract, allowlisted
physical table identifiers, and a deterministic DDL renderer.

The tables, in creation order, are:

    1. source_artifact   — raw uploaded files and their normalized content
    2. reference_dataset — immutable Settings dataset publications
    3. upload            — a single ECN intake batch
    4. source rows       — normalized BOM and reference data rows
    5. upload_dataset    — pinned ECN-to-reference provenance
    6. ecn               — normalized ECN rows linked to an upload batch
    7. analysis_job      — per-ECN analysis lifecycle record
    8. llm_attempt       — one aggregate analysis attempt
    9. provider_call     — per-stage provider diagnostics and usage
   10. analysis          — final accepted analysis result

``ForeignKeyContract.referenced_table_key`` is a logical attribute key
(e.g. ``"source_artifact"``), never a physical default table name.  This
allows suffixed test namespaces created via ``PersistenceTableNames.with_suffix``
to reference their own suffixed tables rather than the default production names.

Usage::

    from app.persistence.table_contracts import (
        build_table_contracts,
        PersistenceTableNames,
        render_create_table,
    )

    names = PersistenceTableNames()
    contracts = build_table_contracts(names)
    for contract in contracts:
        print(render_create_table(contract, names))
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Identifier allowlist
# ---------------------------------------------------------------------------

_TABLE_IDENTIFIER = re.compile(r"^ECN_IMPACT_[A-Z0-9_]{1,110}$")

# Suffix characters must be uppercase letters, digits, or underscores, and
# 1-16 characters long so the resulting full name stays within the 110-char
# cap after "ECN_IMPACT_".
_SUFFIX_PATTERN = re.compile(r"^[A-Z0-9_]{1,16}$")


def quote_identifier(identifier: str) -> str:
    """Quote one allowlisted ECN impact table identifier for HANA SQL.

    Validates ``identifier`` against the ``^ECN_IMPACT_[A-Z0-9_]{1,110}$``
    allowlist and wraps it in double-quotes for safe inclusion in DDL.

    Args:
        identifier: A physical table name that must match the allowlist pattern.

    Returns:
        The identifier enclosed in double-quotes, e.g. ``'"ECN_IMPACT_UPLOAD"'``.

    Raises:
        ValueError: If ``identifier`` does not match the allowlist pattern,
            indicating an unsafe or malformed table identifier.
    """
    if _TABLE_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(
            f"invalid ECN impact table identifier: {identifier!r} does not match "
            r"^ECN_IMPACT_[A-Z0-9_]{1,110}$"
        )
    return f'"{identifier}"'


# ---------------------------------------------------------------------------
# Structural contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnContract:
    """Describe one required HANA column.

    Used as the unit of schema verification: the persistence layer checks that
    every live table column matches the corresponding ``ColumnContract`` by
    name, type, and nullability before the application starts.

    Args:
        name: Exact column name as stored in HANA (uppercase).
        hana_type: HANA base type string, e.g. ``"NVARCHAR"``, ``"BIGINT"``,
            ``"TIMESTAMP"``, ``"NCLOB"``, ``"BLOB"``, ``"TINYINT"``,
            ``"INTEGER"``, ``"DATE"``, ``"DECIMAL"``.
        nullable: ``True`` when the column permits NULL values.
        length: Character length for ``NVARCHAR`` columns; ``None`` otherwise.
        precision: Total digit precision for ``DECIMAL`` columns; ``None``
            otherwise.
        scale: Decimal digit scale for ``DECIMAL`` columns; ``None`` otherwise.
    """

    name: str
    hana_type: str
    nullable: bool
    length: int | None = None
    precision: int | None = None
    scale: int | None = None


@dataclass(frozen=True)
class ForeignKeyContract:
    """Describe a local-to-referenced column tuple using a table-name key.

    ``referenced_table_key`` is a logical attribute key matching one of the
    seven field names on ``PersistenceTableNames`` (e.g. ``"source_artifact"``),
    NOT a physical default table name.  This ensures that suffixed test
    namespaces produced by ``PersistenceTableNames.with_suffix`` self-reference
    correctly without encoding hardcoded default names.

    Args:
        columns: Tuple of local column names forming the FK.
        referenced_table_key: Attribute key on ``PersistenceTableNames`` that
            identifies the referenced table (e.g. ``"upload"``).
        referenced_columns: Tuple of column names in the referenced table.
    """

    columns: tuple[str, ...]
    referenced_table_key: str
    referenced_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableContract:
    """Describe one complete versioned application table.

    Captures the full structural expectation for a HANA column table:
    ordered columns, primary key, uniqueness constraints, and foreign keys.
    The ``key`` is the logical Python-facing name; ``name`` is the physical
    HANA table name resolved from ``PersistenceTableNames``.

    Args:
        key: Logical table key, e.g. ``"source_artifact"``.
        name: Physical HANA table name, e.g. ``"ECN_IMPACT_SOURCE_ARTIFACT"``.
        columns: Ordered tuple of column contracts for this table.
        primary_key: Ordered tuple of column names forming the primary key.
        unique_constraints: Tuple of column-name tuples, each defining one
            unique constraint.
        foreign_keys: Tuple of FK contracts for this table.
    """

    key: str
    name: str
    columns: tuple[ColumnContract, ...]
    primary_key: tuple[str, ...]
    unique_constraints: tuple[tuple[str, ...], ...]
    foreign_keys: tuple[ForeignKeyContract, ...]


@dataclass(frozen=True)
class PersistenceTableNames:
    """Hold the only table identifiers repository SQL may interpolate.

    All fields default to the production physical names.  Call
    ``with_suffix`` to produce an isolated test namespace in which every
    FK target also carries the same suffix, so live integration tests do
    not pollute or depend on production tables.

    Args:
        source_artifact: Physical name of the source-artifact table.
        upload: Physical name of the upload table.
        bom_row: Physical name of the BOM-row table.
        ecn: Physical name of the ECN table.
        analysis_job: Physical name of the analysis-job table.
        llm_attempt: Physical name of the LLM-attempt table.
        analysis: Physical name of the analysis table.

    Raises:
        ValueError: In ``__post_init__`` if any field value does not match
            ``^ECN_IMPACT_[A-Z0-9_]{1,110}$``.
    """

    source_artifact: str = "ECN_IMPACT_SOURCE_ARTIFACT"
    upload: str = "ECN_IMPACT_UPLOAD"
    bom_row: str = "ECN_IMPACT_BOM_ROW"
    ecn: str = "ECN_IMPACT_ECN"
    analysis_job: str = "ECN_IMPACT_ANALYSIS_JOB"
    llm_attempt: str = "ECN_IMPACT_LLM_ATTEMPT"
    analysis: str = "ECN_IMPACT_ANALYSIS"
    reference_dataset: str = "ECN_IMPACT_REFERENCE_DATASET"
    upload_dataset: str = "ECN_IMPACT_UPLOAD_DATASET"
    conditional_rule_row: str = "ECN_IMPACT_CONDITIONAL_RULE_ROW"
    material_master_row: str = "ECN_IMPACT_MATERIAL_MASTER_ROW"
    purchase_info_row: str = "ECN_IMPACT_PURCHASE_INFO_ROW"
    purchase_info_price_row: str = "ECN_IMPACT_PURCHASE_INFO_PRICE_ROW"
    manufacturing_feasibility_assessment: str = "ECN_IMPACT_MANUFACTURING_FEASIBILITY_ASSESSMENT"
    batch_job: str = "ECN_IMPACT_BATCH_JOB"
    generation_batch: str = "ECN_IMPACT_GENERATION_BATCH"
    review: str = "ECN_IMPACT_REVIEW"
    stock_row: str = "ECN_IMPACT_STOCK_ROW"
    provider_call: str = "ECN_IMPACT_PROVIDER_CALL"
    analysis_stage: str = "ECN_IMPACT_ANALYSIS_STAGE"
    rdf_projection: str = "ECN_IMPACT_RDF_PROJECTION"
    plant_mapping: str = "ECN_IMPACT_PLANT_MAPPING"
    app_setting: str = "ECN_IMPACT_APP_SETTING"

    def __post_init__(self) -> None:
        """Validate all table identifiers against the allowlist pattern.

        Calls ``quote_identifier`` on each field value to enforce the
        ``^ECN_IMPACT_[A-Z0-9_]{1,110}$`` pattern.  Raises ``ValueError``
        with "table identifier" in the message if any name is invalid.

        Raises:
            ValueError: If any of the seven physical table names fails the
                identifier allowlist check.
        """
        for field_name in self.__dataclass_fields__:
            quote_identifier(getattr(self, field_name))

    def with_suffix(self, suffix: str) -> "PersistenceTableNames":
        """Return an isolated table namespace for a live integration test.

        Appends ``suffix`` to all seven table names so that FK targets within
        the same namespace remain consistent.  The suffix must match
        ``[A-Z0-9_]{1,16}`` to keep the resulting names within the allowlist.

        Args:
            suffix: An uppercase alphanumeric/underscore string of 1-16
                characters to append to every table name.

        Returns:
            A new ``PersistenceTableNames`` instance with each field
            suffixed; ``__post_init__`` validates the resulting names.

        Raises:
            ValueError: If ``suffix`` contains lowercase letters, special
                characters, is empty, or exceeds 16 characters.
        """
        if _SUFFIX_PATTERN.fullmatch(suffix) is None:
            raise ValueError(
                f"table name suffix {suffix!r} must match [A-Z0-9_]{{1,16}}"
            )
        return PersistenceTableNames(
            source_artifact=self.source_artifact + suffix,
            upload=self.upload + suffix,
            bom_row=self.bom_row + suffix,
            ecn=self.ecn + suffix,
            analysis_job=self.analysis_job + suffix,
            llm_attempt=self.llm_attempt + suffix,
            analysis=self.analysis + suffix,
            reference_dataset=self.reference_dataset + suffix,
            upload_dataset=self.upload_dataset + suffix,
            conditional_rule_row=self.conditional_rule_row + suffix,
            material_master_row=self.material_master_row + suffix,
            purchase_info_row=self.purchase_info_row + suffix,
            purchase_info_price_row=self.purchase_info_price_row + suffix,
            manufacturing_feasibility_assessment=self.manufacturing_feasibility_assessment + suffix,
            batch_job=self.batch_job + suffix,
            generation_batch=self.generation_batch + suffix,
            review=self.review + suffix,
            stock_row=self.stock_row + suffix,
            provider_call=self.provider_call + suffix,
            analysis_stage=self.analysis_stage + suffix,
            rdf_projection=self.rdf_projection + suffix,
            plant_mapping=self.plant_mapping + suffix,
            app_setting=self.app_setting + suffix,
        )


# ---------------------------------------------------------------------------
# Internal column helper
# ---------------------------------------------------------------------------


def _col(
    name: str,
    hana_type: str,
    nullable: bool = False,
    *,
    length: int | None = None,
    precision: int | None = None,
    scale: int | None = None,
) -> ColumnContract:
    """Build a ColumnContract from named arguments.

    All non-nullable columns omit the ``nullable`` argument (defaulting to
    ``False``).  Columns that allow NULL pass ``nullable=True`` as the third
    positional argument.

    Args:
        name: Column name (uppercase).
        hana_type: HANA type string.
        nullable: ``True`` to allow NULL values.
        length: Character length for NVARCHAR.
        precision: Total digit precision for DECIMAL.
        scale: Decimal digit scale for DECIMAL.

    Returns:
        A frozen ``ColumnContract`` instance.
    """
    return ColumnContract(
        name=name,
        hana_type=hana_type,
        nullable=nullable,
        length=length,
        precision=precision,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Per-table contract builders
# ---------------------------------------------------------------------------


def _source_artifact_contract(names: PersistenceTableNames) -> TableContract:
    """Build the source_artifact table contract from the given name set.

    Stores raw uploaded file bytes and their normalized text representation.
    11 columns; PK on ARTIFACT_ID; no UQ or FK constraints.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the source-artifact table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("ARTIFACT_ID", "NVARCHAR", length=36),
        _col("ARTIFACT_KIND", "NVARCHAR", length=40),
        _col("FILE_NAME", "NVARCHAR", length=500),
        _col("MEDIA_TYPE", "NVARCHAR", length=200),
        _col("BYTE_SIZE", "BIGINT"),
        _col("SHA256", "NVARCHAR", length=64),
        _col("CONTENT_BYTES", "BLOB"),
        _col("NORMALIZED_CONTENT", "NCLOB", True),
        _col("PARSE_STATE", "NVARCHAR", length=32),
        _col("PARSE_ERROR_CODE", "NVARCHAR", True, length=80),
        _col("CREATED_AT", "TIMESTAMP"),
    )
    return TableContract(
        key="source_artifact",
        name=names.source_artifact,
        columns=columns,
        primary_key=("ARTIFACT_ID",),
        unique_constraints=(),
        foreign_keys=(),
    )


def _reference_dataset_contract(names: PersistenceTableNames) -> TableContract:
    """Build immutable reference-dataset publication metadata."""
    return TableContract(
        key="reference_dataset",
        name=names.reference_dataset,
        columns=(
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("DATASET_TYPE", "NVARCHAR", length=40),
            _col("ARTIFACT_ID", "NVARCHAR", length=36),
            _col("PUBLICATION_ORDER", "BIGINT"),
            _col("SCOPE_CODE", "NVARCHAR", True, length=40),
            _col("SCOPE_NAME", "NVARCHAR", True, length=200),
            _col("ROW_COUNT", "INTEGER"),
            _col("WARNING_COUNT", "INTEGER"),
            _col("WARNING_JSON", "NCLOB"),
            _col("PARSER_VERSION", "NVARCHAR", length=40),
            _col("CREATED_BY", "NVARCHAR", True, length=128),
            _col("CREATED_AT", "TIMESTAMP"),
            _col("CONTEXT_JSON", "NCLOB", True),
        ),
        primary_key=("DATASET_ID",),
        unique_constraints=(("PUBLICATION_ORDER",),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("ARTIFACT_ID",),
                referenced_table_key="source_artifact",
                referenced_columns=("ARTIFACT_ID",),
            ),
        ),
    )


def _upload_dataset_contract(names: PersistenceTableNames) -> TableContract:
    """Build immutable ECN-upload to active reference-snapshot provenance links."""
    return TableContract(
        key="upload_dataset",
        name=names.upload_dataset,
        columns=(
            _col("UPLOAD_ID", "NVARCHAR", length=36),
            _col("DATASET_TYPE", "NVARCHAR", length=40),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("PINNED_AT", "TIMESTAMP"),
        ),
        primary_key=("UPLOAD_ID", "DATASET_TYPE"),
        unique_constraints=(),
        foreign_keys=(
            ForeignKeyContract(
                columns=("UPLOAD_ID",),
                referenced_table_key="upload",
                referenced_columns=("UPLOAD_ID",),
            ),
            ForeignKeyContract(
                columns=("DATASET_ID",),
                referenced_table_key="reference_dataset",
                referenced_columns=("DATASET_ID",),
            ),
        ),
    )


def _conditional_rule_row_contract(names: PersistenceTableNames) -> TableContract:
    """Build normalized conditional-rule source rows tied to one dataset."""
    return TableContract(
        key="conditional_rule_row",
        name=names.conditional_rule_row,
        columns=(
            _col("ROW_ID", "NVARCHAR", length=36),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("SOURCE_ROW_NUMBER", "INTEGER"),
            _col("RULE_NUMBER", "NVARCHAR", True, length=200),
            _col("PARSE_STATE", "NVARCHAR", length=32),
            _col("LIFECYCLE_STATE", "NVARCHAR", True, length=200),
            _col("SOURCE_TEXT", "NCLOB"),
            _col("NORMALIZED_JSON", "NCLOB"),
        ),
        primary_key=("ROW_ID",),
        unique_constraints=(("DATASET_ID", "SOURCE_ROW_NUMBER"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("DATASET_ID",),
                referenced_table_key="reference_dataset",
                referenced_columns=("DATASET_ID",),
            ),
        ),
    )


def _material_master_row_contract(names: PersistenceTableNames) -> TableContract:
    """Build normalized material-master rows for plant/status lookup."""
    return TableContract(
        key="material_master_row",
        name=names.material_master_row,
        columns=(
            _col("ROW_ID", "NVARCHAR", length=36),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("SHEET_NAME", "NVARCHAR", length=500),
            _col("SOURCE_ROW_NUMBER", "INTEGER"),
            _col("MATERIAL_NUMBER", "NVARCHAR", True, length=200),
            _col("DESCRIPTION", "NVARCHAR", True, length=2000),
            _col("PLANT_CODE", "NVARCHAR", True, length=40),
            _col("PROCUREMENT_TYPE", "NVARCHAR", True, length=40),
            _col("CROSS_PLANT_STATUS", "NVARCHAR", True, length=40),
            _col("PLANT_STATUS", "NVARCHAR", True, length=40),
            _col("MATERIAL_GROUP", "NVARCHAR", True, length=100),
            _col("COMMODITY_CODE", "NVARCHAR", True, length=100),
            _col("MATERIAL_GROUP_DESCRIPTION", "NVARCHAR", True, length=2000),
        ),
        primary_key=("ROW_ID",),
        unique_constraints=(("DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("DATASET_ID",),
                referenced_table_key="reference_dataset",
                referenced_columns=("DATASET_ID",),
            ),
        ),
    )


def _purchase_info_row_contract(names: PersistenceTableNames) -> TableContract:
    """Build normalized purchasing-info records with source provenance."""
    return TableContract(
        key="purchase_info_row",
        name=names.purchase_info_row,
        columns=(
            _col("ROW_ID", "NVARCHAR", length=36),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("SHEET_NAME", "NVARCHAR", length=500),
            _col("SOURCE_ROW_NUMBER", "INTEGER"),
            _col("INFO_RECORD", "NVARCHAR", True, length=100),
            _col("VENDOR", "NVARCHAR", True, length=100),
            _col("PLANT_CODE", "NVARCHAR", True, length=40),
            _col("CATEGORY", "NVARCHAR", True, length=40),
            _col("PURCHASING_GROUP", "NVARCHAR", True, length=40),
            _col("TAX_CODE", "NVARCHAR", True, length=40),
            _col("FIXED_FIRM_DATE", "NVARCHAR", True, length=100),
            _col("MATERIAL_NUMBER", "NVARCHAR", True, length=200),
            _col("SUPPLIER_NAME", "NVARCHAR", True, length=2000),
        ),
        primary_key=("ROW_ID",),
        unique_constraints=(("DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("DATASET_ID",),
                referenced_table_key="reference_dataset",
                referenced_columns=("DATASET_ID",),
            ),
        ),
    )


def _purchase_info_price_row_contract(names: PersistenceTableNames) -> TableContract:
    """Build normalized source price rows independently from purchase setup records."""
    return TableContract(
        key="purchase_info_price_row",
        name=names.purchase_info_price_row,
        columns=(
            _col("ROW_ID", "NVARCHAR", length=36),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("SHEET_NAME", "NVARCHAR", length=500),
            _col("SOURCE_ROW_NUMBER", "INTEGER"),
            _col("INFO_RECORD", "NVARCHAR", True, length=100),
            _col("VENDOR", "NVARCHAR", True, length=100),
            _col("SUPPLIER_NAME", "NVARCHAR", True, length=2000),
            _col("MATERIAL_NUMBER", "NVARCHAR", True, length=200),
            _col("PLANT_CODE", "NVARCHAR", True, length=40),
            _col("PURCHASING_GROUP", "NVARCHAR", True, length=40),
            _col("NET_PRICE", "NVARCHAR", True, length=100),
            _col("CURRENCY", "NVARCHAR", True, length=10),
            _col("PRICE_PER", "NVARCHAR", True, length=100),
            _col("ORDER_PRICE_UNIT", "NVARCHAR", True, length=40),
        ),
        primary_key=("ROW_ID",),
        unique_constraints=(("DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("DATASET_ID",),
                referenced_table_key="reference_dataset",
                referenced_columns=("DATASET_ID",),
            ),
        ),
    )


def _manufacturing_feasibility_assessment_contract(
    names: PersistenceTableNames,
) -> TableContract:
    """Store one automated drawing assessment without retaining drawing bytes."""
    return TableContract(
        key="manufacturing_feasibility_assessment",
        name=names.manufacturing_feasibility_assessment,
        columns=(
            _col("ASSESSMENT_ID", "NVARCHAR", length=36),
            _col("ECN_ROW_ID", "NVARCHAR", length=36),
            _col("MATERIAL_NUMBER", "NVARCHAR", length=200),
            _col("DRAWING_FILE_NAME", "NVARCHAR", length=500),
            _col("DRAWING_SHA256", "NVARCHAR", length=64),
            _col("STATE", "NVARCHAR", length=40),
            _col("REPORT_JSON", "NCLOB", True),
            _col("ERROR_CODE", "NVARCHAR", True, length=100),
            _col("CREATED_AT", "TIMESTAMP"),
            _col("UPDATED_AT", "TIMESTAMP"),
        ),
        primary_key=("ASSESSMENT_ID",),
        unique_constraints=(("ECN_ROW_ID", "MATERIAL_NUMBER", "DRAWING_SHA256"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("ECN_ROW_ID",),
                referenced_table_key="ecn",
                referenced_columns=("ECN_ROW_ID",),
            ),
        ),
    )


def _stock_row_contract(names: PersistenceTableNames) -> TableContract:
    """Store reported stock rows without assuming units, dates or aggregate compatibility."""
    return TableContract(
        key="stock_row", name=names.stock_row,
        columns=(
            _col("ROW_ID", "NVARCHAR", length=36),
            _col("DATASET_ID", "NVARCHAR", length=36),
            _col("SHEET_NAME", "NVARCHAR", length=500),
            _col("SOURCE_ROW_NUMBER", "INTEGER"),
            _col("MATERIAL_NUMBER", "NVARCHAR", length=200),
            _col("PLANT_CODE", "NVARCHAR", length=40),
            _col("STORAGE_LOCATION", "NVARCHAR", length=100),
            _col("UNRESTRICTED_QUANTITY", "NVARCHAR", length=100),
        ),
        primary_key=("ROW_ID",),
        unique_constraints=(("DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),),
        foreign_keys=(ForeignKeyContract(("DATASET_ID",), "reference_dataset", ("DATASET_ID",)),),
    )


def _upload_contract(names: PersistenceTableNames) -> TableContract:
    """Build the upload table contract from the given name set.

    Represents a single intake batch linking three source artifacts.  13
    columns; PK on UPLOAD_ID; UQ on REQUEST_ID and PUBLICATION_ORDER
    separately; three FKs to source_artifact.ARTIFACT_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the upload table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("UPLOAD_ID", "NVARCHAR", length=36),
        _col("REQUEST_ID", "NVARCHAR", length=36),
        _col("PUBLICATION_ORDER", "BIGINT"),
        _col("ECN_ARTIFACT_ID", "NVARCHAR", length=36),
        _col("BOM_ARTIFACT_ID", "NVARCHAR", length=36),
        _col("RULE_ARTIFACT_ID", "NVARCHAR", True, length=36),
        _col("BOM_REUSED", "TINYINT"),
        _col("IMPORTED_ECN_COUNT", "INTEGER"),
        _col("CREATED_BY", "NVARCHAR", True, length=128),
        _col("CREATED_AT", "TIMESTAMP"),
        _col("PARSER_VERSION", "NVARCHAR", length=40),
        _col("DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", length=40),
        _col("RULE_EVALUATOR_VERSION", "NVARCHAR", length=40),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("ECN_ARTIFACT_ID",),
            referenced_table_key="source_artifact",
            referenced_columns=("ARTIFACT_ID",),
        ),
        ForeignKeyContract(
            columns=("BOM_ARTIFACT_ID",),
            referenced_table_key="source_artifact",
            referenced_columns=("ARTIFACT_ID",),
        ),
        ForeignKeyContract(
            columns=("RULE_ARTIFACT_ID",),
            referenced_table_key="source_artifact",
            referenced_columns=("ARTIFACT_ID",),
        ),
    )
    return TableContract(
        key="upload",
        name=names.upload,
        columns=columns,
        primary_key=("UPLOAD_ID",),
        unique_constraints=(("REQUEST_ID",), ("PUBLICATION_ORDER",)),
        foreign_keys=foreign_keys,
    )


def _bom_row_contract(names: PersistenceTableNames) -> TableContract:
    """Build the bom_row table contract from the given name set.

    Holds one row parsed from a BOM workbook sheet.  35 columns; PK on
    BOM_ROW_ID; UQ on (ARTIFACT_ID, SHEET_NAME, SOURCE_ROW_NUMBER); FK to
    source_artifact.ARTIFACT_ID and self-FK for PARENT_BOM_ROW_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the bom_row table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("BOM_ROW_ID", "NVARCHAR", length=36),
        _col("ARTIFACT_ID", "NVARCHAR", length=36),
        _col("BOM_KIND", "NVARCHAR", length=32),
        _col("SHEET_NAME", "NVARCHAR", length=500),
        _col("SOURCE_ROW_NUMBER", "INTEGER"),
        _col("PARENT_BOM_ROW_ID", "NVARCHAR", True, length=36),
        _col("DOTTED_LEVEL", "NVARCHAR", True, length=100),
        _col("HIERARCHY_PATH", "NCLOB", True),
        _col("MAIN_VALUE", "NVARCHAR", True, length=200),
        _col("SUB_VALUE", "NVARCHAR", True, length=200),
        _col("VC_VALUE", "NVARCHAR", True, length=200),
        _col("LEVEL_4_VALUE", "NVARCHAR", True, length=200),
        _col("LEVEL_5_VALUE", "NVARCHAR", True, length=200),
        _col("LEVEL_6_VALUE", "NVARCHAR", True, length=200),
        _col("ITEM_VALUE", "NVARCHAR", True, length=200),
        _col("POSITION_VALUE", "NVARCHAR", True, length=200),
        _col("MATERIAL_NUMBER", "NVARCHAR", True, length=200),
        _col("PARENT_MATERIAL_NUMBER", "NVARCHAR", True, length=200),
        _col("COMPONENT_NUMBER", "NVARCHAR", True, length=200),
        _col("CHANGE_NUMBER", "NVARCHAR", True, length=200),
        _col("CHANGE_NUMBER_TO", "NVARCHAR", True, length=200),
        _col("DESCRIPTION_TEXT", "NCLOB", True),
        _col("QUANTITY", "DECIMAL", True, precision=31, scale=14),
        _col("UNIT", "NVARCHAR", True, length=40),
        _col("PLANT", "NVARCHAR", True, length=80),
        _col("ALTERNATIVE", "NVARCHAR", True, length=200),
        _col("VALID_FROM_RAW", "NVARCHAR", True, length=100),
        _col("VALID_FROM_DATE", "DATE", True),
        _col("VALID_TO_RAW", "NVARCHAR", True, length=100),
        _col("VALID_TO_DATE", "DATE", True),
        _col("ROW_CLASSIFICATION", "NVARCHAR", length=40),
        _col("IS_DOCUMENTATION", "TINYINT"),
        _col("IS_MIGRATION_DATE", "TINYINT"),
        _col("ROW_JSON", "NCLOB"),
        _col("QUALITY_JSON", "NCLOB"),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("ARTIFACT_ID",),
            referenced_table_key="source_artifact",
            referenced_columns=("ARTIFACT_ID",),
        ),
        ForeignKeyContract(
            columns=("PARENT_BOM_ROW_ID",),
            referenced_table_key="bom_row",
            referenced_columns=("BOM_ROW_ID",),
        ),
    )
    return TableContract(
        key="bom_row",
        name=names.bom_row,
        columns=columns,
        primary_key=("BOM_ROW_ID",),
        unique_constraints=(("ARTIFACT_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER"),),
        foreign_keys=foreign_keys,
    )


def _ecn_contract(names: PersistenceTableNames) -> TableContract:
    """Build the ecn table contract from the given name set.

    Stores one normalized ECN row linked to an upload batch.  22 columns;
    PK on ECN_ROW_ID; UQ on (UPLOAD_ID, SOURCE_ORDINAL); FK to
    upload.UPLOAD_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the ecn table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("ECN_ROW_ID", "NVARCHAR", length=36),
        _col("UPLOAD_ID", "NVARCHAR", length=36),
        _col("SOURCE_ORDINAL", "INTEGER"),
        _col("ECN_NUMBER", "NVARCHAR", True, length=80),
        _col("TITLE", "NVARCHAR", True, length=500),
        _col("DESCRIPTION_SUMMARY", "NVARCHAR", True, length=2000),
        _col("DESCRIPTION_TEXT", "NCLOB", True),
        _col("AFFECTED_ITEM_COUNT", "INTEGER"),
        _col("RAW_ECN_MATCH_COUNT", "INTEGER"),
        _col("EBOM_MATCH_COUNT", "INTEGER"),
        _col("EBOM_CONTEXT_COUNT", "INTEGER"),
        _col("MBOM_MATCH_COUNT", "INTEGER"),
        _col("WARNING_COUNT", "INTEGER"),
        _col("SOURCE_JSON", "NCLOB"),
        _col("CHANGE_CANDIDATES_JSON", "NCLOB"),
        _col("EVIDENCE_JSON", "NCLOB"),
        _col("EVIDENCE_HASH", "NVARCHAR", length=64),
        _col("PARSER_VERSION", "NVARCHAR", length=40),
        _col("DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", length=40),
        _col("RULE_EVALUATOR_VERSION", "NVARCHAR", length=40),
        _col("CREATED_AT", "TIMESTAMP"),
        # JSON array of plant labels from the snapshot's plant table, written
        # the first time the snapshot is built; NULL until then. Kept last so
        # the additive ALTER on older tables produces the same column order.
        _col("AFFECTED_PLANTS_JSON", "NVARCHAR", True, length=2000),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("UPLOAD_ID",),
            referenced_table_key="upload",
            referenced_columns=("UPLOAD_ID",),
        ),
    )
    return TableContract(
        key="ecn",
        name=names.ecn,
        columns=columns,
        primary_key=("ECN_ROW_ID",),
        unique_constraints=(("UPLOAD_ID", "SOURCE_ORDINAL"),),
        foreign_keys=foreign_keys,
    )


def _rdf_projection_contract(names: PersistenceTableNames) -> TableContract:
    """Build the per-ECN RDF projection-status table contract.

    Anchors the RDF projection lifecycle relationally: one row per ECN keyed on
    ``ECN_ROW_ID`` (also the FK into ``ecn``), the source hash and projection
    version that produced it, the current ``STATUS`` (plain NVARCHAR — no check
    constraint, since the bootstrap cannot introspect them reliably), the five
    NOT NULL expected-count baselines derived from the deterministic projection,
    the nullable verified node/triple counts and error fields recorded after a
    projection attempt, and the nullable attempt/current timestamps.  15 columns;
    PK on ECN_ROW_ID; no UQ; single FK to ecn.ECN_ROW_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the RDF projection-status table.
    """
    return TableContract(
        key="rdf_projection",
        name=names.rdf_projection,
        columns=(
            _col("ECN_ROW_ID", "NVARCHAR", length=36),
            _col("SOURCE_SHA256", "NVARCHAR", length=64),
            _col("PROJECTION_VERSION", "NVARCHAR", length=40),
            _col("STATUS", "NVARCHAR", length=20),
            _col("EXPECTED_OBJECT_COUNT", "BIGINT"),
            _col("EXPECTED_ARRAY_COUNT", "BIGINT"),
            _col("EXPECTED_SCALAR_COUNT", "BIGINT"),
            _col("EXPECTED_NULL_COUNT", "BIGINT"),
            _col("EXPECTED_NODE_COUNT", "BIGINT"),
            _col("VERIFIED_NODE_COUNT", "BIGINT", True),
            _col("VERIFIED_TRIPLE_COUNT", "BIGINT", True),
            _col("ERROR_CODE", "NVARCHAR", True, length=80),
            _col("ERROR_MESSAGE", "NVARCHAR", True, length=500),
            _col("LAST_ATTEMPT_AT", "TIMESTAMP", True),
            _col("CURRENT_AT", "TIMESTAMP", True),
        ),
        primary_key=("ECN_ROW_ID",),
        unique_constraints=(),
        foreign_keys=(
            ForeignKeyContract(
                columns=("ECN_ROW_ID",),
                referenced_table_key="ecn",
                referenced_columns=("ECN_ROW_ID",),
            ),
        ),
    )


def _analysis_job_contract(names: PersistenceTableNames) -> TableContract:
    """Build the analysis_job table contract from the given name set.

    Tracks the LLM analysis lifecycle for one ECN row.  13 columns; PK on
    JOB_ID; UQ on ECN_ROW_ID; FK to ecn.ECN_ROW_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the analysis_job table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("JOB_ID", "NVARCHAR", length=36),
        _col("ECN_ROW_ID", "NVARCHAR", length=36),
        _col("STATE", "NVARCHAR", length=32),
        _col("ATTEMPT_COUNT", "INTEGER"),
        _col("CREATED_AT", "TIMESTAMP"),
        _col("STARTED_AT", "TIMESTAMP", True),
        _col("FINISHED_AT", "TIMESTAMP", True),
        _col("PROMPT_VERSION", "NVARCHAR", length=40),
        _col("OUTPUT_SCHEMA_VERSION", "NVARCHAR", length=40),
        _col("MODEL_NAME", "NVARCHAR", length=200),
        _col("MODEL_CONFIG_JSON", "NCLOB"),
        _col("ERROR_CODE", "NVARCHAR", True, length=80),
        _col("ERROR_MESSAGE", "NVARCHAR", True, length=2000),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("ECN_ROW_ID",),
            referenced_table_key="ecn",
            referenced_columns=("ECN_ROW_ID",),
        ),
    )
    return TableContract(
        key="analysis_job",
        name=names.analysis_job,
        columns=columns,
        primary_key=("JOB_ID",),
        unique_constraints=(("ECN_ROW_ID",),),
        foreign_keys=foreign_keys,
    )


def _llm_attempt_contract(names: PersistenceTableNames) -> TableContract:
    """Build the llm_attempt table contract from the given name set.

    Records one provider call within an analysis job including token usage.
    21 columns; PK on ATTEMPT_ID; UQ on (JOB_ID, ATTEMPT_NUMBER); FK to
    analysis_job.JOB_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the llm_attempt table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("ATTEMPT_ID", "NVARCHAR", length=36),
        _col("JOB_ID", "NVARCHAR", length=36),
        _col("ATTEMPT_NUMBER", "INTEGER"),
        _col("CORRELATION_ID", "NVARCHAR", length=100),
        _col("PROVIDER", "NVARCHAR", length=100),
        _col("MODEL_NAME", "NVARCHAR", length=200),
        _col("MODEL_DEPLOYMENT", "NVARCHAR", True, length=200),
        _col("LLM_ENDPOINT", "NVARCHAR", length=100),
        _col("STARTED_AT", "TIMESTAMP"),
        _col("FINISHED_AT", "TIMESTAMP", True),
        _col("PROVIDER_OUTCOME", "NVARCHAR", length=32),
        _col("VALIDATION_OUTCOME", "NVARCHAR", length=32),
        _col("ERROR_CODE", "NVARCHAR", True, length=80),
        _col("LATENCY_MS", "BIGINT", True),
        _col("INPUT_TOKENS", "BIGINT", True),
        _col("CACHE_READ_INPUT_TOKENS", "BIGINT", True),
        _col("CACHE_WRITE_INPUT_TOKENS", "BIGINT", True),
        _col("INPUT_TOTAL_TOKENS", "BIGINT", True),
        _col("OUTPUT_TOKENS", "BIGINT", True),
        _col("TOTAL_TOKENS", "BIGINT", True),
        _col("USAGE_AVAILABLE", "TINYINT"),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("JOB_ID",),
            referenced_table_key="analysis_job",
            referenced_columns=("JOB_ID",),
        ),
    )
    return TableContract(
        key="llm_attempt",
        name=names.llm_attempt,
        columns=columns,
        primary_key=("ATTEMPT_ID",),
        unique_constraints=(("JOB_ID", "ATTEMPT_NUMBER"),),
        foreign_keys=foreign_keys,
    )


def _provider_call_contract(names: PersistenceTableNames) -> TableContract:
    """Build per-stage provider-call diagnostics and token-usage storage."""
    return TableContract(
        key="provider_call",
        name=names.provider_call,
        columns=(
            _col("CALL_ID", "NVARCHAR", length=36),
            _col("ATTEMPT_ID", "NVARCHAR", length=36),
            _col("STAGE", "NVARCHAR", length=20),
            _col("CALL_NUMBER", "INTEGER"),
            _col("STARTED_AT", "TIMESTAMP"),
            _col("FINISHED_AT", "TIMESTAMP"),
            _col("OUTCOME", "NVARCHAR", length=32),
            _col("HTTP_STATUS", "INTEGER", True),
            _col("PROVIDER_CODE", "NVARCHAR", True, length=120),
            _col("PROVIDER_REQUEST_ID", "NVARCHAR", True, length=200),
            _col("RETRY_AFTER_SECONDS", "INTEGER", True),
            _col("LATENCY_MS", "BIGINT"),
            _col("MODEL_DEPLOYMENT", "NVARCHAR", True, length=200),
            _col("ERROR_CODE", "NVARCHAR", True, length=80),
            _col("INPUT_TOKENS", "BIGINT", True),
            _col("CACHE_READ_INPUT_TOKENS", "BIGINT", True),
            _col("CACHE_WRITE_INPUT_TOKENS", "BIGINT", True),
            _col("INPUT_TOTAL_TOKENS", "BIGINT", True),
            _col("OUTPUT_TOKENS", "BIGINT", True),
            _col("TOTAL_TOKENS", "BIGINT", True),
            _col("USAGE_AVAILABLE", "TINYINT"),
        ),
        primary_key=("CALL_ID",),
        unique_constraints=(("ATTEMPT_ID", "STAGE"),),
        foreign_keys=(
            ForeignKeyContract(
                columns=("ATTEMPT_ID",),
                referenced_table_key="llm_attempt",
                referenced_columns=("ATTEMPT_ID",),
            ),
        ),
    )


def _analysis_stage_contract(names: PersistenceTableNames) -> TableContract:
    """Store validated partial outputs by attempt so retries can reuse exact evidence."""
    return TableContract(
        key="analysis_stage",
        name=names.analysis_stage,
        columns=(
            _col("RESULT_ID", "NVARCHAR", length=36),
            _col("JOB_ID", "NVARCHAR", length=36),
            _col("ATTEMPT_ID", "NVARCHAR", length=36),
            _col("STAGE", "NVARCHAR", length=20),
            _col("EVIDENCE_HASH", "NVARCHAR", length=64),
            _col("PROMPT_VERSION", "NVARCHAR", length=40),
            _col("OUTPUT_SCHEMA_VERSION", "NVARCHAR", length=40),
            _col("CONTENT_JSON", "NCLOB"),
            _col("CREATED_AT", "TIMESTAMP"),
        ),
        primary_key=("RESULT_ID",),
        unique_constraints=(("ATTEMPT_ID", "STAGE"),),
        foreign_keys=(
            ForeignKeyContract(("JOB_ID",), "analysis_job", ("JOB_ID",)),
            ForeignKeyContract(("ATTEMPT_ID",), "llm_attempt", ("ATTEMPT_ID",)),
        ),
    )


def _analysis_contract(names: PersistenceTableNames) -> TableContract:
    """Build the analysis table contract from the given name set.

    Stores the final accepted analysis result for one ECN.  18 columns; PK
    on ANALYSIS_ID; UQ on ECN_ROW_ID and JOB_ID separately; FKs to
    ecn.ECN_ROW_ID, analysis_job.JOB_ID, and llm_attempt.ATTEMPT_ID.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the analysis table.
    """
    columns: tuple[ColumnContract, ...] = (
        _col("ANALYSIS_ID", "NVARCHAR", length=36),
        _col("ECN_ROW_ID", "NVARCHAR", length=36),
        _col("JOB_ID", "NVARCHAR", length=36),
        _col("SUCCESSFUL_ATTEMPT_ID", "NVARCHAR", length=36),
        _col("COMPLETENESS_STATE", "NVARCHAR", length=40),
        _col("CONTENT_JSON", "NCLOB"),
        _col("EVIDENCE_HASH", "NVARCHAR", length=64),
        _col("PROMPT_VERSION", "NVARCHAR", length=40),
        _col("MODEL_NAME", "NVARCHAR", length=200),
        _col("MODEL_DEPLOYMENT", "NVARCHAR", True, length=200),
        _col("DETERMINISTIC_LOGIC_VERSION", "NVARCHAR", length=40),
        _col("RULE_EVALUATOR_VERSION", "NVARCHAR", length=40),
        _col("OUTPUT_SCHEMA_VERSION", "NVARCHAR", length=40),
        _col("CREATED_AT", "TIMESTAMP"),
        _col("DISCOVERY_STATUS", "NVARCHAR", length=20),
        _col("DISCOVERY_ERROR_CODE", "NVARCHAR", True, length=80),
        _col("DISCOVERY_HASH", "NVARCHAR", length=64),
        _col("DISCOVERED_EVIDENCE_JSON", "NCLOB"),
    )
    foreign_keys: tuple[ForeignKeyContract, ...] = (
        ForeignKeyContract(
            columns=("ECN_ROW_ID",),
            referenced_table_key="ecn",
            referenced_columns=("ECN_ROW_ID",),
        ),
        ForeignKeyContract(
            columns=("JOB_ID",),
            referenced_table_key="analysis_job",
            referenced_columns=("JOB_ID",),
        ),
        ForeignKeyContract(
            columns=("SUCCESSFUL_ATTEMPT_ID",),
            referenced_table_key="llm_attempt",
            referenced_columns=("ATTEMPT_ID",),
        ),
    )
    return TableContract(
        key="analysis",
        name=names.analysis,
        columns=columns,
        primary_key=("ANALYSIS_ID",),
        unique_constraints=(("ECN_ROW_ID",), ("JOB_ID",)),
        foreign_keys=foreign_keys,
    )


def _plant_mapping_contract(names: PersistenceTableNames) -> TableContract:
    """Build the Settings-maintained brand suffix to plant mapping.

    One row per rule brand suffix (``JB`` in ``E_BRAND_JB``). Several suffixes
    may share one plant. 4 columns; PK on BRAND_SUFFIX.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the plant-mapping table.
    """
    return TableContract(
        key="plant_mapping",
        name=names.plant_mapping,
        columns=(
            _col("BRAND_SUFFIX", "NVARCHAR", length=20),
            _col("PLANT_CODE", "NVARCHAR", length=40),
            _col("PLANT_NAME", "NVARCHAR", length=200),
            _col("UPDATED_AT", "TIMESTAMP"),
        ),
        primary_key=("BRAND_SUFFIX",),
        unique_constraints=(),
        foreign_keys=(),
    )


def _app_setting_contract(names: PersistenceTableNames) -> TableContract:
    """Build the small key/value table for non-tabular Settings values.

    Holds, for example, the rule lifecycle states excluded from plant
    assignment. 3 columns; PK on SETTING_KEY.

    Args:
        names: Active ``PersistenceTableNames`` providing the physical name.

    Returns:
        A frozen ``TableContract`` for the application-setting table.
    """
    return TableContract(
        key="app_setting",
        name=names.app_setting,
        columns=(
            _col("SETTING_KEY", "NVARCHAR", length=80),
            _col("VALUE_JSON", "NCLOB"),
            _col("UPDATED_AT", "TIMESTAMP"),
        ),
        primary_key=("SETTING_KEY",),
        unique_constraints=(),
        foreign_keys=(),
    )


# ---------------------------------------------------------------------------
# Public catalog builder
# ---------------------------------------------------------------------------


def build_table_contracts(
    table_names: PersistenceTableNames = PersistenceTableNames(),
) -> tuple[TableContract, ...]:
    """Return all table contracts in dependency-safe creation order.

    Dependencies are created before referencing tables. Reverse this tuple for
    safe drop/cleanup in integration test teardown.

    Args:
        table_names: Optional ``PersistenceTableNames`` override; defaults
            to the production names.  Pass a suffixed instance from
            ``PersistenceTableNames.with_suffix`` for isolated test namespaces.

    Returns:
        Frozen table contracts in creation order.
    """
    from app.persistence.workspace_contracts import workspace_contracts

    return (
        _source_artifact_contract(table_names),
        _reference_dataset_contract(table_names),
        _upload_contract(table_names),
        _bom_row_contract(table_names),
        _conditional_rule_row_contract(table_names),
        _material_master_row_contract(table_names),
        _purchase_info_row_contract(table_names),
        _purchase_info_price_row_contract(table_names),
        _stock_row_contract(table_names),
        _upload_dataset_contract(table_names),
        _ecn_contract(table_names),
        _manufacturing_feasibility_assessment_contract(table_names),
        _rdf_projection_contract(table_names),
        _analysis_job_contract(table_names),
        _llm_attempt_contract(table_names),
        _provider_call_contract(table_names),
        _analysis_stage_contract(table_names),
        _analysis_contract(table_names),
        *workspace_contracts(table_names),
        _plant_mapping_contract(table_names),
        _app_setting_contract(table_names),
    )


# ---------------------------------------------------------------------------
# DDL renderer
# ---------------------------------------------------------------------------


def _column_sql(column: ColumnContract) -> str:
    """Render one exact column declaration from its logical contract.

    For NVARCHAR, renders ``NVARCHAR(length)``.  For DECIMAL, renders
    ``DECIMAL(precision,scale)``.  All other types use their hana_type
    string directly (BIGINT, INTEGER, TINYINT, TIMESTAMP, NCLOB, BLOB,
    DATE).  Nullable columns carry no explicit keyword; NOT NULL columns
    carry the ``NOT NULL`` suffix.

    Args:
        column: The column contract to render.

    Returns:
        A SQL column declaration fragment, e.g.
        ``'"ARTIFACT_ID" NVARCHAR(36) NOT NULL'``.
    """
    if column.hana_type == "NVARCHAR":
        rendered_type = f"NVARCHAR({column.length})"
    elif column.hana_type == "DECIMAL":
        rendered_type = f"DECIMAL({column.precision},{column.scale})"
    else:
        rendered_type = column.hana_type
    nullability = "" if column.nullable else " NOT NULL"
    return f'"{column.name}" {rendered_type}{nullability}'


def render_create_table(
    contract: TableContract,
    table_names: PersistenceTableNames,
) -> str:
    """Render one HANA column-table DDL statement with resolved FK targets.

    Produces a ``CREATE COLUMN TABLE`` statement with:

    * Quoted column declarations in the exact order from ``contract.columns``.
    * An unnamed ``PRIMARY KEY`` clause.
    * Unnamed ``UNIQUE`` clauses for each entry in
      ``contract.unique_constraints``.
    * Unnamed ``FOREIGN KEY ... REFERENCES`` clauses resolved through
      ``table_names`` so that suffixed namespaces self-reference correctly.

    The statement contains no ``IF NOT EXISTS``, no bind parameters, and no
    hand-built constraint names (HANA generates constraint names at DDL time).

    Args:
        contract: The table contract describing columns, PK, UQ, and FK.
        table_names: The active ``PersistenceTableNames`` providing FK targets.

    Returns:
        A single-line DDL string ready for execution against HANA.
    """
    clauses: list[str] = [_column_sql(column) for column in contract.columns]

    # Primary key — unnamed, HANA generates the constraint name
    clauses.append(
        "PRIMARY KEY (" + ", ".join(f'"{c}"' for c in contract.primary_key) + ")"
    )

    # Unique constraints — each rendered unnamed
    for columns in contract.unique_constraints:
        clauses.append(
            "UNIQUE (" + ", ".join(f'"{c}"' for c in columns) + ")"
        )

    # Foreign keys — target resolved through table_names, not hardcoded
    for foreign_key in contract.foreign_keys:
        target_name = getattr(table_names, foreign_key.referenced_table_key)
        local_cols = ", ".join(f'"{c}"' for c in foreign_key.columns)
        remote_cols = ", ".join(f'"{c}"' for c in foreign_key.referenced_columns)
        clauses.append(
            f"FOREIGN KEY ({local_cols}) REFERENCES {quote_identifier(target_name)} ({remote_cols})"
        )

    return (
        f"CREATE COLUMN TABLE {quote_identifier(contract.name)} ("
        + ", ".join(clauses)
        + ")"
    )
