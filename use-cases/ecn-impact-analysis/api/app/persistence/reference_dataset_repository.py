"""Atomic HANA persistence for immutable Settings-managed reference datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine, RowMapping

from app.ecn.errors import PersistenceStateError
from app.ecn.models import JsonValue, canonical_uuid
from app.ecn.reference_data.models import (
    REFERENCE_DATA_PARSER_VERSION,
    NormalizedReferenceRow,
    ReferenceDatasetDetail,
    ReferenceDatasetPublication,
    ReferenceDatasetRecord,
    ReferenceDatasetType,
)
from app.persistence.bulk import insert_chunks
from app.persistence.codec import (
    artifact_to_row,
    bom_row_to_row,
    normalize_row_keys,
    read_lob_text,
)
from app.persistence.hana import from_hana_timestamp, to_hana_timestamp
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier

_ROW_TABLE_BY_TYPE: dict[ReferenceDatasetType, str] = {
    ReferenceDatasetType.STOCK: "stock_row",
    ReferenceDatasetType.CONDITIONAL_RULES: "conditional_rule_row",
    ReferenceDatasetType.MATERIAL_MASTER: "material_master_row",
    ReferenceDatasetType.PURCHASE_INFO_RECORDS: "purchase_info_row",
    ReferenceDatasetType.PURCHASE_INFO_PRICES: "purchase_info_price_row",
}


def _artifact_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the bound source-artifact insert used inside publication."""
    table = quote_identifier(tables.source_artifact)
    return (
        f"INSERT INTO {table} ("
        '"ARTIFACT_ID", "ARTIFACT_KIND", "FILE_NAME", "MEDIA_TYPE", "BYTE_SIZE", '
        '"SHA256", "CONTENT_BYTES", "NORMALIZED_CONTENT", "PARSE_STATE", '
        '"PARSE_ERROR_CODE", "CREATED_AT") VALUES ('
        ":ARTIFACT_ID, :ARTIFACT_KIND, :FILE_NAME, :MEDIA_TYPE, :BYTE_SIZE, "
        ":SHA256, :CONTENT_BYTES, :NORMALIZED_CONTENT, :PARSE_STATE, "
        ":PARSE_ERROR_CODE, :CREATED_AT)"
    )


def _bom_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the bound BOM-row insert for Settings BOM publications."""
    table = quote_identifier(tables.bom_row)
    columns = (
        "BOM_ROW_ID", "ARTIFACT_ID", "BOM_KIND", "SHEET_NAME",
        "SOURCE_ROW_NUMBER", "PARENT_BOM_ROW_ID", "DOTTED_LEVEL",
        "HIERARCHY_PATH", "MAIN_VALUE", "SUB_VALUE", "VC_VALUE",
        "LEVEL_4_VALUE", "LEVEL_5_VALUE", "LEVEL_6_VALUE", "ITEM_VALUE",
        "POSITION_VALUE", "MATERIAL_NUMBER", "PARENT_MATERIAL_NUMBER",
        "COMPONENT_NUMBER", "CHANGE_NUMBER", "CHANGE_NUMBER_TO",
        "DESCRIPTION_TEXT", "QUANTITY", "UNIT", "PLANT", "ALTERNATIVE",
        "VALID_FROM_RAW", "VALID_FROM_DATE", "VALID_TO_RAW", "VALID_TO_DATE",
        "ROW_CLASSIFICATION", "IS_DOCUMENTATION", "IS_MIGRATION_DATE",
        "ROW_JSON", "QUALITY_JSON",
    )
    quoted = ", ".join(f'"{name}"' for name in columns)
    params = ", ".join(f":{name}" for name in columns)
    return f"INSERT INTO {table} ({quoted}) VALUES ({params})"


def _dataset_insert_sql(tables: PersistenceTableNames) -> str:
    """Return the bound immutable dataset metadata insert."""
    table = quote_identifier(tables.reference_dataset)
    return (
        f"INSERT INTO {table} ("
        '"DATASET_ID", "DATASET_TYPE", "ARTIFACT_ID", "PUBLICATION_ORDER", '
        '"SCOPE_CODE", "SCOPE_NAME", "CONTEXT_JSON", "ROW_COUNT", "WARNING_COUNT", '
        '"WARNING_JSON", "PARSER_VERSION", "CREATED_BY", "CREATED_AT") VALUES ('
        ":DATASET_ID, :DATASET_TYPE, :ARTIFACT_ID, :PUBLICATION_ORDER, "
        ":SCOPE_CODE, :SCOPE_NAME, :CONTEXT_JSON, :ROW_COUNT, :WARNING_COUNT, :WARNING_JSON, "
        ":PARSER_VERSION, :CREATED_BY, :CREATED_AT)"
    )


def _row_insert_sql(table_name: str, columns: tuple[str, ...]) -> str:
    """Return a bound insert for one allowlisted normalized row table."""
    table = quote_identifier(table_name)
    quoted = ", ".join(f'"{name}"' for name in columns)
    params = ", ".join(f":{name}" for name in columns)
    return f"INSERT INTO {table} ({quoted}) VALUES ({params})"


def _json(value: object) -> str:
    """Serialize safe normalized data deterministically for HANA NCLOB fields."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _base_row(row: NormalizedReferenceRow, dataset_id: str) -> dict[str, object]:
    """Build source-coordinate parameters shared by all row-table inserts."""
    return {
        "ROW_ID": row.row_id,
        "DATASET_ID": dataset_id,
        "SHEET_NAME": row.sheet_name,
        "SOURCE_ROW_NUMBER": row.source_row_number,
    }


def _normalized_row_params(
    dataset_type: ReferenceDatasetType,
    dataset_id: str,
    row: NormalizedReferenceRow,
) -> dict[str, object]:
    """Map normalized business fields to one source-specific HANA row."""
    values = row.values
    base = _base_row(row, dataset_id)
    if dataset_type is ReferenceDatasetType.CONDITIONAL_RULES:
        return {
            "ROW_ID": row.row_id,
            "DATASET_ID": dataset_id,
            "SOURCE_ROW_NUMBER": row.source_row_number,
            "RULE_NUMBER": values.get("rule_number"),
            "PARSE_STATE": values.get("parse_state"),
            "LIFECYCLE_STATE": values.get("lifecycle_state"),
            "SOURCE_TEXT": values.get("source_text") or "",
            "NORMALIZED_JSON": _json(values),
        }
    field_maps: dict[ReferenceDatasetType, dict[str, str]] = {
        ReferenceDatasetType.STOCK: {"MATERIAL_NUMBER": "material_number", "PLANT_CODE": "plant_code",
            "STORAGE_LOCATION": "storage_location", "UNRESTRICTED_QUANTITY": "unrestricted_quantity"},
        ReferenceDatasetType.MATERIAL_MASTER: {
            "MATERIAL_NUMBER": "material_number", "DESCRIPTION": "description",
            "PLANT_CODE": "plant_code", "PROCUREMENT_TYPE": "procurement_type",
            "CROSS_PLANT_STATUS": "cross_plant_status", "PLANT_STATUS": "plant_status",
            "MATERIAL_GROUP": "material_group", "COMMODITY_CODE": "commodity_code",
            "MATERIAL_GROUP_DESCRIPTION": "material_group_description",
        },
        ReferenceDatasetType.PURCHASE_INFO_RECORDS: {
            "INFO_RECORD": "info_record", "VENDOR": "vendor", "SUPPLIER_NAME": "supplier_name", "PLANT_CODE": "plant_code",
            "CATEGORY": "category", "PURCHASING_GROUP": "purchasing_group",
            "TAX_CODE": "tax_code", "FIXED_FIRM_DATE": "fixed_firm_date",
            "MATERIAL_NUMBER": "material_number",
        },
        ReferenceDatasetType.PURCHASE_INFO_PRICES: {
            "INFO_RECORD": "info_record", "VENDOR": "vendor", "SUPPLIER_NAME": "supplier_name",
            "MATERIAL_NUMBER": "material_number", "PLANT_CODE": "plant_code",
            "PURCHASING_GROUP": "purchasing_group", "NET_PRICE": "net_price",
            "CURRENCY": "currency", "PRICE_PER": "price_per", "ORDER_PRICE_UNIT": "order_price_unit",
        },
    }
    return {**base, **{column: values.get(field) for column, field in field_maps[dataset_type].items()}}


def _row_columns(dataset_type: ReferenceDatasetType) -> tuple[str, ...]:
    """Return the exact insert columns for a normalized source type."""
    by_type = {
        ReferenceDatasetType.STOCK: ("ROW_ID", "DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER",
            "MATERIAL_NUMBER", "PLANT_CODE", "STORAGE_LOCATION", "UNRESTRICTED_QUANTITY"),
        ReferenceDatasetType.CONDITIONAL_RULES: (
            "ROW_ID", "DATASET_ID", "SOURCE_ROW_NUMBER", "RULE_NUMBER",
            "PARSE_STATE", "LIFECYCLE_STATE", "SOURCE_TEXT", "NORMALIZED_JSON",
        ),
        ReferenceDatasetType.MATERIAL_MASTER: (
            "ROW_ID", "DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER",
            "MATERIAL_NUMBER", "DESCRIPTION", "PLANT_CODE", "PROCUREMENT_TYPE",
            "CROSS_PLANT_STATUS", "PLANT_STATUS", "MATERIAL_GROUP", "COMMODITY_CODE", "MATERIAL_GROUP_DESCRIPTION",
        ),
        ReferenceDatasetType.PURCHASE_INFO_RECORDS: (
            "ROW_ID", "DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER", "INFO_RECORD",
            "VENDOR", "SUPPLIER_NAME", "PLANT_CODE", "CATEGORY", "PURCHASING_GROUP", "TAX_CODE",
            "FIXED_FIRM_DATE", "MATERIAL_NUMBER",
        ),
        ReferenceDatasetType.PURCHASE_INFO_PRICES: (
            "ROW_ID", "DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER", "INFO_RECORD",
            "VENDOR", "SUPPLIER_NAME", "MATERIAL_NUMBER", "PLANT_CODE", "PURCHASING_GROUP",
            "NET_PRICE", "CURRENCY", "PRICE_PER", "ORDER_PRICE_UNIT",
        ),
    }
    return by_type[dataset_type]


def _record_from_row(row: RowMapping, active: bool) -> ReferenceDatasetRecord:
    """Decode one metadata projection without materializing artifact content."""
    row = normalize_row_keys(row)
    warnings = json.loads(row["WARNING_JSON"])
    return ReferenceDatasetRecord(
        dataset_id=row["DATASET_ID"],
        dataset_type=ReferenceDatasetType(row["DATASET_TYPE"]),
        artifact_id=row["ARTIFACT_ID"],
        publication_order=int(row["PUBLICATION_ORDER"]),
        file_name=row["FILE_NAME"],
        sha256=row["SHA256"],
        scope_code=row["SCOPE_CODE"],
        scope_name=row["SCOPE_NAME"],
        row_count=int(row["ROW_COUNT"]),
        warning_counts={str(key): int(value) for key, value in warnings.items()},
        parser_version=row["PARSER_VERSION"],
        created_at=from_hana_timestamp(row["CREATED_AT"]),
        active=active,
        context=json.loads(read_lob_text(row.get("CONTEXT_JSON") or "{}") or "{}"),
    )


class ReferenceDatasetRepository:
    """Publish and read immutable source snapshots managed in Settings."""

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames = PersistenceTableNames(),
    ) -> None:
        """Bind a HANA engine and validated physical table identifiers."""
        self._engine = engine
        self._tables = table_names

    def publish(self, publication: ReferenceDatasetPublication) -> ReferenceDatasetRecord:
        """Atomically activate a fully validated source publication.

        Invalid input never reaches this method, so a failed parse cannot replace
        the active publication. A successful transaction becomes active because
        active selection always chooses the greatest publication order by type
        and scope.
        """
        canonical_uuid(publication.dataset_id)
        if publication.artifact.parse_error_code is not None:
            raise ValueError("reference artifact must be valid")
        with self._engine.begin() as connection:
            order = int(
                connection.execute(
                    text(
                        f'SELECT COALESCE(MAX("PUBLICATION_ORDER"), 0) + 1 FROM '
                        f'{quote_identifier(self._tables.reference_dataset)}'
                    )
                ).scalar_one()
            )
            connection.execute(text(_artifact_insert_sql(self._tables)), artifact_to_row(publication.artifact))
            if publication.bom_rows:
                insert_chunks(
                    connection,
                    text(_bom_insert_sql(self._tables)),
                    (bom_row_to_row(row) for row in publication.bom_rows),
                )
            connection.execute(
                text(_dataset_insert_sql(self._tables)),
                {
                    "DATASET_ID": publication.dataset_id,
                    "DATASET_TYPE": publication.dataset_type.value,
                    "ARTIFACT_ID": publication.artifact.artifact_id,
                    "PUBLICATION_ORDER": order,
                    "SCOPE_CODE": publication.scope_code,
                    "SCOPE_NAME": publication.scope_name,
                    "CONTEXT_JSON": _json(publication.context),
                    "ROW_COUNT": publication.row_count,
                    "WARNING_COUNT": sum(publication.warning_counts.values()),
                    "WARNING_JSON": _json(publication.warning_counts),
                    "PARSER_VERSION": REFERENCE_DATA_PARSER_VERSION,
                    "CREATED_BY": publication.created_by,
                    "CREATED_AT": to_hana_timestamp(publication.created_at),
                },
            )
            table_key = _ROW_TABLE_BY_TYPE.get(publication.dataset_type)
            if table_key and publication.rows:
                columns = _row_columns(publication.dataset_type)
                insert_chunks(
                    connection,
                    text(_row_insert_sql(getattr(self._tables, table_key), columns)),
                    (
                        _normalized_row_params(publication.dataset_type, publication.dataset_id, row)
                        for row in publication.rows
                    ),
                )
        return ReferenceDatasetRecord(
            dataset_id=publication.dataset_id,
            dataset_type=publication.dataset_type,
            artifact_id=publication.artifact.artifact_id,
            publication_order=order,
            file_name=publication.artifact.file_name,
            sha256=publication.artifact.sha256,
            scope_code=publication.scope_code,
            scope_name=publication.scope_name,
            row_count=publication.row_count,
            warning_counts=dict(publication.warning_counts),
            parser_version=REFERENCE_DATA_PARSER_VERSION,
            created_at=publication.created_at,
            active=True,
            context=publication.context,
        )

    def list(self, *, connection: Connection | None = None) -> tuple[ReferenceDatasetRecord, ...]:
        """List all publications and active flags on an optional caller-owned connection."""
        dataset = quote_identifier(self._tables.reference_dataset)
        artifact = quote_identifier(self._tables.source_artifact)
        sql = (
            f'SELECT D.*, A."FILE_NAME", A."SHA256" FROM {dataset} D '
            f'JOIN {artifact} A ON A."ARTIFACT_ID" = D."ARTIFACT_ID" '
            'ORDER BY D."PUBLICATION_ORDER" DESC, D."DATASET_ID" DESC'
        )
        with (self._engine.connect() if connection is None else nullcontext(connection)) as connection:
            rows = connection.execute(text(sql)).mappings().all()
        active_keys: set[tuple[str, str | None]] = set()
        records: list[ReferenceDatasetRecord] = []
        for raw_row in rows:
            row = normalize_row_keys(raw_row)
            from app.ecn.reference_data.scopes import scope_key
            record = _record_from_row(row, False)
            key = scope_key(record)
            active = key not in active_keys
            active_keys.add(key)
            records.append(replace(record, active=active))
        return tuple(records)

    def get(self, dataset_id: str, sample_size: int = 20) -> ReferenceDatasetDetail | None:
        """Return one publication plus a bounded source-row sample."""
        canonical_uuid(dataset_id)
        if not 1 <= sample_size <= 100:
            raise ValueError("sample_size must be between 1 and 100")
        records = self.list()
        record = next((item for item in records if item.dataset_id == dataset_id), None)
        if record is None:
            return None
        table_key = _ROW_TABLE_BY_TYPE.get(record.dataset_type)
        if table_key is None:
            return ReferenceDatasetDetail(record, ())
        table = quote_identifier(getattr(self._tables, table_key))
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    f'SELECT * FROM {table} WHERE "DATASET_ID" = :dataset_id '
                    'ORDER BY "SOURCE_ROW_NUMBER", "ROW_ID" LIMIT :sample_size'
                ),
                {"dataset_id": dataset_id, "sample_size": sample_size},
            ).mappings().all()
        samples = tuple(self._sample_from_row(record, row) for row in rows)
        return ReferenceDatasetDetail(record, samples)

    def get_active(
        self,
        dataset_type: ReferenceDatasetType,
        scope_code: str | None = None,
    ) -> ReferenceDatasetRecord | None:
        """Return the latest validated publication for an exact type and scope."""
        return next(
            (
                item for item in self.list()
                if item.active
                and item.dataset_type is dataset_type
                and item.scope_code == scope_code
            ),
            None,
        )

    def delete(self, dataset_id: str) -> bool:
        """Delete an unpinned publication, its rows, and its original workbook.

        Args:
            dataset_id: UUID of the Settings publication to remove.

        Returns:
            True when deleted, or False when the publication does not exist.

        Raises:
            PersistenceStateError: An ECN upload still references the publication.
        """
        canonical_uuid(dataset_id)
        dataset_table = quote_identifier(self._tables.reference_dataset)
        pin_table = quote_identifier(self._tables.upload_dataset)
        upload_table = quote_identifier(self._tables.upload)
        artifact_table = quote_identifier(self._tables.source_artifact)
        bom_table = quote_identifier(self._tables.bom_row)
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    f'SELECT "DATASET_TYPE", "ARTIFACT_ID" FROM {dataset_table} '
                    'WHERE "DATASET_ID" = :dataset_id FOR UPDATE'
                ),
                {"dataset_id": dataset_id},
            ).mappings().one_or_none()
            if row is None:
                return False
            row = normalize_row_keys(row)
            artifact_id = row["ARTIFACT_ID"]
            pinned = connection.execute(
                text(f'SELECT COUNT(1) FROM {pin_table} WHERE "DATASET_ID" = :dataset_id'),
                {"dataset_id": dataset_id},
            ).scalar_one()
            directly_used = connection.execute(
                text(
                    f'SELECT COUNT(1) FROM {upload_table} WHERE '
                    '"BOM_ARTIFACT_ID" = :artifact_id OR "RULE_ARTIFACT_ID" = :artifact_id'
                ),
                {"artifact_id": artifact_id},
            ).scalar_one()
            if pinned or directly_used:
                raise PersistenceStateError(
                    "REFERENCE_DATASET_IN_USE",
                    "This file is used by an ECN upload. Delete that analysis history before removing the file.",
                    status_code=409,
                )
            comparisons = connection.execute(
                text(
                    f'SELECT "DATASET_ID", "CONTEXT_JSON" FROM {dataset_table} '
                    'WHERE "DATASET_ID" <> :dataset_id'
                ),
                {"dataset_id": dataset_id},
            ).mappings().all()
            if any(
                json.loads(read_lob_text(normalize_row_keys(other).get("CONTEXT_JSON") or "{}") or "{}")
                .get("comparison_dataset_id") == dataset_id
                for other in comparisons
            ):
                raise PersistenceStateError(
                    "REFERENCE_DATASET_COMPARISON_IN_USE",
                    "This file is the comparison baseline for another dataset. Delete the dependent version first.",
                    status_code=409,
                )
            dataset_type = ReferenceDatasetType(row["DATASET_TYPE"])
            row_table_key = _ROW_TABLE_BY_TYPE.get(dataset_type)
            if row_table_key:
                row_table = quote_identifier(getattr(self._tables, row_table_key))
                connection.execute(
                    text(f'DELETE FROM {row_table} WHERE "DATASET_ID" = :dataset_id'),
                    {"dataset_id": dataset_id},
                )
            if dataset_type is ReferenceDatasetType.BOM:
                connection.execute(
                    text(f'DELETE FROM {bom_table} WHERE "ARTIFACT_ID" = :artifact_id'),
                    {"artifact_id": artifact_id},
                )
            connection.execute(
                text(f'DELETE FROM {dataset_table} WHERE "DATASET_ID" = :dataset_id'),
                {"dataset_id": dataset_id},
            )
            connection.execute(
                text(f'DELETE FROM {artifact_table} WHERE "ARTIFACT_ID" = :artifact_id'),
                {"artifact_id": artifact_id},
            )
        return True

    def _sample_from_row(
        self, record: ReferenceDatasetRecord, row: Mapping[str, Any]
    ) -> NormalizedReferenceRow:
        """Decode one table row into a generic API-safe sample record."""
        row = normalize_row_keys(row)
        ignored = {"ROW_ID", "DATASET_ID", "SHEET_NAME", "SOURCE_ROW_NUMBER", "NORMALIZED_JSON"}
        values: dict[str, JsonValue] = {
            key.lower(): value
            for key, value in row.items()
            if key not in ignored
            and (value is None or isinstance(value, (str, int, float, bool)))
        }
        if "NORMALIZED_JSON" in row:
            decoded = json.loads(row["NORMALIZED_JSON"])
            if isinstance(decoded, dict):
                values = decoded
        return NormalizedReferenceRow(
            row_id=row["ROW_ID"],
            artifact_id=record.artifact_id,
            sheet_name=row.get("SHEET_NAME") or "Default",
            source_row_number=int(row["SOURCE_ROW_NUMBER"]),
            values=values,
        )
