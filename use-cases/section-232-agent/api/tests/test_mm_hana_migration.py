from __future__ import annotations

from pathlib import Path

from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.mm_hana_migration import (
    HanaMaterialMasterMigrationAdapter,
    InMemoryMaterialMasterMigrationAdapter,
    run_material_master_hana_migration,
)


class _RenameColumnConnection:
    """Capture HANA DDL emitted by the migration adapter during unit tests."""

    def __init__(self) -> None:
        self.executed_sql: list[str] = []

    def table_columns(self, table: str, *, schema: str | None = None) -> list[str]:
        """Return the fake columns available before the rename."""

        assert table == "APP_SETTINGS"
        assert schema == "APP_SCHEMA"
        return ["USE_GCC_TRACKER_METAL_COMPOSITION"]

    def column_exists(self, table: str, column: str, *, schema: str | None = None) -> bool:
        """Return whether the fake table contains the requested column."""

        return column in self.table_columns(table, schema=schema)

    def execute(self, sql: str, params: list[object] | None = None) -> None:
        """Capture the SQL statement instead of sending it to HANA."""

        assert params is None
        self.executed_sql.append(sql)


class _TextReplacementCursor:
    """Serve one fake HANA SELECT result for text-replacement tests."""

    description = [("HISTORY_ID",), ("PAYLOAD_JSON",)]

    def __init__(self) -> None:
        self.sql: str | None = None
        self.params: list[object] | None = None

    def __enter__(self) -> "_TextReplacementCursor":
        """Return this fake cursor from a context manager."""

        return self

    def __exit__(self, *_args: object) -> None:
        """Close the fake cursor when the context manager exits."""

        self.close()

    def execute(self, sql: str, params: list[object] | None = None) -> None:
        """Capture the SELECT statement and parameters."""

        self.sql = sql
        self.params = params or []

    def fetchall(self) -> list[tuple[str, str]]:
        """Return one row whose NCLOB-like payload needs replacement."""

        return [("history-1", '{"item_id":"gcc:1001","mode":"gcc_tracker"}')]

    def close(self) -> None:
        """Mirror the hdbcli cursor close method."""


class _TextReplacementConnection:
    """Capture HANA text replacement SQL without comparing NCLOB values."""

    def __init__(self) -> None:
        self.cursor_instance = _TextReplacementCursor()
        self.executed_sql: list[tuple[str, list[object]]] = []

    def column_exists(self, table: str, column: str, *, schema: str | None = None) -> bool:
        """Return whether the fake history table contains the requested column."""

        assert table == "HISTORY"
        assert schema is None
        return column in {"HISTORY_ID", "PAYLOAD_JSON"}

    def cursor(self) -> _TextReplacementCursor:
        """Return a context-managed cursor-like object."""

        return self.cursor_instance

    def execute(self, sql: str, params: list[object] | None = None) -> None:
        """Capture UPDATE statements and bound values."""

        self.executed_sql.append((sql, list(params or [])))


def _settings(tmp_path: Path) -> MetalCompositionSettings:
    """Build isolated migration settings for fake HANA table tests."""

    return MetalCompositionSettings(
        workbook_path=tmp_path / "Material Master.xlsb",
        api_env_path=tmp_path / ".env",
        ui_state_document_assignments_table="DOC_ASSIGNMENTS",
        ui_state_app_settings_table="APP_SETTINGS",
        ui_state_classification_history_table="HISTORY",
        ui_state_classification_jobs_table="JOBS",
        ui_state_classification_job_items_table="JOB_ITEMS",
        ui_state_classification_ownership_table="OWNERSHIP",
    )


def test_material_master_hana_migration_dry_run_reports_without_mutating(tmp_path: Path) -> None:
    """Verify dry-run reports pending changes without mutating fake HANA rows."""

    legacy_prefix = "".join(("g", "cc"))
    legacy_settings_column = f"USE_{legacy_prefix.upper()}_TRACKER_METAL_COMPOSITION"
    adapter = InMemoryMaterialMasterMigrationAdapter(
        tables={
            "DOC_ASSIGNMENTS": [
                {"ENTRY_ID": "entry-1", "ITEM_ID": f"{legacy_prefix}:1001"},
            ],
            "APP_SETTINGS": [
                {"SETTINGS_ID": "global", legacy_settings_column: 1},
            ],
        }
    )

    result = run_material_master_hana_migration(
        _settings(tmp_path),
        adapter=adapter,
        mode="dry-run",
    )

    assert result.mode == "dry-run"
    assert result.planned_change_count == 2
    assert result.applied_change_count == 0
    assert adapter.tables["DOC_ASSIGNMENTS"][0]["ITEM_ID"] == f"{legacy_prefix}:1001"
    assert legacy_settings_column in adapter.tables["APP_SETTINGS"][0]


def test_material_master_hana_migration_apply_and_rollback_update_contract_values(tmp_path: Path) -> None:
    """Verify apply and rollback transform item ids, JSON values, and settings columns."""

    legacy_prefix = "".join(("g", "cc"))
    legacy_settings_column = f"USE_{legacy_prefix.upper()}_TRACKER_METAL_COMPOSITION"
    adapter = InMemoryMaterialMasterMigrationAdapter(
        tables={
            "DOC_ASSIGNMENTS": [
                {"ENTRY_ID": "entry-1", "ITEM_ID": f"{legacy_prefix}:1001"},
            ],
            "HISTORY": [
                {
                    "HISTORY_ID": "history-1",
                    "ITEM_ID": f"{legacy_prefix}:1001",
                    "PAYLOAD_JSON": (
                        '{"product_code":"'
                        f'{legacy_prefix}:1001'
                        '","selected_source":{"source_kind":"'
                        f'{legacy_prefix}'
                        '"},"final_composition":{"provenance":{"dominant_source":"'
                        f'{legacy_prefix}_tracker'
                        '"}},"timing":{"phases":{"diagram":{"details":{"analysis_mode":"'
                        f'{legacy_prefix}_focused_clues","'
                        f'{legacy_prefix}_material_profile'
                        '":{"confidence":1.0}}},"resolve_source":{"details":{"record_origin":"'
                        f'{legacy_prefix}'
                        '"}}}}}'
                    ),
                    "AGENT_OUTPUTS_JSON": None,
                    "TIMING_JSON": "{}",
                },
            ],
            "APP_SETTINGS": [
                {"SETTINGS_ID": "global", legacy_settings_column: 1},
            ],
        }
    )

    apply_result = run_material_master_hana_migration(
        _settings(tmp_path),
        adapter=adapter,
        mode="apply",
    )

    assert apply_result.applied_change_count == apply_result.planned_change_count
    assert adapter.tables["DOC_ASSIGNMENTS"][0]["ITEM_ID"] == "mm:1001"
    assert adapter.tables["HISTORY"][0]["ITEM_ID"] == "mm:1001"
    assert '"source_kind":"mm"' in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert "material_master" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert "material_master_focused_clues" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert "material_master_material_profile" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert '"record_origin":"mm"' in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert "USE_MATERIAL_MASTER_METAL_COMPOSITION" in adapter.tables["APP_SETTINGS"][0]

    rollback_result = run_material_master_hana_migration(
        _settings(tmp_path),
        adapter=adapter,
        mode="rollback",
    )

    assert rollback_result.applied_change_count == rollback_result.planned_change_count
    assert adapter.tables["DOC_ASSIGNMENTS"][0]["ITEM_ID"] == f"{legacy_prefix}:1001"
    assert adapter.tables["HISTORY"][0]["ITEM_ID"] == f"{legacy_prefix}:1001"
    assert f'"source_kind":"{legacy_prefix}"' in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert f"{legacy_prefix}_tracker" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert f"{legacy_prefix}_focused_clues" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert f"{legacy_prefix}_material_profile" in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert f'"record_origin":"{legacy_prefix}"' in adapter.tables["HISTORY"][0]["PAYLOAD_JSON"]
    assert legacy_settings_column in adapter.tables["APP_SETTINGS"][0]


def test_hana_adapter_uses_hana_rename_column_statement() -> None:
    """Verify the real HANA adapter emits SAP HANA's standalone rename syntax."""

    connection = _RenameColumnConnection()
    adapter = HanaMaterialMasterMigrationAdapter(connection=connection)  # type: ignore[arg-type]

    changed = adapter.rename_column(
        "APP_SETTINGS",
        schema="APP_SCHEMA",
        old_column="USE_GCC_TRACKER_METAL_COMPOSITION",
        new_column="USE_MATERIAL_MASTER_METAL_COMPOSITION",
        dry_run=False,
    )

    assert changed == 1
    assert connection.executed_sql == [
        'RENAME COLUMN "APP_SCHEMA"."APP_SETTINGS"."USE_GCC_TRACKER_METAL_COMPOSITION" '
        'TO "USE_MATERIAL_MASTER_METAL_COMPOSITION"'
    ]


def test_hana_adapter_updates_nclob_text_by_key_columns() -> None:
    """Verify NCLOB text replacements use row keys instead of NCLOB equality."""

    connection = _TextReplacementConnection()
    adapter = HanaMaterialMasterMigrationAdapter(connection=connection)  # type: ignore[arg-type]

    changed = adapter.replace_text(
        "HISTORY",
        ["PAYLOAD_JSON"],
        key_columns=["HISTORY_ID"],
        schema=None,
        replacements=[("gcc:", "mm:"), ("gcc_tracker", "material_master")],
        dry_run=False,
    )

    assert changed == 1
    assert connection.executed_sql == [
        (
            'UPDATE "HISTORY" SET "PAYLOAD_JSON" = ? WHERE "HISTORY_ID" = ?',
            ['{"item_id":"mm:1001","mode":"material_master"}', "history-1"],
        )
    ]
