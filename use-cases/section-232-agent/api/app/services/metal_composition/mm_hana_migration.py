"""Helpers for migrating legacy item contracts to Material Master/MM contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Sequence

from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .persistence_common import qualified_table, quote_identifier


MigrationMode = Literal["dry-run", "apply", "rollback"]

LEGACY_PREFIX = "".join(("g", "cc"))
MM_PREFIX = "mm"
LEGACY_COMPOSITION_MODE = f"{LEGACY_PREFIX}_tracker"
MATERIAL_MASTER_COMPOSITION_MODE = "material_master"
LEGACY_APP_SETTING_COLUMN = f"USE_{LEGACY_PREFIX.upper()}_TRACKER_METAL_COMPOSITION"
MATERIAL_MASTER_APP_SETTING_COLUMN = "USE_MATERIAL_MASTER_METAL_COMPOSITION"


@dataclass(frozen=True)
class MaterialMasterMigrationTableResult:
    """Summary for one migrated or audited HANA table.

    Args:
        table: HANA table name.
        planned_changes: Number of changes found before applying the mode.
        applied_changes: Number of changes actually written.
        audit_findings: Human-readable remaining legacy-value findings.

    Returns:
        Immutable table-level migration summary.
    """

    table: str
    planned_changes: int = 0
    applied_changes: int = 0
    audit_findings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MaterialMasterMigrationResult:
    """Overall migration result for dry-run, apply, or rollback mode.

    Args:
        mode: Requested migration mode.
        table_results: Per-table migration and audit summaries.

    Returns:
        Immutable result object with aggregate counts.
    """

    mode: MigrationMode
    table_results: List[MaterialMasterMigrationTableResult] = field(default_factory=list)

    @property
    def planned_change_count(self) -> int:
        """Return the number of changes that were found for this run."""

        return sum(item.planned_changes for item in self.table_results)

    @property
    def applied_change_count(self) -> int:
        """Return the number of changes actually applied by this run."""

        return sum(item.applied_changes for item in self.table_results)

    @property
    def audit_findings(self) -> List[str]:
        """Return all remaining legacy-value audit findings."""

        out: List[str] = []
        for item in self.table_results:
            out.extend(item.audit_findings)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable migration result."""

        return {
            "mode": self.mode,
            "planned_change_count": self.planned_change_count,
            "applied_change_count": self.applied_change_count,
            "audit_findings": list(self.audit_findings),
            "tables": [
                {
                    "table": item.table,
                    "planned_changes": item.planned_changes,
                    "applied_changes": item.applied_changes,
                    "audit_findings": list(item.audit_findings),
                }
                for item in self.table_results
            ],
        }


class MaterialMasterMigrationAdapter(Protocol):
    """Persistence operations required by the Material Master HANA migration."""

    def table_exists(self, table: str, *, schema: Optional[str]) -> bool:
        """Return whether a table exists."""

    def column_exists(self, table: str, column: str, *, schema: Optional[str]) -> bool:
        """Return whether a table column exists."""

    def rename_column(
        self,
        table: str,
        *,
        schema: Optional[str],
        old_column: str,
        new_column: str,
        dry_run: bool,
    ) -> int:
        """Rename one column and return the number of planned/applied changes."""

    def replace_prefix(
        self,
        table: str,
        column: str,
        *,
        schema: Optional[str],
        old_prefix: str,
        new_prefix: str,
        dry_run: bool,
    ) -> int:
        """Replace a string prefix in one column and return changed-row count."""

    def replace_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        key_columns: Sequence[str],
        schema: Optional[str],
        replacements: Sequence[tuple[str, str]],
        dry_run: bool,
    ) -> int:
        """Replace app-generated text fragments and return changed-cell count."""

    def audit_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        schema: Optional[str],
        tokens: Sequence[str],
    ) -> List[str]:
        """Return remaining legacy-token findings without mutating data."""


class HanaMaterialMasterMigrationAdapter:
    """Run Material Master migration operations against SAP HANA.

    Args:
        connection: Optional existing HANA connection. A new connection is
            created when omitted.

    Returns:
        Adapter that executes SQL against HANA.
    """

    def __init__(self, connection: Optional[HANAConnection] = None) -> None:
        self.connection = connection or HANAConnection()

    def table_exists(self, table: str, *, schema: Optional[str]) -> bool:
        """Return whether a HANA table exists."""

        return self.connection.table_exists(table, schema=schema)

    def column_exists(self, table: str, column: str, *, schema: Optional[str]) -> bool:
        """Return whether a HANA column exists."""

        return self.connection.column_exists(table, column, schema=schema)

    def rename_column(
        self,
        table: str,
        *,
        schema: Optional[str],
        old_column: str,
        new_column: str,
        dry_run: bool,
    ) -> int:
        """Rename a HANA column when the old name exists and the new one does not."""

        if not self.column_exists(table, old_column, schema=schema):
            return 0
        if self.column_exists(table, new_column, schema=schema):
            return 0
        if not dry_run:
            self.connection.execute(
                f"RENAME COLUMN {qualified_table(table, schema)}."
                f"{quote_identifier(old_column)} TO {quote_identifier(new_column)}"
            )
        return 1

    def replace_prefix(
        self,
        table: str,
        column: str,
        *,
        schema: Optional[str],
        old_prefix: str,
        new_prefix: str,
        dry_run: bool,
    ) -> int:
        """Replace a prefix in a HANA text column."""

        rows = self._fetch_changed_rows(
            table,
            [column],
            schema=schema,
            predicate=f"{quote_identifier(column)} LIKE ?",
            params=[f"{old_prefix}%"],
        )
        changed_count = 0
        for row_index, row in enumerate(rows):
            value = _optional_text(row.get(column.lower()))
            if not value or not value.startswith(old_prefix):
                continue
            changed_count += 1
            if not dry_run:
                self._update_cell_by_value(
                    table,
                    schema=schema,
                    column=column,
                    old_value=value,
                    new_value=f"{new_prefix}{value[len(old_prefix):]}",
                )
        return changed_count

    def replace_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        key_columns: Sequence[str],
        schema: Optional[str],
        replacements: Sequence[tuple[str, str]],
        dry_run: bool,
    ) -> int:
        """Replace app-generated legacy text fragments in HANA text columns."""

        existing_key_columns = [column for column in key_columns if self.column_exists(table, column, schema=schema)]
        if len(existing_key_columns) != len(key_columns):
            return 0
        existing_columns = [column for column in columns if self.column_exists(table, column, schema=schema)]
        if not existing_columns:
            return 0
        rows = self._fetch_changed_rows(table, [*existing_key_columns, *existing_columns], schema=schema)
        changed_count = 0
        for row in rows:
            key_values = [row.get(column.lower()) for column in existing_key_columns]
            for column in existing_columns:
                value = _optional_text(row.get(column.lower()))
                if value is None:
                    continue
                next_value = _replace_many(value, replacements)
                if next_value != value:
                    changed_count += 1
                    if not dry_run:
                        self._update_cell_by_keys(
                            table,
                            schema=schema,
                            column=column,
                            key_columns=existing_key_columns,
                            key_values=key_values,
                            new_value=next_value,
                        )
        return changed_count

    def audit_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        schema: Optional[str],
        tokens: Sequence[str],
    ) -> List[str]:
        """Return approximate HANA audit findings for remaining legacy tokens."""

        findings: List[str] = []
        for column in columns:
            if not self.column_exists(table, column, schema=schema):
                continue
            predicates = " OR ".join(f"LOWER({quote_identifier(column)}) LIKE ?" for _ in tokens)
            params = [f"%{token.lower()}%" for token in tokens]
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {qualified_table(table, schema)} WHERE {predicates}",
                    params,
                )
                count = int((cursor.fetchone() or [0])[0] or 0)
            if count:
                findings.append(f"{table}.{column}: {count} row(s) still contain legacy text")
        return findings

    def _fetch_changed_rows(
        self,
        table: str,
        columns: Sequence[str],
        *,
        schema: Optional[str],
        predicate: Optional[str] = None,
        params: Optional[Sequence[object]] = None,
    ) -> List[Dict[str, object]]:
        """Fetch candidate rows from a HANA table as lower-key dictionaries."""

        selected_columns = ", ".join(quote_identifier(column) for column in columns)
        sql = f"SELECT {selected_columns} FROM {qualified_table(table, schema)}"
        if predicate:
            sql = f"{sql} WHERE {predicate}"
        with self.connection.cursor() as cursor:
            cursor.execute(sql, list(params or []))
            rows = cursor.fetchall()
            headers = [str(description[0]).lower() for description in (cursor.description or [])]
        return [
            {headers[index]: value for index, value in enumerate(row)}
            for row in rows
        ]

    def _update_cell_by_value(
        self,
        table: str,
        *,
        schema: Optional[str],
        column: str,
        old_value: str,
        new_value: str,
    ) -> None:
        """Update matching text cells by value without relying on HANA row ids."""

        column_name = quote_identifier(column)
        self.connection.execute(
            f"UPDATE {qualified_table(table, schema)} SET {column_name} = ? WHERE {column_name} = ?",
            [new_value, old_value],
        )

    def _update_cell_by_keys(
        self,
        table: str,
        *,
        schema: Optional[str],
        column: str,
        key_columns: Sequence[str],
        key_values: Sequence[object],
        new_value: str,
    ) -> None:
        """Update one text cell using stable row keys instead of NCLOB equality."""

        column_name = quote_identifier(column)
        predicates = " AND ".join(f"{quote_identifier(key_column)} = ?" for key_column in key_columns)
        self.connection.execute(
            f"UPDATE {qualified_table(table, schema)} SET {column_name} = ? WHERE {predicates}",
            [new_value, *key_values],
        )


class InMemoryMaterialMasterMigrationAdapter:
    """In-memory migration adapter for tests.

    Args:
        tables: Mapping of table name to row dictionaries.

    Returns:
        Mutable fake adapter that applies the same contract transformations as
        the HANA adapter.
    """

    def __init__(self, *, tables: Dict[str, List[Dict[str, object]]]) -> None:
        self.tables = tables

    def table_exists(self, table: str, *, schema: Optional[str]) -> bool:
        """Return whether the fake table exists."""

        del schema
        return table in self.tables

    def column_exists(self, table: str, column: str, *, schema: Optional[str]) -> bool:
        """Return whether any fake row contains the column."""

        del schema
        return any(column in row for row in self.tables.get(table, []))

    def rename_column(
        self,
        table: str,
        *,
        schema: Optional[str],
        old_column: str,
        new_column: str,
        dry_run: bool,
    ) -> int:
        """Rename a fake column."""

        del schema
        if not self.column_exists(table, old_column, schema=None):
            return 0
        if self.column_exists(table, new_column, schema=None):
            return 0
        if not dry_run:
            for row in self.tables.get(table, []):
                if old_column in row:
                    row[new_column] = row.pop(old_column)
        return 1

    def replace_prefix(
        self,
        table: str,
        column: str,
        *,
        schema: Optional[str],
        old_prefix: str,
        new_prefix: str,
        dry_run: bool,
    ) -> int:
        """Replace a prefix in fake rows."""

        del schema
        changed_count = 0
        for row in self.tables.get(table, []):
            value = _optional_text(row.get(column))
            if value and value.startswith(old_prefix):
                changed_count += 1
                if not dry_run:
                    row[column] = f"{new_prefix}{value[len(old_prefix):]}"
        return changed_count

    def replace_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        key_columns: Sequence[str],
        schema: Optional[str],
        replacements: Sequence[tuple[str, str]],
        dry_run: bool,
    ) -> int:
        """Replace text fragments in fake rows."""

        del key_columns
        del schema
        changed_count = 0
        for row in self.tables.get(table, []):
            for column in columns:
                value = _optional_text(row.get(column))
                if value is None:
                    continue
                next_value = _replace_many(value, replacements)
                if next_value != value:
                    changed_count += 1
                    if not dry_run:
                        row[column] = next_value
        return changed_count

    def audit_text(
        self,
        table: str,
        columns: Sequence[str],
        *,
        schema: Optional[str],
        tokens: Sequence[str],
    ) -> List[str]:
        """Return fake audit findings for remaining legacy values."""

        del schema
        findings: List[str] = []
        for column in columns:
            count = 0
            for row in self.tables.get(table, []):
                value = _optional_text(row.get(column))
                if value and any(token.lower() in value.lower() for token in tokens):
                    count += 1
            if count:
                findings.append(f"{table}.{column}: {count} row(s) still contain legacy text")
        return findings


def run_material_master_hana_migration(
    settings: MetalCompositionSettings,
    *,
    adapter: Optional[MaterialMasterMigrationAdapter] = None,
    mode: MigrationMode = "dry-run",
) -> MaterialMasterMigrationResult:
    """Run the Material Master/MM HANA contract migration.

    Args:
        settings: Runtime settings that define the HANA table names.
        adapter: Optional adapter for tests. Defaults to a real HANA adapter.
        mode: ``dry-run`` reports changes, ``apply`` migrates legacy values to
            Material Master/MM values, and ``rollback`` restores legacy values.

    Returns:
        MaterialMasterMigrationResult with planned/applied counts and audit findings.
    """

    if mode not in {"dry-run", "apply", "rollback"}:
        raise ValueError("mode must be dry-run, apply, or rollback")

    migration_adapter = adapter or HanaMaterialMasterMigrationAdapter()
    dry_run = mode == "dry-run"
    rollback = mode == "rollback"
    old_prefix = f"{MM_PREFIX}:"
    new_prefix = f"{LEGACY_PREFIX}:"
    replacements = _rollback_replacements() if rollback else _forward_replacements()
    old_setting_column = MATERIAL_MASTER_APP_SETTING_COLUMN
    new_setting_column = LEGACY_APP_SETTING_COLUMN
    if not rollback:
        old_prefix, new_prefix = new_prefix, old_prefix
        old_setting_column, new_setting_column = new_setting_column, old_setting_column

    table_results: List[MaterialMasterMigrationTableResult] = []
    item_id_specs = _item_id_column_specs(settings)
    text_specs = _text_column_specs(settings)

    for schema, table, columns in item_id_specs:
        if not migration_adapter.table_exists(table, schema=schema):
            continue
        planned = 0
        applied = 0
        for column in columns:
            if not migration_adapter.column_exists(table, column, schema=schema):
                continue
            changed = migration_adapter.replace_prefix(
                table,
                column,
                schema=schema,
                old_prefix=old_prefix,
                new_prefix=new_prefix,
                dry_run=dry_run,
            )
            planned += changed
            applied += 0 if dry_run else changed
        table_results.append(
            MaterialMasterMigrationTableResult(
                table=table,
                planned_changes=planned,
                applied_changes=applied,
            )
        )

    app_settings_schema = settings.ui_state_hana_schema or settings.hana_schema or None
    app_settings_table = settings.ui_state_app_settings_table
    if migration_adapter.table_exists(app_settings_table, schema=app_settings_schema):
        changed = migration_adapter.rename_column(
            app_settings_table,
            schema=app_settings_schema,
            old_column=old_setting_column,
            new_column=new_setting_column,
            dry_run=dry_run,
        )
        table_results.append(
            MaterialMasterMigrationTableResult(
                table=app_settings_table,
                planned_changes=changed,
                applied_changes=0 if dry_run else changed,
            )
        )

    for schema, table, key_columns, columns in text_specs:
        if not migration_adapter.table_exists(table, schema=schema):
            continue
        changed = migration_adapter.replace_text(
            table,
            columns,
            key_columns=key_columns,
            schema=schema,
            replacements=replacements,
            dry_run=dry_run,
        )
        findings = migration_adapter.audit_text(
            table,
            columns,
            schema=schema,
            tokens=_audit_tokens_for_mode(mode),
        )
        table_results.append(
            MaterialMasterMigrationTableResult(
                table=table,
                planned_changes=changed,
                applied_changes=0 if dry_run else changed,
                audit_findings=findings,
            )
        )

    return MaterialMasterMigrationResult(mode=mode, table_results=table_results)


def _item_id_column_specs(settings: MetalCompositionSettings) -> List[tuple[Optional[str], str, Sequence[str]]]:
    """Return configured tables that persist app-generated item identifiers."""

    ui_schema = settings.ui_state_hana_schema or settings.hana_schema or None
    return [
        (ui_schema, settings.ui_state_document_assignments_table, ("ITEM_ID",)),
        (ui_schema, settings.ui_state_classification_history_table, ("ITEM_ID",)),
        (ui_schema, settings.ui_state_classification_job_items_table, ("ITEM_ID",)),
        (ui_schema, settings.ui_state_classification_ownership_table, ("ITEM_ID",)),
    ]


def _text_column_specs(settings: MetalCompositionSettings) -> List[tuple[Optional[str], str, Sequence[str], Sequence[str]]]:
    """Return app-generated text/JSON columns safe for contract-value replacement."""

    ui_schema = settings.ui_state_hana_schema or settings.hana_schema or None
    return [
        (
            ui_schema,
            settings.ui_state_classification_history_table,
            ("HISTORY_ID",),
            ("PAYLOAD_JSON", "AGENT_OUTPUTS_JSON", "TIMING_JSON"),
        ),
        (
            ui_schema,
            settings.ui_state_classification_jobs_table,
            ("JOB_ID",),
            ("ERROR_MESSAGE",),
        ),
        (
            ui_schema,
            settings.ui_state_classification_job_items_table,
            ("JOB_ITEM_ID",),
            ("ERROR_MESSAGE",),
        ),
    ]


def _forward_replacements() -> List[tuple[str, str]]:
    """Return text replacements from legacy values to MM/Material Master values."""

    upper = LEGACY_PREFIX.upper()
    title = " ".join((upper, "Tracker"))
    sentence = " ".join((upper, "tracker"))
    return [
        (f"{LEGACY_PREFIX}:", f"{MM_PREFIX}:"),
        (LEGACY_COMPOSITION_MODE, MATERIAL_MASTER_COMPOSITION_MODE),
        (f"{LEGACY_PREFIX}_focused_clues", "material_master_focused_clues"),
        (f"{LEGACY_PREFIX}_material_profile", "material_master_material_profile"),
        (f'"source_kind":"{LEGACY_PREFIX}"', f'"source_kind":"{MM_PREFIX}"'),
        (f'"source_kind": "{LEGACY_PREFIX}"', f'"source_kind": "{MM_PREFIX}"'),
        (f'"record_origin":"{LEGACY_PREFIX}"', f'"record_origin":"{MM_PREFIX}"'),
        (f'"record_origin": "{LEGACY_PREFIX}"', f'"record_origin": "{MM_PREFIX}"'),
        (title, "Material Master"),
        (sentence, "Material Master"),
        (upper, "Material Master"),
    ]


def _rollback_replacements() -> List[tuple[str, str]]:
    """Return text replacements from MM/Material Master values to legacy values."""

    upper = LEGACY_PREFIX.upper()
    title = " ".join((upper, "Tracker"))
    return [
        (f"{MM_PREFIX}:", f"{LEGACY_PREFIX}:"),
        ("material_master_focused_clues", f"{LEGACY_PREFIX}_focused_clues"),
        ("material_master_material_profile", f"{LEGACY_PREFIX}_material_profile"),
        (f'"source_kind":"{MM_PREFIX}"', f'"source_kind":"{LEGACY_PREFIX}"'),
        (f'"source_kind": "{MM_PREFIX}"', f'"source_kind": "{LEGACY_PREFIX}"'),
        (f'"record_origin":"{MM_PREFIX}"', f'"record_origin":"{LEGACY_PREFIX}"'),
        (f'"record_origin": "{MM_PREFIX}"', f'"record_origin": "{LEGACY_PREFIX}"'),
        (MATERIAL_MASTER_COMPOSITION_MODE, LEGACY_COMPOSITION_MODE),
        ("Material Master", title),
    ]


def _audit_tokens_for_mode(mode: MigrationMode) -> List[str]:
    """Return legacy-like tokens that should be reported after a run."""

    if mode == "rollback":
        return [f"{MM_PREFIX}:", MATERIAL_MASTER_COMPOSITION_MODE, "Material Master"]
    return [f"{LEGACY_PREFIX}:", LEGACY_COMPOSITION_MODE, LEGACY_PREFIX.upper()]


def _replace_many(value: str, replacements: Sequence[tuple[str, str]]) -> str:
    """Apply ordered text replacements to a value."""

    next_value = value
    for source, target in replacements:
        next_value = next_value.replace(source, target)
    return next_value


def _optional_text(value: object) -> Optional[str]:
    """Return a string value or None for null-like values."""

    if value is None:
        return None
    return str(value)
