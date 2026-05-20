"""Persistence helpers for UI document assignments and saved classification state."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.models.metal_composition import (
    MetalCompositionAppSettings,
    MetalCompositionResponse,
)
from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings
from .persistence_common import (
    ensure_index,
    fetch_rows,
    qualified_table as _qualified_table,
)
from .timing import utc_now_iso


@dataclass(frozen=True)
class StoredDocumentReference:
    source: str
    relative_path: str


@dataclass(frozen=True)
class PersistedClassificationSnapshot:
    item_id: str
    dataset_scope: str
    payload: MetalCompositionResponse
    status: str
    last_classified_at: str
    agent_outputs_json: Optional[str] = None
    timing_json: Optional[str] = None


@dataclass(frozen=True)
class PersistedAppSettings:
    use_gcc_tracker_metal_composition: bool = True
    updated_at: Optional[str] = None

    def to_model(self) -> MetalCompositionAppSettings:
        return MetalCompositionAppSettings(
            use_gcc_tracker_metal_composition=bool(self.use_gcc_tracker_metal_composition),
            updated_at=self.updated_at,
        )


class InMemoryMetalCompositionUIStateStore:
    """Test double that preserves the HANA store semantics in memory."""

    def __init__(self) -> None:
        self._document_history: List[tuple[str, str, str, str, List[StoredDocumentReference]]] = []
        self._classification_history: List[PersistedClassificationSnapshot] = []
        self._app_settings = PersistedAppSettings()

    def replace_document_assignments(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        document_refs: Sequence[StoredDocumentReference],
    ) -> List[StoredDocumentReference]:
        self._document_history.append(
            (
                item_id,
                dataset_scope,
                str(uuid.uuid4()),
                utc_now_iso(),
                list(document_refs),
            )
        )
        return list(document_refs)

    def get_document_assignments(self, item_id: str, *, dataset_scope: str) -> List[StoredDocumentReference]:
        for stored_item_id, stored_scope, _revision_id, _updated_at, refs in reversed(self._document_history):
            if stored_item_id == item_id and stored_scope == dataset_scope:
                return list(refs)
        return []

    def get_document_assignment_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], List[StoredDocumentReference]]:
        return {
            key: self.get_document_assignments(key[0], dataset_scope=key[1])
            for key in list(keys)
        }

    def get_document_assignment_keys(self) -> Set[Tuple[str, str]]:
        keys: Set[Tuple[str, str]] = set()
        cleared: Set[Tuple[str, str]] = set()
        for item_id, dataset_scope, _revision_id, _updated_at, refs in reversed(self._document_history):
            key = (item_id, dataset_scope)
            if key in keys or key in cleared:
                continue
            if refs:
                keys.add(key)
            else:
                cleared.add(key)
        return keys

    def save_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        result: MetalCompositionResponse,
        last_classified_at: Optional[str] = None,
    ) -> PersistedClassificationSnapshot:
        payload_dict = result.model_dump(mode="json")
        snapshot = PersistedClassificationSnapshot(
            item_id=item_id,
            dataset_scope=dataset_scope,
            payload=result,
            status=result.status,
            last_classified_at=last_classified_at or utc_now_iso(),
            agent_outputs_json=(
                json.dumps(payload_dict.get("agent_outputs"), sort_keys=True)
                if payload_dict.get("agent_outputs") is not None
                else None
            ),
            timing_json=json.dumps(payload_dict.get("timing") or {}, sort_keys=True),
        )
        self._classification_history.append(snapshot)
        return snapshot

    def get_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
    ) -> Optional[PersistedClassificationSnapshot]:
        for snapshot in reversed(self._classification_history):
            if snapshot.item_id == item_id and snapshot.dataset_scope == dataset_scope:
                return snapshot
        return None

    def get_classification_snapshot_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], PersistedClassificationSnapshot]:
        return {
            key: snapshot
            for key in list(keys)
            if (snapshot := self.get_classification_snapshot(key[0], dataset_scope=key[1])) is not None
        }

    def get_classification_snapshot_keys(self) -> Set[Tuple[str, str]]:
        keys: Set[Tuple[str, str]] = set()
        for snapshot in reversed(self._classification_history):
            keys.add((snapshot.item_id, snapshot.dataset_scope))
        return keys

    def get_classification_stats(self) -> tuple[int, Optional[str]]:
        if not self._classification_history:
            return 0, None
        latest_snapshot_by_key: Dict[Tuple[str, str], PersistedClassificationSnapshot] = {}
        for snapshot in reversed(self._classification_history):
            latest_snapshot_by_key.setdefault((snapshot.item_id, snapshot.dataset_scope), snapshot)
        latest_classified_at = max(
            (snapshot.last_classified_at for snapshot in latest_snapshot_by_key.values()),
            default=None,
        )
        return len(latest_snapshot_by_key), latest_classified_at

    def clear_classification_snapshots(self) -> int:
        count, _latest_classified_at = self.get_classification_stats()
        self._classification_history = []
        return count

    def get_app_settings(self) -> PersistedAppSettings:
        return self._app_settings

    def update_app_settings(
        self,
        *,
        use_gcc_tracker_metal_composition: bool,
    ) -> PersistedAppSettings:
        self._app_settings = PersistedAppSettings(
            use_gcc_tracker_metal_composition=bool(use_gcc_tracker_metal_composition),
            updated_at=utc_now_iso(),
        )
        return self._app_settings

class MetalCompositionUIStateStore:
    """Persist UI state in SAP HANA with append-only history."""

    def __init__(
        self,
        settings: MetalCompositionSettings,
        *,
        connection: Optional[HANAConnection] = None,
    ) -> None:
        self.settings = settings
        self.connection = connection or HANAConnection()
        self.schema = settings.ui_state_hana_schema or settings.hana_schema or None
        self.document_assignments_table = settings.ui_state_document_assignments_table
        self.app_settings_table = settings.ui_state_app_settings_table
        self.classification_history_table = settings.ui_state_classification_history_table
        self._initialize()

    def _initialize(self) -> None:
        if not self.connection.table_exists(self.document_assignments_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.document_assignments_table, self.schema)} (
                    "ENTRY_ID" NVARCHAR(36) PRIMARY KEY,
                    "REVISION_ID" NVARCHAR(36) NOT NULL,
                    "ITEM_ID" NVARCHAR(255) NOT NULL,
                    "DATASET_SCOPE" NVARCHAR(255) NOT NULL,
                    "DOCUMENT_SOURCE" NVARCHAR(32),
                    "DOCUMENT_RELATIVE_PATH" NVARCHAR(5000),
                    "POSITION" INTEGER,
                    "IS_CLEARED" SMALLINT NOT NULL,
                    "UPDATED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )
            ensure_index(
                self.connection,
                self.document_assignments_table,
                schema=self.schema,
                columns=("ITEM_ID", "DATASET_SCOPE"),
            )

        if not self.connection.table_exists(self.app_settings_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.app_settings_table, self.schema)} (
                    "SETTINGS_ID" NVARCHAR(64) PRIMARY KEY,
                    "USE_GCC_TRACKER_METAL_COMPOSITION" SMALLINT NOT NULL,
                    "UPDATED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )

        if not self.connection.table_exists(self.classification_history_table, schema=self.schema):
            self.connection.execute(
                f"""
                CREATE COLUMN TABLE {_qualified_table(self.classification_history_table, self.schema)} (
                    "HISTORY_ID" NVARCHAR(36) PRIMARY KEY,
                    "ITEM_ID" NVARCHAR(255) NOT NULL,
                    "DATASET_SCOPE" NVARCHAR(255) NOT NULL,
                    "PAYLOAD_JSON" NCLOB NOT NULL,
                    "AGENT_OUTPUTS_JSON" NCLOB,
                    "TIMING_JSON" NCLOB,
                    "STATUS" NVARCHAR(64) NOT NULL,
                    "LAST_CLASSIFIED_AT" NVARCHAR(64) NOT NULL,
                    "RECORDED_AT" NVARCHAR(64) NOT NULL
                )
                """
            )
            ensure_index(
                self.connection,
                self.classification_history_table,
                schema=self.schema,
                columns=("ITEM_ID", "DATASET_SCOPE"),
            )

    def _fetch_rows(self, sql: str, params: Sequence[object] | None = None) -> List[Dict[str, object]]:
        return fetch_rows(self.connection, sql, params)

    def replace_document_assignments(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        document_refs: Sequence[StoredDocumentReference],
    ) -> List[StoredDocumentReference]:
        updated_at = utc_now_iso()
        revision_id = str(uuid.uuid4())
        rows: List[List[object]] = []
        if document_refs:
            for position, document_ref in enumerate(document_refs):
                rows.append(
                    [
                        str(uuid.uuid4()),
                        revision_id,
                        item_id,
                        dataset_scope,
                        document_ref.source,
                        document_ref.relative_path,
                        position,
                        0,
                        updated_at,
                    ]
                )
        else:
            rows.append(
                [
                    str(uuid.uuid4()),
                    revision_id,
                    item_id,
                    dataset_scope,
                    None,
                    None,
                    0,
                    1,
                    updated_at,
                ]
            )

        self.connection.executemany(
            f"""
            INSERT INTO {_qualified_table(self.document_assignments_table, self.schema)} (
                "ENTRY_ID", "REVISION_ID", "ITEM_ID", "DATASET_SCOPE", "DOCUMENT_SOURCE",
                "DOCUMENT_RELATIVE_PATH", "POSITION", "IS_CLEARED", "UPDATED_AT"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return self.get_document_assignments(item_id, dataset_scope=dataset_scope)

    def get_document_assignments(self, item_id: str, *, dataset_scope: str) -> List[StoredDocumentReference]:
        rows = self._fetch_rows(
            f"""
            SELECT "REVISION_ID", "DOCUMENT_SOURCE", "DOCUMENT_RELATIVE_PATH", "POSITION", "IS_CLEARED"
            FROM {_qualified_table(self.document_assignments_table, self.schema)}
            WHERE "ITEM_ID" = ? AND "DATASET_SCOPE" = ?
            ORDER BY "UPDATED_AT" DESC, "REVISION_ID" DESC, "POSITION" ASC, "ENTRY_ID" ASC
            """,
            [item_id, dataset_scope],
        )
        return self._extract_latest_document_refs(rows)

    def get_document_assignment_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], List[StoredDocumentReference]]:
        key_list = list(keys)
        if not key_list:
            return {}

        clauses = " OR ".join('("ITEM_ID" = ? AND "DATASET_SCOPE" = ?)' for _ in key_list)
        params: List[str] = []
        for item_id, dataset_scope in key_list:
            params.extend((item_id, dataset_scope))

        rows = self._fetch_rows(
            f"""
            SELECT "ITEM_ID", "DATASET_SCOPE", "REVISION_ID", "DOCUMENT_SOURCE",
                   "DOCUMENT_RELATIVE_PATH", "POSITION", "IS_CLEARED"
            FROM {_qualified_table(self.document_assignments_table, self.schema)}
            WHERE {clauses}
            ORDER BY "ITEM_ID" ASC, "DATASET_SCOPE" ASC, "UPDATED_AT" DESC, "REVISION_ID" DESC,
                     "POSITION" ASC, "ENTRY_ID" ASC
            """,
            params,
        )

        assignment_map: Dict[tuple[str, str], List[StoredDocumentReference]] = {
            (item_id, dataset_scope): []
            for item_id, dataset_scope in key_list
        }
        active_revisions: Dict[tuple[str, str], str] = {}
        cleared_keys: Set[tuple[str, str]] = set()
        for row in rows:
            key = (str(row["item_id"]), str(row["dataset_scope"]))
            revision_id = str(row["revision_id"])
            if key in cleared_keys:
                continue
            if key not in active_revisions:
                active_revisions[key] = revision_id
                if int(row["is_cleared"] or 0):
                    cleared_keys.add(key)
                    assignment_map[key] = []
                    continue
            if active_revisions[key] != revision_id:
                continue
            assignment = self._row_to_document_ref(row)
            if assignment is not None:
                assignment_map[key].append(assignment)
        return assignment_map

    def get_document_assignment_keys(self) -> Set[Tuple[str, str]]:
        rows = self._fetch_rows(
            f"""
            SELECT "ITEM_ID", "DATASET_SCOPE", "REVISION_ID", "IS_CLEARED"
            FROM {_qualified_table(self.document_assignments_table, self.schema)}
            ORDER BY "ITEM_ID" ASC, "DATASET_SCOPE" ASC, "UPDATED_AT" DESC, "REVISION_ID" DESC, "ENTRY_ID" ASC
            """
        )
        keys: Set[Tuple[str, str]] = set()
        processed: Set[Tuple[str, str]] = set()
        for row in rows:
            key = (str(row["item_id"]), str(row["dataset_scope"]))
            if key in processed:
                continue
            processed.add(key)
            if int(row["is_cleared"] or 0) == 0:
                keys.add(key)
        return keys

    def get_app_settings(self) -> PersistedAppSettings:
        rows = self._fetch_rows(
            f"""
            SELECT "USE_GCC_TRACKER_METAL_COMPOSITION", "UPDATED_AT"
            FROM {_qualified_table(self.app_settings_table, self.schema)}
            WHERE "SETTINGS_ID" = ?
            """,
            ["global"],
        )
        if not rows:
            return PersistedAppSettings()
        return self._row_to_app_settings(rows[0])

    def update_app_settings(
        self,
        *,
        use_gcc_tracker_metal_composition: bool,
    ) -> PersistedAppSettings:
        now = utc_now_iso()
        self.connection.execute(
            f"""
            UPSERT {_qualified_table(self.app_settings_table, self.schema)} (
                "SETTINGS_ID", "USE_GCC_TRACKER_METAL_COMPOSITION", "UPDATED_AT"
            ) VALUES (?, ?, ?) WITH PRIMARY KEY
            """,
            [
                "global",
                1 if use_gcc_tracker_metal_composition else 0,
                now,
            ],
        )
        return PersistedAppSettings(
            use_gcc_tracker_metal_composition=bool(use_gcc_tracker_metal_composition),
            updated_at=now,
        )

    def save_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
        result: MetalCompositionResponse,
        last_classified_at: Optional[str] = None,
    ) -> PersistedClassificationSnapshot:
        classified_at = last_classified_at or utc_now_iso()
        payload_dict = result.model_dump(mode="json")
        payload_json = json.dumps(payload_dict, sort_keys=True)
        agent_outputs_json = (
            json.dumps(payload_dict.get("agent_outputs"), sort_keys=True)
            if payload_dict.get("agent_outputs") is not None
            else None
        )
        timing_json = json.dumps(payload_dict.get("timing") or {}, sort_keys=True)
        self.connection.execute(
            f"""
            INSERT INTO {_qualified_table(self.classification_history_table, self.schema)} (
                "HISTORY_ID", "ITEM_ID", "DATASET_SCOPE", "PAYLOAD_JSON", "AGENT_OUTPUTS_JSON",
                "TIMING_JSON", "STATUS", "LAST_CLASSIFIED_AT", "RECORDED_AT"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(uuid.uuid4()),
                item_id,
                dataset_scope,
                payload_json,
                agent_outputs_json,
                timing_json,
                result.status,
                classified_at,
                utc_now_iso(),
            ],
        )
        snapshot = self.get_classification_snapshot(item_id, dataset_scope=dataset_scope)
        if snapshot is None:
            raise RuntimeError(f"Unable to load classification snapshot for {item_id}")
        return snapshot

    def get_classification_snapshot(
        self,
        item_id: str,
        *,
        dataset_scope: str,
    ) -> Optional[PersistedClassificationSnapshot]:
        rows = self._fetch_rows(
            f"""
            SELECT "ITEM_ID", "DATASET_SCOPE", "PAYLOAD_JSON", "AGENT_OUTPUTS_JSON",
                   "TIMING_JSON", "STATUS", "LAST_CLASSIFIED_AT"
            FROM {_qualified_table(self.classification_history_table, self.schema)}
            WHERE "ITEM_ID" = ? AND "DATASET_SCOPE" = ?
            ORDER BY "LAST_CLASSIFIED_AT" DESC, "RECORDED_AT" DESC, "HISTORY_ID" DESC
            """,
            [item_id, dataset_scope],
        )
        if not rows:
            return None
        return self._row_to_snapshot(rows[0])

    def get_classification_snapshot_map(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> Dict[tuple[str, str], PersistedClassificationSnapshot]:
        key_list = list(keys)
        if not key_list:
            return {}

        clauses = " OR ".join('("ITEM_ID" = ? AND "DATASET_SCOPE" = ?)' for _ in key_list)
        params: List[str] = []
        for item_id, dataset_scope in key_list:
            params.extend((item_id, dataset_scope))

        rows = self._fetch_rows(
            f"""
            SELECT "ITEM_ID", "DATASET_SCOPE", "PAYLOAD_JSON", "AGENT_OUTPUTS_JSON",
                   "TIMING_JSON", "STATUS", "LAST_CLASSIFIED_AT"
            FROM {_qualified_table(self.classification_history_table, self.schema)}
            WHERE {clauses}
            ORDER BY "ITEM_ID" ASC, "DATASET_SCOPE" ASC, "LAST_CLASSIFIED_AT" DESC,
                     "RECORDED_AT" DESC, "HISTORY_ID" DESC
            """,
            params,
        )

        snapshot_map: Dict[tuple[str, str], PersistedClassificationSnapshot] = {}
        for row in rows:
            key = (str(row["item_id"]), str(row["dataset_scope"]))
            if key in snapshot_map:
                continue
            snapshot_map[key] = self._row_to_snapshot(row)
        return snapshot_map

    def get_classification_snapshot_keys(self) -> Set[Tuple[str, str]]:
        rows = self._fetch_rows(
            f"""
            SELECT DISTINCT "ITEM_ID", "DATASET_SCOPE"
            FROM {_qualified_table(self.classification_history_table, self.schema)}
            """
        )
        return {(str(row["item_id"]), str(row["dataset_scope"])) for row in rows}

    def get_classification_stats(self) -> tuple[int, Optional[str]]:
        rows = self._fetch_rows(
            f"""
            SELECT COUNT(*) AS "saved_classification_count", MAX("LAST_CLASSIFIED_AT") AS "latest_classified_at"
            FROM (
                SELECT "ITEM_ID", "DATASET_SCOPE", MAX("LAST_CLASSIFIED_AT") AS "LAST_CLASSIFIED_AT"
                FROM {_qualified_table(self.classification_history_table, self.schema)}
                GROUP BY "ITEM_ID", "DATASET_SCOPE"
            )
            """
        )
        if not rows:
            return 0, None
        row = rows[0]
        return int(row["saved_classification_count"] or 0), (
            None if row["latest_classified_at"] is None else str(row["latest_classified_at"])
        )

    def clear_classification_snapshots(self) -> int:
        count, _latest_classified_at = self.get_classification_stats()
        if count <= 0:
            return 0
        self.connection.execute(
            f'DELETE FROM {_qualified_table(self.classification_history_table, self.schema)}'
        )
        return count

    @staticmethod
    def _extract_latest_document_refs(rows: Sequence[Dict[str, object]]) -> List[StoredDocumentReference]:
        if not rows:
            return []
        revision_id = str(rows[0]["revision_id"])
        if int(rows[0]["is_cleared"] or 0):
            return []
        refs: List[StoredDocumentReference] = []
        for row in rows:
            if str(row["revision_id"]) != revision_id:
                break
            assignment = MetalCompositionUIStateStore._row_to_document_ref(row)
            if assignment is not None:
                refs.append(assignment)
        return refs

    @staticmethod
    def _row_to_app_settings(row: Dict[str, object]) -> PersistedAppSettings:
        return PersistedAppSettings(
            use_gcc_tracker_metal_composition=bool(int(row.get("use_gcc_tracker_metal_composition") or 0)),
            updated_at=(None if row.get("updated_at") is None else str(row["updated_at"])),
        )

    @staticmethod
    def _row_to_document_ref(row: Dict[str, object]) -> Optional[StoredDocumentReference]:
        source = row.get("document_source")
        relative_path = row.get("document_relative_path")
        if source is None or relative_path is None:
            return None
        return StoredDocumentReference(
            source=str(source),
            relative_path=str(relative_path),
        )

    @staticmethod
    def _row_to_snapshot(row: Dict[str, object]) -> PersistedClassificationSnapshot:
        return PersistedClassificationSnapshot(
            item_id=str(row["item_id"]),
            dataset_scope=str(row["dataset_scope"]),
            payload=MetalCompositionResponse.model_validate_json(str(row["payload_json"])),
            status=str(row["status"]),
            last_classified_at=str(row["last_classified_at"]),
            agent_outputs_json=(
                None if row.get("agent_outputs_json") is None else str(row["agent_outputs_json"])
            ),
            timing_json=(None if row.get("timing_json") is None else str(row["timing_json"])),
        )
