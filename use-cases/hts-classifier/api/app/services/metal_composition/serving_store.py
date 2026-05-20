"""Serving-store abstractions for PDF-grounded metal composition inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.models.metal_composition import MetalCompositionCandidate
from app.utils.hana import HANAConnection

from .config import MetalCompositionSettings, get_settings
from .timing import finish_timing, utc_now_iso
from .workbook_format import pandas_engine_for_gcc_tracker_workbook

logger = logging.getLogger(__name__)

PRODUCT_CODE_COLUMN = "Product code"
PN_COLUMN = "PN Revised/ Standardized"
PART_DESCRIPTION_COLUMN = "Part description"
NEW_PART_DESCRIPTION_COLUMN = "New Part Description"
SITE_COLUMN = "Site"
BUSINESS_SEGMENT_COLUMN = "Business Segment"
PRIORITY_COLUMN = "Priority"
PRIORITY_DETAIL_COLUMN = "Priority.1"
MATERIAL_CONTENT_METHOD_COLUMN = "Material Content Method"
MATERIAL_IDENTIFIED_COLUMN = "MaterialIdentified"
TOTAL_WEIGHT_COLUMN = "Total Weight (Gram)"
DATE_STARTED_COLUMN = "Date Started"
DATE_COMPLETED_COLUMN = "Date Completed"
STATUS_COLUMN = "Material Content Status"
SAFE_SOURCE_COLUMNS = [
    PRIORITY_COLUMN,
    BUSINESS_SEGMENT_COLUMN,
    SITE_COLUMN,
    PRODUCT_CODE_COLUMN,
    PN_COLUMN,
    PART_DESCRIPTION_COLUMN,
    NEW_PART_DESCRIPTION_COLUMN,
    PRIORITY_DETAIL_COLUMN,
    MATERIAL_CONTENT_METHOD_COLUMN,
    MATERIAL_IDENTIFIED_COLUMN,
    TOTAL_WEIGHT_COLUMN,
    DATE_STARTED_COLUMN,
    DATE_COMPLETED_COLUMN,
]
SNAPSHOT_SOURCE_FILENAME = "source_lookup.parquet"
SNAPSHOT_PREPARED_FILENAME = "prepared_features.parquet"
SNAPSHOT_METADATA_FILENAME = "metadata.json"
SNAPSHOT_SOURCE_PICKLE_FILENAME = "source_lookup.pkl"
SNAPSHOT_PREPARED_PICKLE_FILENAME = "prepared_features.pkl"
SOURCE_PREFIX = "source__"
PREPARED_PREFIX = "prepared__"
TOP_LEVEL_GCC_GRAM_COLUMNS = {
    "steel": "Steel_grams",
    "aluminum": "Aluminum_grams",
    "copper": "Copper_grams",
    "cast_iron": "Cast_Iron_grams",
}
TOP_LEVEL_GCC_WORKBOOK_GRAM_COLUMNS = {
    "Aluminum_grams": "Aluminum  - Gram",
    "Copper_grams": "Copper - Gram",
    "Cast_Iron_grams": "Cast Iron - Gram",
}
STEEL_SUBTYPE_GCC_GRAM_COLUMNS = {
    "electrical_steel": "Electrical_Steel_grams",
    "cold_rolled_coil_steel": "Cold_Rolled_Coil_Steel_grams",
    "hot_rolled_coil_steel": "Hot_Rolled_Coil_Steel_grams",
    "stainless_steel_304": "Stainless_Steel_304_grams",
    "stainless_steel_316": "Stainless_Steel_316_grams",
    "stainless_steel_bar": "Stainless_Steel_Bar_grams",
    "duplex_steel": "Duplex_Steel_grams",
    "cast_steel": "Cast_Steel_grams",
}
STEEL_SUBTYPE_GCC_WORKBOOK_GRAM_COLUMNS = {
    "Electrical_Steel_grams": "Electrical Steel - Grams",
    "Cold_Rolled_Coil_Steel_grams": "Cold-Rolled Coil Steel -Grams",
    "Hot_Rolled_Coil_Steel_grams": "Hot-Rolled Coil Steel - Grams",
    "Stainless_Steel_304_grams": "Stainless Steel 304 - Grams",
    "Stainless_Steel_316_grams": "Stainless Steel 316 - Grams",
    "Stainless_Steel_Bar_grams": "Stainless Steel Bar - Grams",
    "Duplex_Steel_grams": "Duplex Steel - Grams",
    "Cast_Steel_grams": "Cast Steel - Gram",
}


def _is_missing_parquet_engine(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("pyarrow", "fastparquet", "parquet"))


def _clean_text_value(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_numeric_scalar(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_numeric_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    cleaned = series.map(
        lambda value: value.strip().replace(",", ".") if isinstance(value, str) else value
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_lookup_value(value: object) -> str:
    text = _clean_text_value(value)
    return text.lower() if text else ""


def _timestamp_to_iso(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    return ts.date().isoformat()


def _parse_lookup_dates(series: pd.Series) -> pd.Series:
    cleaned = series.map(_clean_text_value)
    numeric = _parse_numeric_series(series)
    excel = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    direct = pd.to_datetime(cleaned.where(numeric.isna()), errors="coerce", dayfirst=True)
    return direct.where(direct.notna(), excel)


def _filter_complete_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    if STATUS_COLUMN not in raw_df.columns:
        return raw_df.copy()
    status = raw_df[STATUS_COLUMN].map(lambda value: (_clean_text_value(value) or "").lower())
    return raw_df.loc[status == "complete"].copy()


def _build_source_frame_from_complete_df(complete_df: pd.DataFrame) -> pd.DataFrame:
    source = pd.DataFrame(index=complete_df.index)
    source["source_row_id"] = complete_df.index.astype(int)
    source["normalized_product_code"] = complete_df[PRODUCT_CODE_COLUMN].map(normalize_lookup_value)

    parsed_started = _parse_lookup_dates(
        complete_df.get(DATE_STARTED_COLUMN, pd.Series(index=complete_df.index, dtype=object))
    )
    parsed_completed = _parse_lookup_dates(
        complete_df.get(DATE_COMPLETED_COLUMN, pd.Series(index=complete_df.index, dtype=object))
    )
    parsed_weight = _parse_numeric_series(
        complete_df.get(TOTAL_WEIGHT_COLUMN, pd.Series(index=complete_df.index, dtype=object))
    )

    for column in SAFE_SOURCE_COLUMNS:
        if column == DATE_STARTED_COLUMN:
            source[column] = parsed_started.map(_timestamp_to_iso)
        elif column == DATE_COMPLETED_COLUMN:
            source[column] = parsed_completed.map(_timestamp_to_iso)
        elif column == TOTAL_WEIGHT_COLUMN:
            source[column] = parsed_weight.astype(float)
        else:
            raw_series = complete_df.get(column, pd.Series(index=complete_df.index, dtype=object))
            source[column] = raw_series.map(_clean_text_value)

    return source.reset_index(drop=True)


def _default_prepared_df(source_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"source_row_id": source_df["source_row_id"].astype(int)})


def _parse_numeric_series_or_default(complete_df: pd.DataFrame, column: str) -> pd.Series:
    if column not in complete_df.columns:
        return pd.Series(0.0, index=complete_df.index, dtype=float)
    return _parse_numeric_series(complete_df[column]).fillna(0.0).astype(float)


def _build_prepared_frame_from_complete_df(complete_df: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(index=complete_df.index)
    prepared["source_row_id"] = complete_df.index.astype(int)

    steel_total = pd.Series(0.0, index=complete_df.index, dtype=float)
    steel_subtype_values: Dict[str, pd.Series] = {}
    for prepared_column, workbook_column in STEEL_SUBTYPE_GCC_WORKBOOK_GRAM_COLUMNS.items():
        grams = _parse_numeric_series_or_default(complete_df, workbook_column)
        steel_subtype_values[prepared_column] = grams
        steel_total = steel_total.add(grams, fill_value=0.0)

    prepared["Steel_grams"] = steel_total.astype(float)
    for prepared_column, workbook_column in TOP_LEVEL_GCC_WORKBOOK_GRAM_COLUMNS.items():
        prepared[prepared_column] = _parse_numeric_series_or_default(complete_df, workbook_column)
    for prepared_column, grams in steel_subtype_values.items():
        prepared[prepared_column] = grams.astype(float)

    return prepared.reset_index(drop=True)


def build_serving_frames_from_raw_df(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    complete_df = _filter_complete_rows(raw_df)
    if complete_df.empty:
        raise ValueError("No completed GCC rows found in workbook")
    source_df = _build_source_frame_from_complete_df(complete_df)
    prepared_df = _build_prepared_frame_from_complete_df(complete_df)
    return source_df, prepared_df


def _coerce_prepared_types(prepared_df: pd.DataFrame) -> pd.DataFrame:
    frame = prepared_df.reset_index(drop=True).copy()
    if "source_row_id" not in frame.columns:
        return pd.DataFrame(columns=["source_row_id"])
    frame["source_row_id"] = pd.to_numeric(frame["source_row_id"], errors="raise").astype(int)
    if "row_id" in frame.columns:
        frame["row_id"] = pd.to_numeric(frame["row_id"], errors="coerce").fillna(-1).astype(int)
    return frame


def build_serving_table_frame(source_df: pd.DataFrame, prepared_df: pd.DataFrame) -> pd.DataFrame:
    source_indexed = source_df.set_index("source_row_id", drop=False).sort_index()
    prepared_indexed = _coerce_prepared_types(prepared_df).set_index("source_row_id", drop=False).sort_index()
    merged = pd.DataFrame(index=source_indexed.index)
    merged["source_row_id"] = source_indexed["source_row_id"].astype(int)
    merged["normalized_product_code"] = source_indexed["normalized_product_code"]

    for column in source_indexed.columns:
        if column in {"source_row_id", "normalized_product_code"}:
            continue
        merged[f"{SOURCE_PREFIX}{column}"] = source_indexed[column]

    for column in prepared_indexed.columns:
        if column == "source_row_id":
            continue
        merged[f"{PREPARED_PREFIX}{column}"] = prepared_indexed[column]

    return merged.reset_index(drop=True)


def split_serving_table_frame(serving_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_columns = [
        "source_row_id",
        "normalized_product_code",
        *[column for column in serving_df.columns if column.startswith(SOURCE_PREFIX)],
    ]
    prepared_columns = [
        "source_row_id",
        *[column for column in serving_df.columns if column.startswith(PREPARED_PREFIX)],
    ]

    source_df = serving_df[source_columns].copy()
    source_df.columns = [
        column.replace(SOURCE_PREFIX, "", 1) if column.startswith(SOURCE_PREFIX) else column
        for column in source_df.columns
    ]
    source_df["source_row_id"] = pd.to_numeric(source_df["source_row_id"], errors="raise").astype(int)
    if TOTAL_WEIGHT_COLUMN in source_df.columns:
        source_df[TOTAL_WEIGHT_COLUMN] = pd.to_numeric(source_df[TOTAL_WEIGHT_COLUMN], errors="coerce")

    prepared_df = serving_df[prepared_columns].copy()
    prepared_df.columns = [
        column.replace(PREPARED_PREFIX, "", 1) if column.startswith(PREPARED_PREFIX) else column
        for column in prepared_df.columns
    ]
    prepared_df = _coerce_prepared_types(prepared_df)
    return source_df.reset_index(drop=True), prepared_df.reset_index(drop=True)


@dataclass(frozen=True)
class ResolvedMaterialRecord:
    product_code: str
    source_row_id: int
    source_row: Dict[str, Any]
    summary: MetalCompositionCandidate


@dataclass(frozen=True)
class GCCMetalProfile:
    source_row_id: int
    top_level_grams: Dict[str, float]
    steel_subtype_grams: Dict[str, float]


class ServingStore:
    """In-memory serving store used on the request path."""

    def __init__(
        self,
        source_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._source_df = source_df.reset_index(drop=True).copy()
        self._prepared_df = _coerce_prepared_types(prepared_df)
        self._metadata = dict(metadata or {})
        self._source_by_row_id = self._source_df.set_index("source_row_id", drop=False).sort_index()
        self._prepared_by_row_id = self._prepared_df.set_index("source_row_id", drop=False).sort_index()
        grouped = (
            self._source_df.groupby("normalized_product_code", sort=False)["source_row_id"]
            .apply(list)
            .to_dict()
        )
        self._lookup_by_product_code = {
            key: [int(value) for value in values]
            for key, values in grouped.items()
            if key
        }

    @property
    def source_df(self) -> pd.DataFrame:
        return self._source_df

    @property
    def prepared_df(self) -> pd.DataFrame:
        return self._prepared_df

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def get_gcc_metal_profile(self, source_row_id: int) -> GCCMetalProfile:
        if int(source_row_id) not in self._prepared_by_row_id.index:
            raise KeyError(f"prepared GCC profile for source_row_id {source_row_id} not found")
        row = self._prepared_by_row_id.loc[int(source_row_id)]
        return GCCMetalProfile(
            source_row_id=int(source_row_id),
            top_level_grams={
                metal: float(_parse_numeric_scalar(row.get(column)) or 0.0)
                for metal, column in TOP_LEVEL_GCC_GRAM_COLUMNS.items()
            },
            steel_subtype_grams={
                subtype: float(_parse_numeric_scalar(row.get(column)) or 0.0)
                for subtype, column in STEEL_SUBTYPE_GCC_GRAM_COLUMNS.items()
            },
        )

    def lookup_candidates(self, product_code: str) -> List[MetalCompositionCandidate]:
        normalized = normalize_lookup_value(product_code)
        row_ids = self._lookup_by_product_code.get(normalized, [])
        return [self._build_candidate(int(row_id)) for row_id in row_ids]

    def get_resolved_record(
        self,
        *,
        product_code: str,
        source_row_id: int,
    ) -> ResolvedMaterialRecord:
        if int(source_row_id) not in self._source_by_row_id.index:
            raise KeyError(f"source_row_id {source_row_id} not found")
        return ResolvedMaterialRecord(
            product_code=_clean_text_value(product_code) or product_code,
            source_row_id=int(source_row_id),
            source_row=self._build_safe_source_context(int(source_row_id)),
            summary=self._build_candidate(int(source_row_id)),
        )

    def resolve(
        self,
        product_code: str,
        selected_source_row_id: Optional[int] = None,
    ) -> Tuple[Optional[ResolvedMaterialRecord], List[MetalCompositionCandidate]]:
        candidates = self.lookup_candidates(product_code)
        if not candidates:
            return None, []
        if len(candidates) == 1:
            row_id = candidates[0].source_row_id
            return self.get_resolved_record(product_code=product_code, source_row_id=row_id), candidates
        if selected_source_row_id is None:
            return None, candidates
        if all(candidate.source_row_id != int(selected_source_row_id) for candidate in candidates):
            return None, candidates
        return (
            self.get_resolved_record(
                product_code=product_code,
                source_row_id=int(selected_source_row_id),
            ),
            candidates,
        )

    def _build_candidate(self, source_row_id: int) -> MetalCompositionCandidate:
        row = self._source_by_row_id.loc[int(source_row_id)]
        return MetalCompositionCandidate(
            source_row_id=int(source_row_id),
            source_kind="gcc",
            pn_revised_standardized=_clean_text_value(row.get(PN_COLUMN)),
            part_description=_clean_text_value(row.get(PART_DESCRIPTION_COLUMN)),
            new_part_description=_clean_text_value(row.get(NEW_PART_DESCRIPTION_COLUMN)),
            priority_detail=_clean_text_value(row.get(PRIORITY_DETAIL_COLUMN)),
            site=_clean_text_value(row.get(SITE_COLUMN)),
            business_segment=_clean_text_value(row.get(BUSINESS_SEGMENT_COLUMN)),
            total_weight_gram=_parse_numeric_scalar(row.get(TOTAL_WEIGHT_COLUMN)),
            date_started=_clean_text_value(row.get(DATE_STARTED_COLUMN)),
            date_completed=_clean_text_value(row.get(DATE_COMPLETED_COLUMN)),
        )

    def _build_safe_source_context(self, source_row_id: int) -> Dict[str, Any]:
        row = self._source_by_row_id.loc[int(source_row_id)]
        safe_context: Dict[str, Any] = {}
        for column in SAFE_SOURCE_COLUMNS:
            if column not in row.index:
                continue
            value = row.get(column)
            if column == TOTAL_WEIGHT_COLUMN:
                safe_context[column] = _parse_numeric_scalar(value)
            else:
                safe_context[column] = _clean_text_value(value)
        return safe_context

class WorkbookStore(ServingStore):
    """Build a serving store directly from the workbook."""

    @classmethod
    def from_settings(cls, settings: MetalCompositionSettings) -> "WorkbookStore":
        """Build a workbook-backed serving store from the configured tracker file.

        Args:
            settings: Metal composition settings containing the workbook path
                and sheet name to load.

        Returns:
            A workbook-backed serving store built from the configured sheet.
        """

        raw_df = pd.read_excel(
            settings.workbook_path,
            sheet_name=settings.sheet_name,
            engine=pandas_engine_for_gcc_tracker_workbook(settings.workbook_path),
        )
        source_df, prepared_df = build_serving_frames_from_raw_df(raw_df)
        metadata = {
            "source": "workbook",
            "workbook_path": str(settings.workbook_path),
            "sheet_name": settings.sheet_name,
            "row_count": int(len(source_df)),
        }
        return cls(source_df=source_df, prepared_df=prepared_df, metadata=metadata)


class LocalSnapshotStore:
    """Persist or load the serving store from local snapshots."""

    def __init__(self, settings: MetalCompositionSettings) -> None:
        self.settings = settings
        self.cache_dir = settings.cache_dir
        self.source_path = self.cache_dir / SNAPSHOT_SOURCE_FILENAME
        self.prepared_path = self.cache_dir / SNAPSHOT_PREPARED_FILENAME
        self.source_pickle_path = self.cache_dir / SNAPSHOT_SOURCE_PICKLE_FILENAME
        self.prepared_pickle_path = self.cache_dir / SNAPSHOT_PREPARED_PICKLE_FILENAME
        self.metadata_path = self.cache_dir / SNAPSHOT_METADATA_FILENAME

    def load(self) -> ServingStore:
        parquet_source_exists = self.source_path.exists()
        parquet_prepared_exists = self.prepared_path.exists()
        pickle_source_exists = self.source_pickle_path.exists()
        pickle_prepared_exists = self.prepared_pickle_path.exists()

        if not parquet_source_exists and not pickle_source_exists:
            raise FileNotFoundError("Metal composition snapshot files are missing")

        if parquet_source_exists:
            source_df = pd.read_parquet(self.source_path)
            if parquet_prepared_exists:
                prepared_df = pd.read_parquet(self.prepared_path)
                snapshot_format = "parquet"
            else:
                prepared_df = _default_prepared_df(source_df)
                snapshot_format = "parquet"
        else:
            source_df = pd.read_pickle(self.source_pickle_path)
            if pickle_prepared_exists:
                prepared_df = pd.read_pickle(self.prepared_pickle_path)
                snapshot_format = "pickle"
            else:
                prepared_df = _default_prepared_df(source_df)
                snapshot_format = "pickle"

        metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["source"] = "local_snapshot"
        metadata["snapshot_format"] = snapshot_format
        return ServingStore(source_df=source_df, prepared_df=prepared_df, metadata=metadata)

    def save(self, store: ServingStore, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_format = "parquet"
        try:
            store.source_df.to_parquet(self.source_path, index=False)
            store.prepared_df.to_parquet(self.prepared_path, index=False)
            if self.source_pickle_path.exists():
                self.source_pickle_path.unlink()
            if self.prepared_pickle_path.exists():
                self.prepared_pickle_path.unlink()
        except Exception as exc:  # noqa: BLE001 - fallback only for missing parquet extras
            if not _is_missing_parquet_engine(exc):
                raise
            snapshot_format = "pickle"
            logger.warning(
                "Parquet snapshot support is unavailable; falling back to pickle snapshots: %s",
                exc,
            )
            store.source_df.to_pickle(self.source_pickle_path)
            store.prepared_df.to_pickle(self.prepared_pickle_path)
            if self.source_path.exists():
                self.source_path.unlink()
            if self.prepared_path.exists():
                self.prepared_path.unlink()
        snapshot_metadata = {
            **store.metadata,
            **dict(metadata or {}),
            "snapshot_format": snapshot_format,
            "saved_at": utc_now_iso(),
        }
        self.metadata_path.write_text(json.dumps(snapshot_metadata, indent=2), encoding="utf-8")


class HanaServingStore:
    """Load or refresh the denormalized serving table from SAP HANA."""

    def __init__(self, settings: MetalCompositionSettings) -> None:
        self.settings = settings
        self.connection = HANAConnection()

    def load(self) -> ServingStore:
        try:
            frame = self.connection.fetch_dataframe(
                self.settings.hana_table,
                schema=self.settings.hana_schema or None,
            )
            if frame.empty:
                raise RuntimeError("HANA serving table is empty")
            source_df, prepared_df = split_serving_table_frame(frame)
            metadata = {
                "source": "hana",
                "hana_schema": self.settings.hana_schema,
                "hana_table": self.settings.hana_table,
                "row_count": int(len(source_df)),
            }
            return ServingStore(source_df=source_df, prepared_df=prepared_df, metadata=metadata)
        finally:
            self.connection.disconnect()

    def refresh_from_store(self, store: ServingStore) -> Dict[str, Any]:
        try:
            serving_df = build_serving_table_frame(store.source_df, store.prepared_df)
            return self.connection.refresh_serving_table(
                frame=serving_df,
                table=self.settings.hana_table,
                schema=self.settings.hana_schema or None,
                primary_key="source_row_id",
                index_columns=["normalized_product_code"],
            )
        finally:
            self.connection.disconnect()


@dataclass(frozen=True)
class ServingStoreLoadResult:
    store: ServingStore
    startup_timing: Dict[str, Any]
    loaded_from: str


def _load_from_hana(settings: MetalCompositionSettings) -> ServingStore:
    return HanaServingStore(settings).load()


def _load_from_snapshot(settings: MetalCompositionSettings) -> ServingStore:
    return LocalSnapshotStore(settings).load()


def load_serving_store(settings: Optional[MetalCompositionSettings] = None) -> ServingStoreLoadResult:
    config = settings or get_settings()
    started_perf = perf_counter()
    started_at = utc_now_iso()
    substeps: Dict[str, Any] = {}
    snapshot = LocalSnapshotStore(config)

    if config.data_source == "workbook":
        logger.warning(
            "Loading metal composition serving data from workbook compatibility mode at %s. "
            "HANA remains the preferred runtime source.",
            config.workbook_path,
        )
        workbook_started_perf = perf_counter()
        workbook_started_at = utc_now_iso()
        store = WorkbookStore.from_settings(config)
        substeps["load_workbook"] = finish_timing(
            workbook_started_perf,
            workbook_started_at,
            details={"workbook_path": str(config.workbook_path)},
        )
        startup_timing = finish_timing(
            started_perf,
            started_at,
            details={"cache_source": "workbook"},
            substeps=substeps,
        )
        return ServingStoreLoadResult(store=store, startup_timing=startup_timing, loaded_from="workbook")

    if config.data_source == "snapshot":
        logger.warning(
            "Loading metal composition serving data from local snapshot compatibility mode in %s. "
            "HANA remains the preferred runtime source.",
            config.cache_dir,
        )
        snapshot_started_perf = perf_counter()
        snapshot_started_at = utc_now_iso()
        store = snapshot.load()
        substeps["load_snapshot"] = finish_timing(snapshot_started_perf, snapshot_started_at)
        startup_timing = finish_timing(
            started_perf,
            started_at,
            details={"cache_source": "local_snapshot"},
            substeps=substeps,
        )
        return ServingStoreLoadResult(
            store=store,
            startup_timing=startup_timing,
            loaded_from="local_snapshot",
        )

    hana_error = None
    hana_started_perf = perf_counter()
    hana_started_at = utc_now_iso()
    try:
        store = _load_from_hana(config)
        substeps["load_hana"] = finish_timing(
            hana_started_perf,
            hana_started_at,
            details={"table": config.hana_table, "schema": config.hana_schema or None},
        )
        snapshot_saved = True
        snapshot_error = None
        snapshot_started_perf = perf_counter()
        snapshot_started_at = utc_now_iso()
        try:
            snapshot.save(
                store,
                metadata={
                    "cache_source": "hana",
                    "hana_table": config.hana_table,
                    "hana_schema": config.hana_schema,
                },
            )
            substeps["save_snapshot"] = finish_timing(snapshot_started_perf, snapshot_started_at)
        except Exception as exc:  # noqa: BLE001 - cache persistence must not fail startup
            snapshot_saved = False
            snapshot_error = str(exc)
            logger.warning("Unable to persist metal composition snapshot: %s", snapshot_error)
            substeps["save_snapshot"] = finish_timing(
                snapshot_started_perf,
                snapshot_started_at,
                status="failed",
                details={"error": snapshot_error},
            )
        startup_timing = finish_timing(
            started_perf,
            started_at,
            details={
                "cache_source": "hana",
                "snapshot_saved": snapshot_saved,
                **({"snapshot_error": snapshot_error} if snapshot_error else {}),
            },
            substeps=substeps,
        )
        return ServingStoreLoadResult(store=store, startup_timing=startup_timing, loaded_from="hana")
    except Exception as exc:  # noqa: BLE001 - fallback path should catch everything
        hana_error = str(exc)
        logger.warning(
            "HANA serving-store load failed, attempting transitional local snapshot fallback: %s",
            hana_error,
        )
        substeps["load_hana"] = finish_timing(
            hana_started_perf,
            hana_started_at,
            status="failed",
            details={"error": hana_error},
        )

    snapshot_started_perf = perf_counter()
    snapshot_started_at = utc_now_iso()
    try:
        store = _load_from_snapshot(config)
        substeps["load_snapshot"] = finish_timing(
            snapshot_started_perf,
            snapshot_started_at,
            details={"fallback_reason": hana_error},
        )
        startup_timing = finish_timing(
            started_perf,
            started_at,
            details={"cache_source": "local_snapshot", "fallback_reason": hana_error},
            substeps=substeps,
        )
        return ServingStoreLoadResult(
            store=store,
            startup_timing=startup_timing,
            loaded_from="local_snapshot",
        )
    except Exception as snapshot_exc:  # noqa: BLE001 - expose both failures together
        substeps["load_snapshot"] = finish_timing(
            snapshot_started_perf,
            snapshot_started_at,
            status="failed",
            details={"error": str(snapshot_exc)},
        )
        raise RuntimeError(
            "Unable to load metal composition serving store from HANA or local snapshot. "
            f"hana_error={hana_error}; snapshot_error={snapshot_exc}"
        ) from snapshot_exc


@lru_cache(maxsize=1)
def get_workbook_store() -> WorkbookStore:
    """Return the workbook-backed store used by refresh jobs or explicit local mode."""

    logger.warning(
        "Using workbook-backed serving store compatibility path. This should be reserved for tests or explicit admin workflows."
    )
    return WorkbookStore.from_settings(get_settings())
