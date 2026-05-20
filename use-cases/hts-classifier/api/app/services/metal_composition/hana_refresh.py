"""Refresh helpers for the GCC Tracker HANA serving table."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .config import MetalCompositionSettings, get_settings
from .serving_store import HanaServingStore, WorkbookStore


class GCCTrackerWorkbookLoadError(ValueError):
    """Raised when the uploaded GCC Tracker workbook cannot be loaded."""


class GCCTrackerHanaRefreshError(RuntimeError):
    """Raised when the HANA serving table refresh fails."""


def build_hana_refresh_settings(
    workbook_path: Path,
    *,
    settings: MetalCompositionSettings | None = None,
) -> MetalCompositionSettings:
    """Build HANA refresh settings for a specific GCC Tracker workbook."""

    base_settings = settings or get_settings()
    resolved_workbook_path = workbook_path.expanduser().resolve()
    if not resolved_workbook_path.is_file():
        raise FileNotFoundError(f"GCC Tracker workbook not found: {resolved_workbook_path}")
    return replace(
        base_settings,
        workbook_path=resolved_workbook_path,
        data_source="hana",
    )


def refresh_metal_composition_hana(
    workbook_path: Path,
    *,
    settings: MetalCompositionSettings | None = None,
) -> Dict[str, Any]:
    """Refresh the configured HANA serving table from a GCC Tracker workbook."""

    refresh_settings = build_hana_refresh_settings(workbook_path, settings=settings)
    try:
        workbook_store = WorkbookStore.from_settings(refresh_settings)
    except Exception as exc:  # noqa: BLE001 - normalize workbook/parser failures for API callers
        raise GCCTrackerWorkbookLoadError(f"Failed to load GCC Tracker workbook: {exc}") from exc

    try:
        refresh_result = HanaServingStore(refresh_settings).refresh_from_store(workbook_store)
    except Exception as exc:  # noqa: BLE001 - normalize HANA driver failures for API callers
        raise GCCTrackerHanaRefreshError(f"Failed to refresh HANA serving table: {exc}") from exc

    return {
        "status": "completed",
        "workbook_path": str(refresh_settings.workbook_path),
        "sheet_name": refresh_settings.sheet_name,
        "hana_schema": refresh_settings.hana_schema,
        "hana_table": refresh_settings.hana_table,
        "source_row_count": int(len(workbook_store.source_df)),
        "prepared_row_count": int(len(workbook_store.prepared_df)),
        "refresh_result": dict(refresh_result),
    }
