from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.metal_composition import hana_refresh as refresh_module
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.hana_refresh import (
    GCCTrackerWorkbookLoadError,
    refresh_metal_composition_hana,
)


def _settings(tmp_path: Path) -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=tmp_path / "configured.xlsb",
        api_env_path=tmp_path / ".env",
        data_source="workbook",
        hana_schema="CONFIG_SCHEMA",
        hana_table="CONFIG_TABLE",
        sheet_name="Material Master",
    )


def test_refresh_uses_uploaded_workbook_and_configured_hana_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "uploaded.xlsb"
    workbook_path.write_bytes(b"uploaded workbook")
    settings = _settings(tmp_path)
    captured = {}

    class FakeWorkbookStore:
        source_df = pd.DataFrame({"source_row_id": [1, 2]})
        prepared_df = pd.DataFrame({"source_row_id": [1, 2]})

        @classmethod
        def from_settings(cls, refresh_settings: MetalCompositionSettings) -> "FakeWorkbookStore":
            captured["workbook_settings"] = refresh_settings
            return cls()

    class FakeHanaServingStore:
        def __init__(self, refresh_settings: MetalCompositionSettings) -> None:
            captured["hana_settings"] = refresh_settings

        def refresh_from_store(self, store: FakeWorkbookStore) -> dict[str, int]:
            captured["store"] = store
            return {"rows_refreshed": len(store.source_df)}

    monkeypatch.setattr(refresh_module, "WorkbookStore", FakeWorkbookStore)
    monkeypatch.setattr(refresh_module, "HanaServingStore", FakeHanaServingStore)

    result = refresh_metal_composition_hana(workbook_path, settings=settings)

    workbook_settings = captured["workbook_settings"]
    hana_settings = captured["hana_settings"]
    assert workbook_settings.workbook_path == workbook_path.resolve()
    assert workbook_settings.data_source == "hana"
    assert workbook_settings.hana_schema == "CONFIG_SCHEMA"
    assert workbook_settings.hana_table == "CONFIG_TABLE"
    assert hana_settings.hana_schema == "CONFIG_SCHEMA"
    assert hana_settings.hana_table == "CONFIG_TABLE"
    assert result["source_row_count"] == 2
    assert result["prepared_row_count"] == 2
    assert result["refresh_result"] == {"rows_refreshed": 2}


def test_refresh_wraps_workbook_load_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "uploaded.xlsb"
    workbook_path.write_bytes(b"invalid workbook")

    class FailingWorkbookStore:
        @classmethod
        def from_settings(cls, refresh_settings: MetalCompositionSettings) -> "FailingWorkbookStore":
            raise ValueError("No completed GCC rows found in workbook")

    monkeypatch.setattr(refresh_module, "WorkbookStore", FailingWorkbookStore)

    with pytest.raises(GCCTrackerWorkbookLoadError, match="No completed GCC rows"):
        refresh_metal_composition_hana(workbook_path, settings=_settings(tmp_path))
