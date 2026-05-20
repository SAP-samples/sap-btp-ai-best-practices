from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import metal_composition_admin as admin_module
from app.routers.metal_composition_admin import router as admin_router
from app.models.metal_composition import ClassificationResetResponse
from app.services.metal_composition.config import MetalCompositionSettings
from app.services.metal_composition.hana_refresh import (
    GCCTrackerHanaRefreshError,
    GCCTrackerWorkbookLoadError,
)


def _settings(tmp_path: Path) -> MetalCompositionSettings:
    return MetalCompositionSettings(
        workbook_path=tmp_path / "configured.xlsb",
        api_env_path=tmp_path / ".env",
        cache_dir=tmp_path / "cache",
        hana_schema="APP_SCHEMA",
        hana_table="METAL_COMPOSITION_SERVING",
        sheet_name="Material Master",
    )


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin_router, prefix="/api/metal-composition/admin")
    return TestClient(app)


def _headers() -> dict[str, str]:
    return {"X-API-Key": "test-api-key"}


def test_refresh_hana_upload_saves_workbook_refreshes_and_invalidates_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    refresh_calls = {}
    cache_invalidations = []
    reset_calls = []

    class FakeMetalCompositionService:
        """Test double that records GCC refresh cleanup calls."""

        def reset_classifications(self) -> ClassificationResetResponse:
            """Clear saved classification state after a tracker refresh."""

            reset_calls.append(True)
            return ClassificationResetResponse(cleared_classification_count=7, cancelled_job_count=2)

    def fake_refresh(workbook_path: Path, *, settings: MetalCompositionSettings) -> dict[str, object]:
        refresh_calls["workbook_path"] = workbook_path
        refresh_calls["settings"] = settings
        return {
            "status": "completed",
            "workbook_path": str(workbook_path),
            "sheet_name": settings.sheet_name,
            "hana_schema": settings.hana_schema,
            "hana_table": settings.hana_table,
            "source_row_count": 3,
            "prepared_row_count": 3,
            "refresh_result": {"rows_refreshed": 3},
        }

    monkeypatch.setattr(admin_module, "get_settings", lambda: settings)
    monkeypatch.setattr(admin_module, "refresh_metal_composition_hana", fake_refresh)
    monkeypatch.setattr(admin_module, "get_metal_composition_service", lambda: FakeMetalCompositionService())
    monkeypatch.setattr(admin_module, "invalidate_metal_composition_service_cache", lambda: cache_invalidations.append(True))

    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        data={"source_path": "/client/data/new_data/GCC Tracker.xlsb"},
        files={"file": ("GCC Tracker.xlsb", b"workbook bytes", "application/vnd.ms-excel.sheet.binary.macroEnabled.12")},
    )

    assert response.status_code == 200
    payload = response.json()
    stored_path = refresh_calls["workbook_path"]
    assert stored_path.is_file()
    assert stored_path.parent == settings.cache_dir / "gcc_tracker_uploads"
    assert refresh_calls["settings"] is settings
    assert reset_calls == [True]
    assert cache_invalidations == [True]
    assert payload["status"] == "completed"
    assert payload["uploaded_filename"] == "GCC Tracker.xlsb"
    assert payload["uploaded_size_bytes"] == len(b"workbook bytes")
    assert payload["source_path"] == "/client/data/new_data/GCC Tracker.xlsb"
    assert payload["stored_workbook_path"] == str(stored_path)
    assert payload["hana_schema"] == "APP_SCHEMA"
    assert payload["hana_table"] == "METAL_COMPOSITION_SERVING"
    assert payload["source_row_count"] == 3
    assert payload["refresh_result"] == {"rows_refreshed": 3}
    assert payload["cleared_classification_count"] == 7
    assert payload["cancelled_job_count"] == 2
    assert payload["service_cache_invalidated"] is True


def test_refresh_hana_accepts_xlsx_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify the admin refresh endpoint accepts standard XLSX workbooks."""

    settings = _settings(tmp_path)
    refresh_calls = {}
    cache_invalidations = []

    class FakeMetalCompositionService:
        """Test double that records classification cleanup for XLSX uploads."""

        def reset_classifications(self) -> ClassificationResetResponse:
            """Return an empty cleanup result after a tracker refresh."""

            return ClassificationResetResponse(cleared_classification_count=0, cancelled_job_count=0)

    def fake_refresh(workbook_path: Path, *, settings: MetalCompositionSettings) -> dict[str, object]:
        """Record the uploaded workbook path and return a successful refresh payload."""

        refresh_calls["workbook_path"] = workbook_path
        refresh_calls["settings"] = settings
        return {
            "status": "completed",
            "workbook_path": str(workbook_path),
            "sheet_name": settings.sheet_name,
            "hana_schema": settings.hana_schema,
            "hana_table": settings.hana_table,
            "source_row_count": 2,
            "prepared_row_count": 2,
            "refresh_result": {"rows_refreshed": 2},
        }

    monkeypatch.setattr(admin_module, "get_settings", lambda: settings)
    monkeypatch.setattr(admin_module, "refresh_metal_composition_hana", fake_refresh)
    monkeypatch.setattr(admin_module, "get_metal_composition_service", lambda: FakeMetalCompositionService())
    monkeypatch.setattr(admin_module, "invalidate_metal_composition_service_cache", lambda: cache_invalidations.append(True))

    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        data={"source_path": "/client/outputs/anonymized_gcc_tracker_sample.xlsx"},
        files={
            "file": (
                "anonymized_gcc_tracker_sample.xlsx",
                b"workbook bytes",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    stored_path = refresh_calls["workbook_path"]
    assert stored_path.is_file()
    assert stored_path.suffix == ".xlsx"
    assert payload["uploaded_filename"] == "anonymized_gcc_tracker_sample.xlsx"
    assert payload["source_path"] == "/client/outputs/anonymized_gcc_tracker_sample.xlsx"
    assert payload["source_row_count"] == 2
    assert cache_invalidations == [True]


def test_refresh_hana_rejects_non_excel_upload() -> None:
    """Verify the admin refresh endpoint rejects unsupported non-Excel files."""

    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        files={"file": ("GCC Tracker.csv", b"workbook bytes", "text/csv")},
    )

    assert response.status_code == 422
    assert ".xlsb or .xlsx" in response.json()["detail"]


def test_refresh_hana_rejects_empty_upload() -> None:
    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        files={"file": ("GCC Tracker.xlsb", b"", "application/vnd.ms-excel.sheet.binary.macroEnabled.12")},
    )

    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


def test_refresh_hana_maps_workbook_errors_to_422(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(admin_module, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        admin_module,
        "refresh_metal_composition_hana",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GCCTrackerWorkbookLoadError("No completed GCC rows found")),
    )

    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        files={"file": ("GCC Tracker.xlsb", b"bad workbook", "application/vnd.ms-excel.sheet.binary.macroEnabled.12")},
    )

    assert response.status_code == 422
    assert "No completed GCC rows" in response.json()["detail"]


def test_refresh_hana_maps_hana_errors_to_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(admin_module, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        admin_module,
        "refresh_metal_composition_hana",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GCCTrackerHanaRefreshError("hana unavailable")),
    )

    response = _client().post(
        "/api/metal-composition/admin/gcc-tracker/refresh-hana",
        headers=_headers(),
        files={"file": ("GCC Tracker.xlsb", b"workbook bytes", "application/vnd.ms-excel.sheet.binary.macroEnabled.12")},
    )

    assert response.status_code == 503
    assert "hana unavailable" in response.json()["detail"]


def test_refresh_hana_returns_409_when_refresh_is_already_running() -> None:
    assert admin_module._REFRESH_LOCK.acquire(blocking=False)
    try:
        response = _client().post(
            "/api/metal-composition/admin/gcc-tracker/refresh-hana",
            headers=_headers(),
            files={"file": ("GCC Tracker.xlsb", b"workbook bytes", "application/vnd.ms-excel.sheet.binary.macroEnabled.12")},
        )
    finally:
        admin_module._REFRESH_LOCK.release()

    assert response.status_code == 409
    assert "already running" in response.json()["detail"]
