"""Frequency codes come from the uploaded workbook, never from the planner's typing.

Covers the inventory read from the FINI table, the ``all_available`` setting
resolved during extraction, the undetermined-frequency warning and the
read-only preview endpoint used by the Datasets page.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
from openpyxl.worksheet.table import Table

from production_wheel.extraction.datasets import _settings, extract_workbooks, preview_workbook

# One FINI per source frequency label, including a value with no weekly code.
FREQUENCIES = ["01W. Weekly", "02W. Every two weeks", "08W. Every two months", "08W. Every two months",
               "999.Not possible to determine"]


def _workbook(path: Path) -> None:
    """Write a minimal main/PV workbook whose FINIs use several production frequencies."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Inputs"
    sheet.append(["Material", "SEFI", "Plant", "Filling line", "Production Frequency",
                  "Total FINI forecast (litres)", "Volume", "Pallet in litres", "Selected PV"])
    for index, frequency in enumerate(FREQUENCIES):
        sheet.append([str(100 + index), "200", "TEST", "L1", frequency, 10000, 5, 500, "PV - 1"])
    sheet.add_table(Table(displayName="FINI_grouping_for_wheel", ref=f"A1:I{len(FREQUENCIES) + 1}"))
    versions = workbook.create_sheet("Versions")
    versions.append(["SEFI", "Clasificador", "Lot size (Litres)"])
    versions.append(["200", "PV - 1", 1000])
    versions.add_table(Table(displayName="Production_versions___List", ref="A1:C2"))
    workbook.save(path)


def test_preview_counts_codes_and_undetermined_values(tmp_path):
    """The preview lists each weekly code with its row count and keeps 999 apart."""
    path = tmp_path / "input.xlsx"
    _workbook(path)
    preview = preview_workbook(path)
    assert preview["plant"] == "TEST"
    inventory = preview["frequency_inventory"]
    assert inventory["codes"] == [
        {"code": "01W", "label": "01W. Weekly", "rows": 1},
        {"code": "02W", "label": "02W. Every two weeks", "rows": 1},
        {"code": "08W", "label": "08W. Every two months", "rows": 2},
    ]
    assert inventory["unrecognized"] == [{"value": "999.Not possible to determine", "rows": 1}]
    assert inventory["total_rows"] == 5


def test_all_available_admits_every_workbook_code(tmp_path):
    """``all_available`` resolves to the workbook's codes and records that origin."""
    path = tmp_path / "input.xlsx"
    _workbook(path)
    result = extract_workbooks(path, settings={"modeled_frequencies": "all_available"})
    metadata = result["metadata"]
    assert metadata["settings"]["modeled_frequencies"] == ["01W", "02W", "08W"]
    assert metadata["parameter_sources"]["modeled_frequencies"] == "user_all_available_in_workbook"
    statuses = {row["production_frequency"]: row["model_status"] for row in result["tables"]["fini_master"]}
    assert statuses["08W. Every two months"] == "modeled"
    # Rows without a weekly code stay out of scope even when every code is requested.
    assert statuses["999.Not possible to determine"] == "out_of_scope"


def test_default_extraction_reports_undetermined_frequency(tmp_path):
    """Undetermined source frequencies always produce a review warning and inventory."""
    path = tmp_path / "input.xlsx"
    _workbook(path)
    result = extract_workbooks(path)
    assert result["metadata"]["settings"]["modeled_frequencies"] == ["01W", "02W"]
    assert result["metadata"]["frequency_inventory"]["unrecognized"][0]["rows"] == 1
    warning = [issue for issue in result["issues"] if issue["rule_id"] == "PRODUCTION_FREQUENCY_UNDETERMINED"]
    assert warning == [{
        "rule_id": "PRODUCTION_FREQUENCY_UNDETERMINED", "severity": "warning", "entity": "fini_master",
        "entity_key": "999.Not possible to determine", "observed": "1 FINI rows",
        "expected": "a weekly frequency code such as 01W",
        "action": "These FINIs stay out of scope. Correct the source frequency to model them.",
    }]


def test_settings_validator_accepts_keyword_and_rejects_other_text():
    """Only the documented keyword is accepted in place of a code list."""
    resolved, origins = _settings({"modeled_frequencies": " ALL_AVAILABLE "})
    assert resolved["modeled_frequencies"] == "all_available"
    assert origins["modeled_frequencies"] == "user"
    with pytest.raises(ValueError):
        _settings({"modeled_frequencies": "everything"})


def test_preview_endpoint_saves_nothing(tmp_path):
    """POST /api/datasets/preview returns the inventory without a service dependency."""
    from app.routers.workspace import router

    path = tmp_path / "input.xlsx"
    _workbook(path)
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post(
        "/api/datasets/preview", files={"primary": ("source.xlsx", path.read_bytes())}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "source.xlsx"
    assert [item["code"] for item in body["frequency_inventory"]["codes"]] == ["01W", "02W", "08W"]
    bad = TestClient(app).post("/api/datasets/preview", files={"primary": ("x.xlsx", b"not a zip")})
    assert bad.status_code == 422
