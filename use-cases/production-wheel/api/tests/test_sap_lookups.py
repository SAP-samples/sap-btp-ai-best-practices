"""SAP-settings lookup tables read from the uploaded inputs workbook.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_lookups.py -q
"""

from __future__ import annotations

import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table

from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService
from production_wheel.extraction.sap_lookups import extract_sap_lookups, merge_sap_lookups


def _add_table(book: Workbook, name: str, headers: list[str], rows: list[list]) -> None:
    """Write one sheet holding one Excel table, as in the customer workbook."""

    sheet = book.create_sheet(name[:30])
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    sheet.add_table(Table(displayName=name, ref=f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"))


def _inputs_workbook(*, pallet_headers: list[str] | None = None) -> bytes:
    """Build a two-plant inputs workbook without the optional recipe characters."""

    book = Workbook()
    book.remove(book.active)
    _add_table(
        book,
        "Network___Primary_DC",
        ["Material", "Production plant", "Primary DC", "Hub", "FINI forecast (litres)", "Has forecast"],
        [[1001, "P1", "D1", "North", 500, "YES"], [1002, "P1", "D2", "South", None, "NO"], [2001, "P2", "D1", "North", 90, "YES"]],
    )
    _add_table(book, "ABC_calculated", ["Material", "Plant", "ABC"], [[1001, "D1", "A"], [2001, "D1", "C"]])
    _add_table(
        book,
        "FINI_pallets",
        pallet_headers or ["Material", "Layer qty", "Pallet qty", "Volume", "Pallet in litres", "Producing plant", "FINI/P-site"],
        [[1001, 11, 44, 10, 440, "P1", "1001/P1"], [2001, 10, 40, 5, 200, "P2", "2001/P2"]],
    )
    _add_table(
        book,
        "Production_versions___List",
        ["SEFI/P-site", "Clasificador", "Lot size", "Lot size (Litres)"],
        [["S1/P1", "PV - 1", 10002, 7623], ["S1/P2", "PV - 1", 500, 400]],
    )
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_lookups_are_filtered_to_the_dataset_plant(tmp_path: Path) -> None:
    """Only P1 evidence survives; numeric materials become text; PV keys split."""

    path = tmp_path / "inputs.xlsx"
    path.write_bytes(_inputs_workbook())
    tables, issues = extract_sap_lookups(path, "P1")

    assert [row["material"] for row in tables["dc_network"]] == ["1001", "1002"]
    assert tables["dc_network"][0]["forecast_litres"] == 500
    assert [row["material"] for row in tables["dc_abc_classes"]] == ["1001"]
    assert tables["dc_pallet_quantities"] == [
        {
            "material": "1001", "plant": "P1", "fini_plant_key": "1001/P1", "pallet_qty": 44,
            "layer_qty": 11, "package_volume": 10, "source_sheet": "FINI_pallets", "source_row": 2,
        }
    ]
    masses = tables["production_version_masses"]
    assert (masses[0]["sefi"], masses[0]["plant"], masses[0]["lot_size_kg"]) == ("S1", "P1", 10002)
    assert [issue["entity_key"] for issue in issues] == ["RecipeCharacters"]
    assert issues[0]["severity"] == "warning"


def test_missing_headers_skip_only_that_view(tmp_path: Path) -> None:
    """A renamed column yields a warning and no view, never a failed import."""

    path = tmp_path / "inputs.xlsx"
    path.write_bytes(_inputs_workbook(pallet_headers=[
        "Material", "Layers", "Pallet qty", "Volume", "Pallet in litres", "Producing plant", "FINI/P-site"]))
    tables, issues = extract_sap_lookups(path, "P1")
    assert "dc_pallet_quantities" not in tables
    assert "dc_network" in tables
    assert any("missing headers" in issue["reason"] for issue in issues)


def test_recipe_characters_merge_without_duplicates() -> None:
    """Primary-workbook characters are kept; the inputs list adds only new ones."""

    tables = {"recipe_characters": [{"list_of_characters_allowed_in_bw": "0"}]}
    merge_sap_lookups(tables, {
        "recipe_characters": [{"list_of_characters_allowed_in_bw": "0"}, {"list_of_characters_allowed_in_bw": "A"}],
        "dc_network": [{"material": "1"}],
    })
    assert [row["list_of_characters_allowed_in_bw"] for row in tables["recipe_characters"]] == ["0", "A"]
    assert tables["dc_network"] == [{"material": "1"}]


def test_backfill_adds_only_lookup_views_to_an_existing_snapshot() -> None:
    """Optimizer inputs stay untouched; counts, issues and revision are refreshed."""

    repo = MemoryRepository()
    service = WorkspaceService(repo)
    repo.insert("datasets", "d", {
        "dataset_id": "d", "revision": 1, "status": "published",
        "metadata": {"plant": "P1", "source_files": ["primary.xlsx", "inputs.xlsx"], "counts": {"fini_master": 1}},
        "issues": [{"rule_id": "OTHER", "severity": "warning"}],
    })
    repo.replace_tables("d", {"fini_master": [{"material": "1001", "plant": "P1"}]})
    repo.put_artifact("d", "inputs.xlsx", _inputs_workbook())

    result = service.backfill_sap_lookups("d")

    stored = repo.get("datasets", "d")
    assert result["counts"]["dc_network"] == 2
    assert stored["revision"] == 2
    assert stored["metadata"]["counts"]["dc_network"] == 2
    assert stored["metadata"]["counts"]["fini_master"] == 1
    assert "sap_lookups_imported_at" in stored["metadata"]
    assert [issue["rule_id"] for issue in stored["issues"]] == ["OTHER", "SAP_LOOKUP_MISSING"]
    assert repo.rows("d", "fini_master") == [{"material": "1001", "plant": "P1"}]
    assert len(repo.rows("d", "dc_abc_classes")) == 1


def test_backfill_requires_a_stored_inputs_workbook() -> None:
    """Snapshots uploaded without an inputs workbook cannot be backfilled."""

    repo = MemoryRepository()
    repo.insert("datasets", "d", {"dataset_id": "d", "revision": 1,
        "metadata": {"plant": "P1", "source_files": ["primary.xlsx"]}, "issues": []})
    service = WorkspaceService(repo)
    try:
        service.backfill_sap_lookups("d")
    except ValueError as error:
        assert "inputs workbook" in str(error)
    else:
        raise AssertionError("backfill without an inputs workbook must fail")
