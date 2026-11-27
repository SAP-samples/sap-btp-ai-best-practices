"""Population differences between production version modes FIXED and OPTIMIZED.

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_pv_mode_scope.py
"""

from app.workspace.pv_mode_scope import LISTED_KEYS, pv_mode_differences
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def _row(material, sefi="S", *, fixed="modeled", optimized="modeled", reason=None, plant="P"):
    """Build one fini_master row with the fields both modes need."""
    return {
        "plant": plant, "material": material, "sefi": sefi, "fini_plant_key": f"{material}/{plant}",
        "model_status": fixed, "exclusion_reason": reason,
        "optimized_pv_model_status": optimized, "optimized_pv_exclusion_reason": None if optimized == "modeled" else reason,
        "forecast_litres_12m": 1000, "package_volume": 1, "pallet_litres_resolved": 10,
        "eligible_lines": "L1", "fixed_pv": "PV1" if fixed == "modeled" else None,
    }


def test_fixed_reports_products_only_optimized_includes():
    """FIXED names the products OPTIMIZED would add, with their reason."""
    rows = [_row("1"), _row("2", fixed="excluded", reason="missing_fixed_pv"),
            _row("3", fixed="excluded", optimized="excluded", reason="missing_fixed_pv")]
    result = pv_mode_differences(rows, [], "FIXED")
    assert [item["fini_plant_key"] for item in result["only_in_other_mode"]] == ["2/P"]
    assert result["only_in_other_mode"][0]["reason_excluded"] == "missing_fixed_pv"
    assert (result["modeled_count"], result["other_mode_modeled_count"]) == (1, 2)
    assert result["warnings"][0].startswith("Production version FIXED leaves out 1 FINI(s) that OPTIMIZED would include: 2/P")
    # With OPTIMIZED nothing is lost, so there is nothing to warn about.
    assert pv_mode_differences(rows, [], "OPTIMIZED")["warnings"] == []


def test_scope_limits_the_comparison_and_long_lists_are_shortened():
    """Only products in the run's plant/SEFI scope count; the text lists ten keys."""
    rows = [_row(str(i), sefi="A", fixed="excluded", reason="missing_fixed_pv") for i in range(LISTED_KEYS + 2)]
    rows.append(_row("x", sefi="B", fixed="excluded", reason="missing_fixed_pv"))
    assert len(pv_mode_differences(rows, [{"plant": "P", "sefi": "B"}], "FIXED")["only_in_other_mode"]) == 1
    result = pv_mode_differences(rows, [{"plant": "P"}], "FIXED")
    assert len(result["only_in_other_mode"]) == LISTED_KEYS + 3
    assert "and 3 more" in result["warnings"][0]


def test_optimized_falls_back_to_fixed_population_without_optimized_rows():
    """Like the optimizer, OPTIMIZED uses the FIXED population when nothing is optimized-modeled."""
    rows = [_row("1", optimized="excluded"), _row("2", optimized="excluded")]
    result = pv_mode_differences(rows, [], "FIXED")
    assert result["only_in_this_mode"] == [] and result["warnings"] == []


def test_validate_draft_returns_pv_mode_scope_and_warning():
    """The Validate panel and the agent receive the comparison for the draft's mode."""
    service = WorkspaceService(MemoryRepository())
    source = service.register_dataset(
        "pv modes",
        {
            "metadata": {"plant": "P", "settings": {}},
            "issues": [],
            "tables": {
                "fini_master": [_row("001"), _row("002", fixed="excluded", reason="missing_fixed_pv")],
                "production_versions": [{"plant": "P", "sefi": "S", "production_version": "PV1", "lot_size_litres": 100, "active": True}],
            },
        },
        [],
    )
    service.publish_dataset(source["dataset_id"])
    draft = service.create_draft(source["dataset_id"])
    validation = service.validate_draft(draft["draft_id"])
    assert validation["valid"], validation
    assert [item["fini_plant_key"] for item in validation["pv_mode_scope"]["only_in_other_mode"]] == ["002/P"]
    assert any("OPTIMIZED would include" in warning for warning in validation["warnings"])

    optimized = service.update_draft(draft["draft_id"], draft["revision"], {"request": {"config": {"pv_mode": "OPTIMIZED"}}})
    validation = service.validate_draft(optimized["draft_id"])
    assert validation["valid"], validation
    assert validation["counts"]["modeled_fini_count"] == 2
    assert validation["pv_mode_scope"]["only_in_other_mode"] == []
