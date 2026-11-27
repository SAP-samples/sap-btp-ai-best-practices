"""Reconciliation helpers: wheel members from workbook columns, partition check, key comparison.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_reconcile.py -q
"""

from __future__ import annotations

import pytest

from production_wheel.sap_settings.reconcile import compare, recipe_partition, wheel_members


def _fini(material, group_key, forecast, pallets=None):
    """Canonical FINI row of the current wheel (group key SEFI/plant/recipe/subgroup)."""

    return {"material": material, "sefi": "S1", "baseline_assigned": 1 if group_key else 0,
            "legacy_baseline_group_key": group_key, "fixed_pv": "PV - 2",
            "legacy_coverage_selected_pv_days": 12.0, "legacy_selected_pv_lot_litres": 1000.0,
            "pallet_litres_resolved": 500.0, "forecast_litres_12m": forecast,
            "legacy_pallets_per_subgroup": pallets}


def test_wheel_members_use_workbook_values_and_the_template_pallet_formula() -> None:
    """Older workbooks lack pallets per subgroup: share x lot / pallet litres fills it."""

    members = wheel_members([_fini("A", "S1/P/x/1", 300), _fini("B", "S1/P/x/1", 100),
                             _fini("C", "S1/P/x/2", 50, pallets=0.7), _fini("D", None, 10)])
    by_material = {member["material"]: member for member in members}
    assert by_material["A"]["proposed_subgroup_code"] == "x/1" and by_material["A"]["selected_pv"] == "PV - 2"
    assert by_material["A"]["legacy_nominal_pallet_count"] == pytest.approx(1000 * 0.75 / 500)
    assert by_material["C"]["legacy_nominal_pallet_count"] == 0.7  # workbook value wins
    assert by_material["D"]["proposed_subgroup_code"] == "" and by_material["A"]["base_group_coverage_days"] == 12.0


def test_recipe_partition_ignores_labels_but_detects_regrouping() -> None:
    """Relabelled subgroups are the same partition; moving one FINI is not."""

    ours = [{"material": m, "sefi": "S1", "recipe_combination": c} for m, c in (("A", "0"), ("B", "0"), ("C", "1"))]
    same, translation = recipe_partition(ours, [{"material": "A", "recipe_combination": "X"},
                                                 {"material": "B", "recipe_combination": "X"},
                                                 {"material": "C", "recipe_combination": "Y"}])
    assert same == {"sefis": 1, "same_partition": 1, "differing": []}
    assert translation == {("S1", "0"): "X", ("S1", "1"): "Y"}
    moved, _ = recipe_partition(ours, [{"material": "A", "recipe_combination": "X"},
                                       {"material": "B", "recipe_combination": "Y"},
                                       {"material": "C", "recipe_combination": "Y"}])
    assert moved["differing"] == ["S1"]


def test_compare_counts_keys_and_field_mismatches() -> None:
    """Numbers compare with tolerance, text case-insensitively; unmatched keys are listed."""

    result = compare(
        [{"material": "1", "code": "05", "kpi": 2.04}, {"material": "2", "code": "9a", "kpi": 1.0}],
        [{"material": "1", "code": "05", "kpi": 2.0}, {"material": "2", "code": "9A", "kpi": 1.2},
         {"material": "3", "code": "10", "kpi": 0.0}],
        ("material",), ("code", "kpi"), tolerance=0.051)
    assert (result["matched"], result["equal"], result["only_reference"], result["only_ours"]) == (2, 1, [("3",)], [])
    assert result["mismatches"] == {"code": [], "kpi": [(("2",), 1.0, 1.2)]}
