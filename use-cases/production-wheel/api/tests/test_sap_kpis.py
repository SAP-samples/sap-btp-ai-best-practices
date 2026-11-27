"""Changeover KPI tab, PV check and end-to-end SAP settings generation.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_kpis.py -q
"""

from __future__ import annotations

import pytest
from production_wheel.metrics import changeover_contribution, group_frequency
from production_wheel.sap_settings.generate import generate
from production_wheel.sap_settings.kpis import changeover_rows, pv_conflict_rows
from production_wheel.sap_settings.policy import resolve_policy, validate_rows

DEMAND_DAYS, PRODUCTIVE_WEEKS = 250, 50


def _group(sefi, subgroup, materials, demand, batch, pcks=None, lines="L1"):
    """Members of one group whose base coverage follows the optimizer's definition."""

    coverage = batch * DEMAND_DAYS / demand
    return [
        {"sefi": sefi, "proposed_subgroup_code": subgroup, "material": material,
         "base_group_coverage_days": coverage, "pck_code": (pcks or ["K1"] * len(materials))[index],
         "package_volume": 5, "common_lines": lines, "selected_pv": "PV - 2",
         "forecast_litres_12m": demand / len(materials), "legacy_nominal_pallet_count": 1.0}
        for index, material in enumerate(materials)
    ]


def test_kpi_changeovers_equal_the_optimizer_j_ch() -> None:
    """With 5 days per week, batches x (FINIs - 1) reproduces J_CH = F x (n - 1)."""

    first = _group("S1", "G1", ["F1", "F2", "F3"], demand=12_500, batch=500, pcks=["K1", "K2", "K2"])
    second = _group("S1", "G2", ["F4", "F5"], demand=25_000, batch=500)
    expected_j_ch = sum(
        float(changeover_contribution(group_frequency(demand, batch, PRODUCTIVE_WEEKS), count))
        for demand, batch, count in ((12_500, 500, 3), (25_000, 500, 2))
    )
    rows = changeover_rows(first + second, "P1", DEMAND_DAYS / PRODUCTIVE_WEEKS)
    total = rows[-1]
    assert total["level"] == "site_total"
    assert total["changeovers_per_week"] == pytest.approx(expected_j_ch)
    subgroup = rows[0]
    assert (subgroup["pack_basis"], subgroup["pack_changeovers_per_week"],
            subgroup["same_pack_changeovers_per_week"]) == ("PCK", 0.5, 0.5)


def test_volume_fallback_unresolved_packs_and_line_split() -> None:
    """Missing PCK falls back to volume; nothing usable leaves changes unclassified."""

    volume = _group("S1", "G1", ["F1", "F2"], demand=12_500, batch=500, pcks=["K1", None], lines="L1|L2")
    unresolved = [dict(row, pck_code=None, package_volume=None) for row in _group("S2", "G1", ["F3", "F4"], 12_500, 500)]
    rows = changeover_rows(volume + unresolved, "P1", 5, line_breakdown=True)
    by_level = {(row["level"], row.get("filling_line") or row.get("sefi")): row for row in rows}
    assert by_level["subgroup", "S1"]["pack_basis"] == "Volume fallback"
    assert by_level["subgroup", "S2"]["unclassified_changeovers_per_week"] == 0.5
    assert by_level["filling_line", "L2"]["changeovers_per_week"] == 0.25


def test_runner_multiple_subgroup_uses_sum_of_inverse_multiples() -> None:
    """Worked example: base 10 days, 0.5 batches/week, multiples 1/2/2/4 -> 0.625 changeovers/week."""

    from production_wheel.runners import CycleStockPolicy, RunnerThresholds
    from production_wheel.sap_settings.roc import runner_adjustments

    example = {"P1": (4000, 12), "P2": (1600, 30), "P3": (690, 70), "P4": (160, 300)}
    members = [
        {"sefi": "S1", "plant": "P1", "proposed_subgroup_code": "G1", "material": fini,
         "base_group_coverage_days": 10, "reference_coverage_source_days": full, "forecast_litres_12m": demand,
         "pck_code": "K1", "package_volume": 5, "common_lines": "L1", "cycle_stock_strategy": "runner_multiple"}
        for fini, (demand, full) in example.items()
    ]
    runner = runner_adjustments(members, CycleStockPolicy(), RunnerThresholds())
    assert [row["runner_multiple"] for row in runner] == [1, 2, 2, 4]
    rows = changeover_rows(members, "P1", 5, runner_rows=runner)
    subgroup, total = rows[0], rows[-1]
    assert subgroup["cycle_stock_strategy"] == "runner_multiple"
    assert subgroup["batches_per_week"] == pytest.approx(0.5)
    assert subgroup["changeovers_per_week"] == pytest.approx(0.625)
    assert subgroup["single_changeovers_per_week"] == pytest.approx(1.5)
    assert (subgroup["base_coverage_days"], round(subgroup["coverage_days"], 2)) == (10, 14.29)
    assert (subgroup["pack_basis"], subgroup["unclassified_changeovers_per_week"]) == ("Runner multiples", pytest.approx(0.625))
    assert (total["changeovers_per_week"], total["single_changeovers_per_week"]) == (pytest.approx(0.625), pytest.approx(1.5))
    # Without runner rows (or for single subgroups) the historical numbers hold.
    assert changeover_rows(members, "P1", 5)[0]["changeovers_per_week"] == pytest.approx(1.5)

    # End to end: generation reads the strategy from the members and reports both KPIs.
    lookups = {"dc_network": None, "dc_pallet_quantities": None, "dc_abc_classes": None}
    tables, summary = generate(members, lookups, resolve_policy([], "P1"), {"point_index": 2}, "P1", "26Q4", 5)
    assert summary["runner_multiple_subgroups"] == 1
    assert summary["kpi_changeovers_per_week"] == pytest.approx(0.625)
    assert summary["kpi_single_changeovers_per_week"] == pytest.approx(1.5)
    assert summary["runner_adjusted_fini"] == 3
    assert round(summary["wheel_roc_days"], 2) == 14.3


def test_pv_conflicts_list_mixed_subgroups() -> None:
    """A subgroup using two production versions is reported."""

    members = _group("S1", "G1", ["F1", "F2"], 12_500, 500)
    members[1]["selected_pv"] = "PV - 3"
    assert pv_conflict_rows(members) == [{"sefi": "S1", "subgroup": "G1", "selected_pvs": "PV - 2|PV - 3"}]


def test_generate_reports_ready_blocked_and_not_applicable_tabs() -> None:
    """Missing lookups and policy block only their tabs; every row carries the point."""

    members = _group("S1", "G1", ["F1", "F2"], 12_500, 500) + [{"sefi": "S9", "material": "X1",
                                                                "production_frequency_code": "01W"}]
    lookups = {"dc_network": None, "dc_pallet_quantities": None, "dc_abc_classes": None,
               "production_version_masses": [{"sefi": "S1", "production_version": "PV - 2", "lot_size_kg": 900}],
               "recipe_characters": [{"list_of_characters_allowed_in_bw": "0"}]}
    policy = resolve_policy(validate_rows([{"plant": "P1", "setting": "multi_tier_dc", "value": "yes"}]), "P1")
    tables, summary = generate(members, lookups, policy, {"point_index": 4, "decision_label": "B"},
                               "P1", "26Q4", 5, point_j_ch=0.5)

    tabs = {name: tab["status"] for name, tab in summary["tabs"].items()}
    assert tabs == {"pv_check": "ready", "kpis": "ready", "cycle_stock": "blocked", "planning_calendar": "blocked",
                    "recipe_combination": "ready", "apo_lot_size": "ready", "central_dc_lot_size": "blocked",
                    "runner_audit": "ready"}
    assert summary["kpi_changeovers_per_week"] == pytest.approx(summary["point_j_ch"])
    assert summary["wheel_roc_days"] == 10.0
    assert tables["sap_apo_lot_sizes"][0]["lot_size_kg"] == 900
    assert all(row["point_index"] == 4 for rows in tables.values() for row in rows)
