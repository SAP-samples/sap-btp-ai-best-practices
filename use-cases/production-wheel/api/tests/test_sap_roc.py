"""Coverage rules, SAP lot-size keys and the runner adjustment of the SAP template.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_roc.py -q
"""

from __future__ import annotations

import pytest

from production_wheel.runners import CycleStockPolicy, RunnerThresholds
from production_wheel.sap_settings.roc import (
    excel_round,
    normalized_days,
    pallet_adjusted_roc,
    runner_adjusted_roc,
    runner_adjustments,
    runner_class,
    sap_lot_size_code,
)

POLICY = CycleStockPolicy()
THRESHOLDS = RunnerThresholds()


def test_excel_round_is_half_away_from_zero() -> None:
    """Excel ROUND differs from Python round on halves."""

    assert excel_round(2.5) == 3.0
    assert excel_round(0.25, 1) == 0.3
    assert excel_round(-2.5) == -3.0
    assert excel_round(None) is None


@pytest.mark.parametrize(
    "roc,pallets,expected",
    [(20, 0.5, 40), (20, 0.95, 20), (20, None, 20), (20, -1, 20), (20, 0, 222), (None, 0.5, None)],
)
def test_pallet_adjustment_covers_one_full_pallet(roc, pallets, expected) -> None:
    """Below 0.9 pallets per batch the DC covers a whole pallet; 0 counts as 0.09."""

    assert pallet_adjusted_roc(roc, pallets) == expected


@pytest.mark.parametrize(
    "days,expected", [(250, 200), (155, 150), (100, 100), (37, 37), (37.5, None), (0, None), (None, None)]
)
def test_days_are_capped_and_rounded_down_to_tens_above_100(days, expected) -> None:
    """Days above the maximum cap; above 100 snap down to tens; fractions have no key."""

    assert normalized_days(days) == expected


@pytest.mark.parametrize(
    "days,code",
    [(1, "01"), (9, "09"), (99, "99"), (100, "9A"), (140, "9E"), (180, "9I"), (190, "9K"), (200, "9L"),
     (105, None), (0, None), (None, None)],
)
def test_sap_lot_size_keys(days, code) -> None:
    """Keys match the 110-row template table: 1-99 padded, 100-200 as 9A..9L without J."""

    assert sap_lot_size_code(days) == code


@pytest.mark.parametrize(
    "full,ratio,expected",
    [(8, 5, "High"), (30, 2, "High"), (30, 5, "Medium"), (100, 12, "Low"), (100, 5, "Medium"), (None, None, None)],
)
def test_runner_class_takes_the_faster_of_absolute_and_relative(full, ratio, expected) -> None:
    """High beats Medium beats Low across the absolute and the within-SEFI view."""

    assert runner_class(full, ratio, THRESHOLDS) == expected


@pytest.mark.parametrize(
    "base,ratio,full,runner,expected",
    [
        (10, 2.0, 8, "High", (20, "adjusted")),        # target 13.0 -> 2 cycles
        (10, 5.0, 30, "Medium", (20, "adjusted")),     # minimum 20 -> 2 cycles beats 30*0.5/10 = 1
        (20, 10.0, 60, "Medium", (20, "adjusted")),    # 60*0.5/20 = 1 still caps ProposedCycles = 2 above 20
        (10, 12.0, 100, "Low", (40, "adjusted")),      # minimum 40 -> 4 cycles
        (90, 50.0, 500, "Low", (180, "adjusted")),     # 3 cycles, capped at floor(200/90) = 2
        (10, 1.0, 10, "High", (10, "reference_runner")),
        (210, 2.0, 50, "Low", (None, "base_above_max")),
        (10, 0.5, 10, "High", (None, "invalid_input")),
        (10, None, 10, "High", (None, "missing_input")),
    ],
)
def test_runner_adjusted_roc_branches(base, ratio, full, runner, expected) -> None:
    """Every branch of the template's adjusted-RoC formula."""

    assert runner_adjusted_roc(base, ratio, full, runner, POLICY) == expected


def test_runner_adjustments_apply_only_to_runner_multiple_subgroups() -> None:
    """The fastest FINI keeps its RoC; a medium runner doubles; single-RoC subgroups stay at base."""

    members = [
        {"sefi": "S1", "plant": "P1", "proposed_subgroup_code": "G1", "material": "F1",
         "reference_coverage_source_days": "10", "base_group_coverage_days": "9.5",
         "cycle_stock_strategy": "runner_multiple", "runner_multiple": "1"},
        {"sefi": "S1", "plant": "P1", "proposed_subgroup_code": "G1", "material": "F2",
         "reference_coverage_source_days": "50", "base_group_coverage_days": "9.5",
         "cycle_stock_strategy": "runner_multiple", "runner_multiple": "2"},
        {"sefi": "S2", "plant": "P1", "proposed_subgroup_code": "G2", "material": "F3",
         "reference_coverage_source_days": "40", "base_group_coverage_days": "12.4"},
    ]
    rows = {row["material"]: row for row in runner_adjustments(members, POLICY, THRESHOLDS)}

    assert rows["F1"]["status"] == "reference_runner"
    assert rows["F1"]["coverage_days"] == 10
    assert rows["F2"]["sefi_ratio"] == 5.0
    assert rows["F2"]["runner_class"] == "Medium"
    assert (rows["F2"]["coverage_days"], rows["F2"]["applied"]) == (20, True)
    assert (rows["F2"]["runner_multiple"], rows["F2"]["optimizer_runner_multiple"]) == (2, 2)
    assert (rows["F1"]["runner_multiple"], rows["F3"]["runner_multiple"]) == (1, 1)
    assert (rows["F3"]["status"], rows["F3"]["coverage_days"]) == ("single_roc", 12)
    assert rows["F3"]["cycle_stock_strategy"] == "single"
