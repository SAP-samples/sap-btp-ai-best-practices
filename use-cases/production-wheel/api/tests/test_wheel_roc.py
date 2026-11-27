"""Wheel RoC rule: whole-day RoC for batches and changeovers, pallets only for inventory.

Covers ``production_wheel.wheel_roc`` and the places that must agree with it:
the cycle stock strategy (runner multiples from ``R``), the SAP settings KPI tab
(same changeovers as the optimizer), the wheel export and the assistant's
assignment derivations. All data is synthetic.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_wheel_roc.py -q
"""

from __future__ import annotations

import math

import pytest

from app.workspace.analytics_evidence import _derivation, _number
from app.workspace.wheel_evidence import has_wheel_rule, wheel_derivations
from app.workspace.wheel_export import wheel_rows
from production_wheel.business_rules import runner_ratios
from production_wheel.candidates import CandidateMember
from production_wheel.cycle_stock import RUNNER_MULTIPLE, SINGLE, strategy_figures
from production_wheel.runners import CycleStockPolicy, RunnerThresholds, excel_round, runner_multiple
from production_wheel.sap_settings.generate import _same_changeovers
from production_wheel.sap_settings.kpis import changeover_rows
from production_wheel.sap_settings.roc import normalized_days, runner_adjustments
from production_wheel.schemas import CoverageBasis, RunConfig
from production_wheel.wheel_roc import (
    inventory_roc_days,
    pallets_per_batch,
    wheel_figures,
    wheel_roc_days,
)

# 80 demand days over 16 productive weeks: 5 days per week, as in the customer calendar.
DAYS, WEEKS = 80, 16
# FINI -> (demand litres, litres per pallet, full-batch coverage days):
# a high, a medium and a low runner of one SEFI.
FINIS = {"A": (54_000.0, 500.0, 10.0), "B": (8_000.0, 450.0, 72.0), "C": (2_000.0, 360.0, 280.0)}
NOMINAL_LOT = 3_500.0
EFFECTIVE = NOMINAL_LOT * 0.9  # base RoC = 80 x 3150 / 64000 = 3.94 days -> R = 4


def _members() -> list[CandidateMember]:
    """Return the synthetic FINIs of one plant/SEFI block."""

    return [CandidateMember(fini_id=fini, plant="P", sefi="S", eligible_lines=frozenset({"L1"}),
                            demand_litres=demand, pallet_litres=pallet, package_volume=1.0,
                            reference_coverage_source_days=full)
            for fini, (demand, pallet, full) in FINIS.items()]


def _config(basis: CoverageBasis = CoverageBasis.ADJUSTED_GROUP) -> RunConfig:
    """Return a run configuration with the customer calendar and the given basis."""

    return RunConfig(demand_days=DAYS, productive_weeks=WEEKS, coverage_basis=basis,
                     cycle_stock_policy={"mode": "optimizer_choice"})


def test_wheel_roc_rounds_like_excel_with_a_one_day_minimum() -> None:
    """R is Excel ROUND to whole days (half away from zero), never below 1 day."""

    assert wheel_roc_days(3.8913) == 4
    assert wheel_roc_days(4.5) == 5
    assert wheel_roc_days(112.5) == 113
    assert wheel_roc_days(4.49) == 4
    assert wheel_roc_days(0.3) == 1
    assert wheel_roc_days(None) is None and wheel_roc_days(0) is None


def test_inventory_roc_reproduces_the_customer_example() -> None:
    """Leading product 5 days, slow runner 10 days (multiple 2) becomes 11 with pallets."""

    # One batch fills 0.85 pallets of the slow runner: floor(10 / 0.85) = 11.
    assert inventory_roc_days(5 * 2, 0.85) == 11.0
    # 0.9 pallets or more: the wheel RoC is already a full pallet.
    assert inventory_roc_days(10, 0.9) == 10
    # Pallets that round to 0.0 count as 0.09, as in the workbook.
    assert inventory_roc_days(4, 0) == float(math.floor(4 / 0.09))
    # Missing pallet data leaves the RoC unchanged.
    assert inventory_roc_days(4, None) == 4


@pytest.mark.parametrize(
    ("multiple", "pallets", "inventory", "sap_days"),
    [(1, 29.1, 96, 96), (2, 5.2, 192, 190), (2, 2.6, 192, 190), (2, 0.8, 240, 200)],
)
def test_confirmed_pallets_of_one_batch_for_a_fini_with_a_multiple(
    multiple: int, pallets: float, inventory: float, sap_days: int
) -> None:
    """Customer-confirmed rule: the pallets of one batch, also for a FINI filled every k-th batch.

    Wheel RoC 96 days: batches and changeovers use 96 x k; inventory uses
    floor(96 x k / pallets) below 0.9 pallets; SAP settings cap at 200 days and
    round values above 100 days down to tens.
    """

    wheel = 96 * multiple
    assert inventory_roc_days(wheel, pallets) == inventory
    assert normalized_days(inventory_roc_days(wheel, pallets), 200) == sap_days


def test_pallets_per_batch_use_the_nominal_lot_and_one_decimal() -> None:
    """Pallets per batch = nominal lot x demand share / pallet litres, Excel-rounded to 0.1."""

    assert pallets_per_batch(1_000.0, 100.0, 2_000.0, 200.0) == 0.3  # 0.25 rounds half up
    assert pallets_per_batch(1_000.0, 1_900.0, 2_000.0, 100.0) == 9.5
    assert pallets_per_batch(1_000.0, 100.0, 2_000.0, None) is None


def test_pallets_never_change_frequency_changeovers_or_wheel_roc() -> None:
    """Changing pallet sizes moves only the inventory RoC."""

    ids = tuple(FINIS)
    demands = [FINIS[f][0] for f in ids]
    small = wheel_figures(ids, demands, [10_000.0] * 3, [1, 5, 10], NOMINAL_LOT, EFFECTIVE, DAYS, WEEKS)
    large = wheel_figures(ids, demands, [FINIS[f][1] for f in ids], [1, 5, 10], NOMINAL_LOT, EFFECTIVE, DAYS, WEEKS)
    for figures in (small, large):
        assert figures.roc_days == 4
        assert figures.frequency_per_week == pytest.approx(5 / 4)
        assert figures.fini_wheel_roc_days == (4.0, 20.0, 40.0)
        assert figures.j_ch == pytest.approx(1.25 * (1 + 1 / 5 + 1 / 10 - 1))
        assert figures.coverage(CoverageBasis.BASE_GROUP) == pytest.approx(
            sum(d * w for d, w in zip(demands, (4, 20, 40))) / sum(demands))
    assert small.fini_inventory_roc_days != large.fini_inventory_roc_days
    assert all(i >= w for i, w in zip(large.fini_inventory_roc_days, large.fini_wheel_roc_days))


def test_runner_multiples_are_computed_from_the_whole_day_wheel_roc() -> None:
    """Multiples use R even when the active basis is the pallet-adjusted coverage."""

    members, config = _members(), _config(CoverageBasis.ADJUSTED_GROUP)
    ratios = runner_ratios(members, config)
    figures = strategy_figures(RUNNER_MULTIPLE, members, NOMINAL_LOT, EFFECTIVE, config, ratios)
    roc = figures.wheel.roc_days
    assert roc == 4
    policy = config.cycle_stock_policy
    for item in figures.members:
        expected, _ = runner_multiple(roc, item.group_ratio, item.full_batch_days, item.runner_class, policy)
        assert item.multiple == expected
    # Every FINI wheel RoC is a whole multiple of R.
    assert all(days % roc == 0 for days in figures.wheel.fini_wheel_roc_days)
    assert figures.coverage_days == figures.wheel.coverage(CoverageBasis.ADJUSTED_GROUP)
    single = strategy_figures(SINGLE, members, NOMINAL_LOT, EFFECTIVE, config, ratios)
    assert single.wheel.fini_wheel_roc_days == (4.0, 4.0, 4.0)
    assert single.j_ch == pytest.approx(1.25 * 2)


def _member_rows(figures, strategy: str) -> list[dict]:
    """Build stored ``members`` rows of the synthetic group as reporting writes them."""

    wheel = figures.wheel
    return [{
        "plant": "P", "sefi": "S", "material": fini, "proposed_subgroup_code": "G001",
        "cycle_stock_strategy": strategy, "runner_multiple": k,
        "reference_coverage_source_days": FINIS[fini][2],
        "base_group_coverage_days": wheel.base_coverage_days,
        "forecast_litres_12m": FINIS[fini][0], "forecast_litres_period": FINIS[fini][0],
        "pck_code": f"PCK{index}", "package_volume": 1.0, "common_lines": "L1",
        "pallet_litres_resolved": FINIS[fini][1], "wheel_roc_days": wheel.roc_days,
        "fini_wheel_roc_days": days, "fini_inventory_roc_days": inventory, "pallets_per_batch": pallets,
    } for index, (fini, k, days, inventory, pallets) in enumerate(zip(
        wheel.fini_ids, wheel.multiples, wheel.fini_wheel_roc_days,
        wheel.fini_inventory_roc_days, wheel.fini_pallets_per_batch))]


@pytest.mark.parametrize("strategy", [SINGLE, RUNNER_MULTIPLE])
def test_optimizer_changeovers_equal_the_sap_settings_kpi(strategy: str) -> None:
    """The KPI tab's changeovers per week equal the optimizer's J_CH under both strategies."""

    members, config = _members(), _config()
    figures = strategy_figures(strategy, members, NOMINAL_LOT, EFFECTIVE, config, runner_ratios(members, config))
    rows = _member_rows(figures, strategy)
    runner = runner_adjustments(rows, CycleStockPolicy(), RunnerThresholds())
    kpis = changeover_rows(rows, "P", DAYS / WEEKS, runner_rows=runner)
    site = next(row for row in kpis if row["level"] == "site_total")
    assert site["batches_per_week"] == pytest.approx(figures.frequency_per_week)
    assert site["changeovers_per_week"] == pytest.approx(figures.j_ch)
    assert _same_changeovers(site["changeovers_per_week"], figures.j_ch) is True
    # SAP settings recompute the same multiples from the same R.
    assert {row["material"]: row["runner_multiple"] for row in runner} == dict(
        zip(figures.wheel.fini_ids, figures.wheel.multiples))


def test_sap_kpi_uses_the_one_day_minimum() -> None:
    """A base RoC under half a day gives R = 1 in SAP settings, not a division by zero."""

    rows = [{"plant": "P", "sefi": "S", "material": "A", "proposed_subgroup_code": "G001",
             "base_group_coverage_days": 0.3, "forecast_litres_12m": 1.0, "package_volume": 1.0}]
    subgroup = changeover_rows(rows, "P", 5.0)[0]
    assert subgroup["base_coverage_days"] == 1
    assert subgroup["batches_per_week"] == 5.0
    assert excel_round(0.3, 0) == 0  # the plain workbook rounding the minimum guards against


def test_wheel_export_writes_wheel_roc_and_keeps_older_runs_unchanged() -> None:
    """RoC selected PV is R for new runs; runs without R keep their stored coverage."""

    headers = ["Material", "RoC selected PV", "Pallets as per subgroup"]
    base = {"source_sheet": "S", "source_row": 1, "material": "A", "proposed_subgroup": "G",
            "cycle_stock_strategy": "single"}
    new = wheel_rows(headers, {}, [{**base, "wheel_roc_days": 4, "pallets_per_batch": 0.3,
                                    "fini_adjusted_coverage_days": 13.0, "selected_pallet_count": 1.0,
                                    "fini_inventory_roc_days": 13.0}], {})[0]
    assert new["RoC selected PV"] == 4
    assert new["Pallets as per subgroup"] == 0.3
    assert new["Inventory RoC with pallets (days)"] == 13.0
    old = wheel_rows(headers, {}, [{**base, "fini_adjusted_coverage_days": 4.64,
                                    "selected_pallet_count": 1.0}], {})[0]
    assert old["RoC selected PV"] == 4.64
    assert old["Pallets as per subgroup"] == 1.0


def test_assignment_derivations_reproduce_stored_wheel_figures() -> None:
    """The assistant's derivations recompute every stored wheel figure of a new run."""

    members, config = _members(), _config()
    figures = strategy_figures(RUNNER_MULTIPLE, members, NOMINAL_LOT, EFFECTIVE, config,
                               runner_ratios(members, config))
    rows = _member_rows(figures, RUNNER_MULTIPLE)
    group = {"wheel_roc_days": figures.wheel.roc_days, "base_group_coverage_days": figures.wheel.base_coverage_days,
             "nominal_lot_litres": NOMINAL_LOT, "group_demand_litres": sum(d for d, _, _ in FINIS.values()),
             "frequency_per_week": figures.frequency_per_week, "j_ch_contribution": figures.j_ch,
             "wheel_coverage_days": figures.wheel.coverage(CoverageBasis.BASE_GROUP),
             "adjusted_group_coverage_days": figures.wheel.coverage(CoverageBasis.ADJUSTED_GROUP),
             "worst_fini_coverage_days": figures.wheel.coverage(CoverageBasis.WORST_FINI),
             "coverage_days": figures.coverage_days, "active_coverage_basis": "ADJUSTED_GROUP"}
    assert has_wheel_rule(group) and not has_wheel_rule({"wheel_roc_days": ""})
    group_derivations, member_derivations = wheel_derivations(
        group, rows, {"demand_days": DAYS, "productive_weeks": WEEKS}, _derivation, _number)
    assert all(item["matches_reported"] for item in group_derivations.values())
    assert all(item["matches_reported"] for per in member_derivations.values() for item in per.values())
