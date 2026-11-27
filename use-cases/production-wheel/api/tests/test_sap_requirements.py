"""Shared SAP input checks: each gap names what is missing and how to fix it.

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_sap_requirements.py
"""
from production_wheel.sap_settings.profile_policy import SapSettingsPolicy
from production_wheel.sap_settings.requirements import (
    calendar_policy_gap,
    central_dc_gap,
    cycle_stock_gaps,
    input_gaps,
    recipe_pool_gap,
)

FULL = {"dc_network": [], "dc_pallet_quantities": [], "dc_abc_classes": [], "production_version_masses": []}


def test_complete_inputs_have_no_gaps():
    """With every lookup, both calendar codes and a character pool nothing is missing."""
    policy = SapSettingsPolicy(calendar_odd_profile="ODD-W1", calendar_even_profile="EVEN-W2").to_resolved()
    assert input_gaps(FULL, policy, ["0", "1"]) == []


def test_each_missing_input_is_reported_once_with_its_kind():
    """Missing views, missing codes, the APO dependency and the intermediate DC are separate gaps."""
    policy = SapSettingsPolicy(multi_tier_dc=True).to_resolved()
    gaps = input_gaps({**FULL, "dc_abc_classes": None}, policy, [])
    assert [(gap.tab, gap.status, gap.code) for gap in gaps] == [
        ("cycle_stock", "partial", "dataset_view"),
        ("planning_calendar", "blocked", "profile_setting"),
        ("recipe_combination", "blocked", "dataset_view"),
        ("apo_lot_size", "blocked", "depends_on"),
        ("central_dc_lot_size", "blocked", "not_ported"),
    ]
    assert all(gap.fix for gap in gaps)


def test_individual_checks():
    """Blocked DC network, one missing calendar code, empty pool, single-tier DC."""
    assert cycle_stock_gaps({**FULL, "dc_network": None})[0].status == "blocked"
    gap = calendar_policy_gap({"calendar_odd_profile": "ODD-W1", "calendar_even_profile": None})
    assert gap.setting == "calendar_even_profile"
    assert recipe_pool_gap(["A"]) is None and recipe_pool_gap([]).setting == "recipe_characters"
    assert central_dc_gap({"multi_tier_dc": "no"}) is None
