"""Planning calendar tab: odd/even week split of two-weekly SEFIs.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_calendar.py -q
"""

from __future__ import annotations

from production_wheel.sap_settings.calendar import calendar_order, planning_calendar_rows
from production_wheel.sap_settings.policy import resolve_policy, validate_rows

POLICY = resolve_policy(validate_rows([
    {"plant": "P1", "setting": "calendar_odd_profile", "value": "ODD-W1"},
    {"plant": "P1", "setting": "calendar_even_profile", "value": "EVEN-W2"},
]), "P1")


def _member(material, sefi, forecast, frequency="02W", subgroup="G1"):
    """One FINI row of the selected point."""

    return {"material": material, "sefi": sefi, "sefi_forecast_litres": forecast,
            "production_frequency_code": frequency, "proposed_subgroup_code": subgroup}


def test_odd_ranks_first_and_cut_at_half_the_forecast() -> None:
    """Forecasts 40/30/20/10 visit 40, 20, 30, 10; the 20 crosses 50% and goes odd."""

    order, cut = calendar_order([40, 30, 20, 10])
    assert order == [0, 2, 1, 3]
    assert cut == 1
    members = [_member("F1", "A", 40), _member("F2", "B", 30), _member("F3", "C", 20), _member("F4", "D", 10),
               _member("F5", "E", None, frequency="01W"), _member("F6", "F", 5, subgroup="")]
    rows, status, _ = planning_calendar_rows(members, POLICY, "P1")
    assigned = {row["sefi"]: row for row in rows if row["action"] == "assign"}
    assert status == "ready"
    assert {sefi: row["lot_size_profile"] for sefi, row in assigned.items()} == {
        "A": "ODD-W1", "C": "ODD-W1", "B": "EVEN-W2", "D": "EVEN-W2"}
    assert assigned["A"]["planning_calendar"] == "W1"
    assert assigned["A"]["lot_size_procedure"] == "PK"
    resets = [row for row in rows if row["action"] == "reset"]
    assert [row["material"] for row in resets] == ["F5"]
    assert resets[0]["lot_size_procedure"] == "EX"


def test_a_dominant_sefi_alone_takes_the_odd_week() -> None:
    """A first SEFI above 50% crosses immediately."""

    assert calendar_order([70, 20, 10]) == ([0, 2, 1], 0)


def test_blocked_without_profiles_or_with_ambiguous_forecasts() -> None:
    """Missing customizing codes and conflicting SEFI forecasts block the tab."""

    rows, status, message = planning_calendar_rows([_member("F1", "A", 40)], resolve_policy([], "P1"), "P1")
    assert (rows, status) == ([], "blocked") and "odd- and even-week lot-size profiles" in message
    rows, status, message = planning_calendar_rows(
        [_member("F1", "A", 40), _member("F2", "A", 41)], POLICY, "P1")
    assert status == "blocked" and "A" in message
    rows, status, _ = planning_calendar_rows([_member("F1", "A", 0)], POLICY, "P1")
    assert status == "blocked"
