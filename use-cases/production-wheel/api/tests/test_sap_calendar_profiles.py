"""Odd/even lot-size profiles learnt from an earlier wheel of the plant.

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_sap_calendar_profiles.py
"""
from production_wheel.sap_settings.calendar_profiles import suggest_calendar_profiles

FINIS = [
    {"material": "1", "sefi": "S1", "sefi_forecast_litres": "900"},
    {"material": "2", "sefi": "S2", "sefi_forecast_litres": "500"},
    {"material": "3", "sefi": "S3", "sefi_forecast_litres": "300"},
]


def _calendar(*pairs, procedure="PK"):
    """Earlier calendar rows as (material, lot-size profile) pairs."""
    return [{"material": m, "lot_size": procedure, "lot_size_profile": code} for m, code in pairs]


def test_highest_forecast_sefi_gives_the_odd_code():
    """S1 has the largest forecast, so its code is the odd-week profile."""
    found = suggest_calendar_profiles(_calendar(("1", "EVEN-W2"), ("2", "ODD-W1"), ("3", "EVEN-W2")), FINIS)
    assert (found["status"], found["odd"], found["even"]) == ("suggested", "EVEN-W2", "ODD-W1")
    assert found["evidence"]["top_sefi"] == "S1" and found["evidence"]["sefis"] == 3


def test_not_exactly_two_codes_or_no_rows_is_not_guessed():
    """Three codes are ambiguous; no rows (or other procedures) mean absent."""
    three = suggest_calendar_profiles(_calendar(("1", "A-W1"), ("2", "B-W2"), ("3", "C-W3")), FINIS)
    assert three["status"] == "ambiguous" and three["odd"] is None and "found 3" in three["message"]
    assert suggest_calendar_profiles([], FINIS)["status"] == "absent"
    assert suggest_calendar_profiles(_calendar(("1", "ODD-W1"), procedure="EX"), FINIS)["status"] == "absent"


def test_unmatched_materials_and_conflicting_sefis_are_reported():
    """FINIs without a SEFI cannot rank; a SEFI with both codes is listed as conflicting."""
    unmatched = suggest_calendar_profiles(_calendar(("8", "ODD-W1"), ("9", "EVEN-W2")), FINIS)
    assert unmatched["status"] == "ambiguous"
    finis = FINIS + [{"material": "4", "sefi": "S1", "sefi_forecast_litres": "900"}]
    found = suggest_calendar_profiles(_calendar(("1", "ODD-W1"), ("4", "EVEN-W2"), ("2", "EVEN-W2")), finis)
    assert found["evidence"]["conflicting_sefis"] == ["S1"]
    assert (found["odd"], found["even"]) == ("EVEN-W2", "ODD-W1")  # S2 is the top unambiguous SEFI
