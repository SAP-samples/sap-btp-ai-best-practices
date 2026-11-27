"""SAP settings rules as a plant-profile section: defaults, validation and round trips.

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_sap_profile_policy.py
"""
import pytest

from production_wheel.sap_settings.policy import resolve_policy, validate_rows
from production_wheel.sap_settings.profile_policy import SapSettingsPolicy


def test_defaults_match_the_policy_defaults():
    """An untouched profile section resolves exactly like an empty rule set."""
    assert SapSettingsPolicy().to_resolved() == resolve_policy([], "P1")


def test_resolved_rules_round_trip_through_the_profile_section():
    """'*' rows, plant rows and class overrides survive from_resolved -> to_resolved."""
    rows = validate_rows([
        {"plant": "*", "setting": "dc_rounding", "value": "layer"},
        {"plant": "P1", "setting": "dc_rounding", "abc_class": "A", "value": "pallet"},
        {"plant": "P1", "setting": "dc_min_lot", "abc_class": "C", "value": "litres", "number": 300},
        {"plant": "P1", "setting": "abc_cycle_days", "abc_class": "*", "number": 15},
        {"plant": "P1", "setting": "calendar_odd_profile", "value": "ODD-W1"},
        {"plant": "P1", "setting": "calendar_even_profile", "value": "EVEN-W2"},
        {"plant": "P1", "setting": "kpi_line_breakdown", "value": "yes"},
    ])
    resolved = resolve_policy(rows, "P1")
    policy, notes = SapSettingsPolicy.from_resolved(resolved)
    assert notes == []
    assert policy.to_resolved() == resolved
    assert policy.dc_min_lot.C.number == 300 and policy.dc_min_lot.A is None
    assert policy.abc_cycle_days.unclassified == 15
    assert policy.kpi_line_breakdown is True


@pytest.mark.parametrize("value, message", [
    ({"dc_min_lot": {"all_classes": {"rule": "litres"}}}, "litres"),
    ({"dc_rounding": {"A": {"rule": "box"}}}, "rule"),
    ({"calendar_odd_profile": "ODDW1"}, "<profile>-<calendar>"),
    ({"calendar_odd_profile": "ODD-W1", "calendar_even_profile": "ODD-W1"}, "must differ"),
    ({"calendar_lot_size": "  "}, "calendar_lot_size"),
    ({"unknown_setting": 1}, "unknown_setting"),
])
def test_invalid_values_are_rejected(value, message):
    """Each malformed value names what is wrong."""
    with pytest.raises(ValueError, match=message):
        SapSettingsPolicy.model_validate(value)


def test_blank_optional_codes_mean_not_set_and_numbers_drop_for_non_litres_rules():
    """Blank calendar codes become None; a pallet rule carries no number."""
    policy = SapSettingsPolicy.model_validate({
        "calendar_odd_profile": "  ", "dc_min_lot": {"all_classes": {"rule": "pallet", "number": 300}}})
    assert policy.calendar_odd_profile is None
    assert policy.dc_min_lot.all_classes.number is None


def test_lenient_reading_keeps_valid_fields_when_one_is_invalid():
    """The migration path never loses a valid code because of another bad value."""
    resolved = resolve_policy([], "P1")
    resolved["calendar_odd_profile"] = "ODD-W1"
    resolved["calendar_even_profile"] = "EVEN-W2"
    resolved["calendar_frequency_code"] = "  "
    policy, notes = SapSettingsPolicy.from_resolved(resolved, strict=False)
    assert policy.calendar_odd_profile == "ODD-W1" and policy.calendar_even_profile == "EVEN-W2"
    assert policy.calendar_frequency_code == "02W"
    assert any(note.startswith("calendar_frequency_code") for note in notes)
    with pytest.raises(ValueError):
        SapSettingsPolicy.from_resolved(resolved, strict=True)


def test_lenient_reading_drops_equal_calendar_codes():
    """Equal odd/even codes cannot both be kept; both are left empty with a note."""
    resolved = resolve_policy([], "P1")
    resolved["calendar_odd_profile"] = resolved["calendar_even_profile"] = "SAME-W1"
    policy, notes = SapSettingsPolicy.from_resolved(resolved, strict=False)
    assert policy.calendar_odd_profile is None and policy.calendar_even_profile is None
    assert notes and "both left empty" in notes[-1]
