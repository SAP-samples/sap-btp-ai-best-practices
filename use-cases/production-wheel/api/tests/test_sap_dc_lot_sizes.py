"""Cycle stock tab: lot-size key, minimum lot and rounding per material and DC.

Run with:
    cd api && ../.venv/bin/python -m pytest tests/test_sap_dc_lot_sizes.py -q
"""

from __future__ import annotations

from production_wheel.sap_settings.dc_lot_sizes import dc_lot_size_rows, half_pallet
from production_wheel.sap_settings.policy import resolve_policy, validate_rows


def _rows():
    """One plant: three wheel FINIs and four other materials at one DC."""

    wheel = {
        "F1": {"coverage_days": 20, "pallets_per_subgroup": 0.5},
        "F2": {"coverage_days": 350, "pallets_per_subgroup": 1.0},
        "F3": {"coverage_days": 155, "pallets_per_subgroup": 2.0},
    }
    network = [
        {"material": m, "dc": "D1", "has_forecast": "YES"} for m in ("F1", "F2", "F3", "N1", "N2", "N3", "N4")
    ] + [{"material": "F1", "dc": "D2", "has_forecast": "NO"}]
    abc = [{"material": "N1", "dc": "D1", "abc_class": "A"}, {"material": "N3", "dc": "D1", "abc_class": "C"},
           {"material": "N4", "dc": "D1", "abc_class": "B"}]
    pallets = [
        {"material": "F1", "pallet_qty": 44, "layer_qty": 11, "package_volume": 10},
        {"material": "F2", "pallet_qty": 44, "layer_qty": 11, "package_volume": 10},
        {"material": "F3", "pallet_qty": 40, "layer_qty": 10, "package_volume": 5},
        {"material": "N1", "pallet_qty": 40, "layer_qty": 10, "package_volume": 5},
        {"material": "N3", "pallet_qty": 100, "layer_qty": 20, "package_volume": 2.5},
    ]
    policy = resolve_policy(validate_rows([
        {"plant": "P1", "setting": "dc_min_lot", "abc_class": "C", "value": "litres", "number": 300}]), "P1")
    return dc_lot_size_rows(wheel, network, abc, pallets, policy)


def test_lot_size_keys_minimum_lots_and_rounding() -> None:
    """Wheel FINIs use pallet-adjusted RoC; others ABC days; site rules come from policy."""

    rows, dropped = _rows()
    by_material = {row["material"]: row for row in rows}
    assert [row["material"] for row in rows] == ["F1", "F2", "F3", "N1", "N3", "N4"]
    assert dropped == 1
    assert (by_material["F1"]["lot_size_code"], by_material["F1"]["min_lot_size_at_dc"],
            by_material["F1"]["rounding_value"]) == ("40", 44, 44)
    assert (by_material["F2"]["cycle_days"], by_material["F2"]["lot_size_code"],
            by_material["F2"]["rounding_value"]) == (200, "9L", 11)
    assert by_material["F3"]["lot_size_code"] == "9F"
    assert (by_material["N1"]["lot_size_code"], by_material["N1"]["cycle_source"]) == ("05", "abc")
    assert (by_material["N3"]["lot_size_code"], by_material["N3"]["min_lot_size_at_dc"]) == ("20", 120)
    assert by_material["F1"]["defined_in_wheel"] == "YES" and by_material["N1"]["defined_in_wheel"] == "NO"


def test_rows_without_pallet_data_go_to_the_without_pallet_output() -> None:
    """No pieces per pallet: no minimum lot and no rounding value."""

    rows, _ = _rows()
    without = [row for row in rows if not row["min_lot_identified"]]
    assert [row["material"] for row in without] == ["N4"]
    assert without[0]["lot_size_code"] == "10"
    assert without[0]["rounding_value"] is None


def test_half_pallet_rounds_up_to_whole_layers() -> None:
    """Half of 44 pieces in layers of 10 is 30 pieces."""

    assert half_pallet(44, 10) == 30
    assert half_pallet(44, None) is None
