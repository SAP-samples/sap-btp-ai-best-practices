"""Cycle stock tab: lot size, minimum lot size and rounding value per material and DC.

Port of the template queries ``DC Lot size inputs`` / ``DC Lot size with pallet``
/ ``DC Lot size without pallet``. One row per material and primary DC with
forecast at that DC:
- **Days.** Wheel FINIs use their (runner-adjusted) RoC raised to cover at
  least one full pallet. Other materials use the ABC cycle days of their class
  at that DC.
- **Lot size.** Days are capped and snapped (``normalized_days``), then mapped
  to the SAP lot-size key.
- **Minimum lot size and rounding value.** In pieces, from the plant policy
  rules per ABC class.
- **Without pallet.** Rows without a minimum lot size (no pallet data, or rule
  ``none``) form the "without pallet" output and carry no minimum or rounding.
"""

from __future__ import annotations

import math
from typing import Any

from production_wheel.extraction.common import as_number, as_text
from production_wheel.sap_settings.policy import class_value
from production_wheel.sap_settings.roc import normalized_days, pallet_adjusted_roc, sap_lot_size_code


def half_pallet(pallet: float | None, layer: float | None) -> float | None:
    """Half a pallet rounded up to whole layers, in pieces (template ``fxHalfPallet``)."""

    if pallet is None or layer is None or layer <= 0:
        return None
    return math.ceil((pallet / 2) / layer) * layer


def _min_lot(rule: tuple[str, float | None], pallet, layer, litres_per_piece) -> float | None:
    """Minimum lot size in pieces for one policy rule."""

    name, number = rule
    if name == "pallet":
        return pallet
    if name == "half_pallet":
        return half_pallet(pallet, layer)
    if name == "one_piece":
        return 1
    if name == "litres":
        return math.ceil(number / litres_per_piece) if litres_per_piece and litres_per_piece > 0 else None
    return None


def _rounding(rule: tuple[str, float | None], pallet, layer, days, maximum) -> float | None:
    """Rounding value in pieces for one policy rule."""

    name = rule[0]
    if name == "layer_at_max_days":
        return layer if days == maximum else pallet
    return {"pallet": pallet, "layer": layer, "half_pallet": half_pallet(pallet, layer), "one_piece": 1}.get(name)


def dc_lot_size_rows(
    wheel: dict[str, dict[str, Any]],
    network: list[dict[str, Any]],
    abc_classes: list[dict[str, Any]],
    pallets: list[dict[str, Any]],
    policy: dict[str, Any],
    max_days: float = 200,
) -> tuple[list[dict[str, Any]], int]:
    """Build cycle-stock rows for every material with forecast at a primary DC.

    Args:
        wheel: Material -> ``{coverage_days, pallets_per_subgroup,
            sales_scenario, sales_network_scenario}`` for FINIs of the selected
            wheel.
        network: ``dc_network`` rows of the plant.
        abc_classes: ``dc_abc_classes`` rows (material, dc, abc_class).
        pallets: ``dc_pallet_quantities`` rows (pieces per pallet and layer,
            litres per piece).
        policy: Resolved plant policy.
        max_days: Cap of cycle-stock days, ``CycleStockPolicy.max_days`` of the
            run's plant profile (template default 200).

    Returns:
        ``(rows, dropped)``. ``dropped`` counts material/DC pairs with neither a
        wheel RoC nor an ABC class (the template drops them silently).
    """

    maximum = max_days
    abc = {(as_text(row["material"]), as_text(row["dc"])): as_text(row["abc_class"]) for row in abc_classes}
    pieces = {as_text(row["material"]): row for row in pallets}
    rows, dropped, seen = [], 0, set()
    for link in network:
        material, dc = as_text(link.get("material")), as_text(link.get("dc"))
        if str(link.get("has_forecast") or "").upper() != "YES" or (material, dc) in seen:
            continue
        seen.add((material, dc))
        abc_class = abc.get((material, dc))
        fini = wheel.get(material)
        if fini is not None:
            days = pallet_adjusted_roc(fini["coverage_days"], fini["pallets_per_subgroup"])
            source = "wheel"
        elif abc_class is not None:
            days = class_value(policy, "abc_cycle_days", abc_class)
            source = "abc"
        else:
            dropped += 1
            continue
        cycle_days = normalized_days(days, maximum)
        quantities = pieces.get(material, {})
        pallet = as_number(quantities.get("pallet_qty"))
        layer = as_number(quantities.get("layer_qty"))
        minimum = _min_lot(class_value(policy, "dc_min_lot", abc_class), pallet, layer,
                           as_number(quantities.get("package_volume")))
        rows.append({
            "plant": dc,
            "material": material,
            "lot_size_code": sap_lot_size_code(cycle_days),
            "min_lot_size_at_dc": minimum,
            "rounding_value": _rounding(class_value(policy, "dc_rounding", abc_class), pallet, layer,
                                        cycle_days, maximum) if minimum is not None else None,
            "defined_in_wheel": "YES" if fini is not None else "NO",
            "sales_scenario": (fini or {}).get("sales_scenario"),
            "sales_network_scenario": (fini or {}).get("sales_network_scenario"),
            "min_lot_identified": minimum is not None,
            "cycle_days": cycle_days,
            "cycle_source": source,
            "abc_class": abc_class,
            "coverage_days": (fini or {}).get("coverage_days"),
            "pallets_per_subgroup": (fini or {}).get("pallets_per_subgroup"),
            "coverage_days_pallet_adjusted": days if source == "wheel" else None,
            "status": "ready" if sap_lot_size_code(cycle_days) else "unmapped_days",
        })
    rows.sort(key=lambda row: (row["defined_in_wheel"] != "YES", row["plant"] or "", row["material"] or ""))
    return rows, dropped
