"""Deterministic quantity / unit-of-measure conversion to the material's sales unit.

Implements requirement R4 ("if the extracted packing unit differs from the material's unit,
apply a conversion"): purchase orders often state litres or gallons while ECC sells the
material per piece (pail, can). The conversion uses the master data, never the LLM:

1. unit text -> SAP unit through the alias Z-table (``lít`` -> ``L``, ``gallon`` -> ``GAL``);
2. unit already valid for the material (base unit, MARM alternative unit or sales unit) ->
   keep it and convert to the sales unit with the MARM factors;
3. volume unit -> divide by the fill volume ``MARA.VOLUM`` (e.g. 4 400 L / 17.6 L = 250 pails).
   A result within 0.02 of a whole piece (or 0.1 % of large counts) is exact; otherwise the nominal pack size (from the
   sales code, e.g. "20L") is offered as an alternative and the result is marked inexact so the
   order goes to review.

Example::

    convert_quantity(material, 4400, "Lít", None, aliases)
    # {"sales_qty": 250.0, "sales_uom": "PC", "method": "volume", "exact": True, ...}
"""

from __future__ import annotations

from typing import Any

from .matching import pack_litres, strip_accents

LITRES_PER_UNIT = {"L": 1.0, "ML": 0.001, "GAL": 3.785, "QT": 0.946, "HG": 1.893}
# A converted count is "exact" when it is within 0.02 of a whole piece, or within 0.1 % of the total
# for large counts (absorbs fill volumes printed with two decimals over hundreds of pails).
WHOLE_NUMBER_ABS_TOLERANCE = 0.02
WHOLE_NUMBER_REL_TOLERANCE = 0.001


def resolve_unit(uom_text: str | None, aliases: dict[str, str]) -> str | None:
    """Map a unit as written on a document to an SAP unit.

    Args:
        uom_text: Unit text, e.g. "Lít", "Ltrs", "PC", "Gallon-".
        aliases: Alias map {lower-case alias: SAP unit} from ZUOM_ALIAS_POAGENT.

    Returns:
        SAP unit, or None when unknown.
    """
    if not uom_text:
        return None
    raw = uom_text.strip()
    candidates = [raw.lower(), strip_accents(raw), raw.lower().rstrip(".-"), strip_accents(raw).rstrip(".-")]
    for candidate in candidates:
        if candidate in aliases:
            return aliases[candidate]
    upper = raw.upper()
    if upper in set(aliases.values()) | set(LITRES_PER_UNIT):
        return upper
    return None


def _volume_litres(material: dict[str, Any]) -> float | None:
    """Fill volume of one piece in litres (MARA.VOLUM in VOLEH)."""
    volume = material.get("VOLUM")
    if not volume:
        return None
    return float(volume) * LITRES_PER_UNIT.get(material.get("VOLEH") or "L", 1.0)


def _near_whole(value: float) -> tuple[bool, int]:
    """Return (is within tolerance of a whole number, rounded value)."""
    rounded = round(value)
    tolerance = max(WHOLE_NUMBER_ABS_TOLERANCE, WHOLE_NUMBER_REL_TOLERANCE * rounded)
    return rounded > 0 and abs(value - rounded) <= tolerance, rounded


def valid_units(material: dict[str, Any]) -> set[str]:
    """Units an order line may use for this material: base unit, MARM units, sales units."""
    units = {material.get("MEINS")}
    units |= {row["MEINH"] for row in material.get("UNITS", [])}
    units |= {row["VRKME"] for row in material.get("SALES", []) if row.get("VRKME")}
    return {unit for unit in units if unit}


def convert_quantity(
    material: dict[str, Any],
    quantity: float,
    uom_text: str | None,
    pack_text: str | None,
    aliases: dict[str, str],
) -> dict[str, Any]:
    """Convert an order-line quantity to the material's sales unit.

    Args:
        material: Material dict from ``master_data.load_materials`` (MEINS, VOLUM, VOLEH, UNITS, SALES, BISMT).
        quantity: Quantity as stated on the document.
        uom_text: Unit as stated (free text).
        pack_text: Pack size text on the line, if any (e.g. "20L").
        aliases: Unit alias map.

    Returns:
        Dict with ``sales_qty``, ``sales_uom``, ``method`` (same_unit | marm | volume | none),
        ``exact`` (bool), ``alternatives`` (list of {sales_qty, basis}), ``note`` and, on failure,
        ``error`` (UNKNOWN_UNIT | UOM_NOT_CONVERTIBLE | INVALID_QUANTITY).
    """
    base_unit = material.get("MEINS") or "PC"
    sales_units = [row["VRKME"] for row in material.get("SALES", []) if row.get("VRKME")]
    sales_unit = sales_units[0] if sales_units else base_unit
    result: dict[str, Any] = {"sales_qty": None, "sales_uom": sales_unit, "method": "none", "exact": False,
                              "alternatives": [], "note": None}
    if quantity is None or quantity <= 0:
        return {**result, "error": "INVALID_QUANTITY", "note": "quantity must be positive"}
    unit = resolve_unit(uom_text, aliases)
    if unit is None and uom_text and uom_text.strip().upper() in valid_units(material):
        unit = uom_text.strip().upper()          # SAP unit code valid for this material (MARM)
    if unit is None and not uom_text:
        unit = base_unit                          # no unit printed: quantity counts base units
    if unit is None:
        return {**result, "error": "UNKNOWN_UNIT", "note": f"unit {uom_text!r} is not known"}

    factors = {row["MEINH"]: float(row["UMREZ"]) / float(row["UMREN"]) for row in material.get("UNITS", [])}
    factors.setdefault(base_unit, 1.0)
    if unit in valid_units(material) and unit in factors and sales_unit in factors:
        sales_qty = quantity * factors[unit] / factors[sales_unit]
        exact, rounded = _near_whole(sales_qty)
        method = "same_unit" if unit == sales_unit else "marm"
        return {**result, "sales_qty": float(rounded) if exact else round(sales_qty, 3), "method": method,
                "exact": exact or unit == sales_unit, "note": None if exact or unit == sales_unit
                else f"{quantity} {unit} is not a whole number of {sales_unit}"}

    if unit in LITRES_PER_UNIT:
        litres = quantity * LITRES_PER_UNIT[unit]
        fill = _volume_litres(material)
        nominal = pack_litres(material.get("BISMT")) or pack_litres(pack_text)
        if not fill and not nominal:
            return {**result, "error": "UOM_NOT_CONVERTIBLE",
                    "note": "material has no volume; quantity cannot be converted from a volume unit"}
        packs = litres / (fill or nominal)
        exact, rounded = _near_whole(packs)
        alternatives = []
        if nominal and fill and abs(nominal - fill) > 1e-6:
            nominal_ok, nominal_rounded = _near_whole(litres / nominal)
            if nominal_ok:
                alternatives.append({"sales_qty": float(nominal_rounded), "basis": f"nominal pack {nominal} L"})
        basis = f"fill volume {fill} L" if fill else f"nominal pack {nominal} L"
        return {**result, "sales_qty": float(rounded) if exact else round(packs, 3), "method": "volume",
                "exact": exact, "alternatives": alternatives,
                "note": f"{quantity} {unit} = {round(litres, 3)} L / {basis}"
                        + ("" if exact else " is not a whole number of pieces")}

    return {**result, "error": "UOM_NOT_CONVERTIBLE",
            "note": f"unit {unit} cannot be converted to {sales_unit} for this material"}
