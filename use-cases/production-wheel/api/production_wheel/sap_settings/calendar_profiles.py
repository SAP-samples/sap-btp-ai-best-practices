"""Suggest the odd- and even-week lot-size profiles from an earlier wheel of the plant.

The planning calendar tab needs two plant customizing codes, one for odd weeks
and one for even weeks. Nothing in SAP standard says which codes a plant uses,
but older product-groups workbooks carry the previous wheel's SAP output table
"FINI Planning calendar at P-site" (dataset view ``planning_calendar_inform``),
where every two-weekly FINI already has one of the two codes.

The split rule is known: SEFIs are sorted by forecast, odd ranks are visited
first, so the SEFI with the largest forecast always gets the odd-week profile
(``calendar.calendar_order``). The suggestion therefore reads the two codes of
that table and calls "odd" the one of the highest-forecast SEFI.

This is an inference from an earlier wheel, so it is only *suggested*: the user
(or the assistant through an approval card) saves it into the plant profile,
and from then on it is the value every run of the profile uses.
"""

from __future__ import annotations

from typing import Any

from production_wheel.extraction.common import as_number, as_text
from production_wheel.sap_settings.calendar import calendar_order


def suggest_calendar_profiles(
    calendar_rows: list[dict[str, Any]],
    fini_rows: list[dict[str, Any]],
    lot_size_procedure: str = "PK",
) -> dict[str, Any]:
    """Infer the odd/even lot-size profiles from an earlier wheel's calendar table.

    Args:
        calendar_rows: ``planning_calendar_inform`` rows (``material``,
            ``lot_size_procedure`` or ``lot_size``, ``lot_size_profile``).
        fini_rows: ``fini_master`` rows of the same dataset (``material``,
            ``sefi``, ``sefi_forecast_litres``), to rank SEFIs by forecast.
        lot_size_procedure: Procedure of calendar-planned FINIs (profile
            setting ``calendar_lot_size``); rows with another procedure are ignored.

    Returns:
        ``{"status", "odd", "even", "message", "evidence"}``. ``status`` is
        ``suggested`` (exactly two codes, odd one identified), ``ambiguous``
        (not exactly two codes, or no forecast to rank by) or ``absent`` (the
        dataset has no calendar assignments). ``evidence`` holds the row and
        SEFI counts, the codes found and SEFIs whose FINIs carry both codes.
    """
    rows = [
        row for row in calendar_rows
        if as_text(row.get("lot_size_profile"))
        and (as_text(row.get("lot_size_procedure") or row.get("lot_size")) or lot_size_procedure) == lot_size_procedure
    ]
    evidence: dict[str, Any] = {"rows": len(rows), "distinct_profiles": sorted({as_text(r["lot_size_profile"]) for r in rows}),
                                "sefis": 0, "conflicting_sefis": []}
    if not rows:
        return {"status": "absent", "odd": None, "even": None, "evidence": evidence,
                "message": "The dataset has no earlier planning-calendar assignments to learn the codes from."}
    codes = evidence["distinct_profiles"]
    if len(codes) != 2:
        return {"status": "ambiguous", "odd": None, "even": None, "evidence": evidence,
                "message": f"Expected exactly two lot-size profiles in the earlier calendar, found {len(codes)}: {', '.join(codes)}."}

    sefi_of = {as_text(r.get("material")): (as_text(r.get("sefi")), as_number(r.get("sefi_forecast_litres")))
               for r in fini_rows if r.get("material") is not None}
    codes_by_sefi: dict[str, set[str]] = {}
    forecast_by_sefi: dict[str, float] = {}
    for row in rows:
        sefi, forecast = sefi_of.get(as_text(row.get("material")), (None, None))
        if not sefi:
            continue
        codes_by_sefi.setdefault(sefi, set()).add(as_text(row["lot_size_profile"]))
        if forecast is not None:
            forecast_by_sefi[sefi] = forecast
    evidence["sefis"] = len(codes_by_sefi)
    evidence["conflicting_sefis"] = sorted(sefi for sefi, found in codes_by_sefi.items() if len(found) > 1)
    ranked = sorted((sefi for sefi in codes_by_sefi if sefi in forecast_by_sefi and len(codes_by_sefi[sefi]) == 1),
                    key=lambda sefi: (-forecast_by_sefi[sefi], sefi))
    if not ranked or sum(forecast_by_sefi[sefi] for sefi in ranked) <= 0:
        return {"status": "ambiguous", "odd": None, "even": None, "evidence": evidence,
                "message": "The earlier calendar's FINIs could not be matched to SEFIs with a forecast."}

    order, _cut = calendar_order([forecast_by_sefi[sefi] for sefi in ranked])
    top = ranked[order[0]]
    odd = next(iter(codes_by_sefi[top]))
    even = next(code for code in codes if code != odd)
    evidence["top_sefi"] = top
    return {
        "status": "suggested", "odd": odd, "even": even, "evidence": evidence,
        "message": (f"From the earlier wheel: SEFI {top} has the largest forecast and used {odd}, "
                    f"so {odd} is the odd-week profile and {even} the even-week profile "
                    f"({evidence['rows']} FINI rows, {evidence['sefis']} SEFIs)."),
    }
