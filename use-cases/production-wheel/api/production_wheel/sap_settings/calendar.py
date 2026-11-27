"""FINI planning calendar tab: split two-weekly SEFIs between odd and even weeks.

Port of the template queries ``FINI Planning calendar at P-site`` and its
``RESET`` variant:

1. Take the wheel FINIs whose source production frequency is two-weekly.
2. List their SEFIs by forecast, largest first, ranked 1..n.
3. Visit the SEFIs odd ranks first (1, 3, 5, ..., then 2, 4, ...), so big
   SEFIs alternate between the halves. Accumulate forecast shares.
4. SEFIs up to and including the one that crosses 50% get the odd-week
   lot-size profile; the rest get the even-week profile. Each FINI inherits its
   SEFI's profile.
5. Every FINI with another frequency is reset: lot-size procedure for
   lot-for-lot, calendar and profile blanked via ticket.

The lot-size profiles are plant customizing codes (not standard SAP), so they
come from plant policy; without them the tab is blocked.
"""

from __future__ import annotations

from typing import Any

from production_wheel.extraction.common import as_number, as_text


def calendar_order(forecasts: list[float]) -> tuple[list[int], int]:
    """Return the visiting order and cut position for forecasts sorted largest first.

    Args:
        forecasts: SEFI forecasts already sorted in descending order.

    Returns:
        ``(order, cut)``: indexes into ``forecasts`` (odd ranks first), and the
        last position in ``order`` assigned to odd weeks (-1 when none).
    """

    total = sum(forecasts)
    order = sorted(range(len(forecasts)), key=lambda index: (index % 2, index))
    cumulative, cut = 0.0, -1
    for position, index in enumerate(order):
        share = forecasts[index] / total
        if cumulative <= 0.5 < cumulative + share:
            cut = position
        cumulative += share
    return order, cut


def planning_calendar_rows(
    members: list[dict[str, Any]],
    policy: dict[str, Any],
    plant: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """Build assigned and reset planning-calendar rows for the selected point.

    Args:
        members: Every ``members`` row of the point (wheel and non-wheel FINIs).
        policy: Resolved plant policy.
        plant: Producing plant.

    Returns:
        ``(rows, status, message)``. Status is ``ready`` or ``blocked``; blocked
        tabs return no rows.
    """

    from production_wheel.sap_settings.requirements import calendar_policy_gap

    gap = calendar_policy_gap(policy)
    if gap:
        return [], gap.status, gap.message
    odd, even = policy["calendar_odd_profile"], policy["calendar_even_profile"]
    frequency = policy["calendar_frequency_code"]
    calendar_fini = [
        member for member in members
        if member.get("proposed_subgroup_code") and as_text(member.get("production_frequency_code")) == frequency
    ]
    forecast_by_sefi: dict[str, set] = {}
    for member in calendar_fini:
        forecast_by_sefi.setdefault(as_text(member.get("sefi")), set()).add(as_number(member.get("sefi_forecast_litres")))
    conflicts = [sefi for sefi, values in forecast_by_sefi.items() if len(values) != 1 or None in values]
    if conflicts:
        return [], "blocked", f"SEFIs with a missing or ambiguous forecast: {', '.join(sorted(conflicts))}"
    sefis = sorted(forecast_by_sefi, key=lambda sefi: (-next(iter(forecast_by_sefi[sefi])), sefi))
    forecasts = [next(iter(forecast_by_sefi[sefi])) for sefi in sefis]
    if sefis and sum(forecasts) <= 0:
        return [], "blocked", "Two-weekly SEFIs have no positive forecast."
    order, cut = calendar_order(forecasts) if sefis else ([], -1)
    profile_by_sefi, cumulative = {}, 0.0
    for position, index in enumerate(order):
        cumulative += forecasts[index] / sum(forecasts)
        profile_by_sefi[sefis[index]] = (odd if position <= cut else even, position + 1, cumulative)
    rows = []
    for member in calendar_fini:
        profile, rank, share = profile_by_sefi[as_text(member.get("sefi"))]
        rows.append({
            "action": "assign", "plant": plant, "material": as_text(member.get("material")),
            "sefi": as_text(member.get("sefi")), "lot_size_procedure": policy["calendar_lot_size"],
            "planning_calendar": profile.partition("-")[2], "lot_size_profile": profile,
            "sefi_forecast_litres": as_number(member.get("sefi_forecast_litres")),
            "sefi_order": rank, "cumulative_share": share,
        })
    reset_text = policy["calendar_reset_text"]
    for member in members:
        if as_text(member.get("production_frequency_code")) == frequency:
            continue
        rows.append({
            "action": "reset", "plant": plant, "material": as_text(member.get("material")),
            "sefi": as_text(member.get("sefi")), "lot_size_procedure": policy["calendar_reset_lot_size"],
            "planning_calendar": reset_text, "lot_size_profile": reset_text,
            "sefi_forecast_litres": None, "sefi_order": None, "cumulative_share": None,
        })
    return rows, "ready", ""
