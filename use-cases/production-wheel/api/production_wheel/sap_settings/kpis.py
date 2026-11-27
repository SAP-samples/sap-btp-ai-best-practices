"""RoC and changeover KPI tab, and the production-version consistency check.

Port of the template queries ``Wheel RoC``, ``Change over estimation`` and
``Check - Several Lot size in Subgroup``.

- **Wheel RoC.** Demand-weighted mean RoC of the wheel FINIs, 1 decimal.
- **Per SEFI subgroup.**
  - Batches per week = days per week / RoC.
  - Changeovers per batch = FINIs - 1.
  - Pack changes per batch = distinct packs - 1, using PCK codes when every
    FINI has one, else package volume, else unresolved.
  - Same-pack changes = changeovers per batch - pack changes per batch.
- **Weekly values** = batches per week times the per-batch counts. The site
  total sums the subgroups.
- **Line breakdown** (optional). Each subgroup's weekly values are split
  equally across its common filling lines (``unassigned`` when there are none).

The subgroup RoC is the wheel RoC ``R`` of its group (``production_wheel.wheel_roc``:
base RoC rounded to whole days, at least 1). The optimizer uses the same ``R`` for
its batches per week and ``J_CH``, so "changeovers per week" equals the point's
``J_CH`` for runs made with the wheel RoC rule.

**Runner multiples** (managed complexity, ``production_wheel.cycle_stock``). A
subgroup whose FINIs are filled only every ``k``-th batch keeps its batches per
week (the SEFI is still made every base cycle), but on average only
``sum(1 / k)`` FINIs are filled per batch:

- changeovers per batch = ``sum(1 / k) - 1``;
- the subgroup ``coverage_days`` is the demand-weighted mean of the FINIs'
  multiplied RoC (``base_coverage_days`` keeps the base RoC);
- ``single_changeovers_per_week`` keeps the single-RoC value for comparison;
- the pack / same-pack split is not estimated for such subgroups: their
  changeovers are reported as unclassified (pack basis "Runner multiples").
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from production_wheel.extraction.common import as_number, as_text
from production_wheel.wheel_roc import wheel_roc_days

_WEEKLY = (
    "batches_per_week",
    "changeovers_per_week",
    "single_changeovers_per_week",
    "pack_changeovers_per_week",
    "same_pack_changeovers_per_week",
    "volume_estimated_pack_changeovers_per_week",
    "unclassified_changeovers_per_week",
)


def wheel_roc(coverage: list[dict[str, Any]], demand: dict[str, float]) -> float | None:
    """Demand-weighted mean RoC (1 decimal) over FINIs with coverage and demand."""

    pairs = [(row["coverage_days"], demand.get(row["material"])) for row in coverage]
    pairs = [(days, litres) for days, litres in pairs if days is not None and litres]
    total = sum(litres for _, litres in pairs)
    return round(sum(days * litres for days, litres in pairs) / total, 1) if total else None


def _pack_basis(members: list[dict[str, Any]]) -> tuple[str, int | None]:
    """Return the pack basis and its distinct pack count for one subgroup."""

    packs = [as_text(member.get("pck_code")) for member in members]
    if all(packs):
        return "PCK", len(set(packs))
    volumes = [as_number(member.get("package_volume")) for member in members]
    if all(volume is not None for volume in volumes):
        return "Volume fallback", len(set(volumes))
    return "Unresolved", None


def _multiple_figures(
    members: list[dict[str, Any]], runner: dict[str, dict[str, Any]],
) -> tuple[float, float] | None:
    """Return ``(fills_per_batch, demand-weighted RoC)`` of a runner-multiple subgroup, else ``None``."""

    rows = [runner.get(as_text(member.get("material"))) or {} for member in members]
    if not any(row.get("cycle_stock_strategy") == "runner_multiple" for row in rows):
        return None
    # Changeovers per batch = sum(1 / k) - 1, confirmed by the customer (open question 2).
    fills = sum(1 / (row.get("runner_multiple") or 1) for row in rows)
    pairs = [(row.get("coverage_days"), as_number(member.get("forecast_litres_12m")))
             for member, row in zip(members, rows)]
    pairs = [(days, litres) for days, litres in pairs if days is not None and litres]
    total = sum(litres for _, litres in pairs)
    return fills, (sum(days * litres for days, litres in pairs) / total if total else None)


def changeover_rows(
    wheel_members: list[dict[str, Any]],
    plant: str,
    days_per_week: float,
    line_breakdown: bool = False,
    runner_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-subgroup, optional per-line and site-total changeover KPI rows.

    Args:
        wheel_members: ``members`` rows of the selected point with a subgroup.
        plant: Producing plant.
        days_per_week: Demand days per productive week.
        line_breakdown: Add per-filling-line rows.
        runner_rows: ``roc.runner_adjustments`` rows; subgroups whose rows use
            runner multiples get the multiple formulas (module docstring).
            ``None`` treats every subgroup as single RoC.

    Returns:
        Subgroup rows, optional filling-line rows and one site-total row.
    """

    runner = {row["material"]: row for row in runner_rows or []}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for member in wheel_members:
        groups[as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code"))].append(member)
    subgroup_rows = []
    for (sefi, subgroup), members in sorted(groups.items()):
        # Wheel RoC R (production_wheel.wheel_roc): whole days, at least 1.
        roc = wheel_roc_days(as_number(members[0].get("base_group_coverage_days")))
        count = len({as_text(member.get("material")) for member in members})
        basis, packs = _pack_basis(members)
        per_batch = max(count - 1, 0)
        single_per_batch = per_batch
        coverage = roc
        multiple = _multiple_figures(members, runner)
        if multiple is not None:
            fills, weighted = multiple
            per_batch, coverage, basis, packs = max(fills - 1, 0.0), weighted, "Runner multiples", None
        pack_per_batch = None if packs is None else max(packs - 1, 0)
        batches = days_per_week / roc if roc else 0.0
        resolved = basis not in ("Unresolved", "Runner multiples")
        subgroup_rows.append({
            "plant": plant, "level": "subgroup", "filling_line": None, "sefi": sefi, "subgroup": subgroup,
            "fini_count": count, "coverage_days": coverage, "pack_basis": basis,
            "cycle_stock_strategy": "runner_multiple" if multiple is not None else "single",
            "base_coverage_days": roc,
            "common_lines": as_text(members[0].get("common_lines")),
            "batches_per_week": batches,
            "changeovers_per_week": batches * per_batch,
            "single_changeovers_per_week": batches * single_per_batch,
            "pack_changeovers_per_week": batches * pack_per_batch if resolved else 0.0,
            "same_pack_changeovers_per_week": batches * (per_batch - pack_per_batch) if resolved else 0.0,
            "volume_estimated_pack_changeovers_per_week": batches * pack_per_batch if basis == "Volume fallback" else 0.0,
            "unclassified_changeovers_per_week": 0.0 if resolved else batches * per_batch,
        })
    rows = list(subgroup_rows)
    if line_breakdown:
        lines: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(_WEEKLY, 0.0))
        for row in subgroup_rows:
            common = [line for line in (row["common_lines"] or "").split("|") if line] or ["unassigned"]
            for line in common:
                for key in _WEEKLY:
                    lines[line][key] += row[key] / len(common)
        rows += [{"plant": plant, "level": "filling_line", "filling_line": line, **values}
                 for line, values in sorted(lines.items())]
    rows.append({"plant": plant, "level": "site_total", "filling_line": None,
                 **{key: sum(row[key] for row in subgroup_rows) for key in _WEEKLY}})
    return rows


def pv_conflict_rows(wheel_members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return SEFI subgroups whose FINIs use more than one selected production version."""

    pvs: dict[tuple[str, str], set] = defaultdict(set)
    for member in wheel_members:
        pvs[as_text(member.get("sefi")), as_text(member.get("proposed_subgroup_code"))].add(as_text(member.get("selected_pv")))
    return [{"sefi": sefi, "subgroup": subgroup, "selected_pvs": "|".join(sorted(v or "" for v in values))}
            for (sefi, subgroup), values in sorted(pvs.items()) if len(values) > 1]
