"""Which finished products each production version mode includes in a run.

A run's ``pv_mode`` changes its population, not only how lot sizes are chosen:

- FIXED (default of a new draft) models the FINIs whose ``model_status`` is
  "modeled": each keeps its source "Selected PV".
- OPTIMIZED models the FINIs whose ``optimized_pv_model_status`` is "modeled":
  the optimizer chooses a production version of the SEFI. A FINI without a
  usable "Selected PV" can be included here.

Before a launch, the planner should know which products the chosen mode leaves
out while the other mode would include them. ``pv_mode_differences`` computes
that from the snapshot's ``fini_master`` rows and the draft's plant/SEFI scope;
``validate_draft`` returns it so both the Validate panel and the agent show it.

Example::

    scope = pv_mode_differences(fini_rows, request["scope"], "FIXED")
    scope["only_in_other_mode"]  # FINIs OPTIMIZED would add
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Mapping

# Product keys quoted in the warning text before "and N more".
LISTED_KEYS = 10

# Status field that decides admission in each production version mode.
STATUS_FIELD = {"FIXED": "model_status", "OPTIMIZED": "optimized_pv_model_status"}


def _in_scope(row: Mapping[str, Any], scope: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether a row's plant/SEFI block matches the run scope.

    An empty scope is the whole snapshot; an unset plant or SEFI is a wildcard,
    as in the optimizer's own scope resolution.
    """
    entries = list(scope or [])
    if not entries:
        return True
    plant, sefi = str(row.get("plant")), str(row.get("sefi"))
    return any(
        (entry.get("plant") in (None, "") or str(entry.get("plant")) == plant)
        and (entry.get("sefi") in (None, "") or str(entry.get("sefi")) == sefi)
        for entry in entries
    )


def _product(row: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Summarize one product for the result: key, SEFI and why the mode excludes it."""
    reason_field = "exclusion_reason" if mode == "FIXED" else "optimized_pv_exclusion_reason"
    return {
        "fini_plant_key": row.get("fini_plant_key"),
        "material": row.get("material"),
        "sefi": row.get("sefi"),
        "reason_excluded": row.get(reason_field),
    }


def _keys_text(products: list[dict[str, Any]]) -> str:
    """List the first product keys, then "and N more"."""
    keys = [str(item["fini_plant_key"]) for item in products[:LISTED_KEYS]]
    more = len(products) - len(keys)
    return ", ".join(keys) + (f" and {more} more" if more > 0 else "")


def pv_mode_differences(
    fini_rows: Iterable[Mapping[str, Any]],
    scope: Iterable[Mapping[str, Any]] | None,
    pv_mode: str,
) -> dict[str, Any]:
    """Compare the run population of the chosen production version mode with the other one.

    Args:
        fini_rows: The snapshot's ``fini_master`` rows (plant, sefi,
            fini_plant_key, model_status, optimized_pv_model_status, reasons).
        scope: The draft request's plant/SEFI scope (list of dicts); empty for all.
        pv_mode: The draft's ``pv_mode``, "FIXED" or "OPTIMIZED".

    Returns:
        Dict with ``pv_mode``, ``other_mode``, ``modeled_count`` and
        ``other_mode_modeled_count`` (in scope), ``only_in_other_mode`` (products
        the other mode would add) and ``only_in_this_mode`` (products the other
        mode would drop), each with key, material, SEFI and the exclusion reason,
        plus ``warnings``: plain sentences for the Validate panel and the agent.
        Only products the chosen mode leaves out produce a warning; products the
        other mode would drop are reported in ``only_in_this_mode`` as context.
    """
    mode = str(pv_mode or "FIXED").upper()
    other = "OPTIMIZED" if mode == "FIXED" else "FIXED"
    rows = [row for row in fini_rows if _in_scope(row, scope)]
    # The optimizer falls back to the FIXED population when no FINI is modeled with
    # OPTIMIZED (CanonicalInputs.members_for); mirror that here.
    optimized_available = any(row.get(STATUS_FIELD["OPTIMIZED"]) == "modeled" for row in fini_rows)

    def modeled(row: Mapping[str, Any], which: str) -> bool:
        """Return whether a row is in the population of one mode."""
        field = STATUS_FIELD["FIXED"] if which == "OPTIMIZED" and not optimized_available else STATUS_FIELD[which]
        return row.get(field) == "modeled"

    only_other = [_product(row, mode) for row in rows if modeled(row, other) and not modeled(row, mode)]
    only_this = [_product(row, other) for row in rows if modeled(row, mode) and not modeled(row, other)]
    warnings = []
    if only_other:
        reasons = collections.Counter(item["reason_excluded"] or "unknown" for item in only_other)
        reason_text = ", ".join(f"{reason} ({count})" for reason, count in reasons.most_common())
        warnings.append(
            f"Production version {mode} leaves out {len(only_other)} FINI(s) that {other} would include: "
            f"{_keys_text(only_other)} (reason with {mode}: {reason_text}). "
            f"Choose {other} to include them, or keep {mode} to leave them out."
        )
    return {
        "pv_mode": mode,
        "other_mode": other,
        "modeled_count": sum(modeled(row, mode) for row in rows),
        "other_mode_modeled_count": sum(modeled(row, other) for row in rows),
        "only_in_other_mode": only_other,
        "only_in_this_mode": only_this,
        "warnings": warnings,
    }
