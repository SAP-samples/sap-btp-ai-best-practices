"""Review findings for finished products (FINIs) the import could not admit.

Every FINI is classified twice during extraction (see
``primary_workbook.classify_fini_admission``):

- ``model_status``: admission when each FINI keeps its "Selected PV" (production
  version mode FIXED, the run default).
- ``optimized_pv_model_status``: admission when the optimizer may choose any
  production version of the SEFI (production version mode OPTIMIZED).

A FINI excluded in FIXED but modeled in OPTIMIZED is not a plain data-quality
exclusion: the planner decides per run whether to include it. It is therefore
reported under its own check, ``FINI_OPTIMIZED_PV_ONLY``, instead of
``FINI_DATA_QUALITY``.

Example::

    issues.extend(exclusion_issues(rows))
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

# FINIs excluded in every production version mode.
DATA_QUALITY_RULE = "FINI_DATA_QUALITY"
# FINIs excluded with FIXED production versions but modeled with OPTIMIZED ones.
OPTIMIZED_PV_ONLY_RULE = "FINI_OPTIMIZED_PV_ONLY"


def exclusion_issues(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one review warning per FINI excluded from the default (FIXED) scope.

    Args:
        rows: Classified ``fini_master`` rows with ``fini_plant_key``,
            ``model_status``, ``exclusion_reason`` and
            ``optimized_pv_model_status``.

    Returns:
        Validation issues in the extraction shape (``rule_id``, ``severity``,
        ``entity_key``, ``reason``, ``observed``, ``action``). Out-of-scope rows
        (frequency settings) produce no issue here; they are scope decisions.
    """
    issues = []
    for row in rows:
        if row.get("model_status") != "excluded":
            continue
        optimized_only = row.get("optimized_pv_model_status") == "modeled"
        issues.append(
            {
                "rule_id": OPTIMIZED_PV_ONLY_RULE if optimized_only else DATA_QUALITY_RULE,
                "severity": "warning",
                "entity_key": row["fini_plant_key"],
                "reason": row.get("exclusion_reason"),
                "observed": (
                    "modeled only with production version OPTIMIZED"
                    if optimized_only
                    else "excluded with production version FIXED and OPTIMIZED"
                ),
                "action": (
                    "Choose production version OPTIMIZED to include this FINI, or fill in its "
                    "Selected PV in the workbook to include it with FIXED"
                    if optimized_only
                    else "Review source evidence before admitting this FINI"
                ),
            }
        )
    return issues
