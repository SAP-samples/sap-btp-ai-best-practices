"""Readiness report: what SAP settings generation would produce, and what is missing.

The report is built before (or instead of) a real generation, so the UI can show
"Planning calendar: blocked, odd/even lot-size profiles missing" next to the
Generate button and the assistant can explain how to fix it.

With a point, the service runs a *dry* generation (``generate`` in memory,
nothing written), so the tab statuses here are exactly what a real generation
returns. Without a point, only the gaps known from the inputs alone are listed
(``requirements.input_gaps``), and the other tabs are ``unknown``.
"""

from __future__ import annotations

from typing import Any

from production_wheel.sap_settings.requirements import Gap

# Tab keys in the customer template's order.
TAB_ORDER = (
    "pv_check", "kpis", "cycle_stock", "planning_calendar",
    "recipe_combination", "apo_lot_size", "central_dc_lot_size", "runner_audit",
)


def last_summary_view(summary: dict[str, Any] | None, profile: dict[str, Any]) -> dict[str, Any] | None:
    """Describe the stored generation and whether the profile changed since.

    Args:
        summary: Stored ``sap_settings`` run summary, or ``None``.
        profile: Profile reference of the current inputs (``origin``, ``revision``).

    Returns:
        ``None`` before any generation; otherwise when, for which point and
        period, with which profile revision, and ``stale`` (True when the live
        profile has another revision than the one the settings were built from,
        or when they predate SAP settings in the plant profile).
    """
    if not summary:
        return None
    used = summary.get("profile_revision")
    stale = profile.get("origin") == "live" and used != profile.get("revision")
    return {
        "generated_at": summary.get("generated_at"),
        "period_label": summary.get("period_label"),
        "point_index": summary.get("point_index"),
        "decision_label": summary.get("decision_label"),
        "profile_revision": used,
        "stale": stale,
        "stale_reason": (
            "" if not stale else
            "generated before SAP settings moved into the plant profile" if used is None else
            f"the plant profile changed (revision {used} -> {profile.get('revision')}); regenerate to apply it"
        ),
    }


def readiness_report(
    *,
    run_id: str,
    point_index: int | None,
    point_source: str,
    selection: dict[str, Any] | None,
    profile: dict[str, Any],
    static_gaps: list[Gap],
    dry_tabs: dict[str, dict[str, Any]] | None,
    policy_values: dict[str, Any],
    policy_sources: dict[str, str],
    multi_tier_dc: bool,
    suggestions: list[dict[str, Any]],
    last_summary: dict[str, Any] | None,
    notes: list[str],
) -> dict[str, Any]:
    """Assemble the readiness payload returned by the API and the assistant tool.

    Args:
        run_id: Run reported on.
        point_index: Point the tab statuses refer to, or ``None``.
        point_source: ``selection`` (stored choice), ``requested`` or ``none``.
        selection: Stored decision selection, or ``None``.
        profile: Profile reference (see ``SapInputsMixin.sap_profile_policy``).
        static_gaps: Gaps known from the inputs alone.
        dry_tabs: Tab summaries of a dry generation, or ``None`` without a point.
        policy_values: ``{setting: value}`` of the rules in use (JSON form).
        policy_sources: ``{setting: source}``.
        multi_tier_dc: Whether the central-DC tab applies.
        suggestions: Data-derived values the user can adopt (may be empty).
        last_summary: Output of ``last_summary_view``.
        notes: Extra explanations (removed profile, run without profile).

    Returns:
        ``{run_id, available, point_index, point_source, selection,
        can_generate, profile, tabs, gaps, blocking_tabs, policy, suggestions,
        last_summary, notes}``. ``tabs`` maps every tab key, in template order,
        to ``{status, message, rows, gaps}``.
    """
    gaps_by_tab: dict[str, list[dict[str, Any]]] = {}
    for gap in static_gaps:
        gaps_by_tab.setdefault(gap.tab, []).append(gap.as_dict())
    tabs: dict[str, dict[str, Any]] = {}
    for name in TAB_ORDER:
        if dry_tabs is not None and name in dry_tabs:
            tab = dict(dry_tabs[name])
        elif gaps_by_tab.get(name):
            first = gaps_by_tab[name][0]
            tab = {"status": first["status"], "message": first["message"], "rows": None}
        elif name == "central_dc_lot_size" and not multi_tier_dc:
            tab = {"status": "not_applicable", "message": "", "rows": None}
        else:
            tab = {"status": "unknown", "message": "Select a decision scenario to check this tab.", "rows": None}
        tab["gaps"] = gaps_by_tab.get(name, [])
        tabs[name] = tab
    return {
        "run_id": run_id,
        "available": True,
        "point_index": point_index,
        "point_source": point_source,
        "selection": selection,
        "can_generate": bool(selection),
        "profile": profile,
        "tabs": tabs,
        "gaps": [gap.as_dict() for gap in static_gaps],
        "blocking_tabs": [name for name, tab in tabs.items() if tab["status"] == "blocked"],
        "policy": {name: {"value": policy_values.get(name), "source": policy_sources.get(name)}
                   for name in policy_values},
        "suggestions": suggestions,
        "last_summary": last_summary,
        "notes": notes,
    }
