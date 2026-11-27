"""Assistant tools for decision scenarios and SAP settings generation.

After a run completes, people pick one scenario (Balanced, changeover- or
inventory-oriented) and turn it into SAP settings. These tools let the
assistant do that on request and explain what is missing:

- ``get_decision_scenarios``: the labelled key points and the stored selection;
- ``select_decision_scenario``: store the chosen point (direct write, on request);
- ``get_sap_settings_status``: what generation would produce, what is missing,
  the rules in use with their source, values learnt from earlier wheels;
- ``generate_sap_settings``: generate for the stored selection (direct write).

SAP settings *rules* (calendar codes, DC lot sizes, ...) live in the plant
profile; the assistant changes them only through ``propose_profile_settings``
with a ``sap_settings`` patch, which the user approves on a card.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.workspace.sap_settings_inputs import PERIOD_LABEL_PATTERN, current_quarter_label

# Keys of a decision point worth showing the model (no evidence blobs).
POINT_KEYS = ("point_index", "decision_label", "decision_class", "coverage_days", "j_ch",
              "sefi_batches_per_week", "group_count")


def sap_tools(service: Any, context_id: str, emit: Callable[[dict[str, Any]], None]) -> list[BaseTool]:
    """Create the decision-scenario and SAP settings tools for one conversation.

    Args:
        service: Shared workspace service (decision and SAP settings mixins).
        context_id: Conversation selection identifier (holds run_id, point_index).
        emit: Callback that forwards UI events (``decision_selected``,
            ``sap_settings_generated``) to the browser.

    Returns:
        Four LangChain tools; run_id defaults to the run selected on the page.
    """

    def selected_run(run_id: str | None) -> str:
        """Use the explicit run or the one selected in this conversation."""
        run_id = run_id or service.context(context_id).get("run_id")
        if not run_id:
            raise ValueError("Select or name a completed run first (list_runs).")
        return run_id

    @tool
    def get_decision_scenarios(run_id: str | None = None) -> dict:
        """List the run's decision scenarios: Balanced (B, the knee), changeover-oriented
        CO1..CO3 (fewer changeovers, more stock) and inventory-oriented INV1..INV3, plus
        the stored selection that SAP settings use. Points without a label are other
        usable frontier points (also selectable). Defaults to the selected run."""
        scenarios = service.decision_scenarios(selected_run(run_id))
        return {
            "run_id": scenarios.get("run_id"),
            "knee_point_index": scenarios.get("knee_point_index"),
            "selection": scenarios.get("selection"),
            "key_points": [{k: p.get(k) for k in POINT_KEYS if k in p} for p in scenarios.get("points", []) if p.get("decision_label")],
            "other_point_indexes": [p["point_index"] for p in scenarios.get("points", []) if not p.get("decision_label")],
            "excluded": scenarios.get("excluded", []),
        }

    @tool
    def select_decision_scenario(run_id: str | None = None, point_index: int | None = None,
                                 decision_label: str | None = None) -> dict:
        """Store the scenario the user chose for SAP settings (direct write).

        Call only when the user asked to select a scenario or to generate SAP
        settings for one. Give point_index, or decision_label such as "B", "CO2",
        "INV1". A new selection deletes SAP settings generated for the previous
        one: when cleared_previous_sap_settings would be true, confirm with the
        user first (get_sap_settings_status shows last_summary)."""
        run_id = selected_run(run_id)
        if point_index is None:
            if not decision_label:
                raise ValueError("Give point_index or decision_label (B, CO1..CO3, INV1..INV3).")
            scenarios = service.decision_scenarios(run_id)
            matches = [p for p in scenarios.get("points", [])
                       if str(p.get("decision_label") or "").upper() == decision_label.strip().upper()]
            if len(matches) != 1:
                labels = [p["decision_label"] for p in scenarios.get("points", []) if p.get("decision_label")]
                raise ValueError(f"Unknown decision label {decision_label!r}; available: {', '.join(labels)}")
            point_index = matches[0]["point_index"]
        had_settings = bool(service.sap_settings(run_id))
        selection = service.select_decision(run_id, int(point_index))
        service.context(context_id, {"run_id": run_id, "point_index": selection["point_index"]})
        emit({"type": "decision_selected", "run_id": run_id, "selection": selection})
        return {"selection": selection, "cleared_previous_sap_settings": had_settings,
                "next": "Check get_sap_settings_status, then generate_sap_settings when the user wants them."}

    @tool
    def get_sap_settings_status(run_id: str | None = None, point_index: int | None = None) -> dict:
        """Check SAP settings for a run before or after generating them.

        Returns, per tab (cycle stock, planning calendar, recipe combination, APO
        lot size, ...), the status a generation would give for point_index
        (default: the stored selection; none: only input gaps), each gap with its
        fix, blocking_tabs, the plant profile whose rules apply (origin live /
        frozen / none, revision), every rule with its value and source (default,
        profile, missing, dataset), suggestions learnt from an earlier wheel of
        the plant, and last_summary (stale=true: the profile changed since the
        settings were generated). Load the sap-settings skill to explain them."""
        return service.sap_settings_readiness(selected_run(run_id), point_index)

    @tool
    def generate_sap_settings(run_id: str | None = None, period_label: str | None = None) -> dict:
        """Generate SAP settings for the run's stored selection (direct write).

        Call only when the user asked for it. period_label is the planning period
        written into the APO comment, e.g. 26Q4; default: the current quarter.
        Blocked tabs are reported, never invented. The user downloads the XLSX
        from the SAP settings card of the workspace."""
        run_id = selected_run(run_id)
        label = (period_label or current_quarter_label()).strip()
        if not re.fullmatch(PERIOD_LABEL_PATTERN, label):
            raise ValueError("period_label may only use letters, digits, space, dot, underscore and dash (max 32).")
        summary = service.generate_sap_settings(run_id, label)
        emit({"type": "sap_settings_generated", "run_id": run_id, "point_index": summary.get("point_index"),
              "generated_at": summary.get("generated_at")})
        return {
            "run_id": run_id,
            "point_index": summary.get("point_index"),
            "decision_label": summary.get("decision_label"),
            "period_label": summary.get("period_label"),
            "tabs": summary.get("tabs"),
            "wheel_roc_days": summary.get("wheel_roc_days"),
            "kpi_changeovers_per_week": summary.get("kpi_changeovers_per_week"),
            "point_j_ch": summary.get("point_j_ch"),
            "kpi_matches_point_j_ch": summary.get("kpi_matches_point_j_ch"),
            "profile": summary.get("sap_settings_source"),
            "download": "The XLSX is downloaded from the SAP settings card (Download SAP settings XLSX).",
        }

    return [get_decision_scenarios, select_decision_scenario, get_sap_settings_status, generate_sap_settings]
