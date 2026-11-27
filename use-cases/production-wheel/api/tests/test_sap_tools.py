"""Assistant tools for decision scenarios and SAP settings, called directly (no LLM).

Run with:
    cd api && ../.venv/bin/python -m pytest -q tests/test_sap_tools.py
"""
import asyncio
from pathlib import Path

import pytest

from app.agent.skills import SkillLoader
from app.agent.tools import sap_tools as sap_tools_module
from app.agent.tools.workspace_tools import workspace_tools
from app.routers.workspace_chat import _public_event
from test_sap_settings_api import _add_calendar_dataset, _client


def _tools():
    """Service with one completed run, its tools, and the emitted UI events."""
    _, service, repo = _client()
    events = []
    tools = {t.name: t for t in workspace_tools(service, "ctx", events.append)}
    service.context("ctx", {"run_id": "r"})
    return service, repo, tools, events


def test_scenarios_and_selection_by_label():
    """The balanced point is found by its label; the selection reaches the page and the context."""
    service, _, tools, events = _tools()
    scenarios = tools["get_decision_scenarios"].invoke({})
    assert scenarios["key_points"][0]["decision_label"] == "B" and scenarios["selection"] is None
    result = tools["select_decision_scenario"].invoke({"decision_label": "b"})
    assert result["selection"]["point_index"] == 1 and result["cleared_previous_sap_settings"] is False
    assert events[-1]["type"] == "decision_selected" and events[-1]["run_id"] == "r"
    assert service.context("ctx")["point_index"] == 1
    with pytest.raises(ValueError, match="available: B"):
        tools["select_decision_scenario"].invoke({"decision_label": "CO9"})
    with pytest.raises(ValueError, match="point_index or decision_label"):
        tools["select_decision_scenario"].invoke({})


def test_status_reports_gaps_and_suggestions_then_generation_uses_the_quarter(monkeypatch):
    """Status explains the blocked calendar with a learnt suggestion; generate defaults the period."""
    service, repo, tools, events = _tools()
    _add_calendar_dataset(repo, "old")
    status = tools["get_sap_settings_status"].invoke({})
    assert status["blocking_tabs"] == ["planning_calendar"]
    assert status["suggestions"][0]["values"] == {"calendar_odd_profile": "ODD-W1", "calendar_even_profile": "EVEN-W2"}
    with pytest.raises(ValueError, match="select a decision scenario"):
        tools["generate_sap_settings"].invoke({})

    tools["select_decision_scenario"].invoke({"point_index": 1})
    monkeypatch.setattr(sap_tools_module, "current_quarter_label", lambda: "26Q4")
    generated = tools["generate_sap_settings"].invoke({})
    assert generated["period_label"] == "26Q4" and generated["tabs"]["planning_calendar"]["status"] == "blocked"
    assert events[-1]["type"] == "sap_settings_generated"
    with pytest.raises(ValueError, match="period_label"):
        tools["generate_sap_settings"].invoke({"period_label": "bad;label"})
    assert tools["select_decision_scenario"].invoke({"point_index": 1})["cleared_previous_sap_settings"] is True


def test_sap_rule_change_goes_through_the_approval_card():
    """A partial sap_settings patch is validated, shown as a card, and keeps other rules on approval."""
    service, _, tools, events = _tools()
    profile_id = tools["get_sap_settings_status"].invoke({})["profile"]["profile_id"]
    before = service.get_plant_profile(profile_id)
    result = tools["propose_profile_settings"].invoke({
        "summary": "Use the plant's odd/even calendar profiles",
        "settings_patch": {"sap_settings": {"calendar_odd_profile": "ODD-W1", "calendar_even_profile": "EVEN-W2"}},
        "profile_id": profile_id})
    assert result["status"] == "awaiting_user_approval" and events[-1]["type"] == "profile_proposal"
    assert service.get_plant_profile(profile_id)["revision"] == before["revision"]
    saved = service.apply_plant_profile_change(profile_id, before["revision"], events[-1]["change"])
    assert saved["settings"]["sap_settings"]["calendar_lot_size"] == "PK"
    assert tools["get_sap_settings_status"].invoke({})["blocking_tabs"] == []
    with pytest.raises(ValueError):
        tools["propose_profile_settings"].invoke({"summary": "bad", "profile_id": profile_id,
                                                  "settings_patch": {"sap_settings": {"calendar_odd_profile": "NODASH"}}})


def test_completed_wait_carries_the_sap_offer():
    """wait_for_run on a finished run adds the offer with blocked inputs."""
    _, _, tools, _ = _tools()
    result = asyncio.run(tools["wait_for_run"].ainvoke({"run_id": "r"}))
    offer = result["sap_settings_offer"]
    assert offer["available"] is True and offer["blocked_inputs"][0]["tab"] == "planning_calendar"
    assert "Offer" in offer["hint"]


def test_ui_events_are_whitelisted():
    """Only the documented keys of the two new events reach the browser."""
    selected = _public_event({"type": "decision_selected", "run_id": "r", "selection": {"point_index": 1}, "secret": 1})
    assert selected == {"type": "decision_selected", "run_id": "r", "selection": {"point_index": 1}}
    generated = _public_event({"type": "sap_settings_generated", "run_id": "r", "point_index": 1,
                               "generated_at": "t", "tabs": {}})
    assert set(generated) == {"type", "run_id", "point_index", "generated_at"}


def test_sap_settings_skill_is_discoverable():
    """The on-demand skill is found by the loader with a description."""
    root = Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_skills"
    summaries = {skill.name: skill for skill in SkillLoader(root).scan()}
    assert "sap-settings" in summaries and "blocked" in summaries["sap-settings"].description
