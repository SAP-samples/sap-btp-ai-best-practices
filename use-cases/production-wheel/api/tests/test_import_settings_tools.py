"""The Workspace Assistant proposes extraction settings for a previewed workbook.

The Datasets page sends the previewed workbook as ``pending_import``; the tools
read it, validate a settings patch and emit an Apply card without saving anything.
"""

import pytest

from app.agent.tools.workspace_tools import workspace_tools
from app.routers.workspace_chat import ChatMessage, _public_event
from app.workspace.repository import MemoryRepository
from app.workspace.service import WorkspaceService


def _pending(overrides=None):
    """Return a previewed workbook with three weekly codes and one undetermined value."""
    return {
        "preview_id": "preview-1",
        "primary_filename": "wheel.xlsx",
        "enrichment_filename": None,
        "plant": "TEST",
        "frequency_inventory": {
            "codes": [{"code": "01W", "label": "01W. Weekly", "rows": 4},
                      {"code": "02W", "label": "02W. Every two weeks", "rows": 3},
                      {"code": "16W", "label": "16W. Every four months", "rows": 5}],
            "unrecognized": [{"value": "999.Not possible to determine", "rows": 2}],
            "total_rows": 14,
        },
        "overrides": overrides or {},
    }


def _tools(pending):
    """Bind the tools to a service whose page session holds ``pending``."""
    service = WorkspaceService(MemoryRepository())
    if pending is not None:
        service.context("ctx", {"pending_import": pending})
    events = []
    return {tool.name: tool for tool in workspace_tools(service, "ctx", events.append)}, events


def test_pending_import_is_read_from_the_page_session():
    """Without a workbook the tool says so; with one it returns the inventory."""
    tools, _ = _tools(None)
    empty = tools["get_pending_import"].invoke({})
    assert empty["pending_import"] is None and "choose the production workbook" in empty["note"]
    tools, _ = _tools(_pending())
    found = tools["get_pending_import"].invoke({})
    assert [item["code"] for item in found["pending_import"]["frequency_inventory"]["codes"]] == ["01W", "02W", "16W"]
    assert found["defaults"]["modeled_frequencies"] == ["01W", "02W"]


def test_all_frequencies_proposal_keeps_keyword_and_warns_about_undetermined_rows():
    """'All frequencies' becomes the keyword for upload and the workbook codes for display."""
    tools, events = _tools(_pending())
    result = tools["propose_extraction_settings"].invoke(
        {"summary": "Model every frequency in the workbook.", "all_available_frequencies": True})
    assert result["status"] == "awaiting_user_approval"
    card = events[-1]
    assert card["type"] == "extraction_settings_proposal" and card["preview_id"] == "preview-1"
    assert card["overrides"] == {"modeled_frequencies": "all_available"}
    assert card["modeled_codes"] == ["01W", "02W", "16W"] and card["modeled_rows"] == 12
    assert card["warnings"] == [
        "2 FINI rows have no weekly frequency code (999.Not possible to determine) and stay out of scope."]


def test_proposals_patch_current_overrides_and_validate():
    """A later change keeps earlier overrides, resets on request and rejects bad values."""
    tools, events = _tools(_pending({"modeled_frequencies": "all_available"}))
    tools["propose_extraction_settings"].invoke({"summary": "Use 240 demand days.", "demand_days": 240})
    assert events[-1]["overrides"] == {"modeled_frequencies": "all_available", "demand_days": 240}
    tools["propose_extraction_settings"].invoke({"summary": "Back to default frequencies, model 04W too.",
        "modeled_frequencies": ["1w", "02W", "04W"]})
    assert events[-1]["overrides"]["modeled_frequencies"] == ["01W", "02W", "04W"]
    assert events[-1]["warnings"][0] == "No FINI in this workbook uses 04W."
    tools["propose_extraction_settings"].invoke({"summary": "Default frequencies.",
        "reset_to_default": ["modeled_frequencies"]})
    assert events[-1]["overrides"] == {}
    count = len(events)
    with pytest.raises(ValueError):
        tools["propose_extraction_settings"].invoke({"summary": "x", "canonical_factor": 1.5})
    with pytest.raises(ValueError, match="summary"):
        tools["propose_extraction_settings"].invoke({"summary": " ", "demand_days": 200})
    with pytest.raises(ValueError, match="Unknown"):
        tools["propose_extraction_settings"].invoke({"summary": "x", "reset_to_default": ["plant"]})
    assert len(events) == count


def test_proposal_requires_a_previewed_workbook():
    """No card is emitted before a production workbook is chosen."""
    tools, events = _tools(None)
    with pytest.raises(ValueError, match="Datasets page"):
        tools["propose_extraction_settings"].invoke({"summary": "All.", "all_available_frequencies": True})
    assert events == []


def test_chat_contract_accepts_pending_import_and_forwards_cards():
    """The chat body validates the preview and the NDJSON whitelist forwards the card."""
    body = ChatMessage(message="use all", context_id="ctx", pending_import=_pending())
    assert body.pending_import.frequency_inventory.codes[0].code == "01W"
    with pytest.raises(ValueError):
        ChatMessage(message="x", context_id="ctx", pending_import={**_pending(), "secret": 1})
    event = _public_event({"type": "extraction_settings_proposal", "proposal_id": "p", "preview_id": "v",
        "primary_filename": "f", "summary": "s", "overrides": {}, "modeled_codes": [], "modeled_rows": 0,
        "total_rows": 0, "all_available": False, "warnings": [], "secret": "no"})
    assert event["proposal_id"] == "p" and "secret" not in event
