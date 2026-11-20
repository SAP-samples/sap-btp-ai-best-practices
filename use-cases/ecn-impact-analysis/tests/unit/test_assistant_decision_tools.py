"""Check the assistant tools that report review status, stored analysis and data quality.

Run with:
    PYTHONPATH=api .venv/bin/python -m pytest tests/unit/test_assistant_decision_tools.py -q
"""

from types import SimpleNamespace

import pytest

from app.ecn.assistant.tools import create_tools

CURRENT = "a" * 64
OLDER = "b" * 64

OVERVIEW = {
    "assessment_id": "selected",
    "evidence_hash": CURRENT,
    "version": "ecn-snapshot-v2",
    "questions": [{
        "code": "PHASE_OUT_STILL_USED", "question": "Is the old part still used?", "why": "BOM still lists it.",
        "resolution": "Confirm the end date.", "affected_count": 2,
        "subjects": [{"number": "QKX4638810"}, {"number": "QKX8914160"}, {"number": "QKX4638810"}],
    }],
    "bom_changes": {"lines": [], "materials": []},
}


def review(outcome, evidence_hash=CURRENT, reviewer="Ana", comments=""):
    """Build one stored review row as WorkspaceRepository.reviews returns it."""
    return {"reviewer": reviewer, "outcome": outcome, "comments": comments, "created_at": "2026-09-29T10:00:00",
            "evidence_hash": evidence_hash, "snapshot_version": "ecn-snapshot-v2"}


def tools_for(reviews=(), analysis=None, bundle=None, overview=OVERVIEW):
    """Create the selected-ECN tools over in-memory fakes, keyed by tool name."""
    snapshots = SimpleNamespace(overview=lambda identity: overview)
    ecns = SimpleNamespace(
        get_ecn_detail=lambda identity: SimpleNamespace(analysis=analysis),
        get_evidence_bundle=lambda identity: bundle,
    )
    workspace = SimpleNamespace(reviews=lambda identity: list(reviews))
    return {t.name: t for t in create_tools(snapshots, ecns, None, "selected", set(), [], set(), workspace=workspace)}


@pytest.mark.asyncio
async def test_review_status_reports_no_decision_yet():
    """An unreviewed ECN must say so instead of letting the model assume approval."""
    result = (await tools_for()["get_review_status"].ainvoke({}))["result"]
    assert result["current_decision"] == "NONE" and result["latest"] is None and result["total"] == 0


@pytest.mark.asyncio
async def test_review_status_flags_decision_made_on_older_evidence():
    """The newest review wins, and a decision on older evidence is marked as such."""
    tools = tools_for([review("CHANGES_REQUESTED", comments="Fix dates"), review("ACCEPTED", evidence_hash=OLDER)])
    result = (await tools["get_review_status"].ainvoke({}))["result"]
    assert result["current_decision"] == "ACCEPTED"
    assert result["latest"]["applies_to_current_snapshot"] is False
    assert [r["outcome"] for r in result["history"]] == ["ACCEPTED", "CHANGES_REQUESTED"]
    assert result["acceptance"]["blocked"] is False and result["acceptance"]["sap_write"] == "simulated"


@pytest.mark.asyncio
async def test_impact_analysis_absent_and_stale():
    """Missing analyses are explicit, and analyses on older evidence are marked not current."""
    assert (await tools_for()["get_impact_analysis"].ainvoke({}))["result"]["available"] is False
    stored = {"evidence_hash": OLDER, "overall_impact": "HIGH", "completeness": "COMPLETE", "model_name": "gpt-5.4",
              "created_at": "2026-09-01", "impact_factors": {"manufacturing": {"severity": "HIGH", "assessment": "Cut-over at PL10."}},
              "sections": {"executive_summary": {"status": "SUFFICIENT", "assessment_markdown": "Summary"}},
              "additional_next_steps": {"manufacturing": ["Plan PL10 cut-over"]}}
    analysis = SimpleNamespace(model_dump=lambda mode: stored)
    result = (await tools_for(analysis=analysis)["get_impact_analysis"].ainvoke({}))["result"]
    assert result["available"] is True and result["is_current"] is False
    assert result["impact_factors"]["manufacturing"]["severity"] == "HIGH"


@pytest.mark.asyncio
async def test_data_quality_questions_name_their_parts():
    """Readiness questions must carry exact part numbers so answers never say 'the two parts'."""
    result = (await tools_for()["get_data_quality_issues"].ainvoke({}))["result"]
    [question] = result["readiness_questions"]
    assert question["parts"] == ["QKX4638810", "QKX8914160"]
    assert result["has_next"] is False


class ReviewStore:
    """In-memory stand-in for WorkspaceRepository's append-only review table."""

    def __init__(self):
        """Start with no saved reviews."""
        self.rows: dict[str, dict] = {}

    def save_review(self, values):
        """Store a row once per REVIEW_ID, like the idempotent repository."""
        self.rows.setdefault(values["REVIEW_ID"], values)

    def reviews(self, identity):
        """Return rows in the repository's lowercase-key shape."""
        return [{k.lower(): v for k, v in row.items()} for row in self.rows.values()]


def decision_tools(history=(), store=None, state=None):
    """Create tools with a writable review store, chat history and per-turn state."""
    snapshots = SimpleNamespace(overview=lambda identity: OVERVIEW)
    ecns = SimpleNamespace(get_ecn_detail=lambda identity: None, get_evidence_bundle=lambda identity: None)
    return {t.name: t for t in create_tools(
        snapshots, ecns, None, "selected", set(), [], set(),
        workspace=store if store is not None else ReviewStore(),
        history=list(history), actor="hashed", review_state=state if state is not None else {})}


COMMENT = "Confirm the PL10 cut-over date for QKX463881A before acceptance."
SHOWN = [{"role": "user", "content": "request changes, reviewer Ana"},
         {"role": "assistant", "content": f"**Decision:** Reject\n**Reviewer:** Ana\n> {COMMENT}\n\nReply yes to save."}]


@pytest.mark.asyncio
async def test_draft_requires_change_description_and_warns_on_open_questions():
    """Drafts follow the review form rules and surface open readiness questions on Accept."""
    tools = decision_tools()
    refused = (await tools["draft_review_decision"].ainvoke(
        {"outcome": "CHANGES_REQUESTED", "reviewer": "Ana", "comments": " "}))["result"]
    assert refused["drafted"] is False and refused["error_code"] == "COMMENTS_REQUIRED"
    accepted = (await tools["draft_review_decision"].ainvoke(
        {"outcome": "ACCEPTED", "reviewer": "Ana", "comments": ""}))["result"]
    assert accepted["drafted"] is True and accepted["confirmation_required"] is True
    assert any("readiness questions" in warning for warning in accepted["warnings"])


@pytest.mark.asyncio
async def test_submit_refuses_a_draft_the_user_never_saw():
    """The server refuses to save comment text missing from the previous assistant answer."""
    store = ReviewStore()
    result = (await decision_tools(history=[], store=store)["submit_review_decision"].ainvoke(
        {"outcome": "CHANGES_REQUESTED", "reviewer": "Ana", "comments": COMMENT}))["result"]
    assert result["saved"] is False and result["error_code"] == "DRAFT_NOT_SHOWN" and not store.rows


@pytest.mark.asyncio
async def test_submit_saves_shown_draft_once_per_answer():
    """A shown draft saves once with a content-derived ID; a second save in the answer is refused."""
    store, state = ReviewStore(), {}
    tools = decision_tools(history=SHOWN, store=store, state=state)
    args = {"outcome": "CHANGES_REQUESTED", "reviewer": "Ana", "comments": COMMENT}
    saved = (await tools["submit_review_decision"].ainvoke(args))["result"]
    assert saved["saved"] is True and saved["sap_updated"] is False and state == {"saved": True}
    [row] = store.rows.values()
    assert row["OUTCOME"] == "CHANGES_REQUESTED" and row["COMMENTS"] == COMMENT and row["ACTOR_ID"] == "hashed"
    again = (await tools["submit_review_decision"].ainvoke(args))["result"]
    assert again["error_code"] == "ALREADY_SAVED"
    # A replay in a later answer reuses the same request ID, so nothing is duplicated.
    await decision_tools(history=SHOWN, store=store)["submit_review_decision"].ainvoke(args)
    assert len(store.rows) == 1


def test_review_decision_skill_is_catalogued_and_loadable():
    """The acceptance procedure is a separate on-demand skill, not part of the base prompt."""
    from pathlib import Path

    from app.ecn.assistant.skills import SkillLoader

    import app.ecn.assistant.runtime as runtime
    loader = SkillLoader(Path(runtime.__file__).parent / "agent_skills", 16000)
    assert "review-decision" in {skill.name for skill in loader.scan()}
    assert "submit_review_decision" in loader.load(["review-decision"])
    assert "review-decision" in runtime.BASE_PROMPT


def test_record_review_rejects_a_stale_snapshot():
    """The shared write path used by the route and the tool refuses outdated evidence."""
    from app.ecn.errors import PersistenceStateError
    from app.ecn.services.reviews import record_review

    store = ReviewStore()
    with pytest.raises(PersistenceStateError) as error:
        record_review(SimpleNamespace(overview=lambda identity: OVERVIEW), store, "selected",
                      request_id="r1", reviewer="Ana", outcome="ACCEPTED", comments="", actor="hashed",
                      evidence_hash=OLDER, snapshot_version="ecn-snapshot-v2")
    assert error.value.code == "REVIEW_SNAPSHOT_MISMATCH" and not store.rows


@pytest.mark.asyncio
async def test_accept_draft_is_refused_while_a_new_part_has_no_price():
    """A blocked ECN cannot even be drafted as accepted; the reason is one line for the user."""
    blocked = OVERVIEW | {"bom_changes": {"lines": [], "materials": [
        {"number": "NEW1", "description": "Decal", "role": "INCOMING", "status": "NEW_PART", "date": "2026-03-13"}]}}
    tools = tools_for(overview=blocked)
    status = (await tools["get_review_status"].ainvoke({}))["result"]
    assert status["acceptance"]["blocked"] is True and status["acceptance"]["missing_price"] == ["NEW1"]
    draft = (await tools["draft_review_decision"].ainvoke({"outcome": "ACCEPTED", "reviewer": "Ana"}))["result"]
    assert draft["drafted"] is False and draft["error_code"] == "ECN_ACCEPT_BLOCKED"
    assert draft["message"] == "Can't accept: 1 new part has no price."
    sent_back = (await tools["draft_review_decision"].ainvoke(
        {"outcome": "CHANGES_REQUESTED", "reviewer": "Ana", "comments": "Add a vendor price for NEW1."}))["result"]
    assert sent_back["drafted"] is True
