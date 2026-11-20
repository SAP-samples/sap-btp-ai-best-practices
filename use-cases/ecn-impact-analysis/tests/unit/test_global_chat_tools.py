"""Check the cross-ECN chat: ECN resolution, the `ecn` argument and ECN matching tools.

Run with:
    PYTHONPATH=api .venv/bin/python -m pytest tests/unit/test_global_chat_tools.py -q
"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ecn.analysis_models import ImpactSeverity
from app.ecn.api_models import AnalysisFilterState
from app.ecn.assistant.catalogue import EcnCatalogue
from app.ecn.assistant.global_tools import create_global_tools
from app.ecn.assistant.tools import create_tools
from app.ecn.evidence.part_index import build_part_index
from app.ecn.rdf import GraphFact, GraphFactPage

NEW_1, OLD_1, ID_2, ID_3 = ("11111111-0000-0000-0000-00000000000" + str(n) for n in range(4))


def item(row_id, number, uploaded_day, impact=None, plants=None, review=None):
    """Build one ECN list row with the fields ``catalogue_entry`` reads."""
    return SimpleNamespace(
        ecn_row_id=row_id, display_ecn_number=number, title=f"{number} title", description_summary="Q900V release",
        resolution_date=None, uploaded_at=datetime(2026, 9, uploaded_day), affected_plants=plants,
        overall_impact=impact, review_outcome=review, analysis_state=AnalysisFilterState.READY,
        counts=SimpleNamespace(affected_item_count=3, ebom_match_count=2, mbom_match_count=1, warning_count=0),
    )


class Repository:
    """ECN list fake: newest upload first, ECN1 imported twice."""

    ITEMS = [
        item(ID_2, "ECN2", 30, ImpactSeverity.MEDIUM, ["PL10 Plant A"], "ACCEPTED"),
        item(NEW_1, "ECN1", 29, ImpactSeverity.HIGH, ["PK01 Plant B"]),
        item(ID_3, "ECN3", 28, ImpactSeverity.LOW, None),
        item(OLD_1, "ECN1", 1, ImpactSeverity.LOW, ["PK01 Plant B"]),
    ]

    def __init__(self):
        """Count list reads so per-turn caching can be checked."""
        self.reads = 0

    def list_ecns(self, query):
        """Return one page of the fixed list."""
        self.reads += 1
        start = (query.page - 1) * query.page_size
        return SimpleNamespace(items=self.ITEMS[start:start + query.page_size], total_pages=1)

    def get_ecn_detail(self, row_id):
        """Every ECN has a stored analysis with one factor."""
        analysis = SimpleNamespace(model_dump=lambda mode: {
            "evidence_hash": "h", "overall_impact": "HIGH", "completeness": "COMPLETE",
            "impact_factors": {"manufacturing": {"severity": "HIGH", "assessment": "long text"}},
            "sections": {"what_changes": {"status": "OK", "assessment_markdown": "details"}},
        })
        return SimpleNamespace(analysis=analysis)


def part(roles, changes=(), plants=()):
    """Build one part-index entry."""
    return {"description": "Bracket", "roles": list(roles), "changes": list(changes), "assemblies": ["Q900VV001"],
            "plants": list(plants), "engineering_dates": [], "manufacturing_dates": [], "end_dates": []}


INDEXES = {
    NEW_1: {"ecn_number": "ECN1", "parts": {"ACX1": part(["INCOMING"], ["ADDED"], ["PK01"]), "ACX9": part(["ECN_OBJECT"])}},
    OLD_1: {"ecn_number": "ECN1", "parts": {"ACX1": part(["ECN_OBJECT"])}},
    ID_2: {"ecn_number": "ECN2", "parts": {"ACX1": part(["OUTGOING"], ["REPLACED"], ["PL10"])}},
}


class Snapshots:
    """Snapshot fake: ECN3's snapshot cannot be built."""

    def part_index(self, row_id):
        """Return the fixed index, or fail for ECN3."""
        if row_id == ID_3:
            raise RuntimeError("snapshot unavailable")
        return INDEXES[row_id]

    def get(self, row_id):
        """Return one BOM change line per ECN."""
        return {"assessment_date": "2026-03-15", "bom_changes": {
            "counts": {"ADDED": 1}, "materials": [],
            "lines": [{"assembly": "Q900VV001", "change": "ADDED", "incoming": {"number": f"P-{row_id[-1]}"}}]}}

    def overview(self, row_id):
        """Return a minimal overview."""
        return {"evidence_hash": "h", "version": "v", "questions": [], "bom_changes": {"lines": [], "materials": []}}


def tools(open_ecn=None, rdf=None):
    """Create the chat tools of one turn, keyed by name, plus the shared catalogue."""
    repository, snapshots = Repository(), Snapshots()
    catalogue = EcnCatalogue(repository, open_ecn)
    workspace = SimpleNamespace(reviews=lambda row_id: [])
    bound = [*create_global_tools(catalogue, snapshots),
             *create_tools(snapshots, repository, rdf, open_ecn, set(), [], set(), workspace=workspace, catalogue=catalogue)]
    return {t.name: t for t in bound}, catalogue, repository


@pytest.mark.asyncio
async def test_catalogue_resolves_the_newest_occurrence_and_reports_problems():
    """ECN numbers map to the newest upload; unknown, missing and too many ECNs are explained."""
    catalogue = EcnCatalogue(Repository())
    targets, problem = await catalogue.targets(["ecn1"], 5)
    assert problem is None and targets[0]["assessment_id"] == NEW_1
    assert (await catalogue.targets(["ECN7"], 5))[1]["error_code"] == "ECN_NOT_FOUND"
    assert (await catalogue.targets([], 5))[1]["error_code"] == "ECN_REQUIRED"
    assert (await catalogue.targets(["ECN1", "ECN2"], 1))[1]["error_code"] == "TOO_MANY_ECNS"
    assert catalogue.ecns.reads == 1
    open_targets, _ = await EcnCatalogue(None, OLD_1).targets("", 5)
    assert open_targets == [{"assessment_id": OLD_1, "ecn_number": None}]
    assert (await EcnCatalogue(None, OLD_1).targets("ECN2", 5))[1]["error_code"] == "ECN_SCOPE_FIXED"


@pytest.mark.asyncio
async def test_list_ecns_filters_by_impact_plant_and_review_without_internal_ids():
    """The catalogue answers 'which ECNs are high impact / at PL10 / not reviewed'."""
    named, catalogue, _ = tools()
    high = (await named["list_ecns"].ainvoke({"impact": "HIGH"}))["result"]
    assert [row["ecn_number"] for row in high["items"]] == ["ECN1"]
    assert "assessment_id" not in high["items"][0]
    pl10 = (await named["list_ecns"].ainvoke({"plant": "pl10"}))["result"]
    assert [row["ecn_number"] for row in pl10["items"]] == ["ECN2"]
    open_items = (await named["list_ecns"].ainvoke({"review": "NOT_REVIEWED"}))["result"]
    assert [row["ecn_number"] for row in open_items["items"]] == ["ECN1", "ECN3"]
    assert catalogue.consulted[NEW_1] == "ECN1"


@pytest.mark.asyncio
async def test_find_ecns_with_part_reports_roles_and_unchecked_ecns():
    """A part found in two ECNs is reported with its role in each; a broken ECN is not a non-match."""
    named, _, _ = tools()
    result = (await named["find_ecns_with_part"].ainvoke({"number": "acx1"}))["result"]
    assert {(row["ecn_number"], tuple(row["roles"])) for row in result["items"]} == {
        ("ECN2", ("OUTGOING",)), ("ECN1", ("INCOMING",))}
    assert result["unavailable_ecns"] == ["ECN3"] and result["ecns_searched"] == 2


@pytest.mark.asyncio
async def test_find_related_ecns_uses_the_open_ecn_and_excludes_itself():
    """'Which ECNs share materials with this ECN?' ranks other ECNs, never the same ECN number."""
    named, catalogue, _ = tools(open_ecn=NEW_1)
    result = (await named["find_related_ecns"].ainvoke({}))["result"]
    assert result["source_ecn"] == "ECN1"
    assert [row["ecn_number"] for row in result["items"]] == ["ECN2"]
    assert result["items"][0]["shared_parts"][0] == {
        "number": "ACX1", "description": "Bracket", "roles_in_source": ["INCOMING"], "roles_in_other": ["OUTGOING"]}
    assert catalogue.consulted == {ID_2: "ECN2"}


@pytest.mark.asyncio
async def test_detail_tools_read_several_named_ecns_with_a_split_budget():
    """One call compares ECNs; each result names its ECN and stays inside its budget share."""
    named, catalogue, _ = tools()
    result = await named["list_bom_changes"].ainvoke({"ecn": ["ECN1", "ECN2"]})
    assert [entry["ecn_number"] for entry in result["results"]] == ["ECN1", "ECN2"]
    assert result["results"][0]["result"]["items"][0]["incoming"]["number"] == "P-0"
    assert set(catalogue.consulted.values()) == {"ECN1", "ECN2"}
    with pytest.raises(ValidationError, match="at most 5"):
        await named["list_bom_changes"].ainvoke({"ecn": ["ECN1", "ECN2", "ECN3", "A", "B", "C"]})


@pytest.mark.asyncio
async def test_global_mode_requires_ecn_numbers_and_has_no_review_tools():
    """Without an open ECN, detail tools ask for ECN numbers and nothing can be saved."""
    named, _, _ = tools()
    result = (await named["get_ecn_overview"].ainvoke({}))["result"]
    assert result["error_code"] == "ECN_REQUIRED"
    assert "draft_review_decision" not in named and "submit_review_decision" not in named
    scoped, _, _ = tools(open_ecn=NEW_1)
    assert "submit_review_decision" in scoped


@pytest.mark.asyncio
async def test_impact_comparison_returns_compact_rows():
    """Comparing impact across ECNs keeps severities and drops section text."""
    named, _, _ = tools()
    result = await named["get_impact_analysis"].ainvoke({"ecn": ["ECN1", "ECN2"]})
    first = result["results"][0]["result"]
    assert first["overall_impact"] == "HIGH" and first["impact_factors"] == {"manufacturing": "HIGH"}
    assert "sections" not in first


class Graph:
    """Graph fake: ECN2's projection is not current."""

    def __init__(self):
        """Record graph reads."""
        self.calls = []

    def current_ids(self, row_ids):
        """Report every ECN except ECN2 as current."""
        return [i for i in row_ids if i != ID_2], [i for i in row_ids if i == ID_2]

    def find_facts(self, scope, **kwargs):
        """Return one fact owned by the newest ECN1."""
        self.calls.append(scope)
        fact = GraphFact("urn:ecn-impact:node:x", (), "Requester", "Ada", None, "/Requester", "a", "h", "v", ecn_row_id=NEW_1)
        return GraphFactPage((fact,), page=1, page_size=10, total_matches=1, has_next=False)


@pytest.mark.asyncio
async def test_find_ecn_facts_searches_every_current_ecn_when_none_is_open():
    """One graph query covers all ready ECNs; facts name their ECN; pending graphs are listed."""
    graph = Graph()
    named, catalogue, _ = tools(rdf=graph)
    result = (await named["find_ecn_facts"].ainvoke({"query": "Ada", "search_in": "value"}))["result"]
    assert graph.calls == [[NEW_1, ID_3]]
    assert result["items"][0]["ecn_number"] == "ECN1" and "ecn_row_id" not in result["items"][0]
    assert result["not_current_ecns"] == ["ECN2"]
    assert catalogue.consulted == {NEW_1: "ECN1"}


def test_part_index_summarizes_ecn_objects_and_bom_change_sides():
    """The index keeps roles, change types, plants and dates per part number."""
    snapshot = {
        "ecn_number": "ECN1",
        "objects": [{"technical": True, "number": "ACX1", "origins": ["ecn"], "description": "Bracket"},
                    {"technical": False, "number": "DOC1", "origins": ["ecn"]}],
        "bom_changes": {"lines": [{
            "change": "REPLACED", "assembly": "Q900VV001", "plants": [{"plant_code": "PK01"}],
            "outgoing": {"number": "ACX0"}, "incoming": {"number": "ACX1"},
            "engineering_date": "2022-11-28", "manufacturing_date": None, "end_date": "2023-01-23"}]},
    }
    index = build_part_index(snapshot)
    assert set(index["parts"]) == {"ACX0", "ACX1"}
    assert index["parts"]["ACX1"]["roles"] == ["ECN_OBJECT", "INCOMING"]
    assert index["parts"]["ACX1"]["engineering_dates"] == ["2022-11-28"]
    assert index["parts"]["ACX0"]["end_dates"] == ["2023-01-23"] and index["parts"]["ACX0"]["plants"] == ["PK01"]
