"""Unit tests for the bounded, server-scoped RDF assistant tools."""

from __future__ import annotations

import pytest

from app.ecn.rdf import (
    GraphFact,
    GraphFactPage,
    GraphField,
    GraphFieldPage,
    ReplacementSourceObjectFact,
    ReplacementSourceObjectPage,
    root_iri,
)


ECN_ROW_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


class _GraphReads:
    """Record trusted graph tool calls and return one bounded fact page."""

    def __init__(self) -> None:
        """Initialize call recording for the graph-tools contract."""
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.fact = GraphFact(
            subject_ref="urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            subject_types=("urn:ecn-impact:vocab:Object",),
            predicate_name="REQUESTER_NAME",
            value="x" * 4000,
            object_ref=None,
            json_pointer="/REQUESTER_NAME",
            artifact_id="11111111-2222-3333-4444-555555555555",
            source_sha256="a" * 64,
            projection_version="1.0.0",
        )
        self.replacement_page = ReplacementSourceObjectPage(
            items=(
                ReplacementSourceObjectFact(
                    subject_ref="urn:affected",
                    role="affected",
                    number="A-1",
                    name="Old pump",
                    description="Hydraulic pump assembly",
                    object_type="WTPart",
                    source_path="/CNAffectLinks/0/AffectedObjects/0",
                    link_path="/CNAffectLinks/0",
                    direct_relation=None,
                    artifact_id="11111111-2222-3333-4444-555555555555",
                    source_sha256="a" * 64,
                    projection_version="1.0.0",
                ),
                ReplacementSourceObjectFact(
                    subject_ref="urn:resulting",
                    role="resulting",
                    number="R-1",
                    name="New pump",
                    description="Hydraulic pump service assembly",
                    object_type="WTPart",
                    source_path="/ImplementationPlan/0/ResultingLinks/0/ResultingObjects/0",
                    link_path="/ImplementationPlan/0/ResultingLinks/0",
                    direct_relation=None,
                    artifact_id="11111111-2222-3333-4444-555555555555",
                    source_sha256="a" * 64,
                    projection_version="1.0.0",
                ),
            ),
            page=1,
            page_size=100,
            total_matches=2,
            has_next=False,
        )

    def find_facts(self, *args, **kwargs) -> GraphFactPage:
        """Record a closed-over fact search and return one source-grounded result."""
        self.calls.append(("find", args, kwargs))
        return GraphFactPage((self.fact,), page=1, page_size=10, total_matches=1, has_next=False)

    def list_fields(self, *args, **kwargs) -> GraphFieldPage:
        """Record exact field-name discovery and return one catalogue entry."""
        self.calls.append(("fields", args, kwargs))
        return GraphFieldPage(
            (
                GraphField(
                    name="CreatedBy",
                    occurrences=1,
                    example_json_pointer="/CreatedBy",
                    subject_types=("urn:ecn-impact:vocab:Object",),
                ),
            ),
            page=1,
            page_size=50,
            total_matches=1,
            has_next=False,
        )

    def expand_nodes(self, *args, **kwargs) -> GraphFactPage:
        """Record a closed-over expansion and return the same fact shape."""
        self.calls.append(("expand", args, kwargs))
        return GraphFactPage((self.fact,), page=1, page_size=100, total_matches=1, has_next=False)

    def replacement_source_objects(self, *args, **kwargs) -> ReplacementSourceObjectPage:
        """Record the fixed source input read for one candidate-tool invocation."""
        self.calls.append(("replacement", args, kwargs))
        return self.replacement_page


def test_tool_budget_retains_graph_pagination_when_detail_is_omitted() -> None:
    """An oversized graph page keeps safe continuation metadata rather than raw facts."""
    from app.ecn.assistant.tools import _bounded

    result = _bounded(
        {
            "items": ["x" * 3000] * 9,
            "page": 2,
            "page_size": 10,
            "total_matches": 21,
            "has_next": True,
        }
    )

    assert result["detail_omitted"] is True
    assert result["page"] == 2
    assert result["page_size"] == 10
    assert result["total_matches"] == 21
    assert result["has_next"] is True


@pytest.mark.asyncio
async def test_graph_tools_close_over_ecn_bound_inputs_and_capture_facts() -> None:
    """Search trims model input, returns bounded data, and issues only trusted nodes."""
    from app.ecn.assistant.tools import create_graph_tools

    reads = _GraphReads()
    captured: list[GraphFact] = []
    issued: set[str] = set()
    tools = create_graph_tools(reads, ECN_ROW_ID, captured, issued)
    find = next(tool for tool in tools if tool.name == "find_ecn_facts")

    result = await find.ainvoke({"query": f"  {'r' * 240}  ", "page": 1})

    assert reads.calls[0] == (
        "find",
        (ECN_ROW_ID,),
        {
            "query": "r" * 200,
            "search_in": "both",
            "node_type": None,
            "role": None,
            "page": 1,
        },
    )
    assert captured == [reads.fact]
    assert reads.fact.subject_ref in issued
    assert result["text_truncated"] is True
    assert result["result"]["items"][0]["value"].endswith("[text truncated]")
    assert all(
        banned not in find.args
        for banned in ("ecn_row_id", "assessment_id", "sparql", "sql", "graph", "path")
    )


@pytest.mark.asyncio
async def test_field_catalogue_tool_discovers_exact_keys_without_exposing_scope() -> None:
    """The model can resolve fuzzy field names before requesting their values."""
    from app.ecn.assistant.tools import create_graph_tools

    reads = _GraphReads()
    tools = create_graph_tools(reads, ECN_ROW_ID, [], set())
    catalogue = next(tool for tool in tools if tool.name == "list_ecn_fields")

    result = await catalogue.ainvoke({"query": "  created  ", "page": 1})

    assert reads.calls == [
        ("fields", (ECN_ROW_ID,), {"query": "created", "page": 1})
    ]
    assert result["result"]["items"][0]["name"] == "CreatedBy"
    assert result["result"]["items"][0]["example_json_pointer"] == "/CreatedBy"
    assert all(
        banned not in catalogue.args
        for banned in ("ecn_row_id", "assessment_id", "sparql", "sql", "graph", "path")
    )


@pytest.mark.asyncio
async def test_graph_expansion_requires_root_or_a_node_issued_this_turn() -> None:
    """The model cannot guess arbitrary graph node identifiers for expansion."""
    from app.ecn.assistant.tools import create_graph_tools

    reads = _GraphReads()
    issued: set[str] = set()
    tools = create_graph_tools(reads, ECN_ROW_ID, [], issued)
    expand = next(tool for tool in tools if tool.name == "expand_ecn_nodes")

    with pytest.raises(ValueError, match="returned by graph discovery"):
        await expand.ainvoke(
            {
                "node_refs": [
                    "urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                ]
            }
        )

    result = await expand.ainvoke({"node_refs": [str(root_iri(ECN_ROW_ID))], "depth": 2})

    assert reads.calls[-1] == (
        "expand",
        (ECN_ROW_ID,),
        {
            "node_refs": (str(root_iri(ECN_ROW_ID)),),
            "direction": "outgoing",
            "predicates": (),
            "depth": 2,
            "page": 1,
        },
    )
    assert result["result"]["page_size"] == 100


@pytest.mark.asyncio
async def test_replacement_candidate_tool_uses_fixed_source_reads_and_q900v_locations() -> None:
    """Candidates are derived on demand from selected-ECN source and snapshot facts only."""
    from types import SimpleNamespace

    from app.ecn.assistant.tools import create_replacement_tools

    reads = _GraphReads()
    snapshots = SimpleNamespace(
        get=lambda _: {
            "objects": [
                {
                    "number": "A-1",
                    "locations": [
                        {
                            "parent": "PARENT-1",
                            "path": [
                                {"number": "PARENT-1"},
                                {"number": "A-1"},
                            ],
                            "bom_scope": "Q900V only",
                        }
                    ],
                },
                {
                    "number": "R-1",
                    "locations": [
                        {
                            "parent": "PARENT-1",
                            "path": [
                                {"number": "PARENT-1"},
                                {"number": "R-1"},
                            ],
                            "bom_scope": "Q900V only",
                        }
                    ],
                },
            ]
        }
    )
    candidate_tool = create_replacement_tools(snapshots, reads, ECN_ROW_ID)[0]

    result = await candidate_tool.ainvoke({"affected_number": "A-1"})

    assert reads.calls == [("replacement", (ECN_ROW_ID,), {"page": 1})]
    payload = result["result"]
    assert payload["bom_scope"] == "Q900V only"
    assert payload["relation_kind"] == "replacementCandidate"
    assert payload["total_matches"] == 1
    assert payload["items"][0]["relation_kind"] == "replacementCandidate"
    assert "description_token_overlap" in payload["items"][0]["supporting_signals"]
    assert payload["authoritative_source_relations"] == []
    assert all(
        banned not in candidate_tool.args
        for banned in ("ecn_row_id", "assessment_id", "sparql", "sql", "graph", "path")
    )
