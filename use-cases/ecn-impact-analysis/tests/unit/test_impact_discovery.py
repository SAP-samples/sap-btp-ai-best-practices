"""Tests for bounded RDF evidence captured before impact synthesis."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.ecn.assistant.discovery import (
    DiscoveryResult,
    DiscoveryStatus,
    GraphEvidenceFact,
    ImpactDiscoveryService,
)
from app.ecn.llm.synthesis import build_synthesis_projection
from app.ecn.rdf.models import GraphFact, GraphFactPage
from app.persistence.records import PinnedGenerationConfig

from tests.fixtures.builders import make_evidence_bundle


def _graph_fact() -> GraphFact:
    """Return one stable source-grounded fact from a projected ECN."""
    return GraphFact(
        subject_ref="urn:ecn:node:requester",
        subject_types=("urn:ecn:Field",),
        predicate_name="REQUESTER_NAME",
        value="Ada Requester",
        object_ref=None,
        json_pointer="/Header/REQUESTER_NAME",
        artifact_id="00000000-0000-4000-8000-000000000001",
        source_sha256="a" * 64,
        projection_version="ecn-rdf-v1",
    )


def test_graph_evidence_identity_is_deterministic_and_source_grounded() -> None:
    """Derive a stable E- identifier from exactly the persisted source fact."""
    first = GraphEvidenceFact.from_graph_fact("ecn-row-1", _graph_fact())
    second = GraphEvidenceFact.from_graph_fact("ecn-row-1", _graph_fact())

    assert first == second
    assert first.evidence_id.startswith("E-")
    assert len(first.evidence_id) == 18
    assert first.json_pointer == "/Header/REQUESTER_NAME"
    assert first.value == "Ada Requester"


def test_failed_discovery_has_no_facts_and_a_stable_empty_digest() -> None:
    """Discard partial graph reads so baseline synthesis cannot cite failed work."""
    result = DiscoveryResult.failed("RDF_PROJECTION_STALE")

    assert result.status is DiscoveryStatus.FAILED
    assert result.facts == ()
    assert result.error_code == "RDF_PROJECTION_STALE"
    assert len(result.discovery_hash) == 64


def test_discovered_facts_are_available_to_final_validation_allowlists() -> None:
    """Allow the final strict schema to cite only graph facts actually returned."""
    discovered = DiscoveryResult.complete(
        (GraphEvidenceFact.from_graph_fact("ecn-row-1", _graph_fact()),)
    )

    projection = build_synthesis_projection(
        make_evidence_bundle(), discovery=discovered
    )

    assert projection.discovery_status == "COMPLETE"
    assert projection.discovered_facts == discovered.facts
    assert all(
        discovered.facts[0].evidence_id in allowed
        for allowed in projection.allowed_evidence_ids_by_section.values()
    )


class _FakeBoundModel:
    """Return one graph search tool call, then a deliberately ignored response."""

    def __init__(self) -> None:
        """Initialize the bounded invocation count."""
        self.calls = 0

    async def ainvoke(self, messages: list[object]) -> AIMessage:
        """Return the predefined call pattern without inspecting source content."""
        del messages
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_ecn_facts",
                        "args": {"query": "requester"},
                        "id": "tool-1",
                    }
                ],
            )
        return AIMessage(content="ignored final discovery prose")


class _FakeDiscoveryModel:
    """Bind the supplied bounded graph tools to the fixed fake response stream."""

    def __init__(self) -> None:
        """Create the reusable fake bound model."""
        self.bound = _FakeBoundModel()

    def bind_tools(self, tools: list[object]) -> _FakeBoundModel:
        """Accept only the service-provided tool list for this contract test."""
        assert len(tools) == 3
        return self.bound


class _FakeRdfService:
    """Expose the current gate and one page of source-grounded graph facts."""

    def __init__(self) -> None:
        """Initialize the synchronous call log used by the bounded tools."""
        self.calls: list[str] = []

    def require_current(self, ecn_row_id: str) -> None:
        """Record the selected ECN projection-currentness check."""
        self.calls.append(f"current:{ecn_row_id}")

    def find_facts(self, ecn_row_id: str, **kwargs: object) -> GraphFactPage:
        """Return one fixed graph fact for the requested scoped ECN."""
        self.calls.append(f"find:{ecn_row_id}")
        assert kwargs["query"] == "requester"
        return GraphFactPage((_graph_fact(),), 1, 10, 1, False)


@pytest.mark.asyncio
async def test_discovery_uses_two_model_calls_and_captures_only_tool_facts() -> None:
    """Ignore discovery prose while retaining the source fact returned by the tool."""
    rdf = _FakeRdfService()
    model = _FakeDiscoveryModel()
    config = PinnedGenerationConfig(
        prompt_version="prompt-v1",
        output_schema_version="schema-v1",
        model_name="gpt-5.4",
        request_timeout_seconds=10,
        max_prompt_evidence_items=10,
        max_prompt_characters=20_000,
        strict_structured_output=True,
    )
    service = ImpactDiscoveryService(rdf, "gpt-5.4", model_factory=lambda: model)

    result = await service.discover(
        build_synthesis_projection(make_evidence_bundle()), config
    )

    assert result.status is DiscoveryStatus.COMPLETE
    assert [call.stage for call in result.calls] == [
        "DISCOVERY_1",
        "DISCOVERY_2",
    ]
    assert result.facts[0].predicate_name == "REQUESTER_NAME"
    assert model.bound.calls == 2
    assert rdf.calls[0].startswith("current:")
    assert rdf.calls[1].startswith("find:")


class _FourCallBoundModel:
    """Request the same safe fact three times before ending discovery."""

    def __init__(self) -> None:
        """Initialize the invocation counter."""
        self.calls = 0

    async def ainvoke(self, messages: list[object]) -> AIMessage:
        """Return three tool-bearing responses and then one terminal response."""
        del messages
        self.calls += 1
        if self.calls == 4:
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "find_ecn_facts",
                    "args": {"query": "requester"},
                    "id": f"tool-{self.calls}",
                }
            ],
        )


class _FourCallDiscoveryModel:
    """Bind the fixed tools to a discovery stream longer than two calls."""

    def __init__(self) -> None:
        """Create the reusable bound response stream."""
        self.bound = _FourCallBoundModel()

    def bind_tools(self, tools: list[object]) -> _FourCallBoundModel:
        """Require field, fact, and expansion tools before returning the stream."""
        assert len(tools) == 3
        return self.bound


@pytest.mark.asyncio
async def test_discovery_continues_until_the_model_stops_requesting_tools() -> None:
    """Discovery has no fixed model/tool iteration count inside its timeout."""
    rdf = _FakeRdfService()
    model = _FourCallDiscoveryModel()
    config = PinnedGenerationConfig(
        prompt_version="prompt-v1",
        output_schema_version="schema-v1",
        model_name="gpt-5.4",
        request_timeout_seconds=10,
        max_prompt_evidence_items=10,
        max_prompt_characters=20_000,
        strict_structured_output=True,
    )

    result = await ImpactDiscoveryService(
        rdf, "gpt-5.4", model_factory=lambda: model
    ).discover(build_synthesis_projection(make_evidence_bundle()), config)

    assert result.status is DiscoveryStatus.COMPLETE
    assert [call.stage for call in result.calls] == [
        "DISCOVERY_1",
        "DISCOVERY_2",
        "DISCOVERY_3",
        "DISCOVERY_4",
    ]
    assert model.bound.calls == 4
    assert len([call for call in rdf.calls if call.startswith("find:")]) == 3
