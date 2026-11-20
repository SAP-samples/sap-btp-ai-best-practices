"""Graph source-field tools and the replacement-candidate tool of the ECN chat.

``list_ecn_fields`` and ``find_ecn_facts`` search the open ECN, named ECNs,
or every stored ECN in one SPARQL query (all ECN graphs share one named
graph). ``expand_ecn_nodes`` and ``find_replacement_candidates`` read one ECN.
"""

from dataclasses import asdict
from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from app.ecn.assistant.catalogue import EcnCatalogue
from app.ecn.assistant.tool_scope import _bounded, _capture_graph_facts, _for_ecns, _graph_page_response
from app.ecn.evidence.material_changes import change_summary
from app.ecn.evidence.replacement_candidates import (
    ReplacementSourceObject,
    derive_replacement_candidates,
)
from app.ecn.rdf import GraphFact, root_iri
from app.ecn.services.blocking_io import run_blocking


async def _graph_targets(catalogue: EcnCatalogue, ecn: list[str]) -> tuple[list[dict], dict | None]:
    """Pick the ECNs a graph search covers: named ECNs, else the open ECN, else all ECNs."""
    if ecn or catalogue.open_ecn_row_id is not None:
        return await catalogue.targets(ecn, None)
    return await catalogue.entries(), None


async def _search_graph(rdf, catalogue: EcnCatalogue, ecn: list[str], read, captured=None, issued=None) -> dict:
    """Run one graph read for one ECN (strict) or several ECNs (skip non-current graphs).

    Args:
        rdf: Projection service.
        catalogue: Per-turn ``EcnCatalogue``.
        ecn: ECN numbers from the tool call.
        read: ``(scope) -> page``; ``scope`` is one assessment ID or a list.
        captured / issued: Per-turn graph state, filled from fact pages.

    Returns:
        The bounded page. Multi-ECN pages name each fact's ``ecn_number`` and
        list ``not_current_ecns`` whose graph is not ready.
    """
    targets, problem = await _graph_targets(catalogue, ecn)
    if problem:
        return _bounded(problem)
    if len(targets) == 1:
        target = targets[0]
        page = await run_blocking(read, target["assessment_id"])
        if captured is not None:
            _capture_graph_facts(page, captured, issued)
        catalogue.note(targets)
        result = _graph_page_response(page)
        return _bounded(result if target["ecn_number"] is None else {"ecn_number": target["ecn_number"], **result})
    numbers = {target["assessment_id"]: target["ecn_number"] for target in targets}
    current, not_current = await run_blocking(rdf.current_ids, list(numbers))
    if not current:
        return _bounded({"error_code": "RDF_PROJECTION_NOT_READY",
                         "message": "No selected ECN has a verified source graph yet.",
                         "not_current_ecns": [numbers[i] for i in not_current]})
    page = await run_blocking(read, current)
    if captured is not None:
        _capture_graph_facts(page, captured, issued)
    owners = {getattr(item, "ecn_row_id", None) for item in page.items}
    catalogue.note([target for target in targets if target["assessment_id"] in owners])
    return _bounded({**_graph_page_response(page, numbers), "ecns_searched": len(current),
                     "not_current_ecns": [numbers[i] for i in not_current]})


def create_graph_tools(
    rdf,
    ecn_row_id: str | None,
    captured_facts: list[GraphFact],
    issued_graph_nodes: set[str],
    catalogue: EcnCatalogue | None = None,
) -> list:
    """Return bounded graph-discovery tools for the open ECN or named ECNs.

    Args:
        rdf: Projection service that gates and executes fixed graph operations.
        ecn_row_id: Open ECN identity omitted from model tool schemas, or
            ``None`` when no ECN is open.
        captured_facts: Per-turn collector of exact returned source facts.
        issued_graph_nodes: Per-turn node references eligible for expansion.
        catalogue: Per-turn ``EcnCatalogue``; defaults to one that only knows
            the open ECN.

    Returns:
        Read-only field-catalogue, fact-search, and local-expansion tools.
    """
    catalogue = catalogue or EcnCatalogue(None, ecn_row_id)

    @tool
    async def list_ecn_fields(
        query: str = "",
        page: Annotated[int, Field(ge=1, le=1000)] = 1,
        ecn: list[str] = Field(default_factory=list, max_length=50),
    ) -> dict:
        """Discover exact source JSON field names before searching their values.

        Use this when the user names a concept but the exact ECN key is unknown,
        such as creator, requester, approver, owner, date, status, or contact.
        ``query`` is an optional case-insensitive fragment of the field name;
        start broad (for example ``created``) and retry with a synonym only when
        no field matches. The result contains up to 50 ``items`` with ``name``,
        ``occurrences``, ``example_json_pointer``, and ``subject_types``, plus
        ``total_matches`` and ``has_next``. Follow pages when ``has_next`` is
        true. This tool returns field metadata, not field values; pass an exact
        returned ``name`` to ``find_ecn_facts`` next.

        ``ecn``: ECN numbers to cover. Empty covers the open ECN, or every
        stored ECN when none is open; then each field also has ``ecn_count``.

        Example input: ``{"query": "created", "page": 1}``.
        """
        return await _search_graph(rdf, catalogue, ecn, lambda scope: rdf.list_fields(
            scope, query=query.strip()[:200], page=page))

    @tool
    async def find_ecn_facts(
        query: str,
        search_in: Literal["predicate", "value", "both"] = "both",
        node_type: str | None = None,
        role: Literal["affected", "resulting"] | None = None,
        page: Annotated[int, Field(ge=1, le=1000)] = 1,
        ecn: list[str] = Field(default_factory=list, max_length=50),
    ) -> dict:
        """Retrieve source facts by exact field name or scalar value.

        Use an exact key returned by ``list_ecn_fields`` with
        ``search_in="predicate"`` for direct field questions. Use
        ``search_in="value"`` only to locate an already known scalar, and
        ``both`` for exploratory lookup. Optional ``node_type`` and ``role``
        filters narrow results but can hide valid root fields, so omit them
        initially. Results contain at most ten facts with exact JSON pointers,
        values or related node references, provenance, ``total_matches``, and
        ``has_next``. Follow pages before declaring a repeated field exhausted.

        ``ecn``: ECN numbers to search. Empty searches the open ECN, or every
        stored ECN when none is open; each fact then names its ``ecn_number``
        (use this to find ECNs sharing a source value, e.g. a requester).
        ``not_current_ecns`` were not searched; never treat them as no match.

        Example input: ``{"query": "CreatedBy", "search_in": "predicate", "page": 1}``.
        """
        return await _search_graph(rdf, catalogue, ecn, lambda scope: rdf.find_facts(
            scope, query=query.strip()[:200], search_in=search_in, node_type=node_type,
            role=role, page=page), captured_facts, issued_graph_nodes)

    @tool
    async def expand_ecn_nodes(
        node_refs: Annotated[list[str], Field(min_length=1, max_length=5)],
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        predicates: list[str] = Field(default_factory=list, max_length=10),
        depth: Literal[1, 2] = 1,
        page: Annotated[int, Field(ge=1, le=1000)] = 1,
        ecn: Annotated[str, Field(max_length=80)] = "",
    ) -> dict:
        """Read local context around graph nodes returned during this turn.

        Use only after fact discovery returns a node reference, or start from
        the ECN root when the nearby document structure is needed. ``direction``
        controls incoming/outgoing edges, ``predicates`` optionally limits exact
        source field names, and ``depth`` is one or two local hops. Each page
        contains at most 100 facts with JSON pointers and continuation metadata.
        It does not search the whole ECN and cannot accept guessed or foreign
        node identifiers.

        ``ecn``: the one ECN number the nodes belong to; leave empty for the open ECN.

        Example input: ``{"node_refs": ["<returned node IRI>"], "depth": 1, "page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Expand nodes of one ECN after checking they were issued this turn."""
            root = str(root_iri(row_id))
            if any(node_ref != root and node_ref not in issued_graph_nodes for node_ref in node_refs):
                raise ValueError("Node references must be returned by graph discovery in this turn.")
            result = await run_blocking(
                rdf.expand_nodes,
                row_id,
                node_refs=tuple(node_refs),
                direction=direction,
                predicates=tuple(predicates),
                depth=depth,
                page=page,
            )
            _capture_graph_facts(result, captured_facts, issued_graph_nodes)
            return _graph_page_response(result)

        return await _for_ecns(catalogue, ecn, run, limit=1)

    return [list_ecn_fields, find_ecn_facts, expand_ecn_nodes]


def create_replacement_tools(snapshots, rdf, ecn_row_id: str | None,
                             catalogue: EcnCatalogue | None = None) -> list:
    """Return the tool for explainable replacement candidates of one ECN.

    Args:
        snapshots: Snapshot service providing observed pinned BOM locations.
        rdf: Projection service providing fixed affected/resulting source reads.
        ecn_row_id: Open ECN identity omitted from the tool schema, or ``None``.
        catalogue: Per-turn ``EcnCatalogue``; defaults to the open ECN only.

    Returns:
        A read-only, paginated candidate tool. It returns no authoritative
        replacement mapping and does not persist any derived relation.
    """
    catalogue = catalogue or EcnCatalogue(None, ecn_row_id)

    @tool
    async def find_replacement_candidates(
        affected_number: str | None = None,
        resulting_number: str | None = None,
        page: Annotated[int, Field(ge=1, le=1000)] = 1,
        ecn: Annotated[str, Field(max_length=80)] = "",
    ) -> dict:
        """Rank possible Q900V affected-to-resulting replacement candidates.

        Use only for a replacement or supersession question. Optional exact
        affected/resulting numbers narrow candidates; ``page`` returns ten at a
        time. Output includes supporting signals, conflicts, source paths, and
        any separate authoritative source relations. A ``replacementCandidate``
        is not proof of replacement and must never be described as ``replaces``.

        ``ecn``: one ECN number; leave empty for the open ECN.

        Example input: ``{"affected_number": "A-1", "page": 1}``.
        """

        async def run(row_id: str) -> dict:
            """Derive the candidates of one ECN."""
            snapshot = await run_blocking(snapshots.get, row_id)
            if snapshot.get("version") == "ecn-snapshot-v2":
                matches = [c for c in snapshot.get("changes", []) if c["action"] == "PHASE_OUT"
                           and (not affected_number or c["number"] == affected_number)
                           and (not resulting_number or resulting_number in c["counterparts"])]
                return {"assessment_id": row_id, "bom_scope": snapshot["bom_scope"],
                        "assessment_date": snapshot["assessment_date"], "total": len(matches),
                        "items": [change_summary(c) for c in matches[(page - 1) * 10:page * 10]]}
            source_objects = []
            for source_page_number in range(1, 21):
                source_page = await run_blocking(
                    rdf.replacement_source_objects,
                    row_id,
                    page=source_page_number,
                )
                if source_page.total_matches > 2_000:
                    return {
                        "assessment_id": row_id,
                        "bom_scope": "Q900V only",
                        "error_code": "REPLACEMENT_SOURCE_OBJECT_LIMIT_EXCEEDED",
                        "message": "Candidate derivation is limited to 2,000 source objects for one ECN.",
                    }
                source_objects.extend(
                    ReplacementSourceObject(
                        subject_ref=item.subject_ref,
                        role=item.role,
                        number=item.number,
                        name=item.name,
                        description=item.description,
                        object_type=item.object_type,
                        source_path=item.source_path,
                        link_path=item.link_path,
                        direct_relation=item.direct_relation,
                    )
                    for item in source_page.items
                )
                if not source_page.has_next:
                    break
            snapshot = await run_blocking(snapshots.get, row_id)
            candidates, authoritative = derive_replacement_candidates(
                source_objects,
                snapshot["objects"],
            )
            affected_filter = (affected_number or "").strip()[:200] or None
            resulting_filter = (resulting_number or "").strip()[:200] or None
            matches = tuple(
                candidate
                for candidate in candidates
                if (affected_filter is None or candidate.affected_number == affected_filter)
                and (resulting_filter is None or candidate.resulting_number == resulting_filter)
            )
            page_size = 10
            start = (page - 1) * page_size
            return {
                "assessment_id": row_id,
                "bom_scope": "Q900V only",
                "relation_kind": "replacementCandidate",
                "items": [
                    asdict(candidate) for candidate in matches[start : start + page_size]
                ],
                "authoritative_source_relations": [
                    asdict(relation) for relation in authoritative
                ],
                "page": page,
                "page_size": page_size,
                "total_matches": len(matches),
                "has_next": start + page_size < len(matches),
            }

        return await _for_ecns(catalogue, ecn, run, limit=1)

    return [find_replacement_candidates]
