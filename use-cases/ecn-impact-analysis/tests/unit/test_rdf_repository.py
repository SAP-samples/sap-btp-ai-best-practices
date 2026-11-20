"""Unit tests for the fixed, application-scoped HANA RDF repository.

The tests use recording SQLAlchemy and DB-API fakes.  They prove that the
repository owns one fixed named graph, never offers generic SPARQL execution,
and keeps relational status writes parameterized without requiring HANA.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from app.ecn.rdf import (
    FIELD_NS,
    PROJECTION_VERSION,
    ProjectionExpectedCounts,
    ProjectionState,
    root_iri,
    project_ecn_json,
)
from app.ecn.errors import PersistenceStateError
from tests.fixtures.persistence import RecordingEngine


ECN_ROW_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
ARTIFACT_ID = "11111111-2222-3333-4444-555555555555"
NOW = datetime(2026, 9, 18, 8, 0, tzinfo=timezone.utc)


class _RawCursor:
    """Record procedure calls and return queued HANA-style output parameters."""

    def __init__(self, connection: "_RawConnection") -> None:
        """Bind the cursor to its short-lived raw connection."""
        self._connection = connection
        self.closed = False

    def callproc(self, procedure: str, arguments: tuple[object, ...]) -> tuple[object, ...]:
        """Record one procedure call and return the next queued output tuple."""
        self._connection.calls.append((procedure, arguments))
        if self._connection.errors:
            raise self._connection.errors.pop(0)
        output = self._connection.outputs.pop(0) if self._connection.outputs else ""
        return (arguments[0], arguments[1], output, None)

    def close(self) -> None:
        """Mark this fake cursor closed for resource-lifecycle assertions."""
        self.closed = True


class _RawConnection:
    """Minimal DB-API connection fake used only by fixed SPARQL operations."""

    def __init__(self) -> None:
        """Initialize empty procedure, output, error, and transaction logs."""
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.outputs: list[str] = []
        self.errors: list[Exception] = []
        self.autocommit_calls: list[bool] = []
        self.commit_count = 0
        self.closed = False

    def cursor(self) -> _RawCursor:
        """Return a fresh recording cursor for one HANA procedure invocation."""
        return _RawCursor(self)

    def commit(self) -> None:
        """Record a transaction commit after a graph mutation."""
        self.commit_count += 1

    def setautocommit(self, enabled: bool) -> None:
        """Record the driver autocommit mode required by HANA SPARQL."""
        self.autocommit_calls.append(enabled)

    def close(self) -> None:
        """Mark this short-lived fake connection closed."""
        self.closed = True


class _RepositoryEngine(RecordingEngine):
    """Combine the shared SQL recorder with a fake raw DB-API connection."""

    def __init__(self) -> None:
        """Initialize both relational and SPARQL call recorders."""
        super().__init__()
        self.raw = _RawConnection()

    def raw_connection(self) -> _RawConnection:
        """Return the application-owned raw connection for one repository call."""
        return self.raw


def _status_row(**overrides: Any) -> dict[str, object]:
    """Build one projected-ECN status row with HANA-like uppercase keys."""
    row: dict[str, object] = {
        "ECN_ROW_ID": ECN_ROW_ID,
        "SOURCE_SHA256": "a" * 64,
        "PROJECTION_VERSION": PROJECTION_VERSION,
        "STATUS": "CURRENT",
        "EXPECTED_OBJECT_COUNT": 2,
        "EXPECTED_ARRAY_COUNT": 1,
        "EXPECTED_SCALAR_COUNT": 3,
        "EXPECTED_NULL_COUNT": 0,
        "EXPECTED_NODE_COUNT": 3,
        "VERIFIED_NODE_COUNT": 3,
        "VERIFIED_TRIPLE_COUNT": 19,
        "ERROR_CODE": None,
        "ERROR_MESSAGE": None,
        "LAST_ATTEMPT_AT": NOW.replace(tzinfo=None),
        "CURRENT_AT": NOW.replace(tzinfo=None),
    }
    row.update(overrides)
    return row


def _sparql_result(**values: object) -> str:
    """Encode one SPARQL JSON result row from simple scalar bindings."""
    return json.dumps(
        {
            "head": {"vars": list(values)},
            "results": {
                "bindings": [
                    {
                        key: {"type": "literal", "value": str(value)}
                        for key, value in values.items()
                    }
                ]
            },
        }
    )


def _count_result(value: int) -> str:
    """Encode one fixed SPARQL aggregate count result."""
    return _sparql_result(count=value)


def _sparql_bindings(*bindings: dict[str, dict[str, str]]) -> str:
    """Encode typed SPARQL JSON bindings for graph-read contract tests."""
    return json.dumps({"head": {"vars": []}, "results": {"bindings": list(bindings)}})


def test_capability_probe_uses_only_fixed_sys_sparql_execute_calls() -> None:
    """Capability probing owns the graph and leaves no probe triple behind."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.raw.outputs.extend(["", "", '{"boolean": true}', "", '{"boolean": false}'])

    RdfRepository(engine).ensure_capabilities()

    assert engine.raw.closed is True
    assert engine.raw.autocommit_calls == [True, False]
    assert {procedure for procedure, _ in engine.raw.calls} == {"SYS.SPARQL_EXECUTE"}
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert queries[0] == "CREATE GRAPH <urn:ecn-impact:graph:ecns>"
    assert "INSERT DATA" in queries[1]
    assert "ASK" in queries[2]
    assert "DELETE DATA" in queries[3]
    assert "ASK" in queries[4]
    assert all("urn:ecn-impact:graph:ecns" in query for query in queries)


def test_status_reads_and_writes_bind_all_variable_values() -> None:
    """Relational status methods use bound values and cap stored diagnostics."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.queue_mapping(_status_row(status="FAILED", error_code="RDF_WRITE_FAILED"))
    engine.queue_rowcounts([1, 1, 1])
    repository = RdfRepository(engine)

    record = repository.get_projection(ECN_ROW_ID)
    repository.mark_pending(ECN_ROW_ID, NOW)
    repository.mark_current(ECN_ROW_ID, 3, 19, NOW)
    repository.mark_failed(ECN_ROW_ID, "RDF_WRITE_FAILED", "x" * 600, NOW)

    assert record is not None
    assert record.status is ProjectionState.FAILED
    assert record.current_at == NOW
    assert record.expected_counts == ProjectionExpectedCounts(2, 1, 3, 0, 3)
    assert engine.begin_count == 3
    for statement, params in engine.executions:
        assert ECN_ROW_ID not in statement
        if "UPDATE" in statement:
            assert params["ecn_row_id"] == ECN_ROW_ID
    assert engine.executions[-1][1]["error_message"] == "x" * 500


def test_replace_is_scoped_to_one_trusted_root_and_chunks_ntriples() -> None:
    """Rebuild deletes only one ECN membership and inserts no chunk over 400 triples."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {f"field_{index}": index for index in range(500)},
    )

    RdfRepository(engine).replace_graph_projection(projection)

    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    deletions = [query for query in queries if query.startswith("DELETE {")]
    assert len(deletions) == 1
    assert "DELETE { GRAPH <urn:ecn-impact:graph:ecns> { ?s ?p ?o } }" in deletions[0]
    assert "belongsToEcn" in deletions[0]
    assert projection.root_iri in deletions[0]
    inserts = [query for query in queries if query.startswith("INSERT DATA")]
    assert len(inserts) > 1
    assert all(
        len([line for line in query.splitlines() if line.startswith("<")]) <= 400
        for query in inserts
    )


def test_sparql_errors_are_reduced_to_a_safe_application_error() -> None:
    """Graph procedure failures never leak a query or source value to callers."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.raw.errors.append(RuntimeError("sensitive source value: alpha"))

    with pytest.raises(PersistenceStateError) as raised:
        RdfRepository(engine).clear_application_graph()

    assert raised.value.code == "RDF_SPARQL_FAILED"
    assert "alpha" not in raised.value.public_message
    assert "GRAPH" not in raised.value.public_message


def test_verification_rejects_members_without_exactly_one_nonnegative_index() -> None:
    """Verification counts missing, negative, and duplicate array-index values as invalid."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"items": ["one"]})
    engine.raw.outputs.extend(
        [
            _sparql_result(
                sourceSha256=projection.source_sha256,
                projectionVersion=projection.projection_version,
            ),
            _count_result(projection.counts.object_count),
            _count_result(projection.counts.array_count),
            _count_result(projection.counts.null_count),
            _count_result(1),
            _count_result(0),
            _count_result(projection.counts.node_count),
            _count_result(1),
            _count_result(0),
            _count_result(projection.counts.triple_count),
        ]
    )

    verification = RdfRepository(engine).verify_graph_projection(projection)

    assert verification.invalid_array_member_count == 1
    assert verification.matches(projection) is False
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert any("HAVING(COUNT(?index) != 1" in query for query in queries)


def test_reconciliation_reduces_malformed_relational_source_to_safe_error() -> None:
    """Reconciliation never exposes malformed stored JSON while calculating source drift."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.queue_mappings(
        [
            {
                "ECN_ROW_ID": ECN_ROW_ID,
                "SOURCE_JSON": "not-json",
                "SOURCE_SHA256": "a" * 64,
                "PROJECTION_VERSION": PROJECTION_VERSION,
                "STATUS": "CURRENT",
            }
        ]
    )

    with pytest.raises(PersistenceStateError) as raised:
        RdfRepository(engine).list_reconcilable(PROJECTION_VERSION)

    assert raised.value.code == "PERSISTED_JSON_INVALID"
    assert "not-json" not in raised.value.public_message


def test_find_facts_uses_scoped_case_insensitive_paging_and_provenance() -> None:
    """Fact discovery decodes source keys and never queries outside the selected ECN."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    subject = "urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    engine.raw.outputs.extend(
        [
            _count_result(11),
            _sparql_bindings(
                {
                    "subject": {"type": "uri", "value": subject},
                    "predicate": {"type": "uri", "value": f"{FIELD_NS}Requester_Name"},
                    "term": {"type": "typed-literal", "value": "Ada", "datatype": "http://www.w3.org/2001/XMLSchema#string"},
                    "pointer": {"type": "typed-literal", "value": ""},
                    "artifact": {"type": "uri", "value": "urn:ecn-impact:artifact:11111111-2222-3333-4444-555555555555"},
                    "sourceSha256": {"type": "typed-literal", "value": "a" * 64},
                    "projectionVersion": {"type": "typed-literal", "value": PROJECTION_VERSION},
                    "subjectTypes": {"type": "typed-literal", "value": "urn:ecn-impact:vocab:Object|urn:ecn-impact:vocab:SourceObject"},
                }
            ),
        ]
    )

    page = RdfRepository(engine).find_facts(
        ECN_ROW_ID,
        query=" requester ",
        search_in="both",
        node_type="SourceObject",
        role="affected",
        page=2,
    )

    assert page.page == 2
    assert page.page_size == 10
    assert page.total_matches == 11
    assert page.has_next is False
    assert page.items[0].predicate_name == "Requester_Name"
    assert page.items[0].value == "Ada"
    assert page.items[0].object_ref is None
    assert page.items[0].json_pointer == "/Requester_Name"
    assert page.items[0].artifact_id == ARTIFACT_ID
    assert page.items[0].subject_types == (
        "urn:ecn-impact:vocab:Object",
        "urn:ecn-impact:vocab:SourceObject",
    )
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert len(queries) == 2
    assert all("GRAPH <urn:ecn-impact:graph:ecns>" in query for query in queries)
    assert all(str(root_iri(ECN_ROW_ID)) in query for query in queries)
    assert all("belongsToEcn" in query for query in queries)
    assert "SELECT DISTINCT ?subject ?predicate ?term" in queries[0]
    assert "?predicate = <urn:ecn-impact:vocab:value>" in queries[0]
    assert 'LCASE(STR(?term)) = LCASE("requester")' in queries[0]
    assert "ORDER BY" in queries[1] and "LIMIT 10 OFFSET 10" in queries[1]


def test_list_fields_returns_exact_selected_ecn_catalogue_with_examples() -> None:
    """Field discovery groups exact source keys without exposing a generic query."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.raw.outputs.extend(
        [
            _count_result(51),
            _sparql_bindings(
                {
                    "predicate": {"type": "uri", "value": f"{FIELD_NS}CreatedBy"},
                    "occurrences": {"type": "literal", "value": "3"},
                    "examplePointer": {"type": "literal", "value": ""},
                    "subjectTypes": {
                        "type": "literal",
                        "value": "urn:ecn-impact:vocab:Object|urn:ecn-impact:vocab:SourceObject",
                    },
                }
            ),
        ]
    )

    page = RdfRepository(engine).list_fields(ECN_ROW_ID, query=" created ", page=2)

    assert page.page == 2
    assert page.page_size == 50
    assert page.total_matches == 51
    assert page.has_next is False
    assert page.items[0].name == "CreatedBy"
    assert page.items[0].occurrences == 3
    assert page.items[0].example_json_pointer == "/CreatedBy"
    assert page.items[0].subject_types == (
        "urn:ecn-impact:vocab:Object",
        "urn:ecn-impact:vocab:SourceObject",
    )
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert len(queries) == 2
    assert all(str(root_iri(ECN_ROW_ID)) in query for query in queries)
    assert all("belongsToEcn" in query for query in queries)
    assert all(FIELD_NS in query for query in queries)
    assert 'CONTAINS(LCASE(STR(?predicate)), LCASE("created"))' in queries[0]
    assert "GROUP BY ?predicate" in queries[1]
    assert "LIMIT 50 OFFSET 50" in queries[1]


def test_replacement_source_pattern_matches_projected_affected_and_resulting_objects() -> None:
    """The fixed source read follows both projected object-array paths."""
    from app.persistence.rdf_repository import RdfRepository

    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {
            "CNAffectLinks": [
                {"AffectedObjects": [{"Number": "A-1", "ObjectType": "WTPart"}]}
            ],
            "ImplementationPlan": [
                {
                    "ResultingLinks": [
                        {
                            "OemReplacesAffected": None,
                            "ResultingObjects": [
                                {"Number": "R-1", "ObjectType": "WTPart"}
                            ],
                        }
                    ]
                }
            ],
        },
    )
    pattern = RdfRepository._replacement_source_object_pattern(projection.root_iri)
    rows = projection.graph.query(f"SELECT ?role WHERE {{ {pattern} }}")

    assert {str(row.role) for row in rows} == {"affected", "resulting"}


def test_replacement_source_objects_reads_only_fixed_selected_ecn_input_fields() -> None:
    """Candidate inputs retain source/link paths without issuing a generic graph query."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    subject = "urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    engine.raw.outputs.extend(
        [
            _count_result(1),
            _sparql_bindings(
                {
                    "subject": {"type": "uri", "value": subject},
                    "role": {"type": "typed-literal", "value": "resulting"},
                    "pointer": {
                        "type": "typed-literal",
                        "value": "/ImplementationPlan/0/ResultingLinks/0/ResultingObjects/0",
                    },
                    "linkPointer": {
                        "type": "typed-literal",
                        "value": "/ImplementationPlan/0/ResultingLinks/0",
                    },
                    "number": {"type": "typed-literal", "value": "R-1"},
                    "name": {"type": "typed-literal", "value": "New pump"},
                    "description": {"type": "typed-literal", "value": "Hydraulic pump"},
                    "objectType": {"type": "typed-literal", "value": "WTPart"},
                    "directRelation": {
                        "type": "typed-literal",
                        "value": "OR:wt.part.WTPart:123",
                    },
                    "artifact": {
                        "type": "uri",
                        "value": "urn:ecn-impact:artifact:11111111-2222-3333-4444-555555555555",
                    },
                    "sourceSha256": {"type": "typed-literal", "value": "a" * 64},
                    "projectionVersion": {"type": "typed-literal", "value": PROJECTION_VERSION},
                }
            ),
        ]
    )

    page = RdfRepository(engine).replacement_source_objects(ECN_ROW_ID, page=1)

    assert page.page_size == 100
    assert page.total_matches == 1
    assert page.has_next is False
    assert page.items[0].number == "R-1"
    assert page.items[0].direct_relation == "OR:wt.part.WTPart:123"
    assert page.items[0].link_path.endswith("/ResultingLinks/0")
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert len(queries) == 2
    assert all(str(root_iri(ECN_ROW_ID)) in query for query in queries)
    assert all("SourceObject" in query for query in queries)
    assert "AffectedLink" in queries[0] and "ResultingLink" in queries[0]
    assert "OemReplacesAffected" in queries[1]
    assert "LIMIT 100 OFFSET 0" in queries[1]


def test_expand_nodes_rejects_a_node_without_exact_selected_ecn_membership() -> None:
    """Expansion refuses a syntactically valid node unless it belongs once to this ECN."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    engine.raw.outputs.append(_count_result(0))
    foreign_node = "urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    with pytest.raises(PersistenceStateError) as raised:
        RdfRepository(engine).expand_nodes(
            ECN_ROW_ID,
            node_refs=(foreign_node,),
            direction="outgoing",
            predicates=(),
            depth=1,
            page=1,
        )

    assert raised.value.code == "RDF_NODE_OUTSIDE_SELECTED_ECN"
    query = str(engine.raw.calls[0][1][0])
    assert "COUNT" in query
    assert str(root_iri(ECN_ROW_ID)) in query


def test_expand_nodes_enforces_static_traversal_limits_before_querying() -> None:
    """Traversal bounds reject oversized starts, filters, depth, and page values."""
    from app.persistence.rdf_repository import RdfRepository

    repository = RdfRepository(_RepositoryEngine())
    root = str(root_iri(ECN_ROW_ID))
    with pytest.raises(ValueError, match="five"):
        repository.expand_nodes(
            ECN_ROW_ID, (root,) * 6, "outgoing", (), 1, 1
        )
    with pytest.raises(ValueError, match="ten"):
        repository.expand_nodes(
            ECN_ROW_ID, (root,), "outgoing", tuple(str(i) for i in range(11)), 1, 1
        )
    with pytest.raises(ValueError, match="depth"):
        repository.expand_nodes(ECN_ROW_ID, (root,), "outgoing", (), 3, 1)
    with pytest.raises(ValueError, match="page"):
        repository.expand_nodes(ECN_ROW_ID, (root,), "outgoing", (), 1, 1001)


def test_expand_nodes_returns_a_deterministic_hundred_fact_page() -> None:
    """Expansion counts unique local triples and pages them in a fixed source order."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    root = str(root_iri(ECN_ROW_ID))
    engine.raw.outputs.extend(
        [
            _count_result(1),
            _count_result(101),
            _sparql_bindings(
                {
                    "subject": {"type": "uri", "value": root},
                    "predicate": {"type": "uri", "value": "urn:ecn-impact:vocab:member"},
                    "term": {"type": "uri", "value": "urn:ecn-impact:node:3f2504e0-4f89-41d3-9a0c-0305e82c3301:cccccccccccccccccccccccccccccccc"},
                    "pointer": {"type": "literal", "value": ""},
                    "artifact": {"type": "uri", "value": "urn:ecn-impact:artifact:11111111-2222-3333-4444-555555555555"},
                    "sourceSha256": {"type": "literal", "value": "a" * 64},
                    "projectionVersion": {"type": "literal", "value": PROJECTION_VERSION},
                    "subjectTypes": {"type": "literal", "value": "urn:ecn-impact:vocab:Object"},
                }
            ),
        ]
    )

    page = RdfRepository(engine).expand_nodes(
        ECN_ROW_ID,
        node_refs=(root,),
        direction="both",
        predicates=("CNAffectLinks",),
        depth=2,
        page=2,
    )

    assert page.page_size == 100
    assert page.total_matches == 101
    assert page.has_next is False
    assert page.items[0].predicate_name == "member"
    assert page.items[0].object_ref is not None
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert "COUNT" in queries[0] and str(root_iri(ECN_ROW_ID)) in queries[0]
    assert "SELECT DISTINCT ?subject ?predicate ?term" in queries[1]
    assert "UNION" in queries[1] and "CNAffectLinks" in queries[1]
    assert "ORDER BY" in queries[2] and "LIMIT 100 OFFSET 100" in queries[2]


def test_find_facts_across_ecns_binds_canonical_roots_and_names_each_fact_owner() -> None:
    """A multi-ECN search is one VALUES-scoped query whose facts carry their ECN."""
    from app.persistence.rdf_repository import RdfRepository

    second_id = "3f2504e0-4f89-41d3-9a0c-0305e82c3302"
    engine = _RepositoryEngine()
    engine.raw.outputs.extend(
        [
            _count_result(1),
            _sparql_bindings(
                {
                    "subject": {"type": "uri", "value": f"urn:ecn-impact:node:{second_id}:aa"},
                    "predicate": {"type": "uri", "value": f"{FIELD_NS}Number"},
                    "term": {"type": "literal", "value": "QKX7702260"},
                    "pointer": {"type": "literal", "value": "/AffectedObjects/0"},
                    "artifact": {"type": "uri", "value": f"urn:ecn-impact:artifact:{ARTIFACT_ID}"},
                    "sourceSha256": {"type": "literal", "value": "a" * 64},
                    "projectionVersion": {"type": "literal", "value": PROJECTION_VERSION},
                    "subjectTypes": {"type": "literal", "value": ""},
                    "root": {"type": "uri", "value": str(root_iri(second_id))},
                }
            ),
        ]
    )

    page = RdfRepository(engine).find_facts(
        [ECN_ROW_ID, second_id], query="QKX7702260", search_in="value",
        node_type=None, role=None, page=1,
    )

    assert page.items[0].ecn_row_id == second_id
    assert page.items[0].value == "QKX7702260"
    queries = [str(arguments[0]) for _, arguments in engine.raw.calls]
    assert all(
        f"VALUES ?root {{ <{root_iri(ECN_ROW_ID)}> <{root_iri(second_id)}> }}" in query
        for query in queries
    )
    assert all("belongsToEcn> ?root" in query for query in queries)
    assert "?projectionVersion ?root" in queries[1]


def test_multi_ecn_reads_reject_non_uuid_identities_before_querying() -> None:
    """Only canonical UUIDs can become root IRIs in a VALUES list."""
    from app.persistence.rdf_repository import RdfRepository

    engine = _RepositoryEngine()
    with pytest.raises(ValueError):
        RdfRepository(engine).list_fields([ECN_ROW_ID, "x> } DROP"], query="", page=1)
    with pytest.raises(ValueError):
        RdfRepository(engine).find_facts([], query="", search_in="both", node_type=None, role=None, page=1)
    assert engine.raw.calls == []
