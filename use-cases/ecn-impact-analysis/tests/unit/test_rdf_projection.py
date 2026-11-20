"""Unit tests for the pure deterministic JSON-to-RDF ECN projection.

These tests pin the public contract of ``app.ecn.rdf`` (Task 2 of the ECN
relational-core / RDF-projection plan).  The projection module is *pure*: it
turns a Python JSON value into an in-memory ``rdflib.Graph`` plus a canonical
source hash, structural counts, and a deterministic chunked N-Triples
serializer.  There is no HANA, SPARQL, network, or file I/O involved, so every
assertion here runs entirely in memory.

The suite covers:

- the full JSON value model (objects, arrays, nested/empty containers,
  date/number-looking strings, booleans, integers, floats, explicit null,
  absent properties, Unicode/reserved property names, duplicate scalar array
  members);
- reversible percent-encoded field predicates (encode -> decode round trip);
- byte-stable canonical source hashing;
- path-proven ``SourceObject``/``AffectedLink``/``ResultingLink`` annotations;
- deterministic reprojection (byte-identical sorted N-Triples + identical
  counts) and stable identifiers under differing ``ecn_row_id`` spellings;
- the bounded, sorted N-Triples chunk serializer.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from app.ecn.rdf import (
    PROJECTION_VERSION,
    VOCAB,
    ProjectionCounts,
    ProjectionState,
    RdfProjection,
    canonical_source_sha256,
    decode_field_predicate,
    field_predicate,
    node_iri,
    project_ecn_json,
    root_iri,
    serialize_ntriples_chunks,
)

# A canonical UUID used as the internal ECN row identity in most tests.
ECN_ROW_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
ARTIFACT_ID = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Small graph-reading helpers (tests never rely on triple insertion order).
# ---------------------------------------------------------------------------


def _subject_at(graph: Graph, pointer: str) -> URIRef:
    """Return the single resource whose ``vocab:jsonPointer`` equals ``pointer``."""
    subjects = list(graph.subjects(VOCAB.jsonPointer, Literal(pointer, datatype=XSD.string)))
    assert len(subjects) == 1, f"expected exactly one node at pointer {pointer!r}, got {subjects}"
    return subjects[0]


def _pointers(graph: Graph) -> set[str]:
    """Return every JSON pointer string carried by a projected resource."""
    return {str(o) for o in graph.objects(None, VOCAB.jsonPointer)}


def _project_comprehensive() -> RdfProjection:
    """Project one rich fixture that exercises every value-model branch."""
    source = {
        "changeNumber": "ECN-001",  # object scalar string
        "revision": 3,  # object scalar integer
        "ratio": 1.5,  # object scalar float -> xsd:double
        "urgent": True,  # object scalar boolean
        "closed": False,  # object scalar boolean
        "owner": None,  # object explicit null
        "tags": ["a", "b", "a"],  # array with a duplicate scalar
        "empty_obj": {},  # empty object
        "empty_arr": [],  # empty array
        "nested": {"inner": {"x": 1}},  # nested objects
        "mixedArr": [{"k": "v"}, None, [10]],  # object, null, array members
    }
    return project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, source)


# ---------------------------------------------------------------------------
# Public contract shape
# ---------------------------------------------------------------------------


def test_projection_state_enum_values() -> None:
    """The four projection lifecycle states carry their exact string values."""
    assert ProjectionState.PENDING == "PENDING"
    assert ProjectionState.CURRENT == "CURRENT"
    assert ProjectionState.FAILED == "FAILED"
    assert ProjectionState.STALE == "STALE"


def test_projection_dataclass_fields() -> None:
    """``project_ecn_json`` returns the frozen ``RdfProjection`` contract."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"a": 1})
    assert isinstance(projection, RdfProjection)
    assert isinstance(projection.counts, ProjectionCounts)
    assert isinstance(projection.graph, Graph)
    assert projection.ecn_row_id == ECN_ROW_ID
    assert projection.root_iri == str(root_iri(ECN_ROW_ID))
    assert projection.projection_version == PROJECTION_VERSION
    assert projection.source_sha256 == canonical_source_sha256({"a": 1})
    # Frozen dataclasses reject attribute mutation.
    with pytest.raises(Exception):
        projection.ecn_row_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Canonical source hashing
# ---------------------------------------------------------------------------


def test_canonical_hash_is_stable_and_key_order_independent() -> None:
    """The hash is byte-stable and independent of dict key insertion order."""
    a = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": True}}
    b = {"a": 1, "nested": {"x": True, "y": [1, 2]}, "b": 2}
    assert canonical_source_sha256(a) == canonical_source_sha256(b)
    # Identical across repeated calls (determinism within a process).
    assert canonical_source_sha256(a) == canonical_source_sha256(a)


def test_canonical_hash_matches_documented_canonical_form() -> None:
    """The hash equals SHA-256 of sorted-key, compact, UTF-8 canonical JSON."""
    source = {"z": "café", "a": [1, {"c": None, "b": False}]}
    expected_json = json.dumps(
        source, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert canonical_source_sha256(source) == expected
    assert len(canonical_source_sha256(source)) == 64


def test_canonical_hash_distinguishes_null_from_absent_and_types() -> None:
    """Explicit null, absent keys, and string-vs-number values hash differently."""
    assert canonical_source_sha256({"a": None}) != canonical_source_sha256({})
    assert canonical_source_sha256({"a": None}) != canonical_source_sha256({"a": "null"})
    assert canonical_source_sha256({"a": 1}) != canonical_source_sha256({"a": "1"})
    assert canonical_source_sha256({"a": True}) != canonical_source_sha256({"a": 1})


# ---------------------------------------------------------------------------
# Reversible field predicates
# ---------------------------------------------------------------------------


RESERVED_KEYS = [
    "changeNumber",
    "Change Number",
    "a/b",
    "a~b",
    "~/~",
    "100%done",
    "a+b=c&d",
    "path:to?thing#frag",
    "café",
    "名前",
    "🚀 rocket",
    "",
    "under_score.dot-dash~tilde",
    'quote"and\\slash',
]


@pytest.mark.parametrize("key", RESERVED_KEYS)
def test_field_predicate_round_trips_exactly(key: str) -> None:
    """Encoding a property name then decoding recovers the exact original key."""
    predicate = field_predicate(key)
    assert isinstance(predicate, URIRef)
    assert str(predicate).startswith("urn:ecn-impact:field:")
    # The encoded IRI is pure ASCII (percent-encoded), so it is URI-safe.
    assert str(predicate).encode("ascii")
    assert decode_field_predicate(predicate) == key


def test_field_predicate_known_encoding() -> None:
    """Spaces and slashes are percent-encoded; unreserved characters are not."""
    assert field_predicate("Change Number") == URIRef("urn:ecn-impact:field:Change%20Number")
    assert field_predicate("a/b") == URIRef("urn:ecn-impact:field:a%2Fb")
    # RFC 3986 unreserved set (letters, digits, - . _ ~) is preserved verbatim.
    assert field_predicate("a-b_c.d~e") == URIRef("urn:ecn-impact:field:a-b_c.d~e")


def test_distinct_keys_yield_distinct_predicates() -> None:
    """Different property names never collide onto the same predicate IRI."""
    seen = {str(field_predicate(k)) for k in RESERVED_KEYS}
    assert len(seen) == len(RESERVED_KEYS)


def test_decode_rejects_foreign_iri() -> None:
    """Decoding an IRI outside the field namespace raises ``ValueError``."""
    with pytest.raises(ValueError):
        decode_field_predicate(URIRef("urn:ecn-impact:vocab:member"))


# ---------------------------------------------------------------------------
# Object scalar field typing (booleans before integers, no reinterpretation)
# ---------------------------------------------------------------------------


def test_object_scalars_are_typed_literals_on_field_predicate() -> None:
    """Object scalar fields become direct typed literals on the field predicate."""
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {"name": "abc", "revision": 3, "ratio": 1.5, "urgent": True, "closed": False},
    )
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)

    name = graph.value(root, field_predicate("name"))
    assert name == Literal("abc", datatype=XSD.string)
    assert name.datatype == XSD.string

    revision = graph.value(root, field_predicate("revision"))
    assert revision.datatype == XSD.integer
    assert revision.value == 3

    ratio = graph.value(root, field_predicate("ratio"))
    assert ratio.datatype == XSD.double
    assert math.isclose(float(ratio.value), 1.5)


def test_booleans_are_checked_before_integers() -> None:
    """``bool`` (a subclass of ``int``) maps to xsd:boolean, not xsd:integer."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"flag": True, "count": 1})
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)

    flag = graph.value(root, field_predicate("flag"))
    assert flag.datatype == XSD.boolean
    assert flag.value is True

    count = graph.value(root, field_predicate("count"))
    assert count.datatype == XSD.integer
    assert count.value == 1


def test_strings_that_resemble_dates_or_numbers_stay_strings() -> None:
    """A source string is an xsd:string even when it looks like a date or number."""
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {"when": "2024-01-01", "num": "42", "flt": "1.5", "bool": "true"},
    )
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)
    for key in ("when", "num", "flt", "bool"):
        literal = graph.value(root, field_predicate(key))
        assert literal.datatype == XSD.string, f"{key} was reinterpreted"


# ---------------------------------------------------------------------------
# Explicit null vs absent property
# ---------------------------------------------------------------------------


def test_each_null_is_its_own_typed_resource() -> None:
    """Every explicit null is a distinct ``NullValue`` resource, not a shared node."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"a": None, "b": None})
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)

    null_a = graph.value(root, field_predicate("a"))
    null_b = graph.value(root, field_predicate("b"))
    assert null_a != null_b  # separate resources, not one shared null node
    for null_node, pointer in ((null_a, "/a"), (null_b, "/b")):
        assert (null_node, RDF.type, VOCAB.NullValue) in graph
        assert graph.value(null_node, VOCAB.jsonPointer) == Literal(pointer, datatype=XSD.string)
        assert graph.value(null_node, VOCAB.belongsToEcn) == root
    assert projection.counts.null_count == 2


def test_absent_property_produces_no_triples() -> None:
    """A property that is not present in the source contributes no predicate."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"present": 1})
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)
    assert graph.value(root, field_predicate("present")) is not None
    # An absent key has neither a literal nor a null resource.
    assert graph.value(root, field_predicate("absent")) is None
    assert projection.counts.null_count == 0


# ---------------------------------------------------------------------------
# Arrays: indices, order, duplicate scalars, empty containers
# ---------------------------------------------------------------------------


def test_array_scalars_get_indexed_wrapper_nodes_preserving_duplicates() -> None:
    """Array scalars become wrapper nodes so duplicate values keep distinct indices."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"tags": ["a", "b", "a"]})
    graph = projection.graph
    array_node = _subject_at(graph, "/tags")
    assert (array_node, RDF.type, VOCAB.Array) in graph

    members = list(graph.objects(array_node, VOCAB.member))
    assert len(members) == 3  # duplicate "a" is NOT collapsed

    by_index = {
        int(graph.value(m, VOCAB.arrayIndex)): graph.value(m, VOCAB.value) for m in members
    }
    assert by_index[0] == Literal("a", datatype=XSD.string)
    assert by_index[1] == Literal("b", datatype=XSD.string)
    assert by_index[2] == Literal("a", datatype=XSD.string)
    # The two "a" members are distinct resources with distinct pointers.
    assert _subject_at(graph, "/tags/0") != _subject_at(graph, "/tags/2")
    for member in members:
        assert (member, RDF.type, VOCAB.ScalarValue) in graph
        assert graph.value(member, VOCAB.belongsToEcn) == root_iri(ECN_ROW_ID)


def test_array_object_and_null_members_carry_array_index() -> None:
    """Object/array/null members of an array each carry a zero-based index."""
    projection = project_ecn_json(
        ECN_ROW_ID, ARTIFACT_ID, {"mixed": [{"k": "v"}, None, [10]]}
    )
    graph = projection.graph
    obj_member = _subject_at(graph, "/mixed/0")
    null_member = _subject_at(graph, "/mixed/1")
    arr_member = _subject_at(graph, "/mixed/2")
    assert graph.value(obj_member, VOCAB.arrayIndex) == Literal(0, datatype=XSD.integer)
    assert graph.value(null_member, VOCAB.arrayIndex) == Literal(1, datatype=XSD.integer)
    assert graph.value(arr_member, VOCAB.arrayIndex) == Literal(2, datatype=XSD.integer)
    assert (obj_member, RDF.type, VOCAB.Object) in graph
    assert (null_member, RDF.type, VOCAB.NullValue) in graph
    assert (arr_member, RDF.type, VOCAB.Array) in graph
    # The inner scalar 10 lives at /mixed/2/0.
    inner = _subject_at(graph, "/mixed/2/0")
    assert graph.value(inner, VOCAB.value) == Literal(10, datatype=XSD.integer)


def test_empty_containers_are_declared_nodes_without_children() -> None:
    """Empty objects and arrays are still declared nodes with no members."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"o": {}, "a": []})
    graph = projection.graph
    empty_obj = _subject_at(graph, "/o")
    empty_arr = _subject_at(graph, "/a")
    assert (empty_obj, RDF.type, VOCAB.Object) in graph
    assert (empty_arr, RDF.type, VOCAB.Array) in graph
    assert list(graph.objects(empty_arr, VOCAB.member)) == []


# ---------------------------------------------------------------------------
# JSON pointer escaping for reserved characters in keys
# ---------------------------------------------------------------------------


def test_json_pointer_escapes_reserved_characters() -> None:
    """Child pointers escape ``~`` as ``~0`` and ``/`` as ``~1`` per RFC 6901.

    The reserved-character keys wrap container values (an empty object) so the
    escaped pointers materialize on resource nodes; a scalar value would be a
    literal with no pointer of its own.
    """
    projection = project_ecn_json(
        ECN_ROW_ID, ARTIFACT_ID, {"a/b": {"c~d": {}}}
    )
    pointers = _pointers(projection.graph)
    assert "" in pointers  # the root pointer
    assert "/a~1b" in pointers  # slash escaped
    assert "/a~1b/c~0d" in pointers  # tilde escaped, nested under escaped slash


# ---------------------------------------------------------------------------
# Path-proven semantic annotations (and nothing else)
# ---------------------------------------------------------------------------


def test_affected_links_and_objects_are_typed_by_path() -> None:
    """Nodes under CNAffectLinks/*/AffectedObjects/* are affected SourceObjects."""
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {"CNAffectLinks": [{"AffectedObjects": [{"objId": "X"}]}]},
    )
    graph = projection.graph
    link = _subject_at(graph, "/CNAffectLinks/0")
    obj = _subject_at(graph, "/CNAffectLinks/0/AffectedObjects/0")
    assert (link, RDF.type, VOCAB.AffectedLink) in graph
    assert (obj, RDF.type, VOCAB.SourceObject) in graph
    assert graph.value(obj, VOCAB.role) == Literal("affected", datatype=XSD.string)


def test_resulting_links_and_objects_are_typed_by_path() -> None:
    """Nodes under ImplementationPlan/*/ResultingLinks/*/ResultingObjects/* are resulting."""
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        {
            "ImplementationPlan": [
                {"ResultingLinks": [{"ResultingObjects": [{"objId": "Y"}]}]}
            ]
        },
    )
    graph = projection.graph
    link = _subject_at(graph, "/ImplementationPlan/0/ResultingLinks/0")
    obj = _subject_at(
        graph, "/ImplementationPlan/0/ResultingLinks/0/ResultingObjects/0"
    )
    assert (link, RDF.type, VOCAB.ResultingLink) in graph
    assert (obj, RDF.type, VOCAB.SourceObject) in graph
    assert graph.value(obj, VOCAB.role) == Literal("resulting", datatype=XSD.string)


def test_no_business_class_inferred_from_unrelated_paths() -> None:
    """Similarly named keys off the proven path receive no business annotation."""
    projection = project_ecn_json(
        ECN_ROW_ID,
        ARTIFACT_ID,
        # AffectedObjects here is NOT under CNAffectLinks -> no SourceObject.
        {"Other": {"AffectedObjects": [{"objId": "Z"}]}},
    )
    graph = projection.graph
    assert (None, RDF.type, VOCAB.SourceObject) not in graph
    assert (None, RDF.type, VOCAB.AffectedLink) not in graph
    assert len(list(graph.objects(None, VOCAB.role))) == 0


# ---------------------------------------------------------------------------
# Root metadata and structural counts
# ---------------------------------------------------------------------------


def test_root_carries_source_artifact_hash_version_and_counts() -> None:
    """The root node carries provenance metadata and the expected structural counts."""
    projection = _project_comprehensive()
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)
    counts = projection.counts

    assert graph.value(root, VOCAB.sourceSha256) == Literal(
        projection.source_sha256, datatype=XSD.string
    )
    assert graph.value(root, VOCAB.projectionVersion) == Literal(
        PROJECTION_VERSION, datatype=XSD.string
    )
    assert graph.value(root, VOCAB.sourceArtifact) is not None
    assert graph.value(root, VOCAB.expectedObjectCount) == Literal(
        counts.object_count, datatype=XSD.integer
    )
    assert graph.value(root, VOCAB.expectedArrayCount) == Literal(
        counts.array_count, datatype=XSD.integer
    )
    assert graph.value(root, VOCAB.expectedScalarCount) == Literal(
        counts.scalar_count, datatype=XSD.integer
    )
    assert graph.value(root, VOCAB.expectedNullCount) == Literal(
        counts.null_count, datatype=XSD.integer
    )
    assert graph.value(root, VOCAB.expectedNodeCount) == Literal(
        counts.node_count, datatype=XSD.integer
    )


def test_structural_counts_are_exact() -> None:
    """The walker reports the exact object/array/scalar/null/node counts."""
    projection = _project_comprehensive()
    counts = projection.counts
    assert counts.object_count == 5
    assert counts.array_count == 4
    assert counts.scalar_count == 11
    assert counts.null_count == 2
    assert counts.node_count == 15


def test_node_count_equals_membership_and_triple_count_equals_len() -> None:
    """``node_count`` counts membership carriers; ``triple_count`` equals len(graph)."""
    projection = _project_comprehensive()
    graph = projection.graph
    membership_subjects = set(graph.subjects(VOCAB.belongsToEcn, root_iri(ECN_ROW_ID)))
    assert len(membership_subjects) == projection.counts.node_count
    # Every membership target is the ECN root; no node escapes to another ECN.
    assert set(graph.objects(None, VOCAB.belongsToEcn)) == {root_iri(ECN_ROW_ID)}
    assert projection.counts.triple_count == len(graph)


def test_graph_contains_no_blank_nodes() -> None:
    """All identifiers are IRIs/literals so the graph is fully deterministic."""
    from rdflib import BNode

    projection = _project_comprehensive()
    for s, _p, o in projection.graph:
        assert not isinstance(s, BNode)
        assert not isinstance(o, BNode)


# ---------------------------------------------------------------------------
# Root that is a bare scalar / null (defensive value-model coverage)
# ---------------------------------------------------------------------------


def test_scalar_root_is_a_scalar_value_node() -> None:
    """A bare scalar source becomes a root ScalarValue carrying vocab:value."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, "just-a-string")
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)
    assert (root, RDF.type, VOCAB.ScalarValue) in graph
    assert graph.value(root, VOCAB.value) == Literal("just-a-string", datatype=XSD.string)
    assert projection.counts.scalar_count == 1
    assert projection.counts.node_count == 1


def test_null_root_is_a_null_value_node() -> None:
    """A bare null source becomes a root NullValue node."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, None)
    graph = projection.graph
    root = root_iri(ECN_ROW_ID)
    assert (root, RDF.type, VOCAB.NullValue) in graph
    assert projection.counts.null_count == 1
    assert projection.counts.node_count == 1


# ---------------------------------------------------------------------------
# Determinism: byte-identical reprojection and stable identifiers
# ---------------------------------------------------------------------------


def test_reprojection_is_byte_identical_and_counts_match() -> None:
    """Reprojecting the same input yields identical N-Triples and identical counts."""
    first = _project_comprehensive()
    second = _project_comprehensive()
    assert list(serialize_ntriples_chunks(first.graph)) == list(
        serialize_ntriples_chunks(second.graph)
    )
    assert first.counts == second.counts
    assert first.source_sha256 == second.source_sha256


def test_key_insertion_order_does_not_change_the_projection() -> None:
    """Object key order carries no meaning; different orders serialize identically."""
    ordered = {"a": 1, "b": {"y": 2, "x": 3}, "c": [1, 2]}
    shuffled = {"c": [1, 2], "b": {"x": 3, "y": 2}, "a": 1}
    left = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, ordered)
    right = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, shuffled)
    assert list(serialize_ntriples_chunks(left.graph)) == list(
        serialize_ntriples_chunks(right.graph)
    )
    assert left.counts == right.counts


def test_ecn_row_id_is_canonicalized_to_stable_uuid_iris() -> None:
    """Different spellings of the same UUID produce identical root/node IRIs."""
    upper = project_ecn_json(ECN_ROW_ID.upper(), ARTIFACT_ID, {"a": {"b": 1}})
    urn = project_ecn_json(f"urn:uuid:{ECN_ROW_ID}", ARTIFACT_ID, {"a": {"b": 1}})
    canonical = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"a": {"b": 1}})
    assert upper.root_iri == canonical.root_iri == urn.root_iri
    assert list(serialize_ntriples_chunks(upper.graph)) == list(
        serialize_ntriples_chunks(canonical.graph)
    )


def test_node_iri_scheme_is_pointer_derived() -> None:
    """Node IRIs derive from the canonical UUID plus the pointer SHA-256 prefix."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"a": {"b": 1}})
    child = node_iri(ECN_ROW_ID, "/a")
    assert child in set(projection.graph.subjects(VOCAB.jsonPointer, Literal("/a", datatype=XSD.string)))
    assert str(child).startswith(f"urn:ecn-impact:node:{ECN_ROW_ID}:")


# ---------------------------------------------------------------------------
# Deterministic bounded N-Triples serialization
# ---------------------------------------------------------------------------


def test_serialize_chunks_are_sorted_valid_and_cover_all_triples() -> None:
    """Chunks are globally sorted, one valid statement per line, covering the graph."""
    projection = _project_comprehensive()
    graph = projection.graph
    chunks = list(serialize_ntriples_chunks(graph, max_triples=4))
    lines: list[str] = []
    for chunk in chunks:
        chunk_lines = chunk.split("\n")
        assert len(chunk_lines) <= 4  # chunk boundary respected
        for line in chunk_lines:
            assert line.endswith(" ."), line  # valid N-Triples statement terminator
        lines.extend(chunk_lines)
    assert len(lines) == len(graph)  # every triple serialized exactly once
    assert lines == sorted(lines)  # globally sorted across chunk boundaries


def test_serialize_chunk_count_matches_ceiling() -> None:
    """The number of chunks equals ceil(triple_count / max_triples)."""
    projection = _project_comprehensive()
    total = len(projection.graph)
    for max_triples in (1, 2, 3, 400):
        chunks = list(serialize_ntriples_chunks(projection.graph, max_triples=max_triples))
        assert len(chunks) == math.ceil(total / max_triples)


def test_serialize_is_deterministic_across_calls() -> None:
    """Serializing the same graph twice yields byte-identical chunk lists."""
    projection = _project_comprehensive()
    assert list(serialize_ntriples_chunks(projection.graph)) == list(
        serialize_ntriples_chunks(projection.graph)
    )


def test_serialize_keeps_multiline_strings_on_one_line() -> None:
    """A newline inside a source string is escaped so each triple stays single-line."""
    projection = project_ecn_json(ECN_ROW_ID, ARTIFACT_ID, {"desc": "line1\nline2"})
    chunks = list(serialize_ntriples_chunks(projection.graph))
    text = "\n".join(chunks)
    # The literal newline must not appear raw; it is escaped as \n within the line.
    assert "line1\\nline2" in text
    for line in text.split("\n"):
        assert line.endswith(" .")
