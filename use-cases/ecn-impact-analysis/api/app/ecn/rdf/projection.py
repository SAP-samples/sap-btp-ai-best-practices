"""Pure, deterministic projection of ECN JSON into an in-memory RDF graph.

This module is the single source of truth for how one ECN's canonical JSON
document becomes RDF.  It is intentionally dependency-light: it performs **no**
HANA access, **no** SPARQL execution, **no** network, and **no** file I/O.  It
only turns a Python JSON value into an :class:`rdflib.Graph`, a canonical source
hash, exact structural counts, and a deterministic, bounded N-Triples
serializer.  Named-graph placement, persistence, and verification live in later
layers.

Determinism is the contract.  Reprojecting the same ``(ecn_row_id,
artifact_id, source_json)`` at the same :data:`PROJECTION_VERSION` yields
byte-identical sorted N-Triples and identical :class:`ProjectionCounts`.  This
is achieved by:

- canonicalizing ``ecn_row_id`` into a stable UUID string for the root/node
  IRIs;
- deriving node IRIs from ``sha256(RFC6901 JSON Pointer)`` rather than any
  dict iteration order;
- preserving array order via a zero-based ``vocab:arrayIndex`` on each member;
- typing scalars by native JSON type (booleans before integers) and never
  reinterpreting date/number-looking strings.

IRI scheme (application-owned, free of customer-specific names)::

    root:  urn:ecn-impact:ecn:{canonical-uuid}
    node:  urn:ecn-impact:node:{canonical-uuid}:{sha256(RFC6901-pointer)[:32]}
    field: urn:ecn-impact:field:{UTF-8 percent-encoded exact key}
    vocab: urn:ecn-impact:vocab:{term}

Public API: :func:`canonical_source_sha256`, :func:`field_predicate`,
:func:`decode_field_predicate`, :func:`root_iri`, :func:`node_iri`,
:func:`project_ecn_json`, and :func:`serialize_ntriples_chunks`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from urllib.parse import quote, unquote
from uuid import UUID

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from app.ecn.rdf.models import JsonValue, ProjectionCounts, RdfProjection

# ---------------------------------------------------------------------------
# Application constants and vocabulary
# ---------------------------------------------------------------------------

# Projection algorithm version.  Bumping this string marks every previously
# written projection as STALE, forcing a deterministic rebuild.
PROJECTION_VERSION: str = "1.0.0"

# IRI namespace prefixes for the four stable identity kinds.
ROOT_NS: str = "urn:ecn-impact:ecn:"
NODE_NS: str = "urn:ecn-impact:node:"
FIELD_NS: str = "urn:ecn-impact:field:"
ARTIFACT_NS: str = "urn:ecn-impact:artifact:"

# Generic application vocabulary.  All predicates and RDF types below live here.
VOCAB = Namespace("urn:ecn-impact:vocab:")

# Compiled JSON-Pointer patterns for the ONLY path-proven semantic annotations.
# No other business class is inferred from labels or values.
_AFFECTED_LINK_PATTERN = re.compile(r"/CNAffectLinks/\d+")
_AFFECTED_OBJECT_PATTERN = re.compile(r"/CNAffectLinks/\d+/AffectedObjects/\d+")
_RESULTING_LINK_PATTERN = re.compile(r"/ImplementationPlan/\d+/ResultingLinks/\d+")
_RESULTING_OBJECT_PATTERN = re.compile(
    r"/ImplementationPlan/\d+/ResultingLinks/\d+/ResultingObjects/\d+"
)


# ---------------------------------------------------------------------------
# Canonical source hashing (frozen public contract)
# ---------------------------------------------------------------------------


def canonical_source_sha256(source_json: JsonValue) -> str:
    """Return the byte-stable SHA-256 hex digest of a canonical JSON encoding.

    The document is serialized with sorted object keys, compact separators, and
    UTF-8 encoding, then hashed.  This makes the digest independent of dict key
    insertion order and stable across runs and process restarts, so later layers
    can recompute it to detect drift between the relational source and the RDF
    projection.

    Args:
        source_json: Any decoded JSON value (object, array, string, number,
            boolean, or null).

    Returns:
        The lowercase 64-character hexadecimal SHA-256 digest of the canonical
        UTF-8 JSON bytes.
    """
    canonical = json.dumps(
        source_json,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Reversible field predicates
# ---------------------------------------------------------------------------


def field_predicate(property_name: str) -> URIRef:
    """Return the reversible, percent-encoded field predicate for a JSON key.

    Every character outside the RFC 3986 unreserved set (letters, digits, and
    ``- . _ ~``) is percent-encoded from the exact UTF-8 bytes of the property
    name.  The encoding is a clean bijection with :func:`decode_field_predicate`,
    so the exact original key -- including Unicode, spaces, slashes, ``~``, and
    reserved characters -- can always be recovered.

    Args:
        property_name: The exact JSON object property name.

    Returns:
        A ``urn:ecn-impact:field:`` :class:`rdflib.URIRef` for that key.
    """
    return URIRef(FIELD_NS + quote(property_name, safe=""))


def decode_field_predicate(predicate: URIRef | str) -> str:
    """Recover the exact JSON property name from a field predicate IRI.

    This is the inverse of :func:`field_predicate`: it strips the
    ``urn:ecn-impact:field:`` namespace and percent-decodes the remainder as
    UTF-8.

    Args:
        predicate: A field predicate IRI produced by :func:`field_predicate`.

    Returns:
        The exact original JSON property name.

    Raises:
        ValueError: If the IRI is not in the field namespace.
    """
    text = str(predicate)
    if not text.startswith(FIELD_NS):
        raise ValueError(f"not an ECN field predicate IRI: {text!r}")
    return unquote(text[len(FIELD_NS) :], encoding="utf-8", errors="strict")


# ---------------------------------------------------------------------------
# Stable identity helpers
# ---------------------------------------------------------------------------


def _canonical_uuid(ecn_row_id: str) -> str:
    """Normalize an ECN row identity into its canonical lowercase UUID string.

    Accepts any spelling that :class:`uuid.UUID` understands (hyphenated,
    uppercase, brace-wrapped, or ``urn:uuid:`` prefixed) and returns the single
    canonical hyphenated lowercase form, so identically-identified ECNs always
    produce identical IRIs.

    Args:
        ecn_row_id: The internal ECN row identity.

    Returns:
        The canonical ``str(UUID)`` form.

    Raises:
        ValueError: If ``ecn_row_id`` is not a valid UUID.
    """
    return str(UUID(str(ecn_row_id)))


def _pointer_hash(json_pointer: str) -> str:
    """Return the first 32 hex characters of ``sha256`` over a JSON pointer.

    Args:
        json_pointer: An RFC 6901 JSON Pointer string (``""`` for the root).

    Returns:
        A 32-character lowercase hex prefix used to build node IRIs.
    """
    return hashlib.sha256(json_pointer.encode("utf-8")).hexdigest()[:32]


def root_iri(ecn_row_id: str) -> URIRef:
    """Return the stable ECN root IRI for an ECN row identity.

    Args:
        ecn_row_id: The internal ECN row identity (any UUID spelling).

    Returns:
        ``urn:ecn-impact:ecn:{canonical-uuid}`` as a :class:`rdflib.URIRef`.
    """
    return URIRef(ROOT_NS + _canonical_uuid(ecn_row_id))


def node_iri(ecn_row_id: str, json_pointer: str) -> URIRef:
    """Return the stable node IRI for a JSON pointer within an ECN.

    Args:
        ecn_row_id: The internal ECN row identity (any UUID spelling).
        json_pointer: The RFC 6901 JSON Pointer of the node.

    Returns:
        ``urn:ecn-impact:node:{canonical-uuid}:{sha256(pointer)[:32]}`` as a
        :class:`rdflib.URIRef`.
    """
    return URIRef(f"{NODE_NS}{_canonical_uuid(ecn_row_id)}:{_pointer_hash(json_pointer)}")


def _rfc6901_token(key: str) -> str:
    """Escape one object key as an RFC 6901 reference token.

    Per RFC 6901 the ``~`` character is escaped as ``~0`` and ``/`` as ``~1``;
    ``~`` must be escaped first so an existing ``~1`` in the source is not
    corrupted.

    Args:
        key: The exact JSON object property name.

    Returns:
        The escaped reference token (without the leading ``/``).
    """
    return key.replace("~", "~0").replace("/", "~1")


def _scalar_literal(value: bool | int | float | str) -> Literal:
    """Return the typed RDF literal for a native JSON scalar.

    Booleans are checked before integers because ``bool`` is a subclass of
    ``int`` in Python.  JSON strings become explicit ``xsd:string`` literals and
    are never reinterpreted as dates or numbers.

    Args:
        value: A JSON boolean, integer, float, or string.

    Returns:
        An :class:`rdflib.Literal` typed as xsd:boolean, xsd:integer,
        xsd:double, or xsd:string respectively.

    Raises:
        TypeError: If ``value`` is not a JSON scalar.
    """
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.double)
    if isinstance(value, str):
        return Literal(value, datatype=XSD.string)
    raise TypeError(f"not a JSON scalar: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Projection builder
# ---------------------------------------------------------------------------


class _ProjectionBuilder:
    """Stateful walker that turns one ECN JSON document into RDF triples.

    The builder is single-use: construct it with the ECN identity, artifact
    identity, and source JSON, then call :meth:`build` exactly once.  It tracks
    the structural counts as it walks and guarantees that ``node_count`` equals
    the number of resources that carry ``vocab:belongsToEcn`` membership.

    Args:
        ecn_row_id: The internal ECN row identity (any UUID spelling).
        artifact_id: The source artifact identity, recorded as provenance.
        source_json: The canonical decoded JSON document for the ECN.
    """

    def __init__(self, ecn_row_id: str, artifact_id: str, source_json: JsonValue) -> None:
        self._ecn_row_id = ecn_row_id
        self._canonical_uuid = _canonical_uuid(ecn_row_id)
        self._source_json = source_json
        self._source_sha256 = canonical_source_sha256(source_json)
        self._root = URIRef(ROOT_NS + self._canonical_uuid)
        self._artifact = URIRef(ARTIFACT_NS + quote(str(artifact_id), safe=""))
        self._graph = Graph()
        # Structural accumulators, finalized during build().
        self._object_count = 0
        self._array_count = 0
        self._scalar_count = 0
        self._null_count = 0
        self._node_count = 0

    # -- IRI construction ---------------------------------------------------

    def _node(self, json_pointer: str) -> URIRef:
        """Return the node IRI for ``json_pointer`` within this ECN."""
        return URIRef(f"{NODE_NS}{self._canonical_uuid}:{_pointer_hash(json_pointer)}")

    # -- Resource declaration ----------------------------------------------

    def _declare_membership(self, subject: URIRef, json_pointer: str, rdf_type: URIRef) -> None:
        """Attach ECN membership, JSON pointer, and RDF type to a resource.

        Every resource node passes through here exactly once, which keeps
        ``node_count`` equal to the number of ``vocab:belongsToEcn`` triples.

        Args:
            subject: The resource IRI being declared.
            json_pointer: The RFC 6901 pointer locating the resource.
            rdf_type: The vocabulary RDF type for the resource.
        """
        self._graph.add((subject, VOCAB.belongsToEcn, self._root))
        self._graph.add((subject, VOCAB.jsonPointer, Literal(json_pointer, datatype=XSD.string)))
        self._graph.add((subject, RDF.type, rdf_type))
        self._node_count += 1

    def _declare_object(self, subject: URIRef, json_pointer: str) -> None:
        """Declare a JSON object resource and apply path-proven annotations."""
        self._declare_membership(subject, json_pointer, VOCAB.Object)
        self._object_count += 1
        self._annotate_by_path(subject, json_pointer)

    def _declare_array(self, subject: URIRef, json_pointer: str) -> None:
        """Declare a JSON array resource."""
        self._declare_membership(subject, json_pointer, VOCAB.Array)
        self._array_count += 1

    def _declare_null(self, subject: URIRef, json_pointer: str) -> None:
        """Declare a distinct ``NullValue`` resource for one explicit JSON null."""
        self._declare_membership(subject, json_pointer, VOCAB.NullValue)
        self._null_count += 1

    def _declare_scalar_wrapper(
        self, subject: URIRef, json_pointer: str, value: bool | int | float | str
    ) -> None:
        """Declare a ``ScalarValue`` wrapper node carrying ``vocab:value``.

        Wrappers give array scalars (and a bare scalar root) their own stable
        identity so duplicate values keep distinct indices and paths.
        """
        self._declare_membership(subject, json_pointer, VOCAB.ScalarValue)
        self._graph.add((subject, VOCAB.value, _scalar_literal(value)))
        self._scalar_count += 1

    def _annotate_by_path(self, subject: URIRef, json_pointer: str) -> None:
        """Add the only path-proven business annotations for object resources.

        Objects at the affected-object and resulting-object pointers become
        ``SourceObject`` resources with an ``affected``/``resulting`` role; the
        containing link objects become ``AffectedLink``/``ResultingLink``.  No
        other business class is inferred.  Patterns are mutually exclusive under
        full-match, so at most one branch applies.

        Args:
            subject: The object resource IRI.
            json_pointer: Its RFC 6901 pointer, matched against the fixed
                affected/resulting path shapes.
        """
        if _AFFECTED_OBJECT_PATTERN.fullmatch(json_pointer):
            self._graph.add((subject, RDF.type, VOCAB.SourceObject))
            self._graph.add((subject, VOCAB.role, Literal("affected", datatype=XSD.string)))
        elif _RESULTING_OBJECT_PATTERN.fullmatch(json_pointer):
            self._graph.add((subject, RDF.type, VOCAB.SourceObject))
            self._graph.add((subject, VOCAB.role, Literal("resulting", datatype=XSD.string)))
        elif _AFFECTED_LINK_PATTERN.fullmatch(json_pointer):
            self._graph.add((subject, RDF.type, VOCAB.AffectedLink))
        elif _RESULTING_LINK_PATTERN.fullmatch(json_pointer):
            self._graph.add((subject, RDF.type, VOCAB.ResultingLink))

    # -- Recursive walk -----------------------------------------------------

    def _project_container(self, value: JsonValue, json_pointer: str, subject: URIRef) -> None:
        """Populate the contents of an already-declared object or array node.

        Args:
            value: The dict or list whose contents are being projected.
            json_pointer: The RFC 6901 pointer of ``value``.
            subject: The already-declared resource representing ``value``.
        """
        if isinstance(value, dict):
            self._project_object_fields(value, json_pointer, subject)
        elif isinstance(value, list):
            self._project_array_members(value, json_pointer, subject)

    def _project_object_fields(
        self, obj: dict[str, JsonValue], json_pointer: str, subject: URIRef
    ) -> None:
        """Project every property of a JSON object onto its resource node."""
        for key, child_value in obj.items():
            child_pointer = f"{json_pointer}/{_rfc6901_token(key)}"
            predicate = field_predicate(key)
            if isinstance(child_value, dict):
                child = self._node(child_pointer)
                self._declare_object(child, child_pointer)
                self._graph.add((subject, predicate, child))
                self._project_container(child_value, child_pointer, child)
            elif isinstance(child_value, list):
                child = self._node(child_pointer)
                self._declare_array(child, child_pointer)
                self._graph.add((subject, predicate, child))
                self._project_container(child_value, child_pointer, child)
            elif child_value is None:
                child = self._node(child_pointer)
                self._declare_null(child, child_pointer)
                self._graph.add((subject, predicate, child))
            else:
                # Object scalar fields are direct typed literals -- not nodes.
                self._graph.add((subject, predicate, _scalar_literal(child_value)))
                self._scalar_count += 1

    def _project_array_members(
        self, array: list[JsonValue], json_pointer: str, subject: URIRef
    ) -> None:
        """Project every member of a JSON array, preserving zero-based order."""
        for index, member in enumerate(array):
            child_pointer = f"{json_pointer}/{index}"
            child = self._node(child_pointer)
            if isinstance(member, dict):
                self._declare_object(child, child_pointer)
                self._project_container(member, child_pointer, child)
            elif isinstance(member, list):
                self._declare_array(child, child_pointer)
                self._project_container(member, child_pointer, child)
            elif member is None:
                self._declare_null(child, child_pointer)
            else:
                self._declare_scalar_wrapper(child, child_pointer, member)
            # Every member resource -- object, array, null, or scalar wrapper --
            # is linked by vocab:member and carries a zero-based array index.
            self._graph.add((subject, VOCAB.member, child))
            self._graph.add((child, VOCAB.arrayIndex, Literal(index, datatype=XSD.integer)))

    # -- Root metadata ------------------------------------------------------

    def _emit_root_metadata(self) -> None:
        """Stamp source provenance and expected structural counts on the root."""
        self._graph.add((self._root, VOCAB.sourceArtifact, self._artifact))
        self._graph.add(
            (self._root, VOCAB.sourceSha256, Literal(self._source_sha256, datatype=XSD.string))
        )
        self._graph.add(
            (
                self._root,
                VOCAB.projectionVersion,
                Literal(PROJECTION_VERSION, datatype=XSD.string),
            )
        )
        self._graph.add(
            (self._root, VOCAB.expectedObjectCount, Literal(self._object_count, datatype=XSD.integer))
        )
        self._graph.add(
            (self._root, VOCAB.expectedArrayCount, Literal(self._array_count, datatype=XSD.integer))
        )
        self._graph.add(
            (self._root, VOCAB.expectedScalarCount, Literal(self._scalar_count, datatype=XSD.integer))
        )
        self._graph.add(
            (self._root, VOCAB.expectedNullCount, Literal(self._null_count, datatype=XSD.integer))
        )
        self._graph.add(
            (self._root, VOCAB.expectedNodeCount, Literal(self._node_count, datatype=XSD.integer))
        )

    # -- Entry point --------------------------------------------------------

    def build(self) -> RdfProjection:
        """Walk the source document once and return the finished projection.

        Returns:
            An :class:`RdfProjection` whose graph holds every projected triple
            and whose counts exactly describe the walked structure.
        """
        value = self._source_json
        if isinstance(value, dict):
            self._declare_object(self._root, "")
            self._project_container(value, "", self._root)
        elif isinstance(value, list):
            self._declare_array(self._root, "")
            self._project_container(value, "", self._root)
        elif value is None:
            self._declare_null(self._root, "")
        elif isinstance(value, (bool, int, float, str)):
            self._declare_scalar_wrapper(self._root, "", value)
        else:
            raise TypeError(f"unsupported JSON value type: {type(value).__name__}")

        # Counts are final only after the full walk; stamp them onto the root.
        self._emit_root_metadata()

        counts = ProjectionCounts(
            object_count=self._object_count,
            array_count=self._array_count,
            scalar_count=self._scalar_count,
            null_count=self._null_count,
            node_count=self._node_count,
            triple_count=len(self._graph),
        )
        return RdfProjection(
            ecn_row_id=self._ecn_row_id,
            root_iri=str(self._root),
            source_sha256=self._source_sha256,
            projection_version=PROJECTION_VERSION,
            counts=counts,
            graph=self._graph,
        )


def project_ecn_json(
    ecn_row_id: str,
    artifact_id: str,
    source_json: JsonValue,
) -> RdfProjection:
    """Project one ECN's canonical JSON into a deterministic RDF graph.

    This is the module's primary entry point.  It walks the entire document,
    assigning stable IRIs, typed literals, membership, JSON-pointer provenance,
    array indices, path-proven annotations, and root metadata, then returns the
    graph together with its canonical source hash and exact structural counts.

    The result is fully deterministic: reprojecting the same inputs at the same
    :data:`PROJECTION_VERSION` produces an identical graph and identical counts.

    Args:
        ecn_row_id: The internal ECN row identity (any UUID spelling; it is
            canonicalized for the IRIs).
        artifact_id: The source artifact identity, recorded as root provenance.
        source_json: The canonical decoded JSON value for the ECN.

    Returns:
        The finished :class:`RdfProjection`.
    """
    return _ProjectionBuilder(ecn_row_id, artifact_id, source_json).build()


# ---------------------------------------------------------------------------
# Deterministic bounded N-Triples serialization
# ---------------------------------------------------------------------------


def serialize_ntriples_chunks(
    graph: Graph,
    max_triples: int = 400,
) -> Iterator[str]:
    """Yield the graph as deterministic, size-bounded N-Triples chunks.

    Every triple is serialized to a single valid N-Triples line (control
    characters and newlines inside literals are escaped, so no statement ever
    spans multiple physical lines).  The complete set of lines is sorted
    lexicographically -- making the output independent of triple insertion order
    -- and then yielded in chunks of at most ``max_triples`` lines.

    Args:
        graph: The projected graph to serialize.
        max_triples: The maximum number of N-Triples lines per yielded chunk
            (default 400).

    Yields:
        Strings of up to ``max_triples`` newline-joined N-Triples statements, in
        globally sorted order.

    Raises:
        ValueError: If ``max_triples`` is less than 1.
    """
    if max_triples < 1:
        raise ValueError("max_triples must be a positive integer")

    # rdflib's N-Triples serializer emits exactly one statement per physical
    # line and escapes LF/CR inside literals, so splitting on "\n" is safe.
    serialized = graph.serialize(format="nt")
    lines = sorted(line for line in serialized.split("\n") if line)

    for start in range(0, len(lines), max_triples):
        yield "\n".join(lines[start : start + max_triples])
