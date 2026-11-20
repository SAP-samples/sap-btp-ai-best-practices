"""Pure, deterministic ECN JSON-to-RDF projection package.

This package converts one ECN's canonical JSON document into an in-memory
:class:`rdflib.Graph` with stable application IRIs, typed RDF terms, RFC 6901
provenance, ECN membership, source hashing, exact structural counts, and a
deterministic bounded N-Triples serializer.  It contains no HANA, SPARQL,
network, or file I/O; those concerns belong to the persistence and lifecycle
layers built on top of it.

Public API:

- Data contracts (:mod:`app.ecn.rdf.models`):
  :class:`ProjectionState`, :class:`ProjectionCounts`, :class:`ProjectionRecord`,
  :class:`ProjectionBatchResult`, :class:`GraphFact`, :class:`GraphFactPage`,
  :class:`ReplacementSourceObjectFact`, :class:`ReplacementSourceObjectPage`,
  :class:`GraphExpansionPage`, :class:`RdfProjection`, and ``JsonValue``.
- Projection functions (:mod:`app.ecn.rdf.projection`):
  :func:`canonical_source_sha256`, :func:`field_predicate`,
  :func:`decode_field_predicate`, :func:`root_iri`, :func:`node_iri`,
  :func:`project_ecn_json`, and :func:`serialize_ntriples_chunks`.
- Constants and vocabulary: :data:`PROJECTION_VERSION`, :data:`VOCAB`,
  :data:`FIELD_NS`, :data:`ROOT_NS`, :data:`NODE_NS`.
"""

from app.ecn.rdf.models import (
    JsonValue,
    GraphExpansionPage,
    GraphFact,
    GraphFactPage,
    GraphField,
    GraphFieldPage,
    ReplacementSourceObjectFact,
    ReplacementSourceObjectPage,
    ProjectionCounts,
    ProjectionBatchResult,
    ProjectionExpectedCounts,
    ProjectionRecord,
    ProjectionSource,
    ProjectionState,
    ProjectionVerification,
    RdfProjection,
)
from app.ecn.rdf.projection import (
    FIELD_NS,
    NODE_NS,
    PROJECTION_VERSION,
    ROOT_NS,
    VOCAB,
    canonical_source_sha256,
    decode_field_predicate,
    field_predicate,
    node_iri,
    project_ecn_json,
    root_iri,
    serialize_ntriples_chunks,
)

__all__ = [
    "JsonValue",
    "GraphFact",
    "GraphFactPage",
    "GraphField",
    "GraphFieldPage",
    "ReplacementSourceObjectFact",
    "ReplacementSourceObjectPage",
    "GraphExpansionPage",
    "ProjectionCounts",
    "ProjectionBatchResult",
    "ProjectionExpectedCounts",
    "ProjectionRecord",
    "ProjectionSource",
    "ProjectionState",
    "ProjectionVerification",
    "RdfProjection",
    "PROJECTION_VERSION",
    "VOCAB",
    "FIELD_NS",
    "NODE_NS",
    "ROOT_NS",
    "canonical_source_sha256",
    "field_predicate",
    "decode_field_predicate",
    "root_iri",
    "node_iri",
    "project_ecn_json",
    "serialize_ntriples_chunks",
]
