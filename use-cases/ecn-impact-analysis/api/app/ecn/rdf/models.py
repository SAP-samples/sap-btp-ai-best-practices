"""Pure data contracts for the ECN JSON-to-RDF projection.

This module holds the dependency-light value objects that describe a projection
result.  They contain no HANA, SPARQL, network, or file I/O logic; the only
third-party import is :mod:`rdflib`, because a completed projection carries an
in-memory graph.

The types defined here are part of the *public* projection contract and are
consumed by later tasks (the HANA projection-status repository and the
projection lifecycle service):

- :class:`ProjectionState` — the four lifecycle states of a per-ECN projection.
- :class:`ProjectionCounts` — the structural counts derived while walking the
  canonical JSON, used both for provenance on the graph root and for readback
  verification.
- :class:`RdfProjection` — the full result of projecting one ECN's JSON.
- :class:`ProjectionRecord` — durable relational lifecycle state for one
  rebuildable projection.
- :class:`ProjectionBatchResult` — aggregate outcome of independent rebuilds.

``JsonValue`` is the recursive type alias for any decoded JSON document.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Union

import rdflib

from app.ecn.models import JsonScalar

# A decoded JSON document: object, array, string, number, boolean, or null.
# Defined as a recursive union so type checkers can follow nested structures.
JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]


class ProjectionState(StrEnum):
    """Lifecycle state of a single ECN's RDF projection.

    Values (exact strings persisted in the relational status record):

    - ``PENDING``: the relational import committed; projection is waiting or
      running.
    - ``CURRENT``: the source hash and projection version match verified RDF
      content.
    - ``FAILED``: the last projection attempt failed; the relational source
      remains valid and the ECN stays visible with a graph-unavailable
      diagnostic.
    - ``STALE``: the source hash or projection version no longer matches the
      verified RDF content and a rebuild is required.
    """

    PENDING = "PENDING"
    CURRENT = "CURRENT"
    FAILED = "FAILED"
    STALE = "STALE"


@dataclass(frozen=True)
class ProjectionCounts:
    """Structural counts computed deterministically while walking the JSON.

    All counts are exact and reproducible for a given source document. They are
    stored on the graph root as provenance and later re-derived from the graph
    during readback verification.

    Attributes:
        object_count: Number of JSON objects (including the root when it is an
            object).
        array_count: Number of JSON arrays.
        scalar_count: Number of JSON scalar values (booleans, integers, floats,
            and strings), counting each occurrence.
        null_count: Number of explicit JSON ``null`` values.
        node_count: Number of RDF resources carrying ECN membership
            (``vocab:belongsToEcn``): the root, every object/array/null
            resource, and every array-scalar wrapper node.
        triple_count: Total number of triples in the projected graph
            (``len(graph)``).
    """

    object_count: int
    array_count: int
    scalar_count: int
    null_count: int
    node_count: int
    triple_count: int


@dataclass(frozen=True)
class ProjectionExpectedCounts:
    """Hold the five structural baselines persisted before graph verification.

    The relational status table intentionally does not duplicate the expected
    total-triple count: that full value remains on the rebuildable RDF root and
    is independently recomputed during verification.

    Args:
        object_count: Expected JSON object count.
        array_count: Expected JSON array count.
        scalar_count: Expected JSON scalar count.
        null_count: Expected explicit JSON null count.
        node_count: Expected resource membership count.
    """

    object_count: int
    array_count: int
    scalar_count: int
    null_count: int
    node_count: int


@dataclass(frozen=True)
class RdfProjection:
    """The complete result of projecting one ECN's canonical JSON to RDF.

    Attributes:
        ecn_row_id: The internal ECN row identity exactly as supplied by the
            caller (not normalized; the IRIs derive from its canonical UUID
            form).
        root_iri: The string form of the ECN root IRI
            (``urn:ecn-impact:ecn:{canonical-uuid}``).
        source_sha256: The canonical SHA-256 hash of the source JSON, matching
            :func:`app.ecn.rdf.projection.canonical_source_sha256`.
        projection_version: The projection algorithm version stamped on the
            root; a change here marks existing projections stale.
        counts: The deterministic :class:`ProjectionCounts` for this document.
        graph: The in-memory default :class:`rdflib.Graph` holding every
            projected triple. Named-graph placement is handled by the
            persistence layer, not here.
    """

    ecn_row_id: str
    root_iri: str
    source_sha256: str
    projection_version: str
    counts: ProjectionCounts
    graph: rdflib.Graph


@dataclass(frozen=True)
class ProjectionRecord:
    """Hold the durable relational status of one derived RDF projection.

    Args:
        ecn_row_id: Canonical selected ECN occurrence identity.
        source_sha256: Canonical hash of the authoritative relational JSON.
        projection_version: Mapper version represented by this record.
        status: Current projection lifecycle state.
        expected_counts: Five structural baselines recorded before graph work.
        verified_node_count: Read-back membership count, when verified.
        verified_triple_count: Read-back owned-triple count, when verified.
        error_code: Safe terminal diagnostic code, when failed.
        error_message: Safe bounded terminal diagnostic, when failed.
        last_attempt_at: Most recent projection attempt in UTC.
        current_at: Successful verification timestamp in UTC.
    """

    ecn_row_id: str
    source_sha256: str
    projection_version: str
    status: ProjectionState
    expected_counts: ProjectionExpectedCounts
    verified_node_count: int | None
    verified_triple_count: int | None
    error_code: str | None
    error_message: str | None
    last_attempt_at: datetime | None
    current_at: datetime | None


@dataclass(frozen=True)
class ProjectionSource:
    """Represent the relational source needed to rebuild one RDF projection.

    Args:
        ecn_row_id: Canonical selected ECN occurrence identity.
        artifact_id: Immutable ECN JSON artifact identity used for provenance.
        source_json: Full decoded authoritative JSON document.
    """

    ecn_row_id: str
    artifact_id: str
    source_json: JsonValue


@dataclass(frozen=True)
class ProjectionVerification:
    """Capture graph read-back facts required before a projection becomes current.

    Args:
        source_sha256: Graph-root source hash, or ``None`` when absent.
        projection_version: Graph-root mapper version, or ``None`` when absent.
        counts: Structural counts recomputed from graph content.
        invalid_array_member_count: Members without exactly one nonnegative index.
        cross_ecn_membership_count: Nodes claiming this and another ECN root.
    """

    source_sha256: str | None
    projection_version: str | None
    counts: ProjectionCounts
    invalid_array_member_count: int
    cross_ecn_membership_count: int

    def matches(self, projection: RdfProjection) -> bool:
        """Return whether this independent graph read-back matches a projection.

        Args:
            projection: The just-built in-memory projection expected in HANA.

        Returns:
            ``True`` only when root provenance, all counts, and structural
            isolation/index invariants match exactly.
        """
        return (
            self.source_sha256 == projection.source_sha256
            and self.projection_version == projection.projection_version
            and self.counts == projection.counts
            and self.invalid_array_member_count == 0
            and self.cross_ecn_membership_count == 0
        )


@dataclass(frozen=True)
class ProjectionBatchResult:
    """Summarize independently attempted projection rebuilds.

    Args:
        requested: Number of trusted selected ECN IDs submitted for rebuild.
        current: Number that reached verified ``CURRENT`` state.
        failed: Number that could not reach ``CURRENT`` during this batch.
    """

    requested: int
    current: int
    failed: int


@dataclass(frozen=True)
class GraphFact:
    """Represent one source-grounded scalar or local structural graph fact.

    Args:
        subject_ref: Stable projected-resource IRI that owns the fact.
        subject_types: Known RDF type IRIs for the subject.
        predicate_name: Exact JSON field name or fixed structural relation name.
        value: Native JSON scalar when the fact points to a literal.
        object_ref: Related resource IRI when the fact points to a node.
        json_pointer: Exact RFC 6901 coordinate of this fact; field facts use
            the subject pointer plus the reversibly decoded field token.
        artifact_id: Authoritative relational source-artifact identity.
        source_sha256: Canonical hash of the projected source JSON.
        projection_version: Version of the mapper that produced this fact.
        ecn_row_id: Owning ECN occurrence, set only by searches that span
            several ECNs; ``None`` when the search was scoped to one ECN.
    """

    subject_ref: str
    subject_types: tuple[str, ...]
    predicate_name: str
    value: JsonScalar | None
    object_ref: str | None
    json_pointer: str
    artifact_id: str
    source_sha256: str
    projection_version: str
    ecn_row_id: str | None = None


@dataclass(frozen=True)
class GraphFactPage:
    """Hold one deterministic page of fact-discovery results.

    Args:
        items: At most ten graph facts in stable source-path order.
        page: One-based requested page number.
        page_size: Fixed number of facts returned per page.
        total_matches: Exact total before pagination for the bounded query.
        has_next: Whether a subsequent page can contain more facts.
    """

    items: tuple[GraphFact, ...]
    page: int
    page_size: int
    total_matches: int
    has_next: bool


@dataclass(frozen=True)
class GraphField:
    """Describe one exact JSON field projected for the selected ECN.

    Args:
        name: Reversibly decoded source JSON property name.
        occurrences: Number of projected uses of this field in the ECN.
        example_json_pointer: One RFC 6901 path showing where the field occurs.
        subject_types: RDF types attached to subjects carrying the field.
        ecn_count: Number of ECNs that carry the field, set only by catalogues
            that span several ECNs; ``None`` for a single-ECN catalogue.
    """

    name: str
    occurrences: int
    example_json_pointer: str
    subject_types: tuple[str, ...]
    ecn_count: int | None = None


@dataclass(frozen=True)
class GraphFieldPage:
    """Hold one deterministic page of exact source-field catalogue entries.

    Args:
        items: At most fifty exact field summaries ordered by field name.
        page: One-based requested page number.
        page_size: Fixed number of field summaries returned per page.
        total_matches: Exact number of distinct matching source field names.
        has_next: Whether a subsequent catalogue page can contain more fields.
    """

    items: tuple[GraphField, ...]
    page: int
    page_size: int
    total_matches: int
    has_next: bool


@dataclass(frozen=True)
class ReplacementSourceObjectFact:
    """Hold the fixed RDF fields needed to derive a replacement candidate.

    The record is a source read, not a replacement assertion.  Its link path
    and direct relation preserve the only source-level relationship that may
    become authoritative in a future export.

    Args:
        subject_ref: Selected-ECN RDF node reference for the source object.
        role: Path-proven affected or resulting source collection role.
        number: Source part number when it is textual.
        name: Source object name when it is textual.
        description: Source object description when it is textual.
        object_type: Source object type when it is textual.
        source_path: RFC 6901 pointer of the source object.
        link_path: RFC 6901 pointer of the enclosing affected/resulting link.
        direct_relation: Non-null scalar ``OemReplacesAffected`` source value.
        artifact_id: Authoritative relational source-artifact identity.
        source_sha256: Canonical hash of the projected source JSON.
        projection_version: Version of the mapper that returned this read.
    """

    subject_ref: str
    role: str
    number: str | None
    name: str | None
    description: str | None
    object_type: str | None
    source_path: str
    link_path: str
    direct_relation: JsonScalar | None
    artifact_id: str
    source_sha256: str
    projection_version: str


@dataclass(frozen=True)
class ReplacementSourceObjectPage:
    """Hold a deterministic, selected-ECN page of candidate source inputs.

    Args:
        items: At most one hundred affected/resulting source objects.
        page: One-based source-object page number.
        page_size: Fixed source-object page size.
        total_matches: Exact selected-ECN source-object total before paging.
        has_next: Whether another bounded source-object page exists.
    """

    items: tuple[ReplacementSourceObjectFact, ...]
    page: int
    page_size: int
    total_matches: int
    has_next: bool


@dataclass(frozen=True)
class GraphExpansionPage:
    """Hold one deterministic page of membership-checked local traversal facts.

    Args:
        items: At most one hundred local graph facts in stable path order.
        page: One-based requested page number.
        page_size: Fixed number of facts returned per page.
        total_matches: Exact total before pagination for the bounded traversal.
        has_next: Whether a subsequent page can contain more traversal facts.
    """

    items: tuple[GraphFact, ...]
    page: int
    page_size: int
    total_matches: int
    has_next: bool
