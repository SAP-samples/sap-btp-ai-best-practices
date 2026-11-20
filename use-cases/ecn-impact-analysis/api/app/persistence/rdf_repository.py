"""Fixed HANA operations for the rebuildable, application-owned ECN RDF graph.

Relational ECN JSON remains authoritative.  This repository records derived
projection state in HANA tables and executes only server-built SPARQL through
``SYS.SPARQL_EXECUTE``.  It deliberately exposes no generic SPARQL API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal
from urllib.parse import unquote

from rdflib import Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ecn.errors import PersistenceStateError
from app.ecn.models import JsonScalar, canonical_uuid
from app.ecn.rdf import (
    FIELD_NS,
    PROJECTION_VERSION,
    ROOT_NS,
    VOCAB,
    GraphExpansionPage,
    GraphFact,
    GraphFactPage,
    GraphField,
    GraphFieldPage,
    ReplacementSourceObjectFact,
    ReplacementSourceObjectPage,
    ProjectionCounts,
    ProjectionExpectedCounts,
    ProjectionRecord,
    ProjectionSource,
    ProjectionState,
    ProjectionVerification,
    RdfProjection,
    canonical_source_sha256,
    decode_field_predicate,
    field_predicate,
    root_iri,
    serialize_ntriples_chunks,
)
from app.ecn.rdf.projection import ARTIFACT_NS, NODE_NS
from app.persistence.codec import normalize_row_keys, read_lob_text
from app.persistence.hana import from_hana_timestamp, to_hana_timestamp
from app.persistence.table_contracts import PersistenceTableNames, quote_identifier


APPLICATION_GRAPH_IRI = "urn:ecn-impact:graph:ecns"
"""The only production named graph owned by this application."""

_SPARQL_PROCEDURE = "SYS.SPARQL_EXECUTE"
_PROBE_IRI = "urn:ecn-impact:probe:capability"
_GRAPH_IRI_RE = re.compile(r"^urn:ecn-impact:graph(?::[A-Za-z0-9._-]{1,80})?$")
_FACT_PAGE_SIZE = 10
_FIELD_PAGE_SIZE = 50
_EXPANSION_PAGE_SIZE = 100
_MAX_PAGE = 1000
_REPLACEMENT_SOURCE_PAGE_SIZE = 100
_MAX_REPLACEMENT_SOURCE_OBJECTS = 2_000
_MAX_REPLACEMENT_SOURCE_PAGES = (
    _MAX_REPLACEMENT_SOURCE_OBJECTS // _REPLACEMENT_SOURCE_PAGE_SIZE
)
_NODE_TYPES = frozenset(
    {
        "Object",
        "Array",
        "NullValue",
        "ScalarValue",
        "SourceObject",
        "AffectedLink",
        "ResultingLink",
    }
)
_STRUCTURAL_PREDICATES = {
    "member": str(VOCAB.member),
    "arrayIndex": str(VOCAB.arrayIndex),
    "value": str(VOCAB.value),
    "role": str(VOCAB.role),
}


def _root_scope(ecn_row_ids: str | Sequence[str]) -> tuple[str, str]:
    """Return the SPARQL membership scope for one or several trusted ECN roots.

    One ECN keeps the original fixed ``<root>`` term, so single-ECN queries are
    unchanged. Several ECNs bind ``?root`` with a ``VALUES`` list built only
    from canonical UUIDs, so one query searches all of them.

    Args:
        ecn_row_ids: One ECN occurrence identity or a sequence of them.

    Returns:
        ``(values_clause, root_term)``: an empty clause and ``<root-iri>`` for
        one ECN, or ``VALUES ?root { ... }`` and ``?root`` for several.

    Raises:
        ValueError: If no ECN is given or an identity is not a canonical UUID.
    """
    ids = (ecn_row_ids,) if isinstance(ecn_row_ids, str) else tuple(ecn_row_ids)
    if not ids:
        raise ValueError("at least one ECN is required")
    roots = list(dict.fromkeys(f"<{root_iri(canonical_uuid(item))}>" for item in ids))
    if len(roots) == 1:
        return "", roots[0]
    # ponytail: one VALUES row per ECN; fine for hundreds of ECNs, chunk the
    # list into several queries if the stored ECN count grows far beyond that.
    return f"VALUES ?root {{ {' '.join(roots)} }}\n", "?root"


def _safe_graph_read_error() -> PersistenceStateError:
    """Return the one safe error shared by malformed fixed graph-read results."""
    return PersistenceStateError(
        "RDF_SPARQL_RESPONSE_INVALID",
        "The RDF projection service returned an invalid graph-read response.",
        status_code=500,
    )


def _decode_json_scalar(binding: dict[str, str]) -> JsonScalar:
    """Decode one SPARQL literal into its native projected JSON scalar.

    Args:
        binding: SPARQL JSON binding with ``value`` and optional ``datatype``.

    Returns:
        The matching JSON scalar type.

    Raises:
        PersistenceStateError: If HANA returns an unsupported or malformed literal.
    """
    value = binding.get("value")
    datatype = binding.get("datatype", "")
    if not isinstance(value, str):
        raise _safe_graph_read_error()
    try:
        if datatype.endswith("#boolean"):
            if value in {"true", "1"}:
                return True
            if value in {"false", "0"}:
                return False
            raise ValueError("invalid boolean")
        if datatype.endswith("#integer"):
            return int(value)
        if datatype.endswith("#double"):
            return float(value)
        return value
    except ValueError as exc:
        raise _safe_graph_read_error() from exc


def _predicate_name(predicate: str) -> str:
    """Map one fixed source or structural predicate IRI to its safe display name."""
    if predicate.startswith(FIELD_NS):
        try:
            return decode_field_predicate(predicate)
        except ValueError as exc:
            raise _safe_graph_read_error() from exc
    for name, iri in _STRUCTURAL_PREDICATES.items():
        if predicate == iri:
            return name
    raise _safe_graph_read_error()


def _is_literal(binding: dict[str, str]) -> bool:
    """Return whether a SPARQL binding is either standard HANA literal form."""
    return binding.get("type") in {"literal", "typed-literal"}


def _artifact_id(artifact_iri: str) -> str:
    """Decode the source artifact identity from the fixed application IRI namespace."""
    if not artifact_iri.startswith(ARTIFACT_NS):
        raise _safe_graph_read_error()
    return unquote(artifact_iri[len(ARTIFACT_NS) :], encoding="utf-8", errors="strict")


def _projection_record_from_row(row: dict[str, object]) -> ProjectionRecord:
    """Decode one normalized HANA projection-status row.

    Args:
        row: Database mapping with any supported key casing.

    Returns:
        Typed immutable projection status.

    Raises:
        PersistenceStateError: If a persisted state or timestamp is malformed.
    """
    values = normalize_row_keys(row)
    try:
        return ProjectionRecord(
            ecn_row_id=str(values["ECN_ROW_ID"]),
            source_sha256=str(values["SOURCE_SHA256"]),
            projection_version=str(values["PROJECTION_VERSION"]),
            status=ProjectionState(str(values["STATUS"])),
            expected_counts=ProjectionExpectedCounts(
                object_count=int(values["EXPECTED_OBJECT_COUNT"]),
                array_count=int(values["EXPECTED_ARRAY_COUNT"]),
                scalar_count=int(values["EXPECTED_SCALAR_COUNT"]),
                null_count=int(values["EXPECTED_NULL_COUNT"]),
                node_count=int(values["EXPECTED_NODE_COUNT"]),
            ),
            verified_node_count=(
                int(values["VERIFIED_NODE_COUNT"])
                if values["VERIFIED_NODE_COUNT"] is not None
                else None
            ),
            verified_triple_count=(
                int(values["VERIFIED_TRIPLE_COUNT"])
                if values["VERIFIED_TRIPLE_COUNT"] is not None
                else None
            ),
            error_code=(str(values["ERROR_CODE"]) if values["ERROR_CODE"] is not None else None),
            error_message=(
                str(values["ERROR_MESSAGE"]) if values["ERROR_MESSAGE"] is not None else None
            ),
            last_attempt_at=(
                from_hana_timestamp(values["LAST_ATTEMPT_AT"])
                if values["LAST_ATTEMPT_AT"] is not None
                else None
            ),
            current_at=(
                from_hana_timestamp(values["CURRENT_AT"])
                if values["CURRENT_AT"] is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistenceStateError(
            "RDF_PROJECTION_RECORD_INVALID",
            "A stored RDF projection record is malformed.",
            status_code=500,
        ) from exc


def _sparql_json(raw: str | bytes | None) -> dict[str, Any]:
    """Decode a HANA SPARQL JSON response without exposing its raw text.

    Args:
        raw: Procedure output returned for a SELECT or ASK request.

    Returns:
        Parsed response object.

    Raises:
        PersistenceStateError: If HANA returned an invalid SPARQL JSON payload.
    """
    try:
        text_value = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
        return json.loads(text_value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistenceStateError(
            "RDF_SPARQL_RESPONSE_INVALID",
            "The RDF projection service returned an invalid verification response.",
            status_code=500,
        ) from exc


def _source_json(raw: object) -> object:
    """Decode one authoritative ECN JSON NCLOB with a safe persisted-data error.

    Args:
        raw: HANA NCLOB value holding the canonical ECN JSON document.

    Returns:
        The decoded JSON value used only for source hashing or rebuilding.

    Raises:
        PersistenceStateError: If the persisted JSON cannot be decoded.
    """
    try:
        return json.loads(read_lob_text(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise PersistenceStateError(
            "PERSISTED_JSON_INVALID",
            "The stored ECN source JSON is malformed.",
            status_code=500,
        ) from exc


class RdfRepository:
    """Persist projection status and execute fixed scoped HANA RDF operations.

    Args:
        engine: SQLAlchemy engine for the current HANA runtime schema.
        table_names: Application-owned table names; tests may use a suffixed set.
        graph_iri: Fixed application graph override used only by isolated tests.
    """

    def __init__(
        self,
        engine: Engine,
        table_names: PersistenceTableNames = PersistenceTableNames(),
        graph_iri: str = APPLICATION_GRAPH_IRI,
    ) -> None:
        """Bind the HANA engine, validated table names, and fixed graph IRI."""
        if not _GRAPH_IRI_RE.fullmatch(graph_iri):
            raise ValueError("RDF graph IRI is outside the application namespace")
        self._engine = engine
        self._tables = table_names
        self._graph_iri = graph_iri

    def _raw_connection(self) -> Any:
        """Open one autocommit DB-API connection required by HANA SPARQL.

        HANA's ``SYS.SPARQL_EXECUTE`` rejects transactional connections.  The
        connection is reset before it returns to SQLAlchemy's shared pool.
        """
        connection = self._engine.raw_connection()
        try:
            connection.setautocommit(True)
        except Exception as exc:  # noqa: BLE001 - driver-specific failure.
            self._close_raw(connection)
            raise self._safe_sparql_error(exc) from exc
        return connection

    @staticmethod
    def _close_raw(connection: Any) -> None:
        """Reset and close a raw connection without masking the operation error."""
        try:
            connection.setautocommit(False)
        except Exception:  # noqa: BLE001 - cleanup cannot replace the operation result.
            pass
        try:
            connection.close()
        except Exception:  # noqa: BLE001 - cleanup cannot replace the operation result.
            return

    @staticmethod
    def _safe_sparql_error(exc: Exception) -> PersistenceStateError:
        """Reduce a database procedure failure to a source-safe public error."""
        return PersistenceStateError(
            "RDF_SPARQL_FAILED",
            "The RDF projection service could not complete its graph operation.",
            status_code=500,
        )

    @staticmethod
    def _callproc(connection: Any, query: str, accept: str = "") -> str | bytes | None:
        """Call the one supported HANA SPARQL procedure and close its cursor.

        Args:
            connection: Short-lived DB-API HANA connection.
            query: Fixed server-built SPARQL text.
            accept: HANA response media type for read operations.

        Returns:
            The procedure's response payload.
        """
        cursor = connection.cursor()
        try:
            result = cursor.callproc(_SPARQL_PROCEDURE, (query, accept, "?", "?"))
            return result[2] if len(result) > 2 else None
        finally:
            cursor.close()

    def _execute(self, connection: Any, query: str, accept: str = "") -> str | bytes | None:
        """Execute one fixed query and translate driver errors to a safe error."""
        try:
            return self._callproc(connection, query, accept)
        except PersistenceStateError:
            raise
        except Exception as exc:  # noqa: BLE001 - hdbcli has driver-specific error types.
            raise self._safe_sparql_error(exc) from exc

    def _ensure_graph(self, connection: Any) -> None:
        """Create the fixed graph, accepting only the expected already-exists race."""
        try:
            self._execute(connection, f"CREATE GRAPH <{self._graph_iri}>")
            connection.commit()
        except PersistenceStateError as exc:
            if "exist" not in str(exc.__cause__ or "").lower():
                raise exc

    def _ask(self, connection: Any, query: str) -> bool:
        """Execute one fixed ASK query and return its required boolean result."""
        payload = _sparql_json(
            self._execute(connection, query, "application/sparql-results+json")
        )
        try:
            return bool(payload["boolean"])
        except (KeyError, TypeError) as exc:
            raise PersistenceStateError(
                "RDF_SPARQL_RESPONSE_INVALID",
                "The RDF projection service returned an invalid verification response.",
                status_code=500,
            ) from exc

    def _select_row(self, connection: Any, query: str) -> dict[str, str] | None:
        """Run one fixed SELECT and return its first literal-binding row, if any."""
        payload = _sparql_json(
            self._execute(connection, query, "application/sparql-results+json")
        )
        try:
            bindings = payload["results"]["bindings"]
            if not bindings:
                return None
            return {
                str(key): str(value["value"])
                for key, value in bindings[0].items()
                if isinstance(value, dict) and "value" in value
            }
        except (KeyError, TypeError, IndexError) as exc:
            raise PersistenceStateError(
                "RDF_SPARQL_RESPONSE_INVALID",
                "The RDF projection service returned an invalid verification response.",
                status_code=500,
            ) from exc

    def _select_bindings(
        self, connection: Any, query: str
    ) -> tuple[dict[str, dict[str, str]], ...]:
        """Run one fixed SELECT and retain typed SPARQL JSON bindings.

        Args:
            connection: Open raw HANA connection for the scoped read.
            query: Fixed server-built SPARQL SELECT text.

        Returns:
            Immutable typed binding rows in HANA's response order.

        Raises:
            PersistenceStateError: If HANA returns malformed result bindings.
        """
        payload = _sparql_json(
            self._execute(connection, query, "application/sparql-results+json")
        )
        try:
            rows = payload["results"]["bindings"]
            if not isinstance(rows, list):
                raise TypeError("bindings must be a list")
            normalized: list[dict[str, dict[str, str]]] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise TypeError("binding row must be an object")
                normalized_row: dict[str, dict[str, str]] = {}
                for key, value in row.items():
                    if not isinstance(key, str) or not isinstance(value, dict):
                        raise TypeError("binding must be an object")
                    if not isinstance(value.get("type"), str) or not isinstance(
                        value.get("value"), str
                    ):
                        raise TypeError("binding needs type and value")
                    normalized_row[key] = {
                        str(field): str(field_value)
                        for field, field_value in value.items()
                        if isinstance(field, str) and isinstance(field_value, str)
                    }
                normalized.append(normalized_row)
            return tuple(normalized)
        except (KeyError, TypeError) as exc:
            raise _safe_graph_read_error() from exc

    @staticmethod
    def _binding(row: dict[str, dict[str, str]], name: str) -> dict[str, str]:
        """Return one required typed binding or raise the safe graph-read error."""
        binding = row.get(name)
        if binding is None:
            raise _safe_graph_read_error()
        return binding

    @classmethod
    def _fact_from_binding(cls, row: dict[str, dict[str, str]]) -> GraphFact:
        """Convert one fixed graph row into a typed source-grounded fact."""
        subject = cls._binding(row, "subject")
        predicate = cls._binding(row, "predicate")
        term = cls._binding(row, "term")
        pointer = cls._binding(row, "pointer")
        artifact = cls._binding(row, "artifact")
        source_sha256 = cls._binding(row, "sourceSha256")
        projection_version = cls._binding(row, "projectionVersion")
        subject_types = cls._binding(row, "subjectTypes")
        if subject.get("type") != "uri" or predicate.get("type") != "uri":
            raise _safe_graph_read_error()
        if term.get("type") == "uri":
            value: JsonScalar | None = None
            object_ref: str | None = term["value"]
        elif _is_literal(term):
            value = _decode_json_scalar(term)
            object_ref = None
        else:
            raise _safe_graph_read_error()
        if (
            not _is_literal(pointer)
            or artifact.get("type") != "uri"
            or not _is_literal(source_sha256)
            or not _is_literal(projection_version)
            or not _is_literal(subject_types)
        ):
            raise _safe_graph_read_error()
        predicate_name = _predicate_name(predicate["value"])
        json_pointer = pointer["value"]
        if predicate["value"].startswith(FIELD_NS):
            token = predicate_name.replace("~", "~0").replace("/", "~1")
            json_pointer = f"{json_pointer}/{token}"
        return GraphFact(
            subject_ref=subject["value"],
            subject_types=tuple(
                sorted(filter(None, subject_types["value"].split("|")))
            ),
            predicate_name=predicate_name,
            value=value,
            object_ref=object_ref,
            json_pointer=json_pointer,
            artifact_id=_artifact_id(artifact["value"]),
            source_sha256=source_sha256["value"],
            projection_version=projection_version["value"],
            ecn_row_id=cls._root_ecn(row),
        )

    @classmethod
    def _root_ecn(cls, row: dict[str, dict[str, str]]) -> str | None:
        """Return the ECN identity of an optional ``?root`` binding, or ``None``."""
        root = row.get("root")
        if root is None:
            return None
        if root.get("type") != "uri" or not root["value"].startswith(ROOT_NS):
            raise _safe_graph_read_error()
        return root["value"][len(ROOT_NS):]

    @staticmethod
    def _optional_text(
        row: dict[str, dict[str, str]], name: str
    ) -> str | None:
        """Return one optional textual source field, ignoring non-text scalars."""
        binding = row.get(name)
        if binding is None:
            return None
        if not _is_literal(binding):
            raise _safe_graph_read_error()
        value = _decode_json_scalar(binding)
        return value if isinstance(value, str) else None

    @classmethod
    def _field_from_binding(cls, row: dict[str, dict[str, str]]) -> GraphField:
        """Decode one grouped field-catalogue result from fixed SPARQL output."""
        predicate = cls._binding(row, "predicate")
        occurrences = cls._binding(row, "occurrences")
        pointer = cls._binding(row, "examplePointer")
        types = row.get("subjectTypes", {"type": "literal", "value": ""})
        if (
            predicate.get("type") != "uri"
            or not predicate.get("value", "").startswith(FIELD_NS)
            or not _is_literal(occurrences)
            or not _is_literal(pointer)
            or not _is_literal(types)
        ):
            raise _safe_graph_read_error()
        name = _predicate_name(predicate["value"])
        token = name.replace("~", "~0").replace("/", "~1")
        base_pointer = pointer["value"]
        try:
            count = int(occurrences["value"])
        except ValueError as exc:
            raise _safe_graph_read_error() from exc
        if count < 1:
            raise _safe_graph_read_error()
        ecn_count = row.get("ecnCount")
        if ecn_count is not None and not _is_literal(ecn_count):
            raise _safe_graph_read_error()
        return GraphField(
            name=name,
            occurrences=count,
            example_json_pointer=f"{base_pointer}/{token}",
            subject_types=tuple(
                sorted(set(filter(None, types["value"].split("|"))))
            ),
            ecn_count=int(ecn_count["value"]) if ecn_count is not None else None,
        )

    @classmethod
    def _replacement_source_object_from_binding(
        cls, row: dict[str, dict[str, str]]
    ) -> ReplacementSourceObjectFact:
        """Decode one fixed source-object/link row for candidate derivation."""
        subject = cls._binding(row, "subject")
        role = cls._binding(row, "role")
        pointer = cls._binding(row, "pointer")
        link_pointer = cls._binding(row, "linkPointer")
        artifact = cls._binding(row, "artifact")
        source_sha256 = cls._binding(row, "sourceSha256")
        projection_version = cls._binding(row, "projectionVersion")
        if (
            subject.get("type") != "uri"
            or not _is_literal(role)
            or role.get("value") not in {"affected", "resulting"}
            or not _is_literal(pointer)
            or not _is_literal(link_pointer)
            or artifact.get("type") != "uri"
            or not _is_literal(source_sha256)
            or not _is_literal(projection_version)
        ):
            raise _safe_graph_read_error()
        direct_binding = row.get("directRelation")
        if direct_binding is None:
            direct_relation: JsonScalar | None = None
        elif _is_literal(direct_binding):
            direct_relation = _decode_json_scalar(direct_binding)
        else:
            raise _safe_graph_read_error()
        return ReplacementSourceObjectFact(
            subject_ref=subject["value"],
            role=role["value"],
            number=cls._optional_text(row, "number"),
            name=cls._optional_text(row, "name"),
            description=cls._optional_text(row, "description"),
            object_type=cls._optional_text(row, "objectType"),
            source_path=pointer["value"],
            link_path=link_pointer["value"],
            direct_relation=direct_relation,
            artifact_id=_artifact_id(artifact["value"]),
            source_sha256=source_sha256["value"],
            projection_version=projection_version["value"],
        )

    def _count(self, connection: Any, pattern: str) -> int:
        """Return one graph-scoped count from a fixed server-built pattern."""
        row = self._select_row(
            connection,
            f"SELECT (COUNT(*) AS ?count) WHERE {{ GRAPH <{self._graph_iri}> {{ {pattern} }} }}",
        )
        try:
            return int((row or {}).get("count", "0"))
        except ValueError as exc:
            raise PersistenceStateError(
                "RDF_SPARQL_RESPONSE_INVALID",
                "The RDF projection service returned an invalid verification response.",
                status_code=500,
            ) from exc

    def _invalid_array_member_count(self, connection: Any, root: str) -> int:
        """Count members lacking exactly one nonnegative ``arrayIndex`` literal.

        Args:
            connection: Open raw HANA connection for the current verification.
            root: Trusted selected ECN root IRI.

        Returns:
            Number of invalid array-member resources in the selected ECN.
        """
        row = self._select_row(
            connection,
            f"SELECT (COUNT(*) AS ?count) WHERE {{ "
            f"{{ SELECT ?member WHERE {{ GRAPH <{self._graph_iri}> {{ "
            f"?member <{VOCAB.belongsToEcn}> <{root}> . "
            f"?array <{VOCAB.member}> ?member . "
            f"OPTIONAL {{ ?member <{VOCAB.arrayIndex}> ?index . }} "
            f"}} }} GROUP BY ?member "
            f"HAVING(COUNT(?index) != 1 || MIN(?index) < 0) }} }}",
        )
        try:
            return int((row or {}).get("count", "0"))
        except ValueError as exc:
            raise PersistenceStateError(
                "RDF_SPARQL_RESPONSE_INVALID",
                "The RDF projection service returned an invalid verification response.",
                status_code=500,
            ) from exc

    @staticmethod
    def _validate_page(page: int) -> None:
        """Reject page values outside the small fixed graph-read range."""
        if not 1 <= page <= _MAX_PAGE:
            raise ValueError(f"page must be between 1 and {_MAX_PAGE}")

    @staticmethod
    def _node_type_iri(node_type: str | None) -> str | None:
        """Return the only allowed projected node-type IRI for an optional filter."""
        if node_type is None:
            return None
        if node_type not in _NODE_TYPES:
            raise ValueError("node_type must be a known projected node type")
        return str(getattr(VOCAB, node_type))

    @staticmethod
    def _fact_filters(
        query: str,
        search_in: Literal["predicate", "value", "both"],
        node_type: str | None,
        role: Literal["affected", "resulting"] | None,
    ) -> str:
        """Build only validated literal/type/role filters for a fact search."""
        if search_in not in {"predicate", "value", "both"}:
            raise ValueError("search_in must be predicate, value, or both")
        if len(query) > 200:
            raise ValueError("query must not exceed 200 characters")
        filters = [
            f'FILTER(STRSTARTS(STR(?predicate), "{FIELD_NS}") || '
            f"?predicate = <{VOCAB.value}>)"
        ]
        if query:
            literal = Literal(query).n3()
            predicate_match = f"CONTAINS(LCASE(STR(?predicate)), LCASE({literal}))"
            value_match = (
                f"(isLiteral(?term) && (LCASE(STR(?term)) = LCASE({literal}) || "
                f"CONTAINS(LCASE(STR(?term)), LCASE({literal}))))"
            )
            expression = (
                predicate_match
                if search_in == "predicate"
                else value_match
                if search_in == "value"
                else f"({predicate_match} || {value_match})"
            )
            filters.append(f"FILTER({expression})")
        type_iri = RdfRepository._node_type_iri(node_type)
        if type_iri is not None:
            filters.append(f"?subject a <{type_iri}> .")
        if role is not None:
            if role not in {"affected", "resulting"}:
                raise ValueError("role must be affected or resulting")
            filters.append(f'?subject <{VOCAB.role}> "{role}" .')
        return "\n".join(filters)

    @staticmethod
    def _fact_result_pattern(root_term: str, filters: str) -> str:
        """Return the fixed membership-scoped source-fact graph pattern.

        ``root_term`` is ``<root-iri>`` for one ECN or ``?root`` for several.
        """
        return f"""
?subject <{VOCAB.belongsToEcn}> {root_term} .
?subject <{VOCAB.jsonPointer}> ?pointer .
?subject ?predicate ?term .
{filters}
""".strip()

    def _read_fact_page(
        self,
        connection: Any,
        root: str,
        graph_pattern: str,
        page: int,
        page_size: int,
        page_type: type[GraphFactPage] | type[GraphExpansionPage],
    ) -> GraphFactPage | GraphExpansionPage:
        """Read one deterministic graph-fact page from a fixed internal pattern.

        ``root`` is one root IRI, or ``?root`` when the pattern binds several
        ECN roots; then each fact also carries its owning ECN.
        """
        root_term = root if root == "?root" else f"<{root}>"
        root_var = " ?root" if root == "?root" else ""
        count_row = self._select_row(
            connection,
            f"""SELECT (COUNT(*) AS ?count) WHERE {{
  {{ SELECT DISTINCT ?subject ?predicate ?term WHERE {{
    GRAPH <{self._graph_iri}> {{
      {graph_pattern}
    }}
  }} }}
}}""",
        )
        try:
            total = int((count_row or {}).get("count", "0"))
        except ValueError as exc:
            raise _safe_graph_read_error() from exc
        offset = (page - 1) * page_size
        rows = self._select_bindings(
            connection,
            f"""SELECT ?subject ?predicate ?term ?pointer ?artifact ?sourceSha256 ?projectionVersion{root_var}
       (GROUP_CONCAT(STR(?subjectType); SEPARATOR = "|"; MAX_LENGTH = 2000) AS ?subjectTypes)
WHERE {{
  GRAPH <{self._graph_iri}> {{
    {graph_pattern}
    {root_term} <{VOCAB.sourceArtifact}> ?artifact ;
             <{VOCAB.sourceSha256}> ?sourceSha256 ;
             <{VOCAB.projectionVersion}> ?projectionVersion .
    OPTIONAL {{ ?subject a ?subjectType . }}
  }}
}}
GROUP BY ?subject ?predicate ?term ?pointer ?artifact ?sourceSha256 ?projectionVersion{root_var}
ORDER BY STR(?pointer) STR(?predicate) STR(?term)
LIMIT {page_size} OFFSET {offset}""",
        )
        items = tuple(self._fact_from_binding(row) for row in rows)
        return page_type(
            items=items,
            page=page,
            page_size=page_size,
            total_matches=total,
            has_next=offset + len(items) < total,
        )

    def find_facts(
        self,
        ecn_row_id: str | Sequence[str],
        query: str,
        search_in: Literal["predicate", "value", "both"],
        node_type: str | None,
        role: Literal["affected", "resulting"] | None,
        page: int,
    ) -> GraphFactPage:
        """Search fixed source-field predicates and literal values in one or several ECNs.

        Args:
            ecn_row_id: Trusted canonical ECN occurrence identity, or a sequence
                of them to search all those ECNs in one query (each returned
                fact then names its ``ecn_row_id``).
            query: Case-insensitive lexical term, capped at 200 characters.
            search_in: Closed predicate/value search mode.
            node_type: Optional known projected node-type filter.
            role: Optional affected/resulting source-role filter.
            page: One-based page number up to the fixed maximum.

        Returns:
            Ten deterministic source-grounded facts and continuation metadata.
        """
        values, root_term = _root_scope(ecn_row_id)
        self._validate_page(page)
        if not isinstance(query, str):
            raise ValueError("query must be text")
        root = "?root" if root_term == "?root" else root_term[1:-1]
        pattern = values + self._fact_result_pattern(
            root_term,
            self._fact_filters(query.strip(), search_in, node_type, role),
        )
        connection = self._raw_connection()
        try:
            result = self._read_fact_page(
                connection,
                root,
                pattern,
                page,
                _FACT_PAGE_SIZE,
                GraphFactPage,
            )
            assert isinstance(result, GraphFactPage)
            return result
        finally:
            self._close_raw(connection)

    def list_fields(
        self,
        ecn_row_id: str | Sequence[str],
        query: str,
        page: int,
    ) -> GraphFieldPage:
        """List exact projected JSON field names for one or several ECNs.

        Args:
            ecn_row_id: Trusted canonical ECN occurrence identity, or a sequence
                of them; then each field also reports ``ecn_count``.
            query: Optional case-insensitive fragment of a field name.
            page: One-based page number up to the fixed maximum.

        Returns:
            Fifty grouped field summaries with occurrence counts and one path.
        """
        values, root_term = _root_scope(ecn_row_id)
        self._validate_page(page)
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("field query must be text no longer than 200 characters")
        term = query.strip()
        query_filter = (
            f"\nFILTER(CONTAINS(LCASE(STR(?predicate)), LCASE({Literal(term).n3()})))"
            if term
            else ""
        )
        ecn_count = (
            "\n       (COUNT(DISTINCT ?root) AS ?ecnCount)" if root_term == "?root" else ""
        )
        pattern = values + f"""
?subject <{VOCAB.belongsToEcn}> {root_term} ;
         <{VOCAB.jsonPointer}> ?pointer ;
         ?predicate ?term .
FILTER(STRSTARTS(STR(?predicate), "{FIELD_NS}")){query_filter}
""".strip()
        connection = self._raw_connection()
        try:
            count_row = self._select_row(
                connection,
                f"""SELECT (COUNT(DISTINCT ?predicate) AS ?count) WHERE {{
  GRAPH <{self._graph_iri}> {{
    {pattern}
  }}
}}""",
            )
            try:
                total = int((count_row or {}).get("count", "0"))
            except ValueError as exc:
                raise _safe_graph_read_error() from exc
            offset = (page - 1) * _FIELD_PAGE_SIZE
            rows = self._select_bindings(
                connection,
                f"""SELECT ?predicate (COUNT(*) AS ?occurrences)
       (MIN(STR(?pointer)) AS ?examplePointer){ecn_count}
       (GROUP_CONCAT(STR(?subjectType); SEPARATOR = "|"; MAX_LENGTH = 2000) AS ?subjectTypes)
WHERE {{
  GRAPH <{self._graph_iri}> {{
    {pattern}
    OPTIONAL {{ ?subject a ?subjectType . }}
  }}
}}
GROUP BY ?predicate
ORDER BY LCASE(STR(?predicate))
LIMIT {_FIELD_PAGE_SIZE} OFFSET {offset}""",
            )
            items = tuple(self._field_from_binding(row) for row in rows)
            return GraphFieldPage(
                items=items,
                page=page,
                page_size=_FIELD_PAGE_SIZE,
                total_matches=total,
                has_next=offset + len(items) < total,
            )
        finally:
            self._close_raw(connection)

    @staticmethod
    def _replacement_source_object_pattern(root: str) -> str:
        """Return the fixed selected-ECN source object/link input pattern.

        The two branches mirror the only path-proven affected/resulting arrays.
        They cannot pair or otherwise infer a relationship between their members.
        """
        return f"""
?subject a <{VOCAB.SourceObject}> ;
         <{VOCAB.belongsToEcn}> <{root}> ;
         <{VOCAB.jsonPointer}> ?pointer .
{{
  {{
    ?subject <{VOCAB.role}> ?role .
    ?link a <{VOCAB.AffectedLink}> ;
          <{VOCAB.belongsToEcn}> <{root}> ;
          <{VOCAB.jsonPointer}> ?linkPointer ;
          <{field_predicate("AffectedObjects")}> ?members .
    ?members <{VOCAB.member}> ?subject .
    FILTER(STR(?role) = "affected")
  }}
  UNION
  {{
    ?subject <{VOCAB.role}> ?role .
    ?link a <{VOCAB.ResultingLink}> ;
          <{VOCAB.belongsToEcn}> <{root}> ;
          <{VOCAB.jsonPointer}> ?linkPointer ;
          <{field_predicate("ResultingObjects")}> ?members .
    ?members <{VOCAB.member}> ?subject .
    FILTER(STR(?role) = "resulting")
  }}
}}
OPTIONAL {{ ?subject <{field_predicate("Number")}> ?number . FILTER(isLiteral(?number)) }}
OPTIONAL {{ ?subject <{field_predicate("Name")}> ?name . FILTER(isLiteral(?name)) }}
OPTIONAL {{ ?subject <{field_predicate("Description")}> ?description . FILTER(isLiteral(?description)) }}
OPTIONAL {{ ?subject <{field_predicate("ObjectType")}> ?objectType . FILTER(isLiteral(?objectType)) }}
OPTIONAL {{ ?link <{field_predicate("OemReplacesAffected")}> ?directRelation . FILTER(isLiteral(?directRelation)) }}
""".strip()

    def replacement_source_objects(
        self, ecn_row_id: str, page: int
    ) -> ReplacementSourceObjectPage:
        """Read a fixed page of source objects for local candidate derivation.

        Args:
            ecn_row_id: Trusted canonical selected ECN occurrence identity.
            page: One-based page in the fixed 2,000-object source ceiling.

        Returns:
            At most 100 path-proven affected/resulting source objects with the
            limited scalar fields and link paths needed by the scorer.
        """
        canonical_uuid(ecn_row_id)
        if not 1 <= page <= _MAX_REPLACEMENT_SOURCE_PAGES:
            raise ValueError(
                f"page must be between 1 and {_MAX_REPLACEMENT_SOURCE_PAGES}"
            )
        root = str(root_iri(ecn_row_id))
        pattern = self._replacement_source_object_pattern(root)
        connection = self._raw_connection()
        try:
            count_row = self._select_row(
                connection,
                f"""SELECT (COUNT(DISTINCT ?subject) AS ?count) WHERE {{
  GRAPH <{self._graph_iri}> {{
    {pattern}
  }}
}}""",
            )
            try:
                total = int((count_row or {}).get("count", "0"))
            except ValueError as exc:
                raise _safe_graph_read_error() from exc
            offset = (page - 1) * _REPLACEMENT_SOURCE_PAGE_SIZE
            rows = self._select_bindings(
                connection,
                f"""SELECT ?subject ?role ?pointer ?linkPointer ?number ?name ?description ?objectType ?directRelation
       ?artifact ?sourceSha256 ?projectionVersion
WHERE {{
  GRAPH <{self._graph_iri}> {{
    {pattern}
    <{root}> <{VOCAB.sourceArtifact}> ?artifact ;
             <{VOCAB.sourceSha256}> ?sourceSha256 ;
             <{VOCAB.projectionVersion}> ?projectionVersion .
  }}
}}
ORDER BY STR(?pointer) STR(?subject)
LIMIT {_REPLACEMENT_SOURCE_PAGE_SIZE} OFFSET {offset}""",
            )
            items = tuple(
                self._replacement_source_object_from_binding(row) for row in rows
            )
            return ReplacementSourceObjectPage(
                items=items,
                page=page,
                page_size=_REPLACEMENT_SOURCE_PAGE_SIZE,
                total_matches=total,
                has_next=offset + len(items) < total,
            )
        finally:
            self._close_raw(connection)

    @staticmethod
    def _validated_node_refs(ecn_row_id: str, node_refs: Sequence[str]) -> tuple[str, ...]:
        """Validate up to five selected-root or same-ECN projected node references."""
        if not node_refs or len(node_refs) > 5:
            raise ValueError("node_refs must contain between one and five nodes")
        canonical = canonical_uuid(ecn_row_id)
        root = str(root_iri(canonical))
        node_pattern = re.compile(
            rf"^{re.escape(NODE_NS)}{re.escape(canonical)}:[0-9a-f]{{32}}$"
        )
        normalized = tuple(dict.fromkeys(node_refs))
        if any(
            not isinstance(node_ref, str)
            or (node_ref != root and node_pattern.fullmatch(node_ref) is None)
            for node_ref in normalized
        ):
            raise ValueError("node_refs must be projected references for the selected ECN")
        return normalized

    @staticmethod
    def _predicate_filter(predicates: Sequence[str]) -> str:
        """Return fixed safe predicates for local expansion, never model-authored IRIs."""
        if len(predicates) > 10:
            raise ValueError("predicates must contain at most ten filters")
        iris: list[str] = []
        for predicate in dict.fromkeys(predicates):
            if not isinstance(predicate, str) or not predicate or len(predicate) > 200:
                raise ValueError("predicate filters must be nonempty source field names")
            iris.append(_STRUCTURAL_PREDICATES.get(predicate, str(field_predicate(predicate))))
        allowed = " || ".join(
            [
                f'STRSTARTS(STR(?predicate), "{FIELD_NS}")',
                *(f"?predicate = <{iri}>" for iri in _STRUCTURAL_PREDICATES.values()),
            ]
        )
        selected = (
            f"\nFILTER(?predicate IN ({', '.join(f'<{iri}>' for iri in iris)}))"
            if iris
            else ""
        )
        return f"FILTER({allowed}){selected}"

    def _selected_membership_count(self, connection: Any, node_ref: str, root: str) -> int:
        """Count all memberships only after proving one belongs to the selected root."""
        return self._count(
            connection,
            f"<{node_ref}> <{VOCAB.belongsToEcn}> <{root}> . "
            f"OPTIONAL {{ <{node_ref}> <{VOCAB.belongsToEcn}> ?owner . }}",
        )

    @staticmethod
    def _expansion_union(
        starts: str,
        root: str,
        direction: Literal["outgoing", "incoming", "both"],
        depth: int,
    ) -> str:
        """Build the closed one/two-hop directed edge templates for local expansion."""
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be outgoing, incoming, or both")
        if depth not in {1, 2}:
            raise ValueError("depth must be 1 or 2")
        branches: list[str] = []
        if direction in {"outgoing", "both"}:
            branches.append(
                f"{{ VALUES ?start {{ {starts} }} ?start ?predicate ?term . BIND(?start AS ?subject) }}"
            )
            if depth == 2:
                branches.append(
                    f"{{ VALUES ?start {{ {starts} }} ?start ?hopPredicate ?hop . "
                    f"?hop <{VOCAB.belongsToEcn}> <{root}> . ?hop ?predicate ?term . "
                    f"BIND(?hop AS ?subject) }}"
                )
        if direction in {"incoming", "both"}:
            branches.append(
                f"{{ VALUES ?start {{ {starts} }} ?subject ?predicate ?start . BIND(?start AS ?term) }}"
            )
            if depth == 2:
                branches.append(
                    f"{{ VALUES ?start {{ {starts} }} ?hop ?hopPredicate ?start . "
                    f"?hop <{VOCAB.belongsToEcn}> <{root}> . ?subject ?predicate ?hop . "
                    f"BIND(?hop AS ?term) }}"
                )
        return " UNION ".join(branches)

    def expand_nodes(
        self,
        ecn_row_id: str,
        node_refs: Sequence[str],
        direction: Literal["outgoing", "incoming", "both"],
        predicates: Sequence[str],
        depth: int,
        page: int,
    ) -> GraphExpansionPage:
        """Return membership-checked local source/structural facts around selected nodes.

        Args:
            ecn_row_id: Trusted canonical selected ECN occurrence identity.
            node_refs: One to five root or projected nodes issued by a trusted tool.
            direction: Closed outgoing, incoming, or bidirectional traversal mode.
            predicates: At most ten source-field or fixed structural relation names.
            depth: One or two local graph hops.
            page: One-based page number up to the fixed maximum.

        Returns:
            At most one hundred deterministic local facts and continuation metadata.

        Raises:
            PersistenceStateError: If any supplied node is not uniquely owned by the ECN.
        """
        canonical_uuid(ecn_row_id)
        self._validate_page(page)
        refs = self._validated_node_refs(ecn_row_id, node_refs)
        if len(predicates) > 10:
            raise ValueError("predicates must contain at most ten filters")
        if depth not in {1, 2}:
            raise ValueError("depth must be 1 or 2")
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction must be outgoing, incoming, or both")
        root = str(root_iri(ecn_row_id))
        connection = self._raw_connection()
        try:
            for node_ref in refs:
                if self._selected_membership_count(connection, node_ref, root) != 1:
                    raise PersistenceStateError(
                        "RDF_NODE_OUTSIDE_SELECTED_ECN",
                        "A graph node is not available in the selected ECN.",
                        status_code=409,
                    )
            starts = " ".join(f"<{node_ref}>" for node_ref in refs)
            pattern = f"""
{self._expansion_union(starts, root, direction, depth)}
?subject <{VOCAB.belongsToEcn}> <{root}> .
?subject <{VOCAB.jsonPointer}> ?pointer .
{self._predicate_filter(predicates)}
""".strip()
            result = self._read_fact_page(
                connection,
                root,
                pattern,
                page,
                _EXPANSION_PAGE_SIZE,
                GraphExpansionPage,
            )
            assert isinstance(result, GraphExpansionPage)
            return result
        finally:
            self._close_raw(connection)

    def ensure_capabilities(self) -> None:
        """Prove fixed-graph create, write, read, and delete capability at startup."""
        connection = self._raw_connection()
        try:
            self._ensure_graph(connection)
            self._execute(
                connection,
                f"INSERT DATA {{ GRAPH <{self._graph_iri}> {{ <{_PROBE_IRI}> <{VOCAB.probe}> \"1\" . }} }}",
            )
            connection.commit()
            if not self._ask(
                connection,
                f"ASK {{ GRAPH <{self._graph_iri}> {{ <{_PROBE_IRI}> <{VOCAB.probe}> \"1\" . }} }}",
            ):
                raise PersistenceStateError(
                    "RDF_CAPABILITY_UNAVAILABLE",
                    "The RDF projection service could not verify graph read capability.",
                    status_code=500,
                )
            self._execute(
                connection,
                f"DELETE DATA {{ GRAPH <{self._graph_iri}> {{ <{_PROBE_IRI}> <{VOCAB.probe}> \"1\" . }} }}",
            )
            connection.commit()
            if self._ask(
                connection,
                f"ASK {{ GRAPH <{self._graph_iri}> {{ <{_PROBE_IRI}> <{VOCAB.probe}> \"1\" . }} }}",
            ):
                raise PersistenceStateError(
                    "RDF_CAPABILITY_UNAVAILABLE",
                    "The RDF projection service could not verify graph cleanup capability.",
                    status_code=500,
                )
        finally:
            self._close_raw(connection)

    def get_projection(self, ecn_row_id: str) -> ProjectionRecord | None:
        """Return the durable projection status for one selected ECN, if present."""
        canonical_uuid(ecn_row_id)
        table = quote_identifier(self._tables.rdf_projection)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(f"SELECT * FROM {table} WHERE \"ECN_ROW_ID\" = :ecn_row_id"),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
        return _projection_record_from_row(row) if row is not None else None

    def list_reconcilable(self, projection_version: str) -> tuple[str, ...]:
        """Return ECNs missing, stale, failed, pending, or built by another mapper version."""
        ecns = quote_identifier(self._tables.ecn)
        statuses = quote_identifier(self._tables.rdf_projection)
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    f"SELECT E.\"ECN_ROW_ID\", E.\"SOURCE_JSON\", P.\"SOURCE_SHA256\", "
                    f"P.\"PROJECTION_VERSION\", P.\"STATUS\" FROM {ecns} E "
                    f"LEFT JOIN {statuses} P ON P.\"ECN_ROW_ID\" = E.\"ECN_ROW_ID\" "
                    "ORDER BY E.\"ECN_ROW_ID\""
                )
            ).mappings().all()
        result = []
        for raw in rows:
            row = normalize_row_keys(raw)
            source_hash = canonical_source_sha256(_source_json(row["SOURCE_JSON"]))
            if (
                row.get("STATUS") != ProjectionState.CURRENT.value
                or row.get("PROJECTION_VERSION") != projection_version
                or row.get("SOURCE_SHA256") != source_hash
            ):
                result.append(str(row["ECN_ROW_ID"]))
        return tuple(result)

    def load_source(self, ecn_row_id: str) -> ProjectionSource | None:
        """Load the complete authoritative ECN JSON and artifact identity for rebuild."""
        canonical_uuid(ecn_row_id)
        ecns = quote_identifier(self._tables.ecn)
        uploads = quote_identifier(self._tables.upload)
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    f"SELECT E.\"ECN_ROW_ID\", E.\"SOURCE_JSON\", U.\"ECN_ARTIFACT_ID\" "
                    f"FROM {ecns} E JOIN {uploads} U ON U.\"UPLOAD_ID\" = E.\"UPLOAD_ID\" "
                    "WHERE E.\"ECN_ROW_ID\" = :ecn_row_id"
                ),
                {"ecn_row_id": ecn_row_id},
            ).mappings().one_or_none()
        if row is None:
            return None
        values = normalize_row_keys(row)
        try:
            return ProjectionSource(
                ecn_row_id=str(values["ECN_ROW_ID"]),
                artifact_id=str(values["ECN_ARTIFACT_ID"]),
                source_json=_source_json(values["SOURCE_JSON"]),
            )
        except (KeyError, TypeError) as exc:
            raise PersistenceStateError(
                "PERSISTED_JSON_INVALID",
                "The stored ECN source JSON is malformed.",
                status_code=500,
            ) from exc

    def mark_pending(self, ecn_row_id: str, attempted_at: datetime) -> None:
        """Mark one existing projection ready for a new post-commit graph attempt."""
        self._update_status(
            ecn_row_id,
            "STATUS = :status, VERIFIED_NODE_COUNT = NULL, VERIFIED_TRIPLE_COUNT = NULL, "
            "ERROR_CODE = NULL, ERROR_MESSAGE = NULL, LAST_ATTEMPT_AT = :attempted_at, CURRENT_AT = NULL",
            {"status": ProjectionState.PENDING.value, "attempted_at": to_hana_timestamp(attempted_at)},
        )

    def mark_current(
        self,
        ecn_row_id: str,
        verified_node_count: int,
        verified_triple_count: int,
        current_at: datetime,
    ) -> None:
        """Persist successful graph verification counts and the current lifecycle state."""
        self._update_status(
            ecn_row_id,
            "STATUS = :status, VERIFIED_NODE_COUNT = :verified_node_count, "
            "VERIFIED_TRIPLE_COUNT = :verified_triple_count, ERROR_CODE = NULL, "
            "ERROR_MESSAGE = NULL, LAST_ATTEMPT_AT = :current_at, CURRENT_AT = :current_at",
            {
                "status": ProjectionState.CURRENT.value,
                "verified_node_count": verified_node_count,
                "verified_triple_count": verified_triple_count,
                "current_at": to_hana_timestamp(current_at),
            },
        )

    def mark_failed(
        self,
        ecn_row_id: str,
        error_code: str,
        error_message: str,
        attempted_at: datetime,
    ) -> None:
        """Persist a bounded safe graph failure without changing relational source facts."""
        self._update_status(
            ecn_row_id,
            "STATUS = :status, VERIFIED_NODE_COUNT = NULL, VERIFIED_TRIPLE_COUNT = NULL, "
            "ERROR_CODE = :error_code, ERROR_MESSAGE = :error_message, "
            "LAST_ATTEMPT_AT = :attempted_at, CURRENT_AT = NULL",
            {
                "status": ProjectionState.FAILED.value,
                "error_code": error_code[:80],
                "error_message": error_message[:500],
                "attempted_at": to_hana_timestamp(attempted_at),
            },
        )

    def _update_status(self, ecn_row_id: str, assignments: str, params: dict[str, object]) -> None:
        """Apply one parameterized lifecycle update to an existing status record."""
        canonical_uuid(ecn_row_id)
        table = quote_identifier(self._tables.rdf_projection)
        with self._engine.begin() as connection:
            connection.execute(
                text(f"UPDATE {table} SET {assignments} WHERE \"ECN_ROW_ID\" = :ecn_row_id"),
                {**params, "ecn_row_id": ecn_row_id},
            )

    def replace_graph_projection(self, projection: RdfProjection) -> None:
        """Replace only one selected ECN's owned triples using sorted 400-line chunks."""
        root = str(root_iri(projection.ecn_row_id))
        connection = self._raw_connection()
        try:
            self._ensure_graph(connection)
            self._execute(
                connection,
                f"DELETE {{ GRAPH <{self._graph_iri}> {{ ?s ?p ?o }} }} "
                f"WHERE {{ GRAPH <{self._graph_iri}> {{ ?s <{VOCAB.belongsToEcn}> <{root}> . ?s ?p ?o }} }}",
            )
            connection.commit()
            for chunk in serialize_ntriples_chunks(projection.graph, max_triples=400):
                self._execute(
                    connection,
                    f"INSERT DATA {{ GRAPH <{self._graph_iri}> {{\n{chunk}\n}} }}",
                )
                connection.commit()
        finally:
            self._close_raw(connection)

    def verify_graph_projection(self, projection: RdfProjection) -> ProjectionVerification:
        """Read graph provenance and structure independently of root metadata counts."""
        root = str(root_iri(projection.ecn_row_id))
        connection = self._raw_connection()
        try:
            metadata = self._select_row(
                connection,
                f"SELECT ?sourceSha256 ?projectionVersion WHERE {{ GRAPH <{self._graph_iri}> {{ "
                f"<{root}> <{VOCAB.sourceSha256}> ?sourceSha256 ; "
                f"<{VOCAB.projectionVersion}> ?projectionVersion . }} }}",
            )
            objects = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; a <{VOCAB.Object}> .",
            )
            arrays = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; a <{VOCAB.Array}> .",
            )
            nulls = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; a <{VOCAB.NullValue}> .",
            )
            wrappers = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; a <{VOCAB.ScalarValue}> .",
            )
            direct_scalars = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; ?field ?value . "
                f"FILTER(isLiteral(?value) && STRSTARTS(STR(?field), \"{FIELD_NS}\"))",
            )
            members = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> .",
            )
            invalid_indices = self._invalid_array_member_count(connection, root)
            cross_membership = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> ; <{VOCAB.belongsToEcn}> ?other . "
                f"FILTER(?other != <{root}>)",
            )
            triples = self._count(
                connection,
                f"?node <{VOCAB.belongsToEcn}> <{root}> . ?node ?predicate ?object .",
            )
            return ProjectionVerification(
                source_sha256=(metadata or {}).get("sourceSha256"),
                projection_version=(metadata or {}).get("projectionVersion"),
                counts=ProjectionCounts(
                    object_count=objects,
                    array_count=arrays,
                    scalar_count=direct_scalars + wrappers,
                    null_count=nulls,
                    node_count=members,
                    triple_count=triples,
                ),
                invalid_array_member_count=invalid_indices,
                cross_ecn_membership_count=cross_membership,
            )
        finally:
            self._close_raw(connection)

    def delete_ecn_graphs(self, ecn_row_ids: Sequence[str]) -> None:
        """Remove every triple owned by the given ECNs, leaving all other ECNs intact.

        Uses the same ``belongsToEcn`` scoped DELETE as ``replace_graph_projection``,
        committed per ECN so one failure does not roll back earlier cleanups.

        Args:
            ecn_row_ids: ECN row IDs already deleted relationally.
        """
        connection = self._raw_connection()
        try:
            for ecn_row_id in ecn_row_ids:
                root = str(root_iri(ecn_row_id))
                self._execute(
                    connection,
                    f"DELETE {{ GRAPH <{self._graph_iri}> {{ ?s ?p ?o }} }} "
                    f"WHERE {{ GRAPH <{self._graph_iri}> {{ ?s <{VOCAB.belongsToEcn}> <{root}> . ?s ?p ?o }} }}",
                )
                connection.commit()
        finally:
            self._close_raw(connection)

    def clear_application_graph(self) -> None:
        """Drop only the application-owned graph during a successful history reset."""
        connection = self._raw_connection()
        try:
            try:
                self._execute(connection, f"DROP GRAPH <{self._graph_iri}>")
                connection.commit()
            except PersistenceStateError as exc:
                if "exist" not in str(exc.__cause__ or "").lower():
                    raise exc
        finally:
            self._close_raw(connection)
