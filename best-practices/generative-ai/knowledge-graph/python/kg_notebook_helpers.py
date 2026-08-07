"""Small, testable helpers used by the HANA Knowledge Graph notebook."""

from __future__ import annotations

from urllib.parse import quote, unquote, urlparse
from uuid import uuid4


GRAPH_RUN_NAMESPACE = "https://kg-demo.example/runs"


def create_run_id() -> str:
    """Create a unique identifier for one notebook execution.

    Returns:
        A lowercase UUID without separators, suitable for one graph namespace.
    """
    return uuid4().hex


def build_run_graph_iri(run_id: str, graph_name: str) -> str:
    """Build a safe, run-isolated named graph IRI.

    Args:
        run_id: Identifier generated once for a notebook execution.
        graph_name: Human-readable scenario label within the execution.

    Returns:
        A bracketed SPARQL IRI that can be used in GRAPH and FROM clauses.

    Raises:
        ValueError: If either identifier is empty after trimming whitespace.
    """
    if not run_id.strip() or not graph_name.strip():
        raise ValueError("run_id and graph_name must be non-empty")

    encoded_run_id = quote(run_id.strip(), safe="")
    encoded_graph_name = quote(graph_name.strip(), safe="")
    return f"<{GRAPH_RUN_NAMESPACE}/{encoded_run_id}/{encoded_graph_name}>"


def predicate_label(predicate_iri: str) -> str:
    """Convert a predicate IRI into a readable label for an LLM prompt.

    Args:
        predicate_iri: Absolute predicate IRI returned by a SPARQL query.

    Returns:
        The decoded final IRI path or fragment component.

    Raises:
        ValueError: If the value is not an absolute IRI.
    """
    normalized_iri = _normalize_absolute_iri(predicate_iri)
    local_name = normalized_iri.rsplit("#", maxsplit=1)[-1].rsplit("/", maxsplit=1)[-1]
    return unquote(local_name).replace("_", " ")


def build_predicate_select_query(
    graph_iri: str,
    predicate_iris: list[str],
) -> str | None:
    """Build a SPARQL_TABLE query restricted to selected predicate IRIs.

    Args:
        graph_iri: Bracketed named-graph IRI created for the current run.
        predicate_iris: Absolute predicate IRIs selected from graph metadata.

    Returns:
        SQL that reads matching triples, or ``None`` when no predicate is selected.

    Raises:
        ValueError: If the graph or a predicate IRI is malformed.
    """
    if not predicate_iris:
        return None

    _validate_bracketed_iri(graph_iri)
    values = " ".join(
        f"<{_normalize_absolute_iri(predicate_iri)}>"
        for predicate_iri in predicate_iris
    )
    return f"""
    SELECT "s", "p", "o"
    FROM SPARQL_TABLE(
      'SELECT ?s ?p ?o
       FROM {graph_iri}
       WHERE {{
         ?s ?p ?o .
         VALUES ?p {{ {values} }}
       }}'
    ) AS "result"
    """.strip()


def _normalize_absolute_iri(value: str) -> str:
    """Validate an absolute IRI and return it without surrounding brackets."""
    normalized_value = value.strip().removeprefix("<").removesuffix(">")
    parsed_value = urlparse(normalized_value)
    if (
        not normalized_value
        or not parsed_value.scheme
        or any(character.isspace() or character in '<>"\'' for character in normalized_value)
    ):
        raise ValueError(f"Invalid absolute IRI: {value!r}")
    return normalized_value


def _validate_bracketed_iri(value: str) -> None:
    """Validate the bracketed IRI format accepted by SPARQL GRAPH clauses."""
    if not value.startswith("<") or not value.endswith(">"):
        raise ValueError(f"Expected a bracketed graph IRI, received: {value!r}")
    _normalize_absolute_iri(value)
