#!/usr/bin/env python3
"""Create and query a portable SAP HANA knowledge-graph RAG application.

Examples:
    python kg_rag.py --question "Which suppliers provide material M-1?" \
        --data-graph https://example.com/kg/data \
        --ontology-graph https://example.com/kg/ontology
    python kg_rag.py --ontology-turtle ontology.ttl --data-turtle data.ttl \
        --reset-graphs \
        --question "Which high-risk suppliers serve Plant 1000?"

The Turtle inputs are optional. If supplied, they are loaded into named HANA
graphs in bounded SPARQL INSERT DATA batches before the QA chain runs.
Required packages include hdbcli, python-dotenv, rdflib, sap-ai-sdk-gen[all],
langchain-hana, and LangChain.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from hdbcli import dbapi
from rdflib import Graph
from tqdm.auto import tqdm

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_hana import HanaRdfGraph, HanaSparqlQAChain


def env_value(name: str, fallback: str | None = None) -> str | None:
    """Read an uppercase environment variable with a lowercase fallback."""
    return os.getenv(name) or os.getenv(name.lower()) or fallback


def connect_hana():
    """Create an encrypted HANA connection from environment variables."""
    address = env_value("HANA_ADDRESS")
    user = env_value("HANA_USER")
    password = env_value("HANA_PASSWORD")
    if not all((address, user, password)):
        raise RuntimeError("HANA_ADDRESS, HANA_USER, and HANA_PASSWORD are required")

    encrypt = str(env_value("HANA_ENCRYPT", "true")).lower() in {
        "true",
        "1",
        "yes",
    }
    return dbapi.connect(
        address=address,
        port=int(env_value("HANA_PORT", "443")),
        user=user,
        password=password,
        encrypt=encrypt,
    )


def sparql_raw(connection, query: str, accept: str = ""):
    """Execute SPARQL and return the response body and metadata."""
    cursor = connection.cursor()
    try:
        result = cursor.callproc("SYS.SPARQL_EXECUTE", (query, accept, "?", "?"))
        return result[2], result[3] if len(result) > 3 else None
    finally:
        cursor.close()


def recreate_graph(connection, graph_uri: str) -> None:
    """Drop and recreate a named graph for an explicit reset operation."""
    try:
        sparql_raw(connection, f"DROP GRAPH <{graph_uri}>")
    except Exception:
        # HANA raises when the graph does not exist; creation remains useful.
        pass
    sparql_raw(connection, f"CREATE GRAPH <{graph_uri}>")


def load_turtle(connection, turtle_path: Path, graph_uri: str, chunk_size: int) -> int:
    """Load a Turtle file into a named graph in bounded insert batches.

    Returns:
        Number of inserted triples.
    """
    graph = Graph()
    graph.parse(turtle_path, format="turtle")
    lines = graph.serialize(format="nt")
    if isinstance(lines, bytes):
        lines = lines.decode("utf-8")
    triples = [line for line in lines.splitlines() if line.strip()]

    for start in tqdm(range(0, len(triples), chunk_size), desc=f"Loading {turtle_path.name}", unit="batch"):
        chunk = "\n".join(triples[start : start + chunk_size])
        sparql_raw(
            connection,
            f"INSERT DATA {{ GRAPH <{graph_uri}> {{ {chunk} }} }}",
        )
    return len(triples)


def build_graph(connection, data_graph: str, ontology_graph: str) -> HanaRdfGraph:
    """Build an ontology-aware HANA graph wrapper."""
    ontology_query = f"""
    CONSTRUCT {{ ?s ?p ?o . }}
    FROM <{ontology_graph}>
    WHERE {{ ?s ?p ?o . }}
    """
    return HanaRdfGraph(
        connection=connection,
        graph_uri=data_graph,
        ontology_query=ontology_query,
    )


def build_qa_chain(graph: HanaRdfGraph, proxy_client, chat_model: str):
    """Build a read-focused natural-language-to-SPARQL QA chain."""
    llm = ChatOpenAI(
        proxy_model_name=chat_model,
        proxy_client=proxy_client,
        temperature=0.0,
    )
    return HanaSparqlQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        # This chain requires an explicit acknowledgement. Use a read-only DB user.
        allow_dangerous_requests=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse graph, loading, and question options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--data-graph", required=True)
    parser.add_argument("--ontology-graph", required=True)
    parser.add_argument("--data-turtle", type=Path)
    parser.add_argument("--ontology-turtle", type=Path)
    parser.add_argument("--reset-graphs", action="store_true")
    parser.add_argument("--chat-model", default="gpt-4.1")
    parser.add_argument("--chunk-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    """Optionally load RDF graphs and answer one grounded graph question."""
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if bool(args.data_turtle) != bool(args.ontology_turtle):
        raise ValueError("provide both --data-turtle and --ontology-turtle")

    load_dotenv(".env", override=True)
    connection = connect_hana()
    if args.reset_graphs:
        recreate_graph(connection, args.data_graph)
        recreate_graph(connection, args.ontology_graph)
    if args.data_turtle:
        load_turtle(connection, args.data_turtle, args.data_graph, args.chunk_size)
        load_turtle(connection, args.ontology_turtle, args.ontology_graph, args.chunk_size)

    proxy_client = get_proxy_client("gen-ai-hub")
    graph = build_graph(connection, args.data_graph, args.ontology_graph)
    chain = build_qa_chain(graph, proxy_client, args.chat_model)
    result = chain.invoke(args.question)
    print(result["result"])


if __name__ == "__main__":
    main()
