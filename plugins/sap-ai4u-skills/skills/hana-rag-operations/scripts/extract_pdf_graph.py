#!/usr/bin/env python3
"""Extract raw document triples from a PDF and store them in SAP HANA.

Examples:
    python extract_pdf_graph.py --pdf specification.pdf \
        --graph-uri https://example.com/kg/raw-extraction \
        --reset-graph
    python extract_pdf_graph.py --pdf specification.pdf \
        --graph-uri https://example.com/kg/raw-extraction \
        --entity-base https://example.com/kg/resource/ \
        --property-base https://example.com/kg/property/

This script uses PyPDFLoader and GraphIndexCreator for exploratory document
extraction. It creates a raw graph, not a reviewed production ontology. Review
the extracted triples and map them to a controlled ontology before KG RAG.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from hdbcli import dbapi
from langchain_core.prompts import PromptTemplate
from tqdm.auto import tqdm

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI


@dataclass(frozen=True)
class ExtractedTriple:
    """A raw document fact in conventional subject-predicate-object order.

    Attributes:
        subject: Extracted entity being described.
        predicate: Extracted relation label.
        object_value: Extracted literal value or target entity text.
        page: One-based source PDF page number.
    """

    subject: str
    predicate: str
    object_value: str
    page: int


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


def sparql_update(connection, update: str) -> None:
    """Execute one controlled SPARQL update against HANA."""
    cursor = connection.cursor()
    try:
        cursor.callproc("SYS.SPARQL_EXECUTE", (update, "", "?", "?"))
    finally:
        cursor.close()


def recreate_graph(connection, graph_uri: str) -> None:
    """Drop and recreate a named graph after explicit user approval."""
    try:
        sparql_update(connection, f"DROP GRAPH <{graph_uri}>")
    except Exception:
        pass
    sparql_update(connection, f"CREATE GRAPH <{graph_uri}>")


def build_extraction_prompt() -> PromptTemplate:
    """Return a GraphIndexCreator-compatible prompt for raw document facts."""
    from langchain_community.graphs.networkx_graph import KG_TRIPLE_DELIMITER

    return PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract factual knowledge triples from the technical text. "
            "Use the exact format (subject, predicate, object). "
            "Do not invent facts or units. Separate triples with "
            f"`{KG_TRIPLE_DELIMITER}`.\n\n"
            "TEXT:\n{text}\n\nOutput:"
        ),
    )


def extract_page_triples(pdf_path: Path, llm) -> list[ExtractedTriple]:
    """Extract raw triples from each PDF page using GraphIndexCreator.

    Returns:
        Extracted facts annotated with the source page. The community graph
        returns triples as ``(subject, object, predicate)``; this function
        converts them to conventional subject-predicate-object order.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.graphs.index_creator import GraphIndexCreator
    except ImportError as error:
        raise RuntimeError(
            "Install langchain-community, networkx, and pypdf to use PDF graph extraction"
        ) from error

    documents = PyPDFLoader(str(pdf_path)).load()
    index_creator = GraphIndexCreator(llm=llm)
    prompt = build_extraction_prompt()
    extracted: list[ExtractedTriple] = []

    for document in tqdm(documents, desc="Extracting PDF triples", unit="page"):
        page_text = document.page_content.strip()
        if not page_text:
            continue
        raw_graph = index_creator.from_text(text=page_text, prompt=prompt)
        page = int(document.metadata.get("page", 0)) + 1
        for subject, object_value, predicate in raw_graph.get_triples():
            extracted.append(
                ExtractedTriple(
                    subject=subject,
                    predicate=predicate,
                    object_value=object_value,
                    page=page,
                )
            )
    return extracted


def iri(base: str, value: str) -> str:
    """Return a safely percent-encoded IRI for an extracted text value."""
    return f"<{base.rstrip('/')}/{quote(value.strip(), safe='')}>"


def literal(value: str) -> str:
    """Return a safely escaped SPARQL string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def store_triples(
    connection,
    graph_uri: str,
    triples: list[ExtractedTriple],
    entity_base: str,
    property_base: str,
    chunk_size: int,
) -> None:
    """Persist raw document facts in bounded HANA SPARQL update batches."""
    statements = [
        f"{iri(entity_base, triple.subject)} "
        f"{iri(property_base, triple.predicate)} "
        f"{literal(triple.object_value)} ."
        for triple in triples
    ]
    for start in tqdm(
        range(0, len(statements), chunk_size), desc="Writing HANA graph", unit="batch"
    ):
        chunk = "\n".join(statements[start : start + chunk_size])
        sparql_update(
            connection,
            f"INSERT DATA {{ GRAPH <{graph_uri}> {{ {chunk} }} }}",
        )


def parse_args() -> argparse.Namespace:
    """Parse document, graph, extraction, and storage options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--graph-uri", required=True)
    parser.add_argument("--entity-base", default="https://example.com/kg/resource")
    parser.add_argument("--property-base", default="https://example.com/kg/property")
    parser.add_argument("--chat-model", default="gpt-4.1")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--reset-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Extract PDF triples and store them in the selected HANA graph."""
    args = parse_args()
    if not args.pdf.is_file():
        raise FileNotFoundError(args.pdf)
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")

    load_dotenv(".env", override=True)
    connection = connect_hana()
    if args.reset_graph:
        recreate_graph(connection, args.graph_uri)

    proxy_client = get_proxy_client("gen-ai-hub")
    llm = ChatOpenAI(
        proxy_model_name=args.chat_model,
        proxy_client=proxy_client,
        temperature=0.0,
    )
    triples = extract_page_triples(args.pdf, llm)
    store_triples(
        connection=connection,
        graph_uri=args.graph_uri,
        triples=triples,
        entity_base=args.entity_base,
        property_base=args.property_base,
        chunk_size=args.chunk_size,
    )
    print(f"Stored {len(triples)} raw triples from {args.pdf.name}.")


if __name__ == "__main__":
    main()
