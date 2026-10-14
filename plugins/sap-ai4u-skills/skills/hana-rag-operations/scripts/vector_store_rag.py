#!/usr/bin/env python3
"""Create and query a portable SAP HANA vector-store RAG application.

Examples:
    python vector_store_rag.py --documents-jsonl documents.jsonl \
        --table-name DOCUMENT_CHUNKS \
        --question "What is the warranty period?"
    python vector_store_rag.py --table-name DOCUMENT_CHUNKS \
        --question "Which documents mention delayed delivery?"

Each JSONL document must contain ``page_content`` and may contain ``metadata``.
Required packages include hdbcli, python-dotenv, tqdm, sap-ai-sdk-gen[all],
langchain-hana, langchain-core, and the installed LangChain retrieval helpers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from dotenv import load_dotenv
from hdbcli import dbapi
from tqdm.auto import tqdm

from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_hana import HanaDB
from langchain_hana.utils import DistanceStrategy


def env_value(name: str, fallback: str | None = None) -> str | None:
    """Read an uppercase environment variable with a lowercase fallback."""
    return os.getenv(name) or os.getenv(name.lower()) or fallback


def connect_hana():
    """Create an encrypted HANA connection from environment variables.

    Returns:
        An authenticated ``hdbcli`` connection.
    """
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


def load_jsonl_documents(path: Path) -> list[Document]:
    """Load LangChain documents from a JSONL file.

    Args:
        path: JSONL file whose records contain ``page_content`` and optional
            ``metadata``.

    Returns:
        Parsed LangChain documents.

    Raises:
        ValueError: If a record is missing text or has invalid metadata.
    """
    documents: list[Document] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(
            tqdm(stream, desc="Reading documents", unit="doc"), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            page_content = record.get("page_content")
            metadata = record.get("metadata", {})
            if not isinstance(page_content, str) or not page_content.strip():
                raise ValueError(f"Line {line_number}: page_content must be non-empty text")
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {line_number}: metadata must be an object")
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def batches(items: list[Document], size: int) -> Iterator[list[Document]]:
    """Yield bounded document batches for database ingestion."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_vector_store(
    connection,
    proxy_client,
    table_name: str,
    embedding_model_name: str,
    distance_strategy: str,
) -> HanaDB:
    """Create a HANA vector store using a Gen AI Hub embedding deployment."""
    embedding_model = OpenAIEmbeddings(
        proxy_model_name=embedding_model_name,
        proxy_client=proxy_client,
    )
    return HanaDB(
        connection=connection,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy[distance_strategy],
        table_name=table_name,
    )


def build_retrieval_chain(retriever, proxy_client, chat_model_name: str):
    """Build a grounded LangChain retrieval chain.

    Returns:
        A runnable that accepts ``{"input": question}`` and returns an answer
        plus the retrieved ``context`` documents.
    """
    try:
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )
    except ImportError:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

    llm = ChatOpenAI(
        proxy_model_name=chat_model_name,
        proxy_client=proxy_client,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied context. "
                "If the context is insufficient, say so.",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def ingest_documents(vector_store: HanaDB, documents: list[Document], batch_size: int) -> None:
    """Add documents to HANA in bounded batches with progress reporting."""
    for batch in tqdm(
        list(batches(documents, batch_size)), desc="Embedding and storing", unit="batch"
    ):
        vector_store.add_documents(batch)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for ingestion and querying."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Grounded question to answer")
    parser.add_argument("--documents-jsonl", type=Path, help="Optional JSONL corpus to ingest")
    parser.add_argument("--table-name", default="DOCUMENT_CHUNKS")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--chat-model", default="gpt-4.1")
    parser.add_argument(
        "--distance-strategy",
        choices=[member.name for member in DistanceStrategy],
        default="COSINE",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Load optional documents, retrieve context, and print a grounded answer."""
    args = parse_args()
    if args.batch_size <= 0 or args.k <= 0:
        raise ValueError("batch size and k must be positive")

    load_dotenv(".env", override=True)
    connection = connect_hana()
    proxy_client = get_proxy_client("gen-ai-hub")
    vector_store = build_vector_store(
        connection=connection,
        proxy_client=proxy_client,
        table_name=args.table_name,
        embedding_model_name=args.embedding_model,
        distance_strategy=args.distance_strategy,
    )

    if args.documents_jsonl:
        documents = load_jsonl_documents(args.documents_jsonl)
        ingest_documents(vector_store, documents, args.batch_size)

    retriever = vector_store.as_retriever(search_kwargs={"k": args.k})
    chain = build_retrieval_chain(retriever, proxy_client, args.chat_model)
    result = chain.invoke({"input": args.question})
    print(result["answer"])
    print(f"Retrieved documents: {len(result.get('context', []))}")


if __name__ == "__main__":
    main()
