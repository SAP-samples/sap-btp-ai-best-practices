---
name: hana-rag-operations
description: Use when creating, adapting, or troubleshooting SAP HANA RAG with LangChain, including HANA vector-store RAG, HanaDB, retrievers, reranking, RDF/SPARQL knowledge-graph RAG, HanaRdfGraph, or HanaSparqlQAChain.
---

# SAP HANA RAG Operations

Create portable Python RAG implementations backed by SAP HANA Cloud. Route the request first: use vector-store RAG for unstructured document similarity, and KG RAG for structured entities, relations, constraints, provenance, and exact graph questions.

## Route the request

| Request | Pattern | Read |
|---|---|---|
| “vector store RAG”, “document RAG”, “semantic search” | `HanaDB` + retriever + standard LangChain retrieval chain | `references/vector-store-rag.md` |
| “KG RAG”, “knowledge graph QA”, “SPARQL QA” | `HanaRdfGraph` + `HanaSparqlQAChain` | `references/kg-rag.md` |
| “create an ontology”, “build a KG”, “extract a KG from PDF” | Ontology-first, deterministic mapping, or document triple extraction | `references/kg-authoring.md` |
| “which operation should I use?” | Compare ingestion, retrieval, reranking, and QA options | `references/operations-matrix.md` |

Use both patterns when the answer needs semantic document evidence and exact master-data relationships; keep their retrieval outputs explicit rather than hiding one behind the other.

## Required conventions

1. Inspect the target project’s installed versions before choosing imports. Current LangChain 1.x uses `langchain_classic.chains` for `create_retrieval_chain` and `create_stuff_documents_chain`; older projects may expose those APIs under `langchain.chains`.
2. Load credentials before constructing SAP clients. In notebooks or long-lived processes, use `load_dotenv(".env", override=True)` so stale process variables do not silently win.
3. Use the SAP Gen AI Hub wrappers for model access: `gen_ai_hub.proxy.langchain.openai.ChatOpenAI` and `OpenAIEmbeddings`, or the native SDK when LangChain is not needed.
4. Keep model names configurable. Use a chat-completions-compatible model for standard LangChain retrieval chains; use Responses-compatible configuration only when the surrounding chain supports it.
5. Store documents, embeddings, graph triples, and tabular metadata in HANA. Treat local files only as explicit user-provided ingestion inputs.
6. Scope database credentials narrowly. `HanaSparqlQAChain` can execute generated requests; set `allow_dangerous_requests=True` only after acknowledging that risk and using a restricted database user.
7. Return sources. Vector RAG should expose retrieved `Document` objects; KG RAG should expose the generated SPARQL, graph scope, and query result when the application needs auditability.

## Use the bundled templates

Adapt the smallest matching script instead of rewriting a notebook:

- `scripts/vector_store_rag.py` — ingest JSONL documents into `HanaDB`, retrieve them, and answer with a standard LangChain retrieval chain.
- `scripts/kg_rag.py` — optionally recreate named RDF graphs from Turtle files, build the ontology-aware graph wrapper, and answer through `HanaSparqlQAChain`.
- `scripts/extract_pdf_graph.py` — extract raw triples from a PDF with `PyPDFLoader` and `GraphIndexCreator`, then persist them in a named HANA graph for exploration or review.

Each script has a module docstring with runnable commands. Copy only the functions needed into the target project, preserving their docstrings and validation boundaries.

## Validate before handoff

- Confirm the HANA connection independently.
- Confirm the embedding deployment and vector dimension before ingestion.
- Run one direct retrieval query and inspect the returned documents.
- Run one grounded question and verify the answer is supported by retrieved context or graph results.
- For KG RAG, inspect the generated SPARQL before enabling any write-capable path.
- Keep a smoke test that exercises the chosen retrieval and answer path without requiring a notebook kernel.

For detailed API choices and portable examples, read the linked references only when that route is selected.
