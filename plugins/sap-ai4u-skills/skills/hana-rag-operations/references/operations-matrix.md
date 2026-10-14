# HANA RAG Operations Matrix

| Operation | Component | Use for | Evidence |
|---|---|---|---|
| Add documents/text | `HanaDB.add_documents`, `add_texts` | Incremental vector ingestion | Stored chunks |
| One-shot ingestion | `HanaDB.from_documents`, `from_texts` | Small initial corpus | `HanaDB` |
| Similarity search | `similarity_search` | Top semantic matches | `Document` list |
| Scores | `similarity_search_with_score` | Thresholds and diagnostics | Documents + scores |
| Metadata filter | `filter=...` | Tenant/type/country/date scope | Filtered documents |
| MMR | `as_retriever(search_type="mmr")` | Diverse context | Diverse documents |
| HANA reranking | `rerank_config` or `HanaReranker` | Better ordering after recall | Reranked documents |
| Vector QA | `create_retrieval_chain` | Classical document RAG | `answer` + `context` |
| Internal embeddings | `HanaInternalEmbeddings` | HANA-managed embedding | HANA vectors |
| Ontology authoring | RDFLib/Turtle ontology-first design | Production classes and predicates | Ontology triples |
| Structured-data mapping | RDFLib mapping functions | Deterministic data-to-RDF conversion | Typed instance triples |
| PDF triple bootstrap | `PyPDFLoader` + `GraphIndexCreator` | Rapid exploratory raw graph | Extracted text triples |
| Typed document extraction | Responses structured output | Schema-constrained document facts | Validated fact objects |
| Graph schema | `HanaRdfGraph` | Ontology-aware access | RDFLib schema |
| Graph read | `HanaRdfGraph.query` | Deterministic SPARQL | Raw response |
| Graph QA | `HanaSparqlQAChain` | Natural-language graph QA | `result` + trace |
| Graph write | `SYS.SPARQL_EXECUTE` | Controlled updates/provenance | Update response |

## Selection

| Question | Prefer |
|---|---|
| “What does this policy say?” | Vector RAG |
| “Find similar maintenance instructions” | Vector RAG |
| “Which suppliers provide material X to plant Y?” | KG RAG |
| “Which records have risk=high after date Z?” | Deterministic SPARQL |
| “Summarize retrieved contract clauses” | Vector RAG with sources |
| “Explain relationships among entities” | KG RAG, optionally augmented with vector evidence |

## Compatibility

- Current `langchain_hana` exports `HanaDB`, `HanaInternalEmbeddings`, `HanaRdfGraph`, `HanaReranker`, and `HanaSparqlQAChain`.
- Current `DistanceStrategy` uses `COSINE`; verify the target package before copying enum names.
- Current LangChain 1.x retrieval helpers are imported from `langchain_classic`.
- `HanaSparqlQAChain` is graph-specific; it is not a vector retriever.
- `HanaDB` retrieves documents but does not generate final answers without a standard LangChain chain.
