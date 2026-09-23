# HANA Knowledge-Graph RAG

## Choose this route

Use KG RAG when answers depend on explicit entities, relations, identifiers, graph constraints, joins, or provenance.

```text
ontology + instance triples -> named HANA RDF graphs
question + schema -> SPARQL -> HANA graph result -> grounded answer
```

The LangChain-specific path is `HanaRdfGraph` plus `HanaSparqlQAChain`. Keep deterministic SPARQL helpers for fixed business queries and writes.

## Graph preparation

Use stable named graphs for separate concerns:

```python
DATA_GRAPH = "https://example.com/kg/data"
ONTOLOGY_GRAPH = "https://example.com/kg/ontology"
PROVENANCE_GRAPH = "https://example.com/kg/provenance"
```

Keep ontology triples separate from instance data. Include class declarations, labels, comments, domains, ranges, and relationship descriptions.

Execute SPARQL through HANA:

```python
def sparql_raw(connection, query: str, accept: str):
    """Execute SPARQL and return the response body and metadata."""
    cursor = connection.cursor()
    try:
        result = cursor.callproc("SYS.SPARQL_EXECUTE", (query, accept, "?", "?"))
        return result[2], result[3] if len(result) > 3 else None
    finally:
        cursor.close()

def sparql_update(connection, update: str):
    """Execute a controlled SPARQL update."""
    return sparql_raw(connection, update, "")
```

For RDF ingestion, parse the user-provided Turtle/RDF, serialize to N-Triples, and insert bounded chunks with:

```sparql
INSERT DATA {
  GRAPH <https://example.com/kg/data> {
    ...triples...
  }
}
```

Create graphs explicitly before loading them. Keep chunk size configurable.

## Ontology-aware graph wrapper

An explicit ontology query is the most portable schema source:

```python
from langchain_hana import HanaRdfGraph

ontology_query = f"""
CONSTRUCT {{ ?s ?p ?o . }}
FROM <{ONTOLOGY_GRAPH}>
WHERE {{ ?s ?p ?o . }}
"""

graph = HanaRdfGraph(
    connection=connection,
    graph_uri=DATA_GRAPH,
    ontology_query=ontology_query,
)

schema = graph.get_schema
print(schema.serialize(format="turtle"))
```

Supported schema sources are `ontology_query`, `ontology_uri`, `ontology_local_file` with its format, and `auto_extract_ontology=True`. Prefer explicit ontology data; auto-extraction is exploratory and may omit useful labels and subclass relationships.

Useful operations include:

```python
graph.query(sparql, inject_from_clause=True)
graph.inject_from_clause(sparql)
graph.refresh_schema(ontology_query=ontology_query)
graph.get_schema
```

## SPARQL QA chain

Use the chain when a question should be translated into SPARQL and rewritten into natural language:

```python
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_hana import HanaSparqlQAChain

proxy_client = get_proxy_client("gen-ai-hub")
llm = ChatOpenAI(
    proxy_model_name="gpt-4.1",
    proxy_client=proxy_client,
    temperature=0.0,
)

chain = HanaSparqlQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

result = chain.invoke(
    "Which high-risk suppliers provide Brake Pad Set to Plant 1000?"
)
print(result["result"])
```

The chain generates a `SELECT`, executes it against the graph, and sends the result context through a second answer prompt. Use custom prompts when the ontology has domain-specific constraints, but keep generated queries read-only.

## Deterministic operations

Do not use the QA chain for fixed contracts. Write explicit SPARQL for:

- `ASK` existence checks;
- exact `SELECT` joins;
- `CONSTRUCT` context extraction;
- graph creation and cleanup;
- controlled risk or master-data updates;
- provenance writes.

Validate user-controlled values before interpolation. Use RDFLib `Literal(...).n3()` for literals and percent-encoded URI builders for external identifiers. Never interpolate arbitrary graph URIs, predicates, or raw user strings into updates.

For agent observations, record the subject, observed property, value, source, confidence, and timestamp in a provenance graph.

## Safety and validation

`allow_dangerous_requests=True` is an acknowledgement, not a safety control. Use a least-privilege database user, inspect generated SPARQL during development, and reject non-`SELECT` statements in read-only paths.

Validate a direct `ASK`/`SELECT`, ontology schema content, generated SPARQL, known-answer parity, absent-evidence behavior, and read-only rejection of `INSERT`, `DELETE`, `DROP`, and `CREATE` statements.
