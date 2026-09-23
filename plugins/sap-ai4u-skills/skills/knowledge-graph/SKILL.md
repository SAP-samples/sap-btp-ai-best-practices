---
name: knowledge-graph
description: Build Knowledge Graph RAG workflows on SAP BTP using Gen AI Hub and SAP HANA KG/SPARQL. Use when tasks require extracting RDF-like triples from domain documents, storing/querying them in HANA, filtering relations by business questions, and generating grounded summaries or comparisons.
---

# Knowledge Graph RAG with SAP HANA

Use this skill to implement KG extraction, storage, and grounded reasoning.

## Set Required Environment Variables

This workflow uses lower-case HANA keys in the existing notebook:

```bash
hana_address=""
hana_port=""
hana_user=""
hana_password=""
```

Also configure AICore/Gen AI Hub variables required by `sap-ai-sdk-gen`.

## Extract Triples from PDF with Gen AI Hub LLM

```python
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

proxy_client = get_proxy_client("gen-ai-hub")
chat_llm = ChatOpenAI(proxy_model_name="gpt-4.1", proxy_client=proxy_client, temperature=0.0)

# Simple, self-contained extraction prompt. Ask for one triple per line as
# "subject | predicate | object" so the response is trivial to parse.
KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = (
    "Extract knowledge-graph triples from the text below. "
    "Return one triple per line as: subject | predicate | object. "
    "Use only facts stated in the text.\n\nText:\n{text}"
)

docs = PyPDFLoader("example.pdf").load()
doc_content = " ".join(d.page_content for d in docs)

response = chat_llm.invoke(KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT.format(text=doc_content))

triples = [
    tuple(part.strip() for part in line.split("|"))
    for line in response.content.splitlines()
    if line.count("|") == 2
]
```

## Bulk Insert Triples into SAP HANA KG

```python
from hdbcli import dbapi

conn = dbapi.connect(address=HANA_ADDRESS, port=HANA_PORT, user=HANA_USER, password=HANA_PASSWORD)
cursor = conn.cursor()

sparql_bulk_insert = f"""
INSERT DATA {{
  GRAPH <http://graph/KG_pdf_tmp_01> {{
    <http://example.com/resource/MotorA> <http://example.com/property/has%20power%20rating> "1500 HP" .
  }}
}}
""".strip()

cursor.callproc("SYS.SPARQL_EXECUTE", [sparql_bulk_insert, "", None, None])
conn.commit()
cursor.close()
conn.close()
```

## Query and Filter for Business Intent

Use these operational helpers:

- `get_unique_relations(...)` to inspect available KG relations.
- `filter_relations_by_business_question(...)` to select relevant relations with LLM.
- `get_triples_by_relations(...)` to fetch focused evidence.
- Build final summary prompt using only filtered triples.


## Apply Best Practices

- Keep triple extraction deterministic with low temperature (`0.0`).
- Normalize entity IDs and escape literals before SPARQL insert.
- Bulk insert triples in one statement for performance.
- Use separate graph names per scenario and clear them after demo runs.
- Select only business-relevant relations before summarization to reduce hallucination.

## Validate with Expected Outputs

Healthy run patterns:

- A triple list like `[("MAC 1-1", "is a", "Equipment"), ...]`.
- Graph visuals produced (`KG_simple.svg`, `KG_advanced.png`).
- Insert log similar to `Bulk insert of <N> triples successful.`
- Relation filter output similar to `Relevant relations: ['has efficiency rating', ...]`.
- Final summary/comparison text grounded in filtered triples.

## Related Skills

- `sap-hana-sql` — HANA dialect rules for any relational tables beside the graph.
- `access-to-generative-ai-models` — LLM call patterns used for extraction/summarization.
- `sap-btp-ai` — routing and shared environment conventions.
