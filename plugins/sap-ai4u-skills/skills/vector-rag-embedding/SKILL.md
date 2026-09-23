---
name: vector-rag-embedding
description: Implement embedding generation and ingestion into SAP HANA Vector Store using SAP Gen AI Hub models. Use when tasks involve chunking source data, creating embeddings, creating/loading vector tables, and storing metadata plus vectors for later RAG retrieval.
---

# Embeddings to SAP HANA Vector Store

Use this skill to populate vector storage correctly for downstream RAG.

## Set Required Environment Variables

```bash
HANA_ADDRESS=""
HANA_PORT="443"
HANA_USER=""
HANA_PASSWORD=""

AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_RESOURCE_GROUP=""
AICORE_BASE_URL=""
```

Install: `pip install "sap-ai-sdk-gen[all]" langchain-hana hdbcli python-dotenv`

## Choose and Record the Embedding Model

Default to `text-embedding-3-small` (1536 dimensions) unless the project already uses another model.

**Critical:** the query side must embed with the SAME model used at ingestion. A mismatch does not error - it silently returns garbage retrieval results. Record the model name (and chunk size/overlap) in the ingestion script's module docstring and reuse it in `vector-rag-query`. The `REAL_VECTOR` column dimension is fixed by the first insert; switching models later requires re-ingesting the table.

## Pattern A: Native Embedding + SQL Insert

```python
from gen_ai_hub.proxy.native.openai import embeddings

EMBEDDING_MODEL = "text-embedding-3-small"

def get_batch_embeddings(text_list, model=EMBEDDING_MODEL):
    """Embed a batch of texts through Gen AI Hub; returns list of vectors."""
    response = embeddings.create(model_name=model, input=text_list)
    return [res.embedding for res in response.data]
```

Create vector table and insert (values are parameterized; the vector is passed as a JSON string to `TO_REAL_VECTOR(?)`):

```python
import json

sql_create = """
CREATE TABLE SCIENCE_DATA (
    MY_TEXT NCLOB,
    MY_METADATA NCLOB,
    MY_VECTOR REAL_VECTOR
)
"""
cursor.execute(sql_create)

sql_insert = """
INSERT INTO SCIENCE_DATA (MY_TEXT, MY_METADATA, MY_VECTOR)
VALUES (?, ?, TO_REAL_VECTOR(?))
"""
rows = [
    (chunk_text, json.dumps(metadata), json.dumps(vector))
    for chunk_text, metadata, vector in zipped_chunks
]
cursor.executemany(sql_insert, rows)
```

`CREATE TABLE IF NOT EXISTS` is not valid HANA SQL - check `SYS.TABLES` first or handle the exception (see `sap-hana-sql` skill for the full idempotent DDL pattern).

## Pattern B: LangChain HanaDB

Import from `langchain_hana` (the old `langchain_community.vectorstores.hanavector` path is deprecated):

```python
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_hana import HanaDB

embeddings_model = OpenAIEmbeddings(proxy_model_name="text-embedding-3-small")

db = HanaDB(
    embedding=embeddings_model,
    connection=connection,
    table_name="SAP_HELP_PUBLIC",
)
db.add_documents(chunks)
docs = db.similarity_search("What is SAP Business AI?", k=2)
```

## Prepare Source Documents

Use one of these ingestion paths:

- CSV path: read rows, choose text column, serialize metadata JSON, chunk text.
- PDF path: load multiple PDFs, split with `RecursiveCharacterTextSplitter`, then store chunks with metadata.

## Apply Best Practices

- Batch embedding calls (`BATCH_SIZE` around `100`) for throughput.
- Batch inserts and commit per batch to avoid large transactions.
- Always store metadata JSON beside text and vector.
- Keep chunk size/overlap explicit and consistent across ingestion runs.
- Use stable table names by environment (`DEV/TEST/PROD`) to prevent accidental overwrite.

## Validate with Expected Outputs

Healthy run patterns:

- Dataframe preview of processed text + metadata.
- HANA version output (`cc.hana_version()`).
- Table create/drop logs.
- Insert logs such as `Inserted batch 1/2`.
- Retrieval smoke test returns top semantic matches.

Run one `similarity_search` (or the SQL cosine query from `vector-rag-query`) against the freshly loaded table before declaring ingestion done.

## Related Skills

- `vector-rag-query` - retrieval and grounded answering against this table (must reuse the same embedding model).
- `sap-hana-sql` - HANA dialect rules for any custom DDL/DML around the vector table.
- `sap-btp-ai` - shared environment setup and routing.
