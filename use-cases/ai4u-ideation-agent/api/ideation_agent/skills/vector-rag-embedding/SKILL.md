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

## Pattern A: Native Embedding + SQL Insert

```python
import json
from gen_ai_hub.proxy.native.openai import embeddings

def get_batch_embeddings(text_list, model="text-embedding-ada-002"):
    response = embeddings.create(model_name=model, input=text_list)
    return [res.embedding for res in response.data]
```

Create vector table and insert:

```python
sql_create = """
CREATE TABLE SCIENCE_DATA_MIT8 (
    MY_TEXT NCLOB,
    MY_METADATA NCLOB,
    MY_VECTOR REAL_VECTOR
);
"""
cursor.execute(sql_create)

sql_insert = """
INSERT INTO SCIENCE_DATA_MIT8 (MY_TEXT, MY_METADATA, MY_VECTOR)
VALUES (?, ?, TO_REAL_VECTOR(?))
"""
cursor.executemany(sql_insert, rows)  # rows contains vector as JSON string
```

## Pattern B: LangChain HanaDB

```python
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB

embeddings_model = OpenAIEmbeddings(proxy_model_name="text-embedding-ada-002")

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

## Reference Implementations

- `best-practices/generative-ai/vector-rag-embedding/python/Native_HANA_VectorStore_Embeddings.ipynb`
- `best-practices/generative-ai/vector-rag-embedding/python/LangChain_HANA_VectorStore_Embeddings.ipynb`
