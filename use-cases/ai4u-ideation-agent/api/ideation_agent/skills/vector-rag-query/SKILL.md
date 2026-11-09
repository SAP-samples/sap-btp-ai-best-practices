---
name: vector-rag-query
description: Implement Retrieval-Augmented Generation query pipelines on top of SAP HANA Vector Store and SAP Gen AI Hub models. Use when tasks involve semantic retrieval, context assembly, prompt composition, optional conversation history, and grounded answer generation.
---

# Query RAG with HANA Vector Store

Use this skill to build runtime retrieval + answer scripts once vectors are already stored.

## Precondition

Assume embeddings are already written to HANA vector table (for example via `vector-rag-embedding` workflow).

## Set Required Environment Variables

```bash
HANA_ADDRESS=""
HANA_PORT=""
HANA_USER=""
HANA_PASSWORD=""
HANA_AUTOCOMMIT="true"
HANA_SSL_CERT_VALIDATE="false"

AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_RESOURCE_GROUP=""
AICORE_BASE_URL=""
```

## Pattern A: Native SQL Retrieval + Completion

```python
from gen_ai_hub.proxy.native.openai import embeddings, chat

def get_embedding(query: str):
    return embeddings.create(model_name="text-embedding-ada-002", input=query).data[0].embedding

def run_vector_search(query, cursor, table_name, k=4):
    qv = get_embedding(query)
    sql = f"""
    SELECT TOP {k} ID, MY_TEXT, MY_METADATA
    FROM {table_name}
    ORDER BY COSINE_SIMILARITY(MY_VECTOR, TO_REAL_VECTOR('{qv}')) DESC
    """
    cursor.execute(sql)
    return cursor.fetchall()
```

Generate answer from retrieved context:

```python
query = "How to test for fat in foods?"
records = run_vector_search(query, cursor, "SCIENCE_DATA_MIT6", k=4)
context = " ".join(r[1] for r in records)

prompt = f"""Use context to answer professionally.\nContext: {context}\nQuestion: {query}\nIf unknown, say you don't know."""
response = chat.completions.create(
    model_name="gpt-4o",
    messages=[{"role": "system", "content": "You are an intelligent assistant."},
              {"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
```

## Pattern B: LangChain Retriever + Optional History

```python
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import HanaDB
from langchain_community.vectorstores.utils import DistanceStrategy

embedding_model = OpenAIEmbeddings(proxy_model_name="text-embedding-ada-002")
vdb = HanaDB(
    embedding=embedding_model,
    connection=connection,
    distance_strategy=DistanceStrategy.COSINE,
    table_name="SAP_HELP_PUBLIC",
)
retriever = vdb.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(proxy_model_name="gpt-4o", max_tokens=2000, temperature=0.5)
```

Use `RunnableWithMessageHistory` only when the user explicitly needs conversational follow-up behavior.

## Apply Best Practices

- Fail fast when table name is wrong; surface DB error clearly.
- Keep retrieval `k` small first (`2` to `4`), then tune.
- Use strict grounded prompt policy (`if unknown, say unknown`).
- Keep retrieval and generation as separate functions for debuggability.
- Log query, retrieved IDs, and latency for each stage.

## Validate with Expected Outputs

Healthy run indicators:

- HANA connection metadata prints (`hana_version`, schema).
- Retrieval returns non-empty records for known table.
- Final answer references retrieved domain context.
- If table missing, explicit DB error similar to `invalid table name`.

## Reference Implementations

- `best-practices/generative-ai/vector-rag-query/python/Native_RAG.ipynb`
- `best-practices/generative-ai/vector-rag-query/python/LangChain_RAG_with_History.ipynb`
