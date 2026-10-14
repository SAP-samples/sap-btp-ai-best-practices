---
name: vector-rag-query
description: Implement Retrieval-Augmented Generation query pipelines on top of SAP HANA Vector Store and SAP Gen AI Hub models. Use when tasks involve semantic retrieval, context assembly, prompt composition, optional conversation history, and grounded answer generation.
---

# Query RAG with HANA Vector Store

Use this skill to build runtime retrieval + answer scripts once vectors are already stored.

## Precondition

Assume embeddings are already written to a HANA vector table (for example via `vector-rag-embedding`).

**Critical:** embed queries with the SAME model that ingested the table (find it in the ingestion script; default in these skills is `text-embedding-3-small`). A mismatched model does not error - it silently returns irrelevant chunks. If retrieval results look random, check this first.

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

## Pattern A: Native SQL Retrieval + Completion

Identifiers cannot be bound as SQL parameters, so validate them against allowlists; bind the query vector as a parameter.

```python
from gen_ai_hub.proxy.native.openai import embeddings, chat
import json

EMBEDDING_MODEL = "text-embedding-3-small"  # must match ingestion

def get_embedding(query: str):
    """Embed one query string through Gen AI Hub."""
    return embeddings.create(model_name=EMBEDDING_MODEL, input=query).data[0].embedding

ALLOWED_METRICS = {"COSINE_SIMILARITY": "DESC", "L2DISTANCE": "ASC"}

def run_vector_search(query: str, cursor, table_name: str, metric: str = "COSINE_SIMILARITY", k: int = 4):
    """Return top-k (text, metadata) rows by vector similarity.

    table_name must come from code/config, never from user input.
    """
    sort_order = ALLOWED_METRICS[metric]  # KeyError on unknown metric = fail fast
    k = int(k)
    query_vector = get_embedding(query)
    sql = f'''
    SELECT TOP {k} MY_TEXT, MY_METADATA
    FROM "{table_name}"
    ORDER BY {metric}(MY_VECTOR, TO_REAL_VECTOR(?)) {sort_order}
    '''
    cursor.execute(sql, (json.dumps(query_vector),))
    return cursor.fetchall()
```

Generate answer from retrieved context:

```python
query = "How to test for fat in foods?"
records = run_vector_search(query, cursor, "SCIENCE_DATA", k=4)
context = " ".join(r[0] for r in records)

prompt = f"""Use context to answer professionally.\nContext: {context}\nQuestion: {query}\nIf the answer is not in the context, say you don't know."""
response = chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "system", "content": "You are an intelligent assistant."},
              {"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
```

## Pattern B: LangChain Retriever + Optional History

Import from `langchain_hana` (the old `langchain_community.vectorstores.hanavector` path is deprecated):

```python
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings, ChatOpenAI
from langchain_hana import HanaDB
from langchain_hana.utils import DistanceStrategy

embedding_model = OpenAIEmbeddings(proxy_model_name="text-embedding-3-small")
vdb = HanaDB(
    embedding=embedding_model,
    connection=connection,
    distance_strategy=DistanceStrategy.COSINE_SIMILARITY,
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
- In production APIs, wrap the LLM call with `token-logger` instrumentation.

## Validate with Expected Outputs

Healthy run indicators:

- HANA connection metadata prints (`hana_version`, schema).
- Retrieval returns non-empty records for known table.
- Final answer references retrieved domain context.
- If table missing, explicit DB error similar to `invalid table name`.

If retrieval returns rows but they are irrelevant to the query, the embedding model almost certainly differs from the ingestion model - fix that before tuning `k` or prompts.

## Related Skills

- `vector-rag-embedding` - how the table was populated (embedding model, chunking).
- `sap-hana-sql` - HANA dialect rules for any custom SQL.
- `token-logger` - token usage logging for production APIs.
- `sap-btp-ai` - shared environment setup and routing.
