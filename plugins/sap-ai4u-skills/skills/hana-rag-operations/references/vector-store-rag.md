# HANA Vector-Store RAG

## Choose this route

Use vector-store RAG for unstructured or semi-structured content where the main operation is semantic similarity search.

```text
documents -> chunks -> embeddings -> HanaDB
question  -> query embedding -> retriever -> context -> LLM -> answer
```

Keep the ingestion embedding model, vector dimension, distance strategy, and table schema consistent between ingestion and query.

## Connection and embeddings

Load the environment before creating SAP clients or the HANA connection:

```python
import os
from dotenv import load_dotenv
from hdbcli import dbapi

load_dotenv(".env", override=True)

connection = dbapi.connect(
    address=os.environ["HANA_ADDRESS"],
    port=int(os.getenv("HANA_PORT", "443")),
    user=os.environ["HANA_USER"],
    password=os.environ["HANA_PASSWORD"],
    encrypt=os.getenv("HANA_ENCRYPT", "true").lower() in {"true", "1", "yes"},
)
```

Use the SAP Gen AI Hub embedding wrapper:

```python
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

proxy_client = get_proxy_client("gen-ai-hub")
embedding_model = OpenAIEmbeddings(
    proxy_model_name="text-embedding-3-small",
    proxy_client=proxy_client,
)
```

Do not switch embedding models for queries against an existing corpus. Re-embed when changing models or dimensions.

## HanaDB operations

```python
from langchain_hana import HanaDB
from langchain_hana.utils import DistanceStrategy

vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
    table_name="DOCUMENT_CHUNKS",
)
```

The current enum members include `COSINE`, `EUCLIDEAN_DISTANCE`, `MAX_INNER_PRODUCT`, `DOT_PRODUCT`, and `JACCARD`. `COSINE_SIMILARITY` is not the current member name.

Ingest LangChain documents:

```python
from langchain_core.documents import Document

vector_store.add_documents([
    Document(
        page_content="The warranty period is 24 months.",
        metadata={"source_id": "policy-001", "page": 4},
    )
])
```

For a one-shot corpus, use `HanaDB.from_documents(...)` or `HanaDB.from_texts(...)`. Use an explicit instance plus `add_documents(...)` for incremental or batched ingestion.

Direct retrieval:

```python
docs = vector_store.similarity_search("What is the warranty period?", k=4)
scored = vector_store.similarity_search_with_score(
    "What is the warranty period?", k=4
)
```

Retriever composition:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke("What is the warranty period?")
```

## Standard LangChain QA

For current LangChain 1.x, use `langchain_classic` retrieval helpers:

```python
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    proxy_model_name="gpt-4.1",
    proxy_client=proxy_client,
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the context. Say when evidence is insufficient."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
result = retrieval_chain.invoke({"input": "What is the warranty period?"})

answer = result["answer"]
sources = result["context"]
```

Prefer this composition over the older `RetrievalQA` abstraction in new code. The retrieved documents remain available under `result["context"]`.

## Filters, MMR, and reranking

Use trusted metadata to scope retrieval:

```python
docs = vector_store.similarity_search(
    "What is the warranty period?",
    k=4,
    filter={"country": "DE", "document_type": "warranty"},
)
```

Use MMR to reduce near-duplicate context:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

Use HANA reranking when the target HANA instance supports the model:

```python
docs = vector_store.similarity_search(
    "What is the warranty period?",
    k=4,
    rerank_config={
        "query": "What is the warranty period?",
        "top_n": 3,
        "model_id": "<supported-hana-reranker-id>",
        "rank_fields": ["content"],
    },
)
```

Alternatively use `HanaReranker(connection=connection, model_id="...")` and call `compress_documents(...)`.

## HANA-managed embeddings

Use `HanaInternalEmbeddings` only when HANA owns the embedding operation:

```python
from langchain_hana import HanaDB, HanaInternalEmbeddings

embedding_model = HanaInternalEmbeddings(
    internal_embedding_model_id="<hana-internal-model-id>",
)
vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    table_name="DOCUMENT_CHUNKS",
)
```

This object intentionally does not embed text in Python. Do not pass it to an external embedding workflow.

## Validation

Check the connection, one embedding, one direct retrieval, one filtered retrieval, and one grounded answer. Verify the answer against `result["context"]`. If no rows return, inspect table name, model, dimension, distance strategy, metadata filters, and transaction state before changing the prompt.
