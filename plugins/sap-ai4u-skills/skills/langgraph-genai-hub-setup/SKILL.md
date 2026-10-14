---
name: langgraph-genai-hub-setup
description: "INVOKE THIS SKILL when initializing LLMs for LangGraph on SAP BTP. Covers gen_ai_hub proxy wrappers, provider selection (OpenAI/Bedrock/VertexAI), make_llm factory pattern, and environment configuration."
---

<overview>
SAP Generative AI Hub provides LangChain-compatible proxy wrappers for multiple LLM providers. These proxies implement the standard `BaseChatModel` interface, so **all LangGraph patterns work unchanged** once the LLM is initialized correctly.

The only difference from standard LangChain is the import path and parameter names. Get these right and everything else follows.
</overview>

---

## Provider Selection

<provider-selection-table>

| Provider | Import Path | Model Name Param | Max Tokens Param | Best For |
|----------|-------------|------------------|------------------|----------|
| OpenAI | `gen_ai_hub.proxy.langchain.openai.ChatOpenAI` | `proxy_model_name` | `max_tokens` | GPT-4.1, broadest tool calling support |
| Bedrock | `gen_ai_hub.proxy.langchain.amazon.ChatBedrock` | `model_name` | `model_kwargs={"max_tokens": N}` | Claude models via AWS |
| VertexAI | `gen_ai_hub.proxy.langchain.google_vertexai.ChatVertexAI` | `model_name` | `max_output_tokens` | Gemini models via Google |

</provider-selection-table>

Model names in examples (`gpt-4.1`, `anthropic--claude-3.5-sonnet`, `gemini-2.5-pro`) must exist as deployments in the target SAP AI Core resource group. Keep model names in variables or config; if a call fails with a deployment/model-not-found error, list available deployments or ask the user rather than guessing another name.

---

## Direct Initialization

<ex-direct-initialization>
<python>
Initialize each provider directly with their specific parameter names.
```python
from dotenv import load_dotenv
load_dotenv()  # Load AICORE credentials from .env

# OpenAI proxy — note: proxy_model_name, NOT model_name
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

llm = ChatOpenAI(
    proxy_model_name="gpt-4.1",
    temperature=0.2,
    max_tokens=2048,
)

# Bedrock proxy — uses model_name and model_kwargs
from gen_ai_hub.proxy.langchain.amazon import ChatBedrock

llm = ChatBedrock(
    model_name="anthropic--claude-3.5-sonnet",
    temperature=0.2,
    model_kwargs={"max_tokens": 2048},
)

# VertexAI proxy — uses model_name and max_output_tokens
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    temperature=0.2,
    max_output_tokens=2048,
)
```
</python>
</ex-direct-initialization>

---

## Factory Pattern

Use a factory function when your code needs to support multiple providers or when you want a single configuration point.

<ex-factory-pattern>
<python>
Centralized LLM factory that handles provider differences.
```python
from typing import Literal, Optional
from dotenv import load_dotenv
load_dotenv()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.amazon import ChatBedrock
from gen_ai_hub.proxy.langchain.google_vertexai import ChatVertexAI


def make_llm(
    provider: Literal["openai", "bedrock", "vertex"],
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    top_p: Optional[float] = None,
):
    """Create a LangChain-compatible LLM from SAP GenAI Hub proxies."""
    if provider == "openai":
        return ChatOpenAI(
            proxy_model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "bedrock":
        return ChatBedrock(
            model_name=model_name,
            temperature=temperature,
            model_kwargs={
                "max_tokens": max_tokens,
                **({"top_p": top_p} if top_p is not None else {}),
            },
        )
    if provider == "vertex":
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    raise ValueError(f"Unknown provider: {provider}")


# Usage
llm = make_llm(provider="openai", model_name="gpt-4.1")
```
</python>
</ex-factory-pattern>

---

## Embeddings

For retrievers or vector stores inside a LangGraph app, initialize embeddings through the same proxy family:

<ex-embeddings>
<python>
Embeddings via the SAP GenAI Hub proxy — also uses `proxy_model_name`.
```python
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(proxy_model_name="text-embedding-3-small")
```
</python>
</ex-embeddings>

For HANA vector store retrieval, see the `vector-rag-query` skill; the embedding model must match the one used at ingestion.

---

## Tool Binding

After proxy initialization, tool binding works identically to standard LangChain.

<ex-tool-binding>
<python>
Bind tools to the proxy LLM — same API as standard LangChain.
```python
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = ChatOpenAI(proxy_model_name="gpt-4.1", temperature=0.2)
llm_with_tools = llm.bind_tools([calculator])

# Use in LangGraph nodes exactly like standard LangChain
response = llm_with_tools.invoke([{"role": "user", "content": "What is 12 * 7?"}])
```
</python>
</ex-tool-binding>

---

## Fixes

<fix-proxy-model-name>
<python>
The OpenAI proxy uses `proxy_model_name`, not `model_name`.
```python
# WRONG: model_name does not work for the OpenAI proxy
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4.1")  # Ignored or errors!

# CORRECT: Use proxy_model_name
llm = ChatOpenAI(proxy_model_name="gpt-4.1")
```
</python>
</fix-proxy-model-name>

<fix-missing-env>
<python>
AICORE credentials must be loaded before creating any proxy LLM.
```python
# WRONG: No environment loaded — proxy cannot authenticate
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
llm = ChatOpenAI(proxy_model_name="gpt-4.1")  # Auth error!

# CORRECT: Load .env with AICORE credentials first
from dotenv import load_dotenv
load_dotenv()

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
llm = ChatOpenAI(proxy_model_name="gpt-4.1")
```
</python>
</fix-missing-env>

<fix-no-direct-langchain-openai>
<python>
Do NOT import from `langchain_openai` directly — use SAP proxy wrappers.
```python
# WRONG: Direct LangChain import bypasses SAP GenAI Hub
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1")  # Not routed through SAP proxy!

# CORRECT: Use the SAP GenAI Hub proxy wrapper
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
llm = ChatOpenAI(proxy_model_name="gpt-4.1")
```
</python>
</fix-no-direct-langchain-openai>

<boundaries>
### What You Should NOT Do

- Import `ChatOpenAI` from `langchain_openai` — always use `gen_ai_hub.proxy.langchain.openai`
- Use `model_name` for the OpenAI proxy — it requires `proxy_model_name`
- Skip loading environment variables — AICORE credentials are required for authentication
- Hardcode API keys — use `.env` files with `python-dotenv`
</boundaries>

## Related Skills

- `langgraph-fundamentals` — graph construction once the LLM is initialized.
- `token-logger` — token-usage logging for production APIs (wrap `llm.invoke` inside nodes).
- `access-to-generative-ai-models` — native (non-LangChain) Gen AI Hub calls.
- `sap-btp-ai` — routing and shared environment conventions.
