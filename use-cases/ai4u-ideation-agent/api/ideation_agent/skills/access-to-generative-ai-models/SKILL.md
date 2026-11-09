---
name: access-to-generative-ai-models
description: Implement Python scripts that call LLMs through SAP Generative AI Hub (native SDK, LangChain wrappers, and orchestration). Use when asked to create or fix code for prompts, completions, multimodal requests, model switching, reasoning controls, or token usage handling via Gen AI Hub.
---

# Access Models via SAP Gen AI Hub

Use this skill to implement direct LLM calls correctly with SAP Gen AI Hub.

## Set Required Environment Variables

Use these variables from `.env`:

```bash
AICORE_AUTH_URL=""
AICORE_CLIENT_ID=""
AICORE_CLIENT_SECRET=""
AICORE_BASE_URL=""
AICORE_RESOURCE_GROUP=""
```

Load them first:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Choose the Integration Pattern

Use this decision rule:

- Use `native` for simple one-shot or multimodal API calls.
- Use `langchain` when building chains, memory, or other LangChain components.
- Use `orchestration` when combining LLM with masking/grounding/filtering modules.

## Use Native Chat Completion

```python
from gen_ai_hub.proxy.native.openai import chat

model = "gpt-4o"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = chat.completions.create(
    messages=messages,
    model=model,
    temperature=0.6,
    max_tokens=1000,
)

print(response.choices[0].message.content)
usage = getattr(response, "usage", None)
if usage:
    print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
```

## Use LangChain Wrapper

```python
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

chat = ChatOpenAI(
    proxy_model_name="gpt-4o",
    max_tokens=1000,
    temperature=0.6,
)
response = chat.invoke("What is the capital of France?")
print(response.content)
```

## Use Orchestration Service

```python
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template
from gen_ai_hub.orchestration.service import OrchestrationService

config = OrchestrationConfig(
    template=Template(messages=[
        SystemMessage("You are a helpful AI assistant."),
        UserMessage("What is the capital of France?"),
    ]),
    llm=LLM(name="gpt-4o", version="latest", parameters={"temperature": 0.6, "max_tokens": 1000}),
)

result = OrchestrationService(config=config).run()
print(result.orchestration_result.choices[0].message.content)
```

## Handle Multimodal Input

For image/video requests, encode media and pass as supported content blocks (`image_url` for OpenAI-compatible calls, `inline_data` for Gemini native client usage).

## Apply Best Practices

- Keep model name in one variable and avoid hardcoding it in many places.
- Read and print token usage for cost and regression checks.
- Use `reasoning_effort` for `gpt-5` style calls and avoid `temperature`/`max_tokens` where unsupported.
- Keep prompts explicit about style and failure behavior (`say you don't know`).
- Add small response checks after each call before building larger flows.

## Validate with Expected Outputs

Treat these as healthy output patterns:

- `"The capital of France is Paris."`
- A chat completion object with `choices[0].message.content`.
- Usage fields present (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- For multimodal logo example, response mentions SAP logo content.

## Reference Implementations

- `best-practices/generative-ai/access-to-generative-ai-models/python/native_implementation.ipynb`
- `best-practices/generative-ai/access-to-generative-ai-models/python/langchain_implementation.ipynb`
- `best-practices/generative-ai/access-to-generative-ai-models/python/orchestration_implementation.ipynb`
