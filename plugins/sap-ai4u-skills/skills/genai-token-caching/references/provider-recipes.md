# Provider and Interface Recipes

These recipes are test shapes, not permanent provider rules. Confirm support
with the active model, installed SDK, SAP Gen AI Hub route, and response
telemetry. Keep model names in variables and load credentials as described by
`$access-to-generative-ai-models`.

## GPT-5.6 LangChain Responses: Implicit Caching

The tested SAP wrapper needs `n` cleared when routing `ChatOpenAI` through the
Responses API. Setting the mode explicitly makes the experiment visible even
when implicit mode is also the model default.

```python
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

model = ChatOpenAI(
    proxy_model_name="gpt-5.6-luna",
    use_responses_api=True,
    prompt_cache_options={"mode": "implicit"},
)
model.n = None

response = model.invoke(messages)
print(response.usage_metadata)
```

For tool-calling agents, bind tools once and keep the message state append-only:

```python
tool_model = model.bind_tools(tools)
response = tool_model.invoke([system_message, *state["messages"]])
```

## GPT-5.6 LangChain Responses: Explicit Breakpoints

Explicit mode disables the automatic breakpoint and uses only the breakpoints
placed on supported content blocks. Keep `prompt_cache_key` stable for the
active run; do not include user- or company-specific data in it.

```python
from uuid import uuid4

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI

run_id = str(uuid4())
model = ChatOpenAI(
    proxy_model_name="gpt-5.6-luna",
    use_responses_api=True,
    prompt_cache_options={"mode": "explicit"},
)
model.n = None

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": stable_system_prompt,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ],
    },
    {"role": "user", "content": "Current request"},
]

response = model.invoke(messages, prompt_cache_key=f"cache-probe-{run_id}")
```

A system-only breakpoint reuses the stable system prefix while later transcript
growth remains uncached. To cache a growing explicit prefix, place a supported
breakpoint on the latest reusable message and verify the resulting reads and
writes.

## Response State Is Not Prompt Caching

`previous_response_id` and `store` can reduce transcript re-upload or preserve
provider-side response state. They are not an explicit prompt-cache mode and
must not be used as the cache-control comparison.

## Bedrock Converse Controls

Run the same request first without cache points, then with cache points if the
active model and route support them. One route producing zero implicit reads
does not prove that every Claude or Bedrock route always requires manual
caching.

```python
from gen_ai_hub.proxy.native.amazon.clients import Session

client = Session().client(model_name=model)
system = [{"text": system_prompt}]
converse_messages = to_converse_messages(messages)

if cache_mode == "explicit":
    system.append({"cachePoint": {"type": "default"}})
    converse_messages[-1]["content"].append({"cachePoint": {"type": "default"}})

response = client.converse(
    system=system,
    messages=converse_messages,
    inferenceConfig={"maxTokens": 1200},
)
```

Avoid unsupported generation knobs when a deployment rejects them. The dated
benchmark used `{"maxTokens": 1200}` without `temperature`.

## Claude On Bedrock LangChain

The currently tested LangChain explicit surface is:

```python
from gen_ai_hub.proxy.langchain.amazon import ChatBedrockConverse
from langchain_core.messages import SystemMessage

llm = ChatBedrockConverse(
    model_name=model,
    model_id=bedrock_model_id(model),
    max_tokens=1200,
)
llm = llm.bind_tools(tools).bind(cache_control={"type": "default"})

response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
```

The tested `ChatBedrockConverse` class exposed `_apply_cache_points` and
`cache_control` handling. In a target project, check whether
`langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse` has
`_apply_cache_points`; if not, run the bundled cache probe before trusting the
wrapper.

New Bedrock model names may lag in SAP SDK maps. Passing `model_id` avoids the
hardcoded allowlist.

```python
def bedrock_model_id(deployment_name: str) -> str:
    """Return the Bedrock model id for an Anthropic deployment name."""

    provider, name = deployment_name.split("--", maxsplit=1)
    return f"{provider}.{name.replace('.', '-')}"
```

## Gemini Controls

Implicit repeated-prefix calls require no cached-content resource. Explicit
cached content is a separate API operation whose SAP route availability must be
tested independently.

```python
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.native.google_genai.clients import Client
from google.genai import types

client = Client(proxy_client=get_proxy_client("gen-ai-hub"))
config = types.GenerateContentConfig(system_instruction=system_prompt)

if cache_mode == "explicit":
    try:
        cached = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_prompt,
                ttl="3600s",
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "Explicit Gemini cache creation failed on this model/API/route; "
            "run the implicit control separately."
        ) from exc
    config = types.GenerateContentConfig(cached_content=cached.name)

contents = [
    {
        "role": "user" if item["role"] == "user" else "model",
        "parts": [{"text": item["content"]}],
    }
    for item in messages
]
response = client.models.generate_content(model=model, contents=contents, config=config)
```

Read cache hits from `response.usage_metadata.cached_content_token_count` when
that field is present. A missing field and a reported zero are different
outcomes.

## Raw and Normalized Telemetry

Always display at least one raw usage payload before using the normalizer:

```python
print(response.usage_metadata)
row = normalize_usage("agent", response)
```

## LangGraph Placement

Cache controls and usage capture belong in the model-calling node, not the tool
node. Keep the raw responses available for inspection.

```python
usage_rows = []
model_responses = []

def assistant(state):
    """Call the model and capture raw plus normalized usage."""

    response = llm.invoke([system_message] + state["messages"])
    model_responses.append(response)
    usage_rows.append(normalize_usage("langgraph_agent", response))
    return {"messages": [response]}
```
