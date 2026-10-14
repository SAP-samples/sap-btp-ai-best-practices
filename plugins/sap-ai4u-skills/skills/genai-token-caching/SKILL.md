---
name: genai-token-caching
description: Build, tune, and debug token or prompt caching for SAP Gen AI Hub agentic workflows using native SDKs, LangChain, and LangGraph. Use when repeated prompt prefixes should reduce input processing, when implicit or explicit cache controls must be selected, or when cache reads and writes need to be verified from model telemetry.
---

# GenAI Token Caching

Use this skill after SAP Gen AI Hub client setup is clear. Prompt-caching
behavior is scoped to the model family, API, SDK, route, tenant, and request
shape being tested. Treat documentation and prior benchmarks as hypotheses;
the active response telemetry is the result.

## Core Workflow

1. Keep stable instructions, tool schemas, and reference context before
   frequently changing user and tool content.
2. Keep the active agent transcript append-only while measuring a cache. Do
   not reorder, flatten, summarize, or edit earlier turns inside the probe.
3. Confirm the exact model, API, wrapper, SDK version, and SAP AI Core route.
4. Inspect one raw response before normalizing it. A missing field means the
   interface did not report it; a reported zero means no activity was reported.
5. Run both implicit and explicit controls when the active interface supports
   them. Do not infer one provider-wide requirement from one route.
6. Compare multiple consecutive calls. Successful prefix reuse normally shows
   cache reads while logical input grows and uncached input grows much more
   slowly.
7. Keep TTL, thresholds, prices, and hit rates conditional on current provider
   documentation and observed telemetry.

## Capability Verification

| Interface to verify | Controls worth testing | Telemetry to inspect |
|---|---|---|
| GPT-5.6 through LangChain `ChatOpenAI` Responses | `prompt_cache_options={"mode": "implicit"}` and explicit content-block breakpoints | `AIMessage.usage_metadata`, especially `input_token_details.cache_read` and `cache_creation` |
| Other OpenAI-compatible model/API combinations | Default request plus any cache controls exposed by that model and installed client | Raw `usage` or LangChain `usage_metadata`; field presence can differ by API |
| Bedrock Converse models | A request without cache points and the same request with supported `cachePoint`/`cache_control` settings | `cacheReadInputTokens`, `cacheWriteInputTokens`, and any TTL-specific detail fields |
| Gemini model/API combinations | Implicit repeated-prefix calls and, where routed, explicit cached content | `cached_content_token_count` or `cachedContentTokenCount`; explicit cache endpoints may have route-specific availability |

These rows identify tests, not permanent provider prescriptions. Read
`references/provider-recipes.md` for exact call shapes.

## Raw Usage Before Normalization

For the tested GPT-5.6 Luna LangChain Responses shape:

```python
raw = response.usage_metadata or {}
details = raw.get("input_token_details") or {}

raw_input_tokens = raw.get("input_tokens")
cache_read_tokens = details.get("cache_read")
cache_write_tokens = details.get("cache_creation")
output_tokens = raw.get("output_tokens")

uncached_input_tokens = None
if raw_input_tokens is not None:
    uncached_input_tokens = max(
        raw_input_tokens - (cache_read_tokens or 0) - (cache_write_tokens or 0),
        0,
    )
```

Here, `input_tokens` is the raw prompt-side total reported by this response
shape. `uncached_input_tokens` is derived and must be labelled as such. Do not
assume another interface uses the same accounting convention.

## Message Shape

Use this shape in native or LangGraph loops:

```python
messages: list[dict[str, str]] = [
    {"role": "user", "content": "Initial task"},
    {"role": "assistant", "content": "Tool/action decision"},
    {"role": "user", "content": "Tool observation"},
]
```

Tool observations should be appended as new `tool` or `user` messages,
depending on the API. They should not be spliced into a rewritten prompt.

For Bedrock Converse, merge consecutive same-role messages because Converse
requires alternating roles:

```python
def to_converse_messages(messages: list[dict[str, str]]) -> list[dict]:
    """Return Bedrock Converse messages from append-only chat history."""

    converse = []
    for message in messages:
        block = {"text": message["content"]}
        if converse and converse[-1]["role"] == message["role"]:
            converse[-1]["content"].append(block)
        else:
            converse.append({"role": message["role"], "content": [block]})
    return converse
```

## Usage Normalization

Copy or import this skill's bundled `normalize_usage.py` helper in the target
project instead of writing provider-specific parsing from scratch.

```python
from normalize_usage import add_derived_cache_writes, normalize_usage

rows = [normalize_usage("agent", response) for response in model_responses]
add_derived_cache_writes(rows)
```

Important normalized fields:

- `provider_input_tokens`: the provider's main raw prompt/input field.
- `input_total_tokens`: normalized prompt-side volume including reported reads
  and writes.
- `uncached_input_tokens`: preferred name for input processed outside the
  cache.
- `input_tokens`: compatibility alias for `uncached_input_tokens`.
- `cache_read_input_tokens`: provider-reported cache reads when present.
- `cache_write_input_tokens`: provider-reported cache writes when present.
- `cache_write_derived_tokens`: an explicitly labelled estimate used only when
  writes are absent and a later cache-read increase can provide evidence.
- `output_tokens`, `reasoning_tokens`, and `total_tokens`: provider output
  totals when present.

## Verification Probe

Run this skill's bundled probe before relying on caching in a new project:

```bash
python path/to/skill/scripts/cache_probe.py --provider openai --model gpt-5.6-luna --cache-mode implicit --turns 4
python path/to/skill/scripts/cache_probe.py --provider openai --model gpt-5.6-luna --cache-mode explicit --turns 4
python path/to/skill/scripts/cache_probe.py --provider bedrock --model anthropic--claude-4.6-sonnet --cache-mode implicit --turns 4
python path/to/skill/scripts/cache_probe.py --provider bedrock --model anthropic--claude-4.6-sonnet --cache-mode explicit --turns 4
python path/to/skill/scripts/cache_probe.py --provider gemini --model gemini-3.5-flash --cache-mode implicit --turns 4 --filler-facts 1200
```

The probe salts the stable prefix for a cold run and prints reported versus
derived values separately. An explicit-mode failure is a result for that
model/API/route; the probe must not silently fall back to implicit mode. Use
`--no-nonce` only when deliberately testing reuse from an earlier run.

## Agent Checklist

- Use `$access-to-generative-ai-models` for SAP Gen AI Hub client setup.
- Apply the relevant LangGraph skills to the graph and tool loop.
- Put cache controls and usage capture at the actual model invocation boundary.
- Keep prompt caching distinct from checkpointing, conversation persistence,
  `previous_response_id`, and application memory.
- Record model, API, SDK, route, and date beside benchmark results.
- Re-run the probe after any model, SDK, tenant, or routing change.
