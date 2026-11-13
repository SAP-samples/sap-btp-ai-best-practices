# Self-contained GPT Responses replay

## Purpose

The portable agent keeps encrypted reasoning for multi-step GPT conversations
without relying on provider response-item storage. This replaces the earlier
workaround that removed reasoning from every generated answer. Graph topology,
tool-call linkage, optional HANA memory, approvals, and other providers retain
existing behavior.

## Evidence and error interpretation

A synthetic four-tool conversation against a GPT-6.1 Sol deployment on 13 November
2026 returned reasoning with encrypted content. A follow-up with the original
output item IDs returned HTTP 404, including when `store=False` was set. Sending
the same full history without provider item IDs, while retaining encrypted
reasoning, returned HTTP 200. Other calls with IDs succeeded, so this is an
intermittent replay-state failure. The gateway does not expose its internal
routing/storage cause.

The earlier template fix was based on 404/hang observations on another GPT
deployment. Those observations remain useful, but do not establish that all
encrypted reasoning is unsupported. A successful ID-free encrypted replay
supports the narrower normalization used here.

- HTTP 404 `Resource not found`: may be a failed response-item lookup or an
  unavailable deployment. Successful earlier steps at the same URL help separate
  these possibilities; test ID-free replay before blaming profile selection.
- HTTP 400 `invalid_encrypted_content`: a distinct verification rejection. The
  template propagates it; use a single targeted retry without reasoning only if
  the application implements that recovery.
- HTTP 429: quota/rate limiting; do not classify it as a replay failure.

## Inputs, outputs and relevant modules

`create_chat_model(ModelSettings)` creates the configured SAP proxy model. For
OpenAI Responses it sets `store=False` and
`include=["reasoning.encrypted_content"]`.
`ResponsesCompatibleChatOpenAI._get_request_payload` accepts LangChain prompt
messages and bound generation/tool parameters. It returns a serialized provider
payload with the unsupported chat-only `n` removed and no top-level `id` on
replayed `message`, `reasoning`, `function_call`, or `custom_tool_call` items.

```python
replay_types = {"message", "reasoning", "function_call", "custom_tool_call"}
payload["input"] = [
    {key: value for key, value in item.items() if key != "id"}
    if item.get("type") in replay_types else item
    for item in payload["input"]
]
```

Encrypted content, assistant text, tool arguments and function `call_id` links
remain intact. Returned `AIMessage` objects and their IDs are not mutated, so
LangGraph's message reducer and HANA's complete history still work. Encrypted
content stays internal; print final text and safe usage/activity, not reasoning
blocks. Provider-managed built-in tools or bare references need their own tested
replay strategy rather than arbitrary ID deletion.

Related files:

- `template_agent/providers.py`: compatibility serializer and shared factory.
- `template_agent/runtime.py`: graph, formatter and summarizer using the factory.
- `template_agent/memory.py`: existing complete-history HANA persistence.
- `requirements.txt`: explicit tested `langchain-openai==1.3.5` dependency.
- `tests/test_providers.py`: actual payload conversion, parallel function calls,
  graph non-mutation, missing opaque content, factory settings, sync/async
  reasoning retention and existing Claude mapping.
- `scripts/live_model_smoke.py`: tool round trip and explicit reasoning follow-up.

## Adapting an existing application

An application copied from a skill is independent of later marketplace updates.
Inspect its existing provider factory; do not overwrite use-case-specific model
callbacks, Claude/Gemini controls, tools, memory, approvals or graph behavior.
Transfer the Responses serialization and factory settings, and remove only the
unconditional reasoning-discard generation overrides. Keep any existing narrow
HTTP 400 verification recovery. Request encrypted content on every stateless
call, preserve complete tool history, and validate before restarting the local
API. Cloud Foundry deployment remains a manual user action.

## Validation

From the generated project root:

```bash
.venv/bin/python -m pytest -q tests/test_providers.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check template_agent/providers.py tests/test_providers.py scripts/live_model_smoke.py
.venv/bin/python -m scripts.live_model_smoke --model openai=YOUR_DEPLOYED_MODEL
```

Offline tests require no SAP credentials. Live smoke calls use synthetic text and
a local arithmetic tool. They report final text and an encrypted item count, not
opaque content or IDs. The reasoning follow-up fails explicitly if no encrypted
reasoning was returned: a simple text or tool-only probe cannot prove replay.
The measured replay stack used SAP AI SDK 7.2.0, `langchain-openai` 1.3.5,
`langchain-core` 1.6.1, and `openai` 2.54.0. Revalidate other model/SDK combinations
in their target AI Core resource group.

Validation completed on 13 November 2026: all 60 template tests passed, the edited
Python files passed Ruff, all 23 Python snippets in the two edited model/setup
skills parsed, and live synthetic GPT-6.1 Sol reasoning replay and arithmetic-tool
round trips succeeded. The reasoning smoke observed one encrypted reasoning item
before replay. No real application data or persistent business records were used.

Marketplace package validation passed for all 38 skills.
