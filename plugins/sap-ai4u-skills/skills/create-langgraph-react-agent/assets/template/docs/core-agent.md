# Core LangGraph agent

## Purpose

The template provides a configurable ReAct agent with CLI and optional A2A
entrypoints but no UI. It is intended to be copied into a use case and extended
through tools and agent-local skills.

## Inputs and outputs

`AgentRuntime.ainvoke()` accepts user text, a required `context_id`, optional
image/PDF `Attachment` objects, and an optional Pydantic model or JSON Schema.
It returns `AgentResult` containing final text, optional parsed output, the
LangGraph message trace, and `status` (`completed`, or `input_required` when a
tool listed in `approvals.tools` awaits the user's approval).

The graph is:

```text
START -> load_skill_node -> compact_context -> agent -> tools -> compact_context -> agent -> END
```

`compact_context` caps tool results and summarizes long history before every
model call; the agent ends the turn early when it requests an approval-gated
tool. Memory, concurrency, and approvals are described in `mcp-and-hana.md`.

```python
from template_agent import AgentRuntime, Attachment

async with await AgentRuntime.create("config/agent.yaml") as runtime:
    result = await runtime.ainvoke(
        "Summarize this document",
        context_id="case-42",
        attachments=[Attachment(path="document.pdf")],
    )
    print(result.output_text)
```

The model factory supports SAP Gen AI Hub OpenAI Responses, Gemini, and Claude
Bedrock Converse wrappers. For SAP AI SDK 7.2.0, the OpenAI subclass removes
the Chat Completions-only `n` field from Responses payloads, while the Claude
factory supplies both the SAP deployment name and its required Bedrock model
ID. Attachment bytes use LangChain standard content blocks; Claude PDF names
are additionally normalized to Bedrock's document-name rules. Attachment bytes
are never stored in conversation memory.

Structured output is a second provider-native formatting call after the ReAct
loop. This keeps tool calling independent from output-schema enforcement.

## Related files

- `template_agent/runtime.py`: graph, invocation, compaction, approvals, structured-output pass.
- `template_agent/memory.py`: HANA conversation store with lease and versioned saves.
- `template_agent/providers.py`: model-supplier factory.
- `template_agent/models.py`: public attachment/result contracts.
- `template_agent/a2a_server.py`: A2A transport adapter and ASGI application.
- `config/agent.yaml`: deployment and integration settings.

## Test

```bash
.venv/bin/python -m pytest -q tests/test_runtime.py tests/test_models_config.py
.venv/bin/ruff check template_agent scripts tests
```
