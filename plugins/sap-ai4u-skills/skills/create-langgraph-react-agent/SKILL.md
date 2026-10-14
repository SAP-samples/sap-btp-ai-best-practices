---
name: create-langgraph-react-agent
description: Scaffold or adapt a minimal portable Python LangGraph ReAct agent for SAP Generative AI Hub use cases. Use when coding agents need to create a drop-in GPT, Gemini, or Claude agent with OpenAI Responses support, image/PDF inputs, structured output, dynamically loaded local skills, stdio/SSE/HTTP MCP tools, dual-stack A2A 1.0 and Joule-compatible 0.3 connectivity, optional SAP HANA conversation memory, a pip requirements workflow, or a small use-case-specific copy.
---

# Create a LangGraph ReAct agent

Scaffold the bundled, tested runtime and adapt only the configuration, tools,
skills, and prompt required by the target use case.

## Scaffold

Run the guarded copier. It refuses to write into a non-empty target:

```bash
python scripts/scaffold_agent.py /absolute/path/to/new-agent
```

Do not copy the skill's own `.venv`, credentials, caches, or test data. The
bundled asset contains no such files.

## Configure

1. Create `.venv` with Python 3.11 or 3.12.
2. Install `requirements.txt` with pip.
3. Copy `.env.example` to `.env` and populate only required credentials.
4. Set `model.provider` and `model.name` in `config/agent.yaml`.
5. Add use-case tools through `AgentRuntime.create(..., extra_tools=[...])`; if
   serving them, use the documented A2A `runtime_factory` so the same tools are
   present remotely.
6. Add agent skills under `agent_skills/<skill-name>/SKILL.md`.
7. Enable only the MCP servers and HANA memory the use case needs.
8. Keep the default A2A listener on loopback until authenticated ingress,
   caller-scoped server contexts, and context-ID authorization are in place.
9. Set `A2A_PUBLIC_URL` to the correct base URL for the selected environment.

Read [references/configuration.md](references/configuration.md) when changing
provider, skill, MCP, A2A, attachment, structured-output, or HANA behavior.

## Serve through A2A

The generated agent exposes the same `AgentRuntime` used by the CLI:

```bash
.venv/bin/python -m template_agent serve
```

It serves the current `/.well-known/agent-card.json` discovery path, Joule's
`/.well-known/agent.json` path, and JSON-RPC at `/`. The pinned A2A 1.1.2 SDK
accepts both current `SendMessage` requests and legacy `message/send` requests
because the card advertises 1.0 and 0.3 interfaces and the route explicitly
enables v0.3 compatibility. The template also corrects the SDK's legacy error
mapping for invalid parameters and task errors. Read
`assets/template/docs/a2a.md` for exact wire examples, security boundaries,
custom-tool serving, memory requirements, and the Joule context handoff.

## Preserve the runtime contract

Keep this graph unless the use case explicitly requires another topology:

```text
START -> load_skill_node -> agent -> tools -> agent -> END
```

Keep `load_skill(skill_names: list[str])` as an ordered batch operation. Each
selected skill must load `SKILL.md` first and then all recursive UTF-8 text
files with explicit boundaries. Keep MCP tools on the official LangChain MCP
adapter and keep HANA memory limited to user/final-assistant conversation turns.
Keep A2A as a transport adapter around `AgentRuntime.ainvoke`; do not duplicate
the graph, tool registry, or memory implementation in the server.

## Validate

Run the smallest relevant checks before handing off:

```bash
.venv/bin/python -m compileall -q template_agent
.venv/bin/python -m pytest -q
.venv/bin/ruff check template_agent scripts tests
.venv/bin/python -m template_agent skills-list
.venv/bin/python -m pytest -q tests/test_a2a.py
```

Use `python -m scripts.live_model_smoke`, `python -m scripts.live_hana_smoke`,
and `python -m template_agent mcp-check` only when credentials and safe external
targets are available. Never deploy the generated application automatically.
