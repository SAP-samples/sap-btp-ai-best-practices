# Portable LangGraph ReAct agent

A minimal SAP Generative AI Hub agent with dynamic skills, LangChain MCP tools,
dual-stack A2A connectivity, and optional SAP HANA conversation memory. The
graph is deliberately small:

`START -> load_skill_node -> agent -> tools -> agent -> END`

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

For tests and linting, install `requirements-dev.txt`, which includes the exact
runtime pins:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/ruff check template_agent scripts tests
```

Python 3.12 can be used when 3.11 is unavailable. Configure the deployment in
`config/agent.yaml`, then run:

```bash
.venv/bin/python -m template_agent skills-list
.venv/bin/python -m template_agent ask "What can you do?" --context-id demo
.venv/bin/python -m template_agent chat --context-id demo
.venv/bin/python -m template_agent serve
```

The A2A server exposes A2A 1.0 and Joule-compatible 0.3 JSON-RPC at `/`, with
agent cards at `/.well-known/agent-card.json` and
`/.well-known/agent.json`. See `docs/` for the public Python API, A2A wire
contract, skills, MCP, HANA, and validation paths. It binds to loopback and has
no authentication by default; follow the A2A security guidance before exposing
it through a public route.
