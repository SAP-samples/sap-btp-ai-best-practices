---
name: opencode-fastapi-embed
description: Embed an OpenCode A2A template into a larger FastAPI application and wire mounted routing, shared API-key security, host-level dotenv loading, and runtime diagnostics. Teach users how to onboard the embedded agent in their own system by customizing system prompts, agent definition, skills, MCP tools, runtime config, and validation steps. Use when migrating `a2a_opencode_*` template agents into an existing `api/app` project or when reproducing the UI5 web components integration pattern.
---

# OpenCode FastAPI Embed

Use this workflow to integrate a standalone OpenCode A2A template into a host FastAPI app.

## Inputs

- Source template directory (for example `a2a_opencode_template`, `a2a_opencode_orchestrator`, `a2a_opencode_perplexity`)
- Host app directory (for example `api/app`)
- Mount path (default `/api/opencode-agent`)
- Router bridge module name (default `opencode_agent`)

## Run the Embed Script

Use the bundled script to copy and patch the template in one pass.

```bash
python3 scripts/embed_opencode_template.py \
  --source-template /path/to/source/a2a_opencode_template \
  --target-app-dir /path/to/api/app \
  --embedded-name a2a_opencode_template \
  --mount-path /api/opencode-agent \
  --router-module opencode_agent \
  --api-env-example /path/to/api/.env.example
```

What it applies:

- Copies template into host app package.
- Patches embedded `env_utils.py` to prioritize `api/.env`.
- Patches embedded `a2a_server.py` to optionally depend on `..security.get_api_key`.
- Adds `/diagnostic` endpoint to embedded `a2a_server.py` when missing.
- Ensures `stdin=asyncio.subprocess.DEVNULL` in embedded `agent.py`.
- Creates router bridge module in host `routers/`.
- Patches host `main.py` import and `app.mount(...)`.
- Optionally updates `api/.env.example` with mounted `A2A_BASE_URL` and sessions file path.

## Validate the Result

Run the verifier script:

```bash
python3 scripts/verify_integration.py \
  --target-app-dir /path/to/api/app \
  --embedded-name a2a_opencode_template \
  --mount-path /api/opencode-agent \
  --router-module opencode_agent \
  --templates-root /path/to/opencode_template_repo
```

## Required Invariants

- Keep every OpenCode runtime subprocess non-interactive:
  - `stdin=asyncio.subprocess.DEVNULL`
  - `OPENCODE_NONINTERACTIVE=1`
- Ensure mounted URL is reflected in `A2A_BASE_URL`.
- Ensure session file path is writable in target runtime.
- Ensure API key enforcement is active for mounted A2A routes in production.

## Teach User Onboarding

When asked how to onboard the agent into a user system, provide a concrete file-by-file change plan covering:

- System behavior: `system_prompt.py`
- Agent identity/role: `.opencode/agent/<slug>.md` and `OPENCODE_AGENT`
- Reusable workflows: `.opencode/skills/*/SKILL.md`
- Tooling surface: `tools/mcp_server.py` and `opencode.json`
- Runtime guardrails: `.opencode/config.json`
- Runtime metadata and secrets: `.env` / `.env.example`
- Integration wiring: host `main.py`, router bridge, `a2a_server.py`, `env_utils.py`
- Verification: agent card, JSON-RPC calls, diagnostics, and `stdin=asyncio.subprocess.DEVNULL` checks

Use this checklist:

- `references/onboarding-checklist.md`

## Manual Follow-up

- Confirm `OPENCODE_AGENT` matches `.opencode/agent/<slug>.md`.
- Confirm `OPENCODE_COMMAND` is valid in deployment (`opencode` or `npx opencode`).
- Confirm MCP tool docs under `.opencode/skills` match actual tools exposed by `tools/mcp_server.py`.

## Reference

- For concrete reference implementation and file-level changes, read:
  - `references/reference-implementation.md`
  - `references/onboarding-checklist.md`
