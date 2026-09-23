# Agent Onboarding Checklist (What to Change)

Use this checklist when teaching a user how to onboard an embedded OpenCode agent in their own system.

## 1) Agent Behavior and Identity

### `system_prompt.py`

- Set domain behavior and global rules.
- Add boundaries (what to do, what not to do).
- Keep instructions deterministic and concise.

### `.opencode/agent/<agent-slug>.md`

- Define role, style, and tool-usage policy.
- Align capabilities with the real MCP tools and skills.
- Keep examples domain-specific to user workflows.

### `.env` (`OPENCODE_AGENT`)

- Set `OPENCODE_AGENT` to the `<agent-slug>` filename without `.md`.
- Ensure exact match with `.opencode/agent/<agent-slug>.md`.

## 2) Skills (Reusable Workflows)

### `.opencode/skills/*/SKILL.md`

- Add domain procedures users need repeatedly.
- Remove generic placeholders that do not match the domain.
- Keep each skill focused and trigger description explicit.

### Skill references/scripts/assets

- Add scripts for repeated deterministic tasks.
- Add references for large domain docs/schemas.
- Keep SKILL.md concise and point to references as needed.

## 3) MCP Tools and Tool Registry

### `tools/mcp_server.py`

- Implement tool functions required by the user domain.
- Validate function signatures and error handling.
- Avoid exposing unnecessary risky actions.

### `opencode.json`

- Register/enable MCP servers that back those tools.
- Remove unused tool servers to reduce attack surface.
- Keep model/tool configuration aligned with deployment.

### `.opencode/skills/mcp-tools/SKILL.md`

- Document exactly the tool names and usage pattern.
- Keep the documented tool list synchronized with `tools/mcp_server.py`.

## 4) Guardrails and Permissions

### `.opencode/config.json`

- Limit command/file/tool permissions to least privilege.
- Keep allowlists explicit and minimal.
- Ensure dangerous operations are blocked by default.

## 5) Integration into Host Application

### Host app router mount

- Add bridge module (for example `routers/opencode_agent.py`) that imports embedded `a2a_server.app`.
- Mount in host `main.py` with desired base path.

### Embedded `a2a_server.py`

- Reuse host security dependency (`Depends(get_api_key)`) when required.
- Keep `/diagnostic` endpoint enabled for troubleshooting.

### Embedded `env_utils.py`

- Load host-level env first (for example `api/.env` before embedded `.env`).

## 6) Runtime Environment

### `.env` / `.env.example`

- Set `A2A_BASE_URL` to include mounted path.
- Set `A2A_SESSIONS_FILE` to writable location.
- Set model/runtime values: `OPENCODE_COMMAND`, `OPENCODE_MODEL`, `OPENCODE_TIMEOUT`.
- Provide required secrets (`AICORE_*`, `API_KEY`, others as needed).

## 7) Non-Interactive Subprocess Requirement

For every OpenCode subprocess call (`asyncio.create_subprocess_exec`), enforce:

- `stdin=asyncio.subprocess.DEVNULL`
- `OPENCODE_NONINTERACTIVE=1`

This prevents hangs in integrated server mode.

## 8) Validation Before Hand-off

- `GET <mount-path>/.well-known/agent-card.json`
- `POST <mount-path>/a2a` with valid auth header
- `GET <mount-path>/diagnostic`
- Verify skill/tool documentation matches actual tool implementations
- Verify `OPENCODE_AGENT` slug consistency
- Verify `stdin=asyncio.subprocess.DEVNULL` still present in runtime subprocess calls
