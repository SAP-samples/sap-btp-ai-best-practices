# Reference Implementation Pattern (Mounted OpenCode Agent)

This skill uses a generic integration pattern:

- Host application structure: `api/app`
- Embedded agent structure: `api/app/a2a_opencode_template`
- Mounted route example: `/api/opencode-agent`

Treat this as a portable pattern, not as a path-specific implementation.

## Host App Wiring

1. Add bridge router:
   - `api/app/routers/opencode_agent.py`
2. Mount in host app:
   - `api/app/main.py`
   - `app.mount("/api/opencode-agent", opencode_agent.app)`
3. Protect with shared API key dependency:
   - host dependency in `api/app/security.py`
   - embedded app consumes `..security.get_api_key`

## Embedded Template Patches

1. `env_utils.py`
   - Add `API_ROOT = PROJECT_ROOT.parent.parent`
   - Load env in this order:
     - `API_ROOT/.env`
     - `PROJECT_ROOT/.env`
     - `API_ROOT/.env.example`
     - `PROJECT_ROOT/.env.example`

2. `a2a_server.py`
   - Add `Depends` import.
   - Add optional import of `..security.get_api_key`.
   - Build FastAPI app with shared dependency.
   - Add `/diagnostic` endpoint for opencode command/runtime checks.

3. `agent.py`
   - Ensure subprocess call contains:
     - `stdin=asyncio.subprocess.DEVNULL`

## Environment

In `api/.env` or `api/.env.example`:

- `A2A_BASE_URL` includes mount path (`/api/opencode-agent`).
- `A2A_SESSIONS_FILE` points to embedded data path or `/tmp/...`.
- `OPENCODE_AGENT` matches `.opencode/agent/<slug>.md`.

## Runtime Checks

- `GET /api/opencode-agent/.well-known/agent-card.json`
- `POST /api/opencode-agent/a2a`
- `GET /api/opencode-agent/diagnostic`
