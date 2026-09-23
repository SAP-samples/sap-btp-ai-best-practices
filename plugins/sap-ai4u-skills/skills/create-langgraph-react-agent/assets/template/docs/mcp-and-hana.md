# MCP tools and SAP HANA memory

## MCP

The runtime uses `MultiServerMCPClient` directly. Enabled servers may use
`stdio`, `sse`, or `streamable_http`; returned tool names are prefixed with the
server name. Required-server failures abort startup, while optional failures
are recorded in diagnostics.

An example remote server, with its URL and key injected from the environment
instead of hardcoded in YAML:

```yaml
mcp:
  servers:
    my_server:
      enabled: true
      required: true
      transport: sse
      url: ${MCP_URL}
      headers:
        x-api-key: ${MCP_KEY}
      timeout_seconds: 30
```

Field meanings:

- `my_server` — the server key. It also prefixes every tool name this server
  exposes (for example `my_server.search`), so keep it short and stable.
- `enabled` — `true` loads the server at startup; `false` skips it entirely.
- `required` — `true` aborts startup if the server fails to connect; `false`
  records the failure in diagnostics and continues without its tools.
- `transport` — how the runtime reaches the server:
  - `stdio` — a local subprocess; set `command` and optional `args` instead of
    `url` (for example `command: npx`, `args: ["-y", "some-mcp-server"]`).
  - `sse` — a remote Server-Sent Events endpoint; requires `url`.
  - `streamable_http` — a remote streamable HTTP endpoint; requires `url`.
- `url` — the server endpoint for `sse` and `streamable_http`. Inject it from an
  environment variable (`${MCP_URL}`) so per-landscape hosts and secrets stay out
  of source control. Not used by `stdio`.
- `headers` — request headers sent to remote servers, typically an API key read
  from the environment (`x-api-key: ${MCP_KEY}`).
- `timeout_seconds` — per-call timeout for this server's tools (defaults to 30).

Check configured connectivity with:

```bash
.venv/bin/python -m template_agent mcp-check
```

## SAP HANA conversation memory

When `memory.enabled` is true, startup connects using the standard `HANA_*`
variables and uses the runtime user's `CURRENT_SCHEMA`. It checks `SYS.TABLES`
and `SYS.TABLE_COLUMNS`, creates `LANGGRAPH_AGENT_MEMORY` when absent, and
rejects an incompatible existing table.

The table contains `CONTEXT_ID NVARCHAR(256)`, `MESSAGES NCLOB`, and
`UPDATED_AT TIMESTAMP`. Values use bound parameters and HANA `MERGE INTO`.
Only the latest configured user/final-assistant turns are retained; tool traces,
system prompts, skill bodies, and attachment bytes are excluded.

```bash
.venv/bin/python -m template_agent clear-context demo
.venv/bin/python -m pytest -q tests/test_memory.py tests/test_mcp.py
```

Only one request should update a given `context_id` at a time.
