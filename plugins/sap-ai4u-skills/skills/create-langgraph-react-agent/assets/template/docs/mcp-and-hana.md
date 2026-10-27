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

### What it is for

With `memory.enabled: true`, a conversation (`context_id`, the A2A `contextId`)
keeps its full history across turns, process restarts, and Cloud Foundry
instances. Without it, every turn starts empty.

### Storage

Startup connects with the standard `HANA_*` variables, uses the runtime user's
`CURRENT_SCHEMA`, checks `SYS.TABLES`/`SYS.TABLE_COLUMNS`, and creates
`LANGGRAPH_AGENT_MEMORY` when absent. One row per context:

| Column | Type | Purpose |
|--------|------|---------|
| `CONTEXT_ID` | `NVARCHAR(256)` PK | Conversation key |
| `MESSAGES` | `NCLOB` | `{"format": 2, "messages": [...]}` from `messages_to_dict`: user, assistant, tool-call and tool-result messages |
| `PENDING_ACTION` | `NCLOB` | Tool calls awaiting approval, or `NULL` |
| `VERSION` | `BIGINT` | Incremented on every save (optimistic concurrency) |
| `LEASE_OWNER` / `LEASE_EXPIRES_AT` | `NVARCHAR(64)` / `TIMESTAMP` | Which turn currently owns the conversation |
| `UPDATED_AT` | `TIMESTAMP` | Last save |

A table created by an earlier template version (only the first three columns,
role/content JSON) is migrated automatically with `ALTER TABLE ... ADD`, and its
rows stay readable. System prompts, skill bodies, and attachment bytes are never
stored; attachments become a short text placeholder.

### One turn

```text
asyncio.Lock(context)            queue same-process turns without touching HANA
acquire lease                    UPDATE ... WHERE lease free or expired, read back owner
                                 busy: retry with backoff up to memory.lock_wait_seconds,
                                 then ContextBusyError
load                             MESSAGES, PENDING_ACTION, VERSION
resolve pending action           approval reply runs held tool calls, other replies reject
run graph                        compact_context keeps history bounded (see below)
save + release                   UPDATE ... SET VERSION = VERSION + 1, LEASE_OWNER = NULL
                                 WHERE VERSION = ? AND LEASE_OWNER = ?
                                 0 rows: ConcurrentUpdateError, nothing overwritten
on failure                       release the lease
```

`memory.lease_seconds` (default 300) must exceed the longest expected turn.
If a lease still expires mid-turn, another instance may start, and the
version check rejects the stale save.

### History size

Before every model call the `compact_context` node:

- truncates tool results above `tool_output.max_output_characters` (30000);
- summarizes older messages with LangChain `SummarizationMiddleware` once the
  history passes `summarization.trigger_tokens` (60000), keeping the last
  `summarization.keep_messages` (6) verbatim. `summarization.model` may name a
  cheaper deployment.

On save, `memory.max_messages` (200) is a hard ceiling applied with
`trim_messages(start_on="human")`, so no tool result is left without its call.

### Tool approvals

`approvals.tools` lists tools that must be approved before they run (requires
memory). A call to one ends the turn with `AgentResult.status ==
"input_required"`, an approval request listing each call and its arguments,
and the calls stored in `PENDING_ACTION`. The next message on the same context
resolves it on any instance: `yes`, `ok`, `approve`, `confirm`, `proceed`,
`go ahead` (case-insensitive, trailing `.`/`!` ignored) runs the calls; any
other reply rejects them and is passed to the model as feedback.

### Commands and tests

```bash
.venv/bin/python -m template_agent clear-context demo
.venv/bin/python -m pytest -q tests/test_memory.py tests/test_runtime.py tests/test_mcp.py
.venv/bin/python -m scripts.live_hana_smoke          # needs HANA_* in .env
```

`live_hana_smoke` uses two store objects as two instances against the real
table: lease exclusivity, versioned save, stale-save rejection, lease expiry.
