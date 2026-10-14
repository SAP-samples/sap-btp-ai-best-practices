# AI4U A2A Connector plugin

## What it is

`ai4u-a2a-connector` is the third plugin in the `sap-ai4u` marketplace. It
lets Claude (Claude Code, the Claude Desktop Code tab, and Claude Desktop)
discover and hold conversations with external **A2A agents** that are
registered as **SAP BTP destinations**. The reference use case is the AI4U
project list agent (destination `AI4U_PROJECT_LIST_AGENT`), a LangGraph agent
that manages the AI4U portal project list in HANA.

It has two components:

| Component | Path | Role |
| --- | --- | --- |
| MCP server `a2a-agents` | `plugins/ai4u-a2a-connector/server/` | Reads destinations with the `btp` CLI, finds agent cards, sends A2A messages |
| Skill `a2a-agents` | `plugins/ai4u-a2a-connector/skills/a2a-agents/SKILL.md` | Tells the assistant when and how to use the tools safely |

## Why

- Agents built from the `create-langgraph-react-agent` template and connected
  to Joule already have a BTP destination. The destination list is therefore
  the team's de facto agent registry, and it already stores each agent's URL
  and credentials.
- Reusing the user's `btp` SSO session means no service keys, tokens, or
  agent URLs have to be configured on the laptop.

## Architecture

```
Claude ──stdio JSON-RPC──> server/mcp_server.py
                             │
                             ├─ destinations.py ── btp --format json list connectivity/destination
                             │                     (session from `btp login ... --sso`, subaccount from `btp target`)
                             │
                             └─ a2a_client.py ──── GET  <destination>/.well-known/agent-card.json (probe)
                                                   POST <card url>  A2A 0.3 JSON-RPC `message/send`
```

1. **Destination listing.** `btp --format json list connectivity/destination`
   returns every destination of the targeted subaccount, including secrets.
   The server keeps them in memory only.
2. **Discovery.** Every `Type: HTTP`, `ProxyType: Internet` destination is
   probed concurrently (32 workers, 5 s timeout). Candidate card URLs, in
   order:
   - `<url>/.well-known/agent-card.json` and `<url>/.well-known/agent.json`
   - the same two under `<url>/a2a`, unless the URL already ends in `/a2a`
   - the same two at the host root

   Probes go out without credentials first. Destination credentials are added
   only when the server answers 401/403. A destination is an agent when a card
   (a JSON object with `name` plus `url` or `supportedInterfaces`) is returned.
   Results are cached for the lifetime of the server process.
3. **Messaging.** The JSON-RPC URL comes from the card: the top-level `url`
   (A2A 0.3), otherwise the 0.3 entry of `supportedInterfaces`. The server
   sends A2A 0.3 `message/send`, the same contract Joule uses, so every
   Joule-registered agent supports it. The reply is flattened to
   `{state, text, context_id, task_id}`.

The destination layouts observed in the Dev subaccount on 2026-09-29, and how
the probe order covers them:

| Destination URL | Card location |
| --- | --- |
| `https://ai4u-project-list-agent-api...` (application root) | `/a2a/.well-known/agent-card.json` |
| `https://pending-loads-a2a-agent.../a2a` (A2A mount) | host root `/.well-known/agent-card.json` (requires `X-API-Key`) |
| `https://weather-forecast-agent...` | `/.well-known/agent-card.json` |

## Tools: inputs and outputs

| Tool | Input | Output |
| --- | --- | --- |
| `list_agents` | `refresh?: boolean` | `{"agents": [{"agent", "name", "description", "version", "authentication", "callable", "skills": [{"name", "description", "examples"}]}], "destinations_scanned": int}` |
| `get_agent_card` | `agent: string` (destination name, case-insensitive) | The agent card JSON |
| `send_message` | `agent: string`, `message: string`, `context_id?: string`, `task_id?: string` | `{"state", "text", "context_id", "task_id"}` |

`send_message` semantics:

- Pass the returned `context_id` on follow-ups to continue the conversation.
- `task_id` is returned only when `state` is `input-required` or
  `auth-required`. It must then be sent back with the answer, which mirrors
  the Joule remote-agent contract.
- JSON-RPC errors come back as `state: "error"`. Transport, login, and auth
  problems come back as MCP tool errors whose text explains the fix.

Example `send_message` result from the live project list agent:

```json
{
  "state": "completed",
  "text": "I found **Document Grounding**. ...",
  "context_id": "9ec5118d-b647-49e5-b1a9-cd70bb33fc31",
  "task_id": null
}
```

## Authentication and security model

| Destination property | Handling |
| --- | --- |
| `URL.headers.<Name>` | Sent as header `<Name>` (e.g. `X-API-Key`) |
| `URL.queries.<Name>` | Added as query parameter `<Name>` |
| `Authentication: NoAuthentication` | No `Authorization` header |
| `Authentication: BasicAuthentication` | `Authorization: Basic` from `User`/`Password` |
| `Authentication: OAuth2ClientCredentials` | Client-credentials token from `tokenServiceURL`, cached until 60 s before expiry |
| Any other type (e.g. `OAuth2UserTokenExchange`) | Agent listed with `callable: false`; calls fail with an explanation |

Guarantees enforced in code and covered by `test_connector.py`:

- **Secrets stay server-side.** Destination secrets never appear in tool
  results or error messages. The raw configuration is excluded from `repr`,
  token-endpoint failures report only the HTTP status, and tool output
  contains no URLs with query secrets.
- **Credentials only go to the destination's own host.** If an agent card
  advertises a JSON-RPC URL on another scheme, host, or port, the agent is
  skipped.
- **Data changes need user confirmation.** The skill requires Claude to show
  the exact message and get an explicit yes before asking an agent to create,
  update, delete, commit, revert, upload, or publish.
- **Agent replies are untrusted.** The skill tells Claude never to act on
  instructions contained in agent output.

Observation (not changed by this plugin): `AI4U_PROJECT_LIST_AGENT` is a
`NoAuthentication` destination on a public route, and the agent can modify
project data. Anyone who knows the URL can call it. Moving it to
`OAuth2ClientCredentials` (the production default in
`sap-agent-joule-cf-bootstrap`) works with this connector without code
changes.

## Configuration

All settings are optional environment variables of the MCP server process.

| Variable | Default | Purpose |
| --- | --- | --- |
| `A2A_BTP_SUBACCOUNT` | `btp target` | Subaccount ID whose destinations are read |
| `A2A_BTP_SUBDOMAIN` | `ai4u` | Global-account subdomain shown in the login hint |
| `BTP_CLI_PATH` | `PATH`, then `/opt/homebrew/bin/btp`, `/usr/local/bin/btp`, `~/bin/btp` | Explicit `btp` binary |
| `A2A_TIMEOUT_SECONDS` | `180` | Timeout of one `message/send` call |

To set them in Claude Code, add an `env` block to the server in
`plugins/ai4u-a2a-connector/.mcp.json`, or export them before starting
`claude`.

## Installation

Prerequisites: `python3` 3.9 or newer (the macOS system Python works), and the
SAP BTP CLI logged in:

```bash
btp login --url https://cli.btp.cloud.sap --subdomain ai4u --sso
```

```bash
btp target
```

### Claude Code (CLI and Desktop Code tab)

```bash
claude plugin marketplace add 'https://github.com/SAP-samples/sap-btp-ai-best-practices.git#sap-ai4u-plugin'
```

```bash
claude plugin install ai4u-a2a-connector@sap-ai4u
```

Restart Claude Code. `/mcp` should list the plugin's `a2a-agents` server as
connected.

### Claude Desktop (`.plugin` upload)

The Desktop app accepts a `.zip` or `.plugin` archive that has
`.claude-plugin/plugin.json` at its root. Build it from the plugin folder's
contents, not the folder itself:

```bash
mkdir -p dist && (cd plugins/ai4u-a2a-connector && zip -qr ../../dist/ai4u-a2a-connector.plugin . -x '*.DS_Store' '*__pycache__*')
```

Upload `dist/ai4u-a2a-connector.plugin` in the Desktop app. `dist/` is
git-ignored.

### Manual MCP-only fallback

When a client cannot install plugins, register only the server. This fallback
has no skill. For example, in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "a2a-agents": {
      "command": "python3",
      "args": ["-B", "/absolute/path/to/sap-btp-ai-best-practices/plugins/ai4u-a2a-connector/server/mcp_server.py"]
    }
  }
}
```

## Key code

Result flattening (`server/a2a_client.py`). A waiting agent asks its question
in the status message; a finished agent answers in artifacts:

```python
if state in PENDING_STATES:
    texts = status_texts or artifact_texts
else:
    texts = artifact_texts or status_texts
return {
    "state": state,
    "text": "\n".join(texts),
    "context_id": result.get("contextId"),
    "task_id": result.get("id") if state in PENDING_STATES else None,
}
```

Host guard (`server/a2a_client.py`):

```python
def same_host(first: str, second: str) -> bool:
    a, b = urllib.parse.urlsplit(first), urllib.parse.urlsplit(second)
    return (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)
```

### Why the standard library only

Claude Desktop starts plugin servers with a minimal GUI `PATH` and no project
virtual environment, and corporate proxies can block PyPI. MCP over stdio
needs only `initialize`, `ping`, `tools/list`, and `tools/call`, so a small
hand-written loop avoids installing the MCP SDK. If resources, prompts, or
streaming are ever needed, move to the official Python MCP SDK (FastMCP).

## Related files

| Path | Role |
| --- | --- |
| `plugins/ai4u-a2a-connector/.mcp.json` | Starts the MCP server |
| `plugins/ai4u-a2a-connector/server/mcp_server.py` | MCP stdio loop, tool schemas, dispatch |
| `plugins/ai4u-a2a-connector/server/destinations.py` | `btp` CLI access, destination parsing, request authentication |
| `plugins/ai4u-a2a-connector/server/a2a_client.py` | Card discovery, `message/send`, result flattening |
| `plugins/ai4u-a2a-connector/server/test_connector.py` | Offline end-to-end self-check |
| `plugins/ai4u-a2a-connector/skills/a2a-agents/SKILL.md` | Usage and safety instructions for the assistant |
| `plugins/ai4u-a2a-connector/{plugin.json,.claude-plugin/,.codex-plugin/}` | Portable, Claude, and Codex manifests |
| `.claude-plugin/marketplace.json`, `.agents/plugins/marketplace.json` | Marketplace entries |

## How to test

Offline, with no BTP session or network. This uses a local fake agent and a
fake `btp`, and drives the real server over stdio:

```bash
python3 plugins/ai4u-a2a-connector/server/test_connector.py
```

Package validation:

```bash
python3 scripts/validate_package.py
```

Live, with a `btp` session. Send JSON-RPC lines to the server by hand:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_agents","arguments":{}}}' | python3 -B plugins/ai4u-a2a-connector/server/mcp_server.py
```

In Claude Code, load the plugin from the checkout without installing:

```bash
claude --plugin-dir plugins/ai4u-a2a-connector
```

Then ask: "Which A2A agents can you reach?" and "Ask the project list agent
to find the Document Grounding project."

Results on 2026-09-29 against the Dev subaccount:

- 73 destinations scanned and 9 agents found in about 8 seconds.
- Two read-only turns with `AI4U_PROJECT_LIST_AGENT` kept the same
  `context_id`.
- 64 real destination secret values were checked, and none appeared in the
  MCP output.

## Limits

- Only A2A 0.3 JSON-RPC `message/send` is used. Add the A2A 1.0
  `SendMessage` branch if an agent appears without a 0.3 interface.
- There is no `tasks/get` polling. Agents that return `working` are reported
  as such.
- Streaming, file parts, and data parts are not sent. Replies are reduced to
  their text parts.
- Destinations with user-token flows (`OAuth2UserTokenExchange`, SAML,
  principal propagation) and on-premise destinations are not reachable from a
  laptop.
- The MCP server is wired for Claude only. Codex receives the skill, not the
  server.
