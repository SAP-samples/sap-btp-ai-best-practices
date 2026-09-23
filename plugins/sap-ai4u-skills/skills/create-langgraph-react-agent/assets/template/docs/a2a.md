# A2A connectivity

## Purpose

`template_agent.a2a_server` exposes the existing skill-aware LangGraph runtime
to current A2A clients and SAP Joule. It is a transport adapter: MCP tools,
provider selection, skill loading, and optional HANA conversation memory still
belong to the single `AgentRuntime` instance created during application startup.

## Inputs and outputs

The first A2A boundary accepts one or more non-empty text parts. It maps the A2A
`contextId` to the runtime `context_id`. Reusing that identifier continues model
history only when HANA memory is enabled; with the generated default
`memory.enabled: false`, it is correlation metadata only. If a client omits
`contextId`, the A2A SDK creates one and the response returns it.

Successful requests return a completed task containing an artifact named
`agent_result`; its first text part is `AgentResult.output_text`. Runtime
failures return a failed task with a generic message so exception text cannot
expose credentials.

Remote file, URL, and data parts are intentionally rejected at this boundary.
The local Python API and CLI continue to support image/PDF `Attachment` inputs.

## Configuration

Set the card metadata and listener in `config/agent.yaml`:

```yaml
a2a:
  enabled: true
  name: Portable LangGraph Agent
  description: Skill-aware LangGraph agent with MCP tools
  version: 0.1.0
  public_url: ${A2A_PUBLIC_URL}
  host: 127.0.0.1
  port: 8080
```

For local testing, `.env` may contain:

```dotenv
A2A_PUBLIC_URL=http://localhost:8080
```

The generated listener is loopback-only. Before a human deploys the application,
add authenticated ingress and caller/tenant-scoped A2A contexts, then change the
listener and public URL for that environment. The advertised URL never falls
back to an invented Cloud Foundry route.

## Security and task retention

The base template intentionally contains no opinionated authentication
middleware. Do not bind it to a public interface or place it on an externally
reachable route as-is: anonymous callers could invoke configured model/MCP
tools, and A2A task operations share one anonymous task namespace. A deployment
adapter must authenticate callers, build caller-scoped server contexts, and
namespace or authorize client-supplied `contextId` values.

`InMemoryTaskStore` is process-local, retains task prompts and answers until the
process exits, and loses them on restart. Replace it with an authorized durable
store only when task lookup must survive restarts. HANA conversation memory is
separate from the A2A task store. Per-context invocation locks prevent lost turns
inside one process, including when cancellation arrives during a blocking
storage call; multi-process deployments need equivalent distributed coordination
if the same HANA context can be handled by multiple workers.

## Run and discover

```bash
.venv/bin/python -m template_agent serve
curl -s http://localhost:8080/.well-known/agent-card.json | python -m json.tool
curl -s http://localhost:8080/.well-known/agent.json | python -m json.tool
```

Both cards advertise JSON-RPC interfaces for A2A 1.0 and 0.3 at `/`. The
server pins `a2a-sdk[http-server]==1.1.2` and uses its v0.3 compatibility adapter
on that route. A narrow wrapper preserves A2A error codes that SDK 1.1.2 would
otherwise collapse into `-32603` on synchronous legacy requests. Streaming is
advertised as unsupported and is outside this Joule `message/send` contract.

## Current A2A request

A2A 1.0 requests select the current protocol with the `A2A-Version` header:

```bash
curl -s -X POST http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -H 'A2A-Version: 1.0' \
  -d '{
    "jsonrpc": "2.0",
    "id": "request-1",
    "method": "SendMessage",
    "params": {
      "message": {
        "messageId": "message-1",
        "contextId": "conversation-1",
        "role": "ROLE_USER",
        "parts": [{"text": "What can you help me with?"}]
      }
    }
  }'
```

## Joule-compatible A2A 0.3 request

Joule's remote `agent-request` uses the legacy JSON-RPC method and part shape.
No `A2A-Version` header is required because the protocol default is 0.3:

```bash
curl -s -X POST http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "request-joule",
    "method": "message/send",
    "params": {
      "message": {
        "messageId": "message-joule",
        "contextId": "conversation-joule",
        "role": "user",
        "parts": [{"kind": "text", "text": "What can you help me with?"}]
      }
    }
  }'
```

The legacy result contains:

```json
{
  "contextId": "conversation-joule",
  "status": {"state": "completed"},
  "artifacts": [
    {
      "name": "agent_result",
      "parts": [{"kind": "text", "text": "...final agent answer..."}]
    }
  ]
}
```

Joule can render `artifacts[0].parts[0].text` and persist `contextId` for the
next turn, matching the contract expected by `sap-agent-joule-cf-bootstrap`.
Enable HANA memory if that round-trip must also retain prior model turns.

## Use-case tools while serving

The built-in `serve` command uses the base configured runtime. If a use case
adds Python tools through `extra_tools`, expose a small application module that
passes the same tools through the A2A runtime factory:

```python
from pathlib import Path

from template_agent.a2a_server import create_a2a_app
from template_agent.runtime import AgentRuntime

from use_case_tools import build_tools


async def create_runtime(config_path: str | Path) -> AgentRuntime:
    """Create the shared runtime with the use case's local tools."""

    return await AgentRuntime.create(config_path, extra_tools=build_tools())


app = create_a2a_app("config/agent.yaml", runtime_factory=create_runtime)
```

Run that module with `uvicorn app:app`. YAML-configured MCP tools already enter
the default runtime and need no custom factory.

## Related files

- `template_agent/a2a_server.py`: card, executor adapter, routes, and lifespan.
- `template_agent/runtime.py`: shared LangGraph runtime invoked by A2A.
- `template_agent/config.py`: validated `A2ASettings`.
- `template_agent/__main__.py`: `serve` command.
- `tests/test_a2a.py`: current and Joule protocol contract tests.
- `config/agent.yaml`: card and listener configuration.

## Test

The protocol tests use an in-memory runtime and make no SAP, MCP, HANA, Joule,
or Cloud Foundry calls:

```bash
.venv/bin/python -m pytest -q tests/test_a2a.py tests/test_models_config.py
.venv/bin/ruff check template_agent tests/test_a2a.py tests/test_models_config.py
```

Do not deploy the generated application automatically. Use
`sap-agent-joule-cf-bootstrap` only after the local A2A tests pass, and leave all
Cloud Foundry and Joule deployment commands to the user.
