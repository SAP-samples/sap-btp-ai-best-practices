---
name: mcp-routing
description: "Use when a request should be handled by MCP tools (external integrations, exact utility calls, or cross-system actions)."
---

# MCP Routing Skill

The template includes MCP tools exposed from `tools/mcp_server.py`.

## Guidance

- Prefer MCP tools for exact arithmetic/time calls when available.
- MCP tools are discovered dynamically at startup from configured server(s).
- Tool names are server-prefixed (for example `example_tools_add`, `example_tools_get_utc_time`).

## Failure Handling

- If MCP call fails, explain the failure briefly.
- Continue with best-effort reasoning when possible.
