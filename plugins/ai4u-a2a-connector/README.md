# AI4U A2A Connector

`ai4u-a2a-connector` lets Claude discover and talk to A2A agents that are
registered as SAP BTP destinations, such as `AI4U_PROJECT_LIST_AGENT`.

## Components

- `.mcp.json` starts the `a2a-agents` MCP server (`server/mcp_server.py`)
  with `python3`. It uses the standard library only, so no virtual
  environment or package install is required.
- `skills/a2a-agents/SKILL.md` teaches the assistant how to use the tools:
  choosing an agent, keeping conversation context, answering input-required
  turns, and confirming data-changing requests.

Tools: `list_agents`, `get_agent_card`, `send_message`.

## Prerequisites

- `python3` 3.9 or newer.
- The SAP BTP CLI, logged in and targeting the subaccount that holds the
  agent destinations:

  ```bash
  btp login --url https://cli.btp.cloud.sap --subdomain ai4u --sso
  ```

## Compatibility

- Claude Code and the Claude Desktop Code tab: install from the `sap-ai4u`
  marketplace (see the repository README).
- Claude Desktop: upload the plugin as a `.plugin` archive (see
  `../../docs/a2a-connector-plugin.md`).
- Codex and Agent Plugins 1.0 clients: the manifests are present and the
  skill is shared. The MCP server is wired for Claude only.

Architecture, security model, configuration, and testing are documented in
`../../docs/a2a-connector-plugin.md`.
