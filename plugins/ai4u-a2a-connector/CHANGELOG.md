# Changelog

All notable changes to `ai4u-a2a-connector` are recorded here.

## 0.1.0 - 2026-09-29

- Initial release:
  - `a2a-agents` MCP server (standard library only) that discovers A2A agents among the BTP destinations of the `btp` CLI target and sends A2A 0.3 `message/send` requests, with destination credentials kept server-side.
  - `a2a-agents` skill for agent selection, multi-turn context, input-required turns, and confirmation of data-changing requests.
