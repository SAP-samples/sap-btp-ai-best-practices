# SAP AI4U Skills

`sap-ai4u-skills` packages 35 curated skills for SAP BTP AI development and
delivery. The package covers SAP Gen AI Hub, LangGraph, MCP and A2A agent
connectivity, HANA, vector and knowledge-graph RAG, hana-ml, RPT-1 tabular
prediction with SAP-RPT-TS time-series forecasting and RPT typed decisions,
Joule, Document AI, token observability, Cloud Foundry, BTP CLI, and supporting
project workflows.

## Compatibility

The plugin contains one shared `skills/` tree and three manifests:

- `plugin.json` for Agent Plugins 1.0 compatible clients.
- `.codex-plugin/plugin.json` for Codex.
- `.claude-plugin/plugin.json` for Claude Code.

Client-specific installation and updates are documented in the repository
README. The 0.2.1 Joule and Cloud Foundry lifecycle redesign is documented in
`../../docs/joule-cf-lifecycle-skill.md`.
The 0.3.0 platform CLI skills are documented in
`../../docs/platform-cli-skills.md`.
The 0.4.0 HANA memory, concurrency, and approval changes are documented in
`../../docs/agent-memory-and-concurrency.md`.

## Provenance

The initial package was imported from `Agent-skills-catalog` commit
`6c8e86cf85a085347a04bc7934d40cd4e557282b`. This repository is the canonical
source for packaged skills after that import; the source repository remains an
unchanged historical snapshot.
