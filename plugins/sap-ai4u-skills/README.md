# SAP AI4U Skills

This plugin packages 33 reusable skills for SAP BTP AI development and delivery,
including SAP Generative AI Hub, LangGraph, HANA, RAG, Joule, and Cloud Foundry
workflows.

## Install

Use the [marketplace guide](../../docs/plugin-marketplace.md) for Codex and
Claude Code installation commands. The plugin lives here under `plugins/`, while
the marketplace catalogs live at the repository root for GitHub discovery.

## Package contents

- `plugin.json`: portable Agent Plugins 1.0 manifest.
- `.codex-plugin/plugin.json`: Codex compatibility metadata.
- `.claude-plugin/plugin.json`: Claude Code compatibility metadata.
- `skills/`: the 33 skills and their supporting files.

## Provenance

This copy was synchronized from the SAP AI4U plugin source at commit
`27e12f44708a42b261765f64eb7552997679b6b2` (package version `0.3.0`).
See [CHANGELOG.md](CHANGELOG.md) for the package history.
