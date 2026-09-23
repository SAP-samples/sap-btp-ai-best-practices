# Changelog

All notable changes to `sap-ai4u-skills` are recorded here.

## 0.3.0 - 2026-09-23

- Add standalone `cf-cli`, `btp-cli`, and `joule-cli` skills for Cloud Foundry
  operations, BTP account and service administration, and Joule Studio CLI
  authoring and lifecycle checks.
- Adapt the pinned A2A Agent Toolkit examples to the current Joule Studio CLI,
  live CLI help, scoped targets, credential-safe output, and manual application
  and Joule deployment.
- Route Joule A2A integration and Destination setup to the existing
  `sap-agent-joule-cf-bootstrap` skill instead of duplicating its workflow.
- Document the new skill inputs, outputs, boundaries, and validation; update
  the catalog description and count to 33 top-level skills.

## 0.2.1 - 2026-09-14

- Redesign `sap-agent-joule-cf-bootstrap` as a focused integration and
  lifecycle skill around an existing LangGraph A2A agent, delegating missing
  agent creation to `create-langgraph-react-agent` instead of duplicating it.
- Add Joule A2A 0.3 context/task continuity, schema 3.28.0 artifacts, and
  distinct direct and interpretability response modes without duplicate final
  messages.
- Add credential-free Cloud Foundry rendering with authenticated production
  ingress guardrails and explicit development-only unauthenticated output.
- Add a dry-run-first, redacted OAuth2 Client Credentials BTP Destination
  upsert using native `connectivity/destination` commands and secure temporary
  configuration files.
- Add separate Cloud Foundry, BTP CLI, Joule Studio CLI, and A2A contract
  references plus focused unit and packaging tests.
- Remove the obsolete full-project bootstrap, duplicate LangGraph application
  template, local SQLite defaults, dependency files, and automatic deployment
  script from this skill.

## 0.2.0 - 2026-09-12

- Add `create-langgraph-react-agent` with the existing SAP Generative AI Hub,
  dynamic-skill, MCP, and optional HANA runtime plus A2A 1.0 and
  Joule-compatible A2A 0.3 JSON-RPC connectivity.
- Add current and legacy agent-card discovery, context propagation, safe task
  failures, runtime lifecycle management, a `serve` command, and offline
  protocol contract tests pinned to `a2a-sdk[http-server]==1.1.2`.
- Preserve legacy invalid-parameter and task error codes around an SDK 1.1.2
  compatibility defect, serialize concurrent turns per context through
  cancellation-safe storage completion, and make loopback the unauthenticated
  development-server default.
- Replace `rpt-1` with the reusable direct AI Core REST client for RPT-1.5 and
  RPT-1.6 model variants, JSON/gzip/Parquet inference, classification,
  regression, multi-target output, confidence intervals, explainability, and
  context modes.
- Harden all RPT response-validation paths so malformed payload diagnostics and
  boundary-truncated excerpts cannot retain echoed credentials.
- Refresh `genai-token-caching` with GPT-5.6 cache controls and telemetry,
  `langgraph-fundamentals` with related-skill routing, and
  `sap-document-ai-client` with the complete schema-field contract.
- Increase the packaged catalog from 30 to 31 skills and add repository-level
  feature, compatibility, and validation documentation.

## 0.1.0 - 2026-08-27

- Package the 30 skills imported from `Agent-skills-catalog` commit
  `6c8e86cf85a085347a04bc7934d40cd4e557282b`.
- Add portable Agent Plugins 1.0, Codex, and Claude Code manifests.
- Add Codex and Claude Code marketplace entries under `sap-ai4u`.
- Replace local-only references in the imported package with portable guidance.
