# Joule and Cloud Foundry lifecycle skill

## Purpose

Release 0.2.1 redesigns `sap-agent-joule-cf-bootstrap` into a focused
integration and lifecycle skill that wraps an **existing** LangGraph A2A agent
with the Joule, Cloud Foundry, and BTP Destination layer it needs to run as a
Joule code-based agent.

The previous version also bootstrapped a full agent project (application source,
model layer, dependencies, local persistence, and a deploy script). That
responsibility now belongs to `create-langgraph-react-agent`. This skill no
longer duplicates it: it prepares, validates, and manages the integration
around an agent that already exists.

This repository packages development assets only. The skill does not deploy an
application to Cloud Foundry, register a Joule capability, or upsert a live BTP
Destination on the user's behalf. It renders assets and prepares commands; the
user submits every deployment command manually.

## Existing agent as the starting point

The workflow begins with a scope decision:

- If an A2A agent already exists, inspect it and continue with this skill.
- If none exists, invoke `create-langgraph-react-agent` to create it, then
  return here and add only the Joule, Cloud Foundry, and Destination layer.
- If another specialist skill owns the target agent, invoke that skill instead
  of duplicating its behavior.

The skill owns integration and lifecycle management. It never recreates the
removed from-scratch bootstrap, a second runtime, a model layer, a dependency
set, or a full project scaffold.

## What the redesign adds

| Area | Behavior |
| --- | --- |
| A2A contract | Joule A2A 0.3 JSON-RPC `message/send`, with `contextId`/`taskId` continuity across multi-turn and `input-required` flows, kept under Joule's 60-second limit |
| Response modes | Distinct `direct` (emit the completed artifact) and `interpretability` (`response_context` generates the final message) modes, without a duplicate final message |
| Cloud Foundry | Credential-free rendering of `manifest.yaml`, authenticated production ingress guardrails, and an explicit development-only unauthenticated path |
| BTP Destination | Dry-run-first, redacted OAuth2 Client Credentials upsert using native `connectivity/destination` CLI commands and a mode-0600 temporary file |
| References | Separate `cloud-foundry.md`, `btp-destination.md`, `joule-cli.md`, and `joule-a2a-contract.md` operational references |

## Inputs

The renderer (`scripts/prepare_joule_integration.py`) and destination helper
(`scripts/manage_destination.py`) take:

- the existing agent repository (`--target`);
- an agent slug and single-line description;
- the planned Cloud Foundry public route (`--public-url`);
- the existing server start command (`--start-command`);
- an XSUAA or IAS auth service instance (`--auth-service`), or
  `--development-no-auth` for an explicit non-production path;
- the response mode (`--mode direct` or `--mode interpretability`);
- the Destination alias, subaccount, and OAuth client credentials supplied as
  environment variables (`JOULE_AGENT_CLIENT_ID`, `JOULE_AGENT_CLIENT_SECRET`,
  `JOULE_AGENT_TOKEN_URL`) rather than command arguments.

## Outputs

The renderer adds only the integration surface, never application code:

- `.cfignore`;
- `manifest.yaml` with route, command, and service bindings but no credentials;
- `da.sapdas.yaml`;
- `joule/a2a/capability.sapdas.yaml` and its capability context (schema
  `3.28.0`, namespace `joule.ext`);
- one remote-agent function and one scenario.

Alongside the rendered files, the skill produces inspected, ready-to-run
commands: the Cloud Foundry `cf` sequence, the redacted BTP Destination upsert
preview (and `--apply`), and the `joule lint`/`compile`/`deploy` lifecycle. The
user runs deployment steps manually.

## Related files

- `plugins/sap-ai4u-skills/skills/sap-agent-joule-cf-bootstrap/SKILL.md` — the
  eight-step integration workflow.
- `scripts/prepare_joule_integration.py` — renders the Joule and CF assets.
- `scripts/manage_destination.py` — dry-run-first BTP Destination upsert.
- `references/joule-a2a-contract.md`, `references/cloud-foundry.md`,
  `references/btp-destination.md`, `references/joule-cli.md`.
- `templates/` — `.cfignore`, `manifest.yaml`, `da.sapdas.yaml`, and the
  `joule/a2a/` capability, function, and scenario templates.

## Tests

Run the skill's unit and packaging tests from the skill directory:

```bash
python3 -m unittest discover -s plugins/sap-ai4u-skills/skills/sap-agent-joule-cf-bootstrap/tests
```

- `tests/test_skill_contract.py` keeps the skill focused: it confirms agent
  creation is delegated, both response modes remain, the legacy bootstrap files
  are gone, the four operational references exist, and the release metadata is
  bumped.
- `tests/test_prepare_joule_integration.py` checks the rendered assets and their
  credential-free, schema-correct content.
- `tests/test_manage_destination.py` checks the destination helper's dry-run,
  redaction, and native-CLI behavior.
