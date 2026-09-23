---
name: joule-cli
description: Use for Joule Studio CLI installation checks, login status, capability linting and compilation, assistant inspection, testing, or deployment preparation. For connecting an existing A2A agent to Joule, combine with sap-agent-joule-cf-bootstrap; the user runs deployment commands.
---

# Joule Studio CLI

Guide the user through the installed Joule Studio CLI lifecycle. Separate local design-time validation from tenant deployment and runtime verification. Prepare deployment commands, but leave `joule deploy` and `joule update` for the user to run.

## Confirm the CLI and tenant

Use the current `@sap/joule-studio-cli` package, not the older `@sap/joule-cli` name. Check the installed Node.js and CLI versions, login status, and command-specific help:

```bash
node --version
joule --version
joule status
joule <command> --help
```

The documented Node.js range is 20.12 through 24. Logged-out `joule --help` may show only a subset of commands; do not infer that deployment or test commands are absent until the user's approved login is active. The user completes interactive authentication in their own terminal. Never put a client secret, password, or SSO passcode in command arguments or project files, and do not recommend insecure credential storage.

## Validate locally

Inspect `da.sapdas.yaml`, its capability folders, and the existing naming conventions. For a code-based remote A2A capability, use capability schema `3.28.0`, assistant descriptor schema `1.4.0`, and namespace `joule.ext`; keep function and scenario names within the current tenant limit. Confirm these values against the tenant and installed CLI when using a different Joule capability type.

From the directory containing the assistant descriptor:

```bash
joule lint
joule compile
```

Fix reported errors before proposing deployment. Lint checks source structure; compile builds artifacts. Neither proves tenant authorization, CF ingress, Destination authentication, A2A responses, or Joule routing.

For a remote agent, use [sap-agent-joule-cf-bootstrap](../sap-agent-joule-cf-bootstrap/SKILL.md) to validate the A2A 0.3 `message/send` contract, `contextId`/`taskId` continuity, production bearer enforcement, and Destination alias. The Destination URL is the deployed API base route, not a discovery or JSON-RPC path.

## Manage the deployed assistant

Inspect `joule deploy --help`, `joule update --help`, `joule list --help`, `joule get --help`, and `joule launch --help` on the logged-in version before constructing commands. The user submits deployment or update commands manually. Afterward, use read-only list/get/status checks and a real Joule conversation to verify the deployed behavior. Use `joule test` only when the installed CLI exposes it and the tenant/test user is authorized; a local compile cannot stand in for that test.

Treat removal and deletion as separate operations with their own target and impact review. Do not run them as part of troubleshooting or validation. When an operation fails, classify it at the correct layer: login, tenant role, lint, compile, deployment, Destination, remote ingress, or A2A response. Do not change roles or weaken authentication to make a connectivity check pass.

## Report

State the CLI version and tenant context when available, which checks were local, which used a live tenant, and which user-run deployment or conversation checks remain.

## Sources

- [SAP Joule Development Guide](https://help.sap.com/doc/7e70b0a09517400f95c7cba8671e60ca/CLOUD/en-US/0cc0df731bc04171a339c682bcf7c39b.pdf)
- [SAP code-based agent contract](https://help.sap.com/docs/joule/joule-development-guide-ba88d1ec6a1b442098863d577c19b0c0/code-based-agents-bring-your-own-agent)
