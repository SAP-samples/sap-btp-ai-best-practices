---
name: btp-cli
description: Use for SAP BTP CLI account, subaccount, entitlement, service, role collection, environment, or Destination operations. Confirm the live command contract and exact account scope before changing resources; use sap-agent-joule-cf-bootstrap for a Joule A2A Destination.
---

# SAP BTP CLI

Use the installed `btp` CLI to discover the target account and prepare precise, scoped BTP operations. Its command definitions can vary with the CLI and backend, so verify the available action and parameters instead of treating a copied example as authoritative.

## Discover before changing state

```bash
btp --info
btp target
btp help <action> <group/object>
btp --format json list accounts/subaccount
```

The usual shape is `btp [options] <action> <group/object> [parameters]`. Confirm the global account, subaccount ID, region, and current target. Prefer an explicit `--subaccount <ID>` on commands that support it when a script or review spans multiple accounts. Use JSON for machine-readable discovery, but inspect its schema before writing a parser.

Have the user complete SSO or the organization's approved interactive login when required. Never ask for a password, token, binding JSON, or service key in chat. Do not print credential-bearing CLI output.

## Route the request

| Request | Read-only discovery | Change preparation |
| --- | --- | --- |
| Account or region | `btp list accounts/subaccount`, `btp get accounts/subaccount <ID>`, `btp list accounts/available-region` | Confirm region, subdomain uniqueness, owner, and scope before create/update. |
| Entitlement or service | `btp list accounts/entitlement`, `btp list services/offering`, `btp list services/plan`, `btp list services/instance` | Check live availability, plan, quota, and charges before assign/create/update. |
| Access | `btp list security/role-collection`, `btp get security/role-collection <name>` | Identify the user, IdP origin, role collection, and subaccount before assignment. |
| Environment | `btp list accounts/environment-instance --subaccount <ID>` | Identify the target landscape from live account data; do not guess CF API URLs or Kyma configuration. |
| Destination | `btp help create connectivity/destination` and corresponding `get`, `update`, `delete` help | Match exact name, subaccount, base URL, auth mode, and readback plan. |

Use `btp help` for each proposed write command. Do not automatically install a CLI extension or create a service to make an example work. If the CLI does not expose the required command, report that and use an approved cockpit or API path rather than guessing syntax.

## Live-change boundary

Perform a live account, entitlement, service, role, or Destination change only when that specific action is authorized. Before it, show the exact target and consequences, including charges, access expansion, or deletion where applicable. After it, read back the resource at the same scope. Do not run application deployment commands; the user deploys applications manually.

Keep secrets out of shell history and shared output. In particular, do not use inline `--password`, client-secret, or credential JSON values, and do not display full service binding or key responses. For a Joule A2A Destination, use [sap-agent-joule-cf-bootstrap](../sap-agent-joule-cf-bootstrap/SKILL.md) and its dry-run-first `manage_destination.py` helper. Its production default is `OAuth2ClientCredentials`; its `NoAuthentication` path is explicitly development-only.

## Verify and report

Report the resolved global account and subaccount, the CLI help used, the before/after resource state, and any unverified live prerequisites. A successful create or update response is not enough when an application must still consume the service or Destination.

## Sources

- [SAP BTP CLI command reference](https://help.sap.com/docs/btp/btp-cli-command-reference/btp-cli-command-reference)
- [SAP BTP CLI command syntax](https://help.sap.com/docs/btp/sap-business-technology-platform/command-syntax-of-btp-cli)
