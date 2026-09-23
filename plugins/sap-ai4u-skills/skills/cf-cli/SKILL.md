---
name: cf-cli
description: Use for SAP BTP Cloud Foundry CLI work involving orgs and spaces, application manifests, routes, service bindings, logs, scaling, or deployment preparation. For Joule A2A integration, combine with sap-agent-joule-cf-bootstrap; application deployment remains user-run.
---

# Cloud Foundry CLI

Help the user inspect and operate the intended Cloud Foundry org and space with the installed `cf` CLI. Prepare application deployments, validate them locally, and give the user the final deployment command to run. Do not run `cf push` or `cf deploy` for the user.

## Establish the target

Before giving a command that can change CF state, inspect the current CLI version, API, org, and space:

```bash
cf version
cf api
cf target
cf help <command>
```

Use the installed CLI's help for flags and any installed plugin. Confirm the target with the user when the intended org or space is unclear; do not infer it from an app name or a sample manifest. Use SSO or the organization's approved login flow. Never put passwords or one-time passcodes in shell arguments or shared output.

## Choose the operation

| Need | Inspect first | Prepare or perform when authorized |
| --- | --- | --- |
| App health or failure | `cf apps`, `cf app <app>`, `cf logs <app> --recent`, `cf events <app>` | Diagnose staging, startup, health check, route, and service failures separately. |
| New or changed app | Existing `manifest.yaml`/`manifest.yml`, `.cfignore`, route, start command, buildpack, services | Validate locally; give the user an inspected `cf push -f <manifest>` command. |
| MTA/CAP app | `mta.yaml`, `mbt --version`, `cf plugins`, `cf help deploy` | Build the `.mtar` locally if requested; give the user the exact `cf deploy <archive>` command. |
| Service or binding | `cf marketplace`, `cf services`, `cf service <service>`, `cf app <app>` | Resolve offering, plan, cost, app, and space before create/bind/update; verify the resulting binding. |
| Route or scale | `cf routes`, `cf help route`, `cf app <app>` | Check mappings, health, quota, and the intended public exposure before a change. |

Keep the existing project's manifest and deployment shape. For CF readiness, verify that a web process binds to `0.0.0.0` and the platform `PORT`, its public URL is configured separately from its listening address, and its health check matches an implemented route. Inspect `.cfignore` so credentials, local data, tests, and unneeded build output are not uploaded, while required application artifacts remain included.

`cf push` handles a CF app manifest; `cf deploy` comes from the MultiApps plugin and handles an MTA archive. Do not substitute one for the other. A successful local build or generated manifest does not establish that the CF route is live.

## Credentials and ingress

- Prefer service bindings and runtime credential delivery. Do not place secrets in a manifest, `cf push --var`, `cf set-env` arguments, or a committed `.env` file.
- `cf env`, `cf service-key`, and `cf oauth-token` can expose credentials. Inspect them only when needed, keep raw output local, and report redacted findings.
- Binding XSUAA or IAS supplies credentials; it does not prove the application validates bearer tokens. For a production API, inspect middleware and verify that an unauthenticated request is rejected with `401` or `403`.
- Do not use `--skip-ssl-validation` to work around a failed connection. Diagnose trust and endpoint configuration instead.

For an existing A2A agent being connected to Joule, continue with [sap-agent-joule-cf-bootstrap](../sap-agent-joule-cf-bootstrap/SKILL.md). Its Destination, A2A contract, and user-run deployment boundary take precedence over generic CF examples.

## Verify and report

After the user runs deployment, check `cf app <app>`, recent logs, the mapped route, the health endpoint, and one real authenticated request when applicable. State which checks ran locally and which require the user's CF deployment. For a service or route change, read back the exact resource in the same org and space.

## Sources

- [Cloud Foundry CLI v8 reference](https://cli.cloudfoundry.org/en-US/v8/)
- [Cloud Foundry manifest attributes](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html)
