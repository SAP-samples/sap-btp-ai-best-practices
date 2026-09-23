# Cloud Foundry Readiness and Operations

Use this reference when adapting the existing A2A runtime, reviewing `manifest.yaml`, preparing manual deployment commands, or diagnosing the deployed route.

## CLI Preflight

Confirm the installed CLI and target explicitly:

```bash
cf version
cf api
cf target
cf orgs
cf spaces
```

Use `cf login` and `cf target -o "<org>" -s "<space>"` interactively as needed. Never put a password in a committed script or shared shell history.

Inspect required services and their plans before referencing them in a manifest:

```bash
cf marketplace
cf services
cf service "<service-instance>"
```

Do not create, update, or delete a service instance without confirming the target org/space and the user's requested scope.

## Existing Runtime Requirements

The application must:

- listen on `0.0.0.0` and the `PORT` supplied by Cloud Foundry;
- advertise the public HTTPS base route in its agent card;
- expose the Joule-compatible A2A v0.3 JSON-RPC endpoint expected by the Destination;
- complete synchronous work within 60 seconds;
- handle shutdown and restart without corrupting conversation state;
- avoid relying on process-local memory or ephemeral local disk for production continuity.

If the existing agent uses the `create-langgraph-react-agent` configuration, update its host, port, and public URL for CF instead of replacing its runtime.

## Bindings and Secrets

Declare existing service instances under `services:` in `manifest.yaml`. Consume their credentials from `VCAP_SERVICES` or an official service-binding library at runtime.

Do not place passwords, OAuth secrets, API keys, or complete service-key JSON in the manifest `env` section or in `cf push --var` arguments. Cloud Foundry CLI commands, environment inspection, logs, and build output can expose them.

Useful read-only checks after the user deploys include:

```bash
cf app "<app-name>"
cf env "<app-name>"
cf logs "<app-name>" --recent
cf events "<app-name>"
```

Never paste the complete `VCAP_SERVICES` value into a report. Summarize binding names and redact credentials.

## Production Authentication

Production requires both sides of the OAuth2 Client Credentials design:

1. The A2A application validates bearer tokens on its protected endpoint.
2. The BTP Destination retrieves and forwards a token using the configured OAuth client.

Binding XSUAA or IAS only makes credentials available. The application's authentication middleware, audience/client checks, scopes, and route policy still have to be implemented and tested.

Before treating the route as production-ready, verify:

```bash
curl -i "https://<route>/.well-known/agent.json"
```

The protected production surface should return `401` or `403` without valid credentials. Then obtain a token through the approved local process and validate the discovery and `message/send` requests with an Authorization bearer header. Do not print or persist the token.

Decide deliberately whether discovery is public while invocation is protected. Do not leave invocation open merely because a public agent card is convenient.

## Persistence and Scaling

Cloud Foundry restarts containers and may run multiple instances. In-memory task stores, local SQLite files, and local checkpoint directories are not shared or durable.

- Use one instance only as a documented development constraint.
- For production continuity, route persistence work to the appropriate HANA or LangGraph persistence skill and use a shared backing service.
- Verify that `contextId` maps deterministically to the persisted LangGraph thread.
- Verify concurrent turns for the same context are serialized or rejected safely.

## Manual Deployment and Management

The assistant prepares these commands but does not run state-changing operations:

```bash
cf push "<app-name>" -f manifest.yaml
cf restage "<app-name>"
cf restart "<app-name>"
cf scale "<app-name>" -i 1
```

Use `cf push` only after local tests, manifest review, binding verification, and an explicit user action. `cf restage`, `cf scale`, route mapping, service rebinding, and deletion are also state-changing operations requiring the user's intended scope.

For rollback, use the organization's approved artifact/version process. Do not assume a prior droplet or build is still available.

## Post-Deployment Checks

- `cf app` reports the intended route and healthy instances.
- Recent logs contain no credential values or unhandled exceptions.
- The agent card advertises the deployed base route, not localhost or `0.0.0.0`.
- An unauthenticated production invocation returns `401` or `403`.
- An authenticated `message/send` returns the expected A2A response within 60 seconds.
- Restarting the app does not lose required production conversation state.

## Sources

- [Cloud Foundry CLI reference](https://cli.cloudfoundry.org/en-US/v8/)
- [Cloud Foundry manifest attributes](https://docs.cloudfoundry.org/devguide/deploy-apps/manifest-attributes.html)
- [Cloud Foundry environment variables](https://docs.cloudfoundry.org/devguide/deploy-apps/environment-variable.html)
