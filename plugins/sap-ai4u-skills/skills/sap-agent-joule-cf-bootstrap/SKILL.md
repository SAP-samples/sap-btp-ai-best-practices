---
name: sap-agent-joule-cf-bootstrap
description: Prepare, connect, validate, deploy, and manage a Joule code-based agent on SAP BTP and Cloud Foundry around an existing LangGraph A2A agent. Use for Joule design-time artifacts, A2A context handling, CF readiness, authenticated BTP Destinations, Joule Studio CLI lifecycle operations, or troubleshooting the linkage between Joule and a deployed agent. Delegate creation of the LangGraph agent itself to a dedicated agent-creation skill.
---

# Connect an Existing A2A Agent to Joule on Cloud Foundry

Build and operate the Joule-facing integration around an existing LangGraph A2A agent. Preserve the agent's business logic and runtime choices. Do not create another agent implementation, model layer, dependency set, or full project scaffold.

## Scope Boundary

Start by deciding whether an A2A agent already exists.

- If it exists, inspect it and continue with this workflow.
- If it does not exist, invoke `create-langgraph-react-agent` to create the agent. After that skill finishes, return here and add only the Joule, Cloud Foundry, and Destination integration.
- If a different specialist skill already owns the target agent, invoke that skill instead of duplicating its behavior here.

Never recreate the removed `bootstrap_agent.py` flow. This skill owns integration and lifecycle management, not LangGraph agent creation.

## Non-Negotiable Rules

- Never deploy an application or Joule assistant on the user's behalf. Prepare, validate, and show the commands; the user submits deployment commands manually.
- Require authenticated production ingress and an `OAuth2ClientCredentials` BTP Destination.
- Allow `NoAuthentication` only when the user explicitly selects a development-only path.
- Treat an XSUAA or IAS service binding as credential delivery, not proof that the app enforces authentication. Verify middleware and confirm an unauthenticated request receives `401` or `403` before calling the route production-ready.
- Keep credentials out of `manifest.yaml`, shell arguments, generated Joule YAML, source control, and shared output. Read credentials from service bindings or environment variables at runtime.
- Use the deployed API base route as the Destination URL. Do not append `/a2a`, `/.well-known/agent.json`, or another endpoint path.
- Use namespace `joule.ext` for custom capabilities and schema version `3.28.0` for the code-based capability.
- Keep Joule function and scenario names at or below 30 characters.
- Keep A2A work synchronous and normally below Joule's 60-second response limit. For longer work, redesign around supported asynchronous behavior rather than increasing an application timeout blindly.

## 1. Inspect Before Editing

Inspect the complete integration surface, not only the LangGraph graph:

```bash
rg -n "StateGraph|langgraph|A2AStarletteApplication|AgentCard|message/send|contextId|taskId|Authorization|JWT|xsuaa|identity" .
rg --files | rg "(^|/)(manifest\.ya?ml|mta\.ya?ml|Procfile|config/agent\.ya?ml|da\.sapdas\.yaml|joule/|xs-security\.json|package\.json|pyproject\.toml|requirements.*\.txt)$"
```

Determine:

- the current A2A protocol versions and discovery endpoints;
- the existing start command and whether it binds to `0.0.0.0` and `${PORT}`;
- how the agent card derives its public URL;
- whether `contextId` and `taskId` survive multi-turn and input-required flows;
- whether task/checkpoint storage is process-local, file-based, or durable;
- how ingress authentication is enforced in application code;
- existing Cloud Foundry manifests and service bindings;
- existing Joule artifacts and naming conventions.

Do not rerender over existing integration files. Adapt them deliberately after comparing them with the templates.

## 2. Confirm the A2A Contract

Read [references/joule-a2a-contract.md](references/joule-a2a-contract.md) before changing the remote-agent function.

The Joule path uses A2A v0.3 JSON-RPC `message/send` with text messages. Validate the existing agent locally before creating platform artifacts:

- discovery returns an agent card with the intended public base URL;
- a text request produces either an `input-required` status message or a completed text artifact;
- the response exposes `contextId` and a task identifier;
- the same identifiers continue the intended conversation;
- error responses do not disclose secrets or internal stack traces.

If the existing agent only exposes A2A 1.x, add or enable the Joule-compatible 0.3 adapter through the agent-owning skill. Do not build a second runtime in this skill.

## 3. Choose the Response Mode

Ask for one mode if it cannot be inferred safely:

- `direct`: emit the remote completed artifact as the Joule message. Do not configure `response_context`, which avoids a second generated response.
- `interpretability`: return `agent_result` as a compact root-level function result and let Joule generate the final response through `response_context`. Keep the direct message only for `input-required`, where the remote agent needs another user value.

Both modes persist root-level `agent_context_id` and `agent_task_id` through capability context. Continue an input-required task with its task ID; clear a completed task ID while retaining the conversation context.

## 4. Render Focused Integration Assets

Use the renderer when the target does not already contain the output paths:

```bash
python3 "<skill-path>/scripts/prepare_joule_integration.py" \
  --target "<existing-agent-repo>" \
  --agent-slug "<agent-slug>" \
  --agent-description "<single-line-description>" \
  --public-url "https://<planned-cf-route>" \
  --start-command "<existing-server-command>" \
  --auth-service "<xsuaa-or-ias-service-instance>" \
  --service "<other-existing-service-instance>" \
  --mode direct
```

Use `--mode interpretability` for the second response mode. Use `--development-no-auth` instead of `--auth-service` only for an explicit non-production environment.

The renderer adds only:

- `.cfignore`;
- `manifest.yaml` with route, command, and service bindings but no credentials;
- `da.sapdas.yaml`;
- `joule/a2a/capability.sapdas.yaml` and capability context;
- one remote-agent function and one scenario.

It must not add application source, dependencies, local persistence, model configuration, a deploy script, or a README. If files collide, inspect and retrofit them manually rather than forcing an overwrite.

## 5. Make the Existing Runtime Cloud Foundry Ready

Read [references/cloud-foundry.md](references/cloud-foundry.md), then adjust the existing application as needed.

At minimum:

- bind the server to `0.0.0.0` and the Cloud Foundry `PORT`;
- set the agent card URL from the public route, not the listening address;
- consume platform credentials from `VCAP_SERVICES` or the supported service-binding library;
- enforce bearer-token validation on the remote-agent endpoint in production;
- avoid SQLite, in-memory task stores, or local disk for production conversation state when restarts or multiple instances must be supported;
- use HANA or another approved durable shared store through its owning skill when persistence is required;
- keep `.env`, local data, tests, build output, and Joule compiled output out of the CF upload.

Compile and unit-test locally. Then give the user the inspected `cf` commands and stop before `cf push`.

## 6. Create or Update the BTP Destination

Read [references/btp-destination.md](references/btp-destination.md). Ensure the Destination name exactly matches both the capability `system_alias` and its `destination` value.

For production, export the OAuth client values locally without echoing them:

```bash
export JOULE_AGENT_CLIENT_ID="<client-id>"
export JOULE_AGENT_CLIENT_SECRET="<client-secret>"
export JOULE_AGENT_TOKEN_URL="https://<authorization-server>/oauth/token"
```

Preview the upsert first:

```bash
python3 "<skill-path>/scripts/manage_destination.py" \
  --name "<destination-alias>" \
  --url "https://<deployed-cf-route>" \
  --subaccount "<subaccount-id>"
```

After the user reviews the preview, they can add `--apply`. The helper checks that the installed `btp` CLI exposes native `connectivity/destination` commands, lists current destinations to choose create or update, uses a mode-0600 temporary configuration file, performs readback, and removes the file.

Use `--development-no-auth` only when the environment and route are explicitly disposable/non-production. Do not downgrade an authenticated production design to make a connectivity test pass.

## 7. Validate and Manage the Joule Assistant

Read [references/joule-cli.md](references/joule-cli.md). Use the current `@sap/joule-studio-cli` package and keep compilation separate from deployment.

From the agent project root:

```bash
joule status
joule lint
joule compile
```

Fix lint and compile errors before proposing deployment. Do not infer that a successful compile proves that the CF route, authentication, Destination, or tenant authorization works.

After the user has manually deployed the CF app and applied the Destination, provide the appropriate `joule deploy`, `joule update`, `joule launch`, `joule get`, `joule list`, or `joule delete` command. Inspect `joule <command> --help` on the installed version before adding flags.

## 8. End-to-End Verification

Verify each boundary separately:

1. Local A2A contract tests pass.
2. Joule YAML uses schema `3.28.0`, namespace `joule.ext`, matching aliases, and names no longer than 30 characters.
3. Direct mode has one completed direct message and no `response_context`; interpretability mode has `response_context` and no duplicate completed message.
4. The production route rejects a request without a bearer token with `401` or `403`.
5. An authorized `message/send` completes within 60 seconds and returns text in the supported response location.
6. The BTP Destination readback shows the base route and `OAuth2ClientCredentials` without printing its secret.
7. `joule lint` and `joule compile` pass.
8. After the user deploys, a Joule conversation preserves context across turns and handles `input-required` without starting an unrelated task.

Report which checks were local, which used a live tenant, and which remain for the user. Never equate generated files or local tests with successful tenant deployment.

## References

- [Joule A2A contract and response modes](references/joule-a2a-contract.md)
- [Cloud Foundry readiness, security, and operations](references/cloud-foundry.md)
- [BTP Destination lifecycle](references/btp-destination.md)
- [Joule Studio CLI lifecycle](references/joule-cli.md)
