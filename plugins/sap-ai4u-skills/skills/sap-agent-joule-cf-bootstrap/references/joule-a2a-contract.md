# Joule A2A Contract

Use this reference when inspecting the existing agent, authoring the Joule remote-agent function, or debugging conversation continuity.

## Supported Baseline

- Code-based Joule agents use capability schema `3.28.0`.
- The remote agent integration uses A2A v0.3 JSON-RPC `message/send`.
- Text content is the supported baseline for requests and responses.
- A synchronous request should finish within 60 seconds.
- Joule delegates response validation and interpretation to the function definition; malformed remote output is therefore an integration defect, not something Joule will repair automatically.

Do not assume an A2A 1.x endpoint also accepts the Joule v0.3 request. Test the exact endpoint and contract used by the Destination.

## Request and Response Shape

The remote-agent action forwards the active user message. Its optional body carries continuity identifiers:

```yaml
parameters:
  - name: agent_context_id
    optional: true
  - name: agent_task_id
    optional: true

action_groups:
  - actions:
      - type: agent-request
        agent_type: remote
        system_alias: inventory-agent-a2a
        body: >
          <? (agent_context_id == null || agent_context_id.isEmpty()) && (agent_task_id == null || agent_task_id.isEmpty())
             ? null
             : (agent_task_id == null || agent_task_id.isEmpty())
               ? '{ "contextId": "' + agent_context_id + '" }'
               : (agent_context_id == null || agent_context_id.isEmpty())
                 ? '{ "taskId": "' + agent_task_id + '" }'
                 : '{ "contextId": "' + agent_context_id + '", "taskId": "' + agent_task_id + '" }' ?>
        result_variable: _agent_response
```

Validate how the concrete server treats absent, null, and empty identifier values. If its SDK rejects empty IDs, adjust the body expression or normalize them in the existing A2A adapter.

Expected outcomes:

- `input-required`: read the prompt from the status message and retain the returned task ID.
- completed: read text from `artifacts[0].parts[0].text`, retain `contextId`, and clear the completed task ID.
- failed/rejected: return a safe user-facing failure while keeping internal traces and credentials out of the Joule message.

## Capability Context

Joule capability context reads root-level function result keys. Flatten nested A2A fields first:

```yaml
result:
  agent_result: <? _agent_response.body.artifacts != null ? _agent_response.body.artifacts[0].parts[0].text : null ?>
  agent_context_id: <? _agent_response.body.contextId ?>
  agent_task_id: <? _agent_response.body.status.state == "input-required" ? _agent_response.body.id : null ?>
```

Then pass and save them in the scenario:

```yaml
target:
  type: function
  name: inventory_agent_a2a
  parameters:
    - name: agent_context_id
      value: $capability_context.agent_context_id
    - name: agent_task_id
      value: $capability_context.agent_task_id

capability_context:
  - name: agent_context_id
    value: $target_result.agent_context_id
  - name: agent_task_id
    value: $target_result.agent_task_id
```

Rules:

- A target parameter must also be an optional function parameter.
- Do not bind the same parameter through both slots and `target.parameters`.
- Do not put scripting expressions in `target.parameters`.
- Keep context compact. Do not store the entire A2A response.
- Exercise null-safety in the function and in the remote server.

## Direct and Interpretability Modes

Direct mode emits the remote completed text as a message and omits `response_context`. Use it when the remote agent owns response wording.

Interpretability mode exposes only a compact root-level `agent_result` and configures one `response_context` entry:

```yaml
response_context:
  - description: Remote agent response
    value: $target_result.agent_result
```

In interpretability mode, do not also emit the completed artifact as a direct message. That creates duplicate responses. Keep the `input-required` message direct because the user must see the remote agent's clarification request.

Joule permits one `response_context` section per scenario, up to 10 fields and 10,000 bytes total. Use root-level result keys and short field descriptions.

## Longer Work

If the agent cannot reliably answer within 60 seconds, decide whether supported asynchronous push notifications fit the product flow. Secure inbound callbacks require the appropriate IAS application-to-application trust. Do not solve this by exposing an unauthenticated callback or by assuming Cloud Foundry timeout changes remove Joule's limit.

## Validation Checklist

- Send an initial text message without continuity IDs.
- Send a second turn with the returned `contextId`.
- Exercise `input-required`, then continue with both `contextId` and `taskId`.
- Confirm a completed task does not accidentally resume on the next unrelated request.
- Confirm direct mode emits one answer.
- Confirm interpretability mode emits one Joule-generated answer.
- Confirm failure output contains no secrets, internal URLs, or stack traces.

## Sources

- [SAP Help: Code-based agents](https://help.sap.com/docs/joule/service-guide/code-based-agents)
