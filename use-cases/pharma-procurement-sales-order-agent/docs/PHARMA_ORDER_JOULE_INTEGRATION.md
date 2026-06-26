# Joule Integration Notes

This project exposes a Joule-compatible streamed-message endpoint for the Pharma Procurement Sales Order Agent.

## Endpoint

```text
POST /api/joule/pharma-order/stream
```

The endpoint returns `text/event-stream` chunks in the structure expected by the Joule function YAML:

```json
{"message":{"parts":[{"text":"..."}]}}
```

## Relevant files

| File | Responsibility |
| --- | --- |
| `joule/capability.sapdas.yaml` | Capability metadata and destination alias. |
| `joule/scenarios/sc_pharma_order.yaml` | Scenario trigger and function reference. |
| `joule/functions/fn_pharma_order.yaml` | Streamed-message action configuration. |
| `api/app/routers/joule.py` | FastAPI adapter for Joule. |
| `api/app/agents/pharma_order/graph.py` | Agent execution and tool orchestration. |

## Duplicate stream handling

Joule may retry streamed requests. The backend keeps a short-lived idempotency cache based on correlation ID and normalized question text. Duplicate requests inside the TTL are suppressed to avoid repeated final answers.

## Destination

Create a BTP destination named `pharma-order-agent-api` pointing to the deployed backend route.
