# ECN model selection and timeout fallback

## Purpose

New ECN generation, RDF discovery, and interactive assistant calls use
`gpt-5.4` by default. If the provider raises a timeout for an individual
model request, the same request is retried once with `gpt-5.4`. Because the
default primary model is now the fallback model itself, a timeout on the
default deployment is not retried; the fallback only applies when `LLM_MODEL`
selects another deployment. Other provider errors never trigger this fallback.

## Inputs and outputs

- `LLM_MODEL` selects the primary SAP AI Core text-generation deployment. Its
  default is `gpt-5.4`.
- The existing per-request timeout bounds each native generation request. Chat
  and discovery use the same provider timeout classification for their
  LangChain Responses requests.
- `gpt-5.4` is the fixed timeout fallback. A deployment that already uses
  `gpt-5.4` does not call itself again as a fallback.
- A successful retry returns the same structured response or agent step as the
  primary request would. No tool operation is replayed; only the timed-out model
  invocation is repeated with its original input and tool bindings.
- Usage telemetry is attributed to the model that returned the response. The
  stored job keeps its pinned primary model for reproducibility.

Fallback is not used for authentication, authorization, missing-model,
rate-limit, server, schema, or validation errors. The provider preflight checks
the configured primary deployment; it does not require the fallback deployment
to be available at startup.

## Implementation

```python
if failure.error_code == "PROVIDER_TIMEOUT" and primary_model != fallback_model:
    response = await call_model(fallback_model)
```

The native generation adapter sends the existing prompt, schema, timeout, and
output limits to the fallback. The shared LangChain helper applies the same rule
to chat and RDF discovery. The analysis stage timeout allows one primary
request and one fallback request; chat allows a bounded 300-second turn and
discovery allows twice the job's pinned request timeout.

## Related files

- `api/app/ecn/settings.py` — primary default and fixed fallback deployment.
- `api/app/ecn/llm/provider.py` — native Responses call and category-aware retry.
- `api/app/ecn/llm/timeout_fallback.py` — timeout-only LangChain fallback.
- `api/app/ecn/assistant/runtime.py` — interactive chat fallback and deadline.
- `api/app/ecn/assistant/discovery.py` — RDF discovery fallback and deadline.
- `api/app/ecn/services/generation.py` — per-stage timeout and actual-model telemetry.
- `api/app/ecn/llm/preflight.py` — primary model deployment check.
- `manifest.yaml`, `api/.env`, and `api/.env.example` — runtime model selection.

## How to test

Use fake provider clients and tool-bound runnables; do not call the live model.
Make the primary raise a timeout and the fallback return a valid response, then
check that the call sequence is `<non-default primary>`, `gpt-5.4`, both requests receive
the same input, and usage is attributed to the fallback. Also check that
non-timeout failures do not invoke the fallback and a fallback timeout remains
a terminal provider timeout. The provider and discovery regression locations are
`tests/unit/test_impact_provider.py` and `tests/unit/test_impact_discovery.py`;
settings defaults are covered by `tests/unit/test_ecn_settings.py`.
