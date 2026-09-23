---
name: token-logger
description: Standardize SAP Cloud Logging observability for SAP Gen AI Hub LLM token usage in Cloud Foundry Python/FastAPI applications, including uncached input, cache-read input, cache-write input, output, and provider totals. Use when adding or auditing token-consumption logging for OpenAI chat/responses, LangChain/LangGraph, Gemini, Bedrock, or orchestration calls, especially with access-to-generative-ai-models and genai-token-caching.
---

# Token Logger

Use this skill with `$access-to-generative-ai-models` when a SAP Gen AI Hub app needs token consumption tracked in SAP Cloud Logging. The app must emit one compact JSON stdout event for every LLM call, and must run in a Cloud Foundry environment with the `Cloud Logging` service bound so token consumption is tracked automatically.

## Workflow

1. Inspect `manifest.yaml`, deployment scripts, and every LLM call site before editing.
2. Ensure API and UI routes use a single, consistent Cloud Foundry landscape domain.
3. Ensure the deploy script binds the API app to the `Cloud Logging` service and restarts it after binding. Do not deploy unless the user explicitly asks.
4. Add a shared `llm_usage_logging.py` helper or adapt the local equivalent.
5. Wrap every LLM call with usage extraction and `emit_llm_usage_event(...)` on both success and error, including cache read and cache write fields when present.
6. Add or update tests that capture stdout JSON and fake provider responses.
7. Document the feature in `docs/` when creating it in an application.

## Cloud Foundry Route Standard

Use one Cloud Foundry landscape domain consistently for both API and UI routes (the example below uses a sample region). In `manifest.yaml`, keep the API route, UI route, API base URL, allowed origin, and Vite API URL aligned:

```yaml
applications:
  - name: example-api
    path: api
    buildpack: python_buildpack
    command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    routes:
      - route: example-api.cfapps.eu10-005.hana.ondemand.com
    env:
      ALLOWED_ORIGIN: https://example-ui.cfapps.eu10-005.hana.ondemand.com
      API_BASE_URL: https://example-api.cfapps.eu10-005.hana.ondemand.com
      APP_ENV: production
      API_KEY: ((api_key))
      AICORE_AUTH_URL: ((aicore_auth_url))
      AICORE_CLIENT_ID: ((aicore_client_id))
      AICORE_CLIENT_SECRET: ((aicore_client_secret))
      AICORE_BASE_URL: ((aicore_base_url))
      AICORE_RESOURCE_GROUP: ((aicore_resource_group))
  - name: example-ui
    path: ui
    buildpack: nodejs_buildpack
    command: npm run build && npx vite preview --host 0.0.0.0 --port $PORT
    routes:
      - route: example-ui.cfapps.eu10-005.hana.ondemand.com
    env:
      VITE_API_BASE_URL: https://example-api.cfapps.eu10-005.hana.ondemand.com
      VITE_APP_HOST: example-ui.cfapps.eu10-005.hana.ondemand.com
      VITE_API_KEY: ((api_key))
```

## Cloud Logging Binding

Put the Cloud Logging bind after `cf push` and restart the API app so the service binding appears in `VCAP_SERVICES`. Include the code in the deploy script; do not only mention a manual step.

```bash
APP_NAME=$(grep -m 1 '\- name:' manifest.yaml | awk '{print $3}')

cf push \
  --var api_key="$API_KEY" \
  --var aicore_auth_url="$AICORE_AUTH_URL" \
  --var aicore_client_id="$AICORE_CLIENT_ID" \
  --var aicore_client_secret="$AICORE_CLIENT_SECRET" \
  --var aicore_base_url="$AICORE_BASE_URL" \
  --var aicore_resource_group="$AICORE_RESOURCE_GROUP"

echo "Binding $APP_NAME to Cloud Logging..."
cf bind-service "$APP_NAME" "Cloud Logging"

echo "Restarting $APP_NAME..."
cf restart "$APP_NAME"
```

If the project has separate API and UI apps, bind only the API app unless the UI also emits LLM usage events.

## Stdout Event Schema

Emit this schema to stdout with `print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)`. Keep field names stable because SAP Cloud Logging dashboards depend on them.

```python
event = {
    "schema_version": "btp.llm_usage.v1",
    "event_type": "llm_usage",
    "event_time": _now_utc(),
    "app_name": vcap.get("application_name") or vcap.get("name"),
    "space_name": vcap.get("space_name"),
    "org_name": vcap.get("organization_name") or vcap.get("org_name"),
    "route": route,
    "method": method,
    "user_hash": _hash_user(user_id),
    "actor_type": actor_type_for_user(user_id, actor_type),
    "client_host": client_host,
    "provider": provider,
    "model": model,
    "llm_endpoint": llm_endpoint,
    "input_tokens": input_tokens,
    "cached_input_tokens": cached_input_tokens,
    "cache_read_input_tokens": cache_read_input_tokens,
    "cache_write_input_tokens": cache_write_input_tokens,
    "input_total_tokens": input_total_tokens,
    "output_tokens": output_tokens,
    "total_tokens": total_tokens,
    "outcome": outcome,
    "latency_ms": int(latency_ms) if latency_ms is not None else None,
    "correlation_id": correlation_id or str(uuid.uuid4()),
}
```

Use these defaults unless project conventions already define compatible values:

- `provider="sap-ai-core"`
- `llm_endpoint="chat.completions"` for OpenAI chat completions and SAP Gen AI Hub LangChain OpenAI proxy calls
- `llm_endpoint="responses"` for OpenAI Responses API
- `llm_endpoint="generateContent"` for Gemini
- `outcome="success"` after a response is returned
- `outcome="error"` when the attempted model call raises, using known token values or `0`

## Helper Pattern

Create a helper near the API code, usually `api/app/observability/llm_usage_logging.py`. Keep functions documented with docstrings.

```python
from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

USER_ID_HEADERS = ("x-client-user-id", "x-user-id", "x-forwarded-user", "x-authenticated-user")
JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")


@dataclass(frozen=True)
class TokenUsage:
    """Normalized LLM token usage values.

    `input_tokens` is uncached input. `cached_input_tokens` is kept as the
    legacy cache-read field and mirrors `cache_read_input_tokens`.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    input_total_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _now_utc() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _vcap_application() -> dict[str, Any]:
    """Parse Cloud Foundry app metadata from VCAP_APPLICATION."""
    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    """Return a salted, non-reversible hash for the user identity."""
    if not user_id:
        return None
    return hashlib.sha256(f"{os.getenv('LOG_USER_HASH_SALT', '')}:{user_id}".encode("utf-8")).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode JWT claims for logging identity only, never authorization."""
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _read_value(source: Any, key: str) -> Any:
    """Read a value from a mapping or object attribute."""
    return source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize OpenAI, LangChain, Gemini, Bedrock, and cached token fields."""
    usage = _read_value(source, "usage") or _read_value(source, "usage_metadata") or source
    response_metadata = _read_value(source, "response_metadata")
    if response_metadata and not _read_value(source, "usage_metadata"):
        usage = _read_value(response_metadata, "token_usage") or _read_value(response_metadata, "usage") or usage

    def first_from(container: Any, *keys: str) -> int:
        for key in keys:
            value = _read_value(container, key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    def first(*keys: str) -> int:
        return first_from(usage, *keys)

    input_details = next(
        (
            _read_value(usage, key)
            for key in ("input_tokens_details", "input_token_details", "prompt_tokens_details", "prompt_token_details")
            if _read_value(usage, key)
        ),
        {},
    )
    provider_input_tokens = first("input_tokens", "prompt_tokens", "prompt_token_count", "promptTokenCount", "inputTokens")
    output_tokens = first("output_tokens", "completion_tokens", "completion_token_count", "candidates_token_count", "candidatesTokenCount", "outputTokens")
    cache_read_input_tokens = first("cached_input_tokens", "cache_read_input_tokens", "cacheReadInputTokens", "cached_content_token_count", "cachedContentTokenCount")
    if not cache_read_input_tokens:
        cache_read_input_tokens = first_from(input_details, "cached_tokens", "cache_read", "cacheRead", "cache_read_input_tokens")
    cache_write_input_tokens = first("cache_write_input_tokens", "cache_creation_input_tokens", "cacheWriteInputTokens")
    if not cache_write_input_tokens:
        cache_write_input_tokens = first_from(input_details, "cache_write", "cache_creation", "cache_write_input_tokens", "cache_creation_input_tokens")
    if not cache_write_input_tokens:
        cache_details = _read_value(usage, "cacheDetails") or _read_value(usage, "cache_details")
        if isinstance(cache_details, list):
            cache_write_input_tokens = sum(first_from(item, "inputTokens", "input_tokens") for item in cache_details)

    input_includes_cache = any(
        (
            first("prompt_token_count", "promptTokenCount"),
            first("cached_content_token_count", "cachedContentTokenCount"),
            first_from(input_details, "cached_tokens", "cache_read", "cache_creation"),
        )
    )
    if input_includes_cache:
        input_tokens = max(provider_input_tokens - cache_read_input_tokens - cache_write_input_tokens, 0)
        input_total_tokens = provider_input_tokens
    else:
        input_tokens = provider_input_tokens
        input_total_tokens = provider_input_tokens + cache_read_input_tokens + cache_write_input_tokens

    total_tokens = first("total_tokens", "total_token_count", "totalTokenCount", "totalTokens") or input_total_tokens + output_tokens
    if output_tokens <= 0 and total_tokens > input_total_tokens:
        output_tokens = total_tokens - input_total_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cache_read_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        input_total_tokens=input_total_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def extract_user_id_from_request(request: Any) -> Optional[str]:
    """Extract a stable logging identity from request headers or JWT claims."""
    headers = getattr(request, "headers", {}) or {}
    for header in USER_ID_HEADERS:
        value = headers.get(header)
        if value:
            return str(value)
    authorization = headers.get("authorization") or headers.get("Authorization")
    if authorization and str(authorization).lower().startswith("bearer "):
        claims = _decode_jwt_payload(str(authorization).split(" ", 1)[1].strip())
        for claim in JWT_IDENTITY_CLAIMS:
            if claims.get(claim):
                return str(claims[claim])
    return None


def extract_client_host_from_request(request: Any) -> Optional[str]:
    """Extract browser, host, or forwarded address for client context."""
    headers = getattr(request, "headers", {}) or {}
    if headers.get("x-client-host"):
        return str(headers["x-client-host"])
    if headers.get("user-agent"):
        return str(headers["user-agent"])
    if headers.get("x-forwarded-for"):
        return str(headers["x-forwarded-for"]).split(",", 1)[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def actor_type_for_user(user_id: Optional[str], explicit: Optional[str] = None) -> str:
    """Return a dashboard-friendly actor type."""
    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    return "human" if user_id else "unknown"


def emit_llm_usage_event(
    *,
    route: str,
    method: str = "POST",
    user_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    client_host: Optional[str] = None,
    provider: str = "sap-ai-core",
    model: str = "unknown",
    llm_endpoint: str = "chat.completions",
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    cache_read_input_tokens: Optional[int] = None,
    cache_write_input_tokens: int = 0,
    input_total_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    output_tokens: int = 0,
    outcome: str = "success",
    latency_ms: Optional[int] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Emit one structured LLM usage event to stdout."""
    vcap = _vcap_application()
    input_tokens = int(input_tokens or 0)
    cached_input_tokens = int(cached_input_tokens or 0)
    cache_read_input_tokens = int(cache_read_input_tokens if cache_read_input_tokens is not None else cached_input_tokens)
    cache_write_input_tokens = int(cache_write_input_tokens or 0)
    input_total_tokens = int(input_total_tokens if input_total_tokens is not None else input_tokens + cache_read_input_tokens + cache_write_input_tokens)
    output_tokens = int(output_tokens or 0)
    total_tokens = int(total_tokens if total_tokens is not None else input_total_tokens + output_tokens)
    event = {
        "schema_version": "btp.llm_usage.v1",
        "event_type": "llm_usage",
        "event_time": _now_utc(),
        "app_name": vcap.get("application_name") or vcap.get("name"),
        "space_name": vcap.get("space_name"),
        "org_name": vcap.get("organization_name") or vcap.get("org_name"),
        "route": route,
        "method": method,
        "user_hash": _hash_user(user_id),
        "actor_type": actor_type_for_user(user_id, actor_type),
        "client_host": client_host,
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "input_total_tokens": input_total_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
```

## Instrumentation Patterns

Wrap each model call at the narrowest point where the real provider request is made. In LangGraph, this means inside the node that calls `llm.invoke` or `llm.ainvoke`, not only around the outer graph invocation.

Native OpenAI chat completions through SAP Gen AI Hub:

```python
import time
from gen_ai_hub.proxy.native.openai import chat

start_time = time.perf_counter()
usage = TokenUsage()
try:
    response = chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="chat.completions",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
except Exception:
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="chat.completions",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    raise
```

LangChain or LangGraph:

```python
start_time = time.perf_counter()
usage = TokenUsage()
try:
    response = await llm_with_tools.ainvoke(messages)
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=route,
        method=method,
        user_id=user_id,
        client_host=client_host,
        provider="sap-ai-core",
        model=model,
        llm_endpoint="chat.completions",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=correlation_id,
    )
except Exception:
    emit_llm_usage_event(
        route=route,
        method=method,
        user_id=user_id,
        client_host=client_host,
        provider="sap-ai-core",
        model=model,
        llm_endpoint="chat.completions",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=correlation_id,
    )
    raise
```

OpenAI Responses API through SAP Gen AI Hub:

```python
import time
from gen_ai_hub.proxy.native.openai import responses

start_time = time.perf_counter()
usage = TokenUsage()
try:
    response = responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        reasoning={"effort": "low"},
    )
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="responses",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    text = response.output_text
except Exception:
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="responses",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    raise
```

Native Gemini through SAP Gen AI Hub:

```python
import time
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.native.google_genai.clients import Client

start_time = time.perf_counter()
usage = TokenUsage()
try:
    proxy_client = get_proxy_client("gen-ai-hub")
    client = Client(proxy_client=proxy_client)
    response = client.models.generate_content(model=model, contents=contents)
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="generateContent",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    text = response.text
except Exception:
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="generateContent",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    raise
```

Gemini responses expose token usage under `response.usage_metadata`: `prompt_token_count`, `candidates_token_count`, `total_token_count`, optional `cached_content_token_count`, and optional `thoughts_token_count`.

Native Amazon Bedrock through SAP Gen AI Hub:

```python
import time
from gen_ai_hub.proxy.native.amazon.clients import Session

start_time = time.perf_counter()
usage = TokenUsage()
try:
    bedrock = Session().client(model_name=model)
    response = bedrock.converse(
        messages=messages,
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="bedrock.converse",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    text = response["output"]["message"]["content"][0]["text"]
except Exception:
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="bedrock.converse",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    raise
```

Native Bedrock `converse` responses expose token usage under `response["usage"]`: `inputTokens`, `outputTokens`, `totalTokens`, optional `cacheReadInputTokens`, optional `cacheWriteInputTokens`, and sometimes nested `cacheDetails` write entries.

Gen AI Hub orchestration service:

```python
import time
from gen_ai_hub.orchestration.service import OrchestrationService

start_time = time.perf_counter()
usage = TokenUsage()
try:
    result = OrchestrationService(config=config).run()
    usage = extract_token_usage(result.orchestration_result)
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="orchestration",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    text = result.orchestration_result.choices[0].message.content
except Exception:
    emit_llm_usage_event(
        route=str(request.url.path),
        method=request.method,
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        provider="sap-ai-core",
        model=model,
        llm_endpoint="orchestration",
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome="error",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
        correlation_id=request.headers.get("x-correlation-id"),
    )
    raise
```

Orchestration responses expose token usage under `result.orchestration_result.usage`, typically with `prompt_tokens`, `completion_tokens`, and `total_tokens`.

Do not count tokens manually unless the provider response lacks usage metadata and the user accepts approximation.

## Tests And Acceptance

Add tests that prove:

- The stdout line is compact JSON with `schema_version="btp.llm_usage.v1"` and `event_type="llm_usage"`.
- `VCAP_APPLICATION` values populate `app_name`, `space_name`, and `org_name`.
- User identifiers are hashed and raw identities are not printed.
- The stdout event includes `cached_input_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens`, and `input_total_tokens`.
- Native chat-completion, Responses API, LangChain, Gemini, and Bedrock usage metadata normalize to uncached input, cache-read input, cache-write input, input total, output, and total tokens.
- Each route or graph node that calls an LLM emits a success event, and error paths emit an error event.

Prefer fake provider responses in unit tests. Run the project's Python test command and `python -m compileall api/app` for FastAPI projects.

## Related Skills

- `access-to-generative-ai-models` — the LLM call patterns this skill instruments.
- `langgraph-genai-hub-setup` — wrap `llm.invoke`/`ainvoke` inside graph nodes, not around the outer graph.
- `sap-btp-ai` — routing and shared environment conventions.
