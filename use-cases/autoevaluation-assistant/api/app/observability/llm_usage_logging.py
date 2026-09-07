"""Structured stdout logging for SAP Gen AI Hub token usage events."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

USER_ID_HEADERS = (
    "x-client-user-id",
    "x-user-id",
    "x-forwarded-user",
    "x-authenticated-user",
)
"""Request headers that may contain a stable end-user identity."""

JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")
"""JWT claims that can identify a caller for hashed usage attribution."""


@dataclass(frozen=True)
class TokenUsage:
    """Normalized LLM token usage values.

    Inputs:
        input_tokens: Uncached prompt or input token count.
        cached_input_tokens: Legacy cache-read token count.
        cache_read_input_tokens: Provider-reported reused prompt tokens.
        cache_write_input_tokens: Provider-reported prompt tokens written to cache.
        input_total_tokens: Prompt tokens before separating cache activity.
        output_tokens: Completion or generated token count.
        total_tokens: Total provider token count, when available.

    Outputs:
        Immutable token usage record consumed by stdout event emitters.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    input_total_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LlmUsageContext:
    """Request or worker context attached to an LLM usage event.

    Inputs:
        route: API route, graph route, or worker operation name.
        method: HTTP method or synthetic method for worker calls.
        user_id: Optional raw user identity; emitted only as a salted hash.
        actor_type: Optional actor category such as human or batch.
        client_host: Optional user-agent, host, or worker identifier.
        correlation_id: Optional request, task, or context correlation ID.

    Outputs:
        Immutable logging context passed through provider wrappers.
    """

    route: str
    method: str = "POST"
    user_id: str | None = None
    actor_type: str | None = None
    client_host: str | None = None
    correlation_id: str | None = None


def _now_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 millisecond format.

    Inputs:
        None.

    Outputs:
        str: Timestamp ending in ``Z`` for Cloud Logging dashboards.
    """

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _vcap_application() -> dict[str, Any]:
    """Parse Cloud Foundry application metadata from ``VCAP_APPLICATION``.

    Inputs:
        None. Reads process environment.

    Outputs:
        dict[str, Any]: Parsed metadata, or an empty dictionary when absent or
        invalid.
    """

    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    """Return a salted, non-reversible hash for a user identity.

    Inputs:
        user_id: Raw user identifier or ``None``.

    Outputs:
        str | None: First 24 hex characters of the salted SHA-256 digest, or
        ``None`` when no identity is supplied.
    """

    if not user_id:
        return None
    salt = os.getenv("LOG_USER_HASH_SALT", "")
    return hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode JWT claims for logging identity only, never authorization.

    Inputs:
        token: Bearer token string without the ``Bearer`` prefix.

    Outputs:
        dict[str, Any]: Decoded JSON claims, or an empty dictionary when the
        token cannot be decoded.
    """

    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _read_value(source: Any, key: str) -> Any:
    """Read a value from a mapping key or object attribute.

    Inputs:
        source: Mapping, object, or ``None``.
        key: Field or attribute name to read.

    Outputs:
        Any: Matching value or ``None`` when unavailable.
    """

    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _first_int(usage: Any, *keys: str) -> int:
    """Return the first integer-like token value found in ``usage``.

    Inputs:
        usage: Mapping or object containing provider usage fields.
        *keys: Candidate field names ordered by preference.

    Outputs:
        int: Parsed non-negative token value, or ``0`` when unavailable.
    """

    for key in keys:
        value = _read_value(usage, key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
    return 0


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize OpenAI, LangChain, Gemini, Bedrock, and mapping token fields.

    Inputs:
        source: Provider response, LangChain message, usage mapping, or object
        exposing usage metadata.

    Outputs:
        TokenUsage: Stable input, output, and total token counters. Missing
        provider usage returns zeroes.
    """

    usage = _read_value(source, "usage") or _read_value(source, "usage_metadata") or source
    response_metadata = _read_value(source, "response_metadata")
    if response_metadata and not _read_value(source, "usage_metadata"):
        usage = (
            _read_value(response_metadata, "token_usage")
            or _read_value(response_metadata, "usage")
            or usage
        )

    provider_input_tokens = _first_int(
        usage,
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
        "promptTokenCount",
        "inputTokens",
    )
    output_tokens = _first_int(
        usage,
        "output_tokens",
        "completion_tokens",
        "completion_token_count",
        "candidates_token_count",
        "candidatesTokenCount",
        "outputTokens",
    )
    input_details = next(
        (
            _read_value(usage, key)
            for key in (
                "input_tokens_details",
                "input_token_details",
                "prompt_tokens_details",
                "prompt_token_details",
            )
            if _read_value(usage, key)
        ),
        {},
    )
    cache_read_input_tokens = _first_int(
        usage,
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cached_content_token_count",
        "cachedContentTokenCount",
    ) or _first_int(
        input_details,
        "cached_tokens",
        "cache_read",
        "cacheRead",
        "cache_read_input_tokens",
    )
    cache_write_input_tokens = _first_int(
        usage,
        "cache_write_input_tokens",
        "cache_creation_input_tokens",
        "cacheWriteInputTokens",
    ) or _first_int(
        input_details,
        "cache_write",
        "cache_creation",
        "cache_write_input_tokens",
        "cache_creation_input_tokens",
    )
    input_includes_cache = any(
        (
            _first_int(usage, "prompt_token_count", "promptTokenCount"),
            _first_int(
                usage,
                "cached_content_token_count",
                "cachedContentTokenCount",
            ),
            _first_int(input_details, "cached_tokens", "cache_read", "cache_creation"),
        )
    )
    if input_includes_cache:
        input_tokens = max(
            provider_input_tokens - cache_read_input_tokens - cache_write_input_tokens,
            0,
        )
        input_total_tokens = provider_input_tokens
    else:
        input_tokens = provider_input_tokens
        input_total_tokens = (
            provider_input_tokens
            + cache_read_input_tokens
            + cache_write_input_tokens
        )

    total_tokens = _first_int(
        usage,
        "total_tokens",
        "total_token_count",
        "totalTokenCount",
        "totalTokens",
    )
    if total_tokens <= 0:
        total_tokens = input_total_tokens + output_tokens
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


def _header_value(headers: Any, name: str) -> str | None:
    """Read a header value from case-sensitive or case-insensitive mappings.

    Inputs:
        headers: Request headers mapping or object.
        name: Lower-case header name to read.

    Outputs:
        str | None: Header value when present.
    """

    if not headers:
        return None
    value = headers.get(name) if hasattr(headers, "get") else None
    if value:
        return str(value)
    title_name = "-".join(part.capitalize() for part in name.split("-"))
    value = headers.get(title_name) if hasattr(headers, "get") else None
    return str(value) if value else None


def extract_user_id_from_request(request: Any) -> Optional[str]:
    """Extract a stable logging identity from request headers or JWT claims.

    Inputs:
        request: FastAPI, Starlette, or request-like object with ``headers``.

    Outputs:
        str | None: Raw identity for hashing by the emitter, or ``None``.
    """

    headers = getattr(request, "headers", {}) or {}
    for header in USER_ID_HEADERS:
        value = _header_value(headers, header)
        if value:
            return value
    authorization = _header_value(headers, "authorization")
    if authorization and authorization.lower().startswith("bearer "):
        claims = _decode_jwt_payload(authorization.split(" ", 1)[1].strip())
        for claim in JWT_IDENTITY_CLAIMS:
            if claims.get(claim):
                return str(claims[claim])
    return None


def extract_client_host_from_request(request: Any) -> Optional[str]:
    """Extract user-agent, browser, host, or forwarded address context.

    Inputs:
        request: FastAPI, Starlette, or request-like object with headers/client.

    Outputs:
        str | None: Client context suitable for operational filtering.
    """

    headers = getattr(request, "headers", {}) or {}
    for header in ("x-client-host", "user-agent", "x-forwarded-for"):
        value = _header_value(headers, header)
        if value:
            return value.split(",", 1)[0].strip() if header == "x-forwarded-for" else value
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def actor_type_for_user(user_id: Optional[str], explicit: Optional[str] = None) -> str:
    """Return a dashboard-friendly actor type.

    Inputs:
        user_id: Optional identity discovered for the request.
        explicit: Optional caller-supplied actor category.

    Outputs:
        str: One of ``human``, ``system``, ``batch``, or ``unknown``.
    """

    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    return "human" if user_id else "unknown"


def usage_context_from_config(
    config: Mapping[str, Any] | None,
    default_route: str,
    default_actor_type: str = "batch",
) -> LlmUsageContext:
    """Build usage context from a LangGraph runtime config mapping.

    Inputs:
        config: Optional LangGraph config containing ``metadata.llm_usage``.
        default_route: Route used when no metadata is supplied.
        default_actor_type: Actor type for synthetic worker/background calls.

    Outputs:
        LlmUsageContext: Normalized context for model-call instrumentation.
    """

    metadata = (config or {}).get("metadata", {})
    llm_usage = metadata.get("llm_usage", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(llm_usage, Mapping):
        llm_usage = {}
    return LlmUsageContext(
        route=str(llm_usage.get("route") or default_route),
        method=str(llm_usage.get("method") or "POST"),
        user_id=(
            str(llm_usage["user_id"])
            if llm_usage.get("user_id") is not None
            else None
        ),
        actor_type=str(llm_usage.get("actor_type") or default_actor_type),
        client_host=(
            str(llm_usage["client_host"])
            if llm_usage.get("client_host") is not None
            else None
        ),
        correlation_id=(
            str(llm_usage["correlation_id"])
            if llm_usage.get("correlation_id") is not None
            else None
        ),
    )


def model_name_from_llm(llm: Any, default: str = "unknown") -> str:
    """Return a stable model name from common LangChain wrapper attributes.

    Inputs:
        llm: LangChain chat model, bound model, or test double.
        default: Value returned when no model-like attribute is present.

    Outputs:
        str: Model or deployment name for usage events.
    """

    for attribute in ("proxy_model_name", "model_name", "model", "deployment_name"):
        value = getattr(llm, attribute, None)
        if value:
            return str(value)
    return default


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
    """Emit one compact structured LLM usage event to stdout.

    Inputs:
        route: API route or synthetic worker operation name.
        method: HTTP method or synthetic operation method.
        user_id: Optional raw identity that will be hashed before emission.
        actor_type: Optional caller category.
        client_host: Optional client or worker context.
        provider: LLM provider label.
        model: Model or deployment name.
        llm_endpoint: Provider endpoint category.
        input_tokens: Uncached input token count.
        cached_input_tokens: Legacy cache-read input token count.
        cache_read_input_tokens: Reused cache token count.
        cache_write_input_tokens: Cache-write input token count.
        input_total_tokens: Full input count before separating cache activity.
        total_tokens: Provider-reported total token count when available.
        output_tokens: Output token count.
        outcome: ``success`` or ``error``.
        latency_ms: Optional call latency in milliseconds.
        correlation_id: Optional request/task/context correlation ID.

    Outputs:
        None. Writes one JSON object to stdout and flushes immediately.
    """

    vcap = _vcap_application()
    normalized_input_tokens = int(input_tokens or 0)
    normalized_cached_input_tokens = int(cached_input_tokens or 0)
    normalized_cache_read_input_tokens = int(
        cache_read_input_tokens
        if cache_read_input_tokens is not None
        else normalized_cached_input_tokens
    )
    normalized_cache_write_input_tokens = int(cache_write_input_tokens or 0)
    normalized_input_total_tokens = int(
        input_total_tokens
        if input_total_tokens is not None
        else (
            normalized_input_tokens
            + normalized_cache_read_input_tokens
            + normalized_cache_write_input_tokens
        )
    )
    normalized_output_tokens = int(output_tokens or 0)
    normalized_total_tokens = int(
        total_tokens
        if total_tokens is not None
        else normalized_input_total_tokens + normalized_output_tokens
    )
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
        "input_tokens": normalized_input_tokens,
        "cached_input_tokens": normalized_cached_input_tokens,
        "cache_read_input_tokens": normalized_cache_read_input_tokens,
        "cache_write_input_tokens": normalized_cache_write_input_tokens,
        "input_total_tokens": normalized_input_total_tokens,
        "output_tokens": normalized_output_tokens,
        "total_tokens": normalized_total_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_llm_usage_from_response(
    *,
    response: Any,
    context: LlmUsageContext,
    model: str,
    llm_endpoint: str,
    outcome: str,
    started_at: float,
) -> TokenUsage:
    """Extract usage from a provider response and emit a usage event.

    Inputs:
        response: Provider response or LangChain message with usage metadata.
        context: Route/user/correlation metadata for the call.
        model: Model or deployment name.
        llm_endpoint: Provider endpoint category.
        outcome: ``success`` or ``error``.
        started_at: ``time.perf_counter`` value captured before the call.

    Outputs:
        TokenUsage: Usage values emitted to stdout.
    """

    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=context.route,
        method=context.method,
        user_id=context.user_id,
        actor_type=context.actor_type,
        client_host=context.client_host,
        model=model,
        llm_endpoint=llm_endpoint,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        input_total_tokens=usage.input_total_tokens,
        total_tokens=usage.total_tokens,
        output_tokens=usage.output_tokens,
        outcome=outcome,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        correlation_id=context.correlation_id,
    )
    return usage
