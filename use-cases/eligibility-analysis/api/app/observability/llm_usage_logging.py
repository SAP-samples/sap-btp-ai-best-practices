"""Structured stdout logging for SAP Gen AI Hub LLM token usage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Optional

USER_ID_HEADERS = (
    "x-client-user-id",
    "x-user-id",
    "x-forwarded-user",
    "x-authenticated-user",
)
JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")

_CURRENT_CONTEXT: ContextVar[Optional["LlmUsageContext"]] = ContextVar(
    "llm_usage_context",
    default=None,
)


@dataclass(frozen=True)
class TokenUsage:
    """Normalized LLM token usage values.

    Attributes:
        input_tokens: Prompt/input tokens reported by the provider.
        cached_input_tokens: Input tokens served from provider cache.
        output_tokens: Completion/output tokens reported by the provider.
        total_tokens: Provider total when available, else input plus output.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LlmUsageContext:
    """Request or system context attached to one LLM usage event.

    Attributes:
        route: Logical API route or background operation label.
        method: HTTP method or operation verb used for dashboards.
        user_id: Raw user identifier, hashed before logging.
        actor_type: Explicit actor type such as human, system, batch, or unknown.
        client_host: Browser, user agent, or forwarded host context.
        correlation_id: Request correlation id; generated if absent.
    """

    route: str
    method: str = "POST"
    user_id: Optional[str] = None
    actor_type: Optional[str] = None
    client_host: Optional[str] = None
    correlation_id: Optional[str] = None


def _now_utc() -> str:
    """Return the current UTC timestamp in ISO-8601 millisecond format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _vcap_application() -> dict[str, Any]:
    """Parse Cloud Foundry app metadata from VCAP_APPLICATION."""
    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    """Return a salted, non-reversible user hash for logging."""
    if not user_id:
        return None
    salt = os.getenv("LOG_USER_HASH_SALT", "")
    return hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode JWT claims for logging identity discovery only."""
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _read_value(source: Any, key: str) -> Any:
    """Read a key from a mapping or object attribute."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize OpenAI, LangChain, Gemini, Bedrock, and cached token fields.

    Args:
        source: Provider response, usage object, or metadata dictionary.

    Returns:
        TokenUsage with integer counts. Missing or malformed values are zero.
    """
    usage = _read_value(source, "usage") or _read_value(source, "usage_metadata") or source
    response_metadata = _read_value(source, "response_metadata")
    if response_metadata and not _read_value(source, "usage_metadata"):
        usage = (
            _read_value(response_metadata, "token_usage")
            or _read_value(response_metadata, "usage")
            or usage
        )

    def first_from(container: Any, *keys: str) -> int:
        """Return the first integer-like value found in a container."""
        for key in keys:
            value = _read_value(container, key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
        return 0

    def first(*keys: str) -> int:
        """Return the first integer-like value from the active usage object."""
        return first_from(usage, *keys)

    input_tokens = first(
        "input_tokens",
        "prompt_tokens",
        "prompt_token_count",
        "promptTokenCount",
        "inputTokens",
    )
    output_tokens = first(
        "output_tokens",
        "completion_tokens",
        "completion_token_count",
        "candidates_token_count",
        "candidatesTokenCount",
        "outputTokens",
    )
    total_tokens = first(
        "total_tokens",
        "total_token_count",
        "totalTokenCount",
        "totalTokens",
    ) or input_tokens + output_tokens
    cached_input_tokens = first(
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
        "cached_content_token_count",
    )
    if not cached_input_tokens:
        for details_key in (
            "input_tokens_details",
            "input_token_details",
            "prompt_tokens_details",
            "prompt_token_details",
        ):
            cached_input_tokens = first_from(
                _read_value(usage, details_key),
                "cached_tokens",
                "cache_read",
                "cacheRead",
            )
            if cached_input_tokens:
                break
    if output_tokens <= 0 and total_tokens > input_tokens:
        output_tokens = total_tokens - input_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _header_value(headers: Any, key: str) -> Optional[str]:
    """Read a header value from case-sensitive or case-insensitive mappings."""
    if not headers:
        return None
    value = headers.get(key)
    if value is None:
        value = headers.get(key.lower())
    if value is None:
        value = headers.get(key.title())
    return str(value) if value else None


def extract_user_id_from_request(request: Any) -> Optional[str]:
    """Extract a stable logging identity from request headers or JWT claims."""
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
    """Extract browser, host, or forwarded address for client context."""
    headers = getattr(request, "headers", {}) or {}
    for header in ("x-client-host", "user-agent", "x-forwarded-for"):
        value = _header_value(headers, header)
        if value:
            return value.split(",", 1)[0].strip() if header == "x-forwarded-for" else value
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def actor_type_for_user(user_id: Optional[str], explicit: Optional[str] = None) -> str:
    """Return a dashboard-friendly actor type."""
    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    return "human" if user_id else "unknown"


def context_from_request(
    request: Any,
    *,
    route: Optional[str] = None,
    actor_type: Optional[str] = None,
) -> LlmUsageContext:
    """Build an LLM usage context from a FastAPI/Starlette request."""
    headers = getattr(request, "headers", {}) or {}
    path = route
    if path is None:
        url = getattr(request, "url", None)
        path = str(getattr(url, "path", "") or "unknown")
    method = str(getattr(request, "method", "POST") or "POST")
    return LlmUsageContext(
        route=path,
        method=method,
        user_id=extract_user_id_from_request(request),
        actor_type=actor_type,
        client_host=extract_client_host_from_request(request),
        correlation_id=_header_value(headers, "x-correlation-id"),
    )


def system_context(
    route: str,
    *,
    method: str = "POST",
    correlation_id: Optional[str] = None,
) -> LlmUsageContext:
    """Build a context for background or system-owned LLM work."""
    return LlmUsageContext(
        route=route,
        method=method,
        actor_type="system",
        correlation_id=correlation_id,
    )


@contextmanager
def llm_usage_context(context: Optional[LlmUsageContext]) -> Iterator[None]:
    """Temporarily attach LLM usage context to the current execution flow."""
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def get_current_llm_usage_context() -> Optional[LlmUsageContext]:
    """Return the current contextvar-based LLM usage context, if any."""
    return _CURRENT_CONTEXT.get()


def emit_llm_usage_event(
    *,
    context: LlmUsageContext,
    provider: str = "sap-ai-core",
    model: str = "unknown",
    llm_endpoint: str = "chat.completions",
    input_tokens: int = 0,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
    outcome: str = "success",
    latency_ms: Optional[int] = None,
) -> None:
    """Emit one compact structured LLM usage event to stdout."""
    vcap = _vcap_application()
    input_tokens = int(input_tokens or 0)
    cached_input_tokens = int(cached_input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    event = {
        "schema_version": "btp.llm_usage.v1",
        "event_type": "llm_usage",
        "event_time": _now_utc(),
        "app_name": vcap.get("application_name") or vcap.get("name"),
        "space_name": vcap.get("space_name"),
        "org_name": vcap.get("organization_name") or vcap.get("org_name"),
        "route": context.route,
        "method": context.method,
        "user_hash": _hash_user(context.user_id),
        "actor_type": actor_type_for_user(context.user_id, context.actor_type),
        "client_host": context.client_host,
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": context.correlation_id or str(uuid.uuid4()),
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
