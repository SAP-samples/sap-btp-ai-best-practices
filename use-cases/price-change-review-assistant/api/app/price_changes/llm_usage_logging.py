from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .token_usage import normalize_token_usage

USER_ID_HEADERS = ("x-client-user-id", "x-user-id", "x-forwarded-user", "x-authenticated-user")
JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")
VALID_ACTOR_TYPES = {"human", "system", "batch", "unknown"}


@dataclass(frozen=True)
class LlmUsageContext:
    """Request metadata included in Cloud Logging LLM usage events.

    Attributes:
        route: FastAPI route path that triggered the LLM call.
        method: HTTP method for the triggering request.
        user_id: Optional raw user identity to hash before logging.
        client_host: Browser, user agent, or source host used for context.
        correlation_id: Stable correlation id reused across model calls in one request.
        actor_type: Optional dashboard-friendly actor category override.
    """

    route: str
    method: str = "POST"
    user_id: str | None = None
    client_host: str | None = None
    correlation_id: str | None = None
    actor_type: str | None = None


def _now_utc() -> str:
    """Return the current UTC timestamp in Cloud Logging-friendly ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _vcap_application() -> dict[str, Any]:
    """Parse Cloud Foundry app metadata from ``VCAP_APPLICATION``.

    Returns:
        Parsed VCAP application metadata, or an empty dictionary when unavailable.
    """
    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: str | None) -> str | None:
    """Return a salted, non-reversible short hash for a raw user identity.

    Args:
        user_id: Raw user identity extracted from headers or JWT claims.

    Returns:
        First 24 hex characters of a SHA-256 hash, or None when no user is known.
    """
    if not user_id:
        return None
    salt = os.getenv("LOG_USER_HASH_SALT", "")
    return hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode JWT claims for logging identity only, never authorization.

    Args:
        token: Bearer token value without the ``Bearer`` prefix.

    Returns:
        Decoded JWT payload dictionary, or an empty dictionary when decoding fails.
    """
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _header_value(headers: Any, name: str) -> str | None:
    """Read a case-insensitive header value from Starlette headers or a mapping.

    Args:
        headers: Request headers object or mapping.
        name: Header name to read.

    Returns:
        Header value as text, or None when missing.
    """
    value = None
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        if value is None:
            value = headers.get(name.title())
    return str(value) if value else None


def extract_user_id_from_request(request: Any) -> str | None:
    """Extract a stable logging identity from request headers or JWT claims.

    Args:
        request: FastAPI/Starlette request-like object.

    Returns:
        User identity before hashing, or None when no identity is available.
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


def extract_client_host_from_request(request: Any) -> str | None:
    """Extract browser, host, or forwarded address for client context.

    Args:
        request: FastAPI/Starlette request-like object.

    Returns:
        Browser/user-agent or client host string, when available.
    """
    headers = getattr(request, "headers", {}) or {}
    for header in ("x-client-host", "user-agent", "x-forwarded-for"):
        value = _header_value(headers, header)
        if value:
            return value.split(",", 1)[0].strip() if header == "x-forwarded-for" else value
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def actor_type_for_user(user_id: str | None, explicit: str | None = None) -> str:
    """Return a dashboard-friendly actor type.

    Args:
        user_id: User identity before hashing.
        explicit: Optional caller-supplied actor type.

    Returns:
        One of human, system, batch, or unknown.
    """
    if explicit in VALID_ACTOR_TYPES:
        return explicit
    return "human" if user_id else "unknown"


def build_llm_usage_context_from_request(request: Any) -> LlmUsageContext:
    """Build reusable LLM usage logging context from one API request.

    Args:
        request: FastAPI/Starlette request-like object.

    Returns:
        LlmUsageContext shared by all LLM calls made for the request.
    """
    headers = getattr(request, "headers", {}) or {}
    return LlmUsageContext(
        route=str(getattr(getattr(request, "url", None), "path", "unknown")),
        method=str(getattr(request, "method", "POST")),
        user_id=extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        correlation_id=_header_value(headers, "x-correlation-id") or str(uuid.uuid4()),
    )


def emit_llm_usage_event(
    *,
    context: LlmUsageContext,
    model: str,
    llm_endpoint: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    outcome: str = "success",
    latency_ms: int | None = None,
    provider: str = "sap-ai-core",
) -> None:
    """Emit one compact SAP Cloud Logging LLM usage event to stdout.

    Args:
        context: Request metadata shared by all LLM calls in one API request.
        model: SAP Gen AI Hub model deployment name.
        llm_endpoint: Provider endpoint family, such as responses or chat.completions.
        input_tokens: Prompt/input token count.
        output_tokens: Completion/output token count.
        outcome: success or error.
        latency_ms: Provider call latency in milliseconds.
        provider: Provider label for dashboards.

    Returns:
        None.
    """
    vcap = _vcap_application()
    input_tokens = int(input_tokens or 0)
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
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": context.correlation_id or str(uuid.uuid4()),
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_llm_usage_for_response(
    *,
    context: LlmUsageContext | None,
    model: str,
    llm_endpoint: str,
    response: Any | None,
    outcome: str,
    started_at: float,
) -> None:
    """Extract response usage and emit a Cloud Logging event when context exists.

    Args:
        context: Optional request logging context.
        model: SAP Gen AI Hub model deployment name.
        llm_endpoint: Provider endpoint family.
        response: Provider response carrying token usage metadata.
        outcome: success or error.
        started_at: ``time.perf_counter()`` value captured before the provider call.

    Returns:
        None.
    """
    if context is None:
        return
    input_tokens, output_tokens, _total_tokens, _usage_available = normalize_token_usage(response)
    emit_llm_usage_event(
        context=context,
        model=model,
        llm_endpoint=llm_endpoint,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        outcome=outcome,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
