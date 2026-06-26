"""Structured stdout logging for LLM token usage.

Cloud Foundry log drains can collect these JSON lines into SAP Cloud Logging.
The emitted schema is intentionally stable because dashboards and alerts may
depend on these field names.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
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

JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")


@dataclass(frozen=True)
class TokenUsage:
    """Normalized LLM token usage values.

    Attributes:
        input_tokens: Prompt or input tokens consumed by the model call.
        output_tokens: Completion or output tokens produced by the model call.
        total_tokens: Total tokens reported by the provider or computed locally.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _now_utc() -> str:
    """Return the current UTC timestamp in Cloud Logging-friendly ISO format."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _vcap_application() -> dict[str, Any]:
    """Parse the Cloud Foundry VCAP_APPLICATION object from the environment."""
    try:
        value = os.getenv("VCAP_APPLICATION", "{}")
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    """Return a salted, non-reversible user hash for logging."""
    if not user_id:
        return None

    salt = os.getenv("LOG_USER_HASH_SALT", "")
    value = f"{salt}:{user_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode a JWT payload without validating it for identity logging only."""
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
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _coerce_int(value: Any) -> int:
    """Convert a token count-like value into an integer with a zero fallback."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_numeric_value(source: Any, keys: tuple[str, ...]) -> int:
    """Return the first non-empty integer value found for the given keys."""
    for key in keys:
        value = _read_value(source, key)
        if value is not None:
            return _coerce_int(value)
    return 0


def _usage_source(source: Any) -> Any:
    """Return the most likely token usage object from a provider response."""
    usage = _read_value(source, "usage")
    if usage is not None:
        return usage

    usage_metadata = _read_value(source, "usage_metadata")
    if usage_metadata is not None:
        return usage_metadata

    response_metadata = _read_value(source, "response_metadata")
    if response_metadata is not None:
        token_usage = _read_value(response_metadata, "token_usage")
        if token_usage is not None:
            return token_usage

    return source


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize provider-specific token metadata into TokenUsage.

    Args:
        source: A model response, usage object, LangChain AIMessage, or mapping
            containing token usage fields.

    Returns:
        TokenUsage: Normalized input, output, and total token counts. Missing or
            unrecognized metadata returns zero counts.
    """
    if source is None:
        return TokenUsage()

    usage = _usage_source(source)
    input_tokens = _first_numeric_value(
        usage,
        (
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
        ),
    )
    output_tokens = _first_numeric_value(
        usage,
        (
            "output_tokens",
            "completion_tokens",
            "completion_token_count",
            "candidatesTokenCount",
        ),
    )
    total_tokens = _first_numeric_value(
        usage,
        (
            "total_tokens",
            "total_token_count",
            "totalTokenCount",
        ),
    )

    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    if output_tokens <= 0 and total_tokens > input_tokens:
        output_tokens = total_tokens - input_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def usage_dict_from_tokens(usage: TokenUsage) -> dict[str, int]:
    """Return API response usage fields from normalized token usage."""
    return {
        "prompt_tokens": int(usage.input_tokens or 0),
        "completion_tokens": int(usage.output_tokens or 0),
        "total_tokens": int(usage.total_tokens or 0),
    }


def extract_user_id_from_request(request: Any) -> Optional[str]:
    """Extract a stable logging identity from request headers or JWT claims.

    JWT payload decoding here is used only to extract a logging identity from an
    already authenticated request. It must not be used for authorization.
    """
    headers = getattr(request, "headers", {}) or {}

    for header in USER_ID_HEADERS:
        value = headers.get(header)
        if value:
            return str(value)

    authorization = headers.get("authorization") or headers.get("Authorization")
    if authorization and str(authorization).lower().startswith("bearer "):
        claims = _decode_jwt_payload(str(authorization).split(" ", 1)[1].strip())
        for claim in JWT_IDENTITY_CLAIMS:
            value = claims.get(claim)
            if value:
                return str(value)

    return None


def extract_client_host_from_request(request: Any) -> Optional[str]:
    """Extract a useful client identifier from request headers or connection info."""
    headers = getattr(request, "headers", {}) or {}

    explicit_host = headers.get("x-client-host")
    if explicit_host:
        return str(explicit_host)

    user_agent = headers.get("user-agent")
    if user_agent:
        return str(user_agent)

    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        return str(forwarded_for).split(",", 1)[0].strip()

    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def actor_type_for_user(user_id: Optional[str], explicit: Optional[str] = None) -> str:
    """Return the actor type to store in the LLM usage event."""
    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    if user_id:
        return "human"
    return "unknown"


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
    output_tokens: int = 0,
    outcome: str = "success",
    latency_ms: Optional[int] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Emit one structured LLM usage event to stdout.

    Args:
        route: HTTP route path or internal call label where the model was used.
        method: HTTP method or "INTERNAL" for non-request model calls.
        user_id: Raw user identifier to hash before logging.
        actor_type: Optional explicit actor type.
        client_host: Browser, host, or forwarded address for client context.
        provider: Provider label expected by Cloud Logging dashboards.
        model: Model deployment name.
        llm_endpoint: Provider endpoint family, such as "chat.completions".
        input_tokens: Prompt or input token count.
        output_tokens: Completion or output token count.
        outcome: "success" or "error" for the attempted model call.
        latency_ms: Model call latency in milliseconds.
        correlation_id: Optional request correlation identifier.
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
        "route": route,
        "method": method,
        "user_hash": _hash_user(user_id),
        "actor_type": actor_type_for_user(user_id, actor_type),
        "client_host": client_host,
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }

    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
