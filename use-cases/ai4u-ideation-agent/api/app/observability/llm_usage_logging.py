"""Structured LLM usage logging for SAP Cloud Logging.

The helpers here emit one compact JSON line to stdout for each production LLM
call. Token extraction intentionally accepts multiple provider shapes observed
in this project: Gen AI Hub orchestration v1/v2, LangChain proxy responses,
Bedrock/Claude metadata, and raw Gemini ``generateContent`` JSON.
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
JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "user_id", "sub")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMUsageMeta:
    route: str = "/unknown"
    method: str = "POST"
    user_id: Optional[str] = None
    client_host: Optional[str] = None
    correlation_id: Optional[str] = None


def _enabled() -> bool:
    raw = os.getenv("LLM_USAGE_LOGGING_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _vcap_application() -> dict[str, Any]:
    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    return hashlib.sha256(
        f"{os.getenv('LOG_USER_HASH_SALT', '')}:{user_id}".encode("utf-8")
    ).hexdigest()[:24]


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _read_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _read_path(source: Any, *path: str) -> Any:
    current = source
    for key in path:
        current = _read_value(current, key)
        if current is None:
            return None
    return current


def _first_int(source: Any, *keys: str) -> int:
    for key in keys:
        value = _read_value(source, key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _usage_candidates(source: Any) -> list[Any]:
    candidates = [
        _read_value(source, "usage"),
        _read_value(source, "usage_metadata"),
        _read_path(source, "llm_output", "token_usage"),
        _read_path(source, "llm_output", "usage"),
        _read_path(source, "response_metadata", "token_usage"),
        _read_path(source, "response_metadata", "usage"),
        _read_path(source, "final_result", "usage"),
        _read_path(source, "orchestration_result", "usage"),
        _read_value(source, "usageMetadata"),
        _read_path(source, "module_results", "llm", "data", "usage"),
        _read_path(source, "module_results", "llm", "data", "usageMetadata"),
        _read_path(source, "module_results", "llm", "data", "completion", "usage"),
        source,
    ]
    return [item for item in candidates if item is not None]


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize token usage from LangChain, orchestration, or raw provider responses."""
    for usage in _usage_candidates(source):
        input_tokens = _first_int(
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
        total_tokens = _first_int(
            usage,
            "total_tokens",
            "total_token_count",
            "totalTokenCount",
            "totalTokens",
        )
        thoughts_tokens = _first_int(usage, "thoughtsTokenCount", "thoughts_token_count")

        # Raw Gemini REST reports candidates separately from hidden thoughts.
        # For cost attribution, output should include all non-prompt tokens.
        if thoughts_tokens > 0 and total_tokens > input_tokens:
            output_tokens = total_tokens - input_tokens

        if total_tokens <= 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens
        if output_tokens <= 0 and total_tokens > input_tokens:
            output_tokens = total_tokens - input_tokens

        if input_tokens or output_tokens or total_tokens:
            return TokenUsage(
                input_tokens=max(0, input_tokens),
                output_tokens=max(0, output_tokens),
                total_tokens=max(0, total_tokens),
            )
    return TokenUsage()


def extract_user_id_from_request(request: Any) -> Optional[str]:
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
    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    return "human" if user_id else "unknown"


def meta_from_mapping(value: Any, *, default_route: str = "/unknown") -> LLMUsageMeta:
    if isinstance(value, LLMUsageMeta):
        return value
    if not isinstance(value, Mapping):
        return LLMUsageMeta(route=default_route)
    return LLMUsageMeta(
        route=str(value.get("route") or default_route),
        method=str(value.get("method") or "POST"),
        user_id=str(value.get("user_id") or "") or None,
        client_host=str(value.get("client_host") or "") or None,
        correlation_id=str(value.get("correlation_id") or "") or None,
    )


def meta_from_request(
    request: Any,
    *,
    route: Optional[str] = None,
    method: Optional[str] = None,
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> LLMUsageMeta:
    return LLMUsageMeta(
        route=route or str(getattr(getattr(request, "url", None), "path", "") or "/unknown"),
        method=method or str(getattr(request, "method", "") or "POST"),
        user_id=user_id or extract_user_id_from_request(request),
        client_host=extract_client_host_from_request(request),
        correlation_id=correlation_id,
    )


def meta_to_mapping(meta: LLMUsageMeta | Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = meta_from_mapping(meta) if meta is not None else LLMUsageMeta()
    return {
        "route": normalized.route,
        "method": normalized.method,
        "user_id": normalized.user_id,
        "client_host": normalized.client_host,
        "correlation_id": normalized.correlation_id,
    }


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
    step_name: Optional[str] = None,
    feature: Optional[str] = None,
) -> None:
    if not _enabled():
        return
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
    if step_name:
        event["step_name"] = step_name
    if feature:
        event["feature"] = feature
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def emit_llm_usage_for_response(
    response: Any,
    *,
    meta: LLMUsageMeta | Mapping[str, Any] | None,
    provider: str = "sap-ai-core",
    model: str = "unknown",
    llm_endpoint: str = "chat.completions",
    outcome: str = "success",
    latency_ms: Optional[int] = None,
    step_name: Optional[str] = None,
    feature: Optional[str] = None,
) -> None:
    normalized_meta = meta_from_mapping(meta)
    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=normalized_meta.route,
        method=normalized_meta.method,
        user_id=normalized_meta.user_id,
        client_host=normalized_meta.client_host,
        provider=provider,
        model=model,
        llm_endpoint=llm_endpoint,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        outcome=outcome,
        latency_ms=latency_ms,
        correlation_id=normalized_meta.correlation_id,
        step_name=step_name,
        feature=feature,
    )


__all__ = [
    "LLMUsageMeta",
    "TokenUsage",
    "actor_type_for_user",
    "emit_llm_usage_event",
    "emit_llm_usage_for_response",
    "extract_client_host_from_request",
    "extract_token_usage",
    "extract_user_id_from_request",
    "meta_from_mapping",
    "meta_from_request",
    "meta_to_mapping",
]
