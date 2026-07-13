"""Structured LLM usage logging for SAP Cloud Logging.

Events are written as single-line JSON to stdout so Cloud Foundry syslog drains
can index them in SAP Cloud Logging / OpenSearch under ``logs-cfsyslog-*``.
Do not log raw prompts, completions, tokens, cookies, API keys, or raw emails.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = "btp.llm_usage.v1"
EVENT_TYPE = "llm_usage"
USER_HEADERS = (
    "x-client-user-id",
    "x-user-id",
    "x-forwarded-user",
    "x-authenticated-user",
)
JWT_IDENTITY_CLAIMS = ("user_name", "email", "user_uuid", "sub")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _headers_from_request(request: Any) -> Mapping[str, str]:
    headers = getattr(request, "headers", None)
    return headers or {}


def _decode_jwt_payload_without_verification(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8"))
        return json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}


def extract_user_id_from_request(request: Any) -> Optional[str]:
    """Extract a stable user identifier for logging only.

    Priority:
    1. explicit client/user headers
    2. identity claims from an already authenticated Bearer JWT
    3. None

    JWT payload decoding here is used only to extract a logging identity from an
    already authenticated request. It must not be used for authorization
    decisions.
    """

    headers = _headers_from_request(request)
    for header_name in USER_HEADERS:
        value = headers.get(header_name) if hasattr(headers, "get") else None
        if value:
            return str(value).strip() or None

    auth_header = headers.get("authorization") if hasattr(headers, "get") else None
    if not auth_header or not str(auth_header).lower().startswith("bearer "):
        return None

    token = str(auth_header).split(" ", 1)[1].strip()
    payload = _decode_jwt_payload_without_verification(token)
    for claim in JWT_IDENTITY_CLAIMS:
        value = payload.get(claim)
        if value:
            return str(value).strip() or None
    return None


def _hash_user_id(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    salt = os.getenv("LOG_USER_HASH_SALT", "")
    digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    return digest[:24]


def _cf_app_context() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw_vcap = os.getenv("VCAP_APPLICATION")
    if not raw_vcap:
        return None, None, None
    try:
        vcap = json.loads(raw_vcap)
    except json.JSONDecodeError:
        return None, None, None
    return (
        vcap.get("application_name") or vcap.get("name"),
        vcap.get("space_name"),
        vcap.get("organization_name") or vcap.get("org_name"),
    )


def _client_host_from_request(request: Any) -> Optional[str]:
    headers = _headers_from_request(request)
    if hasattr(headers, "get"):
        user_agent = headers.get("user-agent")
        if user_agent:
            return str(user_agent)
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return str(host) if host else None


def _correlation_id_from_request(request: Any) -> Optional[str]:
    headers = _headers_from_request(request)
    if not hasattr(headers, "get"):
        return None
    return (
        headers.get("x-correlation-id")
        or headers.get("x-request-id")
        or headers.get("x-vcap-request-id")
    )


def infer_actor_type(user_id: Optional[str], actor_type: Optional[str] = None) -> str:
    if actor_type in {"human", "system", "batch", "unknown"}:
        return actor_type
    if user_id:
        return "human"
    return "unknown"


def extract_token_counts(response: Any) -> Tuple[int, int, int]:
    """Best-effort token extraction from common LLM response shapes."""

    def _as_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                if isinstance(dumped, Mapping):
                    return dict(dumped)
            except Exception:
                pass
        return getattr(value, "__dict__", {}) or {}

    candidates = [
        response,
        getattr(response, "usage_metadata", None),
        getattr(response, "usageMetadata", None),
        getattr(response, "usage", None),
        getattr(response, "response_metadata", None),
        getattr(response, "orchestration_result", None),
        getattr(response, "final_result", None),
    ]
    if isinstance(response, Mapping):
        candidates.extend(
            [
                response.get("usage_metadata"),
                response.get("usageMetadata"),
                response.get("usage"),
                response.get("response_metadata"),
                response.get("orchestration_result"),
                response.get("final_result"),
            ]
        )

    visited = set()
    index = 0
    while index < len(candidates):
        usage = candidates[index]
        index += 1
        if not usage:
            continue

        usage_id = id(usage)
        if usage_id in visited:
            continue
        visited.add(usage_id)

        usage = _as_mapping(usage)
        for nested_key in (
            "token_usage",
            "usage_metadata",
            "usageMetadata",
            "usage",
            "orchestration_result",
            "final_result",
            "response_metadata",
        ):
            nested = usage.get(nested_key)
            if nested:
                candidates.append(nested)
        input_tokens = _safe_int(
            usage.get("promptTokenCount")
            or usage.get("prompt_tokens")
            or usage.get("promptTokens")
            or usage.get("input_tokens")
            or usage.get("inputTokenCount")
        )
        output_tokens = _safe_int(
            usage.get("candidatesTokenCount")
            or usage.get("completion_tokens")
            or usage.get("completionTokens")
            or usage.get("output_tokens")
            or usage.get("outputTokenCount")
        )
        total_tokens = _safe_int(
            usage.get("total_tokens")
            or usage.get("totalTokenCount")
            or (input_tokens + output_tokens)
        )
        if input_tokens or output_tokens or total_tokens:
            if not total_tokens:
                total_tokens = input_tokens + output_tokens
            return input_tokens, output_tokens, total_tokens

    return 0, 0, 0


def emit_llm_usage_event(
    *,
    route: str,
    method: str,
    provider: str,
    model: str,
    llm_endpoint: str,
    input_tokens: Any = 0,
    output_tokens: Any = 0,
    total_tokens: Any = None,
    outcome: str = "success",
    latency_ms: Optional[Any] = None,
    request: Any = None,
    user_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    client_host: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Emit one Cloud Logging-compatible LLM usage event and return it."""

    app_name, space_name, org_name = _cf_app_context()
    detected_user_id = user_id or (extract_user_id_from_request(request) if request is not None else None)
    resolved_client_host = client_host or (_client_host_from_request(request) if request is not None else None)
    resolved_correlation_id = (
        correlation_id
        or (_correlation_id_from_request(request) if request is not None else None)
        or str(uuid.uuid4())
    )

    input_count = _safe_int(input_tokens)
    output_count = _safe_int(output_tokens)
    total_count = _safe_int(total_tokens, input_count + output_count)
    if total_count == 0 and (input_count or output_count):
        total_count = input_count + output_count

    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_TYPE,
        "event_time": _utc_now_iso(),
        "app_name": app_name,
        "space_name": space_name,
        "org_name": org_name,
        "route": route,
        "method": method,
        "user_hash": _hash_user_id(detected_user_id),
        "actor_type": infer_actor_type(detected_user_id, actor_type),
        "client_host": resolved_client_host,
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": total_count,
        "outcome": outcome if outcome in {"success", "error", "timeout", "cancelled"} else "error",
        "latency_ms": None if latency_ms is None else _safe_int(latency_ms),
        "correlation_id": resolved_correlation_id,
    }
    print(json.dumps(event, separators=(",", ":"), ensure_ascii=False), flush=True)
    return event
