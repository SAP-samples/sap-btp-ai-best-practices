"""Structured LLM token-usage events for SAP Cloud Logging.

Every LLM call made by the API emits exactly one compact JSON line to stdout
(schema ``btp.llm_usage.v1``). When the API app is bound to the Cloud Logging
service in Cloud Foundry, these lines are indexed automatically and can be
aggregated in dashboards (tokens per model, per route, per outcome).

Because LLM calls in this app run inside background workers (no HTTP request
object available), the request-level fields (route, method, actor type,
correlation id) are carried through a ``ContextVar`` that the caller sets with
``llm_call_context(...)`` before invoking the LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class TokenUsage:
    """Normalized LLM token usage values.

    Attributes:
        input_tokens: Uncached input tokens.
        cached_input_tokens: Legacy cache-read field, mirrors cache_read_input_tokens.
        cache_read_input_tokens: Input tokens served from the provider prompt cache.
        cache_write_input_tokens: Input tokens written to the provider prompt cache.
        input_total_tokens: All input tokens (uncached + cache read + cache write).
        output_tokens: Generated tokens, including reasoning tokens.
        total_tokens: Provider-reported total, or input_total + output.
    """

    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    input_total_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LlmCallContext:
    """Request-level metadata attached to every usage event.

    Attributes:
        route: API route that triggered the LLM call.
        method: HTTP method of that route.
        actor_type: One of human, system, batch, unknown.
        correlation_id: Id linking all LLM calls of one unit of work (one email).
        user_id: Optional user identity; hashed before logging.
    """

    route: str = "unknown"
    method: str = "POST"
    actor_type: str = "unknown"
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None


_CURRENT_CONTEXT: ContextVar[LlmCallContext] = ContextVar("llm_call_context", default=LlmCallContext())


@contextmanager
def llm_call_context(**fields: Any) -> Iterator[LlmCallContext]:
    """Set the usage-event metadata for LLM calls made inside the block.

    Args:
        **fields: Any LlmCallContext field (route, method, actor_type, correlation_id, user_id).

    Yields:
        The active LlmCallContext.
    """
    token = _CURRENT_CONTEXT.set(LlmCallContext(**fields))
    try:
        yield _CURRENT_CONTEXT.get()
    finally:
        _CURRENT_CONTEXT.reset(token)


def current_llm_call_context() -> LlmCallContext:
    """Return the LlmCallContext active in the current thread/context.

    Returns:
        Active context, or a default "unknown" context when none was set.
    """
    return _CURRENT_CONTEXT.get()


def _now_utc() -> str:
    """Return the current UTC timestamp in ISO format with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _vcap_application() -> dict[str, Any]:
    """Parse Cloud Foundry app metadata from VCAP_APPLICATION.

    Returns:
        Parsed metadata, or an empty dict outside Cloud Foundry.
    """
    try:
        parsed = json.loads(os.getenv("VCAP_APPLICATION", "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _hash_user(user_id: Optional[str]) -> Optional[str]:
    """Return a salted, non-reversible hash for a user identity.

    Args:
        user_id: Raw user identity, or None.

    Returns:
        24-character hex hash, or None when no identity is given.
    """
    if not user_id:
        return None
    return hashlib.sha256(f"{os.getenv('LOG_USER_HASH_SALT', '')}:{user_id}".encode("utf-8")).hexdigest()[:24]


def _read_value(source: Any, key: str) -> Any:
    """Read a value from a mapping or an object attribute.

    Args:
        source: Mapping or object.
        key: Key or attribute name.

    Returns:
        The value, or None when absent.
    """
    return source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)


def extract_token_usage(source: Any) -> TokenUsage:
    """Normalize provider usage metadata into TokenUsage.

    Supports OpenAI Responses/chat completions, LangChain, Gemini and Bedrock
    field names, including prompt-cache read/write details.

    Args:
        source: Provider response object or its usage object.

    Returns:
        Normalized TokenUsage.
    """
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
    output_tokens = first(
        "output_tokens", "completion_tokens", "completion_token_count", "candidates_token_count", "candidatesTokenCount", "outputTokens"
    )
    cache_read_input_tokens = first(
        "cached_input_tokens", "cache_read_input_tokens", "cacheReadInputTokens", "cached_content_token_count", "cachedContentTokenCount"
    )
    if not cache_read_input_tokens:
        cache_read_input_tokens = first_from(input_details, "cached_tokens", "cache_read", "cacheRead", "cache_read_input_tokens")
    cache_write_input_tokens = first("cache_write_input_tokens", "cache_creation_input_tokens", "cacheWriteInputTokens")
    if not cache_write_input_tokens:
        cache_write_input_tokens = first_from(
            input_details, "cache_write", "cache_creation", "cache_write_tokens", "cache_write_input_tokens", "cache_creation_input_tokens"
        )
    if not cache_write_input_tokens:
        cache_details = _read_value(usage, "cacheDetails") or _read_value(usage, "cache_details")
        if isinstance(cache_details, list):
            cache_write_input_tokens = sum(first_from(item, "inputTokens", "input_tokens") for item in cache_details)

    # OpenAI and Gemini report cached tokens as a subset of the input count.
    input_includes_cache = any(
        (
            first("prompt_token_count", "promptTokenCount"),
            first("cached_content_token_count", "cachedContentTokenCount"),
            first_from(input_details, "cached_tokens", "cache_read", "cache_creation", "cache_write_tokens"),
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


def actor_type_for_user(user_id: Optional[str], explicit: Optional[str] = None) -> str:
    """Return a dashboard-friendly actor type.

    Args:
        user_id: Optional user identity.
        explicit: Caller-provided actor type.

    Returns:
        explicit when valid, otherwise human (identity known) or unknown.
    """
    if explicit in {"human", "system", "batch", "unknown"}:
        return explicit
    return "human" if user_id else "unknown"


def emit_llm_usage_event(
    *,
    model: str,
    usage: TokenUsage,
    outcome: str,
    latency_ms: Optional[int],
    llm_endpoint: str = "responses",
    provider: str = "sap-ai-core",
    context: Optional[LlmCallContext] = None,
) -> None:
    """Emit one structured LLM usage event as a compact JSON stdout line.

    Args:
        model: Model name sent to the provider.
        usage: Normalized token usage (zeros on error when unknown).
        outcome: "success" or "error".
        latency_ms: Wall-clock duration of the provider call.
        llm_endpoint: Provider API family, e.g. "responses".
        provider: Provider identifier, "sap-ai-core" for Gen AI Hub.
        context: Request metadata; defaults to the active llm_call_context.
    """
    ctx = context or current_llm_call_context()
    vcap = _vcap_application()
    event = {
        "schema_version": "btp.llm_usage.v1",
        "event_type": "llm_usage",
        "event_time": _now_utc(),
        "app_name": vcap.get("application_name") or vcap.get("name"),
        "space_name": vcap.get("space_name"),
        "org_name": vcap.get("organization_name") or vcap.get("org_name"),
        "route": ctx.route,
        "method": ctx.method,
        "user_hash": _hash_user(ctx.user_id),
        "actor_type": actor_type_for_user(ctx.user_id, ctx.actor_type),
        "client_host": None,
        "provider": provider,
        "model": model,
        "llm_endpoint": llm_endpoint,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_write_input_tokens": usage.cache_write_input_tokens,
        "input_total_tokens": usage.input_total_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "outcome": outcome,
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "correlation_id": ctx.correlation_id or str(uuid.uuid4()),
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
