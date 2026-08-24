"""Normalize token usage from provider-specific SAP Gen AI Hub responses.

This stdlib-only helper supports OpenAI, Bedrock/Anthropic, Gemini, and
LangChain message metadata. It keeps prompt-side token volume separate from
uncached input so cache behavior can be compared without provider-specific
parsing in notebooks.

Examples:
    python normalize_usage.py
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


def to_plain(value: Any, depth: int = 0) -> Any:
    """Convert SDK objects into JSON-friendly builtins.

    Args:
        value: Provider, LangChain, dataclass, or primitive value.
        depth: Current recursion depth.

    Returns:
        A JSON-friendly representation.
    """

    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_plain(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(item, depth + 1) for item in value]
    if is_dataclass(value):
        return to_plain(asdict(value), depth + 1)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_plain(model_dump(mode="json"), depth + 1)
        except TypeError:
            return to_plain(model_dump(), depth + 1)
    if hasattr(value, "__dict__"):
        return {
            key: to_plain(item, depth + 1)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    return repr(value)


def pick(data: Any, *names: str) -> Any:
    """Return the first present field from a mapping or object.

    Args:
        data: Mapping or object to inspect.
        names: Candidate field names.

    Returns:
        The first found value, or ``None``.
    """

    for name in names:
        if isinstance(data, Mapping) and name in data:
            return data[name]
        if hasattr(data, name):
            return getattr(data, name)
    return None


def number(value: Any) -> int | None:
    """Coerce a token value to ``int`` when present.

    Args:
        value: Token value from an SDK response.

    Returns:
        Integer token count, or ``None``.
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def usage_payload(value: Any) -> Any:
    """Extract usage metadata from a raw response or LangChain message.

    Args:
        value: Response object, usage object, mapping, or LangChain message.

    Returns:
        The best available usage payload.
    """

    if hasattr(value, "usage_metadata") and getattr(value, "usage_metadata"):
        payload = to_plain(getattr(value, "usage_metadata"))
        if hasattr(value, "response_metadata"):
            metadata = getattr(value, "response_metadata") or {}
            metadata_usage = pick(metadata, "usage", "token_usage")
            if metadata_usage:
                merged = to_plain(metadata_usage)
                merged.update(payload)
                return merged
        return payload
    if hasattr(value, "usage") and getattr(value, "usage"):
        return getattr(value, "usage")
    if isinstance(value, Mapping) and "usage" in value:
        return value["usage"]
    if isinstance(value, Mapping) and "usage_metadata" in value:
        payload = to_plain(value["usage_metadata"]) or {}
        metadata = value.get("response_metadata") or {}
        metadata_usage = pick(metadata, "usage", "token_usage")
        if metadata_usage:
            merged = to_plain(metadata_usage)
            if isinstance(payload, Mapping):
                merged.update(payload)
            return merged
        return payload
    if hasattr(value, "response_metadata"):
        metadata = getattr(value, "response_metadata") or {}
        token_usage = pick(metadata, "token_usage", "usage")
        if token_usage:
            return token_usage
    if isinstance(value, Mapping) and "response_metadata" in value:
        metadata = value.get("response_metadata") or {}
        token_usage = pick(metadata, "token_usage", "usage")
        if token_usage:
            return token_usage
    return value


def includes_cached_input(usage: Any, input_details: Any) -> bool:
    """Return whether the provider input field includes cached input tokens.

    Args:
        usage: Plain usage payload.
        input_details: Provider input-token detail object.

    Returns:
        ``True`` when cached tokens must be subtracted from provider input.
    """

    if pick(usage, "prompt_token_count", "promptTokenCount") is not None:
        return True
    if pick(usage, "cached_content_token_count", "cachedContentTokenCount") is not None:
        return True
    if pick(input_details, "cached_tokens") is not None:
        return True
    return (
        pick(input_details, "cache_read") is not None
        or pick(input_details, "cache_creation") is not None
    )


def normalize_usage(source: str, value: Any) -> dict[str, Any]:
    """Normalize provider usage into comparable cache-aware fields.

    Args:
        source: Provider or interface name, included for traceability.
        value: Usage payload or response object.

    Returns:
        A dictionary containing preferred ``uncached_input_tokens`` (plus the
        compatibility alias ``input_tokens``), raw provider input, total
        prompt-side input, cache reads/writes, output, reasoning, totals, and
        raw usage.
    """

    usage = to_plain(usage_payload(value))
    input_details = pick(
        usage,
        "input_tokens_details",
        "input_token_details",
        "prompt_tokens_details",
        "prompt_token_details",
    ) or {}
    output_details = pick(
        usage,
        "output_tokens_details",
        "output_token_details",
        "completion_tokens_details",
        "completion_token_details",
    ) or {}

    provider_input_tokens = number(
        pick(
            usage,
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "inputTokens",
        )
    )
    output_tokens = number(
        pick(
            usage,
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
            "outputTokens",
        )
    )
    cache_read_input_tokens = number(
        pick(
            usage,
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
            "cachedContentTokenCount",
            "cacheReadInputTokens",
        )
    )
    if cache_read_input_tokens is None:
        cache_read_input_tokens = number(
            pick(input_details, "cached_tokens", "cache_read", "cache_read_input_tokens")
        )
    cache_write_input_tokens = number(
        pick(
            usage,
            "cache_write_input_tokens",
            "cache_creation_input_tokens",
            "cache_write_tokens",
            "cacheWriteInputTokens",
            "cacheWriteTokens",
        )
    )
    detail_cache_write = number(
        pick(
            input_details,
            "cache_write",
            "cache_creation",
            "cache_write_tokens",
            "cache_write_input_tokens",
            "cache_creation_input_tokens",
        )
    )
    ephemeral_cache_write = sum(
        number(pick(input_details, field)) or 0
        for field in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    # langchain-aws 1.6 reports Bedrock TTL-specific writes in ephemeral
    # fields and deliberately leaves the generic cache_creation field at 0.
    if ephemeral_cache_write:
        cache_write_input_tokens = ephemeral_cache_write
    elif cache_write_input_tokens is None:
        cache_write_input_tokens = detail_cache_write
    if cache_write_input_tokens is None:
        details = pick(usage, "cacheDetails", "cache_details")
        if isinstance(details, list):
            writes = [number(pick(item, "inputTokens", "input_tokens")) for item in details]
            found = [item for item in writes if item is not None]
            if found:
                cache_write_input_tokens = sum(found)

    reasoning_tokens = number(
        pick(usage, "reasoning_tokens", "thoughts_token_count", "thoughtsTokenCount")
    )
    if reasoning_tokens is None:
        reasoning_tokens = number(
            pick(output_details, "reasoning_tokens", "reasoning", "thinking_tokens")
        )
    total_tokens = number(
        pick(usage, "total_tokens", "total_token_count", "totalTokenCount", "totalTokens")
    )

    cache_read = cache_read_input_tokens or 0
    cache_write = cache_write_input_tokens or 0
    if provider_input_tokens is None:
        input_tokens = None
        input_total_tokens = cache_read + cache_write
    elif includes_cached_input(usage, input_details):
        input_tokens = max(provider_input_tokens - cache_read - cache_write, 0)
        input_total_tokens = provider_input_tokens
    else:
        input_tokens = provider_input_tokens
        input_total_tokens = provider_input_tokens + cache_read + cache_write
    if total_tokens is None:
        total_tokens = (input_total_tokens or 0) + (output_tokens or 0)

    return {
        "source": source,
        "uncached_input_tokens": input_tokens,
        "input_tokens": input_tokens,
        "input_total_tokens": input_total_tokens,
        "provider_input_tokens": provider_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "cache_creation_input_tokens": cache_write_input_tokens,
        "cached_input_tokens": cache_read_input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw_usage": usage,
    }


def _self_check() -> None:
    """Run assertion-based checks for OpenAI, Bedrock, and Gemini shapes."""

    openai = normalize_usage(
        "openai",
        {
            "input_tokens": 4059,
            "output_tokens": 43,
            "input_tokens_details": {"cached_tokens": 3840},
        },
    )
    assert openai["input_tokens"] == 219
    assert openai["uncached_input_tokens"] == 219
    assert openai["input_total_tokens"] == 4059
    assert openai["cache_read_input_tokens"] == 3840

    bedrock = normalize_usage(
        "bedrock",
        {
            "inputTokens": 51,
            "outputTokens": 442,
            "cacheReadInputTokens": 100,
            "cacheDetails": [{"inputTokens": 200}],
        },
    )
    assert bedrock["input_tokens"] == 51
    assert bedrock["input_total_tokens"] == 351
    assert bedrock["cache_write_input_tokens"] == 200

    bedrock_ttl = normalize_usage(
        "bedrock-langchain",
        {
            "input_tokens": 4592,
            "output_tokens": 74,
            "input_token_details": {
                "cache_creation": 0,
                "cache_read": 0,
                "ephemeral_5m_input_tokens": 4589,
            },
        },
    )
    assert bedrock_ttl["input_tokens"] == 3
    assert bedrock_ttl["input_total_tokens"] == 4592
    assert bedrock_ttl["cache_write_input_tokens"] == 4589

    luna = normalize_usage(
        "openai-responses",
        {
            "input_tokens": 2862,
            "output_tokens": 5,
            "input_token_details": {"cache_creation": 0, "cache_read": 2830},
        },
    )
    assert luna["provider_input_tokens"] == 2862
    assert luna["uncached_input_tokens"] == 32
    assert luna["cache_read_input_tokens"] == 2830

    gemini = normalize_usage(
        "gemini",
        {
            "prompt_token_count": 5000,
            "candidates_token_count": 80,
            "cached_content_token_count": 4200,
            "total_token_count": 5080,
        },
    )
    assert gemini["input_tokens"] == 800
    assert gemini["input_total_tokens"] == 5000
    assert gemini["cache_read_input_tokens"] == 4200

    print("normalize_usage self-check passed")


if __name__ == "__main__":
    _self_check()
