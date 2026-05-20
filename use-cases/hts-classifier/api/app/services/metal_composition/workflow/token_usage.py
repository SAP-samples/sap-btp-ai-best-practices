"""Request-scoped token usage aggregation for workflow model calls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable, Optional, Tuple

from app.models.metal_composition import TokenUsageEntry, TokenUsageSummary


@dataclass
class _TokenUsageAggregate:
    phase: str
    task: str
    model: str
    call_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    usage_available: bool = False


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_value(usage: Any, *names: str) -> Optional[int]:
    if usage is None:
        return None
    for name in names:
        value = None
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def normalize_token_usage(usage: Any) -> Dict[str, Optional[int]]:
    input_tokens = _usage_value(
        usage,
        "input_tokens",
        "prompt_tokens",
        "inputTokenCount",
        "inputTokens",
    )
    output_tokens = _usage_value(
        usage,
        "output_tokens",
        "completion_tokens",
        "outputTokenCount",
        "outputTokens",
    )
    total_tokens = _usage_value(
        usage,
        "total_tokens",
        "totalTokenCount",
        "totalTokens",
    )

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class TokenUsageRecorder:
    """Aggregate model usage by stable workflow task keys."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: Dict[Tuple[str, str, str], _TokenUsageAggregate] = {}

    def record(self, *, phase: str, task: str, model: str, usage: Any) -> None:
        normalized = normalize_token_usage(usage)
        usage_available = any(value is not None for value in normalized.values())
        key = (phase, task, model)
        with self._lock:
            aggregate = self._entries.get(key)
            if aggregate is None:
                aggregate = _TokenUsageAggregate(phase=phase, task=task, model=model)
                self._entries[key] = aggregate
            aggregate.call_count += 1
            aggregate.usage_available = aggregate.usage_available or usage_available
            if usage_available:
                aggregate.input_tokens = _sum_optional(
                    aggregate.input_tokens,
                    normalized["input_tokens"],
                )
                aggregate.output_tokens = _sum_optional(
                    aggregate.output_tokens,
                    normalized["output_tokens"],
                )
                aggregate.total_tokens = _sum_optional(
                    aggregate.total_tokens,
                    normalized["total_tokens"],
                )

    def build_summary(self) -> TokenUsageSummary:
        with self._lock:
            entries = list(self._entries.values())

        entries.sort(key=lambda item: (item.phase, item.task, item.model))
        payload_entries = [
            TokenUsageEntry(
                phase=item.phase,
                task=item.task,
                model=item.model,
                call_count=item.call_count,
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                total_tokens=item.total_tokens,
                usage_available=item.usage_available,
            )
            for item in entries
        ]
        return TokenUsageSummary(
            entries=payload_entries,
            input_tokens=_sum_many_optional(entry.input_tokens for entry in payload_entries),
            output_tokens=_sum_many_optional(entry.output_tokens for entry in payload_entries),
            total_tokens=_sum_many_optional(entry.total_tokens for entry in payload_entries),
            missing_usage_entry_count=sum(1 for entry in payload_entries if not entry.usage_available),
        )


def _sum_optional(current: Optional[int], incoming: Optional[int]) -> Optional[int]:
    if incoming is None:
        return current
    if current is None:
        return incoming
    return current + incoming


def _sum_many_optional(values: Iterable[Optional[int]]) -> Optional[int]:
    total = 0
    saw_value = False
    for value in values:
        if value is None:
            continue
        total += value
        saw_value = True
    return total if saw_value else None
