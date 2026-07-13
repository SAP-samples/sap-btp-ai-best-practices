from __future__ import annotations

from typing import Any

from .models import TokenUsageGroup, TokenUsageRecord, TokenUsageReport


def _read_int_field(value: Any, field_names: tuple[str, ...]) -> int | None:
    """Read the first integer-like field from an object or dictionary.

    Args:
        value: Provider response object, metadata object, or dictionary.
        field_names: Candidate field names in priority order.

    Returns:
        Integer field value, or None when none of the fields are present.
    """
    for field_name in field_names:
        raw_value = value.get(field_name) if isinstance(value, dict) else getattr(value, field_name, None)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_usage_source(response: Any) -> Any | None:
    """Find provider token-usage metadata on native or LangChain responses.

    Args:
        response: Native GenAI Hub/OpenAI-compatible response, LangChain message,
            or structured-output wrapper response.

    Returns:
        Token usage metadata object or dictionary, or None when unavailable.
    """
    if isinstance(response, dict) and "raw" in response:
        return _extract_usage_source(response.get("raw"))

    if isinstance(response, dict) and "usage" in response:
        return response.get("usage")

    if isinstance(response, dict) and "usage_metadata" in response:
        return response.get("usage_metadata")

    usage = getattr(response, "usage", None)
    if usage is not None:
        return usage

    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        return usage_metadata

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if token_usage is not None:
            return token_usage
        usage_metadata = response_metadata.get("usage_metadata")
        if usage_metadata is not None:
            return usage_metadata
        usage = response_metadata.get("usage")
        if usage is not None:
            return usage

    return None


def normalize_token_usage(response: Any) -> tuple[int, int, int, bool]:
    """Normalize token usage from native and LangChain model responses.

    Args:
        response: Native chat completion response, LangChain AIMessage, or
            structured-output wrapper response.

    Returns:
        Tuple of input tokens, output tokens, total tokens, and whether real
        usage metadata was available.
    """
    usage_source = _extract_usage_source(response)
    if usage_source is None:
        return 0, 0, 0, False

    input_tokens = _read_int_field(
        usage_source,
        ("input_tokens", "prompt_tokens", "prompt_token_count", "promptTokenCount", "inputTokens"),
    ) or 0
    output_tokens = _read_int_field(
        usage_source,
        (
            "output_tokens",
            "completion_tokens",
            "completion_token_count",
            "candidates_token_count",
            "candidatesTokenCount",
            "outputTokens",
        ),
    ) or 0
    total_tokens = _read_int_field(
        usage_source,
        ("total_tokens", "total_token_count", "totalTokenCount", "totalTokens"),
    ) or input_tokens + output_tokens
    if output_tokens <= 0 and total_tokens > input_tokens:
        output_tokens = total_tokens - input_tokens
    return input_tokens, output_tokens, total_tokens, True


class TokenUsageTracker:
    """Collect optional LLM token usage records for one API fetch run."""

    def __init__(self) -> None:
        """Initialize an empty per-run token usage tracker.

        Returns:
            None.
        """
        self.records: list[TokenUsageRecord] = []

    def record_response(
        self,
        stage: str,
        model_name: str,
        response: Any,
        gmail_message_id: str | None = None,
        extraction_id: str | None = None,
        item_index: int | None = None,
    ) -> None:
        """Record token usage for one model response.

        Args:
            stage: Logical application stage, for example extractor or completion.
            model_name: Model name configured for the stage.
            response: Provider response or LangChain message containing usage metadata.
            gmail_message_id: Gmail message associated with the model call.
            extraction_id: Persisted extraction id associated with completion calls.
            item_index: Extracted item index associated with completion calls.

        Returns:
            None.
        """
        input_tokens, output_tokens, total_tokens, usage_available = normalize_token_usage(response)
        self.records.append(
            TokenUsageRecord(
                stage=stage,
                model_name=model_name,
                gmail_message_id=gmail_message_id,
                extraction_id=extraction_id,
                item_index=item_index,
                call_index=len(self.records) + 1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                usage_available=usage_available,
            )
        )

    def report(self) -> TokenUsageReport:
        """Build a JSON-serializable token usage report.

        Returns:
            TokenUsageReport with raw records, overall totals, and grouped totals.
        """
        totals = TokenUsageGroup()
        by_stage: dict[str, TokenUsageGroup] = {}
        by_model: dict[str, TokenUsageGroup] = {}
        for record in self.records:
            _add_record_to_group(totals, record)
            _add_record_to_group(by_stage.setdefault(record.stage, TokenUsageGroup()), record)
            _add_record_to_group(by_model.setdefault(record.model_name, TokenUsageGroup()), record)
        return TokenUsageReport(
            records=list(self.records),
            totals=totals,
            by_stage=by_stage,
            by_model=by_model,
        )


def _add_record_to_group(group: TokenUsageGroup, record: TokenUsageRecord) -> None:
    """Add one token usage record into an aggregate group.

    Args:
        group: Mutable aggregate updated in place.
        record: Token usage record to aggregate.

    Returns:
        None.
    """
    group.calls += 1
    group.input_tokens += record.input_tokens
    group.output_tokens += record.output_tokens
    group.total_tokens += record.total_tokens
