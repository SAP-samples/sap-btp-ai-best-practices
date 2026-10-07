"""LangChain callback that logs token usage for every chat-model call.

``create_chat_model`` attaches one :class:`LLMUsageCallback` to each model, so
every call made through it (agent turns, structured-output formatting, context
summaries, dataset-setting interpretation) emits one ``btp.llm_usage.v1`` stdout
event, on success and on error, including cache read and cache write tokens.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .llm_usage_logging import emit_llm_usage_event, extract_token_usage


@dataclass(frozen=True)
class RequestUsageContext:
    """Who and what triggered the LLM calls made while serving one HTTP request.

    Attributes:
        route: Request path, e.g. ``/api/workspace/chat``.
        method: HTTP method.
        user_id: Raw user identity; hashed before it is logged.
        client_host: Browser, host or forwarded address of the caller.
        correlation_id: Optional ``x-correlation-id`` request header.
    """

    route: str = "internal"
    method: str = "INTERNAL"
    user_id: Optional[str] = None
    client_host: Optional[str] = None
    correlation_id: Optional[str] = None


# Set by the HTTP middleware; calls outside a request (workers, scripts) see the default.
request_usage_context: ContextVar[RequestUsageContext] = ContextVar(
    "request_usage_context", default=RequestUsageContext()
)


class LLMUsageCallback(BaseCallbackHandler):
    """Emit one LLM usage event per chat-model call.

    Args:
        provider: Provider label for dashboards (default ``sap-ai-core``).
        model: Model deployment name.
        llm_endpoint: Endpoint family, e.g. ``chat.completions``, ``responses``,
            ``generateContent`` or ``bedrock.converse``.
    """

    def __init__(self, model: str, llm_endpoint: str, provider: str = "sap-ai-core") -> None:
        """Store static call labels and the per-run start-time table."""
        self.model = model
        self.llm_endpoint = llm_endpoint
        self.provider = provider
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Remember when the model call started so latency can be reported."""
        self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Same as ``on_chat_model_start`` for non-chat LLM runs."""
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Extract usage from the final message and emit a success event."""
        message = None
        generations = getattr(response, "generations", None)
        if generations and generations[0]:
            message = getattr(generations[0][0], "message", None)
        # Prefer the AIMessage usage_metadata; fall back to the raw provider usage.
        usage = extract_token_usage(
            message if getattr(message, "usage_metadata", None) else (getattr(response, "llm_output", None) or message)
        )
        self._emit(run_id, "success", usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Emit an error event; the provider returned no usage, so counts are zero."""
        self._emit(run_id, "error", extract_token_usage(None))

    def _emit(self, run_id: UUID, outcome: str, usage: Any) -> None:
        """Write the event, attributing it to the current request context."""
        started = self._started.pop(run_id, None)
        ctx = request_usage_context.get()
        emit_llm_usage_event(
            route=ctx.route,
            method=ctx.method,
            user_id=ctx.user_id,
            client_host=ctx.client_host,
            provider=self.provider,
            model=self.model,
            llm_endpoint=self.llm_endpoint,
            input_tokens=usage.input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            input_total_tokens=usage.input_total_tokens,
            total_tokens=usage.total_tokens,
            output_tokens=usage.output_tokens,
            outcome=outcome,
            latency_ms=int((time.perf_counter() - started) * 1000) if started is not None else None,
            correlation_id=ctx.correlation_id,
        )
