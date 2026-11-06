"""Single entry point for LLM calls made by the document-request workflow.

Both LLM steps (email extraction and reply composition) go through
``create_response_text`` so that model invocation parameters and token-usage
logging stay identical for every call.
"""

from __future__ import annotations

import time
from importlib import import_module
from typing import Any

from ..observability.llm_usage_logging import TokenUsage, emit_llm_usage_event, extract_token_usage

# Reasoning effort for the Responses API. "low" keeps latency small while still
# letting the reasoning model think briefly before answering.
REASONING_EFFORT = "low"


def default_responses_client() -> Any:
    """Return the SAP Gen AI Hub native OpenAI Responses client.

    Imported lazily so unit tests can run without AI Core credentials.

    Returns:
        The gen_ai_hub ``responses`` client module object.
    """
    return import_module("gen_ai_hub.proxy.native.openai").responses


def create_response_text(responses_client: Any, *, model: str, instructions: str, user_input: str) -> str:
    """Call the Responses API, log token usage, and return the output text.

    Emits one ``btp.llm_usage.v1`` event on success and one on failure (then
    re-raises). No temperature or output-token limit is sent: reasoning models
    reject temperature, and an output cap can truncate the answer because
    reasoning tokens count toward it.

    Args:
        responses_client: Object exposing ``create(**kwargs)`` (Gen AI Hub client or a test fake).
        model: Deployed model name, e.g. "gpt-6-luna".
        instructions: System guidance for the model.
        user_input: User prompt text.

    Returns:
        Plain text returned by the model (``response.output_text``).
    """
    start_time = time.perf_counter()
    try:
        response = responses_client.create(
            model=model,
            instructions=instructions,
            input=user_input,
            reasoning={"effort": REASONING_EFFORT},
        )
    except Exception:
        emit_llm_usage_event(
            model=model,
            usage=TokenUsage(),
            outcome="error",
            latency_ms=int((time.perf_counter() - start_time) * 1000),
        )
        raise
    emit_llm_usage_event(
        model=model,
        usage=extract_token_usage(response),
        outcome="success",
        latency_ms=int((time.perf_counter() - start_time) * 1000),
    )
    text = getattr(response, "output_text", None)
    return text if isinstance(text, str) else str(text or "")
