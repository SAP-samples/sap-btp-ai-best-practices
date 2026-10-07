"""
LLM helper for the Payment Advice Extractor (UC-01).

Provides one JSON-returning entry point, ``complete_json``, over two SAP Gen AI Hub
model families, and logs token usage for every provider call so gpt-5.6-luna and
gemini-3.1-flash-lite can be compared in SAP Cloud Logging.

Model families (verified live against this AI Core resource group):
- ``gpt-5.6-luna`` — reasoning model via the OpenAI-chat proxy. Requires
  ``max_completion_tokens`` (not ``max_tokens``), default temperature, and supports
  ``response_format={"type": "json_object"}``.
- ``gemini-3.1-flash-lite`` — via the Gen AI Hub Google-native client
  (``generate_content`` with ``response_mime_type="application/json"``).

Both share the same JSON-parse retry, transient-error backoff, and usage logging.
The provider callables are injectable so unit tests never hit the network.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from app.observability.llm_usage_logging import emit_llm_usage_event, extract_token_usage

from .config import ALLOWED_MODELS, DEFAULT_MODEL

# Reasoning models spend completion tokens on hidden reasoning, so keep headroom.
DEFAULT_MAX_COMPLETION_TOKENS = 8000

# Transient-error backoff defaults (rate limits, 5xx). Delay grows exponentially,
# capped, because gpt-5.6-luna rate limits in this region can persist for a while.
DEFAULT_TRANSIENT_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 6.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0

# Model routing.
_OPENAI_CHAT_MODELS = {"gpt-5.6-luna"}
_GEMINI_MODELS = {"gemini-3.1-flash-lite"}


class LLMError(RuntimeError):
    """Raised when the model call fails or does not return valid JSON."""


def _endpoint_for(model: str) -> str:
    """Return the Cloud-Logging endpoint label for a model family."""
    return "generateContent" if model in _GEMINI_MODELS else "chat.completions"


def _is_transient(exc: Exception) -> bool:
    """Return whether an SDK error looks transient (retryable)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("rate_limit", "too_many_requests", "timeout", "temporarily", "503", "overloaded")
    )


# --------------------------------------------------------------------------- #
# Provider callables (lazy defaults; injectable for tests)
# --------------------------------------------------------------------------- #
def _default_openai_create() -> Callable[..., Any]:
    """Return the SAP Gen AI Hub OpenAI chat-completions callable."""
    from gen_ai_hub.proxy.native.openai import chat

    return chat.completions.create


# Cached Gemini client. The google-genai Client owns an httpx transport that is
# closed when the object is garbage-collected, so returning a bound method of a
# throwaway client fails with "client has been closed". Keep one alive per process.
_gemini_client = None


def _default_gemini_generate() -> Callable[..., Any]:
    """Return the SAP Gen AI Hub Google-native generate_content callable."""
    global _gemini_client
    if _gemini_client is None:
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.native.google_genai.clients import Client

        _gemini_client = Client(proxy_client=get_proxy_client("gen-ai-hub"))
    return _gemini_client.models.generate_content


def _call_openai(create: Callable[..., Any], messages: list[dict[str, str]], model: str, max_tokens: int):
    """Make one OpenAI-chat call; return (text, response)."""
    response = create(
        messages=messages,
        model=model,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content or ""), response


def _call_gemini(generate: Callable[..., Any], messages: list[dict[str, str]], model: str, max_tokens: int):
    """Make one Gemini generate_content call; return (text, response).

    System messages become the system instruction; the remaining turns are
    concatenated into the text contents. JSON output is requested via the response
    MIME type.
    """
    system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages if m["role"] != "system")
    config: dict[str, Any] = {"response_mime_type": "application/json", "max_output_tokens": max_tokens}
    if system_text:
        config["system_instruction"] = system_text
    response = generate(model=model, contents=convo, config=config)
    return (getattr(response, "text", "") or ""), response


def _invoke_once(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    openai_create: Callable[..., Any] | None,
    gemini_generate: Callable[..., Any] | None,
    route_label: str,
    correlation_id: str | None,
) -> str:
    """Make one provider call, emit a usage event (success or error), return text."""
    started = time.perf_counter()
    endpoint = _endpoint_for(model)
    try:
        if model in _GEMINI_MODELS:
            text, response = _call_gemini(gemini_generate, messages, model, max_tokens)
        else:
            text, response = _call_openai(openai_create, messages, model, max_tokens)
    except Exception:
        emit_llm_usage_event(
            route=route_label, method="INTERNAL", actor_type="batch",
            provider="sap-ai-core", model=model, llm_endpoint=endpoint,
            outcome="error", latency_ms=int((time.perf_counter() - started) * 1000),
            correlation_id=correlation_id,
        )
        raise

    usage = extract_token_usage(response)
    emit_llm_usage_event(
        route=route_label, method="INTERNAL", actor_type="batch",
        provider="sap-ai-core", model=model, llm_endpoint=endpoint,
        input_tokens=usage.input_tokens, cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens, outcome="success",
        latency_ms=int((time.perf_counter() - started) * 1000), correlation_id=correlation_id,
    )
    return text


def complete_json(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    openai_create: Callable[..., Any] | None = None,
    gemini_generate: Callable[..., Any] | None = None,
    retries: int = 1,
    transient_retries: int = DEFAULT_TRANSIENT_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_max: float = DEFAULT_MAX_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    route_label: str = "internal:llm",
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """
    Call the configured model and parse a JSON object from its reply.

    Transient API errors (rate limits, 5xx) are retried with capped exponential
    backoff; a non-JSON reply is retried by nudging the model. Every provider call
    emits a token-usage event to stdout (model, tokens, latency, outcome) for the
    gpt-vs-gemini comparison.

    Args:
        system, user: Prompts.
        model: Model name; must be in ALLOWED_MODELS.
        max_completion_tokens: Output/reasoning token budget.
        openai_create / gemini_generate: Injectable provider callables (tests).
        retries: Extra attempts on non-JSON output.
        transient_retries / backoff_base / backoff_max / sleep: Backoff controls.
        route_label: Call-site label for the usage event (e.g. "mapper:map").
        correlation_id: Optional correlation id for the usage event.

    Returns:
        The parsed JSON object.

    Raises:
        LLMError: Unsupported model, non-transient/exhausted API failure, or
            non-JSON output after all format retries.
    """
    if model not in ALLOWED_MODELS:
        raise LLMError(f"model {model!r} is not in ALLOWED_MODELS {ALLOWED_MODELS}")
    if model not in _OPENAI_CHAT_MODELS and model not in _GEMINI_MODELS:
        raise LLMError(f"model {model!r} has no wired provider path")

    if model in _GEMINI_MODELS:
        gemini_generate = gemini_generate or _default_gemini_generate()
    else:
        openai_create = openai_create or _default_openai_create()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    json_retries_left = retries
    transient_used = 0
    last_error: Exception | None = None
    while True:
        try:
            text = _invoke_once(
                model=model, messages=messages, max_tokens=max_completion_tokens,
                openai_create=openai_create, gemini_generate=gemini_generate,
                route_label=route_label, correlation_id=correlation_id,
            )
        except Exception as exc:
            last_error = exc
            if _is_transient(exc) and transient_used < transient_retries:
                sleep(min(backoff_base * (2 ** transient_used), backoff_max))
                transient_used += 1
                continue
            raise LLMError(f"LLM call failed: {exc}") from exc

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise json.JSONDecodeError("top-level JSON was not an object", text or "", 0)
            return parsed
        except json.JSONDecodeError as exc:
            last_error = exc
            if json_retries_left <= 0:
                raise LLMError(
                    f"model did not return a valid JSON object after {retries + 1} attempts: {exc}"
                ) from exc
            json_retries_left -= 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "That was not valid JSON. Return ONLY a valid JSON object."})
