"""Retry one timed-out LangChain model invocation on a configured fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

from app.ecn.llm.diagnostics import classify_provider_failure
from app.ecn.settings import LLM_TIMEOUT_FALLBACK_MODEL


class AsyncInvoker(Protocol):
    """Describe an async model or tool-bound runnable accepting one input."""

    async def ainvoke(self, input_value: object) -> object:
        """Invoke the runnable and return its response object."""


async def invoke_with_timeout_fallback(
    primary: AsyncInvoker,
    input_value: object,
    primary_model: str,
    fallback_factory: Callable[[], Awaitable[AsyncInvoker]],
    on_fallback: Callable[[str], None] | None = None,
) -> tuple[object, str]:
    """Invoke once and retry the same input on GPT-5.4 only after a timeout.

    Args:
        primary: Configured primary model, already bound to any required tools.
        input_value: Original LangChain message or input value to invoke.
        primary_model: Deployment name associated with ``primary``.
        fallback_factory: Lazy factory for the tool-bound fallback runnable.
        on_fallback: Optional callback notified immediately before fallback invocation.

    Returns:
        The provider response and the model name that returned it.

    Raises:
        Exception: Re-raises non-timeout primary or fallback failures.
        TimeoutError: If the fallback deployment also times out.
        asyncio.CancelledError: Propagates cancellation without starting a fallback.
    """
    try:
        return await primary.ainvoke(input_value), primary_model
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - provider errors are classified safely.
        failure = classify_provider_failure(exc)
        if (
            failure.error_code != "PROVIDER_TIMEOUT"
            or primary_model == LLM_TIMEOUT_FALLBACK_MODEL
        ):
            raise

    fallback = await fallback_factory()
    if on_fallback is not None:
        on_fallback(LLM_TIMEOUT_FALLBACK_MODEL)
    try:
        return (
            await fallback.ainvoke(input_value),
            LLM_TIMEOUT_FALLBACK_MODEL,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize fallback timeout for callers.
        if classify_provider_failure(exc).error_code == "PROVIDER_TIMEOUT":
            raise TimeoutError("fallback model timed out") from None
        raise
