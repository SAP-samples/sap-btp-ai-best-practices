"""Cancellation-safe bridge from async services to synchronous HANA repositories."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import TypeVar

import anyio

T = TypeVar("T")


async def run_blocking(
    function: Callable[..., T], *args: object, **kwargs: object
) -> T:
    """Run synchronous HANA work without abandoning its thread on cancellation.

    Args:
        function: Synchronous repository or engine function to execute.
        *args: Positional arguments forwarded to ``function``.
        **kwargs: Keyword arguments forwarded to ``function``.

    Returns:
        The function's return value.

    Raises:
        asyncio.CancelledError: Only after the worker thread has finished, so an
            in-flight transaction cannot outlive application shutdown teardown.
        Exception: Any exception raised by the synchronous function.
    """
    worker = asyncio.create_task(
        anyio.to_thread.run_sync(
            functools.partial(function, *args, **kwargs),
            abandon_on_cancel=False,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            await worker
        except asyncio.CancelledError:
            pass
        raise
