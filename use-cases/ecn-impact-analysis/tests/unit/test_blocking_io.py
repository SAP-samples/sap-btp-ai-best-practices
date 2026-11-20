"""Tests for cancellation-safe execution of synchronous HANA repository work."""

from __future__ import annotations

import asyncio
import threading

import pytest
from app.ecn.services.blocking_io import run_blocking


@pytest.mark.asyncio
async def test_cancellation_waits_for_the_blocking_transaction_to_finish() -> None:
    """Never abandon a repository thread that may still own a HANA transaction."""
    entered = threading.Event()
    release = threading.Event()

    def transaction() -> str:
        """Block like an in-flight synchronous database transaction."""
        entered.set()
        release.wait(timeout=5)
        return "committed"

    task = asyncio.create_task(run_blocking(transaction))
    await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_blocking_result_and_arguments_are_forwarded() -> None:
    """Return the synchronous result with positional and keyword arguments intact."""

    def calculate(first: int, second: int, *, multiplier: int) -> int:
        """Return a keyword-dependent result from the worker thread."""
        return (first + second) * multiplier

    assert await run_blocking(calculate, 2, 3, multiplier=4) == 20
