"""
ContextVar binding for the active payment advice document in UC-02.

This module provides a thread-safe (and async-safe) way to pass the current
payment advice payload into tool functions without making payload an explicit
argument of every tool.  The agent runtime binds the document once per
invocation via ``set_current_document`` before calling any tool, and restores
the previous state with ``reset_current_document`` afterwards.

Typical usage in an agent run loop::

    token = set_current_document(payload)
    try:
        # invoke LangChain tools — they read via get_current_document()
        ...
    finally:
        reset_current_document(token)

The ``current_document`` ContextVar defaults to None.  Any tool that calls
``get_current_document()`` without a prior ``set_current_document`` will raise
a ``RuntimeError``, making the missing binding immediately obvious rather than
silently producing wrong results.
"""

from __future__ import annotations

import contextvars
from typing import Any

# ---------------------------------------------------------------------------
# ContextVar
# ---------------------------------------------------------------------------

current_document: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("uc02_current_document", default=None)
)
"""The active payment advice payload for the current execution context.

Type: ContextVar[dict | None].  Default is None (no document bound).
Use set_current_document / reset_current_document to manage lifetime.
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def set_current_document(payload: dict[str, Any]) -> contextvars.Token:
    """Bind a payment advice payload to the current execution context.

    Args:
        payload: The payment advice dict to expose to downstream tools.
                 Must contain a ``line_items`` key as expected by the
                 ``app.payment_advice.deductions`` helpers.

    Returns:
        A ``contextvars.Token`` that can be passed to ``reset_current_document``
        to restore the previous binding (important in nested or concurrent
        invocation scenarios).
    """
    return current_document.set(payload)


def get_current_document() -> dict[str, Any]:
    """Return the payload currently bound to this execution context.

    Raises:
        RuntimeError: If no document has been bound via ``set_current_document``
            for the current context.  This is a programming error — the caller
            must always bind before invoking tools.

    Returns:
        The payment advice dict bound by the most recent ``set_current_document``
        call in this context.
    """
    doc = current_document.get()
    if doc is None:
        raise RuntimeError("no document bound to the current context")
    return doc


def reset_current_document(token: contextvars.Token) -> None:
    """Restore the ContextVar to its previous state using the given token.

    Should be called in a ``finally`` block after ``set_current_document`` to
    avoid leaking the binding when running multiple documents in the same
    event-loop task or thread.

    Args:
        token: The token returned by the matching ``set_current_document`` call.

    Returns:
        None
    """
    current_document.reset(token)
