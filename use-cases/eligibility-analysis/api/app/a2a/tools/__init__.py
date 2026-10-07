"""Tool registry for the Invoice Workspace A2A agent (read-only workspace tools only)."""
from __future__ import annotations

from typing import Any, List

from .workspace_tools import WORKSPACE_TOOLS


async def get_all_tools() -> List[Any]:
    """Return the tools bound to the assistant model."""
    return list(WORKSPACE_TOOLS)
