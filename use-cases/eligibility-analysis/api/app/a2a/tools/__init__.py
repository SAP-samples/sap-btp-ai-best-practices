"""Tool registry for the eligibility A2A agent."""
from __future__ import annotations

from typing import Any, List

from .custom import CUSTOM_TOOLS
from .eligibility_pattern_tools import ELIGIBILITY_PATTERN_TOOLS
from .optimizer_tools import OPTIMIZER_TOOLS


async def get_all_tools() -> List[Any]:
    """Return all available tools for the agent."""
    return list(CUSTOM_TOOLS) + list(OPTIMIZER_TOOLS) + list(ELIGIBILITY_PATTERN_TOOLS)
