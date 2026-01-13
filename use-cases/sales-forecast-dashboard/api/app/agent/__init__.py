"""
What-If Forecasting Agent Package.

A LangGraph-based ReAct agent for exploring what-if scenarios
in retail sales forecasting.

This package provides:
- 17 specialized tools for scenario management, predictions, and analysis
- SHAP-based explainability through Model A surrogate
- Support for B&M and WEB channels
- Forecasting horizons from 1-52 weeks

Quick Start:
    from app.agent import run_query

    result = run_query("What if brand awareness increases 10% in NYC?")
    print(result["final_response"])

Interactive Mode:
    python -m forecasting.agent.run --interactive

CLI Usage:
    python -m forecasting.agent.run "Your question here"
    python -m forecasting.agent.run --list-tools
    python -m forecasting.agent.run --info
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Forecasting Team"

# Core agent functions
from .agent import (
    SYSTEM_PROMPT,
    build_agent,
    build_agent_with_checkpointer,
    run_query,
    run_conversation,
    export_agent_graph,
    get_agent_info,
    run_query_stream,
)

# LLM utilities
from .common import make_llm, save_graph_mermaid_png

# Callbacks
from .callbacks import VerboseCallbackHandler

# Tools
from .tools import (
    ALL_TOOLS,
    TOOL_CATEGORIES,
    get_tools_by_category,
    get_tool_count,
    list_tool_names,
)

# State management
from .state import (
    AgentState,
    ScenarioData,
    PredictionResult,
    create_initial_state,
)


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Agent
    "SYSTEM_PROMPT",
    "build_agent",
    "build_agent_with_checkpointer",
    "run_query",
    "run_conversation",
    "export_agent_graph",
    "get_agent_info",
    "run_query_stream",
    # LLM
    "make_llm",
    "save_graph_mermaid_png",
    # Callbacks
    "VerboseCallbackHandler",
    # Tools
    "ALL_TOOLS",
    "TOOL_CATEGORIES",
    "get_tools_by_category",
    "get_tool_count",
    "list_tool_names",
    # State
    "AgentState",
    "ScenarioData",
    "PredictionResult",
    "create_initial_state",
]
