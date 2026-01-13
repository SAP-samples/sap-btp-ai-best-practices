"""
Forecasting Agent Tools Package.

This module exports the 29 core tools for the what-if forecasting agent,
organized following Agent_plan.md specification:

Context & Setup (3 tools):
- lookup_store_metadata: Find stores by criteria
- initialize_forecast_simulation: Set origin date, create baseline
- get_session_state: Return current session state

Scenario Modification (4 tools):
- create_scenario: Fork existing scenario
- modify_business_lever: Update feature columns
- simulate_new_store_opening: Add new store with network effects
- set_active_scenario: Switch active scenario

Execution & Analysis (4 tools):
- run_forecast_model: Generate Model B predictions
- explain_forecast_change: SHAP attribution via Model A
- analyze_sensitivity: Compute lever elasticities
- fetch_previous_year_actuals: Fetch actual sales from the same period last year

Comparison & Export (3 tools):
- compare_scenarios: Delta analysis between scenarios
- list_scenarios: List all scenarios in session
- export_scenario_data: Export model inputs and predictions to CSV

Visualization (1 tool):
- plot_scenario_comparison: Multi-line time series chart

Utility (2 tools):
- get_feature_info: Describe feature bounds and category
- validate_scenario: Check scenario for errors

Query (1 tool):
- get_feature_values: Query current feature values from a scenario

Export (2 tools):
- generate_report: Compile analysis into PDF report with charts and recommendations
- export_baseline_forecasts: Export precomputed baseline forecasts to CSV

Budget (4 tools):
- estimate_budget_for_awareness: Estimate budget needed for target awareness
- estimate_awareness_from_budget: Estimate awareness change from budget investment
- refit_budget_awareness_model: Refit the budget-awareness model with latest data
- get_budget_awareness_info: Get available markets and model quality metrics

Diagnostics (5 tools):
- get_business_health_overview: High-level business health snapshot
- get_dma_diagnostic: Drill down into specific DMAs
- get_underperforming_stores: Find stores below performance thresholds
- get_store_diagnostic: Deep dive with SHAP driver analysis
- get_performance_ranking: Rank stores/DMAs by metrics

Note: plot_driver_waterfall and plot_sensitivity_heatmap were removed due to
using hardcoded/fake data. Use analyze_sensitivity and explain_forecast_change
for accurate model-based analysis.
"""

from __future__ import annotations

# Context & Setup (3 tools)
from .context import (
    lookup_store_metadata,
    initialize_forecast_simulation,
    get_session_state,
)

# Scenario Modification (4 tools)
from .scenario import (
    create_scenario,
    modify_business_lever,
    simulate_new_store_opening,
    set_active_scenario,
)

# Execution & Analysis (4 tools)
from .execution import (
    run_forecast_model,
    explain_forecast_change,
    analyze_sensitivity,
    fetch_previous_year_actuals,
)

# Comparison & Export (3 tools)
from .comparison import (
    compare_scenarios,
    list_scenarios,
    export_scenario_data,
)

# Visualization (1 tool)
from .visualization import (
    plot_scenario_comparison,
)

# Utility (2 tools)
from .validation import (
    get_feature_info,
    validate_scenario,
)

# Query (1 tool)
from .query import (
    get_feature_values,
)

# Export (2 tools)
from .report import (
    generate_report,
)

# Baseline Export
from .export import (
    export_baseline_forecasts,
)

# Budget (4 tools)
from .budget import (
    estimate_budget_for_awareness,
    estimate_awareness_from_budget,
    refit_budget_awareness_model,
    get_budget_awareness_info,
)

# Diagnostics (5 tools)
from .diagnostics import (
    get_business_health_overview,
    get_dma_diagnostic,
    get_underperforming_stores,
    get_store_diagnostic,
    get_performance_ranking,
)


# Master list of all 29 tools for the agent
ALL_TOOLS = [
    # Context & Setup (3)
    lookup_store_metadata,
    initialize_forecast_simulation,
    get_session_state,
    # Scenario Modification (4)
    create_scenario,
    modify_business_lever,
    simulate_new_store_opening,
    set_active_scenario,
    # Execution & Analysis (4)
    run_forecast_model,
    explain_forecast_change,
    analyze_sensitivity,
    fetch_previous_year_actuals,
    # Comparison (3)
    compare_scenarios,
    list_scenarios,
    export_scenario_data,
    # Visualization (1)
    plot_scenario_comparison,
    # Utility (2)
    get_feature_info,
    validate_scenario,
    # Query (1)
    get_feature_values,
    # Export (2)
    generate_report,
    export_baseline_forecasts,
    # Budget (4)
    estimate_budget_for_awareness,
    estimate_awareness_from_budget,
    refit_budget_awareness_model,
    get_budget_awareness_info,
    # Diagnostics (5)
    get_business_health_overview,
    get_dma_diagnostic,
    get_underperforming_stores,
    get_store_diagnostic,
    get_performance_ranking,
]


# Tool categories for documentation and selective loading
TOOL_CATEGORIES = {
    "context_setup": [
        lookup_store_metadata,
        initialize_forecast_simulation,
        get_session_state,
    ],
    "scenario_modification": [
        create_scenario,
        modify_business_lever,
        simulate_new_store_opening,
        set_active_scenario,
    ],
    "execution_analysis": [
        run_forecast_model,
        explain_forecast_change,
        analyze_sensitivity,
        fetch_previous_year_actuals,
    ],
    "comparison": [
        compare_scenarios,
        list_scenarios,
        export_scenario_data,
    ],
    "visualization": [
        plot_scenario_comparison,
    ],
    "utility": [
        get_feature_info,
        validate_scenario,
    ],
    "query": [
        get_feature_values,
    ],
    "export": [
        generate_report,
        export_baseline_forecasts,
    ],
    "budget": [
        estimate_budget_for_awareness,
        estimate_awareness_from_budget,
        refit_budget_awareness_model,
        get_budget_awareness_info,
    ],
    "diagnostics": [
        get_business_health_overview,
        get_dma_diagnostic,
        get_underperforming_stores,
        get_store_diagnostic,
        get_performance_ranking,
    ],
}


def get_tools_by_category(category: str):
    """Get tools for a specific category."""
    return TOOL_CATEGORIES.get(category, [])


def get_tool_count():
    """Get total number of tools."""
    return len(ALL_TOOLS)


def list_tool_names():
    """Get list of all tool names."""
    return [tool.name for tool in ALL_TOOLS]


__all__ = [
    # Master list
    "ALL_TOOLS",
    "TOOL_CATEGORIES",
    # Utility functions
    "get_tools_by_category",
    "get_tool_count",
    "list_tool_names",
    # Context & Setup
    "lookup_store_metadata",
    "initialize_forecast_simulation",
    "get_session_state",
    # Scenario Modification
    "create_scenario",
    "modify_business_lever",
    "simulate_new_store_opening",
    "set_active_scenario",
    # Execution & Analysis
    "run_forecast_model",
    "explain_forecast_change",
    "analyze_sensitivity",
    "fetch_previous_year_actuals",
    # Comparison & Export
    "compare_scenarios",
    "list_scenarios",
    "export_scenario_data",
    # Visualization
    "plot_scenario_comparison",
    # Utility
    "get_feature_info",
    "validate_scenario",
    # Query
    "get_feature_values",
    # Export
    "generate_report",
    "export_baseline_forecasts",
    # Budget
    "estimate_budget_for_awareness",
    "estimate_awareness_from_budget",
    "refit_budget_awareness_model",
    "get_budget_awareness_info",
    # Diagnostics
    "get_business_health_overview",
    "get_dma_diagnostic",
    "get_underperforming_stores",
    "get_store_diagnostic",
    "get_performance_ranking",
]
