"""
Visualization tools for the forecasting agent.

These tools generate plots and charts for scenario analysis:
- plot_scenario_comparison: Multi-line time series chart

Per Agent_plan.md Section 3.4: Visualization Tools

Note: plot_driver_waterfall and plot_sensitivity_heatmap were removed due to
using hardcoded/fake data. Use analyze_sensitivity and explain_forecast_change
for accurate model-based analysis.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.feature_mapping import FEATURE_CATEGORIES
from app.agent.tools.execution import _load_year_over_year_baseline


# Thread lock for matplotlib - prevents race conditions when multiple plots run in parallel
_PLOT_LOCK = threading.Lock()

# Constants
OUTPUT_DIR = Path(__file__).parent.parent / "output"

PLOT_STYLE = {
    "figure.figsize": (12, 8),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

SCENARIO_COLORS = {
    "baseline": "#2E86AB",  # Used for baseline_bm, baseline_web, or any scenario starting with "baseline"
    "default": ["#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#9B59B6", "#1ABC9C"],
}


def _is_baseline_scenario(name: str) -> bool:
    """Check if a scenario name is a baseline (baseline, baseline_bm, baseline_web)."""
    return name.startswith("baseline")


def _ensure_output_dir() -> Path:
    """Ensure output directory exists and return path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _generate_filename(plot_type: str, identifier: str = "") -> str:
    """Generate unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if identifier:
        return f"{plot_type}_{identifier}_{timestamp}.png"
    return f"{plot_type}_{timestamp}.png"


def _setup_plot_style():
    """Configure seaborn and matplotlib styles."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(PLOT_STYLE)


@tool
def plot_scenario_comparison(
    scenario_names: List[str],
    metric: str = "sales",
    include_uncertainty: bool = True,
    include_yoy_actuals: bool = False,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a multi-line time series chart comparing scenarios.

    Creates a line chart with horizon weeks on the X-axis and the specified
    metric on the Y-axis. Each scenario is plotted as a separate line.

    Per Agent_plan.md Section 3.4 - plot_scenario_comparison:
    - X-axis: Time (t0 -> t0+h)
    - Y-axis: Metric (Sales)
    - Series: Line for Baseline, Line for Scenario A, Line for Scenario B
    - Style: Clean lines, different colors per scenario

    Args:
        scenario_names: List of scenario names to compare. All must have predictions.
        metric: Metric to plot - "sales", "aov", "orders" (default: "sales").
        include_uncertainty: Show p10/p90 confidence bands for baseline (default: True).
        include_yoy_actuals: Include actual historical sales from year-over-year
                             baseline for comparison (default: False). When True,
                             plots 2024 actual sales alongside 2025 forecasts.
                             Requires baseline scenario to be initialized.
        title: Custom plot title (optional).

    Returns:
        Dictionary containing:
        - status: "plotted" on success
        - file_path: Path to saved PNG file
        - scenarios: List of scenarios plotted
        - includes_yoy_actuals: Whether YoY actuals were included
        - summary: Description of the comparison

    Example:
        >>> plot_scenario_comparison(["baseline", "high_awareness", "low_staffing"])
        {"status": "plotted", "file_path": "/path/to/plot.png", ...}

        >>> plot_scenario_comparison(["baseline"], include_yoy_actuals=True)
        {"status": "plotted", "includes_yoy_actuals": True, ...}
    """
    session = get_session()

    # Validate scenarios have predictions
    missing_predictions = []
    for name in scenario_names:
        if not session.has_prediction(name):
            missing_predictions.append(name)

    if missing_predictions:
        return {
            "error": f"Missing predictions for: {missing_predictions}. "
            "Run run_forecast_model first."
        }

    # Use lock to prevent race conditions with parallel plot calls
    with _PLOT_LOCK:
        _setup_plot_style()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Track data for each scenario
        scenario_data = {}
        color_idx = 0
        last_horizons = None

        for name in scenario_names:
            pred = session.get_prediction(name)
            df = pred.predictions_df

            # Aggregate by horizon
            metric_col = f"pred_{metric}_p50" if f"pred_{metric}_p50" in df.columns else f"pred_{metric}"
            if metric_col not in df.columns and metric == "sales":
                metric_col = "pred_sales_p50"

            if metric_col not in df.columns:
                continue

            horizon_agg = df.groupby("horizon").agg({
                metric_col: "sum",
            }).reset_index()

            horizons = horizon_agg["horizon"].values
            values = horizon_agg[metric_col].values
            last_horizons = horizons

            scenario_data[name] = {"horizons": horizons, "values": values}

            # Determine color - baselines (baseline_bm, baseline_web) get special color
            if _is_baseline_scenario(name):
                color = SCENARIO_COLORS["baseline"]
            else:
                color = SCENARIO_COLORS["default"][color_idx % len(SCENARIO_COLORS["default"])]
                color_idx += 1

            # Plot line
            ax.plot(horizons, values, marker='o', linewidth=2, color=color, label=name)

            # Add uncertainty band for baseline scenarios
            if include_uncertainty and _is_baseline_scenario(name):
                p10_col = f"pred_{metric}_p10"
                p90_col = f"pred_{metric}_p90"
                if p10_col in df.columns and p90_col in df.columns:
                    p10_agg = df.groupby("horizon")[p10_col].sum().values
                    p90_agg = df.groupby("horizon")[p90_col].sum().values
                    ax.fill_between(horizons, p10_agg, p90_agg, alpha=0.2, color=color)

        # Plot YoY actuals if requested (only for sales metric)
        yoy_plotted = False
        if include_yoy_actuals and metric == "sales":
            # Get baseline scenario for YoY alignment (use channel-specific baseline)
            channel = session.get_channel()
            baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web
            baseline_scenario = session.get_scenario(baseline_name)
            if baseline_scenario is not None:
                yoy_df, _ = _load_year_over_year_baseline(session, baseline_scenario.df)
                if yoy_df is not None and "total_sales" in yoy_df.columns and "horizon" in yoy_df.columns:
                    # Aggregate by horizon
                    yoy_agg = yoy_df.groupby("horizon")["total_sales"].sum().reset_index()
                    horizons_yoy = yoy_agg["horizon"].values
                    values_yoy = yoy_agg["total_sales"].values

                    # Plot with distinctive style (dashed line, different color)
                    ax.plot(horizons_yoy, values_yoy, marker='s', linewidth=2,
                           linestyle='--', color='#888888', label='2024 Actuals')
                    yoy_plotted = True

                    # Store for delta annotation
                    scenario_data["2024_actuals"] = {"horizons": horizons_yoy, "values": values_yoy}

        # Format plot
        ax.set_xlabel('Forecast Horizon (Weeks)')
        ylabel = f'{metric.upper()} ($)' if metric in ['sales', 'aov'] else metric.title()
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Scenario Comparison: {metric.title()}')

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add delta annotations - find a baseline scenario for comparison
        # Look for any baseline (baseline_bm, baseline_web, or legacy "baseline")
        baseline_key = None
        for key in scenario_data.keys():
            if _is_baseline_scenario(key):
                baseline_key = key
                break

        if baseline_key and len(scenario_data) > 1:
            baseline_total = sum(scenario_data[baseline_key]["values"])
            for name, data in scenario_data.items():
                if _is_baseline_scenario(name):
                    continue
                scenario_total = sum(data["values"])
                delta_pct = (scenario_total - baseline_total) / baseline_total * 100 if baseline_total > 0 else 0
                ax.annotate(
                    f'{delta_pct:+.1f}%',
                    xy=(data["horizons"][-1], data["values"][-1]),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                )

        # Save plot with explicit figure reference
        output_dir = _ensure_output_dir()
        filename = _generate_filename("scenario_comparison", f"{len(scenario_names)}_scenarios")
        filepath = output_dir / filename

        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    summary_parts = [f"Plotted {len(scenario_names)} scenarios"]
    if yoy_plotted:
        summary_parts.append("with 2024 actuals")
    summary_parts.append(f"across {len(last_horizons) if last_horizons is not None else 'N'} horizons.")

    plot_path = str(filepath.absolute())

    # Store plot path for report generation
    try:
        session.add_plot_file(plot_path)
    except Exception:
        pass  # Don't fail if storage fails

    return {
        "status": "plotted",
        "file_path": plot_path,
        "scenarios": scenario_names,
        "metric": metric,
        "includes_yoy_actuals": yoy_plotted,
        "summary": " ".join(summary_parts),
        "hint": "Use compare_scenarios for numerical delta analysis.",
    }


# Export all tools
__all__ = [
    "plot_scenario_comparison",
]
