"""
Comparison tools for the forecasting agent.

These tools handle scenario comparison, listing, and data export:
- compare_scenarios: Delta analysis between scenarios
- list_scenarios: List all scenarios in session
- export_scenario_data: Export model inputs and predictions to CSV

Per Agent_plan.md Section 3.3: Execution & Analysis Tools
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.tools.export import (
    _add_fiscal_fields,
    _add_region,
    EXPORT_COLUMNS,
    EXPORT_COLUMN_RENAME,
)


@tool
def compare_scenarios(
    scenario_a: str,
    scenario_b: str,
    metrics: Optional[List[str]] = None,
    by_store: bool = False,
) -> Dict[str, Any]:
    """
    Compare two scenarios and show detailed delta metrics.

    Computes absolute and percentage differences for sales, AOV, orders,
    and traffic (B&M only) between two scenarios.

    Args:
        scenario_a: First scenario name (typically "baseline").
        scenario_b: Second scenario name (the what-if scenario).
        metrics: Optional list of metrics to compare:
            - "sales": Total sales (default)
            - "aov": Average order value
            - "orders": Order count (derived from sales/AOV)
            - "traffic": Store traffic (B&M only)
            Default: ["sales", "aov"]
        by_store: If True, include per-store breakdown showing sales delta for each
            profit_center_nbr. Useful for cannibalization analysis. Default False.

    Returns:
        Dictionary containing:
        - status: "compared" on success
        - scenario_a / scenario_b: Summary of each scenario
        - delta: Absolute differences for each metric
        - delta_pct: Percentage differences for each metric
        - by_horizon: Comparison broken down by horizon week
        - by_store: (only if by_store=True) List of per-store comparisons with
          profit_center_nbr, sales_a, sales_b, delta, delta_pct, and status
          (common/new_in_b/removed_in_b)
        - summary: Human-readable summary

    Example:
        >>> compare_scenarios("baseline", "aggressive_marketing")
        {"status": "compared", "delta": {"sales": 125000}, "delta_pct": {"sales": 5.2}, ...}

        >>> compare_scenarios("baseline", "reopen_store", by_store=True)
        {"by_store": [{"profit_center_nbr": 234, "delta": -50000, "status": "common"}, ...]}
    """
    session = get_session()

    # Validate scenarios exist
    scenario_a_obj = session.get_scenario(scenario_a)
    scenario_b_obj = session.get_scenario(scenario_b)

    if scenario_a_obj is None:
        return {
            "error": f"Scenario '{scenario_a}' not found. "
            f"Available: {list(session.get_state()['scenarios'].keys())}"
        }

    if scenario_b_obj is None:
        return {
            "error": f"Scenario '{scenario_b}' not found. "
            f"Available: {list(session.get_state()['scenarios'].keys())}"
        }

    # Check predictions exist
    if not session.has_prediction(scenario_a):
        return {
            "error": f"No predictions for '{scenario_a}'. "
            "Run run_forecast_model first."
        }

    if not session.has_prediction(scenario_b):
        return {
            "error": f"No predictions for '{scenario_b}'. "
            "Run run_forecast_model first."
        }

    # Default metrics
    if metrics is None:
        metrics = ["sales", "aov"]

    # Get predictions
    pred_a = session.get_prediction(scenario_a)
    pred_b = session.get_prediction(scenario_b)

    df_a = pred_a.predictions_df
    df_b = pred_b.predictions_df

    # Compute aggregate metrics for each scenario
    metrics_a = _compute_aggregate_metrics(df_a, metrics)
    metrics_b = _compute_aggregate_metrics(df_b, metrics)

    # Compute deltas
    delta = {}
    delta_pct = {}

    for metric in metrics:
        key = f"total_{metric}" if metric in ["sales", "traffic", "orders"] else f"mean_{metric}"
        val_a = metrics_a.get(key, 0)
        val_b = metrics_b.get(key, 0)

        if val_a is not None and val_b is not None:
            delta[metric] = round(val_b - val_a, 2)
            if val_a != 0:
                delta_pct[metric] = round((val_b - val_a) / val_a * 100, 2)
            else:
                delta_pct[metric] = 0

    # Compute by-horizon breakdown
    by_horizon = _compute_by_horizon_comparison(df_a, df_b, metrics)

    # Compute per-store breakdown if requested
    by_store_breakdown = []
    if by_store:
        by_store_breakdown = _compute_by_store_comparison(df_a, df_b)

    # Generate summary
    summary = _generate_comparison_summary(scenario_a, scenario_b, delta, delta_pct)

    result = {
        "status": "compared",
        "scenario_a": {
            "name": scenario_a,
            "metrics": metrics_a,
        },
        "scenario_b": {
            "name": scenario_b,
            "metrics": metrics_b,
        },
        "delta": delta,
        "delta_pct": delta_pct,
        "by_horizon": by_horizon,
        "summary": summary,
        "hint": "Use plot_scenario_comparison to visualize this comparison.",
    }

    if by_store:
        result["by_store"] = by_store_breakdown

    # Store results for report generation
    try:
        session.store_scenario_comparison(json.dumps(result, default=str))
    except Exception:
        pass  # Don't fail if storage fails

    return result


def _compute_aggregate_metrics(df: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    """Compute aggregate metrics from predictions DataFrame."""
    result = {}

    if "sales" in metrics and "pred_sales_p50" in df.columns:
        result["total_sales"] = float(df["pred_sales_p50"].sum())
        result["mean_sales"] = float(df["pred_sales_p50"].mean())

    if "aov" in metrics and "pred_aov_p50" in df.columns:
        result["mean_aov"] = float(df["pred_aov_p50"].mean())

    if "orders" in metrics:
        if "pred_log_orders" in df.columns:
            import numpy as np
            result["total_orders"] = float(np.exp(df["pred_log_orders"]).sum())
        elif "pred_sales_p50" in df.columns and "pred_aov_p50" in df.columns:
            # Derive orders from sales / aov
            aov = df["pred_aov_p50"].replace(0, float("nan"))
            orders = df["pred_sales_p50"] / aov
            result["total_orders"] = float(orders.sum())

    if "traffic" in metrics and "pred_traffic_p50" in df.columns:
        result["total_traffic"] = float(df["pred_traffic_p50"].sum())

    result["num_stores"] = int(df["profit_center_nbr"].nunique()) if "profit_center_nbr" in df.columns else 0
    result["num_rows"] = len(df)

    return result


def _compute_by_horizon_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame, metrics: List[str]
) -> List[Dict[str, Any]]:
    """Compute comparison broken down by horizon."""
    by_horizon = []

    if "horizon" not in df_a.columns or "horizon" not in df_b.columns:
        return by_horizon

    horizons = sorted(set(df_a["horizon"].unique()) | set(df_b["horizon"].unique()))

    for h in horizons[:13]:  # Limit to first 13 horizons for brevity
        row_a = df_a[df_a["horizon"] == h]
        row_b = df_b[df_b["horizon"] == h]

        horizon_data = {"horizon": int(h)}

        if "sales" in metrics:
            sales_a = row_a["pred_sales_p50"].sum() if "pred_sales_p50" in row_a.columns else 0
            sales_b = row_b["pred_sales_p50"].sum() if "pred_sales_p50" in row_b.columns else 0
            horizon_data["sales_a"] = round(sales_a, 2)
            horizon_data["sales_b"] = round(sales_b, 2)
            horizon_data["sales_delta"] = round(sales_b - sales_a, 2)
            if sales_a > 0:
                horizon_data["sales_delta_pct"] = round((sales_b - sales_a) / sales_a * 100, 2)

        by_horizon.append(horizon_data)

    return by_horizon


def _compute_by_store_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Compute per-store comparison between two scenarios.

    Aggregates sales by profit_center_nbr and computes delta/delta_pct for each store.
    Handles stores that exist in only one scenario (new stores or removed stores).

    Args:
        df_a: Predictions DataFrame for scenario A (baseline)
        df_b: Predictions DataFrame for scenario B (what-if)

    Returns:
        List of per-store comparisons, each containing:
        - profit_center_nbr: Store ID
        - sales_a: Total sales in scenario A
        - sales_b: Total sales in scenario B
        - delta: Absolute difference (sales_b - sales_a)
        - delta_pct: Percentage difference
        - status: "common" (in both), "new_in_b" (new store), "removed_in_b" (closed)
    """
    by_store_breakdown = []

    if "profit_center_nbr" not in df_a.columns or "profit_center_nbr" not in df_b.columns:
        return by_store_breakdown

    if "pred_sales_p50" not in df_a.columns or "pred_sales_p50" not in df_b.columns:
        return by_store_breakdown

    # Aggregate by store (sum across horizons)
    stores_a = df_a.groupby("profit_center_nbr").agg({
        "pred_sales_p50": "sum"
    }).rename(columns={"pred_sales_p50": "sales"})

    stores_b = df_b.groupby("profit_center_nbr").agg({
        "pred_sales_p50": "sum"
    }).rename(columns={"pred_sales_p50": "sales"})

    # Get all store IDs from both scenarios
    all_stores = set(stores_a.index) | set(stores_b.index)

    for store_id in sorted(all_stores):
        sales_a = stores_a.loc[store_id, "sales"] if store_id in stores_a.index else 0
        sales_b = stores_b.loc[store_id, "sales"] if store_id in stores_b.index else 0

        # Determine store status
        if store_id in stores_a.index and store_id in stores_b.index:
            status = "common"
        elif store_id in stores_b.index:
            status = "new_in_b"
        else:
            status = "removed_in_b"

        delta = sales_b - sales_a
        delta_pct = (delta / sales_a * 100) if sales_a > 0 else (100.0 if sales_b > 0 else 0.0)

        by_store_breakdown.append({
            "profit_center_nbr": int(store_id),
            "sales_a": round(float(sales_a), 2),
            "sales_b": round(float(sales_b), 2),
            "delta": round(float(delta), 2),
            "delta_pct": round(float(delta_pct), 2),
            "status": status,
        })

    return by_store_breakdown


def _generate_comparison_summary(
    scenario_a: str, scenario_b: str, delta: Dict, delta_pct: Dict
) -> str:
    """Generate human-readable comparison summary."""
    parts = [f"Comparing '{scenario_b}' vs '{scenario_a}':"]

    if "sales" in delta:
        direction = "higher" if delta["sales"] > 0 else "lower"
        parts.append(
            f"Sales: ${abs(delta['sales']):,.0f} {direction} "
            f"({delta_pct.get('sales', 0):+.1f}%)"
        )

    if "aov" in delta:
        direction = "higher" if delta["aov"] > 0 else "lower"
        parts.append(
            f"AOV: ${abs(delta['aov']):,.2f} {direction} "
            f"({delta_pct.get('aov', 0):+.1f}%)"
        )

    if "traffic" in delta:
        direction = "higher" if delta["traffic"] > 0 else "lower"
        parts.append(
            f"Traffic: {abs(delta['traffic']):,.0f} {direction} "
            f"({delta_pct.get('traffic', 0):+.1f}%)"
        )

    return " ".join(parts)


@tool
def list_scenarios() -> Dict[str, Any]:
    """
    List all scenarios in the current session with summaries.

    Returns an overview of all scenarios including their names, parent scenarios,
    modification counts, and whether predictions have been generated.

    Returns:
        Dictionary containing:
        - status: "listed" on success
        - active_scenario: Name of the currently active scenario
        - num_scenarios: Total number of scenarios
        - scenarios: List of scenario summaries with:
            - name: Scenario name
            - parent: Parent scenario (if forked)
            - num_rows: Number of data rows
            - num_modifications: Count of modifications applied
            - has_predictions: Whether predictions exist

    Example:
        >>> list_scenarios()
        {"status": "listed", "active_scenario": "baseline", "scenarios": [...]}
    """
    session = get_session()

    scenarios_list = []
    scenarios_dict = session.get_state().get("scenarios", {})

    for name, scenario in scenarios_dict.items():
        summary = scenario.get_summary()
        summary["has_predictions"] = session.has_prediction(name)
        scenarios_list.append(summary)

    # Sort by creation time (baselines first - baseline, baseline_bm, baseline_web)
    scenarios_list.sort(key=lambda x: (not x["name"].startswith("baseline"), x.get("created_at", "")))

    return {
        "status": "listed",
        "active_scenario": session.get_active_scenario_name(),
        "num_scenarios": len(scenarios_list),
        "scenarios": scenarios_list,
        "hint": "Use set_active_scenario to switch scenarios, "
        "or create_scenario to fork a new one.",
    }


@tool
def export_scenario_data(
    scenario_names: Optional[List[str]] = None,
    include_inputs: bool = True,
    include_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Export scenario data (model inputs and/or predictions) to CSV files.

    Useful for debugging unexpected results or external analysis.
    Files are saved to forecasting/agent/output/ directory.

    Args:
        scenario_names: List of scenarios to export. Default: all scenarios.
        include_inputs: Export the input feature DataFrames. Default: True.
        include_predictions: Export prediction DataFrames. Default: True.

    Returns:
        Dictionary containing:
        - status: "exported" on success
        - files: List of exported file paths
        - scenarios_exported: List of scenario names exported
        - output_dir: Path to output directory

    Example:
        >>> export_scenario_data(["baseline", "high_awareness"])
        {"status": "exported", "files": [...], "scenarios_exported": [...]}
    """
    session = get_session()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get scenarios to export
    if scenario_names is None:
        scenario_names = list(session.get_state()["scenarios"].keys())

    if not scenario_names:
        return {
            "error": "No scenarios to export. Initialize simulation first."
        }

    exported_files = []
    scenarios_exported = []

    for name in scenario_names:
        scenario = session.get_scenario(name)
        if scenario is None:
            continue

        scenarios_exported.append(name)

        # Export inputs (feature DataFrame)
        if include_inputs and scenario.df is not None:
            input_path = output_dir / f"inputs_{name}_{timestamp}.csv"
            scenario.df.to_csv(input_path, index=False)
            exported_files.append(str(input_path.absolute()))

        # Export predictions
        if include_predictions and session.has_prediction(name):
            pred = session.get_prediction(name)
            if pred and pred.predictions_df is not None:
                pred_path = output_dir / f"predictions_{name}_{timestamp}.csv"

                # Export predictions with enriched columns (matching export_baseline_forecasts)
                df = pred.predictions_df.copy()

                # Derive orders from log_orders (leave empty if not available)
                if "pred_log_orders" in df.columns:
                    df["pred_orders"] = np.exp(df["pred_log_orders"])
                else:
                    df["pred_orders"] = np.nan

                # Derive conversion_rate from logit_conversion (leave empty if not available)
                if "pred_logit_conversion" in df.columns:
                    df["conversion_rate"] = 1 / (1 + np.exp(-df["pred_logit_conversion"]))
                else:
                    df["conversion_rate"] = np.nan

                # Ensure target_week_date is datetime for fiscal join
                if "target_week_date" in df.columns:
                    df["target_week_date"] = pd.to_datetime(df["target_week_date"])

                # Add fiscal fields via calendar join
                df = _add_fiscal_fields(df)

                # Add region via DMA map join
                df = _add_region(df)

                # Ensure all export columns exist (add missing ones with NaN)
                for col in EXPORT_COLUMNS:
                    if col not in df.columns:
                        df[col] = np.nan

                # Select export columns in correct order
                df = df[EXPORT_COLUMNS].copy()

                # Sort by profit_center_nbr, origin_week_date, and horizon for readability
                sort_cols = ["profit_center_nbr", "origin_week_date", "horizon"]
                df = df.sort_values(sort_cols).reset_index(drop=True)

                # Rename columns to user-friendly names
                df = df.rename(columns=EXPORT_COLUMN_RENAME)

                df.to_csv(pred_path, index=False)
                exported_files.append(str(pred_path.absolute()))

    if not exported_files:
        return {
            "error": "No data exported. Ensure scenarios exist and have data/predictions."
        }

    # Track exported files for chat attachments
    for file_path in exported_files:
        try:
            session.add_export_file(file_path)
        except Exception:
            pass  # Don't fail if tracking fails

    return {
        "status": "exported",
        "files": exported_files,
        "scenarios_exported": scenarios_exported,
        "output_dir": str(output_dir.absolute()),
        "hint": "Files can be opened in Excel or Python for detailed analysis.",
    }


# Export all tools
__all__ = [
    "compare_scenarios",
    "list_scenarios",
    "export_scenario_data",
]
