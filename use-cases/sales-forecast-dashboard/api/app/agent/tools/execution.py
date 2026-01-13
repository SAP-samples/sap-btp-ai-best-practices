"""
Execution and analysis tools for the forecasting agent.

These tools handle running predictions and generating insights:
- run_forecast_model: Generate Model B predictions
- explain_forecast_change: SHAP attribution via Model A (surrogate)
- analyze_sensitivity: Compute lever elasticities

Per Agent_plan.md Section 3.3: Execution & Analysis Tools

Memory Optimization Notes:
- SHAP computation uses batched method to manage memory
- Garbage collection added after heavy computations
- Per-horizon SHAP can be disabled via COMPUTE_PER_HORIZON_SHAP env var
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.state import PredictionResult
from app.agent.feature_mapping import (
    resolve_feature_name,
    get_feature_metadata,
    get_modifiable_features,
    get_features_by_category,
    FEATURE_CATEGORIES,
)
from app.agent.hana_loader import load_model_b_filtered

# Treat features as unchanged unless delta exceeds this tolerance
FEATURE_CHANGE_TOLERANCE = 1e-5


def _load_year_over_year_baseline(
    session,
    scenario_df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Build a year-over-year baseline DataFrame by aligning each scenario row to the
    same store/week 52 weeks earlier from historical model_b data.

    Returns a tuple of (baseline_df, warning_message).
    """
    # Extract store IDs for server-side filtering
    stores = scenario_df["profit_center_nbr"].unique().tolist() if "profit_center_nbr" in scenario_df.columns else None
    channel = session.get_channel()

    try:
        # Use server-side filtering for better performance
        historical_df = load_model_b_filtered(
            profit_center_nbrs=stores,
            channel=channel,
        )
    except Exception as e:
        return None, f"Failed to load model_b from HANA: {e}"

    if "target_week_date" not in historical_df.columns:
        return None, "Historical data missing target_week_date for YearOverYear comparison."

    # Ensure datetime alignment
    historical_df["target_week_date"] = pd.to_datetime(historical_df["target_week_date"])
    if "origin_week_date" in historical_df.columns:
        historical_df["origin_week_date"] = pd.to_datetime(historical_df["origin_week_date"])

    scenario_df = scenario_df.copy()
    if "target_week_date" not in scenario_df.columns:
        return None, "Scenario data missing target_week_date for YearOverYear comparison."

    scenario_df["target_week_date"] = pd.to_datetime(scenario_df["target_week_date"])
    scenario_df["yoy_target_week_date"] = scenario_df["target_week_date"] - timedelta(weeks=52)

    # Rename target_week_date for joining
    historical_df = historical_df.rename(columns={"target_week_date": "yoy_target_week_date"})

    # Left join to preserve scenario ordering - MUST include horizon for 1:1 matching
    join_keys = ["profit_center_nbr", "yoy_target_week_date", "horizon"]
    if "channel" in scenario_df.columns and "channel" in historical_df.columns:
        join_keys.append("channel")

    # Build yoy_df with all join keys present in scenario
    cols_to_copy = ["profit_center_nbr", "yoy_target_week_date"]
    if "horizon" in scenario_df.columns:
        cols_to_copy.append("horizon")
    if "channel" in scenario_df.columns:
        cols_to_copy.append("channel")
    yoy_df = scenario_df[[c for c in cols_to_copy if c in scenario_df.columns]].copy()

    yoy_df = yoy_df.merge(
        historical_df,
        on=[k for k in join_keys if k in historical_df.columns and k in yoy_df.columns],
        how="left",
        sort=False,
    )

    # Validate row count matches - critical for SHAP alignment
    expected_rows = len(scenario_df)
    actual_rows = len(yoy_df)
    if actual_rows != expected_rows:
        return None, (
            f"YoY baseline row count mismatch: expected {expected_rows}, got {actual_rows}. "
            f"Check horizon alignment in historical data."
        )

    # Restore baseline target_week_date column name
    if "yoy_target_week_date" in yoy_df.columns:
        yoy_df = yoy_df.rename(columns={"yoy_target_week_date": "target_week_date"})

    # Check for missing data after merge
    missing = yoy_df.isna().any(axis=1).sum()
    warning = None
    if missing > 0:
        warning = f"YearOverYear baseline has {missing} rows with missing data; results may be partial."

    return yoy_df, warning


def _detect_changed_features(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature_list: List[str],
    tol: float = FEATURE_CHANGE_TOLERANCE,
) -> List[str]:
    """
    Identify features whose values differ between scenario and baseline by more than tol.

    Note: Uses .values to avoid pandas index alignment issues when dataframes
    have different indices. Converts to float to handle boolean columns.
    """
    changed = []
    for f in feature_list:
        if f not in scenario_df.columns or f not in baseline_df.columns:
            continue
        # Use .values to avoid index alignment - both dataframes should have same row count
        # Convert to float to handle boolean columns (np.subtract doesn't work on bool)
        scenario_vals = scenario_df[f].values.astype(float)
        baseline_vals = baseline_df[f].values.astype(float)
        if len(scenario_vals) != len(baseline_vals):
            continue
        diff = np.abs(scenario_vals - baseline_vals)
        # Handle NaN by using nanmax
        max_diff = np.nanmax(diff) if not np.all(np.isnan(diff)) else 0
        if max_diff > tol:
            changed.append(f)
    return changed


def _validate_sign_consistency(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    feature: str,
    shap_impact: float,
    threshold: float = 0.01,
) -> Optional[str]:
    """
    Check if SHAP impact direction matches feature change direction.

    For YoY comparisons, if a feature increased but SHAP shows negative impact
    (or vice versa), this indicates a potential issue with the attribution.

    Args:
        scenario_df: Current scenario feature values.
        baseline_df: Baseline (e.g., YoY) feature values.
        feature: Feature name to check.
        shap_impact: SHAP delta impact value for this feature.
        threshold: Minimum absolute SHAP impact to consider (default 0.01).

    Returns:
        Warning string if inconsistent, None if consistent or below threshold.
    """
    if feature not in scenario_df.columns or feature not in baseline_df.columns:
        return None

    # Skip small impacts - not meaningful enough to validate
    if abs(shap_impact) < threshold:
        return None

    # Use .values and nanmean to handle index alignment and NaN
    scenario_mean = np.nanmean(scenario_df[feature].values)
    baseline_mean = np.nanmean(baseline_df[feature].values)
    delta_feature = scenario_mean - baseline_mean

    # Skip if feature didn't change meaningfully
    if abs(delta_feature) < FEATURE_CHANGE_TOLERANCE:
        return None

    # Check for sign inconsistency
    if delta_feature > 0 and shap_impact < -threshold:
        return (
            f"{feature}: increased ({baseline_mean:.2f} -> {scenario_mean:.2f}) "
            f"but shows negative SHAP impact ({shap_impact:+.4f})"
        )
    if delta_feature < 0 and shap_impact > threshold:
        return (
            f"{feature}: decreased ({baseline_mean:.2f} -> {scenario_mean:.2f}) "
            f"but shows positive SHAP impact ({shap_impact:+.4f})"
        )
    return None


def _compute_feature_deltas(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    features: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute feature value deltas between scenario and baseline.

    Args:
        scenario_df: Current scenario feature values.
        baseline_df: Baseline feature values.
        features: List of feature names to compute deltas for.

    Returns:
        Dictionary mapping feature name to {value_current, value_baseline, delta}.
    """
    deltas = {}
    for f in features:
        if f not in scenario_df.columns or f not in baseline_df.columns:
            continue
        # Use .values and nanmean to handle index alignment and NaN
        scenario_val = np.nanmean(scenario_df[f].values)
        baseline_val = np.nanmean(baseline_df[f].values)
        deltas[f] = {
            "value_current": round(float(scenario_val), 4),
            "value_baseline": round(float(baseline_val), 4),
            "delta": round(float(scenario_val - baseline_val), 4),
        }
    return deltas


def _run_predictions_for_df(
    session,
    df: pd.DataFrame,
    label: str,
) -> Optional[PredictionResult]:
    """
    Run Model B inference for an arbitrary DataFrame to obtain predictions.
    """
    from app.regressor.pipelines import InferencePipeline
    from app.regressor.configs import InferenceConfig

    checkpoint_dir = session.get_checkpoint_dir()
    channel = session.get_channel()

    if not checkpoint_dir.exists():
        return None

    config = InferenceConfig(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir.parent / "infer",
        channels=[channel],
        run_explainability=False,
    )

    try:
        pipeline = InferencePipeline(config)
    except Exception:
        return None

    df_in = df.copy()
    if "channel" not in df_in.columns:
        df_in["channel"] = channel

    try:
        inference_result = pipeline.run(df_in, channels=[channel])
    except Exception:
        return None
    pred_df = inference_result.bm_predictions if channel == "B&M" else inference_result.web_predictions

    if pred_df is None or pred_df.empty:
        return None

    return PredictionResult(
        scenario_name=label,
        predictions_df=pred_df,
        shap_df=None,
        generated_at=datetime.now().isoformat(),
        metadata={"channel": channel, "checkpoint_dir": str(checkpoint_dir)},
    )


def _build_weekly_change_table(
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    scenario_pred: Optional[PredictionResult],
    baseline_pred: Optional[PredictionResult],
    changed_features: List[str],
) -> Optional[pd.DataFrame]:
    """
    Build a week-by-week table with feature deltas and sales deltas.
    """
    if baseline_df is None or scenario_df is None:
        return None

    key_cols = [c for c in ["profit_center_nbr", "horizon"] if c in scenario_df.columns and c in baseline_df.columns]
    if not key_cols:
        return None

    scenario_cols = key_cols + changed_features
    baseline_cols = key_cols + changed_features

    if "target_week_date" in scenario_df.columns:
        scenario_cols.append("target_week_date")
    if "target_week_date" in baseline_df.columns:
        baseline_cols.append("target_week_date")

    sc_feat = scenario_df[scenario_cols].copy()
    base_feat = baseline_df[baseline_cols].copy()

    sc_feat = sc_feat.rename(columns={"target_week_date": "target_week_date_scenario"})
    base_feat = base_feat.rename(columns={"target_week_date": "target_week_date_baseline"})
    base_feat = base_feat.rename(columns={f: f"{f}_baseline" for f in changed_features})

    merged = sc_feat.merge(base_feat, on=key_cols, how="left", sort=False)

    # Attach predictions if available
    if scenario_pred and baseline_pred:
        pred_cols = ["pred_sales_p50"]
        sc_pred = scenario_pred.predictions_df[key_cols + pred_cols].copy()
        base_pred = baseline_pred.predictions_df[key_cols + pred_cols].copy()
        base_pred = base_pred.rename(columns={"pred_sales_p50": "pred_sales_p50_baseline"})

        merged = merged.merge(sc_pred, on=key_cols, how="left", sort=False)
        merged = merged.merge(base_pred, on=key_cols, how="left", sort=False)
        if "pred_sales_p50" in merged.columns and "pred_sales_p50_baseline" in merged.columns:
            merged["delta_sales_p50"] = merged["pred_sales_p50"] - merged["pred_sales_p50_baseline"]

    # Feature deltas
    for f in changed_features:
        base_col = f"{f}_baseline"
        if f in merged.columns and base_col in merged.columns:
            merged[f"{f}_delta"] = merged[f] - merged[base_col]

    # ISO strings for dates
    for col in ["target_week_date_scenario", "target_week_date_baseline"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col]).dt.strftime("%Y-%m-%d")

    return merged
# Treat features as "unchanged" unless they differ by more than this tolerance
FEATURE_CHANGE_TOLERANCE = 1e-5


@tool
def run_forecast_model(
    scenario_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate Model B predictions for scenarios.

    Runs the CatBoost forecasting model (Model B) to produce sales, AOV, orders,
    and traffic predictions with uncertainty quantiles.

    Per Agent_plan.md Section 3.3 - run_forecast_model:
    - Uses Model B for "Hard Numbers" (Sales, AOV, Conversion, Orders)
    - Returns p50/p90 quantiles for uncertainty
    - If B&M channel, also estimates traffic via Monte Carlo

    Args:
        scenario_names: List of scenario names to predict. Default: active scenario only.
                       Use ["baseline", "aggressive_marketing"] to compare multiple.

    Returns:
        Dictionary containing:
        - status: "predicted" on success
        - scenarios_predicted: List of scenarios that were predicted
        - results: Per-scenario prediction summaries including:
            - total_sales_p50, total_sales_p90: Aggregate sales
            - mean_aov_p50: Average AOV
            - weekly_breakdown: List of per-store, per-week predictions with:
                - profit_center_nbr: Store ID
                - horizon: Week number (1-13)
                - target_week_date: Date string (YYYY-MM-DD)
                - sales_p50, sales_p90: Weekly sales
                - aov_p50: Weekly AOV
                - traffic_p50: Weekly traffic (B&M only)
        - comparison: Delta metrics vs baseline (if baseline was predicted)

    Example:
        >>> run_forecast_model(scenario_names=["baseline", "high_awareness"])
        {"status": "predicted", "scenarios_predicted": ["baseline", "high_awareness"], ...}
    """
    session = get_session()

    # Validate session is initialized
    if session.get_origin_date() is None:
        return {
            "error": "Session not initialized. Use initialize_forecast_simulation first."
        }

    # Default to active scenario if not specified
    if scenario_names is None:
        scenario_names = [session.get_active_scenario_name()]

    # Validate all scenarios exist
    missing = []
    for name in scenario_names:
        if session.get_scenario(name) is None:
            missing.append(name)

    if missing:
        available = list(session.get_state()["scenarios"].keys())
        return {
            "error": f"Scenarios not found: {missing}. Available: {available}"
        }

    # Get channel and checkpoint directory
    channel = session.get_channel()
    checkpoint_dir = session.get_checkpoint_dir()

    # Validate checkpoints exist
    if not checkpoint_dir.exists():
        return {
            "error": f"Checkpoint directory not found: {checkpoint_dir}. "
            "Ensure model checkpoints are available."
        }

    required_files = {
        "B&M": ["bm_multi.cbm", "bm_conversion.cbm", "residual_stats.json"],
        "WEB": ["web_multi.cbm", "residual_stats.json"],
    }

    missing_files = []
    for f in required_files.get(channel, []):
        if not (checkpoint_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        return {
            "error": f"Missing checkpoint files in {checkpoint_dir}: {missing_files}"
        }

    # Run predictions for each scenario
    results = {}
    all_predictions = {}

    try:
        # Import and configure inference pipeline
        from app.regressor.pipelines import InferencePipeline
        from app.regressor.configs import InferenceConfig

        config = InferenceConfig(
            checkpoint_dir=checkpoint_dir,
            output_dir=checkpoint_dir.parent / "infer",
            channels=[channel],
            run_explainability=False,  # Handle separately
        )

        pipeline = InferencePipeline(config)

        for scenario_name in scenario_names:
            scenario = session.get_scenario(scenario_name)

            # Validate scenario channel matches session channel
            if scenario.channel and scenario.channel != channel:
                results[scenario_name] = {
                    "error": f"Scenario '{scenario_name}' was created for {scenario.channel} channel, "
                             f"but current session is set to {channel}.",
                    "hint": "Use initialize_forecast_simulation to create a new baseline for this channel, "
                           "or use set_channel to switch channels."
                }
                continue

            df = scenario.df

            # Ensure required columns
            if "channel" not in df.columns:
                df["channel"] = channel

            # Run inference
            inference_result = pipeline.run(df, channels=[channel])

            # Extract predictions based on channel
            if channel == "B&M":
                pred_df = inference_result.bm_predictions
            else:
                pred_df = inference_result.web_predictions

            if pred_df is None or pred_df.empty:
                results[scenario_name] = {"error": "No predictions generated"}
                continue

            # Compute aggregated metrics
            agg_metrics = {
                "total_sales_p50": float(pred_df["pred_sales_p50"].sum()) if "pred_sales_p50" in pred_df.columns else None,
                "total_sales_p90": float(pred_df["pred_sales_p90"].sum()) if "pred_sales_p90" in pred_df.columns else None,
                "mean_aov_p50": float(pred_df["pred_aov_p50"].mean()) if "pred_aov_p50" in pred_df.columns else None,
                "num_stores": int(pred_df["profit_center_nbr"].nunique()) if "profit_center_nbr" in pred_df.columns else 0,
                "num_rows": len(pred_df),
            }

            # Add traffic for B&M
            if channel == "B&M" and "pred_traffic_p50" in pred_df.columns:
                agg_metrics["total_traffic_p50"] = float(pred_df["pred_traffic_p50"].sum())
                agg_metrics["total_traffic_p90"] = float(pred_df["pred_traffic_p90"].sum())

            # Build weekly breakdown for this scenario
            weekly_breakdown = []
            sort_cols = ["horizon"]
            if "target_week_date" in pred_df.columns:
                sort_cols = ["target_week_date", "horizon"]

            sorted_pred = pred_df.sort_values(sort_cols)
            for _, row in sorted_pred.iterrows():
                week_entry = {
                    "horizon": int(row["horizon"]) if "horizon" in row else None,
                }
                # Include store identifier for per-store analysis
                if "profit_center_nbr" in row:
                    week_entry["profit_center_nbr"] = int(row["profit_center_nbr"])
                if "target_week_date" in row:
                    week_entry["target_week_date"] = pd.to_datetime(row["target_week_date"]).strftime("%Y-%m-%d")
                if "pred_sales_p50" in row:
                    week_entry["sales_p50"] = round(float(row["pred_sales_p50"]), 2)
                if "pred_sales_p90" in row:
                    week_entry["sales_p90"] = round(float(row["pred_sales_p90"]), 2)
                if "pred_aov_p50" in row:
                    week_entry["aov_p50"] = round(float(row["pred_aov_p50"]), 2)
                if channel == "B&M" and "pred_traffic_p50" in row:
                    week_entry["traffic_p50"] = round(float(row["pred_traffic_p50"]), 2)
                weekly_breakdown.append(week_entry)

            agg_metrics["weekly_breakdown"] = weekly_breakdown

            results[scenario_name] = agg_metrics

            # Cache prediction result
            pred_result = PredictionResult(
                scenario_name=scenario_name,
                predictions_df=pred_df,
                shap_df=None,
                generated_at=datetime.now().isoformat(),
                metadata={
                    "channel": channel,
                    "checkpoint_dir": str(checkpoint_dir),
                },
            )
            session.add_prediction(pred_result)
            all_predictions[scenario_name] = pred_df

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Compute comparison vs baseline if available
    # Look for channel-specific baseline names (baseline_bm or baseline_web)
    comparison = None
    baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web
    if baseline_name in results and len(results) > 1:
        baseline_sales = results[baseline_name].get("total_sales_p50", 0)
        if baseline_sales and baseline_sales > 0:
            comparison = {}
            for name, metrics in results.items():
                if name == baseline_name:
                    continue
                scenario_sales = metrics.get("total_sales_p50", 0)
                if scenario_sales:
                    delta = scenario_sales - baseline_sales
                    delta_pct = (delta / baseline_sales) * 100
                    comparison[name] = {
                        "delta_sales": round(delta, 2),
                        "delta_sales_pct": round(delta_pct, 2),
                    }

    result_dict = {
        "status": "predicted",
        "channel": channel,
        "scenarios_predicted": scenario_names,
        "results": results,
        "comparison_vs_baseline": comparison,
        "hint": "Use compare_scenarios for detailed delta analysis, "
        "or explain_forecast_change for SHAP attribution.",
    }

    # Store results for report generation
    try:
        session.store_forecast_results(json.dumps(result_dict, default=str))
    except Exception:
        pass  # Don't fail if storage fails

    return result_dict


@tool
def explain_forecast_change(
    scenario_name: str,
    baseline_reference: str = "baseline",
    horizon: Optional[int] = None,
    min_delta_threshold: float = 0.005,
) -> Dict[str, Any]:
    """
    Explain WHY a forecast changed using Model A (Surrogate) SHAP analysis.

    Computes SHAP values for both baseline and scenario, then calculates
    the row-by-row delta to identify which features drove the forecast change.
    Features with identical values in both scenarios (like seasonality for the
    same week) will have zero delta and be filtered out.

    Per Agent_plan.md Section 3.3 - explain_forecast_change:
    1. Define Baseline:
       - "baseline": Compare scenario vs baseline (counterfactual)
       - "YearOverYear": Compare t vs t-52 (removes seasonality effects)
    2. Run Model A: Compute SHAP values for both feature sets
    3. Compute Delta: Row-by-row SHAP delta, then average
    4. Group & Sum: Aggregate delta by category (Financing, Staffing, etc.)
    5. Generate Narrative: "Net Impact: +5% Sales. Drivers: +8% from Financing..."

    Args:
        scenario_name: Scenario to explain (must have predictions).
        baseline_reference: Reference for comparison:
            - "baseline": Compare to baseline scenario
            - "YearOverYear": Compare to same period last year (t-52)
        horizon: Optional specific week to explain (1-52). If None, explains all weeks
                 and includes per-horizon breakdown in response.
        min_delta_threshold: Minimum absolute delta to include a feature (default 0.005).
                            Features with smaller deltas are filtered as noise.

    Returns:
        Dictionary containing:
        - status: "explained" on success
        - horizon_filter: The specific horizon explained (None for aggregate)
        - net_impact: Overall sales change vs baseline
        - top_positive_drivers: Features increasing sales
        - top_negative_drivers: Features decreasing sales
        - by_category: Impact grouped by category (Financing, Staffing, etc.)
        - by_horizon: Per-week breakdown with top drivers for each horizon
        - features_filtered_below_threshold: Count of features filtered as noise
        - narrative: Human-readable explanation

    Example:
        >>> explain_forecast_change("aggressive_marketing", baseline_reference="baseline")
        {"status": "explained", "net_impact": {"delta_sales_pct": 5.2}, "by_horizon": [...], ...}
    """
    session = get_session()
    yoy_warning = None
    warnings = []

    # Validate scenario exists and has predictions
    scenario = session.get_scenario(scenario_name)
    if scenario is None:
        return {
            "error": f"Scenario '{scenario_name}' not found. "
            "Create it first with create_scenario."
        }

    # Extract modified features from scenario audit trail
    modified_features = []
    if scenario.modifications:
        modified_features = list(set(
            m.get("feature", "") for m in scenario.modifications if m.get("feature")
        ))

    # Require scenario predictions
    if not session.has_prediction(scenario_name):
        return {
            "error": f"No predictions for '{scenario_name}'. "
            "Run run_forecast_model first."
        }

    scenario_pred = session.get_prediction(scenario_name)

    # Validate baseline reference
    baseline_df = None
    if baseline_reference == "baseline":
        # Use channel-specific baseline name
        channel = session.get_channel()
        baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web

        if not session.has_prediction(baseline_name):
            return {
                "error": f"No predictions for '{baseline_name}'. "
                "Run run_forecast_model with baseline first."
            }
        baseline_scenario = session.get_scenario(baseline_name)
        if baseline_scenario is None:
            return {"error": f"Baseline scenario '{baseline_name}' not found for comparison."}
        baseline_df = baseline_scenario.df.copy()
        baseline_pred = session.get_prediction(baseline_name)
    elif baseline_reference == "YearOverYear":
        # Build YOY baseline from historical data
        baseline_df, yoy_warning = _load_year_over_year_baseline(session, scenario.df)
        if baseline_df is None or baseline_df.empty:
            return {
                "error": "Could not build YearOverYear baseline for comparison. "
                "Ensure historical model_b.csv is available and contains matching weeks."
            }
        baseline_pred = _run_predictions_for_df(session, baseline_df, "YearOverYear")
    else:
        return {
            "error": f"Unknown baseline_reference: '{baseline_reference}'. "
            "Use 'baseline' or 'YearOverYear'."
        }

    if yoy_warning:
        warnings.append(yoy_warning)

    scenario_df = scenario.df.copy()

    # Try to load surrogate model for SHAP
    checkpoint_dir = session.get_checkpoint_dir()
    channel = session.get_channel()

    try:
        from app.regressor.models import SurrogateExplainer
        from app.regressor.features.model_views import get_model_a_features_for_channel

        # Load surrogate
        surrogate_file = f"surrogate_{'bm' if channel == 'B&M' else 'web'}.cbm"
        surrogate_path = checkpoint_dir / surrogate_file
        meta_path = checkpoint_dir / f"{surrogate_file[:-4]}.meta.json"

        if not surrogate_path.exists():
            # Fall back to feature-based explanation (no SHAP)
            return _explain_without_shap(scenario, baseline_pred, scenario_pred)

        surrogate = SurrogateExplainer(channel=channel)
        surrogate.load_model(surrogate_path, meta_path=meta_path)

        # Get Model A features
        model_a_features = get_model_a_features_for_channel(channel)

        # Prepare data for SHAP
        # Filter to specific horizon if requested
        if horizon is not None:
            if "horizon" not in scenario_df.columns:
                return {"error": "Scenario data missing 'horizon' column for per-week analysis."}
            scenario_df = scenario_df[scenario_df["horizon"] == horizon].copy()
            if baseline_df is not None:
                baseline_df = baseline_df[baseline_df["horizon"] == horizon].copy()
            if scenario_df.empty:
                return {"error": f"No data for horizon {horizon} in scenario '{scenario_name}'."}

        # Filter to Model A features that exist in the data
        available_features = [f for f in model_a_features if f in scenario_df.columns]

        if len(available_features) < 5:
            return _explain_without_shap(scenario, baseline_pred, scenario_pred)

        changed_features = []
        if baseline_df is not None:
            changed_features = _detect_changed_features(
                scenario_df, baseline_df, available_features, tol=FEATURE_CHANGE_TOLERANCE
            )

        # For YoY comparison, ONLY show features that actually changed between years
        # (don't fall back to modified_features - those are for what-if scenarios)
        if baseline_reference == "YearOverYear":
            feature_filter = changed_features  # Strict: only features that changed
        else:
            feature_filter = changed_features or modified_features

        # Compute SHAP values with row-aligned delta
        feature_delta = {}
        features_before_filter = 0
        by_horizon_breakdown = []

        if baseline_df is not None and baseline_reference in ("baseline", "YearOverYear"):
            # Ensure row alignment by sorting on key columns
            key_cols = ["profit_center_nbr", "horizon"]
            available_keys = [k for k in key_cols if k in scenario_df.columns and k in baseline_df.columns]

            if available_keys:
                # Sort to ensure same order for row-by-row delta
                scenario_aligned = scenario_df.sort_values(available_keys).reset_index(drop=True)
                baseline_aligned = baseline_df.sort_values(available_keys).reset_index(drop=True)

                # Compute SHAP for aligned data using batched method for memory efficiency
                scenario_shap = surrogate.compute_shap_values_batched(scenario_aligned[available_features])
                baseline_shap = surrogate.compute_shap_values_batched(baseline_aligned[available_features])

                if scenario_shap is None or baseline_shap is None:
                    return _explain_without_shap(scenario, baseline_pred, scenario_pred)

                # Row-by-row delta, then mean (this ensures identical features cancel out)
                delta_shap = (scenario_shap - baseline_shap).mean(axis=0)
                feature_delta = dict(zip(available_features, delta_shap))

                # Free memory from large SHAP arrays
                del scenario_shap, baseline_shap
                gc.collect()

                # Compute per-horizon breakdown if enabled and not filtering to single horizon
                # Can be disabled via COMPUTE_PER_HORIZON_SHAP=false for memory savings
                from app.regressor.models.surrogate import COMPUTE_PER_HORIZON_SHAP
                if COMPUTE_PER_HORIZON_SHAP and horizon is None and "horizon" in scenario_aligned.columns:
                    horizons = sorted(scenario_aligned["horizon"].unique())
                    for h in horizons:
                        h_mask_s = scenario_aligned["horizon"] == h
                        h_mask_b = baseline_aligned["horizon"] == h

                        if h_mask_s.sum() == 0 or h_mask_b.sum() == 0:
                            continue

                        h_scenario_feat = scenario_aligned.loc[h_mask_s, available_features]
                        h_baseline_feat = baseline_aligned.loc[h_mask_b, available_features]

                        h_shap_s = surrogate.compute_shap_values(h_scenario_feat)
                        h_shap_b = surrogate.compute_shap_values(h_baseline_feat)

                        if h_shap_s is not None and h_shap_b is not None:
                            h_delta = (h_shap_s - h_shap_b).mean(axis=0)
                            h_drivers = {f: v for f, v in zip(available_features, h_delta)
                                        if abs(v) >= min_delta_threshold}

                            # Filter to only changed features if available (removes noise)
                            if feature_filter:
                                h_drivers = {f: v for f, v in h_drivers.items() if f in feature_filter}

                            # Get actual feature values for this horizon
                            h_scenario_data = scenario_aligned.loc[h_mask_s]
                            h_baseline_data = baseline_aligned.loc[h_mask_b]

                            # Build drivers with feature values included
                            def build_driver_entry(feature: str, shap_val: float) -> Dict[str, Any]:
                                driver = {
                                    "feature": feature,
                                    "shap_impact": round(shap_val, 4),
                                }
                                if feature in h_scenario_data.columns and feature in h_baseline_data.columns:
                                    val_current = h_scenario_data[feature].mean()
                                    val_baseline = h_baseline_data[feature].mean()
                                    driver["value_current"] = round(val_current, 4)
                                    driver["value_baseline"] = round(val_baseline, 4)
                                    driver["value_delta"] = round(val_current - val_baseline, 4)
                                return driver

                            # Only add horizon if there are drivers to show
                            top_pos = sorted([(f, v) for f, v in h_drivers.items() if v > 0],
                                           key=lambda x: -x[1])[:3]
                            top_neg = sorted([(f, v) for f, v in h_drivers.items() if v < 0],
                                           key=lambda x: x[1])[:3]

                            entry = {
                                "horizon": int(h),
                                "top_positive": [build_driver_entry(f, v) for f, v in top_pos],
                                "top_negative": [build_driver_entry(f, v) for f, v in top_neg],
                            }

                            # Attach dates for week-by-week traceability
                            if "target_week_date" in scenario_aligned.columns:
                                dates = scenario_aligned.loc[h_mask_s, "target_week_date"]
                                if not dates.empty:
                                    entry["target_week_date"] = pd.to_datetime(dates.iloc[0]).strftime("%Y-%m-%d")
                            if "target_week_date" in baseline_aligned.columns:
                                dates_b = baseline_aligned.loc[h_mask_b, "target_week_date"]
                                if not dates_b.empty:
                                    entry["baseline_target_week_date"] = pd.to_datetime(dates_b.iloc[0]).strftime("%Y-%m-%d")

                            # Always attach scenario (2025) predictions if available
                            if scenario_pred:
                                if (
                                    "pred_sales_p50" in scenario_pred.predictions_df.columns
                                    and "horizon" in scenario_pred.predictions_df.columns
                                ):
                                    sales_s = scenario_pred.predictions_df[
                                        scenario_pred.predictions_df["horizon"] == h
                                    ]["pred_sales_p50"].sum()
                                    entry["sales_p50"] = round(sales_s, 2)

                            # Attach baseline predictions if available (for what-if comparisons)
                            if baseline_pred:
                                if (
                                    "pred_sales_p50" in baseline_pred.predictions_df.columns
                                    and "horizon" in baseline_pred.predictions_df.columns
                                ):
                                    sales_b = baseline_pred.predictions_df[
                                        baseline_pred.predictions_df["horizon"] == h
                                    ]["pred_sales_p50"].sum()
                                    entry["sales_p50_baseline"] = round(sales_b, 2)
                                    if "sales_p50" in entry:
                                        entry["delta_sales_p50"] = round(entry["sales_p50"] - sales_b, 2)

                            # For YoY, include actual historical sales (not just model predictions)
                            if baseline_reference == "YearOverYear" and "total_sales" in h_baseline_data.columns:
                                actual_sales = h_baseline_data["total_sales"].sum()
                                entry["actual_sales_2024"] = round(float(actual_sales), 2)
                                # Compute delta vs actual (more meaningful for YoY)
                                if "sales_p50" in entry:
                                    entry["delta_vs_actual_2024"] = round(entry["sales_p50"] - actual_sales, 2)

                            by_horizon_breakdown.append(entry)
            else:
                # Fallback: compute SHAP without alignment (less accurate)
                scenario_shap = surrogate.compute_shap_values_batched(scenario_df[available_features])
                baseline_shap = surrogate.compute_shap_values_batched(baseline_df[available_features])
                if scenario_shap is not None and baseline_shap is not None:
                    delta_shap = scenario_shap.mean(axis=0) - baseline_shap.mean(axis=0)
                    feature_delta = dict(zip(available_features, delta_shap))
                    # Free memory
                    del scenario_shap, baseline_shap
                    gc.collect()
                else:
                    return _explain_without_shap(scenario, baseline_pred, scenario_pred)
        else:
            # No baseline comparison - use absolute SHAP importance
            scenario_shap = surrogate.compute_shap_values_batched(scenario_df[available_features])
            if scenario_shap is None:
                return _explain_without_shap(scenario, baseline_pred, scenario_pred)
            mean_shap = np.abs(scenario_shap).mean(axis=0)
            feature_delta = dict(zip(available_features, mean_shap))
            # Free memory
            del scenario_shap
            gc.collect()

        # Apply threshold filtering to remove noise
        if feature_filter:
            feature_delta = {f: v for f, v in feature_delta.items() if f in feature_filter}

        features_before_filter = len(feature_delta)
        feature_delta = {f: v for f, v in feature_delta.items() if abs(v) >= min_delta_threshold}
        features_filtered = features_before_filter - len(feature_delta)

        # Sort by absolute impact
        sorted_features = sorted(
            feature_delta.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Separate positive and negative drivers
        positive_drivers = [(f, v) for f, v in sorted_features if v > 0][:5]
        negative_drivers = [(f, v) for f, v in sorted_features if v < 0][:5]

        # Validate sign consistency for top drivers (YoY comparison especially)
        sign_warnings = []
        if baseline_df is not None:
            all_top_drivers = positive_drivers + negative_drivers
            for f, v in all_top_drivers:
                warning = _validate_sign_consistency(scenario_df, baseline_df, f, v)
                if warning:
                    sign_warnings.append(warning)

        # Add sign warnings to main warnings list
        if sign_warnings:
            warnings.extend(sign_warnings)

        # Group by category (use unfiltered delta for category sums)
        by_category = {}
        for category, features in FEATURE_CATEGORIES.items():
            cat_impact = sum(feature_delta.get(f, 0) for f in features if f in feature_delta)
            by_category[category] = round(cat_impact, 4)

        # Compute net impact from predictions
        net_impact = {}
        if baseline_pred and scenario_pred:
            baseline_sales = baseline_pred.predictions_df["pred_sales_p50"].sum()
            scenario_sales = scenario_pred.predictions_df["pred_sales_p50"].sum()
            if baseline_sales > 0:
                delta_sales = scenario_sales - baseline_sales
                delta_pct = (delta_sales / baseline_sales) * 100
                net_impact = {
                    "baseline_sales": round(baseline_sales, 2),
                    "scenario_sales": round(scenario_sales, 2),
                    "delta_sales": round(delta_sales, 2),
                    "delta_sales_pct": round(delta_pct, 2),
                }

        # For YoY comparisons, add actual historical sales totals from by_horizon_breakdown
        if baseline_reference == "YearOverYear" and by_horizon_breakdown:
            actual_2024_total = sum(
                h_entry.get("actual_sales_2024", 0) or 0
                for h_entry in by_horizon_breakdown
            )
            predicted_2025_total = sum(
                h_entry.get("sales_p50", 0) or 0
                for h_entry in by_horizon_breakdown
            )
            if actual_2024_total > 0:
                net_impact["actual_sales_2024"] = round(actual_2024_total, 2)
                net_impact["predicted_sales_2025"] = round(predicted_2025_total, 2)
                net_impact["delta_vs_actual_2024"] = round(predicted_2025_total - actual_2024_total, 2)
                net_impact["delta_vs_actual_2024_pct"] = round(
                    ((predicted_2025_total - actual_2024_total) / actual_2024_total) * 100, 2
                )

        # Build week-by-week change table (features + sales deltas)
        weekly_change_records = []
        weekly_change_path = None
        if baseline_df is not None:
            weekly_features = changed_features or modified_features
            weekly_change_df = _build_weekly_change_table(
                scenario_df, baseline_df, scenario_pred, baseline_pred, weekly_features
            )
            if weekly_change_df is not None and not weekly_change_df.empty:
                output_dir = Path(__file__).parent.parent / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                change_path = output_dir / "change.csv"
                weekly_change_df.to_csv(change_path, index=False)
                weekly_change_path = str(change_path.absolute())
                weekly_change_records = weekly_change_df.to_dict(orient="records")

        # Compute feature value deltas for top drivers (needed for narrative and response)
        feature_value_deltas = {}
        if baseline_df is not None:
            all_driver_features = [f for f, _ in positive_drivers + negative_drivers]
            feature_value_deltas = _compute_feature_deltas(
                scenario_df, baseline_df, all_driver_features
            )

        # Generate narrative with focus on modified features
        narrative = _generate_explanation_narrative(
            net_impact, positive_drivers, negative_drivers, by_category,
            modified_features=modified_features,
            feature_delta=feature_delta,
            feature_value_deltas=feature_value_deltas,
            sign_warnings=sign_warnings,
            baseline_reference=baseline_reference,
        )

        # Compute modified feature impacts for return dict
        modified_feature_impacts = []
        if modified_features and feature_delta:
            for f in modified_features:
                impact = feature_delta.get(f, 0)
                modified_feature_impacts.append({
                    "feature": f,
                    "impact": round(impact, 4),
                    "scale": _classify_impact_scale(impact),
                })

        # Compute interpretation context
        total_modified_impact = sum(abs(m["impact"]) for m in modified_feature_impacts)
        modified_impact_scale = "low" if total_modified_impact < 0.02 else "significant"

        interpretation = {
            "modified_features_impact_scale": modified_impact_scale,
        }
        if modified_impact_scale == "low" and modified_features:
            feature_names = ', '.join(
                f.replace('_roll_mean_4', '').replace('_dma', '').replace('_', ' ')
                for f in modified_features
            )
            interpretation["key_insight"] = (
                f"Low model sensitivity to {feature_names}. "
                f"SHAP deltas are negligible ({total_modified_impact:.3f} total), "
                f"indicating these levers have minimal impact on predicted sales."
            )

        # Build enhanced driver entries with feature values
        # (feature_value_deltas already computed above for narrative)
        def build_driver_with_values(feature: str, impact: float) -> Dict[str, Any]:
            entry = {"feature": feature, "shap_impact": round(impact, 4)}
            if feature in feature_value_deltas:
                entry.update(feature_value_deltas[feature])
            return entry

        # Build week_summary for YoY comparisons (clean format for LLM table generation)
        week_summary = None
        if baseline_reference == "YearOverYear" and by_horizon_breakdown:
            week_summary = []
            for h_entry in by_horizon_breakdown:
                summary_entry = {
                    "horizon": h_entry.get("horizon"),
                    "date_2025": h_entry.get("target_week_date"),
                    "date_2024": h_entry.get("baseline_target_week_date"),
                    "sales_predicted_2025": h_entry.get("sales_p50"),
                    "sales_actual_2024": h_entry.get("actual_sales_2024"),
                    "sales_delta_vs_actual": h_entry.get("delta_vs_actual_2024"),
                    "top_positive_driver": h_entry["top_positive"][0] if h_entry.get("top_positive") else None,
                    "top_negative_driver": h_entry["top_negative"][0] if h_entry.get("top_negative") else None,
                }
                week_summary.append(summary_entry)

        result_dict = {
            "status": "explained",
            "scenario": scenario_name,
            "baseline_reference": baseline_reference,
            "horizon_filter": horizon,
            "min_delta_threshold": min_delta_threshold,
            "net_impact": net_impact,
            "modified_features": modified_features,
            "modified_feature_impacts": modified_feature_impacts,
            "interpretation": interpretation,
            "changed_features_detected": changed_features,
            "top_positive_drivers": [
                build_driver_with_values(f, v) for f, v in positive_drivers
            ],
            "top_negative_drivers": [
                build_driver_with_values(f, v) for f, v in negative_drivers
            ],
            "by_category": by_category,
            "by_horizon": by_horizon_breakdown,
            "week_summary": week_summary,
            "weekly_change": weekly_change_records,
            "weekly_change_file": weekly_change_path,
            "features_filtered_below_threshold": features_filtered,
            "narrative": narrative,
            "sign_consistency_warnings": sign_warnings if sign_warnings else None,
            "warnings": warnings if warnings else None,
            "hint": "Use plot_driver_waterfall to visualize these drivers.",
        }

        # Store results for report generation
        try:
            session.store_explanation(json.dumps(result_dict, default=str))
        except Exception:
            pass  # Don't fail if storage fails

        return result_dict

    except Exception as e:
        return {"error": f"Explanation failed: {str(e)}"}


def _compute_shap_attribution(
    session,
    scenario_name: str,
    baseline_reference: str = "baseline",
    min_delta_threshold: float = 0.005,
    horizon: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP-based attribution between scenario and baseline.

    This is a helper function that extracts the SHAP computation logic for reuse
    by both explain_forecast_change and plot_driver_waterfall.

    Args:
        session: The forecasting session.
        scenario_name: Scenario to explain (must have predictions).
        baseline_reference: Reference for comparison ("baseline" or "YearOverYear").
        min_delta_threshold: Minimum absolute delta to include a feature.
        horizon: Optional specific horizon week to analyze.

    Returns:
        Dictionary containing:
        - by_category: {category: impact_value}
        - top_positive_drivers: list of (feature, impact) tuples
        - top_negative_drivers: list of (feature, impact) tuples
        - feature_delta: {feature: delta_value} for all features above threshold
        - by_horizon: list of per-horizon breakdowns (if horizon is None)
        Returns None if SHAP computation fails.
    """
    # Get checkpoint directory and channel
    checkpoint_dir = session.get_checkpoint_dir()
    channel = session.get_channel()

    try:
        from app.regressor.models import SurrogateExplainer
        from app.regressor.features.model_views import get_model_a_features_for_channel

        # Load surrogate model
        surrogate_file = f"surrogate_{'bm' if channel == 'B&M' else 'web'}.cbm"
        surrogate_path = checkpoint_dir / surrogate_file
        meta_path = checkpoint_dir / f"{surrogate_file[:-4]}.meta.json"

        if not surrogate_path.exists():
            return None

        surrogate = SurrogateExplainer(channel=channel)
        surrogate.load_model(surrogate_path, meta_path=meta_path)

        # Get Model A features
        model_a_features = get_model_a_features_for_channel(channel)

        # Get scenario data
        scenario = session.get_scenario(scenario_name)
        if scenario is None:
            return None

        # Extract modified features from scenario audit trail
        modified_features = []
        if scenario.modifications:
            modified_features = list(set(
                m.get("feature", "") for m in scenario.modifications if m.get("feature")
            ))

        scenario_df = scenario.df.copy()

        if baseline_reference == "baseline":
            # Use channel-specific baseline name
            channel = session.get_channel()
            baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web
            baseline_scenario = session.get_scenario(baseline_name)
            if baseline_scenario is None:
                return None
            baseline_df = baseline_scenario.df.copy()
        elif baseline_reference == "YearOverYear":
            baseline_df, _ = _load_year_over_year_baseline(session, scenario_df)
        else:
            baseline_df = None

        # Filter to specific horizon if requested
        if horizon is not None:
            if "horizon" not in scenario_df.columns:
                return None
            scenario_df = scenario_df[scenario_df["horizon"] == horizon].copy()
            if baseline_df is not None:
                baseline_df = baseline_df[baseline_df["horizon"] == horizon].copy()
            if scenario_df.empty:
                return None

        # Filter to Model A features that exist in the data
        available_features = [f for f in model_a_features if f in scenario_df.columns]

        if len(available_features) < 5:
            return None

        changed_features = []
        if baseline_df is not None:
            changed_features = _detect_changed_features(
                scenario_df, baseline_df, available_features, tol=FEATURE_CHANGE_TOLERANCE
            )

        # For YoY comparison, ONLY show features that actually changed between years
        if baseline_reference == "YearOverYear":
            feature_filter = changed_features
        else:
            feature_filter = changed_features or modified_features

        # Compute SHAP values with row-aligned delta
        feature_delta = {}
        by_horizon_breakdown = []

        if baseline_df is not None and baseline_reference in ("baseline", "YearOverYear"):
            # Ensure row alignment by sorting on key columns
            key_cols = ["profit_center_nbr", "horizon"]
            available_keys = [k for k in key_cols if k in scenario_df.columns and k in baseline_df.columns]

            if available_keys:
                # Sort to ensure same order for row-by-row delta
                scenario_aligned = scenario_df.sort_values(available_keys).reset_index(drop=True)
                baseline_aligned = baseline_df.sort_values(available_keys).reset_index(drop=True)

                # Compute SHAP for aligned data using batched method
                scenario_shap = surrogate.compute_shap_values_batched(scenario_aligned[available_features])
                baseline_shap = surrogate.compute_shap_values_batched(baseline_aligned[available_features])

                if scenario_shap is None or baseline_shap is None:
                    return None

                # Row-by-row delta, then mean (this ensures identical features cancel out)
                delta_shap = (scenario_shap - baseline_shap).mean(axis=0)
                feature_delta = dict(zip(available_features, delta_shap))

                # Free memory
                del scenario_shap, baseline_shap
                gc.collect()

                # Compute per-horizon breakdown if enabled and not filtering to single horizon
                from app.regressor.models.surrogate import COMPUTE_PER_HORIZON_SHAP
                if COMPUTE_PER_HORIZON_SHAP and horizon is None and "horizon" in scenario_aligned.columns:
                    horizons = sorted(scenario_aligned["horizon"].unique())
                    for h in horizons:
                        h_mask_s = scenario_aligned["horizon"] == h
                        h_mask_b = baseline_aligned["horizon"] == h

                        if h_mask_s.sum() == 0 or h_mask_b.sum() == 0:
                            continue

                        h_scenario_feat = scenario_aligned.loc[h_mask_s, available_features]
                        h_baseline_feat = baseline_aligned.loc[h_mask_b, available_features]

                        h_shap_s = surrogate.compute_shap_values(h_scenario_feat)
                        h_shap_b = surrogate.compute_shap_values(h_baseline_feat)

                        if h_shap_s is not None and h_shap_b is not None:
                            h_delta = (h_shap_s - h_shap_b).mean(axis=0)
                            h_drivers = {f: v for f, v in zip(available_features, h_delta)
                                        if abs(v) >= min_delta_threshold}

                            # Filter to only changed or modified features if available (removes noise)
                            if feature_filter:
                                h_drivers = {f: v for f, v in h_drivers.items() if f in feature_filter}

                            # Only add horizon if there are drivers to show
                            if h_drivers:
                                top_pos = sorted([(f, v) for f, v in h_drivers.items() if v > 0],
                                               key=lambda x: -x[1])[:3]
                                top_neg = sorted([(f, v) for f, v in h_drivers.items() if v < 0],
                                               key=lambda x: x[1])[:3]

                                by_horizon_breakdown.append({
                                    "horizon": int(h),
                                    "top_positive": [{"feature": f, "delta": round(v, 4)} for f, v in top_pos],
                                    "top_negative": [{"feature": f, "delta": round(v, 4)} for f, v in top_neg],
                                })
            else:
                # Fallback: compute SHAP without alignment (less accurate)
                scenario_shap = surrogate.compute_shap_values_batched(scenario_df[available_features])
                baseline_shap = surrogate.compute_shap_values_batched(baseline_df[available_features])
                if scenario_shap is not None and baseline_shap is not None:
                    delta_shap = scenario_shap.mean(axis=0) - baseline_shap.mean(axis=0)
                    feature_delta = dict(zip(available_features, delta_shap))
                    # Free memory
                    del scenario_shap, baseline_shap
                    gc.collect()
                else:
                    return None
        else:
            # No baseline comparison - use absolute SHAP importance
            scenario_shap = surrogate.compute_shap_values_batched(scenario_df[available_features])
            if scenario_shap is None:
                return None
            mean_shap = np.abs(scenario_shap).mean(axis=0)
            feature_delta = dict(zip(available_features, mean_shap))
            # Free memory
            del scenario_shap
            gc.collect()

        # Apply threshold filtering to remove noise
        features_before_filter = len(feature_delta)
        if feature_filter:
            feature_delta = {f: v for f, v in feature_delta.items() if f in feature_filter}

        feature_delta_filtered = {f: v for f, v in feature_delta.items() if abs(v) >= min_delta_threshold}
        features_filtered = features_before_filter - len(feature_delta_filtered)

        # Sort by absolute impact
        sorted_features = sorted(
            feature_delta_filtered.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Separate positive and negative drivers
        positive_drivers = [(f, v) for f, v in sorted_features if v > 0][:5]
        negative_drivers = [(f, v) for f, v in sorted_features if v < 0][:5]

        # Group by category
        by_category = {}
        for category, features in FEATURE_CATEGORIES.items():
            cat_impact = sum(feature_delta_filtered.get(f, 0) for f in features if f in feature_delta_filtered)
            by_category[category] = round(cat_impact, 4)

        return {
            "by_category": by_category,
            "top_positive_drivers": positive_drivers,
            "top_negative_drivers": negative_drivers,
            "feature_delta": feature_delta_filtered,
            "by_horizon": by_horizon_breakdown,
            "features_filtered": features_filtered,
        }

    except Exception as e:
        return None


def _explain_without_shap(scenario, baseline_pred, scenario_pred):
    """Generate explanation without SHAP when surrogate unavailable."""
    # Fall back to comparing feature changes
    net_impact = {}
    if baseline_pred and scenario_pred:
        baseline_sales = baseline_pred.predictions_df["pred_sales_p50"].sum()
        scenario_sales = scenario_pred.predictions_df["pred_sales_p50"].sum()
        if baseline_sales > 0:
            delta_sales = scenario_sales - baseline_sales
            delta_pct = (delta_sales / baseline_sales) * 100
            net_impact = {
                "baseline_sales": round(baseline_sales, 2),
                "scenario_sales": round(scenario_sales, 2),
                "delta_sales": round(delta_sales, 2),
                "delta_sales_pct": round(delta_pct, 2),
            }

    # List modifications made to scenario
    modifications = scenario.modifications if scenario else []

    return {
        "status": "explained_partial",
        "message": "SHAP-based explanation unavailable (surrogate model not found). "
        "Showing modification-based summary instead.",
        "net_impact": net_impact,
        "modifications": modifications[-5:],  # Last 5 modifications
        "hint": "To enable SHAP explanations, ensure surrogate models are trained.",
    }


def _classify_impact_scale(impact: float) -> str:
    """
    Classify SHAP delta impact magnitude.

    Args:
        impact: SHAP delta value

    Returns:
        Scale classification: "high", "medium", "low", or "negligible"
    """
    abs_impact = abs(impact)
    if abs_impact >= 0.1:
        return "high"
    elif abs_impact >= 0.03:
        return "medium"
    elif abs_impact >= 0.01:
        return "low"
    else:
        return "negligible"


def _generate_explanation_narrative(
    net_impact,
    positive_drivers,
    negative_drivers,
    by_category,
    modified_features=None,
    feature_delta=None,
    feature_value_deltas=None,
    sign_warnings=None,
    baseline_reference=None,
):
    """Generate human-readable explanation narrative with focus on modified features.

    Args:
        net_impact: Dictionary with sales impact metrics.
        positive_drivers: List of (feature, impact) tuples with positive impact.
        negative_drivers: List of (feature, impact) tuples with negative impact.
        by_category: Dictionary mapping category to total impact.
        modified_features: List of features modified in what-if scenario.
        feature_delta: Dictionary mapping feature to SHAP delta.
        feature_value_deltas: Dictionary mapping feature to value changes.
        sign_warnings: List of sign consistency warning messages.
        baseline_reference: Type of baseline ("baseline" or "YearOverYear").
    """
    parts = []

    # Net impact summary - different format for YoY vs what-if comparisons
    if net_impact:
        if baseline_reference == "YearOverYear":
            # YoY comparison: focus on actual vs predicted
            actual_2024 = net_impact.get("actual_sales_2024")
            predicted_2025 = net_impact.get("predicted_sales_2025")
            delta_vs_actual = net_impact.get("delta_vs_actual_2024")
            delta_pct = net_impact.get("delta_vs_actual_2024_pct", 0)
            if actual_2024 and predicted_2025:
                direction = "higher" if delta_pct > 0 else "lower"
                parts.append(
                    f"YoY Comparison: 2025 forecast (${predicted_2025:,.0f}) is {abs(delta_pct):.1f}% "
                    f"{direction} than 2024 actual (${actual_2024:,.0f}), "
                    f"a ${abs(delta_vs_actual):,.0f} {'increase' if delta_vs_actual > 0 else 'decrease'}."
                )
        else:
            # What-if comparison: standard format
            delta_pct = net_impact.get("delta_sales_pct", 0)
            direction = "increase" if delta_pct > 0 else "decrease"
            parts.append(
                f"Net Impact: {abs(delta_pct):.1f}% {direction} in forecasted sales "
                f"(${net_impact.get('delta_sales', 0):,.0f})."
            )

    # For YoY comparisons, focus on features that actually changed
    if baseline_reference == "YearOverYear" and feature_value_deltas:
        # Build narrative for YoY with actual value changes
        driver_narratives = []
        all_drivers = positive_drivers + negative_drivers
        for f, shap_impact in sorted(all_drivers, key=lambda x: -abs(x[1]))[:3]:
            feature_name = f.replace('_roll_mean_4', '').replace('_dma', '').replace('_', ' ')
            if f in feature_value_deltas:
                val_info = feature_value_deltas[f]
                val_delta = val_info.get("delta", 0)
                val_from = val_info.get("value_baseline", 0)
                val_to = val_info.get("value_current", 0)
                change_dir = "increased" if val_delta > 0 else "decreased"
                impact_dir = "positive" if shap_impact > 0 else "negative"
                driver_narratives.append(
                    f"{feature_name} {change_dir} ({val_from:.1f} -> {val_to:.1f}), "
                    f"{impact_dir} impact ({shap_impact:+.3f})"
                )
            else:
                impact_dir = "positive" if shap_impact > 0 else "negative"
                driver_narratives.append(
                    f"{feature_name}: {impact_dir} impact ({shap_impact:+.3f})"
                )

        if driver_narratives:
            parts.append("Key year-over-year drivers: " + "; ".join(driver_narratives) + ".")

    # Focus on modified features if provided (this is the key insight for what-if analysis)
    elif modified_features and feature_delta:
        modified_impacts = {f: feature_delta.get(f, 0) for f in modified_features}
        total_impact = sum(abs(v) for v in modified_impacts.values())

        if total_impact < 0.02:
            # Low sensitivity - this is the key insight
            feature_names = ', '.join(
                f.replace('_roll_mean_4', '').replace('_dma', '').replace('_', ' ')
                for f in modified_features
            )
            parts.append(
                f"Sensitivity Analysis: The modified feature(s) ({feature_names}) show "
                f"LOW model sensitivity. SHAP delta is only {total_impact:.3f}, indicating "
                f"these levers have minimal impact on sales. The forecast change is due to "
                f"normal model variation, not offsetting factors."
            )
        else:
            # Significant impact from modified features
            for f, v in sorted(modified_impacts.items(), key=lambda x: -abs(x[1])):
                scale = _classify_impact_scale(v)
                feature_name = f.replace('_roll_mean_4', '').replace('_dma', '').replace('_', ' ')
                direction = "positive" if v > 0 else "negative"
                parts.append(f"Modified lever '{feature_name}': {scale} {direction} impact ({v:+.3f}).")
    else:
        # No modified features info - fall back to general driver summary
        if positive_drivers:
            top_pos = positive_drivers[0]
            parts.append(f"Largest positive driver: {top_pos[0]} (+{top_pos[1]:.3f} impact).")

        if negative_drivers:
            top_neg = negative_drivers[0]
            parts.append(f"Largest negative driver: {top_neg[0]} ({top_neg[1]:.3f} impact).")

        # Category summary
        positive_cats = [(c, v) for c, v in by_category.items() if v > 0.001]
        negative_cats = [(c, v) for c, v in by_category.items() if v < -0.001]

        if positive_cats:
            cat_names = ", ".join([c for c, _ in sorted(positive_cats, key=lambda x: -x[1])[:3]])
            parts.append(f"Positive category contributions: {cat_names}.")

        if negative_cats:
            cat_names = ", ".join([c for c, _ in sorted(negative_cats, key=lambda x: x[1])[:3]])
            parts.append(f"Negative category contributions: {cat_names}.")

    # Add sign consistency warnings if any
    if sign_warnings:
        parts.append(
            f"Note: {len(sign_warnings)} driver(s) show counterintuitive effects "
            "(SHAP direction doesn't match feature change direction). "
            "This may indicate model complexity or interaction effects."
        )

    return " ".join(parts) if parts else "No significant drivers identified."


@tool
def analyze_sensitivity(
    store_id: Optional[int] = None,
    feature_category: Optional[str] = None,
    perturbation_pct: float = 20.0,
) -> Dict[str, Any]:
    """
    Analyze sensitivity of sales to changes in business levers.

    Perturbs key levers by +/- perturbation_pct and measures the resulting
    change in predicted sales to compute elasticity coefficients.

    Per Agent_plan.md Section 3.3 - analyze_sensitivity:
    1. Get lever list for category (or all operational levers)
    2. For each lever: Create +perturbation_pct and -perturbation_pct scenarios
    3. Run predictions for perturbed scenarios
    4. Calculate Elasticity: % Delta Sales / % Delta Input
    5. Rank by impact magnitude

    Args:
        store_id: Optional profit_center_nbr to focus on. Default: aggregate across all stores.
        feature_category: Optional category to analyze:
            - "financing": Primary, secondary, tertiary financing
            - "staffing": Unique associates, hours (B&M only)
            - "product_mix": Omni-channel, value, premium, white glove
            - "awareness": Brand awareness, consideration
            Default: all operational levers
        perturbation_pct: Percentage to perturb each lever (default 20%).

    Returns:
        Dictionary containing:
        - status: "analyzed" on success
        - elasticities: Ranked list of (feature, elasticity_coefficient)
        - most_sensitive: Feature with highest elasticity
        - least_sensitive: Feature with lowest elasticity
        - recommendations: Suggestions based on elasticity ranking

    Example:
        >>> analyze_sensitivity(feature_category="financing", perturbation_pct=10)
        {"status": "analyzed", "elasticities": [...], "most_sensitive": "pct_primary_financing_roll_mean_4", ...}
    """
    session = get_session()

    # Validate session
    if session.get_origin_date() is None:
        return {
            "error": "Session not initialized. Use initialize_forecast_simulation first."
        }

    # Use channel-specific baseline name
    channel = session.get_channel()
    baseline_name = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web

    baseline_scenario = session.get_scenario(baseline_name)
    if baseline_scenario is None:
        return {
            "error": f"No baseline scenario '{baseline_name}'. Initialize simulation first."
        }

    if not session.has_prediction(baseline_name):
        return {
            "error": f"No baseline predictions for '{baseline_name}'. Run run_forecast_model with baseline first."
        }

    # Get features to analyze
    channel = session.get_channel()
    if feature_category:
        features_to_test = get_features_by_category(feature_category)
        if not features_to_test:
            return {
                "error": f"Unknown category: '{feature_category}'. "
                f"Available: {list(FEATURE_CATEGORIES.keys())}"
            }
    else:
        # All modifiable features
        features_to_test = get_modifiable_features(channel)

    # Filter to features that exist in baseline
    df = baseline_scenario.df
    features_to_test = [f for f in features_to_test if f in df.columns]

    if not features_to_test:
        return {
            "error": "No analyzable features found in baseline scenario."
        }

    # Get baseline sales
    baseline_pred = session.get_prediction(baseline_name)
    baseline_sales = baseline_pred.predictions_df["pred_sales_p50"].sum()

    if baseline_sales == 0:
        return {"error": "Baseline sales is zero. Cannot compute elasticity."}

    # Analyze each feature by running actual model perturbations
    elasticities = []

    for feature in features_to_test:
        metadata = get_feature_metadata(feature)
        current_mean = df[feature].mean()

        if current_mean == 0 or pd.isna(current_mean):
            continue

        # Compute REAL elasticity via perturbation
        elasticity_result = _compute_real_elasticity(
            session, feature, perturbation_pct, baseline_sales, df
        )

        elasticities.append({
            "feature": feature,
            "category": metadata.category if metadata else "unknown",
            "current_mean": round(current_mean, 4),
            "elasticity_up": elasticity_result["elasticity_up"],
            "elasticity_down": elasticity_result["elasticity_down"],
            "elasticity": elasticity_result["elasticity_avg"],
            "is_asymmetric": elasticity_result["is_asymmetric"],
            "interpretation": _interpret_elasticity(
                elasticity_result["elasticity_avg"],
                elasticity_result["is_asymmetric"]
            ),
        })

    # Sort by absolute elasticity
    elasticities.sort(key=lambda x: abs(x["elasticity"]), reverse=True)

    # Find most and least sensitive
    most_sensitive = elasticities[0]["feature"] if elasticities else None
    least_sensitive = elasticities[-1]["feature"] if elasticities else None

    # Generate recommendations
    recommendations = _generate_sensitivity_recommendations(elasticities[:5])

    result_dict = {
        "status": "analyzed",
        "channel": channel,
        "perturbation_pct": perturbation_pct,
        "features_analyzed": len(elasticities),
        "elasticities": elasticities[:15],  # Top 15
        "most_sensitive": most_sensitive,
        "least_sensitive": least_sensitive,
        "recommendations": recommendations,
        "hint": "Higher elasticity means the lever has more impact on sales. "
        "Focus on high-elasticity levers for maximum effect.",
    }

    # Store results for report generation
    try:
        session.store_sensitivity(json.dumps(result_dict, default=str))
    except Exception:
        pass  # Don't fail if storage fails

    return result_dict


def _compute_real_elasticity(
    session,
    feature: str,
    perturbation_pct: float,
    baseline_sales: float,
    baseline_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute real elasticity by running +/- perturbation scenarios through the model.

    This runs actual CatBoost inference for perturbed feature values to measure
    how sales respond to feature changes. Returns asymmetric elasticity values
    for increases vs decreases.

    Args:
        session: The forecasting session with checkpoint and channel info.
        feature: Feature column name to perturb.
        perturbation_pct: Percentage to perturb (e.g., 20.0 for +/-20%).
        baseline_sales: Total p50 sales from baseline prediction.
        baseline_df: DataFrame with baseline feature values.

    Returns:
        Dictionary containing:
        - elasticity_up: Elasticity for +perturbation
        - elasticity_down: Elasticity for -perturbation
        - elasticity_avg: Average of both (symmetric assumption)
        - is_asymmetric: True if up/down differ significantly (>0.1)
    """
    from app.regressor.pipelines import InferencePipeline
    from app.regressor.configs import InferenceConfig

    # Get metadata for bounds checking
    metadata = get_feature_metadata(feature)
    current_mean = baseline_df[feature].mean()

    if current_mean == 0 or pd.isna(current_mean):
        return {
            "elasticity_up": 0.0,
            "elasticity_down": 0.0,
            "elasticity_avg": 0.0,
            "is_asymmetric": False,
        }

    checkpoint_dir = session.get_checkpoint_dir()
    channel = session.get_channel()

    config = InferenceConfig(
        checkpoint_dir=checkpoint_dir,
        output_dir=checkpoint_dir.parent / "infer",
        channels=[channel],
        run_explainability=False,
    )

    pipeline = InferencePipeline(config)

    results = {}

    for direction, mult in [("up", 1 + perturbation_pct / 100), ("down", 1 - perturbation_pct / 100)]:
        # Create perturbed DataFrame
        perturbed_df = baseline_df.copy()
        new_value = current_mean * mult

        # Clamp to bounds if metadata available
        if metadata and metadata.min_value is not None and metadata.max_value is not None:
            new_value = max(metadata.min_value, min(metadata.max_value, new_value))

        perturbed_df[feature] = new_value

        # Ensure channel column exists
        if "channel" not in perturbed_df.columns:
            perturbed_df["channel"] = channel

        # Run inference
        try:
            inference_result = pipeline.run(perturbed_df, channels=[channel])

            if channel == "B&M":
                pred_df = inference_result.bm_predictions
            else:
                pred_df = inference_result.web_predictions

            if pred_df is None or pred_df.empty:
                results[f"elasticity_{direction}"] = 0.0
                continue

            perturbed_sales = pred_df["pred_sales_p50"].sum()

            # Compute elasticity: % change in sales / % change in feature
            pct_change_feature = (new_value - current_mean) / current_mean * 100
            pct_change_sales = (perturbed_sales - baseline_sales) / baseline_sales * 100

            if abs(pct_change_feature) > 0.01:
                elasticity = pct_change_sales / pct_change_feature
            else:
                elasticity = 0.0

            results[f"elasticity_{direction}"] = round(elasticity, 4)

        except Exception:
            results[f"elasticity_{direction}"] = 0.0

    # Compute average and check for asymmetry
    e_up = results.get("elasticity_up", 0.0)
    e_down = results.get("elasticity_down", 0.0)
    results["elasticity_avg"] = round((e_up + e_down) / 2, 4)
    results["is_asymmetric"] = abs(e_up - e_down) > 0.1

    return results


def _interpret_elasticity(elasticity: float, is_asymmetric: bool = False) -> str:
    """
    Interpret elasticity coefficient with asymmetry indicator.

    Args:
        elasticity: The average elasticity value.
        is_asymmetric: True if up/down elasticities differ significantly.

    Returns:
        Human-readable interpretation of the elasticity.
    """
    abs_e = abs(elasticity)
    direction = "positive" if elasticity > 0 else "negative"

    if abs_e >= 0.20:
        base = f"High {direction} impact"
    elif abs_e >= 0.10:
        base = f"Moderate {direction} impact"
    elif abs_e >= 0.05:
        base = f"Low {direction} impact"
    else:
        base = "Minimal impact"

    if is_asymmetric:
        base += " (asymmetric - increases/decreases have different effects)"

    return base


def _generate_sensitivity_recommendations(top_features: List[Dict]) -> List[str]:
    """Generate recommendations based on sensitivity analysis."""
    recommendations = []

    if not top_features:
        return ["No features analyzed. Initialize baseline first."]

    # Most impactful
    most_impact = top_features[0]
    recommendations.append(
        f"Focus on '{most_impact['feature']}' for maximum impact "
        f"(elasticity: {most_impact['elasticity']:.2f})."
    )

    # Find high-elasticity positive drivers
    positive = [f for f in top_features if f["elasticity"] > 0.15]
    if positive:
        features = ", ".join([f["feature"] for f in positive[:3]])
        recommendations.append(
            f"High-impact positive levers: {features}."
        )

    # Cannibalization warning
    negative = [f for f in top_features if f["elasticity"] < -0.05]
    if negative:
        features = ", ".join([f["feature"] for f in negative[:2]])
        recommendations.append(
            f"Watch for negative impacts from: {features}."
        )

    # Asymmetric warning
    asymmetric = [f for f in top_features if f.get("is_asymmetric", False)]
    if asymmetric:
        features = ", ".join([f["feature"] for f in asymmetric[:3]])
        recommendations.append(
            f"Asymmetric response detected for: {features}. "
            "Increases and decreases have different effects."
        )

    return recommendations


@tool
def fetch_previous_year_actuals(
    store_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Fetch actual sales data from the same period last year.

    Uses the current session's forecast period (origin_date + horizon_weeks)
    and retrieves historical actuals from 52 weeks earlier.

    This data can be used to:
    - Compare forecast predictions against last year's performance
    - Understand year-over-year trends
    - Include in reports showing Forecast vs Previous Year

    Args:
        store_ids: Optional list of store IDs. If None, uses session's store_filter.

    Returns:
        Dictionary containing:
        - status: "fetched" on success
        - period: The date range retrieved (previous year dates)
        - total_sales: Sum of actual sales for the period
        - weekly_breakdown: List of weekly actuals with:
            - target_week_date: Date of the week
            - actual_sales: Actual sales for that week
            - horizon: Week number in sequence
        - num_weeks: Number of weeks retrieved

    Example:
        >>> fetch_previous_year_actuals()
        {"status": "fetched", "total_sales": 4500000, "weekly_breakdown": [...]}
    """
    session = get_session()
    state = session.get_state()

    # Get forecast period from session
    origin_date = state.get("origin_date")
    horizon_weeks = state.get("horizon_weeks", 13)

    if not origin_date:
        return {
            "error": "No forecast initialized. Run initialize_forecast_simulation first.",
            "hint": "Initialize a forecast session before fetching previous year data.",
        }

    # Determine stores
    stores = store_ids or state.get("store_filter", [])
    if not stores:
        return {
            "error": "No stores specified. Provide store_ids or initialize forecast first.",
            "hint": "Either pass store_ids parameter or initialize forecast with stores.",
        }

    channel = state.get("channel", "B&M")

    # Calculate previous year period (52 weeks earlier)
    origin_dt = datetime.strptime(origin_date, "%Y-%m-%d")
    prev_year_origin = origin_dt - timedelta(weeks=52)
    prev_year_end = prev_year_origin + timedelta(weeks=horizon_weeks)

    # Fetch from MODEL_B (contains actual sales in label columns)
    try:
        df = load_model_b_filtered(
            profit_center_nbrs=stores,
            channel=channel,
        )
    except Exception as e:
        return {
            "error": f"Failed to load historical data: {e}",
            "hint": "Check HANA connection and ensure MODEL_B table is accessible.",
        }

    if df.empty:
        return {
            "error": "No historical data found in MODEL_B.",
            "hint": "Check that the stores have historical data.",
        }

    # Ensure datetime columns
    df["target_week_date"] = pd.to_datetime(df["target_week_date"])

    # Filter to previous year period
    # Use horizon=1 rows for actuals (these are the most recent predictions for each target week)
    mask = (
        (df["target_week_date"] >= prev_year_origin) &
        (df["target_week_date"] < prev_year_end)
    )

    # Also filter to horizon=1 if available for cleaner data
    if "horizon" in df.columns:
        mask = mask & (df["horizon"] == 1)

    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        return {
            "error": "No historical data found for previous year period.",
            "period_searched": {
                "start": str(prev_year_origin.date()),
                "end": str(prev_year_end.date()),
            },
            "hint": "The historical data may not cover this period.",
        }

    # Extract actual sales (convert from log if needed)
    if "label_log_sales" in df_filtered.columns:
        df_filtered["actual_sales"] = np.exp(df_filtered["label_log_sales"])
    elif "label_sales" in df_filtered.columns:
        df_filtered["actual_sales"] = df_filtered["label_sales"]
    else:
        return {
            "error": "No sales label column found in historical data.",
            "hint": "Expected 'label_log_sales' or 'label_sales' column.",
        }

    # Build weekly breakdown aggregated across stores
    weekly = df_filtered.groupby("target_week_date").agg({
        "actual_sales": "sum"
    }).reset_index().sort_values("target_week_date")

    weekly_breakdown = []
    for idx, row in enumerate(weekly.itertuples()):
        weekly_breakdown.append({
            "target_week_date": str(row.target_week_date.date()),
            "actual_sales": round(float(row.actual_sales), 2),
            "horizon": idx + 1,
        })

    # Build per-store breakdown if multiple stores
    by_store = None
    if len(stores) > 1:
        store_totals = df_filtered.groupby("profit_center_nbr").agg({
            "actual_sales": "sum"
        }).reset_index()
        by_store = [
            {
                "profit_center_nbr": int(row.profit_center_nbr),
                "total_sales": round(float(row.actual_sales), 2),
            }
            for row in store_totals.itertuples()
        ]

    result = {
        "status": "fetched",
        "period": {
            "start": str(prev_year_origin.date()),
            "end": str(prev_year_end.date()),
            "year": prev_year_origin.year,
        },
        "forecast_period": {
            "start": origin_date,
            "end": str((origin_dt + timedelta(weeks=horizon_weeks)).date()),
            "year": origin_dt.year,
        },
        "total_sales": round(float(df_filtered["actual_sales"].sum()), 2),
        "weekly_breakdown": weekly_breakdown,
        "num_weeks": len(weekly_breakdown),
        "num_stores": len(stores),
        "stores": stores,
    }

    if by_store:
        result["by_store"] = by_store

    # Store for report generation
    try:
        session.store_yoy_actuals(json.dumps(result, default=str))
    except Exception:
        pass  # Don't fail if storage fails

    return result


# Export all tools
__all__ = [
    "run_forecast_model",
    "explain_forecast_change",
    "analyze_sensitivity",
    "fetch_previous_year_actuals",
]
