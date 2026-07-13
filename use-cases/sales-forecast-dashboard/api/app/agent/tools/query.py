"""
Query tools for the forecasting agent.

These tools handle reading/querying scenario data:
- get_feature_values: Query current feature values from a scenario

Per Agent_plan.md requirements for read access to scenario data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.feature_mapping import (
    resolve_feature_name,
    get_feature_metadata,
    get_display_name,
    is_bm_only_feature,
    FEATURE_METADATA,
)


# Valid aggregation modes
_AGGREGATION_MODES = {"statistics", "by_store", "by_horizon", "raw"}
_RAW_LIMIT = 100  # Max rows to return in raw mode


@tool
def get_feature_values(
    feature_names: List[str],
    scenario_name: Optional[str] = None,
    scope_stores: Optional[List[int]] = None,
    scope_dmas: Optional[List[str]] = None,
    horizon_range: Optional[List[int]] = None,
    aggregation: str = "statistics",
) -> Dict[str, Any]:
    """
    Query current feature values from a scenario.

    Use this tool to check feature values before or after modifications.
    Supports natural language feature names (e.g., "brand awareness") for
    Model A features, or exact column names for any DataFrame column.

    Args:
        feature_names: List of features to query. Accepts natural language
                      (e.g., ["brand awareness", "white glove"]) or technical
                      names (e.g., ["brand_awareness_dma_roll_mean_4"]).
                      Can also query any DataFrame column by exact name.
        scenario_name: Scenario to query. Default: active scenario.
        scope_stores: Optional list of profit_center_nbr to filter to.
        scope_dmas: Optional list of DMAs to filter to.
        horizon_range: Optional [start, end] horizon range (e.g., [1, 4]).
                      Use [1, 1] for t0 only. Default: all horizons.
        aggregation: How to aggregate multiple rows:
            - "statistics": Return mean, min, max, std, median (default)
            - "by_store": Return mean per store
            - "by_horizon": Return mean per horizon
            - "raw": Return all individual values (limited to 100 rows)

    Returns:
        Dictionary containing:
        - status: "success" on success
        - scenario_name: Name of the queried scenario
        - features: Dict mapping each feature to its values/statistics
        - scope: Summary of filters applied
        - rows_matched: Number of rows in the query result
        - hint: Contextual guidance

    Example:
        >>> get_feature_values(["brand awareness"])
        {"status": "success", "features": {"brand_awareness_dma_roll_mean_4":
         {"mean": 45.2, "min": 32.1, "max": 58.4, "std": 7.3}}, ...}

        >>> get_feature_values(["white glove"], scope_stores=[160], horizon_range=[1, 4])
        {"status": "success", "features": {"pct_white_glove_roll_mean_4":
         {"mean": 12.5, ...}}, "rows_matched": 4, ...}
    """
    session = get_session()

    # -------------------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------------------

    # Validate feature_names is not empty
    if not feature_names:
        return {"error": "feature_names cannot be empty. Provide at least one feature name."}

    # Validate aggregation mode
    if aggregation not in _AGGREGATION_MODES:
        return {
            "error": f"Invalid aggregation mode: '{aggregation}'. "
            f"Valid modes: {', '.join(sorted(_AGGREGATION_MODES))}"
        }

    # -------------------------------------------------------------------------
    # 2. Get Scenario
    # -------------------------------------------------------------------------

    scenario_name = scenario_name or session.get_active_scenario_name()
    scenario = session.get_scenario(scenario_name)

    if scenario is None:
        available = list(session.get_state().get("scenarios", {}).keys())
        if not available:
            return {
                "error": "No scenarios exist. Initialize simulation first "
                "with initialize_forecast_simulation."
            }
        return {
            "error": f"Scenario '{scenario_name}' not found. "
            f"Available: {available}"
        }

    df = scenario.df
    if df is None or df.empty:
        return {"error": f"Scenario '{scenario_name}' has no data."}

    channel = session.get_channel()

    # -------------------------------------------------------------------------
    # 3. Resolve Feature Names
    # -------------------------------------------------------------------------

    resolved_features = {}  # canonical_name -> user_input
    errors = []

    for user_input in feature_names:
        # First try to resolve via natural language aliases (Model A features)
        resolved = resolve_feature_name(user_input)

        if resolved is not None:
            # Check B&M-only constraint for Model A features
            if channel == "WEB" and is_bm_only_feature(resolved):
                errors.append(
                    f"Feature '{resolved}' is B&M-only and not available for WEB channel."
                )
                continue

            # Check feature exists in DataFrame
            if resolved not in df.columns:
                errors.append(
                    f"Feature '{resolved}' not found in scenario data. "
                    "The scenario may need to be re-initialized."
                )
                continue

            resolved_features[resolved] = user_input

        else:
            # Fallback: try exact column name match (for non-Model A features)
            if user_input in df.columns:
                resolved_features[user_input] = user_input
            else:
                # Try to suggest similar features
                suggestions = _find_similar_features(user_input, df.columns)
                if suggestions:
                    errors.append(
                        f"Unknown feature: '{user_input}'. "
                        f"Did you mean: {suggestions[:3]}?"
                    )
                else:
                    errors.append(f"Unknown feature: '{user_input}'.")

    if errors:
        # Return all errors combined
        return {"error": " ".join(errors)}

    if not resolved_features:
        return {"error": "No valid features to query."}

    # -------------------------------------------------------------------------
    # 4. Apply Scope Filters
    # -------------------------------------------------------------------------

    mask = pd.Series([True] * len(df), index=df.index)

    if scope_stores:
        mask &= df["profit_center_nbr"].isin(scope_stores)

    if scope_dmas and "dma" in df.columns:
        mask &= df["dma"].isin(scope_dmas)

    if horizon_range and len(horizon_range) == 2:
        mask &= (df["horizon"] >= horizon_range[0]) & (df["horizon"] <= horizon_range[1])

    filtered_df = df[mask]

    if len(filtered_df) == 0:
        return {
            "error": "No rows match the specified filters. "
            "Check scope_stores, scope_dmas, and horizon_range."
        }

    # -------------------------------------------------------------------------
    # 5. Compute Feature Values Based on Aggregation Mode
    # -------------------------------------------------------------------------

    features_result = {}

    for canonical_name, user_input in resolved_features.items():
        col_values = filtered_df[canonical_name]

        feature_data = {}

        # Add display name and resolution info for Model A features
        if canonical_name in FEATURE_METADATA:
            feature_data["display_name"] = get_display_name(canonical_name)
            if user_input != canonical_name:
                feature_data["resolved_from"] = user_input

            # Add metadata for Model A features
            metadata = get_feature_metadata(canonical_name)
            if metadata:
                feature_data["metadata"] = {
                    "category": metadata.category,
                    "min_bound": metadata.min_value,
                    "max_bound": metadata.max_value,
                    "value_type": metadata.value_type,
                }
        else:
            # Non-Model A column - just use the column name
            feature_data["display_name"] = canonical_name

        # Compute based on aggregation mode
        if aggregation == "statistics":
            feature_data["statistics"] = _compute_statistics(col_values)

        elif aggregation == "by_store":
            feature_data["by_store"] = _compute_by_store(
                filtered_df, canonical_name
            )

        elif aggregation == "by_horizon":
            feature_data["by_horizon"] = _compute_by_horizon(
                filtered_df, canonical_name
            )

        elif aggregation == "raw":
            values, truncated = _compute_raw_values(
                filtered_df, canonical_name
            )
            feature_data["values"] = values
            feature_data["truncated"] = truncated
            feature_data["total_rows"] = len(filtered_df)
            feature_data["returned_rows"] = len(values)

        features_result[canonical_name] = feature_data

    # -------------------------------------------------------------------------
    # 6. Build Response
    # -------------------------------------------------------------------------

    return {
        "status": "success",
        "scenario_name": scenario_name,
        "features": features_result,
        "scope": {
            "stores": scope_stores,
            "dmas": scope_dmas,
            "horizon_range": horizon_range,
        },
        "rows_matched": len(filtered_df),
        "aggregation_mode": aggregation,
        "hint": "Use modify_business_lever to change Model A feature values.",
    }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _compute_statistics(series: pd.Series) -> Dict[str, float]:
    """Compute summary statistics for a feature column."""
    return {
        "mean": round(float(series.mean()), 4) if not series.isna().all() else None,
        "min": round(float(series.min()), 4) if not series.isna().all() else None,
        "max": round(float(series.max()), 4) if not series.isna().all() else None,
        "std": round(float(series.std()), 4) if len(series) > 1 and not series.isna().all() else None,
        "median": round(float(series.median()), 4) if not series.isna().all() else None,
        "count": int(series.count()),
    }


def _compute_by_store(df: pd.DataFrame, feature: str) -> List[Dict[str, Any]]:
    """Compute feature statistics grouped by store."""
    result = []

    if "profit_center_nbr" not in df.columns:
        return result

    grouped = df.groupby("profit_center_nbr")[feature].agg(["mean", "min", "max", "count"])

    for pcn, row in grouped.iterrows():
        store_data = {
            "profit_center_nbr": int(pcn),
            "mean": round(float(row["mean"]), 4) if pd.notna(row["mean"]) else None,
            "min": round(float(row["min"]), 4) if pd.notna(row["min"]) else None,
            "max": round(float(row["max"]), 4) if pd.notna(row["max"]) else None,
            "count": int(row["count"]),
        }

        # Add DMA if available
        if "dma" in df.columns:
            dma_values = df[df["profit_center_nbr"] == pcn]["dma"]
            if not dma_values.empty:
                store_data["dma"] = dma_values.iloc[0]

        result.append(store_data)

    return result


def _compute_by_horizon(df: pd.DataFrame, feature: str) -> List[Dict[str, Any]]:
    """Compute feature statistics grouped by horizon."""
    result = []

    if "horizon" not in df.columns:
        return result

    grouped = df.groupby("horizon")[feature].agg(["mean", "min", "max", "count"])

    for h, row in grouped.iterrows():
        result.append({
            "horizon": int(h),
            "mean": round(float(row["mean"]), 4) if pd.notna(row["mean"]) else None,
            "min": round(float(row["min"]), 4) if pd.notna(row["min"]) else None,
            "max": round(float(row["max"]), 4) if pd.notna(row["max"]) else None,
            "count": int(row["count"]),
        })

    return sorted(result, key=lambda x: x["horizon"])


def _compute_raw_values(
    df: pd.DataFrame, feature: str
) -> tuple:
    """Return raw values (limited to _RAW_LIMIT rows)."""
    truncated = len(df) > _RAW_LIMIT
    subset = df.head(_RAW_LIMIT)

    result = []
    for _, row in subset.iterrows():
        entry = {
            "profit_center_nbr": int(row["profit_center_nbr"]) if "profit_center_nbr" in row else None,
            "value": round(float(row[feature]), 4) if pd.notna(row[feature]) else None,
        }
        if "horizon" in row.index:
            entry["horizon"] = int(row["horizon"])
        if "dma" in row.index:
            entry["dma"] = row["dma"]
        result.append(entry)

    return result, truncated


def _find_similar_features(user_input: str, available_columns: pd.Index) -> List[str]:
    """Find features with similar names for suggestions."""
    user_lower = user_input.lower()
    suggestions = []

    # Check Model A feature aliases
    for name, meta in FEATURE_METADATA.items():
        if user_lower in name.lower():
            suggestions.append(name)
        elif any(user_lower in alias.lower() for alias in meta.aliases):
            suggestions.append(meta.aliases[0])  # Suggest the primary alias

    # Check DataFrame columns
    for col in available_columns:
        if user_lower in col.lower() and col not in suggestions:
            suggestions.append(col)

    return suggestions[:5]


# Export
__all__ = ["get_feature_values"]
