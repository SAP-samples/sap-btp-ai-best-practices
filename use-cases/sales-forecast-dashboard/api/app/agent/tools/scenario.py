"""
Scenario modification tools for the forecasting agent.

These tools handle scenario creation and modification:
- create_scenario: Fork an existing scenario into a new one
- modify_business_lever: Update feature columns
- simulate_new_store_opening: Add new store with network effects
- set_active_scenario: Switch which scenario is being edited

Per Agent_plan.md Section 3.2: Scenario Modification Tools
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.feature_mapping import (
    resolve_feature_name,
    get_feature_metadata,
    parse_modification,
    apply_modification,
    is_bm_only_feature,
    get_modifiable_features,
)


# =============================================================================
# Awareness Feature Detection
# =============================================================================

# Features that are awareness-related and should trigger budget estimation
_AWARENESS_FEATURES = {
    "brand_awareness_dma_roll_mean_4",
    "brand_consideration_dma_roll_mean_4",
}


def _is_awareness_feature(feature_name: str) -> bool:
    """Check if a feature is awareness-related."""
    return feature_name in _AWARENESS_FEATURES


@tool
def create_scenario(
    new_scenario_name: str,
    source_scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fork an existing scenario into a new one for what-if analysis.

    Creates a copy of the source scenario with a new name. The new scenario
    can then be modified independently using modify_business_lever or
    simulate_new_store_opening.

    Per Agent_plan.md Section 3.2:
        state.scenarios[new_name] = state.scenarios[source].copy()

    Args:
        new_scenario_name: Name for the new scenario (e.g., "aggressive_marketing").
                          Must be unique and not already exist.
        source_scenario: Name of scenario to copy from. Default uses channel-specific
                        baseline (baseline_bm or baseline_web based on current channel).

    Returns:
        Dictionary containing:
        - status: "created" on success
        - scenario_name: Name of the new scenario
        - source_scenario: Name of the source scenario
        - num_rows: Number of rows in the scenario
        - stores: List of stores in the scenario
        - horizons: List of horizons in the scenario

    Example:
        >>> create_scenario("high_awareness", source_scenario="baseline_bm")
        {"status": "created", "scenario_name": "high_awareness", ...}
    """
    session = get_session()

    # Determine source scenario - use channel-specific baseline if not specified
    if source_scenario is None:
        channel = session.get_channel()
        source_scenario = f"baseline_{channel.lower().replace('&', '')}"  # baseline_bm or baseline_web

    # Validate source exists
    source = session.get_scenario(source_scenario)
    if source is None:
        available = list(session.get_state()["scenarios"].keys())
        return {
            "error": f"Source scenario '{source_scenario}' not found. "
            f"Available scenarios: {available}"
        }

    # Check new name doesn't exist
    if new_scenario_name in session.get_state()["scenarios"]:
        return {
            "error": f"Scenario '{new_scenario_name}' already exists. "
            "Use a different name or delete the existing scenario first."
        }

    # Create copy
    new_scenario = source.copy(new_scenario_name)

    # Add to session
    session.add_scenario(new_scenario)
    session.set_active_scenario(new_scenario_name)

    summary = new_scenario.get_summary()

    return {
        "status": "created",
        "scenario_name": new_scenario_name,
        "source_scenario": source_scenario,
        "num_rows": summary["num_rows"],
        "stores": summary["stores"],
        "horizons": summary["horizons"],
        "hint": f"'{new_scenario_name}' is now the active scenario. "
        "Use modify_business_lever to make changes.",
    }


@tool
def modify_business_lever(
    feature_name: str,
    modification: str,
    scope_stores: Optional[List[int]] = None,
    scope_dmas: Optional[List[str]] = None,
    horizon_range: Optional[List[int]] = None,
    scenario_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update specific feature columns in a scenario.

    Supports natural language feature names (e.g., "white glove" -> "pct_white_glove_roll_mean_4")
    and various modification formats (e.g., "set to 50%", "increase by 10%").

    Per Agent_plan.md Section 3.2 - modify_business_lever:
    - Features: Supports all actionable levers from Model A (Financing, Staffing, Product Mix, Awareness)
    - Modification: "set to 0.5", "increase by 10%", "decrease by 5"
    - Scope: DMA list, Store list, or "All"
    - Date_range: Start/End horizons

    Args:
        feature_name: Feature to modify. Accepts:
            - Natural language: "white glove", "brand awareness", "financing"
            - Technical name: "pct_white_glove_roll_mean_4"
        modification: How to change the value:
            - "set to 50" or "set to 50%" - Set to specific value
            - "increase by 10%" - Increase by percentage
            - "decrease by 5" - Decrease by absolute amount
            - "+20%" or "-10%" - Shorthand for increase/decrease
        scope_stores: Optional list of profit_center_nbr to apply to (default: all stores).
        scope_dmas: Optional list of DMAs to apply to (default: all DMAs).
        horizon_range: Optional [start, end] horizon range (default: all horizons).
                       Example: [1, 4] applies only to horizons 1-4.
        scenario_name: Scenario to modify. Default: active scenario.

    Returns:
        Dictionary containing:
        - status: "modified" on success
        - feature: Resolved feature name
        - modification: Modification applied
        - rows_affected: Number of rows changed
        - old_mean: Mean value before modification
        - new_mean: Mean value after modification

    Example:
        >>> modify_business_lever("brand awareness", "increase by 15%")
        {"status": "modified", "feature": "brand_awareness_dma_roll_mean_4", ...}

        >>> modify_business_lever("white glove", "set to 50%", horizon_range=[1, 4])
        {"status": "modified", "feature": "pct_white_glove_roll_mean_4", ...}
    """
    session = get_session()

    # Get target scenario
    scenario_name = scenario_name or session.get_active_scenario_name()
    scenario = session.get_scenario(scenario_name)

    if scenario is None:
        return {
            "error": f"Scenario '{scenario_name}' not found. "
            "Initialize simulation first with initialize_forecast_simulation."
        }

    # Resolve feature name
    resolved_feature = resolve_feature_name(feature_name)
    if resolved_feature is None:
        available = get_modifiable_features(session.get_channel())
        return {
            "error": f"Unknown feature: '{feature_name}'. "
            f"Available features: {available[:10]}... (use get_feature_info for full list)"
        }

    # Check if feature is valid for current channel
    channel = session.get_channel()
    if channel == "WEB" and is_bm_only_feature(resolved_feature):
        return {
            "error": f"Feature '{resolved_feature}' is B&M-only and not available for WEB channel."
        }

    # Check if feature exists in scenario DataFrame
    df = scenario.df
    if resolved_feature not in df.columns:
        return {
            "error": f"Feature '{resolved_feature}' not found in scenario data. "
            "The scenario may need to be re-initialized with complete features."
        }

    # Parse modification
    try:
        parsed_mod = parse_modification(modification)
    except ValueError as e:
        return {"error": str(e)}

    # Get feature metadata for bounds checking
    metadata = get_feature_metadata(resolved_feature)

    # Build mask for scope
    mask = pd.Series([True] * len(df), index=df.index)

    if scope_stores:
        mask &= df["profit_center_nbr"].isin(scope_stores)

    if scope_dmas:
        mask &= df["dma"].isin(scope_dmas)

    if horizon_range and len(horizon_range) == 2:
        mask &= (df["horizon"] >= horizon_range[0]) & (df["horizon"] <= horizon_range[1])

    rows_affected = mask.sum()
    if rows_affected == 0:
        return {
            "error": "No rows match the specified scope. "
            "Check store_ids, dmas, and horizon_range parameters."
        }

    # Store old values for audit
    old_values = df.loc[mask, resolved_feature].copy()
    old_mean = old_values.mean()

    # Apply modification with clamping (no errors for out-of-bounds)
    clamping_warnings = []

    if parsed_mod.operation == "set":
        new_value = parsed_mod.value
        if metadata:
            # Clamp to bounds instead of returning error
            if new_value < metadata.min_value:
                clamping_warnings.append(
                    f"Value clamped from {new_value} to minimum {metadata.min_value}"
                )
                new_value = metadata.min_value
            elif new_value > metadata.max_value:
                clamping_warnings.append(
                    f"Value clamped from {new_value} to maximum {metadata.max_value}"
                )
                new_value = metadata.max_value
        df.loc[mask, resolved_feature] = new_value
    else:
        # Apply row-by-row for increase/decrease/multiply
        # apply_modification now returns (new_value, warning) tuple
        new_values_list = []
        for v in old_values:
            new_v, warning = apply_modification(v, parsed_mod, metadata)
            new_values_list.append(new_v)
            if warning:
                clamping_warnings.append(warning)
        df.loc[mask, resolved_feature] = new_values_list

    new_values = df.loc[mask, resolved_feature]
    new_mean = new_values.mean()

    # Record modification in audit trail
    scenario.add_modification(
        feature_name=resolved_feature,
        modification_type=parsed_mod.operation,
        old_value=float(old_mean),
        new_value=float(new_mean),
        scope={
            "stores": scope_stores,
            "dmas": scope_dmas,
            "horizon_range": horizon_range,
            "rows_affected": int(rows_affected),
        },
    )

    # Invalidate cached predictions
    session.invalidate_prediction(scenario_name)

    response = {
        "status": "modified",
        "feature": resolved_feature,
        "modification": modification,
        "parsed_as": f"{parsed_mod.operation} {parsed_mod.value}"
        + ("%" if parsed_mod.is_percentage else ""),
        "rows_affected": int(rows_affected),
        "old_mean": float(old_mean),
        "new_mean": float(new_mean),
        "change": f"{((new_mean - old_mean) / old_mean * 100) if old_mean != 0 else 0:.1f}%",
        "scenario": scenario_name,
        "hint": "Predictions invalidated. Use run_forecast_model to see updated forecasts.",
    }

    # Add clamping info if any values were clamped
    if clamping_warnings:
        response["clamping_applied"] = True
        # Dedupe warnings and limit to 3 examples
        unique_warnings = list(set(clamping_warnings))[:3]
        response["clamping_warnings"] = unique_warnings

    # Auto-invoke budget estimation for awareness features
    if _is_awareness_feature(resolved_feature):
        try:
            # Import here to avoid circular import
            from app.agent.tools.budget import estimate_budget_for_awareness

            # Determine DMA for estimation
            dma_for_estimate = None
            if scope_dmas and len(scope_dmas) == 1:
                dma_for_estimate = scope_dmas[0]
            else:
                dma_filter = session.get_dma_filter()
                if dma_filter and len(dma_filter) == 1:
                    dma_for_estimate = dma_filter[0]

            if dma_for_estimate:
                # Estimate budget for this awareness change
                budget_result = estimate_budget_for_awareness.invoke({
                    "target_awareness": float(new_mean),
                    "current_awareness": float(old_mean),
                    "dma": dma_for_estimate,
                })

                if budget_result.get("status") == "estimated":
                    response["budget_estimate"] = {
                        "market": budget_result.get("market"),
                        "estimated_monthly_budget": budget_result.get("estimated_budget"),
                        "budget_range": budget_result.get("budget_range"),
                        "confidence": budget_result.get("confidence"),
                        "interpretation": budget_result.get("interpretation"),
                    }
                    # Update hint to include budget info
                    budget_str = budget_result.get("estimated_budget")
                    if budget_str:
                        response["hint"] = (
                            f"Awareness modified. Estimated monthly budget: "
                            f"${budget_str/1000:.0f}K. "
                            "Use run_forecast_model to see updated forecasts."
                        )
        except Exception as e:
            # Don't fail the modification if budget estimation fails
            response["budget_estimate_error"] = str(e)

    return response


@tool
def simulate_new_store_opening(
    latitude: float,
    longitude: float,
    opening_date: str,
    dma: str,
    merchandising_sf: Optional[float] = None,
    is_outlet: bool = False,
    scenario_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simulate the impact of a new store opening on the network.

    Creates a new store entry and updates cannibalization features for
    all nearby existing stores within 20 miles.

    Per Agent_plan.md Section 3.2 - simulate_new_store_opening:
    1. Store DNA: If user doesn't specify, use Mode (categorical) and Median (numerical)
       of existing stores in the target DMA.
    2. Network Effect:
       - Identify all stores in the target DMA
       - Calculate Haversine distance
       - For stores within 20 miles:
         - Update min_dist_new_store_km
         - Update num_new_stores_within_10mi/20mi
         - Recalculate cannibalization_pressure using formula:
           pressure = SUM_j exp(-dist_ij/8km) * (1 + weeks_since_j_open/13)
    3. Add Store: Append new row for the new store

    Args:
        latitude: Latitude of new store location.
        longitude: Longitude of new store location.
        opening_date: Store opening date in YYYY-MM-DD format.
        dma: DMA (market) for the new store (e.g., "CHICAGO").
        merchandising_sf: Optional square footage. If None, uses DMA median.
        is_outlet: Whether new store is an outlet. Default False.
        scenario_name: Scenario to modify. Default: active scenario.

    Returns:
        Dictionary containing:
        - status: "simulated" on success
        - new_store: Details of the new store created
        - network_impact: Summary of impact on existing stores
        - affected_stores: List of stores with updated cannibalization

    Example:
        >>> simulate_new_store_opening(
        ...     latitude=41.8781,
        ...     longitude=-87.6298,
        ...     opening_date="2025-06-01",
        ...     dma="CHICAGO"
        ... )
        {"status": "simulated", "new_store": {...}, "network_impact": {...}}
    """
    session = get_session()

    # Only supported for B&M channel
    if session.get_channel() != "B&M":
        return {
            "error": "simulate_new_store_opening is only supported for B&M channel. "
            "Use set_channel to switch channels if needed."
        }

    # Get target scenario
    scenario_name = scenario_name or session.get_active_scenario_name()
    scenario = session.get_scenario(scenario_name)

    if scenario is None:
        return {
            "error": f"Scenario '{scenario_name}' not found. "
            "Initialize simulation first with initialize_forecast_simulation."
        }

    df = scenario.df

    # Parse opening date
    try:
        opening_dt = pd.to_datetime(opening_date)
    except Exception as e:
        return {"error": f"Invalid opening_date: {opening_date}. Use YYYY-MM-DD format."}

    # Get origin date for calculating weeks_since_open
    origin_date = session.get_origin_date()
    if origin_date:
        origin_dt = pd.to_datetime(origin_date)
    else:
        origin_dt = df["origin_week_date"].max() if "origin_week_date" in df.columns else datetime.now()

    # Calculate weeks since open (can be negative if opening is in future)
    weeks_since_open = max(0, (origin_dt - opening_dt).days / 7)

    # Get DMA defaults from existing stores
    dma_stores = df[df["dma"].str.upper() == dma.upper()] if "dma" in df.columns else df

    # Determine merchandising_sf
    if merchandising_sf is None:
        if "merchandising_sf" in dma_stores.columns and len(dma_stores) > 0:
            merchandising_sf = dma_stores["merchandising_sf"].median()
        else:
            return {
                "error": f"merchandising_sf is required. Could not determine default for DMA '{dma}'. "
                         "Please specify the store's square footage (typical range: 15000-35000 sqft)."
            }

    # Generate a new profit_center_nbr (use max + 1)
    max_pcn = df["profit_center_nbr"].max() if "profit_center_nbr" in df.columns else 0
    new_pcn = int(max_pcn) + 1

    # Calculate distance from new store to all existing stores
    affected_stores = []
    store_master = session.get_store_master()

    # Import haversine function
    from app.regressor.geo import haversine_miles

    # Get existing stores with coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        existing_coords = df[["profit_center_nbr", "latitude", "longitude"]].drop_duplicates()
    elif "latitude" in store_master.columns:
        existing_coords = store_master[
            ["profit_center_nbr", "latitude", "longitude"]
        ].drop_duplicates()
    else:
        existing_coords = pd.DataFrame(columns=["profit_center_nbr", "latitude", "longitude"])

    # Calculate distances
    for _, store_row in existing_coords.iterrows():
        if pd.isna(store_row["latitude"]) or pd.isna(store_row["longitude"]):
            continue

        distance_miles = haversine_miles(
            latitude, longitude,
            store_row["latitude"], store_row["longitude"]
        )
        distance_km = distance_miles * 1.60934

        if distance_miles <= 20:  # Within impact zone
            affected_stores.append({
                "profit_center_nbr": int(store_row["profit_center_nbr"]),
                "distance_miles": round(distance_miles, 2),
                "distance_km": round(distance_km, 2),
            })

    # Update cannibalization features for affected stores
    updated_count = 0
    for affected in affected_stores:
        pcn = affected["profit_center_nbr"]
        dist_km = affected["distance_km"]
        dist_miles = affected["distance_miles"]

        mask = df["profit_center_nbr"] == pcn

        # Update min_dist_new_store_km
        if "min_dist_new_store_km" in df.columns:
            current_min = df.loc[mask, "min_dist_new_store_km"].min()
            if pd.isna(current_min) or dist_km < current_min:
                df.loc[mask, "min_dist_new_store_km"] = dist_km

        # Update num_new_stores_within_10mi_last_52wk
        if "num_new_stores_within_10mi_last_52wk" in df.columns and dist_miles <= 10:
            df.loc[mask, "num_new_stores_within_10mi_last_52wk"] += 1

        # Update num_new_stores_within_20mi_last_52wk
        if "num_new_stores_within_20mi_last_52wk" in df.columns:
            df.loc[mask, "num_new_stores_within_20mi_last_52wk"] += 1

        # Update cannibalization_pressure using formula:
        # pressure_i = SUM_j exp(-dist_ij/8km) * (1 + weeks_since_j_open/13)
        if "cannibalization_pressure" in df.columns:
            # New store contribution to pressure
            lambda_km = 8.0
            tau_weeks = 13.0
            weeks_new_store = weeks_since_open

            pressure_contribution = np.exp(-dist_km / lambda_km) * (
                1 + weeks_new_store / tau_weeks
            )
            df.loc[mask, "cannibalization_pressure"] += pressure_contribution

        updated_count += mask.sum()

    # Create new store rows (one per horizon)
    horizons = sorted(df["horizon"].unique()) if "horizon" in df.columns else [1]
    new_rows = []

    # Get a template row from the DMA
    if not dma_stores.empty:
        template = dma_stores.iloc[0].to_dict()
    else:
        template = df.iloc[0].to_dict()

    for h in horizons:
        new_row = template.copy()
        new_row["profit_center_nbr"] = new_pcn
        new_row["dma"] = dma.upper()
        new_row["latitude"] = latitude
        new_row["longitude"] = longitude
        new_row["horizon"] = h
        new_row["is_outlet"] = 1 if is_outlet else 0
        new_row["is_new_store"] = 1
        new_row["is_comp_store"] = 0
        new_row["merchandising_sf"] = merchandising_sf
        new_row["weeks_since_open"] = weeks_since_open + h  # Increases with horizon
        new_row["weeks_since_open_capped_13"] = min(13, weeks_since_open + h)
        new_row["weeks_since_open_capped_52"] = min(52, weeks_since_open + h)

        # New store has no cannibalization from itself
        new_row["cannibalization_pressure"] = 0.0
        new_row["min_dist_new_store_km"] = 999.0
        new_row["num_new_stores_within_10mi_last_52wk"] = 0
        new_row["num_new_stores_within_20mi_last_52wk"] = 0

        new_rows.append(new_row)

    # Append new store rows to scenario
    new_df = pd.DataFrame(new_rows)
    scenario.df = pd.concat([df, new_df], ignore_index=True)

    # Record modification
    scenario.add_modification(
        feature_name="new_store_opening",
        modification_type="add_store",
        old_value=None,
        new_value={
            "profit_center_nbr": new_pcn,
            "latitude": latitude,
            "longitude": longitude,
            "dma": dma,
            "opening_date": opening_date,
        },
        scope={"affected_stores": len(affected_stores)},
    )

    # Invalidate predictions
    session.invalidate_prediction(scenario_name)

    return {
        "status": "simulated",
        "new_store": {
            "profit_center_nbr": new_pcn,
            "latitude": latitude,
            "longitude": longitude,
            "dma": dma,
            "opening_date": opening_date,
            "merchandising_sf": merchandising_sf,
            "is_outlet": is_outlet,
            "weeks_since_open": round(weeks_since_open, 1),
        },
        "network_impact": {
            "stores_within_10mi": len([s for s in affected_stores if s["distance_miles"] <= 10]),
            "stores_within_20mi": len(affected_stores),
            "rows_updated": updated_count,
        },
        "affected_stores": affected_stores[:20],  # Limit output
        "scenario": scenario_name,
        "hint": "Cannibalization pressure updated for affected stores. "
        "Use run_forecast_model to see impact on forecasts.",
    }


@tool
def set_active_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Switch which scenario is being edited/viewed.

    The active scenario is the default target for modify_business_lever and other
    modification tools. Use this to switch between scenarios for comparison.

    Args:
        scenario_name: Name of scenario to make active (must exist).

    Returns:
        Dictionary containing:
        - status: "switched" on success
        - active_scenario: Name of the now-active scenario
        - scenario_summary: Summary of the active scenario

    Example:
        >>> set_active_scenario("aggressive_marketing")
        {"status": "switched", "active_scenario": "aggressive_marketing", ...}
    """
    session = get_session()

    try:
        session.set_active_scenario(scenario_name)
    except KeyError as e:
        return {"error": str(e)}

    scenario = session.get_scenario(scenario_name)
    summary = scenario.get_summary() if scenario else {}

    return {
        "status": "switched",
        "active_scenario": scenario_name,
        "scenario_summary": summary,
        "has_predictions": session.has_prediction(scenario_name),
        "hint": f"'{scenario_name}' is now active. "
        "Modifications will apply to this scenario by default.",
    }


# Export all tools
__all__ = [
    "create_scenario",
    "modify_business_lever",
    "simulate_new_store_opening",
    "set_active_scenario",
]
