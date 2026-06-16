"""
Validation and utility tools for the forecasting agent.

These tools provide feature information and scenario validation:
- get_feature_info: Describe feature bounds and category
- validate_scenario: Check scenario for errors before prediction

Per Agent_plan.md Section 3 (Utility tools)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

from app.agent.session import get_session
from app.agent.feature_mapping import (
    FEATURE_METADATA,
    FEATURE_CATEGORIES,
    get_feature_metadata,
    get_modifiable_features,
    get_features_by_category,
    is_bm_only_feature,
)


@tool
def get_feature_info(
    feature_name: Optional[str] = None,
    category: Optional[str] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get information about Model A features including bounds and categories.

    Provides metadata for business levers that can be modified in what-if analysis,
    including valid ranges, descriptions, and natural language aliases.

    Args:
        feature_name: Optional specific feature to describe. Accepts natural language
                     (e.g., "white glove") or technical name (e.g., "pct_white_glove_roll_mean_4").
        category: Optional category filter:
            - "financing": Primary, secondary, tertiary financing rates
            - "staffing": Unique associates, hours (B&M only)
            - "product_mix": Omni-channel, value, premium, white glove
            - "awareness": Brand awareness, consideration
            - "cannibalization": Network effects from nearby stores
            - "store_dna": Physical attributes, age
        channel: Optional channel filter ("B&M" or "WEB"). Excludes B&M-only features for WEB.

    Returns:
        Dictionary containing:
        - features: List of feature metadata (name, category, min, max, description, aliases)
        - count: Number of features returned
        - categories: Available categories
        - hint: Usage guidance

    Example:
        >>> get_feature_info(category="financing")
        {"features": [...], "count": 3, ...}

        >>> get_feature_info(feature_name="white glove")
        {"features": [{name: "pct_white_glove_roll_mean_4", ...}], ...}
    """
    session = get_session()

    # Default channel from session
    if channel is None:
        channel = session.get_channel()

    # Validate channel
    channel_upper = channel.upper() if channel else "B&M"
    if channel_upper not in ["B&M", "WEB"]:
        return {"error": f"Invalid channel: {channel}. Use 'B&M' or 'WEB'."}

    # Validate category
    if category and category not in FEATURE_CATEGORIES:
        return {
            "error": f"Unknown category: '{category}'. "
            f"Available categories: {list(FEATURE_CATEGORIES.keys())}"
        }

    features_out = []

    # If specific feature requested
    if feature_name:
        from app.agent.feature_mapping import resolve_feature_name
        resolved = resolve_feature_name(feature_name)

        if resolved is None:
            # Try to find partial matches
            partial_matches = []
            feature_lower = feature_name.lower()
            for name, meta in FEATURE_METADATA.items():
                if feature_lower in name.lower():
                    partial_matches.append(name)
                elif any(feature_lower in alias.lower() for alias in meta.aliases):
                    partial_matches.append(name)

            if partial_matches:
                return {
                    "error": f"Feature '{feature_name}' not found exactly. "
                    f"Did you mean one of: {partial_matches[:5]}?"
                }
            else:
                return {
                    "error": f"Feature '{feature_name}' not found. "
                    "Use get_feature_info() without arguments to see all features."
                }

        meta = FEATURE_METADATA.get(resolved)
        if meta:
            features_out.append({
                "name": meta.name,
                "category": meta.category,
                "description": meta.description,
                "aliases": meta.aliases,
                "min_value": meta.min_value,
                "max_value": meta.max_value,
                "default_value": meta.default_value,
                "channel_applicability": "B&M only" if is_bm_only_feature(meta.name) else "B&M and WEB",
            })
    else:
        # Return all features, optionally filtered
        if category:
            feature_names = get_features_by_category(category)
        else:
            feature_names = get_modifiable_features(channel_upper)

        for name in feature_names:
            meta = FEATURE_METADATA.get(name)
            if meta is None:
                continue

            # Skip B&M-only features for WEB channel
            if channel_upper == "WEB" and is_bm_only_feature(name):
                continue

            features_out.append({
                "name": meta.name,
                "category": meta.category,
                "description": meta.description,
                "aliases": meta.aliases[:3],  # Limit aliases shown
                "min_value": meta.min_value,
                "max_value": meta.max_value,
                "default_value": meta.default_value,
                "channel_applicability": "B&M only" if is_bm_only_feature(name) else "B&M and WEB",
            })

    # Sort by category then name
    features_out.sort(key=lambda x: (x.get("category", ""), x["name"]))

    return {
        "features": features_out,
        "count": len(features_out),
        "filters": {
            "feature_name": feature_name,
            "category": category,
            "channel": channel_upper,
        },
        "categories": list(FEATURE_CATEGORIES.keys()),
        "hint": "Use modify_business_lever with these feature names or aliases to change values.",
    }


@tool
def validate_scenario(
    scenario_name: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Check a scenario for errors and warnings before running predictions.

    Validates:
    1. Required fields are present
    2. Feature values are within valid bounds
    3. Boolean flags are valid (0/1)
    4. Weeks_since_open caps are respected
    5. Channel-appropriate features only
    6. No conflicting flags (e.g., is_new_store and is_comp_store)

    Args:
        scenario_name: Name of scenario to validate. Default: active scenario.
        strict: If True, warnings are treated as errors. Default: True.

    Returns:
        Dictionary containing:
        - is_valid: True if scenario passes all checks
        - errors: List of critical issues (prevent prediction)
        - warnings: List of non-critical issues (proceed with caution)
        - summary: Human-readable validation summary
        - status: "valid", "invalid", or "warnings"

    Example:
        >>> validate_scenario("high_awareness")
        {"is_valid": True, "errors": [], "warnings": [], "status": "valid", ...}
    """
    session = get_session()

    # Get scenario
    scenario_name = scenario_name or session.get_active_scenario_name()
    scenario = session.get_scenario(scenario_name)

    if scenario is None:
        return {
            "error": f"Scenario '{scenario_name}' not found. "
            f"Available: {list(session.get_state()['scenarios'].keys())}"
        }

    df = scenario.df
    channel = session.get_channel()

    errors = []
    warnings = []

    # 1. Check DataFrame is not empty
    if df is None or df.empty:
        errors.append("Scenario DataFrame is empty")
        return {
            "scenario_name": scenario_name,
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "summary": "Validation failed: Scenario has no data",
            "status": "invalid",
        }

    # 2. Check required columns
    required_columns = ["profit_center_nbr", "horizon"]
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # 3. Check feature bounds
    for name, meta in FEATURE_METADATA.items():
        if name not in df.columns:
            continue

        # Skip B&M-only features for WEB
        if channel == "WEB" and is_bm_only_feature(name):
            if df[name].notna().any():
                warnings.append(f"B&M-only feature '{name}' has values for WEB channel")
            continue

        col = df[name]

        # Check min bound
        below_min = col < meta.min_value
        if below_min.any():
            min_val = col[below_min].min()
            count = below_min.sum()
            errors.append(
                f"{name}: {count} values below minimum {meta.min_value} "
                f"(lowest: {min_val:.2f})"
            )

        # Check max bound
        above_max = col > meta.max_value
        if above_max.any():
            max_val = col[above_max].max()
            count = above_max.sum()
            errors.append(
                f"{name}: {count} values above maximum {meta.max_value} "
                f"(highest: {max_val:.2f})"
            )

    # 4. Check boolean flags
    boolean_features = ["is_outlet", "is_comp_store", "is_new_store"]
    for feat in boolean_features:
        if feat in df.columns:
            invalid = ~df[feat].isin([0, 1, 0.0, 1.0, True, False])
            if invalid.any():
                warnings.append(f"{feat} has non-boolean values in {invalid.sum()} rows")

    # 5. Check weeks_since_open consistency
    if "weeks_since_open" in df.columns:
        wso = df["weeks_since_open"]

        if "weeks_since_open_capped_13" in df.columns:
            wso_13 = df["weeks_since_open_capped_13"]
            invalid_13 = wso_13 > 13
            if invalid_13.any():
                errors.append(f"weeks_since_open_capped_13 exceeds 13 in {invalid_13.sum()} rows")

            inconsistent = wso < wso_13
            if inconsistent.any():
                warnings.append(f"weeks_since_open < weeks_since_open_capped_13 in {inconsistent.sum()} rows")

        if "weeks_since_open_capped_52" in df.columns:
            wso_52 = df["weeks_since_open_capped_52"]
            invalid_52 = wso_52 > 52
            if invalid_52.any():
                errors.append(f"weeks_since_open_capped_52 exceeds 52 in {invalid_52.sum()} rows")

    # 6. Check horizon validity
    if "horizon" in df.columns:
        invalid_horizons = (df["horizon"] < 1) | (df["horizon"] > 52)
        if invalid_horizons.any():
            bad_horizons = df.loc[invalid_horizons, "horizon"].unique().tolist()
            errors.append(f"Invalid horizons (must be 1-52): {bad_horizons}")

    # 7. Check conflicting flags
    if "is_new_store" in df.columns and "is_comp_store" in df.columns:
        conflicting = (df["is_new_store"] == 1) & (df["is_comp_store"] == 1)
        if conflicting.any():
            warnings.append(
                f"is_new_store=1 and is_comp_store=1 in {conflicting.sum()} rows "
                "(new stores typically aren't comp stores)"
            )

    # 8. Check for NaN in critical features
    critical_features = ["profit_center_nbr", "horizon", "dma"]
    for feat in critical_features:
        if feat in df.columns:
            na_count = df[feat].isna().sum()
            if na_count > 0:
                errors.append(f"Critical column '{feat}' has {na_count} missing values")

    # 9. Check store count
    if "profit_center_nbr" in df.columns:
        store_count = df["profit_center_nbr"].nunique()
        if store_count == 0:
            errors.append("No stores in scenario")
        elif store_count > 1000:
            warnings.append(f"Large scenario with {store_count} stores - prediction may be slow")

    # Determine validity
    is_valid = len(errors) == 0
    if strict and warnings:
        is_valid = False

    # Generate summary
    if errors:
        status = "invalid"
        summary = f"Validation failed: {len(errors)} error(s), {len(warnings)} warning(s)"
    elif warnings:
        status = "warnings"
        summary = f"Validation passed with {len(warnings)} warning(s)"
    else:
        status = "valid"
        summary = "All validation checks passed"

    return {
        "scenario_name": scenario_name,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "strict_mode": strict,
        "summary": summary,
        "status": status,
        "row_count": len(df),
        "store_count": df["profit_center_nbr"].nunique() if "profit_center_nbr" in df.columns else 0,
        "hint": "Fix errors before running predictions. Warnings are informational.",
    }


# Export all tools
__all__ = [
    "get_feature_info",
    "validate_scenario",
]
