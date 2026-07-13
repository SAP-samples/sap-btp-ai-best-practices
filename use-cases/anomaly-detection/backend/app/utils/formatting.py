from __future__ import annotations

"""Formatting utilities shared across views."""

from typing import Optional

import pandas as pd


def format_value(value, feature_name: Optional[str] = None) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if feature_name and any(token in feature_name.lower() for token in ["price", "value"]):
            return f"${value:,.2f}"
        if feature_name and "percentage" in feature_name.lower():
            return f"{value:.1%}"
        return f"{value:,.2f}"
    return str(value)


def format_feature_name(name: str) -> str:
    return name.replace("_", " ").title()


def format_feature_value(value, feature_name: Optional[str] = None) -> str:
    return format_value(value, feature_name)


def describe_contribution(
    feature_name: str,
    feature_value,
    features_df: pd.DataFrame,
    context_row: pd.Series,
) -> str:
    typical_range = get_typical_range_for_feature(feature_name, feature_value, features_df, context_row)
    base_text = "Anomalous"
    return f"{base_text} {typical_range}" if typical_range else base_text


def get_typical_range_for_feature(
    feature_name: str,
    feature_value,
    features_df: pd.DataFrame,
    row: pd.Series,
) -> Optional[str]:
    if features_df is None or features_df.empty or row is None:
        return None

    sold_to = row.get("Sold To number")
    material_number = row.get("Material Number")

    if sold_to is None or material_number is None:
        return None

    filtered_data = features_df[
        (features_df["Sold To number"] == sold_to)
        & (features_df["Material Number"] == material_number)
    ]

    if feature_name.startswith("is_") or (isinstance(feature_value, (int, float)) and feature_value in [0, 1]):
        if feature_name in filtered_data.columns and len(filtered_data) > 0:
            prevalence = filtered_data[feature_name].mean()
            return f"- Typical for this customer-material: {prevalence:.1%} of orders"
        return "- Typical: Rare occurrence"

    if feature_name == "qty_z_score":
        return "- Typical: 0 (range: -2 to +2)"
    if feature_name == "qty_deviation_from_mean":
        return "- Typical: 0 (Expected Deviation)"

    if feature_name in filtered_data.columns and len(filtered_data) > 0:
        feature_data = filtered_data[feature_name].dropna()

        if not pd.api.types.is_numeric_dtype(feature_data):
            feature_data = pd.to_numeric(feature_data, errors="coerce").dropna()

        if feature_data.empty or not pd.api.types.is_numeric_dtype(feature_data):
            return None

        if len(feature_data) > 1:
            if feature_name == "current_month_total_qty":
                p_low = feature_data.quantile(0.01)
                p_high = feature_data.quantile(0.99)
            else:
                p_low = feature_data.quantile(0.05)
                p_high = feature_data.quantile(0.95)

            if "price" in feature_name.lower() or "value" in feature_name.lower():
                return f"- Typical for this customer-material: ${p_low:,.0f} - ${p_high:,.0f}"
            if "qty" in feature_name.lower() or "quantity" in feature_name.lower():
                return f"- Typical for this customer-material: {p_low:,.0f} - {p_high:,.0f}"
            if "score" in feature_name.lower() and "z_" in feature_name.lower():
                return f"- Typical for this customer-material: {p_low:.2f} - {p_high:.2f}"
            if "days" in feature_name.lower() or "duration" in feature_name.lower():
                return f"- Typical for this customer-material: {p_low:.0f} - {p_high:.0f} days"
            return f"- Typical for this customer-material: {p_low:,.1f} - ${p_high:,.1f}"

        if len(feature_data) == 1:
            single_val = feature_data.iloc[0]
            if "price" in feature_name.lower() or "value" in feature_name.lower():
                return f"- Typical for this customer-material: ${single_val:,.0f}"
            if "qty" in feature_name.lower() or "quantity" in feature_name.lower():
                return f"- Typical for this customer-material: {single_val:,.0f}"
            return f"- Typical for this customer-material: {single_val:,.1f}"

    return None
