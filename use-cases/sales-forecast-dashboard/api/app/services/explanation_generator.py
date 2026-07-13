"""
Explanation Generator Service

Generates human-readable explanations for sales forecast changes.
Converts SHAP deltas (in log-sales space) to dollar impacts and
creates structured text explanations.

Usage:
    from app.services.explanation_generator import generate_static_explanation

    explanation = generate_static_explanation(
        shap_features=shap_deltas,
        baseline_sales=407800,
        current_sales=387480,
        feature_config=config
    )
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def safe_yoy_percentage(
    current: Optional[float],
    baseline: Optional[float]
) -> Optional[float]:
    """
    Calculate YoY percentage change safely.

    Args:
        current: Current year value (2025)
        baseline: Previous year value (2024)

    Returns:
        Percentage change, or None if calculation not possible
    """
    if current is None or baseline is None:
        return None
    if baseline <= 0:
        return None
    return ((current - baseline) / baseline) * 100


def shap_to_dollars(shap_delta: float, base_sales: Optional[float]) -> Optional[float]:
    """
    Convert log-scale SHAP delta to dollar impact.

    Since SHAP values are in log-sales space, the conversion formula is:
        dollar_impact = base_sales * (exp(shap_delta) - 1)

    Uses 2024 baseline as base for "what changed from last year" framing.

    Args:
        shap_delta: SHAP delta value (in log space)
        base_sales: 2024 baseline sales to use as reference

    Returns:
        Dollar impact, or None if base_sales is invalid
    """
    if base_sales is None or base_sales <= 0:
        return None
    return base_sales * (math.exp(shap_delta) - 1)


def generate_static_explanation(
    shap_features: List[Dict],
    baseline_sales: Optional[float],
    current_sales: float,
    feature_config: Dict,
    top_n: int = 3
) -> Dict:
    """
    Generate structured explanation from SHAP deltas.

    Returns dict with summary, drivers (negative contributors),
    offsets (positive contributors), and other_factors (residual).
    Dollar impacts are estimates based on 2024 baseline.

    The actual YoY change may differ from the sum of SHAP-derived dollar
    impacts because:
    1. SHAP values are in log-space (non-linear conversion to dollars)
    2. SHAP explains model behavior, not exact dollar attribution
    3. We only show top N factors; many small factors may contribute

    We calculate "other_factors" as the residual to ensure the explanation
    accounts for the full YoY change, providing business transparency.

    Args:
        shap_features: List of SHAP feature dicts with 'feature', 'value',
                      'baseline_value', 'impact', 'has_baseline' keys
        baseline_sales: 2024 sales for same week (for dollar conversion)
        current_sales: 2025 predicted sales
        feature_config: Display configuration for features (templates, names)
        top_n: Number of top drivers/offsets to include

    Returns:
        Structured explanation dict with summary, drivers, offsets,
        other_factors_impact, is_estimated flag, and generated_by indicator
    """
    # Handle new stores without baseline
    has_baseline = baseline_sales is not None and baseline_sales > 0

    if not has_baseline:
        return {
            "summary": (
                f"This is a new store with no historical data for comparison. "
                f"Forecasted sales for this week: ${current_sales:,.0f}."
            ),
            "drivers": [],
            "offsets": [],
            "other_factors_impact": None,
            "is_estimated": True,
            "no_baseline": True,
            "generated_by": "static"
        }

    yoy_change = current_sales - baseline_sales
    yoy_pct = (yoy_change / baseline_sales) * 100

    # Sort by absolute impact
    sorted_features = sorted(
        shap_features,
        key=lambda x: abs(x.get('impact', 0)),
        reverse=True
    )

    # Separate negative and positive contributors
    negative = [f for f in sorted_features if f.get('impact', 0) < 0][:top_n]
    positive = [f for f in sorted_features if f.get('impact', 0) > 0][:top_n]

    # Build driver descriptions (negative contributors)
    drivers = []
    shown_impact_sum = 0.0
    for f in negative:
        dollar_impact = shap_to_dollars(f.get('impact', 0), baseline_sales)
        config = feature_config.get(f['feature'], {})

        if dollar_impact is not None:
            shown_impact_sum += dollar_impact

        drivers.append({
            "feature": f['feature'],
            "display_name": config.get('display_name', _format_feature_name(f['feature'])),
            "direction": "negative",
            "shap_impact": f.get('impact', 0),
            "dollar_impact": round(dollar_impact, 0) if dollar_impact else None,
            "description": _format_description(f, dollar_impact, config)
        })

    # Build offset descriptions (positive contributors)
    offsets = []
    for f in positive:
        dollar_impact = shap_to_dollars(f.get('impact', 0), baseline_sales)
        config = feature_config.get(f['feature'], {})

        if dollar_impact is not None:
            shown_impact_sum += dollar_impact

        offsets.append({
            "feature": f['feature'],
            "display_name": config.get('display_name', _format_feature_name(f['feature'])),
            "direction": "positive",
            "shap_impact": f.get('impact', 0),
            "dollar_impact": round(dollar_impact, 0) if dollar_impact else None,
            "description": _format_description(f, dollar_impact, config)
        })

    # Calculate "other factors" as the residual
    # This accounts for: factors not shown, non-linearity of SHAP->dollar conversion,
    # model baseline shifts, and any gaps between SHAP explanation and actual change
    other_factors_impact = round(yoy_change - shown_impact_sum, 0)

    # Generate summary
    direction = "lower" if yoy_change < 0 else "higher"
    main_driver = (
        drivers[0]['display_name'] if drivers
        else (offsets[0]['display_name'] if offsets else "various factors")
    )

    summary = (
        f"This week's forecast is ${abs(yoy_change):,.0f} {direction} than last year "
        f"({yoy_pct:+.1f}%), primarily due to {main_driver.lower()}."
    )

    return {
        "summary": summary,
        "drivers": drivers,
        "offsets": offsets,
        "other_factors_impact": other_factors_impact,
        "is_estimated": True,
        "generated_by": "static"
    }


def _format_description(
    feature: Dict,
    dollar_impact: Optional[float],
    config: Dict
) -> str:
    """
    Format a single feature's description, handling missing data.

    Args:
        feature: Feature dict with 'feature', 'value', 'baseline_value', 'impact'
        dollar_impact: Pre-computed dollar impact (can be None)
        config: Feature configuration with display_name and templates

    Returns:
        Human-readable description string
    """
    display_name = config.get('display_name', _format_feature_name(feature['feature']))
    current_value = feature.get('value')
    baseline_value = feature.get('baseline_value')

    # Case 1: No baseline value available
    if baseline_value is None:
        if dollar_impact is not None:
            return f"{display_name} is at {current_value}, contributing ~${abs(dollar_impact):,.0f} to forecast"
        else:
            return f"{display_name} is at {current_value}"

    # Case 2: Full comparison available - try to use template
    if dollar_impact is not None:
        impact = feature.get('impact', 0)
        template_key = f"template_{'positive' if impact > 0 else 'negative'}"
        template = config.get(template_key)

        if template:
            try:
                baseline_num = float(baseline_value)
                current_num = float(current_value)
                pct_change = ((current_num - baseline_num) / baseline_num) * 100 if baseline_num > 0 else 0
            except (ValueError, TypeError):
                pct_change = 0

            try:
                return template.format(
                    baseline=baseline_value,
                    current=current_value,
                    pct_change=f"{pct_change:+.0f}",
                    dollar_impact=f"${abs(dollar_impact):,.0f}"
                )
            except (KeyError, ValueError):
                pass

        # Fallback: generic description with dollar impact
        return f"{display_name} changed from {baseline_value} to {current_value}, impact ~${abs(dollar_impact):,.0f}"

    # Case 3: No dollar impact available
    return f"{display_name} changed from {baseline_value} to {current_value}"


def _format_feature_name(name: str) -> str:
    """
    Convert snake_case feature name to Title Case display name.

    Args:
        name: Feature name like 'staffing_hours_roll_mean_4'

    Returns:
        Formatted name like 'Staffing Hours Roll Mean 4'
    """
    return name.replace('_', ' ').title()
