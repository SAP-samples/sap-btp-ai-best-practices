"""
Text generation module for pharmaceutical anomaly detection explanations.

This module handles generation of human-readable explanations from
SHAP values and other analysis results.
"""

import pandas as pd
from typing import Any


def generate_textual_explanation(
    shap_values_instance: Any, 
    feature_values: pd.Series, 
    feature_columns: list, 
    anomaly_score: float, 
    rule_explanation: str
) -> str:
    """
    Generate human-readable explanation from SHAP values.
    
    Args:
        shap_values_instance: SHAP explanation for a single instance
        feature_values: Feature values for the instance
        feature_columns: Feature column names
        anomaly_score: Anomaly score for the instance
        rule_explanation: Rule-based explanation if available
        
    Returns:
        Human-readable explanation string
    """
    # Get feature contributions
    contributions = []
    base_value = shap_values_instance.base_values
    
    for i, feature in enumerate(feature_columns):
        contribution = shap_values_instance.values[i]
        value = feature_values.iloc[i]
        contributions.append({
            'feature': feature,
            'shap_value': contribution,
            'feature_value': value,
            'abs_contribution': abs(contribution)
        })
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
    
    # Create explanation
    top_contributors = contributions
    explanation_parts = []
    
    for contrib in top_contributors:
        direction = "increases" if contrib['shap_value'] > 0 else "decreases"
        explanation_parts.append(
            f"{contrib['feature']}: {contrib['feature_value']:.3f} "
            f"({direction} anomaly score by {abs(contrib['shap_value']):.3f})"
        )
    
    return "Top contributors: " + "; ".join(explanation_parts)